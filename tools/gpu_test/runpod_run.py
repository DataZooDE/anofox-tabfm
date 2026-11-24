#!/usr/bin/env python3
"""Run a command on a throwaway RunPod GPU pod, then destroy it.

There is no NVIDIA GPU on the dev box, so CUDA-only behaviour (the ggml-cuda
plugin) can otherwise only be reasoned about. This rents a pod by the minute, ships local
files to it, runs a command, streams the output back and terminates the pod.

NOT part of CI: it costs money and needs a key CI does not have. Run it by hand
when a change touches a GPU path.

    tools/gpu_test/runpod_run.py --upload vendor/anofox-ggml-adapter/build/ggml-cuda/libanofox_ggml-cuda_plugin.so \\
        --command 'nvidia-smi && ls -l /workspace'

The API key comes from RUNPOD_API_KEY, or from ~/.bashrc — which exports it but
returns early for non-interactive shells, so it is read out of the file rather
than sourced. It is never printed.

COST: the pod is terminated in a finally block, on exception, and on SIGINT.
If the process is SIGKILLed the pod survives — list and clean up with
``--list`` / ``--terminate <id>``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# REST API v2. This used to be https://rest.runpod.io/v1, a host that no longer resolves (NXDOMAIN from
# every resolver tried); RunPod's own SDK (runpod>=2) and reference (docs.runpod.io/api-reference-v2) use
# api.runpod.io/v2, whose pod body and response differ from v1 (see pod_spec / pod_endpoint).
API = "https://api.runpod.io/v2"
# CUDA 12.8 + python 3.11; sshd starts when PUBLIC_KEY is set.
#
# NOT 12.4, which this default used to be and which is known not to work for
# the thing this harness exists to test: the CUDA-12 ONNX Runtime provider
# fails on a 12.4 image with `undefined symbol: cudaLibraryGetKernel, version
# libcudart.so.12` -- that symbol needs a newer CUDA 12 minor than the image
# ships. Recorded in docs/archive/DYNAMIC_BACKENDS.md ("RunPod-verified" findings 1-2),
# discovered by renting a pod and watching it fail, and then left in place as
# the default so the next person rented a pod and watched it fail too.
# README pins the floor at CUDA userspace >= 12.5.
DEFAULT_IMAGE = "runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
GRAPHQL = "https://api.runpod.io/graphql"
# An EP-level graph bug reproduces on any CUDA device, and these graphs are tiny
# with synthesized weights (~110 MB), so the cheapest card that exists will do.
# "" means: ask the API what is cheapest right now.
DEFAULT_GPU = ""
MIN_VRAM_GB = 8


def api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY")
    if key:
        return key.strip()
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        m = re.search(r"^export RUNPOD_API_KEY=(.*)$", bashrc.read_text(), re.M)
        if m:
            return m.group(1).strip().strip("\"'")
    raise SystemExit("RUNPOD_API_KEY not set and not found in ~/.bashrc")


def call(method: str, path: str, key: str, body=None, timeout=60):
    req = urllib.request.Request(
        f"{API}{path}",
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "anofox-tabfm-gpu-test/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        text = exc.read().decode()
        try:  # v2 errors are application/problem+json: title, status, detail
            problem = json.loads(text)
            text = ": ".join(str(problem[k]) for k in ("title", "detail") if problem.get(k)) or text
        except (ValueError, AttributeError, TypeError):
            pass
        raise SystemExit(f"RunPod {method} {path} -> {exc.code}: {text[:400]}")


# --- pods ---------------------------------------------------------------------------------------------------


def list_pods(key: str) -> list:
    return (call("GET", "/pods", key) or {}).get("pods") or []


def get_pod(key: str, pod_id: str) -> dict:
    return call("GET", f"/pods/{pod_id}", key) or {}


def create_pod(key: str, spec: dict) -> dict:
    return call("POST", "/pods", key, spec)


def terminate_pod(pod_id: str, key: str) -> None:
    call("DELETE", f"/pods/{pod_id}", key)


def pod_endpoint(pod: dict):
    """(host, port) for SSH to the pod's 22/tcp, or None until RunPod has assigned one.

    v1 had ``publicIp`` and ``portMappings``; v2 has ``ssh.direct`` (plain ssh/scp, which is what this script
    needs; ``ssh.proxy`` is an interactive shell only) and, as a fallback, the runtime port table."""
    direct = (pod.get("ssh") or {}).get("direct")
    if direct and direct.get("host") and direct.get("port"):
        return direct["host"], int(direct["port"])
    for port in (pod.get("runtime") or {}).get("ports") or []:
        if port.get("private") == 22 and port.get("public") and port.get("ip"):
            return port["ip"], int(port["public"])
    return None


# --- network volumes ----------------------------------------------------------------------------------------


def list_volumes(key: str) -> list:
    return (call("GET", "/network-volumes", key) or {}).get("networkVolumes") or []


def get_volume(key: str, volume_id: str) -> dict:
    # v2 documents no GET-by-id for a volume, so pick it out of the list.
    for volume in list_volumes(key):
        if volume.get("id") == volume_id:
            return volume
    raise SystemExit(f"no network volume {volume_id!r} on this account (see --list-volumes)")


def create_volume(key: str, size_gb: int, data_center: str) -> dict:
    return call("POST", "/network-volumes", key,
                {"name": "anofox-tabfm-gpu", "size": size_gb, "dataCenter": data_center})


def delete_volume(key: str, volume_id: str) -> None:
    call("DELETE", f"/network-volumes/{volume_id}", key)


def gpus_by_price(key: str, min_vram: int = MIN_VRAM_GB):
    """Rentable GPU types, cheapest first, as (price, id, vram)."""
    req = urllib.request.Request(
        GRAPHQL,
        method="POST",
        data=json.dumps({"query": "{ gpuTypes { id displayName memoryInGb "
                                  "lowestPrice(input:{gpuCount:1}) { uninterruptablePrice } } }"}).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # Default python-urllib UA gets a 403 from the CDN in front of GraphQL.
            "User-Agent": "anofox-tabfm-gpu-test/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    out = []
    for g in data.get("data", {}).get("gpuTypes") or []:
        price = (g.get("lowestPrice") or {}).get("uninterruptablePrice")
        if price and (g.get("memoryInGb") or 0) >= min_vram:
            out.append((float(price), g["id"], g.get("memoryInGb")))
    return sorted(out)


def ensure_keypair(scratch: Path) -> tuple[Path, str]:
    """An ephemeral keypair for this run; RunPod injects the public half."""
    scratch.mkdir(parents=True, exist_ok=True)
    priv = scratch / "runpod_ed25519"
    if not priv.exists():
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(priv), "-C", "anofox-tabfm-gpu-test"],
            check=True,
        )
    return priv, (priv.with_suffix(".pub")).read_text().strip()


def wait_for_ssh(pod_id: str, key: str, timeout: int):
    """Poll until the pod is RUNNING and its port 22 is mapped."""
    deadline = time.time() + timeout
    last = None
    start = time.time()
    while time.time() < deadline:
        pod = get_pod(key, pod_id)
        # RUNNING can precede SSH by minutes (the container may still be pulling a
        # multi-GB image), so the real signal is the direct endpoint appearing.
        status = pod.get("status")
        endpoint = pod_endpoint(pod)
        if status != last:
            print(f"  [{int(time.time() - start):4d}s] {status} ssh={endpoint}", flush=True)
            last = status
        if status in ("ERROR", "EXITED", "TERMINATED"):
            raise SystemExit(f"pod {pod_id} is {status}; it will not expose SSH")
        if endpoint:
            return endpoint
        time.sleep(5)
    raise SystemExit("timed out waiting for the pod to expose SSH")


def ssh_base(priv: Path, ip: str, port: int):
    return [
        "ssh", "-i", str(priv), "-p", str(port),
        # IdentitiesOnly: without it ssh offers every key in the agent first and
        # sshd drops the connection with "Too many authentication failures",
        # which looks exactly like the pod never booting.
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
        "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=10",
        # keepalives: a long silent step (a build logging to a file) otherwise lets a NAT drop the session
        "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6",
        f"root@{ip}",
    ]


def run_detached(priv: Path, ip: str, port: int, command: str, poll: int = 30) -> int:
    """Runs `command` on the pod detached from this ssh session (nohup; output in /workspace/run.log, exit code in
    /workspace/run.exit) and streams the log by reconnecting every `poll` seconds. An hour-long build no longer dies
    with the connection: a dropped poll is retried, and only the exit file ends the run."""
    launch = (f"cd /workspace && rm -f run.exit && nohup bash -c {shlex.quote('(' + command + '); echo $? > /workspace/run.exit')} "
              "> /workspace/run.log 2>&1 < /dev/null &")
    subprocess.run(ssh_base(priv, ip, port) + [launch], check=True)
    shown = 0
    failures = 0
    while True:
        time.sleep(poll)
        probe = subprocess.run(
            ssh_base(priv, ip, port) + [f"tail -c +{shown + 1} /workspace/run.log; echo; echo __EXIT__$(cat /workspace/run.exit 2>/dev/null)"],
            capture_output=True)
        if probe.returncode != 0:
            failures += 1
            if failures > 20:
                print("lost the pod: 20 polls in a row failed")
                return 255
            continue
        failures = 0
        out = probe.stdout.decode("utf-8", "replace")
        body, _, tail = out.rpartition("__EXIT__")
        body = body[:-1] if body.endswith("\n") else body  # the echo before the marker
        if body:
            sys.stdout.write(body)
            sys.stdout.flush()
            shown += len(body.encode("utf-8"))
        code = tail.strip()
        if code:
            return int(code)


def wait_for_sshd(priv: Path, ip: str, port: int, timeout: int = 900):
    """Wait for sshd. The image pull runs first and can take many minutes."""
    start = time.time()
    deadline = start + timeout
    last_err = ""
    while time.time() < deadline:
        r = subprocess.run(ssh_base(priv, ip, port) + ["true"], capture_output=True)
        if r.returncode == 0:
            print(f"  ssh up after {int(time.time() - start)}s")
            return
        last_err = (r.stderr or b"").decode().strip().splitlines()[-1:] or [""]
        print(f"  [{int(time.time() - start):4d}s] waiting for sshd: {last_err[0][:70]}", flush=True)
        time.sleep(15)
    raise SystemExit(f"sshd never came up ({int(timeout)}s): {last_err}")


# RunPod phrases "sold out" more than one way; none of them mean the request
# was malformed, so all of them should fall through to the next candidate.
CAPACITY_ERRORS = (
    "resources to deploy",
    "no instances currently available",
    "no longer any instances available",
    "not enough free gpus",
    # v2 wording is not documented; these are the plausible phrasings. A miss is not silent: main() prints
    # the last error when every candidate failed, so an unrecognised capacity message is visible, not lost.
    "insufficient capacity",
    "no capacity",
    "out of stock",
)


def _is_capacity_error(exc) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in CAPACITY_ERRORS)


def pod_spec(args, gpu_id, cloud, pub, volume=None):
    """The v2 CreatePodRequest (unevaluatedProperties:false: an unknown key is a 400, not ignored).

    No ``startSsh``: it needs SSH keys registered on the ACCOUNT (PUT /v2/account/ssh-keys), which would change
    the account. The pytorch images start sshd themselves when PUBLIC_KEY is in the environment, which is what
    this script has always relied on, and ``ports`` keeps 22/tcp mapped so ``ssh.direct`` gets assigned."""
    spec = {
        "name": "anofox-tabfm-gpu-test",
        "image": args.image,
        "gpu": {"id": gpu_id, "count": 1},
        "cloud": cloud,
        "disk": args.disk,
        "ports": ["22/tcp"],
        "env": {"PUBLIC_KEY": pub},
    }
    if volume:
        # A persistent network volume mounted AT /workspace (S7 of
        # docs/archive/GPU_HARDENING_PLAN.md): uploads, toolchains, build trees and
        # weights survive across pods, which is what turns a 40-90 min CUDA
        # iteration into minutes. Volumes are datacenter-pinned, so the pod is
        # restricted to the volume's datacenter -- capacity misses there are a
        # real possibility the caller sees as the usual "no capacity" fallback.
        spec["mounts"] = {"network": [{"volumeId": volume["id"], "path": "/workspace"}]}
        spec["dataCenterIds"] = [volume["dataCenter"]]
        spec["cloud"] = "SECURE"  # network volumes live in secure DCs
    return spec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default=DEFAULT_GPU,
                    help="comma-separated GPU ids; empty (default) = cheapest available")
    ap.add_argument("--max-candidates", type=int, default=6,
                    help="how far down the cheapest-first list to try")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--upload", nargs="*", type=Path, default=[], help="files to copy to /workspace")
    ap.add_argument("--command", help="command to run on the pod")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the pod")
    ap.add_argument("--disk", type=int, default=30, help="container disk in GB (two extension builds plus weights need ~80)")
    ap.add_argument("--keep", action="store_true", help="do NOT terminate the pod (debugging; costs money)")
    ap.add_argument("--secure", action="store_true",
                    help="rent only from RunPod's own datacenters, never the COMMUNITY tier. Community hosts are "
                         "other people's machines with whatever nvidia-container-toolkit setup they happen to "
                         "have: seen twice on different hosts, nvidia-smi works and every /dev/nvidia* node is "
                         "present, but cudaGetDeviceCount returns 999 and /dev/nvidia-uvm is owned by "
                         "nobody:nogroup -- the host uid/gid is not mapped into the container's user namespace, "
                         "so CUDA enumerates the device and then cannot initialise it. Costs more per hour.")
    ap.add_argument("--volume-id", help="attach this persistent network volume at /workspace (S7: uploads, "
                                        "toolchains and build trees survive across pods)")
    ap.add_argument("--create-volume", type=int, metavar="GB",
                    help="create a network volume of this size and exit. RECURRING COST until deleted "
                         "(~$0.07/GB/month); delete with --delete-volume")
    ap.add_argument("--volume-dc", default="EU-RO-1", help="datacenter for --create-volume")
    ap.add_argument("--list-volumes", action="store_true", help="list network volumes and exit")
    ap.add_argument("--delete-volume", help="delete a network volume id and exit (ends its recurring cost)")
    ap.add_argument("--list", action="store_true", help="list running pods and exit")
    ap.add_argument("--terminate", help="terminate a pod id and exit")
    args = ap.parse_args()

    key = api_key()

    if args.create_volume:
        volume = create_volume(key, args.create_volume, args.volume_dc)
        print(f"created volume {volume['id']} ({args.create_volume} GB, {args.volume_dc}) — "
              f"RECURRING COST until --delete-volume {volume['id']}")
        return 0
    if args.list_volumes:
        for volume in list_volumes(key):
            print(f"{volume['id']}\t{volume.get('size')}GB\t{volume.get('dataCenter')}\t{volume.get('name')}")
        return 0
    if args.delete_volume:
        delete_volume(key, args.delete_volume)
        print(f"deleted volume {args.delete_volume}")
        return 0
    if args.list:
        for pod in list_pods(key):
            print(f"{pod['id']}\t{pod.get('status')}\t{pod.get('name')}\t{(pod.get('gpu') or {}).get('id', '')}")
        return 0
    if args.terminate:
        terminate_pod(args.terminate, key)
        print(f"terminated {args.terminate}")
        return 0

    uploads = list(args.upload)
    command = args.command
    if not command:
        return ap.error("nothing to do: pass --command")

    scratch = Path(os.environ.get("TMPDIR", "/tmp")) / "anofox-runpod"
    priv, pub = ensure_keypair(scratch)

    if args.gpu:
        candidates = [(None, g.strip(), None) for g in args.gpu.split(",") if g.strip()]
    else:
        candidates = gpus_by_price(key)[: args.max_candidates]
        cheapest = ", ".join(f"{i} ${p}/hr" for p, i, _ in candidates[:4])
        print(f"cheapest rentable GPUs: {cheapest}")

    print(f"image={args.image}")
    volume = None
    if args.volume_id:
        volume = get_volume(key, args.volume_id)
        print(f"volume={volume['id']} ({volume.get('size')}GB, {volume['dataCenter']}) mounted at /workspace")
    pod = None
    last_error = None
    # Cheapest first, each on both clouds: capacity is transient and per-cloud,
    # so falling through the list beats failing on one sold-out card. With a
    # volume attached the cloud is forced to SECURE and the datacenter to the
    # volume's, so the inner loop degenerates to one attempt per GPU type.
    for price, gpu_id, vram in candidates:
        for cloud in (("SECURE",) if (volume or args.secure) else ("COMMUNITY", "SECURE")):
            try:
                pod = create_pod(key, pod_spec(args, gpu_id, cloud, pub, volume))
                tag = f"${price}/hr" if price is not None else "?"
                print(f"  got {gpu_id} ({tag}, {vram}GB) on {cloud}")
                break
            except SystemExit as exc:
                if not _is_capacity_error(exc):
                    raise
                last_error = exc
        if pod is not None:
            break
        print(f"  no capacity: {gpu_id}")
    if pod is None:
        raise SystemExit(f"no capacity for any candidate GPU (last answer: {last_error})")
    pod_id = pod["id"]
    print(f"pod {pod_id} created")

    terminated = False

    def cleanup(*_):
        nonlocal terminated
        if terminated or args.keep:
            if args.keep:
                print(f"\n--keep: pod {pod_id} LEFT RUNNING — terminate with --terminate {pod_id}")
            return
        terminated = True
        print(f"\nterminating pod {pod_id}...")
        try:
            terminate_pod(pod_id, key)
            print("terminated")
        except SystemExit as exc:
            print(f"TERMINATION FAILED ({exc}) — terminate manually: --terminate {pod_id}")

    signal.signal(signal.SIGINT, lambda *a: (cleanup(), sys.exit(130)))
    signal.signal(signal.SIGTERM, lambda *a: (cleanup(), sys.exit(143)))

    try:
        ip, port = wait_for_ssh(pod_id, key, args.timeout)
        print(f"ssh root@{ip}:{port}")
        wait_for_sshd(priv, ip, port)

        subprocess.run(ssh_base(priv, ip, port) + ["mkdir -p /workspace"], check=True)
        for f in uploads:
            print(f"  upload {f.name}")
            subprocess.run(
                ["scp", "-i", str(priv), "-P", str(port), "-o", "IdentitiesOnly=yes",
                 "-o", "StrictHostKeyChecking=no",
                 "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
                 str(f), f"root@{ip}:/workspace/{f.name}"],
                check=True,
            )

        print(f"\n--- nvidia-smi ---")
        subprocess.run(ssh_base(priv, ip, port) + ["nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader"])
        print(f"\n--- running ---\n{command}\n")
        return run_detached(priv, ip, port, command)
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())

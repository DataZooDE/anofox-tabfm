#!/usr/bin/env python3
"""Run a command on a throwaway RunPod GPU pod, then destroy it.

There is no NVIDIA GPU on the dev box, so CUDA-only behaviour (issue #21) can
otherwise only be reasoned about. This rents a pod by the minute, ships local
files to it, runs a command, streams the output back and terminates the pod.

NOT part of CI: it costs money and needs a key CI does not have. Run it by hand
when a change touches a GPU path.

    # default: the ScatterND check from issue #21, on the reporter's A40
    tools/gpu_test/runpod_run.py --check-scatternd

    # anything else
    tools/gpu_test/runpod_run.py --upload resources/graph_tabicl_classification.onnx \\
        --command 'python /workspace/ort_ep_check.py /workspace/*.onnx --provider cuda'

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
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
# CUDA 12.4 + python 3.11; sshd starts when PUBLIC_KEY is set.
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
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
        raise SystemExit(f"RunPod {method} {path} -> {exc.code}: {exc.read().decode()[:400]}")


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
        pod = call("GET", f"/pods/{pod_id}", key) or {}
        # desiredStatus is what we ASKED for and is RUNNING immediately; the
        # container may still be pulling a multi-GB image behind it.
        status = f"{pod.get('desiredStatus')}/{pod.get('lastStatus')}"
        ip = pod.get("publicIp")
        ports = pod.get("portMappings") or {}
        if status != last:
            print(f"  [{int(time.time() - start):4d}s] {status} ip={ip} ports={list(ports)}", flush=True)
            last = status
        if ip and ports.get("22"):
            return ip, int(ports["22"])
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
        f"root@{ip}",
    ]


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
)


def _is_capacity_error(exc) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in CAPACITY_ERRORS)


def ort_preamble(version: str) -> str:
    """Install onnxruntime-gpu and PROVE the CUDA EP actually instantiates.

    Two traps, both of which silently produce CPU results that look like GPU
    results — the one way a GPU harness can lie about the thing it exists to
    test:

    * `nvidia-cudnn-cu12` unpinned drifts ahead of what the ORT build wants
      (1.2x needs cuDNN 9.x); the provider library then fails to dlopen.
    * `get_available_providers()` lists what the BUILD supports, not what can
      be instantiated, so it reports CUDAExecutionProvider even when loading it
      fails. Only building a session and reading back its providers is proof.
    """
    return (
        "set -e; "
        f"pip install -q 'onnxruntime-gpu=={version}' onnx numpy 'nvidia-cudnn-cu12<10' || "
        "{ echo 'PREAMBLE: pip install failed'; exit 90; }; "
        "export LD_LIBRARY_PATH=$(python -c 'import os,nvidia.cudnn.lib as l; "
        "print(os.path.dirname(l.__file__))'):$LD_LIBRARY_PATH; "
        "python - <<'PREAMBLE_EOF' || { echo 'PREAMBLE: CUDA EP did not instantiate'; exit 91; }\n"
        "import sys, numpy as np, onnx, onnxruntime as ort\n"
        "from onnx import helper, TensorProto\n"
        "g = helper.make_graph([helper.make_node('Identity', ['i'], ['o'])], 'probe',\n"
        "    [helper.make_tensor_value_info('i', TensorProto.FLOAT, [1])],\n"
        "    [helper.make_tensor_value_info('o', TensorProto.FLOAT, [1])])\n"
        "m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 18)]); m.ir_version = 10\n"
        "onnx.save(m, '/tmp/probe.onnx')\n"
        "so = ort.SessionOptions(); so.log_severity_level = 3\n"
        "s = ort.InferenceSession('/tmp/probe.onnx', so, providers=['CUDAExecutionProvider'])\n"
        "print('ORT', ort.__version__, 'session providers:', s.get_providers())\n"
        "sys.exit(0 if 'CUDAExecutionProvider' in s.get_providers() else 91)\n"
        "PREAMBLE_EOF\n"
        "set +e; "
    )


def _pod_spec(args, gpu_ids, cloud, pub, volume=None):
    spec = {
        "name": "anofox-tabfm-gpu-test",
        "imageName": args.image,
        "gpuTypeIds": gpu_ids,
        "gpuCount": 1,
        "cloudType": cloud,
        "containerDiskInGb": 30,
        "volumeInGb": 0,
        "ports": ["22/tcp"],
        "supportPublicIp": True,
        "env": {"PUBLIC_KEY": pub},
    }
    if volume:
        # A persistent network volume mounted AT /workspace (S7 of
        # docs/GPU_HARDENING_PLAN.md): uploads, toolchains, build trees and
        # weights survive across pods, which is what turns a 40-90 min CUDA
        # iteration into minutes. Volumes are datacenter-pinned, so the pod is
        # restricted to the volume's datacenter -- capacity misses there are a
        # real possibility the caller sees as the usual "no capacity" fallback.
        spec["networkVolumeId"] = volume["id"]
        spec["volumeMountPath"] = "/workspace"
        spec["dataCenterIds"] = [volume["dataCenterId"]]
        spec["cloudType"] = "SECURE"  # network volumes live in secure DCs
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
    ap.add_argument("--with-ort", action="store_true",
                    help="prepend the onnxruntime-gpu + cuDNN setup to --command "
                         "(quoting this by hand through ssh is a trap: a lost "
                         "LD_LIBRARY_PATH silently downgrades the run to CPU)")
    ap.add_argument("--check-scatternd", action="store_true",
                    help="preset: issue #21 — the affected graphs on the CUDA EP")
    # main carries the pins as of #29, so it is no longer a pre-fix baseline:
    # default to the last release that predates them, or the check compares a
    # pinned graph against itself and "passes" without testing anything.
    ap.add_argument("--baseline-ref", default="v2026.08.13",
                    help="git ref supplying the PRE-pin graph (must predate #23)")
    ap.add_argument("--ort-version", default="1.23.2",
                    help="comma-separated onnxruntime-gpu versions to sweep")
    ap.add_argument("--timeout", type=int, default=900, help="seconds to wait for the pod")
    ap.add_argument("--keep", action="store_true", help="do NOT terminate the pod (debugging; costs money)")
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
        volume = call("POST", "/networkvolumes", key,
                      {"name": "anofox-tabfm-gpu", "size": args.create_volume, "dataCenterId": args.volume_dc})
        print(f"created volume {volume['id']} ({args.create_volume} GB, {args.volume_dc}) — "
              f"RECURRING COST until --delete-volume {volume['id']}")
        return 0
    if args.list_volumes:
        for volume in call("GET", "/networkvolumes", key) or []:
            print(f"{volume['id']}\t{volume.get('size')}GB\t{volume.get('dataCenterId')}\t{volume.get('name')}")
        return 0
    if args.delete_volume:
        call("DELETE", f"/networkvolumes/{args.delete_volume}", key)
        print(f"deleted volume {args.delete_volume}")
        return 0
    if args.list:
        for pod in call("GET", "/pods", key) or []:
            print(f"{pod['id']}\t{pod.get('desiredStatus')}\t{pod.get('name')}\t{pod.get('machine', {}).get('gpuTypeId', '')}")
        return 0
    if args.terminate:
        call("DELETE", f"/pods/{args.terminate}", key)
        print(f"terminated {args.terminate}")
        return 0

    uploads = list(args.upload)
    command = args.command
    repo = Path(__file__).resolve().parents[2]
    if args.check_scatternd:
        # Ship BOTH sides: the fix is only demonstrated if the pre-fix graph
        # reproduces the failure on the same pod, in the same run.
        scratch_graphs = Path(os.environ.get("TMPDIR", "/tmp")) / "anofox-runpod"
        scratch_graphs.mkdir(parents=True, exist_ok=True)
        # onnxruntime-gpu needs cuDNN 9 on the loader path; the pytorch images
        # ship cuDNN inside torch rather than somewhere ORT's dlopen finds it.
        # Ship BOTH graphs: a fix is only demonstrated if the pre-fix graph
        # reproduces the failure on the same pod, in the same run.
        for name, ref in (("BASELINE", args.baseline_ref), ("FIXED", "HEAD")):
            for task in ("classification", "regression"):
                dest = scratch_graphs / f"tabicl_{task}_{name}.onnx"
                dest.write_bytes(subprocess.run(
                    ["git", "-C", str(repo), "show", f"{ref}:resources/graph_tabicl_{task}.onnx"],
                    check=True, capture_output=True).stdout)
                uploads.append(dest)
        uploads.append(repo / "tools/gpu_test/ort_ep_check.py")
        command = (
            ort_preamble(args.ort_version.split(",")[0]) +
            "cd /workspace && for task in classification regression; do "
            "echo \"### $task\"; "
            "python ort_ep_check.py tabicl_${task}_BASELINE.onnx tabicl_${task}_FIXED.onnx "
            "--provider cuda --shapes 70x3x60,128x8x100,20x5x9 2>/dev/null | grep -E 'ok |FAIL|failures'; "
            "echo \"### $task CPU-vs-CUDA on the FIXED graph\"; "
            "python ort_ep_check.py tabicl_${task}_FIXED.onnx --provider cuda --shapes 70x3x60 2>/dev/null | grep -E 'ok |FAIL'; "
            "python ort_ep_check.py tabicl_${task}_FIXED.onnx --provider cpu --shapes 70x3x60 2>/dev/null | grep -E 'ok |FAIL'; "
            "done"
        )
    if args.with_ort and command:
        uploads.append(repo / "tools/gpu_test/ort_ep_check.py")
        command = ort_preamble(args.ort_version.split(",")[0]) + command
    if not command:
        return ap.error("nothing to do: pass --command or --check-scatternd")

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
        volume = call("GET", f"/networkvolumes/{args.volume_id}", key)
        print(f"volume={volume['id']} ({volume.get('size')}GB, {volume['dataCenterId']}) mounted at /workspace")
    pod = None
    # Cheapest first, each on both clouds: capacity is transient and per-cloud,
    # so falling through the list beats failing on one sold-out card. With a
    # volume attached the cloud is forced to SECURE and the datacenter to the
    # volume's, so the inner loop degenerates to one attempt per GPU type.
    for price, gpu_id, vram in candidates:
        for cloud in (("SECURE",) if volume else ("COMMUNITY", "SECURE")):
            try:
                pod = call("POST", "/pods", key, _pod_spec(args, [gpu_id], cloud, pub, volume))
                tag = f"${price}/hr" if price is not None else "?"
                print(f"  got {gpu_id} ({tag}, {vram}GB) on {cloud}")
                break
            except SystemExit as exc:
                if not _is_capacity_error(exc):
                    raise
        if pod is not None:
            break
        print(f"  no capacity: {gpu_id}")
    if pod is None:
        raise SystemExit("no capacity for any candidate GPU")
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
            call("DELETE", f"/pods/{pod_id}", key)
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
        result = subprocess.run(ssh_base(priv, ip, port) + [command])
        return result.returncode
    finally:
        cleanup()


if __name__ == "__main__":
    sys.exit(main())

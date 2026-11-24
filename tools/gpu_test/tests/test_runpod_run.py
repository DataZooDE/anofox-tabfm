"""runpod_run.py against RunPod's REST API v2 (https://api.runpod.io/v2).

The script used to talk to https://rest.runpod.io/v1, a host that no longer resolves. Every payload below is
the shape RunPod's v2 reference documents (and, for the two list endpoints, what the live API returned for
this account), so a regression back to v1 names fails here instead of on a rented pod.

Stdlib only, no network, no money:  python3 -m unittest discover -s tools/gpu_test/tests -v
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "runpod_run.py"
spec = importlib.util.spec_from_file_location("runpod_run", SCRIPT)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

# RunPod's documented v2 `Pod` example (GET /v2/pods/{id}), verbatim apart from the id.
POD_RUNNING = {
    "id": "7h9k2m4n6p", "name": "pytorch-training", "image": "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404",
    "args": "", "disk": 50, "ports": ["8888/http", "22/tcp"], "env": {"MODEL_NAME": "llama-3"},
    "registry": None, "status": "RUNNING", "actions": ["stop", "restart", "terminate"],
    "mounts": {"persistent": {"size": 20, "path": "/workspace"}},
    "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1, "vcpuCount": 16, "memory": 64},
    "cloud": "SECURE", "dataCenterId": "US-KS-2", "cudaVersion": "12.8",
    "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22, "username": "7h9k2m4n6p-64411eb2",
                      "command": "ssh 7h9k2m4n6p-64411eb2@ssh.runpod.io"},
            "direct": {"host": "195.26.233.3", "port": 34446, "username": "root",
                       "command": "ssh root@195.26.233.3 -p 34446"}},
    "template": "9x4m2p7v", "cost": 0.44, "locked": False, "globalNetworking": {"enabled": False},
    "runtime": {"uptime": 3600, "ports": [{"private": 22, "public": 34446, "type": "tcp", "ip": "195.26.233.3"}]},
    "createdAt": "2026-06-01T12:00:00Z", "startedAt": "2026-06-01T12:02:00Z",
}
POD_PROVISIONING = dict(POD_RUNNING, status="PROVISIONING", ssh={"proxy": None, "direct": None}, runtime=None,
                        startedAt=None)
# What the live API returned for this account.
VOLUME = {"dataCenter": "EUR-IS-1", "id": "4gmcwywto3", "name": "anofox-tabfm-gpu", "size": 40, "tags": [],
          "type": "STANDARD"}


class FakeCall:
    """Stands in for rp.call: records (method, path, body) and answers from a {(method, path): reply} table."""

    def __init__(self, replies):
        self.replies, self.seen = replies, []

    def __call__(self, method, path, key, body=None, timeout=60):
        self.seen.append((method, path, body))
        reply = self.replies[(method, path)]
        if isinstance(reply, list):          # a sequence of answers, one per call
            return reply.pop(0)
        return reply


class Transport(unittest.TestCase):
    def test_the_api_base_is_v2_on_api_runpod_io(self):
        self.assertEqual(rp.API, "https://api.runpod.io/v2")

    def _urlopen(self, status=200, body=b"{}"):
        if status >= 400:
            err = urllib.error.HTTPError("u", status, "x", {}, io.BytesIO(body))
            return mock.patch.object(rp.urllib.request, "urlopen", side_effect=err)
        resp = mock.MagicMock()
        resp.__enter__.return_value.read.return_value = body
        return mock.patch.object(rp.urllib.request, "urlopen", return_value=resp)

    def test_call_builds_the_v2_url_and_parses_json(self):
        with self._urlopen(body=b'{"pods": []}') as opener:
            self.assertEqual(rp.call("GET", "/pods", "k"), {"pods": []})
        req = opener.call_args.args[0]
        self.assertEqual(req.full_url, "https://api.runpod.io/v2/pods")
        self.assertEqual(req.get_header("Authorization"), "Bearer k")

    def test_a_204_is_none(self):
        with self._urlopen(body=b""):
            self.assertIsNone(rp.call("DELETE", "/pods/x", "k"))

    def test_an_error_names_the_problem_detail(self):
        body = json.dumps({"title": "Conflict", "status": 409, "detail": "retry in 4 minutes"}).encode()
        with self._urlopen(409, body):
            with self.assertRaises(SystemExit) as cm:
                rp.call("DELETE", "/network-volumes/v", "k")
        self.assertIn("409", str(cm.exception))
        self.assertIn("retry in 4 minutes", str(cm.exception))


class Pods(unittest.TestCase):
    def test_list_pods_unwraps_the_pods_key(self):
        fake = FakeCall({("GET", "/pods"): {"pagination": {"hasNextPage": False}, "pods": [POD_RUNNING]}})
        with mock.patch.object(rp, "call", fake):
            self.assertEqual(rp.list_pods("k"), [POD_RUNNING])

    def test_the_endpoint_is_the_direct_ssh_host_and_port(self):
        self.assertEqual(rp.pod_endpoint(POD_RUNNING), ("195.26.233.3", 34446))

    def test_no_endpoint_until_the_direct_ssh_is_assigned(self):
        self.assertIsNone(rp.pod_endpoint(POD_PROVISIONING))
        self.assertIsNone(rp.pod_endpoint({"ssh": {"proxy": None, "direct": None}, "runtime": {"ports": []}}))

    def test_the_endpoint_falls_back_to_the_runtime_port_map(self):
        pod = dict(POD_RUNNING, ssh={"proxy": None, "direct": None})
        self.assertEqual(rp.pod_endpoint(pod), ("195.26.233.3", 34446))

    def test_wait_for_ssh_polls_until_the_endpoint_exists(self):
        fake = FakeCall({("GET", "/pods/7h9k2m4n6p"): [POD_PROVISIONING, POD_PROVISIONING, POD_RUNNING]})
        with mock.patch.object(rp, "call", fake), mock.patch.object(rp.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rp.wait_for_ssh("7h9k2m4n6p", "k", 60), ("195.26.233.3", 34446))
        self.assertEqual(len(fake.seen), 3)

    def test_wait_for_ssh_gives_up_on_a_failed_pod(self):
        fake = FakeCall({("GET", "/pods/p"): dict(POD_PROVISIONING, status="ERROR")})
        with mock.patch.object(rp, "call", fake), mock.patch.object(rp.time, "sleep"), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                rp.wait_for_ssh("p", "k", 60)
        self.assertIn("ERROR", str(cm.exception))

    def test_terminate_is_a_delete_on_the_pod(self):
        fake = FakeCall({("DELETE", "/pods/p"): None})
        with mock.patch.object(rp, "call", fake):
            rp.terminate_pod("p", "k")
        self.assertEqual(fake.seen, [("DELETE", "/pods/p", None)])


class PodSpec(unittest.TestCase):
    ARGS = type("A", (), {"image": "img:1", "disk": 30})()

    # the v2 CreatePodRequest has unevaluatedProperties:false: every key must be one it documents
    V2_KEYS = {"name", "image", "cloud", "gpu", "cpu", "dataCenterIds", "globalNetworking", "mounts",
               "startJupyter", "startSsh", "templateId", "queue", "args", "cmd", "entrypoint", "disk", "env",
               "ports", "registry"}

    def test_a_plain_gpu_pod(self):
        body = rp.pod_spec(self.ARGS, "NVIDIA GeForce RTX 4090", "SECURE", "ssh-ed25519 AAA k")
        self.assertEqual(body, {
            "name": "anofox-tabfm-gpu-test", "image": "img:1", "gpu": {"id": "NVIDIA GeForce RTX 4090", "count": 1},
            "cloud": "SECURE", "disk": 30, "ports": ["22/tcp"], "env": {"PUBLIC_KEY": "ssh-ed25519 AAA k"},
        })
        self.assertLessEqual(set(body), self.V2_KEYS)

    def test_the_v1_names_are_gone(self):
        body = rp.pod_spec(self.ARGS, "g", "COMMUNITY", "k")
        for old in ("imageName", "gpuTypeIds", "gpuCount", "cloudType", "containerDiskInGb", "volumeInGb",
                    "supportPublicIp", "networkVolumeId", "volumeMountPath"):
            self.assertNotIn(old, body)

    def test_a_network_volume_is_mounted_at_workspace_in_its_datacenter(self):
        body = rp.pod_spec(self.ARGS, "g", "COMMUNITY", "k", VOLUME)
        self.assertEqual(body["mounts"], {"network": [{"volumeId": "4gmcwywto3", "path": "/workspace"}]})
        self.assertEqual(body["dataCenterIds"], ["EUR-IS-1"])
        self.assertEqual(body["cloud"], "SECURE")            # network volumes live in secure datacenters
        self.assertLessEqual(set(body), self.V2_KEYS)

    def test_capacity_errors_fall_through_to_the_next_candidate(self):
        for text in ("There are no longer any instances available with the requested specifications",
                     "no instances currently available", "not enough free GPUs", "insufficient capacity"):
            self.assertTrue(rp._is_capacity_error(SystemExit(text)), text)
        self.assertFalse(rp._is_capacity_error(SystemExit("RunPod POST /pods -> 401: bad token")))


class Volumes(unittest.TestCase):
    def test_list_volumes_unwraps_network_volumes(self):
        fake = FakeCall({("GET", "/network-volumes"): {"networkVolumes": [VOLUME], "pagination": {}}})
        with mock.patch.object(rp, "call", fake):
            self.assertEqual(rp.list_volumes("k"), [VOLUME])

    def test_get_volume_picks_one_from_the_list(self):
        fake = FakeCall({("GET", "/network-volumes"): {"networkVolumes": [VOLUME]}})
        with mock.patch.object(rp, "call", fake):
            self.assertEqual(rp.get_volume("k", "4gmcwywto3"), VOLUME)
            with self.assertRaises(SystemExit) as cm:
                rp.get_volume("k", "nope")
        self.assertIn("nope", str(cm.exception))

    def test_create_posts_name_size_and_datacenter(self):
        fake = FakeCall({("POST", "/network-volumes"): dict(VOLUME, id="new")})
        with mock.patch.object(rp, "call", fake):
            made = rp.create_volume("k", 40, "EUR-IS-1")
        self.assertEqual(fake.seen, [("POST", "/network-volumes",
                                      {"name": "anofox-tabfm-gpu", "size": 40, "dataCenter": "EUR-IS-1"})])
        self.assertEqual(made["id"], "new")

    def test_delete_is_a_delete_on_the_volume(self):
        fake = FakeCall({("DELETE", "/network-volumes/v1"): None})
        with mock.patch.object(rp, "call", fake):
            rp.delete_volume("k", "v1")
        self.assertEqual(fake.seen, [("DELETE", "/network-volumes/v1", None)])


class Cli(unittest.TestCase):
    def _run(self, argv, replies):
        fake = FakeCall(replies)
        out = io.StringIO()
        with mock.patch.object(rp, "call", fake), mock.patch.object(rp, "api_key", return_value="k"), \
                mock.patch.object(sys, "argv", ["runpod_run.py"] + argv), contextlib.redirect_stdout(out):
            code = rp.main()
        return code, out.getvalue(), fake.seen

    def test_list_prints_id_status_name_and_gpu(self):
        code, out, _ = self._run(["--list"], {("GET", "/pods"): {"pods": [POD_RUNNING]}})
        self.assertEqual(code, 0)
        self.assertEqual(out.strip().split("\t"),
                         ["7h9k2m4n6p", "RUNNING", "pytorch-training", "NVIDIA GeForce RTX 4090"])

    def test_list_volumes_prints_the_datacenter(self):
        code, out, _ = self._run(["--list-volumes"], {("GET", "/network-volumes"): {"networkVolumes": [VOLUME]}})
        self.assertEqual(code, 0)
        self.assertEqual(out.strip().split("\t"), ["4gmcwywto3", "40GB", "EUR-IS-1", "anofox-tabfm-gpu"])

    def test_terminate_and_delete_volume(self):
        code, out, seen = self._run(["--terminate", "p1"], {("DELETE", "/pods/p1"): None})
        self.assertEqual((code, seen), (0, [("DELETE", "/pods/p1", None)]))
        code, out, seen = self._run(["--delete-volume", "v1"], {("DELETE", "/network-volumes/v1"): None})
        self.assertEqual((code, seen), (0, [("DELETE", "/network-volumes/v1", None)]))

    def test_create_volume_reports_the_recurring_cost(self):
        code, out, seen = self._run(["--create-volume", "40", "--volume-dc", "EUR-IS-1"],
                                    {("POST", "/network-volumes"): dict(VOLUME, id="new")})
        self.assertEqual(code, 0)
        self.assertIn("new", out)
        self.assertIn("RECURRING", out)


if __name__ == "__main__":
    unittest.main()

# /// script
# requires-python = ">=3.10"
# dependencies = ["onnx>=1.16,<2", "numpy>=1.26", "onnxruntime>=1.18"]
# ///
"""Generator for the [ort_engine] C++ test fixture (WS-C).

Builds a tiny weight-free 2-layer MLP graph whose four weight tensors are
external-data stubs pointing at a "weights.bin" that is NEVER created
(S02 license-critical mechanism: the graph carries no weight bytes; the
C++ engine injects them via AddExternalInitializers).

The graph consumes ALL five TabFM engine inputs so the C++ Run() signature
(x, y, train_size, cat_mask, d) is exercised end to end:

    h      = Relu(x @ W1 + b1)                    # [1,T,K]
    base   = h @ W2 + b2                          # [1,T,C]
    s      = 0.001*f32(train_size) + 0.0001*f32(d)
             + 0.01*ReduceSum(f32(cat_mask)) + 0.1*ReduceMean(y)
    logits = base + s                             # [1,T,C] (broadcast)

Outputs (committed):
    mlp_graph.onnx     weight-free graph, external stubs -> weights.bin (absent)
    mlp_weights.f32    raw little-endian f32: W1[H,K] b1[K] W2[K,C] b2[C]
    mlp_expected.json  inputs + numpy-float32 reference logits + shapes

Deterministic: numpy RandomState(1337). Re-running must reproduce the same
bytes (verified at the end).

Run:  uv run gen_mlp_fixture.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import set_external_data

HERE = Path(__file__).parent
H, K, C, T = 4, 8, 3, 6  # T is dynamic in the graph; 6 is the golden run


def build_weights():
	rng = np.random.RandomState(1337)
	W1 = rng.uniform(-1, 1, (H, K)).astype(np.float32)
	b1 = rng.uniform(-1, 1, (K,)).astype(np.float32)
	W2 = rng.uniform(-1, 1, (K, C)).astype(np.float32)
	b2 = rng.uniform(-1, 1, (C,)).astype(np.float32)
	return W1, b1, W2, b2


def make_stub(name, arr, offset):
	"""Weight tensor as an external-data stub: dims/dtype kept, bytes stripped,
	data_location=EXTERNAL pointing at a file that will never exist."""
	t = helper.make_tensor(name, TensorProto.FLOAT, arr.shape, arr.tobytes(), raw=True)
	set_external_data(t, location="weights.bin", offset=offset, length=arr.nbytes)
	t.ClearField("raw_data")  # S02: strip bytes AFTER set_external_data
	return t


def build_graph(W1, b1, W2, b2):
	f32, i64, b8 = TensorProto.FLOAT, TensorProto.INT64, TensorProto.BOOL
	inputs = [
		helper.make_tensor_value_info("x", f32, [1, "T", H]),
		helper.make_tensor_value_info("y", f32, [1, "T"]),
		helper.make_tensor_value_info("train_size", i64, [1]),
		helper.make_tensor_value_info("cat_mask", b8, [1, H]),
		helper.make_tensor_value_info("d", i64, [1]),
	]
	outputs = [helper.make_tensor_value_info("logits", f32, [1, "T", C])]

	offset = 0
	stubs = []
	for name, arr in (("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)):
		stubs.append(make_stub(name, arr, offset))
		offset += arr.nbytes

	consts = [
		helper.make_tensor("c_ts", f32, [], np.array([0.001], np.float32).tobytes(), raw=True),
		helper.make_tensor("c_d", f32, [], np.array([0.0001], np.float32).tobytes(), raw=True),
		helper.make_tensor("c_cm", f32, [], np.array([0.01], np.float32).tobytes(), raw=True),
		helper.make_tensor("c_y", f32, [], np.array([0.1], np.float32).tobytes(), raw=True),
	]

	nodes = [
		helper.make_node("MatMul", ["x", "W1"], ["mm1"]),
		helper.make_node("Add", ["mm1", "b1"], ["pre1"]),
		helper.make_node("Relu", ["pre1"], ["h"]),
		helper.make_node("MatMul", ["h", "W2"], ["mm2"]),
		helper.make_node("Add", ["mm2", "b2"], ["base"]),
		# consume train_size, d, cat_mask, y
		helper.make_node("Cast", ["train_size"], ["ts_f"], to=f32),
		helper.make_node("Mul", ["ts_f", "c_ts"], ["s_ts"]),
		helper.make_node("Cast", ["d"], ["d_f"], to=f32),
		helper.make_node("Mul", ["d_f", "c_d"], ["s_d"]),
		helper.make_node("Cast", ["cat_mask"], ["cm_f"], to=f32),
		helper.make_node("ReduceSum", ["cm_f"], ["cm_sum"], keepdims=0),
		helper.make_node("Mul", ["cm_sum", "c_cm"], ["s_cm"]),
		helper.make_node("ReduceMean", ["y"], ["y_mean"], keepdims=0),
		helper.make_node("Mul", ["y_mean", "c_y"], ["s_y"]),
		helper.make_node("Add", ["s_ts", "s_d"], ["s0"]),
		helper.make_node("Add", ["s0", "s_cm"], ["s1"]),
		helper.make_node("Add", ["s1", "s_y"], ["s"]),
		helper.make_node("Add", ["base", "s"], ["logits"]),
	]

	graph = helper.make_graph(nodes, "tabfm_wsc_mlp_fixture", inputs, outputs,
	                          initializer=stubs + consts)
	model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
	model.ir_version = 9
	# NOTE: onnx.checker rejects weight-free models (stats the external file,
	# S02 friction #4). ORT session creation is the real gate.
	return model


def reference(W1, b1, W2, b2, x, y, train_size, cat_mask, d):
	h = np.maximum(np.float32(0), x @ W1 + b1)
	base = h @ W2 + b2
	s = (np.float32(0.001) * np.float32(train_size)
	     + np.float32(0.0001) * np.float32(d)
	     + np.float32(0.01) * cat_mask.astype(np.float32).sum(dtype=np.float32)
	     + np.float32(0.1) * y.astype(np.float32).mean(dtype=np.float32))
	return (base + s).astype(np.float32)


def main():
	W1, b1, W2, b2 = build_weights()
	model = build_graph(W1, b1, W2, b2)

	rng = np.random.RandomState(4242)
	x = rng.uniform(-2, 2, (1, T, H)).astype(np.float32)
	y = rng.uniform(0, 3, (1, T)).astype(np.float32)
	train_size = np.array([4], np.int64)
	cat_mask = np.array([[True, False, True, False]])
	d = np.array([4], np.int64)

	logits = reference(W1, b1, W2, b2, x, y, train_size[0], cat_mask, d[0])

	# Cross-check with Python ORT via in-memory initializer injection
	import onnxruntime as ort
	patched = onnx.ModelProto()
	patched.CopyFrom(model)
	for init in patched.graph.initializer:
		for name, arr in (("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)):
			if init.name == name:
				init.CopyFrom(onnx.numpy_helper.from_array(arr, name))
	sess = ort.InferenceSession(patched.SerializeToString(),
	                            providers=["CPUExecutionProvider"])
	got = sess.run(None, {"x": x, "y": y, "train_size": train_size,
	                      "cat_mask": cat_mask, "d": d})[0]
	np.testing.assert_allclose(got, logits, rtol=1e-5, atol=1e-6)

	(HERE / "mlp_graph.onnx").write_bytes(model.SerializeToString())
	blob = W1.tobytes() + b1.tobytes() + W2.tobytes() + b2.tobytes()
	(HERE / "mlp_weights.f32").write_bytes(blob)

	spec = {
		"description": "WS-C [ort_engine] fixture; regenerate with: uv run gen_mlp_fixture.py",
		"dims": {"T": T, "H": H, "K": K, "C": C},
		"weights_file": "mlp_weights.f32",
		"tensors": [
			{"name": "W1", "shape": [H, K], "offset": 0, "nbytes": W1.nbytes},
			{"name": "b1", "shape": [K], "offset": W1.nbytes, "nbytes": b1.nbytes},
			{"name": "W2", "shape": [K, C], "offset": W1.nbytes + b1.nbytes, "nbytes": W2.nbytes},
			{"name": "b2", "shape": [C], "offset": W1.nbytes + b1.nbytes + W2.nbytes, "nbytes": b2.nbytes},
		],
		"inputs": {
			"x": x.flatten().tolist(),
			"y": y.flatten().tolist(),
			"train_size": int(train_size[0]),
			"cat_mask": [bool(v) for v in cat_mask.flatten()],
			"d": int(d[0]),
		},
		"logits": logits.flatten().tolist(),
		"logits_shape": [1, T, C],
		"rtol": 1e-5,
	}
	(HERE / "mlp_expected.json").write_text(json.dumps(spec, indent=1) + "\n")

	# determinism proof
	h1 = hashlib.sha256((HERE / "mlp_graph.onnx").read_bytes()).hexdigest()
	m2 = build_graph(*build_weights())
	h2 = hashlib.sha256(m2.SerializeToString()).hexdigest()
	assert h1 == h2, "graph bytes are not deterministic"
	print("graph.onnx sha256:", h1)
	print("weights.f32 sha256:", hashlib.sha256(blob).hexdigest())
	print("OK: python-ORT parity max abs diff:", float(np.abs(got - logits).max()))


if __name__ == "__main__":
	main()

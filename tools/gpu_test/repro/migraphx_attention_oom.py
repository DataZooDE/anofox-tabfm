"""Minimal MIGraphX reproducer: an attention whose score tensor does not fit in VRAM aborts the process.

    q, k, v [N, 6, T, 16] -> softmax(q @ k^T) @ v        score tensor [N, 6, T, T] float32

This is LimiX-2M's per-token sample attention at the shape bucket that aborted (N = 65 tokens, T = 4096 rows:
24.4 GiB of scores) without anything else in the graph. On a 16 GiB gfx1201 with MIGraphX 2.15:

    python migraphx_attention_oom.py 65 4096 attn_65x4096.onnx
    migraphx-driver compile attn_65x4096.onnx --gpu --enable-offload-copy
      -> hip.cpp:155 write_to_gpu: Assertion `host or is_device_ptr(result.get())' failed  (rc 134, 330 s)
    python migraphx_attention_oom.py 33 2048 attn_33x2048.onnx      (3.3 GiB of scores)
      -> compiles (rc 0, 66 s)

The abort is in gpu::compile_ops benchmarking (compile_plan::benchmark -> time_program -> to_gpu): hipMalloc
of a kernel argument fails, allocate_gpu silently retries with hipHostMalloc, and write_to_gpu then asserts that
the buffer is device memory. Builds with assertions enabled (the Arch Linux package) abort; an NDEBUG build
would continue on pinned host memory. Needs the `onnx` package. See docs/ROCM_SINGLE_EVAL_POS.md.
"""

import sys

import onnx
from onnx import TensorProto
from onnx import helper as h


def main() -> None:
    n, t, out_path = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    shape = [n, 6, t, 16]
    ins = [h.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name in ("q", "k", "v")]
    out = h.make_tensor_value_info("o", TensorProto.FLOAT, shape)
    nodes = [
        h.make_node("Transpose", ["k"], ["kt"], perm=[0, 1, 3, 2]),
        h.make_node("MatMul", ["q", "kt"], ["s"]),
        h.make_node("Softmax", ["s"], ["p"], axis=-1),
        h.make_node("MatMul", ["p", "v"], ["o"]),
    ]
    graph = h.make_graph(nodes, "attention_oom", ins, [out])
    onnx.save(h.make_model(graph, opset_imports=[h.make_opsetid("", 17)]), out_path)


if __name__ == "__main__":
    main()

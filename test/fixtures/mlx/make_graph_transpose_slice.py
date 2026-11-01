# Regenerates graph_transpose_slice.onnx: logits = Slice(Transpose(x)), the tail of
# the tabdpt classification graphs. On MLX both ops return strided views.
# usage: python make_graph_transpose_slice.py graph_transpose_slice.onnx
import sys, numpy as np, onnx
from onnx import helper as h, TensorProto as T
c = lambda name, v: h.make_node('Constant', [], [name], value=onnx.numpy_helper.from_array(np.array(v, np.int64)))
g = h.make_graph(
    [h.make_node('Transpose', ['x'], ['xt'], perm=[0, 2, 1]),
     c('st', [1]), c('en', [1 << 62]), c('ax', [2]), c('sp', [1]),
     h.make_node('Slice', ['xt', 'st', 'en', 'ax', 'sp'], ['logits'])],
    'transpose_slice',
    [h.make_tensor_value_info('x', T.FLOAT, [1, 'rows', 'features']),
     h.make_tensor_value_info('y', T.FLOAT, [1, 'train'])],
    [h.make_tensor_value_info('logits', T.FLOAT, [1, 'features', None])])
m = h.make_model(g, opset_imports=[h.make_opsetid('', 18)], producer_name='anofox-tabfm fixture')
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])

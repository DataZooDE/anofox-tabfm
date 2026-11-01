# Regenerates graph_int_div.onnx: logits = x[:, :2*((H+1)/2), :(-7/2)+5], with every Div on int64.
# LimiX-2M computes a Slice end exactly this way (2 * floordiv(H + 1, 2)). ONNX integer Div
# truncates toward zero (ORT: 2*((4+1)/2) = 4, -7/2 = -3), so for H = 4 the output is [1, 4, 2].
# A true division gives [1, 5, 1]; a floor division gives [1, 4, 1].
# usage: python make_graph_int_div.py graph_int_div.onnx
import sys, numpy as np, onnx
from onnx import helper as h, TensorProto as T
c = lambda name, v: h.make_node('Constant', [], [name], value=onnx.numpy_helper.from_array(np.array(v, np.int64)))
g = h.make_graph(
    [h.make_node('Shape', ['x'], ['hdim'], start=2, end=3),
     c('one', [1]), c('two', [2]), c('m7', [-7]), c('five', [5]), c('zero', [0]), c('ax1', [1]), c('ax2', [2]),
     h.make_node('Add', ['hdim', 'one'], ['h1']),
     h.make_node('Div', ['h1', 'two'], ['half']),
     h.make_node('Mul', ['two', 'half'], ['rows_end']),
     h.make_node('Div', ['m7', 'two'], ['neg_half']),
     h.make_node('Add', ['neg_half', 'five'], ['cols_end']),
     h.make_node('Slice', ['x', 'zero', 'rows_end', 'ax1'], ['xr']),
     h.make_node('Slice', ['xr', 'zero', 'cols_end', 'ax2'], ['logits'])],
    'int_div',
    [h.make_tensor_value_info('x', T.FLOAT, [1, 'rows', 'features']),
     h.make_tensor_value_info('y', T.FLOAT, [1, 'train'])],
    [h.make_tensor_value_info('logits', T.FLOAT, [1, None, None])])
m = h.make_model(g, opset_imports=[h.make_opsetid('', 18)], producer_name='anofox-tabfm fixture')
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])

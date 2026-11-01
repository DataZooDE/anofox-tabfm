# Regenerates graph_double.onnx: logits = Pow(x - Cast<float>(centers), scale), where `centers`
# is a float64 INITIALIZER and `scale` a float64 rank-0 CONSTANT. LimiX-2M's numeric encoder
# stores its RBF centers (and a Pow exponent) in float64 exactly like this. MLX has no float64,
# so the bytes must be narrowed on load, not reinterpreted as float32.
# usage: python make_graph_double.py graph_double.onnx
import sys, numpy as np, onnx
from onnx import helper as h, TensorProto as T, numpy_helper as nh
centers = nh.from_array(np.arange(4, dtype=np.float64) / 63.0, 'centers')
g = h.make_graph(
    [h.make_node('Constant', [], ['scale'], value=nh.from_array(np.array(2.0, np.float64))),
     h.make_node('Cast', ['centers'], ['centers_f'], to=T.FLOAT),
     h.make_node('Sub', ['x', 'centers_f'], ['d']),
     h.make_node('Pow', ['d', 'scale'], ['logits'])],
    'double_tensors',
    [h.make_tensor_value_info('x', T.FLOAT, [1, 'rows', 4]),
     h.make_tensor_value_info('y', T.FLOAT, [1, 'train'])],
    [h.make_tensor_value_info('logits', T.FLOAT, [1, 'rows', 4])],
    initializer=[centers])
m = h.make_model(g, opset_imports=[h.make_opsetid('', 18)], producer_name='anofox-tabfm fixture')
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, sys.argv[1])

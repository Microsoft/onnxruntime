import onnx
from onnx import helper
from onnx import TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np
import random

batch = 3
hidden_size = 4
attention_head = 2
hidden_per_attention = 2

relative_attention_num_buckets=32
input_len=8
output_len=8

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [batch, input_len, hidden_size])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [batch, output_len, hidden_size])

q_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
q_weight_initializer = numpy_helper.from_array(q_weight_np_vals, 'encoder.layer.0.SelfAttention.q.weight')

k_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
k_weight_initializer = numpy_helper.from_array(k_weight_np_vals, 'encoder.layer.0.SelfAttention.k.weight')

v_weight_np_vals = (0.01 * np.arange(hidden_size * hidden_size, dtype=np.float32)).reshape((hidden_size, hidden_size))
v_weight_initializer = numpy_helper.from_array(v_weight_np_vals, 'encoder.layer.0.SelfAttention.v.weight')

q_bias_np_vals = (0.01 * np.arange(hidden_size, dtype=np.float32))
q_bias_initializer = numpy_helper.from_array(q_bias_np_vals, 'encoder.layer.0.SelfAttention.q.bias')

k_bias_np_vals = (0.01 * np.arange(hidden_size, dtype=np.float32))
k_bias_initializer = numpy_helper.from_array(k_bias_np_vals, 'encoder.layer.0.SelfAttention.k.bias')

v_bias_np_vals = (0.01 * np.arange(hidden_size, dtype=np.float32))
v_bias_initializer = numpy_helper.from_array(v_bias_np_vals, 'encoder.layer.0.SelfAttention.v.bias')

q_starts_initializer = numpy_helper.from_array(np.asarray([0], dtype=np.int64), 'q_starts')
k_starts_initializer = numpy_helper.from_array(np.asarray([hidden_size], dtype=np.int64), 'k_starts')
v_starts_initializer = numpy_helper.from_array(np.asarray([2*hidden_size], dtype=np.int64), 'v_starts')

q_ends_initializer = numpy_helper.from_array(np.asarray([hidden_size], dtype=np.int64), 'q_ends')
k_ends_initializer = numpy_helper.from_array(np.asarray([2*hidden_size], dtype=np.int64), 'k_ends')
v_ends_initializer = numpy_helper.from_array(np.asarray([9223372036854775807], dtype=np.int64), 'v_ends')

slice_axes_initializer = numpy_helper.from_array(np.asarray([0], dtype=np.int64), 'slice_axes')
slice_steps_initializer = numpy_helper.from_array(np.asarray([1], dtype=np.int64), 'slice_steps')


transpose_q = helper.make_node('Transpose', [q_weight_initializer.name], ['transpose_q'], name='transpose_q', perm=[1,0])
transpose_k = helper.make_node('Transpose', [k_weight_initializer.name], ['transpose_k'], name='transpose_k', perm=[1,0])
transpose_v = helper.make_node('Transpose', [v_weight_initializer.name], ['transpose_v'], name='transpose_v', perm=[1,0])

matmul_q = helper.make_node('MatMul', ['input', 'transpose_q'], ['matmul_q'], name='matmul_q')
matmul_k = helper.make_node('MatMul', ['input', 'transpose_k'], ['matmul_k'], name='matmul_k')
matmul_v = helper.make_node('MatMul', ['input', 'transpose_v'], ['matmul_v'], name='matmul_v')

concat_bias = helper.make_node("Concat", 
            [q_bias_initializer.name,k_bias_initializer.name,v_bias_initializer.name],
            ['concat_bias'],
            axis=0,
            name='concat_bias')

slice_q = helper.make_node('Slice',
            ['concat_bias', q_starts_initializer.name, q_ends_initializer.name, 
            slice_axes_initializer.name,slice_steps_initializer.name],
            ['slice_q'], name = 'slice_q' )

slice_k = helper.make_node('Slice',
            ['concat_bias', k_starts_initializer.name, k_ends_initializer.name, 
            slice_axes_initializer.name,slice_steps_initializer.name],
            ['slice_k'], name = 'slice_k' )

slice_v = helper.make_node('Slice',
            ['concat_bias', v_starts_initializer.name, v_ends_initializer.name, 
            slice_axes_initializer.name,slice_steps_initializer.name],
            ['slice_v'], name = 'slice_v' )

add_q = helper.make_node('Add', ['matmul_q', 'slice_q'], ['add_q'], name='add_q')
add_k = helper.make_node('Add', ['matmul_k', 'slice_k'], ['add_k'], name='add_k')
add_v = helper.make_node('Add', ['matmul_v', 'slice_v'], ['add_v'], name='add_v')

add_1 = helper.make_node('Add', ['add_q','add_k'],['add_1'], name='add_1')
add_2 = helper.make_node('Add', ['add_1','add_v'],['add_2'], name='add_2')
identity = helper.make_node('Identity', ['add_2'], ['output'], name='identity')

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [transpose_q,transpose_k,transpose_v,matmul_q,matmul_k,matmul_v,add_q,add_k,add_v,
    concat_bias,slice_q, slice_k, slice_v, add_1, add_2, identity],
    'concat-slice-test-model',
    [X],
    [Y],
    [q_weight_initializer,k_weight_initializer,v_weight_initializer,q_bias_initializer,k_bias_initializer,
    v_bias_initializer,q_starts_initializer,k_starts_initializer,v_starts_initializer,q_ends_initializer,
    k_ends_initializer,v_ends_initializer,slice_axes_initializer,slice_steps_initializer]
)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = "" # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = 'com.microsoft'

opsets.append(msdomain)
kwargs={}
kwargs['opset_imports'] = opsets

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)
onnx.save(model_def, 'concat_slice_basic_test.onnx')



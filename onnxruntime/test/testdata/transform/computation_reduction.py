import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
from onnx import numpy_helper
import numpy as np

vocab_size=256 #30258

X = helper.make_tensor_value_info('input', TensorProto.FLOAT, ["batch", "seqlen", 128])
unsqueezed_masked_lm_positions = helper.make_tensor_value_info('unsqueezed_masked_lm_positions', 
                                                               TensorProto.INT64, ["batch", "dynamic_prediction_count", 1])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, ["batch", "dynamic_prediction_count", vocab_size])
Gather_Y = helper.make_tensor_value_info('gather_output', TensorProto.FLOAT, ["batch", 128])

layer_norm1_weight_np_vals = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
layer_norm1_weight_initializer = numpy_helper.from_array(layer_norm1_weight_np_vals, "bert.encoder.layer.2.output.LayerNorm.weight")

layer_norm1_bias_np_vals = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
layer_norm1_bias_initializer = numpy_helper.from_array(layer_norm1_bias_np_vals, "bert.encoder.layer.2.output.LayerNorm.bias")

matmul1_np_vals = np.random.uniform(0.0, 1.0, (128, 128)).astype(np.float32).reshape((128, 128))
matmul1_initializer = numpy_helper.from_array(matmul1_np_vals, "matmul1_initializer")

add1_np_vals = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
add1_initializer = numpy_helper.from_array(add1_np_vals, "add1_initializerr")

layer_norm2_weight_np_vals = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
layer_norm2_weight_initializer = numpy_helper.from_array(layer_norm2_weight_np_vals, "cls.predictions.transform.LayerNorm.weight")

layer_norm2_bias_np_vals = np.random.uniform(0.0, 1.0, (128)).astype(np.float32).reshape((128))
layer_norm2_bias_initializer = numpy_helper.from_array(layer_norm2_bias_np_vals, "cls.predictions.transform.LayerNorm.bias")

matmul2_np_vals = np.random.uniform(0.0, 1.0, (128, vocab_size)).astype(np.float32).reshape((128, vocab_size))
matmul2_initializer = numpy_helper.from_array(matmul2_np_vals, "bert.embeddings.word_embeddings.weight_transposed")

add2_np_vals = np.random.uniform(0.0, 1.0, (vocab_size)).astype(np.float32).reshape((vocab_size))
add2_initializer = numpy_helper.from_array(add2_np_vals, "cls.predictions.bias")

gather_indice_np_vals = np.asarray([0]).astype(np.int64).reshape(())
gather_indice_initializer = numpy_helper.from_array(gather_indice_np_vals, "gather_indice_initializer")

nodes=[]
layer_norm1 = helper.make_node('LayerNormalization', 
                               ['input', layer_norm1_weight_initializer.name, layer_norm1_bias_initializer.name],
                               ['layer_norm1', 'saved_mean1', 'saved_inv_std1'],
                               name='layer_norm_1', epsilon=9.999999960041972e-13, axis=-1)
nodes.append(layer_norm1)

gather1 = helper.make_node('Gather', ['layer_norm1', gather_indice_initializer.name], ['gather_output'], name="gather_output", axis=1)
nodes.append(gather1)

matmul1 = helper.make_node('MatMul', ['layer_norm1', matmul1_initializer.name], ['matmul1'], name="matmul_1")
nodes.append(matmul1)

add1 = helper.make_node('Add', [add1_initializer.name, 'matmul1'], ['add1'], name="add_1")
nodes.append(add1)

gelu1 = helper.make_node('Gelu', ['add1'], ['gelu1'], name='gelu_1', domain='com.microsoft')
nodes.append(gelu1)

layer_norm2 = helper.make_node('LayerNormalization',
                               ['gelu1', layer_norm2_weight_initializer.name, layer_norm2_bias_initializer.name],
                               ['layer_norm2', 'saved_mean2', 'saved_inv_std2'],
                               name='layer_norm_2', epsilon=9.999999960041972e-13, axis=-1)
nodes.append(layer_norm2)

matmul2 = helper.make_node('MatMul', ['layer_norm2', matmul2_initializer.name], ['matmul2'], name="matmul_2")
nodes.append(matmul2)

add2 = helper.make_node('Add', ['matmul2', add2_initializer.name], ['add2'], name="add_2")
nodes.append(add2)

gathernd1 = helper.make_node('GatherND', ['add2', 'unsqueezed_masked_lm_positions'], ['gathernd1'], name="gathernd_1", batch_dims=1)
nodes.append(gathernd1)

identity1 = helper.make_node('Identity', ['gathernd1'], ['output'], name="output")
nodes.append(identity1)

initializers=[layer_norm1_weight_initializer, layer_norm1_bias_initializer, matmul1_initializer, add1_initializer,
              layer_norm2_weight_initializer, layer_norm2_bias_initializer, matmul2_initializer, add2_initializer,
              gather_indice_initializer]
# Create the graph (GraphProto)
graph_def = helper.make_graph(nodes, 'test-model', [X, unsqueezed_masked_lm_positions], [Y, Gather_Y], initializers)

opsets = []
onnxdomain = OperatorSetIdProto()
onnxdomain.version = 12
onnxdomain.domain = ""  # The empty string ("") or absence of this field implies the operator set that is defined as part of the ONNX specification.
opsets.append(onnxdomain)

msdomain = OperatorSetIdProto()
msdomain.version = 1
msdomain.domain = "com.microsoft"

opsets.append(msdomain)
kwargs = {}
kwargs["opset_imports"] = opsets

model_def = helper.make_model(graph_def, producer_name='onnx-example', **kwargs)

onnx.save(model_def, "computation_reduction_transformer.onnx")

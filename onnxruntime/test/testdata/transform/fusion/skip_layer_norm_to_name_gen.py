import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum
import numpy as np

#
# TWO SUBGRAPHS: These should all be fused together before quantization!
#    SkipLayerNormGelu
#    SkipLayerNormAttention
#

# Graph for SubLayerNorm with a MatMul -> BiasGelu -> MatMul subgraph
# TODO(kreeger): Eventually rename this one.
def generate_nodes():
    kwargs = {}
    kwargs['epsilon'] = 1e-12
    nodes = [
        helper.make_node('SkipLayerNormalization',
                         ['sln_1_input', 'sln_1_skip', 'sln_1_gamma', 'sln_1_beta', 'sln_1_bias'], ['sln_1_output'],
                         domain="com.microsoft",
                         **kwargs),
        helper.make_node('MatMul', ['sln_1_output', 'matmul_1_b'], ['matmul_1_output']),
        helper.make_node('BiasGelu', ['matmul_1_output', 'bias_gelu_1_bias'], ['bias_gelu_1_output'],
                         domain="com.microsoft"),
        helper.make_node('MatMul', ['bias_gelu_1_output', 'matmul_2_b'], ['matmul_2_output']),
        helper.make_node('SkipLayerNormalization',
                         ['sln_2_input', 'matmul_2_output', 'sln_2_gamma', 'sln_2_beta', 'sln_2_bias'],
                         ['sln_2_output'],
                         domain="com.microsoft",
                         **kwargs),
    ]
    return nodes


# TODO(kreeger): Eventually rename this one.
def generate_initializers():
    initializers = [
        helper.make_tensor('sln_1_gamma', TensorProto.FLOAT, [4], np.random.rand(4)),
        helper.make_tensor('sln_1_beta', TensorProto.FLOAT, [4], np.random.rand(4)),
        helper.make_tensor('sln_1_bias', TensorProto.FLOAT, [4], np.random.rand(4)),

        helper.make_tensor('matmul_1_b', TensorProto.FLOAT, [4, 16], np.random.rand(4 * 16)),

        helper.make_tensor('bias_gelu_1_bias', TensorProto.FLOAT, [16], np.random.rand(16)),

        helper.make_tensor('matmul_2_b', TensorProto.FLOAT, [16, 4], np.random.rand(16 * 4)),

        helper.make_tensor('sln_2_gamma', TensorProto.FLOAT, [4], np.random.rand(4)),
        helper.make_tensor('sln_2_beta', TensorProto.FLOAT, [4], np.random.rand(4)),
        helper.make_tensor('sln_2_bias', TensorProto.FLOAT, [4], np.random.rand(4)),
    ]
    return initializers


def generate_model(model_name):
    # TODO(kreeger): where do attributes go?

    batch_size = 1
    sequence_size = 8
    something_something = 4

    graph = helper.make_graph(
        generate_nodes(),
        'SkipLayerNorm_MatMulBiasGeluMatMul_Subgraph',
        [  # inputs
            helper.make_tensor_value_info('sln_1_input', TensorProto.FLOAT,
                                          [batch_size, sequence_size, something_something]),
            helper.make_tensor_value_info('sln_1_skip', TensorProto.FLOAT,
                                          [batch_size, sequence_size, something_something]),
            helper.make_tensor_value_info('sln_2_input', TensorProto.FLOAT,
                                          [batch_size, sequence_size, something_something])
        ],
        [  # outputs
            helper.make_tensor_value_info('sln_2_output', TensorProto.FLOAT,
                                          [batch_size, sequence_size, something_something])
        ],
        generate_initializers())

    model = helper.make_model(graph);
    onnx.save(model, model_name)
    pass


generate_model('skip_layer_norm_matmul_biasgelu_matmul_subgraph.onnx')
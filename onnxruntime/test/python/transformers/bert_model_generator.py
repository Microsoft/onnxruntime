# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import onnx
import math
import numpy as np
from typing import List
from packaging import version
from onnx import helper, TensorProto


def float_tensor(name: str, shape: List[int], random=False):
    low = 0.0
    high = 1.0
    total_elements = 1
    for x in shape:
        total_elements *= x
    weights = [np.random.uniform(low, high) for _ in range(total_elements)] if random else [1.0] * total_elements
    return helper.make_tensor(name, TensorProto.FLOAT, shape, weights)


def reverse_if(inputs, reverse=False):
    if reverse:
        inputs.reverse()
    return inputs


def create_bert_attention(input_hidden_size=16,
                          num_heads=2,
                          pruned_qk_hidden_size=16,
                          pruned_v_hidden_size=16,
                          use_float_mask=False,
                          switch_add_inputs=False):
    # unsqueeze in opset version 13 has two inputs (axis is moved from attribute to input).
    has_unsqueeze_two_inputs = (version.parse(onnx.__version__) >= version.parse('1.8.0'))

    # nodes in attention subgraph
    nodes = [
        helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm"),
        helper.make_node("LayerNormalization", ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
                         ["layernorm_out"],
                         "layernorm",
                         axis=-1,
                         epsion=0.000009999999747378752),

        # q nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_q_weight"], ["matmul_q_out"], "matmul_q"),
        helper.make_node("Add", reverse_if(["matmul_q_out", "add_q_weight"], switch_add_inputs), ["add_q_out"], "add_q"),
        helper.make_node("Reshape", ["add_q_out", "reshape_weight_qk"], ["reshape_q_out"], "reshape_q"),
        helper.make_node("Transpose", ["reshape_q_out"], ["transpose_q_out"], "transpose_q", perm=[0, 2, 1, 3]),

        # k nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_k_weight"], ["matmul_k_out"], "matmul_k"),
        helper.make_node("Add", reverse_if(["matmul_k_out", "add_k_weight"], switch_add_inputs), ["add_k_out"], "add_k"),
        helper.make_node("Reshape", ["add_k_out", "reshape_weight_qk"], ["reshape_k_out"], "reshape_k"),
        helper.make_node("Transpose", ["reshape_k_out"], ["transpose_k_out"], "transpose_k", perm=[0, 2, 3, 1]),

        # mask nodes
        helper.make_node("Unsqueeze", ["input_mask", "axes_1"], ["unsqueeze0_out"], "unsqueeze0") if has_unsqueeze_two_inputs \
            else helper.make_node("Unsqueeze", ["input_mask"], ["unsqueeze0_out"], "unsqueeze0", axes=[1]),
        helper.make_node("Unsqueeze", ["unsqueeze0_out", "axes_2"], ["unsqueeze1_out"], "unsqueeze1") if has_unsqueeze_two_inputs \
            else helper.make_node("Unsqueeze", ["unsqueeze0_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[2]),

        # when attention_mask is float type, no need to cast
        helper.make_node("Cast", ["unsqueeze1_out"], ["cast_out"], "cast", to=1) if not use_float_mask else None,
        helper.make_node("Sub", ["sub_weight", "unsqueeze1_out" if use_float_mask else "cast_out"], ["sub_out"], "sub"),
        helper.make_node("Mul", ["sub_out", "mul_weight"], ["mul_mask_out"], "mul_mask"),

        # qk nodes
        helper.make_node("MatMul", ["transpose_q_out", "transpose_k_out"], ["matmul_qk_out"], "matmul_qk"),
        helper.make_node("Div", ["matmul_qk_out", "div_weight"], ["div_qk_out"], "div_qk"),
        helper.make_node("Add", reverse_if(["div_qk_out", "mul_mask_out"], switch_add_inputs), ["add_qk_out"], "add_qk"),
        helper.make_node("Softmax", ["add_qk_out"], ["softmax_qk_out"], "softmax_qk", axis=3),

        # v nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_v_weight"], ["matmul_v_out"], "matmul_v"),
        helper.make_node("Add", ["matmul_v_out", "add_v_weight"], ["add_v_out"], "add_v"),
        helper.make_node("Reshape", ["add_v_out", "reshape_weight_v"], ["reshape_v_out"], "reshape_v"),
        helper.make_node("Transpose", ["reshape_v_out"], ["transpose_v_out"], "transpose_v", perm=[0, 2, 1, 3]),

        # qkv nodes
        helper.make_node("MatMul", ["softmax_qk_out", "transpose_v_out"], ["matmul_qkv_1_out"], "matmul_qkv_1"),
        helper.make_node("Transpose", ["matmul_qkv_1_out"], ["transpose_qkv_out"], "transpose_qkv", perm=[0, 2, 1, 3]),
        helper.make_node("Reshape", ["transpose_qkv_out", "reshape_weight_qkv"], ["reshape_qkv_out"], "reshape_qkv"),
        helper.make_node("MatMul", ["reshape_qkv_out", "matmul_qkv_weight"], ["matmul_qkv_2_out"], "matmul_qkv_2"),
        helper.make_node("Add", reverse_if(["matmul_qkv_2_out", "add_qkv_weight"], switch_add_inputs), ["add_qkv_out"], "add_qkv"),
        helper.make_node("Add", reverse_if(["add_qkv_out", "layernorm_out"], switch_add_inputs), ["skip_output"], "add_skip"),
        helper.make_node("LayerNormalization", ["skip_output", "layer_norm_weight", "layer_norm_bias"], ["output"],
                         "layernorm2",
                         axis=-1,
                         epsion=0.000009999999747378752),
    ]

    pruned_qk_head_size = int(pruned_qk_hidden_size / num_heads)
    pruned_v_head_size = int(pruned_v_hidden_size / num_heads)
    initializers = [  # initializers
        float_tensor('layer_norm_weight', [input_hidden_size]),
        float_tensor('layer_norm_bias', [input_hidden_size]),
        float_tensor('matmul_q_weight', [input_hidden_size, pruned_qk_hidden_size]),
        float_tensor('matmul_k_weight', [input_hidden_size, pruned_qk_hidden_size]),
        float_tensor('matmul_v_weight', [input_hidden_size, pruned_v_hidden_size]),
        float_tensor('matmul_qkv_weight', [pruned_v_hidden_size, input_hidden_size]),
        float_tensor('add_q_weight', [pruned_qk_hidden_size]),
        float_tensor('add_k_weight', [pruned_qk_hidden_size]),
        float_tensor('add_v_weight', [pruned_v_hidden_size]),
        float_tensor('add_qkv_weight', [input_hidden_size]),
        helper.make_tensor('div_weight', TensorProto.FLOAT, [1], [math.sqrt(pruned_qk_head_size)]),
        helper.make_tensor('sub_weight', TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor('mul_weight', TensorProto.FLOAT, [1], [-10000]),
        helper.make_tensor('reshape_weight_qk', TensorProto.INT64, [4], [0, 0, num_heads, pruned_qk_head_size]),
        helper.make_tensor('reshape_weight_v', TensorProto.INT64, [4], [0, 0, num_heads, pruned_v_head_size]),
        helper.make_tensor('reshape_weight_qkv', TensorProto.INT64, [3], [0, 0, pruned_v_hidden_size]),
    ]

    if has_unsqueeze_two_inputs:
        initializers.append(helper.make_tensor('axes_1', TensorProto.INT64, [1], [1]))
        initializers.append(helper.make_tensor('axes_2', TensorProto.INT64, [1], [2]))

    batch_size = 1
    sequence_length = 3
    graph = helper.make_graph(
        [node for node in nodes if node],
        "AttentionFusionPrunedModel",  #name
        [  # inputs
            helper.make_tensor_value_info('input_1', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
            helper.make_tensor_value_info('input_2', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
            helper.make_tensor_value_info('input_mask', TensorProto.FLOAT if use_float_mask else TensorProto.INT64,
                                          [batch_size, sequence_length])
        ],
        [  # outputs
            helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
        ],
        initializers)

    model = helper.make_model(graph)
    return model


def create_tf2onnx_attention_3d(input_hidden_size=16, num_heads=4, head_size=4, use_float_mask=False):
    # unsqueeze in opset version 13 has two inputs (axis is moved from attribute to input).
    has_unsqueeze_two_inputs = (version.parse(onnx.__version__) >= version.parse('1.8.0'))

    # nodes in attention subgraph
    nodes = [
        helper.make_node("Add", ["input_1", "input_2"], ["layernorm_input"], "add_layernorm"),
        helper.make_node("LayerNormalization", ["layernorm_input", "layer_norm_weight", "layer_norm_bias"],
                         ["layernorm_out"],
                         "layernorm",
                         axis=-1,
                         epsion=0.000009999999747378752),

        # q nodes
        helper.make_node("Einsum", ["layernorm_out", "einsum_q_weight"], ["einsum_q_out"], "einsum_q", equation="abc,cde->abde"),
        helper.make_node("Add", ["einsum_q_out", "add_q_weight"], ["add_q_out"], "add_q"),

        # k nodes
        helper.make_node("Einsum", ["layernorm_out", "einsum_k_weight"], ["einsum_k_out"], "einsum_k", equation="abc,cde->abde"),
        helper.make_node("Add", ["einsum_k_out", "add_k_weight"], ["add_k_out"], "add_k"),
        helper.make_node("Mul", ["add_k_out", "mul_weight_1"], ["mul_k_out"], "mul_k"),

        # mask nodes
        helper.make_node("Unsqueeze", ["input_mask", "axes_1"], ["unsqueeze0_out"], "unsqueeze0") if has_unsqueeze_two_inputs \
            else helper.make_node("Unsqueeze", ["input_mask"], ["unsqueeze0_out"], "unsqueeze0", axes=[1, 2]),
        helper.make_node("Slice", ["unsqueeze0_out", "slice_start", "slice_end", "slice_axes", "slice_steps"], ["slice_out"], "slice"),

        # when attention_mask is float type, no need to cast
        helper.make_node("Cast", ["slice_out"], ["cast_out"], "cast", to=1) if not use_float_mask else None,
        helper.make_node("Sub", ["sub_weight", "unsqueeze1_out" if use_float_mask else "cast_out"], ["sub_out"], "sub"),
        helper.make_node("Mul", ["sub_out", "mul_weight_2"], ["mul_mask_out"], "mul_mask"),

        # qk nodes
        helper.make_node("Einsum", ["add_q_out", "mul_k_out"], ["einsum_qk_out"], "einsum_qk", equation="aecd,abcd->acbe"),
        helper.make_node("Add", ["einsum_qk_out", "mul_mask_out"], ["add_qk_out"], "add_qk"),
        helper.make_node("Softmax", ["add_qk_out"], ["softmax_qk_out"], "softmax_qk", axis=3),

        # v nodes
        helper.make_node("Einsum", ["layernorm_out", "einsum_v_weight"], ["einsum_v_out"], "einsum_v", equation="abc,cde->abde"),
        helper.make_node("Add", ["einsum_v_out", "add_v_weight"], ["add_v_out"], "add_v"),

        # qkv nodes
        helper.make_node("Einsum", ["softmax_qk_out", "add_v_out"], ["einsum_qkv_1_out"], "einsum_qkv_1", equation="acbe,aecd->abcd"),
        helper.make_node("Einsum", ["einsum_qkv_1_out", "einsum_qkv_weight"], ["einsum_qkv_2_out"], "einsum_qkv_2", equation="abcd,cde->abe"),
        helper.make_node("Add", ["einsum_qkv_2_out", "add_qkv_weight"], ["add_qkv_out"], "add_qkv"),
        helper.make_node("Add", ["add_qkv_out", "layernorm_out"], ["skip_output"], "add_skip"),
        helper.make_node("LayerNormalization", ["skip_output", "layer_norm_weight", "layer_norm_bias"], ["output"],
                         "layernorm2",
                         axis=-1,
                         epsion=0.000009999999747378752),
    ]

    initializers = [  # initializers
        float_tensor('layer_norm_weight', [input_hidden_size]),
        float_tensor('layer_norm_bias', [input_hidden_size]),
        float_tensor('einsum_q_weight', [input_hidden_size, num_heads, head_size]),
        float_tensor('einsum_k_weight', [input_hidden_size, num_heads, head_size]),
        float_tensor('einsum_v_weight', [input_hidden_size, num_heads, head_size]),
        float_tensor('einsum_qkv_weight', [num_heads, head_size, input_hidden_size]),
        float_tensor('add_q_weight', [num_heads, head_size]),
        float_tensor('add_k_weight', [num_heads, head_size]),
        float_tensor('add_v_weight', [num_heads, head_size]),
        float_tensor('add_qkv_weight', [input_hidden_size]),
        helper.make_tensor('sub_weight', TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor('mul_weight_1', TensorProto.FLOAT, [1], [-10000]),
        helper.make_tensor('mul_weight_2', TensorProto.FLOAT, [1], [0.125]),
        helper.make_tensor('reshape_weight_1', TensorProto.INT64, [4], [0, 0, num_heads, head_size]),
        helper.make_tensor('slice_start', TensorProto.INT32, [4], [0, 0, 0, 0]),
        helper.make_tensor('slice_end', TensorProto.INT32, [4], [1000000000, 1000000000, 1000000000, 1000000000]),
        helper.make_tensor('slice_axes', TensorProto.INT32, [4], [0, 1, 2, 3]),
        helper.make_tensor('slice_steps', TensorProto.INT32, [4], [1, 1, 1, 1])
    ]

    if has_unsqueeze_two_inputs:
        initializers.append(helper.make_tensor('axes_1', TensorProto.INT64, [2], [1, 2]))

    batch_size = 1
    sequence_length = 3
    graph = helper.make_graph(
        [node for node in nodes if node],
        "AttentionFusionPrunedModel",  #name
        [  # inputs
            helper.make_tensor_value_info('input_1', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
            helper.make_tensor_value_info('input_2', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
            helper.make_tensor_value_info('input_mask', TensorProto.FLOAT if use_float_mask else TensorProto.INT64,
                                          [batch_size, sequence_length])
        ],
        [  # outputs
            helper.make_tensor_value_info('output', TensorProto.FLOAT,
                                          [batch_size, sequence_length, input_hidden_size]),
        ],
        initializers)

    model = helper.make_model(graph)
    return model


if __name__ == "__main__":
    model = create_bert_attention()
    onnx.save(model, "pruned_bert_attention.onnx")
    model = create_bert_attention(switch_add_inputs=True)
    onnx.save(model, "bert_attention_reverse_add_order.onnx")
    model = create_tf2onnx_attention_3d()
    onnx.save(model, "bert_3d_attention.onnx")

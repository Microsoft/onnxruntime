#!/usr/bin/env python
# coding: utf-8
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import unittest
import onnx
import numpy as np
from onnx import helper, TensorProto
from onnxruntime.quantization import quantize_dynamic
from op_test_utils import TestDataFeeds, check_model_correctness, check_op_type_count


class TestOpEmbedLayerNormalization(unittest.TestCase):
    def input_feeds_int32(self, n, name2shape):
        input_data_list = []
        for i in range(n):
            inputs = {}
            for name, shape in name2shape.items():
                inputs.update({name: np.ones(shape).astype(np.int32)})
            input_data_list.extend([inputs])

        dr = TestDataFeeds(input_data_list)
        return dr

    def construct_model(self, batch, hidden_size, sequence_length, model_path):
        #    <segment_ids>    <input_ids>
        #              \        /
        #       (EmbedLayerNormalization)
        #               /       \
        # <layernorm_output>  <mask_index_output>

        # Inputs to EmbedLayerNormalizationNode
        input_ids_shape = [batch, sequence_length]
        input_ids_tensor = helper.make_tensor_value_info('input_ids', TensorProto.INT32, input_ids_shape)

        segment_ids_shape = [batch, sequence_length]
        segment_ids_tensor = helper.make_tensor_value_info('segment_ids', TensorProto.INT32, segment_ids_shape)

        # EmbedLayerNormalization Node Constants and Weights:
        word_embed_shape = [32, hidden_size]
        word_embed_weights = np.random.random_sample(word_embed_shape).astype(dtype='float32')
        word_embed_initializer = onnx.numpy_helper.from_array(word_embed_weights, name='word_embed')

        pos_embed_shape = [16, hidden_size]
        pos_embed_weights = np.random.random_sample(pos_embed_shape).astype(dtype='float32')
        pos_embed_initializer = onnx.numpy_helper.from_array(pos_embed_weights, name='pos_embed')

        seg_embed_shape = [2, hidden_size]
        seg_embed_weights = np.random.random_sample(seg_embed_shape).astype(dtype='float32')
        seg_embed_initializer = onnx.numpy_helper.from_array(seg_embed_weights, name='seg_embed')

        gamma_shape = [hidden_size]
        gamma = np.random.random_sample(gamma_shape).astype(dtype='float32')
        gamma_initializer = onnx.numpy_helper.from_array(gamma, name='gamma')

        beta_shape = [hidden_size]
        beta = np.random.random_sample(beta_shape).astype(dtype='float32')
        beta_initializer = onnx.numpy_helper.from_array(beta, name='beta')

        # EmbedLayerNormalization Outputs:
        layernorm_out_shape = [batch, sequence_length, hidden_size]
        layernorm_out_tensor = helper.make_tensor_value_info('layernorm_out', TensorProto.FLOAT, layernorm_out_shape)

        mask_index_out_shape = [batch]
        mask_index_out_tensor = helper.make_tensor_value_info('mask_index_out', TensorProto.INT32, mask_index_out_shape)

        # EmbedLayerNormalization Node:
        embed_layer_norm_inputs = [
            'input_ids', 'segment_ids', 'word_embed', 'pos_embed', 'seg_embed', 'gamma', 'beta'
        ]
        embed_layer_norm_outputs = ['layernorm_out', 'mask_index_out']
        embed_layer_norm_node = helper.make_node('EmbedLayerNormalization',
                                                 embed_layer_norm_inputs,
                                                 embed_layer_norm_outputs,
                                                 domain='com.microsoft')

        # Construct the Graph and Model:
        nodes = [embed_layer_norm_node]
        graph_name = 'embed_layernorm_graph'
        inputs = [input_ids_tensor, segment_ids_tensor]
        outputs = [layernorm_out_tensor, mask_index_out_tensor]
        initializers = [
            word_embed_initializer, pos_embed_initializer, seg_embed_initializer, gamma_initializer, beta_initializer
        ]

        graph = helper.make_graph(nodes, graph_name, inputs, outputs, initializer=initializers)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
        model.ir_version = 7 # use stable onnx ir version
        onnx.save(model, model_path)

    def test_quantize_batch_size_1(self):
        batch = 1
        hidden_size = 4
        sequence_length = 4

        model_f32_path = 'test_embed_layer_norm_unit_test_batch1.onnx'
        model_uint8_path = 'test_embed_layer_norm_unit_test_batch1_uint8.onnx'

        self.construct_model(batch, hidden_size, sequence_length, model_f32_path)

        data_reader = self.input_feeds_int32(1, {
            'input_ids': [batch, sequence_length],
            'segment_ids': [batch, sequence_length]
        })

        quantize_dynamic(model_f32_path, model_uint8_path)

        # Quantization should not have any DequantizeLinear nodes:
        qnode_counts = {'DequantizeLinear': 0, 'QEmbedLayerNormalization': 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        data_reader.rewind()

        check_model_correctness(self, model_f32_path, model_uint8_path, data_reader.get_next())

    def test_quantize_batch_size_2(self):
        batch = 2
        hidden_size = 4
        sequence_length = 4

        model_f32_path = 'test_embed_layer_norm_unit_test_batch2.onnx'
        model_uint8_path = 'test_embed_layer_norm_unit_test_batch2_uint8.onnx'

        self.construct_model(batch, hidden_size, sequence_length, model_f32_path)

        data_reader = self.input_feeds_int32(1, {
            'input_ids': [batch, sequence_length],
            'segment_ids': [batch, sequence_length]
        })

        quantize_dynamic(model_f32_path, model_uint8_path)

        # Quantization should not have any DequantizeLinear nodes:
        qnode_counts = {'DequantizeLinear': 0, 'QEmbedLayerNormalization': 1}
        check_op_type_count(self, model_uint8_path, **qnode_counts)
        data_reader.rewind()

        check_model_correctness(self, model_f32_path, model_uint8_path, data_reader.get_next())


if __name__ == '__main__':
    unittest.main()

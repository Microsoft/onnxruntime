#!/usr/bin/env python
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import struct
import tempfile
import unittest

import numpy as np
import onnx

from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config
from onnxruntime.quantization.quant_utils import compute_scale_zp, get_opset_version, get_qmin_qmax_for_qType


class DummyDataReader(CalibrationDataReader):
    def __init__(self, activations):
        self.iterator = ({"INP": act} for act in activations)

    def get_next(self):
        return next(self.iterator, None)


class TestTensorQuantOverridesOption(unittest.TestCase):
    def setUp(self):
        self.activations = [
            np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]], dtype="float32"),
        ]

        self.weight = np.array([[[-1.0, -2.0], [1.0, 2.0]], [[-0.5, -1.5], [0.5, 1.5]]], dtype=np.float32)
        self.bias = np.array([0.0, 1.0], dtype=np.float32)
        self.default_act_qtype = onnx.TensorProto.UINT8
        self.default_wgt_qtype = onnx.TensorProto.UINT8
        self.default_wgt_qtype_per_channel = onnx.TensorProto.UINT8
        self.default_bias_qtype = onnx.TensorProto.INT32

        self.default_zp_scales = {
            "INP": (0, np.float32(0.0235294122248888)),
            "SIG_OUT": (0, np.float32(0.003911871928721666)),
            "WGT": (128, np.float32(0.01568627543747425)),
            "BIAS": (0, np.float32(0.0000613626980339177)),  # zp == 0, scale = weight_scale * sig_out_scale
            "OUT": (0, np.float32(0.005075461231172085)),
        }
        self.default_zp_scales_per_channel = {
            "INP": (0, np.float32(0.0235294122248888)),
            "SIG_OUT": (0, np.float32(0.003911871928721666)),
            # per-channel weights are always symmetric (ie. zp = (qmin + qmax) / 2)
            "WGT": ([128, 128], [np.float32(0.01568627543747425), np.float32(0.0117647061124444)]),
            "BIAS": ([0, 0], [np.float32(0.0000613626980339177), np.float32(0.000046022025344427675)]),
            "OUT": (0, np.float32(0.005075461231172085)),
        }

    def build_float32_model(self, opset=13):
        #    (input)
        #       |
        #    Sigmoid
        #       |
        #     Conv
        #       |
        #    (output)

        inp = onnx.helper.make_tensor_value_info("INP", onnx.TensorProto.FLOAT, self.activations[0].shape)
        sigmoid_node = onnx.helper.make_node("Sigmoid", ["INP"], ["SIG_OUT"])

        out = onnx.helper.make_tensor_value_info("OUT", onnx.TensorProto.FLOAT, [None, None, None])
        wgt_init = onnx.numpy_helper.from_array(self.weight, "WGT")
        bias_init = onnx.numpy_helper.from_array(self.bias, "BIAS")
        conv_node = onnx.helper.make_node("Conv", ["SIG_OUT", "WGT", "BIAS"], ["OUT"])

        graph = onnx.helper.make_graph(
            [sigmoid_node, conv_node], "test", [inp], [out], initializer=[wgt_init, bias_init]
        )
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", opset)])
        onnx.save(model, "model.onnx")

    def perform_qdq_quantization(
        self, output_model_name, extra_options=None, per_channel=False, activation_type=None, opset=13
    ):
        self.build_float32_model(opset)

        if activation_type is None:
            activation_type = self.default_act_qtype

        quantize_static(
            model_input="model.onnx",
            model_output=output_model_name,
            calibration_data_reader=DummyDataReader(self.activations),
            quant_format=QuantFormat.QDQ,
            activation_type=activation_type,
            weight_type=self.default_wgt_qtype,
            per_channel=per_channel,
            op_types_to_quantize=["Conv", "Sigmoid"],
            extra_options=extra_options,
        )

        # Extract quantization parameters: scales and zero points for activations and weights.
        model = onnx.load(output_model_name)
        inp_zp = next(init for init in model.graph.initializer if init.name == "INP_zero_point")
        inp_sc = next(init for init in model.graph.initializer if init.name == "INP_scale")
        sig_out_zp = next(init for init in model.graph.initializer if init.name == "SIG_OUT_zero_point")
        sig_out_sc = next(init for init in model.graph.initializer if init.name == "SIG_OUT_scale")
        wgt_zp = next(init for init in model.graph.initializer if init.name == "WGT_zero_point")
        wgt_sc = next(init for init in model.graph.initializer if init.name == "WGT_scale")
        bias_zp = next(
            init
            for init in model.graph.initializer
            if init.name == "BIAS_quantized_zero_point" or init.name == "BIAS_zero_point"
        )
        bias_sc = next(
            init for init in model.graph.initializer if init.name == "BIAS_quantized_scale" or init.name == "BIAS_scale"
        )
        out_zp = next(init for init in model.graph.initializer if init.name == "OUT_zero_point")
        out_sc = next(init for init in model.graph.initializer if init.name == "OUT_scale")

        # Return quantization parameters
        return inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, bias_zp, bias_sc, out_zp, out_sc

    def test_qdq_default(self):
        """
        Test default behavior without specifying the TensorQuantOverrides option.
        """
        (
            inp_zp,
            inp_sc,
            sig_out_zp,
            sig_out_sc,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            out_zp,
            out_sc,
        ) = self.perform_qdq_quantization(
            "model_default_quant_overrides.onnx",
            extra_options=None,  # default behavior
        )

        # No overrides set. Expect default values
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        self.assertEqual(sig_out_zp.int32_data[0], self.default_zp_scales["SIG_OUT"][0])
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)
        self.assertEqual(sig_out_sc.float_data[0], self.default_zp_scales["SIG_OUT"][1])

        self.assertEqual(wgt_zp.int32_data[0], self.default_zp_scales["WGT"][0])
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype)
        self.assertEqual(wgt_sc.float_data[0], self.default_zp_scales["WGT"][1])

        self.assertEqual(bias_zp.int32_data[0], self.default_zp_scales["BIAS"][0])
        self.assertEqual(bias_zp.data_type, self.default_bias_qtype)
        np_array = onnx.numpy_helper.to_array(bias_sc)
        self.assertEqual(np_array[0], self.default_zp_scales["BIAS"][1])

        self.assertEqual(out_zp.int32_data[0], self.default_zp_scales["OUT"][0])
        self.assertEqual(out_zp.data_type, self.default_act_qtype)
        self.assertEqual(out_sc.float_data[0], self.default_zp_scales["OUT"][1])

    def test_qdq_default_per_channel(self):
        """
        Test default per-channel behavior without specifying the TensorQuantOverrides option.
        """
        (
            inp_zp,
            inp_sc,
            sig_out_zp,
            sig_out_sc,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            out_zp,
            out_sc,
        ) = self.perform_qdq_quantization(
            "model_default_per_channel_quant_overrides.onnx",
            extra_options=None,  # default behavior
            per_channel=True,
        )

        # No overrides set. Expect default values
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        self.assertEqual(sig_out_zp.int32_data[0], self.default_zp_scales["SIG_OUT"][0])
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)
        self.assertEqual(sig_out_sc.float_data[0], self.default_zp_scales["SIG_OUT"][1])

        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype_per_channel)
        for index, zp in enumerate(self.default_zp_scales_per_channel["WGT"][0]):
            self.assertEqual(wgt_zp.int32_data[index], zp)
        for index, scale in enumerate(self.default_zp_scales_per_channel["WGT"][1]):
            self.assertEqual(wgt_sc.float_data[index], scale)

        self.assertEqual(bias_zp.data_type, self.default_bias_qtype)

        num_bias_zps = len(self.default_zp_scales_per_channel["BIAS"][0])
        actual_bias_zps = struct.unpack(f"<{num_bias_zps}i", bias_zp.raw_data)
        for index, zp in enumerate(self.default_zp_scales_per_channel["BIAS"][0]):
            self.assertEqual(actual_bias_zps[index], zp)

        num_bias_scales = len(self.default_zp_scales_per_channel["BIAS"][1])
        actual_bias_scales = struct.unpack(f"<{num_bias_scales}f", bias_sc.raw_data)
        for index, scale in enumerate(self.default_zp_scales_per_channel["BIAS"][1]):
            self.assertEqual(actual_bias_scales[index], scale)

        self.assertEqual(out_zp.int32_data[0], self.default_zp_scales["OUT"][0])
        self.assertEqual(out_zp.data_type, self.default_act_qtype)
        self.assertEqual(out_sc.float_data[0], self.default_zp_scales["OUT"][1])

    def test_qdq_overrides1(self):
        """
        Test overriding:
          - scale/zp for Sigmoid output
          - quant_type, symmetric, reduce_range for Conv weight
          - quant_type, symmetric, reduce_range for Conv bias
        """
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, wgt_zp, wgt_sc, bias_zp, bias_sc, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides1.onnx",
            extra_options={
                "TensorQuantOverrides": {
                    "SIG_OUT": [
                        {"scale": np.array(1.0, dtype=np.float32), "zero_point": np.array(127, dtype=np.uint8)}
                    ],
                    "WGT": [{"quant_type": QuantType.QInt8, "symmetric": True, "reduce_range": True}],
                    "BIAS": [{"quant_type": QuantType.QInt8, "symmetric": True, "reduce_range": True}],
                }
            },
        )

        # Input should have same quant params
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        # Sigmoid output should have overridden scale/zp
        self.assertEqual(sig_out_zp.int32_data[0], 127)
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)
        self.assertEqual(sig_out_sc.float_data[0], np.float32(1.0))

        # Weight should have different type, zero_point, and scale
        self.assertEqual(wgt_zp.data_type, QuantType.QInt8.tensor_type)

        wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(wgt_zp.data_type, reduce_range=True, symmetric=True)
        wgt_rmin, wgt_rmax = np.min(self.weight), np.max(self.weight)
        new_wgt_zp, new_wgt_sc = compute_scale_zp(wgt_rmin, wgt_rmax, wgt_qmin, wgt_qmax, symmetric=True)
        self.assertEqual(wgt_zp.int32_data[0], new_wgt_zp)
        self.assertEqual(wgt_sc.float_data[0], np.float32(new_wgt_sc))

        # Bias should now be treated as a weight and should have different type, zero_point, and scale
        self.assertEqual(bias_zp.data_type, QuantType.QInt8.tensor_type)

        bias_qmin, bias_qmax = get_qmin_qmax_for_qType(bias_zp.data_type, reduce_range=True, symmetric=True)
        bias_rmin, bias_rmax = np.min(self.bias), np.max(self.bias)
        new_bias_zp, new_bias_sc = compute_scale_zp(bias_rmin, bias_rmax, bias_qmin, bias_qmax, symmetric=True)
        self.assertEqual(bias_zp.int32_data[0], new_bias_zp)
        self.assertEqual(bias_sc.float_data[0], np.float32(new_bias_sc))

    def test_qdq_overrides2(self):
        """
        Test overriding rmin/rmax for Sigmoid output.
        """
        sigmoid_rmin, sigmoid_rmax = np.array(0.0, dtype=np.float32), np.array(0.5, dtype=np.float32)
        inp_zp, inp_sc, sig_out_zp, sig_out_sc, _, _, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides2.onnx",
            extra_options={"TensorQuantOverrides": {"SIG_OUT": [{"rmin": sigmoid_rmin, "rmax": sigmoid_rmax}]}},
        )

        # Input should have same quant params
        self.assertEqual(inp_zp.int32_data[0], self.default_zp_scales["INP"][0])
        self.assertEqual(inp_zp.data_type, self.default_act_qtype)
        self.assertEqual(inp_sc.float_data[0], self.default_zp_scales["INP"][1])

        # Sigmoid output should have different scale/zp due to overridden rmin/rmax
        self.assertEqual(sig_out_zp.data_type, self.default_act_qtype)

        sigmoid_qmin, sigmoid_qmax = get_qmin_qmax_for_qType(sig_out_zp.data_type)
        new_sigmoid_zp, new_sigmoid_sc = compute_scale_zp(sigmoid_rmin, sigmoid_rmax, sigmoid_qmin, sigmoid_qmax)
        self.assertEqual(sig_out_zp.int32_data[0], new_sigmoid_zp)
        self.assertEqual(sig_out_sc.float_data[0], np.float32(new_sigmoid_sc))

    def test_qdq_overrides3(self):
        """
        Test overriding rmin and rmax for Conv weight
        """
        wgt_rmin, wgt_rmax = np.array(0.0, dtype=np.float32), np.array(1.0, dtype=np.float32)
        _, _, _, _, wgt_zp, wgt_sc, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides3.onnx",
            extra_options={
                "TensorQuantOverrides": {
                    "WGT": [{"rmin": wgt_rmin, "rmax": wgt_rmax}],
                }
            },
        )

        # Weight should have different zero_point and scale
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype)
        self.assertNotEqual(wgt_rmin, np.min(self.weight))
        self.assertNotEqual(wgt_rmax, np.max(self.weight))

        wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(wgt_zp.data_type)
        new_wgt_zp, new_wgt_sc = compute_scale_zp(wgt_rmin, wgt_rmax, wgt_qmin, wgt_qmax)
        self.assertEqual(wgt_zp.int32_data[0], new_wgt_zp)
        self.assertEqual(wgt_sc.float_data[0], np.float32(new_wgt_sc))

    def test_qdq_overrides4(self):
        """
        Test overriding scale and zero_point for Conv weight
        """
        wgt_zp_val, wgt_scale_val = np.array(4, dtype=np.float32), np.array(0.5, dtype=np.float32)
        _, _, _, _, wgt_zp, wgt_sc, _, _, _, _ = self.perform_qdq_quantization(
            "model_quant_overrides4.onnx",
            extra_options={
                "TensorQuantOverrides": {
                    "WGT": [{"zero_point": wgt_zp_val, "scale": wgt_scale_val}],
                }
            },
        )

        # Weight should have have the expected zero_point and scale
        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype)
        self.assertEqual(wgt_zp.int32_data[0], wgt_zp_val)
        self.assertEqual(wgt_sc.float_data[0], np.float32(wgt_scale_val))

    def test_qdq_overrides_per_channel1(self):
        """
        Test per-channel overriding of scale/zero_point for Conv weight and bias.
        """
        zp_vals, scale_vals = np.array([2, 4], dtype=np.float32), np.array([0.5, 0.2], dtype=np.float32)
        (
            _,
            _,
            _,
            _,
            wgt_zp,
            wgt_sc,
            bias_zp,
            bias_sc,
            _,
            _,
        ) = self.perform_qdq_quantization(
            "model_per_channel_quant_overrides1.onnx",
            extra_options={
                "TensorQuantOverrides": {
                    "WGT": [
                        {"axis": 0, "zero_point": zp_vals[0], "scale": scale_vals[0]},
                        {"zero_point": zp_vals[1], "scale": scale_vals[1]},
                    ],
                    "BIAS": [
                        {"axis": 0, "zero_point": zp_vals[0], "scale": scale_vals[0]},
                        {"zero_point": zp_vals[1], "scale": scale_vals[1]},
                    ],
                }
            },
            per_channel=True,
        )

        self.assertEqual(wgt_zp.data_type, self.default_wgt_qtype_per_channel)
        for index, zp in enumerate(zp_vals):
            self.assertEqual(wgt_zp.int32_data[index], zp)
        for index, scale in enumerate(scale_vals):
            self.assertEqual(wgt_sc.float_data[index], np.float32(scale))

        # NOTE: Bias with overrides is treated as a weight.
        self.assertEqual(bias_zp.data_type, self.default_wgt_qtype_per_channel)
        for index, zp in enumerate(zp_vals):
            self.assertEqual(bias_zp.int32_data[index], zp)
        for index, scale in enumerate(scale_vals):
            self.assertEqual(bias_sc.float_data[index], np.float32(scale))

    def test_qdq_overrides_per_channel2(self):
        """
        Test per-channel overriding of rmin, rmax, reduce_range, and quant_type for Conv weight.
        """
        for reduce_range in (False, True):
            with self.subTest(reduce_range=reduce_range):
                qdq_model_name = f"model_per_chan_overrides_2_reduce_range_{reduce_range}.onnx"
                rmin_vals = [0.0, 0.2]
                rmax_vals = [1.0, 0.8]
                quant_type = QuantType.QUInt8
                (
                    _,
                    _,
                    _,
                    _,
                    wgt_zp,
                    wgt_sc,
                    bias_zp,
                    bias_sc,
                    _,
                    _,
                ) = self.perform_qdq_quantization(
                    qdq_model_name,
                    extra_options={
                        "TensorQuantOverrides": {
                            "WGT": [
                                {
                                    "axis": 0,
                                    "quant_type": quant_type,
                                    "rmin": np.array(rmin_vals[0], dtype=np.float32),
                                    "rmax": np.array(rmax_vals[0], dtype=np.float32),
                                    "reduce_range": reduce_range,
                                },
                                {
                                    "quant_type": quant_type,
                                    "rmin": np.array(rmin_vals[1], dtype=np.float32),
                                    "rmax": np.array(rmax_vals[1], dtype=np.float32),
                                    "reduce_range": reduce_range,
                                },
                            ],
                        }
                    },
                    per_channel=True,
                )

                self.assertEqual(wgt_zp.data_type, quant_type.tensor_type)
                for index, (zp, scale) in enumerate(zip(wgt_zp.int32_data, wgt_sc.float_data, strict=False)):
                    wgt_qmin, wgt_qmax = get_qmin_qmax_for_qType(
                        wgt_zp.data_type,
                        symmetric=True,  # per-channel is always symmetric
                        reduce_range=reduce_range,
                    )
                    expected_zp, expected_scale = compute_scale_zp(
                        np.array(rmin_vals[index], dtype=np.float32),
                        np.array(rmax_vals[index], dtype=np.float32),
                        wgt_qmin,
                        wgt_qmax,
                        symmetric=True,  # per-channel is always symmetric
                    )
                    self.assertEqual(zp, expected_zp)
                    self.assertEqual(scale, np.float32(expected_scale))

    def test_16bit_overrides_bump_opset_to_21(self):
        """
        Test that overriding a tensor to 16-bit (when default is 8-bit) automatically bumps the model
        opset to 21 and emits native ai.onnx Q/DQ ops (not 'com.microsoft' domain ops).

        Previously (before the opset-bump heuristic), a sub-opset-21 model with INT16 overrides would
        use the 'com.microsoft' domain.  Now the model is auto-upgraded so the standard domain is used.
        """
        qdq_model_name = "model_quant_overrides_to_16bit.onnx"
        inp_zp, _, sig_out_zp, _, _, _, _, _, out_zp, _ = self.perform_qdq_quantization(
            qdq_model_name,
            activation_type=onnx.TensorProto.UINT8,  # Default to 8bit activations
            extra_options={
                "TensorQuantOverrides": {
                    "INP": [{"quant_type": QuantType.QUInt16}],
                    "SIG_OUT": [{"quant_type": QuantType.QUInt16}],
                }
            },
            opset=19,
        )

        # Input and Sigmoid's output should be overridden to 16bit
        self.assertEqual(inp_zp.data_type, onnx.TensorProto.UINT16)
        self.assertEqual(sig_out_zp.data_type, onnx.TensorProto.UINT16)

        # Output should be the default uint8 type
        self.assertEqual(out_zp.data_type, onnx.TensorProto.UINT8)

        # The model opset should have been auto-bumped to >= 21
        qdq_model = onnx.load_model(qdq_model_name)
        ai_onnx_opset = get_opset_version(qdq_model)
        self.assertGreaterEqual(ai_onnx_opset, 21)

        # Q/DQ ops should be in the default domain (NOT 'com.microsoft')
        for node in qdq_model.graph.node:
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                self.assertEqual(
                    node.domain,
                    "",
                    f"Expected native ONNX domain for {node.op_type} but got '{node.domain}'",
                )

    def test_16bit_overrides_not_set_ms_domain(self):
        """
        Test that overriding a tensor to 16bit (when default is 8bit) no longer automatically
        sets the 'com.microsoft' domain on DQ and Q ops for opset >= 21.
        Before ONNX 1.16.0, we had to use the 'com.microsoft' domain to be able to use 16-bit quantization.
        """
        qdq_model_name = "model_quant_overrides_to_16bit.onnx"
        inp_zp, _, sig_out_zp, _, _, _, _, _, out_zp, _ = self.perform_qdq_quantization(
            qdq_model_name,
            activation_type=onnx.TensorProto.UINT8,  # Default to 8bit activations
            extra_options={
                "TensorQuantOverrides": {
                    "INP": [{"quant_type": QuantType.QUInt16}],
                    "SIG_OUT": [{"quant_type": QuantType.QUInt16}],
                }
            },
            opset=21,
        )

        # Input and Sigmoid's output should be overridden to 16bit
        self.assertEqual(inp_zp.data_type, onnx.TensorProto.UINT16)
        self.assertEqual(sig_out_zp.data_type, onnx.TensorProto.UINT16)

        # Output should the default uint8 type
        self.assertEqual(out_zp.data_type, onnx.TensorProto.UINT8)

        # Q/DQ ops should be in the default domain (NOT 'com.microsoft')
        qdq_model = onnx.load_model(qdq_model_name)
        for node in qdq_model.graph.node:
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                self.assertEqual(
                    node.domain,
                    "",
                    f"Expected native ONNX domain for {node.op_type} but got '{node.domain}'",
                )

    def test_16bit_convert_quant_type_bumps_opset_to_21(self):
        """
        Regression test: a 16-bit type specified via the 'convert.quant_type' field inside
        TensorQuantOverrides should also trigger the opset-21 auto-bump, even when the top-level
        quant_type for that tensor is 8-bit.

        Verifies that the resulting model has ai.onnx opset >= 21 and that QuantizeLinear /
        DequantizeLinear nodes are in the default domain (not 'com.microsoft').
        """
        qdq_model_name = "model_quant_overrides_convert_16bit.onnx"
        inp_zp, _, sig_out_zp, _, _, _, _, _, out_zp, _ = self.perform_qdq_quantization(
            qdq_model_name,
            activation_type=onnx.TensorProto.UINT8,  # Default to 8bit activations
            extra_options={
                "TensorQuantOverrides": {
                    # quant_type is 8-bit; the 16-bit is only in the convert sub-dict
                    "INP": [{"quant_type": QuantType.QUInt8, "convert": {"quant_type": QuantType.QInt16}}],
                }
            },
            opset=20,
        )

        # INP primary quant type stays uint8
        self.assertEqual(inp_zp.data_type, onnx.TensorProto.UINT8)

        # Output should be the default uint8 type
        self.assertEqual(out_zp.data_type, onnx.TensorProto.UINT8)

        # The model opset should have been auto-bumped to >= 21 due to convert.quant_type = QInt16
        qdq_model = onnx.load_model(qdq_model_name)
        ai_onnx_opset = get_opset_version(qdq_model)
        self.assertGreaterEqual(ai_onnx_opset, 21)

        # Q/DQ ops should be in the default domain (NOT 'com.microsoft')
        for node in qdq_model.graph.node:
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
                self.assertEqual(
                    node.domain,
                    "",
                    f"Expected native ONNX domain for {node.op_type} but got '{node.domain}'",
                )

    def test_override_validation_nonexisting_tensor(self):
        """
        Test that specifying a non-existing tensor should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={
                    "TensorQuantOverrides": {
                        "NON_EXISTING": [
                            {"rmin": np.array(0.0, dtype=np.float32), "rmax": np.array(0.5, dtype=np.float32)}
                        ]
                    }
                },
            )

        self.assertIn("is not present in the model", str(context.exception))

    def test_override_validation_scale_missing_zp(self):
        """
        Test that specifying a scale without zero_point should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={"TensorQuantOverrides": {"SIG_OUT": [{"scale": np.array(0.0, dtype=np.float32)}]}},
            )

        self.assertIn("Must provide both 'scale' and 'zero_point'", str(context.exception))

    def test_override_validation_bad_combination(self):
        """
        Test that specifying a scale/zero_point with rmax/rmin/symmetric/reduce_range should fail.
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={
                    "TensorQuantOverrides": {
                        "SIG_OUT": [
                            {
                                "scale": np.array(0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.int8),
                                "rmax": np.array(10.0, dtype=np.float32),
                            }
                        ]
                    }
                },
            )

        self.assertIn("option(s) [rmax] are invalid with 'scale' and 'zero_point'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={
                    "TensorQuantOverrides": {
                        "SIG_OUT": [
                            {
                                "scale": np.array(0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.int8),
                                "rmax": np.array(10.0, dtype=np.float32),
                            }
                        ]
                    }
                },
            )

        self.assertIn("option(s) [rmax] are invalid with 'scale' and 'zero_point'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={
                    "TensorQuantOverrides": {
                        "SIG_OUT": [
                            {
                                "scale": np.array(0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.int8),
                                "symmetric": True,
                            }
                        ]
                    }
                },
            )

        self.assertIn("option(s) [symmetric] are invalid with 'scale' and 'zero_point'", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={
                    "TensorQuantOverrides": {
                        "SIG_OUT": [
                            {
                                "scale": np.array(0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.int8),
                                "reduce_range": True,
                            }
                        ]
                    }
                },
            )

        self.assertIn("option(s) [reduce_range] are invalid with 'scale' and 'zero_point'", str(context.exception))

    def test_override_validation_per_channel_mismatch(self):
        """
        Test that per-channel overrides that disagree between channels produce an
        error message with the tensor name and channel index actually filled in
        (and not the literal, uninterpolated `{tensor_name}`/`{index}` placeholders).
        """
        with self.assertRaises(ValueError) as context:
            self.perform_qdq_quantization(
                "model_validation.onnx",
                extra_options={
                    "TensorQuantOverrides": {
                        "WGT": [
                            {"axis": 0, "symmetric": True},
                            {"symmetric": False},
                        ]
                    }
                },
                per_channel=True,
            )

        self.assertIn("Channel symmetric value for tensor 'WGT' does not match at index 0.", str(context.exception))

    def test_get_qnn_qdq_config_sigmoid(self):
        """
        Test that the QNN-specific configs override the scale and zero-point of 16-bit Sigmoid.
        """
        # Create float model with a Abs --> Sigmoid
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Abs", ["input_0"], ["abs_out"], name="Abs_0"),
                onnx.helper.make_node("Sigmoid", ["abs_out"], ["output_0"], name="Sigmoid_0"),
            ],
            "sigmoid_graph",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (1, 2, 3))],
            [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (1, 2, 3))],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        float_model_path = "model.onnx"
        onnx.save_model(model, float_model_path)

        other_override_0 = {"abs_out": [{"symmetric": True}]}
        other_override_1 = {
            "abs_out": [
                {
                    "quant_type": QuantType.QUInt8,
                    "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"Sigmoid_0"}},
                }
            ]
        }
        other_override_2 = {
            "abs_out": [
                {
                    "quant_type": QuantType.QInt8,
                    "convert": {"quant_type": QuantType.QInt16, "recv_nodes": {"Sigmoid_0"}},
                }
            ]
        }

        # Enumerate subtests (default_act_qtype, sigmoid_out_qtype, other_override)
        subtest_configs = [
            (QuantType.QUInt16, None, {}),  # Sigmoid gets new scale/zp
            (QuantType.QUInt16, None, other_override_0),  # Sigmoid gets new scale/zp
            (QuantType.QInt16, None, {}),  # Sigmoid gets new scale/zp
            (QuantType.QInt16, None, other_override_0),  # Sigmoid gets new scale/zp
            (QuantType.QUInt8, QuantType.QUInt16, other_override_1),  # Sigmoid gets new scale/zp
            (QuantType.QInt8, QuantType.QInt16, other_override_2),  # Sigmoid gets new scale/zp
            (QuantType.QUInt8, None, other_override_0),  # Sigmoid DOES NOT gets new scale/zp
            (QuantType.QInt8, None, {}),  # Sigmoid DOES NOT gets new scale/zp
            (QuantType.QInt8, QuantType.QInt8, {}),  # Sigmoid DOES NOT gets new scale/zp
        ]

        # Test that Sigmoid's output scale and zp should be overridden for 16-bit Sigmoid.
        for default_act_qtype, sigmoid_out_qtype, abs_override in subtest_configs:
            with self.subTest(
                default_act_qtype=default_act_qtype, sigmoid_out_qtype=sigmoid_out_qtype, abs_override=abs_override
            ):
                init_overrides = {}
                init_overrides.update(abs_override)

                if sigmoid_out_qtype is not None:
                    init_overrides["output_0"] = [{"quant_type": sigmoid_out_qtype}]

                qnn_config = get_qnn_qdq_config(
                    float_model_path,
                    DummyDataReader([]),
                    activation_type=default_act_qtype,
                    init_overrides=(init_overrides if init_overrides else None),
                    add_qtype_converts=False,
                )

                self.assertEqual(set(qnn_config.op_types_to_quantize), {"Abs", "Sigmoid"})

                if default_act_qtype == QuantType.QUInt16 or sigmoid_out_qtype == QuantType.QUInt16:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("output_0", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["output_0"],
                        [
                            {
                                "quant_type": QuantType.QUInt16,
                                "scale": np.array(1.0 / 65536.0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.uint16),
                            }
                        ],
                    )
                elif default_act_qtype == QuantType.QInt16 or sigmoid_out_qtype == QuantType.QInt16:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("output_0", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["output_0"],
                        [
                            {
                                "quant_type": QuantType.QInt16,
                                "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.int16),
                            }
                        ],
                    )

    def test_get_qnn_qdq_config_tanh(self):
        """
        Test that the QNN-specific configs override the scale and zero-point of 16-bit Tanh.
        """

        # Create float model with a Abs --> Tanh
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Abs", ["input_0"], ["abs_out"], name="Abs_0"),
                onnx.helper.make_node("Tanh", ["abs_out"], ["output_0"], name="Tanh_0"),
            ],
            "tanh_graph",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (1, 2, 3))],
            [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (1, 2, 3))],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        float_model_path = "model.onnx"
        onnx.save_model(model, float_model_path)

        other_override_0 = {"abs_out": [{"symmetric": True}]}
        other_override_1 = {
            "abs_out": [
                {"quant_type": QuantType.QUInt8, "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"Tanh_0"}}}
            ]
        }
        other_override_2 = {
            "abs_out": [
                {"quant_type": QuantType.QInt8, "convert": {"quant_type": QuantType.QInt16, "recv_nodes": {"Tanh_0"}}}
            ]
        }

        # Enumerate subtests (default_act_qtype, tanh_out_qtype, other_override)
        subtest_configs = [
            (QuantType.QUInt16, None, {}),  # Tanh gets new scale/zp
            (QuantType.QUInt16, None, other_override_0),  # Tanh gets new scale/zp
            (QuantType.QInt16, None, {}),  # Tanh gets new scale/zp
            (QuantType.QInt16, None, other_override_0),  # Tanh gets new scale/zp
            (QuantType.QUInt8, QuantType.QUInt16, other_override_1),  # Tanh gets new scale/zp
            (QuantType.QInt8, QuantType.QInt16, other_override_2),  # Tanh gets new scale/zp
            (QuantType.QUInt8, None, other_override_0),  # Tanh DOES NOT gets new scale/zp
            (QuantType.QInt8, None, {}),  # Tanh DOES NOT gets new scale/zp
            (QuantType.QInt8, QuantType.QInt8, {}),  # Tanh DOES NOT gets new scale/zp
        ]

        # Test that Tanh's output scale and zp should be overridden for 16-bit Tanh.
        for default_act_qtype, tanh_out_qtype, abs_override in subtest_configs:
            with self.subTest(
                default_act_qtype=default_act_qtype, tanh_out_qtype=tanh_out_qtype, abs_override=abs_override
            ):
                init_overrides = {}
                init_overrides.update(abs_override)

                if tanh_out_qtype is not None:
                    init_overrides["output_0"] = [{"quant_type": tanh_out_qtype}]

                qnn_config = get_qnn_qdq_config(
                    float_model_path,
                    DummyDataReader([]),
                    activation_type=default_act_qtype,
                    init_overrides=(init_overrides if init_overrides else None),
                    add_qtype_converts=False,
                )

                self.assertEqual(set(qnn_config.op_types_to_quantize), {"Abs", "Tanh"})

                if default_act_qtype == QuantType.QUInt16 or tanh_out_qtype == QuantType.QUInt16:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("output_0", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["output_0"],
                        [
                            {
                                "quant_type": QuantType.QUInt16,
                                "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                                "zero_point": np.array(32768, dtype=np.uint16),
                            }
                        ],
                    )
                elif default_act_qtype == QuantType.QInt16 or tanh_out_qtype == QuantType.QInt16:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("output_0", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["output_0"],
                        [
                            {
                                "quant_type": QuantType.QInt16,
                                "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                                "zero_point": np.array(0, dtype=np.int16),
                            }
                        ],
                    )

    def test_get_qnn_qdq_config_matmul(self):
        """
        Test that the QNN-specific configs override MatMul's initializer input type to 8-bit if
        the other input is 16-bit and the default weight type is 8-bit.
        """
        # Create float model with a Abs --> MatMul
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Abs", ["input_0"], ["abs_0_out"], name="Abs_0"),
                onnx.helper.make_node("MatMul", ["abs_0_out", "weight"], ["matmul_0_out"], name="MatMul_0"),
                onnx.helper.make_node("Abs", ["matmul_0_out"], ["output_0"], name="Abs_1"),
            ],
            "matmul_graph",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (2, 3))],
            [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (2, 2))],
            initializer=[onnx.numpy_helper.from_array(np.random.random((3, 2)).astype(np.float32), "weight")],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        float_model_path = "model.onnx"
        onnx.save_model(model, float_model_path)

        q16_qtypes = {QuantType.QUInt16, QuantType.QInt16}
        q8_qtypes = {QuantType.QUInt8, QuantType.QInt8}
        symmetric_wgt_qtypes = {QuantType.QInt8, QuantType.QInt16}

        other_override_0 = {"output_0": [{"symmetric": True}]}
        other_override_1 = {
            "matmul_0_out": [
                {
                    "quant_type": QuantType.QUInt16,
                    "convert": {"quant_type": QuantType.QUInt8, "recv_nodes": {"Abs_1"}},
                }
            ]
        }
        other_override_2 = {
            "matmul_0_out": [
                {
                    "quant_type": QuantType.QInt16,
                    "convert": {"quant_type": QuantType.QInt8, "recv_nodes": {"Abs_1"}},
                }
            ]
        }
        convert_matmul_input = {
            "abs_0_out": [
                {
                    "quant_type": QuantType.QUInt8,
                    "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"MatMul_0"}},
                }
            ]
        }

        # Enumerate subtests (default_act_qtype, default_wgt_qtype, matmul_in_qtype, other_override)
        subtest_configs = [
            (QuantType.QUInt8, QuantType.QUInt8, None, {}),
            (QuantType.QUInt8, QuantType.QUInt8, QuantType.QUInt16, {}),
            (QuantType.QUInt8, QuantType.QUInt8, QuantType.QUInt16, other_override_0),
            (QuantType.QUInt8, QuantType.QUInt8, QuantType.QUInt16, other_override_1),
            (QuantType.QInt8, QuantType.QInt8, QuantType.QInt16, other_override_2),
            (QuantType.QUInt16, QuantType.QUInt8, None, other_override_0),
            (QuantType.QInt16, QuantType.QInt8, None, {}),
            (QuantType.QUInt16, QuantType.QUInt16, None, other_override_0),
            (QuantType.QInt16, QuantType.QInt16, None, {}),
            (QuantType.QUInt8, QuantType.QUInt8, None, {}),
            (QuantType.QUInt8, QuantType.QUInt8, None, convert_matmul_input),
        ]

        # Test if MatMul's weight input is overridden.
        for default_act_qtype, default_wgt_qtype, matmul_input_qtype, other_override in subtest_configs:
            with self.subTest(
                default_act_qtype=default_act_qtype,
                default_wgt_qtype=default_wgt_qtype,
                matmul_input_qtype=matmul_input_qtype,
                other_override=other_override,
            ):
                init_overrides = {}
                init_overrides.update(other_override)

                if matmul_input_qtype is not None:
                    init_overrides["abs_0_out"] = [{"quant_type": matmul_input_qtype}]

                qnn_config = get_qnn_qdq_config(
                    float_model_path,
                    DummyDataReader([]),
                    activation_type=default_act_qtype,
                    weight_type=default_wgt_qtype,
                    init_overrides=(init_overrides if init_overrides else None),
                    add_qtype_converts=False,
                )

                self.assertEqual(set(qnn_config.op_types_to_quantize), {"Abs", "MatMul"})
                input_is_16bit = (
                    (default_act_qtype in q16_qtypes)
                    or (matmul_input_qtype in q16_qtypes)
                    or (other_override == convert_matmul_input)
                )
                weight_is_symmetric = default_wgt_qtype in symmetric_wgt_qtypes

                if input_is_16bit and default_wgt_qtype in q8_qtypes:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("weight", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["weight"],
                        [
                            {
                                "quant_type": default_wgt_qtype,
                                "symmetric": weight_is_symmetric,
                            }
                        ],
                    )
                elif init_overrides:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertNotIn("weight", qnn_config.extra_options["TensorQuantOverrides"])

                self.assertEqual(weight_is_symmetric, qnn_config.extra_options["WeightSymmetric"])

    def test_get_qnn_qdq_config_matmul_per_channel(self):
        """
        When per_channel is enabled, test that the QNN-specific configs explicitly override MatMul's
        initializer inputs to use per-tensor quantization (QNN does not support per-channel MatMul).
        """
        # Create float model with a Abs --> MatMul
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Abs", ["input_0"], ["abs_0_out"], name="Abs_0"),
                onnx.helper.make_node("MatMul", ["abs_0_out", "weight"], ["matmul_0_out"], name="MatMul_0"),
                onnx.helper.make_node("Abs", ["matmul_0_out"], ["output_0"], name="Abs_1"),
            ],
            "matmul_graph",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (2, 3))],
            [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (2, 2))],
            initializer=[onnx.numpy_helper.from_array(np.random.random((3, 2)).astype(np.float32), "weight")],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        float_model_path = "model.onnx"
        onnx.save_model(model, float_model_path)

        symmetric_wgt_qtypes = {QuantType.QInt8, QuantType.QInt16}
        weight_override_16bit = {"weight": [{"quant_type": QuantType.QInt16, "symmetric": True}]}

        # Enumerate subtests (default_wgt_qtype, default_wgt_symmetric, other_override)
        subtest_configs = [
            (QuantType.QUInt8, False, {}),
            (QuantType.QInt8, True, {}),
            (QuantType.QUInt8, None, {}),
            (QuantType.QInt8, None, {}),
            (QuantType.QInt8, None, weight_override_16bit),
        ]

        # Test if MatMul's weight input is overridden to per-tensor correctly.
        for default_wgt_qtype, default_wgt_symmetric, other_override in subtest_configs:
            with self.subTest(
                default_wgt_qtype=default_wgt_qtype,
                default_wgt_symmetric=default_wgt_symmetric,
                other_override=other_override,
            ):
                init_overrides = {}
                init_overrides.update(other_override)

                qnn_config = get_qnn_qdq_config(
                    float_model_path,
                    DummyDataReader([]),
                    weight_type=default_wgt_qtype,
                    weight_symmetric=default_wgt_symmetric,
                    init_overrides=(init_overrides if init_overrides else None),
                    per_channel=True,
                )

                self.assertEqual(set(qnn_config.op_types_to_quantize), {"Abs", "MatMul"})
                weight_is_symmetric = default_wgt_symmetric or default_wgt_qtype in symmetric_wgt_qtypes

                # User did not provide overrides for weight, so get_qnn_qdq_config() should set per-tensor overrides.
                if not init_overrides:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("weight", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["weight"],
                        [
                            {
                                "quant_type": default_wgt_qtype,
                                "symmetric": weight_is_symmetric,
                            }
                        ],
                    )
                else:
                    # Should retain user's overrides.
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("weight", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["weight"], weight_override_16bit["weight"]
                    )

    def test_get_qnn_qdq_config_layernorm(self):
        """
        Test that the QNN-specific configs override LayerNorm's initializer input type to 8-bit if
        the other input is 16-bit and the default weight type is 8-bit.
        """
        # Create float model with a Abs --> LayerNormalization
        graph = onnx.helper.make_graph(
            [
                onnx.helper.make_node("Abs", ["input_0"], ["abs_0_out"], name="Abs_0"),
                onnx.helper.make_node(
                    "LayerNormalization", ["abs_0_out", "weight", "bias"], ["layernorm_0_out"], name="LayerNorm_0"
                ),
                onnx.helper.make_node("Abs", ["layernorm_0_out"], ["output_0"], name="Abs_1"),
            ],
            "layernorm_graph",
            [onnx.helper.make_tensor_value_info("input_0", onnx.TensorProto.FLOAT, (2, 3))],
            [onnx.helper.make_tensor_value_info("output_0", onnx.TensorProto.FLOAT, (2, 3))],
            initializer=[
                onnx.numpy_helper.from_array(np.random.random((2, 3)).astype(np.float32), "weight"),
                onnx.numpy_helper.from_array(np.random.random((2, 3)).astype(np.float32), "bias"),
            ],
        )
        opset_imports = [
            onnx.helper.make_opsetid("", 18),
        ]
        model = onnx.helper.make_model(graph, opset_imports=opset_imports)
        model = onnx.shape_inference.infer_shapes(model)
        float_model_path = "model.onnx"
        onnx.save_model(model, float_model_path)

        q16_qtypes = {QuantType.QUInt16, QuantType.QInt16}
        q8_qtypes = {QuantType.QUInt8, QuantType.QInt8}
        symmetric_wgt_qtypes = {QuantType.QInt8, QuantType.QInt16}

        other_override_0 = {"output_0": [{"symmetric": True}]}
        other_override_1 = {
            "layernorm_0_out": [
                {
                    "quant_type": QuantType.QUInt16,
                    "convert": {"quant_type": QuantType.QUInt8, "recv_nodes": {"Abs_1"}},
                }
            ]
        }
        other_override_2 = {
            "layernorm_0_out": [
                {
                    "quant_type": QuantType.QInt16,
                    "convert": {"quant_type": QuantType.QInt8, "recv_nodes": {"Abs_1"}},
                }
            ]
        }
        convert_layernorm_input = {
            "abs_0_out": [
                {
                    "quant_type": QuantType.QUInt8,
                    "convert": {"quant_type": QuantType.QUInt16, "recv_nodes": {"LayerNorm_0"}},
                }
            ]
        }

        # Enumerate subtests (default_act_qtype, default_wgt_qtype, layernorm_in_qtype, other_override)
        subtest_configs = [
            (QuantType.QUInt8, QuantType.QUInt8, None, {}),
            (QuantType.QUInt8, QuantType.QUInt8, QuantType.QUInt16, {}),
            (QuantType.QUInt8, QuantType.QUInt8, QuantType.QUInt16, other_override_0),
            (QuantType.QUInt8, QuantType.QUInt8, QuantType.QUInt16, other_override_1),
            (QuantType.QInt8, QuantType.QInt8, QuantType.QInt16, other_override_2),
            (QuantType.QUInt16, QuantType.QUInt8, None, other_override_0),
            (QuantType.QInt16, QuantType.QInt8, None, {}),
            (QuantType.QUInt16, QuantType.QUInt16, None, other_override_0),
            (QuantType.QInt16, QuantType.QInt16, None, {}),
            (QuantType.QUInt8, QuantType.QUInt8, None, {}),
            (QuantType.QUInt8, QuantType.QUInt8, None, convert_layernorm_input),
        ]

        # Test if LayerNorm's weight input is overridden.
        for default_act_qtype, default_wgt_qtype, layernorm_input_qtype, other_override in subtest_configs:
            with self.subTest(
                default_act_qtype=default_act_qtype,
                default_wgt_qtype=default_wgt_qtype,
                layernorm_input_qtype=layernorm_input_qtype,
                other_override=other_override,
            ):
                init_overrides = {}
                init_overrides.update(other_override)

                if layernorm_input_qtype is not None:
                    init_overrides["abs_0_out"] = [{"quant_type": layernorm_input_qtype}]

                qnn_config = get_qnn_qdq_config(
                    float_model_path,
                    DummyDataReader([]),
                    activation_type=default_act_qtype,
                    weight_type=default_wgt_qtype,
                    init_overrides=(init_overrides if init_overrides else None),
                    add_qtype_converts=False,
                )

                self.assertEqual(set(qnn_config.op_types_to_quantize), {"Abs", "LayerNormalization"})
                input_is_16bit = (
                    (default_act_qtype in q16_qtypes)
                    or (layernorm_input_qtype in q16_qtypes)
                    or (other_override == convert_layernorm_input)
                )
                weight_is_symmetric = default_wgt_qtype in symmetric_wgt_qtypes

                if input_is_16bit and default_wgt_qtype in q8_qtypes:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertIn("weight", qnn_config.extra_options["TensorQuantOverrides"])
                    self.assertEqual(
                        qnn_config.extra_options["TensorQuantOverrides"]["weight"],
                        [
                            {
                                "quant_type": default_wgt_qtype,
                                "symmetric": weight_is_symmetric,
                            }
                        ],
                    )
                elif init_overrides:
                    self.assertIn("TensorQuantOverrides", qnn_config.extra_options)
                    self.assertNotIn("weight", qnn_config.extra_options["TensorQuantOverrides"])

                self.assertEqual(weight_is_symmetric, qnn_config.extra_options["WeightSymmetric"])
                self.assertNotIn("bias", qnn_config.extra_options["TensorQuantOverrides"])

    def test_get_qnn_qdq_config_ext_data(self):
        """
        Test that get_qnn_qdq_config() returns a config that enables external data
        if the input model has external data.
        """

        # Create model with a weight large enough (> 1024 bytes) to be stored externally.
        large_weight = onnx.numpy_helper.from_array(np.random.random((1, 32, 32)).astype(np.float32), "weight")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Add", ["input", "weight"], ["output"])],
            "add_ext_data",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (1, 32, 32))],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (1, 32, 32))],
            initializer=[large_weight],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 18)],
        )

        # Remove potential leftover external data file.
        if os.path.exists("add_ext_data.bin"):
            os.remove("add_ext_data.bin")

        onnx.save_model(
            model,
            "add_ext_data.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="add_ext_data.bin",
        )

        qnn_config = get_qnn_qdq_config("add_ext_data.onnx", DummyDataReader(self.activations))
        self.assertEqual(set(qnn_config.op_types_to_quantize), {"Add"})
        self.assertTrue(qnn_config.use_external_data_format)

    def test_get_qnn_qdq_config_ext_data_separate_dir(self):
        """
        Test that get_qnn_qdq_config() can validate per-channel quantization overrides for a model with external data
        that is in a separate directory not in the cwd.
        """

        # Create model with a weight large enough (> 1024 bytes) to be stored externally.
        large_weight = onnx.numpy_helper.from_array(np.random.random((1, 2, 32, 32)).astype(np.float32), "weight")
        graph = onnx.helper.make_graph(
            [onnx.helper.make_node("Conv", ["input", "weight"], ["output"])],
            "conv_ext_data",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (1, 2, 64, 64))],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, None)],
            initializer=[large_weight],
        )
        model = onnx.helper.make_model(
            graph,
            opset_imports=[onnx.helper.make_opsetid("", 21)],
        )

        # Make a separate directory in which to save model and its external data.
        model_dir_path = tempfile.mkdtemp(prefix="model_ext_data")
        model_name = "conv_ext_data.onnx"
        model_path = os.path.join(model_dir_path, model_name)

        onnx.save_model(
            model,
            str(model_path),
            save_as_external_data=True,
        )

        # Use tensor quantization overrides to quantize Conv's weight input to 4 bits on axis 0.
        init_overrides = {"weight": [{"quant_type": QuantType.QInt4, "axis": 0, "symmetric": True}]}

        # get_qnn_qdq_config() should be able to validate the per-channel axis without having to load
        # the external weight data.
        qnn_config = get_qnn_qdq_config(
            str(model_path),
            DummyDataReader([]),
            init_overrides=init_overrides,  # Dummy data reader does nothing
        )
        self.assertEqual(set(qnn_config.op_types_to_quantize), {"Conv"})
        self.assertTrue(qnn_config.use_external_data_format)


if __name__ == "__main__":
    t = TestTensorQuantOverridesOption()
    t.setUp()
    t.test_qdq_default_per_channel()
    unittest.main()

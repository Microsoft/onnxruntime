# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
#
# QMoE quantization implementation notes:
#
# Both CPU and CUDA implementations use symmetric quantization centered around 0:
# - 4-bit: range [-8, 7] with no zero-point (symmetric around 0)
# - 8-bit: range [-128, 127] with no zero-point (symmetric around 0)
#
# This follows the _symmetric_quantize_last_axis_of_batched_matrix pattern.
# Tolerance values account for numerical differences between implementations.
#
# Routing Logic:top-k selection first, then softmax
# normalization on the selected experts. This provides proper weight distribution
# while maintaining computational efficiency.
# --------------------------------------------------------------------------
import copy
import json
import os
import time
import unittest
from collections import OrderedDict
from contextlib import nullcontext

import numpy
import torch
import torch.nn.functional as F
from cuda_plugin_ep_helper import resolve_cuda_plugin_ep
from onnx import helper
from parameterized import parameterized
from torch import nn

import onnxruntime
from onnxruntime.capi import _pybind_state as _pybind
from onnxruntime.quantization import CudaQuantizer

try:
    import nvtx

    has_nvtx = True
except ImportError:
    has_nvtx = False
    nvtx = None

try:
    from onnx import TensorProto

    has_onnx = True
except ImportError:
    has_onnx = False

    class TensorProtoPlaceholder:
        FLOAT16 = 10
        FLOAT = 1
        BFLOAT16 = 16


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}
ACT2FN = ClassInstantier(ACT2CLS)

if not has_onnx:

    class TensorProtoPlaceholder:
        FLOAT16 = 10
        FLOAT = 1
        UINT8 = 2
        BFLOAT16 = 16

    TensorProto = TensorProtoPlaceholder

onnxruntime.preload_dlls()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

if torch.cuda.is_available():
    ort_provider = ["CUDAExecutionProvider"]
else:
    ort_provider = ["CPUExecutionProvider"]


def _qmoe_benchmark_nvtx_range(name="benchmark", color="green"):
    if os.getenv("ORT_QMOE_GEMV_BENCHMARK_NVTX") != "1":
        return nullcontext()

    if not has_nvtx:
        return nullcontext()

    return nvtx.annotate(name, color=color)


torch.manual_seed(42)
numpy.random.seed(42)

onnx_to_torch_type_map = {
    TensorProto.FLOAT16: torch.float16,
    TensorProto.FLOAT: torch.float,
    TensorProto.BFLOAT16: torch.bfloat16,
    TensorProto.UINT8: torch.uint8,
}

ort_to_numpy_type_map = {
    TensorProto.FLOAT16: numpy.float16,
    TensorProto.FLOAT: numpy.float32,
    TensorProto.UINT8: numpy.uint8,
}

ort_dtype_name_map = {
    TensorProto.FLOAT16: "FP16",
    TensorProto.FLOAT: "FP32",
    TensorProto.BFLOAT16: "BF16",
}


def print_diff_statistics(diff_tensor: torch.Tensor, prefix: str = ""):
    """
    Print percentile statistics (75%, 95%, 99%) for a difference tensor.
    This helps assess parity quality beyond just max difference.

    Args:
        diff_tensor: Tensor containing absolute differences between expected and actual outputs.
        prefix: Optional prefix string for the output message.
    """
    diff_flat = diff_tensor.flatten().float()
    if diff_flat.numel() == 0:
        print(f"{prefix}Diff statistics: empty tensor")
        return

    # Compute percentiles
    sorted_diff, _ = torch.sort(diff_flat)
    n = sorted_diff.numel()

    p75_idx = min(int(n * 0.75), n - 1)
    p95_idx = min(int(n * 0.95), n - 1)
    p99_idx = min(int(n * 0.99), n - 1)

    p75 = sorted_diff[p75_idx].item()
    p95 = sorted_diff[p95_idx].item()
    p99 = sorted_diff[p99_idx].item()
    max_val = sorted_diff[-1].item()
    mean_val = diff_flat.mean().item()

    print(
        f"{prefix}Diff stats - mean: {mean_val:.6f}, p75: {p75:.6f}, p95: {p95:.6f}, p99: {p99:.6f}, max: {max_val:.6f}"
    )


def quant_dequant_blockwise(weights, block_size, is_4_bit_quantization: bool = True, asymmetric: bool = False):
    bits = 4 if is_4_bit_quantization else 8
    n, k = weights.shape
    block_per_k = (k + block_size - 1) // block_size
    is_symmetric = not asymmetric

    q_weight, scale, zero_point = CudaQuantizer.matmulnbits_blockwise_quantize(
        weights,
        bits,
        block_size,
        symmetric=is_symmetric,
        return_zero_points=True,
        abs_scales=is_symmetric,
        flatten_qweight=False,
    )
    processed_q_weight, _ = CudaQuantizer.qmoe_prepacked_blockwise_quantize(
        weights,
        bits,
        block_size,
        symmetric=is_symmetric,
        abs_scales=is_symmetric,
    )

    scale_torch = scale.to(weights.device).unsqueeze(-1)
    q_weight_torch = q_weight.to(weights.device)

    if is_4_bit_quantization:
        q_low = q_weight_torch & 0x0F
        q_high = (q_weight_torch >> 4) & 0x0F
        q_unpacked = torch.stack((q_low, q_high), dim=-1).view(n, block_per_k, -1)[:, :, :block_size]
        q_unpacked = q_unpacked.to(weights.dtype)
        if is_symmetric:
            dequantized = (q_unpacked - 8.0) * scale_torch
        else:
            zp_torch = zero_point.to(weights.device)
            zp_low = zp_torch & 0x0F
            zp_high = (zp_torch >> 4) & 0x0F
            zp_unpacked = torch.stack((zp_low, zp_high), dim=-1).flatten(1, 2)[:, :block_per_k]
            zp_unpacked = zp_unpacked.contiguous().view(n, block_per_k, 1).to(weights.dtype)
            dequantized = (q_unpacked - zp_unpacked) * scale_torch
    else:
        q_unpacked = q_weight_torch.to(weights.dtype)
        if is_symmetric:
            dequantized = (q_unpacked - 128.0) * scale_torch
        else:
            zp_torch = zero_point.to(weights.device).to(weights.dtype).unsqueeze(-1)
            dequantized = (q_unpacked - zp_torch) * scale_torch

    scale_torch_out = scale.to(weights.device).to(torch.float16)
    processed_q_weight_torch = processed_q_weight.to(weights.device).view(torch.uint8)
    result = dequantized.view(n, k)

    # Symmetric quantization carries no zero-points for either width. The uint8/uint4 storage is
    # offset-binary (centred on 128 / 8), and that offset is baked into the weight converter --
    # FastInterleavedAndBiasedNumericArrayConverter subtracts 1152 = 1024 + 128 in fp16 math for
    # uint8, and 8 for uint4 -- so it is applied whether or not zero-points are supplied. This
    # function used to emit an all-128 tensor for symmetric 8-bit on the theory that "Cutlass
    # expects an explicit Zero Point = 128"; that produced bias = (128 - 128) * scale = 0, a no-op
    # that silently kept every block-wise 8-bit config off the MoE GEMV, whose dispatch rejects any
    # block-wise case with a non-null zeros pointer.
    zero_points_storage = zero_point.to(weights.device).to(torch.uint8) if asymmetric else None

    return scale_torch_out, processed_q_weight_torch, result, zero_points_storage


def _dequantize_unsigned_per_channel_storage(qweight, scales, weights, bits: int):
    n, k = weights.shape
    if bits == 4:
        q_low = qweight & 0x0F
        q_high = (qweight >> 4) & 0x0F
        quantized = torch.stack((q_low, q_high), dim=-1).view(n, -1)[:, :k].to(torch.int16)
        quantized = quantized - 8
    else:
        quantized = qweight.view(n, k).to(torch.int16)
        quantized = quantized - 128

    return quantized.to(weights.device).to(weights.dtype) * scales.to(weights.device).to(weights.dtype).unsqueeze(-1)


def quant_dequant(weights, is_4_bit_quantization: bool = True, asymmetric: bool = False):
    """
    Quantize and dequantize weights for testing purposes.
    Supports symmetric (default) and asymmetric quantization.

    Returns:
        scale, quantized_storage, dequantized, zero_point_storage
    """
    block_size = weights.shape[1]
    if not asymmetric and block_size > 256:
        bits = 4 if is_4_bit_quantization else 8
        qweight, scales = CudaQuantizer.qmoe_symmetric_per_channel_quantize(
            weights,
            bits,
        )
        processed_q_weight, _ = CudaQuantizer.qmoe_per_channel_quantize(
            weights,
            bits,
            True,
        )
        dequantized = _dequantize_unsigned_per_channel_storage(qweight, scales, weights, bits)
        scales = scales.to(weights.device).to(torch.float16).unsqueeze(-1)
        return scales, processed_q_weight.to(weights.device), dequantized, None

    return quant_dequant_blockwise(weights, block_size, is_4_bit_quantization, asymmetric)


def create_moe_onnx_graph(
    hidden_size,
    sequence_length,
    num_experts,
    top_k,
    intermediate_size,
    torch_dtype,
    onnx_dtype,
    fc1_experts_weights,
    fc2_experts_weights,
    fc1_bias=None,
    fc2_bias=None,
    fc1_scales=None,
    fc2_scales=None,
    fc1_zero_points=None,
    fc2_zero_points=None,
    use_swiglu=False,
    use_quant=False,
    quant_bits=4,
    swiglu_fusion=0,
    block_size=0,
):
    if not has_onnx:
        return None

    inter_size = intermediate_size
    topk = top_k

    if fc1_scales is None and use_quant:
        return None
    if fc2_scales is None and use_quant:
        return None
    if not has_onnx:
        return None

    if use_quant:
        assert fc1_experts_weights.dtype == torch.uint8, "FC1 weights must be uint8 for QMoE"
        assert fc2_experts_weights.dtype == torch.uint8, "FC2 weights must be uint8 for QMoE"
        assert fc1_scales is not None, "FC1 scales must be provided for QMoE"
        assert fc2_scales is not None, "FC2 scales must be provided for QMoE"

        # Accept float16 or float32 scales; tests may produce float32 for better precision
        assert fc1_scales.dtype in (torch.float16, torch.float32), "FC1 scales must be float16 or float32 for QMoE"
        assert fc2_scales.dtype in (torch.float16, torch.float32), "FC2 scales must be float16 or float32 for QMoE"

    if not has_onnx:
        return None

    # Set operator name and inputs based on quantization mode
    if use_quant:
        op_name = "QMoE"
        # Match the 14-input schema
        inputs = [
            "input",  # 0
            "router_probs",  # 1
            "fc1_experts_weights",  # 2
            "fc1_scales",  # 3
            "fc1_experts_bias" if fc1_bias is not None else "",  # 4
            "fc2_experts_weights",  # 5
            "fc2_scales",  # 6
            "fc2_experts_bias" if fc2_bias is not None else "",  # 7
            "",  # 8: fc3_weights
            "",  # 9: fc3_scales
            "",  # 10: fc3_bias
            "fc1_zero_points" if fc1_zero_points is not None else "",  # 11
            "fc2_zero_points" if fc2_zero_points is not None else "",  # 12
            "",  # 13: fc3_zero_points
        ]
    else:
        # For regular (non-quantized) MoE, use different operator and input layout
        op_name = "MoE"  # Regular MoE operator
        inputs = [
            "input",
            "router_probs",
            "fc1_experts_weights",
            "fc1_experts_bias" if fc1_bias is not None else "",  # fc1_bias as input 3
            "fc2_experts_weights",
            "fc2_experts_bias" if fc2_bias is not None else "",  # fc2_bias as input 5
            "",  # fc3_experts_weights (not used)
            "",  # fc3_experts_bias (not used)
        ]

    activation = "swiglu" if use_swiglu else "silu"

    nodes = [
        helper.make_node(
            op_name,
            inputs,
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            activation_type=activation,
            # Add new attributes with backwards-compatible default values
            swiglu_fusion=swiglu_fusion,
            swiglu_limit=7.0,
            activation_alpha=1.702,
            activation_beta=1.0,
            domain="com.microsoft",
        ),
    ]

    if use_quant:
        nodes[0].attribute.extend([helper.make_attribute("expert_weight_bits", quant_bits)])

    # Add block_size attribute for block-wise quantization
    if block_size > 0:
        nodes[0].attribute.extend([helper.make_attribute("block_size", block_size)])

    # Weights are store in column major order. Need pack 2 int4 values into uint8.
    # Use the actual tensor shapes instead of calculating them to avoid size mismatches
    fc1_shape = list(fc1_experts_weights.shape)
    fc2_shape = list(fc2_experts_weights.shape)

    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    weight_numpy_type = numpy.uint8 if use_quant else ort_to_numpy_type_map[onnx_dtype]
    weight_onnx_type = TensorProto.UINT8 if use_quant else onnx_dtype

    # Use raw bytes from C-contiguous numpy arrays to ensure the exact memory layout
    # of the packed uint8 weight tensors is preserved when writing the ONNX initializer.
    fc1_np = fc1_experts_weights.detach().cpu().numpy().astype(weight_numpy_type)
    fc2_np = fc2_experts_weights.detach().cpu().numpy().astype(weight_numpy_type)
    fc1_np = numpy.ascontiguousarray(fc1_np)
    fc2_np = numpy.ascontiguousarray(fc2_np)

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            weight_onnx_type,
            fc1_shape,
            fc1_np.tobytes(),
            raw=True,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            weight_onnx_type,
            fc2_shape,
            fc2_np.tobytes(),
            raw=True,
        ),
    ]

    if use_quant:
        if block_size > 0:
            fc1_blocks_per_row = (hidden_size + block_size - 1) // block_size
            fc2_blocks_per_row = (inter_size + block_size - 1) // block_size
            fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size, fc1_blocks_per_row]
            fc2_scale_shape = [num_experts, hidden_size, fc2_blocks_per_row]
        else:
            fc1_scale_shape = [num_experts, 2 * inter_size if use_swiglu else inter_size]
            fc2_scale_shape = [num_experts, hidden_size]

        scale_type = torch.float32 if onnx_dtype == TensorProto.BFLOAT16 else torch_dtype
        fc1_scale_val = fc1_scales.to(scale_type).flatten().detach().cpu().tolist()
        fc2_scale_val = fc2_scales.to(scale_type).flatten().detach().cpu().tolist()
        initializers.extend(
            [
                helper.make_tensor("fc1_scales", onnx_dtype, fc1_scale_shape, fc1_scale_val, raw=False),
                helper.make_tensor("fc2_scales", onnx_dtype, fc2_scale_shape, fc2_scale_val, raw=False),
            ]
        )

    # Add zero-point initializers if provided
    if fc1_zero_points is not None:
        fc1_zp_np = fc1_zero_points.detach().cpu().numpy().astype(numpy.uint8)
        fc1_zp_np = numpy.ascontiguousarray(fc1_zp_np)
        initializers.append(
            helper.make_tensor(
                "fc1_zero_points",
                TensorProto.UINT8,
                list(fc1_zero_points.shape),
                fc1_zp_np.tobytes(),
                raw=True,
            )
        )

    if fc2_zero_points is not None:
        fc2_zp_np = fc2_zero_points.detach().cpu().numpy().astype(numpy.uint8)
        fc2_zp_np = numpy.ascontiguousarray(fc2_zp_np)
        initializers.append(
            helper.make_tensor(
                "fc2_zero_points",
                TensorProto.UINT8,
                list(fc2_zero_points.shape),
                fc2_zp_np.tobytes(),
                raw=True,
            )
        )

    if fc1_bias is not None:
        if onnx_dtype == TensorProto.BFLOAT16:
            fc1_bias_val = fc1_bias.to(torch.float32).flatten().detach().cpu().tolist()
        else:
            fc1_bias_np = fc1_bias.detach().cpu().numpy().astype(ort_to_numpy_type_map[onnx_dtype])
            fc1_bias_val = fc1_bias_np.flatten().tolist()

        initializers.append(
            helper.make_tensor(
                "fc1_experts_bias",
                onnx_dtype,
                list(fc1_bias.shape),
                fc1_bias_val,
                raw=False,
            )
        )

    if fc2_bias is not None:
        if onnx_dtype == TensorProto.BFLOAT16:
            fc2_bias_val = fc2_bias.to(torch.float32).flatten().detach().cpu().tolist()
        else:
            fc2_bias_np = fc2_bias.detach().cpu().numpy().astype(ort_to_numpy_type_map[onnx_dtype])
            fc2_bias_val = fc2_bias_np.flatten().tolist()

        initializers.append(
            helper.make_tensor(
                "fc2_experts_bias",
                onnx_dtype,
                list(fc2_bias.shape),
                fc2_bias_val,
                raw=False,
            )
        )

    graph_inputs = [
        helper.make_tensor_value_info("input", onnx_dtype, [sequence_length, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            onnx_dtype,
            [sequence_length, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", onnx_dtype, [sequence_length, hidden_size]),
    ]

    graph = helper.make_graph(
        nodes,
        "MoE_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


class PhiMoEConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        hidden_act="silu",
        num_experts_per_tok=2,
        num_local_experts=8,
        router_jitter_noise=0.01,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_experts_per_tok = num_experts_per_tok
        self.num_local_experts = num_local_experts
        self.router_jitter_noise = router_jitter_noise


class SwigluMoeConfig:
    def __init__(
        self,
        hidden_size=4096,
        intermediate_size=14336,
        num_local_experts=8,
        num_experts_per_token=2,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts
        self.num_experts_per_token = num_experts_per_token


def swiglu(x: torch.Tensor, alpha: float = 1.702, limit: float = 7.0):
    dim = x.shape[-1]
    x = x.view(-1, dim // 2, 2)
    x_glu, x_linear = x[..., 0], x[..., 1]

    if limit is not None:
        x_glu = x_glu.clamp(max=limit)
        x_linear = x_linear.clamp(min=-limit, max=limit)

    y = x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)
    return y


class MoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class PhiMoEBlockSparseTop2MLP(MoEBlockSparseTop2MLP):
    def __init__(self, config: PhiMoEConfig):
        super().__init__(config)


class PhiMoESwiGLUMLP(nn.Module):
    """
    Phi3 MoE expert converted to 2-weight SwiGLU structure.
    This converts the traditional 3-weight Phi3 structure to SwiGLU format.
    """

    def __init__(self, config: PhiMoEConfig):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

        # Interleave w1 weights and biases to match the fused SwiGLU format
        with torch.no_grad():
            w = (
                self.w1.weight.data.view(2, self.intermediate_size, self.hidden_dim)
                .transpose(0, 1)
                .reshape(-1, self.hidden_dim)
            )
            self.w1.weight.data.copy_(w)
            b = self.w1.bias.data.view(2, self.intermediate_size).transpose(0, 1).reshape(-1)
            self.w1.bias.data.copy_(b)

    def forward(self, x):
        if x.dtype != self.w1.weight.dtype:
            x = x.to(self.w1.weight.dtype)
        x1 = self.w1(x)
        y = swiglu(x1)
        y = self.w2(y)
        return y


class SwigluMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def forward(self, x):
        if x.dtype != self.w1.weight.dtype:
            x = x.to(self.w1.weight.dtype)
        x1 = self.w1(x)
        y = swiglu(x1)
        y = self.w2(y)
        return y


def masked_sampling_omp_inference(scores, top_k, jitter_eps, training):
    """
    Updated to match the CUDA implementation's routing logic for fair comparison.
    This now uses the same complex jitter-based masking approach as the CUDA tests.
    """
    assert top_k == 2
    assert not training

    mask_logits_threshold, selected_experts = torch.topk(scores, 2)

    mask_logits_threshold_1 = mask_logits_threshold[:, 0].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_1)
    logits_mask = ((mask_logits_threshold_1 - scores) / factor) > (2 * jitter_eps)

    multiplier_1 = torch.softmax(scores.masked_fill(logits_mask, float("-inf")), dim=-1).gather(
        dim=-1, index=selected_experts[:, 0].unsqueeze(-1)
    )

    mask_logits_threshold_2 = mask_logits_threshold[:, 1].unsqueeze(-1)

    factor = scores.abs().clamp(min=mask_logits_threshold_2)
    logits_mask = ((mask_logits_threshold_2 - scores) / factor) > (2 * jitter_eps)

    multiplier_2 = torch.softmax(
        torch.scatter(scores, -1, selected_experts[:, 0].unsqueeze(-1), float("-inf")).masked_fill(
            logits_mask, float("-inf")
        ),
        dim=-1,
    ).gather(dim=-1, index=selected_experts[:, 1].unsqueeze(-1))

    multiplier = torch.concat((multiplier_1, multiplier_2), dim=-1)

    return (
        multiplier,
        selected_experts,
    )


class SparseMoeBlockORTHelper(nn.Module):
    def __init__(self, quant_bits=0, onnx_dtype=None, use_asymmetric_quant: bool = False):
        super().__init__()
        self.quant_bits = quant_bits
        self.onnx_dtype = onnx_dtype
        self.np_type = numpy.float16 if self.onnx_dtype == TensorProto.FLOAT16 else numpy.float32
        self.use_asymmetric_quant = use_asymmetric_quant

    def create_ort_session(self, moe_onnx_graph):
        if moe_onnx_graph is None:
            return None

        self.sess_options = onnxruntime.SessionOptions()
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        try:
            ort_session = onnxruntime.InferenceSession(
                moe_onnx_graph, self.sess_options, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
            )
        except Exception as e:
            print(f"ERROR: Failed to create ORT session: {e}")
            return None

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(
        self, hidden_states: torch.Tensor, enable_performance_test=False, enable_debug=False
    ) -> torch.Tensor:
        if self.ort_sess is None:
            print(f"ERROR: ORT session is None for {self.__class__.__name__}")
            return None

        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states_flat)

        if enable_debug:
            print(f"DEBUG: router_logits stats: mean={router_logits.mean():.6f}, std={router_logits.std():.6f}")
            print(
                f"DEBUG: hidden_states_flat stats: mean={hidden_states_flat.mean():.6f}, std={hidden_states_flat.std():.6f}"
            )

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype]

        tensors = {
            "input": hidden_states_flat.clone().to(device=device, dtype=torch_dtype),
            "router_probs": router_logits.clone().to(device=device, dtype=torch_dtype),
            "output": torch.zeros((batch_size * sequence_length, hidden_dim), device=device, dtype=torch_dtype),
        }

        iobinding = self.ort_sess.io_binding()

        for name, tensor in tensors.items():
            if name == "output":
                iobinding.bind_output(
                    name=name,
                    device_type=tensor.device.type,
                    device_id=tensor.device.index or 0,
                    element_type=self.onnx_dtype,
                    shape=tensor.shape,
                    buffer_ptr=tensor.data_ptr(),
                )
            else:
                iobinding.bind_input(
                    name=name,
                    device_type=tensor.device.type,
                    device_id=tensor.device.index or 0,
                    element_type=self.onnx_dtype,
                    shape=tensor.shape,
                    buffer_ptr=tensor.data_ptr(),
                )

        if enable_debug:
            print("DEBUG: About to run ORT inference...")

        iobinding.synchronize_inputs()
        self.ort_sess.run_with_iobinding(iobinding)
        iobinding.synchronize_outputs()

        if enable_debug:
            print("DEBUG: ORT inference completed successfully")

        if enable_performance_test:
            warmup = max(0, int(os.getenv("ORT_QMOE_GEMV_BENCHMARK_WARMUP", "5")))
            repeat = max(1, int(os.getenv("ORT_QMOE_GEMV_BENCHMARK_REPEATS", "100")))
            with _qmoe_benchmark_nvtx_range("warmup", "yellow"):
                for _ in range(warmup):
                    iobinding.synchronize_inputs()
                    self.ort_sess.run_with_iobinding(iobinding)
                    iobinding.synchronize_outputs()

            with _qmoe_benchmark_nvtx_range("benchmark", "green"):
                torch.cuda.synchronize()
                s = time.perf_counter()
                for _ in range(repeat):
                    iobinding.synchronize_inputs()
                    self.ort_sess.run_with_iobinding(iobinding)
                    iobinding.synchronize_outputs()
                torch.cuda.synchronize()
                e = time.perf_counter()
            time_ms = (e - s) / repeat * 1000
            self.last_ort_latency_ms = time_ms
            is_swiglu = hasattr(self, "use_swiglu") and self.use_swiglu
            is_interleaved = getattr(self, "swiglu_fusion", 0) == 1
            act_type = f"SwiGLU(interleaved={is_interleaved})" if is_swiglu else "SiLU"
            print(f"ORT Performance - {act_type} {self.quant_bits}-bit: {time_ms:.3f} ms/inference")

        return tensors["output"].reshape(batch_size, sequence_length, hidden_dim)

    def recreate_onnx_model(self):
        """Recreate the ONNX model with the current weights to reflect any changes to the quantization code."""

        w1_list, w2_list = [], []
        w1_bias_list, w2_bias_list = [], []
        w1_scale_list, w2_scale_list = [], []
        w1_zp_list, w2_zp_list = [], []

        is_4_bit = self.quant_bits == 4

        # Row-wise QMoE (block_size <= 0) does not support zero-points in CUDA kernel path.
        use_effective_asymmetric_quant = self.use_asymmetric_quant and self.block_size > 0
        for i in range(self.num_experts):
            if hasattr(self.experts[i], "w3"):
                w1, w3 = self.experts[i].w1.weight, self.experts[i].w3.weight
                w2 = self.experts[i].w2.weight
                w1_bias = self.experts[i].w1.bias
                w2_bias = self.experts[i].w2.bias
                w3_bias = getattr(self.experts[i].w3, "bias", None)

                # Combine and interleave w1 and w3 for the fused kernel
                w1_combined = torch.cat([w1, w3], dim=0)  # [2*inter, hidden]
                if getattr(self, "swiglu_fusion", 0) == 1:
                    w1_combined = w1_combined.view(2, -1, self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)

                if self.block_size > 0:
                    w1_scale, pre_qweight1, w1_qdq, w1_zp = quant_dequant_blockwise(
                        w1_combined, self.block_size, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                    w2_scale, pre_qweight2, w2_qdq, w2_zp = quant_dequant_blockwise(
                        w2, self.block_size, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                else:
                    w1_scale, pre_qweight1, w1_qdq, w1_zp = quant_dequant(
                        w1_combined, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                    w2_scale, pre_qweight2, w2_qdq, w2_zp = quant_dequant(
                        w2, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )

                if w1_bias is not None and w3_bias is not None:
                    b1_combined = torch.cat([w1_bias, w3_bias], dim=0)
                    if getattr(self, "swiglu_fusion", 0) == 1:
                        b1_combined = b1_combined.view(2, -1).transpose(0, 1).reshape(-1)
                    w1_bias_list.append(b1_combined.detach().cpu())
                elif w1_bias is not None:
                    w1_bias_list.append(w1_bias.detach().cpu())

                if w2_bias is not None:
                    w2_bias_list.append(w2_bias.detach().cpu())
            else:
                # PhiMoESwiGLUMLP already has interleaved weights in w1
                w1 = self.experts[i].w1.weight
                w2 = self.experts[i].w2.weight
                w1_bias = self.experts[i].w1.bias
                w2_bias = self.experts[i].w2.bias

                if self.block_size > 0:
                    w1_scale, pre_qweight1, w1_qdq, w1_zp = quant_dequant_blockwise(
                        w1, self.block_size, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                    w2_scale, pre_qweight2, w2_qdq, w2_zp = quant_dequant_blockwise(
                        w2, self.block_size, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                else:
                    w1_scale, pre_qweight1, w1_qdq, w1_zp = quant_dequant(
                        w1, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                    w2_scale, pre_qweight2, w2_qdq, w2_zp = quant_dequant(
                        w2, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                if w1_bias is not None:
                    w1_bias_list.append(w1_bias.detach().cpu())
                if w2_bias is not None:
                    w2_bias_list.append(w2_bias.detach().cpu())

            torch_dtype = onnx_to_torch_type_map[self.onnx_dtype] if self.onnx_dtype else torch.float32
            # For quantized MoE: keep expert weights in float32 so the PyTorch reference
            # computes in float32 (PhiMoESwiGLUMLP.forward casts input to weight dtype).
            # ORT's CUTLASS grouped GEMM and the decode GEMV kernel both accumulate the
            # weight*activation products in float32 before applying the FP16/BF16 scale, so a
            # float32 reference matches the kernel's accumulation precision. Storing weights in
            # the low-precision dtype causes catastrophic cancellation for near-zero outputs
            # (BF16's 7-bit / FP16's 10-bit mantissa) and makes the reference itself lossy.
            ref_weight_dtype = (
                torch.float32
                if (torch_dtype in (torch.bfloat16, torch.float16) and self.quant_bits > 0)
                else torch_dtype
            )

            if self.use_swiglu:
                if getattr(self, "swiglu_fusion", 0) == 1:
                    # In PhiMoESwiGLUMLP, w1 already contains interleaved gate and linear parts.
                    # We just need to update it with the quantized-dequantized weights.
                    self.experts[i].w1.weight.data = w1_qdq.contiguous().clone().to(ref_weight_dtype)
                else:
                    intermediate_size = self.experts[i].w1.weight.shape[0]
                    gate_dequant = w1_qdq[:intermediate_size].contiguous().clone().to(ref_weight_dtype)
                    value_dequant = w1_qdq[intermediate_size:].contiguous().clone().to(ref_weight_dtype)
                    if hasattr(self.experts[i], "w3"):
                        self.experts[i].w1.weight.data = gate_dequant
                        self.experts[i].w3.weight.data = value_dequant
                    else:
                        self.experts[i].w1.weight.data = w1_qdq.contiguous().clone().to(ref_weight_dtype)
            else:
                self.experts[i].w1.weight.data = w1_qdq.contiguous().clone().to(ref_weight_dtype)

            self.experts[i].w2.weight.data = w2_qdq.contiguous().clone().to(ref_weight_dtype)
            if ref_weight_dtype == torch.float32:
                # Also convert biases so F.linear sees consistent dtypes
                for attr in ("w1", "w2", "w3"):
                    linear_layer = getattr(self.experts[i], attr, None)
                    if linear_layer is not None and linear_layer.bias is not None:
                        linear_layer.bias.data = linear_layer.bias.data.float()

            # DEBUG
            # print(f"DEBUG: Expert {i} w1 dtype={self.experts[i].w1.weight.dtype}, w2 dtype={self.experts[i].w2.weight.dtype}")

            w1_list.append(pre_qweight1)
            w2_list.append(pre_qweight2)
            w1_scale_list.append(w1_scale)
            w2_scale_list.append(w2_scale)

            if self.block_size > 0 and w1_zp is not None:
                w1_zp_list.append(w1_zp)
            if self.block_size > 0 and w2_zp is not None:
                w2_zp_list.append(w2_zp)

        self.moe_experts_weight1 = torch.stack(w1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(w2_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(w1_scale_list, dim=0)
        moe_experts_weight_scale2 = torch.stack(w2_scale_list, dim=0)

        moe_experts_zp1 = torch.stack(w1_zp_list, dim=0) if len(w1_zp_list) > 0 else None
        moe_experts_zp2 = torch.stack(w2_zp_list, dim=0) if len(w2_zp_list) > 0 else None

        # Only squeeze for row-wise (non-blockwise) quantization where scales are [E, N, 1]
        if self.block_size <= 0:
            if moe_experts_weight_scale1.dim() == 3:
                moe_experts_weight_scale1 = moe_experts_weight_scale1.squeeze(-1)
            if moe_experts_weight_scale2.dim() == 3:
                moe_experts_weight_scale2 = moe_experts_weight_scale2.squeeze(-1)

        try:
            self.moe_onnx_graph = create_moe_onnx_graph(
                hidden_size=self.hidden_dim,
                sequence_length=self.batch_size * self.sequence_length,
                num_experts=self.num_experts,
                top_k=self.top_k,
                intermediate_size=self.ffn_dim,
                torch_dtype=torch.float32,
                onnx_dtype=self.onnx_dtype,
                fc1_experts_weights=self.moe_experts_weight1,
                fc2_experts_weights=self.moe_experts_weight2,
                # Pass collected biases
                fc1_bias=torch.stack(w1_bias_list, dim=0) if w1_bias_list else None,
                fc2_bias=torch.stack(w2_bias_list, dim=0) if w2_bias_list else None,
                # Scales are used for dequantization
                fc1_scales=moe_experts_weight_scale1,
                fc2_scales=moe_experts_weight_scale2,
                # Zero points are optional
                fc1_zero_points=moe_experts_zp1,
                fc2_zero_points=moe_experts_zp2,
                use_swiglu=self.use_swiglu,
                use_quant=True,  # Always use QMoE
                quant_bits=self.quant_bits,
                # We use swiglu_fusion=1 (fused and interleaved) based on the kernel implementation.
                # This matches the behavior of the Cutlass/MLAS kernels used in ORT. Tests may set
                # onnx_swiglu_fusion_override to emit a different attribute value (e.g. 0) while keeping
                # the interleaved weight layout, to exercise the kernel's backward-compat remap.
                swiglu_fusion=(
                    self.onnx_swiglu_fusion_override
                    if getattr(self, "onnx_swiglu_fusion_override", None) is not None
                    else getattr(self, "swiglu_fusion", 0)
                ),
                block_size=self.block_size,  # Add block_size for block-wise quantization
            )
        except Exception as e:
            print(f"Failed to create ONNX graph: {e}")
            self.moe_onnx_graph = None
            return False

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None
        return self.ort_sess is not None

    def parity_check(self, recreate_model=True):
        if recreate_model and not self.recreate_onnx_model():
            raise AssertionError("Model update failed")

        dtype = onnx_to_torch_type_map.get(self.onnx_dtype, torch.float32)
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device).to(dtype)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)

        if ort_output is None:
            raise AssertionError("ORT output is None")

        torch_has_nan = torch.isnan(torch_output).any()
        ort_has_nan = torch.isnan(ort_output).any()
        torch_has_inf = torch.isinf(torch_output).any()
        ort_has_inf = torch.isinf(ort_output).any()

        if torch_has_nan or ort_has_nan or torch_has_inf or ort_has_inf:
            torch_output_clean = torch.where(
                torch.isnan(torch_output) | torch.isinf(torch_output), torch.zeros_like(torch_output), torch_output
            )
            ort_output_clean = torch.where(
                torch.isnan(ort_output) | torch.isinf(ort_output), torch.zeros_like(ort_output), ort_output
            )
            max_diff = (torch_output_clean.cpu() - ort_output_clean.cpu()).abs().max()

            if (torch_has_nan and ort_has_nan) or (torch_has_inf and ort_has_inf):
                problematic_torch = torch.isnan(torch_output) | torch.isinf(torch_output)
                problematic_ort = torch.isnan(ort_output) | torch.isinf(ort_output)
                if torch.equal(problematic_torch, problematic_ort):
                    max_diff = 0.0
        else:
            max_diff = (torch_output.cpu() - ort_output.cpu()).abs().max()

        is_swiglu = hasattr(self, "use_swiglu") and self.use_swiglu
        is_interleaved = getattr(self, "swiglu_fusion", 0) == 1
        act_type = f"SwiGLU(interleaved={is_interleaved})" if is_swiglu else "SiLU"
        quant_type = "Asymmetric" if self.use_asymmetric_quant else "Symmetric"
        block_type = f"Block({self.block_size})" if self.block_size > 0 else "Row"

        print(f"Parity check - {act_type} {self.quant_bits}-bit {quant_type} {block_type}: max_diff = {max_diff:.6f}")

        # Print percentile statistics for better parity assessment
        diff = (torch_output.cpu() - ort_output.cpu()).abs()
        print_diff_statistics(diff, prefix=f"  [{act_type} {self.quant_bits}-bit {quant_type}] ")

        # Diagnostic dump: when differences are large, show the index and nearby values
        if max_diff > 1e-3:
            idx = torch.argmax(diff)
            flat_idx = int(idx)
            # Derive coordinates (batch, seq, hidden) from flattened index
            total_elems = torch_output.numel()
            # Work in flattened [batch, seq, hidden] ordering
            hidden_dim = self.hidden_dim
            seq = self.sequence_length
            # Clamp to safe bounds
            flat_idx = min(flat_idx, total_elems - 1)
            i = flat_idx // (hidden_dim)
            j = i // seq
            k = flat_idx % hidden_dim
            print(
                f"Diagnostic - max diff at flat_idx={flat_idx} -> sample (batch_idx={j}, seq_idx={i % seq}, hidden_idx={k})"
            )
            print("Torch sample:", torch_output.cpu().reshape(-1, hidden_dim)[i, k].item())
            print("ORT  sample:", ort_output.cpu().reshape(-1, hidden_dim)[i, k].item())
            # Print routing and per-expert contributions for this token from the PyTorch reference
            try:
                # Use float32 for diagnostic to avoid "unsupported ScalarType BFloat16" on some platforms/ops
                hidden_states_flat = hidden_state.view(-1, hidden_dim).float()
                token_vec = hidden_states_flat[i : i + 1]

                # Copy gate to CPU and float32 for reliable debug
                gate_cpu = copy.deepcopy(self.gate).cpu().float()
                gate_logits = gate_cpu(token_vec.cpu())

                topk_vals, topk_experts = torch.topk(gate_logits, self.top_k, dim=-1)
                topk_soft = F.softmax(topk_vals, dim=1)
                print("Gate logits:", gate_logits.detach().cpu().numpy())
                print("Selected experts:", topk_experts.detach().cpu().numpy())
                print("Routing weights:", topk_soft.detach().cpu().numpy())
                # Compute per-expert contributions for selected experts
                for idx_e, e in enumerate(topk_experts[0].tolist()):
                    expert_layer = copy.deepcopy(self.experts[e]).cpu().float()
                    expert_out = expert_layer(token_vec.cpu())
                    contrib = expert_out[0, k].item() * topk_soft[0, idx_e].item()
                    print(f"Expert {e} contrib at hidden {k}: {contrib}")
            except Exception:
                # Diagnostic dump is best-effort; ignore failures (e.g., unsupported dtype).
                pass

        ort_dtype_quant_bits_tolerance_map = {
            "FP32:0": (5e-3, 1e-3),
            "FP16:0": (5e-2, 1e-3),
            "FP16:4": (0.1, 0.01),
            "FP16:8": (0.1, 0.01),
            "FP32:4": (0.1, 0.01),
            "FP32:8": (0.1, 0.01),
            "BF16:4": (0.1, 0.02),
            "BF16:8": (0.1, 0.02),
        }

        dtype_str = ort_dtype_name_map[self.onnx_dtype]
        tolerance_key = f"{dtype_str}:{self.quant_bits}"
        if tolerance_key in ort_dtype_quant_bits_tolerance_map:
            base_atol, _rtol = ort_dtype_quant_bits_tolerance_map[tolerance_key]

            # Increase tolerance for asymmetric quantization due to different computation path
            if self.use_asymmetric_quant:
                base_atol *= 1.5

            if max_diff > base_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"tolerance {base_atol:.6f} for {tolerance_key} ({quant_type})"
                )
        else:
            fallback_atol = 0.1
            if self.use_asymmetric_quant:
                fallback_atol = 0.15

            if max_diff > fallback_atol:
                raise AssertionError(
                    f"QMoE parity check failed: max difference {max_diff:.6f} exceeds "
                    f"fallback tolerance {fallback_atol:.6f} for unknown config {tolerance_key} ({quant_type})"
                )

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim).to(device)
        self.ort_forward(hidden_state, enable_performance_test=True)


def small_test_cases():
    for batch_size in [1, 4]:
        for sequence_length in [32, 128]:
            yield batch_size, sequence_length


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    def __init__(
        self,
        config: SwigluMoeConfig,
        batch_size: int,
        sequence_length: int,
        quant_bits: int = 0,
        onnx_dtype=None,
        block_size: int = 0,
        use_asymmetric_quant: bool = False,
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype, use_asymmetric_quant=use_asymmetric_quant)
        self.swiglu_fusion = 1
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token
        self.use_swiglu = True
        self.swiglu_fusion = 1
        self.block_size = block_size

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype] if self.onnx_dtype else torch.float32

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True).to(device).to(torch_dtype)

        if self.swiglu_fusion == 1:
            self.experts = nn.ModuleList(
                [PhiMoESwiGLUMLP(config).to(device).to(torch_dtype) for _ in range(self.num_experts)]
            )
        else:
            self.experts = nn.ModuleList(
                [SwigluMlp(config).to(device).to(torch_dtype) for _ in range(self.num_experts)]
            )

            # Weight update and collection is handled in recreate_onnx_model

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.moe_onnx_graph = None
        self.ort_sess = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class PhiMoESparseMoeBlock(SparseMoeBlockORTHelper):
    def __init__(
        self,
        config: PhiMoEConfig,
        batch_size: int,
        sequence_length: int,
        quant_bits: int = 0,
        onnx_dtype=None,
        block_size: int = 0,
        use_asymmetric_quant: bool = False,
    ):
        super().__init__(quant_bits, onnx_dtype=onnx_dtype, use_asymmetric_quant=use_asymmetric_quant)
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.router_jitter_noise = config.router_jitter_noise
        self.use_swiglu = True
        self.swiglu_fusion = 1
        self.block_size = block_size
        use_quant = self.quant_bits > 0

        torch_dtype = onnx_to_torch_type_map[self.onnx_dtype] if self.onnx_dtype else torch.float32

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True).to(device).to(torch_dtype)
        self.experts = nn.ModuleList(
            [PhiMoESwiGLUMLP(config).to(device).to(torch_dtype) for _ in range(self.num_experts)]
        )

        fc1_w_list, fc2_w_list = [], []
        fc1_b_list, fc2_b_list = [], []
        scale_1_list, scale_2_list = [], []
        zp_1_list, zp_2_list = [], []

        use_effective_asymmetric_quant = self.use_asymmetric_quant and self.block_size > 0

        for expert in self.experts:
            fc1_b_list.append(expert.w1.bias)
            fc2_b_list.append(expert.w2.bias)
            if not use_quant:
                # Store original weights
                fc1_w_list.append(expert.w1.weight.detach())
                fc2_w_list.append(expert.w2.weight.detach())
                scale_1_list.append(torch.tensor(1.0))
                scale_2_list.append(torch.tensor(1.0))
            else:
                is_4_bit = self.quant_bits == 4

                if self.block_size > 0:
                    scale1, pre_qweight1, w1_qdq, zp1 = quant_dequant_blockwise(
                        expert.w1.weight, self.block_size, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                    scale2, pre_qweight2, w2_qdq, zp2 = quant_dequant_blockwise(
                        expert.w2.weight, self.block_size, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                else:
                    scale1, pre_qweight1, w1_qdq, zp1 = quant_dequant(
                        expert.w1.weight, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )
                    scale2, pre_qweight2, w2_qdq, zp2 = quant_dequant(
                        expert.w2.weight, is_4_bit, asymmetric=use_effective_asymmetric_quant
                    )

                # For quantized MoE: keep weights in float32 so the PyTorch reference computes
                # in float32, matching ORT's CUTLASS grouped GEMM and decode GEMV kernel that
                # both accumulate weight*activation products in float32 before applying the
                # FP16/BF16 scale. A low-precision reference is itself lossy and would mask the
                # kernel's accumulation precision.
                ref_weight_dtype = (
                    torch.float32
                    if (torch_dtype in (torch.bfloat16, torch.float16) and self.quant_bits > 0)
                    else torch_dtype
                )
                expert.w1.weight.data = w1_qdq.to(ref_weight_dtype)
                expert.w2.weight.data = w2_qdq.to(ref_weight_dtype)
                if ref_weight_dtype == torch.float32:
                    # Also convert biases so F.linear sees consistent dtypes
                    for linear_layer in [expert.w1, expert.w2]:
                        if linear_layer.bias is not None:
                            linear_layer.bias.data = linear_layer.bias.data.float()

                fc1_w_list.append(pre_qweight1)
                fc2_w_list.append(pre_qweight2)
                scale_1_list.append(scale1)
                scale_2_list.append(scale2)
                if self.block_size > 0 and zp1 is not None:
                    zp_1_list.append(zp1)
                if self.block_size > 0 and zp2 is not None:
                    zp_2_list.append(zp2)

        fc1_experts_weights = torch.stack(fc1_w_list, dim=0)
        fc2_experts_weights = torch.stack(fc2_w_list, dim=0)
        fc1_experts_bias = torch.stack(fc1_b_list, dim=0)
        fc2_experts_bias = torch.stack(fc2_b_list, dim=0)

        moe_experts_weight_scale1 = torch.stack(scale_1_list, dim=0) if use_quant else None
        moe_experts_weight_scale2 = torch.stack(scale_2_list, dim=0) if use_quant else None

        moe_experts_zp1 = torch.stack(zp_1_list, dim=0) if len(zp_1_list) > 0 else None
        moe_experts_zp2 = torch.stack(zp_2_list, dim=0) if len(zp_2_list) > 0 else None

        self.batch_size = batch_size
        self.sequence_length = sequence_length

        self.moe_onnx_graph = create_moe_onnx_graph(
            hidden_size=self.hidden_dim,
            sequence_length=self.batch_size * self.sequence_length,
            num_experts=self.num_experts,
            top_k=self.top_k,
            intermediate_size=self.ffn_dim,
            torch_dtype=torch.float32,
            onnx_dtype=self.onnx_dtype,
            fc1_experts_weights=fc1_experts_weights,
            fc2_experts_weights=fc2_experts_weights,
            fc1_bias=fc1_experts_bias,
            fc2_bias=fc2_experts_bias,
            fc1_scales=moe_experts_weight_scale1,
            fc2_scales=moe_experts_weight_scale2,
            fc1_zero_points=moe_experts_zp1,
            fc2_zero_points=moe_experts_zp2,
            use_swiglu=self.use_swiglu,
            use_quant=use_quant,
            quant_bits=self.quant_bits,
            swiglu_fusion=getattr(self, "swiglu_fusion", 0),
            block_size=self.block_size,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph) if self.moe_onnx_graph else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """PyTorch reference forward pass using SwiGLU-style routing"""
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)

        # Match ORT's LaunchSoftmaxTopK tie-breaking semantics.  ORT uses strict
        # `prob > row_scales[j]` insertion, which is equivalent to a stable sort
        # in descending order (lower original index wins on ties).  In low
        # precision dtypes such as bfloat16 distinct fp32 logits often round to
        # the same value, so torch.topk's unstable tie-breaking can pick a
        # different expert than ORT.
        sorted_vals, sorted_idx = torch.sort(router_logits, dim=-1, descending=True, stable=True)
        routing_weights_vals = sorted_vals[..., : self.top_k]
        selected_experts = sorted_idx[..., : self.top_k]
        routing_weights = F.softmax(routing_weights_vals, dim=1, dtype=torch.float)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


# Define test cases for different MoE types.
# NOTE: these all run with hidden_size=128 / intermediate_size=256, which is below the MoE GEMV's
# profiled minimum, so they exercise the CUTLASS grouped GEMM path. See moe_gemv_test_cases below
# for the shapes that select the GEMV.
phi3_test_cases = [
    (1, 1, 4),  # decode-sized INT4 per-channel
    (2, 1, 4),  # 2 tokens x top_k 2 = 4 expanded rows
    (1, 32, 4),
    (1, 32, 8),
    (2, 16, 4),
    (2, 16, 8),
]

# Define test cases for block-wise quantization
phi3_blockwise_test_cases = [
    (1, 1, 4, 32),  # tiny debug case for asymmetric ZP compensation
    (2, 1, 4, 32),  # multi-token decode shape
    (1, 32, 4, 32),  # batch_size, sequence_length, quant_bits, block_size
    (1, 32, 4, 64),
    (1, 32, 4, 128),
    (1, 32, 8, 32),
    (1, 32, 8, 64),
    (1, 32, 8, 128),
    (2, 16, 4, 32),
    (2, 16, 8, 32),
    (2, 16, 8, 64),
]
phi3_blockwise_asymmetric_test_cases = [
    (1, 1, 4, 32),
    (1, 1, 8, 32),
    (1, 32, 4, 64),
    (1, 32, 8, 64),
    (1, 32, 8, 128),
    (2, 16, 8, 64),
]

# These cases use expanded rows > 4 with K < 512, which is outside the profiled
# GEMV range and therefore exercises the CUTLASS grouped GEMM path.
qmoe_cutlass_gemm_blockwise_test_cases = [
    (1, 3, 4, 32),
    (1, 3, 8, 32),
]

qmoe_cutlass_gemm_second_scale_row_test_cases = [
    (4, False),
    (4, True),
    (8, False),
    (8, True),
]

# Cases that actually select the decode-time MoE GEMV. is_moe_gemv_supported() requires every GEMM
# dimension to be >= 512 (kMinProfiledProblemDim): FC1 sees n = 2 * inter_size, k = hidden_size and
# FC2 sees n = hidden_size, k = inter_size, so hidden_size and intermediate_size must both be >= 512
# (and multiples of 64 for the K-tile). Every other config in this file uses 128/256 and therefore
# never reaches the GEMV. Paired with num_experts_per_token=4, batch * seq of 1 and 2 gives 4 and 8
# expanded rows, covering both sides of kMaxProfiledExpandedRowsForSmallProblemDim. The 2-token cases
# are the ones that exercise the multi-token index math that degenerates at num_rows == 1: FC1's
# skip-expand source lookup (permuted_row_to_unpermuted_row % num_rows) and the per-token block
# indexing of the FC2 GEMV's fused finalize epilogue.
# (batch_size, sequence_length, quant_bits, block_size); block_size 0 means per-channel.
moe_gemv_test_cases = [
    (1, 1, 4, 0),
    (2, 1, 4, 0),
    (1, 1, 8, 0),
    (2, 1, 8, 0),
    (2, 1, 4, 32),
    (2, 1, 4, 64),
    # INT8 block-wise. This reached the GEMV only after quant_dequant_blockwise() stopped emitting a
    # redundant all-128 zero-point tensor for symmetric 8-bit: it made weight_zeros non-null, and
    # every MoE GEMV entry point rejects on `group_size > 0 && weight_zeros != nullptr`
    # (has_block_zeros in moe_kernels.cu) before the shape check runs.
    (2, 1, 8, 64),
]


def _run_qmoe_cutlass_gemm_second_scale_row_regression(test_case, quant_bits, use_asymmetric_quant):
    hidden_size = 128
    intermediate_size = 128
    sequence_length = 8
    num_experts = 1
    top_k = 1
    block_size = 32
    onnx_dtype = TensorProto.FLOAT16
    torch_dtype = onnx_to_torch_type_map[onnx_dtype]

    is_4_bit = quant_bits == 4

    fc1 = torch.zeros((intermediate_size, hidden_size), device=device, dtype=torch_dtype)
    fc2 = torch.zeros((hidden_size, intermediate_size), device=device, dtype=torch_dtype)

    fc1[0, :block_size] = 1.0 / 1024.0
    fc1[0, block_size : 2 * block_size] = 1.0
    fc2[0, 0] = 1.0

    fc1_scale, fc1_weight, fc1_qdq, fc1_zp = quant_dequant_blockwise(
        fc1, block_size, is_4_bit, asymmetric=use_asymmetric_quant
    )
    fc2_scale, fc2_weight, fc2_qdq, fc2_zp = quant_dequant_blockwise(
        fc2, block_size, is_4_bit, asymmetric=use_asymmetric_quant
    )

    model = create_moe_onnx_graph(
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        num_experts=num_experts,
        top_k=top_k,
        intermediate_size=intermediate_size,
        torch_dtype=torch.float32,
        onnx_dtype=onnx_dtype,
        fc1_experts_weights=fc1_weight.unsqueeze(0),
        fc2_experts_weights=fc2_weight.unsqueeze(0),
        fc1_scales=fc1_scale.unsqueeze(0),
        fc2_scales=fc2_scale.unsqueeze(0),
        fc1_zero_points=fc1_zp.unsqueeze(0) if fc1_zp is not None else None,
        fc2_zero_points=fc2_zp.unsqueeze(0) if fc2_zp is not None else None,
        use_swiglu=False,
        use_quant=True,
        quant_bits=quant_bits,
        block_size=block_size,
    )

    previous_disable_gemv = os.environ.get("ORT_DISABLE_MOE_GEMV")
    os.environ["ORT_DISABLE_MOE_GEMV"] = "1"
    try:
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = onnxruntime.InferenceSession(
            model,
            sess_options,
            providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")],
        )
    finally:
        if previous_disable_gemv is None:
            os.environ.pop("ORT_DISABLE_MOE_GEMV", None)
        else:
            os.environ["ORT_DISABLE_MOE_GEMV"] = previous_disable_gemv

    x = torch.zeros((sequence_length, hidden_size), device=device, dtype=torch_dtype)
    x[:, block_size : 2 * block_size] = 1.0
    router = torch.zeros((sequence_length, num_experts), device=device, dtype=torch_dtype)

    ort_output = sess.run(None, {"input": x.cpu().numpy(), "router_probs": router.cpu().numpy()})[0]
    fc1_output = torch.matmul(x.float(), fc1_qdq.float().T)
    expected = torch.matmul(F.silu(fc1_output), fc2_qdq.float().T).cpu().numpy().astype(numpy.float16)

    test_case.assertGreater(abs(expected[0, 0]), 20.0)
    numpy.testing.assert_allclose(ort_output, expected, rtol=2e-2, atol=2.5e-1)


@unittest.skipIf(not torch.cuda.is_available(), "skipping QMoE test since it requires CUDA.")
class TestPhiQMoE(unittest.TestCase):
    @parameterized.expand([(0,), (4,)])
    def test_packed_token_input_cuda(self, quant_bits):
        torch.manual_seed(1977 + quant_bits)
        numpy.random.seed(1977 + quant_bits)

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)
        packed_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=1,
            sequence_length=7,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            use_asymmetric_quant=False,
        )

        self.assertIsNotNone(packed_moe.ort_sess)
        packed_moe.parity_check(recreate_model=False)

    @parameterized.expand(phi3_test_cases)
    def test_phi3_qmoe_parity(self, batch_size, sequence_length, quant_bits):
        # Create unique seed based on test parameters to ensure different inputs for each test
        base_seed = 2000  # Different base seed from other tests
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000

        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running Phi3 QMoE test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.float16)

        torch_result = phi3_moe.forward(hidden_states)

        # Verify output shape and basic properties
        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        phi3_moe.parity_check()

    @parameterized.expand(phi3_test_cases)
    def test_phi3_qmoe_parity_bf16(self, batch_size, sequence_length, quant_bits):
        base_seed = 2500
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000
        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed} (BF16)"
        print(f"Running Phi3 QMoE test (BF16): {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.bfloat16)
        _ = phi3_moe.forward(hidden_states)

        phi3_moe.parity_check()

    @parameterized.expand(phi3_test_cases)
    def test_phi3_qmoe_asymmetric_parity(self, batch_size, sequence_length, quant_bits):
        self.skipTest("Row-wise asymmetric QMoE is unsupported on CUDA (zero-points require block-wise mode).")
        base_seed = 3000
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000
        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running Phi3 QMoE Asymmetric test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            use_asymmetric_quant=True,
        )
        phi3_moe.parity_check()

    @parameterized.expand(phi3_blockwise_test_cases)
    def test_phi3_qmoe_blockwise_parity(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running Phi3 QMoE block-wise test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.float16)

        torch_result = phi3_moe.forward(hidden_states)

        # Verify output shape and basic properties
        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        phi3_moe.parity_check()

    @parameterized.expand(phi3_blockwise_test_cases)
    def test_phi3_qmoe_blockwise_parity_bf16(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(142)
        numpy.random.seed(142)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size} (BF16)"
        print(f"Running Phi3 QMoE block-wise test (BF16): {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.bfloat16)
        _ = phi3_moe.forward(hidden_states)

        phi3_moe.parity_check()

    @parameterized.expand(phi3_blockwise_asymmetric_test_cases)
    def test_phi3_qmoe_blockwise_asymmetric_parity(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(43)
        numpy.random.seed(43)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running Phi3 QMoE block-wise Asymmetric test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=True,
        )
        phi3_moe.parity_check()

    @parameterized.expand(qmoe_cutlass_gemm_blockwise_test_cases)
    def test_phi3_qmoe_blockwise_cutlass_gemm_parity(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(44)
        numpy.random.seed(44)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running Phi3 QMoE block-wise CUTLASS GEMM test: {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )
        phi3_moe.parity_check()

    @parameterized.expand(qmoe_cutlass_gemm_blockwise_test_cases)
    def test_phi3_qmoe_blockwise_cutlass_gemm_parity_bf16(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(144)
        numpy.random.seed(144)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size} (BF16)"
        print(f"Running Phi3 QMoE block-wise CUTLASS GEMM test (BF16): {test_config}")

        config = PhiMoEConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2)

        phi3_moe = PhiMoESparseMoeBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )
        phi3_moe.parity_check()

    @parameterized.expand(qmoe_cutlass_gemm_second_scale_row_test_cases)
    def test_phi3_qmoe_blockwise_cutlass_gemm_second_scale_row(self, quant_bits, use_asymmetric_quant):
        _run_qmoe_cutlass_gemm_second_scale_row_regression(self, quant_bits, use_asymmetric_quant)


swiglu_test_cases = [
    (1, 32, 4),
    (1, 32, 8),
    (2, 16, 4),
    (2, 16, 8),
]

# Define test cases for block-wise quantization
swiglu_blockwise_test_cases = [
    (1, 1, 4, 32),  # tiny debug case for asymmetric ZP compensation
    (1, 32, 4, 32),  # batch_size, sequence_length, quant_bits, block_size
    (1, 32, 4, 64),  # New case for group_size=64
    (1, 32, 4, 128),
    (1, 32, 8, 32),
    (1, 32, 8, 64),
    (1, 32, 8, 128),
    (2, 16, 4, 32),
    (2, 16, 8, 32),
    (2, 16, 8, 64),
]
swiglu_blockwise_asymmetric_test_cases = [
    (1, 1, 4, 32),
    (1, 1, 8, 32),
    (1, 32, 4, 64),
    (1, 32, 8, 64),
    (1, 32, 8, 128),
    (2, 16, 8, 64),
]


@unittest.skipIf(not torch.cuda.is_available(), "skipping QMoE test since it requires CUDA.")
class TestSwigluQMoE(unittest.TestCase):
    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_qmoe_parity(self, batch_size, sequence_length, quant_bits):
        # Create unique seed based on test parameters to ensure different inputs for each test
        base_seed = 1000  # Different base seed from regular MoE tests
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000

        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running SwiGLU test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.float16)

        torch_result = swiglu_moe.forward(hidden_states)

        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        swiglu_moe.parity_check()

    def test_swiglu_qmoe_fusion0_remap_parity(self):
        # Backward-compat regression: the published gpt-oss-20b model emits no swiglu_fusion attribute
        # (defaulting to 0) but stores FC1 in the interleaved SwiGLU layout. The CUDA QMoE op remaps
        # swiglu_fusion=0 -> 1 for SwiGLU activation. Here we build an interleaved block, emit
        # swiglu_fusion=0 in the ONNX graph, and verify parity still holds against the interleaved
        # reference. Without the remap this would compute the wrong (non-interleaved) result.
        torch.manual_seed(1234)
        numpy.random.seed(1234)

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=1,
            sequence_length=32,
            quant_bits=4,
            onnx_dtype=TensorProto.FLOAT16,
            use_asymmetric_quant=False,
        )
        # Keep the interleaved reference and weight layout (swiglu_fusion == 1) but emit swiglu_fusion=0
        # in the ONNX attribute so the op's backward-compatibility remap path is exercised.
        self.assertEqual(swiglu_moe.swiglu_fusion, 1)
        swiglu_moe.onnx_swiglu_fusion_override = 0

        hidden_states = torch.randn(1, 32, config.hidden_size).to(device).to(torch.float16)
        _ = swiglu_moe.forward(hidden_states)

        swiglu_moe.parity_check()

    def test_swiglu_qmoe_int_partial_ktile_rejected(self):
        # NaN-hardening regression: the INT4/INT8 weight-only path stores B in the column-interleaved
        # layout, whose CUTLASS K iterator requires each GEMM reduction dim to be a whole multiple of the
        # 64-element interleave tile (fc1.K == hidden_size, fc2.K == inter_size). A partial final K tile
        # is read past the valid range and silently produces garbage/NaN. The CUTLASS mixed-GEMM weight
        # prepacking (shared by the offline CudaQuantizer and ORT's PrePack) therefore rejects such shapes
        # up front instead of computing wrong results. Here inter_size 544 (== 17*32) is block-quant valid
        # (block_size=32) but 544 % 64 == 32, so building the quantized model must raise.
        torch.manual_seed(4321)
        numpy.random.seed(4321)

        config = SwigluMoeConfig(hidden_size=512, intermediate_size=544, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=1,
            sequence_length=1,
            quant_bits=8,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=32,
            use_asymmetric_quant=False,
        )

        # The partial-K-tile guard fires while prepacking the expert weights into the CUTLASS
        # column-interleaved layout, so recreating the ONNX model is rejected before a session
        # is ever created.
        with self.assertRaisesRegex(Exception, "incompatible with column-interleave tiling"):
            swiglu_moe.recreate_onnx_model()

    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_qmoe_parity_bf16(self, batch_size, sequence_length, quant_bits):
        base_seed = 1500
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000
        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed} (BF16)"
        print(f"Running SwiGLU test (BF16): {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.bfloat16)
        _ = swiglu_moe.forward(hidden_states)

        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_test_cases)
    def test_swiglu_qmoe_asymmetric_parity(self, batch_size, sequence_length, quant_bits):
        self.skipTest("Row-wise asymmetric QMoE is unsupported on CUDA (zero-points require block-wise mode).")
        base_seed = 1100
        param_hash = hash((batch_size, sequence_length, quant_bits))
        unique_seed = base_seed + abs(param_hash) % 1000
        torch.manual_seed(unique_seed)
        numpy.random.seed(unique_seed)

        test_config = (
            f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, seed={unique_seed}"
        )
        print(f"Running SwiGLU Asymmetric test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            use_asymmetric_quant=True,
        )
        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_blockwise_test_cases)
    def test_swiglu_qmoe_blockwise_parity(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running SwiGLU block-wise test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.float16)

        torch_result = swiglu_moe.forward(hidden_states)

        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_blockwise_test_cases)
    def test_swiglu_qmoe_blockwise_parity_bf16(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(142)
        numpy.random.seed(142)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size} (BF16)"
        print(f"Running SwiGLU block-wise test (BF16): {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.bfloat16)
        _ = swiglu_moe.forward(hidden_states)

        swiglu_moe.parity_check()

    @parameterized.expand(swiglu_blockwise_asymmetric_test_cases)
    def test_swiglu_qmoe_blockwise_asymmetric_parity(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(43)
        numpy.random.seed(43)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running SwiGLU block-wise Asymmetric test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=True,
        )
        swiglu_moe.parity_check()

    @parameterized.expand(moe_gemv_test_cases)
    def test_swiglu_qmoe_gemv_parity(self, batch_size, sequence_length, quant_bits, block_size):
        # Decode-shaped cases large enough to select the MoE GEMV instead of the CUTLASS grouped GEMM.
        # See the comment on moe_gemv_test_cases for why hidden_size/intermediate_size must be >= 512.
        torch.manual_seed(45)
        numpy.random.seed(45)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running SwiGLU QMoE GEMV test: {test_config}")

        config = SwigluMoeConfig(hidden_size=512, intermediate_size=512, num_local_experts=8, num_experts_per_token=4)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )
        swiglu_moe.parity_check()

    @parameterized.expand(qmoe_cutlass_gemm_blockwise_test_cases)
    def test_swiglu_qmoe_blockwise_cutlass_gemm_parity(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(44)
        numpy.random.seed(44)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size}"
        print(f"Running SwiGLU block-wise CUTLASS GEMM test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.FLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )
        swiglu_moe.parity_check()

    @parameterized.expand(qmoe_cutlass_gemm_blockwise_test_cases)
    def test_swiglu_qmoe_blockwise_cutlass_gemm_parity_bf16(self, batch_size, sequence_length, quant_bits, block_size):
        torch.manual_seed(144)
        numpy.random.seed(144)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}, block_size={block_size} (BF16)"
        print(f"Running SwiGLU block-wise CUTLASS GEMM test (BF16): {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            block_size=block_size,
            use_asymmetric_quant=False,
        )
        swiglu_moe.parity_check()

    @parameterized.expand(qmoe_cutlass_gemm_second_scale_row_test_cases)
    def test_swiglu_qmoe_blockwise_cutlass_gemm_second_scale_row(self, quant_bits, use_asymmetric_quant):
        _run_qmoe_cutlass_gemm_second_scale_row_regression(self, quant_bits, use_asymmetric_quant)


def has_bf16_qmoe():
    """Check if BF16 QMoE is supported (requires Ampere or newer GPU)."""
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers() or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


# BF16 test cases for int4 and int8 quantization
bf16_test_cases = [
    (1, 32, 4),  # batch_size, sequence_length, quant_bits
    (1, 32, 8),
    (2, 16, 4),
    (2, 16, 8),
]


@unittest.skipIf(not has_bf16_qmoe(), "skipping bf16 QMoE tests (requires Ampere+ GPU).")
class TestSwigluQMoEBf16(unittest.TestCase):
    """BF16 QMoE tests for int4 and int8 quantization."""

    @parameterized.expand(bf16_test_cases)
    def test_swiglu_qmoe_bf16_parity(self, batch_size, sequence_length, quant_bits):
        """Test BF16 QMoE with symmetric quantization."""
        torch.manual_seed(42)
        numpy.random.seed(42)

        test_config = f"batch_size={batch_size}, sequence_length={sequence_length}, quant_bits={quant_bits}"
        print(f"Running BF16 SwiGLU QMoE test: {test_config}")

        config = SwigluMoeConfig(hidden_size=128, intermediate_size=256, num_local_experts=4, num_experts_per_token=2)

        swiglu_moe = SwigluMoEBlock(
            config,
            batch_size=batch_size,
            sequence_length=sequence_length,
            quant_bits=quant_bits,
            onnx_dtype=TensorProto.BFLOAT16,
            use_asymmetric_quant=False,
        )

        hidden_states = torch.randn(batch_size, sequence_length, config.hidden_size).to(device).to(torch.bfloat16)

        torch_result = swiglu_moe.forward(hidden_states)

        expected_shape = (batch_size, sequence_length, config.hidden_size)
        self.assertEqual(torch_result.shape, expected_shape)
        self.assertFalse(torch.isnan(torch_result).any())
        self.assertFalse(torch.isinf(torch_result).any())

        swiglu_moe.parity_check()


_QMOE_GEMV_BENCHMARK_RESULT_PREFIX = "QMOE_GEMV_BENCHMARK_RESULT "


def _qmoe_gemv_benchmark_cases():
    return [
        {
            "name": "m1_top2_fp16_128x256",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "m4_top2_fp16_128x256",
            "batch_size": 1,
            "sequence_length": 4,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "m8_top2_fp16_128x256",
            "batch_size": 1,
            "sequence_length": 8,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "m1_top2_bf16_128x256",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_experts": 4,
            "top_k": 2,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "gpt_oss_20b_m1_top4_fp16_2880x2880_e32",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_experts": 32,
            "top_k": 4,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "qwen3_6_35b_a3b_m1_top8_fp16_2048x512_e256",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "top_k": 8,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "gemma4_26b_a4b_m1_top8_fp16_2816x704_e128",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2816,
            "intermediate_size": 704,
            "num_experts": 128,
            "top_k": 8,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "blockwise_int4_b64_m1_top2_fp16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 64,
        },
        {
            "name": "blockwise_int4_b128_m1_top2_fp16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 4,
            "block_size": 128,
        },
        {
            "name": "blockwise_int8_b64_m1_top2_fp16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 8,
            "block_size": 64,
        },
        {
            "name": "blockwise_int8_b128_m1_top2_fp16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 8,
            "block_size": 128,
        },
        {
            "name": "gpt_oss_20b_m1_top4_bf16_2880x2880_e32",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_experts": 32,
            "top_k": 4,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "qwen3_6_35b_a3b_m1_top8_bf16_2048x512_e256",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2048,
            "intermediate_size": 512,
            "num_experts": 256,
            "top_k": 8,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "gemma4_26b_a4b_m1_top8_bf16_2816x704_e128",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2816,
            "intermediate_size": 704,
            "num_experts": 128,
            "top_k": 8,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 4,
            "block_size": 0,
        },
        {
            "name": "blockwise_int4_b64_m1_top2_bf16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 4,
            "block_size": 64,
        },
        {
            "name": "blockwise_int8_b64_m1_top2_bf16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 8,
            "block_size": 64,
        },
        {
            "name": "gpt_oss_20b_m1_top4_int8_fp16_2880x2880_e32",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_experts": 32,
            "top_k": 4,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 8,
            "block_size": 0,
        },
        {
            "name": "gpt_oss_20b_m1_top4_int8_bf16_2880x2880_e32",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 2880,
            "intermediate_size": 2880,
            "num_experts": 32,
            "top_k": 4,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 8,
            "block_size": 0,
        },
        {
            "name": "int8_per_column_m1_top2_fp16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "FLOAT16",
            "quant_bits": 8,
            "block_size": 0,
        },
        {
            "name": "int8_per_column_m1_top2_bf16_1024x4096_e8",
            "batch_size": 1,
            "sequence_length": 1,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "num_experts": 8,
            "top_k": 2,
            "onnx_dtype": "BFLOAT16",
            "quant_bits": 8,
            "block_size": 0,
        },
    ]


def _qmoe_gemv_benchmark_case(case_name):
    for case in _qmoe_gemv_benchmark_cases():
        if case["name"] == case_name:
            return case

    case_names = ", ".join(case["name"] for case in _qmoe_gemv_benchmark_cases())
    raise ValueError(f"Unknown QMoE GEMV benchmark case '{case_name}'. Available cases: {case_names}")


def run_qmoe_gemv_benchmark(case):
    seed = 4242
    torch.manual_seed(seed)
    numpy.random.seed(seed)

    onnx_dtype = getattr(TensorProto, case["onnx_dtype"])
    torch_dtype = onnx_to_torch_type_map[onnx_dtype]
    config = PhiMoEConfig(
        hidden_size=case["hidden_size"],
        intermediate_size=case["intermediate_size"],
        num_local_experts=case["num_experts"],
        num_experts_per_tok=case["top_k"],
    )
    qmoe = PhiMoESparseMoeBlock(
        config,
        batch_size=case["batch_size"],
        sequence_length=case["sequence_length"],
        quant_bits=case.get("quant_bits", 4),
        onnx_dtype=onnx_dtype,
        block_size=case.get("block_size", 0),
        use_asymmetric_quant=False,
    )
    hidden_states = torch.randn(
        case["batch_size"], case["sequence_length"], case["hidden_size"], device=device, dtype=torch_dtype
    )
    output = qmoe.ort_forward(hidden_states, enable_performance_test=True)

    return {
        "case": case["name"],
        "block_size": case.get("block_size", 0),
        "disable_gemv": os.getenv("ORT_DISABLE_MOE_GEMV") == "1",
        "disable_splitk2_swiglu": os.getenv("ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU") == "1",
        "expanded_num_rows": case["batch_size"] * case["sequence_length"] * case["top_k"],
        "has_invalid_output": bool(torch.isnan(output).any() or torch.isinf(output).any()),
        "latency_ms": qmoe.last_ort_latency_ms,
        "quant_bits": case.get("quant_bits", 4),
        "sm": torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1],
    }


def run_qmoe_gemv_benchmark_case(case_name=None):
    case = _qmoe_gemv_benchmark_case(
        case_name or os.getenv("ORT_QMOE_GEMV_BENCHMARK_CASE", _qmoe_gemv_benchmark_cases()[0]["name"])
    )
    return run_qmoe_gemv_benchmark(case)


@unittest.skipIf(not torch.cuda.is_available(), "skipping QMoE GEMV benchmark since it requires CUDA.")
@unittest.skipIf(
    os.getenv("ORT_QMOE_GEMV_BENCHMARK") != "1",
    "Set ORT_QMOE_GEMV_BENCHMARK=1 to run the opt-in QMoE GEMV benchmark.",
)
class TestQMoEGemvBenchmark(unittest.TestCase):
    def test_decode_latency(self):
        result = run_qmoe_gemv_benchmark_case()
        self.assertFalse(result["has_invalid_output"])
        print(_QMOE_GEMV_BENCHMARK_RESULT_PREFIX + json.dumps(result, sort_keys=True))

    @unittest.skipIf(
        os.getenv("ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU") == "1",
        "Unset ORT_DISABLE_MOE_GEMV_SPLITK2_SWIGLU to run the default split-K2 SwiGLU GEMV benchmark.",
    )
    def test_splitk2_swiglu_decode_latency(self):
        result = run_qmoe_gemv_benchmark_case("gpt_oss_20b_m1_top4_fp16_2880x2880_e32")
        self.assertFalse(result["disable_splitk2_swiglu"])
        self.assertFalse(result["has_invalid_output"])
        print(_QMOE_GEMV_BENCHMARK_RESULT_PREFIX + json.dumps(result, sort_keys=True))


@unittest.skipIf(True, "Skipping QMoE benchmark tests")
class TestQMoESwiGLUBenchmark(unittest.TestCase):
    """Benchmark tests for QMoE SwiGLU performance measurement."""

    def test_qmoe_swiglu_throughput_benchmark(self):
        """Comprehensive throughput benchmark for QMoE SwiGLU across different configurations."""

        print("\n=== QMoE SwiGLU Throughput Benchmark ===")

        # Test configurations: (name, hidden_size, intermediate_size, num_experts, top_k, quant_bits)
        configs = [
            ("Medium-4bit", 2880, 2880, 32, 4, 4),
            ("Medium-8bit", 2880, 2880, 32, 4, 8),
        ]

        batch_size = 1
        sequence_length = 512
        num_runs = 30

        results = []

        for config_name, hidden_size, intermediate_size, num_experts, top_k, quant_bits in configs:
            torch.manual_seed(42)
            numpy.random.seed(42)

            print(f"\nTesting {config_name}:")
            print(f"  Hidden: {hidden_size}, Intermediate: {intermediate_size}")
            print(f"  Experts: {num_experts}, Top-K: {top_k}, Quant: {quant_bits}-bit")

            try:
                # Create config and model
                config = PhiMoEConfig(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_local_experts=num_experts,
                    num_experts_per_tok=top_k,
                )

                qmoe_swiglu = PhiMoESparseMoeBlock(
                    config,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    quant_bits=quant_bits,
                    onnx_dtype=TensorProto.FLOAT16,
                )

                # Create test input with fixed sequence length to match ONNX model
                full_hidden_states = torch.randn(batch_size, sequence_length, hidden_size).to(device).to(torch.float16)

                # For TTFT simulation, we'll measure single forward pass time
                # This represents the time to process one token in autoregressive generation

                # Initialize variables
                torch_output = None
                ort_output = None

                # Warm up with full context
                for _ in range(3):
                    _ = qmoe_swiglu.forward(full_hidden_states)

                # Benchmark PyTorch TTFT (Time to First Token)
                # Measure time for a single forward pass (represents token generation time)
                torch.manual_seed(42)

                start_time = time.time()
                for _ in range(num_runs):
                    torch_output = qmoe_swiglu.forward(full_hidden_states)
                end_time = time.time()
                torch_ttft_ms = (end_time - start_time) / num_runs * 1000

                # Calculate tokens per second (throughput)
                # For sequence generation, this represents the rate at which we can generate tokens
                torch_tokens_per_sec = 1000.0 / torch_ttft_ms  # 1 token / (time_ms / 1000)

                print(f"  PyTorch TTFT: {torch_ttft_ms:.3f} ms (per token generation time)")
                print(f"  PyTorch Throughput: {torch_tokens_per_sec:.1f} tokens/sec")

                # Benchmark ONNX Runtime
                ort_ttft_ms = 0
                ort_tokens_per_sec = 0
                speedup = 0
                throughput_ratio = 0
                max_diff = 0

                model_updated = qmoe_swiglu.recreate_onnx_model()
                if model_updated and qmoe_swiglu.ort_sess is not None:
                    # Warm up ORT with full context
                    for _ in range(3):
                        _ = qmoe_swiglu.ort_forward(full_hidden_states)

                    torch.manual_seed(42)

                    # Measure ONNX Runtime TTFT (Time to First Token)
                    start_time = time.time()
                    for _ in range(num_runs):
                        ort_output = qmoe_swiglu.ort_forward(full_hidden_states)
                    end_time = time.time()
                    ort_ttft_ms = (end_time - start_time) / num_runs * 1000

                    # Calculate tokens per second for ONNX Runtime
                    ort_tokens_per_sec = 1000.0 / ort_ttft_ms  # 1 token / (time_ms / 1000)

                    speedup = torch_ttft_ms / ort_ttft_ms if ort_ttft_ms > 0 else 0
                    throughput_ratio = ort_tokens_per_sec / torch_tokens_per_sec if torch_tokens_per_sec > 0 else 0

                    print(f"  ONNX RT TTFT: {ort_ttft_ms:.3f} ms (per token generation time)")
                    print(f"  ONNX RT Throughput: {ort_tokens_per_sec:.1f} tokens/sec")
                    print(f"  TTFT Speedup: {speedup:.2f}x")
                    print(f"  Throughput Gain: {throughput_ratio:.2f}x")
                else:
                    print("  ONNX RT: Not available")

                # Calculate max difference if both outputs available
                if torch_output is not None and ort_output is not None:
                    max_diff = (torch_output.cpu() - ort_output.cpu()).abs().max().item()
                    print(f"  Max diff: {max_diff:.6f}")

                results.append(
                    {
                        "config": config_name,
                        "torch_ttft_ms": torch_ttft_ms,
                        "torch_tokens_per_sec": torch_tokens_per_sec,
                        "ort_ttft_ms": ort_ttft_ms,
                        "ort_tokens_per_sec": ort_tokens_per_sec,
                        "speedup": speedup,
                        "throughput_ratio": throughput_ratio,
                        "max_diff": max_diff,
                    }
                )

            except Exception as e:
                print(f"  Error: {e}")
                continue

        # Summary
        print("\n=== Token Generation Time & Throughput Summary ===")
        print(
            f"{'Config':<15} {'PT Time':<10} {'PT tok/s':<10} {'ORT Time':<11} {'ORT tok/s':<11} {'Time Gain':<10} {'Throughput':<11} {'Max Diff':<10}"
        )
        print("-" * 105)
        for result in results:
            config = result["config"]
            torch_ttft = result["torch_ttft_ms"]
            torch_tps = result["torch_tokens_per_sec"]
            ort_ttft = result["ort_ttft_ms"]
            ort_tps = result["ort_tokens_per_sec"]
            speedup = result["speedup"]
            throughput_ratio = result["throughput_ratio"]
            max_diff = result["max_diff"]

            ort_ttft_str = f"{ort_ttft:.3f}" if ort_ttft > 0 else "N/A"
            ort_tps_str = f"{ort_tps:.1f}" if ort_tps > 0 else "N/A"
            speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
            throughput_str = f"{throughput_ratio:.2f}x" if throughput_ratio > 0 else "N/A"

            print(
                f"{config:<15} {torch_ttft:<10.3f} {torch_tps:<10.1f} {ort_ttft_str:<11} {ort_tps_str:<11} {speedup_str:<10} {throughput_str:<11} {max_diff:<10.6f}"
            )

        print("\nNotes:")
        print("- Time: Token generation time in ms (lower is better)")
        print("- tok/s: Tokens per second throughput (higher is better)")
        print("- Time Gain: ORT speedup for latency (higher is better)")
        print("- Throughput: ORT throughput improvement (higher is better)")


# ============================================================================
# QMoE integer-weight PrePack smoke test.
#
# Validates the PrePack hook added in PR #28749: with `quant_type="int"`, the
# QMoE op should be able to consume raw quantized weights — shape
# `[E, N, K/(8/bits)]` as produced by `quantize_matmul_{4,8}bits` —
# and internally run the CUTLASS fpA_intB layout transform that callers
# previously had to do offline via `pack_weights_for_cuda_mixed_gemm`.
#
# Strategy: build a single ONNX graph with raw (un-prepacked) int4 weight
# initializers and `weights_prepacked=0`, run it through ORT's CUDA QMoE
# kernel, and assert the output is finite and has a plausible magnitude.
# This is a smoke test, not a numerical parity check — see the class
# docstring for why a bit-parity comparison is intentionally omitted.
# ============================================================================


@unittest.skipUnless(torch.cuda.is_available(), "QMoE PrePack smoke test requires CUDA")
class TestQMoEIntPrePackSmoke(unittest.TestCase):
    """Smoke test for the QMoE int4 PrePack hook (issue #28748 / PR #28749).

    Builds a single QMoE node with raw, un-prepacked ``[E, N, K/2]`` int4
    weights straight from ``quantize_matmul_4bits`` and runs it through
    the CUDA QMoE kernel. With the new ``PrePackIntExpertWeights`` hook,
    the kernel should:

    1. Accept the on-disk shape that matches the ``com.microsoft::QMoE``
       schema (``[E, N, K/pack]``), where today's offline tooling has to
       hand-write the transposed pre-prepacked shape ``[E, K, N/pack]``
       and pre-pack the bytes itself via ``pack_weights_for_cuda_mixed_gemm``.
    2. Run the GEMM to completion and produce sensible output (no NaN /
       Inf, output magnitudes consistent with a small weight + small
       input matmul).

    We deliberately do **not** include a bit-parity check against the
    existing offline-pre-pack code path because the existing harness
    (``quant_dequant_blockwise`` → ``pack_weights_for_cuda_mixed_gemm``)
    hardcodes ``force_arch=80`` and produces incorrect output on SM>=90
    hardware (the other ``test_swiglu_qmoe_parity_*`` cases in this file
    fail on H200 / H100 with max-diff > 1.0 on plain main, by
    inspection — pre-existing). A real parity check can be added once
    that harness honors the runtime SM.
    """

    def test_moe_cuda_quantizer_can_emit_full_range_unsigned_offset_storage(self):
        cases = [
            (
                4,
                torch.tensor([[-8.0, -7.0, 0.0, 7.0]], dtype=torch.float32),
                torch.tensor([[0x10, 0xF8]], dtype=torch.uint8),
            ),
            (
                8,
                torch.tensor([[-128.0, -127.0, 0.0, 127.0]], dtype=torch.float32),
                torch.tensor([[0x00, 0x01, 0x80, 0xFF]], dtype=torch.uint8),
            ),
        ]
        for bits, weights, expected_qweight in cases:
            with self.subTest(bits=bits):
                qweight, scales = CudaQuantizer.qmoe_symmetric_per_channel_quantize(
                    weights,
                    bits,
                )

                self.assertTrue(torch.equal(scales, torch.tensor([1.0])))
                self.assertTrue(torch.equal(qweight, expected_qweight))

    def _run_one(
        self,
        *,
        hidden_size,
        inter_size,
        num_experts,
        top_k,
        swiglu_fusion,
        batch_size,
        row_tile_size=None,
    ):
        torch.manual_seed(123)
        numpy.random.seed(123)

        onnx_dtype = TensorProto.FLOAT16
        use_swiglu = True
        # fc1 packs gate+up along the N axis when use_swiglu=True.
        fc1_n = 2 * inter_size if use_swiglu else inter_size
        fc1_k = hidden_size
        fc2_n = hidden_size
        fc2_k = inter_size

        raw_fc1 = numpy.zeros((num_experts, fc1_n, fc1_k // 2), dtype=numpy.uint8)
        raw_fc2 = numpy.zeros((num_experts, fc2_n, fc2_k // 2), dtype=numpy.uint8)
        fc1_scales = numpy.zeros((num_experts, fc1_n), dtype=numpy.float16)
        fc2_scales = numpy.zeros((num_experts, fc2_n), dtype=numpy.float16)

        for e in range(num_experts):
            w1 = (torch.randn(fc1_n, fc1_k) * 0.05).numpy().astype(numpy.float16)
            w2 = (torch.randn(fc2_n, fc2_k) * 0.05).numpy().astype(numpy.float16)
            qw1 = numpy.zeros((fc1_n, 1, fc1_k // 2), dtype=numpy.uint8)
            qw2 = numpy.zeros((fc2_n, 1, fc2_k // 2), dtype=numpy.uint8)
            sc1 = numpy.zeros((fc1_n, 1), dtype=numpy.float32)
            sc2 = numpy.zeros((fc2_n, 1), dtype=numpy.float32)
            zp1 = numpy.zeros((fc1_n, 1), dtype=numpy.uint8)
            zp2 = numpy.zeros((fc2_n, 1), dtype=numpy.uint8)
            _pybind.quantize_matmul_4bits(qw1, numpy.ascontiguousarray(w1.T), sc1, zp1, fc1_k, fc1_n, fc1_k, True)
            _pybind.quantize_matmul_4bits(qw2, numpy.ascontiguousarray(w2.T), sc2, zp2, fc2_k, fc2_n, fc2_k, True)
            raw_fc1[e] = qw1.reshape(fc1_n, fc1_k // 2)
            raw_fc2[e] = qw2.reshape(fc2_n, fc2_k // 2)
            fc1_scales[e] = numpy.abs(sc1).flatten().astype(numpy.float16)
            fc2_scales[e] = numpy.abs(sc2).flatten().astype(numpy.float16)

        qmoe = helper.make_node(
            "QMoE",
            inputs=["x", "router", "fc1_W", "fc1_S", "", "fc2_W", "fc2_S", ""],
            outputs=["y"],
            name="qmoe",
            domain="com.microsoft",
            k=top_k,
            normalize_routing_weights=1,
            activation_type="swiglu" if use_swiglu else "silu",
            swiglu_fusion=swiglu_fusion,
            expert_weight_bits=4,
            quant_type="int",
            # Opt in to the PrePack-hook path; the weights below are raw
            # ``[E, N, K/2]`` outputs of ``quantize_matmul_4bits``, not
            # CUTLASS-prepacked.
            weights_prepacked=0,
        )
        graph = helper.make_graph(
            nodes=[qmoe],
            name="qmoe_only",
            inputs=[
                helper.make_tensor_value_info("x", onnx_dtype, [None, hidden_size]),
                helper.make_tensor_value_info("router", onnx_dtype, [None, num_experts]),
            ],
            outputs=[helper.make_tensor_value_info("y", onnx_dtype, [None, hidden_size])],
            initializer=[
                helper.make_tensor("fc1_W", TensorProto.UINT8, list(raw_fc1.shape), raw_fc1.tobytes(), raw=True),
                helper.make_tensor("fc2_W", TensorProto.UINT8, list(raw_fc2.shape), raw_fc2.tobytes(), raw=True),
                helper.make_tensor("fc1_S", onnx_dtype, list(fc1_scales.shape), fc1_scales.flatten().tolist()),
                helper.make_tensor("fc2_S", onnx_dtype, list(fc2_scales.shape), fc2_scales.flatten().tolist()),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10

        session_options = None
        if row_tile_size is not None:
            session_options = onnxruntime.SessionOptions()
            session_options.add_session_config_entry("ep.cuda.qmoe_row_tile_size", str(row_tile_size))
        sess = onnxruntime.InferenceSession(
            model.SerializeToString(), sess_options=session_options, providers=ort_provider
        )
        x = numpy.random.randn(batch_size, hidden_size).astype(numpy.float16)
        router = numpy.random.randn(batch_size, num_experts).astype(numpy.float16)
        out = sess.run(None, {"x": x, "router": router})[0]

        self.assertEqual(out.shape, (batch_size, hidden_size))
        self.assertEqual(out.dtype, numpy.float16)
        self.assertFalse(numpy.isnan(out).any(), "QMoE raw-weight output has NaN")
        self.assertFalse(numpy.isinf(out).any(), "QMoE raw-weight output has Inf")
        # With weights ~ N(0, 0.05) and input ~ N(0, 1), SwiGLU + routing
        # output magnitudes land well below 10 per element. A loose bound
        # catches accidental near-zero or runaway output that would
        # indicate the PrePack hook silently produced wrong bytes.
        self.assertGreater(numpy.abs(out).mean(), 1e-4, "Output is suspiciously close to zero")
        self.assertLess(numpy.abs(out).max(), 10.0, "Output magnitude is implausibly large")
        return out

    def _run_default_prepacked_model(
        self,
        *,
        hidden_size,
        inter_size,
        num_experts,
        top_k,
        batch_size,
        fc1_weights,
        fc2_weights,
        fc1_scales,
        fc2_scales,
        x,
        router,
    ):
        onnx_dtype = TensorProto.FLOAT16
        fc1_n = 2 * inter_size
        qmoe = helper.make_node(
            "QMoE",
            inputs=["x", "router", "fc1_W", "fc1_S", "", "fc2_W", "fc2_S", ""],
            outputs=["y"],
            name="qmoe",
            domain="com.microsoft",
            k=top_k,
            normalize_routing_weights=1,
            activation_type="swiglu",
            swiglu_fusion=1,
            expert_weight_bits=4,
            quant_type="int",
            # weights_prepacked omitted: default -1 means the INT weights are already in
            # the CUDA EP's offline-prepacked fpA_intB layout.
        )
        graph = helper.make_graph(
            nodes=[qmoe],
            name="qmoe_default_prepacked",
            inputs=[
                helper.make_tensor_value_info("x", onnx_dtype, [batch_size, hidden_size]),
                helper.make_tensor_value_info("router", onnx_dtype, [batch_size, num_experts]),
            ],
            outputs=[helper.make_tensor_value_info("y", onnx_dtype, [batch_size, hidden_size])],
            initializer=[
                helper.make_tensor(
                    "fc1_W", TensorProto.UINT8, list(fc1_weights.shape), fc1_weights.tobytes(), raw=True
                ),
                helper.make_tensor(
                    "fc2_W", TensorProto.UINT8, list(fc2_weights.shape), fc2_weights.tobytes(), raw=True
                ),
                helper.make_tensor("fc1_S", onnx_dtype, [num_experts, fc1_n], fc1_scales.flatten().tolist()),
                helper.make_tensor("fc2_S", onnx_dtype, [num_experts, hidden_size], fc2_scales.flatten().tolist()),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10

        sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=ort_provider)
        return sess.run(None, {"x": x, "router": router})[0]

    def test_int4_default_prepacked_layout_runs_with_moe_cuda_quantizer(self):
        torch.manual_seed(123)
        numpy.random.seed(123)

        hidden_size = 64
        inter_size = 128
        num_experts = 4
        top_k = 2
        batch_size = 8
        bits = 4
        pack = 8 // bits

        fc1_n = 2 * inter_size
        fc1_k = hidden_size
        fc2_n = hidden_size
        fc2_k = inter_size

        cuda_fc1 = numpy.zeros((num_experts, fc1_k, fc1_n // pack), dtype=numpy.uint8)
        cuda_fc2 = numpy.zeros((num_experts, fc2_k, fc2_n // pack), dtype=numpy.uint8)
        cuda_fc1_scales = numpy.zeros((num_experts, fc1_n), dtype=numpy.float16)
        cuda_fc2_scales = numpy.zeros((num_experts, fc2_n), dtype=numpy.float16)
        cuda_quantizer = CudaQuantizer()

        for e in range(num_experts):
            w1 = (torch.randn(fc1_n, fc1_k) * 0.05).numpy().astype(numpy.float16)
            w2 = (torch.randn(fc2_n, fc2_k) * 0.05).numpy().astype(numpy.float16)
            cuda_fc1_t, cuda_fc1_scales_t = cuda_quantizer.qmoe_per_channel_quantize(torch.from_numpy(w1), bits, True)
            cuda_fc2_t, cuda_fc2_scales_t = cuda_quantizer.qmoe_per_channel_quantize(torch.from_numpy(w2), bits, True)
            cuda_fc1[e] = cuda_fc1_t.numpy()
            cuda_fc2[e] = cuda_fc2_t.numpy()
            cuda_fc1_scales[e] = cuda_fc1_scales_t.numpy().astype(numpy.float16)
            cuda_fc2_scales[e] = cuda_fc2_scales_t.numpy().astype(numpy.float16)

        x = numpy.random.randn(batch_size, hidden_size).astype(numpy.float16)
        router = numpy.random.randn(batch_size, num_experts).astype(numpy.float16)
        out = self._run_default_prepacked_model(
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            batch_size=batch_size,
            fc1_weights=cuda_fc1,
            fc2_weights=cuda_fc2,
            fc1_scales=cuda_fc1_scales,
            fc2_scales=cuda_fc2_scales,
            x=x,
            router=router,
        )

        self.assertFalse(numpy.isnan(out).any(), "QMoE output has NaN")
        self.assertFalse(numpy.isinf(out).any(), "QMoE output has Inf")

    def test_int4_default_prepacked_gpt_oss_shape_smoke(self):
        torch.manual_seed(123)
        numpy.random.seed(123)

        hidden_size = 2880
        inter_size = 2880
        num_experts = 32
        top_k = 4
        batch_size = 12
        bits = 4

        fc1_n = 2 * inter_size
        fc1_k = hidden_size
        fc2_n = hidden_size
        fc2_k = inter_size
        pack = 8 // bits
        cuda_quantizer = CudaQuantizer()

        fc1_weights = numpy.zeros((num_experts, fc1_k, fc1_n // pack), dtype=numpy.uint8)
        fc2_weights = numpy.zeros((num_experts, fc2_k, fc2_n // pack), dtype=numpy.uint8)
        fc1_scales = numpy.zeros((num_experts, fc1_n), dtype=numpy.float16)
        fc2_scales = numpy.zeros((num_experts, fc2_n), dtype=numpy.float16)

        for e in range(num_experts):
            w1 = torch.randn(fc1_n, fc1_k, dtype=torch.float16) * 0.01
            w2 = torch.randn(fc2_n, fc2_k, dtype=torch.float16) * 0.01
            q1, s1 = cuda_quantizer.qmoe_per_channel_quantize(w1, bits, True)
            q2, s2 = cuda_quantizer.qmoe_per_channel_quantize(w2, bits, True)
            fc1_weights[e] = q1.numpy()
            fc2_weights[e] = q2.numpy()
            fc1_scales[e] = s1.numpy().astype(numpy.float16)
            fc2_scales[e] = s2.numpy().astype(numpy.float16)

        x = (numpy.random.randn(batch_size, hidden_size) * 0.01).astype(numpy.float16)
        router = numpy.random.randn(batch_size, num_experts).astype(numpy.float16)
        out = self._run_default_prepacked_model(
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            batch_size=batch_size,
            fc1_weights=fc1_weights,
            fc2_weights=fc2_weights,
            fc1_scales=fc1_scales,
            fc2_scales=fc2_scales,
            x=x,
            router=router,
        )

        self.assertEqual(out.shape, (batch_size, hidden_size))
        self.assertFalse(numpy.isnan(out).any(), "QMoE GPT-OSS shape output has NaN")
        self.assertFalse(numpy.isinf(out).any(), "QMoE GPT-OSS shape output has Inf")

    def test_int4_swiglu_interleaved_small(self):
        # inter_size must be a multiple of 64 (the interleaved-weight K tile) for the INT path; a
        # partial final K tile is now rejected up front by QMoE's hardening check.
        self._run_one(hidden_size=64, inter_size=64, num_experts=4, top_k=2, swiglu_fusion=1, batch_size=8)

    def test_int4_swiglu_interleaved_medium(self):
        self._run_one(hidden_size=128, inter_size=64, num_experts=8, top_k=2, swiglu_fusion=1, batch_size=16)

    def test_int4_grouped_moe_row_tiling_parity(self):
        shape = dict(
            hidden_size=128,
            inter_size=128,
            num_experts=8,
            top_k=2,
            swiglu_fusion=1,
            batch_size=65,
        )
        tiled_output = self._run_one(**shape, row_tile_size=16)
        untiled_output = self._run_one(**shape, row_tile_size=1024)
        max_diff = numpy.max(numpy.abs(untiled_output.astype(numpy.float32) - tiled_output.astype(numpy.float32)))

        self.assertLess(
            max_diff,
            0.02,
            f"INT4 grouped-MoE row-tiled output differs from untiled output: max_diff={max_diff:.6f}",
        )


def _sm80_prepack_unsigned_codes(codes, bits):
    """Lay out unsigned ``[N, K]`` codes in the SM80 CUTLASS mixed-GEMM layout for 2/4-bit weights.

    The signed re-bias applied by the CUDA packer cancels with the kernel's unsigned decode, so the layout is
    a row permutation, a sub-byte transpose, a column interleave and a pairwise element shuffle per 32-bit
    word. Returns a uint8 array of shape ``[K, N / (8 / bits)]``.
    """
    n, k = codes.shape
    pack = 8 // bits
    tile = 128 // bits
    permutation = [2 * g + 8 * j + t for g in range(4) for j in range(16 // bits) for t in range(2)]
    row_idx = numpy.array([(r // tile) * tile + permutation[r % tile] for r in range(k)])
    permuted = codes.T[row_idx]  # [K, N]

    def pack_lsb(elements):
        flat = elements.reshape(-1, pack).astype(numpy.uint32)
        return sum(flat[:, i] << (bits * i) for i in range(pack)).astype(numpy.uint8)

    def unpack_lsb(packed):
        mask = (1 << bits) - 1
        return numpy.stack([(packed >> (bits * i)) & mask for i in range(pack)], axis=-1).reshape(-1)

    stream = pack_lsb(permuted.T.reshape(-1))  # sub-byte transpose: rows of N, K packed along the row
    interleave = 16 // bits
    rows_per_tile = 128 * 8 // 16
    stream = (
        stream.reshape(-1, interleave, k // rows_per_tile, rows_per_tile * bits // 8).transpose(0, 2, 1, 3).reshape(-1)
    )
    elements_per_word = 32 // bits
    shuffle = list(range(0, elements_per_word, 2)) + list(range(1, elements_per_word, 2))
    elements = unpack_lsb(stream).reshape(-1, elements_per_word)[:, shuffle].reshape(-1)
    return pack_lsb(elements).reshape(k, n // pack)


class TestQMoEOfflinePrepackedIntParity(unittest.TestCase):
    """weights_prepacked=1 INT2 and mixed-width QMoE must match the PrePack-on-load (weights_prepacked=0) path.

    The offline SM80 layout is generated here (the production packer only handles 4/8-bit), and the original
    block-scale initializers are released by PrePack, so these runs also cover the cached-scale decode path.
    """

    hidden_size = 256
    inter_size = 128
    num_experts = 4

    @staticmethod
    def _random_projection(rng, num_experts, n, k, bits, block_size):
        codes = rng.integers(0, 1 << bits, size=(num_experts, n, k), dtype=numpy.uint8)
        scale_values = 0.004 * (1 + (numpy.arange(n * (k // block_size)) * 7 + 3) % 9)
        scales = numpy.stack(
            [numpy.roll(scale_values, 5 * e).reshape(n, k // block_size) for e in range(num_experts)]
        ).astype(numpy.float16)
        return codes, scales

    @staticmethod
    def _raw_bytes(codes, bits):
        pack = 8 // bits
        flat = codes.reshape(codes.shape[0], codes.shape[1], -1, pack).astype(numpy.uint32)
        packed = sum(flat[..., i] << (bits * i) for i in range(pack))
        return packed.astype(numpy.uint8)

    def _build_session(self, fc1, fc2, bits, block_size, weights_prepacked, fc1_scales=None, fc2_scales=None):
        fc1_codes, fc1_scale_values = fc1
        fc2_codes, fc2_scale_values = fc2
        fc1_bits, fc2_bits = bits
        fc1_scale_values = fc1_scale_values if fc1_scales is None else fc1_scales
        fc2_scale_values = fc2_scale_values if fc2_scales is None else fc2_scales
        if weights_prepacked:
            fc1_w = numpy.stack([_sm80_prepack_unsigned_codes(c, fc1_bits) for c in fc1_codes])
            fc2_w = numpy.stack([_sm80_prepack_unsigned_codes(c, fc2_bits) for c in fc2_codes])
        else:
            fc1_w = self._raw_bytes(fc1_codes, fc1_bits)
            fc2_w = self._raw_bytes(fc2_codes, fc2_bits)

        qmoe = helper.make_node(
            "QMoE",
            inputs=["x", "router", "fc1_W", "fc1_S", "", "fc2_W", "fc2_S", ""],
            outputs=["y"],
            name="qmoe",
            domain="com.microsoft",
            k=2,
            normalize_routing_weights=1,
            activation_type="swiglu",
            swiglu_fusion=1,
            expert_weight_bits=fc1_bits,
            fc1_expert_weight_bits=fc1_bits,
            fc2_expert_weight_bits=fc2_bits,
            fc3_expert_weight_bits=fc1_bits,
            quant_type="int",
            block_size=block_size,
            weights_prepacked=weights_prepacked,
        )
        fp16 = TensorProto.FLOAT16
        graph = helper.make_graph(
            nodes=[qmoe],
            name="qmoe_offline_prepacked",
            inputs=[
                helper.make_tensor_value_info("x", fp16, [None, self.hidden_size]),
                helper.make_tensor_value_info("router", fp16, [None, self.num_experts]),
            ],
            outputs=[helper.make_tensor_value_info("y", fp16, [None, self.hidden_size])],
            initializer=[
                helper.make_tensor("fc1_W", TensorProto.UINT8, list(fc1_w.shape), fc1_w.tobytes(), raw=True),
                helper.make_tensor("fc2_W", TensorProto.UINT8, list(fc2_w.shape), fc2_w.tobytes(), raw=True),
                helper.make_tensor(
                    "fc1_S",
                    fp16,
                    list(fc1_scale_values.shape),
                    fc1_scale_values.astype(numpy.float16).tobytes(),
                    raw=True,
                ),
                helper.make_tensor(
                    "fc2_S",
                    fp16,
                    list(fc2_scale_values.shape),
                    fc2_scale_values.astype(numpy.float16).tobytes(),
                    raw=True,
                ),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        options = onnxruntime.SessionOptions()
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        return onnxruntime.InferenceSession(
            model.SerializeToString(), options, providers=[resolve_cuda_plugin_ep("CUDAExecutionProvider")]
        )

    def _reference(self, fc1, fc2, bits, block_size, x, router):
        def dequantize(projection, width):
            codes, scales = projection
            blocks = numpy.repeat(scales.astype(numpy.float32), block_size, axis=-1)
            weights = (codes.astype(numpy.float32) - (1 << (width - 1))) * blocks
            return weights.astype(numpy.float16).astype(numpy.float32)

        w1 = dequantize(fc1, bits[0])
        w2 = dequantize(fc2, bits[1])
        x32 = x.astype(numpy.float32)
        logits = router.astype(numpy.float32)
        out = numpy.zeros_like(x32)
        for r in range(x32.shape[0]):
            top = numpy.argsort(-logits[r], kind="stable")[:2]
            weights = numpy.exp(logits[r][top] - logits[r][top].max())
            weights /= weights.sum()
            for expert, weight in zip(top, weights, strict=True):
                fused = w1[expert] @ x32[r]
                gate, up = fused[0::2], fused[1::2]
                act = (gate / (1.0 + numpy.exp(-gate)) * up).astype(numpy.float16).astype(numpy.float32)
                out[r] += weight * (w2[expert] @ act)
        return out

    def _make_case(self, bits, block_size):
        rng = numpy.random.default_rng(2024 + bits[0] * 10 + bits[1] + block_size)
        fc1 = self._random_projection(rng, self.num_experts, 2 * self.inter_size, self.hidden_size, bits[0], block_size)
        fc2 = self._random_projection(rng, self.num_experts, self.hidden_size, self.inter_size, bits[1], block_size)
        return fc1, fc2

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_offline_layout_matches_cuda_packer_for_int4(self):
        from onnxruntime.quantization.cuda_quantizer import _pack_weights_for_cuda_mixed_gemm  # noqa: PLC0415

        rng = numpy.random.default_rng(7)
        codes = rng.integers(0, 16, size=(64, 128), dtype=numpy.uint8)
        raw = self._raw_bytes(codes[None], 4)[0]
        expected = _pack_weights_for_cuda_mixed_gemm(raw, 64, 128, 4, 80)
        actual = _sm80_prepack_unsigned_codes(codes, 4)
        numpy.testing.assert_array_equal(actual.reshape(-1), numpy.asarray(expected).view(numpy.uint8))

    @parameterized.expand(
        [
            ("int2_block32", (2, 2), 32),
            ("int2_block128", (2, 2), 128),
            ("int2_int4_block64", (2, 4), 64),
            ("int4_int2_block64", (4, 2), 64),
        ]
    )
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_prepacked_matches_runtime_prepack_and_reference(self, _name, bits, block_size):
        fc1, fc2 = self._make_case(bits, block_size)
        rng = numpy.random.default_rng(99)
        raw_session = self._build_session(fc1, fc2, bits, block_size, 0)
        packed_session = self._build_session(fc1, fc2, bits, block_size, 1)
        for rows in (1, 96):  # GEMV decode and grouped-GEMM prefill
            x = rng.standard_normal((rows, self.hidden_size)).astype(numpy.float16)
            router = rng.standard_normal((rows, self.num_experts)).astype(numpy.float16)
            expected = self._reference(fc1, fc2, bits, block_size, x, router)
            raw_out = raw_session.run(None, {"x": x, "router": router})[0].astype(numpy.float32)
            packed_out = packed_session.run(None, {"x": x, "router": router})[0].astype(numpy.float32)
            self.assertFalse(numpy.isnan(packed_out).any())
            numpy.testing.assert_allclose(packed_out, raw_out, rtol=1e-3, atol=1e-3, err_msg=f"rows={rows}")
            numpy.testing.assert_allclose(packed_out, expected, rtol=0.05, atol=0.05, err_msg=f"rows={rows}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_prepacked_rejects_malformed_scale_shapes(self):
        bits, block_size = (2, 2), 32
        fc1, fc2 = self._make_case(bits, block_size)
        bad_fc1 = fc1[1][:, :, :1].copy()  # [E, N, 1] instead of [E, N, K / block_size]
        bad_fc2 = fc2[1][:, :, :1].copy()
        rng = numpy.random.default_rng(5)
        x = rng.standard_normal((1, self.hidden_size)).astype(numpy.float16)
        router = rng.standard_normal((1, self.num_experts)).astype(numpy.float16)
        for name, kwargs in (("fc1", {"fc1_scales": bad_fc1}), ("fc2", {"fc2_scales": bad_fc2})):
            with self.subTest(name):
                with self.assertRaises(Exception) as ctx:
                    session = self._build_session(fc1, fc2, bits, block_size, 1, **kwargs)
                    session.run(None, {"x": x, "router": router})
                self.assertIn(f"{name}_scales shape [E,N,K/block_size]", str(ctx.exception))


class TestQMoECudaGraph(unittest.TestCase):
    """Regression test for QMoE under CUDA graph capture/replay.

    Before the profiler capture-safety fix, running an fp16 int4 QMoE node inside
    a CUDA-graph-capturing session launched grouped-GEMM profiling kernels and
    temp-allocator traffic on the compute stream *while that stream was being
    captured*. That corrupted the capture and surfaced as a sticky CUDA 700
    (illegal memory access) at a downstream MoE kernel launch on replay. The fix
    detects an in-progress capture, skips profiling, and falls back to a cached /
    default tactic, so capture + replay must now complete cleanly and produce
    finite, deterministic output.
    """

    def _build_int4_qmoe_model(self, *, hidden_size, inter_size, num_experts, top_k, batch_size):
        onnx_dtype = TensorProto.FLOAT16
        bits = 4
        pack = 8 // bits
        fc1_n = 2 * inter_size  # gate + up packed along N for SwiGLU
        fc1_k = hidden_size
        fc2_n = hidden_size
        fc2_k = inter_size

        fc1_weights = numpy.zeros((num_experts, fc1_k, fc1_n // pack), dtype=numpy.uint8)
        fc2_weights = numpy.zeros((num_experts, fc2_k, fc2_n // pack), dtype=numpy.uint8)
        fc1_scales = numpy.zeros((num_experts, fc1_n), dtype=numpy.float16)
        fc2_scales = numpy.zeros((num_experts, fc2_n), dtype=numpy.float16)

        cuda_quantizer = CudaQuantizer()
        for e in range(num_experts):
            w1 = (torch.randn(fc1_n, fc1_k) * 0.05).numpy().astype(numpy.float16)
            w2 = (torch.randn(fc2_n, fc2_k) * 0.05).numpy().astype(numpy.float16)
            q1, s1 = cuda_quantizer.qmoe_per_channel_quantize(torch.from_numpy(w1), bits, True)
            q2, s2 = cuda_quantizer.qmoe_per_channel_quantize(torch.from_numpy(w2), bits, True)
            fc1_weights[e] = q1.numpy()
            fc2_weights[e] = q2.numpy()
            fc1_scales[e] = s1.numpy().astype(numpy.float16)
            fc2_scales[e] = s2.numpy().astype(numpy.float16)

        qmoe = helper.make_node(
            "QMoE",
            inputs=["input", "router_probs", "fc1_W", "fc1_S", "", "fc2_W", "fc2_S", ""],
            outputs=["output"],
            name="qmoe",
            domain="com.microsoft",
            k=top_k,
            normalize_routing_weights=1,
            activation_type="swiglu",
            swiglu_fusion=1,
            expert_weight_bits=bits,
            quant_type="int",
            # weights_prepacked omitted (default): INT weights are already in the
            # CUDA EP's offline-prepacked fpA_intB layout emitted by CudaQuantizer.
        )
        # Static shapes are required for CUDA graph capture.
        graph = helper.make_graph(
            nodes=[qmoe],
            name="qmoe_cuda_graph",
            inputs=[
                helper.make_tensor_value_info("input", onnx_dtype, [batch_size, hidden_size]),
                helper.make_tensor_value_info("router_probs", onnx_dtype, [batch_size, num_experts]),
            ],
            outputs=[helper.make_tensor_value_info("output", onnx_dtype, [batch_size, hidden_size])],
            initializer=[
                helper.make_tensor(
                    "fc1_W", TensorProto.UINT8, list(fc1_weights.shape), fc1_weights.tobytes(), raw=True
                ),
                helper.make_tensor(
                    "fc2_W", TensorProto.UINT8, list(fc2_weights.shape), fc2_weights.tobytes(), raw=True
                ),
                helper.make_tensor("fc1_S", onnx_dtype, [num_experts, fc1_n], fc1_scales.flatten().tolist()),
                helper.make_tensor("fc2_S", onnx_dtype, [num_experts, fc2_n], fc2_scales.flatten().tolist()),
            ],
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10
        return model.SerializeToString()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the QMoE CUDA graph regression test")
    def test_int4_fp16_qmoe_cuda_graph(self):
        hidden_size = 64
        inter_size = 128
        num_experts = 4
        top_k = 2
        batch_size = 8

        torch.manual_seed(123)
        numpy.random.seed(123)
        model_bytes = self._build_int4_qmoe_model(
            hidden_size=hidden_size,
            inter_size=inter_size,
            num_experts=num_experts,
            top_k=top_k,
            batch_size=batch_size,
        )

        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = None
        try:
            sess = onnxruntime.InferenceSession(
                model_bytes,
                sess_options,
                providers=[("CUDAExecutionProvider", {"enable_cuda_graph": True})],
            )
        except Exception as e:
            self.skipTest(f"Could not create a CUDA-graph-enabled CUDA EP session: {e}")
            return

        x = numpy.random.randn(batch_size, hidden_size).astype(numpy.float16)
        router = numpy.random.randn(batch_size, num_experts).astype(numpy.float16)

        # CUDA graph capture requires all inputs/outputs to live in fixed GPU buffers.
        x_ort = onnxruntime.OrtValue.ortvalue_from_numpy(x, "cuda", 0)
        router_ort = onnxruntime.OrtValue.ortvalue_from_numpy(router, "cuda", 0)
        y_ort = onnxruntime.OrtValue.ortvalue_from_shape_and_type([batch_size, hidden_size], numpy.float16, "cuda", 0)

        io_binding = sess.io_binding()
        io_binding.bind_ortvalue_input("input", x_ort)
        io_binding.bind_ortvalue_input("router_probs", router_ort)
        io_binding.bind_ortvalue_output("output", y_ort)

        ro = onnxruntime.RunOptions()

        # First run performs the allocation and captures the CUDA graph. Before the
        # fix, QMoE profiling launched kernels / touched the temp allocator on the
        # captured compute stream here and triggered CUDA 700 (illegal memory access).
        io_binding.synchronize_inputs()
        sess.run_with_iobinding(io_binding, ro)
        io_binding.synchronize_outputs()
        out_capture = y_ort.numpy().copy()

        # Replay the captured graph with the same inputs.
        io_binding.synchronize_inputs()
        sess.run_with_iobinding(io_binding, ro)
        io_binding.synchronize_outputs()
        out_replay = y_ort.numpy().copy()

        for tag, out in (("capture", out_capture), ("replay", out_replay)):
            self.assertEqual(out.shape, (batch_size, hidden_size))
            self.assertEqual(out.dtype, numpy.float16)
            self.assertFalse(numpy.isnan(out).any(), f"QMoE CUDA graph {tag} output has NaN")
            self.assertFalse(numpy.isinf(out).any(), f"QMoE CUDA graph {tag} output has Inf")

        # Replaying with identical inputs must reproduce the captured output exactly.
        numpy.testing.assert_array_equal(out_replay, out_capture)

        # Update the input in place and replay again to exercise the graph past capture.
        x2 = numpy.random.randn(batch_size, hidden_size).astype(numpy.float16)
        x_ort.update_inplace(x2)
        io_binding.synchronize_inputs()
        sess.run_with_iobinding(io_binding, ro)
        io_binding.synchronize_outputs()
        out_updated = y_ort.numpy().copy()
        self.assertFalse(numpy.isnan(out_updated).any(), "QMoE CUDA graph updated-input output has NaN")
        self.assertFalse(numpy.isinf(out_updated).any(), "QMoE CUDA graph updated-input output has Inf")


if __name__ == "__main__":
    unittest.main()

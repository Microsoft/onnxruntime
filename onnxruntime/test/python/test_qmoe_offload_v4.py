#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Unit tests for Portable QMoE Offload v4 (standard com.microsoft::QMoE + session.moe.offload.*)."""

import contextlib
import ctypes
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx

import onnxruntime as ort

TensorProto = onnx.TensorProto
helper = onnx.helper


def _float32_to_bfloat16_u16(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.float32).view(np.uint32)
    return ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16)


def _bfloat16_u16_to_float32(arr: np.ndarray) -> np.ndarray:
    u32 = arr.astype(np.uint32) << 16
    return u32.view(np.float32)


def _build_canonical_nvfp4_qmoe_model(
    model_dir: str,
    num_layers: int = 2,
    num_experts: int = 16,
    hidden_size: int = 128,
    inter_size: int = 64,
    top_k: int = 4,
    seed: int = 42,
    with_bias: bool = False,
    short_profile_pairs: int | None = None,
    embedded_initializers: bool = False,
    non_moe_gpu_init_elements: int = 0,
    layer_experts: list[int] | None = None,
    quant_type: str = "nvfp4",
    use_fp16: bool = False,
) -> str:
    rng = np.random.default_rng(seed)
    ext_filename = "weights.data"
    ext_path = os.path.join(model_dir, ext_filename)
    is_mxfp4 = quant_type in ("fp4", "mxfp4")
    blk_size = 32 if is_mxfp4 else 16
    scale_dtype = TensorProto.FLOAT8E8M0 if is_mxfp4 else TensorProto.FLOAT8E4M3FN
    act_onnx_dtype = TensorProto.FLOAT16 if use_fp16 else TensorProto.BFLOAT16

    nodes = []
    initializers = []
    offset = 0

    def add_tensor(wf, name, arr, onnx_dtype, force_embedded=False):
        nonlocal offset
        raw = arr.tobytes()
        if embedded_initializers or force_embedded:
            t = helper.make_tensor(
                name=name,
                data_type=onnx_dtype,
                dims=list(arr.shape),
                vals=raw,
                raw=True,
            )
            initializers.append(t)
            return
        pad = (4096 - (offset % 4096)) % 4096
        if pad:
            wf.write(b"\x00" * pad)
            offset += pad
        wf.write(raw)
        t = helper.make_tensor(
            name=name,
            data_type=onnx_dtype,
            dims=list(arr.shape),
            vals=raw,
            raw=True,
        )
        onnx.external_data_helper.set_external_data(t, location=ext_filename, offset=offset, length=len(raw))
        t.data_location = TensorProto.EXTERNAL
        t.ClearField("raw_data")
        offset += len(raw)
        initializers.append(t)

    with open(ext_path, "wb") as wf:
        cur_in = "X"
        if non_moe_gpu_init_elements > 0:
            non_moe_arr = np.zeros((1, hidden_size), dtype=np.uint16)
            add_tensor(wf, "non_moe_gpu_bias_init", non_moe_arr, act_onnx_dtype, force_embedded=True)
            nodes.append(
                helper.make_node(
                    "Add",
                    inputs=[cur_in, "non_moe_gpu_bias_init"],
                    outputs=["X_after_non_moe"],
                    name="model/non_moe_pre_add",
                )
            )
            cur_in = "X_after_non_moe"

        for l_idx in range(num_layers):
            ne = layer_experts[l_idx] if layer_experts is not None else num_experts
            fc1_w = rng.integers(0, 256, size=(ne, hidden_size, inter_size), dtype=np.uint8)
            fc2_w = rng.integers(0, 256, size=(ne, inter_size, hidden_size // 2), dtype=np.uint8)
            if is_mxfp4:
                # E8M0 scale codes: 125 = 2^-2 = 0.25, 126 = 2^-1 = 0.5
                scale_choices = np.array([124, 125, 126], dtype=np.uint8)
            else:
                # E4M3 scale bytes: 0x38 = 0.5, 0x30 = 0.25
                scale_choices = np.array([0x30, 0x34, 0x38], dtype=np.uint8)
            fc1_s = rng.choice(scale_choices, size=(ne, 2 * inter_size, hidden_size // blk_size))
            fc2_s = rng.choice(scale_choices, size=(ne, hidden_size, inter_size // blk_size))
            fc1_g = np.full((ne,), 0.05, dtype=np.float32)
            fc2_g = np.full((ne,), 0.05, dtype=np.float32)

            w1_name = f"model.layers.{l_idx}.mlp.fc1_experts_weights"
            s1_name = f"model.layers.{l_idx}.mlp.fc1_scales"
            b1_name = ""
            w2_name = f"model.layers.{l_idx}.mlp.fc2_experts_weights"
            s2_name = f"model.layers.{l_idx}.mlp.fc2_scales"
            b2_name = ""
            g1_name = f"model.layers.{l_idx}.mlp.fc1_global_scales"
            g2_name = f"model.layers.{l_idx}.mlp.fc2_global_scales"

            add_tensor(wf, w1_name, fc1_w, TensorProto.UINT8)
            add_tensor(wf, s1_name, fc1_s, scale_dtype)
            if with_bias:
                b1_name = f"model.layers.{l_idx}.mlp.fc1_bias"
                b2_name = f"model.layers.{l_idx}.mlp.fc2_bias"
                b1_f32 = rng.uniform(-0.1, 0.1, size=(ne, 2 * inter_size)).astype(np.float32)
                b2_f32 = rng.uniform(-0.1, 0.1, size=(ne, hidden_size)).astype(np.float32)
                if use_fp16:
                    fc1_b = b1_f32.astype(np.float16).view(np.uint16)
                    fc2_b = b2_f32.astype(np.float16).view(np.uint16)
                else:
                    fc1_b = _float32_to_bfloat16_u16(b1_f32)
                    fc2_b = _float32_to_bfloat16_u16(b2_f32)
                add_tensor(wf, b1_name, fc1_b, act_onnx_dtype)
                add_tensor(wf, b2_name, fc2_b, act_onnx_dtype)
            add_tensor(wf, w2_name, fc2_w, TensorProto.UINT8)
            add_tensor(wf, s2_name, fc2_s, scale_dtype)
            add_tensor(wf, g1_name, fc1_g, TensorProto.FLOAT)
            add_tensor(wf, g2_name, fc2_g, TensorProto.FLOAT)

            r_in_name = "router_probs" if layer_experts is None else f"router_probs_{l_idx}"
            out_name = "Y" if l_idx == num_layers - 1 else f"H_{l_idx}"
            node = helper.make_node(
                "QMoE",
                inputs=[
                    cur_in,
                    r_in_name,
                    w1_name,
                    s1_name,
                    b1_name,
                    w2_name,
                    s2_name,
                    b2_name,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    g1_name,
                    g2_name,
                ],
                outputs=[out_name],
                name=f"model/layers.{l_idx}/mlp/QMoE_node_{l_idx}",
                domain="com.microsoft",
                activation_type="swiglu",
                normalize_routing_weights=1,
                k=min(top_k, ne),
                expert_weight_bits=4,
                block_size=blk_size,
                swiglu_fusion=1,
                quant_type=quant_type,
                weights_prepacked=1,
            )
            nodes.append(node)
            cur_in = out_name

    if embedded_initializers and os.path.exists(ext_path):
        os.remove(ext_path)

    # Also write a STRP v1 popularity profile (expert-profile.bin)
    prof_path = os.path.join(model_dir, "expert-profile.bin")
    pairs = []
    for e in range(num_experts):
        for l_idx in range(num_layers):
            ne = layer_experts[l_idx] if layer_experts is not None else num_experts
            if e < ne:
                pairs.append((l_idx, (e * 3 + l_idx) % ne))
    if short_profile_pairs is not None:
        pairs = pairs[:short_profile_pairs]
    with open(prof_path, "wb") as pf:
        pf.write(struct.pack("<6I", 0x50525453, 1, num_layers, num_experts, num_layers * num_experts, len(pairs)))
        pf.writelines(struct.pack("<2H", l_id, e_id) for l_id, e_id in pairs)

    graph_inputs = [helper.make_tensor_value_info("X", act_onnx_dtype, ["T", hidden_size])]
    if layer_experts is None:
        graph_inputs.append(helper.make_tensor_value_info("router_probs", act_onnx_dtype, ["T", num_experts]))
    else:
        for l_idx, ne in enumerate(layer_experts):
            graph_inputs.append(helper.make_tensor_value_info(f"router_probs_{l_idx}", act_onnx_dtype, ["T", ne]))

    graph = helper.make_graph(
        nodes,
        "qmoe_v4_test",
        graph_inputs,
        [
            helper.make_tensor_value_info("Y", act_onnx_dtype, ["T", hidden_size]),
        ],
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model_path = os.path.join(model_dir, "model.onnx")
    onnx.save_model(model, model_path)
    return model_path


class TestQMoEOffloadV4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin_path = os.environ.get("QWEN_PROVIDER")
        if plugin_path and os.path.exists(plugin_path):
            with contextlib.suppress(Exception):
                ort.register_execution_provider_library("CUDAExecutionProvider", plugin_path)

    def _make_session(self, model_path: str, config_entries: dict, cuda_options: dict | None = None):
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        for k, v in config_entries.items():
            so.add_session_config_entry(k, str(v))
        providers = [("CUDAExecutionProvider", cuda_options or {})]
        return ort.InferenceSession(model_path, sess_options=so, providers=providers)

    def _run_bf16(self, sess, x_f32: np.ndarray, r_f32: np.ndarray) -> np.ndarray:
        x_bf16 = _float32_to_bfloat16_u16(x_f32)
        r_bf16 = _float32_to_bfloat16_u16(r_f32)
        ort_x = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16)
        ort_r = ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r_bf16, TensorProto.BFLOAT16)
        out = sess.run_with_ort_values(["Y"], {"X": ort_x, "router_probs": ort_r})[0]
        raw_bytes = bytes((ctypes.c_uint16 * int(np.prod(x_f32.shape))).from_address(out.data_ptr()))
        return _bfloat16_u16_to_float32(np.frombuffer(raw_bytes, dtype=np.uint16).reshape(x_f32.shape))

    def test_discarded_prototype_features_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(tmpdir)
            # 1. Unknown session.moe.offload.* key must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "Unknown session.moe.offload"):
                self._make_session(model_path, {"session.moe.offload.unknown_knob": "1"})

            # 2. Discarded prototype moe.cpu_offload.* key must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "discarded and not supported"):
                self._make_session(model_path, {"moe.cpu_offload.experts": "4"})

            # 3. Discarded prototype placement strategy 'basic' must fail with INVALID_ARGUMENT
            with self.assertRaisesRegex(Exception, "expected 'static' or 'adaptive'"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "basic",
                    },
                )

            # 4. Separate QMoEOffload operator schema must no longer exist
            m = onnx.load_model(model_path, load_external_data=False)
            m.graph.node[0].op_type = "QMoEOffload"
            legacy_path = os.path.join(tmpdir, "legacy_qmoe_offload.onnx")
            onnx.save_model(m, legacy_path)
            with self.assertRaisesRegex(Exception, "QMoEOffload"):
                self._make_session(legacy_path, {})

    def test_v4_portable_qmoe_modes_and_numerical_parity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4
            )
            # Per-expert slot bytes for H=128, I=64, block=16:
            # 2*64*64 + 128*32 + 2*64*8 + 128*4 + 8 = 8192 + 4096 + 1024 + 512 + 8 = 13,832 B
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4  # 128 B per layer

            # 1. Baseline: mode=disabled (Full GPU resident)
            sess_base = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})

            # 2. mode=auto with ample VRAM (Full GPU residency admitted via Stage 1->4)
            sess_auto_full = self._make_session(model_path, {"session.moe.offload.mode": "auto"})

            # 3. mode=required, static placement (6 resident slots/layer = 12 total slots across 2 layers, spare=0)
            budget_12_slots = 2 * fixed_table_bytes + 12 * slot_bytes
            sess_static = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.device_expert_bytes": str(budget_12_slots),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )

            # 4. mode=required, adaptive placement (6 resident + 2 spare slots/layer = 16 total physical slots)
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes
            sess_adaptive = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )

            # 5. mode=required with bounded host_expert_bytes (FileExpertStore bounded host cache)
            # Aligned host stride for H=128, I=64 is 14,336 B (8192 + 1024 + 256 + 4096 + 512 + 256).
            host_slot_stride = 14336
            sess_bounded_host = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.host_expert_bytes": str(2 * 4 * host_slot_stride),  # 4 cold slots/layer
                    "session.moe.offload.allow_file_backing": "1",
                    "session.moe.offload.budget_group": "",
                },
            )

            # 6. mode=required with cpu_execution="disabled" (GPU landing-ring weight streaming for all row counts)
            sess_gpu_stream_only = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "static",
                    "session.moe.offload.cpu_execution": "disabled",
                    "session.moe.offload.device_expert_bytes": str(budget_12_slots),
                },
            )

            rng = np.random.default_rng(123)
            # Test both decode regime (T=2 <= async_max_tokens=4) and prefill regime (T=16 > async_max_tokens=4)
            for num_rows in [1, 2, 4, 16]:
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 16), dtype=np.float32)

                y_base_f32 = self._run_bf16(sess_base, x, r)

                for name, s in [
                    ("auto_full", sess_auto_full),
                    ("static_hybrid", sess_static),
                    ("adaptive_hybrid", sess_adaptive),
                    ("bounded_host", sess_bounded_host),
                    ("gpu_stream_only", sess_gpu_stream_only),
                ]:
                    y_cand_f32 = self._run_bf16(s, x, r)
                    max_diff = float(np.max(np.abs(y_base_f32 - y_cand_f32)))
                    self.assertLess(
                        max_diff, 5e-2, f"Numerical mismatch in mode={name} at T={num_rows}: max_diff={max_diff}"
                    )

    def test_v4_eligibility_biases_boundaries_and_multi_session_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_a = os.path.join(tmpdir, "model_a")
            dir_b = os.path.join(tmpdir, "model_b")
            dir_bias = os.path.join(tmpdir, "model_bias")
            os.makedirs(dir_a, exist_ok=True)
            os.makedirs(dir_b, exist_ok=True)
            os.makedirs(dir_bias, exist_ok=True)

            model_a = _build_canonical_nvfp4_qmoe_model(dir_a, seed=101, short_profile_pairs=4)
            model_b = _build_canonical_nvfp4_qmoe_model(dir_b, seed=202, short_profile_pairs=4)
            model_bias = _build_canonical_nvfp4_qmoe_model(dir_bias, seed=303, with_bias=True)

            slot_bytes = 13832
            host_slot_stride = 14336
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            # 1. Exact 1-slot host cache boundary (F07, F10):
            # 0 bytes or 1 byte below 2 * host_slot_stride must fail fast instead of falling back to unbounded mmap.
            with self.assertRaisesRegex(Exception, "host_expert_bytes"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                        "session.moe.offload.host_expert_bytes": "0",
                    },
                )
            with self.assertRaisesRegex(Exception, "host_expert_bytes"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                        "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride - 1),
                    },
                )
            sess_one_slot_host = self._make_session(
                model_a,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride),
                },
            )
            del sess_one_slot_host

            # 2. Missing explicit popularity_file and invalid node_overrides must fail fast (F13, F14).
            with self.assertRaisesRegex(Exception, "popularity_file"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.popularity_file": "nonexistent_profile.bin",
                    },
                )
            with self.assertRaisesRegex(Exception, "node_overrides"):
                self._make_session(
                    model_a,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.node_overrides": "malformed_override_without_colon",
                    },
                )

            # 3. Unsupported swiglu_fusion=2 must be rejected before relocation in mode=required (F03, F04).
            m_bad = onnx.load_model(model_a, load_external_data=False)
            for attr in m_bad.graph.node[0].attribute:
                if attr.name == "swiglu_fusion":
                    attr.i = 2
            bad_path = os.path.join(dir_a, "model_bad_swiglu.onnx")
            onnx.save_model(m_bad, bad_path)
            with self.assertRaisesRegex(Exception, "swiglu_fusion"):
                self._make_session(bad_path, {"session.moe.offload.mode": "required"})

            # 4. Nonzero FC1/FC2 biases parity across cold, hot, and promoted execution (F05).
            fc1_bias_per_expert = 2 * 64 * 2  # 256 B
            fc2_bias_per_layer = 16 * 128 * 2  # 4096 B
            bias_slot_bytes = slot_bytes + fc1_bias_per_expert
            # The node keeps its own full FC1/FC2 bias banks besides the backend's per-slot FC1 bias and FC2 buffer.
            bias_fixed_bytes = fixed_table_bytes + 2 * fc2_bias_per_layer + 16 * fc1_bias_per_expert
            bias_budget = 2 * bias_fixed_bytes + 12 * bias_slot_bytes
            sess_bias_ref = self._make_session(model_bias, {"session.moe.offload.mode": "disabled"})
            sess_bias_off = self._make_session(
                model_bias,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(bias_budget),
                },
            )
            rng = np.random.default_rng(777)
            for step in range(4):
                x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((2, 16), dtype=np.float32)
                y_ref = self._run_bf16(sess_bias_ref, x, r)
                y_off = self._run_bf16(sess_bias_off, x, r)
                self.assertLess(
                    float(np.max(np.abs(y_ref - y_off))),
                    5e-2,
                    f"FC1/FC2 bias parity failed at step {step}",
                )

            # 5. Two same-shaped sessions with different weights on the same GPU (F11, F18).
            sess_a_ref = self._make_session(model_a, {"session.moe.offload.mode": "disabled"})
            sess_b_ref = self._make_session(model_b, {"session.moe.offload.mode": "disabled"})
            sess_a_off = self._make_session(
                model_a,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                    "session.moe.offload.node_overrides": "model/layers.0/mlp/QMoE_node_0:max_resident_slots=6",
                },
            )
            sess_b_off = self._make_session(
                model_b,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.popularity_file": "expert-profile.bin",
                },
            )
            for num_rows in [2, 16, 2]:
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((num_rows, 16), dtype=np.float32)
                ya_ref = self._run_bf16(sess_a_ref, x, r)
                yb_ref = self._run_bf16(sess_b_ref, x, r)
                ya_off = self._run_bf16(sess_a_off, x, r)
                yb_off = self._run_bf16(sess_b_off, x, r)
                self.assertGreater(float(np.max(np.abs(ya_ref - yb_ref))), 1e-2)
                self.assertLess(float(np.max(np.abs(ya_ref - ya_off))), 5e-3)
                self.assertLess(float(np.max(np.abs(yb_ref - yb_off))), 5e-3)

    def test_v4_cuda_graph_capture_and_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            rng = np.random.default_rng(999)
            x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r = rng.standard_normal((2, 16), dtype=np.float32)
            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})

            # Wrap inputs/outputs in FLOAT <-> BFLOAT16 Cast on GPU so CUDA Graph capture has 100% GPU I/O
            m = onnx.load_model(model_path, load_external_data=False)
            m.graph.node.insert(
                0, helper.make_node("Cast", inputs=["R_f32"], outputs=["router_probs"], to=TensorProto.BFLOAT16)
            )
            m.graph.node.insert(0, helper.make_node("Cast", inputs=["X_f32"], outputs=["X"], to=TensorProto.BFLOAT16))
            m.graph.node.append(helper.make_node("Cast", inputs=["Y"], outputs=["Y_f32"], to=TensorProto.FLOAT))
            del m.graph.input[:]
            m.graph.input.extend(
                [
                    helper.make_tensor_value_info("X_f32", TensorProto.FLOAT, ["T", 128]),
                    helper.make_tensor_value_info("R_f32", TensorProto.FLOAT, ["T", 16]),
                ]
            )
            del m.graph.output[:]
            m.graph.output.append(helper.make_tensor_value_info("Y_f32", TensorProto.FLOAT, ["T", 128]))
            cg_model_path = os.path.join(tmpdir, "model_cg.onnx")
            onnx.save_model(m, cg_model_path)

            devices = [
                d
                for d in ort.get_ep_devices()
                if d.ep_name == "CUDAExecutionProvider" and d.device.type == ort.OrtHardwareDeviceType.GPU
            ]
            devices.sort(key=lambda d: int(d.ep_options.get("device_id", d.ep_metadata.get("cuda_device_id", "0"))))
            dev_id = (
                int(devices[0].ep_options.get("device_id", devices[0].ep_metadata.get("cuda_device_id", "0")))
                if devices
                else 0
            )

            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            for k, v in {
                "session.moe.offload.mode": "required",
                "session.moe.offload.placement": "adaptive",
                "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                "session.moe.offload.popularity_file": "expert-profile.bin",
            }.items():
                so.add_session_config_entry(k, v)
            if devices:
                so.add_provider_for_devices(
                    [devices[0]], {"enable_cuda_graph": "1", "min_num_runs_before_cuda_graph_capture": "2"}
                )
                sess_graph = ort.InferenceSession(cg_model_path, sess_options=so)
            else:
                sess_graph = ort.InferenceSession(
                    cg_model_path, sess_options=so, providers=[("CUDAExecutionProvider", {"enable_cuda_graph": "1"})]
                )

            def round_bf16(a: np.ndarray) -> np.ndarray:
                return _bfloat16_u16_to_float32(_float32_to_bfloat16_u16(a))

            cudart = None
            for lib_name in (
                "libcudart.so.13",
                "libcudart.so.12",
                "libcudart.so",
                "cudart64_13.dll",
                "cudart64_12.dll",
            ):
                with contextlib.suppress(OSError):
                    cudart = ctypes.CDLL(lib_name)
                    break
            if cudart is None:
                self.skipTest("CUDA runtime library is required to update captured GPU inputs in place")

            def device_buffer(nbytes: int) -> ctypes.c_void_p:
                ptr = ctypes.c_void_p()
                self.assertEqual(cudart.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), 0)
                return ptr

            def upload(ptr: ctypes.c_void_p, arr: np.ndarray):
                arr = np.ascontiguousarray(arr, dtype=np.float32)
                self.assertEqual(cudart.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p), arr.nbytes, 1), 0)

            # Update the captured GPU inputs in place with different activations and routes on every replay so a
            # stale doorbell, host-result or scratch buffer cannot match the reference by accident.
            x_ptr = device_buffer(x.nbytes)
            r_ptr = device_buffer(r.nbytes)
            try:
                x_in = round_bf16(x)
                r_in = round_bf16(r)
                upload(x_ptr, x_in)
                upload(r_ptr, r_in)
                binding = sess_graph.io_binding()
                binding.bind_input("X_f32", "cuda", dev_id, np.float32, list(x.shape), x_ptr.value)
                binding.bind_input("R_f32", "cuda", dev_id, np.float32, list(r.shape), r_ptr.value)
                binding.bind_output("Y_f32", "cuda", dev_id)

                replay_rng = np.random.default_rng(31337)
                for step in range(8):
                    if step > 0:
                        x_in = round_bf16(replay_rng.standard_normal((2, 128), dtype=np.float32) * 0.5)
                        r_in = round_bf16(replay_rng.standard_normal((2, 16), dtype=np.float32) * 2.0)
                        upload(x_ptr, x_in)
                        upload(r_ptr, r_in)
                    y_ref = self._run_bf16(sess_ref, x_in, r_in)
                    sess_graph.run_with_iobinding(binding)
                    y_cand = binding.copy_outputs_to_cpu()[0]
                    max_diff = float(np.max(np.abs(y_ref - y_cand)))
                    self.assertLess(max_diff, 5e-2, f"CUDA graph replay mismatch at step {step}: max_diff={max_diff}")
            finally:
                cudart.cudaFree(x_ptr)
                cudart.cudaFree(r_ptr)

    def _run_fp16(self, sess, x_f32: np.ndarray, r_f32: np.ndarray) -> np.ndarray:
        x_f16 = x_f32.astype(np.float16)
        r_f16 = r_f32.astype(np.float16)
        out = sess.run(["Y"], {"X": x_f16, "router_probs": r_f16})[0]
        return out.astype(np.float32)

    def test_v4_mxfp4_offload_is_rejected_when_required_and_ignored_when_auto(self):
        # Only NVFP4 with weights_prepacked=1 defines the K-packed row-major layout the offload kernels read, so
        # MXFP4 nodes must neither be relocated nor reinterpreted.
        with tempfile.TemporaryDirectory() as tmpdir:
            for use_fp16 in (False, True):
                sub_dir = os.path.join(tmpdir, "fp16" if use_fp16 else "bf16")
                os.makedirs(sub_dir, exist_ok=True)
                model_path = _build_canonical_nvfp4_qmoe_model(
                    sub_dir,
                    num_layers=2,
                    num_experts=16,
                    hidden_size=128,
                    inter_size=64,
                    top_k=4,
                    seed=515 + int(use_fp16),
                    with_bias=True,
                    quant_type="fp4",
                    use_fp16=use_fp16,
                )
                with self.assertRaisesRegex(Exception, "expected 'nvfp4'"):
                    self._make_session(model_path, {"session.moe.offload.mode": "required"})

                sess_base = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
                sess_auto = self._make_session(model_path, {"session.moe.offload.mode": "auto"})
                rng = np.random.default_rng(808 + int(use_fp16))
                runner = self._run_fp16 if use_fp16 else self._run_bf16
                for num_rows in (1, 4, 16):
                    x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                    r = rng.standard_normal((num_rows, 16), dtype=np.float32)
                    np.testing.assert_array_equal(runner(sess_base, x, r), runner(sess_auto, x, r))

    def test_v4_embedded_heap_weights_bounded_cache_eviction_safety(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Embedded initializers (heap-backed, 2 pages/expert for fc1_w) with bounded host cache (1 cold slot/layer).
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=606,
                embedded_initializers=True,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            host_slot_stride = 14336
            budget_small = 2 * fixed_table_bytes + 8 * slot_bytes  # 4 slots/layer (2 resident + 2 spare)

            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
            sess_bounded_embedded = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_small),
                    "session.moe.offload.host_expert_bytes": str(2 * host_slot_stride),  # 1 cold slot/layer
                    "session.moe.offload.allow_file_backing": "1",
                },
            )
            rng = np.random.default_rng(607)
            # Route specifically to cold uncached experts (experts 12..15) across decode, prefill, and follow-up decode.
            for num_rows in (2, 16, 2):
                x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                r = np.full((num_rows, 16), -10.0, dtype=np.float32)
                for row in range(num_rows):
                    for k_idx in range(4):
                        r[row, 15 - ((row + k_idx) % 8)] = 5.0 - 0.5 * k_idx
                y_ref = self._run_bf16(sess_ref, x, r)
                y_got = self._run_bf16(sess_bounded_embedded, x, r)
                self.assertGreater(float(np.max(np.abs(y_got))), 1e-3)
                self.assertLess(float(np.max(np.abs(y_ref - y_got))), 5e-2)

    def test_v4_named_budget_group_with_non_moe_gpu_initializers_and_failed_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Mixed model with non-MoE GPU initializers allocated during SaveInitializedTensors before CreateKernels.
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir,
                num_layers=2,
                num_experts=16,
                hidden_size=128,
                inter_size=64,
                top_k=4,
                seed=707,
                non_moe_gpu_init_elements=128 * 256,
            )
            slot_bytes = 13832
            fixed_table_bytes = 2 * 16 * 4
            budget_adaptive = 2 * fixed_table_bytes + 16 * slot_bytes

            sess_ref = self._make_session(model_path, {"session.moe.offload.mode": "disabled"})
            sess_grouped = self._make_session(
                model_path,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                    "session.moe.offload.budget_group": "test_mixed_group_v4",
                },
                cuda_options={"gpu_device_mem_limit": str(64 * 1024 * 1024)},
            )
            rng = np.random.default_rng(708)
            x = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r = rng.standard_normal((2, 16), dtype=np.float32)
            y_ref = self._run_bf16(sess_ref, x, r)
            y_grp = self._run_bf16(sess_grouped, x, r)
            self.assertLess(float(np.max(np.abs(y_ref - y_grp))), 5e-2)

            # A failed initialization in a named budget group after non-MoE GPU initializers were allocated
            # must clean up cleanly without underflowing group accounting or terminating the process.
            with self.assertRaisesRegex(Exception, "host_expert_bytes"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_adaptive),
                        "session.moe.offload.host_expert_bytes": "0",
                        "session.moe.offload.budget_group": "test_mixed_group_v4_fail",
                    },
                    cuda_options={"gpu_device_mem_limit": str(32 * 1024 * 1024)},
                )

    def test_v4_prefill_histogram_across_incompatible_expert_counts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_e16 = os.path.join(tmpdir, "e16")
            dir_e64 = os.path.join(tmpdir, "e64")
            dir_het = os.path.join(tmpdir, "het")
            os.makedirs(dir_e16, exist_ok=True)
            os.makedirs(dir_e64, exist_ok=True)
            os.makedirs(dir_het, exist_ok=True)

            m_e16 = _build_canonical_nvfp4_qmoe_model(dir_e16, num_layers=2, num_experts=16, seed=811)
            m_e64 = _build_canonical_nvfp4_qmoe_model(dir_e64, num_layers=2, num_experts=64, seed=812)
            m_het = _build_canonical_nvfp4_qmoe_model(
                dir_het, num_layers=2, num_experts=32, layer_experts=[16, 32], seed=813
            )

            slot_bytes = 13832
            rng = np.random.default_rng(814)

            for m_path, ne in ((m_e16, 16), (m_e64, 64)):
                fixed_bytes = 2 * ne * 4
                budget = 2 * fixed_bytes + 12 * slot_bytes
                s_ref = self._make_session(m_path, {"session.moe.offload.mode": "disabled"})
                s_off = self._make_session(
                    m_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.placement": "adaptive",
                        "session.moe.offload.device_expert_bytes": str(budget),
                    },
                )
                for num_rows in (16, 2):
                    x = rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5
                    r = rng.standard_normal((num_rows, ne), dtype=np.float32)
                    y_ref = self._run_bf16(s_ref, x, r)
                    y_off = self._run_bf16(s_off, x, r)
                    self.assertLess(float(np.max(np.abs(y_ref - y_off))), 5e-2)
                del s_off
                del s_ref

            # Heterogeneous layers (layer 0 has E=16, layer 1 has E=32)
            het_budget = (2 * 16 * 4 + 2 * 32 * 4) + 12 * slot_bytes
            s_het_ref = self._make_session(m_het, {"session.moe.offload.mode": "disabled"})
            s_het_off = self._make_session(
                m_het,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.placement": "adaptive",
                    "session.moe.offload.device_expert_bytes": str(het_budget),
                },
            )
            for num_rows in (16, 2):
                x_bf16 = _float32_to_bfloat16_u16(rng.standard_normal((num_rows, 128), dtype=np.float32) * 0.5)
                r0_bf16 = _float32_to_bfloat16_u16(rng.standard_normal((num_rows, 16), dtype=np.float32))
                r1_bf16 = _float32_to_bfloat16_u16(rng.standard_normal((num_rows, 32), dtype=np.float32))
                feeds = {
                    "X": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(x_bf16, TensorProto.BFLOAT16),
                    "router_probs_0": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r0_bf16, TensorProto.BFLOAT16),
                    "router_probs_1": ort.OrtValue.ortvalue_from_numpy_with_onnx_type(r1_bf16, TensorProto.BFLOAT16),
                }
                out_ref = s_het_ref.run_with_ort_values(["Y"], feeds)[0]
                out_off = s_het_off.run_with_ort_values(["Y"], feeds)[0]
                b_ref = bytes((ctypes.c_uint16 * (num_rows * 128)).from_address(out_ref.data_ptr()))
                b_off = bytes((ctypes.c_uint16 * (num_rows * 128)).from_address(out_off.data_ptr()))
                y_ref = _bfloat16_u16_to_float32(np.frombuffer(b_ref, dtype=np.uint16))
                y_off = _bfloat16_u16_to_float32(np.frombuffer(b_off, dtype=np.uint16))
                self.assertLess(float(np.max(np.abs(y_ref - y_off))), 5e-2)

    def test_v4_pinned_bytes_exact_rounded_and_multilayer_boundaries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dir_1l = os.path.join(tmpdir, "l1")
            dir_2l = os.path.join(tmpdir, "l2")
            os.makedirs(dir_1l, exist_ok=True)
            os.makedirs(dir_2l, exist_ok=True)
            m_1l = _build_canonical_nvfp4_qmoe_model(dir_1l, num_layers=1, num_experts=16, seed=901)
            m_2l = _build_canonical_nvfp4_qmoe_model(dir_2l, num_layers=2, num_experts=16, seed=902)

            slot_bytes = 13832
            fixed_bytes = 2 * 16 * 4
            budget_1l = fixed_bytes + 6 * slot_bytes
            budget_2l = 2 * fixed_bytes + 12 * slot_bytes
            # Actual rounded persistent decode host buffer for E=16:
            # max(min_host_bytes, 256 KiB) + 512 + align256(16 * 4) = 262,144 + 512 + 256 = 262,912 B
            per_layer_decode_pinned = 262912

            rng = np.random.default_rng(903)
            x_dec = rng.standard_normal((2, 128), dtype=np.float32) * 0.5
            r_dec = rng.standard_normal((2, 16), dtype=np.float32)

            # 1-layer below-256KiB (64 KiB) and 1-byte-below boundary (262,911 B) must be rejected before allocation.
            for bad_pinned in (65536, per_layer_decode_pinned - 1):
                s_bad = self._make_session(
                    m_1l,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_1l),
                        "session.moe.offload.pinned_bytes": str(bad_pinned),
                    },
                )
                with self.assertRaisesRegex(Exception, "pinned"):
                    self._run_bf16(s_bad, x_dec, r_dec)
                del s_bad

            # Exact 1-layer boundary (262,912 B) must succeed.
            s_ok_1l = self._make_session(
                m_1l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_1l),
                    "session.moe.offload.pinned_bytes": str(per_layer_decode_pinned),
                },
            )
            _ = self._run_bf16(s_ok_1l, x_dec, r_dec)
            del s_ok_1l

            # 2-layer aggregate limit: 300 KiB or (2 * 262,912 - 1) must reject on the second layer.
            for bad_2l_pinned in (300000, 2 * per_layer_decode_pinned - 1):
                s_bad_2l = self._make_session(
                    m_2l,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_expert_bytes": str(budget_2l),
                        "session.moe.offload.pinned_bytes": str(bad_2l_pinned),
                    },
                )
                with self.assertRaisesRegex(Exception, "pinned"):
                    self._run_bf16(s_bad_2l, x_dec, r_dec)
                del s_bad_2l

            # Exact 2-layer aggregate boundary (525,824 B) must succeed for decode.
            s_ok_2l = self._make_session(
                m_2l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_2l),
                    "session.moe.offload.pinned_bytes": str(2 * per_layer_decode_pinned),
                },
            )
            _ = self._run_bf16(s_ok_2l, x_dec, r_dec)
            del s_ok_2l

            # Prefill host staging (2 * host_pass_bytes = 2 * 111,360 = 222,720 B for E=16, H=128, I=64, wave=8):
            # 222,719 B must be rejected before cudaHostAlloc, while 222,720 B must succeed for prefill.
            prefill_staging_total = 2 * 111360
            s_pref_bad = self._make_session(
                m_1l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_1l),
                    "session.moe.offload.pinned_bytes": str(prefill_staging_total - 1),
                },
            )
            x_pref = rng.standard_normal((16, 128), dtype=np.float32) * 0.5
            r_pref = rng.standard_normal((16, 16), dtype=np.float32)
            with self.assertRaisesRegex(Exception, "pinned"):
                self._run_bf16(s_pref_bad, x_pref, r_pref)
            del s_pref_bad

            s_pref_ok = self._make_session(
                m_1l,
                {
                    "session.moe.offload.mode": "required",
                    "session.moe.offload.device_expert_bytes": str(budget_1l),
                    "session.moe.offload.pinned_bytes": str(prefill_staging_total),
                },
            )
            _ = self._run_bf16(s_pref_ok, x_pref, r_pref)

    def test_v4_ordinary_cuda_ep_and_plugin_controller_budget_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = _build_canonical_nvfp4_qmoe_model(
                tmpdir, num_layers=2, num_experts=16, hidden_size=128, inter_size=64, top_k=4, seed=950
            )
            slot_bytes = 13832
            fixed_bytes = 2 * 16 * 4
            budget_partial = 2 * fixed_bytes + 12 * slot_bytes

            # 1. Explicit ordinary CUDAExecutionProvider providers=[("CUDAExecutionProvider", ...)] partial-residency
            # hybrid decode (cpu_execution="auto", T=1, 2, 4).
            sess_ref = ort.InferenceSession(model_path, providers=[("CUDAExecutionProvider", {})])
            so_ord = ort.SessionOptions()
            so_ord.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            so_ord.add_session_config_entry("session.moe.offload.mode", "required")
            so_ord.add_session_config_entry("session.moe.offload.cpu_execution", "auto")
            so_ord.add_session_config_entry("session.moe.offload.device_expert_bytes", str(budget_partial))
            sess_ord = ort.InferenceSession(model_path, sess_options=so_ord, providers=[("CUDAExecutionProvider", {})])

            rng = np.random.default_rng(951)
            for t_rows in (1, 2, 4):
                x = rng.standard_normal((t_rows, 128), dtype=np.float32) * 0.5
                r = rng.standard_normal((t_rows, 16), dtype=np.float32)
                y_ref = self._run_bf16(sess_ref, x, r)
                y_ord = self._run_bf16(sess_ord, x, r)
                self.assertLess(float(np.max(np.abs(y_ref - y_ord))), 5e-2)
            del sess_ord
            del sess_ref

            # 2. Verify QMoESessionController holds the session's BudgetAccount and enforces gpu_device_mem_limit
            # in both session planning and arena/landing-ring allocations (Finding 6).
            with self.assertRaisesRegex(Exception, "MoE Offload session admission failed"):
                self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.device_working_bytes": str(8 * 1024 * 1024),
                    },
                    cuda_options={"gpu_device_mem_limit": str(4 * 1024 * 1024)},
                )
            with self.assertRaisesRegex(Exception, "(?i)budget|memory|allocation"):
                sess_tight = self._make_session(
                    model_path,
                    {
                        "session.moe.offload.mode": "required",
                        "session.moe.offload.cpu_execution": "disabled",
                        "session.moe.offload.device_expert_bytes": str(budget_partial),
                    },
                    cuda_options={"gpu_device_mem_limit": str(budget_partial + 64 * 1024)},
                )
                self._run_bf16(
                    sess_tight,
                    np.ones((1, 128), dtype=np.float32),
                    np.ones((1, 16), dtype=np.float32),
                )

    @unittest.skipIf(sys.platform == "win32", "Requires POSIX g++ toolchain for standalone header/allocator probe.")
    def test_v4_budget_group_migration_and_concurrent_allocator_address_reuse(self):
        repo_root = Path(__file__).resolve().parents[3]
        budget_hdr = repo_root / "onnxruntime/core/providers/cuda/cuda_device_memory_budget.h"
        if not budget_hdr.is_file():
            self.skipTest(f"Header not found at {budget_hdr}")
        probe_src = r"""
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "core/providers/cuda/cuda_device_memory_budget.h"

int main() {
  using namespace onnxruntime;
  // 1. Committed allocation attached to a new BudgetGroup before BeginFree/FinishFree.
  {
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    SessionBudgetAccount acct(phys, 1024 * 1024);
    acct.Reserve(4096).Commit();
    auto grp = std::make_shared<BudgetGroup>(phys, 512 * 1024);
    acct.SetBudgetGroup(grp);
    assert(acct.GetAccountStats().committed == 4096);
    assert(acct.GetGroupStats().committed == 4096);
    acct.BeginFree(4096);
    acct.FinishFree(4096, true);
    assert(acct.GetAccountStats().committed == 0);
    assert(acct.GetGroupStats().committed == 0);
    assert(phys->GetStats().committed == 0);
  }
  // 2. Pending reservation and in-flight freeing migrated across SetBudgetGroup.
  {
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    SessionBudgetAccount acct(phys, 1024 * 1024);
    auto res_pending = acct.Reserve(2048);
    acct.Reserve(8192).Commit();
    acct.BeginFree(8192);
    auto grp = std::make_shared<BudgetGroup>(phys, 512 * 1024);
    acct.SetBudgetGroup(grp);
    res_pending.Commit();
    acct.FinishFree(8192, true);
    assert(acct.GetGroupStats().committed == 2048);
    acct.BeginFree(2048);
    acct.FinishFree(2048, true);
    assert(acct.GetAccountStats().committed == 0);
    assert(acct.GetGroupStats().committed == 0);
    assert(phys->GetStats().committed == 0);
  }
  // 3. Simulated concurrent address reuse between driver free and metadata cleanup.
  {
    enum class State : uint8_t { kCommitted = 0, kFreeing = 1, kQuarantined = 2 };
    struct Rec { size_t bytes; uint64_t generation; State state; };
    std::mutex mu;
    uint64_t next_gen = 0;
    std::unordered_map<void*, Rec> allocs;
    auto phys = std::make_shared<CudaDeviceMemoryBudget>();
    SessionBudgetAccount acct(phys, 1024 * 1024);
    void* reused_addr = reinterpret_cast<void*>(uintptr_t{0xdeadbeef0000ULL});

    auto alloc_at = [&](void* ptr, size_t sz) {
      auto r = acct.Reserve(sz);
      std::lock_guard<std::mutex> lk(mu);
      const uint64_t g = ++next_gen;
      auto [it, ins] = allocs.emplace(ptr, Rec{sz, g, State::kCommitted});
      if (!ins) {
        assert(it->second.state != State::kCommitted);
        it->second = Rec{sz, g, State::kCommitted};
      }
      r.Commit();
    };

    alloc_at(reused_addr, 4096);
    // Begin Free(reused_addr) on thread A
    uint64_t freeing_gen = 0;
    size_t freeing_bytes = 0;
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = allocs.find(reused_addr);
      assert(it != allocs.end() && it->second.state == State::kCommitted);
      it->second.state = State::kFreeing;
      freeing_bytes = it->second.bytes;
      freeing_gen = it->second.generation;
      acct.BeginFree(freeing_bytes);
    }
    // Driver frees reused_addr; before thread A reacquires mu, thread B allocates the same address!
    alloc_at(reused_addr, 16384);
    // Thread A reacquires mu to finish its free:
    {
      std::lock_guard<std::mutex> lk(mu);
      acct.FinishFree(freeing_bytes, true);
      auto it = allocs.find(reused_addr);
      if (it != allocs.end() && it->second.generation == freeing_gen) {
        allocs.erase(it);
      }
    }
    // The new 16384-byte allocation at reused_addr must still be tracked and committed!
    assert(acct.GetAccountStats().committed == 16384);
    assert(allocs.count(reused_addr) == 1 && allocs[reused_addr].bytes == 16384);
    // Freeing the second allocation must discharge 100% of the budget to 0.
    {
      std::lock_guard<std::mutex> lk(mu);
      auto it = allocs.find(reused_addr);
      assert(it != allocs.end() && it->second.state == State::kCommitted);
      const size_t b2 = it->second.bytes;
      const uint64_t g2 = it->second.generation;
      it->second.state = State::kFreeing;
      acct.BeginFree(b2);
      acct.FinishFree(b2, true);
      if (it->second.generation == g2) allocs.erase(it);
    }
    assert(acct.GetAccountStats().committed == 0);
    assert(phys->GetStats().committed == 0);
  }
  return 0;
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src_file = Path(tmpdir) / "probe.cc"
            bin_file = Path(tmpdir) / "probe"
            src_file.write_text(probe_src)
            subprocess.check_call(
                ["g++", "-O2", "-std=c++17", f"-I{repo_root / 'onnxruntime'}", str(src_file), "-o", str(bin_file)]
            )
            subprocess.check_call([str(bin_file)])


if __name__ == "__main__":
    unittest.main()

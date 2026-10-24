"""Tests for SharedExpertBlockV1: numpy packer, CPU kernel, and (when a GPU is available) the CUDA golden check."""

from __future__ import annotations

import ctypes
import functools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qmoe_shared_layout_ref as ref


def _find_src_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        for cand in (p, p / "onnxruntime"):
            if (cand / "contrib_ops" / "cpu" / "moe" / "qmoe_shared_layout.cc").is_file():
                return cand
    return here.parents[3]


SRC_ROOT = _find_src_root()
KERNEL_SRC = SRC_ROOT / "contrib_ops" / "cpu" / "moe" / "qmoe_shared_layout.cc"


@functools.lru_cache(maxsize=1)
def load_lib():
    if not KERNEL_SRC.is_file():
        raise unittest.SkipTest(f"C++ source {KERNEL_SRC} not found")
    out = Path(tempfile.mkdtemp()) / "libqmoe_shared_layout.so"
    subprocess.check_call(
        [
            "g++",
            "-O2",
            "-std=c++17",
            "-shared",
            "-fPIC",
            f"-I{SRC_ROOT}",
            f"-I{SRC_ROOT.parent / 'include/onnxruntime'}",
            str(KERNEL_SRC),
            "-o",
            str(out),
        ]
    )
    lib = ctypes.CDLL(str(out))
    lib.qmoe_shared_expert_forward.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
    ]
    lib.qmoe_shared_expert_forward.restype = ctypes.c_int
    lib.qmoe_shared_expert_contribution.argtypes = lib.qmoe_shared_expert_forward.argtypes
    lib.qmoe_shared_expert_contribution.restype = ctypes.c_int
    lib.qmoe_shared_expert_contribution_fp16.argtypes = lib.qmoe_shared_expert_forward.argtypes
    lib.qmoe_shared_expert_contribution_fp16.restype = ctypes.c_int
    lib.qmoe_shared_block_bytes.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.qmoe_shared_block_bytes.restype = ctypes.c_int64
    lib.qmoe_shared_cpu_has_avx512.restype = ctypes.c_int
    lib.qmoe_shared_accepts_prepack_contract.argtypes = [ctypes.c_int64, ctypes.c_int64]
    lib.qmoe_shared_accepts_prepack_contract.restype = ctypes.c_int
    lib.qmoe_shared_weight_code_at.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    lib.qmoe_shared_weight_code_at.restype = ctypes.c_int
    lib.qmoe_int4_expert_contribution.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.qmoe_int4_expert_contribution.restype = ctypes.c_int
    lib.qmoe_shared_half_bits_to_float_row.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.c_int,
    ]
    lib.qmoe_shared_half_bits_to_float_row.restype = None
    lib.qmoe_shared_is_file_backed_mapping.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.qmoe_shared_is_file_backed_mapping.restype = ctypes.c_int
    lib.qmoe_shared_evict_file_backed_pages.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.qmoe_shared_evict_file_backed_pages.restype = ctypes.c_size_t
    lib.qmoe_file_store_test_heap_eviction_safety.argtypes = []
    lib.qmoe_file_store_test_heap_eviction_safety.restype = ctypes.c_int
    return lib


def random_expert(rng, hidden, inter):
    n1 = 2 * inter
    return dict(
        fc1_codes=rng.integers(0, 16, (n1, hidden), dtype=np.uint8),
        fc1_scales=(rng.random((n1, hidden // 32)) * 0.02 + 0.001).astype(np.float16),
        fc1_bias=((rng.random(n1) - 0.5) * 0.1).astype(np.float16),
        fc2_codes=rng.integers(0, 16, (hidden, inter), dtype=np.uint8),
        fc2_scales=(rng.random((hidden, inter // 32)) * 0.02 + 0.001).astype(np.float16),
        fc2_bias=((rng.random(hidden) - 0.5) * 0.1).astype(np.float16),
    )


def run_cpu(lib, block, hidden, inter, x, force_scalar, swiglu=(1.0, 0.0, float("inf"))):
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.zeros((x.shape[0], hidden), dtype=np.float32)
    ok = lib.qmoe_shared_expert_forward(
        block.ctypes.data, hidden, inter, x.ctypes.data, x.shape[0], out.ctypes.data, *swiglu, int(force_scalar)
    )
    assert ok == 1
    return out


@unittest.skipIf(sys.platform == "win32", "SharedLayoutTest requires POSIX g++ toolchain to build shared test library.")
class SharedLayoutTest(unittest.TestCase):
    def test_fp16_boundaries_match_explicit_reference(self):
        rng = np.random.default_rng(20261009)
        hidden, inter = 128, 64
        expert = random_expert(rng, hidden, inter)
        expert["fc1_scales"][:] = np.float16(1 / 64)
        expert["fc2_scales"][:] = np.float16(1 / 64)
        block = ref.assemble_block(hidden, inter, **expert)
        original = block.copy()
        weights1 = (expert["fc1_codes"].astype(np.float32) - 8) / 64
        weights2 = (expert["fc2_codes"].astype(np.float32) - 8) / 64
        lib = load_lib()
        for rows in (1, 2, 3, 8):
            inputs = rng.integers(-32, 33, (rows, hidden)).astype(np.float32) / 32
            for alpha, beta, limit in ((1.0, 0.0, float("inf")), (1.702, 1.0, 0.75)):
                projected = (
                    (inputs @ weights1.T + expert["fc1_bias"].astype(np.float32)).astype(np.float16).astype(np.float32)
                )
                gate = np.minimum(projected[:, 0::2], np.float32(limit))
                up = np.clip(projected[:, 1::2], -np.float32(limit), np.float32(limit))
                activated = (gate / (1 + np.exp(-np.float32(alpha) * gate)) * (up + np.float32(beta))).astype(
                    np.float16
                )
                expected = activated.astype(np.float32) @ weights2.T
                for scalar in (True, False):
                    actual = np.empty((rows, hidden), dtype=np.float32)
                    self.assertEqual(
                        lib.qmoe_shared_expert_contribution_fp16(
                            block.ctypes.data,
                            hidden,
                            inter,
                            inputs.ctypes.data,
                            rows,
                            actual.ctypes.data,
                            alpha,
                            beta,
                            limit,
                            int(scalar),
                        ),
                        1,
                    )
                    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6)
        np.testing.assert_array_equal(block, original)

    def test_bias_free_contribution_preserves_legacy_forward(self):
        rng = np.random.default_rng(314)
        hidden, inter = 128, 64
        expert = random_expert(rng, hidden, inter)
        block = ref.assemble_block(hidden, inter, **expert)
        original = block.copy()
        inputs = rng.standard_normal((3, hidden)).astype(np.float32)
        lib = load_lib()
        unbiased = dict(expert, fc2_bias=np.zeros(hidden, dtype=np.float16))
        expected = ref.reference_expert(inputs, **unbiased)
        for scalar in (True, False):
            result = np.empty_like(inputs)
            self.assertEqual(
                lib.qmoe_shared_expert_contribution(
                    block.ctypes.data,
                    hidden,
                    inter,
                    inputs.ctypes.data,
                    len(inputs),
                    result.ctypes.data,
                    1.0,
                    0.0,
                    float("inf"),
                    int(scalar),
                ),
                1,
            )
            np.testing.assert_allclose(result, expected, rtol=2e-4, atol=2e-4)
            legacy = run_cpu(lib, block, hidden, inter, inputs, scalar)
            np.testing.assert_allclose(legacy, result + expert["fc2_bias"], rtol=2e-4, atol=2e-4)
        np.testing.assert_array_equal(block, original)

    def test_qwen_block_bytes_match_design(self):
        lay = ref.block_layout(2560, 640)
        self.assertEqual(lay["fc1_weights"], 0)
        self.assertEqual(lay["fc1_scales"], 1638400)
        self.assertEqual(lay["fc1_bias"], 1843200)
        self.assertEqual(lay["fc2_weights"], 1845760)
        self.assertEqual(lay["fc2_scales"], 2664960)
        self.assertEqual(lay["fc2_bias"], 2767360)
        self.assertEqual(lay["stride"], 2772480)
        self.assertEqual(load_lib().qmoe_shared_block_bytes(2560, 640), 2772480)
        self.assertEqual(load_lib().qmoe_shared_block_bytes(100, 640), -1)

    def test_offline_prepack_contract(self):
        # weights_prepacked == 1 (offline CUDA SM80 order) and block_size == 32 only.
        lib = load_lib()
        self.assertEqual(lib.qmoe_shared_accepts_prepack_contract(1, 32), 1)
        for bad in (0, -1, 2):
            self.assertEqual(lib.qmoe_shared_accepts_prepack_contract(bad, 32), 0)
        for bad_block in (0, 16, 64, 128):
            self.assertEqual(lib.qmoe_shared_accepts_prepack_contract(1, bad_block), 0)

    def test_k_order_table(self):
        # Low nibble of byte b holds original K (b>>1)+16*(b&1); high nibble holds +8.
        codes = np.zeros((4, 64), dtype=np.uint8)
        for k in range(32):
            codes[:] = 0
            codes[0, k] = 15
            packed = ref.pack_sm80_w4(codes)
            nz = np.nonzero(packed)[0]
            self.assertEqual(len(nz), 1)
            byte = int(nz[0])
            self.assertLess(byte, 16)  # channel 0, first scale block
            low = (byte >> 1) + 16 * (byte & 1)
            is_low = (packed[byte] & 0x0F) == 15
            self.assertEqual(low if is_low else low + 8, k)

    def test_weight_code_index_formula(self):
        rng = np.random.default_rng(1)
        lib = load_lib()
        n, k = 16, 192
        codes = rng.integers(0, 16, (n, k), dtype=np.uint8)
        packed = ref.pack_sm80_w4(codes)
        for nn in range(n):
            for kk in range(k):
                self.assertEqual(lib.qmoe_shared_weight_code_at(packed.ctypes.data, n, k, nn, kk), codes[nn, kk])

    def _forward_case(self, hidden, inter, m, force_scalar, seed):
        rng = np.random.default_rng(seed)
        e = random_expert(rng, hidden, inter)
        block = ref.assemble_block(hidden, inter, **e)
        x = (rng.standard_normal((m, hidden)) * 0.5).astype(np.float32)
        got = run_cpu(load_lib(), block, hidden, inter, x, force_scalar)
        want = ref.reference_expert(x, **e)
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)

    def test_prepacked_conversion_matches_codes_path(self):
        rng = np.random.default_rng(8)
        hidden, inter, num_experts = 128, 64, 3
        experts = [random_expert(rng, hidden, inter) for _ in range(num_experts)]
        st = lambda key: np.stack([e[key] for e in experts])  # noqa: E731
        w1 = np.stack([ref.pack_sm80_w4(e["fc1_codes"]).reshape(2 * inter, hidden // 2) for e in experts])
        w2 = np.stack([ref.pack_sm80_w4(e["fc2_codes"]).reshape(hidden, inter // 2) for e in experts])
        got = ref.assemble_blocks_from_prepacked(
            hidden, inter, w1, st("fc1_scales"), st("fc1_bias"), w2, st("fc2_scales"), st("fc2_bias")
        )
        for i, e in enumerate(experts):
            np.testing.assert_array_equal(got[i], ref.assemble_block(hidden, inter, **e))

    def test_forward_small_scalar_and_avx512(self):
        for scalar in (True, False):
            for m in (1, 2, 3, 4, 5, 8):
                self._forward_case(128, 64, m, scalar, 10 + m)

    def test_forward_swiglu_params(self):
        rng = np.random.default_rng(3)
        e = random_expert(rng, 128, 64)
        block = ref.assemble_block(128, 64, **e)
        x = (rng.standard_normal((2, 128)) * 0.5).astype(np.float32)
        p = (1.702, 1.0, 7.0)
        got = run_cpu(load_lib(), block, 128, 64, x, False, p)
        want = ref.reference_expert(x, **e, alpha=p[0], beta=p[1], limit=p[2])
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-4)

    def test_forward_qwen_shape(self):
        self._forward_case(2560, 640, 1, False, 99)
        self._forward_case(2560, 640, 3, True, 100)

    def test_scalar_and_avx512_agree(self):
        if not load_lib().qmoe_shared_cpu_has_avx512():
            self.skipTest("no AVX512")
        rng = np.random.default_rng(5)
        e = random_expert(rng, 512, 128)
        block = ref.assemble_block(512, 128, **e)
        x = rng.standard_normal((4, 512)).astype(np.float32)
        a = run_cpu(load_lib(), block, 512, 128, x, True)
        b = run_cpu(load_lib(), block, 512, 128, x, False)
        np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)

    def test_fp16_subnormal_and_exhaustive_scalar_simd_parity(self):
        lib = load_lib()
        # Check positive and negative subnormal boundary bit patterns explicitly.
        boundary_bits = np.array([0x0001, 0x03FF, 0x0400, 0x8001, 0x83FF, 0x8400], dtype=np.uint16)
        expected_boundary = boundary_bits.view(np.float16).astype(np.float32)
        for force_scalar in (1, 0):
            got_boundary = np.empty(len(boundary_bits), dtype=np.float32)
            lib.qmoe_shared_half_bits_to_float_row(
                boundary_bits.ctypes.data, got_boundary.ctypes.data, len(boundary_bits), force_scalar
            )
            np.testing.assert_array_equal(got_boundary, expected_boundary)

        # Exhaustively test all 65,536 16-bit half patterns (excluding NaNs) for scalar vs SIMD vs IEEE FP16.
        all_bits = np.arange(65536, dtype=np.uint16)
        scalar_out = np.empty(65536, dtype=np.float32)
        simd_out = np.empty(65536, dtype=np.float32)
        lib.qmoe_shared_half_bits_to_float_row(all_bits.ctypes.data, scalar_out.ctypes.data, 65536, 1)
        lib.qmoe_shared_half_bits_to_float_row(all_bits.ctypes.data, simd_out.ctypes.data, 65536, 0)
        ref_out = all_bits.view(np.float16).astype(np.float32)
        non_nan = ~np.isnan(ref_out)
        np.testing.assert_array_equal(scalar_out[non_nan].view(np.uint32), ref_out[non_nan].view(np.uint32))
        np.testing.assert_array_equal(simd_out[non_nan].view(np.uint32), ref_out[non_nan].view(np.uint32))
        self.assertTrue(np.all(np.isnan(scalar_out[~non_nan])))
        self.assertTrue(np.all(np.isnan(simd_out[~non_nan])))

    @unittest.skipUnless(sys.platform == "linux", "Mapped expert-store backing discovery requires Linux.")
    def test_file_expert_store_heap_eviction_preserves_weights(self):
        lib = load_lib()
        self.assertEqual(lib.qmoe_file_store_test_heap_eviction_safety(), 1)

    def test_canonical_int4_qmoe_banks_prepacked_and_raw(self):
        rng = np.random.default_rng(20261023)
        hidden, inter = 128, 64
        expert = random_expert(rng, hidden, inter)
        w1_raw = np.ascontiguousarray(ref.raw_pack(expert["fc1_codes"]))
        w2_raw = np.ascontiguousarray(ref.raw_pack(expert["fc2_codes"]))
        w1_pk = np.ascontiguousarray(ref.pack_sm80_w4(expert["fc1_codes"]))
        w2_pk = np.ascontiguousarray(ref.pack_sm80_w4(expert["fc2_codes"]))
        s1 = np.ascontiguousarray(expert["fc1_scales"].view(np.uint16))
        s2 = np.ascontiguousarray(expert["fc2_scales"].view(np.uint16))
        b1 = np.ascontiguousarray(expert["fc1_bias"].view(np.uint16))
        b2 = np.ascontiguousarray(expert["fc2_bias"].view(np.uint16))

        lib = load_lib()
        for rows in (1, 3, 4, 8):
            x = (rng.standard_normal((rows, hidden)) * 0.5).astype(np.float32)
            want_full = ref.reference_expert(x, **expert, alpha=1.702, beta=1.0, limit=7.0)
            unbiased = dict(expert, fc2_bias=np.zeros(hidden, dtype=np.float16))
            want_contrib = ref.reference_expert(x, **unbiased, alpha=1.702, beta=1.0, limit=7.0)
            for prepacked, (w1, w2) in ((1, (w1_pk, w2_pk)), (0, (w1_raw, w2_raw))):
                for force_scalar in (1, 0):
                    got_full = np.empty((rows, hidden), dtype=np.float32)
                    ok = lib.qmoe_int4_expert_contribution(
                        w1.ctypes.data,
                        s1.ctypes.data,
                        b1.ctypes.data,
                        w2.ctypes.data,
                        s2.ctypes.data,
                        b2.ctypes.data,
                        hidden,
                        inter,
                        x.ctypes.data,
                        rows,
                        got_full.ctypes.data,
                        1.702,
                        1.0,
                        7.0,
                        prepacked,
                        0,
                        1,
                        force_scalar,
                    )
                    self.assertEqual(ok, 1)
                    np.testing.assert_allclose(got_full, want_full, rtol=2e-3, atol=2e-3)

                    got_contrib = np.empty((rows, hidden), dtype=np.float32)
                    ok2 = lib.qmoe_int4_expert_contribution(
                        w1.ctypes.data,
                        s1.ctypes.data,
                        b1.ctypes.data,
                        w2.ctypes.data,
                        s2.ctypes.data,
                        None,
                        hidden,
                        inter,
                        x.ctypes.data,
                        rows,
                        got_contrib.ctypes.data,
                        1.702,
                        1.0,
                        7.0,
                        prepacked,
                        0,
                        0,
                        force_scalar,
                    )
                    self.assertEqual(ok2, 1)
                    np.testing.assert_allclose(got_contrib, want_contrib, rtol=2e-3, atol=2e-3)

    def test_cpu_qmoe_prepacked_and_raw_parity(self):
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError:
            ort = None
        if ort is None:
            self.skipTest("onnxruntime python package is not available on PYTHONPATH")

        rng = np.random.default_rng(20261024)
        hidden, inter, tokens, num_experts, top_k = 128, 64, 4, 4, 2
        experts = [random_expert(rng, hidden, inter) for _ in range(num_experts)]
        n1 = 2 * inter

        def stack(key):
            return np.stack([e[key] for e in experts])

        raw1 = np.stack([ref.raw_pack(e["fc1_codes"]) for e in experts])
        raw2 = np.stack([ref.raw_pack(e["fc2_codes"]) for e in experts])
        pk1 = np.stack([ref.pack_sm80_w4(e["fc1_codes"]).reshape(n1, hidden // 2) for e in experts])
        pk2 = np.stack([ref.pack_sm80_w4(e["fc2_codes"]).reshape(hidden, inter // 2) for e in experts])
        pk1_legacy = pk1.reshape(num_experts, hidden, inter)
        pk2_legacy = pk2.reshape(num_experts, inter, hidden // 2)
        args = (num_experts, hidden, inter, top_k)
        common = (stack("fc1_scales"), stack("fc1_bias"))
        m_raw = build_qmoe_model(*args, raw1, common[0], common[1], raw2, stack("fc2_scales"), stack("fc2_bias"), False)
        m_pk = build_qmoe_model(*args, pk1, common[0], common[1], pk2, stack("fc2_scales"), stack("fc2_bias"), True)
        m_pk_legacy = build_qmoe_model(
            *args,
            pk1_legacy,
            common[0],
            common[1],
            pk2_legacy,
            stack("fc2_scales"),
            stack("fc2_bias"),
            True,
            w1_shape=(num_experts, hidden, inter),
            w2_shape=(num_experts, inter, hidden // 2),
        )
        x = (rng.standard_normal((tokens, hidden)) * 0.5).astype(np.float16)
        router = rng.standard_normal((tokens, num_experts)).astype(np.float16)

        m_invalid = build_qmoe_model(*args, raw1, common[0], common[1], raw2, stack("fc2_scales"), stack("fc2_bias"), 2)
        try:
            ort.InferenceSession(m_invalid, providers=["CPUExecutionProvider"])
        except Exception:
            pass
        else:
            self.skipTest(
                "Loaded onnxruntime package predates CPU QMoE weights_prepacked support; "
                "set PYTHONPATH to the current build directory."
            )

        for disable_prepack in ("0", "1"):
            so = ort.SessionOptions()
            so.add_session_config_entry("session.disable_prepacking", disable_prepack)
            sess_raw = ort.InferenceSession(m_raw, sess_options=so, providers=["CPUExecutionProvider"])
            sess_pk = ort.InferenceSession(m_pk, sess_options=so, providers=["CPUExecutionProvider"])
            sess_pk_legacy = ort.InferenceSession(m_pk_legacy, sess_options=so, providers=["CPUExecutionProvider"])
            y_raw = sess_raw.run(None, {"x": x, "router": router})[0]
            y_pk = sess_pk.run(None, {"x": x, "router": router})[0]
            y_pk_legacy = sess_pk_legacy.run(None, {"x": x, "router": router})[0]
            np.testing.assert_array_equal(y_raw, y_pk)
            np.testing.assert_array_equal(y_raw, y_pk_legacy)

        lib = load_lib()
        blocks = [ref.assemble_block(hidden, inter, **e) for e in experts]
        want = np.zeros((tokens, hidden), dtype=np.float64)
        for t in range(tokens):
            top = np.argsort(-router[t].astype(np.float32), kind="stable")[:top_k]
            logits = router[t][top].astype(np.float64)
            w = np.exp(logits - logits.max())
            w /= w.sum()
            for wi, e in zip(w, top, strict=False):
                y = run_cpu(lib, blocks[e], hidden, inter, x[t : t + 1].astype(np.float32), False)
                want[t] += wi * y[0]
        np.testing.assert_allclose(y_pk.astype(np.float64), want, rtol=1e-2, atol=5e-3)


def build_qmoe_model(
    num_experts,
    hidden,
    inter,
    top_k,
    w1,
    s1,
    b1,
    w2,
    s2,
    b2,
    prepacked,
    w1_shape=None,
    w2_shape=None,
):
    import onnx  # noqa: PLC0415

    n1 = 2 * inter
    tokens = "tokens"
    inits = [
        onnx.numpy_helper.from_array(np.ascontiguousarray(a).reshape(shape), name)
        for name, a, shape in [
            ("w1", w1, w1_shape or (num_experts, n1, hidden // 2)),
            ("s1", s1, (num_experts, n1, hidden // 32)),
            ("b1", b1, (num_experts, n1)),
            ("w2", w2, w2_shape or (num_experts, hidden, inter // 2)),
            ("s2", s2, (num_experts, hidden, inter // 32)),
            ("b2", b2, (num_experts, hidden)),
        ]
    ]
    node = onnx.helper.make_node(
        "QMoE",
        ["x", "router", "w1", "s1", "b1", "w2", "s2", "b2"],
        ["y"],
        domain="com.microsoft",
        k=top_k,
        activation_type="swiglu",
        swiglu_fusion=1,
        normalize_routing_weights=1,
        expert_weight_bits=4,
        block_size=32,
        weights_prepacked=int(prepacked) if not isinstance(prepacked, bool) else (1 if prepacked else 0),
    )
    graph = onnx.helper.make_graph(
        [node],
        "qmoe_golden",
        [
            onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT16, [tokens, hidden]),
            onnx.helper.make_tensor_value_info("router", onnx.TensorProto.FLOAT16, [tokens, num_experts]),
        ],
        [onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT16, [tokens, hidden])],
        initializer=inits,
    )
    model = onnx.helper.make_model(
        graph, opset_imports=[onnx.helper.make_opsetid("", 22), onnx.helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model.SerializeToString()


@unittest.skipUnless(
    os.environ.get("QMOE_GOLDEN_CUDA") == "1", "set QMOE_GOLDEN_CUDA=1 with CUDA_VISIBLE_DEVICES to run"
)
class CudaGoldenTest(unittest.TestCase):
    """The numpy packer must equal the CUDA PrePack, and the CPU kernel must match the CUDA op on the same bytes."""

    def _run(self, hidden, inter, tokens, num_experts=8, top_k=2, seed=0):
        import onnxruntime as ort  # noqa: PLC0415

        rng = np.random.default_rng(seed)
        experts = [random_expert(rng, hidden, inter) for _ in range(num_experts)]
        n1 = 2 * inter

        def stack(key):
            return np.stack([e[key] for e in experts])

        raw1 = np.stack([ref.raw_pack(e["fc1_codes"]) for e in experts])
        raw2 = np.stack([ref.raw_pack(e["fc2_codes"]) for e in experts])
        pk1 = np.stack([ref.pack_sm80_w4(e["fc1_codes"]).reshape(n1, hidden // 2) for e in experts])
        pk2 = np.stack([ref.pack_sm80_w4(e["fc2_codes"]).reshape(hidden, inter // 2) for e in experts])
        args = (num_experts, hidden, inter, top_k)
        common = (stack("fc1_scales"), stack("fc1_bias"))
        m_raw = build_qmoe_model(*args, raw1, common[0], common[1], raw2, stack("fc2_scales"), stack("fc2_bias"), False)
        m_pk = build_qmoe_model(*args, pk1, common[0], common[1], pk2, stack("fc2_scales"), stack("fc2_bias"), True)
        x = (rng.standard_normal((tokens, hidden)) * 0.5).astype(np.float16)
        router = rng.standard_normal((tokens, num_experts)).astype(np.float16)

        outs = []
        for model in (m_raw, m_pk):
            sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider"])
            outs.append(sess.run(None, {"x": x, "router": router})[0])
        np.testing.assert_array_equal(outs[0], outs[1], err_msg="numpy packer differs from CUDA PrePack")

        lib = load_lib()
        blocks = [ref.assemble_block(hidden, inter, **e) for e in experts]
        want = np.zeros((tokens, hidden), dtype=np.float64)
        for t in range(tokens):
            top = np.argsort(-router[t].astype(np.float32), kind="stable")[:top_k]
            logits = router[t][top].astype(np.float64)
            w = np.exp(logits - logits.max())
            w /= w.sum()
            for wi, e in zip(w, top, strict=False):
                y = run_cpu(lib, blocks[e], hidden, inter, x[t : t + 1].astype(np.float32), False)
                want[t] += wi * y[0]
        np.testing.assert_allclose(outs[0].astype(np.float64), want, rtol=1e-2, atol=5e-3)

    def test_grouped_gemm_path(self):
        self._run(256, 128, 16)

    def test_gemv_path(self):
        self._run(512, 512, 1)
        self._run(512, 512, 3)


@unittest.skipUnless(os.environ.get("QMOE_REAL_ARTIFACT"), "set QMOE_REAL_ARTIFACT to an offload artifact directory")
class RealArtifactGoldenTest(unittest.TestCase):
    """Real prepacked banks (first experts of a layer) -> shared blocks; CPU kernel must match the CUDA op."""

    def _banks(self, layer, experts):
        import onnx  # noqa: PLC0415

        root = Path(os.environ["QMOE_REAL_ARTIFACT"])
        model = onnx.load(str(root / "text.offload.onnx"), load_external_data=False)
        init = {t.name: t for t in model.graph.initializer}
        prefix = f"model.layers.{layer}.moe.experts."

        def read(name, dtype, shape):
            t = init[prefix + name]
            ext = {e.key: e.value for e in t.external_data}
            per = int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
            data = np.fromfile(root / ext["location"], dtype=np.uint8, count=per * experts, offset=int(ext["offset"]))
            return data.view(dtype).reshape((experts, *shape[1:]))

        return dict(
            w1=read("gate_up_proj.qweight", np.uint8, (512, 2560, 640)),
            s1=read("gate_up_proj.scales", np.float16, (512, 1280, 80)),
            b1=read("gate_up_proj.bias", np.float16, (512, 1280)),
            w2=read("down_proj.qweight", np.uint8, (512, 640, 1280)),
            s2=read("down_proj.scales", np.float16, (512, 2560, 20)),
            b2=read("down_proj.bias", np.float16, (512, 2560)),
        )

    def _run(self, layer, tokens, experts=8, top_k=2):
        import onnxruntime as ort  # noqa: PLC0415

        hidden, inter = 2560, 640
        banks = self._banks(layer, experts)
        w1 = banks["w1"].reshape(experts, 2 * inter, hidden // 2)
        w2 = banks["w2"].reshape(experts, hidden, inter // 2)
        model = build_qmoe_model(
            experts, hidden, inter, top_k, w1, banks["s1"], banks["b1"], w2, banks["s2"], banks["b2"], True
        )
        rng = np.random.default_rng(layer)
        x = (rng.standard_normal((tokens, hidden)) * 0.5).astype(np.float16)
        router = rng.standard_normal((tokens, experts)).astype(np.float16)
        got = ort.InferenceSession(model, providers=["CUDAExecutionProvider"]).run(None, {"x": x, "router": router})[0]

        blocks = ref.assemble_blocks_from_prepacked(
            hidden, inter, w1, banks["s1"], banks["b1"], w2, banks["s2"], banks["b2"]
        )
        self.assertEqual(blocks.shape[1], 2772480)
        lib = load_lib()
        want = np.zeros((tokens, hidden), dtype=np.float64)
        for tok in range(tokens):
            top = np.argsort(-router[tok].astype(np.float32), kind="stable")[:top_k]
            logits = router[tok][top].astype(np.float64)
            w = np.exp(logits - logits.max())
            w /= w.sum()
            for wi, e in zip(w, top, strict=False):
                want[tok] += wi * run_cpu(lib, blocks[e], hidden, inter, x[tok : tok + 1].astype(np.float32), False)[0]
        err = np.abs(got.astype(np.float64) - want)
        print(f"layer {layer} tokens {tokens}: max abs err {err.max():.5f}, max |want| {np.abs(want).max():.4f}")
        np.testing.assert_allclose(got.astype(np.float64), want, rtol=2e-2, atol=1e-2)

    def test_real_layers(self):
        for layer in (0, 17, 47):
            self._run(layer, 1)
        self._run(0, 4)


if __name__ == "__main__":
    unittest.main()

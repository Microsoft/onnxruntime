# SharedExpertBlockV1 numpy reference: packer, block assembly, and float64 expert forward.
# Layout is documented in dev/docs/design/qwen38_flash_next_int4_24g_offload.md.
from __future__ import annotations

import numpy as np

# kPerm_W4_A16 (contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.h).
K_PERM = np.array(
    [
        0,
        1,
        8,
        9,
        16,
        17,
        24,
        25,
        2,
        3,
        10,
        11,
        18,
        19,
        26,
        27,
        4,
        5,
        12,
        13,
        20,
        21,
        28,
        29,
        6,
        7,
        14,
        15,
        22,
        23,
        30,
        31,
    ],
    dtype=np.int64,
)
BLOCK = 32
COLS_INTERLEAVED = 4
ROWS_PER_TILE = 64
ALIGN = 256


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def nibble_dest(i: int) -> int:
    return i // 2 if i % 2 == 0 else (i - 1) // 2 + 4


def raw_pack(codes: np.ndarray) -> np.ndarray:
    """ONNX raw INT4 [N, K/2]: low nibble is the even K element."""
    n, k = codes.shape
    c = codes.astype(np.uint8).reshape(n, k // 2, 2)
    return (c[:, :, 0] | (c[:, :, 1] << 4)).astype(np.uint8)


def pack_sm80_w4(codes: np.ndarray) -> np.ndarray:
    """Pack unsigned codes [N, K] (0..15, code = q + 8) into the SM80 W4_A16 order; returns uint8 [N*K/2]."""
    n, k = codes.shape
    assert k % ROWS_PER_TILE == 0 and n % COLS_INTERLEAVED == 0
    stored = codes.reshape(n, k // 32, 32)[:, :, K_PERM].reshape(n, k)
    nib = stored.reshape(n, k // 8, 8).astype(np.uint32)
    words = np.zeros((n, k // 8), dtype=np.uint32)
    for i in range(8):
        words |= nib[:, :, i] << np.uint32(4 * nibble_dest(i))
    w = words.reshape(n // COLS_INTERLEAVED, COLS_INTERLEAVED, k // ROWS_PER_TILE, 8)
    w = np.ascontiguousarray(w.transpose(0, 2, 1, 3))
    return w.astype("<u4").view(np.uint8).reshape(-1)


def block_layout(hidden: int, inter: int) -> dict:
    n1 = 2 * inter
    sizes = [
        ("fc1_weights", n1 * hidden // 2),
        ("fc1_scales", hidden // BLOCK * n1 * 2),
        ("fc1_bias", n1 * 2),
        ("fc2_weights", hidden * inter // 2),
        ("fc2_scales", inter // BLOCK * hidden * 2),
        ("fc2_bias", hidden * 2),
    ]
    out, off = {}, 0
    for name, size in sizes:
        out[name] = off
        off += _align(size)
    out["stride"] = off
    return out


def assemble_block(hidden, inter, fc1_codes, fc1_scales, fc1_bias, fc2_codes, fc2_scales, fc2_bias) -> np.ndarray:
    """codes: uint8 [N,K]; scales: fp16 [N,K/32]; bias: fp16 [N]. Returns uint8 [stride]."""
    lay = block_layout(hidden, inter)
    buf = np.zeros(lay["stride"], dtype=np.uint8)

    def put(name, arr):
        raw = np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
        buf[lay[name] : lay[name] + raw.size] = raw

    put("fc1_weights", pack_sm80_w4(fc1_codes))
    put("fc1_scales", np.ascontiguousarray(fc1_scales.astype(np.float16).T))
    put("fc1_bias", fc1_bias.astype(np.float16))
    put("fc2_weights", pack_sm80_w4(fc2_codes))
    put("fc2_scales", np.ascontiguousarray(fc2_scales.astype(np.float16).T))
    put("fc2_bias", fc2_bias.astype(np.float16))
    return buf


def dequant(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    s = np.repeat(scales.astype(np.float64), BLOCK, axis=1)
    return (codes.astype(np.float64) - 8.0) * s


def reference_expert(
    x, fc1_codes, fc1_scales, fc1_bias, fc2_codes, fc2_scales, fc2_bias, alpha=1.0, beta=0.0, limit=np.inf
):
    x = x.astype(np.float64)
    h1 = x @ dequant(fc1_codes, fc1_scales).T + fc1_bias.astype(np.float64)
    gate, value = h1[:, 0::2], h1[:, 1::2]
    gate = np.minimum(gate, limit)
    value = np.clip(value, -limit, limit)
    act = gate / (1.0 + np.exp(-alpha * gate)) * (value + beta)
    return act @ dequant(fc2_codes, fc2_scales).T + fc2_bias.astype(np.float64)


def assemble_blocks_from_prepacked(hidden, inter, w1, s1, b1, w2, s2, b2) -> np.ndarray:
    """Legacy artifact -> shared blocks without touching quantized values.

    w1/w2: uint8 [E, N, K/2] already in the CUDA SM80 W4 order (reused byte-for-byte);
    s1/s2: fp16 [E, N, K/32] output-major (written block-major [K/32, N]); b1/b2: fp16 [E, N].
    Returns uint8 [E, stride].
    """
    lay = block_layout(hidden, inter)
    num_experts = w1.shape[0]
    out = np.zeros((num_experts, lay["stride"]), dtype=np.uint8)

    def put(name, arr):
        raw = np.ascontiguousarray(arr).view(np.uint8).reshape(num_experts, -1)
        out[:, lay[name] : lay[name] + raw.shape[1]] = raw

    put("fc1_weights", w1)
    put("fc1_scales", np.ascontiguousarray(s1.astype(np.float16).transpose(0, 2, 1)))
    put("fc1_bias", b1.astype(np.float16))
    put("fc2_weights", w2)
    put("fc2_scales", np.ascontiguousarray(s2.astype(np.float16).transpose(0, 2, 1)))
    put("fc2_bias", b2.astype(np.float16))
    return out

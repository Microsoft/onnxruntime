# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Profiling script for the CUDA MatMulNBits (4-bit / 8-bit weight-only) op.

Times the op across representative decoder weight matrices and row counts M
(M=1 decode and M>1 small-batch). Mirrors profile_qmoe_gemv.py:
warmup + measured runs wrapped in NVTX ranges so the results can be parsed
with parse_nsys.py for kernel-level timing, while also printing host-observed
average latency for a quick end-to-end view.

Usage:
  # Host-timing table across all cases:
  python profile_matmul_nbits.py --warmup 25 --repeat 200

  # Single case:
  python profile_matmul_nbits.py --k 4096 --n 4096 --m 8 --block-size 32 --bits 4 --dtype fp16

  # Kernel-level via nsys + the repo parser:
  nsys profile -t cuda,nvtx -o mnb --export=sqlite \
      python profile_matmul_nbits.py --k 4096 --n 4096 --m 8
  python parse_nsys.py mnb.sqlite --nvtx-range benchmark --pattern '%'
"""

import argparse
import json
import time
from contextlib import nullcontext

import numpy as np
import torch
from onnx import TensorProto, helper

import onnxruntime

try:
    import ml_dtypes

    bfloat16 = ml_dtypes.bfloat16
except ImportError:
    bfloat16 = None

try:
    import nvtx

    has_nvtx = True
except ImportError:
    has_nvtx = False
    nvtx = None

RESULT_PREFIX = "MATMUL_NBITS_RESULT "

_OT = {"fp16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16}
_TT = {"fp16": torch.float16, "bf16": torch.bfloat16}
_ELEM = {torch.float16: TensorProto.FLOAT16, torch.bfloat16: TensorProto.BFLOAT16}

# Representative decoder weight matrices (K = in features, N = out features),
# sized after a Qwen3-8B-class dense decoder + lm_head.
DEFAULT_CASES = [
    ("qkv", 4096, 4096),
    ("o_proj", 4096, 4096),
    ("gate_up", 4096, 12288),
    ("down", 12288, 4096),
    ("lm_head", 4096, 151936),
]
DEFAULT_MS = [1, 2, 4, 8, 16]


def _nvtx_range(name, color="green"):
    if not has_nvtx:
        return nullcontext()
    return nvtx.annotate(name, color=color)


def build_model(k, n, block_size, bits, onnx_dtype, with_zero_point=True):
    rng = np.random.default_rng(0)
    n_blocks = (k + block_size - 1) // block_size
    blob = block_size // (8 // bits)
    b = rng.integers(0, 256, size=(n, n_blocks, blob), dtype=np.uint8)
    scales_f32 = rng.random(n * n_blocks).astype(np.float32) * 0.02 + 0.01
    if onnx_dtype == TensorProto.FLOAT16:
        scales = scales_f32.astype(np.float16)
    elif onnx_dtype == TensorProto.BFLOAT16:
        if bfloat16 is None:
            raise RuntimeError("ml_dtypes is required for bf16 (pip install ml_dtypes)")
        scales = scales_f32.astype(bfloat16)
    else:
        scales = scales_f32
    inits = [
        helper.make_tensor("B", TensorProto.UINT8, list(b.shape), b.tobytes(), raw=True),
        helper.make_tensor("scales", onnx_dtype, list(scales.shape), scales.tobytes(), raw=True),
    ]
    inputs = ["A", "B", "scales"]
    if with_zero_point:
        # 4-bit zero points are packed two per byte; 8-bit zero points are one byte per block.
        zp_count = n * ((n_blocks + 1) // 2) if bits == 4 else n * n_blocks
        zp = rng.integers(0, 256, size=(zp_count,), dtype=np.uint8)
        inits.append(helper.make_tensor("zero_points", TensorProto.UINT8, list(zp.shape), zp.tobytes(), raw=True))
        inputs.append("zero_points")
    node = helper.make_node(
        "MatMulNBits",
        inputs,
        ["Y"],
        domain="com.microsoft",
        K=k,
        N=n,
        bits=bits,
        block_size=block_size,
        accuracy_level=0,
    )
    graph = helper.make_graph(
        [node],
        "mnb",
        [helper.make_tensor_value_info("A", onnx_dtype, ["M", k])],
        [helper.make_tensor_value_info("Y", onnx_dtype, ["M", n])],
        inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)],
        ir_version=8,  # Match opset 17 instead of inheriting the installed ONNX package's newest IR.
    )
    return model.SerializeToString()


def make_session(model):
    so = onnxruntime.SessionOptions()
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.log_severity_level = 3
    return onnxruntime.InferenceSession(model, so, providers=["CUDAExecutionProvider"])


def run_case(name, m, k, n, block_size, bits, dtype, warmup, repeat):
    onnx_dtype = _OT[dtype]
    torch_dtype = _TT[dtype]
    sess = make_session(build_model(k, n, block_size, bits, onnx_dtype))
    a = np.random.default_rng(m).random((m, k)).astype(np.float32) * 0.02 - 0.01
    at = torch.from_numpy(a).to(torch_dtype).cuda().contiguous()
    y = torch.empty((m, n), dtype=torch_dtype, device="cuda")
    io = sess.io_binding()
    io.bind_input("A", "cuda", 0, _ELEM[torch_dtype], list(at.shape), at.data_ptr())
    io.bind_output("Y", "cuda", 0, _ELEM[torch_dtype], list(y.shape), y.data_ptr())

    with _nvtx_range("warmup", "yellow"):
        for _ in range(warmup):
            sess.run_with_iobinding(io)
    torch.cuda.synchronize()

    best = float("inf")
    trials = 10
    with _nvtx_range("benchmark", "green"):
        for _ in range(trials):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(repeat):
                sess.run_with_iobinding(io)
            torch.cuda.synchronize()
            best = min(best, (time.perf_counter() - t0) / repeat)
    us = best * 1e6
    result = {
        "case": name,
        "m": m,
        "k": k,
        "n": n,
        "block_size": block_size,
        "bits": bits,
        "dtype": dtype,
        "avg_us": round(us, 2),
    }
    print(RESULT_PREFIX + json.dumps(result))
    return result


def main():
    p = argparse.ArgumentParser(description="Profile CUDA MatMulNBits")
    p.add_argument("--k", type=int, help="in features (single-case mode)")
    p.add_argument("--n", type=int, help="out features (single-case mode)")
    p.add_argument("--m", type=int, help="rows (single-case mode)")
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--ms", type=int, nargs="+", default=DEFAULT_MS, help="row counts to sweep in table mode")
    args = p.parse_args()

    if args.k and args.n and args.m:
        run_case("custom", args.m, args.k, args.n, args.block_size, args.bits, args.dtype, args.warmup, args.repeat)
        return

    rows = {}
    for name, k, n in DEFAULT_CASES:
        rows[name] = {
            m: run_case(name, m, k, n, args.block_size, args.bits, args.dtype, args.warmup, args.repeat)["avg_us"]
            for m in args.ms
        }

    print(f"\n  MatMulNBits {args.bits}-bit block{args.block_size} {args.dtype}  (avg us, lower is better)")
    print("  " + f"{'matrix':10s} {'K':>6} {'N':>7} " + " ".join(f"M={m:<5}" for m in args.ms))
    for name, k, n in DEFAULT_CASES:
        print("  " + f"{name:10s} {k:>6} {n:>7} " + " ".join(f"{rows[name][m]:7.1f}" for m in args.ms))


if __name__ == "__main__":
    main()

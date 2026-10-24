# PackedSparseAttentionIndexer — Operator Documentation

## Experimental CUDA decode specializations

Set these process-level options before the first operator invocation. All three
default to `0`; only `0` and `1` are accepted. Existing dispatch remains available
when an option is disabled or its geometry is unsupported.

- `ORT_PACKED_SPARSE_INDEXER_COMPACT_TOPK=1`: compact QSA chunks of 2048 keys
  to 512 keys with device-side 512/1024/2048 sorting probes. Up to five chunks
  use a fused final merge/emit; larger selections use the existing pairwise
  merge. Stable tie ordering, causal tails, capacity stride/padding, and
  overflow handling are preserved. This option is mutually exclusive with
  `ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK=1`.
- `ORT_DSA_INT8_VECTOR_STAGING=1`: the companion `DynamicSparseAttention`
  grouped INT8 path uses eight-byte cache loads and independent per-channel
  key/value scales. Its existing guards still require head size 256, FP16/BF16
  queries, one or two tokens, GQA ratio at least four, more than 256 candidates,
  and sufficient resources. Auxiliary floating-point caches retain their own
  unscaled load path.
- `ORT_DSA_SHARED_SCALE_MERGE=1`: grouped attention merges with at most 32
  splits compute the exponential factor once per split and share it across
  output channels. A nonzero partial sum, not the maximum-score sentinel,
  determines whether a split is active. Active factors that underflow to zero
  remain in the accumulation.

These options change kernels, not graph schemas or cache layouts. Qualify
accuracy and performance on the target runtime, hardware, and workload before
enabling them in a deployment.

This document describes the `com.microsoft::PackedSparseAttentionIndexer` contrib operator: the
packed/variable-length counterpart of `com.microsoft::SparseAttentionIndexer`, built for
continuous-batching (paged) inference engines such as an OgaEngine-style `PagedAttention` model.

Source:
[bert_defs.cc](../../onnxruntime/core/graph/contrib_ops/bert_defs.cc) (schema),
[sparse_attention_indexer_common.h](../../onnxruntime/contrib_ops/cpu/sparse/sparse_attention_indexer_common.h)
(policy enum, selected-capacity formula and CSA window-plan arithmetic, shared unmodified with
`SparseAttentionIndexer`),
[packed_sparse_attention_indexer_common.h](../../onnxruntime/contrib_ops/cpu/sparse/packed_sparse_attention_indexer_common.h)
(fixed 18-input / 7-output slot map),
[packed_sparse_attention_indexer.cc](../../onnxruntime/contrib_ops/cuda/sparse/packed_sparse_attention_indexer.cc) /
[packed_sparse_attention_indexer_impl.cu](../../onnxruntime/contrib_ops/cuda/sparse/packed_sparse_attention_indexer_impl.cu)
(CUDA),
[packed_sparse_attention_indexer.cc](../../onnxruntime/contrib_ops/webgpu/bert/packed_sparse_attention_indexer.cc)
(WebGPU, see also [the WebGPU note](webgpu/packed_sparse_attention_indexer.md)).

---

## 1. Why a separate operator

`SparseAttentionIndexer` uses dense `[batch_size, sequence_length, ...]` tensors, an explicit dense
QSA visibility mask, and state that grows by concatenation every call
(`present_key = concat(past_key, key)`, etc.). A continuous-batching / paged engine instead:

- flattens every request's tokens into one `[total_tokens, ...]` axis (packed layout);
- schedules a different number of new tokens per request per step;
- keeps every request's KV/indexer state in a **fixed-address, fixed-capacity** slot of a state
  pool (so the engine can reuse buffers and support CUDA graph capture), never a tensor that grows;
- derives causal visibility purely from packed offsets, never from a materialized `[B, S, T]` mask.

Restructuring `SparseAttentionIndexer` in place to support both contracts was judged more invasive
and risky than the value of the shared line count (roughly 25-35% of the dense implementation is
directly reusable without change; most of the rest needs new shapes, new state semantics, or a
different launch/index mapping). `PackedSparseAttentionIndexer` is therefore a new op, version 1,
that **does not change `SparseAttentionIndexer`'s schema or behavior**. See [§8](#8-what-is-shared-vs-packed-specific).

## 2. Operator schema

Attributes:

| Attribute | Constraint | Meaning |
|---|---|---|
| `policy_mode` | required, `"qsa"` or `"csa"` | selects the indexer flavour |
| `compress_ratio` | required, `> 0` | tokens folded into one block/compressed entry |
| `state_capacity` | required, `> 0` | fixed capacity (entries) of `past_key_state` |
| `token_budget` | `qsa` only, `> 0`, divisible by `compress_ratio` | selected-token budget |
| `index_topk` | `csa` only, `> 0` | selected compressed-entry count |
| `epsilon` | default `1e-6` | RMSNorm epsilon |
| `scale` | default `1/sqrt(head_size)` | per-head score scale |
| `head_weight_scale` | `csa` only, default `1/sqrt(num_heads)` | head-weight score scale |
| `state_update_capacity` | `qsa` only, default `0`, range `[0, 8]` | maximum captured transitions per request |

Inputs are **fixed at 18 indices** for both policies (unlike the dense op, which uses a different
input/output count per policy). A slot not owned by the active policy is a *positional* optional:
its `NodeProto` input name is empty rather than the slot being removed from the list, so every
later slot keeps its fixed index.

| # | Name | Shape | Type | Policy |
|---|---|---|---|---|
| 0 | `query` | `(total_tokens, num_heads*head_size)`; when qsa omits `key`, `(total_tokens, (num_heads+1)*head_size)` containing row-wise `[query \| key]` | T | both |
| 1 | `key` | `(total_tokens, head_size)` qsa / `(total_tokens, 2*head_size)` csa; optional for qsa when packed into `query` | T | both |
| 2 | `query_norm_weight` | `(head_size)` | T | both |
| 3 | `key_norm_weight` | `(head_size)` | T | both |
| 4 | `cos_cache` | `(max_position, rotary_width)` or `(batch_size, max_position, rotary_width)` | T | both |
| 5 | `sin_cache` | same shape as `cos_cache` | T | both |
| 6 | `cumulative_sequence_lengths` | `(batch_size + 1)` | int32, device-resident | both |
| 7 | `past_sequence_lengths` | `(batch_size)` | int32, device-resident | both |
| 8 | `gate` | `(total_tokens, 2*head_size)` | T | csa only |
| 9 | `position_bias` | `(compress_ratio, 2*head_size)` | T | csa only |
| 10 | `head_weights` | `(total_tokens, num_heads)` | T | csa only |
| 11 | `position_ids` | `(total_tokens)` | int64 | optional qsa / required csa |
| 12 | `past_key_state` | `(batch_size, state_capacity, head_size)` | T_KEY | both (generic) |
| 13 | `past_kv_buffer` | `(batch_size, 2*compress_ratio-1, width)` | T | both (generic) |
| 14 | `past_gate_buffer` | same shape as `past_kv_buffer` | T | csa only |
| 15 | `past_state_lengths` | `(batch_size, 2)` | int32, device-resident | both (generic) |
| 16 | `state_update_capture_count` | `(batch_size)` | int32, device-resident | qsa capture only |
| 17 | `state_update_active` | `(1)` | int32, device-resident | optional qsa capture gate |

Outputs have six fixed indices plus optional slot 6. `present_gate_buffer` is declared with an
empty output name for `qsa`; `state_update` is declared only when QSA capture is enabled:

| # | Name | Shape | Type | Policy |
|---|---|---|---|---|
| 0 | `selected_indices` | `(total_tokens, capacity)` | int32 | both |
| 1 | `selected_counts` | `(total_tokens)` | int32 | both |
| 2 | `present_key_state` | same shape as `past_key_state` | T_KEY | both |
| 3 | `present_kv_buffer` | same shape as `past_kv_buffer` | T | both |
| 4 | `present_gate_buffer` | same shape as `past_gate_buffer` | T | csa only |
| 5 | `present_state_lengths` | same shape as `past_state_lengths` | int32 | both |
| 6 | `state_update` | `(batch_size, state_update_capacity, head_size)` | T | qsa capture only |
| 7 | `index_share_status` | `(total_tokens)` | int32 | index sharing |
| 8 | `state_status` | `(batch_size)` | int32 | optional, CUDA only |
| 9 | `mid_kv_buffer` | same shape as `past_kv_buffer` | T | optional CUDA qsa snapshot |
| 10 | `mid_state_lengths` | same shape as `past_state_lengths` | int32 | optional CUDA qsa snapshot |
| 11 | `validation_status` | `(batch_size)` | int32 | optional CUDA qsa status |

CUDA supports `T_KEY = T` and, for QSA only, BF16 activations with FP32 key state.
The pending buffers and captured state updates remain in the activation type.
`mid_kv_buffer` and `mid_state_lengths` must be connected together and capture the state
after each request's first token. `validation_status` is zero on success; rejected
requests report 9, or `0x99` for requests containing at least two tokens.

Output 8 retains main's `state_status` contract. Graphs built against the earlier
experimental snapshot layout must insert an empty output at slot 8 before the
snapshot outputs, which now occupy slots 9-11. Existing graphs using only the
mainline outputs do not require changes.

`capacity` is `token_budget + compress_ratio - 1` for `qsa` and `index_topk` for `csa`, exactly the
same formula (`SelectedCapacity`) used by `SparseAttentionIndexer`.

**No output shape depends on tensor data.** `total_tokens` and `batch_size` come from input
*shapes* (`query.shape[0]`, `cumulative_sequence_lengths.shape[0] - 1`); every state output has
exactly the same shape as its corresponding state input. This is what makes the fixed-capacity
state design load-bearing: a growing/concatenated state (as in the dense op) would require a
data-dependent output shape, which is incompatible with CUDA graph capture and with pre-allocated
paged state pools.

## 3. Generic state, shared by both policies

Both policies read and write the *same four* state slots — there is no separate
`past_compressed_key` vs. `past_key` naming split as in the dense op:

- `past_key_state` / `present_key_state`: `qsa` stores prepared (already mean-pooled, RMSNorm'd and
  rotated) complete-block keys; `csa` stores compressed keys. Layout-compatible with, or cheaply
  reshaped to, a `[batch_size, capacity, 1, head_size]` auxiliary paged cache when K = V.
- `past_kv_buffer` / `present_kv_buffer` (and `past_gate_buffer` / `present_gate_buffer`, `csa`
  only): the generic pending-token buffer, fixed capacity `2 * compress_ratio - 1`. `qsa` only ever
  uses up to `compress_ratio - 1` of these entries (a raw, not-yet-pooled block); `csa` uses the
  full range for the overlap ("Ca") plus leftover ("Cb") halves of the window-plan arithmetic
  reused from `SparseAttentionIndexer`.
- `past_state_lengths` / `present_state_lengths`: `(batch_size, 2)`. Column 0 is the `key_state`
  entry count (`qsa`: complete-block count; `csa`: compressed-entry count); column 1 is the pending
  buffer length (`qsa`: incomplete-block length in `[0, compress_ratio)`; `csa`: buffer length in
  `[0, 2 * compress_ratio)`, exactly the invariant already documented for
  `CsaWindowPlan`/`TryComputeCsaWindowPlan`).

State never grows. `present_*` always has exactly the same shape as `past_*`; only the *contents*
change. CUDA supports aliasing `past_key_state` with `present_key_state` and `past_state_lengths`
with `present_state_lengths`; it skips the baseline key-state copy when those buffers alias.
WebGPU supports aliasing every corresponding `past_*` / `present_*` pair and reads prior contents
through the matching read-write `present_*` binding. Each request is handled by one invocation,
and buffer compaction reads entries at or above the destination index before overwriting them.

**State overflow.** If a call would close more blocks/windows than
`state_capacity - old_entry_count` allows, that request's step is rejected as a deterministic
no-op: its state and state lengths remain unchanged, and its selection outputs stay empty. This
never reads or writes outside a tensor's fixed extent and never silently truncates state.

**State status.** The rejection above is silent unless the optional output 8, `state_status`, is
requested (outputs 6 and 7 are then declared, possibly with empty names). It holds one int32 per
request, written on the device by the state update kernel, so a CUDA-graph-captured caller can
check it on the host after replay without any synchronization inside the operator: `0` the update
was applied (or intentionally held by IndexShare reuse), `1` invalid device metadata (inconsistent
`cumulative_sequence_lengths`, `past_sequence_lengths` or `past_state_lengths`), `2` the update
would exceed `state_capacity`. The values follow the status convention of
`PackedSparseAttentionIndexerMerge`. A non-zero request leaves its state payloads unchanged, clamps
out-of-range `past_state_lengths` into `present_state_lengths`, and emits an empty selection. The
WebGPU kernel does not implement the output and fails if it is requested.

**Transactional QSA capture.** When `state_update_capacity > 0`,
`state_update_capture_count[b]` requests capture of that request's leading transitions, clamped to
the request length and the configured capacity. `state_update_active == 0` disables capture for the
call. Disabled, unused, and out-of-range slots are zero. A captured token that completes a
compression block stores the prepared block representative; other tokens store their raw key.
After restoring a pre-call state snapshot, a runtime can replay the accepted prefix in order:
append raw entries to the pending buffer, and replace each block-completion transition with its
prepared representative in `key_state`.

## 4. Packed metadata and device-side safety

`cumulative_sequence_lengths` and `past_sequence_lengths` are **device-resident** tensors, read
directly by the kernels — never copied to the host or synchronized on. The device-visible
invariants (validated by well-behaved callers; a malformed value never causes memory corruption,
see below) are:

- `cumulative_sequence_lengths[0] == 0`;
- `cumulative_sequence_lengths[batch_size] == total_tokens`;
- `cumulative_sequence_lengths` is nondecreasing (a repeated offset — a zero-token request row —
  is valid and simply contributes no query rows for that request);
- `past_sequence_lengths[b] >= 0` and, for `qsa`, consistent with `past_state_lengths[b]`
  (`key_state_length == past_sequence_length / compress_ratio`,
  `buffer_length == past_sequence_length % compress_ratio`);
- `past_state_lengths[b, 0] <= state_capacity` and `past_state_lengths[b, 1]` within its policy's
  valid buffer range.

Because there is no host synchronization, the kernels cannot literally raise a C++ exception when
one of these invariants is violated by the input data (as opposed to a mismatched tensor *shape*,
which the host-side `OpKernel::Compute` and the ONNX schema still check the ordinary way). Instead,
every per-request quantity read from these tensors is **clamped into its valid range before use**
(`old_key_len = clamp(past_state_lengths[b,0], 0, state_capacity)`, etc.), and the request-token
lookup (`PackedBatchOfToken`, a binary search over `cumulative_sequence_lengths`) always returns an
index in `[0, batch_size)`. The result is that malformed metadata can make the numeric result wrong
for the affected request, but it can never read or write outside a tensor's allocated extent and
never causes overlapping writes between requests. This mirrors the "prefer deterministic safe
outputs" guidance for EPs that cannot report device-side validation errors asynchronously.

## 5. Policy `qsa`

Each request's packed token range is processed independently, in the same three stages as the
dense `qsa` policy but against fixed-capacity state instead of a growing cache:

1. **Update** (one launch per request): append each raw indexer key to the generic pending buffer;
   whenever it reaches `compress_ratio` tokens, mean-pool it, apply RMSNorm and `key_norm_weight`,
   apply the leading/split-half rotary convention at the block's first logical token position
   (`entry * compress_ratio`, where `entry` is the block's absolute index in `key_state`), and
   append the **prepared** (already normalized and rotated) key to `key_state`. This differs from
   the dense kernel, which stores *raw* concatenated keys and repeats the pooling/RMSNorm/rotate
   work for every query; storing the prepared key once, at update time, is possible only because
   packed `key_state` never needs to be re-windowed the way a dense-mask query can.
2. **Score** (one launch per query token, per candidate block): every causally visible block
   (`block index < min(key_state_length, causal_threshold(position))`) is scored directly against
   `key_state` with `sum_h ReLU(q_h . k)` — a single dot product, no recomputation.
3. **Select** (one launch per query token): keeps the `token_budget / compress_ratio` highest
   scoring blocks (ties broken by ascending index) and appends the request-local logical token
   positions `[j * compress_ratio, ..., j * compress_ratio + compress_ratio - 1]` for each selected
   block `j`, followed by every causally visible position of the current incomplete block, and
   writes the exact active count to `selected_counts`.

"Request-local logical token position" means the same numbering as
`past_sequence_lengths[b] + local_offset` — i.e. the request's own absolute token position, which
is exactly what a per-request main paged KV cache is addressed by. The output is therefore directly
consumable by `SparsePagedAttention` configured with `attention_mode="selected_only"`,
`selected_kv_source="main"`. Optional QSA `position_ids` override query rotary positions and causal
visibility. Callers must keep these positions consistent with prepared block keys and the cache
indices consumed by attention.

## 6. Policy `csa`

Reuses `SparseAttentionIndexer`'s CSA compression, rotary, scoring, causal threshold, and
deterministic TopK semantics verbatim (the *math* is unchanged); only the layout, state, and launch
mapping are packed:

1. **Update** (one launch per request): computes the window plan
   (`overlap_length`, `leftover_length`, `new_window_count`, `present_buffer_length`,
   `present_buffer_start`) for that request's own `buffer_length` and packed token count using
   `TryComputeCsaWindowPlan` — the exact same function used by `SparseAttentionIndexer`'s schema
   and kernel, called directly on the device (it is a small `SAI_HOST_DEVICE` inline function with
   no CUDA-specific code). Every closed window is compressed with the softmax-gated Ca/Cb pooling,
   normalized, rotated with the trailing convention, and appended to `key_state`; the request
   rejects (caps) new windows beyond `state_capacity` as described in [§3](#3-generic-state-shared-by-both-policies).
2. **Score** (one launch per query token, per compressed entry): every entry is scored with
   `sum_h w_h * ReLU(q_h . k)` and masked by the causal threshold from `position_ids` (required for
   `csa`; `qsa` optionally accepts an explicit position override).
3. **Select** (one launch per query token): keeps the `index_topk` highest scoring, causally
   visible entries and writes the exact active count to `selected_counts`.

`selected_indices` values are compressed-entry indices into `key_state`, consumable by
`SparsePagedAttention` configured with `attention_mode="local_plus_selected"`,
`selected_kv_source="auxiliary"`; `key_state` is layout-compatible with, or cheaply reshaped to, the
auxiliary cache contract (`[batch_size, capacity, 1, head_size]` when K = V).

## 7. Provider support

CUDA and WebGPU both implement version 1 of this operator, for `float32`, `float16` (CUDA also
`bfloat16`). There is intentionally no CPU kernel (only the shared constants/helpers/schema are
CPU-agnostic); a production model that uses this op targets a paged-KV engine on an accelerator.
See [the WebGPU note](webgpu/packed_sparse_attention_indexer.md) for WebGPU-specific details.

## 8. What is shared vs. packed-specific

Shared with `SparseAttentionIndexer`, unmodified:

- `Policy` enum, `TryParsePolicy`, `SelectedCapacity` (selected-capacity formula) — from
  `sparse_attention_indexer_common.h`;
- `CsaWindowPlan` / `TryComputeCsaWindowPlan` (CSA window-plan arithmetic) — same header, now
  additionally annotated `SAI_HOST_DEVICE` so CUDA device code can call it directly;
- the CUDA device math (`sparse_attention_indexer_device_math.cuh`, newly extracted from
  `sparse_attention_indexer_impl.cu` with no behavior change): FP32 block reductions
  (`SaiBlockSum`), deterministic argmax/selection (`SaiBlockArgMax`, `SaiScanForNext`), the leading
  and trailing RoPE conventions (`SaiLeadingRope`, `SaiTrailingRope`), and the causal-threshold
  formula (`SaiCausalThreshold`, which turns out to be exactly the "number of complete blocks fully
  visible to a query" formula needed by both policies here, unifying what the dense implementation
  computed two different ways).

Deliberately **not** shared (packed-specific mechanics with no dense equivalent, or dense-only
mechanics with no packed equivalent):

- packed metadata validation and the device-side per-token/per-request lookup
  (`PackedBatchOfToken`);
- fully in-place fixed-capacity state update (no growing/concatenating state, no host-visible
  data-dependent output shape);
- plain causal visibility derived from packed offsets (no dense `[B, 1, S, T]` mask input, no
  mask-compaction step);
- the dense op's batch-major `[B, S, ...]` launch/index mapping and host wrappers, which do not
  apply to a token-major `[total_tokens, ...]` tensor.

## 9. Testing

`onnxruntime/test/contrib_ops/packed_sparse_attention_indexer_op_test.cc` covers:

- shape inference for `qsa` and `csa` (fixed `selected_indices`/`selected_counts` shapes, fixed
  state output shapes, strict per-slot policy validation, and optional capture output);
- multi-request packed batches with unequal token counts, a zero-token request row, and prefill
  followed by decode with independent per-request state (CUDA/WebGPU, skipped without the
  respective execution provider);
- `qsa` state-capacity overflow safety;
- active and disabled transactional QSA capture on CUDA and WebGPU;
- FP32/FP16 (and CUDA-only BF16) numeric coverage against an in-file reference that mirrors this
  document's contract.

## 10. Known limitations and follow-ups

- The reference CUDA/WebGPU kernels prioritize correctness over throughput (see the top-of-file
  comments in the `.cu`/`.cc` implementations). CUDA QSA fuses query rotation into scoring, uses
  the shared-memory TopK path for up to 32 selected blocks, bounded partial TopK for 33–512 blocks,
  and a hierarchical TopK path for common long-context Qwen shapes. Values above 512 that do not
  match the hierarchical specialization retain the correctness-first repeated-scan fallback.
- CUDA has two experimental, default-off dispatch controls. Set them before the first eligible
  call; each value is cached per process. `ORT_PACKED_SPARSE_INDEXER_RADIX_TOPK=1` uses live-key
  radix selection for eligible short QSA calls. `ORT_PACKED_SPARSE_INDEXER_WARP_HEADS=1` uses
  warp-local scoring for eligible packed prefill calls with more than 64 tokens, four heads,
  head size 128 and compression ratio four. Both retain causal visibility, stable tie ordering,
  selected counts and state-update semantics. Broad full-model quality and performance
  qualification remains outstanding; neither option is a general tuning recommendation.
- OgaEngine / Model Builder integration (declaring `past_key_state` etc. as Engine-managed,
  per-request fixed-size state, analogous to a paged auxiliary cache) is out of scope for this
  operator definition and is expected in a follow-up to `microsoft/onnxruntime-genai`.
- No CPU kernel is provided.

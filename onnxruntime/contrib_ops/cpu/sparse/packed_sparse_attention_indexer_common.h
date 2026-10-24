// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Shared constants for com.microsoft.PackedSparseAttentionIndexer. This op reuses the policy
// enum, selected-capacity formula and CSA window-plan arithmetic already defined for
// com.microsoft.SparseAttentionIndexer in sparse_attention_indexer_common.h; it does not modify
// that header's input/output slot layout, which stays specific to the dense operator.
//
// PackedSparseAttentionIndexer instead uses packed [total_tokens, ...] query/key tensors,
// device-resident cumulative_sequence_lengths / past_sequence_lengths, and generic fixed-capacity
// state slots that are shared by both policy_mode values (unlike the dense op's policy-specific
// state names).

#pragma once

#include <cstddef>
#include <cstdint>

#include "contrib_ops/cpu/sparse/sparse_attention_indexer_common.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace packed_sparse_attention_indexer {

// Re-exported so callers only need to include this header.
using sparse_attention_indexer::CsaWindowPlan;
using sparse_attention_indexer::kPolicyModeCsa;
using sparse_attention_indexer::kPolicyModeQsa;
using sparse_attention_indexer::Policy;
using sparse_attention_indexer::SelectedCapacity;
using sparse_attention_indexer::TryComputeCsaWindowPlan;
using sparse_attention_indexer::TryParsePolicy;

// Fixed input slots. Slots 8-10 belong to policy_mode="csa" only; slot 11 (position_ids) is
// forbidden for "qsa" and required for "csa". Every other slot is required for both policies.
enum InputIndex : int {
  kQuery = 0,                      // [total_tokens, num_heads * head_size]
  kKey = 1,                        // qsa: [total_tokens, head_size]; csa: [total_tokens, 2 * head_size]
  kQueryNormWeight = 2,            // [head_size]
  kKeyNormWeight = 3,              // [head_size]
  kCosCache = 4,                   // [max_position, rotary_width] or [batch_size, max_position, rotary_width]
  kSinCache = 5,                   // same shape as cos_cache
  kCumulativeSequenceLengths = 6,  // [batch_size + 1], int32
  kPastSequenceLengths = 7,        // [batch_size], int32
  kGate = 8,                       // csa only: [total_tokens, 2 * head_size]
  kPositionBias = 9,               // csa only: [compress_ratio, 2 * head_size]
  kHeadWeights = 10,               // csa only: [total_tokens, num_heads]
  kPositionIds = 11,               // forbidden (qsa) / required (csa): [total_tokens], int64
  kPastKeyState = 12,              // generic: [batch_size, state_capacity, head_size]
  kPastKvBuffer = 13,              // generic: [batch_size, 2 * compress_ratio - 1, width]
  kPastGateBuffer = 14,            // csa only: same shape as past_kv_buffer
  kPastStateLengths = 15,          // generic: [batch_size, 2], int32
  kStateUpdateCaptureCount = 16,   // qsa only, optional: [batch_size], int32
  kStateUpdateActive = 17,         // qsa only, optional: [1], int32
  kIndexShareMode = 18,
  kMergedIndices = 19,
  kMergedCounts = 20,
  kMergedStatus = 21,
  kInputCount = 22,
};

// Fixed output slots. present_gate_buffer is declared (with an empty name) but not produced for
// policy_mode="qsa".
enum OutputIndex : int {
  kSelectedIndices = 0,      // [total_tokens, selected_capacity], int32, unused entries -1
  kSelectedCounts = 1,       // [total_tokens], int32
  kPresentKeyState = 2,      // same shape as past_key_state
  kPresentKvBuffer = 3,      // same shape as past_kv_buffer
  kPresentGateBuffer = 4,    // csa only: same shape as past_gate_buffer
  kPresentStateLengths = 5,  // [batch_size, 2], int32
  kStateUpdate = 6,          // qsa only, optional: [batch_size, state_update_capacity, head_size]
  kIndexShareStatus = 7,
  kStateStatus = 8,        // optional: [batch_size], int32, see StateStatusCode
  kMidKvBuffer = 9,        // qsa only, optional: snapshot after the first request token
  kMidStateLengths = 10,   // qsa only, optional: lengths after the first request token
  kValidationStatus = 11,  // qsa only, optional: per-request validation status
  kMidKvBuffer2 = 12,
  kMidStateLengths2 = 13,
  kMidKvBuffer3 = 14,
  kMidStateLengths3 = 15,
  kOutputCount = 9,
  kMaxOutputCount = 16,
};

// Values of the optional state_status output. They follow the status convention of
// PackedSparseAttentionIndexerMerge (1: invalid device metadata, 2: capacity overflow).
enum StateStatusCode : int {
  kStateStatusOk = 0,
  kStateStatusInvalidMetadata = 1,
  kStateStatusCapacityOverflow = 2,
};

// Every PackedSparseAttentionIndexer node declares all 6 fixed outputs; present_gate_buffer is an
// empty-name optional output for policy_mode="qsa". State update capture is an optional appended
// seventh output so existing nodes retain their output layout.
constexpr int kFixedOutputCount = kStateUpdate;

// Column layout of past_state_lengths / present_state_lengths.
enum StateLengthColumn : int {
  kKeyStateLength = 0,  // qsa: complete-block count; csa: compressed-entry count
  kBufferLength = 1,    // qsa: incomplete-block length in [0, compress_ratio);
                        // csa: pending buffer length in [0, 2 * compress_ratio)
  kStateLengthColumns = 2,
};

// Generic pending-buffer capacity: qsa only ever uses up to compress_ratio - 1 of these slots.
SAI_HOST_DEVICE inline int64_t GenericBufferCapacity(int64_t compress_ratio) {
  return 2 * compress_ratio - 1;
}

inline Status ValidateDynamicSharedMemory(size_t required_bytes, size_t supported_bytes, const char* kernel_name) {
  ORT_RETURN_IF(required_bytes > supported_bytes,
                "PackedSparseAttentionIndexer: ", kernel_name, " requires ", required_bytes,
                " bytes of dynamic shared memory, but the device supports at most ", supported_bytes);
  return Status::OK();
}

}  // namespace packed_sparse_attention_indexer
}  // namespace contrib
}  // namespace onnxruntime

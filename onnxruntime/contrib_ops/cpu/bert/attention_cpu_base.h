// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "attention_base.h"
#include "attention_helper.h"

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

class AttentionCPUBase : public AttentionBase {
 protected:
  AttentionCPUBase(const OpKernelInfo& info) : AttentionBase(info) {}

  template <typename T>
  Status ApplyAttention(const T* Q,                // Q data. Its size is BxNxSxH
                        const T* K,                // K data. Its size is BxNxSxH
                        const T* V,                // V value with size BxNxSxH
                        const Tensor* mask_index,  // mask index. nullptr if no mask or its size is B
                        const Tensor* past,        // past state
                        Tensor* output,            // output tensor
                        int batch_size,            // batch size
                        int sequence_length,       // sequence length
                        int qk_head_size,          // qk_head_size
                        int v_head_size,           // head_size
                        int v_hidden_size,         // hidden_size
                        const Tensor* extra_add_qk,// extra add in QK. Its size is BxNxSxS
                        OpKernelContext* context) const {
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

    auto* tp = context->GetOperatorThreadPool();

    int past_sequence_length = 0;
    Tensor* present = GetPresent(context, past, batch_size, v_head_size, sequence_length, past_sequence_length);

    // Total sequence length including that of past state: S* = S' + S
    const int all_sequence_length = past_sequence_length + sequence_length;

    // Compute the attention score. It does 2 things:
    //         I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
    //                                           1 x mask_data(B, N, S, S*)
    //         II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
    size_t attention_probs_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * all_sequence_length * sizeof(T);
    auto attention_probs = allocator->Alloc(attention_probs_bytes);
    BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

    void* mask_data = nullptr;
    if (mask_index != nullptr || (is_unidirectional_ && sequence_length > 1)) {
      size_t mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * all_sequence_length * sizeof(T);
      mask_data = allocator->Alloc(mask_data_bytes);
      memset(mask_data, 0, mask_data_bytes);
    }
    BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));

    const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;
    const std::vector<int64_t>* mask_index_dims = mask_index != nullptr ? &(mask_index->Shape().GetDims()) : nullptr;
    const T* past_data = past != nullptr ? past->template Data<T>() : nullptr;
    T* present_data = present != nullptr ? present->template MutableData<T>() : nullptr;

    const T* extra_add_qk_data = nullptr;
    if (extra_add_qk != nullptr) {
      extra_add_qk_data = extra_add_qk->template Data<T>();
    }

    ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), Q, K,
                             mask_index_data, mask_index_dims, static_cast<T*>(mask_data),
                             batch_size, sequence_length, past_sequence_length, qk_head_size == 0 ? v_head_size : qk_head_size,
                             past_data, present_data, tp, extra_add_qk_data);

    // Compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S*) x V(B, N, S*, H)
    auto out_tmp_data =
        allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * v_head_size * sizeof(T));
    BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

    ComputeVxAttentionScore(output->template MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(attention_probs), V,
                            batch_size, sequence_length, past_sequence_length, v_head_size, v_hidden_size,
                            past_data, present_data, tp);

    return Status::OK();
  }

 private:
  // Helper function to compute the attention probs. It does 2 things:
  //  I. attention_probs(B, N, S, S*) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S*, H -> B, N, H, S*) +
  //                                    1 x mask_data(B, N, S, S*)
  //  II.attention_probs(B, N, S, S*) = Softmax(attention_probs)
  template <typename T>
  void ComputeAttentionProbs(T* attention_probs,                           // output buffer for the attention probs. Its size is BxNxSxS
                             const T* Q,                                   // Q data. Its size is BxNxSxH
                             const T* K,                                   // k data. Its size is BxNxSxH
                             const int32_t* mask_index,                    // mask index. nullptr if no mask or its size is B
                             const std::vector<int64_t>* mask_index_dims,  // mask index shape
                             T* mask_data,                                 // buffer for mask data. It is nullptr if mask_index is nullptr, otherwise its shape is BxSxS*
                             int batch_size,                               // batch size of self-attention
                             int sequence_length,                          // sequence length of self-attention
                             int past_sequence_length,                     // sequence length of past state
                             int head_size,                                // head size of self-attention
                             const T* past,                                // past state
                             T* present,                                   // present state
                             ThreadPool* tp,
                             const T* extra_add_qk_data) const {
    const int all_sequence_length = past_sequence_length + sequence_length;                  // S* = S' + S
    const size_t past_chunk_length = static_cast<size_t>(past_sequence_length) * head_size;  // S' x H
    const size_t input_chunk_length = static_cast<size_t>(sequence_length) * head_size;      // S x H
    const size_t present_chunk_length = past_chunk_length + input_chunk_length;              // S* x H

    {
      if (mask_data != nullptr) {
        PrepareMask(mask_index, mask_index_dims, mask_data, is_unidirectional_, batch_size, sequence_length, past_sequence_length);
      } else {  // no any mask
        memset(attention_probs, 0, static_cast<size_t>(batch_size) * num_heads_ * sequence_length * all_sequence_length * sizeof(T));
      }

      const int loop_len = batch_size * num_heads_;
      const float alpha = 1.0f / sqrt(static_cast<float>(head_size));

      // The cost of Gemm
      const double cost = static_cast<double>(head_size) * sequence_length * all_sequence_length;

      ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t i = begin; i != end; ++i) {
          const std::ptrdiff_t batch_index = i / num_heads_;

          // broadcast mask data: (Bx)SxS* -> (BxNx)SxS*
          if (mask_data != nullptr) {
            const T* broadcast_data_src = reinterpret_cast<T*>(mask_data) + batch_index * sequence_length * all_sequence_length;
            T* broadcast_data_dest = reinterpret_cast<T*>(attention_probs) + sequence_length * all_sequence_length * i;
            memcpy(broadcast_data_dest, broadcast_data_src, sequence_length * all_sequence_length * sizeof(T));
          }

          const T* k = K + input_chunk_length * i;
          if (nullptr != present) {
            // concatenate past_K and K : (BxNx)S'xH, (BxNx)SxH -> (BxNx)S*xH
            k = ConcatStateChunk(past, k, present, past_chunk_length, present_chunk_length, i);
          }

          // gemm
          //                     original                 transposed             each iteration
          // A: Q                (B x N x) S x H          (B x N x) S x H        S x H
          // B: K'               (B x N x) S* x H         (B x N x) H x S*       H x S*
          // C: attention_probs  (B x N x) S x S*         (B x N x) S x S*       S x S*
          math::Gemm<T, ThreadPool>(CblasNoTrans, CblasTrans, sequence_length, all_sequence_length, head_size, alpha,
                                    Q + input_chunk_length * i, k, 1.0,
                                    reinterpret_cast<T*>(attention_probs) + sequence_length * all_sequence_length * i, nullptr);
        }
      });
    }

    if (extra_add_qk_data != nullptr) {
      int batch_offset = num_heads_*sequence_length*all_sequence_length;
      int head_offset = sequence_length*all_sequence_length;
      int seq_len_offset = all_sequence_length;

      for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_heads_; j++) {
          for (int k = 0; k < sequence_length; k++) {
            for (int l = 0; l < all_sequence_length; l++) {
              int final_offset =  i*batch_offset + j*head_offset + k*seq_len_offset + l;
              attention_probs[final_offset] += extra_add_qk_data[final_offset];
            }
          }
        }
      }
    }

    //  attention_probs(B, N, S, S*) = Softmax(attention_probs)
    {
      const int N = batch_size * num_heads_ * sequence_length;
      const int D = all_sequence_length;
      ComputeAttentionSoftmaxInplace(attention_probs, N, D, tp);
    }
  }

  template <typename T>
  void ComputeVxAttentionScore(T* output,                 // buffer for the result with size BxSxNxH
                               T* tmp_buffer,             // buffer for temp use with size is BxNxSxH
                               const T* attention_probs,  // Attention probs with size BxNxSxS*
                               const T* V,                // V value with size BxNxSxH
                               int batch_size,            // batch size
                               int sequence_length,       // sequence length
                               int past_sequence_length,  // sequence length in past state
                               int head_size,             // head size
                               int hidden_size,           // hidden size
                               const T* past,             // past state
                               T* present,                // present state
                               ThreadPool* tp) const {
    const int all_sequence_length = past_sequence_length + sequence_length;                  // S* = S' + S
    const size_t past_chunk_length = static_cast<size_t>(past_sequence_length * head_size);  // S' x H
    const size_t input_chunk_length = static_cast<size_t>(sequence_length * head_size);      // S x H
    const size_t present_chunk_length = past_chunk_length + input_chunk_length;              // S* x H

    // Move the pointer of past and present to start of v values.
    if (nullptr != past) {
      past += batch_size * num_heads_ * past_sequence_length * head_size;
    }
    if (nullptr != present) {
      present += batch_size * num_heads_ * all_sequence_length * head_size;
    }

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(sequence_length);

    ThreadPool::TryParallelFor(tp, batch_size * num_heads_, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const T* v = V + input_chunk_length * i;
        if (nullptr != present) {
          // concatenate past_V and V: (BxNx)S'xH, (BxNx)SxH -> (BxNx)S*xH
          v = ConcatStateChunk(past, v, present, past_chunk_length, present_chunk_length, i);
        }

        T* current_tmp_data = reinterpret_cast<T*>(tmp_buffer) + input_chunk_length * i;
        math::MatMul<T>(sequence_length, head_size, all_sequence_length,
                        attention_probs + sequence_length * all_sequence_length * i,
                        v, current_tmp_data, nullptr);

        // transpose: out(B, S, N, H) = transpose out_tmp(B, N, S, H)
        const int batch_index = static_cast<int>(i / num_heads_);
        const int head_index = static_cast<int>(i % num_heads_);
        T* src = current_tmp_data;
        T* dest = output + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
        const auto bytes_to_copy = SafeInt<size_t>(head_size) * sizeof(T);
        for (int j = 0; j < sequence_length; j++) {
          memcpy(dest, src, bytes_to_copy);
          src += head_size;
          dest += hidden_size;
        }
      }
    });
  }
};

}  // namespace contrib
}  // namespace onnxruntime

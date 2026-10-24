// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"

using namespace onnxruntime::common;

// Macros to avoid long line length
#define CUDA_MS_OP_CLASS_NAME(ver, name) \
  ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, ver, name)
#define CUDA_MS_OP_TYPED_CLASS_NAME(ver, type, name) \
  ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, ver, type, name)
#define CUDA_MS_OP_VERSIONED_TYPED_CLASS_NAME(start_ver, end_ver, type, name) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, start_ver, end_ver, type, name)
#define CUDA_MS_OP_VERSIONED_CLASS_NAME(start_ver, end_ver, name) \
  ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, start_ver, end_ver, name)
#define CUDA_MS_OP_TWO_TYPED_CLASS_NAME(ver, type1, type2, name) \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, ver, type1, type2, name)
#define CUDA_MS_OP_THREE_TYPED_CLASS_NAME(ver, type1, type2, type3, name) \
  ONNX_OPERATOR_THREE_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, ver, type1, type2, type3, name)

#define CUDA_ONNX_OP_TYPED_CLASS_NAME(ver, type, name) \
  ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, ver, type, name)
#define CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(start_ver, end_ver, type, name) \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, start_ver, end_ver, type, name)

namespace onnxruntime {
namespace contrib {
namespace cuda {
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GridSample);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FastGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Gelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Gelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, Gelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Gelu);
class CUDA_MS_OP_CLASS_NAME(1, BiasGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasSplitGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasSplitGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, QuickGelu);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, TransposeMatMul);      // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, TransposeMatMul);     // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, TransposeMatMul);  // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, FusedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FusedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedRelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedRelativePositionBias);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RemovePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RemovePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RestorePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RestorePadding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Rfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Rfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Rfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Irfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Irfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Irfft);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMulConj);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMulConj);
class CUDA_MS_OP_CLASS_NAME(1, BiasSoftmax);
class CUDA_MS_OP_CLASS_NAME(1, BiasDropout);
class CUDA_MS_OP_CLASS_NAME(1, BitmaskDropout);
class CUDA_MS_OP_CLASS_NAME(1, BitmaskBiasDropout);
class CUDA_MS_OP_CLASS_NAME(1, NGramRepeatBlock);

// These ops were experimental ops in onnx domain which have been removed now. We add them here as
// contrib ops to maintain backward compatibility
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Affine);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Affine);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Affine);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Attention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Attention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, Attention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedMultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedMultiHeadAttention);
#if !defined(DISABLE_GENERATION_OPS)
class CUDA_MS_OP_CLASS_NAME(1, BeamSearch);
class CUDA_MS_OP_CLASS_NAME(1, WhisperBeamSearch);
#endif
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ConvTransposeWithDynamicPads);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Crop);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Crop);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Crop);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, QMoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QMoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_float, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_MLFloat16, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_float, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_BFloat16, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_float, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, MultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_int8_t, GroupQueryAttention);
#ifdef USE_INT4_KV_CACHE
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_uint8_t, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_uint8_t, GroupQueryAttention);
#endif
#ifdef USE_FP8_KV_CACHE
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_Float8E4M3FN, GroupQueryAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_Float8E4M3FN, GroupQueryAttention);
#endif
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, PagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, PagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, PagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_int8_t, PagedAttention);
#ifdef USE_INT4_KV_CACHE
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_uint8_t, PagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_uint8_t, PagedAttention);
#endif
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_Float8E4M3FN, PagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_Float8E4M3FN, PagedAttention);
#endif
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, SparsePagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, SparsePagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, SparsePagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_int8_t, SparsePagedAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderAttention);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int32_t, DynamicSlice);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int64_t, DynamicSlice);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, EmbedLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, EmbedLayerNormalization);
#if !defined(DISABLE_GENERATION_OPS)
class CUDA_MS_OP_CLASS_NAME(1, GreedySearch);
#endif
class CUDA_MS_OP_CLASS_NAME(1, GroupNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, NhwcConv);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, NhwcConv);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ImageScaler);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ImageScaler);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ImageScaler);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LongformerAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LongformerAttention);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ParametricSoftplus);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ParametricSoftplus);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ParametricSoftplus);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, RotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MRotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MRotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MRotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GemmaRotaryEmbedding);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LinearAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LinearAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, LinearAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedDeltaNet);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedDeltaNet);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GatedDeltaNet);
#if !defined(DISABLE_FLOAT8_TYPES)
class CUDA_MS_OP_TYPED_CLASS_NAME(1, Float8E4M3FN_BFloat16, DequantizeLinear);
#endif
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LinearAttentionGate);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LinearAttentionGate);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, LinearAttentionGate);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedRMSNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedRMSNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GatedRMSNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BranchwiseRMSNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BranchwiseRMSNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, BranchwiseRMSNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ScaledSiLU);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ScaledSiLU);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, ScaledSiLU);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, HyperConnectionPreMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, HyperConnectionPreMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, HyperConnectionPreMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, HyperConnectionPostMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, HyperConnectionPostMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, HyperConnectionPostMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, HyperConnectionLowRankMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, HyperConnectionLowRankMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, HyperConnectionLowRankMix);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, EngramGate);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, EngramGate);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, EngramGate);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int32_t, NGramHashMapping);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, NGramHashMapping);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int32_t, VarlenNGramHashMapping);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, VarlenNGramHashMapping);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GatedAdd);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, CausalConvWithState);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, CausalConvWithState);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, CausalConvWithState);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, VarlenCausalConvWithState);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, VarlenCausalConvWithState);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, VarlenCausalConvWithState);
#if !defined(DISABLE_GENERATION_OPS)
class CUDA_MS_OP_CLASS_NAME(1, Sampling);
#endif
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ScaledTanh);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ScaledTanh);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ScaledTanh);
class CUDA_MS_OP_CLASS_NAME(1, SkipGroupNorm);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SkipLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipSimplifiedLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipSimplifiedLayerNormalization);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SkipSimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ThresholdedRelu);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ThresholdedRelu);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ThresholdedRelu);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_float, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, double_double_double, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_MLFloat16, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_MLFloat16, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_float, LayerNormalization);
class CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, BFloat16_float_BFloat16, LayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_float, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double_double_double, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_MLFloat16, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_float, SimplifiedLayerNormalization);
class CUDA_ONNX_OP_TYPED_CLASS_NAME(1, BFloat16_float_BFloat16, SimplifiedLayerNormalization);
class CUDA_MS_OP_CLASS_NAME(1, Inverse);
class CUDA_MS_OP_CLASS_NAME(1, MatMulBlockQuantizedFp4Weight);
#if !defined(DISABLE_FLOAT8_TYPES)
class CUDA_MS_OP_CLASS_NAME(1, MatMulBlockQuantizedFp8Weight);
#endif
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MatMulNBits);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulNBits);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulNBits);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MatMulBnb4);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulBnb4);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulBnb4);
class CUDA_MS_OP_CLASS_NAME(1, Trilu);
class CUDA_MS_OP_CLASS_NAME(1, UnfoldTensor);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DynamicSparseAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DynamicSparseAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, DynamicSparseAttention);
class CUDA_MS_OP_CLASS_NAME(1, DynamicTimeWarping);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, QuantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, QuantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, DequantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, DequantizeLinear);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_int8_t, QAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, QAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedConv);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, TransposeMatMul);  // backward compatibility
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FusedMatMul);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedMatMul);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedLayerNormalization);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedGelu);
class CUDA_MS_OP_CLASS_NAME(1, QuantizeWithOrder);
class CUDA_MS_OP_CLASS_NAME(1, DequantizeWithOrder);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedAttention);
class CUDA_MS_OP_CLASS_NAME(1, QOrderedLongformerAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderMaskedSelfAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderMaskedSelfAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_float, DecoderMaskedMultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float_MLFloat16, DecoderMaskedMultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_float, DecoderMaskedMultiHeadAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, DecoderMaskedMultiHeadAttention);
class CUDA_MS_OP_CLASS_NAME(1, GemmFloat8);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SparseAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SparseAttention);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SparseAttentionIndexer);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SparseAttentionIndexer);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SparseAttentionIndexer);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedSparseAttentionIndexer);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedSparseAttentionIndexer);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, PackedSparseAttentionIndexer);
class CUDA_MS_OP_CLASS_NAME(1, PackedSparseAttentionIndexerMerge);

class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, BFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, BFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, BFloat16, int64_t, GatherBlockQuantized);

#if !defined(DISABLE_FLOAT8_TYPES)
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, BFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, BFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, BFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, BFloat16, int64_t, GatherBlockQuantized);
#endif  // !defined(DISABLE_FLOAT8_TYPES)

#if !defined(DISABLE_FLOAT4_TYPES)
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, float, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, float, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, MLFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, MLFloat16, int64_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, BFloat16, int32_t, GatherBlockQuantized);
class CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, BFloat16, int64_t, GatherBlockQuantized);
#endif  // !defined(DISABLE_FLOAT4_TYPES)

#ifdef ENABLE_ATEN
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kPytorchAtenDomain, 1, ATen);
#endif

#ifdef ENABLE_TRAINING_OPS
// Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once
// 1). compute optimizer is enabled for inference or
// 2). this is needed by inference for other purpose.
class CUDA_MS_OP_CLASS_NAME(1, ShrunkenGather);
#endif

#if defined(ORT_USE_NCCL)
class CUDA_MS_OP_CLASS_NAME(1, AllReduce);
class CUDA_MS_OP_CLASS_NAME(1, AllGather);
class CUDA_MS_OP_CLASS_NAME(1, AllToAll);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ShardedMoE);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ShardedMoE);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedMatMul);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedMatMul);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSlice);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSlice);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedReshape);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReshape);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReshape);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedExpand);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedExpand);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedExpand);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceSum);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceSum);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMax);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMax);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMean);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMean);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedUnsqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedUnsqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedUnsqueeze);

class CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedSqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSqueeze);
class CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSqueeze);
#endif

template <>
KernelCreateInfo BuildKernelCreateInfo<void>() {
  KernelCreateInfo info;
  return info;
}

Status RegisterCudaContribKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<void>,  // default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GridSample)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FastGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, FastGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FastGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FastGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, Gelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BiasGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasSplitGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasSplitGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BiasAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BiasAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, QuickGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, TransposeMatMul)>,      // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, TransposeMatMul)>,     // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, TransposeMatMul)>,  // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedRelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedRelativePositionBias)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RemovePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RemovePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RestorePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RestorePadding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Rfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Rfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Rfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Irfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, double, Irfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Irfft)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ComplexMulConj)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ComplexMulConj)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, NGramRepeatBlock)>,

      // These ops were experimental ops in onnx domain which have been removed now. We add them here as
      // contrib ops to maintain backward compatibility
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Affine)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Affine)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Affine)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, Attention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, Attention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, Attention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedMultiHeadAttention)>,
#if !defined(DISABLE_GENERATION_OPS)
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BeamSearch)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, WhisperBeamSearch)>,
#endif
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ConvTransposeWithDynamicPads)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, Crop)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, Crop)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, Crop)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, QMoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, QMoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_float, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_MLFloat16, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_float, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_BFloat16, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_float, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, MultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_int8_t, GroupQueryAttention)>,
#ifdef USE_INT4_KV_CACHE
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_uint8_t, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_uint8_t, GroupQueryAttention)>,
#endif
#ifdef USE_FP8_KV_CACHE
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_Float8E4M3FN, GroupQueryAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_Float8E4M3FN, GroupQueryAttention)>,
#endif
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, PagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, PagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, PagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_int8_t, PagedAttention)>,
#ifdef USE_INT4_KV_CACHE
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_uint8_t, PagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_uint8_t, PagedAttention)>,
#endif
#if defined(USE_FP8_KV_CACHE) && !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_Float8E4M3FN, PagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_Float8E4M3FN, PagedAttention)>,
#endif
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, SparsePagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_BFloat16, SparsePagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, SparsePagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16_int8_t, SparsePagedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderAttention)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int32_t, DynamicSlice)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, int64_t, DynamicSlice)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, EmbedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, EmbedLayerNormalization)>,
#if !defined(DISABLE_GENERATION_OPS)
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, GreedySearch)>,
#endif
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, GroupNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, NhwcConv)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, NhwcConv)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ImageScaler)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ImageScaler)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ImageScaler)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LongformerAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LongformerAttention)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ParametricSoftplus)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ParametricSoftplus)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ParametricSoftplus)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, RotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, RotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, RotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MRotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MRotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MRotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GemmaRotaryEmbedding)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LinearAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LinearAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, LinearAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedDeltaNet)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedDeltaNet)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GatedDeltaNet)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, Float8E4M3FN_BFloat16, DequantizeLinear)>,
#endif
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, LinearAttentionGate)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, LinearAttentionGate)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, LinearAttentionGate)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedRMSNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedRMSNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GatedRMSNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, BranchwiseRMSNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, BranchwiseRMSNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, BranchwiseRMSNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ScaledSiLU)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ScaledSiLU)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, ScaledSiLU)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, HyperConnectionPreMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, HyperConnectionPreMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, HyperConnectionPreMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, HyperConnectionPostMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, HyperConnectionPostMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, HyperConnectionPostMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, HyperConnectionLowRankMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, HyperConnectionLowRankMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, HyperConnectionLowRankMix)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, EngramGate)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, EngramGate)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, EngramGate)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int32_t, NGramHashMapping)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, NGramHashMapping)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int32_t, VarlenNGramHashMapping)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, VarlenNGramHashMapping)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, GatedAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, GatedAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, GatedAdd)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, CausalConvWithState)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, CausalConvWithState)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, CausalConvWithState)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, VarlenCausalConvWithState)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, VarlenCausalConvWithState)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, VarlenCausalConvWithState)>,
#if !defined(DISABLE_GENERATION_OPS)
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, Sampling)>,
#endif
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ScaledTanh)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ScaledTanh)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ScaledTanh)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, SkipGroupNorm)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SkipLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SkipSimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SkipSimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SkipSimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float, ThresholdedRelu)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double, ThresholdedRelu)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16, ThresholdedRelu)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_float, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, double_double_double, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_MLFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, float_float_MLFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, MLFloat16_float_float, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_VERSIONED_TYPED_CLASS_NAME(1, 16, BFloat16_float_BFloat16, LayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_float, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, double_double_double, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_MLFloat16, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, float_float_MLFloat16, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, MLFloat16_float_float, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_ONNX_OP_TYPED_CLASS_NAME(1, BFloat16_float_BFloat16, SimplifiedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, Inverse)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, MatMulBlockQuantizedFp4Weight)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, MatMulBlockQuantizedFp8Weight)>,
#endif
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MatMulNBits)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulNBits)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulNBits)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, MatMulBnb4)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, MatMulBnb4)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, MatMulBnb4)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BiasSoftmax)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BiasDropout)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BitmaskDropout)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, BitmaskBiasDropout)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, QuantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int8_t_MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, uint8_t_MLFloat16, DequantizeLinear)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_int8_t, QAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_int8_t, QAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, UnfoldTensor)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DynamicSparseAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DynamicSparseAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, DynamicSparseAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, DynamicTimeWarping)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, Trilu)>,
      // TransposedMatMul is still here for backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, TransposeMatMul)>,  // backward compatibility
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, FusedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, FusedConv)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedLayerNormalization)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedGelu)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QuantizeWithOrder)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, DequantizeWithOrder)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, QOrderedLongformerAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DecoderMaskedSelfAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DecoderMaskedSelfAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_float, DecoderMaskedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float_MLFloat16, DecoderMaskedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_float, DecoderMaskedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16_MLFloat16, DecoderMaskedMultiHeadAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, GemmFloat8)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SparseAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SparseAttention)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, SparseAttentionIndexer)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, SparseAttentionIndexer)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, SparseAttentionIndexer)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, PackedSparseAttentionIndexer)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, PackedSparseAttentionIndexer)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, BFloat16, PackedSparseAttentionIndexer)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, PackedSparseAttentionIndexerMerge)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, uint8_t, BFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, UInt4x2, BFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Int4x2, BFloat16, int64_t, GatherBlockQuantized)>,
#if !defined(DISABLE_FLOAT8_TYPES)
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FN, BFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E4M3FNUZ, BFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2, BFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float8E5M2FNUZ, BFloat16, int64_t, GatherBlockQuantized)>,
#endif  // !defined(DISABLE_FLOAT8_TYPES)
#if !defined(DISABLE_FLOAT4_TYPES)
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, float, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, float, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, MLFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, MLFloat16, int64_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, BFloat16, int32_t, GatherBlockQuantized)>,
      BuildKernelCreateInfo<CUDA_MS_OP_THREE_TYPED_CLASS_NAME(1, Float4E2M1x2, BFloat16, int64_t, GatherBlockQuantized)>,
#endif  // !defined(DISABLE_FLOAT4_TYPES)

#ifdef ENABLE_ATEN
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kPytorchAtenDomain, 1, ATen)>,
#endif

#ifdef ENABLE_TRAINING_OPS
      // Should remove the shrunken_gather include from ENABLE_TRAINING_OPS once
      // 1). compute optimizer is enabled for inference or
      // 2). this is needed by inference for other purpose.
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, ShrunkenGather)>,
#endif

#if defined(ORT_USE_NCCL)
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, AllReduce)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, AllGather)>,
      BuildKernelCreateInfo<CUDA_MS_OP_CLASS_NAME(1, AllToAll)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, ShardedMoE)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, ShardedMoE)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedMatMul)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedMatMul)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSlice)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSlice)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedReshape)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReshape)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReshape)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedExpand)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedExpand)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedExpand)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceSum)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceSum)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMax)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMax)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedReduceMean)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedReduceMean)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedUnsqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedUnsqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedUnsqueeze)>,

      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, int64_t, DistributedSqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, float, DistributedSqueeze)>,
      BuildKernelCreateInfo<CUDA_MS_OP_TYPED_CLASS_NAME(1, MLFloat16, DistributedSqueeze)>,
#endif
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

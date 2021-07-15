// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_fwd.h"
#include "core/providers/cuda/cuda_pch.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, View);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Group);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, PassThrough);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SGDOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ReduceSumTraining);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, ReduceSumTraining);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int32_t, ReduceSumTraining);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ReduceSumTraining);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SplitTraining);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, ConcatTraining);

// Adam
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_float_float_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int64_t_float_MLFloat16_float_float_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_MLFloat16_float_float_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_MLFloat16_MLFloat16_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_MLFloat16_float_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int64_t_float_MLFloat16_MLFloat16_MLFloat16_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int64_t_float_MLFloat16_MLFloat16_float_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_MLFloat16_MLFloat16_MLFloat16_MLFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_MLFloat16_MLFloat16_float_MLFloat16, AdamOptimizer);
// Lamb
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_float_float_float_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_MLFloat16_float_MLFloat16_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_MLFloat16_float_float_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double_double_double_double_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_MLFloat16_MLFloat16_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_MLFloat16_float_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_float_MLFloat16_MLFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_float_float_MLFloat16, LambOptimizer);
// Gradient accumulator
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, InPlaceAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_MLFloat16, InPlaceAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_MLFloat16, InPlaceAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, InPlaceAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ZeroGradient);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ZeroGradient);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SoftmaxCrossEntropy);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SoftmaxCrossEntropyGrad);
// class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropy);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropy);
// class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropyGrad);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropyGrad);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, int64_t, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, int64_t, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, int64_t, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, int64_t, SoftmaxCrossEntropyLoss);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, int64_t, SoftmaxCrossEntropyLossGrad);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, int64_t, SoftmaxCrossEntropyLossGrad);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, int64_t, SoftmaxCrossEntropyLossInternal);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, int64_t, SoftmaxCrossEntropyLossInternal);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, int64_t, SoftmaxCrossEntropyLossInternalGrad);
class ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, int64_t, SoftmaxCrossEntropyLossInternalGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, LogSoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, LogSoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, LogSoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, BatchNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, BatchNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ConvGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, ConvGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ConvGrad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GatherGrad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, DropoutGrad);

// TODO: decprecate GatherND-1 after updating training models to opset-12
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int64_t, GatherND);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int64_t, GatherNDGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DivGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, DivGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DivGrad);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GeluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, GeluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GeluGrad);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, FastGeluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, FastGeluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, FastGeluGrad);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasGeluGrad_dX);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasFastGeluGrad_dX);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ReluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, ReluGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ReluGrad);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, IsFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, IsFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, IsFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, bool, All);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, IsAllFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, IsAllFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, IsAllFinite);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MixedPrecisionScale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, MixedPrecisionScale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_MLFloat16, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_MLFloat16, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, LayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double, LayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, LayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, SimplifiedLayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double, SimplifiedLayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, SimplifiedLayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, InvertibleLayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double, InvertibleLayerNormalizationGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, InvertibleLayerNormalizationGrad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SliceGrad);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GatherElementsGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Scale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Scale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, Scale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistBinarizeEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistBinarizeEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, GistBinarizeEncoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistBinarizeDecoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistBinarizeDecoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, GistBinarizeDecoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, bool, GistPack1Encoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack1Encoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, bool, GistPack1Decoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack1Decoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack8Encoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistPack8Encoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack8Decoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistPack8Decoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack16Encoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack16Decoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPackMsfp15Encoder);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPackMsfp15Decoder);

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
// Adam
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_float_float_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_int64_t_float_BFloat16_float_float_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_BFloat16_float_float_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_BFloat16_BFloat16_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_BFloat16_float_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_int64_t_float_BFloat16_BFloat16_BFloat16_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_int64_t_float_BFloat16_BFloat16_float_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_BFloat16_BFloat16_BFloat16_BFloat16, AdamOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_BFloat16_BFloat16_float_BFloat16, AdamOptimizer);
// Lamb
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_float_float_float_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_BFloat16_float_BFloat16_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_BFloat16_float_float_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double_double_double_double_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_BFloat16_BFloat16_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_BFloat16_float_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_float_BFloat16_BFloat16, LambOptimizer);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_float_float_BFloat16, LambOptimizer);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_BFloat16, InPlaceAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_BFloat16, InPlaceAccumulator);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float, InPlaceAccumulator);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, SoftmaxGrad);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, MixedPrecisionScale);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float, LayerNormalizationGrad);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_BFloat16, ReduceAllL2);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_BFloat16, ReduceAllL2);
#endif

#if defined(ORT_USE_NCCL) || defined(USE_MPI)
// P2P communication operators.
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Send);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Recv);
#endif

#ifdef USE_MPI
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, AdasumAllReduce);
#endif

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, RecordEvent);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, WaitEvent);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, YieldOp);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, ATenOp);

#ifdef ENABLE_TRAINING_TORCH_INTEROP
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, PythonOp);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, PythonOpGrad);
#endif

#ifdef ORT_USE_NCCL
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NcclAllReduce);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NcclAllGather);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NcclReduceScatter);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MegatronF);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MegatronG);
#endif

Status RegisterCudaTrainingKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, View)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Group)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, PassThrough)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SGDOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ReduceSumTraining)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, ReduceSumTraining)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int32_t, ReduceSumTraining)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ReduceSumTraining)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SplitTraining)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, ConcatTraining)>,
    // Adam
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_float_float_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int64_t_float_MLFloat16_float_float_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_MLFloat16_float_float_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_MLFloat16_MLFloat16_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_MLFloat16_float_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int64_t_float_MLFloat16_MLFloat16_MLFloat16_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_int64_t_float_MLFloat16_MLFloat16_float_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_MLFloat16_MLFloat16_MLFloat16_MLFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_MLFloat16_MLFloat16_float_MLFloat16, AdamOptimizer)>,

    // Lamb
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_float_float_float_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_MLFloat16_float_MLFloat16_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_MLFloat16_float_float_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double_double_double_double_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_MLFloat16_MLFloat16_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_MLFloat16_float_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_float_MLFloat16_MLFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float_MLFloat16_float_float_MLFloat16, LambOptimizer)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, InPlaceAccumulator)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_MLFloat16, InPlaceAccumulator)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_MLFloat16, InPlaceAccumulator)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, InPlaceAccumulator)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ZeroGradient)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ZeroGradient)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, DropoutGrad)>,

    // TODO: decprecate GatherND-1 after updating training models to opset-12
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 1, int64_t, GatherND)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, int64_t, GatherNDGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SoftmaxCrossEntropy)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SoftmaxCrossEntropyGrad)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropy)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropy)>,
    // BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int32_t, SparseSoftmaxCrossEntropyGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 9, float, int64_t, SparseSoftmaxCrossEntropyGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, SoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, SoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, SoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, LogSoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, LogSoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, LogSoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, MLFloat16, int64_t, SoftmaxCrossEntropyLoss)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, 12, float, int64_t, SoftmaxCrossEntropyLoss)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, MLFloat16, int64_t, SoftmaxCrossEntropyLoss)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 13, float, int64_t, SoftmaxCrossEntropyLoss)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, int64_t, SoftmaxCrossEntropyLossGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, int64_t, SoftmaxCrossEntropyLossGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, int64_t, SoftmaxCrossEntropyLossInternal)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, int64_t, SoftmaxCrossEntropyLossInternal)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, int64_t, SoftmaxCrossEntropyLossInternalGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, int64_t, SoftmaxCrossEntropyLossInternalGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, BatchNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, BatchNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ConvGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, ConvGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ConvGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GatherGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, DivGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, DivGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, DivGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GeluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, GeluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GeluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, FastGeluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, FastGeluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, FastGeluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasGeluGrad_dX)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BiasFastGeluGrad_dX)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ReluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, ReluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, ReluGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, IsFinite)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, IsFinite)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, IsFinite)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, bool, All)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, IsAllFinite)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, IsAllFinite)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, IsAllFinite)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, MixedPrecisionScale)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, MixedPrecisionScale)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, ReduceAllL2)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, ReduceAllL2)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_MLFloat16, ReduceAllL2)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_MLFloat16, ReduceAllL2)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, LayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double, LayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, LayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, SimplifiedLayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double, SimplifiedLayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, SimplifiedLayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float, InvertibleLayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double, InvertibleLayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16_float, InvertibleLayerNormalizationGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, SliceGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, GatherElementsGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, Scale)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Scale)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, Scale)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistBinarizeEncoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistBinarizeEncoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, GistBinarizeEncoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistBinarizeDecoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistBinarizeDecoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double, GistBinarizeDecoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, bool, GistPack1Encoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack1Encoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, bool, GistPack1Decoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack1Decoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack8Encoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistPack8Encoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack8Decoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MLFloat16, GistPack8Decoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack16Encoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPack16Decoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPackMsfp15Encoder)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, GistPackMsfp15Decoder)>,

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    // Adam
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_float_float_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_int64_t_float_BFloat16_float_float_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_BFloat16_float_float_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_BFloat16_BFloat16_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_float_BFloat16_float_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_int64_t_float_BFloat16_BFloat16_BFloat16_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_int64_t_float_BFloat16_BFloat16_float_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_BFloat16_BFloat16_BFloat16_BFloat16, AdamOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_int64_t_float_BFloat16_BFloat16_float_BFloat16, AdamOptimizer)>,
    // Lamb
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_float_float_float_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_BFloat16_float_BFloat16_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_float_BFloat16_float_float_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, double_double_double_double_double_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_BFloat16_BFloat16_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_BFloat16_float_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_float_BFloat16_BFloat16, LambOptimizer)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float_BFloat16_float_float_BFloat16, LambOptimizer)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_BFloat16, InPlaceAccumulator)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_BFloat16, InPlaceAccumulator)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float, InPlaceAccumulator)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, SoftmaxGrad)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16, MixedPrecisionScale)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float, LayerNormalizationGrad)>,

    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_float, ReduceAllL2)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float_BFloat16, ReduceAllL2)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, BFloat16_BFloat16, ReduceAllL2)>,
#endif

// P2P communication operators.
#if defined(ORT_USE_NCCL) || defined(USE_MPI)
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Send)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, Recv)>,
#endif

#ifdef USE_MPI
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, AdasumAllReduce)>,
#endif

    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, RecordEvent)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, WaitEvent)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, YieldOp)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, ATenOp)>,

#ifdef ENABLE_TRAINING_TORCH_INTEROP
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, PythonOp)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, PythonOpGrad)>,
#endif

#ifdef ORT_USE_NCCL
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NcclAllReduce)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NcclAllGather)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, NcclReduceScatter)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MegatronF)>,
    BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, MegatronG)>,
#endif
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

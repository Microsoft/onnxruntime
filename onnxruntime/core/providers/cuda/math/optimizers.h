// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
void SGDOptimizerImpl(
    const T* eta,
    const T* weights,
    const T* gradients,
    T* weight_out,
    size_t count);

class SGDOptimizer final : public CudaKernel {
 public:
  SGDOptimizer(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
void AdamOptimizerImpl(
    const T1* eta,
    const T2* update_count,
    const T3* weights,
    const T_GRAD* grads,
    const T4* moment_1,
    const T4* moment_2,
    T4 alpha,
    T4 beta,
    T4 lambda,
    T4 epsilon,
    T3* weight_out,
    T4* moment_1_out,
    T4* moment_2_out,
    T2* update_count_out,
    half* fp16_weights_out,
    size_t count);

template <typename T1, typename T2, typename T3, typename T4, typename T_GRAD>
class AdamOptimizer final : public CudaKernel {
 public:
  AdamOptimizer(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("lambda", &lambda_, 0.0f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-6f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float lambda_;
  float epsilon_;
};

// Implementation can be found in cuda file, optimizers_impl.cu
// T1's precision should be higher than T2.
template <typename T1, typename T2, typename T3>
void LambComputeDirectionImpl(
    const T1* weights,
    const T2* grads,
    const T3* moment_1,
    const T3* moment_2,
    T3 alpha,
    T3 beta,
    T1 lambda,
    T3 epsilon,
    T2* update_direction,
    T3* moment_1_out,
    T3* moment_2_out,
    size_t count);

// Implementation can be found in cuda file, optimizers_impl.cu
// T2's precision should be higher than T1.
template <typename T1, typename T2>
void LambUpdateImpl(
    const T1* eta,
    const T2* r_norm,
    const T2* w_norm,
    const T2* weights,
    const T1* update_direction,
    T2* weights_out,
    size_t count);

// Implementation can be found in cuda file, optimizers_impl.cu
template <typename T1, typename T2>
void LambScalarL2NormReductionImpl(
    const T1* value,
    T2* value_out);

template <typename T1, typename T2, typename T3, typename T4>
class LambOptimizer final : public CudaKernel {
 public:
  LambOptimizer(const OpKernelInfo& info) : CudaKernel(info) {
    info.GetAttrOrDefault("alpha", &alpha_, 0.9f);
    info.GetAttrOrDefault("beta", &beta_, 0.999f);
    info.GetAttrOrDefault("lambda", &lambda_, 0.0f);
    info.GetAttrOrDefault("epsilon", &epsilon_, 1e-6f);
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  float alpha_;
  float beta_;
  float lambda_;
  float epsilon_;
};

template <typename T>
class ZeroGradient final : public CudaKernel {
 public:
  ZeroGradient(const OpKernelInfo& info) : CudaKernel(info) {}

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime

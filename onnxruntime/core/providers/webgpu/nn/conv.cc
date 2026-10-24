// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/conv2d_mm.h"
#include "core/providers/webgpu/nn/conv3d_naive.h"
#include "core/providers/webgpu/nn/im2col_matmul.h"
#include "core/providers/webgpu/nn/subgroup_matrix_conv.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/nn/grouped_conv.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

Status TransposeKernel(ComputeContext& context, const Tensor* kernel, const TensorShape& kernel_shape, Tensor* transposed_kernel, const InlinedVector<size_t>& perm) {
  // Transpose weights
  auto rank = kernel_shape.NumDimensions();
  TensorShapeVector transposed_kernel_shape_vector(rank);
  for (size_t i = 0; i < rank; ++i) {
    transposed_kernel_shape_vector[i] = kernel_shape[perm[i]];
  }
  TensorShape transposed_kernel_shape(transposed_kernel_shape_vector);
  *transposed_kernel = context.CreateGPUTensor(kernel->DataType(), transposed_kernel_shape);
  const Tensor reshaped_kernel(kernel->DataType(), kernel_shape, const_cast<void*>(kernel->DataRaw()), kernel->Location());
  return Transpose::DoTranspose(context, perm, reshaped_kernel, *transposed_kernel);
}

template <bool is_channels_last, bool is_fused>
typename Conv<is_channels_last, is_fused>::ConvOptImpl*
Conv<is_channels_last, is_fused>::ConvOptImplCache::GetOrCreate(
    const Conv<is_channels_last, is_fused>& parent, const ComputeContextBase& context) {
  std::call_once(init_flag_, [&]() {
    impl_ = CreateSubgroupMatrixConvImpl(parent, context);
  });
  return impl_.get();
}

template <bool is_channels_last, bool is_fused>
Status Conv<is_channels_last, is_fused>::ComputeInternal(ComputeContext& context) const {
  // Preferred fast path on devices with a vendor subgroup-matrix tiling policy;
  // declines (handled=false) when the kernel/attrs don't qualify.
  if (ConvOptImpl* opt_impl = opt_impl_cache_.GetOrCreate(*this, context); opt_impl != nullptr) {
    bool handled = false;
    ORT_RETURN_IF_ERROR(opt_impl->Compute(context, handled));
    if (handled) {
      return Status::OK();
    }
  }

  bool has_bias = context.InputCount() > 2;
  const auto* input = context.Input<Tensor>(0);
  const Tensor* kernel = prepacked_kernel_ ? prepacked_kernel_.get() : context.Input<Tensor>(1);
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;
  TensorShape input_shape = input->Shape();
  ORT_ENFORCE(kernel != nullptr, "Conv kernel tensor is required.");
  // Prepacked kernels are stored permuted; recover the logical OIHW shape.
  TensorShape kernel_shape = kernel->Shape();
  switch (kernel_layout_) {
    case KernelLayout::OIHW:
      break;
    case KernelLayout::HWIO:
      kernel_shape = TensorShape(TensorShapeVector{kernel_shape[3], kernel_shape[2], kernel_shape[0], kernel_shape[1]});
      break;
    case KernelLayout::OHWI:
      kernel_shape = TensorShape(TensorShapeVector{kernel_shape[0], kernel_shape[3], kernel_shape[1], kernel_shape[2]});
      break;
  }
  ConvAttributes::ConvPadVector local_pads(conv_attrs_.pads.begin(), conv_attrs_.pads.end());
  TensorShapeVector local_dilations(conv_attrs_.dilations.begin(), conv_attrs_.dilations.end());
  TensorShapeVector local_strides(conv_attrs_.strides.begin(), conv_attrs_.strides.end());
  TensorShapeVector kernel_spacial_shape_vector;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(kernel_shape, kernel_spacial_shape_vector, false));
  if (local_pads.empty()) {
    local_pads.resize(kernel_spacial_shape_vector.size() * 2, 0);
  }
  if (local_dilations.empty()) {
    local_dilations.resize(kernel_spacial_shape_vector.size(), 1);
  }
  if (local_strides.empty()) {
    local_strides.resize(kernel_spacial_shape_vector.size(), 1);
  }
  TensorShapeVector input_shape_vector = input_shape.AsShapeVector();
  auto batch = input_shape[0];
  TensorShapeVector output_shape_vector = {batch};
  TensorShape input_spacial_shape = is_channels_last ? TensorShape(TensorShapeVector(std::next(input_shape_vector.begin()), std::prev(input_shape_vector.end()))) : input_shape.Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_spacial_shape, kernel_spacial_shape_vector, local_strides, local_dilations, local_pads, output_shape_vector));
  auto output_channels = kernel_shape[0];
  if (is_channels_last) {
    output_shape_vector.push_back(output_channels);
  } else {
    output_shape_vector.insert(output_shape_vector.begin() + 1, output_channels);
  }
  auto output_shape = TensorShape(output_shape_vector);
  auto* output = context.Output(0, output_shape);
  std::vector<uint32_t> strides;
  std::vector<uint32_t> pads;
  std::vector<uint32_t> dilations;
  auto transform_dim = [](int64_t dim) { return static_cast<int32_t>(dim); };
  std::transform(local_pads.begin(), local_pads.end(), std::back_inserter(pads), transform_dim);
  std::transform(local_strides.begin(), local_strides.end(), std::back_inserter(strides), transform_dim);
  std::transform(local_dilations.begin(), local_dilations.end(), std::back_inserter(dilations), transform_dim);
  auto rank = input_shape.NumDimensions();
  const InlinedVector<size_t> perm = {2, 3, 1, 0};
  if (rank > 5) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only Conv1d, Conv2d, and Conv3d are supported.");
  } else if (rank == 5) {
    // Conv3D - use naive per-element shader (matching JS implementation)
    if (conv_attrs_.group != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Conv3D does not support grouped convolution (group=", conv_attrs_.group, ").");
    }
    const auto output_size = static_cast<uint32_t>(output_shape.Size());
    const auto kernel_depth = static_cast<uint32_t>(kernel_shape[2]);
    const auto kernel_height = static_cast<uint32_t>(kernel_shape[3]);
    const auto kernel_width = static_cast<uint32_t>(kernel_shape[4]);
    // pads: head padding values for each spatial dim (front, top, left)
    std::vector<uint32_t> pads_3d{pads[0], pads[1], pads[2]};
    // Extract spatial dims and channels for explicit uniforms
    const auto x_depth = static_cast<uint32_t>(input_shape[is_channels_last ? 1 : 2]);
    const auto x_height = static_cast<uint32_t>(input_shape[is_channels_last ? 2 : 3]);
    const auto x_width = static_cast<uint32_t>(input_shape[is_channels_last ? 3 : 4]);
    const auto x_channels = static_cast<uint32_t>(input_shape[is_channels_last ? 4 : 1]);
    Conv3DNaiveProgram program(activation_, has_bias, is_channels_last);
    program.CacheHint(activation_.CacheKey(), std::to_string(is_channels_last))
        .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, input_shape, 1})
        .AddInput({kernel, ProgramTensorMetadataDependency::TypeAndRank, kernel_shape, 1})
        .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, output_shape, 1})
        .AddUniformVariables({{output_size},
                              {std::vector<uint32_t>{kernel_depth, kernel_height, kernel_width}},
                              {pads_3d},
                              {strides},
                              {dilations},
                              {std::vector<uint32_t>{x_depth, x_height, x_width}},
                              {x_channels}})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    // Activation uniforms must remain last because definitions and values are matched by index.
    AppendActivationUniformsData(activation_, program);
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::TypeAndRank, bias->Shape(), 1});
    }
    return context.RunProgram(program);
  } else if (rank == 4) {
    // Conv2D
  } else if (rank == 3) {
    // Conv1D
    TensorShapeVector kernel_shape_vector = kernel_shape.AsShapeVector();
    input_shape_vector.insert(input_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
    output_shape_vector.insert(output_shape_vector.begin() + (is_channels_last ? 1 : 2), 1, 1);
    kernel_shape_vector.insert(kernel_shape_vector.begin() + 2, 1);
    input_shape = TensorShape(input_shape_vector);
    kernel_shape = TensorShape(kernel_shape_vector);
    pads.insert(pads.begin(), 0);
    pads.insert(pads.begin() + 2, 0);
    strides.insert(strides.begin(), 1);
    dilations.insert(dilations.begin(), 1);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input and kernel tensors must have at least 3 dimensions");
  }
  std::vector<const Tensor*> inputs(has_bias ? 3 : 2);
  inputs[0] = input;
  inputs[1] = kernel;
  if (has_bias) {
    inputs[2] = bias;
  }
  std::vector<TensorShape> modified_input_output_shapes = {input_shape, kernel_shape};
  if (has_bias) {
    modified_input_output_shapes.push_back(bias->Shape());
  }
  modified_input_output_shapes.push_back(TensorShape(output_shape_vector));

  const auto input_height = input_shape[is_channels_last ? 1 : 2];
  const auto input_width = input_shape[is_channels_last ? 2 : 3];
  const auto input_channels = input_shape[is_channels_last ? 3 : 1];
  const auto kernel_height = kernel_shape[2];
  const auto kernel_width = kernel_shape[3];
  const auto output_height = output_shape_vector[is_channels_last ? 1 : 2];
  const auto output_width = output_shape_vector[is_channels_last ? 2 : 3];

  // pads[0] and pads[1] already contain the correct head (beginning) padding values
  // computed by InferPadsAndOutputShape() which handles auto_pad correctly.
  // For SAME_UPPER: head gets less padding (pad_needed / 2)
  // For SAME_LOWER: head gets more padding ((pad_needed + 1) / 2)
  std::vector<uint32_t> updated_pads{pads[0], pads[1]};

  if (CanApplyIm2ColMatMulProgram(context,
                                  is_channels_last,
                                  activation_,
                                  kernel_shape,
                                  onnxruntime::narrow<uint32_t>(conv_attrs_.group),
                                  kernel->DataType())) {
    // A prepacked kernel must be OHWI here. If it were packed for another consumer, the
    // argument below would be null and ApplyIm2ColMatMulProgram would fall back to
    // transposing input 1 -- which PrePackInternal already had ORT release.
    ORT_ENFORCE(!prepacked_kernel_ || kernel_layout_ == KernelLayout::OHWI,
                "Im2ColMatMul path reached with a kernel prepacked for a different layout.");
    return ApplyIm2ColMatMulProgram(context,
                                    is_channels_last,
                                    activation_,
                                    dilations,
                                    pads,
                                    strides,
                                    kernel_layout_ == KernelLayout::OHWI ? kernel : nullptr,
                                    output);
  }

  // The OHWI layout is only understood by the im2col path above. Reaching here with it
  // would mean PrePackInternal and ComputeInternal disagree on whether the im2col path
  // applies, and the branches below -- which expect either OIHW or HWIO -- would
  // silently misread the layout.
  ORT_ENFORCE(kernel_layout_ != KernelLayout::OHWI,
              "Kernel was prepacked as OHWI but the Im2ColMatMul path was not taken.");

  // Every remaining consumer wants HWIO, so the kernel has to be transposed unless
  // PrePackInternal already produced that layout.
  const bool kernel_needs_transpose = kernel_layout_ != KernelLayout::HWIO;

  if (conv_attrs_.group > 1) {
    Tensor transposed_kernel;
    if (is_channels_last) {
      const Tensor* grouped_kernel = kernel;
      if (kernel_needs_transpose) {
        ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
        grouped_kernel = &transposed_kernel;
      }
      inputs[1] = grouped_kernel;
      modified_input_output_shapes[1] = grouped_kernel->Shape();
    }
    auto output_channels_per_group = output_channels / conv_attrs_.group;
    // A depthwise NHWC conv - one output channel per group, and as many groups as input channels -
    // has a 1:1 channel correspondence, so it can be vectorized across channels even though each
    // group is only one channel wide. The general grouped path's `>= 4` test leaves it scalar,
    // reading every kernel tap one channel at a time.
    const bool is_depthwise_vec = is_channels_last && output_channels_per_group == 1 &&
                                  conv_attrs_.group == input_channels && GetMaxComponents(output_channels) > 1;
    // General grouped-convolution vectors must not cross an output-group boundary.
    auto components = static_cast<int>(is_channels_last && (output_channels_per_group >= 4 || is_depthwise_vec)
                                           ? GetMaxComponents(is_depthwise_vec ? output_channels : output_channels_per_group)
                                           : 1);
    auto output_size = output_shape.Size() / components;
    GroupedConvProgram program(activation_, has_bias, is_channels_last, is_depthwise_vec);
    auto reduced_kernel_shape = ReduceShapeByComponents(modified_input_output_shapes[1], components);
    auto reduced_output_shape = ReduceShapeByComponents(modified_input_output_shapes[has_bias ? 3 : 2], components);
    // Only the depthwise form reads x in whole channel vectors; the general path still needs scalar
    // channel indexing because its input channels do not line up with output vectors.
    auto x_components = is_depthwise_vec ? components : 1;
    auto reduced_x_shape = is_depthwise_vec ? ReduceShapeByComponents(modified_input_output_shapes[0], components)
                                            : modified_input_output_shapes[0];
    program.CacheHint(activation_.CacheKey(), std::to_string(components), std::to_string(is_channels_last),
                      std::to_string(is_depthwise_vec))
        .AddInput({inputs[0], ProgramTensorMetadataDependency::TypeAndRank, reduced_x_shape, x_components})
        .AddInput({inputs[1], ProgramTensorMetadataDependency::TypeAndRank, reduced_kernel_shape, components})
        .AddOutput({output, ProgramTensorMetadataDependency::TypeAndRank, reduced_output_shape, components})
        .AddUniformVariables({{static_cast<uint32_t>(output_size)}, {dilations}, {strides}, {updated_pads}, {static_cast<uint32_t>(output_channels_per_group)}, {static_cast<uint32_t>(components)}})
        .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE);
    // Activation uniforms must remain last because definitions and values are matched by index.
    AppendActivationUniformsData(activation_, program);
    if (has_bias) {
      auto reduced_bias_shape = ReduceShapeByComponents(modified_input_output_shapes[2], components);
      program.AddInput({inputs[2], ProgramTensorMetadataDependency::TypeAndRank, reduced_bias_shape, components});
    }
    return context.RunProgram(program);
  }

  const auto same_size = is_channels_last && input_height == kernel_height && input_width == kernel_width && pads[0] == 0 && pads[1] == 0;
  if (same_size || (kernel_height == 1 && kernel_width == 1 && pads[0] == 0 && pads[1] == 0 && strides[0] == 1 && strides[1] == 1)) {
    Tensor transposed_kernel;
    TensorShape matmul_a_shape;
    TensorShape matmul_b_shape;
    std::vector<const Tensor*> matmul_inputs;
    if (is_channels_last) {
      // Transpose weights
      const Tensor* matmul_kernel = kernel;
      if (kernel_needs_transpose) {
        ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
        matmul_kernel = &transposed_kernel;
      }
      inputs[1] = matmul_kernel;
      if (same_size) {
        const auto shared_dim = input_height * input_width * input_channels;
        matmul_a_shape = TensorShape({1, batch, shared_dim});
        matmul_b_shape = TensorShape({shared_dim, output_channels});
      } else {
        matmul_a_shape = TensorShape({batch, input_height * input_width, input_channels});
        matmul_b_shape = TensorShape({input_channels, output_channels});
      }
      matmul_inputs.push_back(input);
      matmul_inputs.push_back(matmul_kernel);
    } else {
      matmul_a_shape = TensorShape({1, output_channels, input_channels});
      matmul_b_shape = TensorShape({batch, input_channels, input_height * input_width});
      matmul_inputs.push_back(kernel);
      matmul_inputs.push_back(input);
    }
    const bool matmul_b_is_constant =
        is_channels_last && prepacked_kernel_ != nullptr && matmul_inputs[1] == prepacked_kernel_.get();
    Tensor matmul_a = CreateTensorView(*matmul_inputs[0], matmul_a_shape);
    Tensor matmul_b = CreateTensorView(*matmul_inputs[1], matmul_b_shape);
    matmul_inputs[0] = &matmul_a;
    matmul_inputs[1] = &matmul_b;
    if (has_bias) {
      matmul_inputs.push_back(bias);
    }
    return matmul_compute_dispatcher_.Compute(context, activation_, matmul_inputs, output, is_channels_last,
                                              matmul_b_is_constant);
  }
  // Transpose weights when necessary
  Tensor transposed_kernel;
  const Tensor* conv_kernel = kernel;
  if (kernel_needs_transpose) {
    ORT_RETURN_IF_ERROR(TransposeKernel(context, kernel, kernel_shape, &transposed_kernel, perm));
    conv_kernel = &transposed_kernel;
  }
  auto dim_a_outer = static_cast<uint32_t>(is_channels_last ? output_height * output_width : output_channels);
  auto dim_b_outer = static_cast<uint32_t>(is_channels_last ? output_channels : output_height * output_width);
  auto dim_inner = static_cast<uint32_t>(kernel_height * kernel_width * input_channels);
  inputs[1] = conv_kernel;
  TensorShape transposed_kernel_shape = conv_kernel->Shape();
  modified_input_output_shapes[1] = transposed_kernel_shape;
  Conv2dMMProgram conv2d_mm_program = CreateConv2dMMProgram(activation_, inputs, pads, strides, dilations, output, dim_a_outer, dim_b_outer, dim_inner, is_channels_last, modified_input_output_shapes);
  return context.RunProgram(conv2d_mm_program);
}

template <bool is_channels_last, bool is_fused>
Status Conv<is_channels_last, is_fused>::PrePackInternal(ComputeContextBase& context,
                                                         const Tensor& tensor,
                                                         int input_idx,
                                                         AllocatorPtr alloc,
                                                         /*out*/ bool& is_packed) {
  is_packed = false;

  // Only prepack kernel weights (input_idx == 1)
  if (input_idx != 1) {
    return Status::OK();
  }

  const auto& kernel_shape = tensor.Shape();
  const auto& dims = kernel_shape.GetDims();

  // Conv kernels must be 4D: [O, I, H, W]
  if (dims.size() != 4) {
    return Status::OK();
  }

  // Subgroup-matrix Conv needs the weight in OIHW, so skip prepacking for it. Must
  // come before Im2ColMatMul below: prepacking there first would leave
  // prepacked_kernel_ non-null, making SubgroupMatrixConvImpl::Compute silently
  // decline on every constant-weight Conv. Safe even if Compute() later declines
  // for another reason: is_packed stays false, so input 1 stays OIHW.
  if (CanApplySubgroupMatrixConv(context, is_channels_last, kernel_shape,
                                 onnxruntime::narrow<uint32_t>(conv_attrs_.group), tensor.DataType())) {
    return Status::OK();
  }

  // Im2ColMatMul path: transpose OIHW -> OHWI once here instead of on every inference.
  //
  // Placed before the auto_pad check below on purpose:
  //   - Safe: CanApplyIm2ColMatMulProgram() only looks at the adapter, dtype, layout,
  //     fusion, group and kernel H/W -- never at pads -- and ComputeInternal tests it
  //     before every pads-dependent branch. So a true here means the im2col path is
  //     taken at runtime no matter what the padding turns out to be.
  //   - Necessary: otherwise every auto_pad != NOTSET model would bail out below and
  //     keep paying for the transpose on every inference.
  //
  // This call and the one in ComputeInternal must stay in agreement: only the im2col
  // path can read the OHWI layout, so a decision made here that ComputeInternal later
  // reverses would corrupt the weights. If CanApplyIm2ColMatMulProgram() ever gains a
  // condition that is not known at prepack time (pads, strides, input shape), this
  // shortcut must go away. ComputeInternal ORT_ENFORCEs the invariant.
  if (CanApplyIm2ColMatMulProgram(context, is_channels_last, activation_,
                                  kernel_shape, onnxruntime::narrow<uint32_t>(conv_attrs_.group),
                                  tensor.DataType())) {
    ORT_RETURN_IF_ERROR(PrePackIm2ColMatMulWeight(context, tensor, alloc, prepacked_kernel_));
    kernel_layout_ = KernelLayout::OHWI;
    is_packed = true;  // set this flag to true so that ORT will release the initializer tensor
    return Status::OK();
  }

  // Grouped convolution (group > 1):
  //   - Only transposes when is_channels_last
  //   - channels_first: no transpose
  if (conv_attrs_.group > 1) {
    if constexpr (!is_channels_last) {
      // channels_first grouped conv doesn't transpose
      return Status::OK();
    }
    // is_channels_last grouped conv transposes - proceed to transpose below
  }

  // When auto_pad is not NOTSET (i.e., SAME_UPPER, SAME_LOWER, or VALID),
  // the actual padding values are computed at runtime based on input dimensions.
  // We can't predict the execution path without knowing the runtime padding,
  // so skip prepacking to avoid incorrect behavior.
  if (conv_attrs_.auto_pad != AutoPadType::NOTSET) {
    return Status::OK();
  }

  // Analyze execution paths in ComputeInternal to determine if kernel transpose is needed:
  //
  // 0. Subgroup-matrix Conv path: handled above (not prepacked -- stays OIHW)
  // 1. Im2ColMatMul path: handled above (prepacked as OHWI)
  // 2. Grouped conv (group > 1): handled above (skip if !is_channels_last)
  // 3. MatMul optimization (same_size || is_1x1_conv):
  //    - is_channels_last: transposes
  //    - !is_channels_last: does NOT transpose (uses kernel directly)
  // 4. General conv (fallback): ALWAYS transposes
  //
  // The ONLY path that doesn't transpose is: is_1x1_conv && !is_channels_last
  // (same_size requires is_channels_last, so it's always false for channels_first)

  // Get pads and strides (mirroring ComputeInternal logic)
  std::vector<int64_t> pads(conv_attrs_.pads.begin(), conv_attrs_.pads.end());
  std::vector<int64_t> strides(conv_attrs_.strides.begin(), conv_attrs_.strides.end());

  // Default pads and strides if not specified (for 4D/Conv2D: 4 pads, 2 strides)
  if (pads.empty()) {
    pads.resize(4, 0);
  }
  if (strides.empty()) {
    strides.resize(2, 1);
  }

  const int64_t kernel_height = dims[2];
  const int64_t kernel_width = dims[3];

  const bool is_1x1_conv =
      (kernel_height == 1 && kernel_width == 1 && pads[0] == 0 && pads[1] == 0 &&
       strides[0] == 1 && strides[1] == 1);

  if constexpr (!is_channels_last) {
    if (is_1x1_conv) {
      // MatMul optimization for channels_first 1x1 conv does NOT transpose
      return Status::OK();
    }
  }
  // All other paths transpose - proceed to transpose below

  // Perform the transpose using same logic as TransposeKernel
  // For 4D: perm = {2, 3, 1, 0} transforms [O, I, H, W] -> [H, W, I, O]
  const InlinedVector<size_t> perm = InlinedVector<size_t>{2, 3, 1, 0};
  auto rank = kernel_shape.NumDimensions();

  TensorShapeVector transposed_kernel_shape_vector(rank);
  for (size_t i = 0; i < rank; ++i) {
    transposed_kernel_shape_vector[i] = kernel_shape[perm[i]];
  }
  TensorShape transposed_kernel_shape(transposed_kernel_shape_vector);

  // Create the transposed kernel tensor using the prepack allocator.
  // This allocator creates GPU buffers without mapping, suitable for GPU-based operations.
  prepacked_kernel_ = std::make_unique<Tensor>(tensor.DataType(), transposed_kernel_shape, alloc);

  // Perform GPU-based transpose directly from the input GPU tensor
  ORT_RETURN_IF_ERROR(Transpose::DoTranspose(context, perm, tensor, *prepacked_kernel_));

  kernel_layout_ = KernelLayout::HWIO;
  is_packed = true;  // set this flag to true so that ORT will release the initializer tensor

  return Status::OK();
}

// Explicit template instantiation for FusedConv
template class Conv<false, false>;
template class Conv<false, true>;
template class Conv<true, false>;
template class Conv<true, true>;

#define WEBGPU_ONNX_CONV_OPERATOR_KERNEL(VERSION_FROM)                                \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Conv,                                                                           \
      kMSInternalNHWCDomain,                                                          \
      VERSION_FROM,                                                                   \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<true, true>);                                                              \
                                                                                      \
  ONNX_OPERATOR_KERNEL_EX(                                                            \
      Conv,                                                                           \
      kOnnxDomain,                                                                    \
      VERSION_FROM,                                                                   \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<false, false>);

#define WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(VERSION_FROM, VERSION_TO)          \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Conv,                                                                           \
      kOnnxDomain,                                                                    \
      VERSION_FROM, VERSION_TO,                                                       \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<false, false>);                                                            \
                                                                                      \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                                  \
      Conv,                                                                           \
      kMSInternalNHWCDomain,                                                          \
      VERSION_FROM, VERSION_TO,                                                       \
      kWebGpuExecutionProvider,                                                       \
      (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()), \
      Conv<true, true>);

WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(1, 10)
WEBGPU_ONNX_CONV_OPERATOR_VERSIONED_KERNEL(11, 21)
WEBGPU_ONNX_CONV_OPERATOR_KERNEL(22)

}  // namespace webgpu
}  // namespace onnxruntime

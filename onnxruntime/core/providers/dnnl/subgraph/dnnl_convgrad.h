// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "mkldnn_types.h"
#include "core/providers/dnnl/dnnl_fwd.h"
#include "core/providers/dnnl/dnnl_execution_provider.h"
#include "core/providers/dnnl/subgraph/dnnl_kernel.h"
#include "core/providers/dnnl/subgraph/dnnl_conv.h"

namespace onnxruntime {
namespace ort_dnnl {

/*
ConvGrad: (According to OnnxRuntime discovered using code inspection and Onnx documentation)
  Inputs:
    0) dY - Gradient of output Y
    1) X - Input Tensor
    2) W - Weight Tensor
  Outputs:
    0) dX - Gradient of Input X
    1) dW - Gradient of (W) Weight
    2) dB - Gradient of (B) Bias


                    +-----------------+
    (dY) diff_dst   |                 | (dX optional output ) diff_src
    --------------->+                 +----------------->
    (X) src         |                 | (dW optional output ) diff_weights
    --------------->+    ConvGrad     +----------------->
    (W) weights     |                 | (dB optional output ) diff_bias
    --------------->+                 +----------------->
                    |                 |
                    +-----------------+
  diff_dst = DNNL_ARG_DIFF_DST
  src = DNNL_ARG_SRC
  weights = DNNL_ARG_WEIGHTS

  diff_src = DNNL_ARG_DIFF_SRC
  diff_weights = DNNL_ARG_DIFF_WEIGHTS
  diff_bias = DNNL_ARG_DIFF_BIAS

Attributes (auto_pad, dilations, group, kernel_shap, pads, and strides) should be the same as the forward pass Conv operator

To acheive Everything specified in the OnnxRuntime ConvGrad we must use both:
1) dnnl::convolution_backward_data - used to calculate (dX) diff_src
2) dnnl::convolution_backward_weights - used to calculate (dW) diff_weights and (dB) diff_bias
*/
template <typename T>
class DnnlConvGrad : public DnnlKernel {
 public:
  DnnlConvGrad(const DnnlNode& node,
               DNNLExecutionProvider* provider,
               const NodeAttributes& attributes,
               const std::string attributes_prefix = "") : DnnlKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
    assert(node.is_ort_output_required.size() == 3);
    dx_required_ = node.is_ort_output_required[0];
    dw_required_ = node.is_ort_output_required[1];
    db_required_ = node.is_ort_output_required[2];
  }

  void AddForwardDnnlKernel(std::shared_ptr<DnnlConv<T>> conv_fwd) {
    conv_fwd_ = conv_fwd;
  }

  void CreatePrimitives(const OrtCustomOpApi* api,
                        OrtKernelContext* context,
                        const std::unordered_map<dnnl::engine::kind, dnnl::engine>& dnnl_engine,
                        std::vector<dnnl::primitive>& net,
                        std::vector<std::unordered_map<int, dnnl::memory>>& net_args) override {
    dnnl::engine cpu_engine;
    dnnl::engine engine_to_use;
    std::unordered_map<dnnl::engine::kind, dnnl::engine>::const_iterator iter = dnnl_engine.find(dnnl::engine::kind::cpu);
    if (iter != dnnl_engine.end()) {
      cpu_engine = iter->second;
      engine_to_use = cpu_engine;
    }
    gpu_available_ = false;
    dnnl::engine gpu_engine;
    iter = dnnl_engine.find(dnnl::engine::kind::gpu);
    if (iter != dnnl_engine.end()) {
      gpu_engine = iter->second;
      gpu_available_ = true;
      engine_to_use = gpu_engine;
      LOGS_DEFAULT(INFO) << "gpu engine found" << std::endl;
    }
    Ort::CustomOpApi ort{*api};
    stream_ = std::make_unique<dnnl::stream>(dnnl::stream(engine_to_use));

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    // Read tensors and tensor shapes from ORT (three inputs)
    TensorShape dy_shape;
    if (mklnode_ptr_->parent_nodes.empty()) {
      const OrtValue* dy_input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto dy_tensor_info = ort.GetTensorTypeAndShape(dy_input_tensor);
      auto dy_tensor_shape = ort.GetTensorShape(dy_tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(dy_tensor_info);
      auto dyshape = dy_tensor_shape.data();
      auto dydim = dy_tensor_shape.size();
      ort_source_format_ = GetSourceFormat(static_cast<int>(dydim));
      dy_shape = TensorShape(dyshape, dydim);
    } else {
      dy_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = parents_[0].get()->ort_source_format_;
      ort_source_desc_ = parents_[0].get()->ort_source_desc_;
      source_desc_ = parents_[0].get()->primitive_dst_desc_;
    }

    const OrtValue* x_input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    auto x_tensor_info = ort.GetTensorTypeAndShape(x_input_tensor);
    auto x_tensor_shape = ort.GetTensorShape(x_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(x_tensor_info);
    auto xshape = x_tensor_shape.data();
    auto xdim = x_tensor_shape.size();

    TensorShape x_shape(xshape, xdim);

    const OrtValue* w_input_tensor = ort.KernelContext_GetInput(context, input_index + 2);
    auto w_tensor_info = ort.GetTensorTypeAndShape(w_input_tensor);
    auto w_tensor_shape = ort.GetTensorShape(w_tensor_info);
    ort.ReleaseTensorTypeAndShapeInfo(w_tensor_info);
    auto wshape = w_tensor_shape.data();
    auto wdim = w_tensor_shape.size();

    TensorShape w_shape(wshape, wdim);

    const int group_mkl = static_cast<int>(group_);

    primitive_created_status_ = ValidateInputShape(x_shape, w_shape);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    std::vector<int64_t> kernel_shape;
    primitive_created_status_ = ComputeKernelShape(w_shape, kernel_shape);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    const size_t kernel_rank = kernel_shape.size();

    if (kernel_rank + 2 != wdim) {
      primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                                                  " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                                  " W: ", w_shape.ToString().c_str());
      return;
    }

    for (size_t i = 0; i < kernel_rank; ++i) {
      if (kernel_shape[i] != w_shape[i + 2]) {
        primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                                    " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                                    " W: ", w_shape.ToString().c_str());
        return;
      }
    }

    std::vector<int64_t> pads(pads_);
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }
    std::vector<int64_t> dilations(dilations_);
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }
    std::vector<int64_t> strides(strides_);
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    const int64_t N = x_shape[0];
    const int64_t M = w_shape[0];
    std::vector<int64_t> dx_dims;
    dx_dims.insert(dx_dims.begin(), {N, M});
    TensorShape input_shape = x_shape.Slice(2);
    primitive_created_status_ = InferOutputShape(input_shape, kernel_shape, strides, dilations, &pads, &dx_dims);
    if (!primitive_created_status_.IsOK()) {
      return;
    }

    // Setup input and output memory descriptions
    dnnl::memory::dims dy_dims(dy_shape.GetDims().begin(), dy_shape.GetDims().end());
    diff_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({dy_dims}, DnnnType<T>(), ort_source_format_));

    dnnl::memory::dims x_dims(x_shape.GetDims().begin(), x_shape.GetDims().end());
    src_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({x_dims}, DnnnType<T>(), ort_source_format_));

    dnnl::memory::dims w_dims(w_shape.GetDims().begin(), w_shape.GetDims().end());
    weights_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({w_dims}, DnnnType<T>(), ort_source_format_));

    // Verify that the inputs to the ConvGrad operator match the passed in forward Conv operator
    // The Convolution forward operator is passed in from a map that uses the weight name as the key
    // We make the assumption that the name of each weight is unique. This checks that the dimentions
    // of the ConvGrad input tensors match the the tensors from Conv operator.
    // This checks the following:
    //   - the output tensor dimensions from Conv match the dy_dims input of ConvGrad
    //   - the input tensor dimensions from Conv match the x_dims input of ConvGrad
    //   - the input weight tensor dimensions to Conv and ConvGrad match
    // If any of these don't match then the forward operator is incorrect.
    // If the unique name assumption is always true then the following check can be removed.
    dnnl::memory::desc conv_fwd_dst_desc_ = conv_fwd_->GetPrimitiveDesc()->dst_desc();
    dnnl::memory::desc conv_fwd_src_desc_ = conv_fwd_->GetPrimitiveDesc()->src_desc();
    dnnl::memory::desc conv_fwd_weights_desc_ = conv_fwd_->GetPrimitiveDesc()->weights_desc();

    if (conv_fwd_dst_desc_.dims() != dy_dims ||
        conv_fwd_src_desc_.dims() != x_dims ||
        conv_fwd_weights_desc_.dims() != w_dims) {
      primitive_created_status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Model structure is not supported by the DNNL execution provider.");
      return;
    }

    // src and diff_src have the same memory dimensions
    TensorShape dx_shape(x_dims);
    //diff_src_shape_ = dx_shape;
    //diff_src_md_ = std::make_unique < dnnl::memory::desc > (
    //    dnnl::memory::desc({x_dims}, DnnnType<T>(), dnnl::memory::format_tag::any));

    primitive_dst_shape_ = dx_shape;
    primitive_dst_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({x_dims}, DnnnType<T>(), ort_source_format_));

    // weights and diff_weights have the same memory descriptions
    TensorShape dw_shape(w_dims);
    diff_weights_shape_ = dw_shape;
    diff_weights_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({w_dims}, DnnnType<T>(), ort_source_format_));

    TensorShape db_shape({wshape[0]});
    diff_bias_shape_ = db_shape;
    diff_bias_md_ = std::make_unique<dnnl::memory::desc>(
        dnnl::memory::desc({wshape[0]}, DnnnType<T>(), dnnl::memory::format_tag::x));

    dnnl::memory::dims filter_dims_mkl;
    if (group_mkl == 1) {
      filter_dims_mkl.assign(w_shape.GetDims().begin(), w_shape.GetDims().end());
    } else {
      filter_dims_mkl.assign({group_mkl,
                              static_cast<int>(w_shape[0] / group_mkl)});
      filter_dims_mkl.insert(filter_dims_mkl.end(), w_shape.GetDims().begin() + 1, w_shape.GetDims().end());
    }
    dnnl::memory::dims conv_strides(strides.begin(), strides.end());
    dnnl::memory::dims conv_dilations(dilations.begin(), dilations.end());
    // Dnnl dilations start from 0 so we need to subtract 1 from each dim.
    for (size_t dim = 0; dim < kernel_rank; dim++) {
      conv_dilations[dim] -= 1;
    }

    dnnl::memory::dims conv_padding_left(pads.begin(), pads.begin() + kernel_rank);
    dnnl::memory::dims conv_padding_right(pads.begin() + kernel_rank, pads.end());

    auto src_format = dnnl::memory::format_tag::any;
    if (kernel_rank == 1) {
      src_format = dnnl::memory::format_tag::ncw;
      if (group_mkl == 1) {
        filter_format_ = dnnl::memory::format_tag::oiw;
      } else {
        filter_format_ = dnnl::memory::format_tag::goiw;
      }
    } else if (kernel_rank == 2) {
      src_format = dnnl::memory::format_tag::nchw;
      if (group_mkl == 1) {
        filter_format_ = dnnl::memory::format_tag::oihw;
      } else {
        filter_format_ = dnnl::memory::format_tag::goihw;
      }
    } else {
      src_format = dnnl::memory::format_tag::ncdhw;
      if (group_mkl == 1) {
        filter_format_ = dnnl::memory::format_tag::oidhw;
      } else {
        filter_format_ = dnnl::memory::format_tag::goidhw;
      }
    }

    dnnl::memory::dims src_dims_mkl(x_shape.GetDims().begin(), x_shape.GetDims().end());
    if (mklnode_ptr_->parent_nodes.empty()) {
      ort_source_format_ = src_format;
      ort_source_desc_ = dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), src_format);
      source_desc_ = dnnl::memory::desc({src_dims_mkl}, DnnnType<T>(), src_format);
    }

    conv_bwd_data_desc_ = std::make_unique<dnnl::convolution_backward_data::desc>(
        dnnl::convolution_backward_data::desc(
            dnnl::algorithm::convolution_direct,
            *primitive_dst_md_,
            *weights_md_,
            *diff_dst_md_,
            conv_strides,
            conv_dilations,
            conv_padding_left,
            conv_padding_right));

    conv_bwd_data_pd_ = std::make_unique<dnnl::convolution_backward_data::primitive_desc>(
        dnnl::convolution_backward_data::primitive_desc(
            *conv_bwd_data_desc_, engine_to_use, *(conv_fwd_->GetPrimitiveDesc())));

    conv_bwd_weights_desc_ = std::make_unique<dnnl::convolution_backward_weights::desc>(
        dnnl::convolution_backward_weights::desc(
            dnnl::algorithm::convolution_direct,
            *src_md_,
            *diff_weights_md_,
            *diff_bias_md_,
            *diff_dst_md_,
            conv_strides,
            conv_dilations,
            conv_padding_left,
            conv_padding_right));

    conv_bwd_weights_pd_ = std::make_unique<dnnl::convolution_backward_weights::primitive_desc>(
        dnnl::convolution_backward_weights::primitive_desc(
            *conv_bwd_weights_desc_, engine_to_use, *(conv_fwd_->GetPrimitiveDesc())));

    diff_dst_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(conv_bwd_weights_pd_.get()->diff_dst_desc(), cpu_engine, nullptr));

    src_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(conv_bwd_weights_pd_.get()->src_desc(), cpu_engine, nullptr));

    weights_mem_ = std::make_unique<dnnl::memory>(
        dnnl::memory(conv_bwd_data_pd_.get()->weights_desc(), cpu_engine, nullptr));

    //diff_src_mem_ = std::make_unique<dnnl::memory>(
    //    dnnl::memory(conv_bwd_data_pd_.get()->diff_src_desc(), cpu_engine, nullptr));

    if (gpu_available_) {
      diff_dst_mem_gpu_ = std::make_unique<dnnl::memory>(*diff_dst_md_, gpu_engine);
      net.push_back(mkldnn::reorder(*diff_dst_mem_, *diff_dst_mem_gpu_));
      net_args.push_back({{MKLDNN_ARG_SRC, *diff_dst_mem_}, {MKLDNN_ARG_DST, *diff_dst_mem_gpu_}});
      src_mem_gpu_ = std::make_unique<dnnl::memory>(*src_md_, gpu_engine);
      net.push_back(mkldnn::reorder(*src_mem_, *src_mem_gpu_));
      net_args.push_back({{MKLDNN_ARG_SRC, *src_mem_}, {MKLDNN_ARG_DST, *src_mem_gpu_}});
      weights_mem_gpu_ = std::make_unique<dnnl::memory>(*weights_md_, gpu_engine);
      net.push_back(mkldnn::reorder(*weights_mem_, *weights_mem_gpu_));
      net_args.push_back({{MKLDNN_ARG_SRC, *weights_mem_}, {MKLDNN_ARG_DST, *weights_mem_gpu_}});
    }

    primitive_src_desc_ = conv_bwd_data_pd_.get()->diff_dst_desc();
    primitive_dst_desc_ = conv_bwd_data_pd_.get()->diff_src_desc();
    if (dx_required_) {
      if (!gpu_available_) {
        if (mklnode_ptr_->output_index >= 0) {
          // Use Dnnl's internal output buffer
          if (primitive_dst_desc_ != ort_source_desc_) {
            primitive_dst_mem_ = std::make_unique<dnnl::memory>(
                dnnl::memory(conv_bwd_data_pd_.get()->diff_src_desc(), cpu_engine));
          } else {
            primitive_dst_mem_ = std::make_unique<dnnl::memory>(
                dnnl::memory(conv_bwd_data_pd_.get()->diff_src_desc(), cpu_engine, nullptr));
          }
        } else {
          // last node of sub-graph. need to allocate memory for output_tensor
          primitive_dst_mem_ = std::make_unique<dnnl::memory>(
              dnnl::memory(conv_bwd_data_pd_.get()->diff_src_desc(), cpu_engine));
        }
      } else {
        primitive_dst_mem_ = std::make_unique<dnnl::memory>(
            dnnl::memory(conv_bwd_data_pd_.get()->diff_src_desc(), gpu_engine));
      }
    }

    if (dw_required_ || db_required_) {
      diff_weights_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(conv_bwd_weights_pd_.get()->diff_weights_desc(), engine_to_use));

      diff_bias_mem_ = std::make_unique<dnnl::memory>(
          dnnl::memory(conv_bwd_weights_pd_.get()->diff_bias_desc(), engine_to_use));
    }
    if (dx_required_) {
      net.push_back(dnnl::convolution_backward_data(*conv_bwd_data_pd_));
      if (!gpu_available_) {
        net_args.push_back({{DNNL_ARG_DIFF_DST, *diff_dst_mem_},
                            {DNNL_ARG_WEIGHTS, *weights_mem_},
                            {DNNL_ARG_DIFF_SRC, *primitive_dst_mem_}});
      } else {
        net_args.push_back({{DNNL_ARG_DIFF_DST, *diff_dst_mem_gpu_},
                            {DNNL_ARG_WEIGHTS, *weights_mem_gpu_},
                            {DNNL_ARG_DIFF_SRC, *primitive_dst_mem_}});
      }
    }
    if (dw_required_ || db_required_) {
      net.push_back(dnnl::convolution_backward_weights(*conv_bwd_weights_pd_));
      if (!gpu_available_) {
        net_args.push_back({{DNNL_ARG_SRC, *src_mem_},
                            {DNNL_ARG_DIFF_DST, *diff_dst_mem_},
                            {DNNL_ARG_DIFF_BIAS, *diff_bias_mem_},
                            {DNNL_ARG_DIFF_WEIGHTS, *diff_weights_mem_}});
      } else {
        net_args.push_back({{DNNL_ARG_SRC, *src_mem_gpu_},
                            {DNNL_ARG_DIFF_DST, *diff_dst_mem_gpu_},
                            {DNNL_ARG_DIFF_BIAS, *diff_bias_mem_},
                            {DNNL_ARG_DIFF_WEIGHTS, *diff_weights_mem_}});
      }
    }

    if (mklnode_ptr_->output_index >= 0) {
      dnnl::memory::data_type t = DnnnType<T>();
      if (dx_required_) {
        InitDstReorderOutput(cpu_engine, t, net, net_args, gpu_available_);
      }
      // Allocate dst buffer if reorder is necessary
      if (gpu_available_) {
        // reorder to ONNXRuntime format
        if (dw_required_) {
          diff_weights_reorder_mem_to_ = std::make_unique<dnnl::memory>(dnnl::memory(*diff_weights_md_, cpu_engine));
          net.push_back(dnnl::reorder(*diff_weights_mem_, *diff_weights_reorder_mem_to_));
          net_args.push_back({{DNNL_ARG_FROM, *diff_weights_mem_},
                              {DNNL_ARG_TO, *diff_weights_reorder_mem_to_}});
        }
        if (db_required_) {
          diff_bias_reorder_mem_to_ = std::make_unique<dnnl::memory>(dnnl::memory(*diff_bias_md_, cpu_engine));
          net.push_back(dnnl::reorder(*diff_bias_mem_, *diff_bias_reorder_mem_to_));
          net_args.push_back({{DNNL_ARG_FROM, *diff_bias_mem_},
                              {DNNL_ARG_TO, *diff_bias_reorder_mem_to_}});
        }
      }
    }
  }

  Status Bind(const OrtCustomOpApi* api, OrtKernelContext* context) override {
    Ort::CustomOpApi ort{*api};

    ORT_RETURN_IF_ERROR(primitive_created_status_);

    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    const OrtValue* dy_input_tensor = ort.KernelContext_GetInput(context, input_index);
    const T* dy_data = const_cast<T*>(ort.GetTensorData<T>(dy_input_tensor));
    diff_dst_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(dy_data)));

    const OrtValue* x_input_tensor = ort.KernelContext_GetInput(context, input_index + 1);
    const T* x_data = const_cast<T*>(ort.GetTensorData<T>(x_input_tensor));
    src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(x_data)));

    const OrtValue* w_input_tensor = ort.KernelContext_GetInput(context, input_index + 2);
    //const T* w_data = const_cast<T*>(ort.GetTensorData<T>(w_input_tensor));
    const T* w_data = ort.GetTensorData<T>(w_input_tensor);
    weights_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(w_data)));

    if (mklnode_ptr_->output_index >= 0) {
      // Allocate memory for output buffers
      // mklnode_ptr->output_index is the last index value of the node.
      // for that reason we ge the output tensors from last to first. db, dw, then dx
      size_t output_index = mklnode_ptr_->output_index;
      if (db_required_) {
        auto& db_dims = diff_bias_shape_.GetDims();
        OrtValue* db_output = ort.KernelContext_GetOutput(context, output_index, &db_dims[0], static_cast<int>(db_dims.size()));
        T* db_data = ort.GetTensorMutableData<T>(db_output);
        if (!gpu_available_) {
          diff_bias_mem_->set_data_handle(db_data);
        } else {
          diff_bias_reorder_mem_to_->set_data_handle(db_data);
        }
        --output_index;
      }

      if (dw_required_ && output_index >= 0) {
        auto& dw_dims = diff_weights_shape_.GetDims();
        OrtValue* dw_output = ort.KernelContext_GetOutput(context, output_index, &dw_dims[0], static_cast<int>(dw_dims.size()));
        T* dw_data = ort.GetTensorMutableData<T>(dw_output);
        if (!gpu_available_) {
          diff_weights_mem_->set_data_handle(dw_data);
        } else {
          diff_weights_reorder_mem_to_->set_data_handle(dw_data);
        }
        --output_index;
      }

      if (dx_required_ && output_index >= 0) {
        auto& dx_dims = primitive_dst_shape_.GetDims();
        OrtValue* dx_output = ort.KernelContext_GetOutput(context, output_index, &dx_dims[0], static_cast<int>(dx_dims.size()));
        T* diff_src_data = ort.GetTensorMutableData<T>(dx_output);
        if (!gpu_available_) {
          if (primitive_dst_desc_ != ort_source_desc_) {
            reorder_dst_mem_to_->set_data_handle(diff_src_data);
          } else {
            primitive_dst_mem_->set_data_handle(diff_src_data);
          }
        } else {
          reorder_dst_mem_to_->set_data_handle(diff_src_data);
        }
        --output_index;
      }
    }

    return Status::OK();
  }

 private:
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    std::string auto_pad;
    auto attr = attributes.find(attributes_prefix + "auto_pad");
    if (attr != attributes.end() &&
        attr->second().type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING) {
      auto_pad = attr->second().s();
    }
    auto_pad_ = (auto_pad != "") ? StringToAutoPadType(auto_pad) : AutoPadType::NOTSET;

    kernel_shape_specified_ = false;
    attr = attributes.find(attributes_prefix + "kernel_shape");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      Status status = GetIntsAttr(proto, kernel_shape_);
      kernel_shape_specified_ = true;
    }

    attr = attributes.find(attributes_prefix + "strides");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      Status status = GetIntsAttr(proto, strides_);
    }

    bool attr_read = false;
    attr = attributes.find(attributes_prefix + "pads");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      if (GetIntsAttr(proto, pads_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      pads_.resize(kernel_shape_.size() * 2, 0);
    }

    attr_read = false;
    attr = attributes.find(attributes_prefix + "dilations");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      if (GetIntsAttr(proto, dilations_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      dilations_.resize(kernel_shape_.size(), 1);
    }

    attr_read = false;
    attr = attributes.find(attributes_prefix + "group");
    if (attr != attributes.end()) {
      auto& proto = attr->second();
      if (GetIntAttr(proto, group_) == Status::OK())
        attr_read = true;
    }
    if (!attr_read) {
      group_ = 1;
    }
  }

 private:
  bool dx_required_;
  bool dw_required_;
  bool db_required_;
  std::shared_ptr<DnnlConv<T>> conv_fwd_;

  dnnl::memory::desc filter_desc_;
  dnnl::memory::format_tag filter_format_;

  std::shared_ptr<dnnl::memory> src_mem_from_;
  std::unique_ptr<dnnl::memory> src_mem_to_;

  size_t src_size_;
  size_t filter_size_;
  size_t dst_size_;

  std::unique_ptr<dnnl::convolution_backward_data::desc> conv_bwd_data_desc_;
  std::unique_ptr<dnnl::convolution_backward_data::primitive_desc> conv_bwd_data_pd_;

  std::unique_ptr<dnnl::convolution_backward_weights::desc> conv_bwd_weights_desc_;
  std::unique_ptr<dnnl::convolution_backward_weights::primitive_desc> conv_bwd_weights_pd_;

  std::unique_ptr<dnnl::primitive> conv_bwd_;

  std::unique_ptr<dnnl::memory::desc> filter_md_;
  std::unique_ptr<dnnl::memory::desc> bias_md_;

  // input tensors
  std::unique_ptr<dnnl::memory> diff_dst_mem_;
  std::unique_ptr<dnnl::memory> diff_dst_mem_gpu_;
  std::unique_ptr<dnnl::memory::desc> diff_dst_md_;

  std::unique_ptr<dnnl::memory> src_mem_;
  std::unique_ptr<dnnl::memory> src_mem_gpu_;
  std::unique_ptr<dnnl::memory::desc> src_md_;

  std::unique_ptr<dnnl::memory> weights_mem_;
  std::unique_ptr<dnnl::memory> weights_mem_gpu_;
  std::unique_ptr<dnnl::memory::desc> weights_md_;
  //TensorShape bwd_src_shape_;

  // memory for output tensors
  //std::shared_ptr<dnnl::memory> diff_src_mem_;
  //std::shared_ptr<dnnl::memory::desc> diff_src_md_;
  //TensorShape diff_src_shape_;

  std::shared_ptr<dnnl::memory> diff_weights_mem_;
  std::unique_ptr<dnnl::memory::desc> diff_weights_md_;
  TensorShape diff_weights_shape_;
  // memory used for reorders
  std::unique_ptr<dnnl::memory> diff_weights_reorder_mem_to_;

  std::shared_ptr<dnnl::memory> diff_bias_mem_;
  std::unique_ptr<dnnl::memory::desc> diff_bias_md_;
  TensorShape diff_bias_shape_;
  // memory used for reorders
  std::unique_ptr<dnnl::memory> diff_bias_reorder_mem_to_;

  IAllocatorUniquePtr<void> src_reorder_buffer_;
  IAllocatorUniquePtr<void> dst_reorder_buffer_;

  bool gpu_available_;

 private:
  Status ComputeKernelShape(const TensorShape& weight_shape, std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_specified_) {
      kernel_shape = kernel_shape_;
      if (kernel_shape.size() + 2 != weight_shape.NumDimensions()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                               " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                               " W: ", weight_shape.ToString().c_str());
      }
      for (size_t i = 0; i < kernel_shape.size(); ++i) {
        if (kernel_shape[i] != weight_shape[i + 2]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                 " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                 " W: ", weight_shape.ToString().c_str());
        }
      }
    } else {
      auto& weight_dims = weight_shape.GetDims();
      kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }

    return Status::OK();
  }

  Status ValidateInputShape(const TensorShape& X, const TensorShape& W) const {
    const int64_t C = X[1];
    const int64_t M = W[0];

    if (X.NumDimensions() != W.NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "X num_dims does not match W num_dims.",
                             " X: ", X.ToString().c_str(),
                             " W: ", W.ToString().c_str());
    }

    if (C != W[1] * group_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input channels C is not equal to kernel channels * group.",
                             " C: ", C,
                             " kernel channels: ", W[1],
                             " group: ", group_);
    }

    if (M % group_ != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output channels M is not divisible by group.",
                             " M: ", M,
                             " group: ", group_);
    }
    return Status::OK();
  }

  template <bool ForceSymmetricAutoPadding = false>
  Status InferOutputShape(const TensorShape& input_shape,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides,
                          const std::vector<int64_t>& dilations,
                          std::vector<int64_t>* pads,
                          std::vector<int64_t>* output_shape) const {
    size_t rank = gsl::narrow_cast<int>(input_shape.NumDimensions());
    for (size_t dim = 0; dim < rank; ++dim) {
      if (dim >= strides.size() || dim >= kernel_shape.size() ||
          dim >= dilations.size() || dim >= pads->size() ||
          rank + dim >= pads->size()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Out of bound access to array");
      }
      int64_t dim_size = 0;
      ORT_RETURN_IF_ERROR(ComputePadAndOutputShape<ForceSymmetricAutoPadding>(
          input_shape[dim],
          strides[dim],
          kernel_shape[dim],
          dilations[dim],
          auto_pad_,
          &pads->at(dim),
          &pads->at(input_shape.NumDimensions() + dim),
          &dim_size));
      if (dim_size <= 0) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input shape: " + input_shape.ToString());
      }
      output_shape->push_back(dim_size);
    }
    return Status::OK();
  }

 private:
  std::unique_ptr<dnnl::stream> stream_;
  std::vector<int64_t> kernel_shape_;  // must use ComputeKernelShape(...), instead of kernel_shape_
  AutoPadType auto_pad_;
  int64_t group_;
  bool kernel_shape_specified_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> dilations_;
  std::string activation_;
  float alpha_;
};
}  // namespace ort_dnnl
}  // namespace onnxruntime

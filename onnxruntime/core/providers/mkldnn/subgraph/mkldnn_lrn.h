// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/op_kernel.h"
#include "core/providers/mkldnn/mkldnn_fwd.h"
#include "core/providers/mkldnn/mkldnn_execution_provider.h"
#include "core/providers/mkldnn/subgraph/mkldnn_kernel.h"

namespace onnxruntime {
namespace mkl_dnn {

template <typename T>
class MklDnnLrn : public MklDnnKernel {
 public:
  MklDnnLrn(MklDnnNode& node,
            MKLDNNExecutionProvider* provider,
            const NodeAttributes& attributes,
            const std::string attributes_prefix = "") : MklDnnKernel(node, provider) {
    ReadAttributes(attributes, attributes_prefix);
  }

  Status CreatePrimitives(Ort::CustomOpApi ort,
                          OrtKernelContext* context,
                          mkldnn::engine& cpu_engine,
                          std::vector<mkldnn::primitive>& net,
                          mkldnn::memory::format& source_format) {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    TensorShape x_shape;
    if (mklnode_ptr_->parent_nodes.size() == 0) {
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
      auto tensor_shape = ort.GetTensorShape(tensor_info);
      ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      auto xshape = tensor_shape.data();
      auto xdim = tensor_shape.size();

      ort_source_format_ = GetSourceFormat(static_cast<int>(xdim));
      source_format = ort_source_format_;
      x_shape = TensorShape(xshape, xdim);

      mkldnn::memory::dims src_dims_mkl(
          x_shape.GetDims().begin(), x_shape.GetDims().end());

      src_md_.reset(new mkldnn::memory::desc(
          {src_dims_mkl}, MklDnnType<T>(), source_format));
      src_mem_.reset(
          new mkldnn::memory({*src_md_, cpu_engine}, nullptr));
    } else {
      src_md_.reset(
          new mkldnn::memory::desc(parents_[0].get()->primitive_dst_mem_.get()->get_primitive_desc().desc()));
      src_mem_ = parents_[0].get()->primitive_dst_mem_;
      x_shape = parents_[0].get()->primitive_dst_shape_;
      ort_source_format_ = source_format;
    }

    primitive_dst_shape_ = TensorShape(x_shape);

    mkldnn::algorithm algo = mkldnn::algorithm::lrn_across_channels;
    fwd_desc_.reset(new mkldnn::lrn_forward::desc(
        mkldnn::prop_kind::forward_scoring, algo, *src_md_,
        size_, alpha_, beta_, bias_));

    fwd_primitive_desc_.reset(new mkldnn::lrn_forward::primitive_desc(
        *fwd_desc_, cpu_engine));

    primitive_src_format_ = static_cast<mkldnn::memory::format>(
        fwd_primitive_desc_.get()->src_primitive_desc().desc().data.format);
    primitive_dst_format_ = static_cast<mkldnn::memory::format>(
        fwd_primitive_desc_.get()->dst_primitive_desc().desc().data.format);

    if (mklnode_ptr_->output_index >= 0) {
      // last node of sub-graph. need to allocate memory for output_tensor
      if (primitive_dst_format_ != ort_source_format_) {
        // reorder neded. Use primitive output as input to reorder and
        // allocate buffer for reorder output, final output of this subgraph
        primitive_dst_mem_.reset(
            new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc()));
      } else {
        // Last node but re-order not needed. Allocate buffer to output of this node
        primitive_dst_mem_.reset(
            new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc(), nullptr));
      }
    } else {
      // Intermediate node. Use mkldnn kernel internal memory for output and
      // use this as input to next node.
      primitive_dst_mem_.reset(
          new mkldnn::memory(fwd_primitive_desc_.get()->dst_primitive_desc()));
    }

    lrn_fwd_.reset(
        new mkldnn::lrn_forward(*fwd_primitive_desc_, *src_mem_, *primitive_dst_mem_));
    net.push_back(*lrn_fwd_);

    if (mklnode_ptr_->output_index >= 0) {
      // one of the end nodes. Allocate output buffer memory and
      // reorder is necessary
      mkldnn::memory::data_type t = MklDnnType<T>();
      InitDstReorderOutput(cpu_engine, t, net);
    }

    return Status::OK();
  }

  Status Bind(Ort::CustomOpApi ort, OrtKernelContext* context) override {
    int input_index = mklnode_ptr_->input_start_index < 0 ? 0 : mklnode_ptr_->input_start_index;

    if (mklnode_ptr_->parent_nodes.size() == 0) {
      // Sub-graph's first node. Read input from input buffer
      const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
      const T* src_data = const_cast<T*>(ort.GetTensorData<T>(input_tensor));
      src_mem_->set_data_handle(static_cast<void*>(const_cast<T*>(src_data)));
    }

    if (mklnode_ptr_->output_index >= 0) {
      auto& y_dims = primitive_dst_shape_.GetDims();
      // Allocate memory for output bufffer
      OrtValue* output = ort.KernelContext_GetOutput(context, mklnode_ptr_->output_index, &y_dims[0], static_cast<int>(primitive_dst_shape_.GetDims().size()));
      T* dst_data = ort.GetTensorMutableData<T>(output);

      if (primitive_dst_format_ != ort_source_format_) {
        reorder_dst_mem_to_->set_data_handle(dst_data);
      } else {
        primitive_dst_mem_->set_data_handle(dst_data);
      }
    }

    return Status::OK();
  }

 private:
  void ReadAttributes(const NodeAttributes& attributes,
                      const std::string attributes_prefix = "") override {
    auto attr = attributes.find(attributes_prefix + "size");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT) {
      size_ = attr->second.i();
    }
    ORT_ENFORCE(size_ > 0);
    ORT_ENFORCE(size_ % 2 == 1);

    attr = attributes.find(attributes_prefix + "alpha");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      alpha_ = attr->second.f();
    }

    attr = attributes.find(attributes_prefix + "beta");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      beta_ = attr->second.f();
    }

    bias_ = 1.0f;
    attr = attributes.find(attributes_prefix + "bias");
    if (attr != attributes.end() &&
        attr->second.type() == ::ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT) {
      bias_ = attr->second.f();
    }
  }

 private:
  float alpha_ = 0;
  float beta_ = 0;
  float bias_ = 0;
  int size_ = 0;

 private:
  std::shared_ptr<mkldnn::memory> src_mem_;

  std::unique_ptr<mkldnn::lrn_forward::desc> fwd_desc_;
  std::unique_ptr<mkldnn::lrn_forward::primitive_desc> fwd_primitive_desc_;
  std::unique_ptr<mkldnn::primitive> lrn_fwd_;

  std::unique_ptr<mkldnn::memory::desc> src_md_;
};
}  // namespace mkl_dnn
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/common/logging/logging.h>
#include <core/common/safeint.h>
#include <core/framework/tensorprotoutils.h>
#include <core/providers/common.h>
#include <onnx/onnx_pb.h>

#include "helper.h"
#include "model_builder.h"
#include "op_builder.h"

namespace onnxruntime {
namespace nnapi {

using namespace android::nn::wrapper;
using std::vector;

#pragma region helpers

#define ADD_SCALAR_OPERAND(model_builder, input_indices, scalar_value)             \
  {                                                                                \
    uint32_t _index = 0;                                                           \
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromScalar(scalar_value, _index)); \
    input_indices.push_back(_index);                                               \
  }

Status AddTransposeOperator(ModelBuilder& model_builder,
                            const std::string& input,
                            const std::string& perm_name,
                            vector<int32_t> perm,
                            const std::string& output,
                            bool output_is_nhwc) ORT_MUST_USE_RESULT;
Status AddTransposeOperator(ModelBuilder& model_builder,
                            const std::string& input,
                            const std::string& perm_name,
                            vector<int32_t> perm,
                            const std::string& output,
                            bool output_is_nhwc) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  Shape perm_dimen = {SafeInt<uint32_t>(perm.size())};
  OperandType perm_operand_type(Type::TENSOR_INT32, perm_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(perm_name, perm.data(), perm_operand_type));
  uint32_t perm_idx = operand_indices.at(perm_name);

  input_indices.push_back(perm_idx);  // permutation
  ORT_RETURN_IF_ERROR(shaper.Transpose(input, perm, output));
  OperandType output_operand_type = operand_types.at(input);
  output_operand_type.SetDimensions(shaper[output]);
  return model_builder.AddOperation(ANEURALNETWORKS_TRANSPOSE, input_indices, {output},
                                    {output_operand_type}, {output_is_nhwc});
}

Status TransposeBetweenNCHWAndNHWC(ModelBuilder& model_builder,
                                   const std::string& input,
                                   const std::string& output,
                                   bool nchw_to_nhwc) ORT_MUST_USE_RESULT;
Status TransposeBetweenNCHWAndNHWC(ModelBuilder& model_builder,
                                   const std::string& input,
                                   const std::string& output,
                                   bool nchw_to_nhwc) {
  ORT_RETURN_IF_NOT(!model_builder.UseNCHW(), "model_builder.UseNCHW() is on");
  const auto& shaper(model_builder.GetShaper());
  ORT_RETURN_IF_NOT(4 == shaper[input].size(),
                    "TransposeBetweenNCHWAndNHWC input has to be a 4d tensor, actual dimensions: ", shaper[input].size());

  std::string perm_name;
  vector<int32_t> perm;
  if (nchw_to_nhwc) {
    perm_name = model_builder.GetUniqueName(input + "nchw_to_nhwc_perm");
    perm = {0, 2, 3, 1};
  } else {  // nhwc_to_nchw
    perm_name = model_builder.GetUniqueName(input + "nhwc_to_nchw_perm");
    perm = {0, 3, 1, 2};
  }

  ORT_RETURN_IF_ERROR(AddTransposeOperator(model_builder, input, perm_name, perm, output, nchw_to_nhwc));

  if (nchw_to_nhwc) {
    ORT_RETURN_IF_ERROR(model_builder.SetNCHWToNHWCOperandMap(input, output));
  } else {  // nhwc_to_nchw
    ORT_RETURN_IF_ERROR(model_builder.SetNHWCToNCHWOperandMap(input, output));
  }

  LOGS_DEFAULT(VERBOSE) << "Operand [" << input << "] with shape "
                        << Shape2String(shaper[input])
                        << " is transposed "
                        << (nchw_to_nhwc ? "nchw_to_nhwc" : "nhwc_to_nchw")
                        << " to [" << output << "] with shape "
                        << Shape2String(shaper[output]);

  return Status::OK();
}

Status TransposeNHWCToNCHW(ModelBuilder& model_builder,
                           const std::string& input,
                           const std::string& output) ORT_MUST_USE_RESULT;
Status TransposeNHWCToNCHW(ModelBuilder& model_builder,
                           const std::string& input,
                           const std::string& output) {
  return TransposeBetweenNCHWAndNHWC(model_builder, input, output, false /* nchw_to_nhwc */);
}

Status TransposeNCHWToNHWC(ModelBuilder& model_builder,
                           const std::string& input,
                           const std::string& output) ORT_MUST_USE_RESULT;
Status TransposeNCHWToNHWC(ModelBuilder& model_builder,
                           const std::string& input,
                           const std::string& output) {
  return TransposeBetweenNCHWAndNHWC(model_builder, input, output, true /* nchw_to_nhwc */);
}

// Convert the input from nchw to nhwc
Status GetNHWCInput(ModelBuilder& model_builder, const Node& node, size_t input_index, std::string& input) {
  const auto& nchw_input = node.InputDefs()[input_index]->Name();
  ORT_RETURN_IF(model_builder.IsOperandNHWC(input));
  if (!model_builder.GetNHWCOperand(nchw_input, input)) {
    input = model_builder.GetUniqueName(nchw_input + "_nchw_to_nhwc");
    ORT_RETURN_IF_ERROR(TransposeNCHWToNHWC(model_builder, nchw_input, input));
  }
  return Status::OK();
}

// Convert the input from nhwc to nchw
Status GetNCHWInput(ModelBuilder& model_builder, const Node& node, size_t input_index, std::string& input) {
  const auto& nhwc_input = node.InputDefs()[input_index]->Name();
  ORT_RETURN_IF_NOT(model_builder.IsOperandNHWC(input));
  if (!model_builder.GetNCHWOperand(nhwc_input, input)) {
    input = model_builder.GetUniqueName(nhwc_input + "_nhwc_to_nchw");
    ORT_RETURN_IF_ERROR(TransposeNHWCToNCHW(model_builder, nhwc_input, input));
  }
  return Status::OK();
}

static Status AddBinaryOperator(int32_t op_type,
                                ModelBuilder& model_builder,
                                const std::string& input1,
                                const std::string& input2,
                                int32_t fuse_code,
                                const std::string& output,
                                bool output_is_nhwc,
                                float output_scale = 0.0f,
                                int32_t output_zero_point = 0) ORT_MUST_USE_RESULT;
static Status AddBinaryOperator(int32_t op_type,
                                ModelBuilder& model_builder,
                                const std::string& input1,
                                const std::string& input2,
                                int32_t fuse_code,
                                const std::string& output,
                                bool output_is_nhwc,
                                float output_scale,
                                int32_t output_zero_point) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // input 1
  input_indices.push_back(operand_indices.at(input2));  // input 2
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);
  ORT_RETURN_IF_ERROR(shaper.Eltwise(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output],
                                        output_scale, output_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_type, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

enum DataLayout {
  L_0231 = 0,
  L_1230 = 1,
};

// TODO, replace this with more efficient code in optimizers
static Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                        const std::string& name,
                                        const OperandType& source_operand_type,
                                        DataLayout new_layout) ORT_MUST_USE_RESULT;
static Status AddInitializerInNewLayout(ModelBuilder& model_builder,
                                        const std::string& name,
                                        const OperandType& source_operand_type,
                                        DataLayout new_layout) {
  const auto& tensor = model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;
  ORT_RETURN_IF_NOT(shape.size() == 4,
                    "The initializer is not 4D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;
  std::unique_ptr<uint8_t[]> unpacked_tensor;
  size_t tensor_byte_size;

  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      src = reinterpret_cast<const uint8_t*>(GetTensorFloatData(tensor));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor, tensor_byte_size));
      src = unpacked_tensor.get();
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }

  const auto out_t = shape[0], in_t = shape[1],
             h_t = shape[2], w_t = shape[3];
  Shape dest_shape;
  if (new_layout == L_0231)
    dest_shape = {out_t, h_t, w_t, in_t};  // L_0231
  else
    dest_shape = {in_t, h_t, w_t, out_t};  // L_1230 for depthwise conv weight

  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();
  for (uint32_t out = 0; out < out_t; out++) {
    for (uint32_t in = 0; in < in_t; in++) {
      for (uint32_t h = 0; h < h_t; h++) {
        for (uint32_t w = 0; w < w_t; w++) {
          auto onnx_idx = out * in_t * h_t * w_t +
                          in * h_t * w_t +
                          h * w_t +
                          w;

          uint32_t nnapi_idx;
          if (new_layout == L_0231) {  // L_0231
            nnapi_idx = out * h_t * w_t * in_t +
                        h * w_t * in_t +
                        w * in_t +
                        in;
          } else {  // L_1230 for depthwise conv weight
            nnapi_idx = in * h_t * w_t * out_t +
                        h * w_t * out_t +
                        w * out_t +
                        out;
          }

          for (size_t i = 0; i < element_size; i++) {
            buffer[element_size * nnapi_idx + i] = src[element_size * onnx_idx + i];
          }
        }
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

// TODO, replace this with more efficient code in optimizers
static Status AddInitializerTransposed(ModelBuilder& model_builder,
                                       const OperandType& source_operand_type,
                                       const std::string& name) ORT_MUST_USE_RESULT;
static Status AddInitializerTransposed(ModelBuilder& model_builder,
                                       const OperandType& source_operand_type,
                                       const std::string& name) {
  const auto& tensor = model_builder.GetInitializerTensors().at(name);
  const Shape& shape = source_operand_type.dimensions;

  ORT_RETURN_IF_NOT(shape.size() == 2,
                    "The initializer is not 2D: ", name, " actual dim ", shape.size());

  // TODO support other data types
  const uint8_t* src = nullptr;
  std::unique_ptr<uint8_t[]> unpacked_tensor;
  size_t tensor_byte_size;
  switch (tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      src = reinterpret_cast<const uint8_t*>(GetTensorFloatData(tensor));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
      ORT_RETURN_IF_ERROR(
          onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor, tensor_byte_size));
      src = unpacked_tensor.get();
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", name,
                             " doesn't have valid type: ", tensor.data_type());
  }

  const auto x_t = shape[0], y_t = shape[1];
  Shape dest_shape = {y_t, x_t};
  OperandType operand_type = source_operand_type;
  operand_type.SetDimensions(dest_shape);
  std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[operand_type.GetOperandBlobByteSize()]);
  uint8_t* buffer = buffer_holder.get();
  size_t element_size = operand_type.GetElementByteSize();
  for (uint32_t x = 0; x < x_t; x++) {
    for (uint32_t y = 0; y < y_t; y++) {
      for (size_t i = 0; i < element_size; i++) {
        buffer[element_size * (y * x_t + x) + i] = src[element_size * (x * y_t + y) + i];
      }
    }
  }

  return model_builder.AddOperandFromPersistMemoryBuffer(name, &buffer[0], operand_type);
}

static Status ComputeConvPads(
    const Shape& input_dimen,
    const uint32_t weight_size_y, const uint32_t weight_size_x,
    const std::vector<int32_t>& onnx_pads, const std::vector<int32_t>& onnx_strides, const std::vector<int32_t>& onnx_dilations,
    AutoPadType auto_pad_type, bool nchw,
    vector<int32_t>& pads_out) ORT_MUST_USE_RESULT;
static Status ComputeConvPads(
    const Shape& input_dimen,
    const uint32_t weight_size_y, const uint32_t weight_size_x,
    const std::vector<int32_t>& onnx_pads, const std::vector<int32_t>& onnx_strides, const std::vector<int32_t>& onnx_dilations,
    AutoPadType auto_pad_type, bool nchw,
    vector<int32_t>& pads_out) {
  const int32_t input_size_y = nchw ? input_dimen[2] : input_dimen[1];
  const int32_t input_size_x = nchw ? input_dimen[3] : input_dimen[2];
  const int32_t stride_y = onnx_strides[0];
  const int32_t stride_x = onnx_strides[1];
  const int32_t dilation_y = onnx_dilations[0];
  const int32_t dilation_x = onnx_dilations[1];

  int64_t padding_top = onnx_pads[0];
  int64_t padding_bottom = onnx_pads[2];
  int64_t padding_left = onnx_pads[1];
  int64_t padding_right = onnx_pads[3];

  ORT_RETURN_IF_ERROR(ComputePad(input_size_y,
                                 stride_y, weight_size_y, dilation_y,
                                 auto_pad_type,
                                 padding_top, padding_bottom));
  ORT_RETURN_IF_ERROR(ComputePad(input_size_x,
                                 stride_x, weight_size_x, dilation_x,
                                 auto_pad_type,
                                 padding_left, padding_right));

  pads_out = {static_cast<int32_t>(padding_top), static_cast<int32_t>(padding_left),
              static_cast<int32_t>(padding_bottom), static_cast<int32_t>(padding_right)};

  return Status::OK();
}

static Status HandleAutoPad(const Shape& input_shape,
                            const uint32_t weight_size_y,
                            const uint32_t weight_size_x,
                            const vector<int32_t>& onnx_strides,
                            const vector<int32_t>& onnx_dilations,
                            AutoPadType auto_pad_type,
                            bool use_nchw,
                            vector<int32_t>& onnx_pads,
                            int32_t& nnapi_padding_code,
                            bool& use_auto_pad) ORT_MUST_USE_RESULT;
static Status HandleAutoPad(const Shape& input_shape,
                            const uint32_t weight_size_y,
                            const uint32_t weight_size_x,
                            const vector<int32_t>& onnx_strides,
                            const vector<int32_t>& onnx_dilations,
                            AutoPadType auto_pad_type,
                            bool use_nchw,
                            vector<int32_t>& onnx_pads,
                            int32_t& nnapi_padding_code,
                            bool& use_auto_pad) {
  if (auto_pad_type != AutoPadType::NOTSET) {
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        auto_pad_type, use_nchw,
                                        onnx_pads));

    if (AutoPadType::VALID == auto_pad_type || AutoPadType::SAME_UPPER == auto_pad_type) {
      use_auto_pad = true;
      nnapi_padding_code = (AutoPadType::VALID == auto_pad_type) ? ANEURALNETWORKS_PADDING_VALID
                                                                 : ANEURALNETWORKS_PADDING_SAME;
    }
  } else if (onnx_dilations == std::vector<int32_t>{1, 1}) {
    // Since NNAPI runs more efficiently using auto_pad, we try to map the NOTSET padding to auto_pad
    vector<int32_t> same_upper_pads;
    ORT_RETURN_IF_ERROR(ComputeConvPads(input_shape, weight_size_y, weight_size_x,
                                        onnx_pads, onnx_strides, onnx_dilations,
                                        AutoPadType::SAME_UPPER, use_nchw,
                                        same_upper_pads));
    if (onnx_pads == same_upper_pads) {
      use_auto_pad = true;
      nnapi_padding_code = ANEURALNETWORKS_PADDING_SAME;
    }
  }

  return Status::OK();
}

static float GetQuantizationScale(const ModelBuilder& model_builder, const Node& node, size_t idx) {
  const auto& scale_tensor = model_builder.GetInitializerTensors().at(node.InputDefs()[idx]->Name());
  return GetTensorFloatData(scale_tensor)[0];
}

static Status GetQuantizationZeroPoint(const ModelBuilder& model_builder, const Node& node, size_t idx, int32_t& zero_point)
    ORT_MUST_USE_RESULT;
static Status GetQuantizationZeroPoint(const ModelBuilder& model_builder, const Node& node, size_t idx, int32_t& zero_point) {
  std::unique_ptr<uint8_t[]> unpacked_tensor;
  size_t tensor_byte_size;
  const auto& zero_point_tensor = model_builder.GetInitializerTensors().at(node.InputDefs()[idx]->Name());
  ORT_RETURN_IF_ERROR(
      onnxruntime::utils::UnpackInitializerData(zero_point_tensor, unpacked_tensor, tensor_byte_size));
  zero_point = static_cast<int32_t>(unpacked_tensor.get()[0]);
  return Status::OK();
}

// Get scales and zero points for the qlinear binary ops (which has 2 input and 1 output)
// QLinearConv, QLinearMatmul, QLinearAdd
// a, b are inputs, and y is output
static Status GetBinaryOpQuantizationScaleAndZeroPoint(
    const ModelBuilder& model_builder, const Node& node,
    float& a_scale, float& b_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point) ORT_MUST_USE_RESULT;
static Status GetBinaryOpQuantizationScaleAndZeroPoint(
    const ModelBuilder& model_builder, const Node& node,
    float& a_scale, float& b_scale, float& y_scale,
    int32_t& a_zero_point, int32_t& b_zero_point, int32_t& y_zero_point) {
  a_scale = GetQuantizationScale(model_builder, node, 1);
  b_scale = GetQuantizationScale(model_builder, node, 4);
  y_scale = GetQuantizationScale(model_builder, node, 6);

  ORT_RETURN_IF_ERROR(GetQuantizationZeroPoint(model_builder, node, 2, a_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationZeroPoint(model_builder, node, 5, b_zero_point));
  ORT_RETURN_IF_ERROR(GetQuantizationZeroPoint(model_builder, node, 7, y_zero_point));

  return Status::OK();
}

// NNAPI has the quantization scale and zero point embedded in the ANeuralNetworksOperandType
// ONNX has the quantization scale and zero point as the inputs of the qlinear operators
// We want to verify the scale and zeropoint of the ONNX inputs matches the values embedded in the NNAPI inputs
static Status IsValidInputQuantizedType(const ModelBuilder& model_builder,
                                        const std::string& input_name,
                                        float scale,
                                        int32_t zero_point) ORT_MUST_USE_RESULT;
static Status IsValidInputQuantizedType(const ModelBuilder& model_builder,
                                        const std::string& input_name,
                                        float scale,
                                        int32_t zero_point) {
  const OperandType& input_operand_type = model_builder.GetOperandTypes().at(input_name);
  if (input_operand_type.operandType.scale != scale) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input [", input_name,
                           "] NNAPI input scale: ", input_operand_type.operandType.scale,
                           ", ONNX input scale: ", scale);
  }

  if (input_operand_type.operandType.zeroPoint != zero_point) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input [", input_name,
                           "] NNNAPI input zero point: ", input_operand_type.operandType.zeroPoint,
                           ", ONNX input zero point: ", zero_point);
  }

  return Status::OK();
}

static void AddBinaryOpQuantizationScaleAndZeroPointToSkip(ModelBuilder& model_builder, const Node& node) {
  const auto input_defs(node.InputDefs());
  model_builder.AddInitializerToSkip(input_defs[1]->Name());  // a_scale
  model_builder.AddInitializerToSkip(input_defs[2]->Name());  // a_zero_point
  model_builder.AddInitializerToSkip(input_defs[4]->Name());  // b_scale
  model_builder.AddInitializerToSkip(input_defs[5]->Name());  // b_zero_point
  model_builder.AddInitializerToSkip(input_defs[6]->Name());  // y_scale
  model_builder.AddInitializerToSkip(input_defs[7]->Name());  // y_zero_point
}

Status GetQuantizedInputScaleAndZeroPoint(const ModelBuilder& model_builder,
                                          const Node& node,
                                          const std::string& input_name,
                                          float& scale,
                                          int32_t& zero_point) ORT_MUST_USE_RESULT;
Status GetQuantizedInputScaleAndZeroPoint(const ModelBuilder& model_builder,
                                          const Node& node,
                                          const std::string& input_name,
                                          float& scale,
                                          int32_t& zero_point) {
  const auto& op_type = node.OpType();
  auto qlinear_op_type = GetQLinearOpType(node);
  assert(qlinear_op_type != QLinearOpType::Unknown &&
         qlinear_op_type != QLinearOpType::QuantizeLinear);

  size_t scale_idx, zero_point_idx;
  if (qlinear_op_type == QLinearOpType::DequantizeLinear) {
    scale_idx = 1;
    zero_point_idx = 2;
  } else if (IsQLinearBinaryOp(qlinear_op_type)) {
    const auto input_defs(node.InputDefs());
    if (input_name == input_defs[0]->Name()) {
      scale_idx = 1;
      zero_point_idx = 2;
    } else if (input_name == input_defs[3]->Name()) {
      scale_idx = 4;
      zero_point_idx = 5;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Unknown input: ", input_name, ", for op: ", op_type);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported op: ", op_type);
  }

  scale = GetQuantizationScale(model_builder, node, scale_idx);
  zero_point = 0;
  if (node.InputDefs().size() > 2) {
    ORT_RETURN_IF_ERROR(GetQuantizationZeroPoint(model_builder, node, zero_point_idx, zero_point));
  }

  return Status::OK();
}

#pragma endregion helpers

#pragma region op_base

class BaseOpBuilder : public IOpBuilder {
 public:
  virtual ~BaseOpBuilder() = default;
  virtual void AddInitializersToSkip(ModelBuilder& /* model_builder */, const Node& /* node */) override {}

  bool IsOpSupported(ModelBuilder& model_builder, const Node& node) override final;

  Status AddToModelBuilder(ModelBuilder& model_builder, const Node& node) override final ORT_MUST_USE_RESULT;

 protected:
  virtual bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node);

  virtual int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */,
                                        const Node& /* node */) const { return 27; }

  virtual bool HasSupportedInputs(const Node& node);

  virtual Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) ORT_MUST_USE_RESULT = 0;

  bool HasExternalInitializer(ModelBuilder& model_builder, const Node& node);

  virtual int GetMinSupportedOpSet(const Node& /* node */) { return 1; }
  virtual int GetMaxSupportedOpSet(const Node& /* node */) { return 13; }

  bool HasSupportedOpSet(const Node& node);
};

bool BaseOpBuilder::IsOpSupported(ModelBuilder& model_builder, const Node& node) {
#ifdef __ANDROID__
  int32_t android_sdk_ver = model_builder.GetAndroidSdkVer();
  int32_t required_sdk_ver = GetMinSupportedSdkVer(model_builder, node);
  if (required_sdk_ver > android_sdk_ver) {
    LOGS_DEFAULT(VERBOSE) << "Current Android API level [" << android_sdk_ver
                          << "], Operator [" << node.OpType()
                          << "] is only supported on API >" << required_sdk_ver;
    return false;
  }
#endif

  if (!HasSupportedInputs(node))
    return false;

  // We do not support external initializers for now
  if (HasExternalInitializer(model_builder, node))
    return false;

  if (!HasSupportedOpSet(node))
    return false;

  return IsOpSupportedImpl(model_builder, node);
}

bool BaseOpBuilder::HasSupportedInputs(const Node& node) {
  // We only check the type of input 0 by default
  // specific op builder can override this
  const auto& input = *node.InputDefs()[0];

  if (nullptr == input.Shape()) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input shape is null";
    return false;
  }

  int32_t input_type;
  if (!GetType(input, input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

bool BaseOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& /* node */) {
  return true;
}

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node) {
  ORT_RETURN_IF_NOT(model_builder.IsNodeSupported(node), "Unsupported operator ", node.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node));
  LOGS_DEFAULT(VERBOSE) << "Operator name: [" << node.Name()
                        << "] type: [" << node.OpType() << "] was added";
  return Status::OK();
}

bool BaseOpBuilder::HasExternalInitializer(ModelBuilder& model_builder, const Node& node) {
  const auto& initializers(model_builder.GetOnnxGraph().GetAllInitializedTensors());
  for (const auto* node_arg : node.InputDefs()) {
    const auto& input_name(node_arg->Name());
    if (!Contains(initializers, input_name))
      continue;

    const auto* tensor = initializers.at(input_name);
    if (tensor->has_data_location() &&
        tensor->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      LOGS_DEFAULT(VERBOSE) << "Initializer [" << input_name
                            << "] with external data location are not currently supported";
      return true;
    }
  }

  return false;
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node) {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS_DEFAULT(VERBOSE) << node.OpType() << "is only supported for opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

#pragma endregion op_base

#pragma region op_binary

class BinaryOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder, const Node& node) const override;
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;
  bool HasSupportedInputs(const Node& node) override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
  int GetMinSupportedOpSet(const Node& node) override;
};

void BinaryOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  const auto& op = node.OpType();
  if (op == "QLinearAdd") {
    AddBinaryOpQuantizationScaleAndZeroPointToSkip(model_builder, node);
  }
}

int32_t BinaryOpBuilder::GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& node) const {
  const auto& op(node.OpType());
  if (op == "Sub" || op == "Div") {
    return 28;
  }

  return 27;
}

int BinaryOpBuilder::GetMinSupportedOpSet(const Node& node) {
  const auto& op(node.OpType());

  // Add/Sub/Mul/Div opset 6- has broadcast attributes we do not support now
  if (op != "QLinearAdd")
    return 7;

  return 1;
}

bool BinaryOpBuilder::HasSupportedInputs(const Node& node) {
  if (node.OpType() != "QLinearAdd")
    return BaseOpBuilder::HasSupportedInputs(node);

  // QLinearAdd
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

bool BinaryOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  const auto& op_type(node.OpType());
  const auto input_defs(node.InputDefs());
  bool op_is_qlinear = op_type == "QLinearAdd";
  size_t a_idx = 0, b_idx = 1;
  if (op_is_qlinear) {
    b_idx = 3;
  }
  Shape input1_shape, input2_shape;
  if (!GetShape(*input_defs[a_idx], input1_shape) ||
      !GetShape(*input_defs[b_idx], input2_shape))
    return false;

  const auto input1_size = input1_shape.size();
  const auto input2_size = input2_shape.size();
  if (input1_size > 4 || input2_size > 4) {
    LOGS_DEFAULT(VERBOSE) << node.OpType() << " only support up to 4d shape, input1 is "
                          << input1_size << "d shape, input 2 is "
                          << input2_size << "d shape";
    return false;
  }

  if (op_is_qlinear) {
    // For QLinearAdd, we only support uint8 output now
    int32_t output_type;
    if (!GetType(*node.OutputDefs()[0], output_type))
      return false;

    if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      LOGS_DEFAULT(VERBOSE) << "[" << op_type
                            << "] output type: [" << output_type
                            << "] is not supported for now";
      return false;
    }

    // All scale/zero points are initializer scalars
    // a/b/y_scale
    if (!HasValidQuantizationScale(model_builder.GetInitializerTensors(), node, {1, 4, 6}))
      return false;

    // a/b/y_zero_point
    if (!HasValidQuantizationZeroPoint(model_builder.GetInitializerTensors(), node, {2, 5, 7}))
      return false;
  }

  return true;
}

Status BinaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  const auto& op_type(node.OpType());
  const auto input_defs(node.InputDefs());

  int32_t op_code;
  bool op_is_qlinear = op_type == "QLinearAdd";
  if (op_type == "Add" || op_is_qlinear)
    op_code = ANEURALNETWORKS_ADD;
  else if (op_type == "Sub")
    op_code = ANEURALNETWORKS_SUB;
  else if (op_type == "Mul")
    op_code = ANEURALNETWORKS_MUL;
  else if (op_type == "Div")
    op_code = ANEURALNETWORKS_DIV;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }

  size_t a_idx = 0, b_idx = 1;
  if (op_is_qlinear) {
    b_idx = 3;
  }

  std::string input1 = input_defs[a_idx]->Name();
  std::string input2 = input_defs[b_idx]->Name();
  const auto& output = node.OutputDefs()[0]->Name();

  bool input1_is_nhwc = model_builder.IsOperandNHWC(input1);
  bool input2_is_nhwc = model_builder.IsOperandNHWC(input2);
  bool output_is_nhwc = false;

  if (input1_is_nhwc == input2_is_nhwc) {
    output_is_nhwc = input1_is_nhwc;
  } else if (input1_is_nhwc) {
    // need transpose input1 back to nchw
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, a_idx, input1));
  } else {  // input2_is_nhwc
    // need transpose input2 back to nchw
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, b_idx, input2));
  }

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  if (op_is_qlinear) {
    ORT_RETURN_IF_ERROR(GetBinaryOpQuantizationScaleAndZeroPoint(model_builder, node,
                                                                 a_scale, b_scale, y_scale,
                                                                 a_zero_point, b_zero_point, y_zero_point));
  }

  // Verify if the scale and zero point matchs from onnx input and nnapi input
  if (op_is_qlinear) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  int32_t fuse_code = model_builder.FindActivation(node, *node.OutputDefs()[0]);
  return AddBinaryOperator(op_code, model_builder,
                           input1, input2, fuse_code,
                           output, output_is_nhwc, y_scale, y_zero_point);
}

#pragma endregion

#pragma region op_relu

class ReluOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status ReluOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  // skip this relu if it is some op's fuse output
  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Relu Node [" << node.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, output_is_nhwc);
  } else {
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_RELU, input_indices,
                                                   {output}, {output_operand_type}, {output_is_nhwc}));
  }

  return Status::OK();
}

#pragma endregion op_relu

#pragma region op_transpose

class TransposeOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 28;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool TransposeOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Transpose only supports 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

Status TransposeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());

  auto input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  NodeAttrHelper helper(node);
  vector<int32_t> perm = helper.Get("perm", vector<int32_t>());
  auto input_dims = shaper[input].size();
  if (perm.empty()) {
    for (int32_t i = input_dims - 1; i >= 0; i--)
      perm.push_back(i);
  } else {
    ORT_RETURN_IF_NOT(perm.size() == input_dims, "Perm and input should have same dimension");
  }

  if (model_builder.IsOperandNHWC(input)) {
    ORT_RETURN_IF_NOT(input_dims == 4, "Only 4D shape can be nhwc");

    // we are using nhwc here, but the axis is in nchw, need to transpose axis from nchw to nhwc
    const int32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    for (size_t i = 0; i < perm.size(); i++)
      perm[i] = axis_nchw_to_nhwc[perm[i]];
  }

  std::string perm_name = model_builder.GetUniqueName(node.Name() + input + "perm");

  // It is possible this onnx transpose operator can be nchw->nhwc, but so far I don't see
  // any scenario will do this since onnx is nchw only, assume the output is always not nhwc
  // even it is, there will be extra transpose in the onnx model to convert it back to nchw
  // before conv/pool/... operators
  ORT_RETURN_IF_ERROR(AddTransposeOperator(model_builder, input, perm_name, perm, output, false /* is_nhwc */));

  return Status::OK();
}

#pragma endregion op_transpose

#pragma region op_reshape

class ReshapeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;
  static Status AddReshapeOperator(ModelBuilder& model_builder, const Node& node,
                                   const std::string& input, const std::vector<int32_t>& shape) ORT_MUST_USE_RESULT;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
  static bool CanSkipReshape(const Node& node, size_t input_rank, size_t output_rank);

  // Reshape opset 4- uses attributes for new shape which we do not support for now
  int GetMinSupportedOpSet(const Node& /* node */) override { return 5; }
};

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

// We can skip the Reshape if all the output edges satisfies,
// 1. The output of the reshape/flatten is the input 0 of the GEMM/Matmul,
//    and the input rank >= 2 and output_rank == 2
//    This is because Gemm/Matmul will map to ANEURALNETWORKS_FULLY_CONNECTED in NNAPI,
//    ANEURALNETWORKS_FULLY_CONNECTED will flatten the 2+ dim input 0 to 2d
// 2. Or the output the reshape/flatten is the output of the graph
//    (no op in the graph is using the output except can be used by Gemm/Matmul satisfying condition 1 above)
// The reason we want to skip Reshape is that Reshape is not running on Hardware (NPU,...) in NNAPI for
// some CPU (e.g. Qualcomm SD for now), skipping unnecessary Reshape will prevent context switching
// between NNAPI CPU impl and Hardware Accelerator impl and will speed up the execution
// If we are going to skip the reshape, we will still add correct shape and operand type for the output in
// onnxruntime::nnapi::Model.
// If the Reshape output is also a graph output, since NNAPI output is a void* buffer, we can find the shape
// information in onnxruntime::nnapi::Model and pass the correct shape information back to ORT to be used as output shape
/* static */ bool ReshapeOpBuilder::CanSkipReshape(const Node& node, size_t input_rank, size_t output_rank) {
  const auto& output = node.OutputDefs()[0]->Name();
  // We will go through all the output edges
  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    const auto& op_type = it->GetNode().OpType();
    // TODO add quantized matmul when reshape support quantized input
    if (op_type != "Gemm" && op_type != "MatMul") {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when the output is Gemm/Matmul"
                            << " or no op is using the output (output is graph output)"
                            << ", output name, " << output
                            << " is used by " << op_type;
      return false;
    }

    // NNAPI ANEURALNETWORKS_FULLY_CONNECTED will only flatten the input 0
    if (it->GetDstArgIndex() != 0) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when the output is input 0 of Gemm/Matmul"
                            << ", output name, " << output;
      return false;
    }

    // We only support 2d matmul/gemm here
    // And NNAPI ANEURALNETWORKS_FULLY_CONNECTED will only flatten input rank >= 2
    if (input_rank < 2 || output_rank != 2) {
      LOGS_DEFAULT(VERBOSE) << "Reshape/Flatten can only be skipped when input_rank >= 2 and output_rank == 2"
                            << ", output name, " << output
                            << ", the actual input_rank, " << input_rank
                            << ", the actual output_rank, " << output_rank;
      return false;
    }
  }

  // If we reach here, we have either,
  // all the Reshape outputs are used by gemm/matmul, the output can also be a model output [doesn't really matter here]
  // or
  // Reshape has no output edge ==> the output is a graph output or a dead end [which we don't care]
  // we can skip this Reshape now
  LOGS_DEFAULT(VERBOSE) << "Skipping Reshape/Flatten node ["
                        << node.Name() << "] with output, " << output;
  return true;
}

/* static */ Status ReshapeOpBuilder::AddReshapeOperator(ModelBuilder& model_builder,
                                                         const Node& node,
                                                         const std::string& input,
                                                         const std::vector<int32_t>& shape) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& output = node.OutputDefs()[0]->Name();
  ORT_RETURN_IF_ERROR(shaper.Reshape(input, shape, output));
  auto input_rank = shaper[input].size();
  auto output_rank = shaper[output].size();

  // Since Reshape is not running using hardware in NNAPI for some CPU (e.g. Qualcomm SD for now)
  // We will try to see if we the skip the Reshape to prevent context switching between
  // NNAPI CPU impl and NNAPI hardware accelerator impl
  if (CanSkipReshape(node, input_rank, output_rank)) {
    // Since reshape can be skipped, only register the dimension and type, with same index and new name
    const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, false);
  } else {
    // We still need to perform a reshape here
    // Add input
    std::vector<uint32_t> input_indices;
    input_indices.push_back(operand_indices.at(input));
    // Add new shape
    Shape shape_dimen = {static_cast<uint32_t>(shape.size())};
    std::string shape_name = model_builder.GetUniqueName(node.Name() + input + "newshape");
    OperandType shape_operand_type(Type::TENSOR_INT32, shape_dimen);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(shape_name, shape.data(), shape_operand_type));
    input_indices.push_back(operand_indices.at(shape_name));

    const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
    ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_RESHAPE, input_indices, {output}, {output_operand_type}, {false}));
  }

  return Status::OK();
}

bool ReshapeOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& perm_name = node.InputDefs()[1]->Name();
  if (!Contains(initializers, perm_name)) {
    LOGS_DEFAULT(VERBOSE) << "New shape of reshape must be known";
    return false;
  }

  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Reshape only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  const auto& shape_tensor = initializers.at(perm_name);
  const int64_t* rawShape = GetTensorInt64Data(shape_tensor);
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  for (uint32_t i = 0; i < size; i++) {
    // NNAPI reshape does not support 0 as dimension
    if (rawShape[i] == 0 && i < input_shape.size() && input_shape[i] == 0) {
      LOGS_DEFAULT(VERBOSE) << "Reshape doesn't suppport 0 reshape dimension on a dynamic dimension";
      return false;
    }
  }

  return true;
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& initializers(model_builder.GetInitializerTensors());

  auto input = node.InputDefs()[0]->Name();
  if (model_builder.IsOperandNHWC(input)) {
    // We want to transpose nhwc operand back to nchw before reshape
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, 0, input));
  }

  const auto& shape_tensor = initializers.at(node.InputDefs()[1]->Name());
  const int64_t* rawShape = GetTensorInt64Data(shape_tensor);
  const auto size = SafeInt<uint32_t>(shape_tensor.dims()[0]);

  Shape input_shape = shaper[input];
  std::vector<int32_t> shape(size);
  for (uint32_t i = 0; i < size; i++) {
    int32_t dim = SafeInt<int32_t>(rawShape[i]);
    // NNAPI reshape does not support 0 as dimension
    shape[i] = dim == 0 ? input_shape[i] : dim;
  }

  return AddReshapeOperator(model_builder, node, input, shape);
}

#pragma endregion op_reshape

#pragma region op_batchnormalization

class BatchNormalizationOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const Node& /* node */) override { return 7; }
};

void BatchNormalizationOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  // skip everything except input0 for BatchNormalization
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // scale
  model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());  // B
  model_builder.AddInitializerToSkip(node.InputDefs()[3]->Name());  // mean
  model_builder.AddInitializerToSkip(node.InputDefs()[4]->Name());  //var
}

bool BatchNormalizationOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  if (node.OutputDefs().size() != 1) {
    LOGS_DEFAULT(VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  Shape input_shape;
  if (!GetShape(*input_defs[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4) {
    LOGS_DEFAULT(VERBOSE) << "BN only support up to 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS_DEFAULT(VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& scale_name = input_defs[1]->Name();
  const auto& b_name = input_defs[2]->Name();
  const auto& mean_name = input_defs[3]->Name();
  const auto& var_name = input_defs[4]->Name();
  if (!Contains(initializers, scale_name)) {
    LOGS_DEFAULT(VERBOSE) << "Scale of BN must be known";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS_DEFAULT(VERBOSE) << "B of BN must be known";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS_DEFAULT(VERBOSE) << "Mean of BN must be known";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS_DEFAULT(VERBOSE) << "Var of BN must be known";
    return false;
  }

  return true;
}

Status BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);

  // For reshape we are not really doing anything but
  // register a new operand with new shape
  const auto& input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();

  const auto& scale_tensor = initializers.at(node.InputDefs()[1]->Name());
  const auto& bias_tensor = initializers.at(node.InputDefs()[2]->Name());
  const auto& mean_tensor = initializers.at(node.InputDefs()[3]->Name());
  const auto& var_tensor = initializers.at(node.InputDefs()[4]->Name());
  const auto eps = helper.Get("epsilon", 1e-5f);

  const auto size = SafeInt<uint32_t>(scale_tensor.dims()[0]);
  vector<float> a, b;
  a.reserve(size);
  b.reserve(size);

  const float* scale_data = GetTensorFloatData(scale_tensor);
  const float* bias_data = GetTensorFloatData(bias_tensor);
  const float* mean_data = GetTensorFloatData(mean_tensor);
  const float* var_data = GetTensorFloatData(var_tensor);

  for (int64_t i = 0; i < size; i++) {
    a.push_back(scale_data[i] / sqrt(var_data[i] + eps));
    b.push_back((scale_data[i] * -mean_data[i]) / sqrt(var_data[i] + eps) +
                bias_data[i]);
  }

  const auto tensor_a_name = model_builder.GetUniqueName(node.Name() + input + "_imm_a");
  const auto tensor_b_name = model_builder.GetUniqueName(node.Name() + input + "_imm_b");
  const auto tensor_imm_product_name = model_builder.GetUniqueName(node.Name() + input + "_imm_mul");
  Shape tensor_a_dimen = {size};

  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = input_is_nhwc;

  if (!input_is_nhwc) {
    // the batch normalization is applied on C channel,
    // if the input is NC[HW], will need correct shape for tensor_a/b
    // to make sure we are broadcasting on the correct channel,
    // input shape {N, C}       ==> tensor_a/b's shape {size}
    // input shape {N, C, H}    ==> tensor_a/b's shape {size, 1}
    // input shape {N, C, H, W} ==> tensor_a/b's shape {size, 1, 1}
    const auto input_rank = shaper[input].size();
    for (size_t i = 2; i < input_rank; i++)
      tensor_a_dimen.push_back(1);
  }

  shaper.AddShape(tensor_a_name, tensor_a_dimen);
  shaper.AddShape(tensor_b_name, tensor_a_dimen);
  const OperandType a_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_a_name, a.data(), a_operand_type));
  const OperandType b_operand_type(operand_types.at(input).type, tensor_a_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(tensor_b_name, b.data(), b_operand_type));

  // Mul
  ORT_RETURN_IF_ERROR(AddBinaryOperator(ANEURALNETWORKS_MUL,
                                        model_builder,
                                        input, tensor_a_name,
                                        ANEURALNETWORKS_FUSED_NONE,
                                        tensor_imm_product_name,
                                        output_is_nhwc));

  // Add
  int32_t fuse_code = model_builder.FindActivation(node, *node.OutputDefs()[0]);
  ORT_RETURN_IF_ERROR(AddBinaryOperator(ANEURALNETWORKS_ADD,
                                        model_builder,
                                        tensor_imm_product_name, tensor_b_name,
                                        fuse_code,
                                        output,
                                        output_is_nhwc));

  return Status::OK();
}

#pragma endregion op_batchnormalization

#pragma region op_pool

class PoolOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder, const Node& /* node */) const override {
    return model_builder.UseNCHW() ? 29 : 28;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool PoolOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  const auto& op_type = node.OpType();
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE)
        << op_type << " only supports rank-4 tensor, input ["
        << node.InputDefs()[0]->Name() << "] has actual dim count " << input_size;
    return false;
  }

  if (op_type == "AveragePool" || op_type == "MaxPool") {
    NodeAttrHelper helper(node);

    const auto count_include_pad = helper.Get("count_include_pad", 0);
    if (count_include_pad == 1) {
      LOGS_DEFAULT(VERBOSE) << "count_include_pad == 1 is not supported";
      return false;
    }

    const auto storage_order = helper.Get("storage_order", 0);
    if (storage_order == 1) {
      LOGS_DEFAULT(VERBOSE) << "storage_order == 1 is not supported";
      return false;
    }

    if (helper.Get("kernel_shape", std::vector<int32_t>{1, 1}).size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "Only pooling 2d is supported";
      return false;
    }

    if (helper.Get("ceil_mode", 0) == 1) {
      LOGS_DEFAULT(VERBOSE) << "ceil_mode == 1 is not supported for pooling";
      return false;
    }

    if (helper.Get("dilations", std::vector<int32_t>{1, 1}) !=
        std::vector<int32_t>{1, 1}) {
      LOGS_DEFAULT(VERBOSE) << "Dilations of pooling is not supported";
      return false;
    }

    if (node.OutputDefs().size() != 1) {
      LOGS_DEFAULT(VERBOSE) << "Argmax in maxpooling is not supported";
      return false;
    }
  } else if (op_type != "GlobalAveragePool" && op_type != "GlobalMaxPool") {
    LOGS_DEFAULT(VERBOSE) << "PoolOpBuilder, unknown op: " << op_type;
    return false;
  }

  return true;
}

Status PoolOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  NodeAttrHelper helper(node);

  auto input = node.InputDefs()[0]->Name();
  bool use_nchw = model_builder.UseNCHW();
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = false;
  if (use_nchw) {
    ORT_RETURN_IF_NOT(!input_is_nhwc, "model_builder.UseNCHW() but input is NHWC");
  } else {
    output_is_nhwc = true;
    if (!input_is_nhwc) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node, 0, input));
    }
  }

  const auto& output = node.OutputDefs()[0]->Name();
  const auto& op_type = node.OpType();

  int32_t op_code;
  bool is_average_pool = op_type == "AveragePool";
  if (is_average_pool || op_type == "GlobalAveragePool")
    op_code = ANEURALNETWORKS_AVERAGE_POOL_2D;
  else  // (op_type == "MaxPool" || op_type == "GlobalMaxPool")
    op_code = ANEURALNETWORKS_MAX_POOL_2D;

  vector<int32_t> onnx_pads, onnx_strides, kernel_shape;
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
  const auto& input_shape = shaper[input];
  if (is_average_pool || op_type == "MaxPool") {
    const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
    kernel_shape = helper.Get("kernel_shape", vector<int32_t>{0, 0});
    onnx_strides = helper.Get("strides", vector<int>{1, 1});
    onnx_pads = helper.Get("pads", vector<int>{0, 0, 0, 0});
    const auto weight_size_y = static_cast<uint32_t>(kernel_shape[0]);
    const auto weight_size_x = static_cast<uint32_t>(kernel_shape[1]);
    ORT_RETURN_IF_ERROR(
        HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                      onnx_strides, {1, 1} /* onnx_dilations */,
                      auto_pad_type, use_nchw,
                      onnx_pads, nnapi_padding_code, use_auto_pad));
  } else {  // (op_type == "GlobalAveragePool" || op_type == "GlobalMaxPool")
    use_auto_pad = true;
    nnapi_padding_code = ANEURALNETWORKS_PADDING_VALID;
    onnx_strides = vector<int32_t>{1, 1};
    onnx_pads = vector<int32_t>{0, 0, 0, 0};
    if (use_nchw) {
      kernel_shape = vector<int32_t>{static_cast<int32_t>(input_shape[2]),
                                     static_cast<int32_t>(input_shape[3])};
    } else {
      kernel_shape = vector<int32_t>{static_cast<int32_t>(input_shape[1]),
                                     static_cast<int32_t>(input_shape[2])};
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node, *node.OutputDefs()[0]);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));

  if (use_auto_pad) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, nnapi_padding_code);
  } else {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, kernel_shape[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  if (model_builder.GetAndroidSdkVer() > 28) {  // nchw only supported on api 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  ORT_RETURN_IF_ERROR(shaper.Pool(input,
                                  onnx_pads, onnx_strides, kernel_shape,
                                  use_nchw,
                                  output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion op_pool

#pragma region op_conv

class ConvOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder, const Node& /* node */) const override {
    return model_builder.UseNCHW() ? 29 : 28;
  }

  bool HasSupportedInputs(const Node& node) override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool ConvOpBuilder::HasSupportedInputs(const Node& node) {
  if (node.OpType() != "QLinearConv")
    return BaseOpBuilder::HasSupportedInputs(node);

  // QLinearConv only supports input of uint8 for now
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

void ConvOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  const auto& op = node.OpType();
  const auto input_defs = node.InputDefs();

  // skip the weight for conv as we need to transpose
  if (op == "QLinearConv") {
    AddBinaryOpQuantizationScaleAndZeroPointToSkip(model_builder, node);
    model_builder.AddInitializerToSkip(input_defs[3]->Name());  // w
    if (input_defs.size() > 8)
      model_builder.AddInitializerToSkip(input_defs[8]->Name());  // B
  } else {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());  // w
  }
}

bool ConvOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  const auto& op_type = node.OpType();
  const auto input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);

  bool is_qlinear_conv = (op_type == "QLinearConv");
  size_t w_idx = is_qlinear_conv ? 3 : 1;
  const auto group = helper.Get("group", 1);
  const auto weight_name = input_defs[w_idx]->Name();
  if (Contains(initializers, weight_name)) {
    const auto& tensor = initializers.at(weight_name);
    if (tensor.dims().size() != 4) {
      LOGS_DEFAULT(VERBOSE) << "Only conv 2d is supported.";
      return false;
    }

    const auto onnx_dilations = helper.Get("dilations", vector<int>{1, 1});
    if (onnx_dilations != vector<int>{1, 1}) {
      if (group != 1 && tensor.dims()[1] != 1) {
        LOGS_DEFAULT(VERBOSE) << "dilation is not supported on grouped conv";
        return false;
      }

      const auto android_sdk_ver = model_builder.GetAndroidSdkVer();
      if (android_sdk_ver < 29) {
        LOGS_DEFAULT(VERBOSE) << op_type << " dilations is only supported on Android API level 29+, "
                              << "actual API level: " << android_sdk_ver;
        return false;
      }
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "The weight of convolution must be known";
    return false;
  }

  if (is_qlinear_conv) {
    // For QLinearConv, we only support uint8 output now
    int32_t output_type;
    if (!GetType(*node.OutputDefs()[0], output_type))
      return false;

    if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
      LOGS_DEFAULT(VERBOSE) << "[" << op_type
                            << "] output type: [" << output_type
                            << "] is not supported for now";
      return false;
    }

    if (input_defs.size() > 8 && !Contains(initializers, input_defs[8]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Bias of QLinearConv must be known";
      return false;
    }

    // a/b/y_scale
    if (!HasValidQuantizationScale(model_builder.GetInitializerTensors(), node, {1, 4, 6}))
      return false;

    // a/b/y_zero_point
    if (!HasValidQuantizationZeroPoint(model_builder.GetInitializerTensors(), node, {2, 5, 7}))
      return false;
  }

  return true;
}

Status ConvOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);
  const auto input_defs = node.InputDefs();
  const auto& op_type = node.OpType();
  bool is_qlinear_conv = (op_type == "QLinearConv");

  // onnx strides are in the order height, width
  // while nnapi strides are in the order width, height
  const auto onnx_strides = helper.Get("strides", vector<int>{1, 1});

  // onnx pads are in the order top, left, bottom, right
  // while nnapi pads is in the order left, right, top, bottom
  auto onnx_pads = helper.Get("pads", vector<int>{0, 0, 0, 0});

  // onnx dilations is in the order height, width
  // while nnapi dilations are in the order width, height
  const auto onnx_dilations = helper.Get("dilations", vector<int>{1, 1});
  const auto group = helper.Get("group", 1);

  size_t x_idx = 0,
         w_idx = is_qlinear_conv ? 3 : 1,
         b_idx = is_qlinear_conv ? 8 : 2;

  auto input = input_defs[x_idx]->Name();
  bool use_nchw = model_builder.UseNCHW();
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = false;
  if (use_nchw) {
    ORT_RETURN_IF_NOT(!input_is_nhwc, "model_builder.UseNCHW() but input is NHWC");
  } else {
    output_is_nhwc = true;
    if (!input_is_nhwc) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node, x_idx, input));
    }
  }

  float x_scale = 0.0f,
        w_scale = 0.0f,
        y_scale = 0.0f;
  int32_t x_zero_point = 0,
          w_zero_point = 0,
          y_zero_point = 0;

  if (is_qlinear_conv) {
    ORT_RETURN_IF_ERROR(GetBinaryOpQuantizationScaleAndZeroPoint(model_builder, node,
                                                                 x_scale, w_scale, y_scale,
                                                                 x_zero_point, w_zero_point, y_zero_point));
  }

  const auto& weight = input_defs[w_idx]->Name();
  const auto& weight_tensor = initializers.at(weight);
  bool conv_2d = false,
       depthwise_conv_2d = false,
       grouped_conv_2d = false;

  // For ONNX we only have 1 conv ops
  // For NNAPI we have 3
  // Input is (N, C, H, W)
  // group == 1,                                   --> regular conv
  // group != 1 && weight is (M, 1, kH, kW),       --> depthwise conv
  // group != 1 && weight is (M, C/group, kH, kW), --> grouped conv
  if (group == 1)
    conv_2d = true;
  else if ((weight_tensor.dims()[1] == 1))
    depthwise_conv_2d = true;
  else
    grouped_conv_2d = true;

  Shape onnx_weight_shape;
  for (auto dim : weight_tensor.dims())
    onnx_weight_shape.push_back(SafeInt<uint32_t>(dim));

  Type onnx_weight_type;
  switch (weight_tensor.data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      onnx_weight_type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      onnx_weight_type = Type::TENSOR_QUANT8_ASYMM;
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The initializer of graph ", weight, " doesn't have valid type: ", weight_tensor.data_type());
  }

  OperandType onnx_weight_operand_type(onnx_weight_type, onnx_weight_shape, w_scale, w_zero_point);

  // Pre-process weights
  if (conv_2d || grouped_conv_2d) {
    ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight, onnx_weight_operand_type, L_0231));
  } else {  // depthwise_conv_2d
    ORT_RETURN_IF_ERROR(AddInitializerInNewLayout(model_builder, weight, onnx_weight_operand_type, L_1230));
  }

  if (is_qlinear_conv) {
    // Verify if the scale and zero point matchs from onnx input/weight and nnapi input/weight
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, x_scale, x_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, weight, w_scale, w_zero_point));
  }

  bool hasBias = (input_defs.size() > b_idx);
  std::string bias = hasBias ? input_defs[b_idx]->Name() : weight + "_bias";
  if (!hasBias) {
    const auto weight_dimen = shaper[weight];
    Shape bias_dimen;
    if (conv_2d || grouped_conv_2d)
      bias_dimen = {weight_dimen[0]};
    else
      bias_dimen = {weight_dimen[3]};

    const auto& weight_type = operand_types.at(weight).type;
    if (weight_type == Type::TENSOR_FLOAT32) {
      vector<float> buffer(bias_dimen[0], 0.0f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen, x_scale * w_scale);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (weight_type == Type::TENSOR_QUANT8_ASYMM) {
      vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, x_scale * w_scale);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown weight type ", TypeToStr(weight_type));
    }
  } else if (is_qlinear_conv) {  // QLinearConv's bias type need special handling
    const auto& bias_tensor = model_builder.GetInitializerTensors().at(bias);
    ORT_RETURN_IF_NOT(bias_tensor.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32,
                      "bias of QLinearConv should be int32, actual type: ", bias_tensor.data_type());
    Shape bias_dimen;
    for (auto dim : bias_tensor.dims())
      bias_dimen.push_back(SafeInt<uint32_t>(dim));

    const void* buffer = GetTensorInt32Data(bias_tensor);
    OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, x_scale * w_scale);
    ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer, bias_operand_type));
  }

  const auto auto_pad_type = StringToAutoPadType(helper.Get("auto_pad", "NOTSET"));
  bool use_auto_pad = false;
  int32_t nnapi_padding_code = ANEURALNETWORKS_PADDING_SAME;
  const auto& input_shape = shaper[input];
  const auto& kernel_shape = shaper[weight];
  const auto weight_size_y = kernel_shape[1];
  const auto weight_size_x = kernel_shape[2];
  ORT_RETURN_IF_ERROR(
      HandleAutoPad(input_shape, weight_size_y, weight_size_x,
                    onnx_strides, onnx_dilations,
                    auto_pad_type, use_nchw,
                    onnx_pads, nnapi_padding_code, use_auto_pad));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  input_indices.push_back(operand_indices.at(weight));
  input_indices.push_back(operand_indices.at(bias));

  if (use_auto_pad) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, nnapi_padding_code);
  } else {
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[1]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[3]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[0]);
    ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_pads[2]);
  }

  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[1]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_strides[0]);

  if (!conv_2d) {
    if (depthwise_conv_2d) {
      int32_t depthwiseMultiplier = shaper[weight][3] / group;
      ADD_SCALAR_OPERAND(model_builder, input_indices, depthwiseMultiplier);
    } else {  // grouped_conv_2d
      ADD_SCALAR_OPERAND(model_builder, input_indices, group);
    }
  }

  int32_t fuse_code = model_builder.FindActivation(node, *node.OutputDefs()[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  if (model_builder.GetAndroidSdkVer() > 28) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);

    // 1. NNAPI Grouped Conv does not support dilations
    // 2. There is a bug in NNAPI (not sure NNAPI itself or Qualcomm Hexagon driver),
    //    setting dilation (even it is the default (1,1)) will make the execution fall back to CPU
    //    so if dilations == (1,1) we simply ignore it
    if (!grouped_conv_2d &&
        (onnx_dilations[1] != 1 || onnx_dilations[0] != 1)) {
      ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_dilations[1]);
      ADD_SCALAR_OPERAND(model_builder, input_indices, onnx_dilations[0]);
    }
  }

  int32_t operationCode;
  const auto& output = node.OutputDefs()[0]->Name();
  if (conv_2d || grouped_conv_2d) {
    operationCode = conv_2d ? ANEURALNETWORKS_CONV_2D
                            : ANEURALNETWORKS_GROUPED_CONV_2D;
    ORT_RETURN_IF_ERROR(shaper.Conv(input, weight,
                                    onnx_pads, onnx_strides, onnx_dilations,
                                    use_nchw,
                                    output));
  } else {  // depthwise_conv_2d
    operationCode = ANEURALNETWORKS_DEPTHWISE_CONV_2D;
    ORT_RETURN_IF_ERROR(shaper.DepthwiseConv(input, weight,
                                             onnx_pads, onnx_strides, onnx_dilations,
                                             use_nchw,
                                             output));
  }

  const OperandType output_operand_type(operand_types.at(input).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion op_conv

#pragma region op_cast

class CastOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 29;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;

  // Cast opset 5- uses string attribute for to type, is not supported for now
  int GetMinSupportedOpSet(const Node& /* node */) override { return 6; }
};

bool CastOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  NodeAttrHelper helper(node);
  const auto to = helper.Get("to", 0);
  if (to != ONNX_NAMESPACE::TensorProto::FLOAT &&
      to != ONNX_NAMESPACE::TensorProto::INT32) {
    LOGS_DEFAULT(VERBOSE) << "[Cast] Only support cast to int32 or float, actual to type, " << to;
    return false;
  }

  return true;
}

Status CastOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  NodeAttrHelper helper(node);

  const auto& input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  auto to = helper.Get("to", 0);
  Type type;
  switch (to) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      type = Type::TENSOR_FLOAT32;
      break;
    case ONNX_NAMESPACE::TensorProto::INT32:
      type = Type::TENSOR_INT32;
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid cast to type: ", to);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CAST, input_indices, {output},
                                                 {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_softmax

class SoftMaxOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 28;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool SoftMaxOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 2 && input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "SoftMax only support 2d/4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  const auto android_sdk_ver = model_builder.GetAndroidSdkVer();
  if (android_sdk_ver < 29) {
    NodeAttrHelper helper(node);
    int32_t axis = helper.Get("axis", 1);
    if (axis != 1) {
      LOGS_DEFAULT(VERBOSE)
          << "SoftMax only support axis 1 on Android API level: " << android_sdk_ver
          << " input axis: " << axis;
      return false;
    }
  }

  return true;
}

Status SoftMaxOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto android_sdk_ver = model_builder.GetAndroidSdkVer();
  NodeAttrHelper helper(node);

  auto input = node.InputDefs()[0]->Name();
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = input_is_nhwc;
  if (android_sdk_ver < 29) {
    if (model_builder.IsOperandNHWC(input)) {
      output_is_nhwc = false;
      // We want to transpose nhwc operand back to nchw before softmax
      ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, 0, input));
    }
  }

  int32_t axis = helper.Get("axis", 1);
  if (output_is_nhwc) {
    const int32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    axis = axis_nchw_to_nhwc[axis];
  }

  const auto& output = node.OutputDefs()[0]->Name();
  float beta = 1.f;
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  if (android_sdk_ver > 28) {
    // you can only specify axis for android api level 29+
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SOFTMAX, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_identity

class IdentityOpBuilder : public BaseOpBuilder {
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

Status IdentityOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  // Identity is not really going to do anything
  // Just register the dimension and type, with same index and new name
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));  // input

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, output_is_nhwc);
  return Status::OK();
}

#pragma endregion

#pragma region op_gemm

class GemmOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;
  bool HasSupportedInputs(const Node& node) override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
  int GetMinSupportedOpSet(const Node& node) override;
};

bool GemmOpBuilder::HasSupportedInputs(const Node& node) {
  if (node.OpType() != "QLinearMatMul")
    return BaseOpBuilder::HasSupportedInputs(node);

  // QLinearMatMul
  if (!HasValidBinaryOpQuantizedInputs(node))
    return false;

  return true;
}

int GemmOpBuilder::GetMinSupportedOpSet(const Node& node) {
  const auto& op(node.OpType());

  // Gemm opset 6- has broadcast attributes we do not support now
  if (op == "Gemm")
    return 7;

  return 1;
}

bool GemmOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  const auto& op_type = node.OpType();
  const auto input_defs(node.InputDefs());
  const auto& initializers(model_builder.GetInitializerTensors());
  size_t a_idx = 0, b_idx = 1, c_idx = 2;  // A*B+C
  bool is_qlinear_matmul = op_type == "QLinearMatMul";
  if (is_qlinear_matmul) {
    a_idx = 0;
    b_idx = 3;
  }

  Shape a_shape;
  {
    if (!GetShape(*input_defs[a_idx], a_shape))
      return false;

    if (a_shape.size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "A must be 2D";
      return false;
    }
  }

  Shape b_shape;
  {
    if (!GetShape(*input_defs[b_idx], b_shape))
      return false;

    if (b_shape.size() != 2) {
      LOGS_DEFAULT(VERBOSE) << "B must be 2D";
      return false;
    }
  }

  if (op_type == "Gemm") {
    // Only support
    // 1. A*B'+C
    // 2. A*B+C and B is an initializer
    NodeAttrHelper helper(node);
    const auto transA = helper.Get("transA", 0);
    const auto transB = helper.Get("transB", 0);
    const auto alpha = helper.Get("alpha", 1.0f);
    const auto beta = helper.Get("beta", 1.0f);

    if (!(transA == 0 && alpha == 1.f && beta == 1.f)) {
      LOGS_DEFAULT(VERBOSE) << "Only transA == 0, alpha == 1.0 "
                            << "and beta == 1.0 is supported.";
      return false;
    }

    if (transB == 0 && !Contains(initializers, input_defs[b_idx]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of Gemm must be known if transB != 1";
      return false;
    }

    if (input_defs.size() == 3) {
      Shape c_shape;
      if (!GetShape(*input_defs[c_idx], c_shape))
        return false;

      if (c_shape.size() != 1 ||
          c_shape[0] != (transB == 0 ? b_shape[1] : b_shape[0])) {
        LOGS_DEFAULT(VERBOSE) << "C of Gemm must be a vector of b_shape[0]"
                              << " b_shape: " << Shape2String(b_shape)
                              << " c_shape: " << Shape2String(c_shape);

        return false;
      }
    }
  } else if (op_type == "MatMul" || is_qlinear_matmul) {
    // Only support A*B B is an initializer
    if (!Contains(initializers, input_defs[b_idx]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "B of MatMul must be known";
      return false;
    }

    if (is_qlinear_matmul) {
      // For QLinearMatMul, we only support uint8 output now
      int32_t output_type;
      if (!GetType(*node.OutputDefs()[0], output_type))
        return false;

      if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
        LOGS_DEFAULT(VERBOSE) << "[" << op_type
                              << "] output type: [" << output_type
                              << "] is not supported for now";
        return false;
      }

      // All scale/zero points are initializer scalars
      // a/b/y_scale
      if (!HasValidQuantizationScale(model_builder.GetInitializerTensors(), node, {1, 4, 6}))
        return false;

      // a/b/y_zero_point
      if (!HasValidQuantizationZeroPoint(model_builder.GetInitializerTensors(), node, {2, 5, 7}))
        return false;
    }
  } else {
    LOGS_DEFAULT(VERBOSE) << "GemmOpBuilder, unknown op: " << op_type;
  }

  return true;
}

void GemmOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  const auto& op = node.OpType();
  const auto input_defs(node.InputDefs());
  if (op == "MatMul") {
    model_builder.AddInitializerToSkip(input_defs[1]->Name());
  } else if (op == "Gemm") {
    NodeAttrHelper helper(node);
    const auto transB = helper.Get("transB", 0);
    if (transB == 0)
      model_builder.AddInitializerToSkip(input_defs[1]->Name());
  } else if (op == "QLinearMatMul") {
    AddBinaryOpQuantizationScaleAndZeroPointToSkip(model_builder, node);
    model_builder.AddInitializerToSkip(input_defs[3]->Name());  // b
  }
}

Status GemmOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());

  const auto& op = node.OpType();
  const auto input_defs(node.InputDefs());
  NodeAttrHelper helper(node);
  bool is_qlinear_matmul = op == "QLinearMatMul";

  size_t a_idx = 0,
         b_idx = is_qlinear_matmul ? 3 : 1,
         c_idx = 2;  // QLinearMatMul has no bias

  const auto& input1 = input_defs[a_idx]->Name();
  const auto& input2 = input_defs[b_idx]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  const auto transB = helper.Get("transB", 0);

  float a_scale = 0.0f,
        b_scale = 0.0f,
        y_scale = 0.0f;
  int32_t a_zero_point = 0,
          b_zero_point = 0,
          y_zero_point = 0;

  if (is_qlinear_matmul) {
    ORT_RETURN_IF_ERROR(GetBinaryOpQuantizationScaleAndZeroPoint(model_builder, node,
                                                                 a_scale, b_scale, y_scale,
                                                                 a_zero_point, b_zero_point, y_zero_point));
  }

  uint32_t input_2_idx;
  if (transB == 0) {
    Type onnx_mat_b_type;
    if (!is_qlinear_matmul)
      onnx_mat_b_type = Type::TENSOR_FLOAT32;
    else
      onnx_mat_b_type = Type::TENSOR_QUANT8_ASYMM;

    const auto& mat_b_tensor = initializers.at(input2);
    Shape onnx_mat_b_shape;
    for (auto dim : mat_b_tensor.dims())
      onnx_mat_b_shape.push_back(SafeInt<uint32_t>(dim));

    const OperandType onnx_mat_b_operand_type(onnx_mat_b_type, onnx_mat_b_shape, b_scale, b_zero_point);
    ORT_RETURN_IF_ERROR(AddInitializerTransposed(model_builder, onnx_mat_b_operand_type, input2));
  }

  input_2_idx = operand_indices.at(input2);

  // Verify if the scale and zero point matchs from onnx input and nnapi input
  if (is_qlinear_matmul) {
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input1, a_scale, a_zero_point));
    ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input2, b_scale, b_zero_point));
  }

  uint32_t bias_idx;
  bool has_bias = (op == "Gemm") && (input_defs.size() > 2);
  if (has_bias) {
    bias_idx = operand_indices.at(input_defs[c_idx]->Name());
  } else {
    // No C supplied, we need a vector of 0
    std::string bias = node.Name() + op + "_bias";
    const auto& bias_type = operand_types.at(input2).type;
    Shape bias_dimen = {shaper[input2][0]};
    if (bias_type == Type::TENSOR_FLOAT32) {
      std::vector<float> buffer(bias_dimen[0], 0.f);
      OperandType bias_operand_type(Type::TENSOR_FLOAT32, bias_dimen);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else if (bias_type == Type::TENSOR_QUANT8_ASYMM) {
      std::vector<int32_t> buffer(bias_dimen[0], 0);
      OperandType bias_operand_type(Type::TENSOR_INT32, bias_dimen, a_scale * b_scale, 0);
      ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(bias, buffer.data(), bias_operand_type));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown weight type ", TypeToStr(bias_type));
    }

    bias_idx = operand_indices.at(bias);
  }

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input1));  // A
  input_indices.push_back(input_2_idx);                 // B
  input_indices.push_back(bias_idx);                    // C
  int32_t fuse_code = model_builder.FindActivation(node, *node.OutputDefs()[0]);
  ADD_SCALAR_OPERAND(model_builder, input_indices, fuse_code);

  ORT_RETURN_IF_ERROR(shaper.FC(input1, input2, output));
  const OperandType output_operand_type(operand_types.at(input1).type, shaper[output], y_scale, y_zero_point);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_FULLY_CONNECTED, input_indices,
                                                 {output}, {output_operand_type}, {false}));
  return Status::OK();
}

#pragma endregion

#pragma region op_unary

class UnaryOpBuilder : public BaseOpBuilder {
 private:
  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;

  // All ops except "Sin" opset 5- uses consumed_inputs attribute which is not supported for now
  // "Sin" op has support from opset 7, return 6 here for all ops
  int GetMinSupportedOpSet(const Node& /* node */) override { return 6; }
};

int32_t UnaryOpBuilder::GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& node) const {
  const auto& op(node.OpType());
  if (op == "Abs" ||
      op == "Exp" ||
      op == "Neg" ||
      op == "Sin" ||
      op == "Sqrt" ||
      op == "Log") {
    return 29;
  }

  return 27;
}

Status UnaryOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& op_type(node.OpType());

  const auto& input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  int32_t op_code;
  if (op_type == "Abs")
    op_code = ANEURALNETWORKS_ABS;
  else if (op_type == "Exp")
    op_code = ANEURALNETWORKS_EXP;
  else if (op_type == "Floor")
    op_code = ANEURALNETWORKS_FLOOR;
  else if (op_type == "Log")
    op_code = ANEURALNETWORKS_LOG;
  else if (op_type == "Sigmoid")
    op_code = ANEURALNETWORKS_LOGISTIC;
  else if (op_type == "Neg")
    op_code = ANEURALNETWORKS_NEG;
  else if (op_type == "Sin")
    op_code = ANEURALNETWORKS_SIN;
  else if (op_type == "Sqrt")
    op_code = ANEURALNETWORKS_SQRT;
  else if (op_type == "Tanh")
    op_code = ANEURALNETWORKS_TANH;
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "UnaryOpBuilder, unknown op: ", op_type);
  }
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_concat

class ConcatOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool ConcatOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Concat only supports up to 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

Status ConcatOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node);

  std::vector<uint32_t> input_indices;
  const auto& input0 = node.InputDefs()[0]->Name();
  bool all_input_have_same_layout = true;
  bool output_is_nhwc = false;
  const auto node_input_size = node.InputDefs().size();

  // First we want to see if all the input are same layout
  for (size_t i = 0; i < node_input_size - 1; i++) {
    all_input_have_same_layout =
        all_input_have_same_layout &&
        model_builder.IsOperandNHWC(node.InputDefs()[i]->Name()) ==
            model_builder.IsOperandNHWC(node.InputDefs()[i + 1]->Name());
  }

  std::vector<std::string> inputs;
  inputs.reserve(node_input_size);
  if (all_input_have_same_layout) {
    // if all the inputs are of same layout, output will be the same layout
    output_is_nhwc = model_builder.IsOperandNHWC(input0);

    for (size_t i = 0; i < node_input_size; i++) {
      auto input = node.InputDefs()[i]->Name();
      input_indices.push_back(operand_indices.at(input));
      inputs.push_back(input);
    }
  } else {
    // if all the inputs are not same layout,
    // will need transpos those nhwc tensors back to nchw
    for (size_t i = 0; i < node_input_size; i++) {
      auto input = node.InputDefs()[i]->Name();
      if (model_builder.IsOperandNHWC(input)) {
        ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, i, input));
      }
      input_indices.push_back(operand_indices.at(input));
      inputs.push_back(input);
    }
  }

  int rank = shaper[input0].size();
  int32_t axis = static_cast<int32_t>(HandleNegativeAxis(helper.Get("axis", 1), rank));

  if (output_is_nhwc) {
    ORT_RETURN_IF_NOT(rank == 4,
                      "nhwc is only on 4d shape, input ", input0, " has rank: ", rank);
    // we are using nhwc here, but the axis is in nchw, need to transpose axis from nchw to nhwc
    const uint32_t axis_nchw_to_nhwc[4]{0, 3, 1, 2};
    axis = axis_nchw_to_nhwc[axis];
  }
  ADD_SCALAR_OPERAND(model_builder, input_indices, axis);

  const auto& output = node.OutputDefs()[0]->Name();
  ORT_RETURN_IF_ERROR(shaper.Concat(inputs, axis, output));
  const OperandType output_operand_type(operand_types.at(input0).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_CONCATENATION, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_squeeze

class SqueezeOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 28;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;

  // Squeeze opset 13+ uses input for axes, which is not supported yet
  // TODO add support for squeeze opset 13+
  int GetMaxSupportedOpSet(const Node& /* node */) override { return 12; }
};

bool SqueezeOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size > 4 || input_size == 0) {
    LOGS_DEFAULT(VERBOSE) << "Squeeze only supports 1-4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

Status SqueezeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  auto input = node.InputDefs()[0]->Name();
  if (model_builder.IsOperandNHWC(input)) {
    // We want to transpose nhwc operand back to nchw before squeeze
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, 0, input));
  }

  NodeAttrHelper helper(node);
  vector<int32_t> axes = helper.Get("axes", vector<int32_t>());
  const auto& input_shape(shaper[input]);
  auto input_dims = input_shape.size();
  for (auto& axis : axes) {
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, input_dims));
  }

  if (axes.empty()) {  // Squeeze all
    for (size_t i = 0; i < input_dims; i++) {
      if (input_shape[i] == 1)
        axes.push_back(i);
    }
  }

  const auto axes_name = model_builder.GetUniqueName(node.Name() + input + "_axes");
  Shape axes_dimen = {static_cast<uint32_t>(axes.size())};
  shaper.AddShape(axes_name, axes_dimen);
  const OperandType axes_operand_type(Type::TENSOR_INT32, axes_dimen);
  ORT_RETURN_IF_ERROR(model_builder.AddOperandFromPersistMemoryBuffer(axes_name, axes.data(), axes_operand_type));

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));      // input
  input_indices.push_back(operand_indices.at(axes_name));  // axes

  const auto& output = node.OutputDefs()[0]->Name();
  ORT_RETURN_IF_ERROR(shaper.Squeeze(input, axes, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_SQUEEZE, input_indices,
                                                 {output}, {output_operand_type}, {false}));
  return Status::OK();
}

#pragma endregion

#pragma region op_quantizelinear

class QuantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 27;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

void QuantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  const auto input_defs(node.InputDefs());

  model_builder.AddInitializerToSkip(input_defs[1]->Name());

  if (input_defs.size() == 3)  // has zero_point input
    model_builder.AddInitializerToSkip(input_defs[2]->Name());
}

bool QuantizeLinearOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  const auto input_defs(node.InputDefs());
  const auto output_defs(node.OutputDefs());

  int32_t output_type;
  if (!GetType(*output_defs[0], output_type))
    return false;

  if (output_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] output type: [" << output_type
                          << "] is not supported for now";
    return false;
  }

  if (!HasValidQuantizationScale(model_builder.GetInitializerTensors(), node, {1}))
    return false;

  if (input_defs.size() == 3) {  // has zero_point input
    if (!HasValidQuantizationZeroPoint(model_builder.GetInitializerTensors(), node, {2}))
      return false;
  }

  return true;
}

Status QuantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto input_defs(node.InputDefs());

  const auto& input = input_defs[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  float scale = GetQuantizationScale(model_builder, node, 1);
  int32_t zero_point = 0;
  Type output_type = Type::TENSOR_QUANT8_ASYMM;

  if (input_defs.size() == 3) {  // Get zero point
    ORT_RETURN_IF_ERROR(GetQuantizationZeroPoint(model_builder, node, 2, zero_point));
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(output_type, shaper[output], scale, zero_point);
  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_QUANTIZE, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_dequantizelinear

class DequantizeLinearOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 29;
  }

  bool HasSupportedInputs(const Node& node) override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool DequantizeLinearOpBuilder::HasSupportedInputs(const Node& node) {
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    LOGS_DEFAULT(VERBOSE) << "[" << node.OpType()
                          << "] Input type: [" << input_type
                          << "] is not supported for now";
    return false;
  }

  return true;
}

void DequantizeLinearOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  const auto input_defs(node.InputDefs());

  model_builder.AddInitializerToSkip(input_defs[1]->Name());

  if (input_defs.size() == 3)  // has zero_point input
    model_builder.AddInitializerToSkip(input_defs[2]->Name());
}

bool DequantizeLinearOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  const auto input_defs(node.InputDefs());

  if (!HasValidQuantizationScale(model_builder.GetInitializerTensors(), node, {1}))
    return false;

  if (input_defs.size() == 3) {  // has zero_point input
    if (!HasValidQuantizationZeroPoint(model_builder.GetInitializerTensors(), node, {2}))
      return false;
  }

  return true;
}

Status DequantizeLinearOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto input_defs(node.InputDefs());

  const auto& input = input_defs[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  float scale = GetQuantizationScale(model_builder, node, 1);
  int32_t zero_point = 0;
  if (input_defs.size() == 3) {  // Get zero point
    ORT_RETURN_IF_ERROR(GetQuantizationZeroPoint(model_builder, node, 2, zero_point));
  }

  ORT_RETURN_IF_ERROR(IsValidInputQuantizedType(model_builder, input, scale, zero_point));

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(Type::TENSOR_FLOAT32, shaper[output]);

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_DEQUANTIZE, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_LRN

class LRNOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const override {
    return 28;
  }

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

bool LRNOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "LRN only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  return true;
}

Status LRNOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  NodeAttrHelper helper(node);
  const auto android_sdk_ver = model_builder.GetAndroidSdkVer();

  auto input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);
  if (android_sdk_ver < 29) {
    // on android api level 28, we need to transpose the nchw input to nhwc
    output_is_nhwc = true;
    if (!model_builder.IsOperandNHWC(input)) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node, 0, input));
    }
  }

  auto alpha = helper.Get("alpha", 0.0001f);
  const auto beta = helper.Get("beta", 0.75f);
  const auto bias = helper.Get("bias", 1.0f);
  const auto size = helper.Get("size", 1);

  const auto radius = (size - 1) / 2;
  alpha /= size;  // NNAPI's alpha is different than ONNX's alpha

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, radius);
  ADD_SCALAR_OPERAND(model_builder, input_indices, bias);
  ADD_SCALAR_OPERAND(model_builder, input_indices, alpha);
  ADD_SCALAR_OPERAND(model_builder, input_indices, beta);

  // specify axis is only available on api level >= 29
  if (android_sdk_ver > 28) {
    // ONNX LRN is always performed on C dimension
    int32_t axis = output_is_nhwc
                       ? 3   // nhwc
                       : 1;  // nchw
    ADD_SCALAR_OPERAND(model_builder, input_indices, axis);
  }

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(ANEURALNETWORKS_LOCAL_RESPONSE_NORMALIZATION, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_clip

class ClipOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

void ClipOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  if (node.InputDefs().size() > 1)
    model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());  // min

  if (node.InputDefs().size() > 2)
    model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());  // max
}

bool ClipOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  float min, max;
  if (!GetClipMinMax(model_builder.GetInitializerTensors(), node, min, max))
    return false;

  // We only supoort relu6 or relu1
  // TODO, support clip between 2 arbitrary numbers
  if ((min == 0.0f && max == 6.0f) || (min == -1.0f && max == 1.0f)) {
    return true;
  } else {
    LOGS_DEFAULT(VERBOSE) << "Clip only supports [min, max] = [0, 6] or [-1, 1], the input is ["
                          << min << ", " << max << "]";
    return false;
  }

  return true;
}

Status ClipOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());

  const auto& input = node.InputDefs()[0]->Name();
  const auto& output = node.OutputDefs()[0]->Name();
  bool output_is_nhwc = model_builder.IsOperandNHWC(input);

  ORT_RETURN_IF_ERROR(shaper.Identity(input, output));
  const OperandType output_operand_type(operand_types.at(input).type, shaper[output]);

  if (Contains(model_builder.GetFusedActivations(), input)) {
    LOGS_DEFAULT(VERBOSE) << "Clip Node [" << node.Name() << "] fused";
    model_builder.RegisterOperand(output, operand_indices.at(input), output_operand_type, output_is_nhwc);
    return Status::OK();
  }

  float min, max;
  GetClipMinMax(model_builder.GetInitializerTensors(), node, min, max);

  int32_t op_code;
  if (min == 0.0f && max == 6.0f)
    op_code = ANEURALNETWORKS_RELU6;
  else if (min == -1.0f && max == 1.0f)
    op_code = ANEURALNETWORKS_RELU1;
  else
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "ClipOpBuilder, unsupported input [", min, ", ", max, "].",
                           "We should not reach here, ClipOpBuilder::IsOpSupportedImpl should have caught this.");

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(op_code, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));
  return Status::OK();
}

#pragma endregion

#pragma region op_Resize

class ResizeOpBuilder : public BaseOpBuilder {
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) override;

 private:
  bool IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) override;

  int32_t GetMinSupportedSdkVer(ModelBuilder& model_builder, const Node& node) const override;

  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;

  // Resize opset 10- is very different than Resize opset 11+, with many key attributes missing
  // We only support Resize opset 11+ here
  int GetMinSupportedOpSet(const Node& /* node */) override { return 11; }
};

int32_t ResizeOpBuilder::GetMinSupportedSdkVer(ModelBuilder& /* model_builder */, const Node& /* node */) const {
  return 28;
}

void ResizeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) {
  // We will still add scales to the skipped list even sizes are present
  // since there is no use of it, we will not process it later
  model_builder.AddInitializerToSkip(node.InputDefs()[2]->Name());  // scales

  if (node.InputDefs().size() > 3)
    model_builder.AddInitializerToSkip(node.InputDefs()[3]->Name());  // sizes
}

bool ResizeOpBuilder::IsOpSupportedImpl(ModelBuilder& model_builder, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  const auto input_size = input_shape.size();
  if (input_size != 4) {
    LOGS_DEFAULT(VERBOSE) << "Resize only support 4d shape, input is "
                          << input_size << "d shape";
    return false;
  }

  {  // check attributes
    const auto android_sdk_ver = model_builder.GetAndroidSdkVer();

    NodeAttrHelper helper(node);
    const auto mode = helper.Get("mode", "nearest");
    if (mode != "linear") {
      LOGS_DEFAULT(VERBOSE) << "Resize unsupported input mode, " << mode;
      return false;
    }

    const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
    bool using_half_pixel = coord_trans_mode == "half_pixel";
    bool using_align_corners = coord_trans_mode == "align_corners";
    if (!using_half_pixel && !using_align_corners && coord_trans_mode != "asymmetric") {
      LOGS_DEFAULT(VERBOSE) << "Resize, unsupported coord_trans_mode, " << coord_trans_mode;
      return false;
    }

    if (android_sdk_ver < 30 && (using_half_pixel || using_align_corners)) {
      LOGS_DEFAULT(VERBOSE) << "Resize only support half_pixel/align_corners on API level 30+, current API level is "
                            << android_sdk_ver;
      return false;
    }

    const auto exclude_outside = helper.Get("exclude_outside", 0);
    if (exclude_outside != 0) {
      LOGS_DEFAULT(VERBOSE) << "Resize does not support exclude_outside for now";
      return false;
    }
  }

  {  // scales and sizes (if present) must be initializers
    const auto& initializers(model_builder.GetInitializerTensors());
    const auto input_defs = node.InputDefs();
    // scales
    if (input_defs.size() < 3 || !Contains(initializers, input_defs[2]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input scales of Resize must be known";
      return false;
    }

    // sizes
    if (input_defs.size() > 3 && !Contains(initializers, input_defs[3]->Name())) {
      LOGS_DEFAULT(VERBOSE) << "Input sizes of Resize must be known";
      return false;
    }

    // We want to check if the scales or sizes are not trying to resize on N/C channels here
    if (input_defs.size() == 3) {  // we are using scales
      const auto& scales_tensor = initializers.at(input_defs[2]->Name());
      const float* scales_data = GetTensorFloatData(scales_tensor);
      float scale_n = scales_data[0];
      float scale_c = scales_data[1];
      if (scale_n != 1.0f || scale_c != 1.0f) {
        LOGS_DEFAULT(VERBOSE) << "Scales of N/C channel should be 1"
                              << "Resize of N/C channels are not supported"
                              << ", scale_n, " << scale_n << ", scale_c, " << scale_c;
        return false;
      }
    } else {
      // we are using sizes
      const auto& sizes_name = input_defs[3]->Name();
      const auto& sizes_tensor = initializers.at(sizes_name);
      const int64_t* sizes_data = GetTensorInt64Data(sizes_tensor);
      uint32_t size_n = SafeInt<uint32_t>(sizes_data[0]);
      uint32_t size_c = SafeInt<uint32_t>(sizes_data[1]);
      if (size_n != input_shape[0] || size_c != input_shape[1]) {
        LOGS_DEFAULT(VERBOSE) << "Output sizes of N/C chanel should match the input sizes, "
                              << "Resize of N/C channels are not supported"
                              << ", input_size_n, " << input_shape[0] << ", output_size_n, " << size_n
                              << ". input_size_c, " << input_shape[1] << ", output_size_c, " << size_c;
        return false;
      }
    }
  }
  return true;
}

Status ResizeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto& shaper(model_builder.GetShaper());
  const auto& operand_indices(model_builder.GetOperandIndices());
  const auto& operand_types(model_builder.GetOperandTypes());
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);
  const auto input_defs = node.InputDefs();
  const auto android_sdk_ver = model_builder.GetAndroidSdkVer();
  const auto& output = node.OutputDefs()[0]->Name();

  auto input = input_defs[0]->Name();
  bool use_nchw = model_builder.UseNCHW();
  bool input_is_nhwc = model_builder.IsOperandNHWC(input);
  bool output_is_nhwc = false;
  if (use_nchw) {
    ORT_RETURN_IF_NOT(!input_is_nhwc, "model_builder.UseNCHW() but input is NHWC");
  } else {
    output_is_nhwc = true;
    if (!input_is_nhwc) {
      ORT_RETURN_IF_ERROR(GetNHWCInput(model_builder, node, 0, input));
    }
  }

  // TODO, add support for nearest neighbor
  int32_t operationCode = ANEURALNETWORKS_RESIZE_BILINEAR;

  const auto coord_trans_mode = helper.Get("coordinate_transformation_mode", "half_pixel");
  bool using_half_pixel = coord_trans_mode == "half_pixel";
  bool using_align_corners = coord_trans_mode == "align_corners";

  if (input_defs.size() == 3) {  // we are using scales
    const auto& scales_name = input_defs[2]->Name();
    const auto& scales_tensor = initializers.at(scales_name);
    const float* scales_data = GetTensorFloatData(scales_tensor);
    float scale_h = scales_data[2];
    float scale_w = scales_data[3];
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingScales(input, scale_h, scale_w, use_nchw, output));
  } else {  // we are using sizes
    const auto& sizes_name = input_defs[3]->Name();
    const auto& sizes_tensor = initializers.at(sizes_name);
    const int64_t* sizes_data = GetTensorInt64Data(sizes_tensor);
    ORT_RETURN_IF_ERROR(
        shaper.ResizeUsingOutputSizes(input, SafeInt<uint32_t>(sizes_data[2]), SafeInt<uint32_t>(sizes_data[3]), use_nchw, output));
  }

  const auto& output_shape = shaper[output];
  int32_t output_h = use_nchw ? output_shape[2] : output_shape[1];
  int32_t output_w = use_nchw ? output_shape[3] : output_shape[2];

  std::vector<uint32_t> input_indices;
  input_indices.push_back(operand_indices.at(input));
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_w);
  ADD_SCALAR_OPERAND(model_builder, input_indices, output_h);

  if (android_sdk_ver > 28) {
    // using nchw is only available on API level 29
    ADD_SCALAR_OPERAND(model_builder, input_indices, use_nchw);
  }

  if (android_sdk_ver > 29 && (using_align_corners || using_half_pixel)) {
    ADD_SCALAR_OPERAND(model_builder, input_indices, using_align_corners);
    if (using_half_pixel)
      ADD_SCALAR_OPERAND(model_builder, input_indices, using_half_pixel);
  }

  const OperandType output_operand_type(operand_types.at(input).type, output_shape);
  ORT_RETURN_IF_ERROR(model_builder.AddOperation(operationCode, input_indices,
                                                 {output}, {output_operand_type}, {output_is_nhwc}));

  return Status::OK();
}

#pragma endregion

#pragma region op_flatten

class FlattenOpBuilder : public BaseOpBuilder {
 private:
  bool IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) override;
  static void GetFlattenShape(const Node& node, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2);
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) override ORT_MUST_USE_RESULT;
};

/* static */ void FlattenOpBuilder::GetFlattenShape(const Node& node, const Shape& input_shape, int32_t& dim_1, int32_t& dim_2) {
  int32_t rank = static_cast<int>(input_shape.size());
  NodeAttrHelper helper(node);
  int32_t axis = helper.Get("axis", 1);
  // axis == rank is a valid input, but invalid for HandleNegativeAxis
  // Skip non-negative axis here
  if (axis < 0)
    axis = static_cast<int32_t>(HandleNegativeAxis(axis, rank));

  dim_1 = std::accumulate(input_shape.cbegin(), input_shape.cbegin() + axis, 1, std::multiplies<int32_t>());
  dim_2 = std::accumulate(input_shape.cbegin() + axis, input_shape.cend(), 1, std::multiplies<int32_t>());
}

bool FlattenOpBuilder::IsOpSupportedImpl(ModelBuilder& /* model_builder */, const Node& node) {
  Shape input_shape;
  if (!GetShape(*node.InputDefs()[0], input_shape))
    return false;

  if (input_shape.size() > 4 || input_shape.empty()) {
    LOGS_DEFAULT(VERBOSE) << "Flatten only supports up to 1-4d shape, input is "
                          << input_shape.size() << "d shape";
    return false;
  }

  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenShape(node, input_shape, dim_1, dim_2);

  if (dim_1 == 0 && dim_2 == 0) {
    LOGS_DEFAULT(VERBOSE) << "The dynamical input shape " << Shape2String(input_shape)
                          << " is not supported";
    return false;
  }

  return true;
}

Status FlattenOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node) {
  auto input = node.InputDefs()[0]->Name();
  if (model_builder.IsOperandNHWC(input)) {
    // We want to transpose nhwc operand back to nchw before reshape
    ORT_RETURN_IF_ERROR(GetNCHWInput(model_builder, node, 0, input));
  }

  // Flatten is basically a reshape to 2d tensor
  // Get the shape for Reshape here
  Shape input_shape;
  GetShape(*node.InputDefs()[0], input_shape);
  int32_t dim_1 = 1;
  int32_t dim_2 = 1;
  GetFlattenShape(node, input_shape, dim_1, dim_2);
  // If the input is of dynamic shape, replace 0 (dynamic) dimension with -1
  // We cannot have dim_1 and dim_2 both be 0 here, it was checked in IsOpSupportedImpl
  dim_1 = dim_1 == 0 ? -1 : dim_1;
  dim_2 = dim_2 == 0 ? -1 : dim_2;
  std::vector<int32_t> shape{dim_1, dim_2};
  return ReshapeOpBuilder::AddReshapeOperator(model_builder, node, input, shape);
}

#pragma endregion op_reshape

#pragma region CreateOpBuilders

std::unordered_map<std::string, std::shared_ptr<IOpBuilder>>
CreateOpBuilders() {
  std::unordered_map<std::string, std::shared_ptr<IOpBuilder>> op_map;

  {
    auto binary_op_builder = std::make_shared<BinaryOpBuilder>();
    op_map.emplace("Add", binary_op_builder);
    op_map.emplace("Sub", binary_op_builder);
    op_map.emplace("Mul", binary_op_builder);
    op_map.emplace("Div", binary_op_builder);
    op_map.emplace("QLinearAdd", binary_op_builder);
  }

  op_map.emplace("Relu", std::make_shared<ReluOpBuilder>());
  op_map.emplace("Transpose", std::make_shared<TransposeOpBuilder>());
  op_map.emplace("Reshape", std::make_shared<ReshapeOpBuilder>());
  op_map.emplace("BatchNormalization", std::make_shared<BatchNormalizationOpBuilder>());

  {
    auto pool_op_builder = std::make_shared<PoolOpBuilder>();
    op_map.emplace("GlobalAveragePool", pool_op_builder);
    op_map.emplace("GlobalMaxPool", pool_op_builder);
    op_map.emplace("AveragePool", pool_op_builder);
    op_map.emplace("MaxPool", pool_op_builder);
  }

  {
    auto conv_op_builder = std::make_shared<ConvOpBuilder>();
    op_map.emplace("Conv", conv_op_builder);
    op_map.emplace("QLinearConv", conv_op_builder);
  }

  op_map.emplace("Cast", std::make_shared<CastOpBuilder>());
  op_map.emplace("Softmax", std::make_shared<SoftMaxOpBuilder>());
  op_map.emplace("Identity", std::make_shared<IdentityOpBuilder>());

  {
    auto gemm_op_builder = std::make_shared<GemmOpBuilder>();
    op_map.emplace("Gemm", gemm_op_builder);
    op_map.emplace("MatMul", gemm_op_builder);
    op_map.emplace("QLinearMatMul", gemm_op_builder);
  }

  {
    auto unary_op_builder = std::make_shared<UnaryOpBuilder>();
    op_map.emplace("Abs", unary_op_builder);
    op_map.emplace("Exp", unary_op_builder);
    op_map.emplace("Floor", unary_op_builder);
    op_map.emplace("Log", unary_op_builder);
    op_map.emplace("Sigmoid", unary_op_builder);
    op_map.emplace("Neg", unary_op_builder);
    op_map.emplace("Sin", unary_op_builder);
    op_map.emplace("Sqrt", unary_op_builder);
    op_map.emplace("Tanh", unary_op_builder);
  }

  op_map.emplace("Concat", std::make_shared<ConcatOpBuilder>());
  op_map.emplace("Squeeze", std::make_shared<SqueezeOpBuilder>());
  op_map.emplace("QuantizeLinear", std::make_shared<QuantizeLinearOpBuilder>());
  op_map.emplace("DequantizeLinear", std::make_shared<DequantizeLinearOpBuilder>());
  op_map.emplace("LRN", std::make_shared<LRNOpBuilder>());
  op_map.emplace("Clip", std::make_shared<ClipOpBuilder>());
  op_map.emplace("Resize", std::make_shared<ResizeOpBuilder>());
  op_map.emplace("Flatten", std::make_shared<FlattenOpBuilder>());

  return op_map;
}

#pragma endregion

}  // namespace nnapi
}  // namespace onnxruntime
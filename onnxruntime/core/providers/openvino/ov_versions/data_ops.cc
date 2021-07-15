// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <unordered_set>
#include "core/providers/shared_library/provider_api.h"
#include "../backend_utils.h"
#include "../backend_manager.h"
#include <string>
#include <vector>
#include "data_ops.h"
#include "capabilities.h"
#include "utils.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4245 5208)
#elif __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <ngraph/ngraph.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>
#if defined(_MSC_VER)
#pragma warning(default : 4244 4245)
#elif __GNUC__
#pragma GCC diagnostic pop
#endif

namespace onnxruntime {
namespace openvino_ep {

//Ops which are supported only in models(as intermediate nodes) and not in unit tests
std::set<std::string> ops_supported_only_in_model = {
    "Cast",
    "Concat",
    "ConstantOfShape",
    "Dropout",
    "Expand",
    "EyeLike",
    "Exp",
    "GatherND",
    "Identity",
    "Loop",
    "NonMaxSuppression",
    "NonZero",
    "Not",
    "OneHot",
    "Pad",
    "Range",
    "ReduceMin",
    "Resize",
    "Round",
    "Shape",
    "Split",
    "Tile",
    "TopK",
    "QuantizeLinear"};

std::vector<SupportedOp> supported_op_mode = {
    {"Abs", V_2020_4, {"CPU", "GPU"}},
    {"Acos", V_2020_4, {"CPU"}},
    {"Acosh", V_2020_4, {"CPU"}},
    {"Add", V_2020_4, {"All"}},
    {"And", V_2020_4, {"All"}},
    {"ArgMax", V_2020_4, {"CPU"}},
    {"ArgMax", V_2021_1, {"All"}},
    {"ArgMin", V_2020_4, {"CPU"}},
    {"ArgMin", V_2021_2, {"CPU", "MYRIAD"}},
    {"Asin", V_2020_4, {"CPU", "GPU"}},
    {"Asinh", V_2020_4, {"CPU", "GPU"}},
    {"Atan", V_2020_4, {"CPU", "GPU"}},
    {"Atanh", V_2020_4, {"CPU"}},
    {"AveragePool", V_2020_4, {"All"}},
    {"BatchNormalization", V_2020_4, {"All"}},
    {"Cast", V_2020_4, {"All"}},
    {"Ceil", V_2020_4, {"GPU"}},
    {"Ceil", V_2021_3, {"MYRIAD"}},
    {"Ceil", V_2021_2, {"GPU", "MYRIAD"}},
    {"Clip", V_2020_4, {"All"}},
    {"Concat", V_2020_4, {"All"}},
    {"Constant", V_2020_4, {"All"}},
    {"ConstantOfShape", V_2020_4, {"All"}},
    {"Conv", V_2020_4, {"All"}},
    {"ConvTranspose", V_2020_4, {"All"}},
    {"Cos", V_2020_4, {"CPU"}},
    {"Cosh", V_2020_4, {"CPU"}},
    {"DepthToSpace", V_2020_4, {"All"}},
    {"DequantizeLinear", V_2021_4, {"CPU", "GPU"}},
    {"Div", V_2020_4, {"All"}},
    {"Dropout", V_2020_4, {"All"}},
    {"Elu", V_2020_4, {"All"}},
    {"Equal", V_2020_4, {"All"}},
    {"Erf", V_2020_4, {"All"}},
    {"Exp", V_2020_4, {"All"}},
    {"Expand", V_2021_1, {"MYRIAD"}},
    {"Flatten", V_2020_4, {"All"}},
    {"Floor", V_2020_4, {"All"}},
    {"Gather", V_2020_4, {"All"}},
    {"GatherElements", V_2021_3, {"MYRIAD"}},
    {"GatherND", V_2021_2, {"MYRIAD"}},
    {"Gemm", V_2020_4, {"All"}},
    {"GlobalAveragePool", V_2020_4, {"All"}},
    {"GlobalLpPool", V_2020_4, {"CPU", "GPU"}},
    {"Greater", V_2020_4, {"All"}},
    {"Identity", V_2020_4, {"All"}},
    {"InstanceNormalization", V_2020_4, {"All"}},
    {"HardSigmoid", V_2020_4, {"CPU", "GPU"}},
    {"LeakyRelu", V_2020_4, {"All"}},
    {"Less", V_2020_4, {"All"}},
    {"Log", V_2020_4, {"All"}},
    {"Loop", V_2021_3, {"MYRIAD"}},
    {"LRN", V_2020_4, {"All"}},
    {"LSTM", V_2020_4, {"All"}},
    {"MatMul", V_2020_4, {"All"}},
    {"Max", V_2020_4, {"All"}},
    {"MaxPool", V_2020_4, {"All"}},
    {"Mean", V_2020_4, {"All"}},
    {"Min", V_2020_4, {"All"}},
    {"Mul", V_2020_4, {"All"}},
    {"Neg", V_2020_4, {"All"}},
    {"NonMaxSuppression", V_2021_1, {"All"}},
    {"NonZero", V_2021_1, {"CPU", "MYRIAD"}},
    {"Not", V_2021_1, {"All"}},
    {"Not", V_2020_4, {"CPU", "GPU"}},
    {"OneHot", V_2020_4, {"All"}},
    {"Pad", V_2020_4, {"All"}},
    {"Pow", V_2020_4, {"All"}},
    {"PRelu", V_2020_4, {"All"}},
    {"QuantizeLinear", V_2021_4, {"CPU", "GPU"}},
    {"Range", V_2021_2, {"MYRIAD"}},
    {"Reciprocal", V_2020_4, {"All"}},
    {"ReduceLogSum", V_2020_4, {"CPU", "MYRIAD"}},
    {"ReduceMax", V_2020_4, {"All"}},
    {"ReduceMean", V_2020_4, {"All"}},
    {"ReduceMin", V_2020_4, {"All"}},
    {"ReduceProd", V_2020_4, {"CPU"}},
    {"ReduceSum", V_2020_4, {"All"}},
    {"ReduceSumSquare", V_2020_4, {"CPU", "MYRIAD"}},
    {"Relu", V_2020_4, {"All"}},
    {"Resize", V_2020_4, {"CPU"}},
    {"Resize", V_2021_3, {"MYRIAD"}},
    {"Reshape", V_2020_4, {"All"}},
    {"RoiAlign", V_2021_1, {"All"}},
    {"Round", V_2021_2, {"MYRIAD"}},
    {"Scatter", V_2021_1, {"MYRIAD"}},
    {"ScatterElements", V_2021_2, {"MYRIAD"}},
    {"Selu", V_2020_4, {"CPU", "GPU"}},
    {"Shape", V_2020_4, {"All"}},
    {"Sigmoid", V_2020_4, {"All"}},
    {"Sign", V_2020_4, {"CPU"}},
    {"Sign", V_2020_4, {"CPU"}},
    {"Sinh", V_2020_4, {"CPU"}},
    {"SinFloat", V_2020_4, {"MYRIAD"}},
    {"Slice", V_2020_4, {"All"}},
    {"Softmax", V_2020_4, {"All"}},
    {"SpaceToDepth", V_2020_4, {"All"}},
    {"Split", V_2020_4, {"All"}},
    {"Sqrt", V_2020_4, {"All"}},
    {"Squeeze", V_2020_4, {"All"}},
    {"Softsign", V_2020_4, {"CPU"}},
    {"Sub", V_2020_4, {"All"}},
    {"Sum", V_2020_4, {"All"}},
    {"Tan", V_2020_4, {"CPU", "GPU"}},
    {"Tanh", V_2020_4, {"All"}},
    {"Tile", V_2021_3, {"MYRIAD"}},
    {"Transpose", V_2020_4, {"All"}},
    {"TopK", V_2020_4, {"All"}},
    {"Unsqueeze", V_2020_4, {"All"}},
    {"Upsample", V_2021_1, {"CPU"}},
    {"Where", V_2021_2, {"MYRIAD"}},
};

void DataOps::populate_types_supported() {
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_initializer_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_initializer_.insert(std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_initializer_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_initializer_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));

  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_vpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_vpu_.insert(std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));

  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_BOOL));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));
  supported_types_cpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));

  supported_types_gpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));
  supported_types_gpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32));
  supported_types_gpu_.insert(std::make_pair(V_2020_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64));
  supported_types_gpu_.insert(std::make_pair(V_2021_1, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16));
  supported_types_gpu_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8));
  supported_types_gpu_.insert(std::make_pair(V_2021_4, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8));

}

void DataOps::populate_op_mode_supported() {
  no_dimension_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Squeeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Cast", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Gather", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Mul", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Sub", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Min", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Div", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Floor", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Where", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"Range", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"ArgMin", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Max", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Add", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Less", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Greater", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Clip", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Resize", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Equal", V_2021_2, {"MYRIAD"}});

  no_dimension_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Squeeze", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Cast", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Gather", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Mul", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Sub", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Min", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Div", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Floor", V_2020_4, {"All"}});
  no_dimension_supported_.push_back({"Where", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"Range", V_2021_2, {"All"}});
  no_dimension_supported_.push_back({"ArgMin", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Max", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Add", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Less", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Greater", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Clip", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Resize", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Equal", V_2021_2, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Reshape", V_2021_3, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Ceil", V_2021_3, {"MYRIAD"}});
  no_dimension_supported_.push_back({"Loop", V_2021_3, {"MYRIAD"}});
  no_dimension_supported_.push_back({"ReduceMin", V_2021_3, {"MYRIAD"}});
  no_dimension_supported_.push_back({"QuantizeLinear", V_2021_4, {"All"}});
  no_dimension_supported_.push_back({"DequantizeLinear", V_2021_4, {"All"}});
  

  subgraph_supported_.push_back({"Mul", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Transpose", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Unsqueeze", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Cast", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Concat", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Gather", V_2020_4, {"All"}});
  subgraph_supported_.push_back({"Div", V_2020_4, {"MYRIAD"}});
  subgraph_supported_.push_back({"Sub", V_2020_4, {"MYRIAD"}});
  subgraph_supported_.push_back({"Identity", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Div", V_2021_1, {"CPU"}});
  subgraph_supported_.push_back({"Sub", V_2021_1, {"CPU"}});

  //populate unsupportedmode_t
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                 if (node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Abs", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //tensor type does not support select last index
                               auto& attributes = node->GetAttributes();
                               auto last_index_arg = attributes.count("select_last_index") > 0 ? attributes.at("select_last_index").i() : 0;
                               if (last_index_arg != 0)
                                 return true;
                               // tensor type supports float as input for argmax and argmin
                               if (node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"ArgMax", obj});
    op_list_.insert({"ArgMin", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               // ceil_mode attribute is not supported in nGraph
                               const auto& attributes = node->GetAttributes();
                               auto ceil_attr = attributes.find("ceil_mode");
                               // default value of ceil_mode (0) is supported.
                               if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0) return true;
                               return (!dimension_unsupported(node));
                             }};
    op_list_.insert({"AveragePool", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //auto pad null value is not supported
                               const auto& attributes = node->GetAttributes();
                               auto auto_attr = attributes.find("auto_pad");
                               if (auto_attr->second().s() == "") {
                                 return true;
                               }
                               // default value of ceil_mode (0) is supported.
                               auto ceil_attr = attributes.find("ceil_mode");
                               if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0) return true;
                               return (!dimension_unsupported(node));
                             }};
    op_list_.insert({"AveragePool", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //Only float 16, float and double data types are supported
                               const bool data_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
                               const bool data_is_float16 = node->InputDefs()[0]->Type()->find("float16") != std::string::npos;
                               const bool data_is_double = node->InputDefs()[0]->Type()->find("double") != std::string::npos;
                               return !(data_is_float || data_is_float16 || data_is_double);
                             }};
    op_list_.insert({"Clip", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (GetInputCount(node, initializers) > 1)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Conv", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& ) {
                               if (device_id_.find("MYRIAD") != std::string::npos) {
                                  const auto& attributes = node->GetAttributes();
                                  auto conv_filter = attributes.find("kernel_shape");
                                  auto& ints = conv_filter->second().ints();
                                  //If the kernel size is not 2D, the op is rejected in case of MYRIAD
                                  if(ints.size() !=2) {
                                    return true;
                                  }
                               }
                               //If the device is GPU
                               if (device_id_.find("GPU") != std::string::npos) {
                                  bool if_bias = false;
                                  const auto& attributes = node->GetAttributes();
                                  auto conv_filter = attributes.find("kernel_shape");
                                  if (conv_filter != attributes.end()) {
                                  auto& ints = conv_filter->second().ints();
                                  //check if the Input for the op has bias
                                   if(node->InputDefs().size() > 2) {
                                     if(node->InputDefs()[2]->Name() == "B")
                                       if_bias = true;
                                   }
                                  //If the kernel size is 1D and the input has bias and the output precision is FP32, the op is rejected.
                                  if(ints.size() == 1 && if_bias) {
                                    auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                    if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
                                      return true;
                                  }
                                  //If the kernel size is 3D and the input doesnot have bias, the op is rejected in case of GPU
                                  if(ints.size() == 3 && !if_bias)
                                    return true;
                                  }
                                }
                                return false;
                             }};
    op_list_.insert({"Conv", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               auto& attributes = node->GetAttributes();
                               if (attributes.count("auto_pad") == 0 || attributes.at("auto_pad").s() == "") {
                                 return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Conv", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (GetInputCount(node, initializers) > 1)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"ConvTranspose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (device_id_.find("MYRIAD") != std::string::npos) {
                                 if (GetInputCount(node, initializers) > 1)
                                  return true;
                               }
                               bool if_bias = false;
                               const auto& attributes = node->GetAttributes();
                               auto out_shape_attr = attributes.find("output_shape");

                               //tensor type does not support output_shape in Attributes
                               if (out_shape_attr != attributes.end())
                                  return true;

                               // If the device is GPU
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto conv_filter = attributes.find("kernel_shape");
                                 if (conv_filter != attributes.end()) {
                                   auto& kernel_size = conv_filter->second().ints();

                                   //If 3D convolution, reject the op
                                   if(kernel_size.size() == 3)
                                     return true;
                                   //In 1D conv, if the pads are asymmetric, then the op is rejected
                                   if(kernel_size.size() == 1) {
                                     if (attributes.count("pads") > 0) {
                                     auto& pads_attr = attributes.at("pads");
                                     auto int_size = pads_attr.ints_size();
                                     if (int_size > 1 && (pads_attr.ints(0) != pads_attr.ints(1)))
                                       return true;
                                     }
                                   }
                                   //check if the Input for the op has bias
                                   if(node->InputDefs().size() > 2) {
                                     if(node->InputDefs()[2]->Name() == "B")
                                       if_bias = true;
                                   }
                                   //If the kernel size is 2D, the input has no bias, the padding is 0 and the op has dilations, the op is rejected
                                   if(kernel_size.size() == 2 && !if_bias) {
                                     if (attributes.count("pads") > 0) {
                                     auto& pads_attr = attributes.at("pads");
                                     auto int_size = pads_attr.ints_size();
                                     // comparing if all the 4 values in the padding are equal to 1
                                     if (int_size == 4 && (pads_attr.ints(0) == pads_attr.ints(1) == pads_attr.ints(2) == pads_attr.ints(3))) {
                                       if(pads_attr.ints(0) == 1)
                                           return false;
                                     }
                                     // If the op has dilations as an attribute after the above checks, then the op is rejected.
                                     auto dilation_attr = attributes.find("dilations");
                                     if (dilation_attr != attributes.end())
                                       return true;
                                     }
                                   }
                                  }
                                 }
                                 return false;
                             }};
    op_list_.insert({"ConvTranspose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               auto& attributes = node->GetAttributes();
                               if (attributes.count("auto_pad") == 0 || attributes.at("auto_pad").s() == "") {
                                 return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ConvTranspose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("MYRIAD") != std::string::npos) {
                                 const auto& input_arg = node->InputDefs()[0];
                                 auto shape = input_arg->Shape();
                                 if ((shape != nullptr) && (shape->dim(0).value_case() != shape->dim(0).kDimValue)) {
                                   return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"ConvTranspose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               // all ConvInteger zero points need to be constants
                               if (node->InputDefs().size() == 3) {
                                 return (initializers.find(node->InputDefs()[2]->Name()) == initializers.end());
                               } else if (node->InputDefs().size() == 4) {
                                 return initializers.find(node->InputDefs()[2]->Name()) == initializers.end() ||
                                        initializers.find(node->InputDefs()[3]->Name()) == initializers.end();
                               }
                               return false;
                             }};
    op_list_.insert({"ConvInteger", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1},
                             [this](const Node* node, const InitializedTensorSet&) {
                               using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;
                               auto supportedOps = std::set<std::vector<onnx_dtype>>{
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT},
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_FLOAT},
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_INT8},
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT},
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8},
                                   {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8},
                                   {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_UINT8},
                                   {onnx_dtype::TensorProto_DataType_INT8, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT8},
                                   {onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32},
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT},
                                   {onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_UINT8}};

                               if (node->OpType() == "Equal") {
                                 supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_INT32, onnx_dtype::TensorProto_DataType_INT32}),
                                     supportedOps.insert(std::vector<onnx_dtype>{onnx_dtype::TensorProto_DataType_UINT8, onnx_dtype::TensorProto_DataType_FLOAT, onnx_dtype::TensorProto_DataType_FLOAT});
                               }

                               onnx_dtype input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                               onnx_dtype input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
                               onnx_dtype output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                               const std::vector<onnx_dtype> typePair{output_data_type, input_0_data_type, input_1_data_type};
                               const auto match = supportedOps.find(typePair);
                               if (match == supportedOps.end())
                                 return true;
                               else
                                 return false;
                             }};
    op_list_.insert({"Equal", obj});
    op_list_.insert({"And", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_1, V_2021_2, V_2021_3},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 const auto& input = node->InputDefs()[0];
                                 auto graph_inputs = graph_viewer_.GetInputs();
                                 auto it = find(graph_inputs.begin(), graph_inputs.end(), input);
                                 if (it != graph_inputs.end()) {
                                   const auto& indices_arg = node->InputDefs()[1];
                                   if (indices_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64)
                                     return true;
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Gather", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 const auto& input = node->InputDefs()[0];
                                 auto graph_inputs = graph_viewer_.GetInputs();
                                 auto it = find(graph_inputs.begin(), graph_inputs.end(), input);
                                 if (it != graph_inputs.end()) {
                                   const auto& indices_arg = node->InputDefs()[1];
                                   if (indices_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT64)
                                     return true;
                                 }
                                 auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 //If the output of Gather op is INT8 or UINT8, it is rejected for GPU.
                                 if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 ||
                                    output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Gather", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               const auto& indices_arg = node->InputDefs()[0];
                               const auto& output_arg = node->OutputDefs()[0];
                               if (indices_arg->TypeAsProto()->tensor_type().elem_type() != output_arg->TypeAsProto()->tensor_type().elem_type())
                                 return true;
                               if ((indices_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) ||
                                   (indices_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)) {
                                 return false;
                               }
                               return true;
                             }};
    op_list_.insert({"GatherElements", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               const auto& input = node->InputDefs()[0];
                               const auto& output = node->OutputDefs()[0];
                               auto graph_inputs = this->graph_viewer_.GetInputs();
                               auto graph_outputs = this->graph_viewer_.GetOutputs();
                               auto input_it = find(graph_inputs.begin(), graph_inputs.end(), input);
                               auto output_it = find(graph_outputs.begin(), graph_outputs.end(), output);
                               if (input_it != graph_inputs.end() && output_it != graph_outputs.end())
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Identity", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               //Loop has to be initializer
                               const auto& cond = node->InputDefs()[1];
                               return (initializers.find(cond->Name()) == initializers.end());
                             }};
    op_list_.insert({"Loop", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //MaxPool "indices" output is not currently supported.
                               //if (node->OutputDefs().size() > 1)
                               //  return true;
                               const auto& attributes = node->GetAttributes();
                               /* default value of ceil_mode (0) is supported.
      auto ceil_attr = attributes.find("ceil_mode");
      if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0)
        return true;*/
                               auto auto_attr = attributes.find("auto_pad");
                               //auto pad null value is not supported
                               if (auto_attr->second().s() == "")
                                 return true;
                               // dilations attrs are not supported in nGraph
                               if (attributes.find("dilations") != attributes.end())
                                 return true;
                               // uint8 as output type for maxpool op is not supported on GPU
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)
                                  return true;
                               }
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"MaxPool", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //MaxPool "indices" output is not currently supported.
                               if (node->OutputDefs().size() > 1)
                                 return true;
                               const auto& attributes = node->GetAttributes();
                               // default value of ceil_mode (0) is supported.
                               auto ceil_attr = attributes.find("ceil_mode");
                               if (ceil_attr != attributes.end() && ceil_attr->second().i() != 0)
                                 return true;
                               auto auto_attr = attributes.find("auto_pad");
                               //auto pad null value is not supported
                               if (auto_attr->second().s() == "")
                                 return true;
                               // dilations attrs are not supported in nGraph
                               if (attributes.find("dilations") != attributes.end())
                                 return true;
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"MaxPool", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (device_id_.find("MYRIAD") == std::string::npos) {
                                 if (GetInputCount(node, initializers) == 1)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Max", obj});
    op_list_.insert({"Min", obj});
    op_list_.insert({"Mean", obj});
    op_list_.insert({"Sum", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (GetInputCount(node, initializers) == 1)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Mean", obj});
    op_list_.insert({"Sum", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               if (GetInputCount(node, initializers) == 1)
                                 return true;
                               for (size_t i = 0; i < node->InputDefs().size(); i++) {
                                 auto dtype = node->InputDefs()[i]->TypeAsProto()->tensor_type().elem_type();
                                 if (dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8 ||
                                     dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT16)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Max", obj});
    op_list_.insert({"Min", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //All matmuls except float have computation missmatch
                               const bool A_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
                               const bool B_is_float = node->InputDefs()[1]->Type()->find("float") != std::string::npos;
                               return (A_is_float && B_is_float) ? false : true;
                             }};
    op_list_.insert({"MatMul", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               // all MatMulInteger zero points need to be constants
                               if (node->InputDefs().size() == 3) {
                                 // not found in initializers -> not const
                                 return initializers.find(node->InputDefs()[2]->Name()) == initializers.end();
                               } else if (node->InputDefs().size() == 4) {
                                 // not found in initializers -> not const
                                 return ((initializers.find(node->InputDefs()[2]->Name()) == initializers.end()) ||
                                         (initializers.find(node->InputDefs()[2]->Name()) == initializers.end()));
                               }
                               return false;
                             }};
    op_list_.insert({"MatMulInteger", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //Only FP32 data type is allowed
                               auto& attributes = node->GetAttributes();
                               auto fmod = attributes.count("fmod") > 0 ? attributes.at("fmod").i() : 0;
                               if (fmod != 1) return true;
                               //Only FP32 data type is allowed
                               for (const auto& input : node->InputDefs()) {
                                 if (input->Type()->find("float") == std::string::npos)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Mod", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                //If the output of Neg op is INT8, it is rejected for GPU.
                                if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8)
                                  return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Neg", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               auto graph_outputs = graph_viewer_.GetOutputs();
                               const auto& output = node->OutputDefs()[0];
                               auto output_it = find(graph_outputs.begin(), graph_outputs.end(), output);
                               if (output_it != graph_outputs.end())
                                 return true;
                               return false;
                             }};
    op_list_.insert({"NonMaxSuppression", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //Only supported if the data type of both inputs is same
                               auto x_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                               auto y_data_type = node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
                               return x_data_type != y_data_type;
                             }};
    op_list_.insert({"Pow", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                 auto x_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 auto y_data_type = node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
                                 return x_data_type != y_data_type;
                               }
                               //currently both inputs with int32 or int64 datatype are not supported
                               const bool A_is_int32 = node->InputDefs()[0]->Type()->find("int32") != std::string::npos;
                               const bool B_is_int32 = node->InputDefs()[1]->Type()->find("int32") != std::string::npos;
                               const bool A_is_int64 = node->InputDefs()[0]->Type()->find("int64") != std::string::npos;
                               const bool B_is_int64 = node->InputDefs()[1]->Type()->find("int64") != std::string::npos;
                               if ((A_is_int32 && B_is_int32) || (A_is_int64 && B_is_int64))
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Pow", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               auto slope = node->InputDefs()[1];
                               //PRelu slope has to be an initializer or needs to come from a constant node
                               if (initializers.count(slope->Name()))
                                 return false;
                               else {
                                 for (auto input_node = node->InputNodesBegin(); input_node != node->InputNodesEnd(); ++input_node) {
                                   if (GetInputCount(this->graph_viewer_.GetNode((*input_node).Index()), initializers) == 0)
                                     return false;
                                 }
                               }
                               return true;
                             }};
    op_list_.insert({"PRelu", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               bool non_const_zero_point = false;
                               // check if any of the zero points is NOT in the initializers list
                               non_const_zero_point |= initializers.find(node->InputDefs()[2]->Name()) == initializers.end();
                               non_const_zero_point |= initializers.find(node->InputDefs()[5]->Name()) == initializers.end();
                               non_const_zero_point |= initializers.find(node->InputDefs()[7]->Name()) == initializers.end();
                               // QLinearMatMul is not supported if any of the zero points is a dynamic input
                               return non_const_zero_point;
                             }};
    op_list_.insert({"QLinearMatMul", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //Only FP32, INT32 and U8 data types are supported
                               const bool data_is_float = node->InputDefs()[0]->Type()->find("float") != std::string::npos;
                               const bool data_is_int32 = node->InputDefs()[0]->Type()->find("int32") != std::string::npos;
                               const bool data_is_u8 = node->InputDefs()[0]->Type()->find("uint8") != std::string::npos;
                               return !(data_is_float || data_is_int32 || data_is_u8);
                             }};
    op_list_.insert({"ReduceMin", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //Resize opset 11 is not supported
                               if (node->InputDefs().size() > 2)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Resize", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("MYRIAD") != std::string::npos) {
                                 const auto& input_arg = node->InputDefs()[1];
                                 auto shape = input_arg->Shape();
                                 //Reshape op with empty dim is Rejected for Myriad
                                 if (shape != nullptr) {
                                   for (const auto& dim : input_arg->Shape()->dim()) {
                                     if (utils::HasDimValue(dim) && dim.dim_value() == 0)
                                       return true;
                                   }
                                 }
                               }
                               return false;
                             }};
    op_list_.insert({"Reshape", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                                if (device_id_.find("GPU") != std::string::npos) {
                                 auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                 //If the output of Transpose op is INT8 or UINT8, it is rejected for GPU.
                                 if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8 || 
                                   output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8)
                                   return true;
                               }
                               return false;
                             }};
    op_list_.insert({"ReduceMax", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               const auto& attributes = node->GetAttributes();
                               auto axis_attr = attributes.find("axis");
                               //Negative axis is not supported
                               if (axis_attr->second().i() < 0)
                                 return true;
                               if (device_id_.find("MYRIAD") != std::string::npos) {
                                 const auto& input_arg = node->InputDefs()[2];
                                 auto updates_shape = input_arg->Shape();
                                 const auto& output_arg = node->OutputDefs()[0];
                                 auto out_shape = output_arg->Shape();
                                 //If updates attribute dim value greater than output_shape dim value, we reject
                                 if(node->InputDefs()[2]->Name() == "updates")
                                 {
                                  size_t updates_size = updates_shape->dim_size();
                                  if(updates_size == 2) {
                                    if(updates_shape->dim(1).dim_value() > out_shape->dim(1).dim_value())
                                      return true;
                                  }
                                  }
                               }
                               return false;
                             }};
    op_list_.insert({"Scatter", obj});
    op_list_.insert({"ScatterElements", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet& initializers) {
                               //start, end, axes need to be a initializer
                               bool cond_for_slice = false;
                               const auto& data_arg = node->InputDefs()[0];
                               auto graph_inputs = graph_viewer_.GetInputs();

                               auto it = find(graph_inputs.begin(), graph_inputs.end(), data_arg);
                               if (it != graph_inputs.end()) {
                                 if (node->InputDefs().size() > 1) {
                                   const auto& start_arg = node->InputDefs()[1];
                                   const auto& end_arg = node->InputDefs()[2];
                                   cond_for_slice |= initializers.find(start_arg->Name()) == initializers.end();
                                   cond_for_slice |= initializers.find(end_arg->Name()) == initializers.end();
                                 }
                                 if (node->InputDefs().size() > 3) {
                                   const auto& axes_arg = node->InputDefs()[3];
                                   cond_for_slice |= initializers.find(axes_arg->Name()) == initializers.end();
                                 }
                               }

                               return cond_for_slice;
                             }};
    op_list_.insert({"Slice", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //Shape can't have empty axes attribute
                               const auto& attributes = node->GetAttributes();
                               if (attributes.count("axes") == 0)
                                 return true;
                               return false;
                             }};
    op_list_.insert({"Squeeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               return node->InputDefs().size() > 1;
                             }};
    op_list_.insert({"TopK", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               if (device_id_.find("GPU") != std::string::npos) {
                                auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
                                //If the output of Transpose op is INT8, it is rejected for GPU.
                                if (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8)
                                  return true;
                               }
                               return false;
                             }};
    op_list_.insert({"Transpose", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2020_4, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"Unsqueeze", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_1, V_2021_2, V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //check for attributes
                               auto& upsample_attr = node->GetAttributes();
                               if (upsample_attr.count("scales") > 0) {
                                 auto& upsample_arg = upsample_attr.at("scales");
                                 auto float_size = upsample_arg.floats_size();
                                 if (float_size > 2 && (upsample_arg.floats(0) != 1.f || upsample_arg.floats(1) != 1.f)) {
                                   return true;
                                 }
                               }

                               //check for input dimensions
                               const auto& x_arg = node->InputDefs()[0];
                               auto shape = x_arg->Shape();
                               if (shape != nullptr) {
                                 //input tensor rank cannot be of one dimension
                                 if (shape->dim_size() == 1) {
                                   return true;
                                 }
                               }
                               // x_arg supports only float, int8 and float16 type
                               if ((x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
                                   (x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT8) ||
                                   (x_arg->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
                                 return false;
                               } else {
                                 return true;
                               }
                             }};
    op_list_.insert({"Upsample", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_2},
                             [this](const Node* node, const InitializedTensorSet&) {
                               //float data type is not supported
                               const bool data_is_float = node->InputDefs()[1]->Type()->find("float") != std::string::npos;
                               return data_is_float;
                             }};
    op_list_.insert({"Where", obj});
  }
  {
    UnsupportedOpMode obj = {{V_2021_3, V_2021_4},
                             [this](const Node* node, const InitializedTensorSet&) {
                               return (!this->dimension_unsupported(node));
                             }};
    op_list_.insert({"ReduceSum", obj});
  }
}

bool DataOps::op_is_supported(std::string name, std::vector<SupportedOp>& op_list) {
  for (size_t i = 0; i < op_list.size(); i++) {
    if (op_list[i].optype == name) {
      if (op_list[i].version <= version_id_) {
        auto it = op_list[i].device_type.begin();
        while (it != op_list[i].device_type.end()) {
          //if device supported is all then we support it
          if (*it == "All") {
            return true;
          }

          //check for device supported
          if (device_id_.find(*it) != std::string::npos) {
            return true;
          }

          it++;
        }
      }
    }
  }

  return false;
}

bool DataOps::type_is_supported(const NodeArg* node_arg, bool is_initializer) {
  const auto* type_proto = node_arg->TypeAsProto();
  if (!type_proto) {
    return false;
  }

  if (is_initializer) {
    auto dtype = type_proto->tensor_type().elem_type();
    for (auto const& var : supported_types_initializer_) {
      if ((var.first <= version_id_) &&
          (var.second == dtype)) {
        return true;
      }
    }

#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Initializer Data Type is not supported" << std::endl;
    }
#endif
    return false;
  } else {
    auto dtype = type_proto->tensor_type().elem_type();

    if (device_id_ == "MYRIAD" || device_id_ == "HDDL" || device_id_.find("HETERO") != std::string::npos || device_id_.find("MULTI") != std::string::npos) {
      for (auto const& var : supported_types_vpu_) {
        if ((var.first <= version_id_) &&
            (var.second == dtype)) {
          return true;
        }
      }

#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::cout << "I/O data type is not supported" << std::endl;
      }
#endif
      return false;

    } else if (device_id_ == "CPU") {
      for (auto const& var : supported_types_cpu_) {
        if ((var.first <= version_id_) &&
            (var.second == dtype)) {
          return true;
        }
      }
#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::cout << "I/O data type is not supported" << std::endl;
      }
#endif
      return false;

    } else if (device_id_ == "GPU") {
      auto prec_str = openvino_ep::BackendManager::GetGlobalContext().precision_str;
      if (prec_str == "FP32" && dtype == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
        return false;
      for (auto const& var : supported_types_gpu_) {
        if ((var.first <= version_id_) &&
            (var.second == dtype)) {
          return true;
        }
      }
#ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        std::cout << "I/O data type is not supported" << std::endl;
      }
#endif
      return false;
    }
    return true;
  }
}

bool DataOps::unsupported_op_mode(const Node* node) {
  bool result = false;
  const auto& optype = node->OpType();
  const auto& initializers = graph_viewer_.GetAllInitializedTensors();

  auto iter = op_list_.equal_range(optype);
  for (auto it = iter.first; it != iter.second; ++it) {
    auto ob = it->second;
    if (std::find(ob.ver.begin(), ob.ver.end(), version_id_) != ob.ver.end()) {
      return ob.func(node, initializers);
    }
  }
  return result;
}

bool DataOps::dimension_unsupported(const Node* node) {
  auto node_inputs = node->InputDefs();
  size_t input_dims = 0;
  if (node_inputs[0]->Shape() == nullptr) {
    return true;
  } else {
    input_dims = node_inputs[0]->Shape()->dim_size();
    if (node->OpType().find("Pool") != std::string::npos) {
      if (input_dims != 4 && input_dims != 5)
        return false;
    }

    if (node->OpType() == "Unsqueeze") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (input_dims + axes_size > 5 || axes_size == 0)
        return false;
    }

    if (node->OpType() == "ReduceSum") {
      auto& attributes = node->GetAttributes();
      int64_t axes_size = attributes.count("axes") > 0 ? attributes.at("axes").ints().size() : 0;
      if (axes_size == 0)
        return false;
    }
  }
  return true;
}

bool DataOps::node_is_supported(const std::map<std::string, std::set<std::string>>& op_map,
                                const NodeIndex node_idx) {
  const auto& node = graph_viewer_.GetNode(node_idx);
  const auto& optype = node->OpType();

#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "Node " << optype << std::endl;
  }
#endif

  const auto& domain = node->Domain();

  /*
  0. Check if node is in the unsupported list
  1. Check input and output data types are supported.
  2. Check if there is unsupported dimension in input and output shapes
  3. Check Op is supported
   3a. Check if Op is of known unsupported modes (edge cases). If yes return false right away.
   3b. If above is not true, check if the op is available in nGraph.
  */

  //Check 0
  if (!op_is_supported(optype, supported_op_mode)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Node is not in the supported ops list" << std::endl;
    }
#endif
    return false;
  }

  //Check 1
  bool are_types_supported = true;

  node->ForEachDef([this, &are_types_supported](const NodeArg& node_arg, bool is_input) {
    bool is_initializer = false;
    if (is_input) {
      if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
        is_initializer = true;
    }
    bool is_supported = type_is_supported(&node_arg, is_initializer);
    are_types_supported &= is_supported;
  });

  if (!are_types_supported) {
    return false;
  }

  //Check 2

  bool has_unsupported_dimension = false;
  node->ForEachDef([&has_unsupported_dimension, this, &optype](const NodeArg& node_arg, bool is_input) {
    if (is_input) {
      if (this->graph_viewer_.IsConstantInitializer(node_arg.Name(), true))
        return;
    }
    auto shape = node_arg.Shape();
    if (shape != nullptr) {
      //Can't have no dimensions
      if (shape->dim_size() == 0) {
        if (op_is_supported(optype, no_dimension_supported_)) {
          return;
        }
        has_unsupported_dimension = true;
        return;
      } else {
        //Zero dimension check
        for (const auto& dim : shape->dim()) {
          if (utils::HasDimValue(dim) && dim.dim_value() == 0) {
            if ((device_id_.find("MYRIAD") != std::string::npos) && (optype == "Resize"))
              return;
            has_unsupported_dimension = true;
            return;
          }
        }
      }
    }
  });
  if (has_unsupported_dimension) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Dimension check failed" << std::endl;
    }
#endif

    return false;
  }

  //Check 3a
  if (domain == kOnnxDomain && unsupported_op_mode(node)) {
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      std::cout << "Failed in unsupported op mode" << std::endl;
    }
#endif
    return false;
  }

  //Check 3b
  const auto opset = op_map.find(domain);
  if (opset == op_map.end() || opset->second.find(optype) == opset->second.end()) {
    return false;
  } else {
    return true;
  }
}

std::vector<NodeIndex> DataOps::GetUnsupportedNodeIndices(std::unordered_set<std::string>& ng_required_initializers) {
  const auto ng_supported_ops = GetNgSupportedOps(GetOnnxOpSet(graph_viewer_));

  std::vector<NodeIndex> unsupported_nodes_idx;

  for (const auto& node_idx : graph_viewer_.GetNodesInTopologicalOrder()) {
    if (node_is_supported(ng_supported_ops, node_idx)) {
      // Collect inputs that are initializers
      graph_viewer_.GetNode(node_idx)->ForEachDef([&ng_required_initializers, this](const NodeArg& node_arg, bool is_input) {
            if(is_input && this->graph_viewer_.GetAllInitializedTensors().count(node_arg.Name())) {
                ng_required_initializers.insert(node_arg.Name());
              } }, true);
    } else {
      unsupported_nodes_idx.push_back(node_idx);
    }
  }
  return unsupported_nodes_idx;
}

bool DataOps::IsOpSupportedOnlyInModel(std::string name) {
  return ops_supported_only_in_model.find(name) != ops_supported_only_in_model.end();
}

bool DataOps::SpecialConditionForClusterSizeOne(std::unordered_set<std::string>& ng_required_initializers, const Node* node) {
  if (node->OpType() == "Where") {
    if (device_id_.find("MYRIAD") != std::string::npos) {
      if (node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT)
        return true;
    }
  } else if (node->OpType() == "Reshape") {
    const auto& shape_arg = node->InputDefs()[1];
    if (ng_required_initializers.find(shape_arg->Name()) == ng_required_initializers.end()) {
      return true;
    }
  } else if (node->OpType() == "Expand") {
    // nGraph only supports constant shape input values
    const auto& output = node->OutputDefs()[0];
    if (output->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)
      return true;
  } else if (node->OpType() == "RoiAlign") {
    using onnx_dtype = ONNX_NAMESPACE::TensorProto_DataType;

    onnx_dtype input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype input_2_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[2]->TypeAsProto()->tensor_type().elem_type();
    onnx_dtype output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

    if ((input_0_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
        (input_1_data_type != onnx_dtype::TensorProto_DataType_FLOAT16) ||
        (input_2_data_type != onnx_dtype::TensorProto_DataType_FLOAT) ||
        (output_data_type != onnx_dtype::TensorProto_DataType_FLOAT16))
      return true;
  } else if ((node->OpType() == "Greater") || (node->OpType() == "Less")) {
    if (device_id_.find("MYRIAD") != std::string::npos) {
      auto input_0_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      auto input_1_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->InputDefs()[1]->TypeAsProto()->tensor_type().elem_type();
      auto output_data_type = (ONNX_NAMESPACE::TensorProto_DataType)node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

      if (!((output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT) ||
            (output_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16))) {
        return true;
      }

      if ((input_0_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) ||
          (input_1_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16)) {
        return true;
      }
    }
  } else if (node->OpType() == "MaxPool" && device_id_.find("MYRIAD") != std::string::npos) {
    auto output_data_type = node->OutputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
    if (output_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT ||
        output_data_type != ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16) {
      return true;
    }
  }
  return false;
}

bool DataOps::DoNotOmitSubGraph(const std::string& name) {
  return op_is_supported(name, subgraph_supported_);
}

bool DataOps::InsertNode(const Node* node, const std::string& optype) {
  if (optype == "TopK" || optype == "NonZero") {
    return true;
  }
  if (optype == "Gather") {
    if (device_id_.find("MYRIAD") != std::string::npos) {
      auto input_data_type = node->InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();
      if (input_data_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_UINT8) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace openvino_ep
}  // namespace onnxruntime

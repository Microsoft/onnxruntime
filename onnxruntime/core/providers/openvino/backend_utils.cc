// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#include <inference_engine.hpp>

#ifdef OPENVINO_2021_4
using Exception = InferenceEngine::Exception;
#else
using Exception = InferenceEngine::details::InferenceEngineException;
#endif

#include <ngraph/frontend/onnx_import/onnx.hpp>
#include <ngraph/pass/convert_fp32_to_fp16.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "core/providers/shared_library/provider_api.h"

#include "backend_utils.h"

namespace onnxruntime {
namespace openvino_ep {
namespace backend_utils {

#ifndef NDEBUG
bool IsDebugEnabled() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("ORT_OPENVINO_ENABLE_DEBUG");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}
void DumpOnnxModelProto(const ONNX_NAMESPACE::ModelProto& model_proto, std::string file_name) {
  std::fstream outfile(file_name, std::ios::out | std::ios::trunc | std::ios::binary);
  model_proto.SerializeToOstream(outfile);
}

#endif

bool UseCompiledNetwork() {
  const std::string env_name = onnxruntime::GetEnvironmentVar("OV_USE_COMPILED_NETWORK");
  if (!env_name.empty()) {
    return true;
  }
  return false;
}

std::string GetCurrentWorkingDir() {
  std::string curr_dir;
  ORT_UNUSED_PARAMETER(curr_dir);
  char buff[FILENAME_MAX];
  curr_dir = GetCurrentDir(buff, FILENAME_MAX);
  std::string current_working_dir(buff);
  return current_working_dir;
}

bool IsDirExists(const std::string& pathname) {
  struct stat info;
  if(stat(pathname.c_str(), &info) != 0) {
    LOGS_DEFAULT(INFO) << log_tag << "cannot access pathname: " << pathname;
	  return false;
  } else if(info.st_mode & S_IFDIR) {
      LOGS_DEFAULT(INFO) << log_tag << "pathname exists: " << pathname;
	    return true;
  } else {
      LOGS_DEFAULT(INFO) << log_tag << "pathname: " << pathname << ": doesn't contain the directory 'ov_compiled_blobs' ";
  }
  return false;
}

void CreateDirectory(const std::string& ov_compiled_blobs_dir) {
  LOGS_DEFAULT(INFO) << log_tag << "'ov_compiled_blobs' directory doesn't exist at the executable path, so creating one";
#if defined(_WIN32)
  if (_mkdir(ov_compiled_blobs_dir.c_str()) == 0) { // Creating a directory 
	  LOGS_DEFAULT(INFO) << log_tag << "created a directory named 'ov_compiled_blobs' at the executable path";
  } else {
    LOGS_DEFAULT(INFO) << log_tag << "Error creating a directory named 'ov_compiled_blobs' at the executable path";
    throw std::runtime_error("Could not create the directory");
  }
#else
  if (mkdir(ov_compiled_blobs_dir.c_str(), 0777) == 0) { // Creating a directory
    LOGS_DEFAULT(INFO) << log_tag << "created a directory named 'ov_compiled_blobs' at the executable path";
  } else {
    LOGS_DEFAULT(INFO) << log_tag << "Error creating a directory named 'ov_compiled_blobs' at the executable path";
    throw std::runtime_error("Could not create the directory");
  }
#endif
}

struct static_cast_int64 {
  template <typename T1>  // T1 models type statically convertible to T
  int64_t operator()(const T1& x) const { return static_cast<int64_t>(x); }
};

std::shared_ptr<InferenceEngine::CNNNetwork>
CreateCNNNetwork(const ONNX_NAMESPACE::ModelProto& model_proto, const GlobalContext& global_context, const SubGraphContext& subgraph_context, std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map) {
  std::shared_ptr<ngraph::Function> ng_function;

#ifndef NDEBUG
  if (IsDebugEnabled()) {
    DumpOnnxModelProto(model_proto, subgraph_context.subgraph_name + "_static.onnx");
  }
#endif

#if (defined OPENVINO_2020_3) || (defined OPENVINO_2021_1) || (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3)
  ORT_UNUSED_PARAMETER(const_outputs_map);
  std::istringstream model_stream{model_proto.SerializeAsString()};
  try {
    ng_function = ngraph::onnx_import::import_onnx_model(model_stream);
    LOGS_DEFAULT(INFO) << "ONNX Import Done";
  } catch (const std::exception& exp) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Exception while importing model to nGraph Func: " + std::string(exp.what()));
  } catch (...) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while importing model to nGraph Func");
  }
#else
  //ReadNetwork() API flow will be used in OpenVINO-EP starting from OpenVINO 2021.4
  InferenceEngine::CNNNetwork cnn_network;
  const std::string model = model_proto.SerializeAsString();
  InferenceEngine::Blob::Ptr blob = {nullptr};
  try {
    cnn_network = global_context.ie_core.ReadNetwork(model, blob);
    LOGS_DEFAULT(INFO) << "Read network Done";
  } catch (const Exception& e) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Exception while Reading network: " + std::string(e.what()));
  } catch (...) {
    ORT_THROW(log_tag + "[OpenVINO-EP] Unknown exception while Reading network");
  }
  ng_function = cnn_network.getFunction();
#endif

  if (global_context.device_type.find("GPU") != std::string::npos &&
      subgraph_context.precision == InferenceEngine::Precision::FP16) {
    //FP16 transformations
    ngraph::pass::ConvertFP32ToFP16().run_on_function(ng_function);
    ng_function->validate_nodes_and_infer_types();
  }

#if (defined OPENVINO_2021_1) || (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3) || (defined OPENVINO_2021_4)
  if (!global_context.is_wholly_supported_graph) {
    std::map<std::string, std::string> result_to_output;
    for (auto& result : ng_function->get_results()) {
      result_to_output[result->get_friendly_name()] = result->input_value(0).get_node_shared_ptr()->get_friendly_name();
    }

    ngraph::pass::ConstantFolding().run_on_function(ng_function);
    auto& results = const_cast<::ngraph::ResultVector&>(ng_function->get_results());
    size_t index = results.size() - 1;
    for (auto it = results.rbegin(); it != results.rend(); ++it) {
      if (auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>((*it)->input_value(0).get_node_shared_ptr())) {
        const_outputs_map[result_to_output.at((*it)->get_friendly_name())] = const_node;
        results.erase(results.begin() + index);
      }
      --index;
    }
  }
#endif

  try {
    return std::make_shared<InferenceEngine::CNNNetwork>(ng_function);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception thrown while making IE::CNNNetwork: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception thrown while making IE::CNNNetwork");
  }
}

InferenceEngine::Precision ConvertPrecisionONNXToOpenVINO(const ONNX_NAMESPACE::TypeProto& onnx_type, std::string device) {
  ONNX_NAMESPACE::DataType type_string = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType(onnx_type);
  if (*type_string == "float" || *type_string == "tensor(float)") {
    return InferenceEngine::Precision::FP32;
  } else if (*type_string == "float16" || *type_string == "tensor(float16)") {
    return InferenceEngine::Precision::FP16;
  } else if (*type_string == "int32" || *type_string == "tensor(int32)") {
    return InferenceEngine::Precision::I32;
  } else if (*type_string == "int16" || *type_string == "tensor(int16)") {
    return InferenceEngine::Precision::I16;
  } else if (*type_string == "int8" || *type_string == "tensor(int8)") {
    return InferenceEngine::Precision::I8;
  } else if (*type_string == "uint16" || *type_string == "tensor(uint16)") {
    return InferenceEngine::Precision::U16;
  } else if (*type_string == "uint8" || *type_string == "tensor(uint8)") {
    return InferenceEngine::Precision::U8;
  } else if (*type_string == "bool" || *type_string == "tensor(bool)") {
    if (device == "MYRIAD") {
      return InferenceEngine::Precision::I32;
    } else {
      return InferenceEngine::Precision::U8;
    }
  } else if (*type_string == "int64" || *type_string == "tensor(int64)") {
    return InferenceEngine::Precision::I32;
  } else {
    ORT_THROW(log_tag + "Unsupported Data type");
  }
}

void SetIODefs(const ONNX_NAMESPACE::ModelProto& model_proto,
               std::shared_ptr<InferenceEngine::CNNNetwork> network,
               std::unordered_map<std::string, int> output_names,
               std::map<std::string, std::shared_ptr<ngraph::Node>>& const_outputs_map,
               std::string device) {
  // Configure input & output
  // Prepare input blobs

#if defined OPENVINO_2020_3
  ORT_UNUSED_PARAMETER(const_outputs_map);
#endif
  auto inputInfo = network->getInputsInfo();
  int input_idx = 0;
  for (auto iter = inputInfo.begin(); iter != inputInfo.end(); ++iter, ++input_idx) {
    // Get the onnx index for the corresponding input (ignoring initializers)
    auto precision = ConvertPrecisionONNXToOpenVINO(model_proto.graph().input(input_idx).type(), device);
    iter->second->setPrecision(precision);
  }

  // Prepare output blobs
  auto outputInfo = network->getOutputsInfo();
  for (auto iter = outputInfo.begin(); iter != outputInfo.end(); ++iter) {
    auto output_name = iter->first;
#if (defined OPENVINO_2021_1) || (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3) || (defined OPENVINO_2021_4)
    auto it = const_outputs_map.find(output_name);
    //Output is constant and don't need to set precision
    if (it != const_outputs_map.end())
      break;
#endif
    auto itr = output_names.find(output_name);
    if (itr == output_names.end()) {
      ORT_THROW(log_tag + "Output Names Mismatch: " + output_name + " doesn't exist");
    }
    auto precision = ConvertPrecisionONNXToOpenVINO(model_proto.graph().output(itr->second).type(), device);
    iter->second->setPrecision(precision);
  }
}

OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context, size_t batch_size,
                InferenceEngine::InferRequest::Ptr infer_request,
                std::string output_name,
                std::unordered_map<std::string, int> output_names) {
  OrtValue* output_tensor;

  auto graph_output_blob = infer_request->GetBlob(output_name);
  auto graph_output_dims = graph_output_blob->getTensorDesc().getDims();
  if (batch_size > 1) {
    // Add the batch size as dim 0.
    graph_output_dims.insert(graph_output_dims.begin(), batch_size);
  }
  size_t num_dims = graph_output_dims.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(graph_output_dims[j]);
  }
  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;

  output_tensor = ort.KernelContext_GetOutput(context, index, output_shape.get(), num_dims);

  return output_tensor;
}

#if (defined OPENVINO_2021_1) || (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3) || (defined OPENVINO_2021_4)
OrtValue*
GetOutputTensor(Ort::CustomOpApi& ort, OrtKernelContext* context,
                std::string output_name,
                std::unordered_map<std::string, int> output_names,
                std::shared_ptr<ngraph::Node> node) {
  OrtValue* output_tensor;
  auto it = output_names.find(output_name);
  if (it == output_names.end()) {
    ORT_THROW(log_tag + "Output names mismatch between OpenVINO and ONNX");
  }
  int index = it->second;
  auto shape = node->get_shape();

  size_t num_dims = shape.size();
  std::unique_ptr<int64_t[]> output_shape(new int64_t[num_dims]);
  for (size_t j = 0; j < num_dims; j++) {
    output_shape[j] = static_cast<int64_t>(shape[j]);
  }
  output_tensor = ort.KernelContext_GetOutput(context, index, output_shape.get(), num_dims);

  return output_tensor;
}
#endif

int GetFirstAvailableDevice(GlobalContext& global_context) {
  int i = 0;
  //Get the first available VAD-M device and set the device to busy
  while (i < 8) {
    bool device = global_context.deviceAvailableList[i];
    if (device) {
      global_context.deviceAvailableList[i] = false;
      break;
    }
    i++;
  }
  //If all of the devices are busy, assign the first device and
  //make all remaining devices free
  if (i == 8) {
    i = 0;
    global_context.deviceAvailableList[i] = false;
    for (int j = 1; j < 8; j++) {
      global_context.deviceAvailableList[j] = true;
    }
  }
  return i;
}

#if (defined OPENVINO_2021_1) || (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3) || (defined OPENVINO_2021_4)
void FillOutputsWithConstantData(Ort::CustomOpApi& ort, std::shared_ptr<ngraph::Node> node, OrtValue* out_tensor) {
  switch (node->get_element_type()) {
    case ngraph::element::Type_t::f32: {
      FillOutputHelper<float>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::boolean: {
      FillOutputHelper<char>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::i32: {
      FillOutputHelper<int32_t>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::i64: {
      FillOutputHelper<int64_t>(ort, out_tensor, node);
      break;
    }
    case ngraph::element::Type_t::f16: {
      FillOutputHelper<float>(ort, out_tensor, node);
      break;
    }
    default:
      ORT_THROW(log_tag + "Unsupported output data type");
  }
}
#endif

#if (defined OPENVINO_2021_1) || (defined OPENVINO_2021_2) || (defined OPENVINO_2021_3) || (defined OPENVINO_2021_4)
template <typename T>
void FillOutputHelper(Ort::CustomOpApi& ort, OrtValue* out_tensor, std::shared_ptr<ngraph::Node> node) {
  auto const_node = std::dynamic_pointer_cast<ngraph::op::Constant>(node);
  auto res = const_node->cast_vector<T>();
  T* tensor_data = ort.GetTensorMutableData<T>(out_tensor);
  std::copy(res.begin(), res.end(), tensor_data);
}
#endif

void FillInputBlob(InferenceEngine::Blob::Ptr& inputBlob, size_t request_id, size_t batch_slice_idx,
                   std::string input_name, Ort::CustomOpApi& ort, OrtKernelContext* context,
                   InferenceEngine::Precision precision, const SubGraphContext& subgraph_context) {
  auto minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inputBlob);
  auto minputHolder = minput->wmap();

  auto input_data = minputHolder.as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
  size_t input_data_size = inputBlob->byteSize();

#if defined OPENVINO_2020_3
  ORT_UNUSED_PARAMETER(input_name);
  const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context.input_indexes[request_id]);
#else
  ORT_UNUSED_PARAMETER(request_id);
  const OrtValue* tensor = ort.KernelContext_GetInput(context, subgraph_context.input_names.at(input_name));
#endif
  auto tensor_shape = ort.GetTensorTypeAndShape(tensor);
  auto elem_type = ort.GetTensorElementType(tensor_shape);

  if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
      (precision == InferenceEngine::Precision::I32)) {
    const int64_t* tensor_data_64 = ort.GetTensorData<int64_t>(tensor);
    auto data_len = (input_data_size * 2) / sizeof(int64_t);
    const int64_t* batch_memory_offset = tensor_data_64 + data_len * batch_slice_idx;

    std::copy(batch_memory_offset, batch_memory_offset + data_len, (uint32_t*)input_data);
  } else {
    // Copy input data into OpenVINO's input buffer
    const char* tensor_data = ort.GetTensorData<char>(tensor);
    const char* batch_memory_offset = tensor_data + input_data_size * batch_slice_idx;
    std::memcpy(input_data, batch_memory_offset, input_data_size);
  }
}

void FillOutputBlob(InferenceEngine::Blob::Ptr& outputBlob, OrtValue* output_tensor,
                    Ort::CustomOpApi& ort, InferenceEngine::Precision precision, size_t batch_slice_idx) {
  auto moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outputBlob);

  auto moutputHolder = moutput->rmap();

  const auto output_data = moutputHolder.as<const InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();

  size_t output_data_size = outputBlob->byteSize();
  auto tensor_shape = ort.GetTensorTypeAndShape(output_tensor);
  auto elem_type = ort.GetTensorElementType(tensor_shape);

  if ((elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) &&
      (precision == InferenceEngine::Precision::I32)) {
    int64_t* tensor_data = ort.GetTensorMutableData<int64_t>(output_tensor);
    auto data_len = output_data_size / sizeof(int32_t);
    int64_t* batch_memory_offset = tensor_data + data_len * batch_slice_idx;

    std::transform((int32_t*)output_data, ((int32_t*)output_data) + data_len, batch_memory_offset, static_cast_int64());

  } else {
    char* tensor_data = ort.GetTensorMutableData<char>(output_tensor);
    char* batch_memory_offset = tensor_data + output_data_size * batch_slice_idx;

    std::memcpy(batch_memory_offset, output_data, output_data_size);
  }
}

std::vector<std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>>
perfCountersSorted(std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfMap) {
  using perfItem = std::pair<std::string, InferenceEngine::InferenceEngineProfileInfo>;
  std::vector<perfItem> sorted;
  for (auto& kvp : perfMap) sorted.push_back(kvp);

  std::stable_sort(sorted.begin(), sorted.end(),
                   [](const perfItem& l, const perfItem& r) {
                     return l.second.execution_index < r.second.execution_index;
                   });

  return sorted;
}

void printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performanceMap,
                            std::ostream& stream, std::string deviceName) {
  long long totalTime = 0;
  // Print performance counts
  stream << std::endl
         << "performance counts:" << std::endl
         << std::endl;

  auto performanceMapSorted = perfCountersSorted(performanceMap);

  for (const auto& it : performanceMapSorted) {
    std::string toPrint(it.first);
    const int maxLayerName = 30;

    if (it.first.length() >= maxLayerName) {
      toPrint = it.first.substr(0, maxLayerName - 4);
      toPrint += "...";
    }
    stream << std::setw(maxLayerName) << std::left << toPrint;
    switch (it.second.status) {
      case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
        stream << std::setw(15) << std::left << "EXECUTED";
        break;
      case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
        stream << std::setw(15) << std::left << "NOT_RUN";
        break;
      case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
        stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
        break;
    }
    stream << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
    stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
    stream << std::setw(20) << std::left << "cpu: " + std::to_string(it.second.cpu_uSec);
    stream << " execType: " << it.second.exec_type << std::endl;
    if (it.second.realTime_uSec > 0) {
      totalTime += it.second.realTime_uSec;
    }
  }
  stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
  std::cout << std::endl;
  std::cout << "Full device name: " << deviceName << std::endl;
  std::cout << std::endl;
}

void printPerformanceCounts(InferenceEngine::InferRequest::Ptr request, std::ostream& stream, std::string deviceName) {
  auto performanceMap = request->GetPerformanceCounts();
  printPerformanceCounts(performanceMap, stream, deviceName);
}

void printPerformanceCounts(InferenceEngine::InferRequest request, std::ostream& stream, std::string deviceName) {
  auto performanceMap = request.GetPerformanceCounts();
  printPerformanceCounts(performanceMap, stream, deviceName);
}

}  // namespace backend_utils
}  // namespace openvino_ep
}  // namespace onnxruntime

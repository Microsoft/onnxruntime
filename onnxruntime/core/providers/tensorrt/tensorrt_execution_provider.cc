// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <fstream>
#include <list>
#include <unordered_set>
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#include "core/common/safeint.h"
#include "tensorrt_execution_provider.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "cuda_runtime_api.h"
#include "gsl/gsl"
#include <experimental/filesystem>
#include <unordered_map>
#include <utility>
#include <limits>
#include <map>
#include <memory>
#include "flatbuffers/idl.h"
#include "ort_trt_int8_cal_table.fbs.h"

#ifdef _WIN32
#include <windows.h>
#define LIBTYPE HINSTANCE
#define OPENLIB(libname) LoadLibrary(libname)
#define LIBFUNC(lib, fn) GetProcAddress((lib), (fn))
#else
#include <dlfcn.h>
#define LIBTYPE void*
#define OPENLIB(libname) dlopen((libname), RTLD_LAZY)
#define LIBFUNC(lib, fn) dlsym((lib), (fn))
#endif

#define CUDA_RETURN_IF_ERROR(expr)               \
  ORT_RETURN_IF_ERROR(CUDA_CALL(expr)            \
                          ? common::Status::OK() \
                          : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error executing ", #expr))

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
namespace fs = std::experimental::filesystem;
namespace {
std::string GetCachePath(const std::string& root, const std::string& name) {
  if (root.empty()) {
    return name;
  } else {
    fs::path path = root;
    path.append(name);
    return path.string();
  }
}

float ConvertSinglePrecisionIEEE754ToFloat(unsigned long input) {
  int s = (input >> 31) & 0x01;
  int e = ((input & 0x7f800000) >> 23) - 127;
  int p = -1;
  double m = 0.0;
  for (int i = 0; i < 23; ++i) {
    m += ((input >> (23 - i - 1)) & 0x01) * pow(2.0, p--);
  }
  return (s ? -1 : 1) * pow(2.0, e) * (m + 1.0);
}

/*
* Seralize engine profile
* The profile contains min/max shape ranges of dynamic shape dimensions of each input tensor
* For example, assume tensor_a has two dynamic shape dimensions: dim_0 and dim_2, and tensor_b
* has one dynamic shape dimension: dim_1. The data in profile will be,
* key: tensor_a, value: dim_0 min_shape max_shape dim_2 min_shape max_shape
* key: tensor_b, value: dim_1 min_shape max_shape
*/
void SerializeProfile(const std::string& file_name, std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>>& shape_ranges) {
  // Serialize profile
  flexbuffers::Builder builder;
  auto profile_start = builder.StartMap();
  for (auto outer_it = shape_ranges.begin(); outer_it != shape_ranges.end(); ++outer_it) {
    builder.TypedVector(outer_it->first.c_str(), [&] {
      for (auto inner_it = outer_it->second.begin(); inner_it != outer_it->second.end(); ++inner_it) {
        builder.Int(inner_it->first);
        builder.Int(inner_it->second.first);
        builder.Int(inner_it->second.second);
      }
    });
  }
  builder.EndMap(profile_start);
  builder.Finish();

  // Save flexbuffer
  std::ofstream file(file_name, std::ios::binary | std::ios::out);
  auto buf = builder.GetBuffer();
  size_t size = builder.GetSize();
  file.write(reinterpret_cast<const char*>(&buf[0]), size);
  file.close();
}

// Deserialize engine profile
std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>> DeserializeProfile(std::ifstream& infile) {
  // Load flexbuffer
  infile.seekg(0, std::ios::end);
  int length = infile.tellg();
  infile.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data{new char[length]};
  infile.read((char*)data.get(), length);
  infile.close();

  // Deserialize profile
  std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>> shape_ranges;
  auto tensors_range_entries = flexbuffers::GetRoot((const uint8_t*)data.get(), length).AsMap();
  auto keys = tensors_range_entries.Keys();
  auto values = tensors_range_entries.Values();
  for (size_t i = 0, end = keys.size(); i < end; ++i) {
    auto dim_range_vectors = values[i].AsTypedVector();
    std::unordered_map<int, std::pair<int64_t, int64_t>> inner_map;
    for (size_t j = 0, end = dim_range_vectors.size() / 3; j < end; ++j) {
      size_t idx = 3 * j;
      inner_map[dim_range_vectors[idx].AsInt64()] = std::make_pair(dim_range_vectors[idx + 1].AsInt64(), dim_range_vectors[idx + 2].AsInt64());
    }
    shape_ranges[keys[i].AsString().c_str()] = inner_map;
  }
  return shape_ranges;
}

// Check if cycle exists in the graph after partitioning
bool FindCycleHelper(int i, const std::list<int>* adjacency_map, bool visited[], bool* st, std::vector<int>& cycles) {
  if (!visited[i]) {
    visited[i] = true;
    st[i] = true;
    for (auto iter = adjacency_map[i].begin(); iter != adjacency_map[i].end(); ++iter) {
      if (!visited[*iter] && FindCycleHelper(*iter, adjacency_map, visited, st, cycles)) {
        cycles.push_back(*iter);
        return true;
      } else if (st[*iter]) {
        cycles.push_back(*iter);
        return true;
      }
    }
  }
  st[i] = false;
  return false;
}

/*
* Read calibration table for INT8 quantization
* Two kind of calibration tables are supported,
* 1. ORT generated calibration table
* The table is pre-serialized by flatbuffers. 
* Each entry in the table is a key-value pair,
* key: tensor name, value: maximum absolute value in floating point
* For example,
*   data_0 2.008338
*   ...
* 2. Native TensorRT generated calibration table
* Data format is defined by TensorRT as,
* tensor name : scale in 32-bit single precision IEEE754 format
* For example,
*   TRT-7103-EntropyCalibration2
*   data_0: 4000889d
*   ...
*/
bool ReadDynamicRange(const std::string file_name, const bool is_trt_calibration_table, std::unordered_map<std::string, float>& dynamic_range_map) {
  std::ifstream infile(file_name, std::ios::binary | std::ios::in);
  if (!infile) {
    return false;
  }

  if (is_trt_calibration_table) {
    // Native TensorRT generated calibration table
    std::string line;
    char delim = ':';
    if (std::getline(infile, line)) {
      std::istringstream first_line(line);
      std::string version;
      std::getline(first_line, version, delim);
      std::size_t found = version.find("TRT-");
      if (found != std::string::npos) {
        while (std::getline(infile, line)) {
          std::istringstream in_line(line);
          std::string str;
          std::getline(in_line, str, delim);
          std::string tensor_name = str;
          std::getline(in_line, str, delim);
          unsigned long scale_int = std::strtoul(str.c_str(), nullptr, 16);
          float scale_float = ConvertSinglePrecisionIEEE754ToFloat(scale_int);
          float dynamic_range = scale_float * 127.0;
          dynamic_range_map[tensor_name] = dynamic_range;
        }
      } else {
        throw std::runtime_error("This is not a TensorRT generated calibration table " + file_name);
      }
    }
  } else {
    // ORT generated calibration table
    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    std::unique_ptr<char[]> data{new char[length]};
    infile.read((char*)data.get(), length);
    infile.close();
    auto flat_table = flatbuffers::GetRoot<CalTableFlatBuffers::TrtTable>((const uint8_t*)data.get());
    auto flat_dict = flat_table->dict();
    for (size_t i = 0, end = flat_dict->size(); i < end; ++i) {
      dynamic_range_map[flat_dict->Get(i)->key()->str()] = std::stof(flat_dict->Get(i)->value()->str());
    }
  }
  return true;
}

bool SetDynamicRange(nvinfer1::INetworkDefinition& network, std::unordered_map<std::string, float>& dynamic_range_map) {
  // Set dynamic range for input tensors
  for (int i = 0; i < network.getNbInputs(); ++i) {
    const std::string tensor_name = network.getInput(i)->getName();
    auto dynamic_range_iter = dynamic_range_map.find(tensor_name);
    if (dynamic_range_iter != dynamic_range_map.end()) {
      if (!network.getInput(i)->setDynamicRange(-dynamic_range_iter->second, dynamic_range_iter->second)) {
        return false;
      }
    }
  }

  // Set dynamic range for activations and weights
  for (int i = 0; i < network.getNbLayers(); ++i) {
    auto trt_layer = network.getLayer(i);
    for (int j = 0, e = trt_layer->getNbOutputs(); j < e; ++j) {
      const std::string tensor_name = trt_layer->getOutput(j)->getName();
      auto dynamic_range_iter = dynamic_range_map.find(tensor_name);
      if (dynamic_range_iter != dynamic_range_map.end()) {
        if (!trt_layer->getOutput(j)->setDynamicRange(-dynamic_range_iter->second, dynamic_range_iter->second)) {
          return false;
        }
      } else if (trt_layer->getType() == nvinfer1::LayerType::kCONSTANT) {
        nvinfer1::IConstantLayer* const_layer = static_cast<nvinfer1::IConstantLayer*>(trt_layer);
        auto trt_weights = const_layer->getWeights();
        double max_weight = std::numeric_limits<double>::min();
        for (int64_t k = 0, end = trt_weights.count; k < end; ++k) {
          double weight{};
          switch (trt_weights.type) {
            case nvinfer1::DataType::kFLOAT:
              weight = static_cast<const float*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kBOOL:
              weight = static_cast<const bool*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kINT8:
              weight = static_cast<const int8_t*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kHALF:
              weight = static_cast<const uint16_t*>(trt_weights.values)[k];
              break;
            case nvinfer1::DataType::kINT32:
              weight = static_cast<const int32_t*>(trt_weights.values)[k];
              break;
          }
          max_weight = std::max(max_weight, std::abs(weight));
        }
        if (!trt_layer->getOutput(j)->setDynamicRange(-max_weight, max_weight)) {
          return false;
        }
      }
    }
  }
  return true;
}
}  // namespace

namespace google {
namespace protobuf {
void ShutdownProtobufLibrary();
}
}  // namespace google

struct ShutdownProtobuf {
  ~ShutdownProtobuf() {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
} g_protobuf;

namespace onnxruntime {

namespace cuda {
template <>
void Impl_Cast(
    cudaStream_t stream,
    const int64_t* input_data, int32_t* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(static_cast<void*>(stream), input_data, output_data, count);
}

template <>
void Impl_Cast(
    cudaStream_t stream,
    const int32_t* input_data, int64_t* output_data,
    size_t count) {
  return g_host->cuda__Impl_Cast(static_cast<void*>(stream), input_data, output_data, count);
}

}  // namespace cuda

template <>
bool CudaCall<cudaError, false>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg) {
  return g_host->CudaCall_false(retCode, exprString, libName, successCode, msg);
}

template <>
bool CudaCall<cudaError, true>(cudaError retCode, const char* exprString, const char* libName, cudaError successCode, const char* msg) {
  return g_host->CudaCall_true(retCode, exprString, libName, successCode, msg);
}

class Memcpy final : public OpKernel {
 public:
  Memcpy(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* ctx) const override {
    const auto* X = ctx->Input<Tensor>(0);
    Tensor* Y = ctx->Output(0, X->Shape());
    Status retval = Info().GetDataTransferManager().CopyTensor(*X, *Y, Info().GetKernelDef().ExecQueueId());
    return retval;
  }
};

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

ONNX_OPERATOR_KERNEL_EX(
    MemcpyFromHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)
        .ExecQueueId(kCudaStreamCopyIn)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

ONNX_OPERATOR_KERNEL_EX(
    MemcpyToHost,
    kOnnxDomain,
    1,
    kTensorrtExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUOutput, 0)
        .ExecQueueId(kCudaStreamCopyOut)
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    Memcpy);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost);

static Status RegisterTensorrtKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyFromHost)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kTensorrtExecutionProvider, kOnnxDomain, 1, MemcpyToHost)>,
  };

  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(kernel_registry.Register(function_table_entry()));
  }
  return Status::OK();
}

static std::shared_ptr<KernelRegistry> s_kernel_registry;

void Shutdown_DeleteRegistry() {
  s_kernel_registry.reset();
}

std::shared_ptr<KernelRegistry> TensorrtExecutionProvider::GetKernelRegistry() const {
  if (!s_kernel_registry) {
    s_kernel_registry = KernelRegistry::Create();
    auto status = RegisterTensorrtKernels(*s_kernel_registry);
    if (!status.IsOK())
      s_kernel_registry.reset();
    ORT_THROW_IF_ERROR(status);
  }

  return s_kernel_registry;
}

// Per TensorRT documentation, logger needs to be a singleton.
TensorrtLogger& GetTensorrtLogger() {
  static TensorrtLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
  return trt_logger;
}

TensorrtExecutionProvider::TensorrtExecutionProvider(const TensorrtExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kTensorrtExecutionProvider, true}, device_id_(info.device_id) {
  CUDA_CALL_THROW(cudaSetDevice(device_id_));
  if (info.has_user_compute_stream) {
    external_stream_ = true;
    stream_ = static_cast<cudaStream_t>(info.user_compute_stream);
  } else {
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  // Get environment variables
  if (info.has_trt_options) {
    max_partition_iterations_ = info.max_partition_iterations;
  } else {
    const std::string max_partition_iterations_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMaxPartitionIterations);
    if (!max_partition_iterations_env.empty()) {
      max_partition_iterations_ = std::stoi(max_partition_iterations_env);
    }
  }

  if (info.has_trt_options) {
    min_subgraph_size_ = info.min_subgraph_size;
  } else {
    const std::string min_subgraph_size_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMinSubgraphSize);
    if (!min_subgraph_size_env.empty()) {
      min_subgraph_size_ = std::stoi(min_subgraph_size_env);
    }
  }

  if (info.has_trt_options) {
    max_workspace_size_ = info.max_workspace_size;
  } else {
    const std::string max_workspace_size_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kMaxWorkspaceSize);
    if (!max_workspace_size_env.empty()) {
      max_workspace_size_ = std::stoull(max_workspace_size_env);
    }
  }

  if (info.has_trt_options) {
    fp16_enable_ = info.fp16_enable;
  } else {
    const std::string fp16_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kFP16Enable);
    if (!fp16_enable_env.empty()) {
      fp16_enable_ = (std::stoi(fp16_enable_env) == 0 ? false : true);
    }
  }

  if (info.has_trt_options) {
    int8_enable_ = info.int8_enable;
  } else {
    const std::string int8_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kINT8Enable);
    if (!int8_enable_env.empty()) {
      int8_enable_ = (std::stoi(int8_enable_env) == 0 ? false : true);
    }
  }

  if (int8_enable_) {
    if (info.has_trt_options) {
      int8_calibration_cache_name_ = info.int8_calibration_table_name;
    } else {
      const std::string int8_calibration_cache_name_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kINT8CalibrationTableName);
      if (!int8_calibration_cache_name_env.empty()) {
        int8_calibration_cache_name_ = int8_calibration_cache_name_env;
      }
    }

    if (info.has_trt_options) {
      int8_use_native_tensorrt_calibration_table_ = info.int8_use_native_calibration_table;
    } else {
      const std::string int8_use_native_tensorrt_calibration_table_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kINT8UseNativeTensorrtCalibrationTable);
      if (!int8_use_native_tensorrt_calibration_table_env.empty()) {
        int8_use_native_tensorrt_calibration_table_ = (std::stoi(int8_use_native_tensorrt_calibration_table_env) == 0 ? false : true);
      }
    }
  }

  if (info.has_trt_options) {
    dump_subgraphs_ = info.dump_subgraphs;
  } else {
    const std::string dump_subgraphs_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDumpSubgraphs);
    if (!dump_subgraphs_env.empty()) {
      dump_subgraphs_ = (std::stoi(dump_subgraphs_env) == 0 ? false : true);
    }
  }

  if (info.has_trt_options) {
    engine_cache_enable_ = info.engine_cache_enable;
  } else {
    const std::string engine_cache_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCacheEnable);
    if (!engine_cache_enable_env.empty()) {
      engine_cache_enable_ = (std::stoi(engine_cache_enable_env) == 0 ? false : true);
    }
  }

  if (engine_cache_enable_ || int8_enable_) {
    const std::string engine_cache_path = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kEngineCachePath);
    if (info.has_trt_options) {
      cache_path_ = info.cache_path;
    } else {
      cache_path_ = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kCachePath);
    }

    if (!engine_cache_path.empty() && cache_path_.empty()) {
      cache_path_ = engine_cache_path;
      LOGS_DEFAULT(WARNING) << "[TensorRT EP] ORT_TENSORRT_ENGINE_CACHE_PATH is deprecated! Please use ORT_TENSORRT_CACHE_PATH to specify engine cache path";
    }
    if (!cache_path_.empty() && !fs::is_directory(cache_path_)) {
      if (!fs::create_directory(cache_path_)) {
        throw std::runtime_error("Failed to create directory " + cache_path_);
      }
    }
    runtime_ = tensorrt_ptr::unique_pointer<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(GetTensorrtLogger()));
  }

  const std::string engine_decryption_enable_env = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDecryptionEnable);
  if (!engine_decryption_enable_env.empty()) {
    engine_decryption_enable_ = (std::stoi(engine_decryption_enable_env) == 0 ? false : true);
  }

  if (engine_decryption_enable_) {
    std::string engine_decryption_lib_path = onnxruntime::GetEnvironmentVar(tensorrt_env_vars::kDecryptionLibPath);
    LIBTYPE handle = OPENLIB(engine_decryption_lib_path.c_str());
    if (handle == nullptr) {
      ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                      "TensorRT EP could not open shared library from " + engine_decryption_lib_path);
    }
    engine_decryption_ = (int (*)(const char*, char*, size_t*))LIBFUNC(handle, "decrypt");
  }
}

TensorrtExecutionProvider::~TensorrtExecutionProvider() {
  if (!external_stream_ && stream_) {
    CUDA_CALL(cudaStreamDestroy(stream_));
  }
}

AllocatorPtr TensorrtExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  if (mem_type == OrtMemTypeDefault) {
    return allocator_;
  } else {
    return IExecutionProvider::GetAllocator(id, mem_type);
  }
}

void TensorrtExecutionProvider::RegisterAllocator(std::shared_ptr<AllocatorManager> allocator_manager) {
  allocator_ = AllocatorManager__GetAllocator(allocator_manager.get(), device_id_, OrtMemTypeDefault);
  if (nullptr == allocator_) {
    AllocatorCreationInfo default_memory_info(
        [](OrtDevice::DeviceId device_id) { return CreateCUDAAllocator(device_id, onnxruntime::CUDA); }, device_id_);
    allocator_ = CreateAllocator(default_memory_info);
    AllocatorManager__InsertAllocator(allocator_manager.get(), allocator_);
  }
  TryInsertAllocator(allocator_);

  auto cuda_pinned_alloc = AllocatorManager__GetAllocator(allocator_manager.get(), DEFAULT_CPU_ALLOCATOR_DEVICE_ID, OrtMemTypeCPUOutput);
  if (nullptr == cuda_pinned_alloc) {
    AllocatorCreationInfo pinned_allocator_info(
        [](OrtDevice::DeviceId device_id) {
          return CreateCUDAPinnedAllocator(device_id, onnxruntime::CUDA_PINNED);
        },
        DEFAULT_CPU_ALLOCATOR_DEVICE_ID);
    cuda_pinned_alloc = CreateAllocator(pinned_allocator_info);
    AllocatorManager__InsertAllocator(allocator_manager.get(), cuda_pinned_alloc);
  }
  TryInsertAllocator(cuda_pinned_alloc);
}

std::unique_ptr<IDataTransfer> TensorrtExecutionProvider::GetDataTransfer() const {
  return onnxruntime::CreateGPUDataTransfer(static_cast<void*>(GetComputeStream()));
}

Status TensorrtExecutionProvider::OnRunEnd() {
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(GetComputeStream())));
  return Status::OK();
}

Status TensorrtExecutionProvider::SetComputeStream(void* stream) {
  if (stream != stream_) {
    if (stream_) {
      CUDA_RETURN_IF_ERROR(cudaStreamDestroy(stream_));
    }

    external_stream_ = true;
    stream_ = static_cast<cudaStream_t>(stream);
  }
  return Status::OK();
}

// Convert GraphViewer graph to GraphProto
void ToGraphProtoInternal(const GraphViewer& graph, ONNX_NAMESPACE::GraphProto& graph_proto) {
  for (const auto* input_arg : graph.GetInputs()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  // Add all graph's initializers to the subgraph
  const auto& init_tensors = graph.GetAllInitializedTensors();
  for (const auto& tensor : init_tensors) {
    *(graph_proto.mutable_initializer()->Add()) = *(tensor.second);
  }

  for (const auto* output_arg : graph.GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : graph.GetValueInfo()) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph.GetNodesInTopologicalOrder()) {
    const gsl::not_null<ONNX_NAMESPACE::NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{graph.GetNode(node_idx)};
    p_node->ToProto(*node_proto);
  }
}

std::unique_ptr<IndexedSubGraph> TensorrtExecutionProvider::GetSubGraph(SubGraph_t graph_nodes_index, const GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  std::unordered_set<size_t> node_set;
  node_set.reserve(graph_nodes_index.first.size());
  for (const auto& index : graph_nodes_index.first) {
    node_set.insert(node_index[index]);
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  // Find inputs and outputs of the subgraph
  std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::IndexedSubGraph::Create();
  std::unordered_map<const NodeArg*, int> fused_inputs, fused_outputs, fused_outputs_to_add, graph_outputs_to_add;
  std::unordered_set<const NodeArg*> erased;
  int input_order = 0;
  int output_order = 0;

  for (const auto& index : graph_nodes_index.first) {
    sub_graph->Nodes().push_back(node_index[index]);
    const auto& node = graph.GetNode(node_index[index]);
    for (const auto& input : node->InputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    for (const auto& input : node->ImplicitInputDefs()) {
      const auto& it = fused_outputs.find(input);
      if (it != fused_outputs.end()) {
        fused_outputs.erase(it);
        erased.insert(input);
      } else if (erased.find(input) == erased.end()) {
        // Only when input is neither in output list nor erased list, add the input to input list
        fused_inputs[input] = input_order++;
      }
    }

    // For output searching, there are two special cases,
    // One is, if node's OutputEdges are more than its outputs, meaning certain output is used more than once,
    // if the output is connected to nodes that don't belong to the subgraph, the output need to be added
    // to the output list
    // The other one is, if subgraph's node output is parent graph's output. the node output should
    // be also added to the subgraph's output list
    if (node->GetOutputEdgesCount() > node->OutputDefs().size()) {
      for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
        const auto& node_idx = it->GetNode().Index();
        const auto& output = (it->GetNode()).InputDefs()[it->GetDstArgIndex()];
        if (node_set.find(node_idx) != node_set.end()) {
          const auto& iter = fused_inputs.find(output);
          if (iter != fused_inputs.end()) {
            fused_inputs.erase(iter);
            erased.insert(output);
          } else if (erased.find(output) == erased.end()) {
            if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
              graph_outputs_to_add[output] = output_order;
            }
            fused_outputs[output] = output_order++;
          }
        } else {
          fused_outputs_to_add[output] = output_order++;
        }
      }
    } else {
      for (const auto& output : node->OutputDefs()) {
        const auto& it = fused_inputs.find(output);
        if (it != fused_inputs.end()) {
          fused_inputs.erase(it);
          erased.insert(output);
        }
        // Only when output is neither in input list nor erased list, add the output to output list
        else if (erased.find(output) == erased.end()) {
          if (graph_output_names.find(output->Name()) != graph_output_names.end()) {
            graph_outputs_to_add[output] = output_order;
          }
          fused_outputs[output] = output_order++;
        }
      }
    }
  }

  fused_outputs.insert(fused_outputs_to_add.begin(), fused_outputs_to_add.end());
  fused_outputs.insert(graph_outputs_to_add.begin(), graph_outputs_to_add.end());

  // Sort inputs and outputs by the order they were added
  std::multimap<int, const NodeArg*> inputs, outputs;
  for (auto it = fused_inputs.begin(), end = fused_inputs.end(); it != end; ++it) {
    inputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  for (auto it = fused_outputs.begin(), end = fused_outputs.end(); it != end; ++it) {
    outputs.insert(std::pair<int, const NodeArg*>(it->second, it->first));
  }

  // Generate unique kernel name for TRT subgraph
  uint64_t model_hash = 0;
  int id = GenerateMetaDefId(graph, model_hash);
  std::string subgraph_id = std::to_string(model_hash) + "_" + std::to_string(id);
  auto meta_def = IndexedSubGraph_MetaDef::Create();
  const std::string graph_type = graph.IsSubgraph() ? "subgraph" : "graph";
  meta_def->name() = "TRTKernel_" + graph_type + "_" + graph.Name() + "_" + subgraph_id;

  // Assign inputs and outputs to subgraph's meta_def
  for (const auto& input : inputs) {
    if (input.second->Exists()) {
      meta_def->inputs().push_back(input.second->Name());
    }
  }

  for (const auto& output : outputs) {
    if (output.second->Exists()) {
      meta_def->outputs().push_back(output.second->Name());
    }
  }

  meta_def->domain() = kMSDomain;
  meta_def->since_version() = 1;
  sub_graph->SetMetaDef(std::move(meta_def));

  return sub_graph;
}

SubGraphCollection_t TensorrtExecutionProvider::GetSupportedList(SubGraphCollection_t nodes_vector_input, int iterations, const int max_iterations,
                                                                 const GraphViewer& graph, bool* early_termination) const {
  // Return if iterations are exceeding predefined number
  SubGraphCollection_t nodes_list_output;
  if (iterations > max_iterations) {
    *early_termination = true;
    return nodes_list_output;
  }

  // Get parent graph output names
  std::unordered_set<std::string> graph_output_names;
  for (const auto* output_arg : graph.GetOutputs()) {
    graph_output_names.insert(output_arg->Name());
  }

  iterations++;
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  for (const auto& group : nodes_vector_input) {
    // Construct subgraph
    if (!group.first.empty()) {
      if (group.second) {
        nodes_list_output.push_back(group);
      } else {
        auto model_build = graph.CreateModel(*GetLogger());
        auto& graph_build = model_build->MainGraph();

        // Add node and node args
        // If node output is also parent graph output, the  output will be added to the
        // subgraph's output list
        std::vector<std::string> subgraph_output_names;
        for (const auto& index : group.first) {
          const auto& node = graph.GetNode(node_index[index]);
          std::vector<onnxruntime::NodeArg*> inputs, outputs;
          for (auto input : node->InputDefs()) {
            auto& n_input = graph_build.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
            inputs.push_back(&n_input);
            const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
            if (graph.GetInitializedTensor(input->Name(), initializer)) {
              const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
              if (!graph_build.GetInitializedTensor(input->Name(), subgraph_initializer)) {
                graph_build.AddInitializedTensor(*(initializer));
              }
            }
          }

          for (auto input : node->ImplicitInputDefs()) {
            const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
            if (graph.GetInitializedTensor(input->Name(), initializer)) {
              const ONNX_NAMESPACE::TensorProto* subgraph_initializer = nullptr;
              if (!graph_build.GetInitializedTensor(input->Name(), subgraph_initializer)) {
                graph_build.AddInitializedTensor(*(initializer));
              }
            }
          }
          for (auto output : node->OutputDefs()) {
            auto& n_output = graph_build.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
            outputs.push_back(&n_output);
            const auto name = output->Name();
            if (graph_output_names.find(name) != graph_output_names.end()) {
              subgraph_output_names.push_back(name);
            }
          }
          graph_build.AddNode(node->Name(), node->OpType(), node->Description(), inputs, outputs, &node->GetAttributes(), node->Domain());
        }

        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Add parent graph output to the subgraph
        int i = 0;
        std::vector<const NodeArg*> subgraph_outputs;
        subgraph_outputs.resize(subgraph_output_names.size());
        for (auto& name : subgraph_output_names) {
          auto output_arg = graph.GetNodeArg(name);
          auto& subgraph_output_arg = graph_build.GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
          subgraph_outputs[i] = &subgraph_output_arg;
          ++i;
        }
        auto& graph_build_outputs = graph_build.GetOutputs();
        subgraph_outputs.insert(subgraph_outputs.begin(), graph_build_outputs.begin(), graph_build_outputs.end());
        graph_build.SetOutputs(graph_build_outputs);
        ORT_ENFORCE(graph_build.Resolve().IsOK());

        // Check if input tensors have shapes
        if (iterations > 1) {
          for (const auto* input_arg : graph_build.GetInputs()) {
            if (input_arg->Shape() == nullptr) {
              ORT_THROW_IF_ERROR(ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                                 "TensorRT input: " + input_arg->Name() + " has no shape specified. " +
                                                     "Please run shape inference on the onnx model first. Details can be found in " +
                                                     "https://github.com/microsoft/onnxruntime/blob/master/docs/execution_providers/TensorRT-ExecutionProvider.md#shape-inference-for-tensorrt-subgraphs"));
            }
          }
        }

        // Serialize modelproto to string
        auto graph_viewer = graph_build.CreateGraphViewer();
        auto model = graph_viewer->CreateModel(*GetLogger());
        auto model_proto = model->ToProto();
        ToGraphProtoInternal(*graph_viewer, *model_proto->mutable_graph());
        model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

        std::string string_buf;
        model_proto->SerializeToString(string_buf);

        if (dump_subgraphs_) {
          // Dump TensorRT subgraph for debugging if enabled via ORT_TENSORRT_DUMP_SUBGRAPHS env variable.
          std::fstream dump("TensorrtExecutionProvider_TRT_Subgraph.onnx", std::ios::out | std::ios::trunc | std::ios::binary);
          model_proto->SerializeToOstream(dump);
        }

        // Get supported node list recursively
        SubGraphCollection_t parser_nodes_list;
        TensorrtLogger& trt_logger = GetTensorrtLogger();
        auto trt_builder = tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto trt_network = tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));

        auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
        trt_parser->supportsModel(string_buf.data(), string_buf.size(), parser_nodes_list, model_path_);

        SubGraphCollection_t next_nodes_list;
        const std::vector<NodeIndex>& subgraph_node_index = graph_viewer->GetNodesInTopologicalOrder();
        next_nodes_list = GetSupportedList(parser_nodes_list, iterations, max_iterations, *graph_viewer, early_termination);
        for (int i = 0, end = next_nodes_list.size(); i < end; ++i) {
          for (int j = 0, end = next_nodes_list[i].first.size(); j < end; ++j) {
            next_nodes_list[i].first[j] = group.first[subgraph_node_index[next_nodes_list[i].first[j]]];
          }
          nodes_list_output.push_back(next_nodes_list[i]);
        }
      }
    }
  }
  return nodes_list_output;
}

// Detect and remove cycles from supported node list
void TensorrtExecutionProvider::RemoveTensorRTGraphCycles(SubGraphCollection_t& supported_nodes_vector, const GraphViewer& graph) const {
  const std::vector<NodeIndex>& node_index = graph.GetNodesInTopologicalOrder();
  bool trt_cycle = true;
  while (trt_cycle) {
    trt_cycle = false;
    std::unordered_map<std::string, int> node_to_index_map;
    std::unordered_map<int, std::string> index_to_node_map;
    std::unordered_map<std::string, std::unordered_set<std::string>> input_to_nodes_map, node_to_outputs_map;
    std::unordered_set<int> non_trt_node_index(node_index.begin(), node_index.end());
    int id = 0;
    for (const auto& group : supported_nodes_vector) {
      if (!group.first.empty()) {
        // Construct subgraph from node list
        std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, graph);

        // Create node to inputs/outputs/index maps
        const auto& meta_def = sub_graph->GetMetaDef();
        const std::string node_name = meta_def->name();
        if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
          index_to_node_map[id] = node_name;
          node_to_index_map[node_name] = id++;
        }

        if (meta_def != nullptr) {
          for (const auto& input : meta_def->inputs()) {
            input_to_nodes_map[input].insert(node_name);
          }
          for (const auto& output : meta_def->outputs()) {
            node_to_outputs_map[node_name].insert(output);
          }
        }

        // Remove TensorRT nodes from node index list
        for (const auto& index : group.first) {
          non_trt_node_index.erase(node_index[index]);
        }
      }
    }

    // Add non TensorRT nodes to the maps
    for (const auto& index : non_trt_node_index) {
      const auto& node = graph.GetNode(index);
      const std::string node_name = node->Name();
      if (node_to_index_map.find(node_name) == node_to_index_map.end()) {
        index_to_node_map[id] = node_name;
        node_to_index_map[node_name] = id++;
      }

      for (const auto& input : node->InputDefs()) {
        input_to_nodes_map[input->Name()].insert(node_name);
      }

      for (const auto& output : node->OutputDefs()) {
        node_to_outputs_map[node_name].insert(output->Name());
      }
    }

    // Create adjacency list
    int graph_size = node_to_index_map.size();
    std::list<int>* adjacency_map = new std::list<int>[graph_size];
    for (const auto& node : node_to_outputs_map) {
      for (auto iter = node.second.begin(); iter != node.second.end(); ++iter) {
        const auto& loc = input_to_nodes_map.find(*iter);
        if (loc != input_to_nodes_map.end()) {
          int parent_node_index = node_to_index_map.find(node.first)->second;
          for (auto child_node : loc->second) {
            int child_node_index = node_to_index_map.find(child_node)->second;
            adjacency_map[parent_node_index].push_back(child_node_index);
          }
        }
      }
    }

    // Check cycle in the graph
    bool* visited = new bool[graph_size];
    bool* st = new bool[graph_size];
    for (int i = 0; i < graph_size; ++i) {
      visited[i] = false;
      st[i] = false;
    }

    std::vector<int> cycles;
    bool has_cycle = false;
    for (int i = 0; i < graph_size; ++i) {
      if (FindCycleHelper(i, adjacency_map, visited, st, cycles)) {
        has_cycle = true;
        break;
      }
    }

    // Remove TensorRT subgraph from the supported node list if it's part of the cycle
    if (has_cycle) {
      for (int i = 0; i < static_cast<int>(cycles.size()); ++i) {
        auto loc = index_to_node_map.find(cycles[i]);
        if (loc != index_to_node_map.end() && loc->second.find("TRTKernel") != std::string::npos) {
          std::size_t found = loc->second.rfind("_");
          if (found != std::string::npos) {
            int trt_node_index = std::stoi(loc->second.substr(found + 1));
            supported_nodes_vector.erase(supported_nodes_vector.begin() + trt_node_index);
            trt_cycle = true;
            break;
          }
        }
      }
    }

    delete[] adjacency_map;
    delete[] visited;
    delete[] st;
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
TensorrtExecutionProvider::GetCapability(const GraphViewer& graph,
                                         const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  // Get ModelPath
  const auto& path_string = graph.ModelPath().ToPathString();
#ifdef _WIN32
  wcstombs(model_path_, path_string.c_str(), sizeof(model_path_));
#else
  strcpy(model_path_, path_string.c_str());
#endif

  // Get supported node list from TensorRT parser
  const int number_of_ort_nodes = graph.NumberOfNodes();
  std::vector<size_t> nodes_vector(number_of_ort_nodes);
  std::iota(std::begin(nodes_vector), std::end(nodes_vector), 0);
  SubGraphCollection_t supported_nodes_vector, parser_nodes_vector = {{nodes_vector, false}};
  bool early_termination = false;
  supported_nodes_vector = GetSupportedList(parser_nodes_vector, 0, max_partition_iterations_, graph, &early_termination);
  if (early_termination) {
    supported_nodes_vector.clear();
  }

  // Remove subgraphs if its size is less than the predefined minimal size
  for (auto it = supported_nodes_vector.begin(); it != supported_nodes_vector.end(); ++it) {
    const int subgraph_size = it->first.size();
    if (subgraph_size < min_subgraph_size_) {
      supported_nodes_vector.erase(it--);
    }
  }

  // Detect and remove cycles from supported node list
  RemoveTensorRTGraphCycles(supported_nodes_vector, graph);

  // Construct subgraph capability from node list
  std::vector<std::unique_ptr<ComputeCapability>> result;
  int number_of_trt_nodes = 0;
  for (const auto& group : supported_nodes_vector) {
    if (!group.first.empty()) {
      std::unique_ptr<IndexedSubGraph> sub_graph = GetSubGraph(group, graph);
      result.push_back(ComputeCapability::Create(std::move(sub_graph)));
      number_of_trt_nodes += group.first.size();
    }
  }

  const int number_of_subgraphs = supported_nodes_vector.size();
  if (number_of_trt_nodes == 0) {
    LOGS_DEFAULT(WARNING) << "[TensorRT EP] No graph will run on TensorRT exeuction provider";
  } else if (number_of_trt_nodes == number_of_ort_nodes) {
    LOGS_DEFAULT(INFO) << "[TensorRT EP] Whole graph will run on TensorRT exeuction provider";
  } else {
    LOGS_DEFAULT(INFO) << "[TensorRT EP] Graph is partitioned and number of subgraphs running on TensorRT exeuction provider is " << number_of_subgraphs;
  }

  return result;
}

common::Status TensorrtExecutionProvider::Compile(const std::vector<Node*>& fused_nodes,
                                                  std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto* fused_node : fused_nodes) {
    // Build map from input name to its index in input definitions
    std::unordered_map<std::string, int> input_map;
    const auto& input_defs = fused_node->InputDefs();
    input_map.reserve(input_defs.size());
    for (int i = 0, end = input_defs.size(); i < end; ++i) {
      input_map[input_defs[i]->Name()] = i;
    }

    // Build map from output name to its index in output definitions
    std::unordered_map<std::string, int> output_map;
    const auto& output_defs = fused_node->OutputDefs();
    output_map.reserve(output_defs.size());
    for (int i = 0, end = output_defs.size(); i < end; ++i) {
      output_map[output_defs[i]->Name()] = i;
    }

    // Reconstruct graph proto from fused node's function body
    const auto* func_body = fused_node->GetFunctionBody();
    if (!func_body) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Function body is empty");
    }
    const Graph& graph_body = func_body->Body();
    auto graph_body_viewer = graph_body.CreateGraphViewer();
    auto model = graph_body_viewer->CreateModel(*GetLogger());
    auto model_proto = model->ToProto();
    *model_proto->mutable_graph() = *graph_body.ToGraphProto();
    model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    std::string string_buf;
    model_proto->SerializeToString(string_buf);

    if (dump_subgraphs_) {
      // Dump the TensorRT subgraph if enabled via ORT_TENSORRT_DUMP_SUBGRAPHS env variable.
      std::fstream dump(fused_node->Name() + ".onnx", std::ios::out | std::ios::trunc | std::ios::binary);
      model_proto->SerializeToOstream(dump);
    }

    TensorrtLogger& trt_logger = GetTensorrtLogger();
    auto trt_builder = tensorrt_ptr::unique_pointer<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto trt_network = tensorrt_ptr::unique_pointer<nvinfer1::INetworkDefinition>(trt_builder->createNetworkV2(explicitBatch));
    auto trt_config = tensorrt_ptr::unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
    auto trt_parser = tensorrt_ptr::unique_pointer<nvonnxparser::IParser>(nvonnxparser::createParser(*trt_network, trt_logger));
    trt_parser->parse(string_buf.data(), string_buf.size(), model_path_);
    trt_config->setMaxWorkspaceSize(max_workspace_size_);

    int num_inputs = trt_network->getNbInputs();
    int num_outputs = trt_network->getNbOutputs();
    std::unordered_map<std::string, int> input_indexes(num_inputs);
    std::unordered_map<std::string, std::unordered_map<int, std::pair<int64_t, int64_t>>> input_shape_ranges;
    std::unordered_map<std::string, int> output_indexes(num_outputs);
    std::unordered_map<std::string, int> output_types(num_outputs);

    // Initialize shape range for dynamic shape tensors
    bool has_dynamic_shape = false;
    for (unsigned int i = 0, end = num_inputs; i < end; ++i) {
      auto input = trt_network->getInput(i);
      const std::string& input_name = input->getName();
      nvinfer1::Dims dims = input->getDimensions();
      int nb_dims = dims.nbDims;
      if (input->isShapeTensor()) {
        // Shape tensor
        input_shape_ranges[input_name][0] = std::make_pair(INT_MAX, INT_MIN);
        has_dynamic_shape = true;
      } else {
        // Execution tensor
        for (int j = 0, end = nb_dims; j < end; ++j) {
          if (dims.d[j] == -1) {
            input_shape_ranges[input_name][j] = std::make_pair(INT_MAX, INT_MIN);
            has_dynamic_shape = true;
          }
        }
      }
    }

    // Check platform availability for low precision
    if (fp16_enable_) {
      if (!trt_builder->platformHasFastFp16()) {
        fp16_enable_ = false;
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] ORT_TENSORRT_FP16_ENABLE is set, but platform doesn't support fast native fp16";
      }
    }

    if (int8_enable_) {
      if (!trt_builder->platformHasFastInt8()) {
        int8_enable_ = false;
        LOGS_DEFAULT(WARNING) << "[TensorRT EP] ORT_TENSORRT_INT8_ENABLE is set, but platform doesn't support fast native int8";
      }
    }

    // Load INT8 calibration table
    std::unordered_map<std::string, float> dynamic_range_map;
    if (int8_enable_) {
      const std::string calibration_cache_path = GetCachePath(cache_path_, int8_calibration_cache_name_);
      if (!ReadDynamicRange(calibration_cache_path, int8_use_native_tensorrt_calibration_table_, dynamic_range_map)) {
        throw std::runtime_error("Failed to read INT8 calibration table " + calibration_cache_path);
      }
    }

    // Set precision flags
    std::string trt_node_name_with_precision = fused_node->Name();
    if (fp16_enable_ && int8_enable_) {
      trt_config->setFlags(1U << static_cast<uint32_t>(nvinfer1::BuilderFlag::kFP16) | 1U << static_cast<uint32_t>(nvinfer1::BuilderFlag::kINT8));
      trt_node_name_with_precision += "_fp16_int8";
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] FP16 and INT8 mode is enabled";
    } else if (fp16_enable_) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
      trt_node_name_with_precision += "_fp16";
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] FP16 mode is enabled";
    } else if (int8_enable_) {
      trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
      trt_node_name_with_precision += "_int8";
      LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] INT8 mode is enabled";
    }

    // Build TRT engine here if the graph doesn't have dynamic shape input. Otherwise engine will
    // be built at runtime
    tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine> trt_engine;
    tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext> trt_context;
    if (!has_dynamic_shape) {
      const std::string cache_path = GetCachePath(cache_path_, trt_node_name_with_precision);
      const std::string engine_cache_path = cache_path + ".engine";
      std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
      if (engine_cache_enable_ && engine_file) {
        engine_file.seekg(0, std::ios::end);
        int engine_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        engine_file.read((char*)engine_buf.get(), engine_size);
        trt_engine = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size, nullptr));
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not deserialize engine from cache: " + engine_cache_path);
        }
      } else if (engine_decryption_enable_ && engine_cache_enable_ && !engine_file) {
        // Decrypt engine
        size_t engine_size = 0;
        if (!engine_decryption_(engine_cache_path.c_str(), nullptr, &engine_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not get engine buffer size");
        }
        std::unique_ptr<char[]> engine_buf{new char[engine_size]};
        if (!engine_decryption_(engine_cache_path.c_str(), &engine_buf[0], &engine_size)) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not call engine decryption function decrypt");
        }
        // Deserialize engine
        trt_engine = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engine_buf.get(), engine_size, nullptr));
        LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not deserialize engine from encrypted cache: " + engine_cache_path);
        }
      } else {
        // Set INT8 per tensor dynamic range
        if (int8_enable_ && trt_builder->platformHasFastInt8()) {
          trt_config->setInt8Calibrator(nullptr);
          if (!SetDynamicRange(*trt_network, dynamic_range_map)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not set INT8 dynamic range for fused node: " + fused_node->Name());
          }
        }

        // Build engine
        trt_engine = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(trt_builder->buildEngineWithConfig(*trt_network, *trt_config));
        if (trt_engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                 "TensorRT EP could not build engine for fused node: " + fused_node->Name());
        }
        if (engine_cache_enable_) {
          nvinfer1::IHostMemory* serializedModel = trt_engine->serialize();
          std::ofstream file(engine_cache_path, std::ios::binary | std::ios::out);
          file.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
          serializedModel->destroy();
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + engine_cache_path;
        }
      }

      // Build context
      trt_context = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());
      if (trt_context == nullptr) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                               "TensorRT EP could not build execution context for fused node: " + fused_node->Name());
      }
    }

    // Create input to index map
    for (int i = 0; i < num_inputs; ++i) {
      auto input = trt_network->getInput(i);
      const std::string& input_name = input->getName();
      const auto& iter = input_map.find(input_name);
      if (iter != input_map.end()) {
        input_indexes[input_name] = iter->second;
      }
    }

    // Create output to index and type maps
    const auto& graph_output = model_proto->graph().output();
    for (int i = 0; i < num_outputs; ++i) {
      const std::string& output_name = trt_network->getOutput(i)->getName();
      const auto& iter = output_map.find(output_name);
      if (iter != output_map.end()) {
        output_indexes[output_name] = iter->second;
      }
      const auto& tensor_type = graph_output[i].type().tensor_type();
      output_types[output_name] = tensor_type.elem_type();
    }

    // Save engine, context and input/output info to map
    parsers_.emplace(fused_node->Name(), std::move(trt_parser));
    engines_.emplace(fused_node->Name(), std::move(trt_engine));
    contexts_.emplace(fused_node->Name(), std::move(trt_context));
    builders_.emplace(fused_node->Name(), std::move(trt_builder));
    networks_.emplace(fused_node->Name(), std::move(trt_network));
    input_info_[fused_node->Name()].push_back(input_indexes);
    output_info_[fused_node->Name()].push_back(output_indexes);
    output_info_[fused_node->Name()].push_back(output_types);
    input_shape_ranges_[fused_node->Name()] = input_shape_ranges;

    // Create function state
    // TODO: remove default capture
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [=](ComputeContext* context, FunctionState* state) {
      std::unique_ptr<TensorrtFuncState> p = onnxruntime::make_unique<TensorrtFuncState>();
      *p = {context->allocate_func, context->release_func, context->allocator_handle, &parsers_[context->node_name],
            &engines_[context->node_name], &contexts_[context->node_name], &builders_[context->node_name],
            &networks_[context->node_name], input_info_[context->node_name], output_info_[context->node_name],
            input_shape_ranges_[context->node_name], &tensorrt_mu_, &fp16_enable_, &int8_enable_, &max_workspace_size_,
            trt_node_name_with_precision, engine_cache_enable_, cache_path_, runtime_.get(), nullptr,
            allocator_, dynamic_range_map, engine_decryption_enable_, engine_decryption_};
      *state = p.release();
      return 0;
    };

    // Release function state
    compute_info.release_state_func = [](FunctionState state) {
      if (state)
        delete static_cast<TensorrtFuncState*>(state);
    };
    // Create compute function
    compute_info.compute_func = [this](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      TensorrtFuncState* trt_state = reinterpret_cast<TensorrtFuncState*>(state);
      std::lock_guard<OrtMutex> lock(*(trt_state->tensorrt_mu_ptr));
      const std::unordered_map<std::string, int>& input_indexes = (trt_state->input_info)[0];
      const std::unordered_map<std::string, int>& output_indexes = (trt_state->output_info)[0];
      const std::unordered_map<std::string, int>& output_types = (trt_state->output_info)[1];
      auto& shape_ranges = trt_state->input_shape_ranges;
      auto trt_builder = trt_state->builder->get();
      auto trt_engine = trt_state->engine->get();
      auto trt_context = trt_state->context->get();
      auto trt_profile = &(trt_state->trt_profile);
      auto alloc = trt_state->scratch_allocator;
      int num_inputs = input_indexes.size();
      int num_outputs = output_indexes.size();
      bool engine_update = false;
      std::unordered_set<std::string> input_names;
      std::unordered_map<std::string, std::vector<int32_t>> tensor_shape_values;

      cudaStream_t stream = static_cast<cudaStream_t>(this->GetComputeStream());

      // Load serialized engine
      const std::string cache_path = GetCachePath(trt_state->engine_cache_path, trt_state->trt_node_name_with_precision);
      const std::string engine_cache_path = cache_path + ".engine";
      const std::string profile_cache_path = cache_path + ".profile";
      if (trt_state->engine_cache_enable && trt_engine == nullptr) {
        std::ifstream engine_file(engine_cache_path, std::ios::binary | std::ios::in);
        std::ifstream profile_file(profile_cache_path, std::ios::binary | std::ios::in);
        if (engine_file && profile_file) {
          // Deserialize profile
          shape_ranges = DeserializeProfile(profile_file);
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + profile_cache_path;
          // Deserialize engine
          trt_state->context->reset();
          trt_state->engine->reset();
          engine_file.seekg(0, std::ios::end);
          int engine_size = engine_file.tellg();
          engine_file.seekg(0, std::ios::beg);
          std::unique_ptr<char[]> engine_buf{new char[engine_size]};
          engine_file.read((char*)engine_buf.get(), engine_size);
          *(trt_state->engine) = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(
              trt_state->runtime->deserializeCudaEngine(engine_buf.get(), engine_size, nullptr));
          if (trt_state->engine == nullptr) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP Failed to Build Engine.");
          }
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
          trt_engine = trt_state->engine->get();
          *(trt_state->context) = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(
              trt_state->engine->get()->createExecutionContext());
          if (trt_state->context == nullptr) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to create context.");
          }
          trt_context = trt_state->context->get();
        } else if (trt_state->engine_decryption_enable && !engine_file && profile_file) {
          shape_ranges = DeserializeProfile(profile_file);
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + profile_cache_path;
          // Decrypt engine
          size_t engine_size = 0;
          if (!trt_state->engine_decryption(engine_cache_path.c_str(), nullptr, &engine_size)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not get engine buffer size");
          }
          std::unique_ptr<char[]> engine_buf{new char[engine_size]};
          if (!trt_state->engine_decryption(engine_cache_path.c_str(), &engine_buf[0], &engine_size)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not call engine decryption function decrypt");
          }
          // Deserialize engine
          trt_state->context->reset();
          trt_state->engine->reset();
          *(trt_state->engine) = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(trt_state->runtime->deserializeCudaEngine(engine_buf.get(), engine_size, nullptr));
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
          if (trt_state->engine == nullptr) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP could not deserialize engine from encrypted cache: " + engine_cache_path);
          }
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] DeSerialized " + engine_cache_path;
          trt_engine = trt_state->engine->get();
          *(trt_state->context) = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(
              trt_state->engine->get()->createExecutionContext());
          if (trt_state->context == nullptr) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to create context.");
          }
          trt_context = trt_state->context->get();
        }
      }

      for (int i = 0, end = num_inputs; i < end; ++i) {
        auto input = trt_state->network->get()->getInput(i);
        const std::string& input_name = input->getName();
        nvinfer1::Dims dims = input->getDimensions();
        int nb_dims = dims.nbDims;
        // Check and update shape ranges for dynamic shape inputs
        input_names.insert(input_name);
        if (shape_ranges.find(input_name) != shape_ranges.end()) {
          int input_index = 0;
          const auto& iter = input_indexes.find(input_name);
          if (iter != input_indexes.end()) {
            input_index = iter->second;
          }

          const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
          auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
          const auto& tensor_shapes = ort.GetTensorShape(tensor_info);
          auto& shape_range = shape_ranges[input_name];

          // Create shape profile
          if (input->isShapeTensor()) {
            // Get shape values for shape tensor input
            const auto& tensor_type = ort.GetTensorElementType(tensor_info);
            int shape_size = nb_dims == 0 ? 1 : tensor_shapes[0];
            tensor_shape_values[input_name].resize(shape_size);
            switch (tensor_type) {
              case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
                int32_t* input = new int32_t[shape_size];
                CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input, ort.GetTensorData<int32_t>(input_tensor), shape_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
                CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
                for (int j = 0; j < shape_size; ++j) {
                  tensor_shape_values[input_name][j] = input[j];
                }
                delete[] input;
                break;
              }
              case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
                int64_t* input = new int64_t[shape_size];
                CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(input, ort.GetTensorData<int64_t>(input_tensor), shape_size * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
                CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
                for (int j = 0; j < shape_size; ++j) {
                  tensor_shape_values[input_name][j] = static_cast<int32_t>(input[j]);
                }
                delete[] input;
                break;
              }
              default: {
                return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                       "TensorRT shape tensor data type: " + std::to_string(tensor_type) + " not supported.");
              }
            }

            // Update shape ranges
            std::vector<int32_t> shapes_min(shape_size), shapes_opt(shape_size), shapes_max(shape_size);
            int shape_range_size = shape_range.size();
            if (shape_size == shape_range_size) {
              // If shape size matches, check/update shape range
              for (int j = 0; j < shape_size; ++j) {
                shapes_min[j] = shape_range[j].first;
                shapes_opt[j] = shape_range[j].second;
                shapes_max[j] = shape_range[j].second;

                const auto& tensor_shape_value = tensor_shape_values[input_name][j];
                // Update shape range lower bound
                if (tensor_shape_value < shape_range[j].first) {
                  shape_range[j].first = tensor_shape_value;
                  shapes_min[j] = tensor_shape_value;
                  engine_update = true;
                }
                // Update shape range upper bound
                if (tensor_shape_value > shape_range[j].second) {
                  shape_range[j].second = tensor_shape_value;
                  shapes_max[j] = tensor_shape_value;
                  shapes_opt[j] = tensor_shape_value;
                  engine_update = true;
                }
              }
            } else {
              // If shape size doesn't match, initialize shape_range with the new shape value
              shape_range.clear();
              for (int j = 0; j < shape_size; ++j) {
                const auto& tensor_shape_value = tensor_shape_values[input_name][j];
                shape_range[j] = std::make_pair(tensor_shape_value, tensor_shape_value);
                shapes_min[j] = tensor_shape_value;
                shapes_opt[j] = tensor_shape_value;
                shapes_max[j] = tensor_shape_value;
              }
              engine_update = true;
            }

            if (*trt_profile == nullptr) {
              *trt_profile = trt_builder->createOptimizationProfile();
            }
            (*trt_profile)->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, &shapes_min[0], shape_size);
            (*trt_profile)->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, &shapes_opt[0], shape_size);
            (*trt_profile)->setShapeValues(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, &shapes_max[0], shape_size);

          } else {  // Execution tensor
            nvinfer1::Dims dims_min(dims), dims_opt(dims), dims_max(dims);
            for (int j = 0, end = nb_dims; j < end; ++j) {
              const auto& tensor_shape = tensor_shapes[j];
              if (shape_range.find(j) != shape_range.end()) {
                dims_min.d[j] = shape_range[j].first;
                dims_opt.d[j] = shape_range[j].second;
                dims_max.d[j] = shape_range[j].second;

                // Update minimum dimension
                if (tensor_shape < shape_range[j].first) {
                  shape_range[j].first = tensor_shape;
                  dims_min.d[j] = tensor_shape;
                  engine_update = true;
                }
                // Update maximum dimension
                if (tensor_shape > shape_range[j].second) {
                  shape_range[j].second = tensor_shape;
                  dims_max.d[j] = tensor_shape;
                  dims_opt.d[j] = tensor_shape;
                  engine_update = true;
                }
              }
            }

            if (*trt_profile == nullptr) {
              *trt_profile = trt_builder->createOptimizationProfile();
            }
            (*trt_profile)->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, dims_min);
            (*trt_profile)->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, dims_opt);
            (*trt_profile)->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, dims_max);
          }
          ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
        }
      }

      // Regenerate engine
      // Only one profile is generated, so no need to explicitly set optimization profile
      if (engine_update) {
        trt_state->context->reset();
        trt_state->engine->reset();
        auto trt_config = tensorrt_ptr::unique_pointer<nvinfer1::IBuilderConfig>(trt_builder->createBuilderConfig());
        trt_config->setMaxWorkspaceSize(*(trt_state->max_workspace_size_ptr));
        trt_config->addOptimizationProfile(*trt_profile);

        // Set INT8 Per Tensor Dynamic range
        if (*(trt_state->int8_enable_ptr) && trt_builder->platformHasFastInt8()) {
          trt_config->setInt8Calibrator(nullptr);
          if (!SetDynamicRange(*trt_state->network->get(), trt_state->dynamic_range_map)) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to set INT8 dynamic range.");
          }
        }

        // Set precision
        if (*(trt_state->fp16_enable_ptr) && *(trt_state->int8_enable_ptr)) {
          trt_config->setFlags(1U << static_cast<uint32_t>(nvinfer1::BuilderFlag::kFP16) | 1U << static_cast<uint32_t>(nvinfer1::BuilderFlag::kINT8));
        } else if (*(trt_state->fp16_enable_ptr)) {
          trt_config->setFlag(nvinfer1::BuilderFlag::kFP16);
        } else if (*(trt_state->int8_enable_ptr)) {
          trt_config->setFlag(nvinfer1::BuilderFlag::kINT8);
        }

        // Build engine
        *(trt_state->engine) = tensorrt_ptr::unique_pointer<nvinfer1::ICudaEngine>(
            trt_builder->buildEngineWithConfig(*trt_state->network->get(), *trt_config));
        if (trt_state->engine == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP Failed to Build Engine.");
        }
        trt_engine = trt_state->engine->get();
        if (trt_state->engine_cache_enable) {
          // Serialize engine profile
          SerializeProfile(profile_cache_path, shape_ranges);
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + profile_cache_path;

          // Serialize engine
          nvinfer1::IHostMemory* serializedModel = trt_engine->serialize();
          std::ofstream file(engine_cache_path, std::ios::binary | std::ios::out);
          file.write(reinterpret_cast<char*>(serializedModel->data()), serializedModel->size());
          serializedModel->destroy();
          LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Serialized " + engine_cache_path;
        }

        // Build context
        *(trt_state->context) = tensorrt_ptr::unique_pointer<nvinfer1::IExecutionContext>(
            trt_state->engine->get()->createExecutionContext());
        if (trt_state->context == nullptr) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL, "TensorRT EP failed to create context.");
        }
        trt_context = trt_state->context->get();
      }

      // Get input and output binding names
      int total_bindings = trt_engine->getNbBindings();
      std::vector<void*> buffers(total_bindings);
      std::vector<std::string> input_binding_names, output_binding_names;
      for (int i = 0, end = total_bindings; i < end; ++i) {
        if (trt_engine->bindingIsInput(i)) {
          input_binding_names.push_back(trt_engine->getBindingName(i));
        } else {
          output_binding_names.push_back(trt_engine->getBindingName(i));
        }
      }

      // Set input shapes and assign input buffers
      std::vector<IAllocatorUniquePtr<void>> scratch_buffers;
      for (int i = 0, end = input_binding_names.size(); i < end; ++i) {
        const std::string& input_name = input_binding_names[i];
        int binding_index = trt_engine->getBindingIndex(input_name.c_str());
        if (binding_index == -1) {
          continue;
        }

        int input_index = 0;
        const auto& iter = input_indexes.find(input_name);
        if (iter != input_indexes.end()) {
          input_index = iter->second;
        }
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_index);
        auto tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        const auto& tensor_shapes = ort.GetTensorShape(tensor_info);

        // Set dynamic shapes
        nvinfer1::Dims dimensions = trt_engine->getBindingDimensions(static_cast<int>(binding_index));
        int nb_dims = dimensions.nbDims;
        if (input_names.count(input_name) == 1) {
          if (trt_engine->isShapeBinding(binding_index)) {
            trt_context->setInputShapeBinding(binding_index, &tensor_shape_values[input_name][0]);
          } else {
            for (int j = 0, end = nb_dims; j < end; ++j) {
              dimensions.d[j] = tensor_shapes[j];
            }
            trt_context->setBindingDimensions(binding_index, dimensions);
          }
        }

        const auto& input_type = ort.GetTensorElementType(tensor_info);
        switch (input_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            auto input_tensor_ptr = ort.GetTensorData<float>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(float)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = const_cast<float*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            auto input_tensor_ptr = ort.GetTensorData<uint16_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(uint16_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = const_cast<uint16_t*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            auto input_tensor_ptr = ort.GetTensorData<bool>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(bool)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = const_cast<bool*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            auto input_tensor_ptr = ort.GetTensorData<int8_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(int8_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = const_cast<int8_t*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            auto input_tensor_ptr = ort.GetTensorData<int32_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(int32_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = const_cast<int32_t*>(input_tensor_ptr);
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
            auto input_tensor_ptr = ort.GetTensorData<int64_t>(input_tensor);
            if (input_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(int32_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              SafeInt<int> input_dim_size = 1;
              for (int j = 0, end = nb_dims; j < end; ++j) {
                if (tensor_shapes[j] == 0) {
                  input_dim_size = 1;
                  break;
                } else {
                  input_dim_size *= tensor_shapes[j];
                }
              }
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, input_dim_size * sizeof(int32_t)));
              buffers[binding_index] = scratch_buffers.back().get();
              cuda::Impl_Cast<int64_t, int32_t>(stream, input_tensor_ptr, reinterpret_cast<int32_t*>(buffers[binding_index]), input_dim_size);
            }
            break;
          }
          default: {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP input onnx tensor data type: " + std::to_string(input_type) + " not supported.");
          }
        }
        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      // Set output shapes and assign output buffers
      std::vector<int> output_dim_sizes(num_outputs, 1);
      std::vector<OrtValue*> output_tensor(num_outputs, nullptr);
      for (int i = 0, end = output_binding_names.size(); i < end; ++i) {
        // Set dynamic shapes
        const std::string& output_name = output_binding_names[i];
        int binding_index = trt_engine->getBindingIndex(output_name.c_str());
        if (binding_index == -1) {
          continue;
        }

        int output_index = 0;
        const auto& index_iter = output_indexes.find(output_name);
        if (index_iter != output_indexes.end()) {
          output_index = index_iter->second;
        }
        nvinfer1::Dims dimensions = trt_context->getBindingDimensions(static_cast<int>(binding_index));
        int nb_dims = dimensions.nbDims;
        std::vector<int64_t> output_shapes(nb_dims);
        for (int j = 0, end = nb_dims; j < end; ++j) {
          output_shapes[j] = dimensions.d[j];
        }
        output_tensor[i] = ort.KernelContext_GetOutput(context, output_index, output_shapes.data(), output_shapes.size());

        int output_type = 0;
        const auto& type_iter = output_types.find(output_name);
        if (type_iter != output_types.end()) {
          output_type = type_iter->second;
        }

        switch (output_type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            auto output_tensor_ptr = ort.GetTensorMutableData<float>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(float)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
            auto output_tensor_ptr = ort.GetTensorMutableData<uint16_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(uint16_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
            auto output_tensor_ptr = ort.GetTensorMutableData<bool>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(bool)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
            auto output_tensor_ptr = ort.GetTensorMutableData<int8_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(int8_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
            auto output_tensor_ptr = ort.GetTensorMutableData<int32_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(int32_t)));
              buffers[binding_index] = scratch_buffers.back().get();
            } else {
              buffers[binding_index] = output_tensor_ptr;
            }
            break;
          }
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            // Allocate INT32 CUDA memory for INT64 output type because TensorRT doesn't fully support INT64
            auto output_tensor_ptr = ort.GetTensorMutableData<int64_t>(output_tensor[i]);
            if (output_tensor_ptr == nullptr) {
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, sizeof(int32_t)));
              buffers[binding_index] = scratch_buffers.back().get();
              output_dim_sizes[i] = 1;
            } else {
              SafeInt<int> output_dim_size(output_dim_sizes[i]);
              for (int j = 0, end = nb_dims; j < end; ++j) {
                if (dimensions.d[j] == 0) {
                  output_dim_size = 1;
                  break;
                } else {
                  output_dim_size *= dimensions.d[j];
                }
              }
              scratch_buffers.push_back(IAllocator::MakeUniquePtr<void>(alloc, output_dim_size * sizeof(int32_t)));
              buffers[binding_index] = scratch_buffers.back().get();
              output_dim_sizes[i] = output_dim_size;
            }
            break;
          }
          default: {
            return ORT_MAKE_STATUS(ONNXRUNTIME, EP_FAIL,
                                   "TensorRT EP output tensor data type: " + std::to_string(output_type) + " not supported.");
          }
        }
      }

      // Run TRT inference
      if (!trt_context->enqueueV2(&buffers[0], stream, nullptr)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "TensorRT EP execution context enqueue failed.");
      }

      // Cast INT64 input to INT32 because TensorRT doesn't fully support INT64
      for (int i = 0, end = output_binding_names.size(); i < end; ++i) {
        const std::string& output_name = output_binding_names[i];
        size_t binding_index = trt_engine->getBindingIndex(output_name.c_str());
        int output_type = 0;
        const auto& iter = output_types.find(output_name);
        if (iter != output_types.end()) {
          output_type = iter->second;
        }
        if (output_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
          auto output_tensor_ptr = ort.GetTensorMutableData<int64_t>(output_tensor[i]);
          if (output_tensor_ptr != nullptr) {
            cuda::Impl_Cast<int32_t, int64_t>(stream, reinterpret_cast<int32_t*>(buffers[binding_index]), output_tensor_ptr, output_dim_sizes[i]);
          }
        }
      }

      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
}  // namespace onnxruntime

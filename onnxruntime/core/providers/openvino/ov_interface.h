// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <utility>
#include <optional>
#include <algorithm>
#include <unordered_map>

#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "openvino/pass/convert_fp32_to_fp16.hpp"
#include "openvino/frontend/manager.hpp"
#include "openvino/core/dimension.hpp"
#include "openvino/core/partial_shape.hpp"
#include "weak_singleton.h"

#include <string>

// Helper macro to test OpenVINO version at compile time.
// Usage: #if OPENVINO_VERSION_AT_LEAST(2025, 3)
// Falls back to 0 if OPENVINO_VERSION_MAJOR/MINOR are not defined.
#if defined(OPENVINO_VERSION_MAJOR) && defined(OPENVINO_VERSION_MINOR)
#define OPENVINO_VERSION_AT_LEAST(major, minor) \
  ((OPENVINO_VERSION_MAJOR > (major)) ||        \
   (OPENVINO_VERSION_MAJOR == (major) && OPENVINO_VERSION_MINOR >= (minor)))
#else
#define OPENVINO_VERSION_AT_LEAST(major, minor) 0
#endif

namespace onnxruntime {
namespace openvino_ep {
class OVCore;
class OVInferRequest;
class OVExeNetwork;
struct ModelBlobWrapper;
struct SessionContext;

typedef ov::Tensor OVTensor;
typedef ov::ProfilingInfo OVProfilingInfo;
typedef ov::Model OVNetwork;
typedef std::shared_ptr<OVInferRequest> OVInferRequestPtr;
typedef std::shared_ptr<OVTensor> OVTensorPtr;

std::optional<bool> queryOVProperty(const std::string& property, const std::string& device_type);

struct OVCore : WeakSingleton<OVCore> {
  ov::Core core;

  // OV Interface For Reading Model
  std::shared_ptr<OVNetwork> ReadModel(std::string&& model_stream, const std::string& model_path);

  OVExeNetwork StatefulCompileModel(std::shared_ptr<OVNetwork>& model,
                                    std::string& hw_target,
                                    const ov::AnyMap& device_config);
  // OV Interface for Compiling OV Model Type
  OVExeNetwork CompileModel(std::shared_ptr<const OVNetwork>& ie_cnn_network,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            bool enable_causallm,
                            const std::string& name);
  // OV Interface for Fast Compile
  OVExeNetwork CompileModel(const std::string& onnx_model,
                            std::string& hw_target,
                            ov::AnyMap& device_config,
                            const std::string& name);
  // OV Interface for Import model Stream
  OVExeNetwork ImportModel(ModelBlobWrapper& model_blob,
                           std::string hw_target,
                           const ov::AnyMap& device_config,
                           std::string name);
  OVExeNetwork ImportEPCtxOVIREncapsulation(std::istream& model_stream,
                                            std::string& hw_target,
                                            const ov::AnyMap& device_config,
                                            bool enable_causallm,
                                            std::filesystem::path xml_file_path,
                                            const SessionContext& session_context);

  std::vector<std::string> GetAvailableDevices() const;
  std::vector<std::string> GetAvailableDevices(const std::string& device_type) const;
  void SetCache(const std::string& cache_dir_path);
  void SetStreams(const std::string& device_type, int num_streams);
};

class OVExeNetwork {
  ov::CompiledModel compiled_model_obj;
  std::string target_device;
  bool is_stateful_causallm;
  bool is_kvcache_reorder_added = false;

 public:
  explicit OVExeNetwork(ov::CompiledModel compiled_model, std::string device, bool stateful_causallm = false, bool kvcache_reorder_added = false)
      : compiled_model_obj(std::move(compiled_model)), target_device(std::move(device)), is_stateful_causallm(stateful_causallm), is_kvcache_reorder_added(kvcache_reorder_added) {}
  OVExeNetwork() : compiled_model_obj(ov::CompiledModel()), is_stateful_causallm(false) {}
  ov::CompiledModel& Get() { return compiled_model_obj; }
  std::shared_ptr<OVInferRequest> CreateInferRequest();
};

class OVInferRequest {
  struct ov_tensor_data_t {
    OVTensorPtr tensor_ptr;
    const void* ort_ptr;
  };

 protected:
  ov::InferRequest ovInfReq;
  std::unordered_map<std::string, ov_tensor_data_t> bindings_cache_;

 public:
  uint32_t GetNumInputs();
  virtual OVTensorPtr GetTensor(const std::string& name);
  std::string GetInputTensorName(uint32_t index);

  // Set tensor call infer req tensor if ort_ptr differs from last set ptr.
  void SetTensor(const std::string& name, const ov::element::Type& type, const ov::Shape& shape, void* ort_ptr) {
    auto& cached_binding = bindings_cache_[name];
    if (cached_binding.ort_ptr != ort_ptr ||
        !cached_binding.tensor_ptr ||
        cached_binding.tensor_ptr->get_shape() != shape) {
      cached_binding.tensor_ptr.reset();
      auto ov_tensor = std::make_shared<ov::Tensor>(type, shape, const_cast<void*>(ort_ptr));
      ovInfReq.set_tensor(name, *ov_tensor);
      cached_binding = {std::move(ov_tensor), ort_ptr};
    }
  }

  void SetTensor(const std::string& name, OVTensorPtr& blob);
  virtual void Infer();
  explicit OVInferRequest(ov::InferRequest obj) : ovInfReq(std::move(obj)) {}
  OVInferRequest() : ovInfReq(ov::InferRequest()) {}
  ov::InferRequest& GetInfReq() {
    return ovInfReq;
  }
  virtual void RewindKVCache([[maybe_unused]] size_t index) {}
  virtual void SetReorderKVCacheStatus([[maybe_unused]] const std::vector<int32_t>& src_indices, [[maybe_unused]] const std::vector<int32_t>& dst_indices) {}
};

class StatefulOVInferRequest : public OVInferRequest {
 public:
  explicit StatefulOVInferRequest(ov::InferRequest infer_request, std::string device, bool kvcache_reorder_added = false);

  void Infer() override;
  void RewindKVCache(size_t index) override;
  void SetReorderKVCacheStatus(const std::vector<int32_t>& src_indices, const std::vector<int32_t>& dst_indices) override;
  void FillTensor(const std::string& tensor_name, const ov::element::Type& type,
                  const std::vector<size_t>& shape, int32_t fill_value);
  void CacheTensor(const std::string& tensor_name, std::vector<int64_t>& cache);
  void SetTensorFromCache(const std::string& tensor_name, const std::vector<int64_t>& cache_data);
  std::optional<ov::Tensor> FindTensor(const std::string& tensor_name);
  OVTensorPtr GetTensor(const std::string& name) override;

 private:
  void PreProcessInferRequest();
  void PostProcessInferRequest();
  void CleanReorderKVCacheStatus();
  std::string target_device;

  std::vector<int64_t> cached_input_ids;
  std::vector<int64_t> cached_position_ids;
  std::vector<int32_t> kv_src_indices;
  std::vector<int32_t> kv_dst_indices;

  // If prefill_use_full_chat_history is true, cache the "input_ids" & "position_ids" tensors,
  // and ensure that full chat history is passed for each prefill call.
  bool prefill_use_full_chat_history = false;
  // If kvcache_reorder is added, will include kv_src/dst_indices as input
  bool is_kvcache_reorder_added = false;

  bool IsNPULogitsSliceRequired();
  bool _npu_logits_slice_required = false;
};

}  // namespace openvino_ep
}  // namespace onnxruntime

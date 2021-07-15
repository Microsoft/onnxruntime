// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/shared_library/provider_api.h"
#include "openvino_execution_provider.h"
#include "contexts.h"
#include "backend_manager.h"
#include "ov_versions/capabilities.h"

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {

constexpr const char* OpenVINO = "OpenVINO";

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const OpenVINOExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider} {
  openvino_ep::BackendManager::GetGlobalContext().device_type = info.device_type_;
  openvino_ep::BackendManager::GetGlobalContext().precision_str = info.precision_;
  openvino_ep::BackendManager::GetGlobalContext().enable_vpu_fast_compile = info.enable_vpu_fast_compile_;
  openvino_ep::BackendManager::GetGlobalContext().use_compiled_network = info.use_compiled_network_;
  openvino_ep::BackendManager::GetGlobalContext().blob_dump_path = info.blob_dump_path_;

  if ((int)info.num_of_threads_ <= 0) {
    openvino_ep::BackendManager::GetGlobalContext().num_of_threads = 8;
  } else {
    openvino_ep::BackendManager::GetGlobalContext().num_of_threads = info.num_of_threads_;
  }
  if (info.device_id_ != "") {
    bool device_found = false;
    auto available_devices = openvino_ep::BackendManager::GetGlobalContext().ie_core.GetAvailableDevices();
    for (auto device : available_devices) {
      if (device == info.device_id_) {
        device_found = true;
        break;
      }
    }
    if (!device_found) {
      std::string err_msg = std::string("Device not found : ") + info.device_id_ + "\nChoose one of:\n";
      for (auto device : available_devices) {
        err_msg = err_msg + device + "\n";
      }
      ORT_THROW(err_msg);
    }
  }
  openvino_ep::BackendManager::GetGlobalContext().device_id = info.device_id_;

  AllocatorCreationInfo device_info(
      [](int) {
        return CreateCPUAllocator(OrtMemoryInfo(OpenVINO, OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));
}

std::vector<std::unique_ptr<ComputeCapability>>
OpenVINOExecutionProvider::GetCapability(const GraphViewer& graph_viewer, const std::vector<const KernelRegistry*>& kernel_registries) const {
  ORT_UNUSED_PARAMETER(kernel_registries);

  std::vector<std::unique_ptr<ComputeCapability>> result;
  openvino_ep::BackendManager::GetGlobalContext().onnx_model_name = graph_viewer.Name();
#ifdef _WIN32
  std::wstring onnx_path = graph_viewer.ModelPath().ToPathString();
  openvino_ep::BackendManager::GetGlobalContext().onnx_model_path_name = std::string(onnx_path.begin(), onnx_path.end());
#else
  openvino_ep::BackendManager::GetGlobalContext().onnx_model_path_name = graph_viewer.ModelPath().ToPathString();
#endif
  openvino_ep::BackendManager::GetGlobalContext().onnx_opset_version = graph_viewer.DomainToVersionMap().at(kOnnxDomain);

#if defined OPENVINO_2020_3
  result = openvino_ep::GetCapability_2020_3(graph_viewer,
                                             openvino_ep::BackendManager::GetGlobalContext().device_type);
#elif defined (OPENVINO_2021_1)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_1");
  result = obj.Execute();
#elif defined (OPENVINO_2021_2)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_2");
  result = obj.Execute();
#elif defined (OPENVINO_2021_3)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_3");
  result = obj.Execute();
#elif defined (OPENVINO_2021_4)
  openvino_ep::GetCapability obj(graph_viewer,
                                 openvino_ep::BackendManager::GetGlobalContext().device_type, "V_2021_4");
  result = obj.Execute();
#endif

  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<onnxruntime::Node*>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node : fused_nodes) {
    NodeComputeInfo compute_info;
    std::shared_ptr<openvino_ep::BackendManager> backend_manager = std::make_shared<openvino_ep::BackendManager>(fused_node, *GetLogger());

    compute_info.create_state_func =
        [backend_manager](ComputeContext* context, FunctionState* state) {
          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState();
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          p->backend_manager = backend_manager;
          *state = static_cast<FunctionState>(p);
          return 0;
        };
    compute_info.compute_func = [](FunctionState state, const OrtApi* api, OrtKernelContext* context) {
      auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
      try {
        function_state->backend_manager->Compute(*api, context);
      } catch (const char* msg) {
        return common::Status(common::ONNXRUNTIME, common::FAIL, msg);
      }
      return Status::OK();
    };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
            delete function_state;
          }
        };
    node_compute_funcs.push_back(compute_info);
  }

  return Status::OK();
}
}  // namespace onnxruntime

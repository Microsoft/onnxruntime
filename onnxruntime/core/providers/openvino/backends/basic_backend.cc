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
using WaitMode = InferenceEngine::InferRequest::WaitMode;
#else
using Exception = InferenceEngine::details::InferenceEngineException;
using WaitMode = InferenceEngine::IInferRequest::WaitMode;
#endif

#include "core/providers/shared_library/provider_api.h"

#include "../backend_utils.h"
#include <ngraph/frontend/onnx_import/onnx.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "basic_backend.h"
#include "../backend_manager.h"

namespace onnxruntime {
namespace openvino_ep {

using namespace backend_utils;

BasicBackend::BasicBackend(const ONNX_NAMESPACE::ModelProto& model_proto,
                           GlobalContext& global_context,
                           const SubGraphContext& subgraph_context)
    : global_context_(global_context), subgraph_context_(subgraph_context) {
  std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
  bool vpu_status = false;
  bool import_blob_status = false;
  std::string model_blob_name;
  std::ifstream blob_path;
  std::string ov_compiled_blobs_dir = "";

  if(hw_target == "MYRIAD" && global_context_.use_compiled_network == true) {
    if(!openvino_ep::backend_utils::UseCompiledNetwork()) {
      std::size_t model_index = global_context_.onnx_model_path_name.find_last_of("/\\");
      std::string model_name= global_context_.onnx_model_path_name.substr(model_index+1);
      std::size_t model_extension_index = model_name.find_last_of(".");
      if(openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph) {
          model_blob_name = global_context_.onnx_model_name + "_" + "op_v_" + std::to_string(global_context_.onnx_opset_version) + "_" + model_name.substr(0,model_extension_index) + "_" + hw_target + "_" + subgraph_context_.subgraph_name + "_ov_" + "fully" + ".blob";
      }
      else {
          model_blob_name = global_context_.onnx_model_name + "_" + "op_v_" + std::to_string(global_context_.onnx_opset_version) + "_" + model_name.substr(0,model_extension_index) + "_" + hw_target + "_" + subgraph_context_.subgraph_name + "_ov_" + "partially" + ".blob";
      }
      if(global_context_.blob_dump_path == "" || global_context_.blob_dump_path == "\"" || global_context_.blob_dump_path.empty()) {
        ov_compiled_blobs_dir = openvino_ep::backend_utils::GetCurrentWorkingDir() + "/ov_compiled_blobs/";
      } else {
        ov_compiled_blobs_dir = global_context_.blob_dump_path + "/ov_compiled_blobs";
      }
      if(openvino_ep::backend_utils::IsDirExists(ov_compiled_blobs_dir)) {
        LOGS_DEFAULT(INFO) << log_tag << "'ov_compiled_blobs' directory already exists at the executable path";
      }
      else {
        CreateDirectory(ov_compiled_blobs_dir);
      }
      blob_path.open(ov_compiled_blobs_dir + "/" + model_blob_name);
      if (!blob_path.is_open()) {
          LOGS_DEFAULT(INFO) << log_tag << "Device specific Compiled blob doesn't exist for this model";
      } else {
          LOGS_DEFAULT(INFO) << log_tag << "Device specific Compiled blob already exists for this model";
          vpu_status = true;
      }
    }
  }

  //validate const subgraphs
  if(!openvino_ep::BackendManager::GetGlobalContext().is_wholly_supported_graph) {
    ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
    SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);
  #if defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
    if (const_outputs_map_.size() == subgraph_context_.output_names.size())
      subgraph_context_.is_constant = true;
  #endif

    // Loading model to the plugin
    if (subgraph_context_.is_constant) {
      LOGS_DEFAULT(INFO) << log_tag << "The subgraph is a const. Directly moving to Infer stage.";
      return;
    }
  }

  if (vpu_status == true || openvino_ep::backend_utils::UseCompiledNetwork()) {
    const std::string model_blob_path = ov_compiled_blobs_dir + "/" + model_blob_name;
    const std::string compiled_blob_path = onnxruntime::GetEnvironmentVar("OV_BLOB_PATH");
    try {
      if(vpu_status == true) {
        LOGS_DEFAULT(INFO) << log_tag << "Importing the pre-compiled blob for this model which already exists in the directory 'ov_compiled_blobs'";
        exe_network_ = global_context_.ie_core.ImportNetwork(model_blob_path, hw_target, {});
      } else {
        LOGS_DEFAULT(INFO) << log_tag << "Importing the pre-compiled blob from the path set by the user";
        if (compiled_blob_path.empty())
          throw std::runtime_error("The compiled blob path is not set");
        exe_network_ = global_context_.ie_core.ImportNetwork(compiled_blob_path, hw_target, {});
      }
    } catch (Exception &e) {
      ORT_THROW(log_tag + " Exception while Importing Network for graph: " + subgraph_context_.subgraph_name + ": " + e.what());
    } catch(...) {
      ORT_THROW(log_tag + " Exception while Importing Network for graph: " + subgraph_context_.subgraph_name);
    }
    import_blob_status = true;
    LOGS_DEFAULT(INFO) << log_tag << "Succesfully Created an executable network from a previously exported network";
  }

  if ((global_context_.use_compiled_network == true && import_blob_status == false) || vpu_status == false) {
    if(!openvino_ep::backend_utils::UseCompiledNetwork()) {
      ie_cnn_network_ = CreateCNNNetwork(model_proto, global_context_, subgraph_context_, const_outputs_map_);
      SetIODefs(model_proto, ie_cnn_network_, subgraph_context_.output_names, const_outputs_map_, global_context_.device_type);
    #if defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
      if (const_outputs_map_.size() == subgraph_context_.output_names.size())
        subgraph_context_.is_constant = true;
    #endif

      // Loading model to the plugin
      if (subgraph_context_.is_constant)
        return;
      std::map<std::string, std::string> config;
    #ifndef NDEBUG
      if (openvino_ep::backend_utils::IsDebugEnabled()) {
        config["PERF_COUNT"] = CONFIG_VALUE(YES);
      }
    #endif
      if (global_context_.device_type.find("MYRIAD") != std::string::npos) {
    #if defined(OPENVINO_2020_3)
        if (subgraph_context_.set_vpu_config) {
          config["VPU_DETECT_NETWORK_BATCH"] = CONFIG_VALUE(NO);
        }
        if (global_context_.enable_vpu_fast_compile) {
          config["VPU_HW_INJECT_STAGES"] = CONFIG_VALUE(NO);
          config["VPU_COPY_OPTIMIZATION"] = CONFIG_VALUE(NO);
        }
    #else
        if (subgraph_context_.set_vpu_config) {
          config["MYRIAD_DETECT_NETWORK_BATCH"] = CONFIG_VALUE(NO);
        }
        if (global_context_.enable_vpu_fast_compile) {
          config["MYRIAD_HW_INJECT_STAGES"] = CONFIG_VALUE(NO);
          config["MYRIAD_COPY_OPTIMIZATION"] = CONFIG_VALUE(NO);
        }
    #endif
    #if defined(OPENVINO_2021_4)
      //to check preprocessing inside model
      config["MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL"] = CONFIG_VALUE(NO);
    #endif
      }
      try {
        exe_network_ = global_context_.ie_core.LoadNetwork(*ie_cnn_network_, hw_target, config);
      } catch (const Exception& e) {
        ORT_THROW(log_tag + " Exception while Loading Network for graph: " + subgraph_context_.subgraph_name + ": " + e.what());
      } catch (...) {
        ORT_THROW(log_tag + " Exception while Loading Network for graph " + subgraph_context_.subgraph_name);
      }
      LOGS_DEFAULT(INFO) << log_tag << "Loaded model to the plugin";
      if(global_context_.use_compiled_network && hw_target == "MYRIAD") {
        LOGS_DEFAULT(INFO) << log_tag << "Dumping the compiled blob for this model into the directory 'ov_compiled_blobs'";
        std::ofstream compiled_blob_dump{ov_compiled_blobs_dir + "/" + model_blob_name};
        exe_network_.Export(compiled_blob_dump);
      }
    }
  }
  //The infer_requests_ pool will be intialized with a default value of 8 infer_request's
  //The nireq value can also be configured to any num_of_threads during runtime
  size_t nireq = global_context_.num_of_threads;
  LOGS_DEFAULT(INFO) << log_tag << "The value of nireq being used is: " << nireq;
#ifndef NDEBUG
  if (openvino_ep::backend_utils::IsDebugEnabled()) {
    std::cout << "The value of nireq being used is: " << nireq << std::endl;
  }
#endif
  inferRequestsQueue_ = std::unique_ptr<InferRequestsQueue>(new InferRequestsQueue(exe_network_, nireq));
}

// Starts an asynchronous inference request for data in slice indexed by batch_slice_idx on
// an Infer Request indexed by infer_req_idx
void BasicBackend::StartAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, std::shared_ptr<InferenceEngine::InferRequest> infer_request) {
  auto graph_input_info = exe_network_.GetInputsInfo();

  size_t index = 0;
  for (auto input_info_iter = graph_input_info.begin();
       input_info_iter != graph_input_info.end(); ++input_info_iter, ++index) {
    // Get OpenVINO's input buffer
    InferenceEngine::Blob::Ptr graph_input_blob;
    std::string input_name = input_info_iter->first;
    try {
      graph_input_blob = infer_request->GetBlob(input_name);

    } catch (const Exception& e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for input: " + input_name);
    }
    auto precision = input_info_iter->second->getPrecision();
    size_t batch_slice = 0;
    FillInputBlob(graph_input_blob, index, batch_slice, input_name, ort, context, precision, subgraph_context_);
  }
  // Start Async inference
  try {
    infer_request->StartAsync();
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Couldn't start Inference: " + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Couldn't start Inference");
  }
}

// Wait for asynchronous inference completion on an Infer Request object indexed by infer_req_idx
// and copy the results into a slice location within the batched output buffer indexed by batch_slice_idx
void BasicBackend::CompleteAsyncInference(Ort::CustomOpApi& ort, OrtKernelContext* context, std::shared_ptr<InferenceEngine::InferRequest> infer_request) {
  // Wait for Async inference completion
  try {
    infer_request->Wait(WaitMode::RESULT_READY);
  } catch (const Exception& e) {
    ORT_THROW(log_tag + " Exception with completing Inference" + e.what());
  } catch (...) {
    ORT_THROW(log_tag + " Exception with completing Inference");
  }
  auto graph_output_info = exe_network_.GetOutputsInfo();

  for (auto output_info_iter = graph_output_info.begin();
       output_info_iter != graph_output_info.end(); ++output_info_iter) {
    // Get OpenVINO's output blob
    InferenceEngine::Blob::Ptr graph_output_blob;
    auto output_name = output_info_iter->first;
    try {
      graph_output_blob = infer_request->GetBlob(output_name);
    } catch (const Exception& e) {
      ORT_THROW(log_tag + " Cannot access IE Blob for output: " + output_name + e.what());
    } catch (...) {
      ORT_THROW(log_tag + " Cannot access IE Blob for output: " + output_name);
    }
    size_t batch_size = 1;
    auto output_tensor = GetOutputTensor(ort, context, batch_size, infer_request, output_name, subgraph_context_.output_names);
    auto precision = output_info_iter->second->getPrecision();

    size_t batch_slice = 0;
    FillOutputBlob(graph_output_blob, output_tensor, ort, precision, batch_slice);
  }
#if defined(OPENVINO_2021_1) || defined(OPENVINO_2021_2) || defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
  if (!const_outputs_map_.empty()) {
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ort, context, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(ort, node, output_tensor);
    }
  }
#endif
}

void BasicBackend::Infer(Ort::CustomOpApi& ort, OrtKernelContext* context) {
  // Preliminary Thread safety mechanism
  // currently allows a maximum of 8 Infer request's to paralelly execute at the same time

  LOGS_DEFAULT(INFO) << log_tag << "Running graph " << subgraph_context_.subgraph_name;
  LOGS_DEFAULT(INFO) << log_tag << "In Infer";

  if (subgraph_context_.is_constant) {
#if defined(OPENVINO_2021_1)  || defined(OPENVINO_2021_2) || defined(OPENVINO_2021_3) || defined(OPENVINO_2021_4)
    for (auto item : const_outputs_map_) {
      auto out_name = item.first;
      auto node = item.second;
      auto output_tensor = GetOutputTensor(ort, context, out_name, subgraph_context_.output_names, node);
      FillOutputsWithConstantData(ort, node, output_tensor);
    }
#endif
    // Get Output tensors
    LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
  } else {
      //Requesting for an idle infer_request from a pool of infer_requests_
      std::shared_ptr<InferenceEngine::InferRequest> infer_request;
      try {
      infer_request = inferRequestsQueue_->getIdleRequest();
	    } catch (const Exception& e) {
      ORT_THROW(log_tag + " No idle Infer Requests found from the infer_requests_ pool! " + e.what());
	    } catch (...) {
      ORT_THROW(log_tag + "No idle Infer Requests!");
      }
      StartAsyncInference(ort, context, infer_request);
      CompleteAsyncInference(ort, context, infer_request);
  
      // Get Output tensors
      LOGS_DEFAULT(INFO) << log_tag << "Inference successful";
      //Once the inference is completed, the infer_request becomes free and is placed back into pool of infer_requests_
      inferRequestsQueue_->putIdleRequest(infer_request);
#ifndef NDEBUG
    if (openvino_ep::backend_utils::IsDebugEnabled()) {
      inferRequestsQueue_->printstatus();  //Printing the elements of infer_requests_ vector pool only in debug mode
      std::string& hw_target = (global_context_.device_id != "") ? global_context_.device_id : global_context_.device_type;
      printPerformanceCounts(infer_request, std::cout, hw_target);
    }
#endif
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime

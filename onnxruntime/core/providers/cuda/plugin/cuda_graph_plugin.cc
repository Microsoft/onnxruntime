// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_graph_plugin.h"

#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace onnxruntime {
namespace cuda_plugin {

// --- CudaGraphSet ---

CudaGraphSet::~CudaGraphSet() {
  Clear();
}

void CudaGraphSet::Clear() {
  for (auto& it : cuda_graphs_) {
    cudaGraphExecDestroy(it.second);
  }
  cuda_graphs_.clear();
}

bool CudaGraphSet::Contains(CudaGraphAnnotation_t id) const {
  return cuda_graphs_.find(id) != cuda_graphs_.end();
}

void CudaGraphSet::Put(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec) {
  if (Contains(id)) {
    throw std::runtime_error(
        "CudaGraphSet: trying to capture a graph with annotation id " +
        std::to_string(id) + " that is already used. Use a different annotation id.");
  }
  cuda_graphs_.emplace(id, graph_exec);
}

cudaGraphExec_t CudaGraphSet::Get(CudaGraphAnnotation_t id) const {
  auto it = cuda_graphs_.find(id);
  if (it == cuda_graphs_.end()) {
    throw std::runtime_error(
        "CudaGraphSet: no captured graph for annotation id " + std::to_string(id));
  }
  return it->second;
}

// --- CudaGraphManager ---

CudaGraphManager::CudaGraphManager(cudaStream_t stream) : stream_(stream) {
}

CudaGraphManager::~CudaGraphManager() {
  Reset();
}

void CudaGraphManager::SetStream(cudaStream_t stream) {
  stream_ = stream;
}

void CudaGraphManager::CaptureBegin(CudaGraphAnnotation_t id) {
  if (!IsGraphCaptureAllowedOnRun(id)) {
    throw std::runtime_error("CudaGraphManager: capture not allowed for annotation id " +
                             std::to_string(id));
  }

  if (cuda_graph_set_.Contains(id)) {
    throw std::runtime_error("CudaGraphManager: graph with annotation id " +
                             std::to_string(id) + " already captured");
  }

  PL_CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
  PL_CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
}

void CudaGraphManager::CaptureEnd(CudaGraphAnnotation_t id) {
  cudaGraph_t graph = nullptr;
  PL_CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph));
  if (graph == nullptr) {
    throw std::runtime_error("CudaGraphManager: cudaStreamEndCapture returned NULL graph");
  }

  cudaGraphExec_t graph_exec = nullptr;
  cudaError_t instantiate_err = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
  cudaError_t destroy_err = cudaGraphDestroy(graph);
  if (instantiate_err != cudaSuccess) {
    if (graph_exec != nullptr) {
      cudaGraphExecDestroy(graph_exec);
    }
    PL_CUDA_CALL_THROW(instantiate_err);
  }
  if (destroy_err != cudaSuccess) {
    cudaGraphExecDestroy(graph_exec);
    PL_CUDA_CALL_THROW(destroy_err);
  }

  try {
    cuda_graph_set_.Put(id, graph_exec);
    has_captured_any_ = true;
  } catch (...) {
    cudaGraphExecDestroy(graph_exec);
    throw;
  }
}

OrtStatus* CudaGraphManager::Replay(CudaGraphAnnotation_t id, bool sync) {
  if (!cuda_graph_set_.Contains(id)) {
    return Ort::GetApi().CreateStatus(
        ORT_INVALID_ARGUMENT,
        (std::string("CUDA graph replay error: graph not found for id ") +
         std::to_string(id))
            .c_str());
  }

  cudaGraphExec_t graph_exec = cuda_graph_set_.Get(id);

  cudaError_t err = cudaGraphLaunch(graph_exec, stream_);
  if (err != cudaSuccess) {
    return Ort::GetApi().CreateStatus(
        ORT_EP_FAIL,
        (std::string("CUDA graph launch error: ") + cudaGetErrorName(err) +
         ": " + cudaGetErrorString(err))
            .c_str());
  }

  if (sync) {
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph sync error: ") + cudaGetErrorName(err) +
           ": " + cudaGetErrorString(err))
              .c_str());
    }
  } else {
    // Flush WDDM command queue so cudaGraphLaunch is submitted to the GPU hardware immediately.
    err = cudaStreamQuery(stream_);
    if (err != cudaSuccess && err != cudaErrorNotReady) {
      return Ort::GetApi().CreateStatus(
          ORT_EP_FAIL,
          (std::string("CUDA graph stream query error: ") + cudaGetErrorName(err) +
           ": " + cudaGetErrorString(err))
              .c_str());
    }
  }

  return nullptr;
}

bool CudaGraphManager::IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t id) const {
  return id != kCudaGraphAnnotationSkip;
}

bool CudaGraphManager::IsGraphCaptured(CudaGraphAnnotation_t id) const {
  return cuda_graph_set_.Contains(id);
}

void CudaGraphManager::Reset() {
  if (has_captured_any_) {
    recapture_immediately_ = true;
  }
  cuda_graph_set_.Clear();
  run_count_.clear();
}

bool CudaGraphManager::IsGraphCaptureAllowed(CudaGraphAnnotation_t id, int min_runs) const {
  if (!IsGraphCaptureAllowedOnRun(id)) {
    return false;
  }
  if (recapture_immediately_) {
    return true;
  }
  auto it = run_count_.find(id);
  if (it == run_count_.end()) {
    return false;
  }
  return it->second >= min_runs;
}

void CudaGraphManager::IncrementRegularRunCount(CudaGraphAnnotation_t id) {
  auto it = run_count_.find(id);
  if (it == run_count_.end()) {
    run_count_[id] = 1;
    return;
  }
  it->second++;
}

}  // namespace cuda_plugin
}  // namespace onnxruntime

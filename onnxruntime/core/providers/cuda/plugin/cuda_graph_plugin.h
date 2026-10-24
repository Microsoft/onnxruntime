// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Plugin-side CUDA graph manager. Manages cudaGraph_t / cudaGraphExec_t lifecycle
// for CUDA graph capture and replay in the plugin EP. This is a standalone port
// of the bundled CUDAGraphManager (cuda_graph.h) without framework dependencies.

#pragma once

#include "cuda_plugin_utils.h"

#include <unordered_map>

namespace onnxruntime {
namespace cuda_plugin {

using CudaGraphAnnotation_t = int;

constexpr CudaGraphAnnotation_t kCudaGraphAnnotationSkip = -1;
constexpr CudaGraphAnnotation_t kCudaGraphAnnotationDefault = 0;

/// Storage for captured CUDA graph executables, keyed by annotation ID.
class CudaGraphSet {
 public:
  CudaGraphSet() = default;
  ~CudaGraphSet();

  CudaGraphSet(const CudaGraphSet&) = delete;
  CudaGraphSet& operator=(const CudaGraphSet&) = delete;

  void Clear();
  bool Contains(CudaGraphAnnotation_t id) const;
  void Put(CudaGraphAnnotation_t id, cudaGraphExec_t graph_exec);
  cudaGraphExec_t Get(CudaGraphAnnotation_t id) const;

 private:
  std::unordered_map<CudaGraphAnnotation_t, cudaGraphExec_t> cuda_graphs_;
};

/// Orchestrates CUDA graph capture, instantiation, and replay.
class CudaGraphManager {
 public:
  CudaGraphManager() = default;
  explicit CudaGraphManager(cudaStream_t stream);
  ~CudaGraphManager();

  CudaGraphManager(const CudaGraphManager&) = delete;
  CudaGraphManager& operator=(const CudaGraphManager&) = delete;

  void SetStream(cudaStream_t stream);

  /// Begin capturing CUDA operations on the stream.
  void CaptureBegin(CudaGraphAnnotation_t id);

  /// End capture, instantiate the graph, and store the executable.
  void CaptureEnd(CudaGraphAnnotation_t id);

  /// Launch a previously captured graph. Returns OrtStatus on error.
  OrtStatus* Replay(CudaGraphAnnotation_t id, bool sync = true);

  /// Returns false if the annotation ID indicates capture should be skipped.
  bool IsGraphCaptureAllowedOnRun(CudaGraphAnnotation_t id) const;

  /// Returns true if a graph has been captured for the given annotation ID.
  bool IsGraphCaptured(CudaGraphAnnotation_t id) const;

  /// Destroy all captured graph executables.
  void Reset();

  /// Track warm-up runs before allowing graph capture.
  /// Returns true when enough warm-up runs have occurred for the given annotation ID.
  bool IsGraphCaptureAllowed(CudaGraphAnnotation_t id, int min_runs) const;

  /// Increment the warm-up run count for a given annotation ID.
  void IncrementRegularRunCount(CudaGraphAnnotation_t id);

 private:
  CudaGraphSet cuda_graph_set_;
  cudaStream_t stream_ = nullptr;  // Does not own the stream.

  /// Tracks the number of regular (non-captured) runs per annotation ID.
  /// Graph capture is deferred until the count reaches `min_num_runs_before_cuda_graph_capture`.
  std::unordered_map<CudaGraphAnnotation_t, int> run_count_;
  bool has_captured_any_ = false;
  bool recapture_immediately_ = false;
};

}  // namespace cuda_plugin
}  // namespace onnxruntime

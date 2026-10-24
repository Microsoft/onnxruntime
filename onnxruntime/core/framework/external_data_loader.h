// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <filesystem>
#include <memory>
#include <string_view>
#include <vector>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/ortdevice.h"
#include "core/platform/env.h"

struct OrtMemoryInfo;

namespace onnxruntime {
class IAllocator;
#ifndef SHARED_PROVIDER
class Tensor;
#endif
class Stream;

namespace common {
class Status;
}

// Data transfer interface.
// Keep default virtual methods inline so shared providers do not require a non-exported RTTI symbol.
class IExternalDataLoader {
 public:
  virtual ~IExternalDataLoader() = default;

  virtual bool CanLoad(const OrtMemoryInfo& target_memory_info) const = 0;

  virtual bool SupportsDataType([[maybe_unused]] int32_t tensor_data_type) const { return true; }

#if defined(ENABLE_D3D12_FILE_LOADING)
  // Returns true when the loader creates the tensor's backing allocation instead of
  // writing into a tensor allocated by the framework.
  virtual bool CreatesTensorForDevice([[maybe_unused]] const OrtDevice& target_device) const { return false; }

  // Begins one external-weight load transaction for a SessionState initialization.
  // A transaction has two phases:
  // 1. ORT registers each eligible initializer as it is discovered, then commits the
  //    complete candidate set to the loader.
  // 2. ORT asks the loader for each committed tensor through LoadTensor().
  // EndLoad() ends the transaction after tensor loading or any earlier failure.
  //
  // Implementations should discard stale transaction state and initialize any
  // transaction-wide resources. Returning an error prevents the transaction from
  // continuing and causes EndLoad() to be called.
  virtual common::Status BeginLoad() const { return common::Status::OK(); }

  // Registers one external tensor with the active transaction. The path, offset, and length
  // have already been resolved and validated against the TensorProto by the caller.
  // This incremental registration lets ORT report candidates as it discovers them
  // without retaining a separate core-side candidate collection. The loader owns the
  // candidate metadata needed to plan file access, destination allocation, and shared
  // device work. ORT does not call LoadTensor() during this phase.
  //
  // RegisterLoadCandidate() calls occur after a successful BeginLoad() and before
  // CommitLoadCandidates(). The same file may be referenced by multiple tensors, and
  // a transaction may contain no candidates. Returning an error ends the transaction.
  // The default implementation retains no candidate state.
  virtual common::Status RegisterLoadCandidate([[maybe_unused]] const Env& env,
                                               [[maybe_unused]] const std::filesystem::path& data_file_path,
                                               [[maybe_unused]] std::string_view tensor_name,
                                               [[maybe_unused]] FileOffsetType data_offset,
                                               [[maybe_unused]] SafeInt<size_t> data_length) const {
    return common::Status::OK();
  }

  // Commits the complete set of registered candidates before tensor loading begins.
  // This is where a loader can coalesce file ranges, allocate destinations, perform
  // shared I/O or device work, and synchronize the results. On success, the loader
  // must either make each accepted candidate ready for a subsequent LoadTensor() call
  // or make CanLoad() return false so normal loading can take over.
  //
  // ORT then makes a separate initializer-deserialization pass and calls LoadTensor()
  // for each tensor still claimed by this loader. The loader must retain unclaimed
  // committed tensors until LoadTensor() transfers their ownership or EndLoad()
  // releases them. This method is called even when no candidates were registered.
  // Long-running work should check is_canceled and return MODEL_LOAD_CANCELED when
  // cancellation is observed. Returning any error causes EndLoad() to be called.
  // The default implementation has no work to commit.
  virtual common::Status CommitLoadCandidates(
      [[maybe_unused]] const std::function<bool()>& is_canceled) const {
    return common::Status::OK();
  }

  // Ends the active transaction. Cancel outstanding work and release candidate
  // metadata and resources still owned by the loader, without invalidating allocations
  // already transferred by successful LoadTensor() calls. This is called after all
  // expected LoadTensor() calls or after any earlier failure or cancellation. It may
  // be called more than once, so implementations must be noexcept and idempotent.
  virtual void EndLoad() const noexcept {}
#endif

  // Tensor should be allocated with the correct memory info and size. A loader that
  // creates tensors for the target device replaces it with one backed by memory
  // owned through allocator.
#if defined(ENABLE_D3D12_FILE_LOADING)
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] std::string_view tensor_name,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] const std::shared_ptr<IAllocator>& allocator,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#endif

  // Tensor should be already allocated with the correct memory info and size.
#if defined(__wasm__)
  virtual common::Status LoadTensor([[maybe_unused]] const Env& env,
                                    [[maybe_unused]] const std::filesystem::path& data_file_path,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#else
  virtual common::Status LoadTensor([[maybe_unused]] const RandomAccessFile& file,
                                    [[maybe_unused]] FileOffsetType data_offset,
                                    [[maybe_unused]] SafeInt<size_t> data_length,
                                    [[maybe_unused]] Tensor& tensor) const {
    ORT_NOT_IMPLEMENTED(__FUNCTION__, " is not implemented");
  }
#endif
};

#if defined(__wasm__)

enum class ExternalDataLoadType {
  CPU = 0,
#if defined(USE_JSEP) || defined(USE_WEBGPU)
  WEBGPU_BUFFER = 1,
#endif
};

// Check a mounted file's range before allocating a destination buffer.
common::Status ValidateWebAssemblyExternalDataRange(const std::filesystem::path& data_file_path,
                                                    FileOffsetType data_offset,
                                                    SafeInt<size_t> data_length);

// Entry point for loading external data implementation using inline JavaScript.
common::Status LoadWebAssemblyExternalData(const Env& env,
                                           const std::filesystem::path& data_file_path,
                                           FileOffsetType data_offset,
                                           SafeInt<size_t> data_length,
                                           ExternalDataLoadType load_type,
                                           void* tensor_data);

#endif

}  // namespace onnxruntime

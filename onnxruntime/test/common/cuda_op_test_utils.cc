// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>

#if defined(USE_CUDA)
#include "cuda_runtime_api.h"
#endif

namespace onnxruntime {
namespace test {

int GetCudaArchitecture() {
  // This will cache the result so we only call cudaGetDeviceProperties once.
  // Usually, we test on a single GPU or multiple GPUs of same architecture, so it's fine to cache the result.
  static int cuda_arch = -1;

#if defined(USE_CUDA)
  if (cuda_arch == -1) {
    int current_device_id = 0;
    cudaGetDevice(&current_device_id);
    // must wait GPU idle, otherwise cudaGetDeviceProperties might fail
    cudaDeviceSynchronize();
    cudaDeviceProp prop;

    // When cudaGetDeviceProperties fails, just return -1 and no error is raised.
    // If cuda device has issue, test will fail anyway so no need to raise error here.
    if (cudaSuccess == cudaGetDeviceProperties(&prop, current_device_id)) {
      cuda_arch = prop.major * 100 + prop.minor * 10;
    }

    // Log GPU compute capability
    if (cuda_arch == -1) {
      std::cout << "WARNING: CUDA is not available or failed to initialize" << std::endl;
    } else {
      std::cout << "GPU Compute Capability: SM "
                << cuda_arch / 100 << "." << (cuda_arch % 100) / 10
                << " (value: " << cuda_arch << ")" << std::endl;
    }
  }
#endif

  return cuda_arch;
}

}  // namespace test
}  // namespace onnxruntime

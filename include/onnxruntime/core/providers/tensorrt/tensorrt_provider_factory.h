// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_Tensorrt, _In_ OrtSessionOptions* options, int device_id);
ORT_API_STATUS(SessionOptionsAppendExecutionProvider_TensorRT, _In_ OrtSessionOptions* options, const OrtTensorRTProviderOptions* tensorrt_options);

#ifdef __cplusplus
}
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

bool InitProvidersSharedLibrary();
void UnloadSharedProviders();

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "migraphx_inc.h"

namespace onnxruntime {

// -----------------------------------------------------------------------
// Error handling
// -----------------------------------------------------------------------

template <typename ERRTYPE, bool THRW>
bool RocmCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg = "");

#define HIP_CALL(expr) (RocmCall<hipError_t, false>((expr), #expr, "HIP", hipSuccess))
#define HIP_CALL_THROW(expr) (RocmCall<hipError_t, true>((expr), #expr, "HIP", hipSuccess))

}  // namespace onnxruntime

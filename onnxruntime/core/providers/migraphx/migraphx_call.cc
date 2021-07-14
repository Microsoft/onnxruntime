// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unistd.h>
#include <string.h>
#include "migraphx_call.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {

using namespace common;

template <typename ERRTYPE>
const char* RocmErrString(ERRTYPE x) {
  ORT_NOT_IMPLEMENTED();
}

#define CASE_ENUM_TO_STR(x) \
  case x:                   \
    return #x

template <>
const char* RocmErrString<hipError_t>(hipError_t x) {
  (void)hipDeviceSynchronize();
  return hipGetErrorString(x);
}

template <typename ERRTYPE, bool THRW>
bool RocmCall(ERRTYPE retCode, const char* exprString, const char* libName, ERRTYPE successCode, const char* msg) {
  if (retCode != successCode) {
    try {
      char hostname[HOST_NAME_MAX];
      if (gethostname(hostname, HOST_NAME_MAX) != 0)
        strcpy(hostname, "?");
      int currentHipDevice;
      (void)hipGetDevice(&currentHipDevice);
      (void)hipGetLastError();  // clear last HIP error
      static char str[1024];
      snprintf(str, 1024, "%s failure %d: %s ; GPU=%d ; hostname=%s ; expr=%s; %s",
               libName, (int)retCode, RocmErrString(retCode), currentHipDevice,
               hostname,
               exprString, msg);
      if (THRW) {
        // throw an exception with the error info
        ORT_THROW(str);
      } else {
        LOGS_DEFAULT(ERROR) << str;
      }
    } catch (const std::exception& e) {  // catch, log, and rethrow since HIP code sometimes hangs in destruction, so we'd never get to see the error
      if (THRW) {
        ORT_THROW(e.what());
      } else {
        LOGS_DEFAULT(ERROR) << e.what();
      }
    }
    return false;
  }
  return true;
}

template bool RocmCall<hipError_t, false>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg);
template bool RocmCall<hipError_t, true>(hipError_t retCode, const char* exprString, const char* libName, hipError_t successCode, const char* msg);

}  // namespace onnxruntime

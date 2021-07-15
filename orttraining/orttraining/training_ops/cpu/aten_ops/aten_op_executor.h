// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <dlpack/dlpack.h>
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {
namespace aten_ops {

typedef bool (*IsTensorArgumentFunc)(const char* op_name, const char* overload_name, size_t index);
typedef std::vector<DLManagedTensor*> (*ExecuteATenOperatorFunc)(const char* op_name, const char* overload_name,
                                                                 const std::vector<DLManagedTensor*>& dlpacks);

class ATenOperatorExecutor {
 public:
  static ATenOperatorExecutor& Instance() { return InstanceImpl(); }

  static void Initialize(void* p_is_tensor_argument_func_raw, void* p_execute_aten_op_func_raw) {
    InstanceImpl(p_is_tensor_argument_func_raw, p_execute_aten_op_func_raw);
  }

  bool IsTensorArgument(const std::string& op_name, const std::string& overload_name, size_t index) {
    ORT_ENFORCE(p_is_tensor_argument_func_, "ATenOperatorExecutor is not initialized.");
    return p_is_tensor_argument_func_(op_name.c_str(), overload_name.c_str(), index);
  }
  std::vector<DLManagedTensor*> operator()(const std::string& op_name, const std::string& overload_name,
                                           const std::vector<DLManagedTensor*>& dlpacks) {
    ORT_ENFORCE(p_execute_aten_op_func_, "ATenOperatorExecutor is not initialized.");
    return p_execute_aten_op_func_(op_name.c_str(), overload_name.c_str(), dlpacks);
  }

 private:
  static ATenOperatorExecutor& InstanceImpl(void* p_is_tensor_argument_func_raw = nullptr,
                                            void* p_execute_aten_op_func_raw = nullptr) {
    static ATenOperatorExecutor instance(p_is_tensor_argument_func_raw, p_execute_aten_op_func_raw);
    return instance;
  }

  ATenOperatorExecutor(void* p_is_tensor_argument_func_raw, void* p_execute_aten_op_func_raw) {
    ORT_ENFORCE(p_is_tensor_argument_func_raw && p_execute_aten_op_func_raw);
    p_is_tensor_argument_func_ = reinterpret_cast<IsTensorArgumentFunc>(p_is_tensor_argument_func_raw);
    p_execute_aten_op_func_ = reinterpret_cast<ExecuteATenOperatorFunc>(p_execute_aten_op_func_raw);
  }

  IsTensorArgumentFunc p_is_tensor_argument_func_;
  ExecuteATenOperatorFunc p_execute_aten_op_func_;
};

}  // namespace aten_ops
}  // namespace contrib
}  // namespace onnxruntime

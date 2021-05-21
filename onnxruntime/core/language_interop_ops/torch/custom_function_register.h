// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <mutex>
#include <unordered_map>
#include <vector>
#include <Python.h>

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

class OrtTorchFunctionPool final {
 public:
  static OrtTorchFunctionPool& GetInstance() {
    static OrtTorchFunctionPool instance_;
    return instance_;
  }

  /// AutogradFunction includes ForwardCore and BackwardCore.
  /// ForwardCore is the apply() function pointer.
  /// BackwardCore is the backward() function pointer.
  void RegisterTorchAutogradFunction(const std::string& key, PyObject* obj, const bool override = false);
  void UnregisterTorchAutogradFunction(const std::string& key);
  PyObject* GetForwardCore(const std::string& key);   // The "key" is the "name" attribute in PythonOp.
  PyObject* GetBackwardCore(const std::string& key);  // The "key" is the "name" attribute in PythonOpGrad.

  /// Context is torch backward gradient function pointer, and
  /// it is a property of forward run outputs (tensors), its lifecycle
  /// is along with forward run outputs in PyTorch design.
  int64_t RegisterContext(PyObject* auto_grad_context);
  void UnRegisterContext(int64_t context_index);
  PyObject* GetContext(int64_t context_index);

  /// ForwardRunner/BackwardRunner are "glue" codes written code that interacting
  /// with C++ kernels during Python function invoking.
  void RegisterForwardRunner(PyObject* obj, const bool override = false);
  void UnregisterForwardRunner();
  void RegisterBackwardRunner(PyObject* obj, const bool override = false);
  void UnregisterBackwardRunner();
  PyObject* GetForwardRunner();
  PyObject* GetBackwardRunner();

 private:
  OrtTorchFunctionPool();
  ~OrtTorchFunctionPool();
  OrtTorchFunctionPool(const OrtTorchFunctionPool&) = delete;
  OrtTorchFunctionPool& operator=(const OrtTorchFunctionPool&) = delete;

  PyObject* forward_runner;
  PyObject* backward_runner;
  std::unordered_map<std::string, PyObject*> forward_core_pool;
  std::unordered_map<std::string, PyObject*> backward_core_pool;

  std::unordered_map<int64_t, PyObject*> func_context_pool;
  std::mutex func_context_pool_mutex_;
};
}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
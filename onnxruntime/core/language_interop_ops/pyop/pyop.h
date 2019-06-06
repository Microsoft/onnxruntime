#pragma once
#ifdef _WIN32
#include <Windows.h>
#define LIB_PYOP "onnxruntime_pywrapper.dll"
#define LOAD_PYOP_LIB(n,v,m) ORT_ENFORCE((v=LoadLibraryA(n))!=nullptr,m)
#else
#ifdef __APPLE__
#define LIB_PYOP "./libonnxruntime_pywrapper.dylib"
#else
#define LIB_PYOP "./libonnxruntime_pywrapper.so"
#endif
#define LOAD_PYOP_LIB(n,v,m) ORT_ENFORCE((v=dlopen(n,RTLD_NOW|RTLD_GLOBAL))!=nullptr,m)
#define HMODULE void*
#include "dlfcn.h"
#endif
#include "core/platform/env.h"
#define LOAD_PYOP_SYM(n,v,m) ORT_ENFORCE(Env::Default().GetSymbolFromLibrary(handle_,n,reinterpret_cast<void**>(&v))==Status::OK(),m)
#include "core/framework/ml_value.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/op_kernel_context_internal.h"
#include <iostream>
#include <vector>
#include <unordered_map>

namespace onnxruntime {

using OnnxTypes   = std::vector<ONNXTensorElementDataType>;
using OnnxAttrs   = std::unordered_map<std::string, std::string>;
using PyOpLogFunc = std::function<void(const char*)>;

typedef bool Initialize();
typedef void ReleaseInstance(void*);
typedef bool InvokePythonFunc(void*,
                              const char*,
                              const std::vector<const void*>&,
                              const std::vector<int32_t>&,
                              const std::vector<std::vector<int64_t>>&,
                              std::vector<std::unique_ptr<char[]>>&,
                              std::vector<int32_t>&,
                              std::vector<std::vector<int64_t>>&,
                              std::function<void(const char*)>);
typedef const char* GetLastErrorMessage(std::string&);
typedef void* NewInstance(const char*, const char*, const OnnxAttrs&);

class PyOpLibProxy {

public:
    static const PyOpLibProxy& GetInstance();
    HMODULE              handle_                 = nullptr;
    Initialize*          initialize_             = nullptr;
    NewInstance*         new_instance_           = nullptr;
    InvokePythonFunc*    invoke_python_func_     = nullptr;
    ReleaseInstance*     release_instance_       = nullptr;
    GetLastErrorMessage* get_last_error_message_ = nullptr;
private:
    PyOpLibProxy();
    ~PyOpLibProxy();
};

struct PyCustomKernel {

    PyCustomKernel(Ort::CustomOpApi   ort,
                   const OnnxAttrs&   attrs,
                   const std::string& module,
                   const std::string& class_name,
                   const std::string& compute,
                   PyOpLogFunc        logging_func);
    ~PyCustomKernel();
    void    GetOutputShape(OrtKernelContext*, size_t, OrtTensorTypeAndShapeInfo*);
    void    Compute(OrtKernelContext* context);
    int32_t GetType(const OrtValue* input) const;
private:
    Ort::CustomOpApi ort_;
    OnnxAttrs        attrs_;
    std::string      module_;
    std::string      class_name_;
    std::string      compute_;
    void*            instance_ = nullptr;
    PyOpLogFunc      logging_func_;
};

struct PyCustomOp: Ort::CustomOpBase<PyCustomOp, PyCustomKernel> {

    PyCustomOp(const OnnxAttrs&    attrs,
               const OnnxTypes&    inputs_type,
               const OnnxTypes&    outputs_type,
               const std::string&  module,
               const std::string&  class_name,
               const std::string&  compute      = "compute",
               PyOpLogFunc         logging_func = [](const char*){});
    void* CreateKernel(Ort::CustomOpApi api, const OrtKernelInfo*);
    const char* GetName() const;
    size_t GetInputTypeCount() const;
    ONNXTensorElementDataType GetInputType(size_t index) const;
    size_t GetOutputTypeCount() const;
    ONNXTensorElementDataType GetOutputType(size_t index) const;
private:
    OnnxAttrs      attrs_;
    OnnxTypes      inputs_type_;
    OnnxTypes      outputs_type_;
    std::string    module_;
    std::string    class_name_;
    std::string    compute_;
    PyOpLogFunc    logging_func_;
};//struct PyCustomOp

PyCustomOp* LoadPyOp(const ONNX_NAMESPACE::NodeProto& node_proto, PyOpLogFunc log_func);
}//namespace onnxruntime

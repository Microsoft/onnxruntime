// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_c_api.h"
#include "core/session/allocator_impl.h"
#include "core/session/inference_session_utils.h"
#include "core/session/IOBinding.h"
#include "core/framework/allocator.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/execution_provider.h"
#include "core/framework/utils.h"
#include <cassert>
#include <cstring>
#include <functional>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/ml_value.h"
#include "core/session/environment.h"
#include "core/framework/callback.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/inference_session.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/framework/data_types.h"
#include "abi_session_options_impl.h"
#include "core/framework/TensorSeq.h"
#include "core/platform/ort_mutex.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_factory.h"
#endif

#include <comdef.h>
#include "runner/training_runner.h"
#include "runner/orttraining_c_api.h"
#include "runner/orttraining_apis.h"

#include "core/framework/bfc_arena.h"
#include "cxxopts.hpp"
#include <condition_variable>
#include <mutex>
#include <tuple>
#include <map>
#include <sstream>
#include <iostream>

using namespace onnxruntime::logging;
using onnxruntime::BFloat16;
using onnxruntime::DataTypeImpl;
using onnxruntime::Environment;
using onnxruntime::IAllocator;
using onnxruntime::InputDefList;
using onnxruntime::MLFloat16;
using onnxruntime::OutputDefList;
using onnxruntime::Tensor;
using onnxruntime::ToOrtStatus;
using onnxruntime::common::Status;

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace std;
using namespace training;

namespace onnxruntime {
std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_CUDA(OrtDevice::DeviceId device_id,
                                                                               OrtCudnnConvAlgoSearch cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE,
                                                                               size_t cuda_mem_limit = std::numeric_limits<size_t>::max(),
                                                                               onnxruntime::ArenaExtendStrategy arena_extend_strategy = ArenaExtendStrategy::kNextPowerOfTwo,
                                                                               bool do_copy_in_default_stream = true);
}

#define MAX_BUFFER 1024

#ifndef ORT_STATUS_PTR
#ifdef _WIN32
#define ORT_STATUS_PTR _Check_return_ _Ret_maybenull_ OrtStatusPtr
#else
#define ORT_STATUS_PTR OrtStatus*
#endif
#endif

// Return the OrtStatus if it indicates an error
#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status)                      \
      return _status;                 \
  } while (0)

// Convert internal onnxruntime::Status to OrtStatus and return if there's an error
#define ORT_API_RETURN_IF_STATUS_NOT_OK(expr) \
  do {                                        \
    auto _status = (expr);                    \
    if (!_status.IsOK())                      \
      return ToOrtStatus(_status);            \
  } while (0)

#define TENSOR_READ_API_BEGIN                          \
  API_IMPL_BEGIN                                       \
  auto v = reinterpret_cast<const ::OrtValue*>(value); \
  auto& tensor = v->Get<onnxruntime::Tensor>();

#define TENSOR_READWRITE_API_BEGIN \
  API_IMPL_BEGIN                   \
  auto v = (value);                \
  auto tensor = v->GetMutable<onnxruntime::Tensor>();


static char* StrDup(const std::string& str, _Inout_ OrtAllocator* allocator) {
  char* output_string = reinterpret_cast<char*>(allocator->Alloc(allocator, str.size() + 1));
  memcpy(output_string, str.c_str(), str.size());
  output_string[str.size()] = '\0';
  return output_string;
}

static char* StrDup(const wchar_t* str, _Inout_ OrtAllocator* allocator) {
  _bstr_t b(str);
  return StrDup((char*)b, allocator);
}

static char* StrDup(const onnxruntime::PathString& str, _Inout_ OrtAllocator* allocator) {
  return StrDup(str.c_str(), allocator);
}

static char* StrDup(const wchar_t* val) {
  _bstr_t b(val);
  return _strdup((char*)b);
}


//=================================================================================================
//  Helper Classes 
//=================================================================================================

struct OrtShape {
  std::vector<size_t> dim_;
};

struct OrtValueCollection {
  size_t capacity_;
  size_t count_;
  OrtValue** rgvalues_;
  char** rgnames_;
  bool* rgname_owned_;
};

class OrtValueCollectionEx : public OrtValueCollection {
 public:
  OrtValueCollectionEx(int nCapacity) {
    capacity_ = nCapacity;
    count_ = 0;
    rgvalues_ = new OrtValue*[nCapacity];
    rgnames_ = new char*[nCapacity];
    rgname_owned_ = new bool[nCapacity];

    for (int i = 0; i < nCapacity; i++) {
      rgvalues_[i] = nullptr;
      rgnames_[i] = nullptr;
      rgname_owned_[i] = false;
    }
  }

  ~OrtValueCollectionEx() {
    // Do not delete the OrtValue pointers, for the memory is not owned by the array.
    if (rgvalues_ != nullptr)
      delete rgvalues_;

    for (int i = 0; i < capacity_; i++) {
      if (rgname_owned_[i])
        free(rgnames_[i]);
    }

    if (rgnames_ != nullptr)
      delete rgnames_;

    if (rgname_owned_ != nullptr)
      delete rgname_owned_;
  }

  bool Add(OrtValue* val, char* szName, bool bNameOwned = false) {
    if (count_ == capacity_)
      return false;

    rgvalues_[count_] = val;
    rgnames_[count_] = szName;
    rgname_owned_[count_] = bNameOwned;
    count_++;
    return true;
  }

  bool SetAt(size_t nIdx, OrtValue* val, char* szName, bool bNameOwned) {
    if (nIdx >= capacity_)
      return false;

    rgvalues_[nIdx] = val;

    if (rgnames_[nIdx] != nullptr && rgname_owned_[nIdx])
      free(rgnames_[nIdx]);

    rgnames_[nIdx] = szName;
    rgname_owned_[nIdx] = bNameOwned;
    return true;
  }

  void BeforeUsingAsInput(int queue_id) {
    for (size_t i = 0; i < count_; i++) {
      if (rgvalues_[i]->Fence())
        rgvalues_[i]->Fence()->BeforeUsingAsInput(onnxruntime::kCpuExecutionProvider, queue_id);
    }
  }
};

struct InternalParameters {
 public:
  std::vector<std::string> rgstr_datafeed_names_;
  OrtShape expected_input_shape_;
  OrtShape expected_output_shape_;
};

struct InternalParameters;
typedef struct InternalParameters InternalTrainingParameters;

// The actual types defined have an Ort prefix
//ORT_RUNTIME_CLASS(TrainingParameters);
struct OrtTrainingParameters {
  struct onnxruntime::training::TrainingRunner::Parameters* prunner_param_;
  InternalTrainingParameters* pinternal_param_;
  char* pszinput_labels_;
  char* pszoutput_predictions_;
  char* pszoutput_loss_;
  char* pszloss_function_type_;
  char* pszinit_feed_names_;
  bool use_cuda_;
  bool use_tensorboard_;
  OrtDataGetBatchCallback fntraining_data_getbatch_;
  OrtDataGetBatchCallback fntesting_data_getbatch_;
  OrtValueCollection* ptraining_data_;
  OrtValueCollection* ptesting_data_;
  onnxruntime::training::TrainingRunner* ptraining_runner_;
  onnxruntime::training::IDataLoader* ptraining_data_loader_;
  onnxruntime::training::IDataLoader* ptesting_data_loader_;
};
typedef struct OrtTrainingParameters OrtTrainingParameters;

class OrtTrainingParametersEx : public OrtTrainingParameters {
 public:
  OrtTrainingParametersEx() {
    use_cuda_ = false;
    use_tensorboard_ = true;
    fntraining_data_getbatch_ = nullptr;
    fntesting_data_getbatch_ = nullptr;

    ptraining_data_ = nullptr;
    ptesting_data_ = nullptr;
    ptraining_data_loader_ = nullptr;
    ptesting_data_loader_ = nullptr;
    pszinput_labels_ = nullptr;
    pszoutput_predictions_ = nullptr;
    pszoutput_loss_ = nullptr;
    pszloss_function_type_ = nullptr; // Do not free, set to static string.
    pszinit_feed_names_ = nullptr;
    ptraining_runner_ = nullptr;
    prunner_param_ = new struct onnxruntime::training::TrainingRunner::Parameters(); 
    pinternal_param_ = new InternalParameters();
  }

  OrtTrainingParametersEx(OrtTrainingParameters p) {
    use_cuda_ = p.use_cuda_;
    use_tensorboard_ = p.use_tensorboard_;
    fntraining_data_getbatch_ = p.fntraining_data_getbatch_;
    fntesting_data_getbatch_ = p.fntesting_data_getbatch_;
    ptraining_data_ = p.ptraining_data_;
    ptesting_data_ = p.ptesting_data_;
    ptraining_data_loader_ = p.ptraining_data_loader_;
    ptesting_data_loader_ = p.ptesting_data_loader_;
    pszinput_labels_ = _strdup(p.pszinput_labels_);
    pszoutput_predictions_ = _strdup(p.pszoutput_predictions_);
    pszoutput_loss_ = _strdup(p.pszoutput_loss_);
    pszloss_function_type_ = p.pszloss_function_type_;  // Do not free, set to static string.
    pszinit_feed_names_ = _strdup(p.pszinit_feed_names_);
    ptraining_runner_ = nullptr;
    prunner_param_ = new struct onnxruntime::training::TrainingRunner::Parameters(*p.prunner_param_);
    pinternal_param_ = new InternalParameters();
  }

  ~OrtTrainingParametersEx() {
    CleanUp();
  }

  void CleanUp() {
    if (ptraining_data_loader_ != nullptr) {
      delete ptraining_data_loader_;
      ptraining_data_loader_ = nullptr;
    }

    if (ptesting_data_loader_ != nullptr) {
      delete ptesting_data_loader_;
      ptesting_data_loader_ = nullptr;
    }

    if (ptraining_data_ != nullptr) {
      delete ((OrtValueCollectionEx*)ptraining_data_);
      ptraining_data_ = nullptr;
    }

    if (ptesting_data_ != nullptr) {
      delete ((OrtValueCollectionEx*)ptesting_data_);
      ptesting_data_ = nullptr;
    }

    if (pszinput_labels_ != nullptr) {
      free(pszinput_labels_);
      pszinput_labels_ = nullptr;
    }

    if (pszoutput_predictions_ != nullptr) {
      free(pszoutput_predictions_);
      pszoutput_predictions_ = nullptr;
    }

    if (pszoutput_loss_ != nullptr) {
      free(pszoutput_loss_);
      pszoutput_loss_ = nullptr;
    }

    if (pszinit_feed_names_ != nullptr) {
      free(pszinit_feed_names_);
      pszinit_feed_names_ = nullptr;
    }

    if (ptraining_runner_ != nullptr) {
      delete ptraining_runner_;
      ptraining_runner_ = nullptr;
    }

    if (prunner_param_ != nullptr) {
      delete prunner_param_;
      prunner_param_ = nullptr;
    }

    if (pinternal_param_ != nullptr) {
      delete pinternal_param_;
      pinternal_param_ = nullptr;
    }
  }
};

class DataSetEx : public DataSet {
 public:
  DataSetEx(OrtTrainingParameters* pParam, OrtDataUse dataUse)
      : DataSet(((OrtTrainingParametersEx*)pParam)->pinternal_param_->rgstr_datafeed_names_) {
    m_pParam = pParam;
    m_dataUse = dataUse;
  }

  virtual size_t NumSamples() const {
    return m_pParam->prunner_param_->batch_size;
  }

  virtual std::vector<OrtValue> GetKthBatch(size_t batch_size, size_t k_th, AllocatorPtr allocator = nullptr) const {
    OrtDataGetBatchCallback fnGetData;
    OrtValueCollection* pData;

    if (m_dataUse == ORT_DATAUSE_TRAINING) {
      fnGetData = m_pParam->fntraining_data_getbatch_;
      pData = m_pParam->ptraining_data_;
    } else {
      fnGetData = m_pParam->fntesting_data_getbatch_;
      pData = m_pParam->ptesting_data_;
    }

    for (int i = 0; i < pData->capacity_; i++) {
      pData->rgvalues_[i] = nullptr;
    }

    fnGetData(m_pParam->prunner_param_->batch_size, pData, &m_pParam->pinternal_param_->expected_input_shape_, &m_pParam->pinternal_param_->expected_output_shape_);

    std::vector<OrtValue> result;
    for (int i = 0; i < pData->capacity_; i++) {
      if (pData->rgvalues_[i] != nullptr)
        result.push_back(*pData->rgvalues_[i]);
    }

    return result;
  }

 private:
  OrtTrainingParameters* m_pParam;
  OrtDataUse m_dataUse;
};

//=================================================================================================
//  C API Implementations - OrtTrainingApis
//=================================================================================================

ORT_API_STATUS_IMPL(OrtTrainingApis::CreateTrainingParameters, OrtTrainingParameters** out) {
  API_IMPL_BEGIN
  *out = new OrtTrainingParametersEx();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtTrainingApis::ReleaseTrainingParameters, _Frees_ptr_opt_ OrtTrainingParameters* ptr) {
  delete ((OrtTrainingParametersEx*)ptr);
}

ORT_API_STATUS_IMPL(OrtTrainingApis::CloneTrainingParameters, const OrtTrainingParameters* input, OrtTrainingParameters** out) {
  API_IMPL_BEGIN
  *out = new OrtTrainingParametersEx(*input);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_string, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingStringParameter key, _In_ const ORTCHAR_T* value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) 
  {
    case OrtTrainingStringParameter::ORT_TRAINING_MODEL_PATH:
      pParam->prunner_param_->model_name = _bstr_t(value);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_LOG_PATH:
      pParam->prunner_param_->log_dir = ToPathString(value);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_INPUT_LABELS:
      pParam->pszinput_labels_ = _strdup(_bstr_t(value));
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_PREDICTIONS:
      pParam->pszoutput_predictions_ = _strdup(_bstr_t(value));
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_LOSS:
      pParam->pszoutput_loss_ = _strdup(_bstr_t(value));
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_string, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingStringParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingStringParameter::ORT_TRAINING_MODEL_PATH:
      *ppvalue = StrDup(pParam->prunner_param_->model_name.c_str(), allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_LOG_PATH:
      *ppvalue = StrDup(pParam->prunner_param_->log_dir.c_str(), allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_INPUT_LABELS:
      *ppvalue = StrDup(pParam->pszinput_labels_, allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_PREDICTIONS:
      *ppvalue = StrDup(pParam->pszoutput_predictions_, allocator);
      break;

    case OrtTrainingStringParameter::ORT_TRAINING_OUTPUT_LOSS:
      *ppvalue = StrDup(pParam->pszoutput_loss_, allocator);
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingBooleanParameter key, _In_ const bool value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_CUDA:
      pParam->use_cuda_ = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_GIST:
      pParam->prunner_param_->use_gist = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_PROFILER:
      pParam->prunner_param_->use_profiler = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_TENSORBOARD:
      pParam->use_tensorboard_ = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_IS_PERFTEST:
      pParam->prunner_param_->is_perf_test = value;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_SHUFFLE_DATA:
      pParam->prunner_param_->shuffle_data = value;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingBooleanParameter key, _Out_ bool* pvalue) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_CUDA:
      *pvalue = pParam->use_cuda_;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_GIST:
      *pvalue = pParam->prunner_param_->use_gist;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_PROFILER:
      *pvalue = pParam->prunner_param_->use_profiler;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_USE_TENSORBOARD:
      *pvalue = pParam->use_tensorboard_;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_IS_PERFTEST:
      *pvalue = pParam->prunner_param_->is_perf_test;
      break;

    case OrtTrainingBooleanParameter::ORT_TRAINING_SHUFFLE_DATA:
      *pvalue = pParam->prunner_param_->shuffle_data;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_long, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingLongParameter key, _In_ const long value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_BATCH_SIZE:
      pParam->prunner_param_->eval_batch_size = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_PERIOD:
      pParam->prunner_param_->evaluation_period = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_NUM_TRAIN_STEPS:
      pParam->prunner_param_->num_train_steps = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_TRAIN_BATCH_SIZE:
      pParam->prunner_param_->batch_size = (size_t)value;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_DISPLAY_LOSS_STEPS:
      pParam->prunner_param_->display_loss_steps = (size_t)value;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_long, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingLongParameter key, _Out_ long* pvalue) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_BATCH_SIZE:
      *pvalue = (long)pParam->prunner_param_->eval_batch_size;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_EVAL_PERIOD:
      *pvalue = (long)pParam->prunner_param_->evaluation_period;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_NUM_TRAIN_STEPS:
      *pvalue = (long)pParam->prunner_param_->num_train_steps;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_TRAIN_BATCH_SIZE:
      *pvalue = (long)pParam->prunner_param_->batch_size;
      break;

    case OrtTrainingLongParameter::ORT_TRAINING_DISPLAY_LOSS_STEPS:
      *pvalue = (long)pParam->prunner_param_->display_loss_steps;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingParameter_double, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingNumericParameter key, _In_ const double value) {
  API_IMPL_BEGIN
  Status status;

  switch (key) {
    case OrtTrainingNumericParameter::ORT_TRAINING_LEARNING_RATE:
      pParam->prunner_param_->lr_params.initial_lr = (float)value;
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingParameter_double, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingNumericParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue) {
  API_IMPL_BEGIN
  Status status;
  char szBuffer[MAX_BUFFER];

  switch (key) {
    case OrtTrainingNumericParameter::ORT_TRAINING_LEARNING_RATE:
      snprintf(szBuffer, MAX_BUFFER - 1, "%f", pParam->prunner_param_->lr_params.initial_lr);
      *ppvalue = StrDup(szBuffer, allocator); 
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingOptimizer, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingOptimizer opt) {
  API_IMPL_BEGIN
  Status status;

  switch (opt) {
    case OrtTrainingOptimizer::ORT_TRAINING_OPTIMIZER_SGD:
      pParam->prunner_param_->training_optimizer_name = "SGDOptimizer";
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingOptimizer, _In_ OrtTrainingParameters* pParam,
                    _Out_ OrtTrainingOptimizer* popt) {
  API_IMPL_BEGIN
  Status status;

  if (pParam->prunner_param_->training_optimizer_name == "SGDOptimizer")
    *popt = OrtTrainingOptimizer::ORT_TRAINING_OPTIMIZER_SGD;

  else
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  return ToOrtStatus(status);
  API_IMPL_END
}


ORT_API_STATUS_IMPL(OrtTrainingApis::SetTrainingLossFunction, _In_ OrtTrainingParameters* pParam,
                    _In_ const OrtTrainingLossFunction loss) {
  API_IMPL_BEGIN
  Status status;

  switch (loss) {
    case OrtTrainingLossFunction::ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY:
      pParam->pszloss_function_type_ = "SoftmaxCrossEntropy";
      break;

    default:
      status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
      break;
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetTrainingLossFunction, _In_ OrtTrainingParameters* pParam,
                    _Out_ OrtTrainingLossFunction* ploss) {
  API_IMPL_BEGIN
  Status status;

  if (strcmp(pParam->pszloss_function_type_, "SoftmaxCrossEntropy") == 0)
    *ploss = OrtTrainingLossFunction::ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY;

  else
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  return ToOrtStatus(status);
  API_IMPL_END
}


// Currently these are only used to show training with C# interaction,  
// A more long term change is to add to the training_runner a uniqe key passed in 
// to the RunWithUpdate training_runner function.
std::map<std::string, std::tuple<OrtErrorFunctionCallback, OrtEvaluationFunctionCallback>> m_fnFunctionMap;
std::string m_strLastMapUsed; 

static void error_function_callback(const std::vector<std::string>& feed_names,
                           const std::vector<OrtValue>& feeds,
                           const std::vector<std::string>& fetch_names,
                           const std::vector<OrtValue>& fetches,
                           size_t /*step*/) {
  const OrtValue* label_o = &feeds[1];
  const std::string label = feed_names[1];
  const OrtValue* predict_o = &fetches[0];
  const std::string predict = fetch_names[0];
  const OrtValue* loss_o = &fetches[1];
  const std::string loss = fetch_names[1];
  const int queue_id = 0;

  std::string strLabel = feed_names[1];
  m_strLastMapUsed = strLabel;
  OrtErrorFunctionCallback errorFn = std::get<0>(m_fnFunctionMap[strLabel]);

  OrtValueCollectionEx col(3);
  col.Add((OrtValue*)label_o, (char*)label.c_str(), false);
  col.Add((OrtValue*)predict_o, (char*)predict.c_str(), false);
  col.Add((OrtValue*)loss_o, (char*)loss.c_str(), false);
  col.BeforeUsingAsInput(queue_id);

  errorFn((OrtValueCollection*)&col);
}

static void evaluation_function_callback(size_t num_samples, size_t step, const std::string str) {
  OrtEvaluationFunctionCallback evalFn = std::get<1>(m_fnFunctionMap[m_strLastMapUsed]);

  evalFn(num_samples, step);
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetupTrainingParameters, _In_ OrtTrainingParameters* pParam,
                                        OrtErrorFunctionCallback errorFn,
                                        OrtEvaluationFunctionCallback evalFn) {
  API_IMPL_BEGIN
  Status status;

  pParam->prunner_param_->model_path = ToPathString(pParam->prunner_param_->model_name) + ORT_TSTR(".onnx");
  pParam->prunner_param_->model_with_loss_func_path = ToPathString(pParam->prunner_param_->model_name) + ORT_TSTR("_with_cost.onnx");
  pParam->prunner_param_->model_with_training_graph_path = ToPathString(pParam->prunner_param_->model_name) + ORT_TSTR("_bw.onnx");
  pParam->prunner_param_->model_actual_running_graph_path = ToPathString(pParam->prunner_param_->model_name) + ORT_TSTR("_bw_running.onnx");
  pParam->prunner_param_->output_dir = ORT_TSTR(".");

  // Gist encode
  pParam->prunner_param_->model_gist_encode_path = ToPathString(pParam->prunner_param_->model_name) + ORT_TSTR("_encode_gist.onnx");

  if (strcmp(pParam->pszloss_function_type_, "SoftmaxCrossEntropy") == 0) {
    pParam->prunner_param_->loss_func_info = LossFunctionInfo(OpDef("SoftmaxCrossEntropy", kMSDomain, 1),
                                                      pParam->pszoutput_loss_,
                                                      {pParam->pszoutput_predictions_, pParam->pszinput_labels_});
  } 
  else {
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
    return ToOrtStatus(status);
  }

  pParam->prunner_param_->fetch_names = {pParam->pszoutput_predictions_, pParam->pszoutput_loss_};
  pParam->prunner_param_->error_function = error_function_callback;
  pParam->prunner_param_->post_evaluation_callback = evaluation_function_callback;
  m_fnFunctionMap.insert(std::make_pair(pParam->pszinput_labels_, std::make_pair(errorFn, evalFn)));

  // Setup CUDA
  if (pParam->use_cuda_) {
    // Use local rank as device ID of the associated CUDA EP.
    OrtDevice::DeviceId device_id = static_cast<OrtDevice::DeviceId>(MPIContext::GetInstance().GetLocalRank());
    pParam->prunner_param_->providers.emplace(kCudaExecutionProvider, CreateExecutionProviderFactory_CUDA(device_id));
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetupTrainingData, _In_ OrtTrainingParameters* pParam,
                    _In_ OrtDataGetBatchCallback trainingdataqueryFn, 
                    _In_ OrtDataGetBatchCallback testingdataqueryFn,
                    _In_ const ORTCHAR_T* szFeedNames) {
  API_IMPL_BEGIN
  Status status;

  pParam->fntraining_data_getbatch_ = trainingdataqueryFn;
  pParam->fntesting_data_getbatch_ = testingdataqueryFn;

  pParam->pszinit_feed_names_ = StrDup(szFeedNames);
  istringstream fstrm(pParam->pszinit_feed_names_);
  string strName;

  while (getline(fstrm, strName, ';')) {
    pParam->pinternal_param_->rgstr_datafeed_names_.push_back(strName);
  }

  int nInputCount = 2;

  pParam->ptraining_data_ = (OrtValueCollection*)new OrtValueCollectionEx(nInputCount);
  pParam->ptesting_data_ = (OrtValueCollection*)new OrtValueCollectionEx(nInputCount);

  // TODO: load these from the actual model once loaded.
  pParam->pinternal_param_->expected_input_shape_.dim_.push_back(784);
  pParam->pinternal_param_->expected_output_shape_.dim_.push_back(10);

  auto device_count = MPIContext::GetInstance().GetWorldSize();
  auto trainingData = std::make_shared<DataSetEx>(pParam, OrtDataUse::ORT_DATAUSE_TRAINING);
  auto testingData = std::make_shared<DataSetEx>(pParam, OrtDataUse::ORT_DATAUSE_TESTING);

  pParam->ptraining_data_loader_ = new SingleDataLoader(std::dynamic_pointer_cast<DataSet>(trainingData), pParam->pinternal_param_->rgstr_datafeed_names_);
  pParam->ptesting_data_loader_ = new SingleDataLoader(std::dynamic_pointer_cast<DataSet>(testingData), pParam->pinternal_param_->rgstr_datafeed_names_);

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::InitializeTraining, _In_ OrtEnv* pEnv, _In_ OrtTrainingParameters* pParam) {
  API_IMPL_BEGIN
  Status status;

  if (pParam->ptraining_runner_ != nullptr)
    delete pParam->ptraining_runner_;

  pParam->ptraining_runner_ = new TrainingRunner(*pParam->prunner_param_, pEnv->GetEnvironment());    
  status = pParam->ptraining_runner_->Initialize();

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::RunTraining, _In_ OrtTrainingParameters* pParam) {
  API_IMPL_BEGIN
  Status status;

  status = pParam->ptraining_runner_->Run(pParam->ptraining_data_loader_, pParam->ptesting_data_loader_);

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::EndTraining, _In_ OrtTrainingParameters* pParam) {
  API_IMPL_BEGIN
  Status status;

  status = pParam->ptraining_runner_->EndTraining(pParam->ptesting_data_loader_);

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetCount, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount) {
  API_IMPL_BEGIN
  Status status;

  *pnCount = pCol->count_;

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetCapacity, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount) {
  API_IMPL_BEGIN
  Status status;

  *pnCount = pCol->capacity_;

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _Outptr_ OrtValue** output, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppName) {
  API_IMPL_BEGIN
  Status status;

  if (nIdx < pCol->count_) {
    *output = pCol->rgvalues_[nIdx];
    if (pCol->rgnames_[nIdx] != nullptr)
      *ppName = StrDup(pCol->rgnames_[nIdx], allocator);
    else
      *ppName = nullptr;
  } 
  else {
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::SetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _In_ OrtValue* input, _In_ const ORTCHAR_T* pwszName) {
  API_IMPL_BEGIN
  Status status;

  if (nIdx < pCol->capacity_) {
    bool bNameOwned = false;
    char* pszName = "";

    if (pwszName != nullptr) {
      pszName = _strdup(_bstr_t(pwszName));
      bNameOwned = true;
    }

    ((OrtValueCollectionEx*)pCol)->SetAt(nIdx, input, pszName, bNameOwned);
  } 
  else {
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");
  }

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetDimCount, _In_ OrtShape* pShape, _Out_ size_t* pnCount) {
  API_IMPL_BEGIN
  Status status;

  *pnCount = pShape->dim_.size();

  return ToOrtStatus(status);
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtTrainingApis::GetDimAt, _In_ OrtShape* pShape, _In_ size_t nIdx, _Out_ size_t* output) {
  API_IMPL_BEGIN
  Status status;

  if (nIdx < pShape->dim_.size())
    *output = pShape->dim_[nIdx];
  else
    status = Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::INVALID_ARGUMENT, "");

  return ToOrtStatus(status);
  API_IMPL_END
}


static constexpr OrtTrainingApiBase ort_training_api_base = {
    &OrtTrainingApis::GetApi,
    &OrtTrainingApis::GetVersionString,
};

/* Rules on how to add a new Ort API version

In general, NEVER remove or rearrange the members in this structure unless a new version is being created. The
goal is for newer shared libraries of the Onnx Runtime to work with binaries targeting the previous versions.
In order to do that we need to ensure older binaries get the older interfaces they are expecting.

See \onnxruntime\core\session\onnxruntime_c_api.cc notes for OrtApi table.

*/
static constexpr OrtTrainingApi ort_training_api_1_to_6 = {
    // NOTE: The ordering of these fields MUST not change after that version has shipped since existing binaries depend on this ordering.
    &OrtTrainingApis::CreateTrainingParameters,
    &OrtTrainingApis::CloneTrainingParameters,

    &OrtTrainingApis::SetTrainingParameter_string,
    &OrtTrainingApis::GetTrainingParameter_string,

    &OrtTrainingApis::SetTrainingParameter_bool,
    &OrtTrainingApis::GetTrainingParameter_bool,

    &OrtTrainingApis::SetTrainingParameter_long,
    &OrtTrainingApis::GetTrainingParameter_long,

    &OrtTrainingApis::SetTrainingParameter_double,
    &OrtTrainingApis::GetTrainingParameter_double,

    &OrtTrainingApis::SetTrainingOptimizer,
    &OrtTrainingApis::GetTrainingOptimizer,

    &OrtTrainingApis::SetTrainingLossFunction,
    &OrtTrainingApis::GetTrainingLossFunction,

    &OrtTrainingApis::SetupTrainingParameters,
    &OrtTrainingApis::SetupTrainingData,

    &OrtTrainingApis::InitializeTraining,
    &OrtTrainingApis::RunTraining,
    &OrtTrainingApis::EndTraining,

    &OrtTrainingApis::GetCount,
    &OrtTrainingApis::GetCapacity,
    &OrtTrainingApis::GetAt,
    &OrtTrainingApis::SetAt,

    &OrtTrainingApis::GetDimCount,
    &OrtTrainingApis::GetDimAt,

    &OrtTrainingApis::ReleaseTrainingParameters,
};


ORT_API(const OrtTrainingApi*, OrtTrainingApis::GetApi, uint32_t version) {
  if (version >= 1 && version <= 6)
    return &ort_training_api_1_to_6;

  return nullptr;  // Unsupported version
}

ORT_API(const char*, OrtTrainingApis::GetVersionString) {
  return ORT_VERSION;
}

const OrtTrainingApiBase* ORT_API_CALL OrtGetTrainingApiBase(void) NO_EXCEPTION {
  return &ort_training_api_base;
}

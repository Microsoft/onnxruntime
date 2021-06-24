// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/environment.h"
#include "core/framework/allocatormgr.h"
#include "core/graph/constants.h"
#include "core/graph/op.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/operator_sets_ml.h"
#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
#include "onnx/defs/operator_sets_training.h"
#endif
#endif
#ifndef DISABLE_CONTRIB_OPS
#include "core/graph/contrib_ops/contrib_defs.h"
#endif
#ifdef ML_FEATURIZERS
#include "core/graph/featurizers_ops/featurizers_defs.h"
#endif
#ifdef USE_DML
#include "core/graph/dml_ops/dml_defs.h"
#endif

#include "core/platform/env.h"
#include "core/util/thread_utils.h"
#include "core/session/allocator_impl.h"

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include "core/platform/tracing.h"
#endif

#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
#include "orttraining/core/graph/training_op_defs.h"
#endif
#ifdef ENABLE_TRAINING
#include "orttraining/core/graph/gradient_builder_registry.h"
#include "orttraining/core/graph/loss_function_registry.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder_registry.h"
#endif

namespace onnxruntime {
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

std::once_flag schemaRegistrationOnceFlag;

Status Environment::Create(std::unique_ptr<logging::LoggingManager> logging_manager,
                           std::unique_ptr<Environment>& environment,
                           const OrtThreadingOptions* tp_options,
                           bool create_global_thread_pools) {
  environment = std::unique_ptr<Environment>(new Environment());
  auto status = environment->Initialize(std::move(logging_manager), tp_options, create_global_thread_pools);
  return status;
}

Status Environment::RegisterAllocator(AllocatorPtr allocator) {
  const auto& mem_info = allocator->Info();
  // We don't expect millions of allocators getting registered. Hence linear search should be fine.
  auto ite = std::find_if(std::begin(shared_allocators_),
                          std::end(shared_allocators_),
                          [&mem_info](const AllocatorPtr& alloc_ptr) { return alloc_ptr->Info() == mem_info; });
  if (ite != shared_allocators_.end()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Allocator with this OrtMemoryInfo is already registered.");
  }
  shared_allocators_.insert(ite, allocator);
  return Status::OK();
}

Status Environment::CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg* arena_cfg) {
  // TODO should we allow sharing of non-CPU allocators?
  if (mem_info.device.Type() != OrtDevice::CPU) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Only CPU devices are supported for now.");
  }

  // determine if arena should be used
  bool create_arena = mem_info.alloc_type == OrtArenaAllocator;

#ifdef USE_JEMALLOC
#if defined(USE_MIMALLOC_ARENA_ALLOCATOR) || defined(USE_MIMALLOC_STL_ALLOCATOR)
#error jemalloc and mimalloc should not both be enabled
#endif
  //JEMalloc already has memory pool, so just use device allocator.
  create_arena = false;
#elif !(defined(__amd64__) || defined(_M_AMD64))
  //Disable Arena allocator for x86_32 build because it may run into infinite loop when integer overflow happens
  create_arena = false;
#endif

  AllocatorPtr allocator_ptr;
  // create appropriate DeviceAllocatorRegistrationInfo and allocator based on create_arena
  if (create_arena) {
    // defaults in case arena_cfg is nullptr (not supplied by the user)
    size_t max_mem = 0;
    int arena_extend_strategy = -1;
    int initial_chunk_size_bytes = -1;
    int max_dead_bytes_per_chunk = -1;
    int initial_growth_chunk_size_bytes = -1;

    // override with values from the user supplied arena_cfg object
    if (arena_cfg) {
      max_mem = arena_cfg->max_mem;

      arena_extend_strategy = arena_cfg->arena_extend_strategy;
      // validate the value here
      if (!(arena_extend_strategy == -1 || arena_extend_strategy == 0 || arena_extend_strategy == 1)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Received invalid value for arena extend strategy."
                               " Valid values can be either 0, 1 or -1.");
      }

      initial_chunk_size_bytes = arena_cfg->initial_chunk_size_bytes;
      max_dead_bytes_per_chunk = arena_cfg->max_dead_bytes_per_chunk;
      initial_growth_chunk_size_bytes = arena_cfg->initial_growth_chunk_size_bytes;
    }

    OrtArenaCfg l_arena_cfg{max_mem, arena_extend_strategy, initial_chunk_size_bytes, max_dead_bytes_per_chunk,
                            initial_growth_chunk_size_bytes};
    AllocatorCreationInfo alloc_creation_info{
        [mem_info](int) { return std::make_unique<TAllocator>(mem_info); },
        0,
        create_arena,
        l_arena_cfg};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  } else {
    AllocatorCreationInfo alloc_creation_info{[](int) { return std::make_unique<TAllocator>(); },
                                              0, create_arena};
    allocator_ptr = CreateAllocator(alloc_creation_info);
  }

  return RegisterAllocator(allocator_ptr);
}

Status Environment::Initialize(std::unique_ptr<logging::LoggingManager> logging_manager,
                               const OrtThreadingOptions* tp_options,
                               bool create_global_thread_pools) {
  auto status = Status::OK();

  logging_manager_ = std::move(logging_manager);

  // create thread pools
  if (create_global_thread_pools) {
    create_global_thread_pools_ = true;
    OrtThreadPoolParams to = tp_options->intra_op_thread_pool_params;
    if (to.name == nullptr) {
      to.name = ORT_TSTR("intra-op");
    }
    intra_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTRA_OP);
    to = tp_options->inter_op_thread_pool_params;
    if (to.name == nullptr) {
      to.name = ORT_TSTR("inter-op");
    }
    inter_op_thread_pool_ = concurrency::CreateThreadPool(&Env::Default(), to, concurrency::ThreadPoolType::INTER_OP);
  }

  ORT_TRY {
#if !defined(ORT_MINIMAL_BUILD)
    // Register Microsoft domain with min/max op_set version as 1/1.
    std::call_once(schemaRegistrationOnceFlag, []() {
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSExperimentalDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSNchwcDomain, 1, 1);
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSFeaturizersDomain, 1, 1);
#ifdef USE_DML
      ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(onnxruntime::kMSDmlDomain, 1, 1);
#endif
// Register contributed schemas.
// The corresponding kernels are registered inside the appropriate execution provider.
#ifndef DISABLE_CONTRIB_OPS
      contrib::RegisterContribSchemas();
#endif
#ifdef ML_FEATURIZERS
      featurizers::RegisterMSFeaturizersSchemas();
#endif
#ifdef USE_DML
      dml::RegisterDmlSchemas();
#endif

#ifndef DISABLE_ML_OPS
      RegisterOnnxMLOperatorSetSchema();
#endif

#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
      RegisterOnnxTrainingOperatorSetSchema();
#endif

#if defined(ENABLE_TRAINING) || defined(ENABLE_TRAINING_OPS)
      // preserve this order until <training schemas>: this depends on operatorsetschema registration.
      training::RegisterTrainingOpSchemas();
#endif
#ifdef ENABLE_TRAINING
      training::GradientBuilderRegistry::GetInstance().RegisterGradientBuilders();
      training::LossFunctionRegistry::GetInstance().RegisterNonOperatorLossFunctions();
      training::OptimizerBuilderRegistry::GetInstance().RegisterBuilders();
      training::OptimizerGraphBuilderRegistry::GetInstance().RegisterGraphBuilders();
      // <training schemas>
#endif
    });

    // Register MemCpy schema;

    // These ops are internal-only, so register outside of onnx
    static std::vector<std::string> all_fixed_size_types = []() {
      std::vector<std::string> all_types;
      std::vector<std::string> all_tensor_types = OpSchema::all_tensor_types_with_bfloat();
      std::vector<std::string> all_sequence_types = OpSchema::all_tensor_sequence_types();
      all_types.insert(all_types.end(), all_tensor_types.begin(), all_tensor_types.end());
      all_types.insert(all_types.end(), all_sequence_types.begin(), all_sequence_types.end());
      all_types.emplace_back("seq(tensor(bfloat16))");
      all_types.erase(std::remove_if(all_types.begin(), all_types.end(), 
                      [](const std::string& s) { return s.find("string") != std::string::npos; }), all_types.end());
      return all_types; }();

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyFromHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            all_fixed_size_types,
            "Constrain to all fixed size tensor and sequence types. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

    ORT_ATTRIBUTE_UNUSED ONNX_OPERATOR_SCHEMA(MemcpyToHost)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "T")
        .TypeConstraint(
            "T",
            all_fixed_size_types,
            "Constrain to all fixed size tensor and sequence types. If the dtype attribute is not provided this must be a valid output type.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput)
        .SetDoc(R"DOC(
Internal copy node
)DOC");

#endif  // !defined(ORT_MINIMAL_BUILD)
    // fire off startup telemetry (this call is idempotent)
    const Env& env = Env::Default();
    env.GetTelemetryProvider().LogProcessInfo();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = Status(ONNXRUNTIME, common::RUNTIME_EXCEPTION, std::string{"Exception caught: "} + ex.what());
    });
  }
  ORT_CATCH(...) {
    status = Status{ONNXRUNTIME, common::RUNTIME_EXCEPTION};
  }
  return status;
}

}  // namespace onnxruntime

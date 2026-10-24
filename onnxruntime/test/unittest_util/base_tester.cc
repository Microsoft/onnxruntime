// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/unittest_util/base_tester.h"

#include <csignal>
#include "gmock/gmock.h"

#include "core/common/logging/logging.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/model_load_utils.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "test/unittest_util/run_options_config_keys.h"
#include "test/unittest_util/test_allocator_manager.h"
#include "test/unittest_util/test_dynamic_plugin_ep.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"
#include "test/util/include/test_utils.h"

#ifdef ENABLE_TRAINING
#include "orttraining/core/session/training_session.h"
#endif

using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {
namespace {

#ifndef NDEBUG
void DebugTrap() {
#if _MSC_VER
  __debugbreak();
#else
  raise(SIGTRAP);
#endif
}
#endif

bool ShouldRouteCudaToDynamicPluginEp(const std::optional<std::string>& dynamic_plugin_ep_name) {
  // Route CUDA requests to the CUDA plugin EP when unit test main has initialized
  // dynamic plugin EP infrastructure with the CUDA plugin registration.
  return dynamic_plugin_ep_name.has_value() &&
         *dynamic_plugin_ep_name == dynamic_plugin_ep_infra::kCudaExecutionProviderPluginName;
}

}  // namespace

BaseTester::~BaseTester() {
#ifndef NDEBUG
  if (!testing_function_called_) {
    std::cerr << "Test was not executed." << std::endl;
    DebugTrap();
  }
#endif
}

void BaseTester::AddInitializers(onnxruntime::Graph& graph) {
  for (auto index : initializer_indexes_) {
    auto& data = input_data_[index];
    auto& tensor = data.data.Get<Tensor>();
    ONNX_NAMESPACE::TensorProto tensor_proto;

    // 1. set dimension
    auto& shape = tensor.Shape();
    for (auto& dim : shape.GetDims()) {
      tensor_proto.add_dims(dim);
    }

    // 2. set type
    tensor_proto.set_data_type(data.def.TypeAsProto()->tensor_type().elem_type());

    // 3. data
    if (data.def.TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      const std::string* string_data = tensor.Data<std::string>();
      for (auto i = 0; i < shape.Size(); i++) {
        tensor_proto.add_string_data(string_data[i]);
      }
    } else {
      // Note: need to use Tensor::CalculateTensorStorageSize (instead of shape.Size() * elem_size) to properly
      // calculate the storage size for sub-byte types (e.g., Int4 or Int2)
      auto buffer_size = Tensor::CalculateTensorStorageSize(tensor.DataType(), shape);
      utils::SetRawDataInTensorProto(tensor_proto, tensor.DataRaw(), buffer_size);
    }

    // 4. name
    tensor_proto.set_name(data.def.Name());
    graph.AddInitializedTensor(tensor_proto);
  }
}

void BaseTester::FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
                                         std::vector<std::string>& output_names) {
  for (auto& output : output_data_) {
    if (output.def.Exists()) {
      output_names.push_back(output.def.Name());
    }
  }

  FillFeeds(feeds);
}

void BaseTester::FillFeeds(std::unordered_map<std::string, OrtValue>& feeds) {
  for (size_t i = 0; i < input_data_.size(); ++i) {
    if (std::find(initializer_indexes_.begin(), initializer_indexes_.end(), i) == initializer_indexes_.end() &&
        input_data_[i].def.Exists() &&
        // We don't include optional type OrtValues of None because this is
        // how we expect users to deal with sending through "None"s as graph inputs
        // (i.e.) don't send them through at all
        input_data_[i].data.IsAllocated()) {
      feeds[input_data_[i].def.Name()] = input_data_[i].data;
    }
  }
}

void BaseTester::SetOutputAbsErr(const char* name, float v) {
  auto it = std::find_if(output_data_.begin(), output_data_.end(),
                         [name](Data& data) { return (data.def.Name() == name); });
  ORT_ENFORCE(it != output_data_.end());
  it->validation_params.absolute_error = optional<float>(v);
}

void BaseTester::SetOutputRelErr(const char* name, float v) {
  auto it = std::find_if(output_data_.begin(), output_data_.end(),
                         [name](Data& data) { return (data.def.Name() == name); });
  ORT_ENFORCE(it != output_data_.end());
  it->validation_params.relative_error = optional<float>(v);
}

void BaseTester::SetOutputTolerance(float abs_error, float rel_error) {
  for (auto& output : output_data_) {
    if (output.def.Exists()) {
      if (abs_error >= 0.0f) {
        output.validation_params.absolute_error = optional<float>(abs_error);
      }

      if (rel_error >= 0.0f) {
        output.validation_params.relative_error = optional<float>(rel_error);
      }
    }
  }
}

std::vector<int64_t> BaseTester::GetDimsForProto(gsl::span<const int64_t> dims) {
  std::vector<int64_t> dims_for_proto{dims.begin(), dims.end()};
  if (add_symbolic_dim_to_tensor_data_ >= 0 &&
      dims.size() > static_cast<size_t>(add_symbolic_dim_to_tensor_data_)) {
    dims_for_proto[add_symbolic_dim_to_tensor_data_] = -1;
  }
  return dims_for_proto;
}

void BaseTester::AddShapeToTensorData(NodeArg& node_arg, gsl::span<const int64_t> dims,
                                      const std::vector<std::string>* dim_params) {
  if (dim_params && !(dim_params->empty()) && add_shape_to_tensor_data_) {
    // If dim_params presents, configure node_arg's dim value based on dim_params, which supports symbolic dim and dim broadcast.
    const auto& dim_params_data = *dim_params;
    onnx::TensorShapeProto new_shape;

    // currently hard-code the reserved symbolic names.
    // TODO: when the list grows longer, consider move it to a better place.
    const static std::unordered_set<std::string> reserved_symbolic{"batch", "seq"};

    for (size_t i = 0; i < dim_params_data.size(); ++i) {
      if (reserved_symbolic.find(dim_params_data[i]) != reserved_symbolic.end()) {
        new_shape.add_dim()->set_dim_param(dim_params_data[i]);
      } else {
        ASSERT_TRUE(std::stoi(dim_params_data[i]) == dims[i]);
        new_shape.add_dim()->set_dim_value(dims[i]);
      }
    }
    node_arg.SetShape(new_shape);
  }
}

#if !defined(DISABLE_SPARSE_TENSORS)
static std::unique_ptr<SparseTensor> MakeSparseTensor(MLDataType data_type, const gsl::span<const int64_t>& dims) {
  TensorShape shape{dims};
  auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);
  auto p_tensor = std::make_unique<SparseTensor>(data_type, shape, std::move(allocator));
  return p_tensor;
}

void BaseTester::CopyDataToTensor(gsl::span<const std::byte> data, Tensor& dst) {
  ORT_ENFORCE(dst.SizeInBytes() >= data.size_bytes(), "Not enough space in the destination tensor");
  memcpy(dst.MutableDataRaw(), data.data(), data.size_bytes());
}

NodeArg BaseTester::MakeSparseNodeArg(int32_t dtype, const char* name, const gsl::span<const int64_t>& dims,
                                      const std::vector<std::string>* dim_params) {
  std::vector<int64_t> dims_for_proto = GetDimsForProto(dims);
  TSparseTensorProto type_proto(dtype, add_shape_to_tensor_data_ ? &dims_for_proto : nullptr);
  NodeArg node_arg(name, &type_proto.proto);
  AddShapeToTensorData(node_arg, dims, dim_params);
  return node_arg;
}

void BaseTester::AddSparseTensorData(std::vector<Data>& data, NodeArg node_arg,
                                     std::unique_ptr<SparseTensor> p_tensor,
                                     const ValidateOutputParams& check_params) {
  OrtValue value;
  auto ml_type = DataTypeImpl::GetType<SparseTensor>();
  value.Init(p_tensor.release(), ml_type, ml_type->GetDeleteFunc());
  data.push_back(Data(std::move(node_arg), std::move(value),
                      optional<float>(check_params.relative_error), optional<float>(check_params.absolute_error),
                      check_params.sort_output));
}

void BaseTester::AddSparseCooTensorData(std::vector<Data>& data,
                                        MLDataType data_type,
                                        const char* name,
                                        gsl::span<const int64_t> dims,
                                        gsl::span<const std::byte> values,
                                        gsl::span<const int64_t> indices,
                                        const ValidateOutputParams& check_params,
                                        const std::vector<std::string>* dim_params) {
  const auto elem_size = data_type->Size();
  const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();
  const auto nnz = values.size_bytes() / elem_size;
  ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
  ORT_ENFORCE((nnz == indices.size() || 2 * nnz == indices.size()), "Expecting indices to have either nnz or (2 * nnz) length");
  auto p_tensor = MakeSparseTensor(data_type, dims);
  auto mutator = p_tensor->MakeCooData(nnz, indices.size());
  CopyDataToTensor(values, mutator.Values());
  CopyDataToTensor(gsl::as_bytes(indices), mutator.Indices());

  NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
  AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor), check_params);
}

void BaseTester::AddSparseCooTensorStrings(std::vector<Data>& data,
                                           const char* name,
                                           gsl::span<const int64_t> dims,
                                           gsl::span<const std::string> values,
                                           gsl::span<const int64_t> indices,
                                           const std::vector<std::string>* dim_params) {
  auto data_type = DataTypeImpl::GetType<std::string>();
  const auto nnz = values.size();
  const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();
  ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
  ORT_ENFORCE((nnz == indices.size() || 2 * nnz == indices.size()), "Expecting indices to have either nnz or (2 * nnz) length");
  auto p_tensor = MakeSparseTensor(data_type, dims);
  // linear index is 1-D index, otherwise 2-D index
  auto mutator = p_tensor->MakeCooData(nnz, indices.size());
  auto mutable_values = mutator.Values().MutableDataAsSpan<std::string>();
  ORT_ENFORCE(values.size() == mutable_values.size(), "Must allocate space for values");
  std::copy(values.begin(), values.end(), mutable_values.begin());
  CopyDataToTensor(gsl::as_bytes(indices), mutator.Indices());
  NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
  AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor), ValidateOutputParams());
}

void BaseTester::AddSparseCsrTensorData(std::vector<Data>& data,
                                        MLDataType data_type,
                                        const char* name,
                                        gsl::span<const int64_t> dims,
                                        gsl::span<const std::byte> values,
                                        gsl::span<const int64_t> inner_indices,
                                        gsl::span<const int64_t> outer_indices,
                                        const ValidateOutputParams& check_params,
                                        const std::vector<std::string>* dim_params) {
  const auto elem_size = data_type->Size();
  const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();
  const auto nnz = values.size_bytes() / elem_size;
  ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
  ORT_ENFORCE(nnz == inner_indices.size(), "Expecting the same number of inner_indices as nnz");
  auto p_tensor = MakeSparseTensor(data_type, dims);

  auto mutator = p_tensor->MakeCsrData(nnz, inner_indices.size(), outer_indices.size());
  CopyDataToTensor(values, mutator.Values());
  CopyDataToTensor(gsl::as_bytes(inner_indices), mutator.Inner());
  CopyDataToTensor(gsl::as_bytes(outer_indices), mutator.Outer());

  NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
  AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor), check_params);
}

void BaseTester::AddSparseCsrTensorStrings(std::vector<Data>& data,
                                           const char* name,
                                           gsl::span<const int64_t> dims,
                                           gsl::span<const std::string> values,
                                           gsl::span<const int64_t> inner_indices,
                                           gsl::span<const int64_t> outer_indices,
                                           const std::vector<std::string>* dim_params) {
  auto data_type = DataTypeImpl::GetType<std::string>();
  const auto nnz = values.size();
  const auto dtype = data_type->AsPrimitiveDataType()->GetDataType();

  ORT_ENFORCE(dims.size() == 2U, "Expecting a 2-D dense shape");
  ORT_ENFORCE(nnz == inner_indices.size(), "Expecting the same number of inner_indices as nnz");
  auto p_tensor = MakeSparseTensor(data_type, dims);

  auto mutator = p_tensor->MakeCsrData(nnz, inner_indices.size(), outer_indices.size());
  auto mutable_values = mutator.Values().MutableDataAsSpan<std::string>();
  ORT_ENFORCE(values.size() == mutable_values.size(), "Must allocate space for values");
  std::copy(values.begin(), values.end(), mutable_values.begin());
  CopyDataToTensor(gsl::as_bytes(inner_indices), mutator.Inner());
  CopyDataToTensor(gsl::as_bytes(outer_indices), mutator.Outer());
  NodeArg node_arg = MakeSparseNodeArg(dtype, name, dims, dim_params);
  AddSparseTensorData(data, std::move(node_arg), std::move(p_tensor), ValidateOutputParams());
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

template <class SessionType>
void BaseTester::ExecuteModel(Model& model, SessionType& session,
                              ExpectResult expect_result,
                              const std::string& expected_failure_string,
                              const RunOptions* run_options,
                              const std::unordered_map<std::string, OrtValue>& feeds,
                              const std::vector<std::string>& output_names,
                              const std::string& provider_type,
                              bool allow_released_onnx_opset_only) {
  fetches_.clear();

  std::string s1;
  const bool rc = model.ToProto().SerializeToString(&s1);
  ASSERT_TRUE(rc) << "Failed to serialize proto to string";

  std::stringstream sstr(s1);
  EXPECT_STATUS_OK(session.Load(sstr, allow_released_onnx_opset_only));

  auto status = session.Initialize();
  if (!status.IsOK()) {
    ASSERT_EQ(expect_result, ExpectResult::kExpectFailure) << "Initialize failed but expected success: "
                                                           << status.ErrorMessage();

    // No need to check expected failure string if empty string is given.
    if (expected_failure_string.empty()) {
      return;
    }

    // Disable expected_failure_string checks for OpenVINO EP
    if (provider_type != kOpenVINOExecutionProvider) {
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
    }

    return;
  }

  RunOptions default_run_options;

  for (int i = 0; i < num_run_calls_; ++i) {
    fetches_.clear();
    status = session.Run(run_options ? *run_options : default_run_options, feeds, output_names, &fetches_);

    if (status.IsOK()) {
      ASSERT_EQ(expect_result, ExpectResult::kExpectSuccess) << "Run succeeded but expected failure.";
    } else {
      ASSERT_EQ(expect_result, ExpectResult::kExpectFailure) << "Run failed but expected success: "
                                                             << status.ErrorMessage();

      // No need to check expected failure string if empty string is given.
      if (expected_failure_string.empty()) {
        return;
      }

      // Disable expected_failure_string checks for MKL-DNN and OpenVINO EP's
      if (provider_type != kDnnlExecutionProvider &&
          provider_type != kOpenVINOExecutionProvider) {
        ASSERT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      }

      return;
    }
  }

  // Verify the outputs
  // Todo: support check output with map/sequence/....
  if (verify_output_) {
    if (custom_output_verifier_) {
      // do custom verification if provided
      custom_output_verifier_(fetches_, provider_type);
    } else {
      // default verification
      size_t idx = 0;
      for (auto& expected_data : output_data_) {
        OrtValue& ort_value = fetches_[idx];

        if (expected_data.def.Exists()) {  // optional edges won't exist (so skip them)
          const auto& name = expected_data.def.Name();
          if (!expected_data.data.IsAllocated()) {  // optional type output (None)
            EXPECT_TRUE(!ort_value.IsAllocated()) << "Expected to see an output of None for " << name
                                                  << " but instead got an output that wasn't None";

            // Make sure types align
            EXPECT_EQ(expected_data.data.Type(), ort_value.Type())
                << "Expected optional type: " << expected_data.data.Type() << " for " << name
                << " but instead got optional type: " << ort_value.Type();
          }

          else if (expected_data.data.IsTensor()) {
            // verify output shape inference when input defs have shape
            if (add_shape_to_tensor_data_) {
              auto out_shape_proto = expected_data.def.Shape();
              EXPECT_TRUE(out_shape_proto != nullptr);

              const auto tensor_shape = utils::GetTensorShapeFromTensorShapeProto(*out_shape_proto);
              const auto inferred_dims = tensor_shape.GetDims();
              const auto& expected_shape = expected_data.data.Get<Tensor>().Shape();
              EXPECT_TRUE(inferred_dims.size() == expected_shape.NumDimensions());

              for (size_t d = 0; d < inferred_dims.size(); ++d) {
                // check equal unless the input involved a symbolic dimension
                if (inferred_dims[d] != -1) {
                  EXPECT_EQ(expected_shape[d], inferred_dims[d]) << "Output idx = " << idx << " dim = " << d;
                }
              }
            }

            CheckOrtValuesAreEqual(name, expected_data.data, ort_value, expected_data.validation_params,
                                   provider_type);
          } else {
            CheckOrtValuesAreEqual(name, expected_data.data, ort_value, expected_data.validation_params,
                                   provider_type);
          }

          ++idx;

          // skip missing trailing optional outputs
          if (idx == fetches_.size()) {
            break;
          }
        }
      }
    }
  }
}

bool SetEpsForAllNodes(Graph& graph,
                       const std::vector<std::unique_ptr<IExecutionProvider>>& execution_providers,
                       const std::vector<std::shared_ptr<CustomRegistry>>* custom_registries,
                       const std::function<bool(const IExecutionProvider&)>& ep_only_uses_kernel_registry_fn) {
  const OpSchemaKernelTypeStrResolver kernel_type_str_resolver{};
  const KernelRegistry::TypeConstraintMap type_constraint_map{};

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == kConstant)
      continue;

    bool found = false;
    const auto& logger = DefaultLoggingManager().DefaultLogger();

    for (const auto& ep : execution_providers) {
      auto provider_type = ep->Type();

      node.SetExecutionProviderType(provider_type);

      if (!ep_only_uses_kernel_registry_fn(*ep)) {
        found = true;
        break;
      }

      if (std::shared_ptr<KernelRegistry> ep_kernel_registry = ep->GetKernelRegistry();
          ep_kernel_registry != nullptr) {
        // Check the EP has an impl for the node from builtin registry.
        if (KernelRegistry::HasImplementationOf(*ep_kernel_registry, node, ep->Type(), kernel_type_str_resolver,
                                                logger)) {
          found = true;
          break;
        }

        // check the internal NHWC domain if EP requests NHWC as it may only have a kernel registered in that domain
        if (ep->GetPreferredLayout() == DataLayout::NHWC) {
          const KernelCreateInfo* kci = nullptr;
          auto status = ep_kernel_registry->TryFindKernel(ep->Type(),
                                                          std::string_view(node.OpType()),
                                                          std::string_view(kMSInternalNHWCDomain),
                                                          node.SinceVersion(),
                                                          type_constraint_map,
                                                          logger,
                                                          &kci);
          if (status.IsOK() && kci != nullptr) {
            found = true;
            break;
          }
        }
      }

      // Check the EP has an impl for the node from custom_registries
      if (custom_registries != nullptr &&
          std::any_of(custom_registries->cbegin(), custom_registries->cend(),
                      [&](auto reg) {
                        return KernelRegistry::HasImplementationOf(*reg->GetKernelRegistry(), node, ep->Type(),
                                                                   kernel_type_str_resolver, logger);
                      })) {
        found = true;
        break;
      }
    }

    // We will reach here:
    //  - either we could not find an impl in all possible kernel registries
    //  - or we skip the registry search and blindly assign the node to the EP due to impl details.
    if (!found) {
      return false;
    }
  }

  // all nodes have been assigned an EP
  return true;
}

BaseTester& BaseTester::Config(const SessionOptions& sess_options) {
  ctx_.session_options = sess_options;
  return *this;
}

BaseTester& BaseTester::Config(ExpectResult expect_result, const std::string& expected_failure_string) {
  ctx_.expect_result = expect_result;
  ctx_.expected_failure_string = expected_failure_string;
  return *this;
}

BaseTester& BaseTester::ConfigExcludeEps(const std::unordered_set<std::string>& excluded_provider_types) {
  ctx_.excluded_provider_types = excluded_provider_types;
  return *this;
}

BaseTester& BaseTester::Config(const RunOptions* run_options) {
  ctx_.run_options = run_options;
  return *this;
}

BaseTester& BaseTester::ConfigEps(std::vector<std::unique_ptr<IExecutionProvider>>&& execution_providers) {
  ORT_ENFORCE(execution_providers.size() > 0);
  ctx_.run_with_specified_eps = true;
  ctx_.execution_providers = std::move(execution_providers);
  return *this;
}

BaseTester& BaseTester::Config(const Graph::ResolveOptions& resolve_options) {
  ctx_.resolve_options = resolve_options;
  return *this;
}

void BaseTester::Run(ExpectResult expect_result, const std::string& expected_failure_string,
                     const std::unordered_set<std::string>& excluded_provider_types,
                     const RunOptions* run_options,
                     std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers,
                     ExecutionMode execution_mode,
                     const Graph::ResolveOptions& options) {
  SessionOptions so;
  so.use_per_session_threads = false;
  so.session_logid = test_name_;
  so.session_log_verbosity_level = 1;
  so.execution_mode = execution_mode;
  so.use_deterministic_compute = use_determinism_;
  so.graph_optimization_level = TransformerLevel::Default;  // 'Default' == off

  Run(so, expect_result, expected_failure_string, excluded_provider_types, run_options, execution_providers, options);
}

#define ASSERT_PROVIDER_STATUS_OK(function)                                                         \
  do {                                                                                              \
    Status _tmp_status = function;                                                                  \
    ASSERT_TRUE(_tmp_status.IsOK()) << "provider: " << provider_type << ", error: " << _tmp_status; \
  } while (false)

void BaseTester::Run(SessionOptions so,
                     ExpectResult expect_result, const std::string& expected_failure_string,
                     const std::unordered_set<std::string>& excluded_provider_types,
                     const RunOptions* run_options,
                     std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers,
                     const Graph::ResolveOptions& options,
                     /*out*/ size_t* number_of_pre_packed_weights_counter,
                     /*out*/ size_t* number_of_shared_pre_packed_weights_counter) {
  if (execution_providers == nullptr) {
    ctx_.run_with_specified_eps = false;
    ctx_.execution_providers.clear();
  } else {
    ConfigEps(std::move(*execution_providers));
    // NOTE: some callsites do the following:
    //
    //   std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    //   execution_providers.push_back(DefaultCPUExecutionProvider());
    //   test.run(..., &execution_providers, ...);
    //   execution_providers[0] =  DefaultCUDAExecutionProvider();     //  <-- std::move cause segfault here.
    //   test.run(..., &execution_providers, ...);
    //
    // So we need to restore the old vector's size.
    execution_providers->resize(ctx_.execution_providers.size());
  }

  Config(so);
  Config(expect_result, expected_failure_string);
  Config(run_options);
  ConfigExcludeEps(excluded_provider_types);
  Config(options);

  RunWithConfig(number_of_pre_packed_weights_counter, number_of_shared_pre_packed_weights_counter);
}

void BaseTester::RunWithConfig(size_t* number_of_pre_packed_weights_counter,
                               size_t* number_of_shared_pre_packed_weights_counter) {
  std::string cur_provider = "not set";
  ORT_TRY {
    testing_function_called_ = true;
    fetches_.clear();

    // IsAllowReleasedONNXOpsetsOnlySet() checks for the appropriate env var in the process (i.e.) process-wide
    // `test_allow_released_onnx_opset_only_` is for this specific OpTester instance
    // We will only support released opsets iff IsAllowReleasedONNXOpsetsOnlySet() and
    // `test_allow_released_onnx_opset_only_` are both true
    auto allow_released_onnx_opset_only =
        test_allow_released_onnx_opset_only_ && model_load_utils::IsAllowReleasedONNXOpsetsOnlySet();

    if (allow_released_onnx_opset_only) {
      auto& onnx_released_versions =
          ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange::Instance().LastReleaseVersionMap();
      auto it = onnx_released_versions.find(domain_);
      if (it != onnx_released_versions.end() && opset_version_ > it->second) {
        LOGS_DEFAULT(WARNING) << "Encountered model with opset version greater than released onnx opset version. "
                              << "Skipping this test. To run this test set environment variable ALLOW_RELEASED_ONNX_OPSET_ONLY to \"0\". "
                              << "Opset version of current model is " << opset_version_
                              << ", the latest released onnx opset version is " << it->second << ".";
        GTEST_SKIP();
      }
    }

    const bool strict_shape_type_inference = ctx_.session_options.config_options.GetConfigOrDefault(
                                                 kOrtSessionOptionsConfigStrictShapeTypeInference, "1") == "1";
    const ModelOptions model_options(allow_released_onnx_opset_only, strict_shape_type_inference);

    Model* p_model = nullptr;
    CreateModelToTest(model_options, p_model);
    if (!p_model) {
      ASSERT_EQ(ctx_.expect_result, ExpectResult::kExpectFailure) << "Failed to create model to test.";
      return;
    }

    Model& model = *p_model;

    // Hookup the inputs and outputs
    std::unordered_map<std::string, OrtValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names);
    number_of_nodes_ = model.MainGraph().NumberOfNodes();

    // Run the model
    if (ctx_.run_with_specified_eps) {
      ExecuteModelForEps(std::move(ctx_.execution_providers), model, ctx_.session_options,
                         ctx_.expect_result, ctx_.expected_failure_string,
                         ctx_.run_options, feeds, output_names,
                         /*custom_registries=*/nullptr,
                         /*assign_ep_for_nodes=*/false,
                         allow_released_onnx_opset_only,
                         number_of_pre_packed_weights_counter,
                         number_of_shared_pre_packed_weights_counter);
    } else {
      // synthetic EP name for testing CoreML EP with ML Program
      constexpr const char* kCoreMLExecutionProviderMLProgram = "CoreMLExecutionProvider_MLProgram";

#ifdef USE_TENSORRT
      // only run trt ep to reduce test time
      static const std::vector<std::string> all_provider_types = {
          kTensorrtExecutionProvider,
      };
#else
      static const std::vector<std::string> all_provider_types = {
          kCpuExecutionProvider,
          kCudaExecutionProvider,
#ifdef ENABLE_CUDA_NHWC_OPS
          kCudaNHWCExecutionProvider,
#endif
          kDnnlExecutionProvider,
          kTensorrtExecutionProvider,
          kOpenVINOExecutionProvider,
          kDmlExecutionProvider,
          kAclExecutionProvider,
          kNnapiExecutionProvider,
          kVSINPUExecutionProvider,
          kCoreMLExecutionProvider,
          kCoreMLExecutionProviderMLProgram,
          kQnnExecutionProvider,
          kSnpeExecutionProvider,
          kXnnpackExecutionProvider,
#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS)
          kWebGpuExecutionProvider,
#endif
      };

      // need to special case any synthetic EP names in the exclude list
      if (ctx_.excluded_provider_types.count(kCoreMLExecutionProvider) > 0) {
        ctx_.excluded_provider_types.insert(kCoreMLExecutionProviderMLProgram);
      }
#endif

      const auto dynamic_plugin_ep_name = dynamic_plugin_ep_infra::GetEpName();
      const bool route_cuda_to_dynamic_plugin_ep = ShouldRouteCudaToDynamicPluginEp(dynamic_plugin_ep_name);

      std::optional<std::vector<std::string>> provider_types_including_dynamic_plugin_ep{};
      if (dynamic_plugin_ep_name.has_value() && !route_cuda_to_dynamic_plugin_ep) {
        ORT_ENFORCE(std::find(all_provider_types.begin(), all_provider_types.end(),
                              *dynamic_plugin_ep_name) == all_provider_types.end(),
                    "Dynamic plugin EP name conflicts with a known EP name: ", *dynamic_plugin_ep_name);
        provider_types_including_dynamic_plugin_ep = all_provider_types;
        provider_types_including_dynamic_plugin_ep->push_back(*dynamic_plugin_ep_name);
      }

      const auto all_provider_types_span =
          provider_types_including_dynamic_plugin_ep.has_value()
              ? gsl::span<const std::string>{*provider_types_including_dynamic_plugin_ep}
              : gsl::span<const std::string>{all_provider_types};

      bool has_run = false;

      for (const std::string& provider_type : all_provider_types_span) {
        if (ctx_.excluded_provider_types.count(provider_type) > 0)
          continue;

        cur_provider = provider_type;

        std::unique_ptr<IExecutionProvider> execution_provider;
        if (provider_type == onnxruntime::kCpuExecutionProvider)
          execution_provider = DefaultCpuExecutionProvider();
        else if (provider_type == onnxruntime::kCudaExecutionProvider)
          execution_provider = route_cuda_to_dynamic_plugin_ep ? dynamic_plugin_ep_infra::MakeEp() : DefaultCudaExecutionProvider();
#ifdef ENABLE_CUDA_NHWC_OPS
        else if (provider_type == onnxruntime::kCudaNHWCExecutionProvider)
          execution_provider = DefaultCudaNHWCExecutionProvider();
#endif
        else if (provider_type == onnxruntime::kDnnlExecutionProvider)
          execution_provider = DefaultDnnlExecutionProvider();
        else if (provider_type == onnxruntime::kOpenVINOExecutionProvider)
          execution_provider = DefaultOpenVINOExecutionProvider();
        else if (provider_type == onnxruntime::kTensorrtExecutionProvider)
          execution_provider = DefaultTensorrtExecutionProvider();
        else if (provider_type == onnxruntime::kNnapiExecutionProvider)
          execution_provider = DefaultNnapiExecutionProvider();
        else if (provider_type == onnxruntime::kVSINPUExecutionProvider)
          execution_provider = DefaultVSINPUExecutionProvider();
        else if (provider_type == onnxruntime::kRknpuExecutionProvider)
          execution_provider = DefaultRknpuExecutionProvider();
        else if (provider_type == onnxruntime::kAclExecutionProvider)
          execution_provider = DefaultAclExecutionProvider();
        else if (provider_type == onnxruntime::kCoreMLExecutionProvider)
          execution_provider = DefaultCoreMLExecutionProvider();
        else if (provider_type == kCoreMLExecutionProviderMLProgram)
          execution_provider = DefaultCoreMLExecutionProvider(/*use_mlprogram*/ true);
        else if (provider_type == onnxruntime::kSnpeExecutionProvider)
          execution_provider = DefaultSnpeExecutionProvider();
        else if (provider_type == onnxruntime::kQnnExecutionProvider)
          execution_provider = DefaultQnnExecutionProvider();
        else if (provider_type == onnxruntime::kXnnpackExecutionProvider)
          execution_provider = DefaultXnnpackExecutionProvider();
        else if (provider_type == onnxruntime::kDmlExecutionProvider)
          execution_provider = DefaultDmlExecutionProvider();
        else if (provider_type == onnxruntime::kWebGpuExecutionProvider)
          execution_provider = DefaultWebGpuExecutionProvider();
        else if (provider_type == dynamic_plugin_ep_name) {
          execution_provider = dynamic_plugin_ep_infra::MakeEp();
        }

        // skip if execution provider is disabled
        if (execution_provider == nullptr)
          continue;

        ExecuteModelForEps(
            [&execution_provider]() {
              std::vector<std::unique_ptr<IExecutionProvider>> ret;
              ret.emplace_back(std::move(execution_provider));
              return ret;
            }(),
            model, ctx_.session_options,
            ctx_.expect_result, ctx_.expected_failure_string,
            ctx_.run_options, feeds, output_names,
            &custom_session_registries_,
            /*try_assign_ep_for_nodes=*/true,
            allow_released_onnx_opset_only,
            number_of_pre_packed_weights_counter,
            number_of_shared_pre_packed_weights_counter);

        has_run = true;
        cur_provider = "not set";
      }

#ifdef USE_TENSORRT
      // We are allowing tests to be run with only TensorRT EP, but TensorRT EP may not support all tests and may be in excluded providers list.
      // So, no registered EPs were able to run the model is okay for this situation.
      ORT_UNUSED_PARAMETER(has_run);
#else
      EXPECT_TRUE(has_run) << "No registered execution providers were able to run the model.";
#endif
    }
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what() << "\nProvider:" << cur_provider << "\n";
    });
    // rethrow as some tests for error handling expect this
    ORT_RETHROW;
  }
}

int BaseTester::GetNumberOfNodesAfterRun() const { return number_of_nodes_; }

void BaseTester::ExecuteModelForEps(
    std::vector<std::unique_ptr<IExecutionProvider>>&& execution_providers,
    onnxruntime::Model& model,
    SessionOptions sess_options,  // session options is copied to avoid the side effect in this function
    onnxruntime::test::BaseTester::ExpectResult expect_result,
    const std::string& expected_failure_string,
    const onnxruntime::RunOptions* run_options,
    const std::unordered_map<std::string, OrtValue>& feeds,
    const std::vector<std::string>& output_names,
    const std::vector<std::shared_ptr<CustomRegistry>>* custom_registries,
    bool try_assign_ep_for_nodes,
    bool allow_released_onnx_opset_only,
    size_t* number_of_pre_packed_weights_counter,
    size_t* number_of_shared_pre_packed_weights_counter) {
  const auto dynamic_plugin_ep_name = dynamic_plugin_ep_infra::GetEpName();
  const bool route_cuda_to_dynamic_plugin_ep = ShouldRouteCudaToDynamicPluginEp(dynamic_plugin_ep_name);

  for (auto& entry : execution_providers) {
    // Be noted, entry in execution providers passed in OpTester will be std::moved in the first BaseTester::Run(),
    // To make the error more obvious to debug (instead of a segment fault), we do check explicitly here.
    ASSERT_TRUE(entry) << "Execution provider entry invalid.";

    if (route_cuda_to_dynamic_plugin_ep && entry->Type() == kCudaExecutionProvider) {
      auto plugin_ep = dynamic_plugin_ep_infra::MakeEp();
      ASSERT_TRUE(plugin_ep) << "Failed to create CUDA plugin EP while routing from CUDAExecutionProvider.";
      entry = std::move(plugin_ep);
    }

    if (entry->Type() == kDmlExecutionProvider) {
      sess_options.enable_mem_pattern = false;
      sess_options.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
      break;
    }
  }

  InferenceSession session_object{sess_options, GetEnvironment()};

  if (add_prepacked_shared_container_to_sessions_) {
    ASSERT_STATUS_OK(session_object.AddPrePackedWeightsContainer(&prepacked_weights_container_));
  }

  ASSERT_TRUE(!execution_providers.empty()) << "Empty execution providers vector.";
  if (try_assign_ep_for_nodes) {
    auto ep_only_uses_kernel_registry = [](const IExecutionProvider& ep) {
      const auto& provider_type = ep.Type();

      constexpr std::array kEpsThatCompileNodes{
          kOpenVINOExecutionProvider,
          kTensorrtExecutionProvider,  // uses kernel registry for Memcpy* nodes only
          kNnapiExecutionProvider,
          kVSINPUExecutionProvider,
          kCoreMLExecutionProvider,
          kDnnlExecutionProvider,
          kQnnExecutionProvider,
          kSnpeExecutionProvider,
      };

      // check list of known EPs that compile nodes
      if (const auto ep_it = std::find(kEpsThatCompileNodes.begin(), kEpsThatCompileNodes.end(),
                                       provider_type);
          ep_it != kEpsThatCompileNodes.end()) {
        return false;
      }

      const OrtEp* ort_ep = ep.GetOrtEp();

      if (ort_ep != nullptr) {  // This is a plugin EP

        if (ep.GetKernelRegistry() == nullptr) {
          // assume that a dynamic plugin EP which does not return a kernel registry does not use one
          return false;
        }

        if (ort_ep->Compile != nullptr) {
          // assume that a plugin EP that compiles nodes does not use a kernel registry for all nodes
          return false;
        }
      }

      // otherwise, assume that the EP uses a kernel registry
      return true;
    };

    if (!SetEpsForAllNodes(model.MainGraph(), execution_providers, custom_registries, ep_only_uses_kernel_registry)) {
      std::string providers;
      for (const auto& ep : execution_providers) {
        providers.append(ep->Type() + " ");
      }
      LOGS_DEFAULT(WARNING) << "registered execution providers " << providers << " were unable to run the model.";
      return;
    }
  }

  std::string provider_type;
  for (auto&& ep : execution_providers) {
    provider_type += ep->Type() + ":";
  }

  provider_type.resize(provider_type.size() - 1);  // remove the trailing ':'

  SCOPED_TRACE(MakeString("registered execution providers: ", provider_type));

  if (custom_registries != nullptr) {
    for (const auto& reg : *custom_registries) {
      ASSERT_PROVIDER_STATUS_OK(session_object.RegisterCustomRegistry(reg));
    }
  }

  for (auto&& entry : execution_providers) {
    ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(entry)));
  }

  ExecuteModel<InferenceSession>(
      model, session_object, expect_result, expected_failure_string,
      run_options, feeds, output_names, provider_type, allow_released_onnx_opset_only);

  // After the model has initialized (happens in ExecuteModel),
  // we should be able to tell how many constant initializers were pre-packed
  // and out of these pre-packed ones how many of them used a "cached" version
  // from the shared container.
  // Populate these value if the user has requested this information.
  if (number_of_pre_packed_weights_counter != nullptr) {
    *number_of_pre_packed_weights_counter = session_object.GetSessionState().GetNumberOfPrepacksCounter();
  }

  if (number_of_shared_pre_packed_weights_counter != nullptr) {
    *number_of_shared_pre_packed_weights_counter =
        session_object.GetSessionState().GetUsedSharedPrePackedWeightCounter();
  }
}

void BaseTester::AddReferenceOutputs(const std::string& model_path, float abs_error,
                                     std::unique_ptr<IExecutionProvider> ep) {
  SessionOptions so;
  so.session_logid = test_name_;
  so.session_log_verbosity_level = 1;
  so.graph_optimization_level = TransformerLevel::Default;

  RunOptions run_options;
  run_options.run_tag = test_name_;
  run_options.run_log_verbosity_level = 1;

  Status status;
  InferenceSession subgraph_session_object{so, GetEnvironment()};
  status = subgraph_session_object.RegisterExecutionProvider(std::move(ep));
  ASSERT_TRUE((status = subgraph_session_object.Load(model_path)).IsOK()) << status;
  ASSERT_TRUE((status = subgraph_session_object.Initialize()).IsOK()) << status;

  // Retrieve output names
  auto model_outputs = subgraph_session_object.GetModelOutputs();
  ASSERT_TRUE(model_outputs.first.IsOK());
  std::vector<std::string> output_names;
  std::transform(model_outputs.second->begin(),
                 model_outputs.second->end(),
                 std::back_inserter(output_names),
                 [](const onnxruntime::NodeArg* node_arg) -> std::string { return node_arg->Name(); });

  NameMLValMap feeds;
  for (size_t i = 0; i < input_data_.size(); ++i) {
    if (input_data_[i].def.Exists()) {
      feeds[input_data_[i].def.Name()] = input_data_[i].data;
    }
  }

  std::vector<OrtValue> subgraph_fetches;
  ASSERT_TRUE((status = subgraph_session_object.Run(run_options, feeds, output_names, &subgraph_fetches)).IsOK()) << status;

  for (size_t out_idx = 0; out_idx < subgraph_fetches.size(); out_idx++) {
    // Retrieve TypeProto
    ASSERT_TRUE(subgraph_fetches[out_idx].Type()->IsTensorType()) << status;
    const Tensor& t = subgraph_fetches[out_idx].Get<Tensor>();
    const TensorTypeBase* tensor_type = DataTypeImpl::TensorTypeFromONNXEnum(t.GetElementType());

    // Construct a temp TypeProto with shape information
    ONNX_NAMESPACE::TypeProto tmp_type_proto(*(tensor_type->GetTypeProto()));
    auto mutable_shape = tmp_type_proto.mutable_tensor_type()->mutable_shape();
    for (auto i : t.Shape().GetDims()) {
      auto* mutable_dim = mutable_shape->add_dim();
      mutable_dim->set_dim_value(i);
    }

    if (abs_error != 0.0f) {
      output_data_.push_back(Data(NodeArg(output_names[out_idx], &tmp_type_proto),
                                  std::move(subgraph_fetches[out_idx]),
                                  optional<float>(), optional<float>(abs_error)));
    } else {
      output_data_.push_back(Data(NodeArg(output_names[out_idx], &tmp_type_proto),
                                  std::move(subgraph_fetches[out_idx]),
                                  optional<float>(), optional<float>()));
    }
  }
}

#ifdef ENABLE_TRAINING
// Deprecated code. Remove this when training::TrainingSession is removed.
template void BaseTester::ExecuteModel<training::TrainingSession>(
    Model& model, training::TrainingSession& session,
    ExpectResult expect_result, const std::string& expected_failure_string,
    const RunOptions* run_options,
    const std::unordered_map<std::string, OrtValue>& feeds,
    const std::vector<std::string>& output_names, const std::string& provider_type,
    bool allow_released_onnx_opset_only);
#endif

}  // namespace test
}  // namespace onnxruntime

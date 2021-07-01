// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include <algorithm>
#include <cfloat>
#include <functional>
#include <iterator>
#include <thread>
#include <fstream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "core/common/denormal.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/profiler.h"
#include "core/framework/compute_capability.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/bfc_arena.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/platform/env.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#ifdef USE_CUDA
#include "core/providers/cuda/gpu_data_transfer.h"
#elif USE_ROCM
#include "core/providers/rocm/gpu_data_transfer.h"
#endif
#include "core/session/environment.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "dummy_provider.h"
#include "test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/providers/provider_test_utils.h"
#include "test/optimizer/dummy_graph_transformer.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;
using namespace onnxruntime::concurrency;

namespace {
struct KernelRegistryAndStatus {
  std::shared_ptr<onnxruntime::KernelRegistry> kernel_registry = std::make_shared<onnxruntime::KernelRegistry>();
  onnxruntime::Status st;
};
}  // namespace
namespace onnxruntime {

#ifdef USE_CUDA
ProviderInfo_CUDA& GetProviderInfo_CUDA();
#endif

class FuseAdd : public OpKernel {
 public:
  explicit FuseAdd(const OpKernelInfo& info) : OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override {
    auto X = context->Input<Tensor>(0);
    auto Y = context->Input<Tensor>(1);
    auto Z = context->Input<Tensor>(2);
    auto& shape = X->Shape();
    auto M = context->Output(0, shape)->template MutableData<float>();
    for (int i = 0; i < shape.Size(); ++i) {
      *(M + i) = *(X->template Data<float>() + i) + *(Y->template Data<float>() + i) + *(Z->template Data<float>() + i);
    }
    return Status::OK();
  }
};
constexpr const char* kFuseTest = "FuseTest";
constexpr const char* kFuseExecutionProvider = "FuseExecutionProvider";
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kFuseExecutionProvider, kFuseTest, 1, FuseAdd);
ONNX_OPERATOR_KERNEL_EX(FuseAdd,
                        kFuseTest,
                        1,
                        kFuseExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        FuseAdd);

Status RegisterOperatorKernels(KernelRegistry& kernel_registry) {
  return kernel_registry.Register(
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kFuseExecutionProvider, kFuseTest, 1, FuseAdd)>());
}

KernelRegistryAndStatus GetFusedKernelRegistry() {
  KernelRegistryAndStatus ret;
  ret.st = RegisterOperatorKernels(*ret.kernel_registry);
  return ret;
}

class FuseExecutionProvider : public IExecutionProvider {
 public:
  explicit FuseExecutionProvider() : IExecutionProvider{kFuseExecutionProvider} {
    AllocatorCreationInfo device_info{
        [](int) {
          return std::make_unique<CPUAllocator>(OrtMemoryInfo("Fuse", OrtAllocatorType::OrtDeviceAllocator));
        }};
    InsertAllocator(device_info.device_alloc_factory(0));
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override {
    // Fuse two add into one.
    std::vector<std::unique_ptr<ComputeCapability>> result;
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    for (auto& node : graph.Nodes()) {
      sub_graph->nodes.push_back(node.Index());
    }
    auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = "FuseAdd";
    meta_def->domain = "FuseTest";
    meta_def->inputs = {"X", "Y", "Z"};
    meta_def->outputs = {"M"};
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    meta_def->type_and_shape_inference_function = [](::onnx::InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      ::onnx::TensorShapeProto intermediary_shape;
      bidirectionalBroadcastShapeInference(
          ctx.getInputType(0)->tensor_type().shape(),
          ctx.getInputType(1)->tensor_type().shape(),
          intermediary_shape);
      bidirectionalBroadcastShapeInference(
          ctx.getInputType(1)->tensor_type().shape(),
          intermediary_shape,
          *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
    };
    sub_graph->SetMetaDef(std::move(meta_def));
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    return result;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    static KernelRegistryAndStatus k = GetFusedKernelRegistry();
    // throw if the registry failed to initialize
    ORT_THROW_IF_ERROR(k.st);
    return k.kernel_registry;
  }
};

namespace test {
static void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                          const std::vector<float>& expected_values);
static constexpr const ORTCHAR_T* MODEL_URI = ORT_TSTR("testdata/mul_1.onnx");
static constexpr const ORTCHAR_T* MODEL_URI_NO_OPSET = ORT_TSTR("testdata/mul_1.noopset.onnx");
//static const std::string MODEL_URI = "./testdata/squeezenet/model.onnx"; // TODO enable this after we've weights?

static void CreateMatMulModel(std::unique_ptr<onnxruntime::Model>& p_model, ProviderType provider_type) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  // Generate the input & output def lists
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  p_model = std::make_unique<Model>("test", true, ModelMetaData(), PathString(),
                                    IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                    model_specific_functions, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = p_model->MainGraph();

  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  std::vector<onnxruntime::NodeArg*> input_defs;
  auto& input_arg_a = graph.GetOrCreateNodeArg("A", &tensor_float);
  input_defs.push_back(&input_arg_a);

  auto& input_arg_b = graph.GetOrCreateNodeArg("B", &tensor_float);
  input_defs.push_back(&input_arg_b);

  std::vector<onnxruntime::NodeArg*> output_defs;
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &tensor_float);
  output_defs.push_back(&output_arg);

  // Create a simple model
  auto& node = graph.AddNode("node1", "MatMul", "MatMul", input_defs, output_defs, nullptr, onnxruntime::kOnnxDomain);
  if (provider_type == kCpuExecutionProvider) {
    node.SetExecutionProviderType(provider_type);
  } else {
#ifdef USE_CUDA
    node.SetExecutionProviderType(provider_type);
#endif
  }
  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

template <typename T = float>
void VerifyOutputs(const Tensor& tensor, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  TensorShape expected_shape(expected_dims);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape), *reinterpret_cast<const std::vector<int64_t>*>(&tensor.Shape()));
  const std::vector<T> found(tensor.template Data<T>(),
                             tensor.template Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<float>& expected_values) {
  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  VerifyOutputs(rtensor, expected_dims, expected_values);
}

void RunModel(InferenceSession& session_object,
              const RunOptions& run_options,
              bool is_preallocate_output_vec = false) {
  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  if (is_preallocate_output_vec) {
    fetches.resize(output_names.size());
    for (auto& elem : fetches) {
      CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                           &elem);
    }
  }

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_y, expected_values_mul_y);
}

void RunModelWithBindingMatMul(InferenceSession& session_object,
                               const RunOptions& run_options,
                               ProviderType bind_provider_type,
                               bool is_preallocate_output_vec,
                               ProviderType allocation_provider,
                               IExecutionProvider* gpu_provider,
                               OrtDevice* output_device) {
  unique_ptr<IOBinding> io_binding;
  Status st = session_object.NewIOBinding(&io_binding);
  ASSERT_TRUE(st.IsOK());
  auto input_allocator = io_binding->GetCPUAllocator(0, bind_provider_type);

  // bind a value to A with input that will produce invalid output in order to test replacement of a feed
  std::vector<float> values_mul_x_tmp = {12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};
  std::vector<int64_t> dims_mul_x_A_tmp = {3, 4};
  OrtValue input_tmp;
  CreateMLValue<float>(input_allocator, dims_mul_x_A_tmp, values_mul_x_tmp, &input_tmp);
  io_binding->BindInput("A", input_tmp);
  const void* tmp_A = io_binding->GetInputs()[0].Get<Tensor>().DataRaw();  // location of data post binding

  // prepare inputs
  std::vector<float> values_mul_x = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};

  /*
      0 1 2 3     0 1 2
      4 5 6 7     3 4 5
      8 9 10 11   6 7 8
      9 10 11
      */
  // bind one input to cpu allocator from bind_provider_type, and another on user provided CPU memory
  // so both code pathes are covered
  OrtValue input_ml_value_A;
  std::vector<int64_t> dims_mul_x_A = {3, 4};
  CreateMLValue<float>(input_allocator, dims_mul_x_A, values_mul_x, &input_ml_value_A);

  OrtValue input_ml_value_B;
  std::vector<int64_t> dims_mul_x_B = {4, 3};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x_B, values_mul_x,
                       &input_ml_value_B);

  io_binding->BindInput("A", input_ml_value_A);
  io_binding->BindInput("B", input_ml_value_B);

  // check location of 'A' post-binding has changed to validate that the previous value was replaced
  ASSERT_TRUE(io_binding->GetInputs()[0].Get<Tensor>().DataRaw() != tmp_A);

  // prepare outputs
  std::vector<int64_t> expected_output_dims = {3, 3};
  OrtValue output_ml_value;
  if (is_preallocate_output_vec) {
    if (allocation_provider == kCpuExecutionProvider) {
      AllocateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), expected_output_dims,
                             &output_ml_value);
    } else if (allocation_provider == kCudaExecutionProvider || allocation_provider == kRocmExecutionProvider) {
      AllocateMLValue<float>(gpu_provider->GetAllocator(0, OrtMemTypeDefault), expected_output_dims, &output_ml_value);
    } else {
      ORT_THROW("Unsupported provider");
    }
  }

  if (output_device) {
    // output should be allocated on specified device (if not preallocated here)
    io_binding->BindOutput("Y", *output_device);
  } else {
    io_binding->BindOutput("Y", output_ml_value);
  }

  ASSERT_TRUE(io_binding->SynchronizeInputs().IsOK());

  // prepare expected inputs and outputs
  std::vector<float> expected_values_mul_y = {42, 48, 54, 114, 136, 158, 186, 224, 262};

  // Now run
  st = session_object.Run(run_options, *io_binding.get());

  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  ASSERT_TRUE(st.IsOK());

  if ((is_preallocate_output_vec && (allocation_provider == kCudaExecutionProvider || allocation_provider == kRocmExecutionProvider)) ||
      (output_device && output_device->Type() == OrtDevice::GPU)) {
#if defined(USE_CUDA) || defined(USE_ROCM)
    // in this case we need to copy the tensor from cuda to cpu
    vector<OrtValue>& outputs = io_binding->GetOutputs();
    ASSERT_EQ(1, outputs.size());
    auto& rtensor = outputs.front().Get<Tensor>();
    auto element_type = rtensor.DataType();
    auto& shape = rtensor.Shape();
    auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
    std::unique_ptr<Tensor> cpu_tensor = std::make_unique<Tensor>(element_type,
                                                                  shape,
                                                                  cpu_allocator);
#ifdef USE_CUDA
    cudaStream_t stream = static_cast<cudaStream_t>(gpu_provider->GetComputeStream());
    st = GetProviderInfo_CUDA().CreateGPUDataTransfer(stream)->CopyTensor(rtensor, *cpu_tensor.get(), 0);
#elif USE_ROCM
    hipStream_t stream = static_cast<hipStream_t>(gpu_provider->GetComputeStream());
    st = GPUDataTransfer(stream).CopyTensor(rtensor, *cpu_tensor.get(), 0);
#endif
    ASSERT_TRUE(st.IsOK());
    OrtValue ml_value;
    ml_value.Init(cpu_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    VerifyOutputs({ml_value}, expected_output_dims, expected_values_mul_y);
#endif
  } else {
    if (allocation_provider == kCudaExecutionProvider || allocation_provider == kRocmExecutionProvider) {
      gpu_provider->Sync();
    }
    VerifyOutputs(io_binding->GetOutputs(), expected_output_dims, expected_values_mul_y);
  }
}

TEST(InferenceSessionTests, NoTimeout) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, GetEnvironment()};
  Status st;
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}

TEST(InferenceSessionTests, OnlyExecutePathToFetches) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.OnlyExecutePathToFetches";

  InferenceSession session_object{so, GetEnvironment()};
  Status st;
  ASSERT_TRUE((st = session_object.Load(MODEL_URI)).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  run_options.only_execute_path_to_fetches = true;
  RunModel(session_object, run_options);
}

TEST(InferenceSessionTests, DisableCPUArena) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.DisableCPUArena";
  so.enable_cpu_mem_arena = false;

  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}

TEST(InferenceSessionTests, TestModelSerialization) {
  // Load model with level 0 transform level
  // and assert that the model has Identity nodes.
  SessionOptions so;
  const string test_model = "testdata/transform/abs-id-max.onnx";
  so.session_logid = "InferenceSessionTests.TestModelSerialization";
  so.graph_optimization_level = TransformerLevel::Default;
  InferenceSessionWrapper session_object_noopt{so, GetEnvironment()};
  ASSERT_TRUE(session_object_noopt.Load(test_model).IsOK());
  ASSERT_TRUE(session_object_noopt.Initialize().IsOK());

  // Assert that model has Identity Nodes.
  const auto& graph_noopt = session_object_noopt.GetGraph();
  std::map<std::string, int> op_to_count_noopt = CountOpsInGraph(graph_noopt);
  ASSERT_TRUE(op_to_count_noopt["Identity"] > 0);

  // Load model with level 1 transform level.
  so.graph_optimization_level = TransformerLevel::Level1;
  so.optimized_model_filepath = ToWideString(test_model + "-TransformLevel-" + std::to_string(static_cast<uint32_t>(so.graph_optimization_level)));
  InferenceSessionWrapper session_object{so, GetEnvironment()};
  ASSERT_TRUE(session_object.Load(test_model).IsOK());
  ASSERT_STATUS_OK(session_object.Initialize());

  // Assert that model has been transformed and identity Node is removed.
  const auto& graph = session_object.GetGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);

  // Serialize model to the same file path again to make sure that rewrite doesn't fail.
  InferenceSession overwrite_session_object{so, GetEnvironment()};
  ASSERT_TRUE(overwrite_session_object.Load(test_model).IsOK());
  ASSERT_TRUE(overwrite_session_object.Initialize().IsOK());

  // Load serialized model with no transform level and serialize model.
  SessionOptions so_opt;
  so_opt.session_logid = "InferenceSessionTests.TestModelSerialization";
  so_opt.graph_optimization_level = TransformerLevel::Default;
  so_opt.optimized_model_filepath = ToWideString(so.optimized_model_filepath) + ToWideString("-TransformLevel-" + std::to_string(static_cast<uint32_t>(so_opt.graph_optimization_level)));
  InferenceSession session_object_opt{so_opt, GetEnvironment()};
  ASSERT_TRUE(session_object_opt.Load(so.optimized_model_filepath).IsOK());
  ASSERT_TRUE(session_object_opt.Initialize().IsOK());

  // Assert that re-feed of optimized model with default transform level results
  // in same runtime model as abs-id-max.onnx with TransformLevel-1.
  std::ifstream model_fs_session1(so.optimized_model_filepath, ios::in | ios::binary);
  ASSERT_TRUE(model_fs_session1.good());
  std::ifstream model_fs_session2(so_opt.optimized_model_filepath, ios::in | ios::binary);
  ASSERT_TRUE(model_fs_session2.good());
  ASSERT_TRUE(model_fs_session1.tellg() == model_fs_session2.tellg());
  model_fs_session1.seekg(0, std::ifstream::beg);
  model_fs_session2.seekg(0, std::ifstream::beg);
  ASSERT_TRUE(std::equal(std::istreambuf_iterator<char>(model_fs_session1.rdbuf()),
                         std::istreambuf_iterator<char>(),
                         std::istreambuf_iterator<char>(model_fs_session2.rdbuf())));

  // Assert that empty optimized model file-path doesn't fail loading.
  so_opt.optimized_model_filepath = ToWideString("");
  InferenceSession session_object_emptyValidation{so_opt, GetEnvironment()};
  ASSERT_TRUE(session_object_emptyValidation.Load(test_model).IsOK());
  ASSERT_TRUE(session_object_emptyValidation.Initialize().IsOK());
}

#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
static bool Compare(const InputDefList& f_arg, const InputDefList& s_arg) {
  if (f_arg.size() != s_arg.size()) {
    cout << "Sizes differ: f_arg size: " << f_arg.size() << " s_arg size: " << s_arg.size() << endl;
    return false;
  }

  for (size_t i = 0; i < f_arg.size(); ++i) {
    const onnxruntime::NodeArg* x = f_arg[i];
    const onnxruntime::NodeArg* y = s_arg[i];
    if ((x->Shape() == nullptr) ^ (y->Shape() == nullptr)) {
      return false;
    }
    if (!x->Shape()) {
      continue;
    }
    auto x_shape = utils::GetTensorShapeFromTensorShapeProto(*x->Shape());
    auto y_shape = utils::GetTensorShapeFromTensorShapeProto(*y->Shape());
    if (x->Name() == y->Name() && x_shape == y_shape && *x->Type() == *y->Type()) {
      continue;
    }
    return false;
  }

  return true;
}

TEST(InferenceSessionTests, ModelMetadata) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.ModelMetadata";
  InferenceSession session_object{so, GetEnvironment()};
  auto model_uri = ORT_TSTR("../models/opset8/test_squeezenet/model.onnx");
  ASSERT_STATUS_OK(session_object.Load(model_uri));

  std::shared_ptr<onnxruntime::Model> p_model;
  ASSERT_STATUS_OK(onnxruntime::Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()));
  const onnxruntime::Graph& graph = p_model->MainGraph();

  // 1. first test the model meta
  {
    auto retval = session_object.GetModelMetadata();
    ASSERT_TRUE(retval.first.IsOK());
    const ModelMetadata* m = retval.second;
    ASSERT_TRUE(m->custom_metadata_map == p_model->MetaData() &&
                m->description == p_model->DocString() &&
                m->domain == p_model->Domain() &&
                m->graph_name == graph.Name() &&
                m->producer_name == p_model->ProducerName() &&
                m->version == p_model->ModelVersion());
  }

  {
    // 2. test inputs
    auto& inputs = graph.GetInputs();
    auto weights = graph.GetAllInitializedTensors();

    // skip the weights
    InputDefList inputs_no_weights;
    for (auto& elem : inputs) {
      if (weights.find(elem->Name()) != weights.end()) {
        continue;
      } else {
        inputs_no_weights.push_back(elem);
      }
    }

    auto retval = session_object.GetModelInputs();
    cout << "weights size: " << weights.size()
         << " inputs.size(): " << inputs.size()
         << " from session: " << retval.second->size() << endl;
    ASSERT_TRUE(retval.first.IsOK());
    ASSERT_TRUE(Compare(inputs_no_weights, *retval.second));
  }

  // 3. test outputs
  {
    auto retval = session_object.GetModelOutputs();
    ASSERT_TRUE(retval.first.IsOK());

    auto& outputs = graph.GetOutputs();
    retval = session_object.GetModelOutputs();
    ASSERT_TRUE(retval.first.IsOK());
    ASSERT_TRUE(Compare(outputs, *retval.second));
  }
}
#endif
TEST(InferenceSessionTests, CheckRunLogger) {
  SessionOptions so;

  so.session_logid = "CheckRunLogger";

  // create CapturingSink. LoggingManager will own it, but as long as the logging_manager
  // is around our pointer stays valid.
  auto capturing_sink = new CapturingSink();

  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(capturing_sink), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  InferenceSession session_object{so, *env.get()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "RunTag";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  RunModel(session_object, run_options);

#ifndef NDEBUG
  // check for some VLOG output to make sure tag was correct. VLOG is not enabled in release build
  auto& msgs = capturing_sink->Messages();
  std::copy(msgs.begin(), msgs.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  bool have_log_entry_with_run_tag =
      (std::find_if(msgs.begin(), msgs.end(),
                    [&run_options](std::string msg) {
                      return msg.find(run_options.run_tag) != string::npos;
                    }) != msgs.end());

  ASSERT_TRUE(have_log_entry_with_run_tag);
#endif
}

TEST(InferenceSessionTests, CheckRunProfilerWithSessionOptions) {
  SessionOptions so;

  so.session_logid = "CheckRunProfiler";
  so.enable_profiling = true;
  so.profile_file_prefix = ORT_TSTR("onnxprofile_profile_test");

  InferenceSession session_object(so, GetEnvironment());
#ifdef USE_CUDA
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
#endif
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "RunTag";

  RunModel(session_object, run_options);
  std::string profile_file = session_object.EndProfiling();

  std::ifstream profile(profile_file);
  ASSERT_TRUE(profile);
  std::string line;
  std::vector<std::string> lines;

  while (std::getline(profile, line)) {
    lines.push_back(line);
  }

  auto size = lines.size();
  ASSERT_TRUE(size > 1);
  ASSERT_TRUE(lines[0].find("[") != string::npos);
  ASSERT_TRUE(lines[1].find("model_loading_uri") != string::npos);
  ASSERT_TRUE(lines[size - 1].find("]") != string::npos);
  std::vector<std::string> tags = {"pid", "dur", "ts", "ph", "X", "name", "args"};

  for (size_t i = 1; i < size - 1; ++i) {
    for (auto& s : tags) {
      ASSERT_TRUE(lines[i].find(s) != string::npos);
    }
  }
}

TEST(InferenceSessionTests, CheckRunProfilerWithStartProfile) {
  SessionOptions so;

  so.session_logid = "CheckRunProfiler";

  InferenceSession session_object(so, GetEnvironment());
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "RunTag";

  session_object.StartProfiling("onnxruntime_profile_custom");
  RunModel(session_object, run_options);
  std::string profile_file = session_object.EndProfiling();

  std::ifstream profile(profile_file);
  std::string line;

  std::vector<std::string> tags = {"pid", "dur", "ts", "ph", "X", "name", "args"};
  int count = 0;
  while (std::getline(profile, line)) {
    if (count == 0) {
      ASSERT_TRUE(line.find("[") != string::npos);
    } else if (count <= 5) {
      for (auto& s : tags) {
        ASSERT_TRUE(line.find(s) != string::npos);
      }
    } else {
      ASSERT_TRUE(line.find("]") != string::npos);
    }

    if (count == 1) {
      ASSERT_TRUE(line.find("mul_1_fence_before") != string::npos);
    }
    count++;
  }
}

TEST(InferenceSessionTests, CheckRunProfilerStartTime) {
  // Test whether the InferenceSession can access the profiler's start time
  SessionOptions so;

  so.session_logid = "CheckRunProfiler";
  so.enable_profiling = true;
  so.profile_file_prefix = ORT_TSTR("onnxprofile_profile_test");

  InferenceSession session_object(so, GetEnvironment());
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  uint64_t before_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::high_resolution_clock::now().time_since_epoch())
                                   .count();  // get current time
  session_object.StartProfiling("onnxruntime_profile_start");
  uint64_t profiling_start_time = session_object.GetProfiling().GetStartTimeNs();
  uint64_t after_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now().time_since_epoch())
                                  .count();

  // the profiler's start time needs to be between before_time and after_time
  ASSERT_TRUE(before_start_time <= profiling_start_time && profiling_start_time <= after_start_time);
}

TEST(InferenceSessionTests, MultipleSessionsNoTimeout) {
  SessionOptions session_options;

  session_options.session_logid = "InferenceSessionTests.MultipleSessionsNoTimeout";
  InferenceSession session_object{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  std::thread thread1{[&session_object]() {
    RunOptions run_options;
    run_options.run_tag = "one session/thread 1";
    RunModel(session_object, run_options);
  }};

  std::thread thread2{[&session_object]() {
    RunOptions run_options;
    run_options.run_tag = "one session/thread 2";
    RunModel(session_object, run_options);
  }};

  thread1.join();
  thread2.join();
}

TEST(InferenceSessionTests, PreAllocateOutputVector) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.PreAllocateOutputVector";

  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "InferenceSessionTests.PreAllocateOutputVector";
  bool is_preallocate_output_vec = true;
  RunModel(session_object, run_options, is_preallocate_output_vec);
}

TEST(InferenceSessionTests, ConfigureVerbosityLevel) {
  SessionOptions so;

  so.session_logid = "ConfigureVerbosityLevel";
  so.session_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  so.session_log_verbosity_level = 1;

  // create CapturingSink. LoggingManager will own it, but as long as the logging_manager
  // is around our pointer stays valid.
  auto capturing_sink = new CapturingSink();

  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(capturing_sink),
      logging::Severity::kVERBOSE,
      false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  InferenceSession session_object{so, *env.get()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "ConfigureVerbosityLevel";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  run_options.run_log_verbosity_level = 1;
  RunModel(session_object, run_options);

#ifndef NDEBUG
  // check for some VLOG output to make sure tag was correct. VLOG is not enabled in release build
  auto& msgs = capturing_sink->Messages();
  std::copy(msgs.begin(), msgs.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  bool have_log_entry_with_vlog_session_msg =
      (std::find_if(msgs.begin(), msgs.end(),
                    [&](std::string msg) { return msg.find("Added input argument with name") != string::npos; }) !=
       msgs.end());

  ASSERT_TRUE(have_log_entry_with_vlog_session_msg);

  bool have_log_entry_with_vlog_run_msg =
      (std::find_if(msgs.begin(), msgs.end(),
                    [&](std::string msg) { return msg.find("Size of execution plan vector") != string::npos; }) !=
       msgs.end());

  ASSERT_TRUE(have_log_entry_with_vlog_run_msg);
#endif
}

TEST(InferenceSessionTests, TestWithIstream) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.TestWithIstream";

  InferenceSession session_object{so, GetEnvironment()};

  std::ifstream model_file_stream(MODEL_URI, ios::in | ios::binary);
  ASSERT_TRUE(model_file_stream.good());
  ASSERT_TRUE(session_object.Load(model_file_stream).IsOK());
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "InferenceSessionTests.TestWithIstream";
  RunModel(session_object, run_options);
}

TEST(InferenceSessionTests, TestRegisterExecutionProvider) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.TestWithIstream";

  InferenceSession session_object{so, GetEnvironment()};
  CPUExecutionProviderInfo epi;
  ASSERT_TRUE(session_object.RegisterExecutionProvider(std::make_unique<CPUExecutionProvider>(epi)).IsOK());

  std::ifstream model_file_stream(MODEL_URI, ios::in | ios::binary);
  ASSERT_TRUE(model_file_stream.good());
  ASSERT_TRUE(session_object.Load(model_file_stream).IsOK());
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "InferenceSessionTests.TestWithIstream";
  RunModel(session_object, run_options);
}

static void TestBindHelper(const std::string& log_str,
                           ProviderType bind_provider_type,
                           ProviderType run_provider_type,
                           bool preallocate_output,
                           ProviderType allocation_provider = kCpuExecutionProvider,
                           OrtDevice* output_device = nullptr) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests." + log_str;
  so.session_log_verbosity_level = 1;  // change to 1 for detailed logging

  InferenceSession session_object{so, GetEnvironment()};
  IExecutionProvider* gpu_provider{};

  if (bind_provider_type == kCudaExecutionProvider || bind_provider_type == kRocmExecutionProvider) {
#ifdef USE_CUDA
    auto provider = DefaultCudaExecutionProvider();
    gpu_provider = provider.get();
    ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(provider)));
#elif USE_ROCM
    ROCMExecutionProviderInfo epi;
    epi.device_id = 0;

    auto provider = std::make_unique<ROCMExecutionProvider>(epi);
    gpu_provider = provider.get();

    ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(provider)));
#endif
  }

  std::unique_ptr<Model> p_model;
  CreateMatMulModel(p_model, run_provider_type);

  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::stringstream sstr(s1);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;

  RunModelWithBindingMatMul(session_object,
                            run_options,
                            bind_provider_type,
                            preallocate_output,
                            allocation_provider,
                            gpu_provider,
                            output_device);
}

TEST(InferenceSessionTests, TestBindCpu) {
  TestBindHelper("TestBindCpu",
                 kCpuExecutionProvider,
                 kCpuExecutionProvider,
                 false /* don't preallocate output */);
}

TEST(InferenceSessionTests, TestIOBindingReuse) {
  SessionOptions so;
  InferenceSession session_object(so, GetEnvironment());
  std::unique_ptr<Model> p_model;
  CreateMatMulModel(p_model, kCpuExecutionProvider);

  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::stringstream sstr(s1);
  ASSERT_TRUE(session_object.Load(sstr).IsOK());
  ASSERT_STATUS_OK(session_object.Initialize());
  unique_ptr<IOBinding> io_binding;
  Status st = session_object.NewIOBinding(&io_binding);
  ASSERT_TRUE(st.IsOK());

  OrtValue ml_value1;
  vector<float> v1{2.f};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, v1, &ml_value1);
  io_binding->BindOutput("foo", ml_value1);
  ASSERT_TRUE(io_binding->GetOutputs().size() == 1);
  auto span = io_binding->GetOutputs()[0].Get<Tensor>().DataAsSpan<float>();
  ASSERT_TRUE(static_cast<size_t>(span.size()) == v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_TRUE(v1[i] == span[i]);
  }

  OrtValue ml_value2;
  vector<float> v2{3.f};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, v2, &ml_value2);
  io_binding->BindOutput("foo", ml_value2);
  ASSERT_TRUE(io_binding->GetOutputs().size() == 1);
  span = io_binding->GetOutputs()[0].Get<Tensor>().DataAsSpan<float>();
  ASSERT_TRUE(static_cast<size_t>(span.size()) == v2.size());
  for (size_t i = 0; i < v2.size(); ++i) {
    ASSERT_TRUE(v2[i] == span[i]);
  }
}

TEST(InferenceSessionTests, InvalidInputTypeOfTensorElement) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.InvalidInputTypeOfTensorElement";

  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  OrtValue ml_value;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                         &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(!st.IsOK());
}

#if defined(USE_CUDA) || defined(USE_ROCM)
#if USE_CUDA
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif USE_ROCM
constexpr const char* kGpuExecutionProvider = kRocmExecutionProvider;
#endif
TEST(InferenceSessionTests, TestBindCuda) {
  TestBindHelper("TestBindCuda",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 false /* don't preallocate output */);
}

TEST(InferenceSessionTests, TestBindCudaPreallocateOutputOnCuda) {
  TestBindHelper("TestBindCudaPreallocateOutputOnCuda",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 true /* preallocate output on GPU */,
                 kGpuExecutionProvider);
}

TEST(InferenceSessionTests, TestBindCudaPreallocateOutputOnCpu) {
  TestBindHelper("TestBindCudaPreallocateOutputOnCpu",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 true /* preallocate output on CPU */,
                 kCpuExecutionProvider);
}

TEST(InferenceSessionTests, TestBindCudaPreallocateOutputOnCpu2) {
  TestBindHelper("TestBindCudaPreallocateOutputOnCpu2",
                 kGpuExecutionProvider,
                 kCpuExecutionProvider,
                 true /* preallocate output on CPU */,
                 kCpuExecutionProvider);
}

TEST(InferenceSessionTests, TestBindCudaSpecifyOutputDeviceOnCuda) {
  OrtDevice device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);

  TestBindHelper("TestBindCudaPreallocateOutputOnCuda",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 false /* preallocate output on GPU */,
                 kGpuExecutionProvider,
                 &device /* specify output device */);
}

#endif

TEST(InferenceSessionTests, ModelWithoutOpset) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.ModelWithoutOpset";

  InferenceSession session_object{so, GetEnvironment()};
  Status retval = session_object.Load(MODEL_URI_NO_OPSET);
  ASSERT_FALSE(retval.IsOK());
  if (!retval.IsOK()) {
    ASSERT_TRUE(retval.ErrorMessage().find("Missing opset in the model") != std::string::npos);
  }
}

static common::Status RunOptionalInputTest(bool add_required_input,
                                           bool add_optional_input,
                                           bool add_invalid_input,
                                           int model_ir_version,
                                           const Environment& sess_env) {
  SessionOptions so;
  so.session_logid = "RunOptionalInputTest";
  InferenceSession session_object{so, sess_env};
  Status status;
  std::string model_path = "testdata/optional_inputs_ir" + std::to_string(model_ir_version) + ".onnx";

  ORT_RETURN_IF_ERROR(session_object.Load(model_path));
  ORT_RETURN_IF_ERROR(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  // prepare inputs
  std::vector<int64_t> dims = {1};
  std::vector<float> required_input_val = {1.f};
  std::vector<float> other_required_input_val = {0.f};
  std::vector<float> optional_input_val = {10.f};  // override initializer value of 1
  std::vector<float> unknown_input_val = {20.f};

  OrtValue required_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, required_input_val, &required_input_mlvalue);

  OrtValue other_required_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, other_required_input_val, &other_required_input_mlvalue);

  OrtValue optional_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, optional_input_val, &optional_input_mlvalue);

  OrtValue unknown_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, unknown_input_val, &unknown_input_mlvalue);

  NameMLValMap feeds;

  if (add_required_input)
    feeds.insert(std::make_pair("required_input", required_input_mlvalue));

  // always add this one
  feeds.insert(std::make_pair("other_required_input", other_required_input_mlvalue));

  if (add_optional_input)
    feeds.insert(std::make_pair("optional_input", optional_input_mlvalue));

  if (add_invalid_input)
    feeds.insert(std::make_pair("unknown_input", unknown_input_mlvalue));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("add_output");
  std::vector<OrtValue> fetches;

  float expected_value = required_input_val[0];
  expected_value += add_optional_input ? optional_input_val[0] : 1.f;

  status = session_object.Run(run_options, feeds, output_names, &fetches);

  if (status.IsOK()) {
    OrtValue& output = fetches.front();
    const auto& tensor = output.Get<Tensor>();
    float output_value = *tensor.Data<float>();
    if (output_value != expected_value) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output of ", output_value, " != ", expected_value);
    }
  }

  return status;
}

// test the change in handling of graph inputs that match initializers between IR version 3 and 4
// in V3 disallow overriding an initializer via the feeds
// for V4 allow it
TEST(InferenceSessionTests, TestOptionalInputs) {
  std::vector<int> ir_versions{3, 4};
  const auto& sess_env = GetEnvironment();
  for (auto version : ir_versions) {
    // required input only
    auto status = RunOptionalInputTest(true, false, false, version, sess_env);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

    // required and optional input
    status = RunOptionalInputTest(true, true, false, version, sess_env);
    if (version == 3) {
      ASSERT_FALSE(status.IsOK()) << status.ErrorMessage();
    } else {
      ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    }
    // required, optional and invalid input
    status = RunOptionalInputTest(true, true, true, version, sess_env);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Invalid Feed Input Name"));

    // missing required
    status = RunOptionalInputTest(false, true, false, version, sess_env);
    ASSERT_FALSE(status.IsOK());
    if (version == 3) {
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Invalid Feed Input Name"));
    } else {
      EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Missing Input:"));
    }
  }
}

TEST(ExecutionProviderTest, FunctionTest) {
  onnxruntime::Model model("graph_1", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "execution_provider_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "ExecutionProviderTest.FunctionTest";
  InferenceSession session_object{so, GetEnvironment()};
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);

  InferenceSession session_object_2{so, GetEnvironment()};
  ASSERT_STATUS_OK(
      session_object_2.RegisterExecutionProvider(std::make_unique<::onnxruntime::FuseExecutionProvider>()));
  status = session_object_2.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object_2.Initialize();
  ASSERT_TRUE(status.IsOK());
  status = session_object_2.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(ExecutionProviderTest, ShapeInferenceForFusedFunctionTest) {
  onnxruntime::Model model("graph_1", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "fused_node_shape_inference_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "ExecutionProviderTest.ShapeInferenceForFusedFunctionTest";
  InferenceSessionWrapper session{so, GetEnvironment()};
  ASSERT_STATUS_OK(
      session.RegisterExecutionProvider(std::make_unique<::onnxruntime::FuseExecutionProvider>()));
  status = session.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session.Initialize();
  ASSERT_TRUE(status.IsOK());

  Graph& fused_graph = session.GetMutableGraph();
  ASSERT_TRUE(fused_graph.NumberOfNodes() == 1);
  auto& fused_node = *fused_graph.Nodes().begin();
  ASSERT_TRUE(fused_node.NodeType() == Node::Type::Fused);
  ASSERT_TRUE(fused_node.Op()->has_type_and_shape_inference_function());

  // Clear shape inference data from output node to verify that assigned inference function is called
  auto& fused_node_output = *fused_node.MutableOutputDefs()[0];
  fused_node_output.ClearShape();
  fused_graph.SetGraphResolveNeeded();
  fused_graph.Resolve();

  ASSERT_TRUE(fused_node_output.Shape() != nullptr);
  ASSERT_TRUE(utils::GetTensorShapeFromTensorShapeProto(*fused_node_output.Shape()) == utils::GetTensorShapeFromTensorShapeProto(float_tensor.tensor_type().shape()));
}

TEST(InferenceSessionTests, Test3LayerNestedSubgraph) {
  // The main graph contains a 'If' node: 'graph_0__if_0'
  // Inside the then-branch of 'graph_0__if_0', there is a nested 'If' node: 'graph_0__if_0__else__if_0'
  // This 3-layer nested graph consumes the same initializer in different sub-graph, used by operators that partitioned in different EP.

  // the then-branch subgraph of main graph's If node 'graph_0__if_0'
  ONNX_NAMESPACE::GraphProto graph_0__if_0__then;
  {
    onnxruntime::Model model("graph_0__if_0__then__graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    {
      ONNX_NAMESPACE::TypeProto float_tensor;
      float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("__graph_0__if_0__then__unknown");

      // implicit input
      auto& data_0 = graph.GetOrCreateNodeArg("data_0", &float_tensor);
      graph.AddOuterScopeNodeArg("data_0");

      // graph output
      auto& graph_if_output = graph.GetOrCreateNodeArg("graph_0__if_0__then__output_0", &float_tensor);

      {
        std::vector<onnxruntime::NodeArg*> inputs = {&data_0};
        std::vector<onnxruntime::NodeArg*> outputs = {&graph_if_output};
        graph.AddNode("graph_0__if_0__then__abs_0", "Abs", "node abs", inputs, outputs);
      }
      auto status = graph.Resolve();
      ASSERT_TRUE(status.IsOK());
      graph_0__if_0__then = graph.ToGraphProto();
      ASSERT_TRUE(status.IsOK());
    }
  }

  // the then-branch (and else-branch, they are the same graph in this test case) subgraph of "graph_0__if_0__else"'s If node 'graph_0__if_0__else__if_0'
  ONNX_NAMESPACE::GraphProto graph_0__if_0__else__if_0__thenelse;
  {
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("__iii_then__unknown");

    onnxruntime::Model model("graph_if_else___then", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    // implicit inputs
    auto& data_0 = graph.GetOrCreateNodeArg("data_0", &float_tensor);
    auto& graph_if_output_else = graph.GetOrCreateNodeArg("graph_if_output_else", &float_tensor);
    graph.AddOuterScopeNodeArg("data_0");
    graph.AddOuterScopeNodeArg("graph_if_output_else");

    // output
    auto& output = graph.GetOrCreateNodeArg("graph_if_else___then_output", &float_tensor);

    // operators
    {
      std::vector<onnxruntime::NodeArg*> inputs = {&graph_if_output_else, &data_0};
      std::vector<onnxruntime::NodeArg*> outputs = {&output};
      graph.AddNode("add_1", "Add", "node add", inputs, outputs);
    }
    auto status = graph.Resolve();
    ASSERT_TRUE(status.IsOK());
    graph_0__if_0__else__if_0__thenelse = graph.ToGraphProto();
    ASSERT_TRUE(status.IsOK());
  }

  // the else-branch subgraph of main graph's If node 'graph_0__if_0'
  ONNX_NAMESPACE::GraphProto graph_0__if_0__else;
  {
    onnxruntime::Model model("graph_if_else", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    {
      ONNX_NAMESPACE::TypeProto float_tensor;
      float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
      float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("__graph_if_else__unknown");
      ONNX_NAMESPACE::TypeProto bool_tensor;
      bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

      // implicit inputs
      auto& graph_if_input = graph.GetOrCreateNodeArg("graph_if_input", &float_tensor);
      auto& if_cond_input = graph.GetOrCreateNodeArg("if_cond_input", &bool_tensor);
      auto& data_0 = graph.GetOrCreateNodeArg("data_0", nullptr);
      graph.AddOuterScopeNodeArg("graph_if_input");
      graph.AddOuterScopeNodeArg("if_cond_input");
      graph.AddOuterScopeNodeArg("data_0");

      // intermediate value nodes
      auto& node_1 = graph.GetOrCreateNodeArg("graph_if_else_node_1", nullptr);
      auto& node_2 = graph.GetOrCreateNodeArg("graph_if_else_node_2", nullptr);
      auto& node_4 = graph.GetOrCreateNodeArg("graph_if_else_node_4", &float_tensor);

      // output nodes
      auto& graph_if_output = graph.GetOrCreateNodeArg("graph_if_output_else", &float_tensor);

      {
        std::vector<onnxruntime::NodeArg*> inputs = {&graph_if_input};
        std::vector<onnxruntime::NodeArg*> outputs = {&node_1};
        graph.AddNode("shape_1", "Shape", "node 1", inputs, outputs);
      }
      {
        std::vector<onnxruntime::NodeArg*> inputs = {&node_1};
        std::vector<onnxruntime::NodeArg*> outputs = {&node_2};
        auto& cast_node = graph.AddNode("cast_1", "Cast", "node 2", inputs, outputs);
        ONNX_NAMESPACE::AttributeProto to;
        to.set_name("to");
        to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
        to.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
        cast_node.AddAttribute("to", to);
      }
      {
        std::vector<onnxruntime::NodeArg*> inputs = {&node_2, &data_0};
        std::vector<onnxruntime::NodeArg*> outputs = {&graph_if_output};
        graph.AddNode("sub_1", "Sub", "node 3", inputs, outputs);
      }
      {
        std::vector<onnxruntime::NodeArg*> inputs = {&if_cond_input};
        std::vector<onnxruntime::NodeArg*> outputs = {&node_4};

        auto& if_node = graph.AddNode("graph_0__if_0__else__if_0", "If", "If node", inputs, outputs);

        if_node.AddAttribute("then_branch", graph_0__if_0__else__if_0__thenelse);
        if_node.AddAttribute("else_branch", graph_0__if_0__else__if_0__thenelse);
      }

      {
        std::vector<const onnxruntime::NodeArg*> outputs = {&node_4};
        graph.SetOutputs(outputs);
      }

      auto status = graph.Resolve();
      ASSERT_TRUE(status.IsOK());
      graph_0__if_0__else = graph.ToGraphProto();
      ASSERT_TRUE(status.IsOK());
    }
  }

  // the main graph 'graph_0'
  onnxruntime::Model model("graph_0", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto& if_cond_input = graph.GetOrCreateNodeArg("if_cond_input", &bool_tensor);
  auto& graph_if_input = graph.GetOrCreateNodeArg("graph_if_input", nullptr);
  auto& if_cond_output = graph.GetOrCreateNodeArg("if_cond_output", &float_tensor);

  {
    std::vector<onnxruntime::NodeArg*> inputs = {&if_cond_input};
    std::vector<onnxruntime::NodeArg*> outputs = {&graph_if_input};
    auto& cast_node = graph.AddNode("cast_9", "Cast", "node 2", inputs, outputs);
    ONNX_NAMESPACE::AttributeProto to;
    to.set_name("to");
    to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    to.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    cast_node.AddAttribute("to", to);
  }

  std::vector<onnxruntime::NodeArg*> inputs = {&if_cond_input};
  std::vector<onnxruntime::NodeArg*> outputs = {&if_cond_output};

  auto& if_node = graph.AddNode("graph_0__if_0", "If", "If node", inputs, outputs);

  if_node.AddAttribute("then_branch", graph_0__if_0__then);
  if_node.AddAttribute("else_branch", graph_0__if_0__else);

  // initializer data_0
  ONNX_NAMESPACE::TensorProto data_0{};
  data_0.set_name("data_0");
  data_0.add_dims(1);
  data_0.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  data_0.add_float_data(0);
  graph.AddInitializedTensor(data_0);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "3-layer-nested-subgraph-test.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK());

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.Test3LayerNestedSubgraph";
  InferenceSession session_object{so, GetEnvironment()};

#ifdef USE_CUDA
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
#elif USE_ROCM
  ROCMExecutionProviderInfo epi;
  epi.device_id = 0;
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<ROCMExecutionProvider>(epi)));
#endif

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  std::vector<int64_t> dim = {1};
  std::vector<bool> va = {false};
  OrtValue ml_value_x;
  CreateMLValue<bool>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dim, va,
                      &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("if_cond_input", ml_value_x));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("if_cond_output");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims = {1};
  std::vector<float> expected_values = {1.0f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims, expected_values);
}

TEST(InferenceSessionTests, Test2LayerNestedSubgraph) {
  // The main graph contains a 'If' node which has a subgraph that consumes implicit inputs

  // the then-branch (and else-branch, they are the same graph in this test case) subgraph of main graph's If node 'graph_0__if_0'
  ONNX_NAMESPACE::GraphProto graph_0__if_0__thenelse;
  {
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("__graph_0__if_0__thenelse__unknown");

    onnxruntime::Model model("graph_0__if_0__thenelse__graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    // implicit inputs
    auto& input_0 = graph.GetOrCreateNodeArg("input_0", &float_tensor);
    auto& graph_0__value_3 = graph.GetOrCreateNodeArg("graph_0__value_3", &float_tensor);
    graph.AddOuterScopeNodeArg("input_0");
    graph.AddOuterScopeNodeArg("graph_0__value_3");

    // output
    auto& output = graph.GetOrCreateNodeArg("graph_0__if_0__thenelse__output", &float_tensor);

    // operators
    {
      std::vector<onnxruntime::NodeArg*> inputs = {&graph_0__value_3, &input_0};
      std::vector<onnxruntime::NodeArg*> outputs = {&output};
      graph.AddNode("graph_0__if_0__thenelse__add_0", "Add", "node add", inputs, outputs);
    }
    auto status = graph.Resolve();
    ASSERT_TRUE(status.IsOK());
    graph_0__if_0__thenelse = graph.ToGraphProto();
    ASSERT_TRUE(status.IsOK());
  }

  // the main graph 'graph_0'
  onnxruntime::Model model("graph_0", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto float_tensor_input;
  float_tensor_input.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("__graph_0__float_unknown");
  ONNX_NAMESPACE::TypeProto bool_tensor;
  bool_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  // graph inputs
  auto& input_0 = graph.GetOrCreateNodeArg("input_0", &float_tensor_input);
  auto& input_1 = graph.GetOrCreateNodeArg("input_1", &bool_tensor);

  // intermediate values
  auto& graph_0__value_1 = graph.GetOrCreateNodeArg("graph_0__value_1", nullptr);
  auto& graph_0__value_2 = graph.GetOrCreateNodeArg("graph_0__value_2", nullptr);
  auto& graph_0__value_3 = graph.GetOrCreateNodeArg("graph_0__value_3", &float_tensor);

  // graph output
  auto& output_0 = graph.GetOrCreateNodeArg("output_0", &float_tensor);

  // operator nodes
  {
    std::vector<onnxruntime::NodeArg*> inputs = {&input_1};
    std::vector<onnxruntime::NodeArg*> outputs = {&graph_0__value_1};
    graph.AddNode("graph_0__shape_0", "Shape", "shape node in main graph", inputs, outputs);
  }
  {
    std::vector<onnxruntime::NodeArg*> inputs = {&graph_0__value_1};
    std::vector<onnxruntime::NodeArg*> outputs = {&graph_0__value_2};
    auto& cast_node = graph.AddNode("graph_0__cast_0", "Cast", "cast node in main graph", inputs, outputs);
    ONNX_NAMESPACE::AttributeProto to;
    to.set_name("to");
    to.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    to.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    cast_node.AddAttribute("to", to);
  }
  {
    std::vector<onnxruntime::NodeArg*> inputs = {&graph_0__value_2, &input_0};
    std::vector<onnxruntime::NodeArg*> outputs = {&graph_0__value_3};
    graph.AddNode("graph_0__sub_0", "Sub", "sub node in main graph", inputs, outputs);
  }
  {
    std::vector<onnxruntime::NodeArg*> inputs = {&input_1};
    std::vector<onnxruntime::NodeArg*> outputs = {&output_0};

    auto& if_node = graph.AddNode("graph_0__if_0", "If", "if node in main graph", inputs, outputs);

    if_node.AddAttribute("then_branch", graph_0__if_0__thenelse);
    if_node.AddAttribute("else_branch", graph_0__if_0__thenelse);
  }

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "2-layer-nested-subgraph-test.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);
  ASSERT_TRUE(status.IsOK());

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.Test2LayerNestedSubgraph";
  InferenceSession session_object{so, GetEnvironment()};

#ifdef USE_CUDA
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
#elif USE_ROCM
  ROCMExecutionProviderInfo epi;
  epi.device_id = 0;
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::make_unique<ROCMExecutionProvider>(epi)));
#endif

  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  std::vector<int64_t> dim_input_0 = {1};
  std::vector<float> data_input_0 = {0.0f};
  OrtValue ml_value_input_0;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dim_input_0, data_input_0,
                       &ml_value_input_0);
  std::vector<int64_t> dim_input_1 = {1};
  std::vector<bool> data_input_1 = {false};
  OrtValue ml_value_input_1;
  CreateMLValue<bool>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dim_input_1, data_input_1,
                      &ml_value_input_1);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("input_0", ml_value_input_0));
  feeds.insert(std::make_pair("input_1", ml_value_input_1));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("output_0");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims = {1};
  std::vector<float> expected_values = {1.0f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims, expected_values);
}

TEST(InferenceSessionTests, TestTruncatedSequence) {
  // model/data generated by <repo>/onnxruntime/test/testdata/CNTK/gen.py GenScan()
  // Manually updated to have IR version of 4.
  static const std::string LSTM_MODEL_URI = "testdata/scan_1.onnx";
  // This model is a 4x forward LSTM. Parse it to find out mapping between init_state input/output
  ONNX_NAMESPACE::ModelProto model_proto;
  int model_fd;
  auto status = Env::Default().FileOpenRd(LSTM_MODEL_URI, model_fd);
  ASSERT_TRUE(status.IsOK());
  google::protobuf::io::FileInputStream f(model_fd);
  f.SetCloseOnDelete(true);
  ASSERT_TRUE(model_proto.ParseFromZeroCopyStream(&f));
  GraphProto& graph_proto = *model_proto.mutable_graph();

  auto find_attr = [&](const NodeProto& node, const std::string& attr_name) -> const AttributeProto* {
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto& attr = node.attribute(i);
      if (attr.name() == attr_name)
        return &attr;
    }
    return nullptr;
  };

  std::unordered_map<std::string, std::string> init_state_map;
  for (int i_node = 0; i_node < graph_proto.node_size(); ++i_node) {
    auto& node = *graph_proto.mutable_node(i_node);
    if (node.op_type() == "Scan") {
      // only works in forward, and do not allow bidirection
      auto attr_directions = find_attr(node, "scan_input_directions");
      if (attr_directions != nullptr) {
        ASSERT_TRUE(attr_directions->ints_size() == 1);

        if (attr_directions->ints(0) == 1)
          continue;  // skip backward Scan
      }

      // input 0 is optional sequence length, 1..N are for initial states
      // and N+1..N+num_scan_inputs are actual inputs
      // output 0..N-1 are for output states, and N.. are actual outputs
      auto attr_num_scan_inputs = find_attr(node, "num_scan_inputs");
      ASSERT_TRUE(attr_num_scan_inputs != nullptr);
      int num_scan_inputs = gsl::narrow_cast<int>(attr_num_scan_inputs->i());
      ASSERT_TRUE(node.input_size() - num_scan_inputs < node.output_size());
      for (int i = 0; i < node.input_size() - num_scan_inputs; ++i) {
        init_state_map.insert(std::make_pair(node.output(i), node.input(i)));
      }
    }
  }

  // now run the truncated model
  SessionOptions so;
  InferenceSession session_object(so, GetEnvironment());
  ASSERT_TRUE(session_object.Load(LSTM_MODEL_URI).IsOK());
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> X_dims = {5, 1, 3};
  std::vector<float> X = {0.5488135f, 0.71518934f, 0.60276335f,
                          0.5448832f, 0.4236548f, 0.6458941f,
                          0.4375872f, 0.891773f, 0.96366274f,
                          0.3834415f, 0.79172504f, 0.5288949f,
                          0.56804454f, 0.92559665f, 0.07103606f};

  std::vector<int64_t> Y_dims = {5, 1, 2};
  std::vector<float> Y_data = {-1.1730184e-04f, -3.1204990e-04f,
                               -2.9978977e-04f, -1.0602647e-03f,
                               -3.8115133e-04f, -2.0684483e-03f,
                               -2.5120965e-04f, -2.9920202e-03f,
                               3.0980256e-05f, -3.5933927e-03f};

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), X_dims, X, &ml_value);

  std::string input_name = "Input13165";
  NameMLValMap feeds = {{input_name, ml_value}};

  // prepare outputs for whole sequence
  std::string final_output_name = "";
  int final_output_index = -1;
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    if (init_state_map.find(graph_proto.output(i).name()) == init_state_map.end()) {
      ASSERT_TRUE(final_output_name.empty());
      final_output_name = graph_proto.output(i).name();
      final_output_index = i;
    }
  }

  std::vector<std::string> output_names = {final_output_name};
  std::vector<OrtValue> fetches;

  // Now run the full sequence
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(Y_dims);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape), *reinterpret_cast<const std::vector<int64_t>*>(&rtensor.Shape()));
  for (size_t i = 0; i < Y_data.size(); ++i)
    EXPECT_NEAR(Y_data[i], rtensor.template Data<float>()[i], FLT_EPSILON);

  // run truncated sequence
  output_names.clear();
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    output_names.push_back(graph_proto.output(i).name());
  }
  fetches.clear();

  std::vector<int> truncated_lengths = {2, 2, 1};              // sums to non-truncated length
  auto seq_stride = TensorShape(X_dims).SizeFromDimension(1);  // sequence is the first dimension of input shape
  int seq_start = 0;
  for (auto truncated_len : truncated_lengths) {
    std::vector<int64_t> truncated_input_dims = X_dims;
    truncated_input_dims[0] = truncated_len;
    OrtValue truncated_ml_value;
    std::vector<float> truncated_input(X.begin() + seq_start * seq_stride, X.begin() + (seq_start + truncated_len) * seq_stride);
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), truncated_input_dims, truncated_input, &truncated_ml_value);
    NameMLValMap truncated_feeds = {{input_name, truncated_ml_value}};
    if (seq_start > 0) {
      // continue from truncated sequence
      ASSERT_TRUE(fetches.size() == output_names.size());
      for (size_t i_output = 0; i_output < output_names.size(); ++i_output) {
        auto iter = init_state_map.find(output_names[i_output]);
        if (iter != init_state_map.end())
          truncated_feeds.insert(std::make_pair(iter->second, fetches[i_output]));
      }
    }
    std::vector<OrtValue> truncated_fetches;
    st = session_object.Run(run_options, truncated_feeds, output_names, &truncated_fetches);
    if (!st.IsOK()) {
      std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
    }
    ASSERT_TRUE(st.IsOK());

    // check truncated output
    auto& truncated_rtensor = truncated_fetches[final_output_index].Get<Tensor>();
    std::vector<int64_t> truncated_output_dims = Y_dims;
    truncated_output_dims[0] = truncated_len;
    TensorShape truncated_shape(truncated_output_dims);
    //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
    ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&truncated_shape), *reinterpret_cast<const std::vector<int64_t>*>(&truncated_rtensor.Shape()));
    auto seq_output_stride = truncated_shape.SizeFromDimension(1);
    for (int i = 0; i < truncated_shape.Size(); ++i)
      EXPECT_NEAR(Y_data[i + seq_start * seq_output_stride], truncated_rtensor.template Data<float>()[i], FLT_EPSILON);

    // prepare for next truncated input
    fetches = truncated_fetches;
    seq_start += truncated_len;
  }
}

// create the feeds and fetches using the dummy allocator so that we have to copy to CPU to execute, and from
// CPU to return in utils::ExecuteGraph. Call InferenceSession::Run twice to test the caching of the copy logic.
TEST(InferenceSessionTests, TestCopyToFromDevices) {
  SessionOptions so;
  so.session_logid = "InferenceSessionTests.TestCopyToFromDevices";
  InferenceSession session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));

  auto dummy_provider = std::make_unique<DummyExecutionProvider>();
  auto* p_dummy_provider = dummy_provider.get();
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(dummy_provider)));

  ASSERT_STATUS_OK(session_object.Initialize());

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value;
  CreateMLValue<float>(p_dummy_provider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value);

  std::vector<std::string> feed_names;
  std::vector<OrtValue> feeds;
  feed_names.push_back("X");
  feeds.push_back(ml_value);

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  auto run_test = [&](int run_num) {
    // prepare outputs
    std::vector<std::string> output_names;
    std::vector<OrtValue> fetches;
    output_names.push_back("Y");

    fetches.resize(output_names.size());
    for (auto& elem : fetches) {
      CreateMLValue<float>(p_dummy_provider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                           &elem);
    }

    // Now run
    RunOptions run_options;
    run_options.run_tag = "run:" + std::to_string(run_num);

    common::Status st = session_object.Run(run_options, feed_names, feeds, output_names, &fetches, nullptr);
    ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

    VerifyOutputs(fetches, expected_dims_mul_y, expected_values_mul_y);
  };

  int run_number = 0;
  run_test(run_number++);
  run_test(run_number++);
}

// This test validates the RegisterTransformer API
// It creates and registers a dummy transformer and after session initialize
// validates that this transformer was called regardless of the graph optimization level set.
TEST(InferenceSessionTests, TestRegisterTransformers) {
  string model_uri = "testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";

  for (int i = static_cast<int>(TransformerLevel::Default); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    SessionOptions so;
    so.session_logid = "InferenceSessionTests.TestL1AndL2Transformers";
    so.graph_optimization_level = static_cast<TransformerLevel>(i);
    InferenceSession session_object{so, GetEnvironment()};

    // Create and register dummy graph transformer
    auto dummy_transformer_unique_ptr = std::make_unique<DummyGraphTransformer>("DummyTransformer");
    const auto* dummy_transformer = dummy_transformer_unique_ptr.get();
    ASSERT_STATUS_OK(session_object.RegisterGraphTransformer(std::move(dummy_transformer_unique_ptr)));

    ASSERT_STATUS_OK(session_object.Load(model_uri));
    ASSERT_STATUS_OK(session_object.Initialize());

    // Validate transformer was called after Session.Initialize
    ASSERT_TRUE(dummy_transformer->IsTransformerInvoked());
  }
}

// This test validates session initialize is successful when all the pre-defined
// L1 and L2 transformers are enabled.
TEST(InferenceSessionTests, TestL1AndL2Transformers) {
  // Models which cover all transformers.
  std::vector<std::string> test_model_uris = {"testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx",
                                              "testdata/transform/abs-id-max.onnx",
                                              "testdata/transform/slice-v11-elim.onnx",
                                              "testdata/transform/matmul_add_fusion/2Input/model.onnx",
                                              "testdata/transform/matmul_add_fusion/3Input/gemm_relu.onnx",
                                              "testdata/transform/fusion/fuse-conv-bn-add-mul-float16.onnx"};

  for (const auto& model_uri : test_model_uris) {
    SessionOptions so;
    so.session_logid = "InferenceSessionTests.TestL1AndL2Transformers";
    so.graph_optimization_level = TransformerLevel::Level2;
    InferenceSession session_object{so, GetEnvironment()};
    ASSERT_STATUS_OK(session_object.Load(model_uri));
    ASSERT_STATUS_OK(session_object.Initialize());
  }
}

// fallback to lenient merging of shape info if model opset is not the latest
TEST(InferenceSessionTests, TestLenientShapeInferencing) {
  // latest opset should fail
  std::vector<int64_t> input_shape{2, 2};
  std::vector<float> input_data{0.f, 1.f, 2.f, 3.f};
  std::vector<int64_t> invalid_output_shape{1, 2};  // valid shape is {2} as output data is input_shape
  std::vector<int64_t> output_data{2, 2};

  OpTester latest_opset("Shape", -1);  // use latest opset for shape inference errors
  latest_opset.AddInput("data", input_shape, input_data);
  latest_opset.AddOutput<int64_t>("output", invalid_output_shape, output_data);
  latest_opset.Run(OpTester::ExpectResult::kExpectFailure,
                   "Mismatch between number of source and target dimensions. Source=1 Target=2");

  // older opset should allow the mismatch with a warning.
  // we also need for the output to be valid so OpTester doesn't throw so add an Unsqueeze after the Shape.
  // This should result in a warning log message but successful run.
  class OpTesterWithReshape : public OpTester {
   public:
    OpTesterWithReshape() : OpTester("Shape", 7) {
    }

   protected:
    void AddNodes(onnxruntime::Graph& graph,
                  std::vector<onnxruntime::NodeArg*>& graph_input_defs,
                  std::vector<onnxruntime::NodeArg*>& graph_output_defs,
                  std::vector<std::function<void(onnxruntime::Node& node)>>& add_attribute_funcs) override {
      // we need to create an intermediate output with a different name
      auto tmp_output_defs = graph_output_defs;
      auto type_info = *tmp_output_defs[0]->TypeAsProto();  // copy
      auto& shape_output = graph.GetOrCreateNodeArg("shape_output", &type_info);
      tmp_output_defs[0] = &shape_output;

      // call base implementation to add the Shape node with invalid output shape
      OpTester::AddNodes(graph, graph_input_defs, tmp_output_defs, add_attribute_funcs);

      // add Unsqueeze node to fix the output shape

      auto& unsqueeze = graph.AddNode("unsqueeze", "Unsqueeze", "Fix output shape", tmp_output_defs, graph_output_defs);
      unsqueeze.AddAttribute("axes", std::vector<int64_t>{0});
    }
  };

  OpTesterWithReshape old_opset;

  old_opset.AddInput("data", input_shape, input_data);
  old_opset.AddOutput<int64_t>("output", invalid_output_shape, output_data);
  // TensorRT doesn't handle Unsqueeze
  // OpenVINO: Disabled temporarily
  old_opset.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

#ifdef USE_CUDA

TEST(InferenceSessionTests, TestParallelExecutionWithCudaProvider) {
  string model_uri = "testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";

  SessionOptions so;
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  so.session_logid = "InferenceSessionTests.TestParallelExecutionWithCudaProvider";
  InferenceSession session_object{so, GetEnvironment()};

  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));

  ASSERT_STATUS_OK(session_object.Load(model_uri));

  auto status = session_object.Initialize();

  ASSERT_TRUE(status.IsOK());

  const auto& so_queried = session_object.GetSessionOptions();

  // execution mode is sequential since we have registered the CUDA EP
  // (which isn't supported by the parallel execution mode)
  ASSERT_TRUE(so_queried.execution_mode == ExecutionMode::ORT_SEQUENTIAL);
}

TEST(InferenceSessionTests, TestArenaShrinkageAfterRun) {
  OrtArenaCfg arena_cfg;
  arena_cfg.arena_extend_strategy = 1;  // kSameAsRequested

  SessionOptions so;
  InferenceSession session_object{so, GetEnvironment()};
  OrtCUDAProviderOptions provider_options{};
  provider_options.default_memory_arena_cfg = &arena_cfg;
  provider_options.device_id = 0;
  auto factory = CreateExecutionProviderFactory_Cuda(&provider_options);

  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(factory->CreateProvider()));
  ASSERT_STATUS_OK(session_object.Initialize());

  // Fetch the CUDA allocator to analyze its stats
  OrtMemoryInfo mem_info(CUDA, OrtArenaAllocator);
  auto cuda_alloc = session_object.GetAllocator(mem_info);

  AllocatorStats alloc_stats;
  static_cast<BFCArena*>(cuda_alloc.get())->GetStats(&alloc_stats);
#ifdef ENABLE_TRAINING
  // In training builds, initializers are allocated using the Reserve() call which
  // will not cause an arena extension
  ASSERT_EQ(alloc_stats.num_arena_extensions, 0);
#else
  // The arena would have made an extension to accommodate the sole initializer on CUDA
  ASSERT_EQ(alloc_stats.num_arena_extensions, 1);
#endif

  // no shrinkages should have occurred during this time (sanity check)
  ASSERT_EQ(alloc_stats.num_arena_shrinkages, 0);

  auto allocated_memory_before_run = alloc_stats.total_allocated_bytes;

  {
    // First Run - no shrinkage
    RunOptions run_options_1;
    RunModel(session_object, run_options_1);

    static_cast<BFCArena*>(cuda_alloc.get())->GetStats(&alloc_stats);

    // The arena would have made 2 more extensions as part of servicing memory requests within Run()
    // 1) - To take the solitary feed to cuda memory
    // 2) - Allocate output of the solitary node
#ifdef ENABLE_TRAINING
    // In training - that is a total of 2 extensions
    ASSERT_EQ(alloc_stats.num_arena_extensions, 2);
#else
    // In inferencing - that is a total of 3 extensions
    ASSERT_EQ(alloc_stats.num_arena_extensions, 3);
#endif

    // Assert that there have been no shrinkages after this Run()
    ASSERT_EQ(alloc_stats.num_arena_shrinkages, 0);
  }

  {
    // Second Run - with shrinkage
    RunOptions run_options_2;
    run_options_2.config_options.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");
    RunModel(session_object, run_options_2);

    static_cast<BFCArena*>(cuda_alloc.get())->GetStats(&alloc_stats);

    // The arena would have made no extensions in this Run() as the freed memory after the first Run()
    // will be re-used

#ifdef ENABLE_TRAINING
    // In training - that is a total of 2 extensions
    ASSERT_EQ(alloc_stats.num_arena_extensions, 2);
#else
    // In inferencing - that is a total of 3 extensions
    ASSERT_EQ(alloc_stats.num_arena_extensions, 3);
#endif

    // The arena would have shrunk both extensions it made as part of Run() - because these allocations
    // would have been left unused after this Run()
    // (The allocation for the sole initializer will not be shrunk as it is still being "used" by the session)
    ASSERT_EQ(alloc_stats.num_arena_shrinkages, 2);
  }

  // Assert that allocated memory before and after Run() are the same
  // Because any memory allocated during Run would have been de-allocated as pat of the shrinkage
  auto allocated_memory_after_run = alloc_stats.total_allocated_bytes;
  ASSERT_EQ(allocated_memory_before_run, allocated_memory_after_run);
}

#endif

// The model being tested here triggers a case where the allocation planner (AP) tries to reuse a tensor of type
// double for a string tensor. The reuse logic of AP works correctly on Windows and Ubuntu 16.x
// since there the sizeof(double) != sizeof(std::string). However, on CentOS (gcc 4.8.x), the 2 sizes are equal.
TEST(InferenceSessionTests, ModelThatTriggersAllocationPlannerToReuseDoubleTensorForStringTensor) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.ModelThatTriggersAllocationPlannerBug";

  InferenceSession session_object{so, GetEnvironment()};
  Status st;
  ASSERT_TRUE((st = session_object.Load("testdata/test_cast_back_to_back_non_const_mixed_types_origin.onnx")).IsOK())
      << st.ErrorMessage();
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  // prepare inputs
  std::vector<int64_t> dims_x = {1, 2, 3};
  std::vector<float> values_x = {1.6f, -0.6f, -0.5f, -1.0f, 0.8f, -2.3f};
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x,
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("u", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("res");
  output_names.push_back("res2");
  output_names.push_back("res3");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_res = {1, 2, 3};
  std::vector<int64_t> expected_values_res = {1, 0, 0, -1, 0, -2};

  std::vector<int64_t> expected_dims_res2 = {1, 2, 3};
  std::vector<int64_t> expected_values_res2 = {1, 0, 0, -1, 0, -2};

  std::vector<int64_t> expected_dims_res3 = {1, 2, 3};
  std::vector<int8_t> expected_values_res3 = {1, 0, 0, 1, 0, 1};

  // Now run
  st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  ASSERT_EQ(3u, fetches.size());
  VerifyOutputs(fetches[0].Get<Tensor>(), expected_dims_res, expected_values_res);
  VerifyOutputs(fetches[1].Get<Tensor>(), expected_dims_res2, expected_values_res2);
  VerifyOutputs(fetches[2].Get<Tensor>(), expected_dims_res3, expected_values_res3);
}

// The following test is to cover the feature of InferenceSession that allows some session options
// to flow in from a model file, and use defaults for missing session options/session options not supported for parsing
// from the model
static char ort_load_config_from_model_env_var_enabled[] = "ORT_LOAD_CONFIG_FROM_MODEL=1";
static char ort_load_config_from_model_env_var_disabled[] = "ORT_LOAD_CONFIG_FROM_MODEL=0";

TEST(InferenceSessionTests, LoadModelWithValidOrtConfigJson) {
  // Part 1 - Load config from model feature enabled
#ifdef _WIN32
  (void)_putenv(ort_load_config_from_model_env_var_enabled);
#else
  putenv(ort_load_config_from_model_env_var_enabled);
#endif

  SessionOptions so;
  std::string model_path = "testdata/model_with_valid_ort_config_json.onnx";

  // Create session
  InferenceSession session_object_1{so, GetEnvironment(), model_path};

  // Load() and Initialize() the session
  Status st;
  ASSERT_TRUE((st = session_object_1.Load()).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object_1.Initialize()).IsOK()) << st.ErrorMessage();

  // The default value for inter_op_param.thread_pool_size is 0
  // The model requests for inter_op_param.thread_pool_size to be 5
  ASSERT_TRUE(session_object_1.GetSessionOptions().inter_op_param.thread_pool_size == 5);

  // The default value for intra_op_param.thread_pool_size is 0
  // The model requests for intra_op_param.thread_pool_size to be 2
  ASSERT_TRUE(session_object_1.GetSessionOptions().intra_op_param.thread_pool_size == 2);

  // The default value for execution_mode is ORT_SEQUENTIAL
  // The model's config doesn't explicitly request a mode in the ORT config Json - hence the default should be used
  ASSERT_TRUE(session_object_1.GetSessionOptions().execution_mode == ExecutionMode::ORT_SEQUENTIAL);

  // The default value for graph_optimization_level is Level1
  // The model requests Level3 - hence that should be used
  ASSERT_TRUE(session_object_1.GetSessionOptions().graph_optimization_level == TransformerLevel::Level3);

  // The default value for enable_profiling is false
  // The model requests true - hence that should be used
  ASSERT_TRUE(session_object_1.GetSessionOptions().enable_profiling);

  // Part 2 - Load config from model feature disabled
#ifdef _WIN32
  (void)_putenv(ort_load_config_from_model_env_var_disabled);
#else
  putenv(ort_load_config_from_model_env_var_disabled);
#endif

  // Change from default value for one option
  so.intra_op_param.thread_pool_size = 2;

  // Create session
  InferenceSession session_object_2{so, GetEnvironment(), model_path};

  // Load() and Initialize() the session
  ASSERT_TRUE((st = session_object_2.Load()).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object_2.Initialize()).IsOK()) << st.ErrorMessage();

  // The default value for enable_profiling is false
  // Even though the model requests enable_profiling to be true in the ORT config Json,
  // the default value should be used as the feature is disabled
  ASSERT_FALSE(session_object_2.GetSessionOptions().enable_profiling);

  // In the session options object fed in at session creation,
  // the request was for intra_op_param.thread_pool_size to be 2 - that should be honored
  ASSERT_TRUE(session_object_2.GetSessionOptions().intra_op_param.thread_pool_size == 2);
}

TEST(InferenceSessionTests, LoadModelWithInValidOrtConfigJson) {
  // Part 1 - Load config from model feature enabled
#ifdef _WIN32
  (void)_putenv(ort_load_config_from_model_env_var_enabled);
#else
  putenv(ort_load_config_from_model_env_var_enabled);
#endif

  SessionOptions so;
  std::string model_path = "testdata/model_with_invalid_ort_config_json.onnx";

  // Create session (should throw as the json within the model is invalid/improperly formed)
  ORT_TRY {
    InferenceSession session_object_1{so, GetEnvironment(), model_path};
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&e]() {
      std::string e_message(std::string(e.what()));
      ASSERT_TRUE(e_message.find("Could not finalize session options while constructing the inference session. Error Message:") != std::string::npos);
      ASSERT_TRUE(e_message.find("Json stored in the `ort_config` key cannot be parsed.") != std::string::npos);
    });
  }

  // Part 2 - Load config from model feature disabled
  // The invalid/improperly formed config json in the model should not come into the picture here
#ifdef _WIN32
  ORT_IGNORE_RETURN_VALUE(_putenv(ort_load_config_from_model_env_var_disabled));
#else
  putenv(ort_load_config_from_model_env_var_disabled);
#endif

  // Change from default value for one option
  so.intra_op_param.thread_pool_size = 2;

  // Create session
  InferenceSession session_object_2{so, GetEnvironment(), model_path};

  // Load() and Initialize() the session
  Status st;
  ASSERT_TRUE((st = session_object_2.Load()).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object_2.Initialize()).IsOK()) << st.ErrorMessage();

  // Default value for execution_mode
  ASSERT_TRUE(session_object_2.GetSessionOptions().execution_mode == ExecutionMode::ORT_SEQUENTIAL);

  // In the session options object fed in at session creation,
  // the request was for intra_op_param.thread_pool_size to be 2 - that should be honored
  ASSERT_TRUE(session_object_2.GetSessionOptions().intra_op_param.thread_pool_size == 2);
}

TEST(InferenceSessionTests, LoadModelWithNoOrtConfigJson) {
  // Part 1 - Load config from model feature enabled
#ifdef _WIN32
  (void)_putenv(ort_load_config_from_model_env_var_enabled);
#else
  putenv(ort_load_config_from_model_env_var_enabled);
#endif

  SessionOptions so;
  // Change from default value for one option
  so.intra_op_param.thread_pool_size = 2;

  std::string model_path = "testdata/transform/abs-id-max.onnx";

  // Create session
  InferenceSession session_object_1{so, GetEnvironment(), model_path};

  // Load() and Initialize() the session
  Status st;
  ASSERT_TRUE((st = session_object_1.Load()).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object_1.Initialize()).IsOK()) << st.ErrorMessage();

  // The custom session options instance requested intra_op_param.thread_pool_size == 2,
  // but since the session tried to look into the model for the config, and didn't find any
  // the defaults would be used for session creation
  ASSERT_TRUE(session_object_1.GetSessionOptions().intra_op_param.thread_pool_size == 0);

  // Part 2 - Load config from model feature disabled
  // The missing config json should not come into the picture
#ifdef _WIN32
  (void)_putenv(ort_load_config_from_model_env_var_disabled);
#else
  putenv(ort_load_config_from_model_env_var_disabled);
#endif

  // Create session
  InferenceSession session_object_2{so, GetEnvironment(), model_path};  // so has inter_op_param.thread_pool_size set to 2

  // Load() and Initialize() the session
  ASSERT_TRUE((st = session_object_2.Load()).IsOK()) << st.ErrorMessage();
  ASSERT_TRUE((st = session_object_2.Initialize()).IsOK()) << st.ErrorMessage();

  // In the session options object fed in at session creation,
  // the request was for intra_op_param.thread_pool_size to be 2 - that should be honored
  ASSERT_TRUE(session_object_2.GetSessionOptions().intra_op_param.thread_pool_size == 2);
}

TEST(InferenceSessionTests, LoadModelWithEnvVarSetToUnsupportedVal) {
  // "10" is unsupported for ORT_LOAD_CONFIG_FROM_MODEL
  char env_var_value_set_to_unsupported_val[] = "ORT_LOAD_CONFIG_FROM_MODEL=10";
#ifdef _WIN32
  (void)_putenv(env_var_value_set_to_unsupported_val);
#else
  putenv(env_var_value_set_to_unsupported_val);
#endif
  SessionOptions so;
  std::string model_path = "testdata/model_with_valid_ort_config_json.onnx";

  // Create session (should throw because of the unsupported value for the env var - ORT_LOAD_CONFIG_FROM_MODEL)
  ORT_TRY {
    InferenceSession session_object_1{so, GetEnvironment(), model_path};
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&e]() {
      std::string e_message(std::string(e.what()));
      ASSERT_TRUE(e_message.find("Could not finalize session options while constructing the inference session. Error Message:") != std::string::npos);
      ASSERT_TRUE(e_message.find("The only supported values for the environment variable ") != std::string::npos);
      ASSERT_TRUE(e_message.find("The environment variable contained the value: 10") != std::string::npos);
    });
  }

  // Disable the feature before exiting the test as this process is likely to be used for running other tests
#ifdef _WIN32
  (void)
      _putenv(ort_load_config_from_model_env_var_disabled);
#else
  putenv(ort_load_config_from_model_env_var_disabled);
#endif
}

// Global threadpool related tests
// We test for 4 combinations
class InferenceSessionTestGlobalThreadPools : public InferenceSessionWrapper {
 public:
  InferenceSessionTestGlobalThreadPools(const SessionOptions& session_options,
                                        const Environment& env)
      : InferenceSessionWrapper(session_options, env) {
  }

  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPoolToUse() const {
    return InferenceSession::GetIntraOpThreadPoolToUse();
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPoolToUse() const {
    return InferenceSession::GetInterOpThreadPoolToUse();
  }
};

// Test 1: env created WITHOUT global tp / use per session tp (default case): in this case per session tps should be in use
TEST(InferenceSessionTests, CheckIfPerSessionThreadPoolsAreBeingUsed) {
  SessionOptions so;
  so.use_per_session_threads = true;

  so.session_logid = "CheckIfPerSessionThreadPoolsAreBeingUsed";
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  ASSERT_TRUE(st.IsOK());

  InferenceSessionTestGlobalThreadPools session_object{so, *env.get()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  // make sure we're using the per session threadpools
  auto intra_tp_from_session = session_object.GetIntraOpThreadPoolToUse();
  auto intra_tp_from_session_state = session_object.GetSessionState().GetThreadPool();
  auto inter_tp_from_session = session_object.GetInterOpThreadPoolToUse();
  auto inter_tp_from_session_state = session_object.GetSessionState().GetInterOpThreadPool();
  auto intra_tp_from_env = env->GetIntraOpThreadPool();
  auto inter_tp_from_env = env->GetInterOpThreadPool();

  // ensure threadpools were set correctly in the session state
  ASSERT_TRUE(intra_tp_from_session == intra_tp_from_session_state);
  ASSERT_TRUE(inter_tp_from_session == inter_tp_from_session_state);

  ASSERT_TRUE(intra_tp_from_env == nullptr);
  ASSERT_TRUE(inter_tp_from_env == nullptr);

  RunOptions run_options;
  run_options.run_tag = "RunTag";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  RunModel(session_object, run_options);
}

// Test 2: env created with global tp / DONT use per session tp: in this case global tps should be in use
TEST(InferenceSessionTests, CheckIfGlobalThreadPoolsAreBeingUsed) {
  SessionOptions so;
  so.use_per_session_threads = false;

  so.session_logid = "CheckIfGlobalThreadPoolsAreBeingUsed";
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  OrtThreadingOptions tp_options;
  auto st = Environment::Create(std::move(logging_manager), env, &tp_options, true /*create_global_thread_pools*/);
  ASSERT_TRUE(st.IsOK());

  InferenceSessionTestGlobalThreadPools session_object{so, *env.get()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  // make sure we're using the global threadpools in both session and session state
  auto intra_tp_from_session = session_object.GetIntraOpThreadPoolToUse();
  auto intra_tp_from_session_state = session_object.GetSessionState().GetThreadPool();
  auto inter_tp_from_session = session_object.GetInterOpThreadPoolToUse();
  auto inter_tp_from_session_state = session_object.GetSessionState().GetInterOpThreadPool();
  auto intra_tp_from_env = env->GetIntraOpThreadPool();
  auto inter_tp_from_env = env->GetInterOpThreadPool();

  ASSERT_TRUE(intra_tp_from_session == intra_tp_from_env);
  ASSERT_TRUE(inter_tp_from_session == inter_tp_from_env);
  ASSERT_TRUE(intra_tp_from_session_state == intra_tp_from_env);
  ASSERT_TRUE(inter_tp_from_session_state == inter_tp_from_env);

  RunOptions run_options;
  run_options.run_tag = "RunTag";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  RunModel(session_object, run_options);
}

// Test 3: env created with global tp / use per session tp: in this case per session tps should be in use
TEST(InferenceSessionTests, CheckIfPerSessionThreadPoolsAreBeingUsed2) {
  SessionOptions so;
  so.use_per_session_threads = true;

  so.session_logid = "CheckIfPerSessionThreadPoolsAreBeingUsed2";
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  OrtThreadingOptions tp_options;
  auto st = Environment::Create(std::move(logging_manager), env, &tp_options, true /*create_global_thread_pools*/);
  ASSERT_TRUE(st.IsOK());

  InferenceSessionTestGlobalThreadPools session_object{so, *env.get()};
  ASSERT_STATUS_OK(session_object.Load(MODEL_URI));
  ASSERT_STATUS_OK(session_object.Initialize());

  // make sure we're using the per session threadpools
  auto intra_tp_from_session = session_object.GetIntraOpThreadPoolToUse();
  auto intra_tp_from_session_state = session_object.GetSessionState().GetThreadPool();
  auto inter_tp_from_session = session_object.GetInterOpThreadPoolToUse();
  auto inter_tp_from_session_state = session_object.GetSessionState().GetInterOpThreadPool();
  auto intra_tp_from_env = env->GetIntraOpThreadPool();
  auto inter_tp_from_env = env->GetInterOpThreadPool();

  // ensure threadpools were set correctly in the session state
  ASSERT_TRUE(intra_tp_from_session == intra_tp_from_session_state);
  ASSERT_TRUE(inter_tp_from_session == inter_tp_from_session_state);

  // ensure per session thread pools in use are different from the
  // env threadpools
  if (intra_tp_from_session && intra_tp_from_env) {  // both tps could be null on 1 core machines
    ASSERT_FALSE(intra_tp_from_session == intra_tp_from_env);
  }

  if (inter_tp_from_session && inter_tp_from_env) {  // both tps could be null on 1 core machines
    ASSERT_FALSE(inter_tp_from_session == inter_tp_from_env);
  }

  RunOptions run_options;
  run_options.run_tag = "RunTag";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  RunModel(session_object, run_options);
}

// Test 4: env created WITHOUT global tp / DONT use per session tp --> this should throw an exception
TEST(InferenceSessionTests, InvalidSessionEnvCombination) {
  SessionOptions so;
  so.use_per_session_threads = false;

  so.session_logid = "InvalidSessionEnvCombination";
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  ASSERT_TRUE(st.IsOK());

  ORT_TRY {
    InferenceSessionTestGlobalThreadPools session_object{so, *env.get()};
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&e]() {
      std::string e_message(std::string(e.what()));
      ASSERT_TRUE(e_message.find(
                      "When the session is not configured to use per session"
                      " threadpools, the env must be created with the the CreateEnvWithGlobalThreadPools API") !=
                  std::string::npos);
    });
  }
}

// Tests for sharing allocators between sessions
class InferenceSessionTestSharingAllocator : public InferenceSessionWrapper {
 public:
  InferenceSessionTestSharingAllocator(const SessionOptions& session_options,
                                       const Environment& env)
      : InferenceSessionWrapper(session_options, env) {
  }
};

// Ensure sessions use the same allocator. It uses ORT created allocator.
TEST(InferenceSessionTests, AllocatorSharing_EnsureSessionsUseSameOrtCreatedAllocator) {
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  ASSERT_TRUE(st.IsOK());
  // create allocator to register with the env
  bool use_arena = true;
#if !(defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64))
  use_arena = false;
#endif
  OrtMemoryInfo mem_info{onnxruntime::CPU, use_arena ? OrtArenaAllocator : OrtDeviceAllocator};
  AllocatorCreationInfo device_info{
      [mem_info](int) { return std::make_unique<TAllocator>(mem_info); },
      0, use_arena};

  AllocatorPtr allocator_ptr = CreateAllocator(device_info);
  st = env->RegisterAllocator(allocator_ptr);
  ASSERT_STATUS_OK(st);
  // create sessions to share the allocator

  SessionOptions so1;
  so1.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
  InferenceSessionTestSharingAllocator sess1(so1, *env);
  ASSERT_STATUS_OK(sess1.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess1.Initialize());

  SessionOptions so2;
  so2.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
  InferenceSessionTestSharingAllocator sess2(so2, *env);
  ASSERT_STATUS_OK(sess2.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess2.Initialize());

  // This line ensures the allocator in the session is the same as that in the env
  ASSERT_EQ(sess1.GetSessionState().GetAllocator(mem_info).get(),
            allocator_ptr.get());

  // This line ensures the underlying IAllocator* is the same across 2 sessions.
  ASSERT_EQ(sess1.GetSessionState().GetAllocator(mem_info).get(),
            sess2.GetSessionState().GetAllocator(mem_info).get());
}

// Ensure sessions don't use the same allocator. It uses ORT created allocator.
TEST(InferenceSessionTests, AllocatorSharing_EnsureSessionsDontUseSameOrtCreatedAllocator) {
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  ASSERT_TRUE(st.IsOK());
  // create allocator to register with the env
  bool use_arena = true;
#if !(defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64))
  use_arena = false;
#endif
  OrtMemoryInfo mem_info{onnxruntime::CPU, use_arena ? OrtArenaAllocator : OrtDeviceAllocator};
  AllocatorCreationInfo device_info{
      [mem_info](int) { return std::make_unique<TAllocator>(mem_info); },
      0, use_arena};

  AllocatorPtr allocator_ptr = CreateAllocator(device_info);
  st = env->RegisterAllocator(allocator_ptr);
  ASSERT_STATUS_OK(st);
  // create sessions to share the allocator

  SessionOptions so1;
  so1.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
  InferenceSessionTestSharingAllocator sess1(so1, *env);
  ASSERT_STATUS_OK(sess1.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess1.Initialize());

  SessionOptions so2;
  so2.config_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "0");
  InferenceSessionTestSharingAllocator sess2(so2, *env);
  ASSERT_STATUS_OK(sess2.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess2.Initialize());

  // This line ensures the allocator in the session is the same as that in the env
  ASSERT_EQ(sess1.GetSessionState().GetAllocator(mem_info).get(),
            allocator_ptr.get());

  // This line ensures the underlying OrtAllocator* is the same across 2 sessions.
  ASSERT_NE(sess1.GetSessionState().GetAllocator(mem_info).get(),
            sess2.GetSessionState().GetAllocator(mem_info).get());
}

class InferenceSessionTestSharingInitializer : public InferenceSessionWrapper {
 public:
  InferenceSessionTestSharingInitializer(const SessionOptions& session_options,
                                         const Environment& env)
      : InferenceSessionWrapper(session_options, env) {
  }
};

TEST(InferenceSessionTests, InitializerSharing_EnsureSessionsUseUserAddedInitializer) {
  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  ASSERT_TRUE(st.IsOK());

  // create initializer to share between sessions
  const char* init_name = "W";
  OrtValue val_to_share_from_allocator;
  OrtValue val_to_share;
  std::vector<float> input_data_vec{1., 2., 3., 4., 5., 6.};

  auto allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  CreateMLValue<float>(allocator, {3, 2}, input_data_vec, &val_to_share_from_allocator);

  OrtMemoryInfo mem_info{CPU, OrtArenaAllocator};
  CreateMLValue<float>({3, 2}, input_data_vec.data(), mem_info, &val_to_share);

  // create sessions to share the allocator
  SessionOptions so1;
  ASSERT_STATUS_OK(so1.AddInitializer(init_name, &val_to_share));

  // ensure an error is returned when an initializer with the same name is added.
  ASSERT_FALSE(so1.AddInitializer(init_name, &val_to_share).IsOK());

  // ensure an error is returned when an initializer with a buffer NOT owned by the user is added.
  ASSERT_FALSE(so1.AddInitializer(init_name, &val_to_share_from_allocator).IsOK());

  InferenceSessionTestSharingInitializer sess1(so1, *env);
  ASSERT_STATUS_OK(sess1.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess1.Initialize());

  SessionOptions so2;
  ASSERT_STATUS_OK(so2.AddInitializer(init_name, &val_to_share));
  InferenceSessionTestSharingInitializer sess2(so2, *env);
  ASSERT_STATUS_OK(sess2.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess2.Initialize());

  SessionOptions so3;
  InferenceSessionTestSharingInitializer sess3(so3, *env);
  ASSERT_STATUS_OK(sess3.Load(MODEL_URI));
  ASSERT_STATUS_OK(sess3.Initialize());

  int so1_idx;
  ASSERT_STATUS_OK(sess1.GetSessionState().GetOrtValueNameIdxMap().GetIdx(init_name, so1_idx));
  const auto* so1_init_buffer = sess1.GetSessionState().GetInitializedTensors().at(so1_idx).Get<Tensor>().Data<float>();

  int so2_idx;
  ASSERT_STATUS_OK(sess2.GetSessionState().GetOrtValueNameIdxMap().GetIdx(init_name, so2_idx));
  const auto* so2_init_buffer = sess2.GetSessionState().GetInitializedTensors().at(so2_idx).Get<Tensor>().Data<float>();

  // Ensure session1 stores the same data ptr as the one supplied by the user
  ASSERT_EQ(so1_init_buffer, val_to_share.Get<Tensor>().Data<float>());

  // Ensure both sessions share the same data ptr
  ASSERT_EQ(so1_init_buffer, so2_init_buffer);

  int so3_idx;
  ASSERT_STATUS_OK(sess3.GetSessionState().GetOrtValueNameIdxMap().GetIdx(init_name, so3_idx));
  const auto* so3_init_buffer = sess3.GetSessionState().GetInitializedTensors().at(so3_idx).Get<Tensor>().Data<float>();

  // Ensure session 3 doesn't share the same data ptr as any other session
  ASSERT_NE(so3_init_buffer, so1_init_buffer);
  ASSERT_NE(so3_init_buffer, so2_init_buffer);

  // Ensure session 3 doesn't share the same data ptr as the one supplied by the user for any of the other sessions
  ASSERT_NE(so3_init_buffer, val_to_share.Get<Tensor>().Data<float>());
}

void RunModelWithDenormalAsZero(InferenceSession& session_object,
                                const RunOptions& run_options,
                                bool set_denormal_as_zero) {
  const float denormal_float = 1e-38f;

  // prepare input X
  std::vector<int64_t> dims_mul{3, 2};
  std::vector<float> values_mul(6);
  std::fill(values_mul.begin(), values_mul.end(), denormal_float);
  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims_mul, values_mul, &ml_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare output C
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul{3, 1};
  std::vector<float> expected_values_mul(3);
  std::fill(expected_values_mul.begin(), expected_values_mul.end(),
            (set_denormal_as_zero) ? 0.0f : denormal_float * 3);

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, expected_dims_mul, expected_values_mul);
}

void VerifyThreadPoolWithDenormalAsZero(onnxruntime::concurrency::ThreadPool* tp,
                                        bool set_denormal_as_zero) {
  const int num_tasks = 4;
  const float denormal_float = 1e-38f;
  const double denormal_double = 1e-308;

  std::array<float, num_tasks> input_float;
  input_float.fill(denormal_float);
  std::array<double, num_tasks> input_double;
  input_double.fill(denormal_double);

  ThreadPool::TrySimpleParallelFor(tp, num_tasks, [&](std::ptrdiff_t i) {
    input_float[i] *= 2;
    input_double[i] *= 2;
  });
  std::for_each(input_float.begin(), input_float.end(), [&](float f) {
    EXPECT_EQ(f, (set_denormal_as_zero) ? 0.0f : denormal_float * 2);
  });
  std::for_each(input_double.begin(), input_double.end(), [&](double d) {
    EXPECT_EQ(d, (set_denormal_as_zero) ? 0.0 : denormal_double * 2);
  });
}

// test global thread pool with setting denormal as zero
TEST(InferenceSessionTests, GlobalThreadPoolWithDenormalAsZero) {
  // test if denormal-as-zero mode is supported
  if (!SetDenormalAsZero(false)) {
    return;
  }

  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  OrtThreadingOptions tp_options;
  tp_options.inter_op_thread_pool_params.thread_pool_size = 2;
  tp_options.inter_op_thread_pool_params.set_denormal_as_zero = true;
  tp_options.intra_op_thread_pool_params.thread_pool_size = 2;
  tp_options.intra_op_thread_pool_params.set_denormal_as_zero = true;
  auto st = Environment::Create(std::move(logging_manager), env, &tp_options, true);
  ASSERT_TRUE(st.IsOK());

  SessionOptions so;
  so.config_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");
  so.use_per_session_threads = false;

  std::string configValue;
  so.config_options.TryGetConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, configValue);
  EXPECT_EQ(configValue, "1");

  // Since only the first session option for flush-to-zero and denormal-as-zero are effective,
  // set them manually here for a test.
#ifdef _OPENMP
  InitializeWithDenormalAsZero(true);
#endif
  SetDenormalAsZero(true);

  InferenceSessionTestGlobalThreadPools session{so, *env};
  ASSERT_STATUS_OK(session.Load("testdata/matmul_1.onnx"));
  ASSERT_STATUS_OK(session.Initialize());

  RunOptions run_options;
  run_options.run_tag = "global_thread_pool_denormal_as_zero";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  RunModelWithDenormalAsZero(session, run_options, true);

#ifndef _OPENMP
  VerifyThreadPoolWithDenormalAsZero(env->GetIntraOpThreadPool(), true);
#endif
  VerifyThreadPoolWithDenormalAsZero(env->GetInterOpThreadPool(), true);

  // Set back to default.
#ifdef _OPENMP
  InitializeWithDenormalAsZero(false);
#endif
  SetDenormalAsZero(false);
}

// test inter thread pool with setting denormal as zero
TEST(InferenceSessionTests, InterThreadPoolWithDenormalAsZero) {
  // test if denormal-as-zero mode is supported
  if (!SetDenormalAsZero(false)) {
    return;
  }

  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(new CLogSink()), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  std::unique_ptr<Environment> env;
  auto st = Environment::Create(std::move(logging_manager), env);
  ASSERT_TRUE(st.IsOK());

  SessionOptions so;

  // inference session without denormal as zero.
  so.execution_mode = ExecutionMode::ORT_PARALLEL;
  // inference session with denormal as zero
  so.config_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

  // Since only the first session option for flush-to-zero and denormal-as-zero are effective,
  // set them manually here for a test.
#ifdef _OPENMP
  InitializeWithDenormalAsZero(true);
#endif
  SetDenormalAsZero(true);

  InferenceSessionTestGlobalThreadPools session1{so, *env};
  ASSERT_STATUS_OK(session1.Load("testdata/matmul_1.onnx"));
  ASSERT_STATUS_OK(session1.Initialize());

  RunOptions run_options;
  run_options.run_tag = "inter_thread_pool_denormal_as_zero";
  run_options.run_log_severity_level = static_cast<int>(Severity::kVERBOSE);
  RunModelWithDenormalAsZero(session1, run_options, true);

#ifndef _OPENMP
  VerifyThreadPoolWithDenormalAsZero(session1.GetIntraOpThreadPoolToUse(), true);
#endif
  VerifyThreadPoolWithDenormalAsZero(session1.GetInterOpThreadPoolToUse(), true);

  // inference session without denormal as zero.
  so.config_options.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "0");

  // Since only the first session option for flush-to-zero and denormal-as-zero are effective,
  // set them manually here for a test.
#ifdef _OPENMP
  InitializeWithDenormalAsZero(false);
#endif
  SetDenormalAsZero(false);

  InferenceSessionTestGlobalThreadPools session2{so, *env};
  ASSERT_STATUS_OK(session2.Load("testdata/matmul_1.onnx"));
  ASSERT_STATUS_OK(session2.Initialize());

  // Since it's parallel, it runs on threads.
  RunModelWithDenormalAsZero(session2, run_options, false);

#ifndef _OPENMP
  VerifyThreadPoolWithDenormalAsZero(session2.GetIntraOpThreadPoolToUse(), false);
#endif
  VerifyThreadPoolWithDenormalAsZero(session2.GetInterOpThreadPoolToUse(), false);
}

}  // namespace test
}  // namespace onnxruntime

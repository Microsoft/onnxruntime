// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include "gtest/gtest.h"

#ifdef ORT_ENABLE_STREAM
#include "nlohmann/json.hpp"
using json = nlohmann::json;
#endif

#include "core/framework/session_state.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "test/framework/model_builder_utils.h"
#include "core/framework/allocation_planner.h"
#include "core/session/inference_session.h"
#include "core/graph/model.h"
#include "core/graph/graph_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/util/thread_utils.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/file_util.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#endif  // USE_CUDA
#include "core/session/onnxruntime_session_options_config_keys.h"
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
#ifdef USE_CUDA
ProviderInfo_CUDA& GetProviderInfo_CUDA();
#endif
namespace test {

namespace modelbuilder {

class NodeCounter {
 private:
  static int node_count_;

 public:
  static int Next() { return ++node_count_; }
};

int NodeCounter::node_count_ = 0;

struct UnaryNode {
  std::vector<onnxruntime::NodeArg*> input_args;
  std::vector<onnxruntime::NodeArg*> output_args;
  onnxruntime::Node* p_node;

  UnaryNode(onnxruntime::Graph& graph, const std::string& op, onnxruntime::NodeArg* p_input_arg,
            onnxruntime::NodeArg* p_output_arg)
      : input_args({p_input_arg}), output_args({p_output_arg}) {
    int num = NodeCounter::Next();
    p_node = &graph.AddNode("node" + std::to_string(num), op, "test op", input_args, output_args);
  }

  UnaryNode(onnxruntime::Graph& graph, onnxruntime::NodeArg* p_input_arg, onnxruntime::NodeArg* p_output_arg)
      : UnaryNode(graph, "Transpose", p_input_arg, p_output_arg) {}

  UnaryNode(onnxruntime::Graph& graph, std::string& node_name, const std::string& op, std::vector<onnxruntime::NodeArg*>& inputs,
            std::vector<onnxruntime::NodeArg*>& outputs) : input_args(inputs), output_args(outputs) {
    p_node = &graph.AddNode(node_name, op, "test op", input_args, output_args);
  }
};

class DummyOpKernel : public OpKernel {
 public:
  DummyOpKernel(const OpKernelInfo& p) : OpKernel(p) {}
  Status Compute(OpKernelContext* context) const {
    ORT_UNUSED_PARAMETER(context);
    return Status::OK();
  }
  Status ComputeAsync(OpKernelContext* context, DoneCallback done) const {
    ORT_UNUSED_PARAMETER(context);
    ORT_UNUSED_PARAMETER(done);
    return Status::OK();
  }
};

}  // namespace modelbuilder

using namespace modelbuilder;

class AllocationPlanTestUtility {
 public:
  static void CheckAllocationKind(const SequentialExecutionPlan& plan, std::vector<AllocKind>& expected) {
    ASSERT_EQ(plan.allocation_plan.size(), expected.size()) << "Allocation plan of wrong size";
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(plan.allocation_plan[i].alloc_kind, expected[i]) << "Error in allocation kind at position " << i;
    }
  }
  // The free list has been re-implmented.
  // remove those checkers first.
  // TODO: add the tests for new release plan.
};

typedef std::unordered_map<const onnxruntime::NodeArg*, TensorShapeProto*> ShapeMap;

class SequentialPlannerTestContext : public ISequentialPlannerContext {
 public:
  SequentialPlannerTestContext(ShapeMap* shape_map) : shape_map_(shape_map) {}

  TensorShapeProto* GetShape(const onnxruntime::NodeArg& arg) const override {
    ++shape_queries_;
    auto iter = shape_map_->find(&arg);
    return (shape_map_->end() != iter) ? iter->second : nullptr;
  }

  size_t GetShapeQueryCount() const { return shape_queries_; }

 private:
  ShapeMap* shape_map_;
  mutable size_t shape_queries_ = 0;
};

class ParallelPlannerTestContext : public SequentialPlannerTestContext {
 public:
  ParallelPlannerTestContext(ShapeMap* shape_map) : SequentialPlannerTestContext(shape_map) {
  }
  bool IsParallelExecutionEnabled() const override { return true; }
  ExecutionOrder GetExecutionOrder() const override { return ExecutionOrder::DEFAULT; }
  bool GetEnableMemoryReuse() const override { return false; }
};

class PlannerTest : public ::testing::Test {
 private:
  void index(const std::string& name, int& out) {
    ASSERT_TRUE(state_->GetOrtValueNameIdxMap().GetIdx(name, out).IsOK());
  }

  onnxruntime::Model model_;
  onnxruntime::Graph& graph_;

  // some standard components used to build test-cases:
  Type float_type_;

  std::unique_ptr<::onnxruntime::KernelDef> std_kernel_;               // a unary kernel with no-aliasing and no-in-place
  std::unique_ptr<::onnxruntime::KernelDef> in_place_kernel_;          // a unary kernel with in-place
  std::unique_ptr<::onnxruntime::KernelDef> external_outputs_kernel_;  // an unary kernel with external outputs
#ifdef ENABLE_STRIDED_TENSORS
  std::unique_ptr<::onnxruntime::KernelDef> may_strided_input_kernel_;   // an uinary kernel with may_strided_input
  std::unique_ptr<::onnxruntime::KernelDef> may_strided_output_kernel_;  // an unary kernel with may_strided_output
#endif

  std::unordered_map<std::string, onnxruntime::NodeArg*> name_to_arg_;
  std::vector<std::unique_ptr<UnaryNode>> nodes_;
  std::vector<std::unique_ptr<OpKernelInfo>> op_kernel_infos_;
  std::vector<std::pair<onnxruntime::Node*, KernelDef&>> kernel_bindings_;
  ExecutionProviders execution_providers_;
  std::unique_ptr<concurrency::ThreadPool> tp_;
  DataTransferManager dtm_;
  ExternalDataLoaderManager edlm_;
  profiling::Profiler profiler_;
  std::unique_ptr<SessionOptions> sess_options_;
  std::unique_ptr<SessionState> state_;
  ShapeMap shape_map_;
  std::optional<SequentialExecutionPlan> plan_;
  size_t shape_queries_ = 0;

 public:
  PlannerTest()
      : model_("test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 10}}, {}, DefaultLoggingManager().DefaultLogger()),
        graph_(model_.MainGraph()),
        tp_(concurrency::CreateThreadPool(&onnxruntime::Env::Default(), OrtThreadPoolParams(),
                                          concurrency::ThreadPoolType::INTRA_OP)) {
    std_kernel_ = KernelDefBuilder().SetName("Transpose").Provider(kCpuExecutionProvider).SinceVersion(1, 10).Build();
    in_place_kernel_ =
        KernelDefBuilder().SetName("Relu").Provider(kCpuExecutionProvider).SinceVersion(1, 10).MayInplace(0, 0).Build();
    external_outputs_kernel_ =
        KernelDefBuilder().SetName("Tanh").Provider(kCpuExecutionProvider).SinceVersion(1, 10).ExternalOutputs().Build();
#ifdef ENABLE_STRIDED_TENSORS
    may_strided_input_kernel_ = KernelDefBuilder()
                                    .SetName("Abs")
                                    .Provider(kCpuExecutionProvider)
                                    .SinceVersion(1, 10)
                                    .MayStridedInput(0)
                                    .Build();
    may_strided_output_kernel_ = KernelDefBuilder()
                                     .SetName("Neg")
                                     .Provider(kCpuExecutionProvider)
                                     .SinceVersion(1, 10)
                                     .MayStridedOutput(0, 0)
                                     .Build();
#endif
    CPUExecutionProviderInfo epi;
    auto execution_provider = std::make_unique<CPUExecutionProvider>(epi);
    ORT_THROW_IF_ERROR(execution_providers_.Add("CPUExecutionProvider", std::move(execution_provider)));
    sess_options_ = std::make_unique<SessionOptions>();
    sess_options_->enable_mem_pattern = false;
    sess_options_->use_deterministic_compute = false;
    sess_options_->enable_mem_reuse = true;
    state_.reset(new SessionState(graph_, execution_providers_, tp_.get(), nullptr, dtm_, edlm_,
                                  DefaultLoggingManager().DefaultLogger(), profiler_, *sess_options_));
  }

  onnxruntime::NodeArg* Arg(const std::string& name) {
    auto iter = name_to_arg_.find(name);
    if (name_to_arg_.end() != iter) return iter->second;
    return (name_to_arg_[name] = &graph_.GetOrCreateNodeArg(name, &float_type_.value));
  }

  template <typename T>
  void SetTensorType(const std::string& name) {
    name_to_arg_[name] = &graph_.GetOrCreateNodeArg(name, DataTypeImpl::GetTensorType<T>()->GetTypeProto());
  }

  onnxruntime::Node* AddNode(::onnxruntime::KernelDef& kernel_def, std::string& input, std::string& output) {
    auto node = std::make_unique<UnaryNode>(graph_, kernel_def.OpName(), Arg(input), Arg(output));
    auto* p_node = node->p_node;
    p_node->SetExecutionProviderType(kernel_def.Provider());
    nodes_.push_back(std::move(node));
    kernel_bindings_.emplace_back(p_node, kernel_def);
    return p_node;
  }

  onnxruntime::Node* AddNode(::onnxruntime::KernelDef& kernel_def, std::string& node_name, std::vector<onnxruntime::NodeArg*>& input, std::vector<onnxruntime::NodeArg*>& output) {
    auto node = std::make_unique<UnaryNode>(graph_, node_name, kernel_def.OpName(), input, output);
    auto* p_node = node->p_node;
    p_node->SetExecutionProviderType(kernel_def.Provider());
    nodes_.push_back(std::move(node));
    kernel_bindings_.emplace_back(p_node, kernel_def);
    return p_node;
  }

  onnxruntime::Node* AddNormalNode(std::string& input, std::string& output) {
    return AddNode(*std_kernel_, input, output);
  }

  onnxruntime::Node* AddInplaceNode(std::string& input, std::string& output) {
    return AddNode(*in_place_kernel_, input, output);
  }

  onnxruntime::Node* AddExternalOutputsNode(std::string& input, std::string& output) {
    return AddNode(*external_outputs_kernel_, input, output);
  }

#ifdef ENABLE_STRIDED_TENSORS
  onnxruntime::Node* AddMayStridedInputNode(std::string& input, std::string& output) {
    return AddNode(*may_strided_input_kernel_, input, output);
  }

  onnxruntime::Node* AddMayStridedOutputNode(std::string& input, std::string& output) {
    return AddNode(*may_strided_output_kernel_, input, output);
  }
#endif

  void BindKernel(onnxruntime::Node* p_node, ::onnxruntime::KernelDef& kernel_def, KernelRegistry* reg,
                  std::unordered_map<NodeIndex, gsl::not_null<const KernelCreateInfo*>>& kernel_create_info_map) {
    const auto& logger = DefaultLoggingManager().DefaultLogger();
    const IExecutionProvider* ep = execution_providers_.Get(*p_node);
    ASSERT_NE(ep, nullptr);
    auto info = std::make_unique<OpKernelInfo>(
        *p_node, kernel_def, *ep, state_->GetInitializedTensors(), state_->GetOrtValueNameIdxMap(),
        state_->GetDataTransferMgr(), state_->GetAllocators(), state_->GetSessionOptions().config_options);

    op_kernel_infos_.push_back(std::move(info));
    const auto kernel_type_str_resolver = OpSchemaKernelTypeStrResolver{};
    if (!KernelRegistry::HasImplementationOf(*reg, *p_node, onnxruntime::kCpuExecutionProvider,
                                             kernel_type_str_resolver, logger)) {
      ASSERT_STATUS_OK(reg->Register(
          KernelCreateInfo(std::make_unique<KernelDef>(kernel_def),
                           [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
                             out = std::make_unique<DummyOpKernel>(info);
                             return Status::OK();
                           })));
    }

    const KernelCreateInfo* kci;
    ASSERT_STATUS_OK(reg->TryFindKernel(*p_node, "", kernel_type_str_resolver, logger, &kci));
    kernel_create_info_map.insert({p_node->Index(), gsl::not_null<const KernelCreateInfo*>(kci)});
  }

  void SetShape(std::string& name, TensorShapeProto* shape) { shape_map_[Arg(name)] = shape; }

  void SetShape(std::initializer_list<std::pair<std::string&, TensorShapeProto*>> shapes) {
    for (auto& pair : shapes) {
      SetShape(pair.first, pair.second);
    }
  }

  void CreatePlan(const std::vector<const NodeArg*>& outer_scope_node_args = {},
                  bool invoke_createPlan_explicityly = true) {
    state_.reset(new SessionState(graph_, execution_providers_, tp_.get(), nullptr, dtm_, edlm_,
                                  DefaultLoggingManager().DefaultLogger(), profiler_, *sess_options_));
    EXPECT_EQ(graph_.Resolve(), Status::OK());

    std::shared_ptr<KernelRegistry> reg = std::make_shared<KernelRegistry>();
    std::unordered_map<NodeIndex, gsl::not_null<const KernelCreateInfo*>> kernel_create_info_map;

    for (auto& binding : kernel_bindings_) {
      BindKernel(binding.first, binding.second, reg.get(), kernel_create_info_map);
    }

    auto cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
    KernelRegistryManager kernel_registry_manager;
    kernel_registry_manager.RegisterKernelRegistry(reg);
    auto status = kernel_registry_manager.RegisterKernels(execution_providers_);
    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

    // CreatePlan is called inside FinalizeSessionState and usually the initializers are removed following that.
    // Leave initializers so we can duplicate the call to CreatePlan from here to validate.
    constexpr bool remove_initializers = false;
    status = state_->FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager, {}, remove_initializers);

    EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
    SequentialPlannerTestContext test_context(&shape_map_);
    plan_.emplace();

    class MockStreamHandleRegsitry : public IStreamCommandHandleRegistry {
     public:
      // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
      // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
      virtual WaitNotificationFn GetWaitHandle(const OrtDevice& /*notification_owner_device*/,
                                               const OrtDevice& /*executor_device*/) const override {
        return nullptr;
      }

      virtual CreateStreamFn GetCreateStreamFn(const OrtDevice::DeviceType /*execution_provider_type*/) const override {
        return nullptr;
      }

      virtual void RegisterWaitFn(const OrtDevice::DeviceType /*notification_ep_type*/, const OrtDevice::DeviceType /*ep_type*/, WaitNotificationFn /*fn*/) override {}

      virtual void RegisterCreateStreamFn(const OrtDevice::DeviceType /*ep_type*/, CreateStreamFn /*f*/) override {}
    };

    if (invoke_createPlan_explicityly) {
      onnxruntime::GraphViewer graph_viewer{graph_};
      status = SequentialPlanner::CreatePlan(
          nullptr,
          graph_viewer,
          outer_scope_node_args,
          execution_providers_,
          kernel_create_info_map,
          {},
          {},
          state_->GetOrtValueNameIdxMap(),
          test_context,
#ifdef ORT_ENABLE_STREAM
          MockStreamHandleRegsitry(),
#endif
          /* {{kCpuExecutionProvider, 1}}, {},*/
          ORT_TSTR(""),
          DefaultLoggingManager().DefaultLogger(),
          plan_);

      EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
      shape_queries_ = test_context.GetShapeQueryCount();
      // AllocationPlanTestUtility::BasicIntegrityCheck(*plan_, name_to_arg_.size());
    }
  }

  void CheckAllocKind(const std::string& name, AllocKind kind) {
    int id;
    index(name, id);
    EXPECT_EQ(plan_->allocation_plan[id].alloc_kind, kind) << "Error in allocation kind for " << name;
  }

  void CheckReusedBuffer(const std::string& name, const std::string& expected) {
    int id, expected_id;
    index(name, id);
    index(expected, expected_id);
    EXPECT_EQ(plan_->allocation_plan[id].reused_buffer, expected_id);
  }

  void CheckFreed(int step_number, std::initializer_list<std::string> freed_items) {
    // TODO: add the checker for new implementation of release plan
    //// create set and check equality
    std::unordered_set<int> expected;
    for (auto& name : freed_items) {
      int id;
      index(name, id);
      expected.insert(id);
    }
    std::unordered_set<int> plan_result;
    // todo - support multi-stream
    EXPECT_EQ(plan_->execution_plan.size(), 1U);
    int list_size = static_cast<int>(plan_->node_release_list.size());
    EXPECT_GT(list_size, step_number);
    for (auto action_idx : plan_->node_release_list[step_number]) {
      const size_t ortvalue_id = plan_->release_actions[action_idx].value_index;
      plan_result.insert(static_cast<int>(ortvalue_id));
    }
    EXPECT_EQ(plan_result, expected) << "Freed items incorrect for step " << step_number;
  }

 protected:
  Graph& GetGraph() { return graph_; }
  size_t GetShapeQueryCount() const { return shape_queries_; }
  template <typename T>
  void CheckStorageSizedReuse(AllocKind expected) {
    std::string input = "input", a = "a", b = "b", c = "c", output = "output";
    SetTensorType<T>(b);
    SetTensorType<T>(c);
    SetTensorType<T>(output);
    AddNormalNode(input, a);
    auto cast_kernel = KernelDefBuilder().SetName("Cast").Provider(kCpuExecutionProvider).SinceVersion(9).Build();
    AddNode(*cast_kernel, a, b)->AddAttribute("to", static_cast<int64_t>(utils::ToTensorProtoElementType<T>()));
    AddNormalNode(b, c);
    AddNormalNode(c, output);
    Shape compatible{"batch", "width"}, different{"batch", "other"};
    SetShape({{a, &compatible.value}, {b, &different.value}, {c, &compatible.value}});

    CreatePlan();

    CheckAllocKind(c, expected);
    if (expected == AllocKind::kReuse) CheckReusedBuffer(c, a);
  }
  const SequentialExecutionPlan& GetPlan() const { return *plan_; }
  const SessionState& GetState() const { return *state_; }
  ExecutionProviders& GetExecutionProviders() { return execution_providers_; }
  void SetNodePartitionConfigFilePath(const char* config_file_path) {
    ORT_THROW_IF_ERROR(sess_options_->config_options.AddConfigEntry(kNodePartitionConfigFile, config_file_path));
  }
  std::unique_ptr<::onnxruntime::KernelDef>& GetStdKernel() { return std_kernel_; }
  void SetExecutionOrder(ExecutionOrder order) { sess_options_->execution_order = order; }
#ifdef USE_CUDA
  void MemcpyToHostInCuda_TransposeInCudaAndCpu(const char* partitionConfigFile = nullptr) {
    std::unique_ptr<::onnxruntime::KernelDef> cudaKernel = KernelDefBuilder().SetName("MemcpyToHost").Provider(kCudaExecutionProvider).SetDefaultOutputMemoryType(OrtMemTypeCPUOutput).Build();
    std::unique_ptr<::onnxruntime::KernelDef> cudaKernelTrans = KernelDefBuilder().SetName("Transpose").Provider(kCudaExecutionProvider).SinceVersion(1, 10).Build();
    std::string Graph_input("Graph_input"), Arg1("Arg1"), Arg2("Arg2"), Arg3("Arg3"), node1("node1"), node2("node2"), node3("node3");
    std::vector<onnxruntime::NodeArg*> input1{Arg(Graph_input)}, output1{Arg(Arg1)}, output2{Arg(Arg2)}, output3{Arg(Arg3)};
    AddNode(*cudaKernel, node1, input1, output1);
    AddNode(*GetStdKernel(), node2, output1, output2);
    AddNode(*cudaKernelTrans, node3, output1, output3);

    CUDAExecutionProviderInfo epi;
    onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA();
    auto epFactory = ep.CreateExecutionProviderFactory(epi);
    std::unique_ptr<IExecutionProvider> execution_provider = epFactory->CreateProvider();
    ORT_THROW_IF_ERROR(GetExecutionProviders().Add("CUDAExecutionProvider", std::move(execution_provider)));

    if (partitionConfigFile != nullptr) SetNodePartitionConfigFilePath(partitionConfigFile);
    CreatePlan({}, false);
  }
#endif  // USE_CUDA
};

TEST_F(PlannerTest, ChainTest) {
  // tensor variables:
  std::string W("W"), X("X"), B("B"), Y("Y"), Z("Z");

  // graph structure:

  ONNX_NAMESPACE::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("W");
  GetGraph().AddInitializedTensor(tensor);

  AddNormalNode(W, X);
  AddNormalNode(X, B);
  AddNormalNode(B, Y);
  AddNormalNode(Y, Z);

  // simulate shape-inference results:
  Shape shape1{50, 100};
  auto shape = &shape1.value;
  SetShape({{X, shape}, {B, shape}, {Y, shape}, {Z, shape}});

  CreatePlan();

  // Expected plan:
  //   W: kAllocateStatically; X: kAllocate; B: kAllocate; Y: kReuse (X); post-node3: free(B); Z is returned output
  CheckAllocKind(W, AllocKind::kAllocateStatically);
  CheckAllocKind(X, AllocKind::kAllocate);
  CheckAllocKind(B, AllocKind::kAllocate);
  CheckAllocKind(Y, AllocKind::kReuse);
  CheckAllocKind(Z, AllocKind::kAllocateOutput);

  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {B});
  CheckFreed(3, {X});
}

/* InputOutputTest: Test that:
(a) All inputs are classified as kPreExisting,
(b) All outer scope node args are classified as kPreExisting,
(c) All outputs are classified as kAllocate (in this example),
(d) Neither input nor outputs are freed.
*/
TEST_F(PlannerTest, InputOutputTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), Y1("Y1"), Y2("Y2"), Outer1("Outer1"), Y3("Y3");

  // graph structure:
  AddNormalNode(X1, Y1);
  AddNormalNode(X2, Y2);

  // add node that consumes an outer scope node arg
  auto outer_node = AddNormalNode(Outer1, Y3);
  const NodeArg* outer_scope_node_arg = outer_node->InputDefs().at(0);
  GetGraph().AddOuterScopeNodeArg(Outer1);

  // simulate no shape-inference:

  CreatePlan({outer_scope_node_arg});

  // X1: kPreExisting, X2: kPreExisting, Outer1: kPreExisting, Y1: kAllocate, Y2: kAllocate, Y3: kAllocate
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kPreExisting);
  CheckAllocKind(Outer1, AllocKind::kPreExisting);
  CheckAllocKind(Y1, AllocKind::kAllocateOutput);
  CheckAllocKind(Y2, AllocKind::kAllocateOutput);
  CheckAllocKind(Y3, AllocKind::kAllocateOutput);

  // Nothing should be freed (since they are either inputs or outputs)
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {});
}

// InPlaceTest: Check that we reuse when Inplace allows us to.

TEST_F(PlannerTest, InPlaceTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kReuse);
  CheckAllocKind(X4, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X2});
}

TEST_F(PlannerTest, ExternalOutputsTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddExternalOutputsNode(X1, X2);  // external-outputs operator; X1: input; X2: temporary
  AddNormalNode(X2, X3);           // normal operator; X3: temporary
  AddNormalNode(X3, X4);           // normal operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocatedExternally);
  CheckAllocKind(X3, AllocKind::kAllocate);
  CheckAllocKind(X4, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  // X2 will not be reused but will be freed (to release the current reference). X3 will be allocated and will be freed.
  CheckFreed(0, {});
  CheckFreed(1, {X2});
  CheckFreed(2, {X3});
}

TEST_F(PlannerTest, ExternalOutputsNoReuseTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4"), X5("X5");

  // graph structure:
  AddExternalOutputsNode(X1, X2);  // external-outputs operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);          // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);           // normal operator; X4: temporary
  AddNormalNode(X4, X5);           // normal operator; X5: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}, {X5, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocatedExternally);
  CheckAllocKind(X3, AllocKind::kAllocate);  // Should not be Reused.
  CheckAllocKind(X4, AllocKind::kAllocate);
  CheckAllocKind(X5, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  // X2 will not be reused. X3 will be allocated and will be freed.
  CheckFreed(0, {});
  CheckFreed(1, {X2});
  CheckFreed(2, {X3});
  CheckFreed(3, {X4});
}

#ifdef ENABLE_STRIDED_TENSORS
TEST_F(PlannerTest, MayStridedTest1) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3");

  // graph structure:
  AddNormalNode(X1, X2);
  AddMayStridedOutputNode(X2, X3);  // may_strided_output as graph output.

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  // X2 will not be reused because X3 is a graph output. X3 will be allocated and will be freed.
  CheckFreed(0, {});
  CheckFreed(1, {X2});
}

TEST_F(PlannerTest, MayStridedTest2) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddMayStridedOutputNode(X1, X2);  // X2 can reuse X1, and is a strided output.
  AddMayStridedInputNode(X2, X3);   // X3 is a graph output, cannot reuse.
  AddMayStridedInputNode(X2, X4);   // X4 is a graph output, cannot reuse.

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kReuse);
  CheckAllocKind(X3, AllocKind::kAllocateOutput);
  CheckAllocKind(X4, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  // X2 will not be reused and will not be freed. X3 will be allocated and will be freed.
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {});
}

TEST_F(PlannerTest, MayStridedTest3) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddMayStridedOutputNode(X1, X2);  // X2 cannot strided reuse X1 because,
  // one of X2's consumers is a node not supporting strided input. So X2 is a allocate.
  AddMayStridedInputNode(X2, X3);  // X3 is a graph output, cannot reuse.
  AddNormalNode(X2, X4);

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kAllocateOutput);
  CheckAllocKind(X4, AllocKind::kAllocateOutput);

  // Be noted: the last two nodes added can run in two different orders, we need figure out the exact order
  // we planned then we know how to check the free order.
  const GraphViewer& graph_viewer = GetState().GetGraphViewer();
  // Normal node index is 2.
  bool does_normal_node_run_at_last = graph_viewer.GetNodesInTopologicalOrder()[2] == 2;

  // check each ml-value is freed at appropriate step

  CheckFreed(0, {});

  if (does_normal_node_run_at_last) {
    // Normal node has node index to be 2, but it is possible that the normal node is executed after the strided node.
    // Then X2 will released once the normal node is executed.
    CheckFreed(1, {});
    CheckFreed(2, {X2});
  } else {
    // Normal node has node index to be 2, and is executed before the strided node.
    // So X2 will be released after the strided node (node index to be 1) is executed.
    CheckFreed(2, {});
    CheckFreed(1, {X2});
  }
}
#endif

// InPlaceSizeMismatchTest: Check that Inplace reuse is not allowed when sizes don't match.
// Also tests reuse of disjoint lifetime tensors.
TEST_F(PlannerTest, InPlaceSizeMismatchTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4"), X5("X5");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: temporary (reuse X2)
  AddInplaceNode(X4, X5);  // may-in-place operator; X5 output

  // simulate shape-inference results:
  Shape shape1w{"M", "N"};
  auto shape1 = &shape1w.value;
  Shape shape2w{"M", "K"};
  auto shape2 = &shape2w.value;
  SetShape({{X1, shape1}, {X2, shape1}, {X3, shape2}, {X4, shape1}, {X5, shape1}});

  CreatePlan();

  // check allocation kind:
  CheckAllocKind(X1, AllocKind::kPreExisting);
  CheckAllocKind(X2, AllocKind::kAllocate);
  CheckAllocKind(X3, AllocKind::kAllocate);
  CheckAllocKind(X4, AllocKind::kReuse);
  CheckAllocKind(X5, AllocKind::kAllocateOutput);

  // check each ml-value is freed at appropriate step
  CheckFreed(0, {});
  CheckFreed(1, {});
  CheckFreed(2, {X3});
  CheckFreed(3, {X2});
}

TEST_F(PlannerTest, ReuseIndexPreservesMostRecentlyFreedCompatibleBuffer) {
  std::string input = "input", a = "a", b = "b", c = "c";
  std::string joined = "joined", reused = "reused", output = "output";
  AddNormalNode(input, a);
  AddNormalNode(input, b);
  AddNormalNode(input, c);
  auto sum_kernel = KernelDefBuilder().SetName("Sum").Provider(kCpuExecutionProvider).SinceVersion(8).Build();
  std::string sum_name = "sum";
  std::vector<NodeArg*> inputs{Arg(a), Arg(b), Arg(c)}, outputs{Arg(joined)};
  AddNode(*sum_kernel, sum_name, inputs, outputs);
  AddNormalNode(joined, reused);
  AddNormalNode(reused, output);
  Shape compatible{"batch", "width"}, incompatible{"batch", "other"}, join_shape{"batch", "joined"};
  SetShape({{a, &compatible.value}, {b, &compatible.value}, {c, &incompatible.value}, {joined, &join_shape.value}, {reused, &compatible.value}, {output, &compatible.value}});

  CreatePlan();

  // Sum frees a, b, then c. Skip the newer incompatible c, but prefer b to a.
  CheckAllocKind(reused, AllocKind::kReuse);
  CheckReusedBuffer(reused, b);
  CheckAllocKind(output, AllocKind::kAllocateOutput);
}

TEST_F(PlannerTest, ReuseIndexPreservesEmptyAndUnknownShapeEligibility) {
  std::string input = "input", a = "a", b = "b", c = "c", output = "output";
  AddNormalNode(input, a);
  AddNormalNode(a, b);
  AddNormalNode(b, c);
  AddNormalNode(c, output);
  Shape empty_symbol{""}, different{"other"};
  SetShape({{a, &empty_symbol.value}, {b, &different.value}, {c, &empty_symbol.value}});

  CreatePlan();

  CheckAllocKind(a, AllocKind::kAllocate);
  CheckAllocKind(b, AllocKind::kAllocate);
  CheckAllocKind(c, AllocKind::kAllocate);
  CheckAllocKind(output, AllocKind::kAllocateOutput);
}

TEST_F(PlannerTest, ReuseIndexAllowsZeroSizedTensors) {
  std::string input = "input", a = "a", b = "b", c = "c", output = "output";
  AddNormalNode(input, a);
  AddNormalNode(a, b);
  AddNormalNode(b, c);
  AddNormalNode(c, output);
  Shape empty{0, 8}, different{1, 8};
  SetShape({{a, &empty.value}, {b, &different.value}, {c, &empty.value}});

  CreatePlan();

  CheckAllocKind(c, AllocKind::kReuse);
  CheckReusedBuffer(c, a);
}

TEST_F(PlannerTest, ReuseIndexDoesNotReuseScalarBuffers) {
  std::string input = "input", a = "a", b = "b", c = "c", output = "output";
  AddNormalNode(input, a);
  AddNormalNode(a, b);
  AddNormalNode(b, c);
  AddNormalNode(c, output);
  Shape scalar(std::initializer_list<int>{}), different{1};
  SetShape({{a, &scalar.value}, {b, &different.value}, {c, &scalar.value}});

  CreatePlan();

  CheckAllocKind(c, AllocKind::kAllocate);
}

TEST_F(PlannerTest, ReuseIndexAllowsEqualWidthDifferentElementTypes) {
  CheckStorageSizedReuse<int32_t>(AllocKind::kReuse);
}

TEST_F(PlannerTest, ReuseIndexRejectsDifferentElementWidths) {
  CheckStorageSizedReuse<int64_t>(AllocKind::kAllocate);
}

TEST_F(PlannerTest, ReuseIndexDoesNotReuseStringBuffers) {
  std::string input = "input", a = "a", b = "b", c = "c", output = "output";
  for (const auto& name : {input, a, b, c, output}) SetTensorType<std::string>(name);
  AddNormalNode(input, a);
  AddNormalNode(a, b);
  AddNormalNode(b, c);
  AddNormalNode(c, output);
  Shape shape{"batch", "width"};
  SetShape({{a, &shape.value}, {b, &shape.value}, {c, &shape.value}});

  CreatePlan();

  CheckAllocKind(c, AllocKind::kAllocate);
}

// The indexed complexity contract applies to full builds, not the minimal linear fallback.
#if !defined(ORT_MINIMAL_BUILD)
TEST_F(PlannerTest, ReuseIndexAvoidsQuadraticShapeQueries) {
  constexpr int node_count = 512;
  InlinedVector<std::string> names;
  InlinedVector<Shape> shapes;
  names.reserve(node_count + 1);
  shapes.reserve(node_count + 1);
  for (int i = 0; i <= node_count; ++i) {
    names.push_back("value_" + std::to_string(i));
    shapes.emplace_back(std::initializer_list<int>{i + 1});
    SetShape(names.back(), &shapes.back().value);
    if (i > 0) AddNormalNode(names[i - 1], names[i]);
  }

  CreatePlan();

  // Count existing shape-context queries instead of relying on wall-clock time.
  // Every freed buffer has a distinct shape, so no full-list scans are needed.
  RecordProperty("shape_queries", std::to_string(GetShapeQueryCount()));
  EXPECT_LE(GetShapeQueryCount(), 3U * node_count);
  for (int i = 1; i < node_count; ++i) CheckAllocKind(names[i], AllocKind::kAllocate);
}
#endif

// Test operator<< to output details of an allocation & execution plan.
TEST_F(PlannerTest, PlanOutputTest) {
  // tensor variables:
  std::string X1("X1"), X2("X2"), X3("X3"), X4("X4");

  // graph structure:
  AddNormalNode(X1, X2);   // no in-place operator; X1: input; X2: temporary
  AddInplaceNode(X2, X3);  // may-in-place operator; X3: temporary
  AddNormalNode(X3, X4);   // no in-place operator; X4: output

  // simulate shape-inference results:
  Shape shape1{"M", "N"};
  auto shape = &shape1.value;
  SetShape({{X1, shape}, {X2, shape}, {X3, shape}, {X4, shape}});

  CreatePlan();

  ORT_TRY {
    std::ostringstream output;
    output << std::make_pair(&GetPlan(), &GetState());
    auto output_size = output.str().size();
    // Currently, we don't check details of the output, as it may change over time.
    EXPECT_GT(output_size, 0u);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&ex]() {
      EXPECT_TRUE(false) << "Exception in producing output: " << ex.what();
    });
  }
}

TEST_F(PlannerTest, PriorityBasedExecutionOrder) {
  std::string X("X"), Y1("Y1"), Y2("Y2"), Y3("Y3");
  AddNormalNode(X, Y1);
  AddNormalNode(X, Y2)->SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
  AddNormalNode(Y2, Y3);

  SetExecutionOrder(ExecutionOrder::PRIORITY_BASED);
  CreatePlan({}, false);

  GraphViewer graph_viewer(GetGraph());
  const auto& expected = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);
  ASSERT_NE(expected, graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT));

  const auto* plan = GetState().GetExecutionPlan();
  ASSERT_EQ(plan->execution_plan.size(), 1u);
  std::vector<NodeIndex> launched;
  for (const auto& step : plan->execution_plan[0]->steps_) {
    if (step->ToString().rfind("LaunchKernel", 0) == 0) {
      launched.push_back(step->GetNodeIndex());
    }
  }
  EXPECT_EQ(launched, expected);
}

#ifdef USE_CUDA
TEST_F(PlannerTest, LocationPlanningForPassThroughExplicitAndImplicitSubgraphInputs) {
  // Types
  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim_param");

  TypeProto int64_scalar;
  int64_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int64_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // The model has a main graph and 2 levels of nested subgraphs
  // Main graph: 2 Abs nodes + one Loop node
  // First level (Loop) subgraph: Identity (condition pass-through) + If node
  // Second level subgraph(s): Then and Else branches: Both have an Add node
  // The Add node adds 2 values:
  // One value from the main graph ("abs_data_0_out") that is "implicitly"
  // consumed by the Loop node and "passed through" to the If subgraphs.
  // Another value from the main graph ("abs_data_1_out") that is "explicitly"
  // consumed by the Loop node as a loop carried dependency and its name in
  // the scope of the Loop node is "loop_state_var".

  // In the Loop subgraph, there are no explicit consumers of "abs_data_0_out"
  // and "loop_state_var", there is only one implicit consumer - "If".
  // We want to ensure that since there are no explicit consumers, the planned locations
  // for these values in this subgraph are the same locations as their corresponding
  // values in the outer scope, thus deferring any copies (if required) till the actual
  // subgraph(s) they are explicitly consumed in.
  auto create_model = [&float_tensor, &int64_scalar, &bool_scalar]() -> Model {
    auto create_if_subgraph = [&float_tensor](bool is_then) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      auto& outer_scope_0 = graph.GetOrCreateNodeArg("loop_state_var", &float_tensor);
      graph.AddOuterScopeNodeArg("loop_state_var");

      auto& outer_scope_1 = graph.GetOrCreateNodeArg("abs_data_0_out", &float_tensor);
      graph.AddOuterScopeNodeArg("abs_data_0_out");

      auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &float_tensor);
      graph.AddNode("if_out", "Add", "add", {&outer_scope_0, &outer_scope_1}, {&if_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    auto create_loop_subgraph = [&create_if_subgraph, &float_tensor, &int64_scalar, &bool_scalar]() -> GraphProto {
      Model model("loop_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      std::vector<NodeArg*> inputs;
      std::vector<NodeArg*> outputs;

      /*  Inputs: iter_num, cond_in, loop carried state variables.
         iter_num_in    cond_in     [loop_state_var]
           (unused)        |               |
                       [Identity]         [If]
                           |               |
                        cond_out     loop_state_var_out
    */

      // graph inputs
      auto& iter_num_in = graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
      auto& cond_in = graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
      auto& loop_state_var = graph.GetOrCreateNodeArg("loop_state_var", &float_tensor);

      // graph outputs
      auto& cond_out = graph.GetOrCreateNodeArg("cond_out", &bool_scalar);
      auto& loop_state_var_out = graph.GetOrCreateNodeArg("loop_state_var_out", &float_tensor);

      // outer scope args
      ORT_IGNORE_RETURN_VALUE(graph.GetOrCreateNodeArg("abs_data_0_out", &float_tensor));
      graph.AddOuterScopeNodeArg("abs_data_0_out");

      // cond_in -> cond_out
      {
        inputs = {&cond_in};
        outputs = {&cond_out};

        graph.AddNode("cond_in_identity", "Identity", "Forward cond_in to cond_out", inputs, outputs);
      }

      // loop_state_var -> If(cond_in) -> loop_state_var_out
      {
        inputs = {&cond_in};
        outputs = {&loop_state_var_out};

        auto& node = graph.AddNode("loop_var_out", "If", "If with loop_state_var as implicit_input", inputs, outputs);
        node.AddAttribute("then_branch", create_if_subgraph(true));
        node.AddAttribute("else_branch", create_if_subgraph(false));
      }

      graph.SetInputs({&iter_num_in, &cond_in, &loop_state_var});
      graph.SetOutputs({&cond_out, &loop_state_var_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();

    // Abs-0
    auto& abs_data_0_in = main_graph.GetOrCreateNodeArg("abs_data_0_in", &float_tensor);
    auto& abs_data_0_out = main_graph.GetOrCreateNodeArg("abs_data_0_out", &float_tensor);
    std::vector<onnxruntime::NodeArg*> abs_0_inputs = {&abs_data_0_in};
    std::vector<onnxruntime::NodeArg*> abs_0_outputs = {&abs_data_0_out};
    main_graph.AddNode("abs_0", "Abs", "node abs", abs_0_inputs, abs_0_outputs);

    // Abs-1
    auto& abs_data_1_in = main_graph.GetOrCreateNodeArg("abs_data_1_in", &float_tensor);
    auto& abs_data_1_out = main_graph.GetOrCreateNodeArg("abs_data_1_out", &float_tensor);
    const std::array<onnxruntime::NodeArg*, 1> abs_1_inputs = {&abs_data_1_in};
    const std::array<onnxruntime::NodeArg*, 1> abs_1_outputs = {&abs_data_1_out};
    main_graph.AddNode("abs_1", "Abs", "node abs", abs_1_inputs, abs_1_outputs);

    // Loop
    auto& iter_num_in = main_graph.GetOrCreateNodeArg("iter_num_in", &int64_scalar);
    auto& cond_in = main_graph.GetOrCreateNodeArg("cond_in", &bool_scalar);
    auto& loop_state_out_var = main_graph.GetOrCreateNodeArg("loop_state_out_var", &float_tensor);

    auto& loop_node = main_graph.AddNode("loop", "Loop", "Loop node",
                                         {&iter_num_in, &cond_in, &abs_data_1_out},
                                         {&loop_state_out_var});
    loop_node.AddAttribute("body", create_loop_subgraph());

    main_graph.SetInputs({&abs_data_0_in, &abs_data_1_in, &iter_num_in, &cond_in});
    main_graph.SetOutputs({&loop_state_out_var});

    auto status = main_graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return model;
  };

  // Create and load session
  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};

  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  std::string s1;
  const bool rc = create_model().ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Check planned locations of values in the main graph that are implicit subgraph inputs
  // and explicit subgraph inputs to the Loop node

  // Main graph (L0 graph)
  const auto& main_graph_session_state = sess.GetSessionState();

  {
    const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
    const auto* main_graph_plan = main_graph_session_state.GetExecutionPlan();

    OrtValueIndex abs_data_0_out_index;
    ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("abs_data_0_out", abs_data_0_out_index));

    OrtValueIndex abs_data_1_out_index;
    ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("abs_data_1_out", abs_data_1_out_index));

    EXPECT_EQ(main_graph_plan->allocation_plan[abs_data_0_out_index].location.Type(), OrtDevice::GPU);
    EXPECT_EQ(main_graph_plan->allocation_plan[abs_data_1_out_index].location.Type(), OrtDevice::GPU);
  }

  // First subgraph (Loop) (L1 graph)
  // There are 3 nodes in the main level- Only one of them has a subgraph (Loop).
  // Find that.
  const SessionState* find_first_subgraph_session_state = nullptr;
  for (size_t i = 0; i < 3; ++i) {
    find_first_subgraph_session_state = main_graph_session_state.GetSubgraphSessionState(i, "body");
    if (find_first_subgraph_session_state) {
      break;
    }
  }

  const auto& first_subgraph_session_state = *find_first_subgraph_session_state;

  {
    const auto& first_subgraph_ort_value_index_map = first_subgraph_session_state.GetOrtValueNameIdxMap();
    const auto* first_subgraph_plan = first_subgraph_session_state.GetExecutionPlan();

    OrtValueIndex abs_data_0_out_index;
    ASSERT_STATUS_OK(first_subgraph_ort_value_index_map.GetIdx("abs_data_0_out", abs_data_0_out_index));

    // "abs_data_1_out" is "loop_state_var" in this scope as it was consumed as an explicit subgraph input
    // to Loop's body subgraph
    OrtValueIndex abs_data_1_out_index;
    ASSERT_STATUS_OK(first_subgraph_ort_value_index_map.GetIdx("loop_state_var", abs_data_1_out_index));

    // There are no explicit consumers of "abs_data_0_out" and "loop_state_var (abs_data_1_out)" in this scope.
    // There is only one implicit consumer "If". Hence, check that we are preserving the locations of these values
    // from the outer scope, thus deferring any copies till the actual nested subgraph these values are used in.
    EXPECT_EQ(first_subgraph_plan->allocation_plan[abs_data_0_out_index].location.Type(), OrtDevice::GPU);
    EXPECT_EQ(first_subgraph_plan->allocation_plan[abs_data_1_out_index].location.Type(), OrtDevice::GPU);
  }
}

TEST_F(PlannerTest, LocationPlanningForInitializersOnlyUsedInANestedSubgraph) {
  // This a simple model that has one outer scope initializer and an `If` node
  // and that initializer is ONLY used in nested subgraphs (both the `If` subgraphs).
  // We want to test that the location planned for this initializer accounts for
  // its usage in the nested subgraphs and statically determines the right location
  // for it (without defaulting to CPU).

  // Types
  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim_param");

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_model = [&float_tensor, &bool_scalar]() -> Model {
    auto create_if_subgraph = [&float_tensor](bool is_then) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      auto& outer_scope_0 = graph.GetOrCreateNodeArg("abs_data_out", &float_tensor);
      graph.AddOuterScopeNodeArg("abs_data_out");

      auto& outer_scope_1 = graph.GetOrCreateNodeArg("init_data", &float_tensor);
      graph.AddOuterScopeNodeArg("init_data");

      auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &float_tensor);
      graph.AddNode("if_out", "Add", "add", {&outer_scope_0, &outer_scope_1}, {&if_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();

    // Abs-0
    auto& abs_data_in = main_graph.GetOrCreateNodeArg("abs_data_in", &float_tensor);
    auto& abs_data_out = main_graph.GetOrCreateNodeArg("abs_data_out", &float_tensor);
    main_graph.AddNode("abs_0", "Abs", "node abs", {&abs_data_in}, {&abs_data_out});

    // If
    auto& if_in = main_graph.GetOrCreateNodeArg("if_in", &bool_scalar);
    auto& if_out = main_graph.GetOrCreateNodeArg("if_out", &float_tensor);
    auto& node = main_graph.AddNode("if_out", "If", "If", {&if_in}, {&if_out});
    node.AddAttribute("then_branch", create_if_subgraph(true));
    node.AddAttribute("else_branch", create_if_subgraph(false));

    // Add initializer to the graph
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.add_dims(1);
    tensor.add_float_data(1.0f);
    tensor.set_data_type(TensorProto_DataType_FLOAT);
    tensor.set_name("init_data");
    graph_utils::AddInitializerWithOrtValue(main_graph, tensor);

    // Main graph's inputs/outputs
    main_graph.SetInputs({&abs_data_in, &if_in});
    main_graph.SetOutputs({&if_out});

    auto status = main_graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return model;
  };

  // Create and load session
  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};

  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  std::string s1;
  const bool rc = create_model().ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Check planned locations for the initializer
  const auto& main_graph_session_state = sess.GetSessionState();
  const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
  const auto* main_graph_plan = main_graph_session_state.GetExecutionPlan();

  OrtValueIndex init_data_index;
  ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("init_data", init_data_index));

  EXPECT_EQ(main_graph_plan->allocation_plan[init_data_index].location.Type(), OrtDevice::GPU);
}

TEST_F(PlannerTest, LocationPlanningForInitializersUsedOnDifferentDevicesInMainGraphAndSubgraph) {
  // This a simple model that has one outer scope initializer, an `If` node followed
  // by a `TopK` node. The initializer is used in both nested subgraphs(`Add` consumes it
  // and requires it on GPU) and main graph(the second input of `TopK` is required on CPU).
  // The right location for the initializer should be CPU as no Memcpy will be inserted
  // for a node in main graph that requires the input(initializer) on CPU if that initializer
  // is placed on GPU by allocation planner.
  TypeProto int_tensor;
  int_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_INT64);
  int_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim_param");

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_model = [&int_tensor, &bool_scalar]() -> Model {
    auto create_if_subgraph = [&int_tensor](bool is_then) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      auto& outer_scope_0 = graph.GetOrCreateNodeArg("abs_data_out", &int_tensor);
      graph.AddOuterScopeNodeArg("abs_data_out");

      auto& outer_scope_1 = graph.GetOrCreateNodeArg("init_data", &int_tensor);
      graph.AddOuterScopeNodeArg("init_data");

      auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &int_tensor);
      graph.AddNode("if_out", "Add", "add", {&outer_scope_0, &outer_scope_1}, {&if_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();

    // Abs-0
    auto& abs_data_in = main_graph.GetOrCreateNodeArg("abs_data_in", &int_tensor);
    auto& abs_data_out = main_graph.GetOrCreateNodeArg("abs_data_out", &int_tensor);
    main_graph.AddNode("abs_0", "Abs", "node abs", {&abs_data_in}, {&abs_data_out});

    // If
    auto& if_in = main_graph.GetOrCreateNodeArg("if_in", &bool_scalar);
    auto& if_out = main_graph.GetOrCreateNodeArg("if_out", &int_tensor);
    auto& node = main_graph.AddNode("if_out", "If", "If", {&if_in}, {&if_out});
    node.AddAttribute("then_branch", create_if_subgraph(true));
    node.AddAttribute("else_branch", create_if_subgraph(false));

    // TopK
    auto& topk_data_in_0 = main_graph.GetOrCreateNodeArg("if_out", &int_tensor);
    auto& topk_data_in_1 = main_graph.GetOrCreateNodeArg("init_data", &int_tensor);
    auto& topk_data_out_0 = main_graph.GetOrCreateNodeArg("topk_data_out_0", &int_tensor);
    auto& topk_data_out_1 = main_graph.GetOrCreateNodeArg("topk_data_out_1", &int_tensor);
    main_graph.AddNode("topk_0", "TopK", "node topk", {&topk_data_in_0, &topk_data_in_1},
                       {&topk_data_out_0, &topk_data_out_1});

    // Add initializer to the graph
    ONNX_NAMESPACE::TensorProto tensor;
    tensor.add_dims(1);
    tensor.add_int64_data(1);
    tensor.set_data_type(TensorProto_DataType_INT64);
    tensor.set_name("init_data");
    graph_utils::AddInitializerWithOrtValue(main_graph, tensor);

    // Main graph's inputs/outputs
    main_graph.SetInputs({&abs_data_in, &if_in});
    main_graph.SetOutputs({&topk_data_out_0, &topk_data_out_1});

    auto status = main_graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return model;
  };

  // Create and load session
  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};

  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  std::string s1;
  const bool rc = create_model().ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Check planned locations for the initializer
  const auto& main_graph_session_state = sess.GetSessionState();
  const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
  const auto* main_graph_plan = main_graph_session_state.GetExecutionPlan();

  OrtValueIndex init_data_index;
  ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("init_data", init_data_index));

  EXPECT_EQ(main_graph_plan->allocation_plan[init_data_index].location.Type(), OrtDevice::CPU);

  // TODO: test para exe plan on subgraph supported
  // const auto* para_graph_plan = const_cast<SessionState&>(main_graph_session_state).GetParallelExecutionPlan();
  // EXPECT_EQ(para_graph_plan->allocation_plan[init_data_index].location.device.Type(), OrtDevice::GPU);
}

TEST_F(PlannerTest, LocationPlanningForImplicitInputsWithoutExplicitConsumersInMainGraph) {
  // This a simple model that has two inputs and an `If` node.
  // The first input is the condition for the `If` node and the second input
  // is an input consumed implicitly by the `If` node to be used in its subgraphs.
  // Note that there are no other explicit consumers of this input in the main graph.

  // We want to test that the location planned for this implicit input is the default device
  // of the EP that the `If` node is partitioned to (which will be CUDA)
  // and that it doesn't default to CPU.

  // Types
  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_param("dim_param");

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_model = [&float_tensor, &bool_scalar]() -> Model {
    auto create_if_subgraph = [&float_tensor](bool is_then) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      auto& outer_scope_0 = graph.GetOrCreateNodeArg("image_data_in", &float_tensor);
      graph.AddOuterScopeNodeArg("image_data_in");

      auto& if_out = graph.GetOrCreateNodeArg(is_then ? "if_then_out" : "if_else_out", &float_tensor);
      graph.AddNode("if_out", "Relu", "relu", {&outer_scope_0}, {&if_out});

      auto status = graph.Resolve();
      EXPECT_EQ(status, Status::OK());

      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();
    auto& image_data_in = main_graph.GetOrCreateNodeArg("image_data_in", &float_tensor);

    // If
    auto& if_in = main_graph.GetOrCreateNodeArg("if_in", &bool_scalar);
    auto& if_out = main_graph.GetOrCreateNodeArg("if_out", &float_tensor);
    auto& node = main_graph.AddNode("if_out", "If", "If", {&if_in}, {&if_out});
    node.AddAttribute("then_branch", create_if_subgraph(true));
    node.AddAttribute("else_branch", create_if_subgraph(false));

    // Main graph's inputs/outputs
    main_graph.SetInputs({&image_data_in, &if_in});
    main_graph.SetOutputs({&if_out});

    auto status = main_graph.Resolve();
    EXPECT_EQ(status, Status::OK());

    return model;
  };

  // Create and load session
  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};

  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  std::string s1;
  const bool rc = create_model().ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  // Check planned locations for the implicit input
  const auto& main_graph_session_state = sess.GetSessionState();
  const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
  const auto* main_graph_plan = main_graph_session_state.GetExecutionPlan();

  OrtValueIndex input_data_index;
  ASSERT_STATUS_OK(main_graph_ort_value_index_map.GetIdx("image_data_in", input_data_index));

  EXPECT_EQ(main_graph_plan->allocation_plan[input_data_index].location.Type(), OrtDevice::GPU);

  // TODO: test para exe plan on subgraph supported
  // const auto* para_graph_plan = const_cast<SessionState&>(main_graph_session_state).GetParallelExecutionPlan();
  // EXPECT_EQ(para_graph_plan->allocation_plan[input_data_index].location.device.Type(), OrtDevice::GPU);
}

void ExpectExecutionStepTypeContains(const SessionState& state,
                                     size_t stream_idx,
                                     size_t step_idx,
                                     const char* expected_type_name,
                                     const char* message) {
  const auto* execution_plan = state.GetExecutionPlan();
  ASSERT_NE(execution_plan, nullptr) << message;
  ASSERT_LT(stream_idx, execution_plan->execution_plan.size()) << message;
  const auto& steps = execution_plan->execution_plan[stream_idx]->steps_;
  ASSERT_LT(step_idx, steps.size()) << message;
  const auto* step = steps[step_idx].get();
  ASSERT_NE(step, nullptr) << message;
  EXPECT_NE(strstr(typeid(*step).name(), expected_type_name), nullptr) << message;
}

// Test MultiStream scenario for the graph:
// node1(CPU ep)->node2(CPU ep)->node3(CUDA ep)->node4(CPU ep)
TEST_F(PlannerTest, MultiStream) {
  ONNX_NAMESPACE::TensorProto tensor;
  tensor.add_dims(1);
  tensor.add_float_data(1.0f);
  tensor.set_data_type(TensorProto_DataType_FLOAT);
  tensor.set_name("Graph_input");
  GetGraph().AddInitializedTensor(tensor);

  std::string Graph_input("Graph_input"), Arg1("Arg1"), Arg2("Arg2"), Arg3("Arg3"), Arg4("Arg4");
  AddNormalNode(Graph_input, Arg1);
  AddNormalNode(Arg1, Arg2);
  std::unique_ptr<::onnxruntime::KernelDef> cudaKernel = KernelDefBuilder().SetName("Transpose").Provider(kCudaExecutionProvider).SinceVersion(1, 10).Build();
  AddNode(*cudaKernel, Arg2, Arg3);
  AddNormalNode(Arg3, Arg4);

  CUDAExecutionProviderInfo epi;
  onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA();
  auto epFactory = ep.CreateExecutionProviderFactory(epi);
  std::unique_ptr<IExecutionProvider> execution_provider = epFactory->CreateProvider();
  ORT_THROW_IF_ERROR(GetExecutionProviders().Add("CUDAExecutionProvider", std::move(execution_provider)));

  CreatePlan({}, false);

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan.size(), 2) << "2 logic streams for CPU and CUDA separately";
  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[0]->steps_.size(), 6) << "CPU stream has 6 steps";
  ExpectExecutionStepTypeContains(GetState(), 0, 0, "LaunchKernelStep", "0th step: LaunchKernelStep for node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 1, "LaunchKernelStep", "1st step: LaunchKernelStep for node 2");
  ExpectExecutionStepTypeContains(GetState(), 0, 2, "TriggerDownstreamStep", "2nd step: TriggerDownstreamStep for node 3, no Activate/Wait step between node 2 and node 3");
  ExpectExecutionStepTypeContains(GetState(), 0, 3, "BarrierStep", "3rd step: BarrierStep for node 4");
  ExpectExecutionStepTypeContains(GetState(), 0, 4, "WaitOnEPStep", "4th step: WaitOnEPStep for node 4");
  ExpectExecutionStepTypeContains(GetState(), 0, 5, "LaunchKernelStep", "5th step: LaunchKernelStep for node 4");

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[1]->steps_.size(), 4) << "CUDA stream has 4 steps";
  ExpectExecutionStepTypeContains(GetState(), 1, 0, "BarrierStep", "0th step: BarrierStep for node 3");
  ExpectExecutionStepTypeContains(GetState(), 1, 1, "LaunchKernelStep", "1st step: LaunchKernelStep for node 3");
  ExpectExecutionStepTypeContains(GetState(), 1, 2, "ActivateNotificationStep", "2nd step: ActivateNofiticationStep by node 3");
  ExpectExecutionStepTypeContains(GetState(), 1, 3, "TriggerDownstreamStep", "3rd step: TriggerDownstreamStep for node 4");
}

// Test execution plan for the graph:
// node1   node2
//   \       /
//    \     /
//      node3
// All 3 nodes are CUDA EP, node1 is in stream0, node2 is in stream1, node3 is in stream2
TEST_F(PlannerTest, MultiStream1StreamWaitFor2Streams) {
  std::unique_ptr<::onnxruntime::KernelDef> cudaKernel = KernelDefBuilder().SetName("Transpose").Provider(kCudaExecutionProvider).SinceVersion(1, 10).Build();
  std::unique_ptr<::onnxruntime::KernelDef> cudaKernelAdd = KernelDefBuilder().SetName("Add").Provider(kCudaExecutionProvider).SinceVersion(1, 10).Build();
  std::string Graph_input("Graph_input"), Arg1("Arg1"), Arg2("Arg2"), Arg3("Arg3"), node1("node1"), node2("node2"), node3("node3");
  std::vector<onnxruntime::NodeArg*> input1{Arg(Graph_input)}, output1{Arg(Arg1)}, output2{Arg(Arg2)}, input3{Arg(Arg1), Arg(Arg2)}, output3{Arg(Arg3)};
  AddNode(*cudaKernel, node1, input1, output1);
  AddNode(*cudaKernel, node2, input1, output2);
  AddNode(*cudaKernelAdd, node3, input3, output3);

  CUDAExecutionProviderInfo epi;
  onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA();
  auto epFactory = ep.CreateExecutionProviderFactory(epi);
  std::unique_ptr<IExecutionProvider> execution_provider = epFactory->CreateProvider();
  ORT_THROW_IF_ERROR(GetExecutionProviders().Add("CUDAExecutionProvider", std::move(execution_provider)));

  SetNodePartitionConfigFilePath("./testdata/multi_stream_models/3_gpu_streams.json");
  CreatePlan({}, false);

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan.size(), 3) << "3 logic streams";
  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[0]->steps_.size(), 3) << "stream 0 has 3 steps";
  ExpectExecutionStepTypeContains(GetState(), 0, 0, "LaunchKernelStep", "0th step: LaunchKernelStep for node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 1, "ActivateNotificationStep", "1st step: ActivateNofiticationStep by node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 2, "TriggerDownstreamStep", "2nd step: TriggerDownstreamStep for node 3");

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[1]->steps_.size(), 3) << "stream 1 has 3 steps";
  ExpectExecutionStepTypeContains(GetState(), 1, 0, "LaunchKernelStep", "0th step: LaunchKernelStep for node 2");
  ExpectExecutionStepTypeContains(GetState(), 1, 1, "ActivateNotificationStep", "1st step: ActivateNofiticationStep by node 2");
  ExpectExecutionStepTypeContains(GetState(), 1, 2, "TriggerDownstreamStep", "2nd step: TriggerDownstreamStep for node 3");

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[2]->steps_.size(), 5) << "stream 2 has 5 steps";
  ExpectExecutionStepTypeContains(GetState(), 2, 0, "BarrierStep", "0th step: BarrierStep for node 3, for TriggerDownstreamStep in stream 1");
  ExpectExecutionStepTypeContains(GetState(), 2, 1, "BarrierStep", "1st step: BarrierStep for node 3, for TriggerDownstreamStep in stream 2");
  ExpectExecutionStepTypeContains(GetState(), 2, 2, "WaitOnEPStep", "2nd step: WaitOnEPStep for node 3, for ActivateNotificationStep in stream 1");
  ExpectExecutionStepTypeContains(GetState(), 2, 3, "WaitOnEPStep", "3rd step: WaitOnEPStep for node 3, for ActivateNotificationStep in stream 2");
  ExpectExecutionStepTypeContains(GetState(), 2, 4, "LaunchKernelStep", "4th step: LaunchKernelStep for node 3");
}

// Test execution plan for the graph:
// stream 0: node1 (MemcpyToHost, CUDA EP) -> node3 (Transpose, CUDA EP)
// stream 1: node2 (CPU EP)
// node1's output, which is consumed by both node2 and node3, is in CPU.
TEST_F(PlannerTest, MultiStreamCudaEPNodeCPUOutput) {
  MemcpyToHostInCuda_TransposeInCudaAndCpu("./testdata/multi_stream_models/memcpyToHost_same_stream_with_transpose.json");
  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan.size(), 2) << "2 logic streams";
  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[0]->steps_.size(), 5) << "stream 0 has 5 steps";
  ExpectExecutionStepTypeContains(GetState(), 0, 0, "LaunchKernelStep", "0th step: LaunchKernelStep for node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 1, "ActivateNotificationStep", "1st step: ActivateNofiticationStep by node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 2, "TriggerDownstreamStep", "2nd step: TriggerDownstreamStep for node 3");
  ExpectExecutionStepTypeContains(GetState(), 0, 3, "WaitOnEPStep", "3rd step: WaitOnEPStep for node 3 in the same stream, as node 1's output is to CPU");
  ExpectExecutionStepTypeContains(GetState(), 0, 4, "LaunchKernelStep", "4th step: LaunchKernelStep for node 3");

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[1]->steps_.size(), 3) << "stream 1 has 3 steps";
  ExpectExecutionStepTypeContains(GetState(), 1, 0, "BarrierStep", "0th step: BarrierStep for node 2, for TriggerDownstreamStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 1, "WaitOnEPStep", "1st step: WaitOnEPStep for node 2, for ActivateNotificationStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 2, "LaunchKernelStep", "2nd step: LaunchKernelStep for node 2");
}

// Test execution plan for the graph:
// node1 has 2 outputs which are both consumed by node2, node1 and node2 are in different streams
// Only 1 WaitOnEPStep is expected before launching node2
// TODO(leca): there is a bug in the corresponding graph that node2 will be visited twice when traversing node1's output nodes
// (see: for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) in BuildExecutionPlan()). We can just break the loop and don't need the extra variables once it is fixed
TEST_F(PlannerTest, MultiStreamMultiOutput) {
  std::unique_ptr<::onnxruntime::KernelDef> cudaKernel = KernelDefBuilder().SetName("RNN").Provider(kCudaExecutionProvider).SinceVersion(7).Build();
  std::string Graph_input1("Graph_input1"), Graph_input2("Graph_input2"), Graph_input3("Graph_input3"), Arg1("Arg1"), Arg2("Arg2"), Arg3("Arg3"), node1("node1"), node2("node2");
  std::vector<onnxruntime::NodeArg*> input1{Arg(Graph_input1), Arg(Graph_input2), Arg(Graph_input3)}, output1{Arg(Arg1), Arg(Arg2)}, input2{Arg(Arg1), Arg(Arg2)}, output2{Arg(Arg3)};
  AddNode(*cudaKernel, node1, input1, output1);

  std::unique_ptr<::onnxruntime::KernelDef> cpuKernel = KernelDefBuilder().SetName("Add").Provider(kCpuExecutionProvider).SinceVersion(7, 12).Build();
  AddNode(*cpuKernel, node2, input2, output2);

  CUDAExecutionProviderInfo epi;
  onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA();
  auto epFactory = ep.CreateExecutionProviderFactory(epi);
  std::unique_ptr<IExecutionProvider> execution_provider = epFactory->CreateProvider();
  ORT_THROW_IF_ERROR(GetExecutionProviders().Add("CUDAExecutionProvider", std::move(execution_provider)));

  CreatePlan({}, false);

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan.size(), 2) << "2 logic streams";
  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[0]->steps_.size(), 3) << "stream 0 has 3 steps";
  ExpectExecutionStepTypeContains(GetState(), 0, 0, "LaunchKernelStep", "0th step: LaunchKernelStep for node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 1, "ActivateNotificationStep", "1st step: ActivateNofiticationStep by node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 2, "TriggerDownstreamStep", "2nd step: TriggerDownstreamStep for node 2");

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[1]->steps_.size(), 3) << "stream 1 has 3 steps";
  ExpectExecutionStepTypeContains(GetState(), 1, 0, "BarrierStep", "0th step: BarrierStep for node 2, for TriggerDownstreamStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 1, "WaitOnEPStep", "1st step: WaitOnEPStep for node 2, for ActivateNotificationStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 2, "LaunchKernelStep", "2nd step: LaunchKernelStep for node 2");
}

// Test execution plan for the graph:
// node1   node2
//   \       /
//    \     /
//      node3
// node1 and node2 are in the same stream, both has an output which will be consumed by node3 in a different stream
// TODO(leca): the ideal case is there is only 1 wait step before launching node3,
// as there is a specific order between node1 and node2 if they are in the same stream, thus node3 will only need to wait the latter one
TEST_F(PlannerTest, MultiStream2NodesSameStreamConsumedBy1NodeInDifferentStream) {
  std::unique_ptr<::onnxruntime::KernelDef> cudaKernel = KernelDefBuilder().SetName("Transpose").Provider(kCudaExecutionProvider).SinceVersion(1, 10).Build();
  std::string Graph_input1("Graph_input1"), Graph_input2("Graph_input2"), Graph_input3("Graph_input3"), Arg1("Arg1"), Arg2("Arg2"), Arg3("Arg3"), node1("node1"), node2("node2"), node3("node3");
  std::vector<onnxruntime::NodeArg*> input1{Arg(Graph_input1)}, input2{Arg(Graph_input2)}, output1{Arg(Arg1)}, output2{Arg(Arg2)}, input3{Arg(Arg1), Arg(Arg2)}, output3{Arg(Arg3)};
  AddNode(*cudaKernel, node1, input1, output1);
  AddNode(*cudaKernel, node2, input2, output2);

  std::unique_ptr<::onnxruntime::KernelDef> cpuKernel = KernelDefBuilder().SetName("Add").Provider(kCpuExecutionProvider).SinceVersion(7, 12).Build();
  AddNode(*cpuKernel, node3, input3, output3);

  CUDAExecutionProviderInfo epi;
  onnxruntime::ProviderInfo_CUDA& ep = onnxruntime::GetProviderInfo_CUDA();
  auto epFactory = ep.CreateExecutionProviderFactory(epi);
  std::unique_ptr<IExecutionProvider> execution_provider = epFactory->CreateProvider();
  ORT_THROW_IF_ERROR(GetExecutionProviders().Add("CUDAExecutionProvider", std::move(execution_provider)));

  CreatePlan({}, false);

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan.size(), 2) << "2 logic streams";
  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[0]->steps_.size(), 6) << "stream 0 has 6 steps";
  ExpectExecutionStepTypeContains(GetState(), 0, 0, "LaunchKernelStep", "0th step: LaunchKernelStep for node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 1, "ActivateNotificationStep", "1st step: ActivateNofiticationStep by node 1");
  ExpectExecutionStepTypeContains(GetState(), 0, 2, "TriggerDownstreamStep", "2nd step: TriggerDownstreamStep for node 3");
  ExpectExecutionStepTypeContains(GetState(), 0, 3, "LaunchKernelStep", "3rd step: LaunchKernelStep for node 2");
  ExpectExecutionStepTypeContains(GetState(), 0, 4, "ActivateNotificationStep", "4th step: ActivateNofiticationStep by node 2");
  ExpectExecutionStepTypeContains(GetState(), 0, 5, "TriggerDownstreamStep", "5th step: TriggerDownstreamStep for node 3");

  EXPECT_EQ(GetState().GetExecutionPlan()->execution_plan[1]->steps_.size(), 5) << "stream 1 has 5 steps";
  ExpectExecutionStepTypeContains(GetState(), 1, 0, "BarrierStep", "0th step: BarrierStep for node 1, for TriggerDownstreamStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 1, "BarrierStep", "1st step: BarrierStep for node 2, for TriggerDownstreamStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 2, "WaitOnEPStep", "2nd step: WaitOnEPStep for node 1, for ActivateNotificationStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 3, "WaitOnEPStep", "3rd step: WaitOnEPStep for node 2, for ActivateNotificationStep in stream 0");
  ExpectExecutionStepTypeContains(GetState(), 1, 4, "LaunchKernelStep", "4th step: LaunchKernelStep for node 3");
}
#endif

#if defined(USE_CUDA) || defined(USE_WEBGPU)
TEST_F(PlannerTest, LocationPlanningForInitializerAsIfSubgraphOutput) {
  constexpr int64_t kNumElements = 4;
  const std::vector<float> then_data{1.f, 2.f, 3.f, 4.f};
  const std::vector<float> else_data{-1.f, -2.f, -3.f, -4.f};

  TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(kNumElements);

  TypeProto bool_scalar;
  bool_scalar.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  bool_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_model = [&]() -> Model {
    auto create_if_subgraph = [&](const std::string& init_name, const std::vector<float>& data) -> GraphProto {
      Model model("if_branch_subgraph", true, DefaultLoggingManager().DefaultLogger());
      auto& graph = model.MainGraph();

      ONNX_NAMESPACE::TensorProto tensor;
      tensor.add_dims(kNumElements);
      for (float v : data) {
        tensor.add_float_data(v);
      }
      tensor.set_data_type(TensorProto_DataType_FLOAT);
      tensor.set_name(init_name);
      graph_utils::AddInitializerWithOrtValue(graph, tensor);

      auto& out = graph.GetOrCreateNodeArg(init_name, &float_tensor);
      graph.SetOutputs({&out});

      EXPECT_STATUS_OK(graph.Resolve());
      return graph.ToGraphProto();
    };

    onnxruntime::Model model("main_graph", false, ModelMetaData(),
                             PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                             {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
    auto& main_graph = model.MainGraph();

    auto& if_cond = main_graph.GetOrCreateNodeArg("if_cond", &bool_scalar);
    auto& if_out = main_graph.GetOrCreateNodeArg("if_out", &float_tensor);
    auto& node = main_graph.AddNode("if_node", "If", "If", {&if_cond}, {&if_out});
    node.AddAttribute("then_branch", create_if_subgraph("then_init", then_data));
    node.AddAttribute("else_branch", create_if_subgraph("else_init", else_data));

    main_graph.SetInputs({&if_cond});
    main_graph.SetOutputs({&if_out});

    EXPECT_STATUS_OK(main_graph.Resolve());
    return model;
  };

  SessionOptions so;
  InferenceSession sess{so, GetEnvironment()};
#if defined(USE_CUDA)
  ASSERT_STATUS_OK(sess.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
#else
  ASSERT_STATUS_OK(sess.RegisterExecutionProvider(DefaultWebGpuExecutionProvider()));
#endif

  std::string s1;
  ASSERT_TRUE(create_model().ToProto().SerializeToString(&s1));
  std::stringstream sstr(s1);
  ASSERT_STATUS_OK(sess.Load(sstr));
  ASSERT_STATUS_OK(sess.Initialize());

  const auto& main_graph_session_state = sess.GetSessionState();
  const Node* if_node = nullptr;
  for (const auto& n : main_graph_session_state.GetGraphViewer().Nodes()) {
    if (n.OpType() == "If") {
      if_node = &n;
    }
  }
  ASSERT_NE(if_node, nullptr);
  ASSERT_NE(if_node->GetExecutionProviderType(), kCpuExecutionProvider) << "If must run on the GPU EP for this test";

  auto check_branch = [&](const std::string& attr_name, const std::string& init_name) {
    const auto* subgraph_session_state = main_graph_session_state.GetSubgraphSessionState(if_node->Index(), attr_name);
    ASSERT_NE(subgraph_session_state, nullptr);

    OrtValueIndex init_index;
    ASSERT_STATUS_OK(subgraph_session_state->GetOrtValueNameIdxMap().GetIdx(init_name, init_index));
    EXPECT_EQ(subgraph_session_state->GetExecutionPlan()->allocation_plan[init_index].location.Type(),
              OrtDevice::GPU)
        << init_name << " should be planned on the If node's device";
  };
  check_branch("then_branch", "then_init");
  check_branch("else_branch", "else_init");

  // The output must still be correct for both branches.
  auto run_and_check = [&](bool cond, const std::vector<float>& expected) {
    OrtValue cond_value;
    CreateMLValue<bool>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], {1}, {cond}, &cond_value);
    NameMLValMap feeds{{"if_cond", cond_value}};
    std::vector<OrtValue> fetches;
    std::vector<std::string> output_names{"if_out"};
    ASSERT_STATUS_OK(sess.Run(RunOptions{}, feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    const auto& out = fetches[0].Get<Tensor>();
    ASSERT_EQ(out.Shape().Size(), kNumElements);
    EXPECT_EQ(std::vector<float>(out.Data<float>(), out.Data<float>() + kNumElements), expected);
  };
  run_and_check(true, then_data);
  run_and_check(false, else_data);
}
#endif

#if !defined(__wasm__) && defined(ORT_ENABLE_STREAM)
TEST_F(PlannerTest, ParaPlanCreation) {
  TypeProto graph_in_type;
  graph_in_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* graph_in_shape = graph_in_type.mutable_tensor_type()->mutable_shape();
  graph_in_shape->add_dim()->set_dim_value(3L);
  graph_in_shape->add_dim()->set_dim_value(3L);
  graph_in_shape->add_dim()->set_dim_value(300L);
  graph_in_shape->add_dim()->set_dim_value(300L);

  TypeProto relu_0_out_type, relu_1_out_type, relu_2_out_type;
  relu_0_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  relu_1_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  relu_2_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto maxpool_0_out_type;
  maxpool_0_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto conv_0_weight_type, conv_1_weight_type, conv_2_weight_type, conv_3_weight_type, conv_4_weight_type;

  conv_0_weight_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_0_weight_shape = conv_0_weight_type.mutable_tensor_type()->mutable_shape();
  conv_0_weight_shape->add_dim()->set_dim_value(64L);
  conv_0_weight_shape->add_dim()->set_dim_value(3L);
  conv_0_weight_shape->add_dim()->set_dim_value(7L);
  conv_0_weight_shape->add_dim()->set_dim_value(7L);

  conv_1_weight_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_1_weight_shape = conv_1_weight_type.mutable_tensor_type()->mutable_shape();
  conv_1_weight_shape->add_dim()->set_dim_value(64L);
  conv_1_weight_shape->add_dim()->set_dim_value(64L);
  conv_1_weight_shape->add_dim()->set_dim_value(1L);
  conv_1_weight_shape->add_dim()->set_dim_value(1L);

  conv_2_weight_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_2_weight_shape = conv_2_weight_type.mutable_tensor_type()->mutable_shape();
  conv_2_weight_shape->add_dim()->set_dim_value(64L);
  conv_2_weight_shape->add_dim()->set_dim_value(64L);
  conv_2_weight_shape->add_dim()->set_dim_value(3L);
  conv_2_weight_shape->add_dim()->set_dim_value(3L);

  conv_3_weight_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_3_weight_shape = conv_3_weight_type.mutable_tensor_type()->mutable_shape();
  conv_3_weight_shape->add_dim()->set_dim_value(256L);
  conv_3_weight_shape->add_dim()->set_dim_value(64L);
  conv_3_weight_shape->add_dim()->set_dim_value(1L);
  conv_3_weight_shape->add_dim()->set_dim_value(1L);

  conv_4_weight_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_4_weight_shape = conv_4_weight_type.mutable_tensor_type()->mutable_shape();
  conv_4_weight_shape->add_dim()->set_dim_value(256L);
  conv_4_weight_shape->add_dim()->set_dim_value(64L);
  conv_4_weight_shape->add_dim()->set_dim_value(1L);
  conv_4_weight_shape->add_dim()->set_dim_value(1L);

  TypeProto conv_0_bias_type, conv_1_bias_type, conv_2_bias_type, conv_3_bias_type, conv_4_bias_type;
  conv_0_bias_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_0_bias_shape = conv_0_bias_type.mutable_tensor_type()->mutable_shape();
  conv_0_bias_shape->add_dim()->set_dim_value(64L);

  conv_1_bias_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_1_bias_shape = conv_1_bias_type.mutable_tensor_type()->mutable_shape();
  conv_1_bias_shape->add_dim()->set_dim_value(64L);

  conv_2_bias_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_2_bias_shape = conv_2_bias_type.mutable_tensor_type()->mutable_shape();
  conv_2_bias_shape->add_dim()->set_dim_value(64L);

  conv_3_bias_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_3_bias_shape = conv_3_bias_type.mutable_tensor_type()->mutable_shape();
  conv_3_bias_shape->add_dim()->set_dim_value(256L);

  conv_4_bias_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  auto* conv_4_bias_shape = conv_4_bias_type.mutable_tensor_type()->mutable_shape();
  conv_4_bias_shape->add_dim()->set_dim_value(256L);

  TypeProto conv_0_out_type, conv_1_out_type, conv_2_out_type, conv_3_out_type, conv_4_out_type;
  conv_0_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  conv_1_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  conv_2_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  conv_3_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  conv_4_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto graph_out_type;
  graph_out_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  onnxruntime::Model model("main_graph", false, ModelMetaData(),
                           PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                           {{kOnnxDomain, 14}}, {}, DefaultLoggingManager().DefaultLogger());
  auto& main_graph = model.MainGraph();

  auto& graph_in = main_graph.GetOrCreateNodeArg("graph_in", &graph_in_type);

  auto& maxpool_0_out = main_graph.GetOrCreateNodeArg("maxpool_out", &maxpool_0_out_type);
  auto& relu_0_out = main_graph.GetOrCreateNodeArg("relu_0_out", &relu_0_out_type);
  auto& relu_1_out = main_graph.GetOrCreateNodeArg("relu_1_out", &relu_1_out_type);
  auto& relu_2_out = main_graph.GetOrCreateNodeArg("relu_2_out", &relu_2_out_type);

  ONNX_NAMESPACE::TensorProto conv_0_weight_tensor;
  conv_0_weight_tensor.add_dims(64L);
  conv_0_weight_tensor.add_dims(3L);
  conv_0_weight_tensor.add_dims(7L);
  conv_0_weight_tensor.add_dims(7L);
  for (int i = 0; i < 64 * 3 * 7 * 7; ++i) conv_0_weight_tensor.add_float_data(0.234f);
  conv_0_weight_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_0_weight_tensor.set_name("conv_0_weight");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_0_weight_tensor);

  ONNX_NAMESPACE::TensorProto conv_1_weight_tensor;
  conv_1_weight_tensor.add_dims(64L);
  conv_1_weight_tensor.add_dims(64L);
  conv_1_weight_tensor.add_dims(1L);
  conv_1_weight_tensor.add_dims(1L);
  conv_1_weight_tensor.set_data_type(TensorProto_DataType_FLOAT);
  for (int i = 0; i < 64 * 64; ++i) conv_1_weight_tensor.add_float_data(1.017f);
  conv_1_weight_tensor.set_name("conv_1_weight");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_1_weight_tensor);

  ONNX_NAMESPACE::TensorProto conv_2_weight_tensor;
  conv_2_weight_tensor.add_dims(64L);
  conv_2_weight_tensor.add_dims(64L);
  conv_2_weight_tensor.add_dims(3L);
  conv_2_weight_tensor.add_dims(3L);
  for (int i = 0; i < 64 * 64 * 3 * 3; ++i) conv_2_weight_tensor.add_float_data(2.317f);
  conv_2_weight_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_2_weight_tensor.set_name("conv_2_weight");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_2_weight_tensor);

  ONNX_NAMESPACE::TensorProto conv_3_weight_tensor;
  conv_3_weight_tensor.add_dims(256L);
  conv_3_weight_tensor.add_dims(64L);
  conv_3_weight_tensor.add_dims(1L);
  conv_3_weight_tensor.add_dims(1L);
  for (int i = 0; i < 256 * 64; ++i) conv_3_weight_tensor.add_float_data(1.256f);
  conv_3_weight_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_3_weight_tensor.set_name("conv_3_weight");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_3_weight_tensor);

  ONNX_NAMESPACE::TensorProto conv_4_weight_tensor;
  conv_4_weight_tensor.add_dims(256L);
  conv_4_weight_tensor.add_dims(64L);
  conv_4_weight_tensor.add_dims(1L);
  conv_4_weight_tensor.add_dims(1L);
  for (int i = 0; i < 256 * 64; ++i) conv_4_weight_tensor.add_float_data(1.913f);
  conv_4_weight_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_4_weight_tensor.set_name("conv_4_weight");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_4_weight_tensor);

  auto& conv_0_weight = main_graph.GetOrCreateNodeArg("conv_0_weight", &conv_0_weight_type);
  auto& conv_1_weight = main_graph.GetOrCreateNodeArg("conv_1_weight", &conv_1_weight_type);
  auto& conv_2_weight = main_graph.GetOrCreateNodeArg("conv_2_weight", &conv_2_weight_type);
  auto& conv_3_weight = main_graph.GetOrCreateNodeArg("conv_3_weight", &conv_3_weight_type);
  auto& conv_4_weight = main_graph.GetOrCreateNodeArg("conv_4_weight", &conv_4_weight_type);

  ONNX_NAMESPACE::TensorProto conv_0_bias_tensor;
  conv_0_bias_tensor.add_dims(64L);
  conv_0_bias_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_0_bias_tensor.set_name("conv_0_bias");
  for (int i = 0; i < 64; ++i) conv_0_bias_tensor.add_float_data(1.123f);
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_0_bias_tensor);

  ONNX_NAMESPACE::TensorProto conv_1_bias_tensor;
  conv_1_bias_tensor.add_dims(64L);
  for (int i = 0; i < 64; ++i) conv_1_bias_tensor.add_float_data(2.234f);
  conv_1_bias_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_1_bias_tensor.set_name("conv_1_bias");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_1_bias_tensor);

  ONNX_NAMESPACE::TensorProto conv_2_bias_tensor;
  conv_2_bias_tensor.add_dims(64L);
  for (int i = 0; i < 64; ++i) conv_2_bias_tensor.add_float_data(0.121f);
  conv_2_bias_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_2_bias_tensor.set_name("conv_2_bias");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_2_bias_tensor);

  ONNX_NAMESPACE::TensorProto conv_3_bias_tensor;
  conv_3_bias_tensor.add_dims(256L);
  for (int i = 0; i < 256; ++i) conv_3_bias_tensor.add_float_data(1.201f);
  conv_3_bias_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_3_bias_tensor.set_name("conv_3_bias");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_3_bias_tensor);

  ONNX_NAMESPACE::TensorProto conv_4_bias_tensor;
  conv_4_bias_tensor.add_dims(256L);
  for (int i = 0; i < 256; ++i) conv_4_bias_tensor.add_float_data(0.897f);
  conv_4_bias_tensor.set_data_type(TensorProto_DataType_FLOAT);
  conv_4_bias_tensor.set_name("conv_4_bias");
  graph_utils::AddInitializerWithOrtValue(main_graph, conv_4_bias_tensor);

  auto& conv_0_bias = main_graph.GetOrCreateNodeArg("conv_0_bias", &conv_0_bias_type);
  auto& conv_1_bias = main_graph.GetOrCreateNodeArg("conv_1_bias", &conv_1_bias_type);
  auto& conv_2_bias = main_graph.GetOrCreateNodeArg("conv_2_bias", &conv_2_bias_type);
  auto& conv_3_bias = main_graph.GetOrCreateNodeArg("conv_3_bias", &conv_3_bias_type);
  auto& conv_4_bias = main_graph.GetOrCreateNodeArg("conv_4_bias", &conv_4_bias_type);

  auto& conv_0_out = main_graph.GetOrCreateNodeArg("conv_0_out", &conv_0_out_type);
  auto& conv_1_out = main_graph.GetOrCreateNodeArg("conv_1_out", &conv_1_out_type);
  auto& conv_2_out = main_graph.GetOrCreateNodeArg("conv_2_out", &conv_2_out_type);
  auto& conv_3_out = main_graph.GetOrCreateNodeArg("conv_3_out", &conv_3_out_type);
  auto& conv_4_out = main_graph.GetOrCreateNodeArg("conv_4_out", &conv_4_out_type);

  auto& graph_out = main_graph.GetOrCreateNodeArg("graph_out", &graph_out_type);

  NodeAttributes conv_0_attributes;

  ONNX_NAMESPACE::AttributeProto dilation;
  dilation.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  dilation.add_ints(1);
  dilation.add_ints(1);
  dilation.set_name("dilations");
  conv_0_attributes["dilations"] = dilation;

  ONNX_NAMESPACE::AttributeProto group;
  group.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  group.set_i(1);
  group.set_name("group");
  conv_0_attributes["group"] = group;

  ONNX_NAMESPACE::AttributeProto conv_0_kernel_shape;
  conv_0_kernel_shape.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_0_kernel_shape.add_ints(7);
  conv_0_kernel_shape.add_ints(7);
  conv_0_kernel_shape.set_name("kernel_shape");
  conv_0_attributes["kernel_shape"] = conv_0_kernel_shape;

  ONNX_NAMESPACE::AttributeProto conv_0_pads;
  conv_0_pads.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_0_pads.add_ints(3);
  conv_0_pads.add_ints(3);
  conv_0_pads.add_ints(3);
  conv_0_pads.add_ints(3);
  conv_0_pads.set_name("pads");
  conv_0_attributes["pads"] = conv_0_pads;

  ONNX_NAMESPACE::AttributeProto conv_0_strides;
  conv_0_strides.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_0_strides.add_ints(2);
  conv_0_strides.add_ints(2);
  conv_0_strides.set_name("strides");
  conv_0_attributes["strides"] = conv_0_strides;

  main_graph.AddNode("conv_0", "Conv", "", {&graph_in, &conv_0_weight, &conv_0_bias}, {&conv_0_out}, &conv_0_attributes);
  main_graph.AddNode("relu_0", "Relu", "", {&conv_0_out}, {&relu_0_out});

  NodeAttributes maxpool_0_attributes;
  ONNX_NAMESPACE::AttributeProto ceil_mode;
  ceil_mode.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  ceil_mode.set_i(0);
  ceil_mode.set_name("ceil_mode");
  maxpool_0_attributes["ceil_mode"] = ceil_mode;

  ONNX_NAMESPACE::AttributeProto maxpool_0_kernel_shape;
  maxpool_0_kernel_shape.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  maxpool_0_kernel_shape.add_ints(3);
  maxpool_0_kernel_shape.add_ints(3);
  maxpool_0_kernel_shape.set_name("kernel_shape");
  maxpool_0_attributes["kernel_shape"] = maxpool_0_kernel_shape;

  ONNX_NAMESPACE::AttributeProto maxpool_0_pads;
  maxpool_0_pads.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  maxpool_0_pads.add_ints(1);
  maxpool_0_pads.add_ints(1);
  maxpool_0_pads.add_ints(1);
  maxpool_0_pads.add_ints(1);
  maxpool_0_pads.set_name("pads");
  maxpool_0_attributes["pads"] = maxpool_0_pads;

  ONNX_NAMESPACE::AttributeProto maxpool_0_strides;
  maxpool_0_strides.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  maxpool_0_strides.add_ints(1);
  maxpool_0_strides.add_ints(1);
  maxpool_0_strides.set_name("strides");
  maxpool_0_attributes["strides"] = maxpool_0_strides;

  main_graph.AddNode("maxpool_0", "MaxPool", "", {&relu_0_out}, {&maxpool_0_out}, &maxpool_0_attributes);

  NodeAttributes conv_1_attributes;
  conv_1_attributes["dilations"] = dilation;
  conv_1_attributes["group"] = group;

  ONNX_NAMESPACE::AttributeProto conv_1_kernel_shape;
  conv_1_kernel_shape.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_1_kernel_shape.add_ints(1);
  conv_1_kernel_shape.add_ints(1);
  conv_1_kernel_shape.set_name("kernel_shape");
  conv_1_attributes["kernel_shape"] = conv_1_kernel_shape;

  ONNX_NAMESPACE::AttributeProto conv_1_pads;
  conv_1_pads.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_1_pads.add_ints(0);
  conv_1_pads.add_ints(0);
  conv_1_pads.add_ints(0);
  conv_1_pads.add_ints(0);
  conv_1_pads.set_name("pads");
  conv_1_attributes["pads"] = conv_1_pads;

  ONNX_NAMESPACE::AttributeProto conv_1_strides;
  conv_1_strides.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_1_strides.add_ints(1);
  conv_1_strides.add_ints(1);
  conv_1_strides.set_name("strides");
  conv_1_attributes["strides"] = conv_1_strides;

  main_graph.AddNode("conv_1", "Conv", "", {&maxpool_0_out, &conv_1_weight, &conv_1_bias}, {&conv_1_out}, &conv_1_attributes);
  main_graph.AddNode("relu_1", "Relu", "", {&conv_1_out}, {&relu_1_out});

  NodeAttributes conv_2_attributes;
  conv_2_attributes["dilations"] = dilation;
  conv_2_attributes["group"] = group;

  ONNX_NAMESPACE::AttributeProto conv_2_kernel_shape;
  conv_2_kernel_shape.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_2_kernel_shape.add_ints(3);
  conv_2_kernel_shape.add_ints(3);
  conv_2_kernel_shape.set_name("kernel_shape");
  conv_2_attributes["kernel_shape"] = conv_2_kernel_shape;

  ONNX_NAMESPACE::AttributeProto conv_2_pads;
  conv_2_pads.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_2_pads.add_ints(1);
  conv_2_pads.add_ints(1);
  conv_2_pads.add_ints(1);
  conv_2_pads.add_ints(1);
  conv_2_pads.set_name("pads");
  conv_2_attributes["pads"] = conv_2_pads;

  ONNX_NAMESPACE::AttributeProto conv_2_strides;
  conv_2_strides.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_2_strides.add_ints(1);
  conv_2_strides.add_ints(1);
  conv_2_strides.set_name("strides");
  conv_2_attributes["strides"] = conv_2_strides;

  main_graph.AddNode("conv_2", "Conv", "", {&relu_1_out, &conv_2_weight, &conv_2_bias}, {&conv_2_out}, &conv_2_attributes);
  main_graph.AddNode("relu_2", "Relu", "", {&conv_2_out}, {&relu_2_out});

  NodeAttributes conv_3_attributes;
  conv_3_attributes["dilations"] = dilation;
  conv_3_attributes["group"] = group;

  ONNX_NAMESPACE::AttributeProto conv_3_kernel_shape;
  conv_3_kernel_shape.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_3_kernel_shape.add_ints(1);
  conv_3_kernel_shape.add_ints(1);
  conv_3_kernel_shape.set_name("kernel_shape");
  conv_3_attributes["kernel_shape"] = conv_3_kernel_shape;

  ONNX_NAMESPACE::AttributeProto conv_3_pads;
  conv_3_pads.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_3_pads.add_ints(0);
  conv_3_pads.add_ints(0);
  conv_3_pads.add_ints(0);
  conv_3_pads.add_ints(0);
  conv_3_pads.set_name("pads");
  conv_3_attributes["pads"] = conv_3_pads;

  ONNX_NAMESPACE::AttributeProto conv_3_strides;
  conv_3_strides.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_3_strides.add_ints(1);
  conv_3_strides.add_ints(1);
  conv_3_strides.set_name("strides");
  conv_3_attributes["strides"] = conv_3_strides;

  main_graph.AddNode("conv_3", "Conv", "", {&relu_2_out, &conv_3_weight, &conv_3_bias}, {&conv_3_out}, &conv_3_attributes);

  NodeAttributes conv_4_attributes;
  conv_4_attributes["dilations"] = dilation;
  conv_4_attributes["group"] = group;

  ONNX_NAMESPACE::AttributeProto conv_4_kernel_shape;
  conv_4_kernel_shape.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_4_kernel_shape.add_ints(1);
  conv_4_kernel_shape.add_ints(1);
  conv_4_kernel_shape.set_name("kernel_shape");
  conv_4_attributes["kernel_shape"] = conv_4_kernel_shape;

  ONNX_NAMESPACE::AttributeProto conv_4_pads;
  conv_4_pads.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_4_pads.add_ints(0);
  conv_4_pads.add_ints(0);
  conv_4_pads.add_ints(0);
  conv_4_pads.add_ints(0);
  conv_4_pads.set_name("pads");
  conv_4_attributes["pads"] = conv_4_pads;

  ONNX_NAMESPACE::AttributeProto conv_4_strides;
  conv_4_strides.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  conv_4_strides.add_ints(1);
  conv_4_strides.add_ints(1);
  conv_4_strides.set_name("strides");
  conv_4_attributes["strides"] = conv_4_strides;

  main_graph.AddNode("conv_4", "Conv", "", {&maxpool_0_out, &conv_4_weight, &conv_4_bias}, {&conv_4_out}, &conv_4_attributes);
  main_graph.AddNode("add_0", "Add", "", {&conv_3_out, &conv_4_out}, {&graph_out});

  main_graph.SetInputs({&graph_in});
  main_graph.SetOutputs({&graph_out});

  auto status = main_graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Default;
  ASSERT_TRUE(so.config_options.AddConfigEntry(kNodePartitionConfigFile,
                                               "./testdata/multi_stream_models/simplified_ssd_cpu.json")
                  .IsOK());
  InferenceSession sess{so, GetEnvironment()};

  status = sess.RegisterExecutionProvider(DefaultCpuExecutionProvider());
  ASSERT_TRUE(status.IsOK());
  ASSERT_TRUE(model.Save(model, ORT_TSTR("./simplified_ssd.onnx")).IsOK());

  std::string s1;
  const bool rc = model.ToProto().SerializeToString(&s1);
  EXPECT_EQ(rc, true);
  std::stringstream sstr(s1);

  status = sess.Load(sstr);
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  const auto& main_graph_session_state = sess.GetSessionState();
  const auto& main_graph_ort_value_index_map = main_graph_session_state.GetOrtValueNameIdxMap();
  auto* exe_plan = const_cast<onnxruntime::SessionState&>(main_graph_session_state).GetExecutionPlan();
  auto& per_value_plans = exe_plan->GetAllocationPlan();
  InlinedHashMap<std::string, std::string> reuse_pairs;
  reuse_pairs.emplace("conv_0_out", "relu_0_out");  // conv_0_out is reused by relu_0_out
  reuse_pairs.emplace("conv_1_out", "relu_1_out");  // conv_1_out is reused by relu_1_out
  reuse_pairs.emplace("conv_2_out", "relu_2_out");  // conv_2_out is reused by relu_2_out
  for (size_t i = 0; i < per_value_plans.size(); ++i) {
    auto& per_value_plan = per_value_plans[i];
    if (per_value_plan.alloc_kind == AllocKind::kReuse) {
      std::string reused;
      ORT_ENFORCE(main_graph_ort_value_index_map.GetName(per_value_plan.reused_buffer, reused).IsOK());
      reuse_pairs.erase(reused);
    }  // if
  }  // for
  ASSERT_TRUE(reuse_pairs.empty());
}

TEST_F(PlannerTest, TestMultiStreamConfig) {
  const char* type = "DeviceBasedPartitioner";
  constexpr size_t type_len = 22;

  auto graph_partitioner_cpu = IGraphPartitioner::CreateGraphPartitioner(
      DefaultLoggingManager().DefaultLogger(),
      ORT_TSTR("./testdata/multi_stream_models/multi_stream_single_stream.json"));

  ASSERT_TRUE(graph_partitioner_cpu &&
              strncmp(graph_partitioner_cpu->Type(), type, type_len) == 0 &&
              graph_partitioner_cpu->Streams() == 1);

  auto graph_partitioner_cpu_gpu = IGraphPartitioner::CreateGraphPartitioner(
      DefaultLoggingManager().DefaultLogger(),
      ORT_TSTR("./testdata/multi_stream_models/multi_stream_double_stream.json"));

  ASSERT_TRUE(graph_partitioner_cpu_gpu &&
              strncmp(graph_partitioner_cpu_gpu->Type(), type, type_len) == 0 &&
              graph_partitioner_cpu_gpu->Streams() == 2);
}

TEST_F(PlannerTest, InvalidMultiStreamConfigUsesDefaultPartition) {
  const PathString config_path = ORT_TSTR("./test_invalid_partition_config.json");
  ScopedFileDeleter cleanup{config_path};
  const std::array<const char*, 13> invalid_configs = {
      "not-json",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]]})",
      R"({"type":"DeviceBasedPartitioner","streams":1,"devices":["0"]})",
      R"({"type":"DeviceBasedPartitioner","streams":["node1"],"devices":["0"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[[1]],"devices":["0"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":[0]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":["bad"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":["0","1"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"],["node2"]],"devices":["0","bad"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":["999999999999999999999"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":["-1"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":["5"]})",
      R"({"type":"DeviceBasedPartitioner","streams":[["node1"]],"devices":["256"]})",
  };

  for (const char* config : invalid_configs) {
    {
      std::ofstream file{std::filesystem::path(config_path)};
      ASSERT_TRUE(file.is_open());
      file << config;
    }
    auto partitioner = IGraphPartitioner::CreateGraphPartitioner(DefaultLoggingManager().DefaultLogger(), config_path);
    ASSERT_NE(partitioner, nullptr);
    EXPECT_EQ(partitioner->Streams(), 0u) << config;
  }

  {
    std::ofstream file{std::filesystem::path(config_path)};
    ASSERT_TRUE(file.is_open());
    file << R"({"type":"DeviceBasedPartitioner"})";
  }
  auto partitioner = IGraphPartitioner::CreateGraphPartitioner(DefaultLoggingManager().DefaultLogger(), config_path);
  ASSERT_NE(partitioner, nullptr);
  EXPECT_EQ(partitioner->Streams(), 0u);
}

// Save partition config to a file and check its completeness
TEST_F(PlannerTest, TestMultiStreamSaveConfig) {
  const char* config_file_path = "./testdata/multi_stream_models/conv_add_relu_single_stream.json";
  {
    SessionOptions sess_opt;
    sess_opt.graph_optimization_level = TransformerLevel::Default;
    ASSERT_TRUE(sess_opt.config_options.AddConfigEntry(kNodePartitionConfigFile,
                                                       config_file_path)
                    .IsOK());

    InferenceSession sess(sess_opt, GetEnvironment(), ORT_TSTR("./testdata/multi_stream_models/conv_add_relu.onnx"));
    auto status = sess.RegisterExecutionProvider(DefaultCpuExecutionProvider());
    ASSERT_TRUE(status.IsOK());

    status = sess.Load();
    ASSERT_TRUE(status.IsOK());

    status = sess.Initialize();
    ASSERT_TRUE(status.IsOK());
  }

  std::ifstream if_stream(config_file_path);
  ASSERT_TRUE(if_stream.is_open());
  std::set<std::string> node_set{"model_41/conv2d_34/Conv2D__2321",
                                 "model_41/conv2d_34/Conv2D",
                                 "model_41/lambda_9/add",
                                 "model_41/activation_27/Relu",
                                 "Transpose__2331"};

  try {
    json json_config = json::parse(if_stream);
    ASSERT_TRUE(json_config["type"] == "DeviceBasedPartitioner");

    for (const auto& node_stream : json_config["streams"]) {
      ASSERT_TRUE(node_stream.is_array());

      for (const auto& node_name : node_stream) {
        ASSERT_TRUE(node_name.is_string());
        auto iter = node_set.find(node_name);

        ASSERT_TRUE(iter != node_set.end());
        node_set.erase(iter);
      }
    }
  } catch (...) {
    ASSERT_TRUE(false);
  }
  if_stream.close();
  ASSERT_TRUE(node_set.empty());
}

// Load with partition config where a node is missing, session load expected to fail.
TEST_F(PlannerTest, TestMultiStreamMissingNodeConfig) {
  const char* config_file_path = "./testdata/multi_stream_models/conv_add_relu_single_stream_missing_node.json";
  SessionOptions sess_opt;
  sess_opt.graph_optimization_level = TransformerLevel::Default;
  ASSERT_TRUE(sess_opt.config_options.AddConfigEntry(kNodePartitionConfigFile,
                                                     config_file_path)
                  .IsOK());

  InferenceSession sess(sess_opt, GetEnvironment(), ORT_TSTR("./testdata/multi_stream_models/conv_add_relu.onnx"));
  auto status = sess.RegisterExecutionProvider(DefaultCpuExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  status = sess.Load();
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(!status.IsOK());
}

// A mismatched stream/device count falls back to default partitioning.
TEST_F(PlannerTest, TestMultiStreamMismatchDeviceFallsBackToDefault) {
  const char* config_file_path = "./testdata/multi_stream_models/conv_add_relu_single_stream_mismatch_device.json";
  SessionOptions sess_opt;
  sess_opt.graph_optimization_level = TransformerLevel::Default;
  ASSERT_TRUE(sess_opt.config_options.AddConfigEntry(kNodePartitionConfigFile,
                                                     config_file_path)
                  .IsOK());

  InferenceSession sess(sess_opt, GetEnvironment(), ORT_TSTR("./testdata/multi_stream_models/conv_add_relu.onnx"));
  auto status = sess.RegisterExecutionProvider(DefaultCpuExecutionProvider());
  ASSERT_TRUE(status.IsOK());

  status = sess.Load();
  ASSERT_TRUE(status.IsOK());

  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());
}
#endif

#if defined(USE_CUDA) && defined(ORT_ENABLE_STREAM)
TEST_F(PlannerTest, TestCpuIf) {
  SessionOptions sess_opt;
  sess_opt.graph_optimization_level = TransformerLevel::Default;

  InferenceSession sess(sess_opt, GetEnvironment(), ORT_TSTR("./testdata/multi_stream_models/cpu_if.onnx"));
  ASSERT_STATUS_OK(sess.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
  ASSERT_STATUS_OK(sess.Load());
  ASSERT_STATUS_OK(sess.Initialize());

  auto& sess_state = const_cast<onnxruntime::SessionState&>(sess.GetSessionState());
  const auto& exe_plan = sess_state.GetExecutionPlan()->execution_plan;
  if (exe_plan.size() == 2 &&
      exe_plan[1]->device_.Type() == OrtDevice::CPU &&
      exe_plan[1]->steps_.size() == 9 &&
      exe_plan[1]->steps_[7]->GetNodeIndex() == 7) {
    // there must be a wait before cpu If node
    static const std::string WaitOnEPStep = "WaitOnEPStep";
    ASSERT_EQ(exe_plan[1]->steps_[6]->ToString().substr(0, WaitOnEPStep.size()), WaitOnEPStep);
  }
}

// model looks like:
//                                                 |-----------> Gather
//                                                 |-----------> Gather
//                                                 |-----------> Gather
//                                                 |-----------> Gather
// Shape ----------------> Reshape --> Shape ------------------> Reshape
//                           ^                                     ^
// InstanceNormalization ----|         InstanceNormalization ------|
//
// Python script to create this model:
// def CreateModelFor19480():
//    #shape->reshape->shape->reshape, 4 gather
//    graphNodes = []
//    graphNodes.append(h.make_node('Shape', inputs=['shape_input'], outputs=['9']))
//    graphNodes.append(h.make_node('InstanceNormalization', inputs=['in0_input', 'scale0', 'B0'], outputs=['8']))
//    graphNodes.append(h.make_node('Reshape', inputs=['8', '9'], outputs=['Reshape15_output']))
//    graphNodes.append(h.make_node('Shape', inputs=['Reshape15_output'], outputs=['281']))
//    graphNodes.append(h.make_node('InstanceNormalization', inputs=['in1_input', 'scale1', 'B1'], outputs=['293']))
//    graphNodes.append(h.make_node('Reshape', inputs=['293', '281'], outputs=['output0']))
//    graphNodes.append(h.make_node('Gather', inputs=['281', 'indices1'], outputs=['output1']))
//    graphNodes.append(h.make_node('Gather', inputs=['281', 'indices2'], outputs=['output2']))
//    graphNodes.append(h.make_node('Gather', inputs=['281', 'indices3'], outputs=['output3']))
//    graphNodes.append(h.make_node('Gather', inputs=['281', 'indices4'], outputs=['output4']))
//    g = h.make_graph(graphNodes, 'issue_19480',
//                     [h.make_tensor_value_info('shape_input', tp.FLOAT, ['batch', 128, None, None]),
//                      h.make_tensor_value_info('in0_input', tp.FLOAT, ['batch', 32, None]),
//                      h.make_tensor_value_info('scale0', tp.FLOAT, [32]),
//                      h.make_tensor_value_info('B0', tp.FLOAT, [32]),
//                      h.make_tensor_value_info('in1_input', tp.FLOAT, ['batch', 32, None]),
//                      h.make_tensor_value_info('scale1', tp.FLOAT, [32]),
//                      h.make_tensor_value_info('B1', tp.FLOAT, [32]),
//                      h.make_tensor_value_info('indices1', tp.INT32, []),
//                      h.make_tensor_value_info('indices2', tp.INT32, []),
//                      h.make_tensor_value_info('indices3', tp.INT32, []),
//                      h.make_tensor_value_info('indices4', tp.INT32, [])],
//                     [h.make_tensor_value_info('output0', tp.FLOAT, None),
//                      h.make_tensor_value_info('output1', tp.INT64, None),
//                      h.make_tensor_value_info('output2', tp.INT64, None),
//                      h.make_tensor_value_info('output3', tp.INT64, None),
//                      h.make_tensor_value_info('output4', tp.INT64, None)])
//    model = h.make_model(g, opset_imports=[h.make_operatorsetid("", 17)], producer_name='producer_name')
//    onnx.save(model, 'issue_19480.onnx')
//
TEST(AllocationPlannerTest, ReusedInputCrossDifferentStreams) {
  SessionOptions sess_opt;
  sess_opt.graph_optimization_level = TransformerLevel::Default;

  InferenceSession sess(sess_opt, GetEnvironment(), ORT_TSTR("./testdata/multi_stream_models/issue_19480.onnx"));
  auto status = sess.RegisterExecutionProvider(DefaultCudaExecutionProvider());
  status = sess.Load();
  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK()) << "No crash";
  const SequentialExecutionPlan* plan = sess.GetSessionState().GetExecutionPlan();
  ASSERT_EQ(plan->allocation_plan[14].alloc_kind, AllocKind::kReuse) << "The input of reshape and gather will reuse the output of shape";

  int gather_count = 0;
  ASSERT_GT(plan->execution_plan.size(), 1) << "Number of execution plans should be greater than 1";
  for (size_t i = 0; i < plan->execution_plan[1]->steps_.size(); i++) {
    const auto* step = plan->execution_plan[1]->steps_[i].get();
    if (strstr(typeid(*step).name(), "LaunchKernelStep")) {
      const Node* node = sess.GetSessionState().GetGraphViewer().GetNode(plan->execution_plan[1]->steps_[i]->GetNodeIndex());
      if (node->OpType() == "Gather")
        gather_count++;
      else
        FAIL() << "CPU stream should contain only gather ops";
    }
  }
  ASSERT_EQ(gather_count, 4) << "4 gather ops are all placed in CPU stream";
}
#endif

#ifdef ENABLE_TRAINING_OPS
// use a carefully constructed model to re-produce a customer reported issue where a model produced invalid output.
// this issue required:
// - buffer A that is re-used later in the model
//   - output of the first Shape node
//   - first usage completes after the following Cast node
// - buffer B which has the same size requirement and is used after the first usage of A is complete
//   - buffer B is used for the output from `squeeze2` and a number of other nodes in that part of the model.
// - re-use of buffer A for an output of a node that has no consumers whilst buffer B is still in use
//   - this is the `per_input_length` output of the ConcatTraining node
//
// Because the logic to determine when a buffer can be freed is based on consumers, buffer A gets freed after the
// Cast node. It is then re-used as buffer B because the memory pattern planner believes that block to be available.
// When we re-use buffer A for the ConcatTraining output we are using the same address for two different node output
// buffers, leading to corruption of the output.
// This tests that the change in allocation planner to not re-use a buffer for outputs with no consumers prevents this.
TEST(AllocationPlannerTest, AvoidReuseOfBufferForNodeOutputWithNoConsumers) {
  SessionOptions sess_opt;
  sess_opt.graph_optimization_level = TransformerLevel::Default;

  InferenceSession sess(sess_opt, GetEnvironment(), ORT_TSTR("./testdata/avoid_reuse_of_buffer_for_node_output_with_no_consumers.onnx"));
  auto status = sess.Load();
  status = sess.Initialize();
  ASSERT_TRUE(status.IsOK());

  const auto& session_state = sess.GetSessionState();
  const auto& ort_value_index_map = session_state.GetOrtValueNameIdxMap();
  const SequentialExecutionPlan* plan = session_state.GetExecutionPlan();

  OrtValueIndex concat_training_unused_out_index;
  // Here per_input_length output of the ConcatTraining node has no consumers, so it should not reuse the buffer.
  ASSERT_STATUS_OK(ort_value_index_map.GetIdx("per_input_length", concat_training_unused_out_index));
  EXPECT_EQ(plan->allocation_plan[concat_training_unused_out_index].alloc_kind, AllocKind::kAllocate);
}
#endif

// Regression test for a heap buffer overflow caused by reusing a packed sub-byte buffer for a full-byte tensor.
//
// A packed sub-byte tensor (uint4[N]) needs ceil(N/2) storage bytes, while a full-byte tensor (uint8[N]) of the
// same logical shape needs N bytes. Both types have a one-byte C++ carrier size, so the allocation planner's
// SameSize() check used to treat them as the same size (carrier size 1 == 1 and identical shape) and let the
// uint8 output reuse the smaller uint4 buffer. Writing the uint8 tensor into that half-sized buffer then
// overflows it (CWE-131 -> CWE-787). The uint8 output must not reuse the uint4 buffer.
//
// Graph (opset 21):  X(float) -Cast-> A(uint4) -Cast-> B(float) -Cast-> C(uint8) -Cast-> Y(float)
// A is fully consumed by the second Cast and freed, so before the fix the planner reused A's 512-byte buffer for
// the 1023-byte uint8 tensor C.
//
// The dimension is deliberately odd so the ceiling division in the packed storage size (ceil(1023 / 2) = 512, not
// 1023 / 2 = 511) is covered as well.
TEST(AllocationPlannerTest, AvoidReuseOfPackedSubByteBufferForFullByteTensor) {
  constexpr int64_t kDim = 1023;

  auto make_tensor_type = [](TensorProto_DataType elem_type) {
    TypeProto t;
    t.mutable_tensor_type()->set_elem_type(elem_type);
    t.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(kDim);
    return t;
  };

  auto create_model = [&](TensorProto_DataType source_elem_type, TensorProto_DataType dest_elem_type) -> Model {
    Model model("packed_subbyte_reuse", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 21}}, {},
                DefaultLoggingManager().DefaultLogger());
    Graph& graph = model.MainGraph();

    TypeProto float_type = make_tensor_type(TensorProto_DataType_FLOAT);
    TypeProto source_type = make_tensor_type(source_elem_type);
    TypeProto dest_type = make_tensor_type(dest_elem_type);

    auto& X = graph.GetOrCreateNodeArg("X", &float_type);
    auto& A = graph.GetOrCreateNodeArg("A", &source_type);
    auto& B = graph.GetOrCreateNodeArg("B", &float_type);  // consumes A -> A becomes dead here
    auto& C = graph.GetOrCreateNodeArg("C", &dest_type);
    auto& Y = graph.GetOrCreateNodeArg("Y", &float_type);

    auto add_cast = [&graph](const std::string& name, NodeArg& in, NodeArg& out, TensorProto_DataType to) {
      auto& node = graph.AddNode(name, "Cast", name, {&in}, {&out});
      node.AddAttribute("to", static_cast<int64_t>(to));
    };
    add_cast("cast_to_source", X, A, source_elem_type);
    add_cast("cast_a_to_float", A, B, TensorProto_DataType_FLOAT);
    add_cast("cast_to_dest", B, C, dest_elem_type);
    add_cast("cast_c_to_float", C, Y, TensorProto_DataType_FLOAT);

    graph.SetInputs({&X});
    graph.SetOutputs({&Y});
    EXPECT_STATUS_OK(graph.Resolve());
    return model;
  };

  struct StorageCase {
    TensorProto_DataType source;
    TensorProto_DataType destination;
    bool reuse;
  };
  // Exercise both packing-density rejection and compatible packed/dense reuse, plus string exclusion.
  for (const auto& storage_case : {
           StorageCase{TensorProto_DataType_UINT4, TensorProto_DataType_UINT8, false},
           StorageCase{TensorProto_DataType_UINT8, TensorProto_DataType_UINT4, false},
           StorageCase{TensorProto_DataType_UINT4, TensorProto_DataType_UINT4, true},
           StorageCase{TensorProto_DataType_UINT4, TensorProto_DataType_INT4, true},
           StorageCase{TensorProto_DataType_UINT8, TensorProto_DataType_INT8, true},
           StorageCase{TensorProto_DataType_STRING, TensorProto_DataType_STRING, false},
       }) {
    SCOPED_TRACE(::testing::Message() << "source=" << storage_case.source
                                      << ", destination=" << storage_case.destination);
    SessionOptions so;
    // Keep memory reuse on (default) and avoid graph optimizations that could rewrite the Cast chain.
    so.graph_optimization_level = TransformerLevel::Default;
    InferenceSession sess{so, GetEnvironment()};

    std::string serialized;
    ASSERT_TRUE(create_model(storage_case.source, storage_case.destination).ToProto().SerializeToString(&serialized));
    std::stringstream sstr(serialized);
    ASSERT_STATUS_OK(sess.Load(sstr));
    ASSERT_STATUS_OK(sess.Initialize());

    const auto& session_state = sess.GetSessionState();
    const auto& ort_value_index_map = session_state.GetOrtValueNameIdxMap();
    const SequentialExecutionPlan* plan = session_state.GetExecutionPlan();

    OrtValueIndex a_index, c_index;
    ASSERT_STATUS_OK(ort_value_index_map.GetIdx("A", a_index));
    ASSERT_STATUS_OK(ort_value_index_map.GetIdx("C", c_index));

    const auto& c_plan = plan->allocation_plan[c_index];
    const bool reuses_a = (c_plan.alloc_kind == AllocKind::kReuse) && (c_plan.reused_buffer == a_index);
    EXPECT_EQ(reuses_a, storage_case.reuse);

    // The model must still run and produce correct results after round-tripping through both storage types.
    constexpr float kValue = 3.0f;
    std::vector<int64_t> dims{kDim};
    std::vector<float> input_data(static_cast<size_t>(kDim), kValue);
    OrtValue input_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims, input_data, &input_value);

    NameMLValMap feeds{{"X", input_value}};
    std::vector<std::string> output_names{"Y"};
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(sess.Run(feeds, output_names, &fetches));

    ASSERT_EQ(fetches.size(), 1u);
    const Tensor& out = fetches[0].Get<Tensor>();
    ASSERT_EQ(out.Shape().Size(), kDim);
    const float* out_data = out.Data<float>();
    for (int64_t i = 0; i < kDim; ++i) {
      ASSERT_EQ(out_data[i], kValue) << "mismatch at index " << i;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

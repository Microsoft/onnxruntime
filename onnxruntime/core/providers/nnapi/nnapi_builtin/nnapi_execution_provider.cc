// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"

#include "core/framework/allocatormgr.h"
#include "core/framework/compute_capability.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

#ifdef __ANDROID__
#include "core/providers/nnapi/nnapi_builtin/builders/model_builder.h"
#include "core/providers/nnapi/nnapi_builtin/model.h"
#endif

namespace onnxruntime {

constexpr const char* NNAPI = "Nnapi";

constexpr std::array kDefaultPartitioningStopOps{
    "NonMaxSuppression",
};

NnapiExecutionProvider::NnapiExecutionProvider(uint32_t nnapi_flags)
    : IExecutionProvider{onnxruntime::kNnapiExecutionProvider, true},
      nnapi_flags_(nnapi_flags),
      // TODO make this configurable
      partitioning_stop_ops_(kDefaultPartitioningStopOps.begin(), kDefaultPartitioningStopOps.end()) {
  AllocatorCreationInfo device_info(
      [](int) {
        return std::make_unique<CPUAllocator>(OrtMemoryInfo(NNAPI, OrtAllocatorType::OrtDeviceAllocator));
      });

  InsertAllocator(CreateAllocator(device_info));

  AllocatorCreationInfo cpu_memory_info(
      [](int) {
        return std::make_unique<CPUAllocator>(
            OrtMemoryInfo(NNAPI, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
      });

  InsertAllocator(CreateAllocator(cpu_memory_info));
}

NnapiExecutionProvider::~NnapiExecutionProvider() {}

std::vector<std::unique_ptr<ComputeCapability>>
NnapiExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                                      const std::vector<const KernelRegistry*>& /*kernel_registries*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // TODO: Task 812756: NNAPI EP, add support for subgraph (If and Loop operators)
  if (graph_viewer.IsSubgraph()) {
    return result;
  }

  // We need to get the Android system API level to ensure the GetCapability giving the correct result
  // based on the system.
  // If we are actually running on Android system, we can get the API level by querying the system
  // However, since we also allow the NNAPI EP run GetCapability for model conversion on a non-Android system,
  // since we cannot get the runtime system API level, we have to specify it using compile definition.
  static const int32_t android_feature_level = []() {
#ifdef __ANDROID__
    const auto* nnapi = NnApiImplementation();
    return nnapi->nnapi_runtime_feature_level;
#else
    return ORT_NNAPI_MAX_SUPPORTED_API_LEVEL;
#endif
  }();

  const nnapi::OpSupportCheckParams params{
      android_feature_level,
      !!(nnapi_flags_ & NNAPI_FLAG_USE_NCHW),
  };

  if (params.android_feature_level < ORT_NNAPI_MIN_API_LEVEL) {
    LOGS_DEFAULT(WARNING) << "All ops will fallback to CPU EP, because system NNAPI feature level ["
                          << params.android_feature_level
                          << "] is lower than minimal supported NNAPI API feature level ["
                          << ORT_NNAPI_MIN_API_LEVEL
                          << "] of this build for NNAPI";
    return result;
  }

  // Disable NNAPI if the graph has any unsupported inputs
  for (const auto* input : graph_viewer.GetInputs()) {
    if (!nnapi::IsInputSupported(*input, "graph")) {
      return result;
    }
  }

  const auto excluded_nodes = utils::CreateExcludedNodeSet(graph_viewer, partitioning_stop_ops_);
  const bool check_excluded_nodes = !excluded_nodes.empty();

  std::unordered_set<std::string> node_outputs_in_current_group{};

  const auto is_node_supported = [&](const Node& node) -> bool {
    const bool excluded = check_excluded_nodes && Contains(excluded_nodes, &node);
    const bool supported = !excluded &&
                           nnapi::IsNodeSupportedInGroup(node, graph_viewer, params,
                                                         node_outputs_in_current_group);
    LOGS_DEFAULT(VERBOSE) << "Operator type: [" << node.OpType()
                          << "] index: [" << node.Index()
                          << "] name: [" << node.Name()
                          << "] supported: [" << supported
                          << "]";

    if (supported) {
      // We want to save all the output names of nodes in the current group for easy query
      // See nnapi::IsNodeSupportedInGroup()
      for (const auto* output : node.OutputDefs()) {
        node_outputs_in_current_group.insert(output->Name());
      }
    }

    return supported;
  };

  const auto on_group_closed = [&](const std::vector<const Node*>& group) -> bool {
    // reset per-partition node group tracking
    node_outputs_in_current_group.clear();
    return nnapi::IsValidSupportedNodeGroup(group);
  };

  const auto gen_metadef_name = [&]() {
    uint64_t model_hash;
    int metadef_id = GenerateMetaDefId(graph_viewer, model_hash);
    return MakeString(NNAPI, "_", model_hash, "_", metadef_id);
  };

  result = utils::CreateSupportedPartitions(graph_viewer, is_node_supported, on_group_closed,
                                            gen_metadef_name, NNAPI);

  const auto num_of_partitions = result.size();
  const auto num_of_supported_nodes = std::transform_reduce(
      result.begin(), result.end(),
      size_t{0}, std::plus<>{},
      [](const auto& partition) -> size_t {
        return partition && partition->sub_graph ? partition->sub_graph->nodes.size() : 0;
      });

  const auto summary_msg = MakeString(
      "NnapiExecutionProvider::GetCapability,",
      " number of partitions supported by NNAPI: ", num_of_partitions,
      " number of nodes in the graph: ", graph_viewer.NumberOfNodes(),
      " number of nodes supported by NNAPI: ", num_of_supported_nodes);

  // If the graph is partitioned in multiple subgraphs, and this may impact performance,
  // we want to give users a summary message at warning level.
  if (num_of_partitions > 1) {
    LOGS_DEFAULT(WARNING) << summary_msg;
  } else {
    LOGS_DEFAULT(INFO) << summary_msg;
  }

  return result;
}

#ifdef __ANDROID__
static Status GetOutputBuffer(Ort::CustomOpApi& ort,
                              OrtKernelContext* context,
                              const nnapi::Model& model,
                              const std::string& output_name,
                              const std::vector<uint32_t>& output_shape,
                              const android::nn::wrapper::Type output_type,
                              void** output_buffer) ORT_MUST_USE_RESULT;

static Status GetOutputBuffer(Ort::CustomOpApi& ort,
                              OrtKernelContext* context,
                              const nnapi::Model& model,
                              const std::string& output_name,
                              const std::vector<uint32_t>& output_shape,
                              const android::nn::wrapper::Type output_type,
                              void** output_buffer) {
  using namespace android::nn::wrapper;
  std::vector<int64_t> int64_output_shape(output_shape.begin(),
                                          output_shape.end());
  auto output_idx = model.GetMappedOutputIdx(output_name);
  auto* output_tensor = ort.KernelContext_GetOutput(context, output_idx,
                                                    int64_output_shape.data(),
                                                    int64_output_shape.size());

  switch (output_type) {
    case Type::TENSOR_FLOAT32:
      *output_buffer = ort.GetTensorMutableData<float>(output_tensor);
      break;
    case Type::TENSOR_INT32:
      *output_buffer = ort.GetTensorMutableData<int32_t>(output_tensor);
      break;
    case Type::TENSOR_QUANT8_ASYMM:
      *output_buffer = ort.GetTensorMutableData<uint8_t>(output_tensor);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported output type: ", TypeToStr(output_type));
      break;
  }

  return Status::OK();
}

common::Status NnapiExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  using namespace android::nn::wrapper;
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    Node& fused_node = fused_node_and_graph.fused_node;
    const onnxruntime::GraphViewer& graph_viewer(fused_node_and_graph.filtered_graph);

    nnapi::ModelBuilder builder(graph_viewer);
    builder.SetUseNCHW(nnapi_flags_ & NNAPI_FLAG_USE_NCHW);
    builder.SetUseFp16(nnapi_flags_ & NNAPI_FLAG_USE_FP16);
    if (nnapi_flags_ & NNAPI_FLAG_CPU_DISABLED) {
      builder.SetTargetDeviceOption(nnapi::ModelBuilder::TargetDeviceOption::CPU_DISABLED);
    }

    std::unique_ptr<nnapi::Model> nnapi_model;
    ORT_RETURN_IF_ERROR(builder.Compile(nnapi_model));

    // Build map from input name to its index in input definitions
    {
      std::unordered_map<std::string, size_t> input_map;
      const auto& input_defs = fused_node.InputDefs();
      input_map.reserve(input_defs.size());
      for (size_t i = 0, end = input_defs.size(); i < end; ++i) {
        input_map[input_defs[i]->Name()] = i;
      }
      nnapi_model->SetInputMap(std::move(input_map));
    }

    // Build map from output name to its index in output definitions
    {
      std::unordered_map<std::string, size_t> output_map;
      const auto& output_defs = fused_node.OutputDefs();
      output_map.reserve(output_defs.size());
      for (size_t i = 0, end = output_defs.size(); i < end; ++i) {
        output_map[output_defs[i]->Name()] = i;
      }
      nnapi_model->SetOutputMap(std::move(output_map));
    }

    nnapi_models_.emplace(fused_node.Name(), std::move(nnapi_model));

    NodeComputeInfo compute_info;
    compute_info.create_state_func = [&](ComputeContext* context, FunctionState* state) {
      *state = nnapi_models_[context->node_name].get();
      return 0;
    };

    compute_info.release_state_func = [](FunctionState state) {
      // the `state` is a nnapi::model managed by unique_ptr
      ORT_UNUSED_PARAMETER(state);
    };

    compute_info.compute_func = [](FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
      Ort::CustomOpApi ort{*api};
      nnapi::Model* model = reinterpret_cast<nnapi::Model*>(state);
      const size_t num_inputs = ort.KernelContext_GetInputCount(context);
      const size_t num_outputs = ort.KernelContext_GetOutputCount(context);
      const auto& model_inputs = model->GetInputs();
      const auto& model_outputs = model->GetOutputs();

      ORT_RETURN_IF_NOT(model_inputs.size() <= num_inputs, "Inconsistent input sizes");
      ORT_RETURN_IF_NOT(model_outputs.size() == num_outputs, "Inconsistent output sizes");

      std::vector<nnapi::Execution::InputBuffer> inputs;
      inputs.reserve(model_inputs.size());
      for (size_t i = 0; i < model_inputs.size(); i++) {
        const auto& input_name = model_inputs[i];
        const auto& model_input_type = model->GetInputType(input_name);

        auto input_idx = model->GetMappedInputIdx(input_name);
        const OrtValue* input_tensor = ort.KernelContext_GetInput(context, input_idx);
        auto* tensor_info = ort.GetTensorTypeAndShape(input_tensor);
        std::vector<uint32_t> dimensions;
        for (const auto& dim : ort.GetTensorShape(tensor_info))
          dimensions.push_back(static_cast<uint32_t>(dim));

        // If we have an empty shape, this is a scalar input,
        // since NNAPI will treat empty shape input as dynamic ranking input, (onnx does not support dynamic ranking)
        // we will make the scalar input as a {1} tensor
        if (dimensions.empty()) {
          dimensions.push_back(1);
        }

        // it is possible that the input has the detailed size while
        // the model has an operand with unknown size, use the size
        // of the actual input
        OperandType input_type = model_input_type;
        input_type.SetDimensions(dimensions);

        // We have some op has input can have {0} shapes, such as Resize.scales/roi, these are valid input
        // We still want to log the shape info, in case we get an input shape with some zero dim and some non-zero dim
        if (input_type.GetOperandBlobByteSize() == 0) {
          LOGS_DEFAULT(INFO) << "The actual input [" << input_name << "] has "
                             << nnapi::Shape2String(dimensions) << " shape";
        }

        if (input_type.dimensions != model_input_type.dimensions && model_input_type.GetOperandBlobByteSize() != 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "The actual input dimensions should match the model input "
                                 "dimensions, or model input dimension has 0 (dynamic)");
        }

        const void* inputBuffer = ort.GetTensorData<void>(input_tensor);
        inputs.push_back({input_name, inputBuffer, std::move(input_type)});

        ort.ReleaseTensorTypeAndShapeInfo(tensor_info);
      }

      // From this point we will need to take the exclusive lock on the model until the Predict is
      // performed, to block other threads to perform Predict on the same model
      // TODO, investigate concurrent runs for different executions from the same model
      {
        std::unique_ptr<nnapi::Execution> execution;
        std::unique_lock<OrtMutex> lock(model->GetMutex());
        ORT_RETURN_IF_ERROR(model->PrepareForExecution(execution));

        ORT_RETURN_IF_ERROR(execution->SetInputBuffers(inputs));
        std::vector<nnapi::Execution::OutputBuffer> outputs;
        outputs.reserve(num_outputs);
        std::vector<int32_t> dynamic_shape_output_indices;
        std::vector<OperandType> dynamic_shape_output_types;
        std::vector<std::unique_ptr<uint8_t[]>> dynamic_shape_output_buffers;
        for (size_t i = 0; i < num_outputs; i++) {
          const auto& output_name = model_outputs[i];

          // Below 2 need to be copied since we will modify or take ownership
          const auto model_output_type = model->GetOutputType(output_name, *execution);
          auto output_shape = model_output_type.dimensions;

          bool is_dynamic_shape_output = false;
          if (model_output_type.GetOperandBlobByteSize() == 0) {
            if (!model->SupportsDynamicOutputShape()) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                     "We do not support dynamic output shape or empty output for now");
            }

            is_dynamic_shape_output = true;
          }

          void* output_buffer = nullptr;
          size_t output_buffer_byte_size;
          if (!is_dynamic_shape_output) {
            // Since NNAPI use {1} tensor as scalar, if the model output should have empty shape
            // We are going to replace the {1} shape of the output back to {}
            if (model->IsScalarOutput(output_name))
              output_shape.clear();

            ORT_RETURN_IF_ERROR(GetOutputBuffer(ort, context,
                                                *model,
                                                output_name, output_shape, model_output_type.type,
                                                &output_buffer));
            output_buffer_byte_size = model_output_type.GetOperandBlobByteSize();
          } else {
            // This output is dynamic (size unknown), will need allocate a buffer for the result
            // and copy the content to ORT output tensors after the execution (will know output shape after the execution)
            output_buffer_byte_size = model->GetDynamicOutputBufferSize() * model_output_type.GetElementByteSize();
            std::unique_ptr<uint8_t[]> buffer_holder(new uint8_t[output_buffer_byte_size]);
            output_buffer = buffer_holder.get();
            dynamic_shape_output_types.push_back(model_output_type);
            dynamic_shape_output_indices.push_back(static_cast<int32_t>(i));
            dynamic_shape_output_buffers.push_back(std::move(buffer_holder));
          }

          outputs.push_back({output_buffer, std::move(model_output_type), output_buffer_byte_size});
        }

        ORT_RETURN_IF_ERROR(execution->SetOutputBuffers(outputs));
        std::vector<std::vector<uint32_t>> dynamic_output_shapes;
        ORT_RETURN_IF_ERROR(
            execution->Predict(dynamic_shape_output_indices, dynamic_output_shapes));

        // We have dynamic output buffers, need to copy the content from temp buffers to ORT output tensors
        for (size_t i = 0; i < dynamic_shape_output_indices.size(); i++) {
          const int32_t model_output_idx = dynamic_shape_output_indices[i];
          const auto output_name = model_outputs[model_output_idx];

          const auto& output_shape = dynamic_output_shapes[i];
          auto model_output_type = dynamic_shape_output_types[i];
          model_output_type.SetDimensions(output_shape);
          ORT_RETURN_IF_NOT(model_output_type.GetOperandBlobByteSize() != 0, "We do not support 0 size output for now");

          void* model_output_buffer = dynamic_shape_output_buffers[i].get();
          void* onnx_output_buffer = nullptr;
          ORT_RETURN_IF_ERROR(GetOutputBuffer(ort, context,
                                              *model,
                                              output_name, output_shape, model_output_type.type,
                                              &onnx_output_buffer));

          size_t output_buffer_byte_size = model_output_type.GetOperandBlobByteSize();
          memcpy(onnx_output_buffer, model_output_buffer, output_buffer_byte_size);
        }
      }
      return Status::OK();
    };

    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
#else
common::Status NnapiExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& fused_nodes_and_graphs,
                                               std::vector<NodeComputeInfo>& node_compute_funcs) {
  for (const auto& fused_node_and_graph : fused_nodes_and_graphs) {
    ORT_UNUSED_PARAMETER(fused_node_and_graph);
    NodeComputeInfo compute_info;
    compute_info.create_state_func = [](ComputeContext* /*context*/, FunctionState* /*state*/) { return 0; };
    compute_info.release_state_func = [](FunctionState /*state*/) {};
    compute_info.compute_func = [](FunctionState /* state */, const OrtCustomOpApi* /* api */,
                                   OrtKernelContext* /* context */) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "Compute is not supported in this build.");
    };
    node_compute_funcs.push_back(compute_info);
  }
  return Status::OK();
}
#endif  // __ANDROID__

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/constant_folding.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

namespace onnxruntime {

ConstantFolding::ConstantFolding(const IExecutionProvider& execution_provider,
                                 bool skip_dequantize_linear,
                                 const std::unordered_set<std::string>& compatible_execution_providers,
                                 const std::unordered_set<std::string>& excluded_initializers) noexcept
    : GraphTransformer("ConstantFolding", compatible_execution_providers),
      skip_dequantize_linear_(skip_dequantize_linear),
      excluded_initializers_(excluded_initializers),
      execution_provider_(execution_provider) {
}

// We need to handle a Shape node separately as the input doesn't need to be a constant initializer for
// Shape to be able to be constant folded.
static bool ConstantFoldShapeNode(Graph& graph, Node& node) {
  auto shape = node.MutableInputDefs()[0]->Shape();
  bool is_concrete_shape = true;
  std::vector<int64_t> dim_values;
  if (shape != nullptr) {
    for (int dim_index = 0; dim_index < shape->dim_size(); dim_index++) {
      auto dim = shape->dim(dim_index);
      if (!utils::HasDimValue(dim)) {
        is_concrete_shape = false;
        break;
      }
      dim_values.push_back(dim.dim_value());
    }
  } else {
    is_concrete_shape = false;
  }

  if (is_concrete_shape) {
    ONNX_NAMESPACE::TensorProto shape_constant;
    auto* constant_arg_out = node.MutableOutputDefs()[0];
    shape_constant.set_name(constant_arg_out->Name());
    shape_constant.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_constant.add_dims(dim_values.size());
    shape_constant.set_raw_data(dim_values.data(), dim_values.size() * sizeof(int64_t));
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(dim_values.size());
    constant_arg_out->SetShape(result_shape);
    graph.AddInitializedTensor(shape_constant);
  }

  return is_concrete_shape;  // convert to constant if this is true
}

Status ConstantFolding::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  bool have_updated_nodes = false;
  GraphViewer graph_viewer(graph);
  auto& order = graph_viewer.GetNodesInTopologicalOrder();

  std::function<bool(const std::string&)> is_constant_initializer_check = [&graph](const std::string& name) -> bool {
    return graph.IsSparseInitializer(name);
  };

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    if (!node) {
      continue;
    }

    // avoid to constant fold DequantizeLinear for QDQ format
    if (skip_dequantize_linear_ && node->OpType().compare("DequantizeLinear") == 0) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // Updating a node may allow shape inferencing to infer output shapes of following nodes,
    // so re-run the shape inferencing. use have_updated_nodes as that only applies to this Graph
    // (vs. 'modified' which is passed into subgraphs and applies to the main graph and all subgraphs)
    // Ignore any control flow node containing subgraphs as UpdateShapeInference is not intended to be used on it.
    if (have_updated_nodes && !node->ContainsSubgraph()) {
      ORT_RETURN_IF_ERROR(graph.UpdateShapeInference(*node));
    }

    bool converted_to_constant = false;
    if (node->OpType().compare("Shape") == 0) {
      converted_to_constant = ConstantFoldShapeNode(graph, *node);
    } else {
      InitializedTensorSet constant_inputs;

      // we currently constant fold using the CPU EP only.
      // if the node is assigned to a different EP we can run it if it's an ONNX op as we have CPU based
      // implementations for all ONNX ops. If the node/op is from a different op domain or if the CPU implementation
      // does not support the specific input type(s) required by the node (currently we only support a subset of
      // types in some CPU kernels) then we can't proceed with constant folding for the node.
      auto ep_type = node->GetExecutionProviderType();
      bool cpu_ep = ep_type == kCpuExecutionProvider;
      if (!cpu_ep && node->Domain() != kOnnxDomain) {
        continue;
      }

      // Check if constant folding can be applied on this node.
      if (!graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
          !optimizer_utils::IsOperationDeterministic(node->Domain(), node->OpType()) ||
          // constant folding does not support executing a node that includes subgraphs (control flow operators,
          // such as If/Loop/Scan, fall into this category). individual nodes in the subgraph will be processed
          // by the Recurse call above
          node->ContainsSubgraph() || !graph_utils::AllNodeInputsAreConstant(graph, *node, constant_inputs, excluded_initializers_)) {
        continue;
      }

      // Create execution frame for executing constant nodes.
      OptimizerExecutionFrame::Info info({node}, constant_inputs, graph.ModelPath(), execution_provider_,
                                         is_constant_initializer_check);

      std::vector<int> fetch_mlvalue_idxs;
      for (const auto* node_out : node->OutputDefs()) {
        fetch_mlvalue_idxs.push_back(info.GetMLValueIndex(node_out->Name()));
      }

      // override the EP assigned to the node so that it will use the CPU kernel for Compute.
      if (!cpu_ep) {
        node->SetExecutionProviderType(kCpuExecutionProvider);
      }

      auto kernel = info.CreateKernel(node);

      // undo the EP change to the value that was assigned at graph partitioning time
      if (!cpu_ep) {
        node->SetExecutionProviderType(ep_type);
      }

      if (kernel == nullptr) {
        LOGS(logger, WARNING) << "Could not find a CPU kernel and hence "
                              << "can't constant fold " << node->OpType() << " node '" << node->Name() << "'";

        // Move on to the next candidate node
        continue;
      }

      OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

      OpKernelContext op_kernel_context(&frame, kernel.get(), nullptr, logger);
      ORT_RETURN_IF_ERROR(kernel->Compute(&op_kernel_context));

      std::vector<OrtValue> fetches;
      ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));

      // Go over all output node args and substitute them with the newly computed tensors, which will be
      // added to the graph as initializers.
      ORT_ENFORCE(fetches.size() == node->OutputDefs().size());
      converted_to_constant = true;
      for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
        OrtValue& ort_value = fetches[fetch_idx];
        // XXX: Add support for SparseTensors outputs when we have sparse outputs
        if (!ort_value.IsTensor()) {
          LOGS(logger, WARNING) << "Unsupported output type of " << ort_value.Type()
                                << ". Can't constant fold " << node->OpType() << " node '" << node->Name() << "'";
          converted_to_constant = false;
          break;
        }
      }

      if (converted_to_constant) {
        for (size_t fetch_idx = 0; fetch_idx < fetches.size(); ++fetch_idx) {
          OrtValue& ort_value = fetches[fetch_idx];
          // Build the TensorProto that corresponds to the computed OrtValue and add it as initializer to the graph.
          auto* constant_arg_out = node->MutableOutputDefs()[fetch_idx];
          const Tensor& out_tensor = ort_value.Get<Tensor>();
          ONNX_NAMESPACE::TensorProto out_tensorproto = utils::TensorToTensorProto(out_tensor, constant_arg_out->Name());

          ONNX_NAMESPACE::TensorShapeProto result_shape;
          for (auto& dim : out_tensor.Shape().GetDims()) {
            result_shape.add_dim()->set_dim_value(dim);
          }

          constant_arg_out->SetShape(result_shape);
          graph.AddInitializedTensor(out_tensorproto);
        }
      }
    }

    if (converted_to_constant) {
      // Remove single-output node chain for inputs of the node
      auto p_ip_node = node->InputNodesBegin();
      const auto p_ip_node_end = node->InputNodesEnd();
      while (p_ip_node != p_ip_node_end) {
        const auto& input_node = *p_ip_node;
        // Update the node iterator before removing the corresponding node because removing
        // the node will invalidate the node iterator
        ++p_ip_node;
        graph_utils::RemoveNodesWithOneOutputBottomUp(graph, input_node);
      }

      // Remove the output edges of the constant node and then remove the node itself.
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
      modified = true;
      have_updated_nodes = true;
    }
  }

  return Status::OK();
}
}  // namespace onnxruntime

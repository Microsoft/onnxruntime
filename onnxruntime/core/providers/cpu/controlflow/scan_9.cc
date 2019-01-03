// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// there's no way to use a raw pointer as the copy destination with std::copy_n
// (which gsl::copy uses with span::data() which returns a raw pointer) with the 14.11 toolset
// without generating a 4996 warning. going through an iterator is way too much overhead so turn off the warning.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)
#endif
#include "core/providers/cpu/controlflow/scan.h"
#include "core/providers/cpu/controlflow/scan_utils.h"

#include "core/framework/framework_common.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"

#include "core/providers/common.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/transpose.h"

#include "gsl/gsl_algorithm"

#ifdef _MSC_VER
#pragma warning(pop)
#endif

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
using namespace onnxruntime::scan::detail;

namespace onnxruntime {
/*
ONNX_OPERATOR_SET_SCHEMA(
    Scan,
    9,
    OpSchema()
        .SetDoc(scan_9_doc)
        .Input(
            0,
            "initial_state_and_scan_inputs",
            "Initial values of the loop's N state variables followed by M scan_inputs",
            "V",
            OpSchema::Variadic,
            false)
        .Output(
            0,
            "final_state_and_scan_outputs",
            "Final values of the loop's N state variables followed by K scan_outputs",
            "V",
            OpSchema::Variadic,
            false)
        .Attr(
            "body",
            "The graph run each iteration. It has N+M inputs: "
            "(loop state variables..., scan_input_elts...). It has N+K outputs: "
            "(loop state variables..., scan_output_elts...). Each "
            "scan_output is created by concatenating the value of the specified "
            "scan_output_elt value at the end of each iteration of the loop. It is an error"
            " if the dimensions of these values change across loop iterations.",
            AttributeProto::GRAPH,
            true)
        .Attr(
            "num_scan_inputs",
            "An attribute specifying the number of scan_inputs M. ",
            AttributeProto::INT,
            true)
        .Attr(
            "scan_input_directions",
            "An optional list of M flags. The i-th element of the list specifies the direction "
            "to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 "
            "indicates reverse direction. "
            "If omitted, all scan_input tensors will be scanned in the forward direction.",
            AttributeProto::INTS,
            false)
        .Attr(
            "scan_output_directions",
            "An optional list of K flags, one for each scan_output. The i-th element of the list "
            "specifies whether the i-th scan_output should be constructed by appending or "
            "prepending a new value in each iteration: 0 indicates appending and 1 "
            "indicates prepending. "
            "If omitted, all scan_output tensors will be produced by appending a value "
            "in each iteration.",
            AttributeProto::INTS,
            false)
        .Attr(
            "axes",
            "An optional list of M flags. The i-th element of the list specifies the axis "
            "to be scanned (the sequence axis) for the i-th scan_input. If omitted, 0 will "
            "be used as the scan axis for every scan_input.",
            AttributeProto::INTS,
            false)
        .TypeConstraint("I", {"tensor(int64)"}, "Int64 tensor")
        .TypeConstraint("V", OpSchema::all_tensor_types(), "All Tensor types")
}
*/

class ScanImpl {
 public:
  ScanImpl(OpKernelContextInternal& context,
           const SessionState& session_state,
           int64_t num_scan_inputs,
           const std::vector<int64_t>& input_directions,
           const std::vector<int64_t>& output_directions,
           const std::vector<int64_t>& axes);

  // Initialize by validating all the inputs, and allocating the output tensors
  Status Initialize();

  // Execute the batch, by iterating the sequence in each batch entry
  // and calling the subgraph with each item in the sequence.
  Status Execute();

 private:
  // validate inputs and setup batch size and max sequence length.
  Status ValidateInput();

  Status ValidateSubgraphInput(int start_input, int end_input,
                               const std::vector<const NodeArg*>& graph_inputs);

  // setup inputs to subgraph, transposing if necessary
  Status SetupInputs();

  Status AllocateOutputTensors();
  Status CreateLoopStateVariables(std::vector<LoopStateVariable>& loop_state_variables);

  using ConstTensorSlicerIterators = std::vector<MLValueTensorSlicer<const MLValue>::Iterator>;
  using MutableTensorSlicerIterators = std::vector<MLValueTensorSlicer<MLValue>::Iterator>;

  OpKernelContextInternal& context_;
  const SessionState& session_state_;
  const GraphViewer& subgraph_;

  int num_loop_state_variables_;
  int num_scan_inputs_;
  int num_variadic_inputs_;
  int num_variadic_outputs_;

  int64_t sequence_len_ = -1;

  const std::vector<int64_t>& input_directions_;
  const std::vector<int64_t>& output_directions_;
  const std::vector<int64_t>& axes_from_attribute_;
  std::vector<int64_t> axes_;

  // inputs for graph. either original input value or transposed input if an axis other than 0 was specified
  std::vector<MLValue> inputs_;

  std::vector<std::string> subgraph_output_names_;
  std::vector<std::unique_ptr<OutputIterator>> output_iterators_;

  std::unordered_map<std::string, const MLValue*> implicit_inputs_;
};

template <>
Scan<9>::Scan(const OpKernelInfo& info) : OpKernel(info) {
  // make sure the attribute was present even though we don't need it here.
  // The GraphProto is loaded as a Graph instance by main Graph::Resolve,
  // and a SessionState instance for executing the subgraph is created by InferenceSession.
  // This is available via Info().GetSubgraphSessionState("attribute_name") when Compute is called.
  ONNX_NAMESPACE::GraphProto proto;
  ORT_ENFORCE(info.GetAttr<ONNX_NAMESPACE::GraphProto>("body", &proto).IsOK());
  (void)proto;

  ORT_ENFORCE(info.GetAttr<int64_t>("num_scan_inputs", &num_scan_inputs_).IsOK());

  ReadDirections(info, "scan_input_directions", input_directions_, num_scan_inputs_);

  // we don't know the concrete number of outputs until Compute is called, as the subgraph may have optional inputs
  // so proto.input_size() - num_scan_inputs_ does not necessarily == num state variables.
  // num scan outputs == proto.output_size() - num state variables.
  // read the provided attribute with a size that will be >= num scan outputs and validate later
  ReadDirections(info, "scan_output_directions", output_directions_, -1);

  if (info.GetAttrs<int64_t>("axes", axes_).IsOK()) {
    ORT_ENFORCE(gsl::narrow_cast<int64_t>(axes_.size()) == num_scan_inputs_,
                "Number of entries in 'axes' was ", axes_.size(), " but expected ", num_scan_inputs_);

    // can't do any other validation if we support negatively indexed axes as the min/max value allowed is
    // dependent on the input received by Compute
  } else {
    axes_ = std::vector<int64_t>(num_scan_inputs_, 0);
  }
}

template <>
Status Scan<9>::Compute(OpKernelContext* ctx) const {
  auto ctx_internal = static_cast<OpKernelContextInternal*>(ctx);
  auto* session_state = ctx_internal->SubgraphSessionState("body");
  ORT_ENFORCE(session_state, "Subgraph SessionState was not found for 'body' attribute.");

  // TODO:
  //       Consider how usage of ExecutionFrame and SequentialExecutor can be optimized
  //         - initial implementation is focused on making it work, rather than optimizing.

  ScanImpl scan_impl{*ctx_internal, *session_state, num_scan_inputs_, input_directions_, output_directions_, axes_};

  auto status = scan_impl.Initialize();
  ORT_RETURN_IF_ERROR(status);

  status = scan_impl.Execute();

  return status;
}

ScanImpl::ScanImpl(OpKernelContextInternal& context,
                   const SessionState& session_state,
                   int64_t num_scan_inputs,
                   const std::vector<int64_t>& input_directions,
                   const std::vector<int64_t>& output_directions,
                   const std::vector<int64_t>& axes)
    : context_{context},
      session_state_{session_state},
      subgraph_{*session_state.GetGraphViewer()},
      num_scan_inputs_{gsl::narrow_cast<int>(num_scan_inputs)},
      input_directions_{input_directions},
      output_directions_{output_directions},
      axes_from_attribute_{axes},
      implicit_inputs_{context_.GetImplicitInputs()} {
  num_variadic_inputs_ = context_.NumVariadicInputs(0);
  num_variadic_outputs_ = context_.OutputCount();
  num_loop_state_variables_ = num_variadic_inputs_ - gsl::narrow_cast<int>(num_scan_inputs);

  inputs_.reserve(num_scan_inputs_);
  axes_.reserve(num_scan_inputs_);
}

/**
Calculate the transpose permutations and output shape by shifting the chosen axis to the first dimension.
The other dimension indexes or values are pushed in order after the chosen axis.

e.g. if shape is {2, 3, 4} and axis 1 is chosen the permutations will be {1, 0, 2} and output shape will be {3, 2, 4}
     if axis 2 is chosen the permutations will be {2, 0, 1} and the output shape will be {4, 2, 3}
*/
static void CalculateTransposedShape(const TensorShape& input_shape, int64_t axis,
                                     std::vector<int64_t>& permutations, std::vector<int64_t>& output_shape) {
  int64_t rank = input_shape.NumDimensions();
  const auto& dims = input_shape.GetDims();

  permutations.reserve(rank);
  permutations.push_back(axis);

  output_shape.reserve(rank);
  output_shape.push_back(dims[axis]);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != axis) {
      permutations.push_back(i);
      output_shape.push_back(dims[i]);
    }
  }
}

Status ScanImpl::Initialize() {
  auto status = ValidateInput();
  ORT_RETURN_IF_ERROR(status);

  status = SetupInputs();
  ORT_RETURN_IF_ERROR(status);

  auto& subgraph_outputs = subgraph_.GetOutputs();
  subgraph_output_names_.reserve(subgraph_outputs.size());

  // save list of subgraph output names in their provided order to use when fetching the results
  // from each subgraph execution. the Scan outputs will match this order.
  for (auto& output : subgraph_outputs) {
    subgraph_output_names_.push_back(output->Name());
  }

  status = AllocateOutputTensors();
  ORT_RETURN_IF_ERROR(status);

  return Status::OK();
}

Status ScanImpl::ValidateSubgraphInput(int start_input, int end_input,
                                       const std::vector<const NodeArg*>& graph_inputs) {
  // sequence dim is all that's required as a scalar input will only have that
  auto min_dims_required = 1;

  for (int i = start_input; i < end_input; ++i) {
    auto& input_tensor = *context_.Input<Tensor>(i);
    const auto& input_shape = input_tensor.Shape();

    if (input_shape.NumDimensions() < static_cast<size_t>(min_dims_required))
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid scan input:", graph_inputs[i]->Name(),
                             " Expected ", min_dims_required,
                             " dimensions or more but input had shape of ", input_shape);

    auto seq_len_dim = axes_[i - num_loop_state_variables_];
    auto this_seq_len = input_shape[seq_len_dim];

    if (sequence_len_ < 0) {
      sequence_len_ = this_seq_len;
    } else {
      if (sequence_len_ != this_seq_len) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Scan inputs have inconsistent sequence lengths. Previous value was ",
                               sequence_len_, " but input '", graph_inputs[i]->Name(),
                               "' dimension ", seq_len_dim, " has length of ", this_seq_len);
      }
    }
  }

  return Status::OK();
}

Status ScanImpl::ValidateInput() {
  auto& graph_inputs = subgraph_.GetInputs();  // required inputs
  auto num_graph_inputs = graph_inputs.size();

  if (static_cast<size_t>(num_variadic_inputs_) < num_graph_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The subgraph in 'body' requires ", num_graph_inputs,
                           " inputs but Scan was only given ", num_variadic_inputs_);
  }

  // validate/calculate the axes values and populate axes_.
  // we already checked that axes_from_attribute_.size() == num_scan_inputs_.
  for (int i = 0; i < num_scan_inputs_; ++i) {
    auto axis = axes_from_attribute_[i];

    // zero is always valid, so only do extra checks for non-zero values
    if (axis != 0) {
      int64_t input_rank = context_.Input<Tensor>(i + num_loop_state_variables_)->Shape().NumDimensions();
      // check axis is valid for input_rank and also handle any negative axis value
      if (axis >= -input_rank && axis <= input_rank - 1)
        axis = HandleNegativeAxis(axis, input_rank);
      else
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid value in axes for input ", i,
                               " of ", axis, ". Input tensor rank was ", input_rank);
    }

    axes_.push_back(axis);
  }

  // no validation for loop state variables

  // validate the scan inputs
  auto status = ValidateSubgraphInput(num_loop_state_variables_, num_variadic_inputs_, graph_inputs);
  ORT_RETURN_IF_ERROR(status);

  // validate the output directions match the number of Scan outputs if provided
  if (output_directions_.size() > 0 &&
      output_directions_.size() != static_cast<size_t>(num_variadic_outputs_ - num_loop_state_variables_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Number of entries in 'scan_output_directions' was ", output_directions_.size(),
                           " but expected ", num_variadic_outputs_ - num_loop_state_variables_);
  }

  return Status::OK();
}  // namespace onnxruntime

Status ScanImpl::SetupInputs() {
  auto status = Status::OK();
  AllocatorPtr alloc;

  for (int i = 0; i < num_scan_inputs_; ++i) {
    auto sequence_dim = axes_[i];

    if (sequence_dim == 0) {
      // no transpose required
      inputs_.push_back(*context_.GetInputMLValue(i + num_loop_state_variables_));
    } else {
      auto& input_tensor = *context_.Input<Tensor>(i + num_loop_state_variables_);
      const auto& input_shape = input_tensor.Shape();

      std::vector<int64_t> permutations;
      std::vector<int64_t> new_shape;
      CalculateTransposedShape(input_shape, sequence_dim, permutations, new_shape);

      if (!alloc) {
        status = context_.GetTempSpaceAllocator(&alloc);
        ORT_RETURN_IF_ERROR(status);
      }

      MLValue transpose_output = scan::detail::AllocateTensorInMLValue(input_tensor.DataType(), new_shape, alloc);

      status = TransposeBase::DoTranspose(permutations, input_tensor, *transpose_output.GetMutable<Tensor>());
      ORT_RETURN_IF_ERROR(status);

      inputs_.push_back(transpose_output);
    }
  }

  return status;
}

Status ScanImpl::AllocateOutputTensors() {
  Status status = Status::OK();
  auto& graph_outputs = subgraph_.GetOutputs();

  if (graph_outputs.size() != static_cast<size_t>(num_variadic_outputs_)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph in 'body' produces ", graph_outputs.size(),
                           " outputs but Scan expects ", num_variadic_outputs_);
  }

  std::unique_ptr<OutputIterator> output_iter;

  for (int i = 0; i < num_loop_state_variables_; ++i) {
    status = AllocateOutput(context_, subgraph_, i, true, -1, sequence_len_, output_iter);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  for (int i = num_loop_state_variables_, end = num_variadic_outputs_; i < end; ++i) {
    ScanDirection direction = ScanDirection::kForward;
    const int scan_output_index = i - num_loop_state_variables_;
    if (static_cast<size_t>(scan_output_index) < output_directions_.size()) {
      direction = static_cast<ScanDirection>(output_directions_[scan_output_index]);
    }

    status = AllocateOutput(context_, subgraph_, i, false, -1, sequence_len_, output_iter, direction);
    ORT_RETURN_IF_ERROR(status);
    output_iterators_.push_back(std::move(output_iter));
  }

  return Status::OK();
}

// setup the loop state variables for each batch item
Status ScanImpl::CreateLoopStateVariables(std::vector<LoopStateVariable>& loop_state_variables) {
  AllocatorPtr alloc;
  auto status = context_.GetTempSpaceAllocator(&alloc);
  ORT_RETURN_IF_ERROR(status);

  loop_state_variables.reserve(num_loop_state_variables_);

  for (int i = 0; i < num_loop_state_variables_; ++i) {
    const MLValue& input_mlvalue = *context_.GetInputMLValue(i);
    MLValue* output_mlvalue = context_.GetOutputMLValue(i);
    ORT_ENFORCE(output_mlvalue, "Output MLValue has not been created for loop state variable output ", i);

    loop_state_variables.push_back(LoopStateVariable(input_mlvalue, *output_mlvalue, sequence_len_, alloc));
  }

  return status;
}

Status ScanImpl::Execute() {
  Status status = Status::OK();

  std::vector<LoopStateVariable> loop_state_variables;
  status = CreateLoopStateVariables(loop_state_variables);
  ORT_RETURN_IF_ERROR(status);

  // Setup input MLValue streams
  std::vector<MLValueTensorSlicer<const MLValue>::Iterator> scan_input_stream_iterators;
  scan_input_stream_iterators.reserve(num_variadic_inputs_ - num_loop_state_variables_);

  for (int i = 0, end = num_scan_inputs_; i < end; ++i) {
    const auto& mlvalue = inputs_[i];

    // forward
    if (input_directions_[i] == static_cast<int64_t>(ScanDirection::kForward)) {
      // the iterator is self contained, so we don't need to keep the MLValueTensorSlicer instance around
      scan_input_stream_iterators.push_back(MLValueTensorSlicer<const MLValue>::Create(mlvalue).begin());
    } else {  // reverse
      scan_input_stream_iterators.push_back(MLValueTensorSlicer<const MLValue>::Create(mlvalue).rbegin());
    }
  }

  // Call the subgraph for each item in the sequence
  status = IterateSequence(context_, session_state_, subgraph_, loop_state_variables,
                           scan_input_stream_iterators, sequence_len_, num_loop_state_variables_,
                           num_variadic_inputs_, num_variadic_outputs_, implicit_inputs_,
                           subgraph_output_names_, output_iterators_);

  return status;
}

ONNX_CPU_OPERATOR_KERNEL(Scan,
                         9,
                         KernelDefBuilder()
                             .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>())
                             .TypeConstraint("V", DataTypeImpl::AllTensorTypes()),
                         Scan<9>);
}  // namespace onnxruntime

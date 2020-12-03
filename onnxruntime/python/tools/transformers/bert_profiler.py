import os
import argparse
import json
import onnx
import psutil
import numpy


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, type=str, help="onnx model path")

    parser.add_argument('--batch_size', required=False, type=int, default=1, help="batch size of input")

    parser.add_argument('--sequence_length', required=False, type=int, default=32, help="sequence length of input")

    parser.add_argument('--past_sequence_length', required=False, type=int, default=1, help="past sequence length for gpt2")

    parser.add_argument('--global_length', required=False, type=int, default=1, help="number of global tokens for longformer")

    parser.add_argument('--samples',
                        required=False,
                        type=int,
                        default=1000,
                        help="number of test cases to be generated")

    parser.add_argument("--thread_num", required=False, type=int, default=-1, help="number of threads to use")

    parser.add_argument('--input_ids_name', required=False, type=str, default=None, help="input name for input ids")
    parser.add_argument('--segment_ids_name', required=False, type=str, default=None, help="input name for segment ids")
    parser.add_argument('--input_mask_name',
                        required=False,
                        type=str,
                        default=None,
                        help="input name for attention mask")

    parser.add_argument('--dummy_inputs', required=False, default='default', choices=['bert', 'gpt2', 'longformer', 'default'], help="dummy inputs")

    parser.add_argument('--use_gpu', required=False, action='store_true', help="use GPU")
    parser.set_defaults(use_gpu=False)

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args(argv)
    return args


def create_bert_inputs(model, batch_size, sequence_length, samples, input_ids_name, segment_ids_name, input_mask_name):
    from bert_test_data import get_bert_inputs, generate_test_data
    input_ids, segment_ids, input_mask = get_bert_inputs(model, input_ids_name, segment_ids_name, input_mask_name)
    all_inputs = generate_test_data(batch_size,
                                    sequence_length,
                                    test_cases=samples,
                                    seed=123,
                                    verbose=False,
                                    input_ids=input_ids,
                                    segment_ids=segment_ids,
                                    input_mask=input_mask,
                                    random_mask_length=False)

    return all_inputs

def run_profile(onnx_model_path,
                use_gpu,
                thread_num,
                batch_size,
                sequence_length,
                all_inputs):
    from benchmark_helper import create_onnxruntime_session

    session = create_onnxruntime_session(onnx_model_path, use_gpu, num_threads=thread_num, enable_profiling=True)

    for inputs in all_inputs:
        _ = session.run(None, inputs)

    profile_file = session.end_profiling()
    return profile_file


def parse_profile_results(profile_file):
    print(f"loading profile output {profile_file} ...")

    with open(profile_file, "r") as f:
        sess_time = json.load(f)

    assert isinstance(sess_time, list)

    node_time = {}
    node_provider = {}
    total = 0
    for item in sess_time:
        if item["cat"] == "Node" and "args" in item and "provider" in item["args"]:
            device = "CPU" if item["args"]["provider"] == "CPUExecutionProvider" else "CUDA"
            if item["name"] not in node_provider:
                node_provider[item["name"]] = device
            else:
                assert node_provider[item["name"]] == device

            if item["name"] in node_time:
                node_time[item["name"]] += item["dur"]
            else:
                node_time[item["name"]] = item["dur"]
            total += item["dur"]

    results = [f"Duration\tPercentage\tProvider\tName"]
    for k, v in sorted(node_time.items(), key=lambda x: x[1], reverse=True):
        results.append(f"{v}\t{v * 100.0 / total:5.2f}\t{node_provider[k]}\t{k}")

    return results


def get_dim_from_type_proto(dim):
    return getattr(dim, dim.WhichOneof('value')) if type(dim.WhichOneof('value')) == str else None


def get_shape_from_type_proto(type_proto):
    return [get_dim_from_type_proto(d) for d in type_proto.tensor_type.shape.dim]


def create_dummy_inputs(onnx_model_path, batch_size, sequence_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    dummy_inputs = {}
    for input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(input.type)
        symbol_dims = []
        for i, dim in enumerate(shape):
            if type(dim) == str:
                symbol_dims.append(i)

        # allowed symbolic dimensions: batch_size and sequence_length
        if len(symbol_dims) > 2:
            return None
        if len(symbol_dims) > 0:
            shape[symbol_dims[0]] = batch_size
        if len(symbol_dims) > 1:
            shape[symbol_dims[1]] = sequence_length

        elem_type = input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)
        data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs

def create_gpt2_inputs(onnx_model_path, batch_size, sequence_length, past_sequence_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    # The symbolic name shall be same as those used in Gpt2Helper.export_onnx(...) function.
    symbols = {
        'batch_size' : batch_size,
        'seq_len':sequence_length, 
        'past_seq_len': past_sequence_length,
        'total_seq_len': sequence_length + past_sequence_length
    }

    dummy_inputs = {}
    for input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(input.type)
        for i, dim in enumerate(shape):
            if type(dim) == str and dim not in symbols.keys():
                raise RuntimeError(f"symbol is not supported: {dim}")
            else:
                shape[i] = symbols[dim]

        elem_type = input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)
        data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs

def create_longformer_inputs(onnx_model_path, batch_size, sequence_length, global_length, samples):
    from onnx import TensorProto
    from onnx_model import OnnxModel

    onnx_model = OnnxModel(onnx.load(onnx_model_path))
    symbols = {
        'batch_size' : batch_size,
        'sequence_length':sequence_length
    }

    dummy_inputs = {}
    for input in onnx_model.get_graph_inputs_excluding_initializers():
        shape = get_shape_from_type_proto(input.type)
        for i, dim in enumerate(shape):
            if type(dim) == str and dim not in symbols.keys():
                raise RuntimeError(f"symbol is not supported: {dim}")
            else:
                shape[i] = symbols[dim]

        elem_type = input.type.tensor_type.elem_type
        assert elem_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]
        data_type = numpy.float32 if elem_type == TensorProto.FLOAT else (
            numpy.int64 if elem_type == TensorProto.INT64 else numpy.int32)

        if "global" in input.name:
            data = numpy.zeros(shape, dtype=data_type)
            data[:, :global_length] = 1
        else:
            data = numpy.ones(shape, dtype=data_type)
        dummy_inputs[input.name] = data

    all_inputs = [dummy_inputs for _ in range(samples)]
    return all_inputs


def run(args):
    num_threads = args.thread_num if args.thread_num > 0 else psutil.cpu_count(logical=False)

    # Set OMP environment variable before importing onnxruntime. Needed for cpu only, and no impact for onnxruntime-gpu package.
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = str(num_threads)

    all_inputs = None
    if args.dummy_inputs == 'bert':
        all_inputs = create_bert_inputs(args.model, args.batch_size, args.sequence_length,
                               args.samples, args.input_ids_name, args.segment_ids_name, args.input_mask_name)
    elif args.dummy_inputs == 'gpt2':
        all_inputs = create_gpt2_inputs(args.model, args.batch_size, args.sequence_length, args.past_sequence_length, args.samples)
    elif args.dummy_inputs == 'longformer':
        all_inputs = create_longformer_inputs(args.model, args.batch_size, args.sequence_length, args.global_length, args.samples)
    else: # default
        all_inputs = create_dummy_inputs(args.model, args.batch_size, args.sequence_length, args.samples)
               
    profile_file = run_profile(args.model, args.use_gpu, args.thread_num, args.batch_size, args.sequence_length, all_inputs)

    return parse_profile_results(profile_file)


if __name__ == '__main__':
    args = parse_arguments()
    print("Arguments", args)

    from benchmark_helper import setup_logger
    setup_logger(args.verbose)

    results = run(args)

    print("Results:")
    print("-" * 64)
    for line in results:
        print(line)

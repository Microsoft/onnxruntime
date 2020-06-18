# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script benchmarks gpt2 model with past state.
# For gpt2 model without past state, use benchmark.py to measure performance.

import os
import sys
import numpy
import time
import psutil
import argparse
import logging
import torch
from transformers import GPT2Model, GPT2LMHeadModel, GPT2Tokenizer, AutoConfig

logger = logging.getLogger('')

# Map alias to a tuple of Model Class, Tokenizer, pretrained model name, use LMHead or not, use attention mask or not
MODEL_CLASSES = {
    "gpt2": (GPT2Model, GPT2Tokenizer, "gpt2", False, False),
    "distilgpt2": (GPT2LMHeadModel, GPT2Tokenizer, "distilgpt2", True, True),
}


def dump_environment():
    if "OMP_NUM_THREADS" in os.environ:
        logger.info("OMP_NUM_THREADS={}".format(os.environ["OMP_NUM_THREADS"]))
    else:
        logger.info("no environment variable of OMP_NUM_THREADS")

    if "OMP_WAIT_POLICY" in os.environ:
        logger.info("OMP_WAIT_POLICY={}".format(os.environ["OMP_WAIT_POLICY"]))
    else:
        logger.info("no environment variable of OMP_WAIT_POLICY")


def setup_environment():
    # ATTENTION: these environment variables must be set before importing onnxruntime.
    os.environ["OMP_NUM_THREADS"] = str(psutil.cpu_count(logical=True))
    os.environ["OMP_WAIT_POLICY"] = 'ACTIVE'
    dump_environment()


def pytorch_inference(model, input_ids, past=None, attention_mask=None, total_runs=100):
    latency = []
    with torch.no_grad():
        for _ in range(total_runs):
            start = time.time()
            outputs = model(input_ids=input_ids, past=past, attention_mask=attention_mask)
            latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.debug("PyTorch Inference time = {} ms".format(format(average_latency, '.2f')))
    return outputs, average_latency


def onnxruntime_inference(ort_session, input_ids, past=None, attention_mask=None, total_runs=100):
    ort_inputs = {'input_ids': numpy.ascontiguousarray(input_ids.cpu().numpy())}

    if attention_mask is not None:
        ort_inputs['attention_mask'] = numpy.ascontiguousarray(attention_mask.cpu().numpy())

    if past is not None:
        for i, past_i in enumerate(past):
            ort_inputs[f'past_{i}'] = numpy.ascontiguousarray(past[i].cpu().numpy())

    latency = []
    for _ in range(total_runs):
        start = time.time()
        ort_outputs = ort_session.run(None, ort_inputs)
        latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.debug("OnnxRuntime Inference time = {} ms".format(format(average_latency, '.2f')))

    return ort_outputs, average_latency


def onnxruntime_inference_with_binded_io(ort_session,
                                         input_ids,
                                         output_buffers,
                                         max_last_state_size,
                                         last_state_shape,
                                         past=None,
                                         attention_mask=None,
                                         max_present_size=None,
                                         present_shape=None,
                                         total_runs=100):
    # Bind inputs and outputs to onnxruntime session
    io_binding = ort_session.io_binding()
    # Bind inputs
    io_binding.bind_input('input_ids', input_ids.device.type, 0, numpy.longlong, list(input_ids.size()),
                          input_ids.data_ptr())
    if attention_mask is not None:
        io_binding.bind_input('attention_mask', attention_mask.device.type, 0, numpy.float32,
                              list(attention_mask.size()), attention_mask.data_ptr())
    n_layer = None
    if past is not None:
        n_layer = len(past)
        for i, past_i in enumerate(past):
            io_binding.bind_input(f'past_{i}', past[i].device.type, 0, numpy.float32, list(past[i].size()),
                                  past[i].data_ptr())

    # Bind outputs
    if output_buffers["last_state"] is None or output_buffers["present"] is None:
        # Allocate output buffers with the largest size need by current model
        allocateOutputBuffers(output_buffers, max_last_state_size, max_present_size, n_layer, input_ids.device)

    last_state_buffer = output_buffers["last_state"]
    present_buffers = output_buffers["present"]
    io_binding.bind_output(ort_session.get_outputs()[0].name, last_state_buffer.device.type, 0, numpy.float32,
                           last_state_shape, last_state_buffer.data_ptr())
    if present_buffers is not None:
        for i, present_i in enumerate(present_buffers):
            io_binding.bind_output(f'present_{i}', present_buffers[i].device.type, 0, numpy.float32, present_shape,
                                   present_buffers[i].data_ptr())

    latency = []
    for _ in range(total_runs):
        start = time.time()
        # Run onnxruntime with io binding
        ort_session.run_with_iobinding(io_binding)
        latency.append(time.time() - start)

    average_latency = sum(latency) * 1000 / len(latency)
    logger.debug("OnnxRuntime with IO binding inference time = {} ms".format(format(average_latency, '.2f')))

    # Copy results to cpu
    ort_outputs = [last_state_buffer[0:numpy.prod(last_state_shape)].reshape(last_state_shape).cpu()]
    if present_buffers is not None:
        for i, present_i in enumerate(present_buffers):
            ort_outputs.append(present_buffers[i][0:numpy.prod(present_shape)].reshape(present_shape).cpu())
    return ort_outputs, average_latency


def inference(model,
              ort_session,
              input_ids,
              past=None,
              attention_mask=None,
              output_buffers=None,
              max_last_state_size=None,
              last_state_shape=None,
              max_present_size=None,
              present_shape=None,
              total_runs=100,
              verify_outputs=True,
              disable_ort_io_binding=False):
    outputs, torch_latency = pytorch_inference(model, input_ids, past, attention_mask, total_runs)
    ort_outputs, ort_latency = onnxruntime_inference(ort_session, input_ids, past, attention_mask, total_runs)
    latencies = [torch_latency, ort_latency]
    if not disable_ort_io_binding:
        ort_io_outputs, ort_io_latency = onnxruntime_inference_with_binded_io(ort_session, input_ids, output_buffers,
                                                                              max_last_state_size, last_state_shape,
                                                                              past, attention_mask, max_present_size,
                                                                              present_shape, total_runs)
        latencies.append(ort_io_latency)
    if verify_outputs:
        logger.debug('Verifying Pytorch and ONNX Runtime outputs.')
        verify_ort_outputs(model, outputs, ort_outputs)
        if not disable_ort_io_binding:
            logger.debug('Verifying Pytorch and ONNX Runtime with io binding outputs.')
            verify_ort_outputs(model, outputs, ort_io_outputs)

    return latencies


def verify_ort_outputs(model, torch_outputs, ort_outputs):
    is_close = numpy.allclose(ort_outputs[0], torch_outputs[0].cpu(), rtol=1e-05, atol=1e-04)
    logger.debug(f'PyTorch and OnnxRuntime output 0 (last_state) are close: {is_close}')

    is_all_close = is_close
    for layer in range(model.config.n_layer):
        is_close = numpy.allclose(ort_outputs[1 + layer], torch_outputs[1][layer].cpu(), rtol=1e-05, atol=1e-04)
        logger.debug(f'PyTorch and OnnxRuntime layer {layer} state (present_{layer}) are close:{is_close}')
        is_all_close = is_all_close and is_close

    if not is_all_close:
        logger.warning(f'PyTorch and OnnxRuntime results are not all close.')


def allocateOutputBuffers(output_buffers, max_last_state_size, max_present_size, n_layer, device):
    # Allocate output tensors with the largest test size needed. So the allocated memory can be reused
    # for each test run.
    # dummy last state
    if output_buffers["last_state"] is None:
        output_buffers["last_state"] = torch.empty(max_last_state_size, dtype=torch.float32, device=device)
    # create dummy present
    if n_layer is not None and output_buffers["present"] is None:
        present_buffers = [torch.empty(max_present_size, dtype=torch.float32, device=device) for _ in range(n_layer)]
        output_buffers["present"] = present_buffers


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_type',
                        required=True,
                        type=str,
                        choices=list(MODEL_CLASSES.keys()),
                        help='Model type selected in the list: ' + ', '.join(MODEL_CLASSES.keys()))

    parser.add_argument('-c',
                        '--cache_dir',
                        required=False,
                        type=str,
                        default='./cache_models',
                        help='Directory to cache pre-trained models')

    parser.add_argument('--onnx_dir',
                        required=False,
                        type=str,
                        default='./onnx_models',
                        help='Directory to store onnx models')

    parser.add_argument('-t',
                        '--test_times',
                        required=False,
                        default=100,
                        type=int,
                        help='Number of repeat times to get average inference latency.')

    parser.add_argument('-v', '--validate_onnx', required=False, action='store_true', help='Validate ONNX model')

    parser.add_argument('-o',
                        '--optimize_onnx',
                        required=False,
                        action='store_true',
                        help='Use optimizer.py to optimize onnx model')
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument('--disable_ort_io_binding',
                        required=False,
                        action='store_true',
                        help='Disable running ONNX Runtime with binded inputs and outputs. ')
    parser.set_defaults(disable_ort_io_binding=False)

    parser.add_argument('--use_gpu', required=False, action='store_true')
    parser.set_defaults(use_gpu=False)

    parser.add_argument('-b', '--batch_sizes', nargs='+', type=int, default=[1])

    parser.add_argument('-s', '--sequence_lengths', nargs='+', type=int, default=[8, 16, 32, 64, 128, 256])

    parser.add_argument('--verbose', required=False, action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()

    return args


def setup_logger(verbose=True):
    # output logging to stdout
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter('[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s'))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter('%(filename)20s: %(message)s'))
        logging_level = logging.INFO
        logging.getLogger("transformers").setLevel(logging.ERROR)
    log_handler.setLevel(logging_level)

    # Avoid duplicated handlers when runing this script in multiple cells of Jupyter Notebook.
    if not logger.hasHandlers():
        logger.addHandler(log_handler)

    logger.setLevel(logging_level)


def export_onnx(model, config, tokenizer, device, output_dir, use_LMHead=False, use_attention_mask=False):
    """ Export GPT-2 model with past state to ONNX model
    """
    model.to(device)

    inputs = tokenizer.encode_plus("Here is an example input for GPT2 model",
                                   add_special_tokens=True,
                                   return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    logger.debug(f"input_ids={input_ids}")

    # Use example input to generate an example of past state.
    outputs = model(input_ids=input_ids, past=None)
    assert len(outputs) == 2
    logger.debug(f"output 0 shape={outputs[0].shape}")
    logger.debug(f"outputs[1][0] shape={outputs[1][0].shape}")

    num_layer = model.config.n_layer
    past_names = [f'past_{i}' for i in range(num_layer)]
    present_names = [f'present_{i}' for i in range(num_layer)]

    # GPT2Model  output last_state has shape (batch_size, all_seq_len, hidden_size)
    # GPT2LMHeadModel output prediction_scores has shape (batch_size, all_seq_len, vocab_size)
    # where all_seq_len = past_seq_len + seq_len
    output_names = ["prediction_scores" if use_LMHead else "last_state"] + present_names

    # Shape of input tensors:
    #    input_ids: (batch_size, seq_len)
    #    past_{i}:  (2, batch_size, num_heads, past_seq_len, hidden_size/num_heads)
    #    attention_mask: (batch_size, seq_len)
    # Shape of output tensors:
    #    last_state: (batch_size, all_seq_len, hidden_size)
    #    present_{i}:  (2, batch_size, num_heads, all_seq_len, hidden_size/num_heads)
    dynamic_axes = {'input_ids': {0: 'batch_size', 1: 'seq_len'}, output_names[0]: {0: 'batch_size', 1: 'all_seq_len'}}
    for name in past_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'past_seq_len'}
    for name in present_names:
        dynamic_axes[name] = {1: 'batch_size', 3: 'all_seq_len'}

    if use_attention_mask:
        dynamic_axes['attention_mask'] = {0: 'batch_size', 1: 'seq_len'}

    dummy_input_ids = torch.randint(low=0,
                                    high=model.config.vocab_size - 1,
                                    size=(1, 1),
                                    dtype=torch.int64,
                                    device=device)
    # Use the example past state to create dummy past state inputs.
    dummy_past = [torch.zeros(list(outputs[1][0].shape), dtype=torch.float32, device=device) for _ in range(num_layer)]

    dummy_mask = torch.ones([1, 1], dtype=torch.float32, device=device) if use_attention_mask else None

    model_name = "gpt2{}_past{}.onnx".format("_lm" if use_LMHead else "", "_mask" if use_attention_mask else "")
    export_model_path = os.path.join(output_dir, model_name)

    with torch.no_grad():
        outputs = model(input_ids=dummy_input_ids, past=dummy_past, attention_mask=dummy_mask)
    logger.debug(f"present_0 shape={outputs[1][0].shape}")

    torch.onnx.export(model,
                      args=(dummy_input_ids, tuple(dummy_past), dummy_mask) if use_attention_mask else
                      (dummy_input_ids, tuple(dummy_past)),
                      f=export_model_path,
                      input_names=['input_ids'] + past_names + (['attention_mask'] if use_attention_mask else []),
                      output_names=output_names,
                      example_outputs=outputs,
                      dynamic_axes=dynamic_axes,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=False)
    return export_model_path


def main():
    args = parse_arguments()
    setup_logger(args.verbose)
    dump_environment()

    cache_dir = args.cache_dir
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    output_dir = args.onnx_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    use_torchscript = False
    (model_class, tokenizer_class, model_name, use_LMHead, use_attention_mask) = MODEL_CLASSES[args.model_type]
    config = AutoConfig.from_pretrained(model_name, torchscript=use_torchscript, cache_dir=cache_dir)
    model = model_class.from_pretrained(model_name, config=config, cache_dir=cache_dir)
    tokenizer = tokenizer_class.from_pretrained(model_name, cache_dir=cache_dir)
    #if use_torchscript:
    #    model = torch.jit.trace(model, (input_ids, past))

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    export_model_path = export_onnx(model, config, tokenizer, device, output_dir, use_LMHead, use_attention_mask)

    # setup environment variables before importing onnxruntime.
    setup_environment()
    import onnxruntime

    if not args.optimize_onnx:
        onnx_model_path = export_model_path
    else:
        from optimizer import optimize_model
        m = optimize_model(export_model_path,
                           model_type='gpt2',
                           num_heads=config.num_attention_heads,
                           hidden_size=config.hidden_size,
                           opt_level=0,
                           optimization_options=None,
                           use_gpu=args.use_gpu)
        onnx_model_path = os.path.join(output_dir, 'gpt2_past_optimized.onnx')
        m.save_model_to_file(onnx_model_path)

    if args.use_gpu and 'CUDAExecutionProvider' not in onnxruntime.get_available_providers():
        logger.warning("Please install onnxruntime-gpu package to test GPU inference.")

    sess_options = onnxruntime.SessionOptions()
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    logger.info(f"Session option: intra_op_num_threads={sess_options.intra_op_num_threads}")

    logger.info(f"Start inferencing onnx model: {onnx_model_path}")
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options)

    # Calculate the largest size needed for each output
    max_batch_size = max(args.batch_sizes)
    max_seq_len = max(args.sequence_lengths)
    max_present_size = numpy.prod([
        2, max_batch_size, config.num_attention_heads, max_seq_len + 1,
        int(config.hidden_size / config.num_attention_heads)
    ])

    # dummy last state
    if use_LMHead:
        max_last_state_size = numpy.prod([max_batch_size, 1, config.vocab_size])
    else:
        max_last_state_size = numpy.prod([max_batch_size, 1, config.hidden_size])

    output_buffers = {"last_state": None, "present": None}
    for batch_size in args.batch_sizes:
        for sequence_length in args.sequence_lengths:
            past_shape = [
                2, batch_size, config.num_attention_heads, sequence_length,
                int(config.hidden_size / config.num_attention_heads)
            ]
            dummy_past = [torch.rand(past_shape, dtype=torch.float32, device=device) for _ in range(config.n_layer)]
            dummy_input_ids = torch.randint(low=0,
                                            high=model.config.vocab_size - 1,
                                            size=(batch_size, 1),
                                            dtype=torch.int64,
                                            device=device)
            dummy_mask = torch.ones([batch_size, 1], dtype=torch.float32, device=device) if use_attention_mask else None

            # Calculate the expected output shapes
            last_state_shape = [batch_size, 1, config.vocab_size] if use_LMHead else [batch_size, 1, config.hidden_size]
            present_shape = [
                2, batch_size, config.num_attention_heads, sequence_length + 1,
                int(config.hidden_size / config.num_attention_heads)
            ]

            latencies = inference(model,
                                  session,
                                  dummy_input_ids,
                                  dummy_past,
                                  dummy_mask,
                                  output_buffers,
                                  max_last_state_size,
                                  last_state_shape,
                                  max_present_size,
                                  present_shape,
                                  args.test_times,
                                  verify_outputs=args.validate_onnx,
                                  disable_ort_io_binding=args.disable_ort_io_binding)
            ort_io_latency_info = f", ort_io_latency={latencies[2]}" if not args.disable_ort_io_binding else ""
            logger.info(
                f"batch_size={batch_size}, sequence_length={sequence_length}, torch_latency={latencies[0]}, ort_latency={latencies[1]}{ort_io_latency_info}"
            )


if __name__ == '__main__':
    main()

---
title: CUDA
parent: Execution Providers
grand_parent: Reference
nav_order: 1
---

# CUDA Execution Provider
{: .no_toc }

The CUDA Execution Provider enables hardware accelerated computation on Nvidia CUDA-enabled GPUs.


## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Requirements
Please reference table below for official GPU packages dependencies.

|ONNX Runtime|CUDA|cuDNN|Notes|
|---|---|---|---|
|1.8|11.0.3|8.0.4 (Linux)<br/>8.0.2.39 (Windows)|libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4<br/>libcupti.so 2020.1.1|
|1.7|11.0.3|8.0.4 (Linux)<br/>8.0.2.39 (Windows)|libcudart 11.0.221<br/>libcufft 10.2.1.245<br/>libcurand 10.2.1.245<br/>libcublasLt 11.2.0.252<br/>libcublas 11.2.0.252<br/>libcudnn 8.0.4|
|1.5-1.6|10.2|8.0.3|CUDA 11 can be built from source|
|1.2-1.4|10.1|7.6.5|Requires cublas10-10.2.1.243; cublas 10.1.x will not work|
|1.0-1.1|10.0|7.6.4|CUDA versions from 9.1 up to 10.1, and cuDNN versions from 7.1 up to 7.4 should also work with Visual Studio 2017|

For older versions, please reference the readme and build pages on the release branch.

## Build
For build instructions, please see the [BUILD page](../../how-to/build/eps.md#cuda).

## Configuration Options
The CUDA Execution Provider supports the following configuration options.

### device_id
The device ID.

Default value: 0

### gpu_mem_limit
The size limit of the device memory arena in bytes. This size limit is only for the execution provider's arena. The total device memory usage may be higher.

Default value: max value of C++ size_t type (effectively unlimited)

### arena_extend_strategy
The strategy for extending the device memory arena.

Value                   | Description
-|-
kNextPowerOfTwo (0)     | subsequent extensions extend by larger amounts (multiplied by powers of two)
kSameAsRequested (1)    | extend by the requested amount

Default value: kNextPowerOfTwo

### cudnn_conv_algo_search
The type of search done for cuDNN convolution algorithms.

Value           | Description
-|-
EXHAUSTIVE (0)  | expensive exhaustive benchmarking using cudnnFindConvolutionForwardAlgorithmEx
HEURISTIC (1)   | lightweight heuristic based search using cudnnGetConvolutionForwardAlgorithm_v7
DEFAULT (2)     | default algorithm using CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM

Default value: EXHAUSTIVE

### do_copy_in_default_stream
Whether to do copies in the default stream or use separate streams. The recommended setting is true. If false, there are race conditions and possibly better performance.

Default value: true

## Samples

### Python

```python
import onnxruntime as ort

model_path = '<path to model>'

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider',
]

session = ort.InferenceSession(model_path, providers=providers)
```

### C/C++

```c++
OrtSessionOptions* session_options = /* ... */;

OrtCUDAProviderOptions options;
options.device_id = 0;
options.arena_extend_strategy = 0;
options.gpu_mem_limit = 2 * 1024 * 1024 * 1024;
options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
options.do_copy_in_default_stream = 1;

SessionOptionsAppendExecutionProvider_CUDA(session_options, &options);
```


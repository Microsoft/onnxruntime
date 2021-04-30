---
title: Direct ML
parent: Execution Providers
grand_parent: Reference
nav_order: 4
---

# DirectML Execution Provider
{: .no_toc }

DirectML is a high-performance, hardware-accelerated DirectX 12 library for machine learning on Windows.  DirectML provides GPU acceleration for common machine learning tasks across a broad range of supported hardware and drivers.

When used standalone, the DirectML API is a low-level DirectX 12 library and is suitable for high-performance, low-latency applications such as frameworks, games, and other real-time applications. The seamless interoperability of DirectML with Direct3D 12 as well as its low overhead and conformance across hardware makes DirectML ideal for accelerating machine learning when both high performance is desired, and the reliability and predictability of results across hardware is critical.

The *DirectML Execution Provider* is an optional component of ONNX Runtime that uses DirectML to accelerate inference of ONNX models. The DirectML execution provider is capable of greatly improving evaluation time of models using commodity GPU hardware, without sacrificing broad hardware support or requiring vendor-specific extensions to be installed.

The DirectML Execution Provider currently uses DirectML version 1.4.2.

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Minimum requirements

The DirectML execution provider requires any DirectX 12 capable device. Almost all commercially-available graphics cards released in the last several years support DirectX 12. Examples of compatible hardware include:

* NVIDIA Kepler (GTX 600 series) and above
* AMD GCN 1st Gen (Radeon HD 7000 series) and above
* Intel Haswell (4th-gen core) HD Integrated Graphics and above

DirectML is compatible with Windows 10, version 1709 (10.0.16299; RS3, "Fall Creators Update") and newer.

## Building from source

For general information about building onnxruntime, see [BUILD](../../how-to/build.md).

Requirements for building the DirectML execution provider:
1. Visual Studio 2017 toolchain (see [cmake configuration instructions](../../how-to/build.md))
2. [The Windows 10 SDK (10.0.18362.0) for Windows 10, version 1903](https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk) (or newer)

To build onnxruntime with the DML EP included, supply the `--use_dml` parameter to `build.bat`. e.g.

```powershell
    build.bat --config RelWithDebInfo --build_shared_lib --parallel --use_dml
```

The DirectML execution provider supports building for both x64 (default) and x86 architectures.

Note that building onnxruntime with the DirectML execution provider enabled causes the the DirectML redistributable package to be automatically downloaded as part of the build.  Its use is governed by a license whose text may be found as part of the NuGet package.

## Using the DirectML execution provider

When using the [C API](../api/c-api.md) with a DML-enabled build of onnxruntime (see [Building from source](#building-from-source)), the DirectML execution provider can be enabled using one of the two factory functions included in `include/onnxruntime/core/providers/dml/dml_provider_factory.h`.

### `OrtSessionOptionsAppendExecutionProvider_DML` function

 Creates a DirectML Execution Provider which executes on the hardware adapter with the given `device_id`, also known as the adapter index. The device ID corresponds to the enumeration order of hardware adapters as given by [IDXGIFactory::EnumAdapters](https://docs.microsoft.com/windows/win32/api/dxgi/nf-dxgi-idxgifactory-enumadapters). A `device_id` of 0 always corresponds to the default adapter, which is typically the primary display GPU installed on the system. A negative `device_id` is invalid.

```c
OrtStatus* OrtSessionOptionsAppendExecutionProvider_DML(
    _In_ OrtSessionOptions* options,
    int device_id
    );
```

### `OrtSessionOptionsAppendExecutionProviderEx_DML` function

Creates a DirectML Execution Provider using the given DirectML device, and which executes work on the supplied D3D12 command queue. The DirectML device and D3D12 command queue must have the same parent [ID3D12Device](https://docs.microsoft.com/windows/win32/api/d3d12/nn-d3d12-id3d12device), or an error will be returned. The D3D12 command queue must be of type `DIRECT` or `COMPUTE` (see [D3D12_COMMAND_LIST_TYPE](https://docs.microsoft.com/windows/win32/api/d3d12/ne-d3d12-d3d12_command_list_type)). If this function succeeds, the inference session once created will maintain a strong reference on both the `dml_device` and `command_queue` objects.

```c
OrtStatus* OrtSessionOptionsAppendExecutionProviderEx_DML(
    _In_ OrtSessionOptions* options,
    _In_ IDMLDevice* dml_device,
    _In_ ID3D12CommandQueue* cmd_queue
    );
```

### See Also

[DMLCreateDevice function](https://docs.microsoft.com/windows/win32/api/directml/nf-directml-dmlcreatedevice)  
[ID3D12Device::CreateCommandQueue method](https://docs.microsoft.com/windows/win32/api/d3d12/nf-d3d12-id3d12device-createcommandqueue)  
[Direct3D 12 programming guide](https://docs.microsoft.com/windows/win32/direct3d12/directx-12-programming-guide)

### ONNX opset support

The DirectML execution provider currently supports ONNX opset 12 ([ONNX v1.7](https://github.com/onnx/onnx/releases/tag/v1.7.0)). Evaluating models which require a higher opset version is not supported, and may produce unexpected results.

### Multi-threading and supported session options

The DirectML execution provider does not support the use of memory pattern optimizations or parallel execution in onnxruntime. When supplying session options during InferenceSession creation, these options must be disabled or an error will be returned.

If using the onnxruntime C API, you must call `DisableMemPattern` and `SetSessionExecutionMode` functions to set the options required by the DirectML execution provider.

See [onnxruntime\include\onnxruntime\core\session\onnxruntime_c_api.h](https://github.com/microsoft/onnxruntime/tree/master/include//onnxruntime/core/session/onnxruntime_c_api.h).

    OrtStatus*(ORT_API_CALL* DisableMemPattern)(_Inout_ OrtSessionOptions* options)NO_EXCEPTION;

    OrtStatus*(ORT_API_CALL* SetSessionExecutionMode)(_Inout_ OrtSessionOptions* options, ExecutionMode execution_mode)NO_EXCEPTION;

If creating the onnxruntime InferenceSession object directly, you must set the appropriate fields on the `onnxruntime::SessionOptions` struct. Specifically, `execution_mode` must be set to `ExecutionMode::ORT_SEQUENTIAL`, and `enable_mem_pattern` must be `false`.

Additionally, as the DirectML execution provider does not support parallel execution, it does not support multi-threaded calls to `Run` on the same inference session. That is, if an inference session using the DirectML execution provider, only one thread may call `Run` at a time. Multiple threads are permitted to call `Run` simultaneously if they operate on different inference session objects.

## Samples

A complete sample of onnxruntime using the DirectML execution provider can be found under [samples/c_cxx/fns_candy_style_transfer](https://github.com/microsoft/onnxruntime/tree/master/samples//c_cxx/fns_candy_style_transfer).

## Performance best practices
The DirectML execution provider works most efficiently when tensor shapes are known at the time a session is created.  This provides a few performance benefits:
1) Because constant folding can occur more often, there may be fewer CPU / GPU copies and stalls during evaluations.
2) More initialization work occurs when sessions are created rather than during the first evaluation.
3) Weights may be pre-processed within DirectML, enabling more efficient algorithms to be used.
4) Graph optimization occurs within DirectML. For example, Concat operators may be removed, and more optimal tensor layouts may be used for the input and output of operators.

Normally when the shapes of model inputs are known during session creation, the shapes for the rest of the model are inferred by OnnxRuntime when a session is created.  However if a model input contains a free dimension (such as for batch size), steps must be taken to retain the above performance benefits.

In this case, there are three options:
- Edit the model to replace an input's free dimension (specified through ONNX using "dim_param") with a fixed size (specified through ONNX using "dim_value").
- Specify values of named dimensions within model inputs when creating the session using the OnnxRuntime *AddFreeDimensionOverrideByName* ABI.
- Edit the model to ensure that an input's free dimension has a [denotation](https://github.com/onnx/onnx/blob/master/docs/DimensionDenotation.md) (such as "DATA_BATCH," or a custom denotation).  Then when creating the session, specify the dimension size for each denotation.  This can be done using the OnnxRuntime *AddFreeDimensionOverride* ABI.


## See also

[DirectML documentation \(docs.microsoft.com\)](https://docs.microsoft.com/en-us/windows/win32/direct3d12/dml)

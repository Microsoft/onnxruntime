---
title: Install
parent: Get Started
nav_order: 1
---

# Install ONNX Runtime
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

Use this guide to install ONNX Runtime and its dependencies, for your target operating system, hardware, accelerator, and language.

For an overview, see this [installation matrix](https://onnxruntime.ai).

## Prerequisites

### Linux / CPU

1. English language package with the `en_US.UTF-8` locale

    * Install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
    * Run `locale-gen en_US.UTF-8`
    * Run `update-locale LANG=en_US.UTF-8`

### Linux / GPU

1. English language package with the `en_US.UTF-8` locale

    * Install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
    * Run `locale-gen en_US.UTF-8`
    * Run `update-locale LANG=en_US.UTF-8`

2. CUDA 11.0.3 and cuDNN 8.0.2.4
   * libcudart 11.0.221
   * libcufft 10.2.1.245
   * libcurand 10.2.1.245
   * libcublasLt 11.2.0.252
   * libcublas 11.2.0.252
   * libcudnn 8.0.4

Version dependencies for older ONNX Runtime releases are listed [here](../reference/execution-providers/CUDA-ExecutionProvider.html#version-dependency).

### Windows / CPU

1. English language package with the `en_US.UTF-8` locale

2. [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)


  
### Windows / GPU

1. English language package with the `en_US.UTF-8` locale

2. [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)

3. CUDA 11.0.3 and cuDNN 8.0.2.39

Version dependencies for older ONNX Runtime releases are listed [here](../reference/execution-providers/CUDA-ExecutionProvider.html#version-dependency).

### MacOS / CPU

1. The system must have libomp.dylib which can be installed using `brew install libomp`.

## Install

* Default CPU Provider (Eigen + MLAS)
* GPU Provider - NVIDIA CUDA
* GPU Provider - DirectML (Windows)
  * *On Windows, the [DirectML execution provider](../reference/execution-providers/DirectML-ExecutionProvider.html) is recommended for optimal performance and compatibility with a broad set of GPUs.*

If using pip, run `pip install --upgrade pip` prior to downloading.

|Repository|Official build|Nightly build|
|---|---|---|
|Python|CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime)| [ort-nightly (dev)](https://test.pypi.org/project/ort-nightly)|
||GPU: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) | [ort-gpu-nightly (dev)](https://test.pypi.org/project/ort-gpu-nightly)|
|C#/C/C++|CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|
||GPU: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|
|Java|CPU: [**com.microsoft.onnxruntime/onnxruntime**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)|
||GPU: [**com.microsoft.onnxruntime/onnxruntime_gpu**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)|
|nodejs|CPU: [**onnxruntime**](https://www.npmjs.com/package/onnxruntime)|
|Other|[Contributed non-official packages](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp) (including Homebrew, Linuxbrew, and nixpkgs)|
||These are not maintained by the core ONNX Runtime team and may have limited support; use at your discretion.|

Note: Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.

## Docker Images

* [ONNX-Ecosystem](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem): includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started
* [Additional dockerfiles](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles)

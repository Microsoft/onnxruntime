---
title: Install ORT
parent: How to
nav_order: 1
---

# Install ONNX Runtime
{: .no_toc }

See the [installation matrix](https://onnxruntime.ai) for recommended instructions for desired combinations of target operating system, hardware, accelerator, and language. 

Details on OS versions, compilers, language versions, dependent libraries , etc can be found under [Compatibility](../resources/compatibility.md#Environment-compatibility).

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Inference

The following build variants are available as officially supported packages. Others can be [built from source](../how-to/build.md) from each release branch.

1. Default CPU Provider
2. GPU Provider - NVIDIA CUDA
3. GPU Provider - [DirectML](../reference/execution-providers/DirectML-ExecutionProvider.md) (Windows) - *recommended for optimized performance and compatibility with a broad set of GPUs on Windows devices*

||Official build|Nightly build|
|---|---|---|
|Python|If using pip, run `pip install --upgrade pip` prior to downloading.||
||CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime)| [ort-nightly (dev)](https://test.pypi.org/project/ort-nightly)|
||GPU: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) | [ort-gpu-nightly (dev)](https://test.pypi.org/project/ort-nightly-gpu)|
|C#/C/C++|CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) | [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|
||GPU - CUDA: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|
||GPU - DirectML: [**Microsoft.ML.OnnxRuntime.DirectML**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML)|[ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)|
|WinML|[**Microsoft.AI.MachineLearning**](https://www.nuget.org/packages/Microsoft.AI.MachineLearning)||
|Java|CPU: [**com.microsoft.onnxruntime:onnxruntime**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime)|
||GPU: [**com.microsoft.onnxruntime:onnxruntime_gpu**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu)|
|Android|[**com.microsoft.onnxruntime:onnxruntime-mobile**](https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime-mobile) ||
|iOS (C/C++)|CocoaPods: **onnxruntime-mobile-c**||
|Objective-C|CocoaPods: **onnxruntime-mobile-objc**||
|React Native|[**onnxruntime-react-native**](https://www.npmjs.com/package/onnxruntime-react-native)||
|Node.js|[**onnxruntime-node**](https://www.npmjs.com/package/onnxruntime-node)||
|Web|[**onnxruntime-web**](https://www.npmjs.com/package/onnxruntime-web)||



*Note: Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.*

### Requirements
{: .no_toc }

* All builds require the English language package with `en_US.UTF-8` locale. On Linux, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
by running `locale-gen en_US.UTF-8` and `update-locale LANG=en_US.UTF-8`

* The GPU CUDA build requires installation of compatible CUDA and cuDNN libraries: see [CUDA Execution Provider requirements](../reference/execution-providers/CUDA-ExecutionProvider.html#requirements).

* Windows builds require [Visual C++ 2019 runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads).


## Training
 
||Official build|Nightly build|
|---|---|---|
|PyTorch 1.8.1 (CUDA 10.2)|[**onnxruntime_stable_torch181.cu102**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.cu102.html)|[onnxruntime_nightly_torch181.cu102](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.cu102.html)|
|PyTorch 1.8.1 (CUDA 11.1)|[**onnxruntime_stable_torch181.cu111**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.cu111.html )|[onnxruntime_nightly_torch181.cu111](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.cu111.html)|
|PyTorch 1.9 (CUDA 10.2)|[**onnxruntime-training**](https://pypi.org/project/onnxruntime-training/)|[onnxruntime_nightly_torch190.cu102](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.cu102.html)|
|PyTorch 1.9 (CUDA 11.1)|[**onnxruntime_stable_torch190.cu111**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.cu111.html)|[onnxruntime_nightly_torch190.cu111](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.cu111.html)|
|[*Preview*] PyTorch 1.8.1 (ROCm 4.2)|[**onnxruntime_stable_torch181.rocm42**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch181.rocm42.html)|[onnxruntime_nightly_torch181.rocm42](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch181.rocm42.html)|
|[*Preview*] PyTorch 1.9 (ROCm 4.2)|[**onnxruntime_stable_torch190.rocm42**](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_stable_torch190.rocm42.html)|[onnxruntime_nightly_torch190.rocm42](https://onnxruntimepackages.z14.web.core.windows.net/onnxruntime_nightly_torch190.rocm42.html)|

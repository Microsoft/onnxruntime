<p align="center"><img width="50%" src="docs/images/ONNX_Runtime_logo_dark.png" /></p>

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20CPU%20CI%20Pipeline?label=Windows+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=9)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Windows%20GPU%20CI%20Pipeline?label=Windows+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=10)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20CPU%20CI%20Pipeline?label=Linux+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=11)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/Linux%20GPU%20CI%20Pipeline?label=Linux+GPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=12)
[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status/MacOS%20CI%20Pipeline?label=MacOS+CPU)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=13)

**ONNX Runtime** is a cross-platform model **inferencing and training accelerator** compatible with most popular ML/DNN frameworks, including PyTorch, Tensorflow/Keras, scikit-learn, CoreML, and more.

For data scientists and ML engineers, ONNX runtime provides a high performance solution and cross platform APIs for systems to integrate with a single runtime across a variety of hardware options. Extensibility options support models with custom operators and the execution provider interface enables a growing list of hardware accelerators.

[ONNX Runtime Inferencing](./onnxruntime) APIs are stable and production-ready since the [1.0 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.0.0) in October 2019 and provide inferencing latency acceleration compared with the native framework.

[ONNX Runtime Training](./orttraining) APIs were introduced in May 2020 and currently supports PyTorch training acceleration on NVIDIA GPUs, with more to come soon.

***

# Table of Contents

* **[Overview](#overview)**
  * [Framework Interoperability](#framework-interoperability)
    * [High Performance Model Inferencing](#high-performance-model-inferencing)
    * [Training Acceleration](#training-acceleration)
* **[Get Started](#get-started)**
  * [ONNX Runtime Inferencing](#inferencing-start)
  * [ONNX Runtime Training](#training-start)
* **[Data/Telemetry](#Data/Telemetry)**
* **[Contributions and Feedback](#contribute)**
* **[License](#license)**

***

# Overview

## Framework Interoperability

Supporting models based on the standard [ONNX](https://onnx.ai) format, the runtime is compatible with PyTorch, scikit-learn, Tensorflow, Keras, CoreML, and all other popular frameworks that support [conversion or export to the interoperable format](https://github.com/onnx/tutorials#getting-onnx-models).

ONNX Runtime is up to date and backwards compatible with all operators (both DNN and traditional ML) since ONNX v1.2.1+. [(ONNX opset compatibility details)](docs/Versioning.md). Newer versions of ONNX Runtime support all models that worked with prior versions, so updates should not break integrations.

* [Supported operators/types](./docs/OperatorKernels.md)
  * *Operators not supported in the current ONNX spec may be available as a [Contrib Operator](./docs/ContribOperators.md)*
* [Extensibility: Add a custom operator/kernel](docs/AddingCustomOp.md)

## High Performance Model Inferencing

The inference runtime provides a cross platform API compatible with Windows, Linux, and Mac across a variety of hardware architectures. Using graph optimizations and accelerators, ONNX Runtime can provide lower latency compared to other runtimes for faster end-to-end customer experiences and minimized machine utilization costs.

* [Supported languages and architectures](#apis-and-official-builds)
* [High level architectural design](docs/HighLevelDesign.md)
* [Performance Tuning](./docs/ONNX_Runtime_Perf_Tuning.md)
* [Extensibility: Add a new graph transform](include/onnxruntime/core/optimizer/graph_transformer.h)
* [Extensibility: Add a new rewrite rule](include/onnxruntime/core/optimizer/rewrite_rule.h)

### Supported accelerators ("[Execution Providers](./docs/execution_providers))"

|CPU|GPU|IoT/Edge/Mobile|Other|
|---|---|---|---|
|<ul><li>Default CPU - *MLAS (Microsoft Linear Algebra Subprograms) + Eigen*</li><li>[Intel DNNL](./docs/execution_providers/DNNL-ExecutionProvider.md)</li><li>[Intel nGraph](./docs/execution_providers/nGraph-ExecutionProvider.md)</li><li>Intel MKL-ML</ul>|<ul><li>NVIDIA CUDA</li><li>[NVIDIA TensorRT](./docs/execution_providers/TensorRT-ExecutionProvider.md)</li><li>[DirectML](./docs/execution_providers/DirectML-ExecutionProvider.md)</li></ul>|<ul><li>[Intel OpenVINO](./docs/execution_providers/OpenVINO-ExecutionProvider.md)</li><li>[ARM Compute Library](./docs/execution_providers/ACL-ExecutionProvider.md) (*preview*)</li><li>[Android Neural Networks API](./docs/execution_providers/NNAPI-ExecutionProvider.md) (*preview*)</li></ul>|<ul><li>[Nuphar Model Compiler](./docs/execution_providers/Nuphar-ExecutionProvider.md)</li></ul>|

* **[Get Started with ONNX Runtime Inferencing](inferencing-start)**
* [Roadmap: Upcoming accelerators](./docs/Roadmap.md#accelerators-and-execution-providers)
* [Extensibility: Add an execution provider](docs/AddingExecutionProvider.md)

## Training Acceleration

The training runtime works with existing training code from supported frameworks to accelerate computation of the operators in the model.

The current version supports training acceleration for PyTorch-trained Transformer models on NVIDIA GPUs.

* **[Get Started with ONNX Runtime Training](training-start)**

# Get Started

[Frequently Asked Questions](./docs/FAQ.md)

## Inferencing: Start

To get an ONNX model, see [ONNX Tutorials](https://github.com/onnx/tutorials#getting-onnx-models).
ONNX Runtime supports all versions of ONNX 1.2+. ([Full versioning compatibility](docs/Versioning.md#tool-compatibility))

To use ONNX Runtime, refer to the table on [aka.ms/onnxruntime](https://aka.ms/onnxruntime) for instructions for different build combinations. 

* [Binaries](#binaries)
* [Build from source (includes additional combinations)](#build-from-source)
* [Docker images](#docker-images)
* [API Documentation](#api-documentation)
* [Deploy ONNX Runtime Inferencing](#deploying-onnx-runtime)
* [Samples](./samples)

### Binaries

Official builds are available on PyPi (Python) and Nuget (C#/C/C++):

* Default CPU Provider (Eigen + MLAS)
* GPU Provider - NVIDIA CUDA
* GPU Provider - DirectML (Windows)
  * *On Windows, the [DirectML execution provider](./docs/execution_providers/DirectML-ExecutionProvider.md) is recommended for optimal performance and compatibility with a broad set of GPUs.*

Dev builds created from the master branch are available for testing newer changes between official releases. Please use these at your own risk. We strongly advise against deploying these to production workloads as support is limited for dev builds.

|Pypi (Python)|Nuget (C#/C/C++)|Other package repositories|
|---|---|---|
*If using pip, run `pip install --upgrade pip` prior to downloading.*<br><br>CPU: [**onnxruntime**](https://pypi.org/project/onnxruntime) / [ort-nightly (dev)](https://test.pypi.org/project/ort-nightly)<br><br>GPU: [**onnxruntime-gpu**](https://pypi.org/project/onnxruntime-gpu) / [ort-gpu-nightly (dev)](https://test.pypi.org/project/ort-gpu-nightly) | CPU: [**Microsoft.ML.OnnxRuntime**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime) / [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly) <br><br>GPU: [**Microsoft.ML.OnnxRuntime.Gpu**](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.gpu) / [ort-nightly (dev)](https://aiinfra.visualstudio.com/PublicPackages/_packaging?_a=feed&feed=ORT-Nightly)</li></ul>|[Contributed non-official packages](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-uwp) (including Homebrew, Linuxbrew, and nixpkgs)<br><br>*These are not maintained by the core ONNX Runtime team and may have limited support; use at your discretion.*|

#### System Requirements

* System language
  * Installation of the **English language package** and configuring `en_US.UTF-8 locale` is required, as certain operators makes use of system locales.
  * For Ubuntu, install [language-pack-en package](https://packages.ubuntu.com/search?keywords=language-pack-en)
    * Run the following commands:
      `locale-gen en_US.UTF-8`
      `update-locale LANG=en_US.UTF-8`
    * Follow similar procedure to configure other locales on other platforms.
  
* Default CPU
  * ONNX Runtime binaries in the CPU packages use OpenMP and depend on the library being available at runtime in the system.
    * For Windows, **OpenMP** support comes as part of VC runtime. It is also available as redist packages:
      [vc_redist.x64.exe](https://aka.ms/vs/16/release/vc_redist.x64.exe) and [vc_redist.x86.exe](https://aka.ms/vs/16/release/vc_redist.x86.exe)
    * For Linux, the system must have **libgomp.so.1** which can be installed using `apt-get install libgomp1`.

* Default GPU (CUDA)
  * The default GPU build requires CUDA runtime libraries being installed on the system:
    * Version: **CUDA 10.1** and **cuDNN 7.6.5**
  * Version dependencies from older ONNX Runtime releases can be found in [prior release notes](https://github.com/microsoft/onnxruntime/releases).

### Build from Source

For production scenarios, it's strongly recommended to build only from an [official release branch](https://github.com/microsoft/onnxruntime/releases).

* [Instructions for additional build flavors](./BUILD.md)

### Docker Images

* [ONNX-Ecosystem](https://github.com/onnx/onnx-docker/tree/master/onnx-ecosystem): includes ONNX Runtime (CPU, Python), dependencies, tools to convert from various frameworks, and Jupyter notebooks to help get started
* [Additional dockerfiles](./dockerfiles)

### API Documentation

|API|Supported Versions|Samples|
|---|---|---|
[Python](https://aka.ms/onnxruntime-python)| 3.5, 3.6, 3.7<br>[Python Dev Notes](./docs/Python_Dev_Notes.md)| [Samples](./samples#python)|
|[C#](docs/CSharp_API.md)| | [Samples](./samples#C)|
|[C++](./include/onnxruntime/core/session/onnxruntime_cxx_api.h)| |[Samples](./samples#CC)|
|[C](docs/C_API.md)| | [Samples](./samples#CC)|
|[WinRT](docs/WinRT_API.md) | [Windows.AI.MachineLearning](https://docs.microsoft.com/en-us/windows/ai/windows-ml/api-reference)| [Samples](https://github.com/microsoft/windows-Machine-Learning)|
|[Java](docs/Java_API.md)|8-13|[Samples](./samples#Java)| 
[Ruby](https://github.com/ankane/onnxruntime) (external project)| 2.4-2.7| [Samples](https://ankane.org/tensorflow-ruby)|
|[Javascript (node.js)](./nodejs) | | [Samples](./nodejs/examples/README.md) |

### Deploying ONNX Runtime

#### Cloud

* ONNX Runtime can be deployed to the cloud for model inferencing using [Azure Machine Learning Services](https://azure.microsoft.com/en-us/services/machine-learning-service).
  * [Detailed instructions](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-build-deploy-onnx)
  * [AzureML sample notebooks](https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/deployment/onnx)

* **ONNX Runtime Server (beta)** is a hosted application for serving ONNX models using ONNX Runtime, providing a REST API for prediction.
  * [Usage details](./docs/ONNX_Runtime_Server_Usage.md)
  * [Image installation instructions](./dockerfiles#onnx-runtime-server-preview)

#### IoT and edge devices

* [Reference implementations](https://github.com/Azure-Samples/onnxruntime-iot-edge)

The expanding focus and selection of IoT devices with sensors and consistent signal streams introduces new opportunities to move AI workloads to the edge.
This is particularly important when there are massive volumes of incoming data/signals that may not be efficient or useful to push to the cloud due to storage or latency considerations. Consider: surveillance tapes where 99% of footage is uneventful, or real-time person detection scenarios where immediate action is required. In these scenarios, directly executing model inferencing on the target device is crucial for optimal assistance.

#### Client applications

* Install or build the package you need to use in your application. ([sample implementations](https://github.com/microsoft/onnxruntime/tree/master/samples/c_cxx) using the C++ API)

* On newer Windows 10 devices (1809+), ONNX Runtime is available by default as part of the OS and is accessible via the [Windows Machine Learning APIs](https://docs.microsoft.com/en-us/windows/ai/windows-ml/). ([Tutorials for Windows Desktop or UWP app](https://docs.microsoft.com/en-us/windows/ai/windows-ml/get-started-desktop))

***

## Training: Start

**[End-to-End Sample Notebook]()**

1. Build the ONNX Runtime wheel for training. ([Build instructions](BUILD.md#training))

2. Use ONNX Runtime Training in your PyTorch pre-training script.
High-level code fragment to include in your pre-training code:

  ```python
  import torch
  ...
  import onnxruntime
  from onnxruntime.capi.ort_trainer import IODescription, ModelDescription, ORTTrainer

  # Model definition
  class Net(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
      ...
    def forward(self, x):
      ...

  model = Net(D_in, H, H_out)
  criterion = torch.nn.Functional.cross_entropy
  description = ModelDescription(...)
  optimizer = 'SGDOptimizer'
  trainer = ORTTrainer(model, criterion, description, optimizer, ...)

  # Training Loop
  for t in range(1000):
    # forward + backward + weight update
    loss, y_pred = trainer.train_step(x, y, learning_rate)
    ...
  ```

# Data/Telemetry

This project may collect usage data and send it to Microsoft to help improve our products and services. See the [privacy statement](docs/Privacy.md) for more details.

# Contributions and Feedback

We welcome contributions! Please see the [contribution guidelines](CONTRIBUTING.md).

For any feedback or to report a bug, please file a [GitHub Issue](https://github.com/Microsoft/onnxruntime/issues).

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# License

This project is licensed under the [MIT License](LICENSE).

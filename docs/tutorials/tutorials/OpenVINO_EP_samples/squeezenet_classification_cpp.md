---
nav_exclude: true
---

# Image classification with Squeezenet in CPP using OpenVINO Execution Provider:
{: .no_toc }

1. The image classification uses a Squeezenet Deep Learning ONNX Model from the ONNX Model Zoo.

2. The sample involves presenting an image to the ONNX Runtime (RT), which uses the OpenVINO Execution Provider for ONNX RT to run inference on various Intel hardware devices like Intel CPU, GPU, VPU and more. The sample uses OpenCV for image processing and ONNX Runtime OpenVINO EP for inference. After the sample image is inferred, the terminal will output the predicted label classes in order of their confidence.

The source code for this sample is available [here](https://github.com/microsoft/onnxruntime/tree/master/samples/c_cxx/OpenVINO_EP/squeezenet_classification).

# How to build

## Prerequisites
1. [The Intel<sup>®</sup> Distribution of OpenVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html)

2. Use opencv (use the same opencv package that comes builtin with Intel<sup>®</sup> Distribution of OpenVINO toolkit)

3. Use any sample image as input to the sample.

4. Download the latest Squeezenet model from the ONNX Model Zoo.
   This example was adapted from [ONNX Model Zoo](https://github.com/onnx/models).Download the latest version of the [Squeezenet](https://github.com/onnx/models/tree/master/vision/classification/squeezenet) model from here.

## Install ONNX Runtime for OpenVINO Execution Provider

## Build steps
[build instructions](https://www.onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html#build)


## Reference Documentation
[Documentation](https://www.onnxruntime.ai/docs/reference/execution-providers/OpenVINO-ExecutionProvider.html)

If you build it by yourself, you must append the "--build_shared_lib" flag to your build command. 
```
./build.sh --config Release --use_openvino CPU_FP32 --build_shared_lib 
```

## Build the sample C++ Application

1. Navigate to the directory /onnxruntime/build/Linux/Release/

2. Now copy all the files required to run this sample at this same location (/onnxruntime/build/Linux/Release/)

3. compile the sample

```
g++ -o run_squeezenet squeezenet_cpp_app.cpp -I ../../../include/onnxruntime/core/session/ -I /opt/intel/openvino_2021.4.582/opencv/include/ -I /opt/intel/openvino_2021.4.582/opencv/lib/ -L ./ -lonnxruntime_providers_openvino -lonnxruntime_providers_shared -lonnxruntime -L /opt/intel/openvino_2021.4.582/opencv/lib/ -lopencv_imgcodecs -lopencv_dnn -lopencv_core -lopencv_imgproc
```

Note: This build command is using the opencv location from OpenVINO 2021.4 Release Installation. You can use any version of OpenVINO and change the location path accordingly.

4. Run the sample

    To Run

(using Intel OpenVINO-EP)

```
./run_squeezenet --use_openvino <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
```

Example:

```
./run_squeezenet --use_openvino squeezenet1.1-7.onnx demo.jpeg synset.txt   (using Intel OpenVINO-EP)
```

(using Default CPU)

```
./run_squeezenet --use_cpu <path_to_onnx_model> <path_to_sample_image> <path_to_labels_file>
```

Example:

```
./run_squeezenet --use_cpu squeezenet1.1-7.onnx demo.jpeg synset.txt   (using Default CPU)
```
---
title: Overview
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
has_children: false
nav_order: 1
---
{::options toc_levels="2" /}

# Overview

ONNX Runtime Mobile is a special build of ONNX Runtime that minimizes the binary size and reduces memory usage. It does this primarily by only including specified operators and types in the build, and by saving a pre-optimized ONNX model to an internal format ('ORT format model').

An ONNX model must be converted to an ORT format model to be used with ONNX Runtime Mobile.

![Steps to build for mobile platforms](../../../images/mobile.png){:width="60%"}


There are two options for deploying ONNX Runtime mobile.

* TOC
{:toc}

## Pre-built package 

The pre-built package includes support for selected operators and ONNX opset versions based on the requirements for popular models. If you choose to use the pre-built package you will not need a development environment to perform a custom build of ONNX Runtime, however the binary size will be larger than if you do a custom build with just the operators required by your model/s. Your model can only use the opsets and operators supported by the pre-built package. 


### Available pre-built packages

| Platform | Package location | Included Execution Providers |
|----------|------------------|----------|
| Android | onnxruntime-mobile package in Maven  | CPU Execution Provider <br>NNAPI Execution Provider |
| iOS (preview) | onnxruntime-mobile CocoaPod (C/C++ APIs) <br>onnxruntime-mobile-objc CocoaPod (Objective-C API) | CPU Execution Provider <br>CoreML Execution Provider |

### Operators and Types supported

ONNX operator and types supported by the pre-built package for each ONNX Runtime release are documented [here](../../reference/mobile/prebuilt-package).


## Custom build

Performing a custom build will ensure the smallest possibly binary size and that the build will work with your model/s. You can also choose whether to enable features such as exceptions or traditional ML operators. You will however need a development environment to build for all platforms you're targeting. 


-------
Next: [Initial setup](initial-setup)


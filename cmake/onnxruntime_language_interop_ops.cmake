# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
include(onnxruntime_pyop.cmake)
file (GLOB onnxruntime_language_interop_ops_src "${ONNXRUNTIME_ROOT}/core/language_interop_ops/language_interop_ops.cc")
add_library(onnxruntime_language_interop ${onnxruntime_language_interop_ops_src})
onnxruntime_add_include_to_target(onnxruntime_language_interop onnxruntime_common onnxruntime_framework onnxruntime_pyop gsl onnx onnx_proto protobuf::libprotobuf)
target_include_directories(onnxruntime_language_interop PRIVATE ${ONNXRUNTIME_ROOT} ${eigen_INCLUDE_DIRS})
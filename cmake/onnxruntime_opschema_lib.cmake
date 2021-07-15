# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Schema library (for contrib ops and training ops)

set (CONTRIB_OPS_DIR ${ONNXRUNTIME_ROOT}/core/graph/contrib_ops)
set (TRAINING_OPS_DIR ${ORTTRAINING_ROOT}/orttraining/core/graph)

file(GLOB_RECURSE contrib_ops_schema_src
   "${CONTRIB_OPS_DIR}/*.cc"
   "${TRAINING_OPS_DIR}/training_op_defs.cc"
)

# The nchwc op schemas are platform-specific and not currently required.
list(REMOVE_ITEM contrib_ops_schema_src ${CONTRIB_OPS_DIR}/nchwc_schema_defs.cc)

# Use of ORT_ENFORCE introduces a dependency on GetStackTrace.
# Currently, we just use an empty implementation to avoid bringing in other dependencies.
if(WIN32)
   list(APPEND contrib_ops_schema_src "${ONNXRUNTIME_ROOT}/core/platform/windows/stacktrace.cc")
else()
   list(APPEND contrib_ops_schema_src "${ONNXRUNTIME_ROOT}/core/platform/posix/stacktrace.cc")
endif()

onnxruntime_add_static_library(ort_opschema_lib ${contrib_ops_schema_src})
target_compile_options(ort_opschema_lib PRIVATE -D_OPSCHEMA_LIB_=1)

set (OPSCHEMA_LIB_DEPENDENCIES onnx onnx_proto protobuf::libprotobuf flatbuffers)

# ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" is found
target_include_directories(ort_opschema_lib PRIVATE ${ONNXRUNTIME_ROOT} ${ORTTRAINING_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
onnxruntime_add_include_to_target(ort_opschema_lib onnxruntime_common onnx onnx_proto protobuf::libprotobuf flatbuffers)
add_dependencies(ort_opschema_lib ${OPSCHEMA_LIB_DEPENDENCIES})


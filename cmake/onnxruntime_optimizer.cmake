# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB onnxruntime_optimizer_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.h"
    "${ONNXRUNTIME_ROOT}/core/optimizer/*.cc"
    )

if (onnxruntime_ENABLE_TRAINING)
    file(GLOB orttraining_optimizer_srcs CONFIGURE_DEPENDS
        "${ORTTRAINING_SOURCE_DIR}/core/optimizer/*.h"
        "${ORTTRAINING_SOURCE_DIR}/core/optimizer/*.cc"
        )
    set(onnxruntime_optimizer_srcs ${onnxruntime_optimizer_srcs} ${orttraining_optimizer_srcs})
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_optimizer_srcs})

add_library(onnxruntime_optimizer ${onnxruntime_optimizer_srcs})
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/optimizer  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
onnxruntime_add_include_to_target(onnxruntime_optimizer onnxruntime_common onnxruntime_framework onnx onnx_proto protobuf::libprotobuf safeint_interface)
if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
   target_compile_options(onnxruntime_optimizer PRIVATE "/wd4244")   
endif()
target_include_directories(onnxruntime_optimizer PRIVATE ${ONNXRUNTIME_ROOT})
if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_optimizer PRIVATE ${ORTTRAINING_ROOT})
endif()
add_dependencies(onnxruntime_optimizer ${onnxruntime_EXTERNAL_DEPENDENCIES})
set_target_properties(onnxruntime_optimizer PROPERTIES FOLDER "ONNXRuntime")

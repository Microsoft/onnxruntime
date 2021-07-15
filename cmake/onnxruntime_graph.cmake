# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_graph_src CONFIGURE_DEPENDS
  "${ONNXRUNTIME_INCLUDE_DIR}/core/graph/*.h"
  "${ONNXRUNTIME_ROOT}/core/graph/*.h"
  "${ONNXRUNTIME_ROOT}/core/graph/*.cc"
  )

# create empty list for any excludes
set(onnxruntime_graph_src_exclude_patterns)

if (onnxruntime_MINIMAL_BUILD)
  # remove schema registration support
  list(APPEND onnxruntime_graph_src_exclude_patterns
    "${ONNXRUNTIME_INCLUDE_DIR}/core/graph/schema_registry.h"
    "${ONNXRUNTIME_ROOT}/core/graph/schema_registry.cc"
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*defs.h"
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*defs.cc"
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/onnx_function_util.h"
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/onnx_function_util.cc"
  )

  # no Function support initially
  list(APPEND onnxruntime_graph_src_exclude_patterns
    "${ONNXRUNTIME_ROOT}/core/graph/function*"
  )

  # no optimizer support initially
  list(APPEND onnxruntime_graph_src_exclude_patterns
    "${ONNXRUNTIME_ROOT}/core/graph/graph_utils.*"
  )
endif()

if (onnxruntime_DISABLE_CONTRIB_OPS)
  list(APPEND onnxruntime_graph_src_exclude_patterns
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*.h"
    "${ONNXRUNTIME_ROOT}/core/graph/contrib_ops/*.cc"
    )
endif()

if(NOT onnxruntime_USE_FEATURIZERS)
  list(APPEND onnxruntime_graph_src_exclude_patterns
    "${ONNXRUNTIME_ROOT}/core/graph/featurizers_ops/*.h"
    "${ONNXRUNTIME_ROOT}/core/graph/featurizers_ops/*.cc"
  )
endif()

if(NOT onnxruntime_USE_DML)
  list(APPEND onnxruntime_graph_src_exclude_patterns
    "${ONNXRUNTIME_ROOT}/core/graph/dml_ops/*.h"
    "${ONNXRUNTIME_ROOT}/core/graph/dml_ops/*.cc"
    )
endif()

file(GLOB onnxruntime_graph_src_exclude ${onnxruntime_graph_src_exclude_patterns})
list(REMOVE_ITEM onnxruntime_graph_src ${onnxruntime_graph_src_exclude})

file(GLOB_RECURSE onnxruntime_ir_defs_src CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/defs/*.cc"
)

if (onnxruntime_ENABLE_TRAINING_OPS AND NOT onnxruntime_ENABLE_TRAINING)
  set(orttraining_graph_src
      "${ORTTRAINING_SOURCE_DIR}/core/graph/training_op_defs.cc"
      "${ORTTRAINING_SOURCE_DIR}/core/graph/training_op_defs.h"
      )
endif()
if (onnxruntime_ENABLE_TRAINING)
  file(GLOB_RECURSE orttraining_graph_src CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/core/graph/*.h"
      "${ORTTRAINING_SOURCE_DIR}/core/graph/*.cc"
      )
endif()

set(onnxruntime_graph_lib_src ${onnxruntime_graph_src} ${onnxruntime_ir_defs_src})
if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    list(APPEND onnxruntime_graph_lib_src ${orttraining_graph_src})
endif()

onnxruntime_add_static_library(onnxruntime_graph ${onnxruntime_graph_lib_src})
add_dependencies(onnxruntime_graph onnx_proto flatbuffers)
onnxruntime_add_include_to_target(onnxruntime_graph onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers)

if (onnxruntime_ENABLE_TRAINING)
  #TODO: the graph library should focus on ONNX IR, it shouldn't depend on math libraries like MKLML/OpenBlas
  target_include_directories(onnxruntime_graph PRIVATE ${MKLML_INCLUDE_DIR})
  target_link_libraries(onnxruntime_graph PRIVATE nlohmann_json::nlohmann_json)
endif()

target_include_directories(onnxruntime_graph PRIVATE ${ONNXRUNTIME_ROOT})

if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(onnxruntime_graph PRIVATE ${ORTTRAINING_ROOT})

    if (onnxruntime_USE_NCCL)
        target_include_directories(onnxruntime_graph PRIVATE ${NCCL_INCLUDE_DIRS})
    endif()
endif()

set_target_properties(onnxruntime_graph PROPERTIES FOLDER "ONNXRuntime")
set_target_properties(onnxruntime_graph PROPERTIES LINKER_LANGUAGE CXX)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/graph  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_graph_src} ${onnxruntime_ir_defs_src})
if (onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_TRAINING_OPS)
    source_group(TREE ${ORTTRAINING_ROOT} FILES ${orttraining_graph_src})
endif()

if (onnxruntime_BUILD_MS_EXPERIMENTAL_OPS)
  target_compile_definitions(onnxruntime_graph PRIVATE BUILD_MS_EXPERIMENTAL_OPS=1)
endif()

if (WIN32)
  set(onnxruntime_graph_static_library_flags
      -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
  )

  set_target_properties(onnxruntime_graph PROPERTIES
      STATIC_LIBRARY_FLAGS "${onnxruntime_graph_static_library_flags}")

  if (NOT onnxruntime_DISABLE_EXCEPTIONS)  
    target_compile_options(onnxruntime_graph PRIVATE
        /EHsc   # exception handling - C++ may throw, extern "C" will not
    )
  endif()

  # Add Code Analysis properties to enable C++ Core checks. Have to do it via a props file include.
  set_target_properties(onnxruntime_graph PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
endif()

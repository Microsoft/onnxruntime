# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_framework_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.h"
    "${ONNXRUNTIME_ROOT}/core/framework/*.cc"
)

if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  file(GLOB_RECURSE onnxruntime_training_framework_torch_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.h"
      "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.cc"
  )
  list(APPEND onnxruntime_framework_srcs ${onnxruntime_training_framework_torch_srcs})
  if (onnxruntime_ENABLE_TRITON)
    file(GLOB_RECURSE onnxruntime_training_framework_triton_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.h"
      "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.cc"
    )
    list(APPEND onnxruntime_framework_srcs ${onnxruntime_training_framework_triton_srcs})
  endif()
elseif(onnxruntime_ENABLE_TRITON)
  # Triton executor shares some code from torch_interop, such as python and dlpack related code files.
  # When torch_interop is enabled, all these dependencies are already included.
  # But if not, we need to include them explicitly.
  file(GLOB_RECURSE onnxruntime_training_framework_triton_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/dlpack_python.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/dlpack_python.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/gil.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/python_common.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.cc"
  )
  list(APPEND onnxruntime_framework_srcs ${onnxruntime_training_framework_triton_srcs})
endif()

if (onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_framework_src_exclude
    "${ONNXRUNTIME_ROOT}/core/framework/kernel_pilot.h"
    "${ONNXRUNTIME_ROOT}/core/framework/kernel_pilot.cc"
    "${ONNXRUNTIME_ROOT}/core/framework/kernel_pilot_moe_expert_selection.h"
    "${ONNXRUNTIME_ROOT}/core/framework/kernel_pilot_moe_expert_selection.cc"
    "${ONNXRUNTIME_ROOT}/core/framework/kernel_pilot_moe_expert_state.h"
    "${ONNXRUNTIME_ROOT}/core/framework/kernel_pilot_moe_expert_state.cc"
  )

  # custom ops support must be explicitly enabled in a minimal build. exclude if not.
  if (NOT onnxruntime_MINIMAL_BUILD_CUSTOM_OPS)
    list(APPEND onnxruntime_framework_src_exclude
      "${ONNXRUNTIME_INCLUDE_DIR}/core/framework/customregistry.h"
      "${ONNXRUNTIME_ROOT}/core/framework/customregistry.cc"
    )
  endif()

  list(REMOVE_ITEM onnxruntime_framework_srcs ${onnxruntime_framework_src_exclude})
endif()

source_group(TREE ${REPO_ROOT} FILES ${onnxruntime_framework_srcs})

onnxruntime_add_static_library(onnxruntime_framework ${onnxruntime_framework_srcs})

if (MSVC)
  set(ORT_FRAMEWORK_NATVIS_FILE "onnxruntime_framework.natvis")
  target_sources(
      onnxruntime_framework
      INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/${ORT_FRAMEWORK_NATVIS_FILE}>)
endif()

if(onnxruntime_ENABLE_INSTRUMENT)
  target_compile_definitions(onnxruntime_framework PRIVATE ONNXRUNTIME_ENABLE_INSTRUMENT)
endif()
if(onnxruntime_USE_TENSORRT OR onnxruntime_USE_NCCL)
# TODO: for now, core framework depends on CUDA. It should be moved to TensorRT EP
# TODO: provider_bridge_ort.cc should not include nccl.h
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
else()
target_include_directories(onnxruntime_framework PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
endif()
# Needed for the provider interface, as it includes training headers when training is enabled
if (onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(onnxruntime_framework PRIVATE ${ORTTRAINING_ROOT})
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OR onnxruntime_ENABLE_TRITON)
    onnxruntime_add_include_to_target(onnxruntime_framework Python::Module dlpack::dlpack)
  endif()
endif()

if (onnxruntime_ENABLE_ATEN)
  onnxruntime_add_include_to_target(onnxruntime_framework dlpack::dlpack)
endif()
onnxruntime_add_include_to_target(onnxruntime_framework onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers safeint_interface Boost::mp11 nlohmann_json::nlohmann_json Eigen3::Eigen)

if (onnxruntime_USE_MIMALLOC)
    onnxruntime_add_include_to_target(onnxruntime_framework mimalloc-static)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  target_link_libraries(onnxruntime_framework ${ABSEIL_LIBS})
endif()

set_target_properties(onnxruntime_framework PROPERTIES FOLDER "ONNXRuntime")
# need onnx to build to create headers that this project includes
add_dependencies(onnxruntime_framework ${onnxruntime_EXTERNAL_DEPENDENCIES})

# In order to find the shared provider libraries we need to add the origin to the rpath for all executables we build
# For the shared onnxruntime library, this is set in onnxruntime.cmake through CMAKE_SHARED_LINKER_FLAGS
# But our test files don't use the shared library so this must be set for them.
# For Win32 it generates an absolute path for shared providers based on the location of the executable/onnxruntime.dll
if (UNIX AND NOT APPLE AND NOT onnxruntime_MINIMAL_BUILD AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath='$ORIGIN'")
endif()

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB)
  find_package (SQLite3 REQUIRED)
  include_directories(${SQLite3_INCLUDE_DIR})
  target_link_libraries (onnxruntime_framework ${SQLite3_LIBRARY})
  target_compile_definitions(onnxruntime_framework PRIVATE DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB)
endif()

if (WIN32)
  target_compile_definitions(onnxruntime_framework PRIVATE _SCL_SECURE_NO_WARNINGS)
endif()

if (onnxruntime_BUILD_SHARED_LIB)
  install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/framework/provider_options.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)
else()
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/framework  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core)
  install(TARGETS onnxruntime_framework EXPORT ${PROJECT_NAME}Targets
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

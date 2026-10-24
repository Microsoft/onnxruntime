# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

file(GLOB_RECURSE onnxruntime_providers_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/cpu/*.cc"
)

if(onnxruntime_DISABLE_ML_OPS)
  list(FILTER onnxruntime_providers_srcs EXCLUDE REGEX ".*/ml/.*")
endif()

file(GLOB_RECURSE onnxruntime_cpu_contrib_ops_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/*.cc"
)
list(REMOVE_ITEM onnxruntime_cpu_contrib_ops_srcs
  "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/moe/qmoe_shared_layout_avx512.cc"
)
if(onnxruntime_ENABLE_MLAS_QMOE_OFFLOAD AND NOT onnxruntime_DISABLE_CONTRIB_OPS)
  list(REMOVE_ITEM onnxruntime_cpu_contrib_ops_srcs
    "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/moe/qmoe_shared_layout.cc"
  )
endif()

file(GLOB_RECURSE onnxruntime_js_contrib_ops_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/js/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/js/*.cc"
)

file(GLOB_RECURSE onnxruntime_webgpu_contrib_ops_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/contrib_ops/webgpu/*.h"
  "${ONNXRUNTIME_ROOT}/contrib_ops/webgpu/*.cc"
)

file(GLOB onnxruntime_providers_common_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/*.cc"
  "${ONNXRUNTIME_ROOT}/core/providers/op_kernel_type_control_overrides.inc"
)


source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

set(onnxruntime_providers_src ${onnxruntime_providers_common_srcs} ${onnxruntime_providers_srcs})

# disable contrib ops conditionally
if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
  if (NOT onnxruntime_ENABLE_ATEN)
    list(REMOVE_ITEM onnxruntime_cpu_contrib_ops_srcs
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/aten_ops/aten_op.h"
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/aten_ops/aten_op.cc"
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/aten_ops/aten_op_executor.cc"
    )
  endif()
  # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
  source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cpu_contrib_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_contrib_ops_srcs})
endif()

if (onnxruntime_ENABLE_TRAINING_OPS AND NOT onnxruntime_ENABLE_TRAINING)
  file(GLOB_RECURSE onnxruntime_cpu_training_ops_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.cc"
  )

  source_group(TREE ${ORTTRAINING_ROOT}/ FILES ${onnxruntime_cpu_training_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_training_ops_srcs})

  file(GLOB_RECURSE onnxruntime_cpu_full_training_only_srcs
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/communication/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/communication/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/record.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/record.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/wait.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/wait.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/yield.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/controlflow/yield.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/gist/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/gist/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/tensorboard/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/tensorboard/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/torch/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/torch/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/triton/triton_op.cc"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/triton/triton_op.h"
  )

  list(REMOVE_ITEM onnxruntime_providers_src ${onnxruntime_cpu_full_training_only_srcs})
endif()

if (onnxruntime_ENABLE_DLPACK)
  file(GLOB_RECURSE onnxruntime_providers_dlpack_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/dlpack/dlpack_converter.cc"
    "${ONNXRUNTIME_ROOT}/core/dlpack/dlpack_converter.h"
  )
  set(onnxruntime_providers_dlpack_srcs ${onnxruntime_providers_dlpack_srcs})
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_dlpack_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_providers_dlpack_srcs})
endif()

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB_RECURSE onnxruntime_cpu_training_ops_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.h"
    "${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/*.h"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/*.cc"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/adasum/*"
    "${ORTTRAINING_SOURCE_DIR}/core/framework/communication/*"
  )

  # This is already built in framework.cmake
  file(GLOB_RECURSE onnxruntime_training_framework_excude_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.h"
      "${ORTTRAINING_SOURCE_DIR}/core/framework/torch/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.h"
      "${ORTTRAINING_SOURCE_DIR}/core/framework/triton/*.cc"
  )

  list(REMOVE_ITEM onnxruntime_cpu_training_ops_srcs ${onnxruntime_training_framework_excude_srcs})

  source_group(TREE ${ORTTRAINING_ROOT}/ FILES ${onnxruntime_cpu_training_ops_srcs})
  list(APPEND onnxruntime_providers_src ${onnxruntime_cpu_training_ops_srcs})
endif()

if (onnxruntime_REDUCED_OPS_BUILD)
  substitute_op_reduction_srcs(onnxruntime_providers_src)
endif()
# Only one of these targets exists per configuration, and consumers link
# ${onnxruntime_providers_target} rather than either name, so the choice lives here alone.
#
# With onnxruntime_ENABLE_LTO on MSVC the objects carry whole-program IR, which pushes the archive
# past the 4 GiB limit lib.exe enforces on the COFF archive format (LNK1248). No lib.exe option
# raises the limit, and a shared-library build only needs the objects for the final link, so no
# archive is produced there.
#
# The archive cannot instead be materialized from $<TARGET_OBJECTS:...> of an always-present object
# library: the Xcode generator hash-suffixes object filenames when source basenames collide, as
# some of these sources do, while TARGET_OBJECTS expands to the unsuffixed names, so libtool fails
# to find them.
if (MSVC AND onnxruntime_BUILD_SHARED_LIB)
  onnxruntime_add_object_library(onnxruntime_providers_obj ${onnxruntime_providers_src})
  set(onnxruntime_providers_target onnxruntime_providers_obj)
else()
  onnxruntime_add_static_library(onnxruntime_providers ${onnxruntime_providers_src})
  set(onnxruntime_providers_target onnxruntime_providers)
endif()
if (onnxruntime_REDUCED_OPS_BUILD)
  add_op_reduction_include_dirs(${onnxruntime_providers_target})
endif()

if (HAS_BITWISE_INSTEAD_OF_LOGICAL)
  target_compile_options(${onnxruntime_providers_target} PRIVATE "-Wno-bitwise-instead-of-logical")
endif()

if (MSVC)
   target_compile_options(${onnxruntime_providers_target} PRIVATE "/bigobj")
#   if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
#      target_compile_options(${onnxruntime_providers_target} PRIVATE "/wd4244")
#   endif()
endif()
onnxruntime_add_include_to_target(${onnxruntime_providers_target} onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers Boost::mp11 safeint_interface)

if (onnxruntime_BUILD_MS_EXPERIMENTAL_OPS)
  target_compile_definitions(${onnxruntime_providers_target} PRIVATE BUILD_MS_EXPERIMENTAL_OPS=1)
endif()

if(HAS_DEPRECATED_COPY)
  #temporarily ignore this warning
  #see: https://en.wikipedia.org/wiki/Rule_of_three_(C%2B%2B_programming)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/math/matmul_integer.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/math/quantize_linear_matmul.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/qlinearconv.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/nn/conv_integer.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/generator/random.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/onehot.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
  set_source_files_properties("${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/where_op.cc" PROPERTIES COMPILE_FLAGS -Wno-deprecated-copy)
endif()

# This is enabled only for Adasum files in training mode.
# The flags won't be applied globally since some high-precision training and inferencing ops will incur precision loss.
if (onnxruntime_ENABLE_CPU_FP16_OPS)
  set_source_files_properties(${ORTTRAINING_SOURCE_DIR}/core/framework/adasum/adasum_mpi.cc PROPERTIES COMPILE_FLAGS " -fassociative-math -ffast-math -ftree-vectorize -funsafe-math-optimizations -mf16c -mavx -mfma ")
  set_source_files_properties(${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/adasum_kernels.cc PROPERTIES COMPILE_FLAGS " -fassociative-math -ffast-math -ftree-vectorize -funsafe-math-optimizations -mf16c -mavx -mfma ")
  set_source_files_properties(${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/adasum_kernels.cc PROPERTIES COMPILE_FLAGS " -fassociative-math -ffast-math -ftree-vectorize -funsafe-math-optimizations -mf16c -mavx -mfma ")
endif()

target_include_directories(${onnxruntime_providers_target} PRIVATE ${ONNXRUNTIME_ROOT})
onnxruntime_add_include_to_target(${onnxruntime_providers_target} re2::re2 Eigen3::Eigen)
add_dependencies(${onnxruntime_providers_target} onnx ${onnxruntime_EXTERNAL_DEPENDENCIES})

if (onnxruntime_ENABLE_TRAINING_OPS)
  target_include_directories(${onnxruntime_providers_target} PRIVATE ${ORTTRAINING_ROOT})
endif()

if (onnxruntime_ENABLE_ATEN)
  target_compile_definitions(${onnxruntime_providers_target} PRIVATE ENABLE_ATEN)
endif()

if (onnxruntime_ENABLE_DLPACK)
  target_compile_definitions(${onnxruntime_providers_target} PRIVATE ENABLE_DLPACK)
  # DLPack is a header-only dependency
  onnxruntime_add_include_to_target(${onnxruntime_providers_target} dlpack::dlpack)
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_dependencies(${onnxruntime_providers_target} tensorboard)
  onnxruntime_add_include_to_target(${onnxruntime_providers_target} tensorboard)
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OR onnxruntime_ENABLE_TRITON)
    onnxruntime_add_include_to_target(${onnxruntime_providers_target} Python::Module)
  endif()

  if (onnxruntime_USE_NCCL)
    target_include_directories(${onnxruntime_providers_target} PUBLIC ${MPI_CXX_INCLUDE_DIRS})
  endif()
endif()

install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/cpu/cpu_provider_factory.h  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/)
install(FILES ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/resource.h ${PROJECT_SOURCE_DIR}/../include/onnxruntime/core/providers/custom_op_context.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers)
set_target_properties(${onnxruntime_providers_target} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${onnxruntime_providers_target} PROPERTIES FOLDER "ONNXRuntime")

if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD
                                  AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS|visionOS|tvOS"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android"
                                  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  file(GLOB onnxruntime_providers_shared_cc_srcs CONFIGURE_DEPENDS
  "${ONNXRUNTIME_ROOT}/core/providers/shared/*.h"
  "${ONNXRUNTIME_ROOT}/core/providers/shared/*.cc"
  )

  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_shared_cc_srcs})
  onnxruntime_add_shared_library(onnxruntime_providers_shared ${onnxruntime_providers_shared_cc_srcs} "${ONNXRUNTIME_ROOT}/core/dll/onnxruntime.rc")
  set_target_properties(onnxruntime_providers_shared PROPERTIES FOLDER "ONNXRuntime")
  set_target_properties(onnxruntime_providers_shared PROPERTIES LINKER_LANGUAGE CXX)
  #ON AIX, to call dlopen on onnxruntime_providers_shared, we need to generate this library as shared object .so file.
  #Latest cmake behavior is changed and  cmake will remove shared object .so file after generating the shared archive.
  #To prevent that, making AIX_SHARED_LIBRARY_ARCHIVE as OFF for onnxruntime_providers_shared.
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    set_target_properties(onnxruntime_providers_shared PROPERTIES AIX_SHARED_LIBRARY_ARCHIVE OFF)
  endif()

  target_compile_definitions(onnxruntime_providers_shared PRIVATE FILE_NAME=\"onnxruntime_providers_shared.dll\")


  # On Apple/Unix we don't directly link with this library as we load it with RTLD_GLOBAL, so this is only set to the actual library on WIN32
  # But, in exchange we need to manually add Boost::mp11 to include dirs for every EP.
  # It is because "provider_api.h" includes core/framework/op_kernel.h which includes op_kernel.h which includes "boost/mp11.hpp"
  set(ONNXRUNTIME_PROVIDERS_SHARED)

  if(APPLE)
  set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/shared/exported_symbols.lst")
  elseif(UNIX)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      target_link_options(onnxruntime_providers_shared PRIVATE
                          "LINKER:--version-script=${ONNXRUNTIME_ROOT}/core/providers/shared/version_script.lds"
                          "LINKER:--gc-sections")
    endif()
  elseif(WIN32)
  set_property(TARGET onnxruntime_providers_shared APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/shared/symbols.def")
  set(ONNXRUNTIME_PROVIDERS_SHARED onnxruntime_providers_shared)
  else()
  message(FATAL_ERROR "onnxruntime_providers_shared unknown platform, need to specify shared library exports for it")
  endif()

  install(TARGETS onnxruntime_providers_shared
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS onnxruntime_providers EXPORT ${PROJECT_NAME}Targets
          ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
          FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

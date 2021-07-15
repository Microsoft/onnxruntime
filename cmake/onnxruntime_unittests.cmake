# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
  find_package(XCTest REQUIRED)
endif()

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT})
if (onnxruntime_ENABLE_TRAINING)
  list(APPEND TEST_INC_DIR ${ORTTRAINING_ROOT})
endif()
if (onnxruntime_USE_TVM)
  list(APPEND TEST_INC_DIR ${TVM_INCLUDES})
endif()

set(disabled_warnings)
function(AddTest)
  cmake_parse_arguments(_UT "DYN" "TARGET" "LIBS;SOURCES;DEPENDS" ${ARGN})
  list(REMOVE_DUPLICATES _UT_SOURCES)

  if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    onnxruntime_add_executable(${_UT_TARGET} ${TEST_SRC_DIR}/xctest/orttestmain.m)
  else()
    onnxruntime_add_executable(${_UT_TARGET} ${_UT_SOURCES})
  endif()

  if (_UT_DEPENDS)
    list(REMOVE_DUPLICATES _UT_DEPENDS)
  endif(_UT_DEPENDS)

  if(_UT_LIBS)
    list(REMOVE_DUPLICATES _UT_LIBS)
  endif()

  source_group(TREE ${REPO_ROOT} FILES ${_UT_SOURCES})

  if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings, they are dangerous
    target_compile_options(${_UT_TARGET} PRIVATE "/wd4244")
  endif()
  if (MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "/wd6330")
  endif()
  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest")

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)
  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
            Threads::Threads)
    target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_ONNXRUNTIME_DLL)
  else()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()
  onnxruntime_add_include_to_target(${_UT_TARGET} date_interface flatbuffers)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
  if (onnxruntime_USE_CUDA)
    target_include_directories(${_UT_TARGET} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${onnxruntime_CUDNN_HOME}/include)
    if (onnxruntime_USE_NCCL)
      target_include_directories(${_UT_TARGET} PRIVATE ${NCCL_INCLUDE_DIRS})
    endif()
  endif()
  if(MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()
  if (WIN32)
    # include dbghelp in case tests throw an ORT exception, as that exception includes a stacktrace, which requires dbghelp.
    target_link_libraries(${_UT_TARGET} PRIVATE debug dbghelp)

    if (onnxruntime_USE_CUDA)
      # disable a warning from the CUDA headers about unreferenced local functions
      if (MSVC)
        target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4505>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4505>")
      endif()
    endif()
    if (MSVC)
      # warning C6326: Potential comparison of a constant with another constant.
      # Lot of such things came from gtest
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
    endif()
    target_compile_options(${_UT_TARGET} PRIVATE ${disabled_warnings})
  else()
    target_compile_options(${_UT_TARGET} PRIVATE ${DISABLED_WARNINGS_FOR_TVM})
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    target_compile_options(${_UT_TARGET} PRIVATE "-Wno-error=uninitialized")
  endif()

  set(TEST_ARGS)
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    if (onnxruntime_BUILD_WEBASSEMBLY)
      # WebAssembly use a memory file system, so we do not use full path
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<TARGET_FILE_NAME:${_UT_TARGET}>.$<CONFIG>.results.xml")
    else()
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
    endif()
  endif(onnxruntime_GENERATE_TEST_REPORTS)

  if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    # target_sources(${_UT_TARGET} PRIVATE ${TEST_SRC_DIR}/xctest/orttestmain.m)
    set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}
      MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.${_UT_TARGET}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_CLANG_ENABLE_MODULES "YES"
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO"
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO")

    xctest_add_bundle(${_UT_TARGET}_xc ${_UT_TARGET}
      ${TEST_SRC_DIR}/xctest/ortxctest.m
      ${TEST_SRC_DIR}/xctest/xcgtest.mm
      ${_UT_SOURCES})

    if(_UT_DYN)
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
              Threads::Threads)
      target_compile_definitions(${_UT_TARGET}_xc PRIVATE USE_ONNXRUNTIME_DLL)
    else()
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
    endif()
    onnxruntime_add_include_to_target(${_UT_TARGET}_xc date_interface flatbuffers)
    target_include_directories(${_UT_TARGET}_xc PRIVATE ${TEST_INC_DIR})
    get_target_property(${_UT_TARGET}_DEFS ${_UT_TARGET} COMPILE_DEFINITIONS)
    target_compile_definitions(${_UT_TARGET}_xc PRIVATE ${_UT_TARGET}_DEFS)

    set_target_properties(${_UT_TARGET}_xc PROPERTIES FOLDER "ONNXRuntimeXCTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}_xc
      MACOSX_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.${_UT_TARGET}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")

    xctest_add_test(xctest.${_UT_TARGET} ${_UT_TARGET}_xc)
  else()
    if (onnxruntime_BUILD_WEBASSEMBLY)
      find_program(NODE_EXECUTABLE node required)
      if (NOT NODE_EXECUTABLE)
        message(FATAL_ERROR "Node is required for unit tests")
      endif()

      set(TEST_NODE_FLAGS)
      if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
        list(APPEND TEST_NODE_FLAGS "--experimental-wasm-threads")
        list(APPEND TEST_NODE_FLAGS "--experimental-wasm-bulk-memory")
      endif()
      if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
        list(APPEND TEST_NODE_FLAGS "--experimental-wasm-simd")
      endif()

      add_test(NAME ${_UT_TARGET}
        COMMAND ${NODE_EXECUTABLE} ${TEST_NODE_FLAGS} ${_UT_TARGET}.js ${TEST_ARGS}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
      )
    else()
      add_test(NAME ${_UT_TARGET}
        COMMAND ${_UT_TARGET} ${TEST_ARGS}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
      )
    endif()
  endif()
endfunction(AddTest)

# general program entrypoint for C++ unit tests
set(onnxruntime_unittest_main_src "${TEST_SRC_DIR}/unittest_main/test_main.cc")

#Do not add '${TEST_SRC_DIR}/util/include' to your include directories directly
#Use onnxruntime_add_include_to_target or target_link_libraries, so that compile definitions
#can propagate correctly.

file(GLOB onnxruntime_test_utils_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/util/include/*.h"
  "${TEST_SRC_DIR}/util/*.cc"
)

file(GLOB onnxruntime_test_common_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/common/*.cc"
  "${TEST_SRC_DIR}/common/*.h"
  "${TEST_SRC_DIR}/common/logging/*.cc"
  "${TEST_SRC_DIR}/common/logging/*.h"
)

file(GLOB onnxruntime_test_quantiztion_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/quantization/*.cc"
  "${TEST_SRC_DIR}/quantization/*.h"
)

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/ir/*.cc"
    "${TEST_SRC_DIR}/ir/*.h"
    )

  file(GLOB onnxruntime_test_optimizer_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/optimizer/*.cc"
    "${TEST_SRC_DIR}/optimizer/*.h"
    )

  set(onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/framework/*.cc"
    "${TEST_SRC_DIR}/framework/*.h"
    "${TEST_SRC_DIR}/platform/*.cc"
    )

else()  # minimal and/or reduced ops build

  set(onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/*.cc"
    )

  if (onnxruntime_MINIMAL_BUILD)
    list(APPEND onnxruntime_test_framework_src_patterns
      "${TEST_SRC_DIR}/framework/ort_model_only_test.cc"
    )

  else() # reduced ops build
    file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/ir/*.cc"
      "${TEST_SRC_DIR}/ir/*.h"
      )
  endif()
endif()

file(GLOB onnxruntime_test_training_src
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/framework/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.cc"
  )

if(WIN32)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
endif()

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  if(onnxruntime_USE_CUDA)
    list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
  endif()

  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/providers/*.h"
    "${TEST_SRC_DIR}/providers/*.cc"
    "${TEST_SRC_DIR}/opaque_api/test_opaque_api.cc"
    "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
    "${TEST_SRC_DIR}/framework/TestAllocatorManager.h"
    "${TEST_SRC_DIR}/framework/test_utils.cc"
    "${TEST_SRC_DIR}/framework/test_utils.h"
  )

  if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    list(APPEND onnxruntime_test_providers_src_patterns
      "${TEST_SRC_DIR}/contrib_ops/*.h"
      "${TEST_SRC_DIR}/contrib_ops/*.cc")
  endif()

  if(onnxruntime_USE_FEATURIZERS)
    list(APPEND onnxruntime_test_providers_src_patterns
      "${TEST_SRC_DIR}/featurizers_ops/*.h"
      "${TEST_SRC_DIR}/featurizers_ops/*.cc")
  endif()

else()
  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/framework/test_utils.cc"
    "${TEST_SRC_DIR}/framework/test_utils.h"
    # TODO: Add anything that is needed for testing a minimal build
  )
endif()

file(GLOB onnxruntime_test_providers_src CONFIGURE_DEPENDS ${onnxruntime_test_providers_src_patterns})

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_cpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cpu/*"
    )
endif()

if(onnxruntime_DISABLE_ML_OPS)
  list(FILTER onnxruntime_test_providers_cpu_src EXCLUDE REGEX ".*/ml/.*")
endif()

list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cpu_src})

if (onnxruntime_USE_CUDA AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_cuda_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_src})
endif()

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB_RECURSE orttraining_test_trainingops_cpu_src CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/test/training_ops/compare_provider_test_utils.cc"
    "${ORTTRAINING_SOURCE_DIR}/test/training_ops/function_op_test_utils.cc"
    "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/*"
    )
  list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cpu_src})

  if (onnxruntime_USE_CUDA OR onnxruntime_USE_ROCM)
    file(GLOB_RECURSE orttraining_test_trainingops_cuda_src CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cuda/*"
      )
    list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cuda_src})
  endif()
endif()

if (onnxruntime_USE_DNNL)
  file(GLOB_RECURSE onnxruntime_test_providers_dnnl_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/dnnl/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_dnnl_src})
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  file(GLOB_RECURSE onnxruntime_test_providers_nnapi_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/nnapi/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_nnapi_src})
endif()

if (onnxruntime_USE_RKNPU)
  file(GLOB_RECURSE onnxruntime_test_providers_rknpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/rknpu/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_rknpu_src})
endif()

if (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_internal_testing_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/internal_testing/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_internal_testing_src})
endif()

set (ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR "${TEST_SRC_DIR}/shared_lib")
set (ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR "${TEST_SRC_DIR}/global_thread_pools")
set (ONNXRUNTIME_API_TESTS_WITHOUT_ENV_SRC_DIR "${TEST_SRC_DIR}/api_tests_without_env")

set (onnxruntime_shared_lib_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_session_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_run_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_allocator.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_nontensor_types.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_loading.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_ort_format_models.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.cc)

if (NOT onnxruntime_MINIMAL_BUILD)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc)
endif()

if(onnxruntime_RUN_ONNX_TESTS)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_io_types.cc)
endif()

set (onnxruntime_global_thread_pools_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_main.cc
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_inference.cc)

set (onnxruntime_api_tests_without_env_SRC
          ${ONNXRUNTIME_API_TESTS_WITHOUT_ENV_SRC_DIR}/test_apis_without_env.cc)

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  onnxruntime_test_utils
  onnxruntime_common
)

set(onnxruntime_test_ir_libs
  onnxruntime_test_utils
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_optimizer_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
)

set(onnxruntime_test_framework_libs
  onnxruntime_test_utils
  onnxruntime_framework
  onnxruntime_util
  onnxruntime_graph
  onnxruntime_common
  onnxruntime_mlas
  )

set(onnxruntime_test_server_libs
  onnxruntime_test_utils
  onnxruntime_test_utils_for_server
)

if(WIN32)
    list(APPEND onnxruntime_test_framework_libs Advapi32)
endif()

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_FEATURIZERS)
   list(APPEND onnxruntime_test_providers_dependencies onnxruntime_featurizers)
   list(APPEND onnxruntime_test_providers_libs onnxruntime_featurizers re2)
   list(APPEND TEST_INC_DIR ${RE2_INCLUDE_DIR})
endif()

if(onnxruntime_USE_DML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dml)
endif()

if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_migraphx)
endif()

if(onnxruntime_USE_ROCM)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rocm)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml onnxruntime_coreml_proto)
endif()

file(GLOB_RECURSE onnxruntime_test_tvm_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/tvm/*.h"
  "${TEST_SRC_DIR}/tvm/*.cc"
  )

if(onnxruntime_USE_NUPHAR)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/nuphar/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nuphar)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nuphar)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nuphar)
endif()

if(onnxruntime_USE_ACL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_acl)
endif()

if(onnxruntime_USE_ARMNN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_armnn)
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  set(ONNXRUNTIME_INTEROP_TEST_LIBS PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()

set(ONNXRUNTIME_TEST_LIBS
    onnxruntime_session
    ${ONNXRUNTIME_INTEROP_TEST_LIBS}
    ${onnxruntime_libs}
    # CUDA, TENSORRT, DNNL, and OpenVINO are dynamically loaded at runtime
    ${PROVIDERS_MIGRAPHX}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_ROCM}
    ${PROVIDERS_COREML}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_flatbuffers
)

if (onnxruntime_ENABLE_TRAINING)
  set(ONNXRUNTIME_TEST_LIBS onnxruntime_training_runner onnxruntime_training ${ONNXRUNTIME_TEST_LIBS})
endif()

set(onnxruntime_test_providers_libs
    onnxruntime_test_utils
    ${ONNXRUNTIME_TEST_LIBS}
  )

if(onnxruntime_USE_TENSORRT)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/tensorrt/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_tensorrt)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_tensorrt onnxruntime_providers_shared)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nnapi/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/rknpu/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/coreml/*)
  list(APPEND onnxruntime_test_framework_libs onnxruntime_providers_coreml onnxruntime_coreml_proto)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml onnxruntime_coreml_proto)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_coreml onnxruntime_coreml_proto)
endif()


if(WIN32)
  if (onnxruntime_USE_TVM)
    list(APPEND disabled_warnings ${DISABLED_WARNINGS_FOR_TVM})
  endif()
endif()

file(GLOB onnxruntime_test_framework_src CONFIGURE_DEPENDS
  ${onnxruntime_test_framework_src_patterns}
  )

#without auto initialize onnxruntime
onnxruntime_add_static_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
if(MSVC)
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
else()
  target_compile_definitions(onnxruntime_test_utils PUBLIC -DNSYNC_ATOMIC_CPP11)
  target_include_directories(onnxruntime_test_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT}
          "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
endif()
if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_test_utils PRIVATE ${NCCL_INCLUDE_DIRS})
endif()
onnxruntime_add_include_to_target(onnxruntime_test_utils onnxruntime_common onnxruntime_framework onnxruntime_session GTest::gtest GTest::gmock onnx onnx_proto flatbuffers)


if (onnxruntime_USE_DML)
  target_add_dml(onnxruntime_test_utils)
endif()
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE
        ${eigen_INCLUDE_DIRS} ${ONNXRUNTIME_ROOT})
set_target_properties(onnxruntime_test_utils PROPERTIES FOLDER "ONNXRuntimeTest")

set(onnx_test_runner_src_dir ${TEST_SRC_DIR}/onnx)
file(GLOB onnx_test_runner_common_srcs CONFIGURE_DEPENDS
    ${onnx_test_runner_src_dir}/*.h
    ${onnx_test_runner_src_dir}/*.cc)

list(REMOVE_ITEM onnx_test_runner_common_srcs ${onnx_test_runner_src_dir}/main.cc)

onnxruntime_add_static_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
if(MSVC)
  target_compile_options(onnx_test_runner_common PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
else()
  target_compile_definitions(onnx_test_runner_common PUBLIC -DNSYNC_ATOMIC_CPP11)
  target_include_directories(onnx_test_runner_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT}
          "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
endif()
if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  #TODO: fix the warnings, they are dangerous
  target_compile_options(onnx_test_runner_common PRIVATE "/wd4244")
endif()
onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_common onnxruntime_framework
        onnxruntime_test_utils onnx onnx_proto re2::re2 flatbuffers)

add_dependencies(onnx_test_runner_common onnx_test_data_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnx_test_runner_common PRIVATE ${eigen_INCLUDE_DIRS} ${RE2_INCLUDE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})

set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "ONNXRuntimeTest")

set(all_tests ${onnxruntime_test_common_src} ${onnxruntime_test_ir_src} ${onnxruntime_test_optimizer_src}
        ${onnxruntime_test_framework_src} ${onnxruntime_test_providers_src} ${onnxruntime_test_quantiztion_src})
if(NOT TARGET onnxruntime AND NOT onnxruntime_BUILD_WEBASSEMBLY)
  list(APPEND all_tests ${onnxruntime_shared_lib_test_SRC})
endif()

if (onnxruntime_USE_CUDA)
  onnxruntime_add_static_library(onnxruntime_test_cuda_ops_lib ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu)
  list(APPEND onnxruntime_test_common_libs onnxruntime_test_cuda_ops_lib)
endif()

set(all_dependencies ${onnxruntime_test_providers_dependencies} )

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND all_tests ${onnxruntime_test_training_src})
endif()

if (onnxruntime_USE_TVM)
  list(APPEND all_tests ${onnxruntime_test_tvm_src})
endif()
if (onnxruntime_USE_OPENVINO)
  list(APPEND all_tests ${onnxruntime_test_openvino_src})
endif()

# this is only added to onnxruntime_test_framework_libs above, but we use onnxruntime_test_providers_libs for the onnxruntime_test_all target.
# for now, add it here. better is probably to have onnxruntime_test_providers_libs use the full onnxruntime_test_framework_libs
# list given it's built on top of that library and needs all the same dependencies.
if(WIN32)
  list(APPEND onnxruntime_test_providers_libs Advapi32)
endif()

if (onnxruntime_BUILD_WEBASSEMBLY)
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    # WebAssembly threading support in node is an experimental feature yet
    # and that makes some intensive threadpool tests fail randomly.
    # Will enable this test when node.js releases a stable version supporting multi-threads.
    list(REMOVE_ITEM all_tests
      "${TEST_SRC_DIR}/platform/threadpool_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/nn/string_normalizer_test.cc"
    )
  else()
    list(REMOVE_ITEM all_tests
      "${TEST_SRC_DIR}/framework/execution_frame_test.cc"
      "${TEST_SRC_DIR}/framework/inference_session_test.cc"
      "${TEST_SRC_DIR}/platform/barrier_test.cc"
      "${TEST_SRC_DIR}/platform/threadpool_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/controlflow/loop_test.cc"
      "${TEST_SRC_DIR}/providers/cpu/nn/string_normalizer_test.cc"
      "${TEST_SRC_DIR}/providers/memcpy_test.cc"
    )
  endif()
endif()

AddTest(
  TARGET onnxruntime_test_all
  SOURCES ${all_tests} ${onnxruntime_unittest_main_src}
  LIBS
    onnx_test_runner_common ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs} re2::re2
    onnx_test_data_proto nlohmann_json::nlohmann_json
  DEPENDS ${all_dependencies}
)

# the default logger tests conflict with the need to have an overall default logger
# so skip in this type of
target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
  target_compile_definitions(onnxruntime_test_all_xc PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
endif()
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  target_compile_options(onnxruntime_test_all PUBLIC "-Wno-unused-const-variable")
endif()
if(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE)
  target_compile_definitions(onnxruntime_test_all PUBLIC -DRUN_MODELTEST_IN_DEBUG_MODE)
endif()
if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_test_all PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()
if (onnxruntime_USE_FEATURIZERS)
  target_include_directories(onnxruntime_test_all PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/external/FeaturizersLibrary/src)
endif()
if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  target_link_libraries(onnxruntime_test_all PRIVATE onnxruntime_language_interop onnxruntime_pyop)
endif()
if (onnxruntime_USE_ROCM)
  target_include_directories(onnxruntime_test_all PRIVATE  ${onnxruntime_ROCM_HOME}/hipfft/include ${onnxruntime_ROCM_HOME}/include ${onnxruntime_ROCM_HOME}/hiprand/include ${onnxruntime_ROCM_HOME}/rocrand/include ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/onnxruntime ${CMAKE_CURRENT_BINARY_DIR}/amdgpu/orttraining)
endif()
if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  target_link_libraries(onnxruntime_test_all PRIVATE Python::Python)
endif()
if (onnxruntime_BUILD_WEBASSEMBLY)
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_DEPENDS ${TEST_SRC_DIR}/wasm/dump-test-result-in-nodejs.js)
  set_target_properties(onnxruntime_test_all PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 --pre-js \"${TEST_SRC_DIR}/wasm/dump-test-result-in-nodejs.js\" -s \"EXPORTED_RUNTIME_METHODS=['FS']\" --preload-file ${CMAKE_CURRENT_BINARY_DIR}/testdata@/testdata -s EXIT_RUNTIME=1")
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    set_property(TARGET onnxruntime_test_all APPEND_STRING PROPERTY LINK_FLAGS " -s USE_PTHREADS=1 -s PROXY_TO_PTHREAD=1")
  endif()
endif()

set(test_data_target onnxruntime_test_all)

onnxruntime_add_static_library(onnx_test_data_proto ${TEST_SRC_DIR}/proto/tml.proto)
add_dependencies(onnx_test_data_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
#onnx_proto target should mark this definition as public, instead of private
target_compile_definitions(onnx_test_data_proto PRIVATE "-DONNX_API=")
if(WIN32)
  target_compile_options(onnx_test_data_proto PRIVATE "/wd4125" "/wd4456" "/wd4100" "/wd4267" "/wd6011" "/wd6387" "/wd28182")
else()
  if(HAS_UNUSED_PARAMETER)
    target_compile_options(onnx_test_data_proto PRIVATE "-Wno-unused-parameter")
  endif()
  if(HAS_UNUSED_VARIABLE)
    target_compile_options(onnx_test_data_proto PRIVATE "-Wno-unused-variable")
  endif()
  if(HAS_UNUSED_BUT_SET_VARIABLE)
    target_compile_options(onnx_test_data_proto PRIVATE "-Wno-unused-but-set-variable")
  endif()
endif()
add_dependencies(onnx_test_data_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
onnxruntime_add_include_to_target(onnx_test_data_proto onnx_proto)
target_include_directories(onnx_test_data_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(onnx_test_data_proto PROPERTIES FOLDER "ONNXRuntimeTest")
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS external/onnx TARGET onnx_test_data_proto)

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  #
  # onnxruntime_ir_graph test data
  #
  set(TEST_DATA_SRC ${TEST_SRC_DIR}/testdata)
  set(TEST_DATA_DES $<TARGET_FILE_DIR:${test_data_target}>/testdata)

  set(TEST_SAMPLES_SRC ${REPO_ROOT}/samples)
  set(TEST_SAMPLES_DES $<TARGET_FILE_DIR:${test_data_target}>/samples)

  # Copy test data from source to destination.
  add_custom_command(
    TARGET ${test_data_target} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TEST_DATA_SRC}
    ${TEST_DATA_DES})

  # Copy test samples from source to destination.
  add_custom_command(
    TARGET ${test_data_target} PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${TEST_SAMPLES_SRC}
    ${TEST_SAMPLES_DES})

  if (onnxruntime_USE_DNNL)
    list(APPEND onnx_test_libs dnnl)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DNNL_DLL_PATH} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()
  if(WIN32)
    if (onnxruntime_USE_TVM)
      add_custom_command(
        TARGET ${test_data_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tvm> $<TARGET_FILE_DIR:${test_data_target}>
        )
    endif()
  endif()

  if(WIN32)
    set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
    onnxruntime_add_static_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
    target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
    set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
    set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})
    set(GETOPT_LIB_WIDE win_getopt_wide)
  endif()


  set(onnx_test_libs
    onnxruntime_test_utils
    ${ONNXRUNTIME_TEST_LIBS}
    onnx_test_data_proto
    ${onnxruntime_EXTERNAL_LIBRARIES})

  if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
    list(APPEND onnx_test_libs onnxruntime_language_interop onnxruntime_pyop)
  endif()

  onnxruntime_add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
  if(MSVC)
    target_compile_options(onnx_test_runner PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()
  if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set_target_properties(onnx_test_runner PROPERTIES
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    )
  endif()

  target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
  target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})
  set_target_properties(onnx_test_runner PROPERTIES FOLDER "ONNXRuntimeTest")

  if (onnxruntime_USE_TVM)
    if (WIN32)
      target_link_options(onnx_test_runner PRIVATE "/STACK:4000000")
    endif()
  endif()

  install(TARGETS onnx_test_runner
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          BUNDLE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

  if(onnxruntime_BUILD_BENCHMARKS)
    SET(BENCHMARK_DIR ${TEST_SRC_DIR}/onnx/microbenchmark)
    onnxruntime_add_executable(onnxruntime_benchmark
      ${BENCHMARK_DIR}/main.cc
      ${BENCHMARK_DIR}/modeltest.cc
      ${BENCHMARK_DIR}/pooling.cc
      ${BENCHMARK_DIR}/batchnorm.cc
      ${BENCHMARK_DIR}/batchnorm2.cc
      ${BENCHMARK_DIR}/tptest.cc
      ${BENCHMARK_DIR}/eigen.cc
      ${BENCHMARK_DIR}/copy.cc
      ${BENCHMARK_DIR}/gelu.cc
      ${BENCHMARK_DIR}/activation.cc
      ${BENCHMARK_DIR}/quantize.cc
      ${BENCHMARK_DIR}/reduceminmax.cc)
    target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    if(WIN32)
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd4141>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4141>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    target_link_libraries(onnxruntime_benchmark PRIVATE onnx_test_runner_common benchmark::benchmark ${onnx_test_libs})
    add_dependencies(onnxruntime_benchmark ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")

    SET(MLAS_BENCH_DIR ${TEST_SRC_DIR}/mlas/bench)
    file(GLOB_RECURSE MLAS_BENCH_SOURCE_FILES "${MLAS_BENCH_DIR}/*.cpp" "${MLAS_BENCH_DIR}/*.h")
    onnxruntime_add_executable(onnxruntime_mlas_benchmark ${MLAS_BENCH_SOURCE_FILES})
    target_include_directories(onnxruntime_mlas_benchmark PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_link_libraries(onnxruntime_mlas_benchmark PRIVATE benchmark::benchmark onnxruntime_mlas onnxruntime_common onnxruntime_framework onnxruntime_util)
    if(NOT WIN32)
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE nsync_cpp)
    endif()
    set_target_properties(onnxruntime_mlas_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  if(WIN32)
    target_compile_options(onnx_test_runner_common PRIVATE -D_CRT_SECURE_NO_WARNINGS)
  endif()

  if (NOT onnxruntime_REDUCED_OPS_BUILD AND NOT onnxruntime_BUILD_WEBASSEMBLY)
    add_test(NAME onnx_test_pytorch_converted
      COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-converted)
    add_test(NAME onnx_test_pytorch_operator
      COMMAND onnx_test_runner ${PROJECT_SOURCE_DIR}/external/onnx/onnx/backend/test/data/pytorch-operator)
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND android_shared_libs log android)
  endif()

  #eager mode test
  if(onnxruntime_ENABLE_EAGER_MODE)
    file(GLOB onnxruntime_eager_mode_test_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/eager/*.cc"
      )
    add_executable(onnxruntime_eager_mode_test ${onnxruntime_eager_mode_test_src})
    target_include_directories(onnxruntime_eager_mode_test PRIVATE ${ONNXRUNTIME_ROOT}
            ${onnxruntime_graph_header}
            ${onnxruntime_exec_src_dir}
            ${CMAKE_CURRENT_BINARY_DIR}
            "${TEST_SRC_DIR}/util/include")
    set(onnxruntime_eager_mode_libs
            onnxruntime_eager
            onnxruntime_session
            onnxruntime_optimizer
            onnxruntime_providers
            onnxruntime_util
            onnxruntime_framework
            flatbuffers
            onnxruntime_graph
            onnxruntime_common
            onnxruntime_mlas
            onnx
            onnx_proto
            ${PROTOBUF_LIB}
            GTest::gtest
            re2::re2
            onnxruntime_flatbuffers
            ${CMAKE_DL_LIBS}
            )
    if(onnxruntime_ENABLE_TRAINING)
      list(APPEND onnxruntime_eager_mode_libs onnxruntime_training tensorboard)
    endif()
    IF(NOT WIN32)
      list(APPEND onnxruntime_eager_mode_libs nsync_cpp)
    endif()
    target_link_libraries(onnxruntime_eager_mode_test PRIVATE ${onnxruntime_eager_mode_libs} Threads::Threads ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()

  #perf test runner
  set(onnxruntime_perf_test_src_dir ${TEST_SRC_DIR}/perftest)
  set(onnxruntime_perf_test_src_patterns
  "${onnxruntime_perf_test_src_dir}/*.cc"
  "${onnxruntime_perf_test_src_dir}/*.h")

  if(WIN32)
    list(APPEND onnxruntime_perf_test_src_patterns
      "${onnxruntime_perf_test_src_dir}/windows/*.cc"
      "${onnxruntime_perf_test_src_dir}/windows/*.h" )
  else ()
    list(APPEND onnxruntime_perf_test_src_patterns
      "${onnxruntime_perf_test_src_dir}/posix/*.cc"
      "${onnxruntime_perf_test_src_dir}/posix/*.h" )
  endif()

  file(GLOB onnxruntime_perf_test_src CONFIGURE_DEPENDS
    ${onnxruntime_perf_test_src_patterns}
    )
  onnxruntime_add_executable(onnxruntime_perf_test ${onnxruntime_perf_test_src} ${ONNXRUNTIME_ROOT}/core/platform/path_lib.cc)
  if(MSVC)
    target_compile_options(onnxruntime_perf_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()
  target_include_directories(onnxruntime_perf_test PRIVATE ${onnx_test_runner_src_dir} ${ONNXRUNTIME_ROOT}
          ${eigen_INCLUDE_DIRS} ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir}
          ${CMAKE_CURRENT_BINARY_DIR})
  if (WIN32)
    target_compile_options(onnxruntime_perf_test PRIVATE ${disabled_warnings})
    if (NOT DEFINED SYS_PATH_LIB)
      set(SYS_PATH_LIB shlwapi)
    endif()
  endif()
  if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set_target_properties(onnxruntime_perf_test PROPERTIES
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    )
  endif()

  if (onnxruntime_BUILD_SHARED_LIB)
    set(onnxruntime_perf_test_libs
            onnx_test_runner_common onnxruntime_test_utils onnxruntime_common
            onnxruntime onnxruntime_flatbuffers  onnx_test_data_proto
            ${onnxruntime_EXTERNAL_LIBRARIES}
            ${GETOPT_LIB_WIDE} ${SYS_PATH_LIB} ${CMAKE_DL_LIBS})
    if(NOT WIN32)
      list(APPEND onnxruntime_perf_test_libs nsync_cpp)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND onnxruntime_perf_test_libs ${android_shared_libs})
    endif()
    target_link_libraries(onnxruntime_perf_test PRIVATE ${onnxruntime_perf_test_libs} Threads::Threads)
    if(WIN32)
      target_link_libraries(onnxruntime_perf_test PRIVATE debug dbghelp advapi32)
    endif()
    if(tensorflow_C_PACKAGE_PATH)
      target_include_directories(onnxruntime_perf_test PRIVATE ${tensorflow_C_PACKAGE_PATH}/include)
      target_link_directories(onnxruntime_perf_test PRIVATE ${tensorflow_C_PACKAGE_PATH}/lib)
      target_link_libraries(onnxruntime_perf_test PRIVATE tensorflow)
      target_compile_definitions(onnxruntime_perf_test PRIVATE HAVE_TENSORFLOW)
    endif()
  else()
    target_link_libraries(onnxruntime_perf_test PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs})
  endif()
  set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "ONNXRuntimeTest")

  if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS AND NOT onnxruntime_BUILD_SHARED_LIB)
    target_link_libraries(onnxruntime_perf_test PRIVATE onnxruntime_language_interop onnxruntime_pyop)
  endif()

  if (onnxruntime_USE_TVM)
    if (WIN32)
      target_link_options(onnxruntime_perf_test PRIVATE "/STACK:4000000")
    endif()
  endif()

  # shared lib
  if (onnxruntime_BUILD_SHARED_LIB)
    onnxruntime_add_static_library(onnxruntime_mocked_allocator ${TEST_SRC_DIR}/util/test_allocator.cc)
    target_include_directories(onnxruntime_mocked_allocator PUBLIC ${TEST_SRC_DIR}/util/include)
    set_target_properties(onnxruntime_mocked_allocator PROPERTIES FOLDER "ONNXRuntimeTest")

    #################################################################
    # test inference using shared lib
    set(onnxruntime_shared_lib_test_LIBS onnxruntime_mocked_allocator onnxruntime_test_utils onnxruntime_common onnx_proto)
    if(NOT WIN32)
      list(APPEND onnxruntime_shared_lib_test_LIBS nsync_cpp)
    endif()
    if (onnxruntime_USE_CUDA)
      list(APPEND onnxruntime_shared_lib_test_LIBS onnxruntime_test_cuda_ops_lib cudart)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND onnxruntime_shared_lib_test_LIBS ${android_shared_libs})
    endif()
    AddTest(DYN
            TARGET onnxruntime_shared_lib_test
            SOURCES ${onnxruntime_shared_lib_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_shared_lib_test_LIBS}
            DEPENDS ${all_dependencies}
    )
    if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
      add_custom_command(
        TARGET onnxruntime_shared_lib_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_DES}
        $<TARGET_FILE_DIR:onnxruntime_shared_lib_test>/testdata)
    endif()

    # test inference using global threadpools
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Android|iOS" AND NOT onnxruntime_MINIMAL_BUILD)
      AddTest(DYN
              TARGET onnxruntime_global_thread_pools_test
              SOURCES ${onnxruntime_global_thread_pools_test_SRC}
              LIBS ${onnxruntime_shared_lib_test_LIBS}
              DEPENDS ${all_dependencies}
      )
    endif()

  # A separate test is needed to test the APIs that don't rely on the env being created first.
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Android|iOS")
      AddTest(DYN
              TARGET onnxruntime_api_tests_without_env
              SOURCES ${onnxruntime_api_tests_without_env_SRC}
              LIBS ${onnxruntime_shared_lib_test_LIBS}
              DEPENDS ${all_dependencies}
      )
    endif()
  endif()

  # the debug node IO functionality uses static variables, so it is best tested
  # in its own process
  if(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
    AddTest(
      TARGET onnxruntime_test_debug_node_inputs_outputs
      SOURCES
        "${TEST_SRC_DIR}/debug_node_inputs_outputs/debug_node_inputs_outputs_utils_test.cc"
        "${TEST_SRC_DIR}/framework/TestAllocatorManager.cc"
        "${TEST_SRC_DIR}/framework/test_utils.cc"
        "${TEST_SRC_DIR}/providers/provider_test_utils.cc"
        ${onnxruntime_unittest_main_src}
      LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
      DEPENDS ${all_dependencies}
    )

    target_compile_definitions(onnxruntime_test_debug_node_inputs_outputs
      PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
  endif(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)

  #some ETW tools
  if(WIN32 AND onnxruntime_ENABLE_INSTRUMENT)
    onnxruntime_add_executable(generate_perf_report_from_etl ${ONNXRUNTIME_ROOT}/tool/etw/main.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(generate_perf_report_from_etl PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(generate_perf_report_from_etl PRIVATE tdh Advapi32)

    onnxruntime_add_executable(compare_two_sessions ${ONNXRUNTIME_ROOT}/tool/etw/compare_two_sessions.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/eparser.h ${ONNXRUNTIME_ROOT}/tool/etw/eparser.cc
            ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.h ${ONNXRUNTIME_ROOT}/tool/etw/TraceSession.cc)
    target_compile_definitions(compare_two_sessions PRIVATE "_CONSOLE" "_UNICODE" "UNICODE")
    target_link_libraries(compare_two_sessions PRIVATE ${GETOPT_LIB_WIDE} tdh Advapi32)
  endif()

  file(GLOB onnxruntime_mlas_test_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/mlas/unittest/*.h"
    "${TEST_SRC_DIR}/mlas/unittest/*.cpp"
  )
  onnxruntime_add_executable(onnxruntime_mlas_test ${onnxruntime_mlas_test_src})
  if(MSVC)
    target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
  endif()
  if(${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set_target_properties(onnxruntime_mlas_test PROPERTIES
      XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
    )
  endif()
  target_include_directories(onnxruntime_mlas_test PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}
          ${CMAKE_CURRENT_BINARY_DIR})
  set(onnxruntime_mlas_test_libs GTest::gtest GTest::gmock onnxruntime_mlas onnxruntime_common)
  if(NOT WIN32)
    list(APPEND onnxruntime_mlas_test_libs nsync_cpp ${CMAKE_DL_LIBS})
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    list(APPEND onnxruntime_mlas_test_libs ${android_shared_libs})
  endif()
  if (onnxruntime_USE_OPENMP)
    list(APPEND onnxruntime_mlas_test_libs OpenMP::OpenMP_CXX)
  endif()
  list(APPEND onnxruntime_mlas_test_libs Threads::Threads)
  target_link_libraries(onnxruntime_mlas_test PRIVATE ${onnxruntime_mlas_test_libs})
  if(WIN32)
    target_link_libraries(onnxruntime_mlas_test PRIVATE debug Dbghelp Advapi32)
  endif()
  if (onnxruntime_LINK_LIBATOMIC)
    target_link_libraries(onnxruntime_mlas_test PRIVATE atomic)
  endif()
  set_target_properties(onnxruntime_mlas_test PROPERTIES FOLDER "ONNXRuntimeTest")
  if (onnxruntime_BUILD_WEBASSEMBLY)
    if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
      set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s USE_PTHREADS=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
    else()
      set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
    endif()
  endif()

  onnxruntime_add_shared_library_module(custom_op_library ${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.cc)
  target_include_directories(custom_op_library PRIVATE ${REPO_ROOT}/include)
  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.def")
  endif()
  set_property(TARGET custom_op_library APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})

  if (onnxruntime_BUILD_JAVA AND NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
      message(STATUS "Running Java tests")
      # native-test is added to resources so custom_op_lib can be loaded
      # and we want to symlink it there
      set(JAVA_NATIVE_TEST_DIR ${JAVA_OUTPUT_DIR}/native-test)
      file(MAKE_DIRECTORY ${JAVA_NATIVE_TEST_DIR})

      # delegate to gradle's test runner
      if(WIN32)
        add_custom_command(TARGET custom_op_library POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:custom_op_library>
                        ${JAVA_NATIVE_TEST_DIR}/$<TARGET_FILE_NAME:custom_op_library>)
        # On windows ctest requires a test to be an .exe(.com) file
        # So there are two options 1) Install Chocolatey and its gradle package
        # That package would install gradle.exe shim to its bin so ctest could run gradle.exe
        # 2) With standard installation we get gradle.bat. We delegate execution to a separate .cmake file
        # That can handle both .exe and .bat
        add_test(NAME onnxruntime4j_test COMMAND ${CMAKE_COMMAND}
          -DGRADLE_EXECUTABLE=${GRADLE_EXECUTABLE}
          -DBIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
          -DREPO_ROOT=${REPO_ROOT}
          ${ORT_PROVIDER_CMAKE_FLAGS}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_java_unittests.cmake)
      else()
        add_custom_command(TARGET custom_op_library POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:custom_op_library>
                        ${JAVA_NATIVE_TEST_DIR}/$<TARGET_LINKER_FILE_NAME:custom_op_library>)
        if (onnxruntime_USE_CUDA)
          add_test(NAME onnxruntime4j_test COMMAND ${GRADLE_EXECUTABLE} cmakeCheck -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR} -DUSE_CUDA=1
                  WORKING_DIRECTORY ${REPO_ROOT}/java)
        else()
          add_test(NAME onnxruntime4j_test COMMAND ${GRADLE_EXECUTABLE} cmakeCheck -DcmakeBuildDir=${CMAKE_CURRENT_BINARY_DIR}
                  WORKING_DIRECTORY ${REPO_ROOT}/java)
        endif()
      endif()
      set_property(TEST onnxruntime4j_test APPEND PROPERTY DEPENDS onnxruntime4j_jni)
  endif()

  # limit to only test on windows first, due to a runtime path issue on linux
  if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_EXTENDED_MINIMAL_BUILD
                                    AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS"
                                    AND NOT (CMAKE_SYSTEM_NAME STREQUAL "Android")
                                    AND NOT onnxruntime_BUILD_WEBASSEMBLY
                                    AND MSVC)
    file(GLOB_RECURSE test_execution_provider_srcs
      "${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/*.h"
      "${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/*.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
    )

    onnxruntime_add_shared_library_module(test_execution_provider ${test_execution_provider_srcs})
    add_dependencies(test_execution_provider onnxruntime_providers_shared)
    target_link_libraries(test_execution_provider PRIVATE onnxruntime_providers_shared)
    target_include_directories(test_execution_provider PRIVATE $<TARGET_PROPERTY:onnx,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(test_execution_provider PRIVATE $<TARGET_PROPERTY:onnxruntime_common,INTERFACE_INCLUDE_DIRECTORIES>)
    target_include_directories(test_execution_provider PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} ${ORTTRAINING_ROOT})
    if(APPLE)
      set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/exported_symbols.lst")
    elseif(UNIX)
      set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/version_script.lds -Xlinker --gc-sections")
    elseif(WIN32)
      set_property(TARGET test_execution_provider APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${REPO_ROOT}/onnxruntime/test/testdata/custom_execution_provider_library/symbols.def")
    else()
      message(FATAL_ERROR "test_execution_provider unknown platform, need to specify shared library exports for it")
    endif()
  endif()

  include(onnxruntime_fuzz_test.cmake)
endif()

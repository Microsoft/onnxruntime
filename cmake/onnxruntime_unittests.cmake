# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
if (IOS)
  find_package(XCTest REQUIRED)
endif()

set(TEST_SRC_DIR ${ONNXRUNTIME_ROOT}/test)
set(TEST_INC_DIR ${ONNXRUNTIME_ROOT})
if(NOT CMAKE_CROSSCOMPILING)
  add_test(NAME onnxruntime_public_exports_test
    COMMAND "${Python_EXECUTABLE}" "${REPO_ROOT}/tools/ci_build/test_gen_def.py")
endif()
if (onnxruntime_ENABLE_TRAINING)
  list(APPEND TEST_INC_DIR ${ORTTRAINING_ROOT})
endif()

# Exclude files based on CMake options.
function(filter_test_srcs test_srcs_var)
  set(excluded_path_prefixes)

  if(onnxruntime_DISABLE_CONTRIB_OPS)
    list(APPEND excluded_path_prefixes ${TEST_SRC_DIR}/contrib_ops)
  endif()

  if(onnxruntime_DISABLE_ML_OPS)
    list(APPEND excluded_path_prefixes ${TEST_SRC_DIR}/providers/cpu/ml)
  endif()

  list(LENGTH excluded_path_prefixes num_excluded_path_prefixes)

  if("${num_excluded_path_prefixes}" GREATER "0")
    set(filtered_test_srcs)

    foreach(test_src ${${test_srcs_var}})
      set(is_excluded false)

      foreach(excluded_path_prefix ${excluded_path_prefixes})
        cmake_path(ABSOLUTE_PATH test_src OUTPUT_VARIABLE test_src_absolute)

        cmake_path(IS_PREFIX excluded_path_prefix ${test_src_absolute} NORMALIZE is_excluded)

        if (is_excluded)
          break()
        endif()
      endforeach()

      if(NOT is_excluded)
        list(APPEND filtered_test_srcs ${test_src})
      endif()
    endforeach()

    set(${test_srcs_var} ${filtered_test_srcs} PARENT_SCOPE)
  endif()
endfunction()

set(disabled_warnings)

function(onnxruntime_disable_gtest_character_conversion_as_error target_name)
  if (HAS_NO_ERROR_CHARACTER_CONVERSION)
    target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=character-conversion>")
  endif()
endfunction()

function(AddTest)
  cmake_parse_arguments(_UT "DYN" "TARGET;TEST_NAME" "LIBS;SOURCES;DEPENDS;TEST_ARGS" ${ARGN})
  if (NOT _UT_TEST_NAME)
    set(_UT_TEST_NAME ${_UT_TARGET})
  endif()
  list(REMOVE_DUPLICATES _UT_SOURCES)

  filter_test_srcs(_UT_SOURCES)

  message(VERBOSE "AddTest() TARGET: ${_UT_TARGET}")
  message(VERBOSE "AddTest() SOURCES:")
  foreach(ut_src ${_UT_SOURCES})
    message(VERBOSE "  ${ut_src}")
  endforeach()

  if (IOS)
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
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")
  endif()
  if (MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6330>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6330>")
    #Abseil has a lot of C4127/C4324 warnings.
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4127>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4127>")
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4324>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4324>")
  endif()

  set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest")

  if (MSVC)
    # set VS debugger working directory to the test program's directory
    set_target_properties(${_UT_TARGET} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>)
  endif()

  if (_UT_DEPENDS)
    add_dependencies(${_UT_TARGET} ${_UT_DEPENDS})
  endif(_UT_DEPENDS)

  if(_UT_DYN)
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
            Threads::Threads)
    target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_ONNXRUNTIME_DLL)
  else()
    if(onnxruntime_USE_CUDA OR onnxruntime_USE_NV)
      #XXX: we should not need to do this. onnxruntime_test_all.exe should not have direct dependency on CUDA DLLs,
      # otherwise it will impact when CUDA DLLs can be unloaded.
      target_link_libraries(${_UT_TARGET} PRIVATE CUDA::cudart)
    endif()
    if(onnxruntime_USE_CUDA AND NOT onnxruntime_CUDA_MINIMAL)
      target_link_libraries(${_UT_TARGET} PRIVATE cudnn_frontend)
    endif()
    target_link_libraries(${_UT_TARGET} PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
  endif()

  onnxruntime_add_include_to_target(${_UT_TARGET} date::date flatbuffers::flatbuffers)
  target_include_directories(${_UT_TARGET} PRIVATE ${TEST_INC_DIR})
  if (onnxruntime_USE_CUDA)
    target_include_directories(${_UT_TARGET} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    if(NOT onnxruntime_CUDA_MINIMAL)
      target_include_directories(${_UT_TARGET} PRIVATE ${CUDNN_INCLUDE_DIR})
    endif()
    if (onnxruntime_USE_NCCL)
      target_include_directories(${_UT_TARGET} PRIVATE ${NCCL_INCLUDE_DIRS})
    endif()
    if(onnxruntime_CUDA_MINIMAL)
      target_compile_definitions(${_UT_TARGET} PRIVATE -DUSE_CUDA_MINIMAL)
    endif()
  endif()
  if (onnxruntime_USE_TENSORRT)
    # used for instantiating placeholder TRT builder to mitigate TRT library load/unload overhead
    target_include_directories(${_UT_TARGET} PRIVATE ${TENSORRT_INCLUDE_DIR})
  endif()
  if (onnxruntime_USE_NV)
    # used for instantiating placeholder TRT builder to mitigate TRT library load/unload overhead
    target_include_directories(${_UT_TARGET} PRIVATE ${NV_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
  endif()

  if(MSVC)
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  endif()

  if (WIN32)
    # include dbghelp in case tests throw an ORT exception, as that exception includes a stacktrace, which requires dbghelp.
    target_link_libraries(${_UT_TARGET} PRIVATE debug dbghelp)

    if (MSVC)
      # warning C6326: Potential comparison of a constant with another constant.
      # Lot of such things came from gtest
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      # Raw new and delete. A lot of such things came from googletest.
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      # "Global initializer calls a non-constexpr function."
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
    endif()
    target_compile_options(${_UT_TARGET} PRIVATE ${disabled_warnings})
  else()
    target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-error=sign-compare>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
    if (${HAS_NOERROR})
      target_compile_options(${_UT_TARGET} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-error=uninitialized>")
    endif()
    onnxruntime_disable_gtest_character_conversion_as_error(${_UT_TARGET})
  endif()

  set(TEST_ARGS ${_UT_TEST_ARGS})
  if (onnxruntime_GENERATE_TEST_REPORTS)
    # generate a report file next to the test program
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # WebAssembly use a memory file system, so we do not use full path
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<TARGET_FILE_NAME:${_UT_TARGET}>.$<CONFIG>.results.xml")
    else()
      list(APPEND TEST_ARGS
        "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:${_UT_TARGET}>.$<CONFIG>.results.xml>")
    endif()
  endif(onnxruntime_GENERATE_TEST_REPORTS)

  if (IOS)
    # target_sources(${_UT_TARGET} PRIVATE ${TEST_SRC_DIR}/xctest/orttestmain.m)

    set(_UT_IOS_BUNDLE_GUI_IDENTIFIER com.onnxruntime.utest.${_UT_TARGET})
    # replace any characters that are not valid in a bundle identifier with '-'
    string(REGEX REPLACE "[^a-zA-Z0-9\\.-]" "-" _UT_IOS_BUNDLE_GUI_IDENTIFIER ${_UT_IOS_BUNDLE_GUI_IDENTIFIER})

    set_target_properties(${_UT_TARGET} PROPERTIES FOLDER "ONNXRuntimeTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}
      MACOSX_BUNDLE_GUI_IDENTIFIER ${_UT_IOS_BUNDLE_GUI_IDENTIFIER}
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
    onnxruntime_configure_target(${_UT_TARGET}_xc)
    if(_UT_DYN)
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock onnxruntime ${CMAKE_DL_LIBS}
              Threads::Threads)
      target_compile_definitions(${_UT_TARGET}_xc PRIVATE USE_ONNXRUNTIME_DLL)
    else()
      target_link_libraries(${_UT_TARGET}_xc PRIVATE ${_UT_LIBS} GTest::gtest GTest::gmock ${onnxruntime_EXTERNAL_LIBRARIES})
    endif()
    onnxruntime_add_include_to_target(${_UT_TARGET}_xc date::date flatbuffers::flatbuffers)
    target_include_directories(${_UT_TARGET}_xc PRIVATE ${TEST_INC_DIR})
    get_target_property(${_UT_TARGET}_DEFS ${_UT_TARGET} COMPILE_DEFINITIONS)
    target_compile_definitions(${_UT_TARGET}_xc PRIVATE ${_UT_TARGET}_DEFS)

    set_target_properties(${_UT_TARGET}_xc PROPERTIES FOLDER "ONNXRuntimeXCTest"
      MACOSX_BUNDLE_BUNDLE_NAME ${_UT_TARGET}_xc
      MACOSX_BUNDLE_GUI_IDENTIFIER ${_UT_IOS_BUNDLE_GUI_IDENTIFIER}
      MACOSX_BUNDLE_LONG_VERSION_STRING ${ORT_VERSION}
      MACOSX_BUNDLE_BUNDLE_VERSION ${ORT_VERSION}
      MACOSX_BUNDLE_SHORT_VERSION_STRING ${ORT_VERSION}
      XCODE_ATTRIBUTE_ENABLE_BITCODE "NO")

    # This is a workaround for an Xcode 16 / CMake issue:
    #   error: Multiple commands produce '<build>/Debug/Debug-iphonesimulator/onnxruntime_test_all.app/PlugIns'
    #       note: CreateBuildDirectory <build>/Debug/Debug-iphonesimulator/onnxruntime_test_all.app/PlugIns
    #       note: Target 'onnxruntime_test_all' (project 'onnxruntime') has create directory command with output
    #             '<build>/Debug/Debug-iphonesimulator/onnxruntime_test_all.app/PlugIns'
    #
    # It seems related to the test target (e.g., onnxruntime_test_all_xc) LIBRARY_OUTPUT_DIRECTORY property getting set
    # to "$<TARGET_BUNDLE_CONTENT_DIR:${testee}>/PlugIns" in xctest_add_bundle():
    # https://github.com/Kitware/CMake/blob/9c4a0a9ff09735b847bbbc38caf6da7f6c7238f2/Modules/FindXCTest.cmake#L159-L168
    #
    # This is the related CMake issue: https://gitlab.kitware.com/cmake/cmake/-/issues/26301
    #
    # Unsetting LIBRARY_OUTPUT_DIRECTORY avoids the build error.
    set_property(TARGET ${_UT_TARGET}_xc PROPERTY LIBRARY_OUTPUT_DIRECTORY)

    # Don't bother calling xctest_add_test() because we don't use CTest to run tests on iOS.
    # Instead, we can call 'xcodebuild test-without-building' and specify a '-destination' referring to an iOS
    # simulator or device.
    # xctest_add_test(xctest.${_UT_TARGET} ${_UT_TARGET}_xc)
  else()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      # Include the Node.js helper for finding and validating Node.js and NPM
      include(node_helper.cmake)

      if (onnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER)
        add_custom_command(TARGET ${_UT_TARGET} POST_BUILD
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/package.json $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/package-lock.json $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TEST_SRC_DIR}/wasm/karma.conf.js $<TARGET_FILE_DIR:${_UT_TARGET}>
          COMMAND ${NPM_CLI} ci
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )

        set(TEST_NPM_FLAGS)
        if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
          list(APPEND TEST_NPM_FLAGS "--wasm-threads")
        endif()
        add_test(NAME ${_UT_TEST_NAME}
          COMMAND ${NPM_CLI} test -- ${TEST_NPM_FLAGS} --entry=${_UT_TARGET} ${TEST_ARGS}
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
      else()
        set(TEST_NODE_FLAGS)

        if (onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD)
          message(WARNING "Use system `node` to test Wasm relaxed SIMD. Please make sure to install node v21 or newer.")
          set(NODE_EXECUTABLE node)
        # prefer Node from emsdk so the version is more deterministic
        elseif (DEFINED ENV{EMSDK_NODE})
          set(NODE_EXECUTABLE $ENV{EMSDK_NODE})
        else()
          message(WARNING "EMSDK_NODE environment variable was not set. Falling back to system `node`.")
          set(NODE_EXECUTABLE node)
        endif()

        add_test(NAME ${_UT_TEST_NAME}
          COMMAND ${NODE_EXECUTABLE} ${TEST_NODE_FLAGS} ${_UT_TARGET}.js ${TEST_ARGS}
          WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
        )
      endif()
      # Set test timeout to 3 hours.
      set_tests_properties(${_UT_TEST_NAME} PROPERTIES TIMEOUT 10800)
    else()
      add_test(NAME ${_UT_TEST_NAME}
        COMMAND ${_UT_TARGET} ${TEST_ARGS}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${_UT_TARGET}>
      )
      # Set test timeout to 3 hours.
      set_tests_properties(${_UT_TEST_NAME} PROPERTIES TIMEOUT 10800)
    endif()
  endif()
endfunction(AddTest)

# Given a list of test source files, in variable `all_srcs_var`, partition it into two lists:
# - a list of the provider-related test source files, stored in `provider_test_srcs_var`
# - a list of the other remaining test source files, stored in `other_srcs_var`
#
# In particular, provider-related test source files are located in these root paths:
# - onnxruntime/test/contrib_ops
# - onnxruntime/test/providers
function(partition_provider_test_srcs
         all_srcs_var provider_test_srcs_var other_srcs_var)
  set(provider_test_src_roots
    ${TEST_SRC_DIR}/contrib_ops
    ${TEST_SRC_DIR}/providers
  )

  function(is_provider_test_src src_var result_var)
    cmake_path(ABSOLUTE_PATH ${src_var} OUTPUT_VARIABLE src_absolute)

    foreach(provider_test_src_root ${provider_test_src_roots})
      cmake_path(IS_PREFIX provider_test_src_root ${src_absolute} NORMALIZE src_matches_root)
      if(src_matches_root)
        set(${result_var} true PARENT_SCOPE)
        return()
      endif()
    endforeach()

    set(${result_var} false PARENT_SCOPE)
  endfunction()

  set(provider_test_srcs)
  set(other_srcs)

  foreach(src ${${all_srcs_var}})
    is_provider_test_src(src is_provider_test_src_result)
    if(is_provider_test_src_result)
      list(APPEND provider_test_srcs ${src})
    else()
      list(APPEND other_srcs ${src})
    endif()
  endforeach()

  set(${provider_test_srcs_var} ${provider_test_srcs} PARENT_SCOPE)
  set(${other_srcs_var} ${other_srcs} PARENT_SCOPE)
endfunction()

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

file(GLOB onnxruntime_test_quantization_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/quantization/*.cc"
  "${TEST_SRC_DIR}/quantization/*.h"
)

file(GLOB onnxruntime_test_flatbuffers_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/flatbuffers/*.cc"
  "${TEST_SRC_DIR}/flatbuffers/*.h"
)

file(GLOB onnxruntime_test_lora_src CONFIGURE_DEPENDS
  "${TEST_SRC_DIR}/lora/*.cc"
  "${TEST_SRC_DIR}/lora/*.h"
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

  if (onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
    list(APPEND onnxruntime_test_framework_src_patterns
      "${TEST_SRC_DIR}/framework/ort_model_only_test.cc"
    )
  endif()

  if (NOT onnxruntime_MINIMAL_BUILD)
    file(GLOB onnxruntime_test_ir_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/ir/*.cc"
      "${TEST_SRC_DIR}/ir/*.h"
      )
  endif()
endif()

if((NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
   AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_optimizer_src
       "${TEST_SRC_DIR}/optimizer/runtime_optimization/graph_runtime_optimization_test.cc")
endif()

file(GLOB onnxruntime_test_training_src
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/model/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/gradient/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/graph/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/session/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/optimizer/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/framework/*.cc"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.h"
  "${ORTTRAINING_SOURCE_DIR}/test/distributed/*.cc"
  )

# TODO (baijumeswani): Remove the minimal build check here.
#                      The training api tests should be runnable even on a minimal build.
#                      This requires converting all the *.onnx files to ort format.
if (NOT onnxruntime_MINIMAL_BUILD)
  if (onnxruntime_ENABLE_TRAINING_APIS)
    file(GLOB onnxruntime_test_training_api_src
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.h"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/core/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/core/*.h"
      )
  endif()
endif()

if(WIN32)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/windows/*.cc"
    "${TEST_SRC_DIR}/platform/windows/logging/*.cc" )
  if(onnxruntime_USE_1DS_TELEMETRY)
    list(APPEND onnxruntime_test_framework_src_patterns
      "${TEST_SRC_DIR}/platform/windows_telemetry/*.cc" )
  endif()
endif()

if(LINUX AND NOT onnxruntime_DISABLE_DEVICE_DISCOVERY)
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/linux/npu_device_discovery_test.cc"
    "${TEST_SRC_DIR}/platform/linux/pci_device_discovery_test.cc")
endif()

if(onnxruntime_USE_TELEMETRY AND NOT WIN32 AND NOT ANDROID AND NOT CMAKE_SYSTEM_NAME STREQUAL "iOS")
  list(APPEND onnxruntime_test_framework_src_patterns
    "${TEST_SRC_DIR}/platform/posix/device_id_test.cc")
endif()

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)

  if(onnxruntime_USE_CUDA)
    list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/framework/cuda/*)
  endif()

  set(onnxruntime_test_providers_src_patterns
    "${TEST_SRC_DIR}/providers/*.h"
    "${TEST_SRC_DIR}/providers/*.cc"
    "${TEST_SRC_DIR}/opaque_api/test_opaque_api.cc"
    "${TEST_SRC_DIR}/contrib_ops/*.h"
    "${TEST_SRC_DIR}/contrib_ops/*.cc"
    "${TEST_SRC_DIR}/contrib_ops/math/*.h"
    "${TEST_SRC_DIR}/contrib_ops/math/*.cc"
  )

else()
  set(onnxruntime_test_providers_src_patterns
    # TODO: Add anything that is needed for testing a minimal build
  )
endif()

list(LENGTH onnxruntime_test_providers_src_patterns onnxruntime_test_providers_src_patterns_length)
if(onnxruntime_test_providers_src_patterns_length GREATER 0)
  file(GLOB onnxruntime_test_providers_src CONFIGURE_DEPENDS ${onnxruntime_test_providers_src_patterns})
else()
  set(onnxruntime_test_providers_src)
endif()

if(NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_cpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cpu/*"
    )
endif()

list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cpu_src})

if (onnxruntime_USE_CUDA AND NOT onnxruntime_BUILD_CUDA_EP_AS_PLUGIN AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB onnxruntime_test_providers_cuda_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_src})

  if (onnxruntime_USE_CUDA_NHWC_OPS AND CUDNN_MAJOR_VERSION GREATER 8)
    file(GLOB onnxruntime_test_providers_cuda_nhwc_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/providers/cuda/nhwc/*.cc"
    )
    list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_nhwc_src})
  endif()
endif()

if (onnxruntime_USE_CUDA AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB onnxruntime_test_providers_cuda_plugin_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/plugin/*.cc"
  )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cuda_plugin_src})
endif()

if (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  file(GLOB onnxruntime_test_providers_webgpu_plugin_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/webgpu/plugin/*.cc"
    "${TEST_SRC_DIR}/providers/webgpu/plugin/*.h"
  )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_webgpu_plugin_src})
endif()

if (onnxruntime_USE_CANN)
  file(GLOB_RECURSE onnxruntime_test_providers_cann_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cann/*"
    )
  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_cann_src})
endif()

# Disable training ops test for minimal build as a lot of these depend on loading an onnx model.
if (NOT onnxruntime_MINIMAL_BUILD)
  if (onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE orttraining_test_trainingops_cpu_src CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/compare_provider_test_utils.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/function_op_test_utils.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/*"
      )

    if (NOT onnxruntime_ENABLE_TRAINING)
      list(REMOVE_ITEM orttraining_test_trainingops_cpu_src
        "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cpu/tensorboard/summary_op_test.cc"
        )
    endif()

    list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cpu_src})

    if (onnxruntime_USE_CUDA)
      file(GLOB_RECURSE orttraining_test_trainingops_cuda_src CONFIGURE_DEPENDS
        "${ORTTRAINING_SOURCE_DIR}/test/training_ops/cuda/*"
        )
      list(APPEND onnxruntime_test_providers_src ${orttraining_test_trainingops_cuda_src})
    endif()
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

set(onnxruntime_test_internal_testing_ep_src)
if (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_EXTENDED_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_test_providers_internal_testing_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/internal_testing_ep/*"
    )
  list(APPEND onnxruntime_test_internal_testing_ep_src ${onnxruntime_test_providers_internal_testing_src})
endif()

set (ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR "${TEST_SRC_DIR}/shared_lib")
set (ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR "${TEST_SRC_DIR}/global_thread_pools")
set (ONNXRUNTIME_CUSTOM_OP_REGISTRATION_TEST_SRC_DIR "${TEST_SRC_DIR}/custom_op_registration")
set (ONNXRUNTIME_LOGGING_APIS_TEST_SRC_DIR "${TEST_SRC_DIR}/logging_apis")
set (ONNXRUNTIME_AUTOEP_TEST_SRC_DIR "${TEST_SRC_DIR}/autoep")
set (ONNXRUNTIME_EP_GRAPH_TEST_SRC_DIR "${TEST_SRC_DIR}/ep_graph")

set (onnxruntime_shared_lib_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/custom_op_utils.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_allocator.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_data_copy.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_ep_context_data_api.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_experimental_api.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_loading.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_nontensor_types.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_ort_format_models.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_run_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_runtime_path.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_session_options.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_version.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.h
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/utils.cc
          )

if (NOT onnxruntime_MINIMAL_BUILD)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_inference.cc)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_model_builder_api.cc)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_env_creation.cc)
endif()

if(onnxruntime_RUN_ONNX_TESTS)
  list(APPEND onnxruntime_shared_lib_test_SRC ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_io_types.cc)
endif()

set (onnxruntime_global_thread_pools_test_SRC
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/test_fixture.h
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_main.cc
          ${ONNXRUNTIME_GLOBAL_THREAD_POOLS_TEST_SRC_DIR}/test_inference.cc)

set (onnxruntime_webgpu_external_dawn_test_SRC
          ${TEST_SRC_DIR}/webgpu/external_dawn/main.cc)

set (onnxruntime_webgpu_delay_load_test_SRC
          ${TEST_SRC_DIR}/webgpu/delay_load/main.cc)

# tests from lowest level library up.
# the order of libraries should be maintained, with higher libraries being added first in the list

set(onnxruntime_test_common_libs
  onnxruntime_unittest_utils
  onnxruntime_common
)

set (onnxruntime_test_providers_dependencies ${onnxruntime_EXTERNAL_DEPENDENCIES})

if(onnxruntime_USE_CUDA AND NOT onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda)
endif()

if(onnxruntime_USE_CUDA AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cuda_plugin)
endif()

if(onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_webgpu)
endif()

if(onnxruntime_USE_CANN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_cann)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_VSINPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_vsinpu)
endif()

if(onnxruntime_USE_JSEP)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_js)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_DML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dml)
endif()

if(onnxruntime_USE_DNNL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_dnnl)
endif()

if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_migraphx)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml coreml_proto)
endif()

if(onnxruntime_USE_ACL)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_acl)
endif()

set(ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS
    # CUDA, TENSORRT, MIGRAPHX, DNNL, and OpenVINO are dynamically loaded at runtime.
    # QNN EP can be built as either a dynamic and static libs.
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_VSINPU}
    ${PROVIDERS_JS}
    ${PROVIDERS_SNPE}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_COREML}
    ${PROVIDERS_XNNPACK}
    ${PROVIDERS_AZURE}
)

if (onnxruntime_BUILD_QNN_EP_STATIC_LIB)
  list(APPEND ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS onnxruntime_providers_qnn)
endif()
if (onnxruntime_WEBGPU_LINKED_INTO_HOST)
  list(APPEND ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS onnxruntime_providers_webgpu)
endif()

set(ONNXRUNTIME_TEST_LIBS
    onnxruntime_session
    ${ONNXRUNTIME_INTEROP_TEST_LIBS}
    ${onnxruntime_libs}
    ${ONNXRUNTIME_TEST_STATIC_PROVIDER_LIBS}
    onnxruntime_optimizer
    ${onnxruntime_providers_target}
    onnxruntime_util
    onnxruntime_lora
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    ${ONNXRUNTIME_MLAS_LIBS}
    onnxruntime_common
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
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/tensorrt/tensorrt_execution_provider_utils.h")
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_tensorrt onnxruntime_providers_shared)
  list(APPEND onnxruntime_test_providers_libs ${TENSORRT_LIBRARY_INFER})
endif()

if(onnxruntime_USE_NV)
  # If an external project (e.g. dawn from Webgpu EP has already added a Vulkan::Headers target we shouldn't try to import another version of the Vulkan headers)
  if (NOT TARGET Vulkan::Headers)
    onnxruntime_fetchcontent_declare(
      vulkan_headers
      URL ${DEP_URL_vulkan_headers}
      URL_HASH SHA1=${DEP_SHA1_vulkan_headers}
      EXCLUDE_FROM_ALL
    )
    onnxruntime_fetchcontent_makeavailable(vulkan_headers)
  endif()
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nv_tensorrt_rtx/*)
  list(APPEND onnxruntime_test_framework_src_patterns  "${ONNXRUNTIME_ROOT}/core/providers/nv_tensorrt_rtx/nv_execution_provider_utils.h")
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nv_tensorrt_rtx onnxruntime_providers_shared)
  list(APPEND onnxruntime_test_providers_libs ${TENSORRT_LIBRARY_INFER})
  list(APPEND onnxruntime_test_providers_libs Vulkan::Headers)
endif()


if(onnxruntime_USE_MIGRAPHX)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/migraphx/*)
endif()

if(onnxruntime_USE_NNAPI_BUILTIN)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/nnapi/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_nnapi)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_nnapi)
endif()

if(onnxruntime_USE_JSEP)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/js/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_js)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_js)
endif()

if(onnxruntime_USE_WEBGPU AND NOT onnxruntime_USE_EP_API_ADAPTERS)
  file(GLOB_RECURSE onnxruntime_test_providers_webgpu_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/webgpu/*.cc"
    "${TEST_SRC_DIR}/providers/webgpu/*.h")

  list(APPEND onnxruntime_test_providers_src ${onnxruntime_test_providers_webgpu_src})
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_webgpu)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_webgpu)
endif()

# QNN EP tests require CPU EP op implementations for accuracy evaluation, so disable on minimal
# or reduced op builds.
if(onnxruntime_USE_QNN AND NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/qnn_node_group/*)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/qnn/optimizer/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_qnn)
  if(NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
    list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_shared)
  endif()
endif()

if(onnxruntime_USE_SNPE)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/snpe/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_snpe)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_snpe)
endif()

if(onnxruntime_USE_RKNPU)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/rknpu/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_rknpu)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_rknpu)
endif()

if(onnxruntime_USE_COREML)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/coreml/*.cc)
  if(APPLE)
    list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/coreml/*.mm)
  endif()
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_coreml coreml_proto)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_coreml coreml_proto)
endif()

if(onnxruntime_USE_XNNPACK)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/xnnpack/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_xnnpack)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_xnnpack)
endif()

if(onnxruntime_USE_AZURE)
  list(APPEND onnxruntime_test_framework_src_patterns  ${TEST_SRC_DIR}/providers/azure/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_azure)
  list(APPEND onnxruntime_test_providers_libs onnxruntime_providers_azure)
endif()

if (onnxruntime_USE_OPENVINO)
  list(APPEND onnxruntime_test_framework_src_patterns ${TEST_SRC_DIR}/providers/openvino/*)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_openvino)
  list(APPEND onnxruntime_test_providers_dependencies onnxruntime_providers_shared)
endif()

file(GLOB onnxruntime_test_framework_src CONFIGURE_DEPENDS
  ${onnxruntime_test_framework_src_patterns}
  )

if(IOS)
  # The ONNX test runner library is not built for iOS.
  list(REMOVE_ITEM onnxruntime_test_framework_src "${TEST_SRC_DIR}/framework/onnx_test_loader_test.cc")
endif()

#This is a small wrapper library that shouldn't use any onnxruntime internal symbols(except onnxruntime_common).
#Because it could dynamically link to onnxruntime. Otherwise you will have two copies of onnxruntime in the same
#process and you won't know which one you are testing.
onnxruntime_add_static_library(onnxruntime_test_utils ${onnxruntime_test_utils_src})
if(MSVC)
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
          "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
  target_compile_options(onnxruntime_test_utils PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
else()
  target_include_directories(onnxruntime_test_utils PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  onnxruntime_disable_gtest_character_conversion_as_error(onnxruntime_test_utils)
endif()
if (onnxruntime_USE_NCCL)
  target_include_directories(onnxruntime_test_utils PRIVATE ${NCCL_INCLUDE_DIRS})
endif()
onnxruntime_add_include_to_target(onnxruntime_test_utils onnxruntime_common onnxruntime_framework onnxruntime_session GTest::gtest GTest::gmock onnx onnx_proto flatbuffers::flatbuffers nlohmann_json::nlohmann_json Boost::mp11 safeint_interface Eigen3::Eigen)
if (onnxruntime_USE_DML)
  target_add_dml(onnxruntime_test_utils)
endif()
add_dependencies(onnxruntime_test_utils ${onnxruntime_EXTERNAL_DEPENDENCIES})
target_include_directories(onnxruntime_test_utils PUBLIC "${TEST_SRC_DIR}/util/include" PRIVATE
        ${ONNXRUNTIME_ROOT})
if (onnxruntime_USE_CUDA AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
  target_compile_definitions(onnxruntime_test_utils PRIVATE
    ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE
    ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP=1)
endif()
set_target_properties(onnxruntime_test_utils PROPERTIES FOLDER "ONNXRuntimeTest")
source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_test_utils_src})

# onnxruntime_unittest_utils
# This is static library containing utilities that are specifically for unit tests.
# Unlike onnxruntime_test_utils, the source files here may have dependencies on internal onnxruntime code.
# Thus, onnxruntime_unittest_utils is not suitable for use in programs that don't link with internal onnxruntime
# libraries.
block()

file(GLOB onnxruntime_unittest_utils_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/unittest_util/*.h"
    "${TEST_SRC_DIR}/unittest_util/*.cc")

if(onnxruntime_MINIMAL_BUILD OR onnxruntime_REDUCED_OPS_BUILD)
  # some exclusions from a minimal or reduced ops build
  list(REMOVE_ITEM onnxruntime_unittest_utils_src
    "${TEST_SRC_DIR}/unittest_util/base_tester.cc"
    "${TEST_SRC_DIR}/unittest_util/base_tester.h"
    "${TEST_SRC_DIR}/unittest_util/function_test_util.cc"
    "${TEST_SRC_DIR}/unittest_util/function_test_util.h"
    "${TEST_SRC_DIR}/unittest_util/graph_transform_test_builder.cc"
    "${TEST_SRC_DIR}/unittest_util/graph_transform_test_builder.h"
    "${TEST_SRC_DIR}/unittest_util/model_tester.h"
    "${TEST_SRC_DIR}/unittest_util/op_tester.cc"
    "${TEST_SRC_DIR}/unittest_util/op_tester.h"
    "${TEST_SRC_DIR}/unittest_util/qdq_test_utils.cc"
    "${TEST_SRC_DIR}/unittest_util/qdq_test_utils.h"
  )

  if (onnxruntime_MINIMAL_BUILD)
    list(REMOVE_ITEM onnxruntime_unittest_utils_src
      "${TEST_SRC_DIR}/unittest_util/test_dynamic_plugin_ep.cc"
      "${TEST_SRC_DIR}/unittest_util/test_dynamic_plugin_ep.h"
    )
  endif()
endif()

onnxruntime_add_static_library(onnxruntime_unittest_utils ${onnxruntime_unittest_utils_src})

set(onnxruntime_unittest_utils_libs ${ONNXRUNTIME_TEST_LIBS})
get_target_property(onnxruntime_providers_type ${onnxruntime_providers_target} TYPE)
if (onnxruntime_providers_type STREQUAL "OBJECT_LIBRARY")
  # Linking an object library into a static library copies its objects into that archive. Every
  # target that links onnxruntime_unittest_utils also links the providers target directly, so only
  # its usage requirements are needed here.
  list(REMOVE_ITEM onnxruntime_unittest_utils_libs ${onnxruntime_providers_target})
  list(APPEND onnxruntime_unittest_utils_libs $<COMPILE_ONLY:${onnxruntime_providers_target}>)
endif()

target_link_libraries(onnxruntime_unittest_utils PUBLIC
                      onnx
                      GTest::gtest
                      GTest::gmock
                      onnxruntime_test_utils
                      ${onnxruntime_unittest_utils_libs}
                      ${onnxruntime_EXTERNAL_LIBRARIES}
                      )

set_target_properties(onnxruntime_unittest_utils PROPERTIES FOLDER "ONNXRuntimeTest")

source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_unittest_utils_src})

endblock()

if(NOT IOS)
    set(onnx_test_runner_src_dir ${TEST_SRC_DIR}/onnx)
    file(GLOB onnx_test_runner_common_srcs CONFIGURE_DEPENDS
        ${onnx_test_runner_src_dir}/*.h
        ${onnx_test_runner_src_dir}/*.cc)

    list(REMOVE_ITEM onnx_test_runner_common_srcs ${onnx_test_runner_src_dir}/main.cc)

    # if training is disabled, endian_utils are still used in tests
    if (NOT onnxruntime_ENABLE_TRAINING)
      list(APPEND onnx_test_runner_common_srcs
          ${ONNXRUNTIME_ROOT}/core/framework/endian_utils.cc
          ${ONNXRUNTIME_ROOT}/core/framework/endian_utils.h
          )
    endif ()

    onnxruntime_add_static_library(onnx_test_runner_common ${onnx_test_runner_common_srcs})
    if(MSVC)
      target_compile_options(onnx_test_runner_common PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    else()
      target_include_directories(onnx_test_runner_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
    endif()
    if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
      #TODO: fix the warnings, they are dangerous
      target_compile_options(onnx_test_runner_common PRIVATE "/wd4244")
    endif()
    onnxruntime_add_include_to_target(onnx_test_runner_common onnxruntime_common onnxruntime_framework
            onnxruntime_test_utils onnx onnx_proto re2::re2 flatbuffers::flatbuffers Boost::mp11 safeint_interface Eigen3::Eigen)

    add_dependencies(onnx_test_runner_common onnx_test_data_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
    target_include_directories(onnx_test_runner_common PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})

    set_target_properties(onnx_test_runner_common PROPERTIES FOLDER "ONNXRuntimeTest")
    set(onnx_test_runner_common_lib onnx_test_runner_common)
endif()

set(all_tests
    ${onnxruntime_test_common_src}
    ${onnxruntime_test_ir_src}
    ${onnxruntime_test_optimizer_src}
    ${onnxruntime_test_framework_src}
    ${onnxruntime_test_providers_src}
    ${onnxruntime_test_internal_testing_ep_src}
    ${onnxruntime_test_quantization_src}
    ${onnxruntime_test_flatbuffers_src}
    ${onnxruntime_test_lora_src}
)

set(onnxruntime_test_providers_runtime_dependencies)
if (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS AND NOT onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
  if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
    set(onnxruntime_test_cuda_kernels_src_patterns "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/*.cc")
  endif()

  file(GLOB onnxruntime_test_providers_cuda_ut_src CONFIGURE_DEPENDS
    "${TEST_SRC_DIR}/providers/cuda/test_cases/*"
    ${onnxruntime_test_cuda_kernels_src_patterns}
  )

  if (onnxruntime_DISABLE_CONTRIB_OPS OR onnxruntime_CUDA_MINIMAL)
    list(REMOVE_ITEM onnxruntime_test_providers_cuda_ut_src
      "${TEST_SRC_DIR}/providers/cuda/test_cases/generation_cuda_impl_test.cc")
  endif()

  # cuda_plugin_test_shims.cc provides onnxruntime::GetEnvironmentVar for the
  # BUILD_CUDA_EP_AS_PLUGIN unit-test object library, where provider_bridge_provider.cc
  # is not linked. In the non-plugin onnxruntime_providers_cuda_ut module the symbol is
  # already defined by provider_bridge_provider.cc (via onnxruntime_providers_cuda_obj),
  # so exclude the shim here to avoid a duplicate-definition link error.
  list(REMOVE_ITEM onnxruntime_test_providers_cuda_ut_src
    "${TEST_SRC_DIR}/providers/cuda/test_cases/cuda_plugin_test_shims.cc")

  # onnxruntime_providers_cuda_ut is only for unittests.
  set(onnxruntime_cuda_ut_compile_targets onnxruntime_providers_cuda_ut)
  if (WIN32)
    # Inspect the compiled tests before linking the host, without depending on the module's link.
    onnxruntime_add_object_library(onnxruntime_providers_cuda_ut_objects ${onnxruntime_test_providers_cuda_ut_src})
    set(onnxruntime_cuda_ut_sources $<TARGET_OBJECTS:onnxruntime_providers_cuda_ut_objects>)
    list(APPEND onnxruntime_cuda_ut_compile_targets onnxruntime_providers_cuda_ut_objects)
  else()
    set(onnxruntime_cuda_ut_sources ${onnxruntime_test_providers_cuda_ut_src})
  endif()
  onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_ut ${onnxruntime_cuda_ut_sources} $<TARGET_OBJECTS:onnxruntime_providers_cuda_obj>)
  foreach(cuda_ut_target IN LISTS onnxruntime_cuda_ut_compile_targets)
    config_cuda_provider_shared_module(${cuda_ut_target})
    target_compile_options(${cuda_ut_target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_NVCC_THREADS}\">")
    onnxruntime_add_include_to_target(${cuda_ut_target} GTest::gtest GTest::gmock onnxruntime_test_utils)
    add_dependencies(${cuda_ut_target} onnxruntime_test_utils)
    target_include_directories(${cuda_ut_target} PRIVATE ${ONNXRUNTIME_ROOT}/core/mickey)
    target_compile_definitions(${cuda_ut_target} PRIVATE ORT_API_MANUAL_INIT)
    target_link_libraries(${cuda_ut_target} PRIVATE GTest::gtest GTest::gmock ${ONNXRUNTIME_MLAS_LIBS}
                                                  ${PROTOBUF_LIB})
    if(WIN32)
      target_link_libraries(${cuda_ut_target} PRIVATE onnx_proto)
    endif()
    if (MSVC)
      # Cutlass code has an issue with warning C4100: 'magic': unreferenced formal parameter.
      target_compile_options(${cuda_ut_target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4100>"
                    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4100>")
    endif()
  endforeach()
  # Link architecture-specific OBJECT libraries (same as onnxruntime_providers_cuda).
  if(TARGET onnxruntime_providers_cuda_sm90_tma)
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE onnxruntime_providers_cuda_sm90_tma)
  endif()
  if(TARGET onnxruntime_providers_cuda_sm120_tma)
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE onnxruntime_providers_cuda_sm120_tma)
  endif()
  if(TARGET onnxruntime_providers_cuda_flash_attention)
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE onnxruntime_providers_cuda_flash_attention)
  endif()
  if(TARGET onnxruntime_providers_cuda_xqa)
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE onnxruntime_providers_cuda_xqa)
  endif()
  if(TARGET onnxruntime_providers_cuda_llm)
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE onnxruntime_providers_cuda_llm)
  endif()
  if(TARGET onnxruntime_providers_cuda_llm_fp4)
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE onnxruntime_providers_cuda_llm_fp4)
  endif()
  list(APPEND onnxruntime_test_providers_runtime_dependencies onnxruntime_providers_cuda_ut)
endif()

if (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN AND
    NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
  set(onnxruntime_test_providers_cuda_plugin_internal_test_src
    "${TEST_SRC_DIR}/providers/cuda/test_cases/allocator_cuda_test.cc"
    "${TEST_SRC_DIR}/providers/cuda/test_cases/cuda_utils_test.cc"
    "${TEST_SRC_DIR}/providers/cuda/test_cases/group_query_attention_workspace_header_test.cc"
    "${TEST_SRC_DIR}/providers/cuda/test_cases/packed_attention_workspace_header_test.cc"
    "${TEST_SRC_DIR}/providers/cuda/test_cases/reduction_functions_test.cc"
  )
  # matmul_nbits_workspace_test.cc / matmul_nbits_e2e_workspace_test.cc are intentionally excluded
  # from the plugin-internal test list: they construct the concrete in-tree CUDAExecutionProvider
  # and MatMulNBits<T> kernel types and downcast a plugin-world OpKernel to MatMulNBits<MLFloat16>,
  # which is undefined behavior against a real PluginEpOpKernel (whose DeclareWorkspaceRequirements()
  # is the default no-op under BUILD_CUDA_EP_AS_PLUGIN). Re-enable them here once workspace
  # estimation/declaration is bridged through OrtKernelImpl for plugin kernels.

  if (NOT onnxruntime_DISABLE_CONTRIB_OPS)
    list(APPEND onnxruntime_test_providers_cuda_plugin_internal_test_src
      "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/gemm_tactic_cache_test.cc"
      "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/moe_gemm_int2_test.cc"
      "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/qmoe_fp4_to_fp8_kernel_test.cc"
      "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/qmoe_fp8_compaction_test.cc"
      "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/softmax_topk_kernel_test.cc"
    )
  endif()

  list(APPEND all_tests ${onnxruntime_test_providers_cuda_plugin_internal_test_src})

  if (TARGET onnxruntime_providers_cuda_plugin)
    if (NOT onnxruntime_DISABLE_CONTRIB_OPS)
      set_property(SOURCE "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/moe_gemm_int2_test.cc"
        APPEND PROPERTY INCLUDE_DIRECTORIES
        $<TARGET_PROPERTY:onnxruntime_providers_cuda_plugin,INCLUDE_DIRECTORIES>)
      set_property(SOURCE "${TEST_SRC_DIR}/contrib_ops/cuda_kernels/moe_gemm_int2_test.cc"
        APPEND PROPERTY COMPILE_DEFINITIONS
        BUILD_CUDA_EP_AS_PLUGIN ORT_API_MANUAL_INIT ORT_USE_EP_API_ADAPTERS=1)
    endif()
    set(onnxruntime_providers_cuda_plugin_ut_impl_src
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_allocator.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_call.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_utils.cu"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cudnn_common.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cudnn_loader.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/cufft_loader.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/fpgeneric.cu"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/reduction/reduction_functions.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/reduction/reduction_functions.cu"
      "${TEST_SRC_DIR}/providers/cuda/test_cases/cuda_plugin_test_shims.cc"
    )

    if (NOT onnxruntime_DISABLE_CONTRIB_OPS)
      list(APPEND onnxruntime_providers_cuda_plugin_ut_impl_src
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/llm/gemm_tactic_cache.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_fp16_uint2.cu"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/qmoe_kernels.cu"
      )
      if (onnxruntime_USE_FP4_QMOE)
        list(APPEND onnxruntime_providers_cuda_plugin_ut_impl_src
          "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/llm/moe_gemm/moe_gemv_fp4.cu")
      endif()
    endif()

    onnxruntime_add_object_library(onnxruntime_providers_cuda_plugin_ut_impl ${onnxruntime_providers_cuda_plugin_ut_impl_src})
    set_target_properties(onnxruntime_providers_cuda_plugin_ut_impl PROPERTIES
      CUDA_STANDARD 20
      CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(onnxruntime_providers_cuda_plugin_ut_impl PRIVATE
      $<TARGET_PROPERTY:onnxruntime_providers_cuda_plugin,INCLUDE_DIRECTORIES>)
    target_compile_definitions(onnxruntime_providers_cuda_plugin_ut_impl PRIVATE
      BUILD_CUDA_EP_AS_PLUGIN
      ONNX_ML=1
      ONNX_NAMESPACE=onnx
      ONNX_USE_LITE_PROTO=1
      ORT_API_MANUAL_INIT
      ORT_USE_EP_API_ADAPTERS=1)
    target_compile_options(onnxruntime_providers_cuda_plugin_ut_impl PRIVATE
      ${_cuda_plugin_shared_compile_options}
      "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
      "$<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>"
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_plugin_nvcc_threads}\">")
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
      # The internal INT2 test TU exposes third-party header diagnostics under CUDA 12 with -Werror.
      target_compile_options(onnxruntime_providers_cuda_plugin_ut_impl PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=970>"
        "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2189>")
    endif()
    add_dependencies(onnxruntime_providers_cuda_plugin_ut_impl ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set(onnxruntime_providers_cuda_plugin_ut_impl_objects $<TARGET_OBJECTS:onnxruntime_providers_cuda_plugin_ut_impl>)
  endif()
endif()

set(all_dependencies ${onnxruntime_test_providers_dependencies} ${onnxruntime_test_providers_runtime_dependencies})

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND all_tests ${onnxruntime_test_training_src})
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
    list(APPEND all_tests ${onnxruntime_test_training_api_src})
endif()


if (onnxruntime_USE_OPENVINO)
  list(APPEND all_tests ${onnxruntime_test_openvino_src})
endif()

if(WIN32)
  list(APPEND onnxruntime_test_providers_libs Advapi32)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (NOT onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
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
  list(REMOVE_ITEM all_tests
    "${TEST_SRC_DIR}/providers/cpu/reduction/reduction_ops_test.cc"
    "${TEST_SRC_DIR}/providers/cpu/tensor/grid_sample_test.cc"
    "${TEST_SRC_DIR}/providers/cpu/tensor/grid_sample_test_custom.cc")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR IOS)
   # Because we do not run these model tests in our web or iOS CI build pipelines, and some test code uses C++17
   # filesystem functions that are not available in the iOS version we target.
   message("Disable model tests")
   list(REMOVE_ITEM all_tests
      "${TEST_SRC_DIR}/providers/cpu/model_tests.cc"
    )
endif()

set(test_all_args)
if (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV)
  # TRT EP CI takes much longer time when updating to TRT 8.2
  # So, we only run trt ep and exclude other eps to reduce CI test time.
  #
  # The test names of model tests were using sequential number in the past.
  # This PR https://github.com/microsoft/onnxruntime/pull/10220 (Please see ExpandModelName function in model_tests.cc for more details)
  # made test name contain the "ep" and "model path" information, so we can easily filter the tests using cuda ep or other ep with *cpu_* or *xxx_*.
  list(APPEND test_all_args "--gtest_filter=-*cpu_*:*cuda_*" )
endif ()
if(NOT onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
  list(REMOVE_ITEM all_tests ${TEST_SRC_DIR}/providers/cuda/cuda_provider_test.cc)
endif()

partition_provider_test_srcs(all_tests onnxruntime_provider_test_srcs onnxruntime_test_all_srcs)

if ((onnxruntime_MINIMAL_BUILD OR onnxruntime_REDUCED_OPS_BUILD)
    AND NOT onnxruntime_DISABLE_CONTRIB_OPS)
  # Shape-only validation does not require ONNX loading; minimal builds have no provider test binary.
  list(APPEND onnxruntime_test_all_srcs
    "${TEST_SRC_DIR}/contrib_ops/attention_validation_test.cc"
  )
endif()

if (onnxruntime_USE_OPENVINO)
  # ov_protobuf_utils.cpp lives under core/providers (not test/), so partition_provider_test_srcs
  # would route it to onnxruntime_test_all. Append it here after the partition so it is compiled into
  # onnxruntime_provider_test alongside openvino_ov_protobuf_utils_test.cc, because the OpenVINO EP
  # is a dynamically-loaded module and is not statically linked into the test binary. Same for 
  # ov_factory.cc, which is used by openvino_ep_supported_devices_test.cc to exercise 
  # GetSupportedDevices directly.
  list(APPEND onnxruntime_provider_test_srcs
       ${ONNXRUNTIME_ROOT}/core/providers/openvino/ov_protobuf_utils.cpp
       ${ONNXRUNTIME_ROOT}/core/providers/openvino/ov_factory.cc)
endif()

# Workarounds for onnxruntime test targets.
function(onnxruntime_apply_test_target_workarounds target)
  if (MSVC)
    # TODO: The test code for OpenVINO, QNN, and WebGPU is getting flagged with a warning from ABSL for unreachable code.
    # Need to figure out how those particular targets/build variants are failing, but regular windows is not.
    target_compile_options(${target} PRIVATE "/wd4702")
  endif()

  # TODO fix shorten-64-to-32 warnings
  # there are some in builds where sizeof(size_t) != sizeof(int64_t), e.g., in 'ONNX Runtime Web CI Pipeline'
  if (HAS_SHORTEN_64_TO_32 AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    target_compile_options(${target} PRIVATE -Wno-error=shorten-64-to-32)
  endif()
endfunction()

# Set environment variables for plugin EP tests when run via CTest.
function(onnxruntime_set_plugin_ep_test_environment target)
  if(onnxruntime_WEBGPU_STATIC_PLUGIN)
    # The WebGPU plugin EP is linked into the test binary and registered by ORT core, so there is no
    # library to register by path here. onnxruntime_set_webgpu_plugin_ep_test_definitions() below gives
    # test_main.cc an equivalent compiled-in default config.
  elseif(onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS)
    set(ORT_PLUGIN_EP_JSON_CONFIG "{\"ep_library_registration_name\": \"WebGPU_PluginEP\", \"ep_library_path\": \"$<TARGET_FILE_NAME:onnxruntime_providers_webgpu>\", \"selected_ep_name\": \"WebGpuExecutionProvider\"}")
    set_tests_properties(${target} PROPERTIES
      ENVIRONMENT "ORT_UNIT_TEST_MAIN_DYNAMIC_PLUGIN_EP_CONFIG_JSON=${ORT_PLUGIN_EP_JSON_CONFIG}"
    )
  # TODO: add for other plugin EPs if needed
  # elseif()
  endif()
endfunction()

# Route the WebGPU EP through the dynamic plugin EP infrastructure in plugin builds
# (--use_webgpu shared_lib or --use_webgpu static_plugin).
# Without initializing the infra in test_main.cc, WebGpuExecutionProviderWithOptions() (default_providers.cc,
# adapters branch) returns null and every WebGPU test skips itself, leaving the plugin path with no test coverage.
# This must be applied to every test target that links test_main.cc and runs WebGPU tests.
function(onnxruntime_set_webgpu_plugin_ep_test_definitions target)
  if(NOT (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS))
    return()
  endif()

  target_compile_definitions(${target} PRIVATE
    ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE
    ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP=1)

  if (onnxruntime_WEBGPU_STATIC_PLUGIN)
    # The plugin EP is linked into the test binary and registered by ORT core during environment creation,
    # so there is no library path to dlopen and no separate build-order dependency needed.
    target_compile_definitions(${target} PRIVATE ORT_UNIT_TEST_HAS_WEBGPU_STATIC_PLUGIN_EP=1)
  else()
    target_compile_definitions(${target} PRIVATE
      ORT_UNIT_TEST_WEBGPU_PLUGIN_EP_LIBRARY_PATH="$<TARGET_FILE_NAME:onnxruntime_providers_webgpu>")
    # The plugin EP DLL is dlopen'd at test-run time (not linked), so add an explicit build-order
    # dependency to ensure it (and its co-located dawn/dxcompiler DLLs) exist before the tests run.
    add_dependencies(${target} onnxruntime_providers_webgpu)
  endif()
endfunction()

function(onnxruntime_apply_emscripten_test_link_settings target)
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set_target_properties(${target} PROPERTIES LINK_DEPENDS ${TEST_SRC_DIR}/wasm/onnxruntime_test_adapter.js)
    set_target_properties(${target} PROPERTIES LINK_DEPENDS ${ONNXRUNTIME_ROOT}/wasm/pre.js)
    set_target_properties(${target} PROPERTIES LINK_FLAGS "-s STACK_SIZE=5242880 -s INITIAL_MEMORY=536870912 -s ALLOW_MEMORY_GROWTH=1 -s MAXIMUM_MEMORY=4294967296 -s INCOMING_MODULE_JS_API=[preRun,locateFile,arguments,onExit,wasmMemory,buffer,instantiateWasm] --pre-js \"${TEST_SRC_DIR}/wasm/onnxruntime_test_adapter.js\" --pre-js \"${ONNXRUNTIME_ROOT}/wasm/pre.js\" -s \"EXPORTED_RUNTIME_METHODS=['FS']\" --preload-file ${CMAKE_CURRENT_BINARY_DIR}/testdata@/testdata -s EXIT_RUNTIME=1")
    if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -s DEFAULT_PTHREAD_STACK_SIZE=131072 -s PROXY_TO_PTHREAD=1")
    endif()
    if (onnxruntime_USE_JSEP)
      set_target_properties(${target} PROPERTIES LINK_DEPENDS ${ONNXRUNTIME_ROOT}/wasm/pre-jsep.js)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " --pre-js \"${ONNXRUNTIME_ROOT}/wasm/pre-jsep.js\"")
    endif()

    ###
    ### if you want to investigate or debug a test failure in ${target}, replace the following line.
    ### those flags slow down the CI test significantly, so we don't use them by default.
    ###
    #   set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=2 -s SAFE_HEAP=1 -s STACK_OVERFLOW_CHECK=2")
    set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS " -s ASSERTIONS=0 -s SAFE_HEAP=0 -s STACK_OVERFLOW_CHECK=1")
  endif()
endfunction()

list(APPEND onnxruntime_test_all_srcs ${onnxruntime_unittest_main_src})
AddTest(
  TARGET onnxruntime_test_all
  SOURCES ${onnxruntime_test_all_srcs}
  LIBS
    ${onnx_test_runner_common_lib} ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
    onnx_test_data_proto
  DEPENDS ${all_dependencies}
  TEST_ARGS ${test_all_args}
)
target_include_directories(onnxruntime_test_all PRIVATE ${ONNXRUNTIME_ROOT}/core/flatbuffers/schema) # ort.fbs.h

onnxruntime_apply_test_target_workarounds(onnxruntime_test_all)

if(NOT onnxruntime_MINIMAL_BUILD AND NOT CMAKE_CROSSCOMPILING
  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  foreach(fixture IN ITEMS valid missing_export)
    onnxruntime_add_shared_library(onnxruntime_provider_bridge_${fixture}_fixture
      "${TEST_SRC_DIR}/shared_lib/provider_bridge_test_library.cc")
    set_target_properties(onnxruntime_provider_bridge_${fixture}_fixture PROPERTIES FOLDER "ONNXRuntimeTest")
    if(CMAKE_SYSTEM_NAME MATCHES "AIX")
      set_target_properties(onnxruntime_provider_bridge_${fixture}_fixture PROPERTIES AIX_SHARED_LIBRARY_ARCHIVE OFF)
    endif()
  endforeach()
  target_compile_definitions(onnxruntime_provider_bridge_valid_fixture PRIVATE ORT_TEST_PROVIDER_SET_HOST)

  foreach(mode IN ITEMS normal no_exceptions)
    set(bridge_test_target onnxruntime_provider_bridge_${mode}_test)
    set(bridge_test_sources "${TEST_SRC_DIR}/shared_lib/provider_bridge_test.cc" ${onnxruntime_unittest_main_src})
    if(mode STREQUAL "no_exceptions")
      list(APPEND bridge_test_sources
        "${ONNXRUNTIME_ROOT}/core/session/provider_bridge_ort.cc"
        "${ONNXRUNTIME_ROOT}/core/common/helper.cc")
    endif()
    AddTest(TARGET ${bridge_test_target}
      SOURCES ${bridge_test_sources}
      LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
      DEPENDS onnxruntime_provider_bridge_valid_fixture onnxruntime_provider_bridge_missing_export_fixture)
    set_target_properties(${bridge_test_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${bridge_test_target}")
    target_compile_definitions(${bridge_test_target} PRIVATE
      ORT_PROVIDER_BRIDGE_VALID_TEST_LIBRARY="$<TARGET_FILE_NAME:onnxruntime_provider_bridge_valid_fixture>"
      ORT_PROVIDER_BRIDGE_MISSING_EXPORT_TEST_LIBRARY="$<TARGET_FILE_NAME:onnxruntime_provider_bridge_missing_export_fixture>")
    add_custom_command(TARGET ${bridge_test_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:onnxruntime_provider_bridge_valid_fixture>
        $<TARGET_FILE:onnxruntime_provider_bridge_missing_export_fixture>
        $<TARGET_FILE_DIR:${bridge_test_target}>)
    if(onnxruntime_providers_webgpu_dll_deps)
      add_custom_command(TARGET ${bridge_test_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${onnxruntime_providers_webgpu_dll_deps}" "$<TARGET_FILE_DIR:${bridge_test_target}>"
        COMMAND_EXPAND_LISTS
        VERBATIM)
    endif()
    if(mode STREQUAL "no_exceptions")
      target_compile_definitions(${bridge_test_target} PRIVATE ORT_NO_EXCEPTIONS)
      target_include_directories(${bridge_test_target} PRIVATE
        $<TARGET_PROPERTY:onnxruntime_session,INCLUDE_DIRECTORIES>)
    endif()
  endforeach()
endif()

if(NOT onnxruntime_MINIMAL_BUILD AND NOT CMAKE_CROSSCOMPILING
  AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten"
  AND NOT CMAKE_SYSTEM_NAME STREQUAL "AIX")
  foreach(fixture IN ITEMS shared valid missing_export)
    set(probe_fixture onnxruntime_optional_probe_${fixture}_fixture)
    onnxruntime_add_shared_library(${probe_fixture}
      "${TEST_SRC_DIR}/shared_lib/optional_provider_probe_test_library.cc")
    target_include_directories(${probe_fixture} PRIVATE
      $<TARGET_PROPERTY:onnxruntime_session,INCLUDE_DIRECTORIES>)
    target_link_libraries(${probe_fixture} PRIVATE onnxruntime_common onnx)
    target_compile_definitions(${probe_fixture} PRIVATE ORT_TEST_OPTIONAL_PROVIDER_${fixture})
    set_target_properties(${probe_fixture} PROPERTIES FOLDER "ONNXRuntimeTest")
  endforeach()
  foreach(mode IN ITEMS normal no_exceptions)
    set(probe_target onnxruntime_optional_provider_probe_${mode}_test)
    set(probe_sources "${TEST_SRC_DIR}/shared_lib/optional_provider_probe_test.cc" ${onnxruntime_unittest_main_src})
    if(mode STREQUAL "no_exceptions")
      list(APPEND probe_sources
        "${ONNXRUNTIME_ROOT}/core/session/provider_bridge_ort.cc"
        "${ONNXRUNTIME_ROOT}/core/common/helper.cc")
    endif()
    AddTest(TARGET ${probe_target}
      SOURCES ${probe_sources}
      LIBS ${onnxruntime_test_providers_libs} ${onnxruntime_test_common_libs}
      DEPENDS onnxruntime_optional_probe_shared_fixture onnxruntime_optional_probe_valid_fixture
              onnxruntime_optional_probe_missing_export_fixture)
    set_target_properties(${probe_target} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/${probe_target}")
    foreach(fixture IN ITEMS shared valid missing_export)
      target_compile_definitions(${probe_target} PRIVATE
        ORT_OPTIONAL_PROBE_${fixture}_LIBRARY="$<TARGET_FILE_NAME:onnxruntime_optional_probe_${fixture}_fixture>")
      add_custom_command(TARGET ${probe_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${probe_target}>/fixtures"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "$<TARGET_FILE:onnxruntime_optional_probe_${fixture}_fixture>"
          "$<TARGET_FILE_DIR:${probe_target}>/fixtures"
        VERBATIM)
    endforeach()
    if(onnxruntime_providers_webgpu_dll_deps)
      add_custom_command(TARGET ${probe_target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${onnxruntime_providers_webgpu_dll_deps}" "$<TARGET_FILE_DIR:${probe_target}>"
        COMMAND_EXPAND_LISTS
        VERBATIM)
    endif()
    if(mode STREQUAL "no_exceptions")
      target_compile_definitions(${probe_target} PRIVATE ORT_NO_EXCEPTIONS ONNX_NO_EXCEPTIONS)
      if(MSVC)
        target_compile_options(${probe_target} PRIVATE /EHs-c- /wd4530)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
          target_compile_options(${probe_target} PRIVATE /clang:-fno-exceptions)
        endif()
      else()
        target_compile_options(${probe_target} PRIVATE -fno-exceptions)
      endif()
      target_include_directories(${probe_target} PRIVATE
        $<TARGET_PROPERTY:onnxruntime_session,INCLUDE_DIRECTORIES>)
    endif()
  endforeach()
endif()

if (onnxruntime_USE_CUDA AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
  # ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE is required so that test_main.cc initializes
  # the dynamic plugin EP infrastructure. onnxruntime_test_utils (which compiles default_providers.cc)
  # already routes DefaultCudaExecutionProvider() through the plugin infrastructure in plugin builds;
  # without initializing it here, that routing returns a null EP and tests crash.
  target_compile_definitions(onnxruntime_test_all PRIVATE
    ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE
    ORT_UNIT_TEST_CUDA_PLUGIN_EP_LIBRARY_PATH="$<TARGET_FILE_NAME:onnxruntime_providers_cuda_plugin>"
    ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP=1)
endif()

if (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS)
  onnxruntime_set_webgpu_plugin_ep_test_definitions(onnxruntime_test_all)
endif()

if (MSVC)
  # The warning means the type of two integral values around a binary operator is narrow than their result.
  # If we promote the two input values first, it could be more tolerant to integer overflow.
  # However, this is test code. We are less concerned.
  target_compile_options(onnxruntime_test_all PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26451>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26451>")
  target_compile_options(onnxruntime_test_all PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4244>"
                "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4244>")

  # Avoid this compile error in graph_transform_test.cc and qdq_transformer_test.cc:
  # fatal error C1128: number of sections exceeded object file format limit: compile with /bigobj
  set_property(SOURCE "${TEST_SRC_DIR}/optimizer/graph_transform_test.cc"
                      "${TEST_SRC_DIR}/optimizer/qdq_transformer_test.cc"
               APPEND PROPERTY COMPILE_OPTIONS "/bigobj")
else()
  target_compile_options(onnxruntime_test_all PRIVATE "-Wno-parentheses")
endif()

if (MSVC AND onnxruntime_ENABLE_STATIC_ANALYSIS)
# attention_op_test.cc: Function uses '49152' bytes of stack:  exceeds /analyze:stacksize '16384'..
target_compile_options(onnxruntime_test_all PRIVATE  "/analyze:stacksize 131072")
endif()

#In AIX + gcc compiler ,crash is observed with the usage of googletest EXPECT_THROW,
#because some needed symbol is garbaged out by linker.
#So, fix is to exports the symbols from executable.
#Another way is to use -Wl,-bkeepfile for each object file where EXPECT_THROW is used like below
#target_link_options(onnxruntime_test_all PRIVATE "-Wl,-bkeepfile:CMakeFiles/onnxruntime_test_all.dir${TEST_SRC_DIR}/framework/tensor_test.cc.o")
if (CMAKE_SYSTEM_NAME MATCHES "AIX" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set_target_properties(onnxruntime_test_all PROPERTIES ENABLE_EXPORTS 1)
endif()

# the default logger tests conflict with the need to have an overall default logger
# so skip in this type of
target_compile_definitions(onnxruntime_test_all PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
if (IOS)
  target_compile_definitions(onnxruntime_test_all_xc PUBLIC -DSKIP_DEFAULT_LOGGER_TESTS)
endif()
if(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE)
  target_compile_definitions(onnxruntime_test_all PUBLIC -DRUN_MODELTEST_IN_DEBUG_MODE)
endif()
if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  target_compile_definitions(onnxruntime_test_all PRIVATE DEBUG_NODE_INPUTS_OUTPUTS)
endif()

if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  target_link_libraries(onnxruntime_test_all PRIVATE Python::Python)
endif()
onnxruntime_apply_emscripten_test_link_settings(onnxruntime_test_all)

if (onnxruntime_ENABLE_ATEN)
  target_compile_definitions(onnxruntime_test_all PRIVATE ENABLE_ATEN)
endif()

set(test_data_target onnxruntime_test_all)

onnxruntime_add_static_library(onnx_test_data_proto ${TEST_SRC_DIR}/proto/tml.proto)
add_dependencies(onnx_test_data_proto onnx_proto ${onnxruntime_EXTERNAL_DEPENDENCIES})
#onnx_proto target should mark this definition as public, instead of private
target_compile_definitions(onnx_test_data_proto PRIVATE "-DONNX_API=")
onnxruntime_add_include_to_target(onnx_test_data_proto onnx_proto ${PROTOBUF_LIB})
if (MSVC)
    # Cutlass code has an issue with the following:
    # warning C4100: 'magic': unreferenced formal parameter
    target_compile_options(onnx_test_data_proto PRIVATE "/wd4100")
endif()
target_include_directories(onnx_test_data_proto PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
set_target_properties(onnx_test_data_proto PROPERTIES FOLDER "ONNXRuntimeTest")
if(NOT DEFINED onnx_SOURCE_DIR)
  find_path(onnx_SOURCE_DIR NAMES "onnx/onnx-ml.proto3" "onnx/onnx-ml.proto" REQUIRED)
endif()
onnxruntime_protobuf_generate(APPEND_PATH IMPORT_DIRS ${onnx_SOURCE_DIR} TARGET onnx_test_data_proto)

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

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if (onnxruntime_USE_SNPE)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${SNPE_SO_FILES} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()

  if (onnxruntime_USE_DNNL)
    if(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND onnxruntime_DNNL_OPENCL_ROOT STREQUAL "")
      message(FATAL_ERROR "--dnnl_opencl_root required")
    elseif(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
      message(FATAL_ERROR "--dnnl_gpu_runtime required")
    elseif(onnxruntime_DNNL_GPU_RUNTIME STREQUAL "ocl" AND NOT (onnxruntime_DNNL_OPENCL_ROOT STREQUAL ""))
      #file(TO_CMAKE_PATH ${onnxruntime_DNNL_OPENCL_ROOT} onnxruntime_DNNL_OPENCL_ROOT)
      #set(DNNL_OCL_INCLUDE_DIR ${onnxruntime_DNNL_OPENCL_ROOT}/include)
      #set(DNNL_GPU_CMAKE_ARGS "-DDNNL_GPU_RUNTIME=OCL " "-DOPENCLROOT=${onnxruntime_DNNL_OPENCL_ROOT}")
      target_compile_definitions(onnxruntime_test_all PUBLIC -DDNNL_GPU_RUNTIME=OCL)
    endif()
    list(APPEND onnx_test_libs dnnl)
    add_custom_command(
      TARGET ${test_data_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${DNNL_DLL_PATH} $<TARGET_FILE_DIR:${test_data_target}>
      )
  endif()
  if(WIN32)
    set(wide_get_opt_src_dir ${TEST_SRC_DIR}/win_getopt/wide)
    onnxruntime_add_static_library(win_getopt_wide ${wide_get_opt_src_dir}/getopt.cc ${wide_get_opt_src_dir}/include/getopt.h)
    target_include_directories(win_getopt_wide INTERFACE ${wide_get_opt_src_dir}/include)
    set_target_properties(win_getopt_wide PROPERTIES FOLDER "ONNXRuntimeTest")
    set(onnx_test_runner_common_srcs ${onnx_test_runner_common_srcs})
    set(GETOPT_LIB_WIDE win_getopt_wide)
  endif()
endif()

# onnxruntime_provider_test
# Execution provider-related tests.
# These also have some support for dynamically specified plugin EPs.
if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD)
set(onnxruntime_provider_test_target onnxruntime_provider_test)
if (WIN32 AND TARGET onnxruntime_providers_cuda_ut)
  set(onnxruntime_provider_test_target onnxruntime_provider_test_executable)
endif()
block()
  set(supporting_test_srcs
    ${TEST_SRC_DIR}/common/cuda_op_test_utils.cc
    ${TEST_SRC_DIR}/common/cuda_op_test_utils.h
    ${TEST_SRC_DIR}/common/tensor_op_test_utils.cc
    ${TEST_SRC_DIR}/common/tensor_op_test_utils.h
  )

  if (onnxruntime_USE_DNNL)
    list(APPEND supporting_test_srcs
      ${TEST_SRC_DIR}/common/dnnl_op_test_utils.cc
      ${TEST_SRC_DIR}/common/dnnl_op_test_utils.h
    )
  endif()

  list(APPEND onnxruntime_provider_test_srcs
    ${supporting_test_srcs}
    ${onnxruntime_providers_cuda_plugin_ut_impl_objects}
    ${onnxruntime_unittest_main_src}
  )

  set(onnxruntime_provider_test_libs
    ${onnx_test_runner_common_lib}
    ${onnxruntime_test_providers_libs}
    ${onnxruntime_test_common_libs}
    onnx_test_data_proto
  )

  set(onnxruntime_provider_test_deps ${onnxruntime_test_providers_dependencies})
  if (onnxruntime_provider_test_target STREQUAL "onnxruntime_provider_test")
    list(APPEND onnxruntime_provider_test_deps ${onnxruntime_test_providers_runtime_dependencies})
  endif()

  AddTest(
    TARGET ${onnxruntime_provider_test_target}
    TEST_NAME onnxruntime_provider_test
    SOURCES ${onnxruntime_provider_test_srcs}
    LIBS ${onnxruntime_provider_test_libs}
    DEPENDS ${onnxruntime_provider_test_deps}
  )

  if (MSVC)
    set_property(SOURCE "${TEST_SRC_DIR}/providers/cpu/tensor/cast_op_test.cc"
                 APPEND PROPERTY COMPILE_OPTIONS "/bigobj")
  endif()

  # Match the non-plugin WebGPU test-source selection above.
  if (onnxruntime_USE_WEBGPU AND NOT onnxruntime_USE_EP_API_ADAPTERS)
    target_include_directories(${onnxruntime_provider_test_target} PRIVATE ${WGSL_GENERATED_ROOT})
  endif()

  if (onnxruntime_USE_OPENVINO)
    target_compile_definitions(${onnxruntime_provider_test_target} PRIVATE ONNXRUNTIME_OPENVINO_TEST)
    target_link_libraries(${onnxruntime_provider_test_target} PRIVATE openvino::runtime)
  endif()
  
  if (NOT onnxruntime_provider_test_target STREQUAL "onnxruntime_provider_test")
    # Keep the public build target responsible for both runtime artifacts without
    # making the executable depend on the module that imports its symbols.
    set_target_properties(${onnxruntime_provider_test_target} PROPERTIES OUTPUT_NAME onnxruntime_provider_test)
    add_custom_target(onnxruntime_provider_test ALL)
    add_dependencies(onnxruntime_provider_test
      ${onnxruntime_provider_test_target} ${onnxruntime_test_providers_runtime_dependencies})
    set_target_properties(onnxruntime_provider_test PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  onnxruntime_apply_test_target_workarounds(${onnxruntime_provider_test_target})
  onnxruntime_set_plugin_ep_test_environment(onnxruntime_provider_test)

  if (onnxruntime_USE_CUDA AND onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS AND
      NOT onnxruntime_DISABLE_CONTRIB_OPS)
    set(fp8_decode_filter "CUDA_EP_Unittest.QMoEFp8Gemv*")
    if (onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
      set(fp8_decode_outer_filter "${fp8_decode_filter}")
    else()
      set(fp8_decode_outer_filter "CUDA_EP_Unittest.All")
    endif()
    foreach(native_decode 0 1)
      add_test(NAME onnxruntime_fp8_native_decode_${native_decode}
        COMMAND $<TARGET_FILE:${onnxruntime_provider_test_target}>
          "--gtest_filter=${fp8_decode_outer_filter}"
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${onnxruntime_provider_test_target}>)
      set_tests_properties(onnxruntime_fp8_native_decode_${native_decode} PROPERTIES
        ENVIRONMENT "GTEST_FILTER=${fp8_decode_filter};ORT_QMOE_FP8_GEMV_NATIVE_DECODE=${native_decode}"
        RUN_SERIAL TRUE)
    endforeach()
  endif()

  if (onnxruntime_USE_CUDA AND onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS AND
      onnxruntime_USE_FP4_QMOE AND NOT onnxruntime_DISABLE_CONTRIB_OPS)
    set(nvfp4_scale_filter "CUDA_EP_Unittest.QMoENvfp4Gemv*")
    if (onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
      set(nvfp4_scale_outer_filter "${nvfp4_scale_filter}")
    else()
      set(nvfp4_scale_outer_filter "CUDA_EP_Unittest.All")
    endif()
    foreach(native_scale 0 1)
      add_test(NAME onnxruntime_nvfp4_native_scale_${native_scale}
        COMMAND $<TARGET_FILE:${onnxruntime_provider_test_target}>
          "--gtest_filter=${nvfp4_scale_outer_filter}"
        WORKING_DIRECTORY $<TARGET_FILE_DIR:${onnxruntime_provider_test_target}>)
      set_tests_properties(onnxruntime_nvfp4_native_scale_${native_scale} PROPERTIES
        ENVIRONMENT "GTEST_FILTER=${nvfp4_scale_filter};ORT_NVFP4_GEMV_NATIVE_SCALE=${native_scale}"
        RUN_SERIAL TRUE)
    endforeach()
  endif()

  if (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS AND
      NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_REDUCED_OPS_BUILD AND
      NOT IOS AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # Run legacy-safe regressions separately; concurrent-success tests require modern mode.
    string(JOIN ":" webgpu_legacy_test_filter
      "WebGpuPluginRecordingModeTest.*"
      "WebGpuPluginSharedAllocatorRegistrationTest.*"
      "WebGpuPluginSharedAllocatorTest.*"
      "WebGpuSessionAllocatorDeathTest.*"
      "PluginEpWebGpuConcurrency.DeferredProducerThen*"
      "PluginEpWebGpuConcurrency.CpuInputAndOutputRun"
      "PluginEpWebGpuConcurrency.MixedCpuAndGpuFeeds"
      "PluginEpWebGpuConcurrency.MixedCpuAndGpuFeedsReversed"
      "PluginEpWebGpuConcurrency.CpuFeedsWithCpuPartition"
      "PluginEpWebGpuConcurrency.CpuOutputBoundToGpu"
      "PluginEpWebGpuConcurrency.CpuOutputBoundToGpuFirst"
      "PluginEpWebGpuConcurrency.PreallocatedGpuInputWithMixedOutputBindings"
      "PluginEpWebGpuConcurrency.PreallocatedGpuInputWithMixedOutputBindingsReversed"
      "PluginEpWebGpuConcurrency.CpuOnlyGraphOutputBoundToGpu"
      "PluginEpWebGpuConcurrency.RepeatedKernelScratchBufferReuse"
      "PluginEpWebGpuConcurrency.CpuPartitionBetweenGpuKernels"
      "PluginEpWebGpuConcurrency.CpuInputAndGpuOutputRun"
      "PluginEpWebGpuConcurrency.CpuBindInputReusesDirtyGpuBuffer"
      "PluginEpWebGpuConcurrency.Serial*"
      "PluginEpWebGpuConcurrency.GraphCaptureReplayInterleavedWithIdleSessionCpuBindInput"
      "PluginEpWebGpuConcurrency.GpuInputAndCpuOutputRun"
      "PluginEpWebGpuConcurrency.SharedGpuCopyIsSubmittedBeforeSessionRun"
    )
    set(webgpu_legacy_test_args "--gtest_filter=${webgpu_legacy_test_filter}")
    if (onnxruntime_GENERATE_TEST_REPORTS)
      list(APPEND webgpu_legacy_test_args
        "--gtest_output=xml:onnxruntime_webgpu_legacy_test.$<CONFIG>.results.xml")
    endif()
    add_test(NAME onnxruntime_webgpu_legacy_test
      COMMAND ${onnxruntime_provider_test_target} ${webgpu_legacy_test_args}
      WORKING_DIRECTORY $<TARGET_FILE_DIR:${onnxruntime_provider_test_target}>
    )
    set_tests_properties(onnxruntime_webgpu_legacy_test PROPERTIES
      TIMEOUT 10800
      FAIL_REGULAR_EXPRESSION
        "0 tests from 0 test suites"
    )
    onnxruntime_set_plugin_ep_test_environment(onnxruntime_webgpu_legacy_test)
    set_property(TEST onnxruntime_webgpu_legacy_test APPEND PROPERTY
      ENVIRONMENT "ORT_WEBGPU_EP_FORCE_LEGACY=1")
  endif()

  # The CUDA EP internal unit tests (onnxruntime_providers_cuda_ut) are built as a shared-library
  # module that is dlopen'd at runtime by this binary (see CUDA_EP_Unittest.All -> TestAll()). Some
  # of those tests (e.g. the MatMulNBits two-level workspace end-to-end test) run a full
  # InferenceSession, whose symbols are statically linked into this executable. Mirror
  # onnxruntime_test_all and export them so the dlopen'd module can resolve them at load time;
  # without this the module fails to load with an undefined-symbol error.
  set_target_properties(${onnxruntime_provider_test_target} PROPERTIES ENABLE_EXPORTS 1)

  # Export only host symbols referenced by the module, including definitions in static libraries.
  # Exporting every provider-test symbol exceeds the Windows import-library limit.
  if (WIN32 AND TARGET onnxruntime_providers_cuda_ut)
    include("${CMAKE_CURRENT_LIST_DIR}/onnxruntime_test_exports.cmake")
    get_target_property(provider_test_link_libraries ${onnxruntime_provider_test_target} LINK_LIBRARIES)
    get_target_property(cuda_ut_link_libraries onnxruntime_providers_cuda_ut LINK_LIBRARIES)
    onnxruntime_export_test_symbols(${onnxruntime_provider_test_target}
      OBJECT_TARGET onnxruntime_providers_cuda_ut_objects
      HOST_LIBS ${provider_test_link_libraries}
      MODULE_LIBS onnxruntime_providers_cuda_obj ${cuda_ut_link_libraries})
    target_link_libraries(onnxruntime_providers_cuda_ut PRIVATE ${onnxruntime_provider_test_target})
  endif()

  if (onnxruntime_USE_CUDA AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
    target_compile_definitions(${onnxruntime_provider_test_target} PRIVATE
      ORT_UNIT_TEST_CUDA_PLUGIN_EP_LIBRARY_PATH="$<TARGET_FILE_NAME:onnxruntime_providers_cuda_plugin>"
      ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP=1)
  endif()

  if (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EP_API_ADAPTERS)
    target_compile_definitions(${onnxruntime_provider_test_target} PRIVATE
      ORT_UNIT_TEST_HAS_WEBGPU_PLUGIN_EP=1)
  endif()

  if (onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
    target_link_libraries(${onnxruntime_provider_test_target} PRIVATE
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
      CUDA::curand
      CUDA::cuda_driver
      CUDNN::cudnn
      cudnn_frontend)
  endif()

  # Expose QNN SDK headers to unit tests via an interface target
  if(onnxruntime_USE_QNN)
    add_library(qnn_sdk_headers_include INTERFACE)
    target_include_directories(qnn_sdk_headers_include INTERFACE
      ${onnxruntime_QNN_HOME}/include
      ${onnxruntime_QNN_HOME}/include/QNN)
    target_link_libraries(${onnxruntime_provider_test_target} PRIVATE qnn_sdk_headers_include)
  endif()

  # enable dynamic plugin EP usage
  target_compile_definitions(${onnxruntime_provider_test_target} PRIVATE ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP_USAGE)
  onnxruntime_set_webgpu_plugin_ep_test_definitions(${onnxruntime_provider_test_target})
  onnxruntime_apply_emscripten_test_link_settings(${onnxruntime_provider_test_target})

  if (IOS)
    add_custom_command(
      TARGET ${onnxruntime_provider_test_target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_directory
      ${TEST_DATA_SRC}
      $<TARGET_FILE_DIR:${onnxruntime_provider_test_target}>/testdata)
  endif()
endblock()
endif()

set(onnx_test_libs
  onnxruntime_test_utils
  ${ONNXRUNTIME_TEST_LIBS}
  onnx_test_data_proto
  ${onnxruntime_EXTERNAL_LIBRARIES})

if (NOT IOS)
    onnxruntime_add_executable(onnx_test_runner ${onnx_test_runner_src_dir}/main.cc)
    if(MSVC)
      target_compile_options(onnx_test_runner PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
        set_target_properties(onnx_test_runner PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
      else()
        set_target_properties(onnx_test_runner PROPERTIES LINK_FLAGS "-s NODERAWFS=1 -s ALLOW_MEMORY_GROWTH=1")
      endif()
    endif()

    target_link_libraries(onnx_test_runner PRIVATE onnx_test_runner_common ${GETOPT_LIB_WIDE} ${onnx_test_libs} nlohmann_json::nlohmann_json)
    target_include_directories(onnx_test_runner PRIVATE ${ONNXRUNTIME_ROOT})

    if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
      target_link_libraries(onnx_test_runner PRIVATE Python::Python)
    endif()
    set_target_properties(onnx_test_runner PROPERTIES FOLDER "ONNXRuntimeTest")

    install(TARGETS onnx_test_runner
            ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
            BUNDLE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if(onnxruntime_BUILD_BENCHMARKS)
    SET(BENCHMARK_DIR ${TEST_SRC_DIR}/onnx/microbenchmark)
    onnxruntime_add_executable(onnxruntime_benchmark
      ${BENCHMARK_DIR}/main.cc
      ${BENCHMARK_DIR}/modeltest.cc
      ${BENCHMARK_DIR}/pooling.cc
      ${BENCHMARK_DIR}/resize.cc
      ${BENCHMARK_DIR}/batchnorm.cc
      ${BENCHMARK_DIR}/batchnorm2.cc
      ${BENCHMARK_DIR}/tptest.cc
      ${BENCHMARK_DIR}/eigen.cc
      ${BENCHMARK_DIR}/copy.cc
      ${BENCHMARK_DIR}/gelu.cc
      ${BENCHMARK_DIR}/activation.cc
      ${BENCHMARK_DIR}/quantize.cc
      ${BENCHMARK_DIR}/reduceminmax.cc
      ${BENCHMARK_DIR}/layer_normalization.cc
      $<$<OR:$<BOOL:${onnxruntime_USE_WEBGPU}>,$<BOOL:${onnxruntime_USE_CUDA}>>:${BENCHMARK_DIR}/paged_attention.cc>)
    target_include_directories(onnxruntime_benchmark PRIVATE ${ONNXRUNTIME_ROOT} ${onnxruntime_graph_header} ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_compile_definitions(onnxruntime_benchmark PRIVATE BENCHMARK_STATIC_DEFINE)
    target_compile_definitions(onnxruntime_benchmark PRIVATE ${mlas_private_compile_definitions})
    if(WIN32)
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd4141>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd4141>")
      # Avoid using new and delete. But this is a benchmark program, it's ok if it has a chance to leak.
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26400>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26400>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26814>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26814>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26814>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26497>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                        "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
      target_compile_options(onnxruntime_benchmark PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    target_link_libraries(onnxruntime_benchmark PRIVATE onnx_test_runner_common benchmark::benchmark ${onnx_test_libs})
    add_dependencies(onnxruntime_benchmark ${onnxruntime_EXTERNAL_DEPENDENCIES})
    set_target_properties(onnxruntime_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")

    SET(MLAS_BENCH_DIR ${TEST_SRC_DIR}/mlas/bench)
    file(GLOB_RECURSE MLAS_BENCH_SOURCE_FILES "${MLAS_BENCH_DIR}/*.cpp" "${MLAS_BENCH_DIR}/*.h")
    list(FILTER MLAS_BENCH_SOURCE_FILES EXCLUDE REGEX "${MLAS_BENCH_DIR}/riscv64/.*")
    onnxruntime_add_executable(onnxruntime_mlas_benchmark ${MLAS_BENCH_SOURCE_FILES} ${ONNXRUNTIME_ROOT}/core/framework/error_code.cc)
    target_include_directories(onnxruntime_mlas_benchmark PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_link_libraries(onnxruntime_mlas_benchmark PRIVATE benchmark::benchmark onnxruntime_util ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_benchmark PRIVATE BENCHMARK_STATIC_DEFINE)
    target_compile_definitions(onnxruntime_mlas_benchmark PRIVATE ${mlas_private_compile_definitions})
    if(WIN32)
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE debug Dbghelp)
      # Avoid using new and delete. But this is a benchmark program, it's ok if it has a chance to leak.
      target_compile_options(onnxruntime_mlas_benchmark PRIVATE /wd26409)
      # "Global initializer calls a non-constexpr function." BENCHMARK_CAPTURE macro needs this.
      target_compile_options(onnxruntime_mlas_benchmark PRIVATE /wd26426)
    else()
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE  ${CMAKE_DL_LIBS})
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      target_link_libraries(onnxruntime_mlas_benchmark PRIVATE cpuinfo)
    endif()
    set_target_properties(onnxruntime_mlas_benchmark PROPERTIES FOLDER "ONNXRuntimeTest")

  endif()

  if(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
    set(MLAS_RISCV64_BENCH_DIR ${TEST_SRC_DIR}/mlas/bench/riscv64)

    onnxruntime_add_executable(
      onnxruntime_mlas_sgemm_riscv_bench
      ${MLAS_RISCV64_BENCH_DIR}/sgemm_riscv_bench.cpp)
    target_include_directories(onnxruntime_mlas_sgemm_riscv_bench PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_link_libraries(
      onnxruntime_mlas_sgemm_riscv_bench
      PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_sgemm_riscv_bench PRIVATE ${mlas_private_compile_definitions})
    set_target_properties(onnxruntime_mlas_sgemm_riscv_bench PROPERTIES FOLDER "ONNXRuntimeTest")

    onnxruntime_add_executable(
      onnxruntime_mlas_softmax_riscv_compare
      ${MLAS_RISCV64_BENCH_DIR}/softmax_rvv_compare.cpp)
    target_include_directories(
      onnxruntime_mlas_softmax_riscv_compare
      PRIVATE ${ONNXRUNTIME_ROOT} ${ONNXRUNTIME_ROOT}/core/mlas/inc)
    target_link_libraries(
      onnxruntime_mlas_softmax_riscv_compare
      PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_softmax_riscv_compare PRIVATE ${mlas_private_compile_definitions})
    set_target_properties(onnxruntime_mlas_softmax_riscv_compare PROPERTIES FOLDER "ONNXRuntimeTest")

    onnxruntime_add_executable(
      onnxruntime_mlas_halfgemm_rvv_bench
      ${MLAS_RISCV64_BENCH_DIR}/halfgemm_rvv_bench.cpp)
    target_include_directories(onnxruntime_mlas_halfgemm_rvv_bench PRIVATE
      ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
    target_link_libraries(
      onnxruntime_mlas_halfgemm_rvv_bench
      PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_halfgemm_rvv_bench PRIVATE ${mlas_private_compile_definitions})
    set_target_properties(onnxruntime_mlas_halfgemm_rvv_bench PROPERTIES FOLDER "ONNXRuntimeTest")

    onnxruntime_add_executable(
      onnxruntime_mlas_cast_rvv_bench
      ${MLAS_RISCV64_BENCH_DIR}/cast_rvv_bench.cpp)
    target_include_directories(onnxruntime_mlas_cast_rvv_bench PRIVATE
      ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
    target_link_libraries(
      onnxruntime_mlas_cast_rvv_bench
      PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_cast_rvv_bench PRIVATE ${mlas_private_compile_definitions})
    set_target_properties(onnxruntime_mlas_cast_rvv_bench PROPERTIES FOLDER "ONNXRuntimeTest")

    onnxruntime_add_executable(
      onnxruntime_mlas_rope_rvv_bench
      ${MLAS_RISCV64_BENCH_DIR}/rope_rvv_bench.cpp)
    target_include_directories(onnxruntime_mlas_rope_rvv_bench PRIVATE
      ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
    target_link_libraries(
      onnxruntime_mlas_rope_rvv_bench
      PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_rope_rvv_bench PRIVATE ${mlas_private_compile_definitions})
    set_target_properties(onnxruntime_mlas_rope_rvv_bench PROPERTIES FOLDER "ONNXRuntimeTest")

    onnxruntime_add_executable(
      onnxruntime_mlas_rmsnorm_rvv_bench
      ${MLAS_RISCV64_BENCH_DIR}/rmsnorm_rvv_bench.cpp)
    target_include_directories(onnxruntime_mlas_rmsnorm_rvv_bench PRIVATE
      ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
    target_link_libraries(
      onnxruntime_mlas_rmsnorm_rvv_bench
      PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common ${CMAKE_DL_LIBS})
    target_compile_definitions(onnxruntime_mlas_rmsnorm_rvv_bench PRIVATE ${mlas_private_compile_definitions})
    set_target_properties(onnxruntime_mlas_rmsnorm_rvv_bench PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  if(WIN32)
    target_compile_options(onnx_test_runner_common PRIVATE -D_CRT_SECURE_NO_WARNINGS)
  endif()

  if (NOT onnxruntime_REDUCED_OPS_BUILD AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    add_test(NAME onnx_test_pytorch_converted
      COMMAND onnx_test_runner ${onnx_SOURCE_DIR}/onnx/backend/test/data/pytorch-converted)
    add_test(NAME onnx_test_pytorch_operator
      COMMAND onnx_test_runner ${onnx_SOURCE_DIR}/onnx/backend/test/data/pytorch-operator)
  endif()

  # ---------------------------------------------------------------------------
  # Materialized ONNX node-test corpus.
  #
  # ONNX PR #7959 deletes the on-disk node-test corpus (onnx/backend/test/data/node) from the ONNX
  # source archive, leaving only the in-memory Python generators. To detach ORT from that on-disk
  # artifact, we regenerate the corpus from ONNX's own generators into the build tree and point the
  # C++ onnx_test_runner consumers (this ctest + QNN CI) at the materialized copy.
  #
  # Gated on onnxruntime_MATERIALIZE_ONNX_NODE_TESTS (opt-in; default OFF -- enabled explicitly on
  # the provisioned CI legs via --cmake_extra_defines). When ON, the gate
  # FAILS LOUDLY at configure time if Python/onnx/numpy are missing or mismatched -- it never
  # silently skips, because a silent skip would recreate the exact "green CI with zero node tests"
  # coverage gap this feature exists to close.
  # ---------------------------------------------------------------------------
  if (onnxruntime_MATERIALIZE_ONNX_NODE_TESTS AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # Derive the expected wheel version from the archive URL so this also works with installed ONNX packages.
    string(REGEX MATCH "v([0-9]+\\.[0-9]+\\.[0-9]+)" _onnx_url_version "${DEP_URL_onnx}")
    if(CMAKE_MATCH_1)
      set(_onnx_pinned_version ${CMAKE_MATCH_1})
    else()
      message(FATAL_ERROR "Could not parse the pinned ONNX version from DEP_URL_onnx='${DEP_URL_onnx}' "
        "(expected a vX.Y.Z tag). Fix cmake/deps.txt or this parser.")
    endif()

    # Python interpreter is not guaranteed for static test-only builds (the top-level
    # find_package(Python) in cmake/CMakeLists.txt is gated on BUILD_SHARED_LIB OR ENABLE_PYTHON).
    #
    # Guard on Python_VERSION, NOT Python_EXECUTABLE: Python_EXECUTABLE is a CACHE variable that
    # persists across reconfigures, whereas Python_VERSION is a normal variable re-derived only when
    # find_package actually runs. On any second `cmake` pass in an existing build dir, a cached
    # Python_EXECUTABLE would skip find_package and leave Python_VERSION EMPTY -- and the
    # numpy-pin-by-version selection below (`Python_VERSION VERSION_LESS 3.11`) treats empty as 0,
    # silently picking the < 3.11 pin regardless of the real interpreter. Keying the guard on
    # Python_VERSION forces find_package to repopulate the version before that selection every pass.
    if(NOT Python_VERSION)
      find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
    endif()

    # Configure-time fail-loud gate: onnx present and at the pinned version?
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import onnx,sys; sys.stdout.write(onnx.__version__)"
      RESULT_VARIABLE _onnx_rc OUTPUT_VARIABLE _onnx_ver ERROR_VARIABLE _onnx_err)
    if(NOT _onnx_rc EQUAL 0)
      message(FATAL_ERROR
        "onnxruntime_MATERIALIZE_ONNX_NODE_TESTS=ON but 'import onnx' failed for ${Python_EXECUTABLE}.\n"
        "  Fix: pip install onnx==${_onnx_pinned_version} numpy\n"
        "  OR reconfigure with -Donnxruntime_MATERIALIZE_ONNX_NODE_TESTS=OFF (node-test coverage will be dropped).\n"
        "  Details: ${_onnx_err}")
    endif()
    # The source and wheel are both final releases, so require exact version parity.
    if(NOT _onnx_ver STREQUAL _onnx_pinned_version)
      message(FATAL_ERROR
        "onnx ${_onnx_ver} != pinned ${_onnx_pinned_version} "
        "(cmake/deps.txt). A mismatched wheel bakes the wrong opset/IR into the materialized "
        "corpus (silent drift).\n"
        "  Fix: pip install onnx==${_onnx_pinned_version}")
    endif()

    # numpy version: assert the build's numpy equals the CI-pinned numpy (the SAME pin the QNN
    # legs install and that the ONNX corpus reproduction is validated under). Kept in sync with
    # tools/ci_build/github/linux/docker/scripts/requirements.txt (python-conditional):
    #   numpy==2.2.6 ; python_version <  "3.11"
    #   numpy==2.4.2 ; python_version >= "3.11"
    # The version ASSERT (installed == pin) and the equivalence ctest's <=4-ULP output band are
    # SEPARATE guards: the assert catches env drift (wrong/unexpected numpy) at CONFIGURE time with
    # a clear, early message; the ULP band separately absorbs the residual float drift between the
    # pinned numpy and the numpy the archived ONNX release used to recompute reference outputs.
    # Passing the pin (not the tautological installed value) is what makes the materializer's
    # --expected-numpy-version assert meaningful. Both import failure AND mismatch FAIL LOUD.
    if(Python_VERSION VERSION_LESS 3.11)
      set(_numpy_pinned_version 2.2.6)
    else()
      set(_numpy_pinned_version 2.4.2)
    endif()
    execute_process(
      COMMAND ${Python_EXECUTABLE} -c "import numpy,sys; sys.stdout.write(numpy.__version__)"
      RESULT_VARIABLE _np_rc OUTPUT_VARIABLE _np_ver ERROR_VARIABLE _np_err)
    if(NOT _np_rc EQUAL 0)
      message(FATAL_ERROR
        "onnxruntime_MATERIALIZE_ONNX_NODE_TESTS=ON but 'import numpy' failed for ${Python_EXECUTABLE}.\n"
        "  Fix: pip install numpy==${_numpy_pinned_version}\n"
        "  OR reconfigure with -Donnxruntime_MATERIALIZE_ONNX_NODE_TESTS=OFF.\n"
        "  Details: ${_np_err}")
    endif()
    if(NOT _np_ver STREQUAL _numpy_pinned_version)
      message(FATAL_ERROR
        "numpy ${_np_ver} != pinned ${_numpy_pinned_version} (requirements.txt, for Python "
        "${Python_VERSION}). The materialized corpus must be reproduced under the pinned numpy so "
        "the equivalence oracle stays within its <=4-ULP band.\n"
        "  Fix: pip install numpy==${_numpy_pinned_version}\n"
        "  OR reconfigure with -Donnxruntime_MATERIALIZE_ONNX_NODE_TESTS=OFF.")
    endif()

    set(_materialize_script ${REPO_ROOT}/tools/python/materialize_onnx_node_tests.py)
    set(_materialized_node_root ${CMAKE_BINARY_DIR}/onnx_node_tests)
    set(_materialized_node_dir  ${_materialized_node_root}/node)
    # Single shared floor VALUE for the "silently-empty/undersized materialization" guard. Wired to
    # three consumption sites; all use the same 1500 literal, but they DO NOT all count the same
    # population -- the floor is one stable value, not a claim that the counts are identical:
    #   * MIN_NODE_CASES in onnxruntime/test/python/onnx_node_test_equivalence_test.py
    #   * MIN_NODE_CASES in onnxruntime/test/python/onnx_backend_test_series.py -- the PRE-exclude
    #     population (counted in the base runner's _add_model_test, BEFORE backend_test.exclude()
    #     drops filtered cases), i.e. the full materialized case count.
    #   * onnx_test_runner -m (the C++ consumption-point tripwire, wired below) -- checked against
    #     tests.size() in main.cc, which is the POST-skip COLLECTED count: LoadTests (TestCase.cc)
    #     already `continue`s past broken/disabled/no-test-data/training-domain cases before they
    #     reach that tally, so the C++ population is the Python one MINUS the CPU skip set.
    # These two populations are INTENTIONALLY different; 1500 is a single conservative floor that
    # clears both today by a wide margin: ~1799 collected minus a handful of CPU skips (~5:
    # convinteger + 4 dft_*) is ~1794, still >> 1500 (~17% headroom on the smaller C++ count). 1500
    # is a STABLE floor, not a moving target -- it does not track the corpus size. If the CPU
    # broken/skip set ever grew past that margin, the C++ `-m` floor would red-fail LOUDLY (the safe
    # direction) -- never a silent pass. Passed to the build-time materialize step (build fails
    # loud), the node ctest (-m, consumption-point), AND the equivalence ctest.
    set(ORT_ONNX_NODE_MIN_CASES 1500)

    add_custom_command(
      OUTPUT  ${_materialized_node_root}/.stamp
      COMMAND ${CMAKE_COMMAND} -E make_directory ${_materialized_node_root}
      COMMAND ${Python_EXECUTABLE} ${_materialize_script}
                --out ${_materialized_node_root}
                --expected-onnx-version ${_onnx_pinned_version}
                --expected-numpy-version ${_numpy_pinned_version}
                --min-cases ${ORT_ONNX_NODE_MIN_CASES}
                --stamp ${_materialized_node_root}/.stamp
      DEPENDS ${_materialize_script} ${REPO_ROOT}/cmake/deps.txt
      COMMENT "Materializing ONNX node-test corpus -> ${_materialized_node_dir}"
      VERBATIM)
    add_custom_target(onnx_node_tests_materialized ALL
      DEPENDS ${_materialized_node_root}/.stamp)
    # Keep the corpus available for targeted onnx_test_runner builds as well as
    # normal ALL builds. The runner is the direct C++ consumer of this artifact.
    if(TARGET onnx_test_runner)
      add_dependencies(onnx_test_runner onnx_node_tests_materialized)
    endif()

    if (NOT onnxruntime_REDUCED_OPS_BUILD)
      # First-class ctest over the materialized node corpus (the durable replacement for the
      # on-disk corpus; previously node tests were only run via the Python series + QNN CI).
      # -m enforces the case-count floor at CONSUMPTION (the runner's tally), so a silently
      # empty/partial/truncated materialized tree fails loud instead of exiting green with
      # near-zero coverage (onnx_test_runner skips non-existent dirs). Single-sourced with the
      # build-time --min-cases via ORT_ONNX_NODE_MIN_CASES.
      add_test(NAME onnx_test_node_materialized
        COMMAND onnx_test_runner -m ${ORT_ONNX_NODE_MIN_CASES} ${_materialized_node_dir})
      # No ctest DEPENDS on onnx_node_tests_materialized: it is an add_custom_target(... ALL),
      # not a registered test, so a test-level DEPENDS on it is a silent no-op. The real ordering
      # guarantee is (a) the ALL target materializes the corpus during the normal build, before
      # any ctest runs, and (b) the -m case-count tripwire above fails LOUD if the corpus is
      # missing/partial at consumption time.

      # Equivalence oracle: while pinned BELOW #7959 the on-disk corpus still ships in the archive,
      # so we byte-compare the materialized copy against it. Gated on if(EXISTS ...) at CONFIGURE
      # time, so the test simply stops being registered (clean auto-retirement, no perpetual skip,
      # no red CI) once the pin advances past #7959 and the oracle disappears.
      set(_onnx_disk_node ${onnx_SOURCE_DIR}/onnx/backend/test/data/node)
      if(EXISTS ${_onnx_disk_node})
        add_test(NAME onnx_node_tests_equivalence
          COMMAND ${Python_EXECUTABLE} ${REPO_ROOT}/tools/python/compare_node_test_corpora.py
                  --oracle       ${_onnx_disk_node}
                  --materialized ${_materialized_node_dir}
                  --expected-onnx-version  ${_onnx_pinned_version}
                  --expected-numpy-version ${_numpy_pinned_version}
                  --min-cases ${ORT_ONNX_NODE_MIN_CASES})
        # No test-level DEPENDS on onnx_node_tests_materialized here either: it is a custom
        # target (ALL), not a test, so ctest DEPENDS on it is a silent no-op. The ALL target
        # materializes the corpus during the build (before ctest); the comparator's own
        # --min-cases floor fails loud on a missing/partial corpus.
      else()
        message(STATUS "ONNX on-disk node corpus absent (post-#7959 pin) -- equivalence oracle "
          "retired; skipping onnx_node_tests_equivalence.")
      endif()
    endif()
  endif()

  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND android_shared_libs log android)
  endif()
endif()


if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  if(NOT IOS)
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

    # ABSL_FLAGS_STRIP_NAMES is set to 1 by default to disable flag registration when building for Android, iPhone, and "embedded devices".
    # See the issue: https://github.com/abseil/abseil-cpp/issues/1875
    # We set it to 0 for all builds to be able to use ABSL flags for onnxruntime_perf_test.
    target_compile_definitions(onnxruntime_perf_test PRIVATE ABSL_FLAGS_STRIP_NAMES=0)

    if(MSVC)
      target_compile_options(onnxruntime_perf_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
            "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
    endif()
    target_include_directories(onnxruntime_perf_test PRIVATE   ${onnx_test_runner_src_dir} ${ONNXRUNTIME_ROOT}
          ${onnxruntime_graph_header} ${onnxruntime_exec_src_dir}
          ${CMAKE_CURRENT_BINARY_DIR})

    if (WIN32)
      target_compile_options(onnxruntime_perf_test PRIVATE ${disabled_warnings})
      if (NOT DEFINED SYS_PATH_LIB)
        set(SYS_PATH_LIB shlwapi)
      endif()
    endif()

    if (onnxruntime_BUILD_SHARED_LIB)
      #It will dynamically link to onnxruntime. So please don't add onxruntime_graph/onxruntime_framework/... here.
      #onnxruntime_common is kind of ok because it is thin, tiny and totally stateless.
      set(onnxruntime_perf_test_libs
            onnx_test_runner_common onnxruntime_test_utils onnxruntime_common
            onnxruntime onnxruntime_flatbuffers onnx_test_data_proto
            ${onnxruntime_EXTERNAL_LIBRARIES}
            absl::flags absl::flags_parse ${SYS_PATH_LIB} ${CMAKE_DL_LIBS})
      if(NOT WIN32)
        if(onnxruntime_USE_SNPE)
          list(APPEND onnxruntime_perf_test_libs onnxruntime_providers_snpe)
        endif()
      endif()
      if (CMAKE_SYSTEM_NAME STREQUAL "Android")
        list(APPEND onnxruntime_perf_test_libs ${android_shared_libs})
      endif()
      if (CMAKE_SYSTEM_NAME MATCHES "AIX")
        list(APPEND onnxruntime_perf_test_libs onnxruntime_graph onnxruntime_session ${onnxruntime_providers_target} onnxruntime_framework onnxruntime_util onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 gtest absl_failure_signal_handler absl_examine_stack absl_flags_parse  absl_flags_usage absl_flags_usage_internal)
      endif()
      target_link_libraries(onnxruntime_perf_test PRIVATE ${onnxruntime_perf_test_libs} Threads::Threads)
      if (onnxruntime_USE_CUDA OR onnxruntime_USE_NV OR onnxruntime_USE_TENSORRT)
        target_link_libraries(onnxruntime_perf_test PRIVATE CUDA::cudart)
      endif()
      if(WIN32)
        target_link_libraries(onnxruntime_perf_test PRIVATE debug dbghelp advapi32)
      endif()
    else()
      target_link_libraries(onnxruntime_perf_test PRIVATE onnx_test_runner_common absl::flags absl::flags_parse ${onnx_test_libs})
      #  When onnxruntime_BUILD_SHARED_LIB is OFF (the plugin build path), perf test was missing CUDA include directories and CUDA::cudart linkage.
      if (onnxruntime_USE_CUDA OR onnxruntime_USE_NV OR onnxruntime_USE_TENSORRT)
        target_include_directories(onnxruntime_perf_test PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_link_libraries(onnxruntime_perf_test PRIVATE CUDA::cudart)
      endif()
    endif()
    set_target_properties(onnxruntime_perf_test PROPERTIES FOLDER "ONNXRuntimeTest")

endif()

  # Build ep_weight_sharing_ctx_gen for all supported EPs (QNN, TensorRT, OpenVINO, VitisAI)
  if(onnxruntime_USE_QNN OR onnxruntime_USE_TENSORRT OR onnxruntime_USE_OPENVINO OR onnxruntime_USE_VITISAI)
    #qnn ctx generator
    set(ep_weight_sharing_ctx_gen_src_dir ${TEST_SRC_DIR}/ep_weight_sharing_ctx_gen)
    set(ep_weight_sharing_ctx_gen_src_patterns
    "${ep_weight_sharing_ctx_gen_src_dir}/*.cc"
    "${ep_weight_sharing_ctx_gen_src_dir}/*.h")

    file(GLOB ep_weight_sharing_ctx_gen_src CONFIGURE_DEPENDS
      ${ep_weight_sharing_ctx_gen_src_patterns}
      )
    onnxruntime_add_executable(ep_weight_sharing_ctx_gen ${ep_weight_sharing_ctx_gen_src})
    target_include_directories(ep_weight_sharing_ctx_gen PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR})
    if (WIN32)
      target_compile_options(ep_weight_sharing_ctx_gen PRIVATE ${disabled_warnings})
      if (NOT DEFINED SYS_PATH_LIB)
        set(SYS_PATH_LIB shlwapi)
      endif()
    endif()

    if (onnxruntime_BUILD_SHARED_LIB)
      set(ep_weight_sharing_ctx_gen_libs onnxruntime_common onnxruntime ${onnxruntime_EXTERNAL_LIBRARIES} ${GETOPT_LIB_WIDE})
      target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE ${ep_weight_sharing_ctx_gen_libs})
      if (WIN32)
        target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE debug dbghelp advapi32)
      endif()
    else()
      target_link_libraries(ep_weight_sharing_ctx_gen PRIVATE onnxruntime_session ${onnxruntime_test_providers_libs} ${onnxruntime_EXTERNAL_LIBRARIES} ${GETOPT_LIB_WIDE})
    endif()

    set_target_properties(ep_weight_sharing_ctx_gen PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()

  if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(onnxruntime_cpuinfo_test_library cpuinfo)
    get_target_property(onnxruntime_cpuinfo_aliased_target cpuinfo ALIASED_TARGET)
    if(onnxruntime_cpuinfo_aliased_target)
      set(onnxruntime_cpuinfo_test_library ${onnxruntime_cpuinfo_aliased_target})
    endif()
    get_target_property(onnxruntime_cpuinfo_library_type ${onnxruntime_cpuinfo_test_library} TYPE)
    if(onnxruntime_cpuinfo_library_type STREQUAL "STATIC_LIBRARY")
      onnxruntime_add_executable(
        onnxruntime_cpuinfo_refcount_test
        ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cpuinfo_refcount_test.cc)
      target_compile_definitions(
        onnxruntime_cpuinfo_refcount_test
        PRIVATE ORT_CPUINFO_TEST_HAS_INTERNAL_STATE)
      if(onnxruntime_USE_XNNPACK)
        target_compile_definitions(
          onnxruntime_cpuinfo_refcount_test
          PRIVATE ORT_CPUINFO_TEST_USE_XNNPACK)
        if(onnxruntime_USE_VCPKG)
          target_include_directories(onnxruntime_cpuinfo_refcount_test PRIVATE ${XNNPACK_HDR})
        else()
          target_include_directories(onnxruntime_cpuinfo_refcount_test PRIVATE ${XNNPACK_INCLUDE_DIR})
        endif()
        target_link_libraries(
          onnxruntime_cpuinfo_refcount_test
          PRIVATE ${onnxruntime_EXTERNAL_LIBRARIES_XNNPACK})
      endif()
      target_link_libraries(onnxruntime_cpuinfo_refcount_test PRIVATE cpuinfo Threads::Threads)
      add_test(
        NAME onnxruntime_cpuinfo_refcount_test
        COMMAND onnxruntime_cpuinfo_refcount_test)
      set_target_properties(
        onnxruntime_cpuinfo_refcount_test
        PROPERTIES FOLDER "ONNXRuntimeTest")
    endif()
  endif()

  # shared lib
  if (onnxruntime_BUILD_SHARED_LIB)
    if(WIN32)
      onnxruntime_add_executable(onnxruntime_shared_lib_dlopen_test ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/dlopen_main.cc)
      add_dependencies(onnxruntime_shared_lib_dlopen_test ${all_dependencies} onnxruntime)
      add_test(NAME onnxruntime_shared_lib_dlopen_test COMMAND onnxruntime_shared_lib_dlopen_test WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_shared_lib_dlopen_test>)
      set_target_properties(onnxruntime_shared_lib_dlopen_test PROPERTIES FOLDER "ONNXRuntimeTest")

      if(onnxruntime_cpuinfo_library_type STREQUAL "STATIC_LIBRARY")
        onnxruntime_add_shared_library(
          onnxruntime_cpuinfo_dlopen_test_library
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cpuinfo_dlopen_test_library.cc
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cpuinfo_dlopen_test_library.def)
        target_include_directories(
          onnxruntime_cpuinfo_dlopen_test_library
          PRIVATE ${ONNXRUNTIME_ROOT})
        target_link_libraries(
          onnxruntime_cpuinfo_dlopen_test_library
          PRIVATE onnxruntime_common cpuinfo)

        onnxruntime_add_executable(
          onnxruntime_shared_lib_cpuinfo_dlopen_test
          ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cpuinfo_dlopen_test.cc)
        add_dependencies(
          onnxruntime_shared_lib_cpuinfo_dlopen_test
          onnxruntime_cpuinfo_dlopen_test_library)
        target_compile_definitions(
          onnxruntime_shared_lib_cpuinfo_dlopen_test
          PRIVATE
          ORT_CPUINFO_DLOPEN_TEST_LIBRARY=L"$<TARGET_FILE_NAME:onnxruntime_cpuinfo_dlopen_test_library>")
        add_test(
          NAME onnxruntime_shared_lib_cpuinfo_dlopen_test
          COMMAND onnxruntime_shared_lib_cpuinfo_dlopen_test
          WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_cpuinfo_dlopen_test_library>)
        set_target_properties(
          onnxruntime_cpuinfo_dlopen_test_library
          onnxruntime_shared_lib_cpuinfo_dlopen_test
          PROPERTIES FOLDER "ONNXRuntimeTest")
      endif()

      if (MSVC)
        # set VS debugger working directory to the test program's directory
        set_target_properties(onnxruntime_shared_lib_dlopen_test PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_shared_lib_dlopen_test>)
      endif()
    endif()
    onnxruntime_add_static_library(onnxruntime_mocked_allocator ${TEST_SRC_DIR}/util/test_allocator.cc)
    target_include_directories(onnxruntime_mocked_allocator PUBLIC ${TEST_SRC_DIR}/util/include)
    target_link_libraries(onnxruntime_mocked_allocator PRIVATE ${GSL_TARGET})
    set_target_properties(onnxruntime_mocked_allocator PROPERTIES FOLDER "ONNXRuntimeTest")

    # onnxruntime_runtime_path_test_shared_library
    block()
      set(onnxruntime_runtime_path_test_shared_library_src
          "${TEST_SRC_DIR}/shared_lib/runtime_path_test_shared_library/runtime_path_test_shared_library.h"
          "${TEST_SRC_DIR}/shared_lib/runtime_path_test_shared_library/runtime_path_test_shared_library.cc")

      onnxruntime_add_shared_library(onnxruntime_runtime_path_test_shared_library
                                     ${onnxruntime_runtime_path_test_shared_library_src})

      if (CMAKE_SYSTEM_NAME MATCHES "AIX")
        target_link_libraries(onnxruntime_runtime_path_test_shared_library PRIVATE
                              onnxruntime_common ${CMAKE_DL_LIBS})
        set_target_properties(onnxruntime_runtime_path_test_shared_library PROPERTIES AIX_SHARED_LIBRARY_ARCHIVE OFF)
      else()
        if (CPUINFO_SUPPORTED)
          target_link_libraries(onnxruntime_runtime_path_test_shared_library PRIVATE
                                onnxruntime_common cpuinfo ${CMAKE_DL_LIBS})
        else()
          target_link_libraries(onnxruntime_runtime_path_test_shared_library PRIVATE
                                onnxruntime_common ${CMAKE_DL_LIBS})
        endif()
      endif()
      target_include_directories(onnxruntime_runtime_path_test_shared_library PRIVATE ${ONNXRUNTIME_ROOT})

      if(UNIX)
        if (APPLE)
          set(onnxruntime_runtime_path_test_shared_library_link_flags "-Xlinker -dead_strip")
        elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
          string(CONCAT onnxruntime_runtime_path_test_shared_library_link_flags
                 "-Xlinker --version-script=${TEST_SRC_DIR}/shared_lib/runtime_path_test_shared_library/runtime_path_test_shared_library.lds "
                 "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
        endif()
      else()
        set(onnxruntime_runtime_path_test_shared_library_link_flags
            "-DEF:${TEST_SRC_DIR}/shared_lib/runtime_path_test_shared_library/runtime_path_test_shared_library.def")
      endif()

      set_property(TARGET onnxruntime_runtime_path_test_shared_library APPEND_STRING PROPERTY LINK_FLAGS
                   ${onnxruntime_runtime_path_test_shared_library_link_flags})

      set_target_properties(onnxruntime_runtime_path_test_shared_library PROPERTIES FOLDER "ONNXRuntimeTest")
      source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_runtime_path_test_shared_library_src})
    endblock()

    #################################################################
    # test inference using shared lib
    set(onnxruntime_shared_lib_test_LIBS
        onnxruntime_mocked_allocator
        onnxruntime_test_utils
        onnxruntime_common
        onnx_proto
        onnxruntime_runtime_path_test_shared_library)

    if(NOT WIN32)
      if(onnxruntime_USE_SNPE)
        list(APPEND onnxruntime_shared_lib_test_LIBS onnxruntime_providers_snpe)
      endif()
    endif()
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      list(APPEND onnxruntime_shared_lib_test_LIBS cpuinfo)
    endif()
    if (onnxruntime_USE_CUDA)
      list(APPEND onnxruntime_shared_lib_test_LIBS CUDA::cudart)
    endif()

    if (onnxruntime_USE_TENSORRT)
      list(APPEND onnxruntime_shared_lib_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (onnxruntime_USE_NV)
      list(APPEND onnxruntime_shared_lib_test_LIBS ${TENSORRT_LIBRARY_INFER} CUDA::cudart)
    endif()
    if (onnxruntime_USE_DML)
      list(APPEND onnxruntime_shared_lib_test_LIBS d3d12.lib)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      list(APPEND onnxruntime_shared_lib_test_LIBS ${android_shared_libs})
    endif()

    if (CMAKE_SYSTEM_NAME MATCHES "AIX")
      list(APPEND onnxruntime_shared_lib_test_LIBS onnxruntime_graph onnxruntime_session ${onnxruntime_providers_target} onnxruntime_framework onnxruntime_util onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 onnx)
    endif()

    AddTest(DYN
            TARGET onnxruntime_shared_lib_test
            SOURCES ${onnxruntime_shared_lib_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_shared_lib_test_LIBS}
            DEPENDS ${all_dependencies}
    )

    target_include_directories(onnxruntime_shared_lib_test PRIVATE ${ONNXRUNTIME_ROOT})

    if (onnxruntime_USE_CUDA)
      target_include_directories(onnxruntime_shared_lib_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_sources(onnxruntime_shared_lib_test PRIVATE ${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu)
    endif()
    if (onnxruntime_USE_NV)
      target_include_directories(onnxruntime_shared_lib_test PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    endif()


    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_sources(onnxruntime_shared_lib_test PRIVATE
        "${ONNXRUNTIME_ROOT}/core/platform/android/cxa_demangle.cc"
        "${TEST_SRC_DIR}/platform/android/cxa_demangle_test.cc"
      )
      target_compile_definitions(onnxruntime_shared_lib_test PRIVATE USE_DUMMY_EXA_DEMANGLE=1)
    endif()

    if (CMAKE_SYSTEM_NAME MATCHES "AIX" AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
      set_target_properties(onnxruntime_shared_lib_test PROPERTIES ENABLE_EXPORTS 1)
    endif()

    if (IOS)
      add_custom_command(
        TARGET onnxruntime_shared_lib_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_SRC}
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
  endif()

  # the debug node IO functionality uses static variables, so it is best tested
  # in its own process
  if(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
    AddTest(
      TARGET onnxruntime_test_debug_node_inputs_outputs
      SOURCES
        "${TEST_SRC_DIR}/debug_node_inputs_outputs/debug_node_inputs_outputs_utils_test.cc"
        "${TEST_SRC_DIR}/providers/provider_test_utils.h"
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

  if(NOT onnxruntime_target_platform STREQUAL "ARM64EC")
    file(GLOB onnxruntime_mlas_test_src CONFIGURE_DEPENDS
      "${TEST_SRC_DIR}/mlas/unittest/*.h"
      "${TEST_SRC_DIR}/mlas/unittest/*.cpp"
    )
    onnxruntime_add_executable(onnxruntime_mlas_test ${onnxruntime_mlas_test_src})
    if(MSVC)
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>"
              "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6326>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd6326>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26426>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26426>")
      target_compile_options(onnxruntime_mlas_test PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /bigobj>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/bigobj>")
    endif()
    if(IOS)
      set_target_properties(onnxruntime_mlas_test PROPERTIES
        XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED "NO"
      )
    endif()
    target_compile_definitions(onnxruntime_mlas_test PRIVATE ${mlas_private_compile_definitions})
    target_include_directories(onnxruntime_mlas_test PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}
            ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(onnxruntime_mlas_test PRIVATE GTest::gtest GTest::gmock ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common)
    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      target_link_libraries(onnxruntime_mlas_test PRIVATE cpuinfo)
    endif()
    if(NOT WIN32)
      target_link_libraries(onnxruntime_mlas_test PRIVATE  ${CMAKE_DL_LIBS})
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Android")
      target_link_libraries(onnxruntime_mlas_test PRIVATE ${android_shared_libs})
    endif()
    if(WIN32)
      target_link_libraries(onnxruntime_mlas_test PRIVATE debug Dbghelp Advapi32)
    endif()
    if (onnxruntime_LINK_LIBATOMIC)
      target_link_libraries(onnxruntime_mlas_test PRIVATE atomic)
    endif()
    target_link_libraries(onnxruntime_mlas_test PRIVATE Threads::Threads)
    set_target_properties(onnxruntime_mlas_test PROPERTIES FOLDER "ONNXRuntimeTest")
    if (onnxruntime_RUN_MLAS_TESTS)
      # The full suite is too slow for per-PR ARM64 CI, so focus on architecture-specific activation and FP16 paths.
      set(onnxruntime_mlas_test_args "--gtest_filter=*FP16*:*Fp16*:Exp.*:Softmax*:Activation*")
      if (onnxruntime_GENERATE_TEST_REPORTS)
        list(APPEND onnxruntime_mlas_test_args
          "--gtest_output=xml:$<SHELL_PATH:$<TARGET_FILE:onnxruntime_mlas_test>.$<CONFIG>.results.xml>")
      endif()
      add_test(NAME onnxruntime_mlas_test
        COMMAND onnxruntime_mlas_test ${onnxruntime_mlas_test_args}
        WORKING_DIRECTORY $<TARGET_FILE_DIR:onnxruntime_mlas_test>)
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
        set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
      else()
        set_target_properties(onnxruntime_mlas_test PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
      endif()
    endif()
endif()
  # Training API Tests
  # Disabling training_api_test_trainer. CXXOPT generates a ton of warnings because of which nuget pipeline is failing.
  # TODO(askhade): Fix the warnings.
  # This has no impact on the release as the release package and the pipeline, both do not use this.
  # This is used by devs for testing training apis.
  #if (onnxruntime_ENABLE_TRAINING_APIS)
  if (0)
    # Only files in the trainer and common folder will be compiled into test trainer.
    file(GLOB training_api_test_trainer_src
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/common/*.h"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/trainer/*.cc"
      "${ORTTRAINING_SOURCE_DIR}/test/training_api/trainer/*.h"
    )
    onnxruntime_add_executable(onnxruntime_test_trainer ${training_api_test_trainer_src})

    onnxruntime_add_include_to_target(onnxruntime_test_trainer onnxruntime_session
      onnxruntime_framework onnxruntime_common onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)

    set(CXXOPTS ${cxxopts_SOURCE_DIR}/include)
    target_include_directories(onnxruntime_test_trainer PRIVATE
      ${CMAKE_CURRENT_BINARY_DIR}
      ${ONNXRUNTIME_ROOT}
      ${ORTTRAINING_ROOT}
      ${CXXOPTS}
      ${extra_includes}
      ${onnxruntime_graph_header}
      ${onnxruntime_exec_src_dir}
    )

    set(ONNXRUNTIME_TEST_LIBS
      onnxruntime_session
      ${onnxruntime_libs}
      # CUDA is dynamically loaded at runtime
      onnxruntime_optimizer
      ${onnxruntime_providers_target}
      onnxruntime_util
      onnxruntime_lora
      onnxruntime_framework
      onnxruntime_util
      onnxruntime_graph
      ${ONNXRUNTIME_MLAS_LIBS}
      onnxruntime_common
      onnxruntime_flatbuffers
    )

    target_link_libraries(onnxruntime_test_trainer PRIVATE
      ${ONNXRUNTIME_TEST_LIBS}
      ${onnxruntime_EXTERNAL_LIBRARIES}
    )
    set_target_properties(onnxruntime_test_trainer PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_CUDA_MINIMAL)

  set(custom_op_src_patterns
    "${TEST_SRC_DIR}/testdata/custom_op_library/*.h"
    "${TEST_SRC_DIR}/testdata/custom_op_library/*.cc"
    "${TEST_SRC_DIR}/testdata/custom_op_library/cpu/cpu_ops.*"
  )

  set(custom_op_lib_include ${REPO_ROOT}/include)
  set(custom_op_lib_option)
  set(custom_op_lib_link ${GSL_TARGET})

  if (onnxruntime_USE_CUDA)
    list(APPEND custom_op_src_patterns
        "${ONNXRUNTIME_SHARED_LIB_TEST_SRC_DIR}/cuda_ops.cu"
        "${TEST_SRC_DIR}/testdata/custom_op_library/cuda/cuda_ops.*")
    list(APPEND custom_op_lib_include ${CUDAToolkit_INCLUDE_DIRS})
    if(NOT onnxruntime_CUDA_MINIMAL)
      list(APPEND custom_op_lib_include ${CUDNN_INCLUDE_DIR})
    endif()
    if (HAS_QSPECTRE)
      list(APPEND custom_op_lib_option "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /Qspectre>")
    endif()
    set(custom_op_lib_link ${custom_op_lib_link} CUDA::cudart)
  endif()

  file(GLOB custom_op_src ${custom_op_src_patterns})
  onnxruntime_add_shared_library(custom_op_library ${custom_op_src})
  target_compile_options(custom_op_library PRIVATE ${custom_op_lib_option})
  target_include_directories(custom_op_library PRIVATE ${REPO_ROOT}/include ${custom_op_lib_include})
  target_link_libraries(custom_op_library PRIVATE ${GSL_TARGET} ${custom_op_lib_link})

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.lds -Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG "-DEF:${TEST_SRC_DIR}/testdata/custom_op_library/custom_op_library.def")
    if (NOT onnxruntime_USE_CUDA)
      target_compile_options(custom_op_library PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd26409>"
                    "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/wd26409>")
    endif()
  endif()
  set_property(TARGET custom_op_library APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_CUSTOM_OP_LIB_LINK_FLAG})
  #ON AIX, to call dlopen on custom_op_library, we need to generate this library as shared object .so file.
  #Latest cmake behavior is changed and  cmake will remove shared object .so file after generating the shared archive.
  #To prevent that, making AIX_SHARED_LIBRARY_ARCHIVE as OFF for custom_op_library.
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    set_target_properties(custom_op_library PROPERTIES AIX_SHARED_LIBRARY_ARCHIVE OFF)
  endif()

  if (NOT onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    if (onnxruntime_BUILD_JAVA AND NOT onnxruntime_ENABLE_STATIC_ANALYSIS)
      block()
        message(STATUS "Enabling Java tests")

        # native-test is added to resources so custom_op_lib can be loaded
        # and we want to copy it there
        set(JAVA_NATIVE_TEST_DIR ${JAVA_OUTPUT_DIR}/native-test)
        file(MAKE_DIRECTORY ${JAVA_NATIVE_TEST_DIR})

        set(CUSTOM_OP_LIBRARY_DST_FILE_NAME
            $<IF:$<BOOL:${WIN32}>,$<TARGET_FILE_NAME:custom_op_library>,$<TARGET_LINKER_FILE_NAME:custom_op_library>>)

        add_custom_command(TARGET custom_op_library POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:custom_op_library>
                ${JAVA_NATIVE_TEST_DIR}/${CUSTOM_OP_LIBRARY_DST_FILE_NAME})

        # also copy other library dependencies that may be required by tests to native-test
        if(onnxruntime_USE_QNN)
          add_custom_command(TARGET onnxruntime_providers_qnn POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy ${QNN_LIB_FILES} ${JAVA_NATIVE_TEST_DIR})
        endif()
	if (WIN32)
          set(EXAMPLE_PLUGIN_EP_DST_FILE_NAME $<IF:$<BOOL:${WIN32}>,$<TARGET_FILE_NAME:example_plugin_ep>,$<TARGET_LINKER_FILE_NAME:example_plugin_ep>>)
          add_custom_command(TARGET custom_op_library POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:example_plugin_ep> ${JAVA_NATIVE_TEST_DIR}/${EXAMPLE_PLUGIN_EP_DST_FILE_NAME})
	endif()

        # delegate to gradle's test runner

        # On Windows, ctest requires a test to be an .exe(.com) file. With gradle wrapper, we get gradlew.bat.
        # To work around this, we delegate gradle execution to a separate .cmake file that can be run with cmake.
        # For simplicity, we use this setup for all supported platforms and not just Windows.

        # Note: Here we rely on the values in ORT_PROVIDER_FLAGS to be of the format "-Doption=value".
        # This happens to also match the gradle command line option for specifying system properties.
        set(GRADLE_SYSTEM_PROPERTY_DEFINITIONS ${ORT_PROVIDER_FLAGS})

        if(onnxruntime_ENABLE_TRAINING_APIS)
          message(STATUS "Enabling Java tests for training APIs")

          list(APPEND GRADLE_SYSTEM_PROPERTY_DEFINITIONS "-DENABLE_TRAINING_APIS=1")
        endif()

        add_test(NAME onnxruntime4j_test COMMAND
            ${CMAKE_COMMAND}
                -DGRADLE_EXECUTABLE=${GRADLE_EXECUTABLE}
                -DBIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
                -DREPO_ROOT=${REPO_ROOT}
                # Note: Quotes are important here to pass a list of values as a single property.
                "-DGRADLE_SYSTEM_PROPERTY_DEFINITIONS=${GRADLE_SYSTEM_PROPERTY_DEFINITIONS}"
                -P ${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_java_unittests.cmake)

        set_property(TEST onnxruntime4j_test APPEND PROPERTY DEPENDS onnxruntime4j_jni)
      endblock()
    endif()
  endif()

  if (onnxruntime_BUILD_SHARED_LIB AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
    set (onnxruntime_customopregistration_test_SRC
            ${ONNXRUNTIME_CUSTOM_OP_REGISTRATION_TEST_SRC_DIR}/test_registercustomops.cc)

    set(onnxruntime_customopregistration_test_LIBS custom_op_library onnxruntime_common onnxruntime_test_utils)

    if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      list(APPEND onnxruntime_customopregistration_test_LIBS cpuinfo)
    endif()
    if (onnxruntime_USE_TENSORRT)
      list(APPEND onnxruntime_customopregistration_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (onnxruntime_USE_NV)
      list(APPEND onnxruntime_customopregistration_test_LIBS ${TENSORRT_LIBRARY_INFER})
    endif()
    if (CMAKE_SYSTEM_NAME MATCHES "AIX")
      list(APPEND onnxruntime_customopregistration_test_LIBS onnxruntime_graph onnxruntime_session ${onnxruntime_providers_target} onnxruntime_lora onnxruntime_framework onnxruntime_util onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 ${PROTOBUF_LIB} onnx onnx_proto)
    endif()
    AddTest(DYN
            TARGET onnxruntime_customopregistration_test
            SOURCES ${onnxruntime_customopregistration_test_SRC} ${onnxruntime_unittest_main_src}
            LIBS ${onnxruntime_customopregistration_test_LIBS}
            DEPENDS ${all_dependencies}
    )

    if (IOS)
      add_custom_command(
        TARGET onnxruntime_customopregistration_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${TEST_DATA_SRC}
        $<TARGET_FILE_DIR:onnxruntime_customopregistration_test>/testdata)
    endif()

  endif()
endif()

# Build custom op library that returns an error OrtStatus when the exported RegisterCustomOps function is called.
if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
  onnxruntime_add_shared_library_module(custom_op_invalid_library
                                        ${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.h
                                        ${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.cc)
  target_include_directories(custom_op_invalid_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_invalid_library/custom_op_library.def")
  endif()

  set_property(TARGET custom_op_invalid_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_INVALID_LIB_LINK_FLAG})
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))

  file(GLOB_RECURSE custom_op_get_const_input_test_library_src
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op.h"
        "${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op.cc"
  )

  onnxruntime_add_shared_library_module(custom_op_get_const_input_test_library ${custom_op_get_const_input_test_library_src})

  onnxruntime_add_include_to_target(custom_op_get_const_input_test_library onnxruntime_common GTest::gtest GTest::gmock)
  target_include_directories(custom_op_get_const_input_test_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session
                                                                            ${REPO_ROOT}/include/onnxruntime/core/common)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_get_const_input_test_library/custom_op_lib.def")
  endif()

  set_property(TARGET custom_op_get_const_input_test_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_GET_CONST_INPUT_TEST_LIB_LINK_FLAG})
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND (NOT onnxruntime_MINIMAL_BUILD OR onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))

  file(GLOB_RECURSE custom_op_local_function_test_library_src
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.h"
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/dummy_gemm.cc"
        "${TEST_SRC_DIR}/testdata/custom_op_local_function/dummy_gemm.h"
  )

  onnxruntime_add_shared_library_module(custom_op_local_function ${custom_op_local_function_test_library_src})

  onnxruntime_add_include_to_target(custom_op_local_function onnxruntime_common GTest::gtest GTest::gmock)
  target_include_directories(custom_op_local_function PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session
                                                                            ${REPO_ROOT}/include/onnxruntime/core/common)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif(NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_local_function/custom_op_local_function.def")
  endif()

  set_property(TARGET custom_op_local_function APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_lOCAL_FUNCTION_TEST_LIB_LINK_FLAG})
endif()

# Build library that can be used with RegisterExecutionProviderLibrary and automatic EP selection
# We need a shared lib build to use that as a dependency for the test library
if (onnxruntime_BUILD_SHARED_LIB AND
    NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND
    NOT onnxruntime_MINIMAL_BUILD)

  #
  # example_plugin_ep
  #
  file(GLOB onnxruntime_autoep_test_library_src "${TEST_SRC_DIR}/autoep/library/example_plugin_ep/*.h"
                                                "${TEST_SRC_DIR}/autoep/library/example_plugin_ep/*.cc"
                                                "${TEST_SRC_DIR}/autoep/library/ep_context_data_utils.h"
                                                "${TEST_SRC_DIR}/autoep/library/plugin_ep_utils.h")
  onnxruntime_add_shared_library_module(example_plugin_ep ${onnxruntime_autoep_test_library_src})
  target_include_directories(example_plugin_ep PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(example_plugin_ep PRIVATE onnxruntime ${GSL_TARGET})

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG "-Xlinker -dead_strip")
    elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/autoep/library/example_plugin_ep/example_plugin_ep_library.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/autoep/library/example_plugin_ep/example_plugin_ep_library.def")
  endif()

  set_property(TARGET example_plugin_ep APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_AUTOEP_LIB_LINK_FLAG})

  set_target_properties(example_plugin_ep PROPERTIES FOLDER "ONNXRuntimeTest")
  source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_autoep_test_library_src})

  #
  # example_plugin_ep_virt_gpu
  #
  set(onnxruntime_autoep_test_example_plugin_ep_virt_gpu_src
          "${TEST_SRC_DIR}/autoep/library/plugin_ep_utils.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep_lib_entry.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep_factory.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep_factory.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep.cc")
  onnxruntime_add_shared_library_module(example_plugin_ep_virt_gpu ${onnxruntime_autoep_test_example_plugin_ep_virt_gpu_src})
  target_include_directories(example_plugin_ep_virt_gpu PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(example_plugin_ep_virt_gpu PRIVATE onnxruntime ${GSL_TARGET})

  if(UNIX)
    if (APPLE)
	    set(ONNXRUNTIME_AUTOEP_EP_LIB_VIRT_GPU_LINK_FLAG "-Xlinker -dead_strip")
    elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_AUTOEP_EP_LIB_VIRT_GPU_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_AUTOEP_EP_LIB_VIRT_GPU_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/autoep/library/example_plugin_ep_virt_gpu/ep_lib.def")
  endif()

  set_property(TARGET example_plugin_ep_virt_gpu APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_AUTOEP_EP_LIB_VIRT_GPU_LINK_FLAG})

  set_target_properties(example_plugin_ep_virt_gpu PROPERTIES FOLDER "ONNXRuntimeTest")
  source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_autoep_test_example_plugin_ep_virt_gpu_src})

  #
  # example_plugin_ep_kernel_registry
  #
  set(onnxruntime_autoep_test_example_plugin_ep_kernel_registry_src
          "${TEST_SRC_DIR}/autoep/library/plugin_ep_utils.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_lib_entry.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_factory.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_factory.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_allocator.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_data_transfer.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_data_transfer.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_kernel_registration.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_kernel_registration.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_profiling.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_profiling.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/utils.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/squeeze.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/squeeze.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/relu.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/relu.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/binary_op.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/binary_op.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/scan.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/scan.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/loop.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/loop.cc"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/if.h"
          "${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/kernels/if.cc")
  onnxruntime_add_shared_library_module(example_plugin_ep_kernel_registry ${onnxruntime_autoep_test_example_plugin_ep_kernel_registry_src})
  target_include_directories(example_plugin_ep_kernel_registry PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(example_plugin_ep_kernel_registry PRIVATE onnxruntime ${GSL_TARGET})

  if(UNIX)
    if (APPLE)
	    set(ONNXRUNTIME_EXAMPLE_PLUGIN_EP_KERNEL_REGISTRY_LINK_FLAG "-Xlinker -dead_strip")
    elseif (NOT CMAKE_SYSTEM_NAME MATCHES "AIX")
      string(CONCAT ONNXRUNTIME_EXAMPLE_PLUGIN_EP_KERNEL_REGISTRY_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_EXAMPLE_PLUGIN_EP_KERNEL_REGISTRY_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/autoep/library/example_plugin_ep_kernel_registry/ep_lib.def")
  endif()

  set_property(TARGET example_plugin_ep_kernel_registry APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_EXAMPLE_PLUGIN_EP_KERNEL_REGISTRY_LINK_FLAG})

  set_target_properties(example_plugin_ep_kernel_registry PROPERTIES FOLDER "ONNXRuntimeTest")
  source_group(TREE ${TEST_SRC_DIR} FILES ${onnxruntime_autoep_test_example_plugin_ep_kernel_registry_src})

  #
  # test library
  #
  file(GLOB onnxruntime_autoep_test_SRC "${ONNXRUNTIME_AUTOEP_TEST_SRC_DIR}/*.h"
                                        "${ONNXRUNTIME_AUTOEP_TEST_SRC_DIR}/*.cc")

  set(onnxruntime_autoep_test_LIBS onnxruntime_mocked_allocator ${ONNXRUNTIME_TEST_LIBS} onnxruntime_test_utils
                                   onnx_proto onnx ${onnxruntime_EXTERNAL_LIBRARIES})

  if (onnxruntime_USE_TENSORRT)
    list(APPEND onnxruntime_autoep_test_LIBS ${TENSORRT_LIBRARY_INFER})
  endif()

  if (onnxruntime_USE_CUDA)
    list(APPEND onnxruntime_autoep_test_LIBS CUDA::cudart)
  endif()

  if (onnxruntime_USE_DML)
    list(APPEND onnxruntime_autoep_test_LIBS d3d12.lib)
  endif()

  if (CPUINFO_SUPPORTED)
    list(APPEND onnxruntime_autoep_test_LIBS cpuinfo)
  endif()

  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND onnxruntime_autoep_test_LIBS onnxruntime_graph onnxruntime_session ${onnxruntime_providers_target}
                onnxruntime_optimizer onnxruntime_mlas onnxruntime_framework onnxruntime_util onnxruntime_flatbuffers
                iconv re2 onnx)
  endif()

  AddTest(DYN
          TARGET onnxruntime_autoep_test
          SOURCES ${onnxruntime_autoep_test_SRC} ${onnxruntime_unittest_main_src}
          LIBS ${onnxruntime_autoep_test_LIBS}
          DEPENDS ${all_dependencies} example_plugin_ep example_plugin_ep_virt_gpu example_plugin_ep_kernel_registry
  )
endif()

if (onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  set (onnxruntime_logging_apis_test_SRC
       ${ONNXRUNTIME_LOGGING_APIS_TEST_SRC_DIR}/test_logging_apis.cc)

  set(onnxruntime_logging_apis_test_LIBS onnxruntime_common onnxruntime_test_utils)
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND onnxruntime_logging_apis_test_LIBS onnxruntime_session onnxruntime_util onnxruntime_lora onnxruntime_framework onnxruntime_common onnxruntime_graph  ${onnxruntime_providers_target} onnxruntime_mlas onnxruntime_optimizer onnxruntime_flatbuffers iconv re2 ${PROTOBUF_LIB} onnx onnx_proto)
     endif()

  if(NOT WIN32)
    list(APPEND onnxruntime_logging_apis_test_LIBS  ${CMAKE_DL_LIBS})
  endif()

  AddTest(DYN
          TARGET onnxruntime_logging_apis_test
          SOURCES ${onnxruntime_logging_apis_test_SRC}
          LIBS ${onnxruntime_logging_apis_test_LIBS}
          DEPENDS ${all_dependencies}
  )
endif()

if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_USE_OPENVINO AND (NOT onnxruntime_MINIMAL_BUILD OR
                                                                        onnxruntime_MINIMAL_BUILD_CUSTOM_OPS))
  onnxruntime_add_shared_library_module(custom_op_openvino_wrapper_library
                                        ${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.cc
                                        ${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/openvino_wrapper.cc)
  target_include_directories(custom_op_openvino_wrapper_library PRIVATE ${REPO_ROOT}/include/onnxruntime/core/session)
  target_link_libraries(custom_op_openvino_wrapper_library PRIVATE openvino::runtime)

  if(UNIX)
    if (APPLE)
      set(ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG "-Xlinker -dead_strip")
    else()
      string(CONCAT ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG
             "-Xlinker --version-script=${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.lds "
             "-Xlinker --no-undefined -Xlinker --gc-sections -z noexecstack")
    endif()
  else()
    set(ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG
        "-DEF:${TEST_SRC_DIR}/testdata/custom_op_openvino_wrapper_library/custom_op_lib.def")
  endif()

  set_property(TARGET custom_op_openvino_wrapper_library APPEND_STRING PROPERTY LINK_FLAGS
               ${ONNXRUNTIME_CUSTOM_OP_OPENVINO_WRAPPER_LIB_LINK_FLAG})
endif()

if (onnxruntime_USE_WEBGPU AND onnxruntime_USE_EXTERNAL_DAWN AND TARGET dawn::dawn_native)
  if (onnxruntime_BUILD_SHARED_LIB)
    AddTest(DYN TARGET onnxruntime_webgpu_external_dawn_test
            SOURCES ${onnxruntime_webgpu_external_dawn_test_SRC}
            LIBS dawn::dawn_native dawn::dawn_proc
            DEPENDS ${all_dependencies})
  else()
    AddTest(TARGET onnxruntime_webgpu_external_dawn_test
            SOURCES ${onnxruntime_webgpu_external_dawn_test_SRC}
            LIBS dawn::dawn_native dawn::dawn_proc ${onnxruntime_test_providers_libs}
            DEPENDS ${all_dependencies})
  endif()
  onnxruntime_add_include_to_target(onnxruntime_webgpu_external_dawn_test dawn::dawncpp_headers dawn::dawn_headers)
  target_compile_features(onnxruntime_webgpu_external_dawn_test PRIVATE cxx_std_20)
endif()

if (onnxruntime_USE_WEBGPU AND WIN32 AND onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  AddTest(DYN
          TARGET onnxruntime_webgpu_delay_load_test
          SOURCES ${onnxruntime_webgpu_delay_load_test_SRC}
          LIBS ${SYS_PATH_LIB}
          DEPENDS ${all_dependencies}
  )
endif()

# onnxruntime_ep_graph_test tests the implementation of the public OrtGraph APIs for use in plugin EPs (OrtEp).
if (onnxruntime_BUILD_SHARED_LIB AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND NOT onnxruntime_MINIMAL_BUILD)
  file(GLOB_RECURSE onnxruntime_ep_graph_test_SRC "${ONNXRUNTIME_EP_GRAPH_TEST_SRC_DIR}/*.h"
                                                  "${ONNXRUNTIME_EP_GRAPH_TEST_SRC_DIR}/*.cc")

  set(onnxruntime_ep_graph_test_LIBS ${ONNXRUNTIME_TEST_LIBS} onnxruntime_test_utils ${onnxruntime_EXTERNAL_LIBRARIES})
  if (CMAKE_SYSTEM_NAME MATCHES "AIX")
    list(APPEND onnxruntime_ep_graph_test_LIBS onnxruntime_session onnxruntime_util onnxruntime_lora onnxruntime_framework
                                               onnxruntime_common onnxruntime_graph ${onnxruntime_providers_target} onnxruntime_mlas
                                               onnxruntime_optimizer onnxruntime_flatbuffers iconv re2
                                               ${PROTOBUF_LIB} onnx onnx_proto)
  endif()

  if(NOT WIN32)
    list(APPEND onnxruntime_ep_graph_test_LIBS  ${CMAKE_DL_LIBS})
  endif()

  if (onnxruntime_USE_TENSORRT OR onnxruntime_USE_NV)
    # Need this because unittest_main_src defines a global nvinfer1::IBuilder variable.
    list(APPEND onnxruntime_ep_graph_test_LIBS ${TENSORRT_LIBRARY_INFER})
  endif()

  AddTest(DYN
          TARGET onnxruntime_ep_graph_test
          SOURCES ${onnxruntime_ep_graph_test_SRC} ${onnxruntime_unittest_main_src}
          LIBS ${onnxruntime_ep_graph_test_LIBS}
          DEPENDS ${all_dependencies}
  )
endif()

if(WIN32 AND onnxruntime_USE_1DS_TELEMETRY)
  if(NOT GDK_PLATFORM AND NOT CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
    set(telemetry_lifecycle_test_dir "${TEST_SRC_DIR}/platform/windows_telemetry/lifecycle")
    onnxruntime_add_shared_library(onnxruntime_telemetry_shutdown_test_library
      "${telemetry_lifecycle_test_dir}/telemetry_shutdown_test_library.cc"
      "${telemetry_lifecycle_test_dir}/telemetry_shutdown_test_library.def")
    target_link_libraries(onnxruntime_telemetry_shutdown_test_library PRIVATE onnxruntime_common mat)
    onnxruntime_add_executable(onnxruntime_telemetry_shutdown_test
      "${telemetry_lifecycle_test_dir}/telemetry_shutdown_test.cc")
    add_dependencies(onnxruntime_telemetry_shutdown_test onnxruntime_telemetry_shutdown_test_library)
    foreach(mode process_exit normal_destruction explicit_unload)
      add_test(NAME onnxruntime_telemetry_${mode}_test
        COMMAND onnxruntime_telemetry_shutdown_test
          "$<TARGET_FILE:onnxruntime_telemetry_shutdown_test_library>" ${mode}
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:onnxruntime_telemetry_shutdown_test>")
      set_tests_properties(onnxruntime_telemetry_${mode}_test PROPERTIES TIMEOUT 15)
    endforeach()
    set_target_properties(onnxruntime_telemetry_shutdown_test_library onnxruntime_telemetry_shutdown_test
      PROPERTIES FOLDER "ONNXRuntimeTest")
  endif()
endif()

include(onnxruntime_fuzz_test.cmake)

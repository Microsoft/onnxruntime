
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)

cmake_policy(SET CMP0092 NEW)
cmake_policy(SET CMP0091 NEW)
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.20")
  cmake_policy(SET CMP0117 NEW)
endif()

# Project
project(onnxruntime C CXX)
# Needed for Java
set (CMAKE_C_STANDARD 99)

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(CMakeDependentOption)

set(CMAKE_CXX_STANDARD 17)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Enable CTest
enable_testing()
include(Dart)

if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Options
option(onnxruntime_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(onnxruntime_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(onnxruntime_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(onnxruntime_ENABLE_PYTHON "Enable python buildings" OFF)
# Enable it may cause LNK1169 error
option(onnxruntime_ENABLE_MEMLEAK_CHECKER "Experimental: Enable memory leak checker in Windows debug build" OFF)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
option(onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO "When building with CUDA support, generate device code line number information." OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_COREML "Build with CoreML support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_RKNPU "Build with RKNPU support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_FEATURIZERS "Build ML Featurizers support" OFF)
option(onnxruntime_DEV_MODE "Enable developer warnings and treat most of them as error." OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_BUILD_CSHARP "Build C# library" OFF)
option(onnxruntime_BUILD_OBJC "Build Objective-C library" OFF)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)

option(onnxruntime_USE_TVM "Build tvm for code-gen" OFF)
option(onnxruntime_USE_LLVM "Build tvm with LLVM" OFF)

option(onnxruntime_BUILD_FOR_NATIVE_MACHINE "Enable this option for turning on optimization specific to this machine" OFF)
option(onnxruntime_USE_AVX "Use AVX instructions" OFF)
option(onnxruntime_USE_AVX2 "Use AVX2 instructions" OFF)
option(onnxruntime_USE_AVX512 "Use AVX512 instructions" OFF)

option(onnxruntime_USE_OPENMP "Build with OpenMP support" OFF)
option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_BUILD_APPLE_FRAMEWORK "Build a macOS/iOS framework" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)
option(onnxruntime_USE_NUPHAR "Build with Nuphar" OFF)
option(onnxruntime_USE_VITISAI "Build with Vitis-AI" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)
option(onnxruntime_GCOV_COVERAGE "Compile with options necessary to run code coverage" OFF)

#It's preferred to turn it OFF when onnxruntime is dynamically linked to PROTOBUF. But Tensort always required the full version of protobuf.
cmake_dependent_option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF "NOT onnxruntime_USE_TENSORRT" ON)
option(tensorflow_C_PACKAGE_PATH "Path to tensorflow C package installation dir")
option(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS "Enable operator implemented in language other than cpp" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump debug information about node inputs and outputs when executing the model." OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_MIGRAPHX "Build with AMDMIGraphX support" OFF)
option(onnxruntime_USE_WINML "Build with WinML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_USE_ACL_1902 "Build with ACL version 1902 support" OFF)
option(onnxruntime_USE_ACL_1905 "Build with ACL version 1905 support" OFF)
option(onnxruntime_USE_ACL_1908 "Build with ACL version 1908 support" OFF)
option(onnxruntime_USE_ACL_2002 "Build with ACL version 2002 support" OFF)
option(onnxruntime_USE_ARMNN "Build with ArmNN support" OFF)
option(onnxruntime_ARMNN_RELU_USE_CPU "Use the CPU implementation for the Relu operator for the ArmNN EP" ON)
option(onnxruntime_ARMNN_BN_USE_CPU "Use the CPU implementation for the Batch Normalization operator for the ArmNN EP" ON)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
#The onnxruntime_PREFER_SYSTEM_LIB is mainly designed for package managers like apt/yum/vcpkg.
#Please note, by default Protobuf_USE_STATIC_LIBS is OFF but it's recommended to turn it ON on Windows. You should set it properly when onnxruntime_PREFER_SYSTEM_LIB is ON otherwise you'll hit linkage errors.
#If you have already installed protobuf(or the others) in your system at the default system paths(like /usr/include), then it's better to set onnxruntime_PREFER_SYSTEM_LIB ON. Otherwise onnxruntime may see two different protobuf versions and we won't know which one will be used, the worst case could be onnxruntime picked up header files from one of them but the binaries from the other one.
#It's default OFF because it's experimental now.
option(onnxruntime_PREFER_SYSTEM_LIB "Experimental: Build with the preinstalled libraries in your system" OFF)
option(onnxruntime_USE_ROCM "Build with AMD GPU support" OFF)

# Options related to reducing the binary size produced by the build
option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF)
option(onnxruntime_DISABLE_ML_OPS "Disable traditional ML ops" OFF)
cmake_dependent_option(onnxruntime_DISABLE_RTTI "Disable RTTI" ON "NOT onnxruntime_ENABLE_PYTHON" OFF)
# For now onnxruntime_DISABLE_EXCEPTIONS will only work with onnxruntime_MINIMAL_BUILD, more changes (ONNX, non-CPU EP, ...) are required to run this standalone
option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling. Requires onnxruntime_MINIMAL_BUILD currently." OFF)
option(onnxruntime_MINIMAL_BUILD "Exclude as much as possible from the build. Support ORT format models. No support for ONNX format models." OFF)
option(onnxruntime_EXTENDED_MINIMAL_BUILD "onnxruntime_MINIMAL_BUILD with support for execution providers that compile kernels." OFF)
option(onnxruntime_MINIMAL_BUILD_CUSTOM_OPS "Add custom operator kernels support to a minimal build." OFF)
option(onnxruntime_REDUCED_OPS_BUILD "Reduced set of kernels are registered in build via modification of the kernel registration source files." OFF)
option(onnxruntime_DISABLE_ORT_FORMAT_LOAD "Disable loading an ORT format model when onnxruntime_MINIMAL_BUILD=OFF (i.e. in a full build)." OFF)
option(onnxruntime_DISABLE_EXTERNAL_INITIALIZERS "Don't allow models to load external data" OFF)

#A special option just for debugging and sanitize check. Please do not enable in option in retail builds.
#The option has no effect on Windows.
option(onnxruntime_USE_VALGRIND "Build with valgrind hacks" OFF)

# A special build option only used for gathering code coverage info
option(onnxruntime_RUN_MODELTEST_IN_DEBUG_MODE "Run model tests even in debug mode" OFF)

# options for security fuzzing
# build configuration for fuzz testing is in onnxruntime_fuzz_test.cmake
option(onnxruntime_FUZZ_TEST "Enable Fuzz testing" OFF)

# training options
option(onnxruntime_ENABLE_NVTX_PROFILE "Enable NVTX profile." OFF)
option(onnxruntime_ENABLE_MEMORY_PROFILE "Enable memory profile." OFF)
option(onnxruntime_ENABLE_TRAINING "Enable training functionality." OFF)
option(onnxruntime_ENABLE_TRAINING_OPS "Include training operators but no training session support." OFF)
option(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP "Enable training kernels interop with torch." OFF)
option(onnxruntime_ENABLE_TRAINING_E2E_TESTS "Enable training end-to-end tests." OFF)
option(onnxruntime_ENABLE_CPU_FP16_OPS "Build with advanced instruction sets" ON)
option(onnxruntime_USE_NCCL "Build with NCCL support" OFF)
option(onnxruntime_USE_MPI "Build with MPI support" OFF)

# build WebAssembly
option(onnxruntime_BUILD_WEBASSEMBLY "Enable this option to create WebAssembly byte codes" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "Enable this option to create WebAssembly byte codes with multi-threads support" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "Enable this option to turn on exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO "Enable this option to turn on DWARF format debug info" OFF)

# Enable bitcode for iOS
option(onnxruntime_ENABLE_BITCODE "Enable bitcode for iOS only" OFF)

# build eager mode
option(onnxruntime_ENABLE_EAGER_MODE "build ort eager mode")

# build separate library of schemas of (custom) ops used by ORT (for ONNX to MLIR translation)
option(onnxruntime_BUILD_OPSCHEMA_LIB "Build op schema library" ON)

# option to enable custom operators in onnxruntime-extensions
option(onnxruntime_ENABLE_EXTENSION_CUSTOM_OPS "Enable custom operators in onnxruntime-extensions" OFF)

# Single output director for all binaries
set (RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Single output directory for all binaries.")

function(set_msvc_c_cpp_compiler_warning_level warning_level)
  if (NOT "${warning_level}" MATCHES "^[0-4]$")
    message(FATAL_ERROR "Expected warning_level value of 0-4, got '${warning_level}'.")
  endif()

  if(MSVC)
    set(warning_flag "/W${warning_level}")
    get_property(opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    # only match the generator expression added by this function
    list(FILTER opts
         EXCLUDE REGEX "^\\$<\\$<OR:\\$<COMPILE_LANGUAGE:C>,\\$<COMPILE_LANGUAGE:CXX>>:/W[0-4]>$")
    list(APPEND opts "$<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:${warning_flag}>")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${opts}")
  endif()
endfunction()

# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)

# Fuzz test has only been tested with BUILD_SHARED_LIB option,
# using the MSVC compiler and on windows OS.
if(MSVC AND WIN32 AND onnxruntime_FUZZ_TEST AND onnxruntime_BUILD_SHARED_LIB AND onnxruntime_USE_FULL_PROTOBUF)
  # Fuzz test library dependency, protobuf-mutator,
  # needs the onnx message to be compiled using "non-lite protobuf version"
  set(onnxruntime_FUZZ_ENABLED ON)
endif()

if(onnxruntime_USE_VALGRIND AND NOT WIN32)
  add_definitions(-DRE2_ON_VALGRIND=1)
endif()

if (onnxruntime_ENABLE_NVTX_PROFILE)
  message(WARNING "NTVX profile temporarily disabled, will be fixed soon")
# TODO: This doesn't work with the shared cuda provider. Disabling temporarily to do a clean fix later as it wasn't trivial
#  add_definitions(-DENABLE_NVTX_PROFILE=1)
endif()

if (onnxruntime_ENABLE_BITCODE)
  if (NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS"))
    message(FATAL_ERROR "iOS platform required for onnxruntime_ENABLE_BITCODE")
  endif()
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE YES)
  set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE "bitcode")
else()
  set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO)
endif()

if (onnxruntime_ENABLE_MEMORY_PROFILE)
  add_definitions(-DORT_MEMORY_PROFILE=1)
endif()

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
#nsync tests failed on Mac Build
set(NSYNC_ENABLE_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
set(ONNX_ML 1)
if(NOT onnxruntime_ENABLE_PYTHON)
  set(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS OFF)
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  add_compile_definitions(ENABLE_LANGUAGE_INTEROP_OPS)
endif()

if (NOT (UNIX AND onnxruntime_ENABLE_PYTHON AND onnxruntime_ENABLE_TRAINING AND (NOT onnxruntime_BUILD_SHARED_LIB)))
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    message(WARNING "onnxruntime_ENABLE_TRAINING_TORCH_INTEROP is turned OFF due to incompatible build combinations.")
  endif()
  set(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OFF)
endif()

set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB OFF)
if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  add_compile_definitions(ENABLE_TRAINING_TORCH_INTEROP)

  # Python::Python is required for building unit test executables. 
  if (onnxruntime_BUILD_UNIT_TESTS)
    set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB ON)
  endif()
endif()

# General C# properties
if (onnxruntime_BUILD_CSHARP)
  check_language(CSharp)
  if (CMAKE_CSharp_COMPILER)
    enable_language(CSharp)
    set(CMAKE_CSharp_FLAGS ${CMAKE_CSharp_FLAGS} "/langversion:6")
    message(STATUS "CMAKE_Csharp_Compiler = ${CMAKE_CSharp_COMPILER}")
  else()
    message(WARNING "Language Csharp is not found in the system")
  endif()
endif()

if(NOT WIN32)
  #TODO: On Linux we may try https://github.com/microsoft/TraceLogging
  if(onnxruntime_ENABLE_INSTRUMENT)
    message(WARNING "Instrument is only supported on Windows now")
    set(onnxruntime_ENABLE_INSTRUMENT OFF)
  endif()
else()
  if(WINDOWS_STORE)
    # cmake/external/protobuf/src/google/protobuf/compiler/subprocess.cc and onnxruntime/core/platform/windows/env.cc call a bunch of Win32 APIs.
    # For now, we'll set the API family to desktop globally to expose Win32 symbols in headers; this must be fixed!
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/FI${CMAKE_CURRENT_SOURCE_DIR}/set_winapi_family_desktop.h>")
  endif()
endif()

if(onnxruntime_USE_OPENMP)
  find_package(OpenMP)
  if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    include_directories(${OpenMP_CXX_INCLUDE_DIR})
  else()
    message(WARNING "Flag --use_openmp is specified, but OpenMP is not found in current build environment. Setting it to OFF.")
    set(onnxruntime_USE_OPENMP OFF)
  endif()
endif()

# 'extended' implies minimal.
if (onnxruntime_EXTENDED_MINIMAL_BUILD AND NOT onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_MINIMAL_BUILD ON)
endif()

# Enable space optimization for gcc/clang
# Cannot use "-ffunction-sections -fdata-sections" if we enable bitcode (iOS)
if (NOT MSVC AND NOT onnxruntime_ENABLE_BITCODE)
  string(APPEND CMAKE_CXX_FLAGS " -ffunction-sections -fdata-sections")
  string(APPEND CMAKE_C_FLAGS " -ffunction-sections -fdata-sections")
endif()

if (onnxruntime_BUILD_WEBASSEMBLY)
  # Enable LTO for release single-thread build
  if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug" AND NOT onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    # NOTES:
    # (1) LTO does not work for WebAssembly multi-thread. (segment fault in worker)
    # (2) "-flto=thin" does not work correctly for wasm-ld.
    #     we don't set onnxruntime_ENABLE_LTO because it appends flag "-flto=thin"
    #     instead, we manually set CMAKE_CXX_FLAGS "-flto"
    string(APPEND CMAKE_CXX_FLAGS " -flto")
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO)
    # "-g3" generates DWARF format debug info.
    # NOTE: With debug info enabled, web assembly artifacts will be very huge (>1GB). So we offer an option to build without debug info.
    set(CMAKE_CXX_FLAGS_DEBUG "-g3")
  else()
    # do not generate any debug info. This helps to accelerate building process and reduce binary size.
    set(CMAKE_CXX_FLAGS_DEBUG "-g0")
  endif()

  # Build WebAssembly with multi-threads support.
  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    string(APPEND CMAKE_CXX_FLAGS " -pthread")
  endif()
endif()

# ORT build with as much excluded as possible. Supports ORT flatbuffers models only.
if (onnxruntime_MINIMAL_BUILD)
  add_compile_definitions(ORT_MINIMAL_BUILD)
  add_compile_definitions(ENABLE_ORT_FORMAT_LOAD)

  if (onnxruntime_EXTENDED_MINIMAL_BUILD)
    # enable EPs that compile kernels at runtime
    add_compile_definitions(ORT_EXTENDED_MINIMAL_BUILD)
  endif()

  if (onnxruntime_MINIMAL_BUILD_CUSTOM_OPS)
    add_compile_definitions(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  endif()

  set(onnxruntime_REDUCED_OPS_BUILD ON)

  if (NOT onnxruntime_ENABLE_PYTHON)
    # Python bindings use typeid so can't automatically disable RTTI in that case
    set(onnxruntime_DISABLE_RTTI ON)
  endif()

  if (MSVC)
    # turn on LTO (which adds some compiler flags and turns on LTCG) unless it's a Debug build to minimize binary size
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
      set (onnxruntime_ENABLE_LTO ON)
    endif()

    # undocumented internal flag to allow analysis of a minimal build binary size
    if (ADD_DEBUG_INFO_TO_MINIMAL_BUILD)
      string(APPEND CMAKE_CXX_FLAGS " /Zi")
      string(APPEND CMAKE_C_FLAGS " /Zi")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS " /debug")
    endif()
  else()
    if (CMAKE_HOST_SYSTEM MATCHES "Darwin")
      add_link_options(-Wl, -dead_strip)
    else()
      add_link_options(-Wl,--gc-sections)
    endif()

    if (ADD_DEBUG_INFO_TO_MINIMAL_BUILD)
      string(APPEND CMAKE_CXX_FLAGS " -g")
      string(APPEND CMAKE_C_FLAGS " -g")
    endif()
  endif()
else()
  # support ORT format model loading unless onnxruntime_DISABLE_ORT_FORMAT_LOAD is set
  if (NOT onnxruntime_DISABLE_ORT_FORMAT_LOAD)
    add_compile_definitions(ENABLE_ORT_FORMAT_LOAD)
  endif()
endif()

if(onnxruntime_ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ipo_enabled OUTPUT ipo_output)
    if(NOT ipo_enabled)
      message(WARNING "IPO is not supported by this compiler")
      set(onnxruntime_ENABLE_LTO OFF)
    else()
      set (CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
    endif()
endif()

if (onnxruntime_REDUCED_OPS_BUILD)
  add_compile_definitions(REDUCED_OPS_BUILD)
endif()

if(onnxruntime_DISABLE_EXTERNAL_INITIALIZERS)
  add_definitions(-DDISABLE_EXTERNAL_INITIALIZERS=1)
endif()

if(onnxruntime_DISABLE_RTTI)
  add_compile_definitions(ORT_NO_RTTI GOOGLE_PROTOBUF_NO_RTTI)
  if(MSVC)
    # Disable RTTI and turn usage of dynamic_cast and typeid into errors
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GR->" "$<$<COMPILE_LANGUAGE:CXX>:/we4541>")
  else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>")
  endif()
else()
  #MSVC RTTI flag /GR is not added to CMAKE_CXX_FLAGS by default. But, anyway VC++2019 treats "/GR" default on.
  #So we don't need the following three lines. But it's better to make it more explicit.
  if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/GR>")
  endif()
endif()

# If this is only enabled in an onnxruntime_ORT_MODEL_FORMAT_ONLY build we don't need ONNX changes
# as we (currently) only pull in data_type_utils.cc/h which doesn't throw
if(onnxruntime_DISABLE_EXCEPTIONS)
  if(NOT onnxruntime_MINIMAL_BUILD)
    message(FATAL_ERROR "onnxruntime_MINIMAL_BUILD required for onnxruntime_DISABLE_EXCEPTIONS")
  endif()

  add_compile_definitions("ORT_NO_EXCEPTIONS")
  add_compile_definitions("MLAS_NO_EXCEPTION")
  add_compile_definitions("ONNX_NO_EXCEPTIONS")

  if(MSVC)
    string(REGEX REPLACE "/EHsc" "/EHs-c-" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    # Eigen throw_std_bad_alloc calls 'new' instead of throwing which results in a nodiscard warning.
    # It also has unreachable code as there's no good way to avoid EIGEN_EXCEPTIONS being set in macros.h
    # TODO: see if we can limit the code this is disabled for.
    string(APPEND CMAKE_CXX_FLAGS " /wd4834 /wd4702")
    add_compile_definitions("_HAS_EXCEPTIONS=0")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-exceptions -fno-unwind-tables -fno-asynchronous-unwind-tables")
  endif()
endif()

# We need to link with libatomic on systems that do not have built-in atomics, or
# don't have built-in support for 8 byte atomics
# Derived from https://github.com/protocolbuffers/protobuf/blob/master/cmake/CMakeLists.txt
set(onnxruntime_LINK_LIBATOMIC false)
# We don't need to link libatomic on iOS
if (NOT MSVC AND NOT (CMAKE_SYSTEM_NAME STREQUAL "iOS"))
  include(CheckCXXSourceCompiles)
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} -std=c++11)
  check_cxx_source_compiles("
    #include <atomic>
    int main() {
      return std::atomic<int64_t>{};
    }
  " onnxruntime_HAVE_BUILTIN_ATOMICS)
  if (NOT onnxruntime_HAVE_BUILTIN_ATOMICS)
    set(onnxruntime_LINK_LIBATOMIC true)
  endif ()
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endif ()

set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)
set(ORTTRAINING_ROOT ${PROJECT_SOURCE_DIR}/../orttraining)
set(ORTTRAINING_SOURCE_DIR ${ORTTRAINING_ROOT}/orttraining)

file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" ORT_VERSION)

# Guarantee that the Eigen code that you are #including is licensed
# under the MPL2 and possibly more permissive licenses (like BSD).
add_definitions(-DEIGEN_MPL2_ONLY)
if(MSVC)
  add_definitions(-DEIGEN_HAS_CONSTEXPR -DEIGEN_HAS_VARIADIC_TEMPLATES -DEIGEN_HAS_CXX11_MATH -DEIGEN_HAS_CXX11_ATOMIC
          -DEIGEN_STRONG_INLINE=inline)
endif()

if(onnxruntime_CROSS_COMPILING)
  set(CMAKE_CROSSCOMPILING ON)
  check_cxx_compiler_flag(-Wno-error HAS_NOERROR)
  if(HAS_NOERROR)
    string(APPEND CMAKE_CXX_FLAGS " -Wno-error=attributes")
    string(APPEND CMAKE_C_FLAGS " -Wno-error=attributes")
  endif()
endif()

# Mark symbols to be invisible, for macOS/iOS target only
# Due to many dependencies have different symbol visibility settings, set global compile flags here.
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS")
  foreach(flags CMAKE_CXX_FLAGS CMAKE_OBJC_FLAGS CMAKE_OBJCXX_FLAGS)
    string(APPEND ${flags} " -fvisibility=hidden -fvisibility-inlines-hidden")
  endforeach()
endif()

#must after OpenMP settings
find_package(Threads)

macro(check_nvcc_compiler_flag _FLAG _RESULT)
    execute_process(COMMAND ${onnxruntime_CUDA_HOME}/bin/nvcc "${_FLAG}" RESULT_VARIABLE NVCC_OUT ERROR_VARIABLE NVCC_ERROR)
    message("NVCC_ERROR = ${NVCC_ERROR}")
    message("NVCC_OUT = ${NVCC_OUT}")
    if("${NVCC_OUT}" MATCHES "0")
        set(${_RESULT} 1)
    else()
        set(${_RESULT} 0)
    endif()
endmacro()

#Set global compile flags for all the source code(including third_party code like protobuf)
#This section must be before any add_subdirectory, otherwise build may fail because /MD,/MT mismatch
if (MSVC)
  #Always enable exception handling, even for Windows ARM
  if(NOT onnxruntime_DISABLE_EXCEPTIONS)
    string(APPEND CMAKE_CXX_FLAGS " /EHsc /wd26812")
    string(APPEND CMAKE_C_FLAGS " /EHsc /wd26812")
  endif()
  string(APPEND CMAKE_CXX_FLAGS " /external:W0 /external:templates- /external:I ${CMAKE_CURRENT_SOURCE_DIR} /external:I ${CMAKE_CURRENT_BINARY_DIR}")
  if(onnxruntime_USE_AVX)
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX")
  elseif(onnxruntime_USE_AVX2)
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX2")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX2")
  elseif(onnxruntime_USE_AVX512)
    string(APPEND CMAKE_CXX_FLAGS " /arch:AVX512")
    string(APPEND CMAKE_C_FLAGS " /arch:AVX512")
  endif()

  if (onnxruntime_ENABLE_LTO AND NOT onnxruntime_USE_CUDA)
    SET (CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /Gw /GL")
    SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /Gw /GL")
    SET (CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /Gw /GL")
  endif()

  # The WinML build tool chain builds ARM/ARM64, and the internal tool chain does not have folders for spectre mitigation libs.
  # WinML performs spectre mitigation differently.
  if (NOT DEFINED onnxruntime_DISABLE_QSPECTRE_CHECK)
    check_cxx_compiler_flag(-Qspectre HAS_QSPECTRE)
    if (HAS_QSPECTRE)
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /Qspectre")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Qspectre")
    endif()
  endif()
  SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /DYNAMICBASE")
  check_cxx_compiler_flag(-guard:cf HAS_GUARD_CF)
  if (HAS_GUARD_CF)
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} /guard:cf")
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /guard:cf")
    SET(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} /guard:cf")
    SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /guard:cf")
    SET(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} /guard:cf")
    SET(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} /guard:cf")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /guard:cf")
  endif()
else()
  if(onnxruntime_BUILD_FOR_NATIVE_MACHINE)
    string(APPEND CMAKE_CXX_FLAGS " -march=native -mtune=native")
    string(APPEND CMAKE_C_FLAGS " -march=native -mtune=native")
  elseif(onnxruntime_USE_AVX)
    string(APPEND CMAKE_CXX_FLAGS " -mavx")
    string(APPEND CMAKE_C_FLAGS " -mavx")
  elseif(onnxruntime_USE_AVX2)
    string(APPEND CMAKE_CXX_FLAGS " -mavx2")
    string(APPEND CMAKE_C_FLAGS " -mavx2")
  elseif(onnxruntime_USE_AVX512)
    string(APPEND CMAKE_CXX_FLAGS " -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")
    string(APPEND CMAKE_C_FLAGS " -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl")
  endif()

  # Check support for AVX and f16c.
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-mf16c" COMPILER_SUPPORT_MF16C)
  if(NOT COMPILER_SUPPORT_MF16C)
    message("F16C instruction set is not supported.")
  endif()

  check_cxx_compiler_flag("-mfma" COMPILER_SUPPORT_FMA)
  if(NOT COMPILER_SUPPORT_FMA)
    message("FMA instruction set is not supported.")
  endif()

  check_cxx_compiler_flag("-mavx" COMPILER_SUPPORT_AVX)
  if(NOT COMPILER_SUPPORT_AVX)
    message("AVX instruction set is not supported.")
  endif()

  if(NOT (COMPILER_SUPPORT_MF16C AND COMPILER_SUPPORT_FMA AND COMPILER_SUPPORT_AVX))
    message("One or more AVX/F16C instruction flags are not supported. ")
    set(onnxruntime_ENABLE_CPU_FP16_OPS FALSE)
  endif()

  # This is enabled only for Adasum files in training mode.
  # The flags won't be applied globally since some high-precision training and inferencing ops will incur precision loss.
  if(onnxruntime_ENABLE_CPU_FP16_OPS)
    set_source_files_properties(${ORTTRAINING_SOURCE_DIR}/core/framework/adasum/adasum_mpi.cc PROPERTIES COMPILE_FLAGS " -fassociative-math -ffast-math -ftree-vectorize -funsafe-math-optimizations -mf16c -mavx -mfma ")
    set_source_files_properties(${ORTTRAINING_SOURCE_DIR}/training_ops/cpu/collective/adasum_kernels.cc PROPERTIES COMPILE_FLAGS " -fassociative-math -ffast-math -ftree-vectorize -funsafe-math-optimizations -mf16c -mavx -mfma ")
    set_source_files_properties(${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/adasum_kernels.cc PROPERTIES COMPILE_FLAGS " -fassociative-math -ffast-math -ftree-vectorize -funsafe-math-optimizations -mf16c -mavx -mfma ")
    add_definitions(-DENABLE_CPU_FP16_TRAINING_OPS)
  endif()
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    #For Mac compliance
    message("Adding flags for Mac builds")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fstack-protector-strong")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "iOSCross")
    #For ios compliance
    message("Adding flags for ios builds")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target arm64-apple-darwin-macho")
    elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "arm")
      set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target armv7a-apple-darwin-macho")
    endif()
endif()

#Dependencies begin
if (onnxruntime_BUILD_UNIT_TESTS)
  if(onnxruntime_PREFER_SYSTEM_LIB)
    find_package(GTest CONFIG)
  endif()
  if(NOT TARGET GTest::gtest)
    message("Use gtest from submodule")

    # WebAssembly threading support in Node.js is still an experimental feature and
    # not working properly with googletest suite.
    if (onnxruntime_BUILD_WEBASSEMBLY)
      set(gtest_disable_pthreads ON)
    endif()

    # gtest and gmock
    set_msvc_c_cpp_compiler_warning_level(4)
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
    set_msvc_c_cpp_compiler_warning_level(3)
    set_target_properties(gmock PROPERTIES FOLDER "External/GTest")
    # disable treating all warnings as errors for gmock
    target_compile_options(gmock PRIVATE "-w")
    target_compile_options(gtest PRIVATE "-w")
    set_target_properties(gmock_main PROPERTIES FOLDER "External/GTest")
    set_target_properties(gtest PROPERTIES FOLDER "External/GTest")
    set_target_properties(gtest_main PROPERTIES FOLDER "External/GTest")
  endif()
endif()

set(ENABLE_DATE_TESTING  OFF CACHE BOOL "" FORCE)
set(USE_SYSTEM_TZ_DB  ON CACHE BOOL "" FORCE)
set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)

if(CMAKE_CROSSCOMPILING)
  message("Doing crosscompiling")
endif()

#Need python to generate def file
if(onnxruntime_BUILD_SHARED_LIB OR onnxruntime_ENABLE_PYTHON)
  if(onnxruntime_ENABLE_PYTHON)
    if(onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS OR onnxruntime_REQUIRE_PYTHON_EMBED_LIB)
      find_package(Python 3.6 COMPONENTS Interpreter Development NumPy)
    else()
      find_package(Python 3.6 COMPONENTS Interpreter Development.Module NumPy)
    endif()
  else()
    find_package(Python 3.6 COMPONENTS Interpreter)
  endif()
endif()

#target begins from here

if(onnxruntime_BUILD_BENCHMARKS)
  if(onnxruntime_PREFER_SYSTEM_LIB)
    find_package(benchmark)
  endif()
  if(NOT TARGET benchmark::benchmark)
    message("Use benchmark from submodule")
    # We will not need to test benchmark lib itself.
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
    # We will not need to install benchmark since we link it statically.
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")
    add_subdirectory(${PROJECT_SOURCE_DIR}/external/googlebenchmark EXCLUDE_FROM_ALL)
    #add_library(benchmark::benchmark ALIAS benchmark)
  endif()
endif()

if(NOT WIN32 AND NOT onnxruntime_PREFER_SYSTEM_LIB)
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/nsync EXCLUDE_FROM_ALL)
endif()
# External dependencies
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)

#protobuf begin

#Here we support two build mode:
#1. if ONNX_CUSTOM_PROTOC_EXECUTABLE is set, build Protobuf from source, except protoc.exe. This mode is mainly
#   for cross-compiling
#2. if ONNX_CUSTOM_PROTOC_EXECUTABLE is not set, Compile everything(including protoc) from source code.

if(onnxruntime_PREFER_SYSTEM_LIB)
  set(protobuf_MODULE_COMPATIBLE ON)
  find_package(Protobuf)
endif()


if(Protobuf_FOUND OR Protobuf_FOUND)
  message("Use protobuf from preinstalled system lib")
  if (onnxruntime_USE_FULL_PROTOBUF)
    set(PROTOBUF_LIB protobuf::libprotobuf)
    #We have a check here but most of the cmake users don't know the Protobuf_USE_STATIC_LIBS
    # variable exists and may leave it in a wrong state.
    if(NOT Protobuf_USE_STATIC_LIBS)
      # ONNX Runtime itself can work in such a setting but it may cause compatibility issues
      # when ONNX Runtime is integrated with the other ONNX ecosystem softwares.
      message(WARNING "Use Protobuf_USE_STATIC_LIBS to ensure compatibility with other ONNX ecosystem components.")
    endif()
  else()
    set(PROTOBUF_LIB protobuf::libprotobuf-lite)
  endif()
else()
  message("Use protobuf from submodule")
  # use protobuf as a submodule
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build protobuf tests" FORCE)
  endif()
  if(onnxruntime_DISABLE_RTTI)
    set(protobuf_DISABLE_RTTI OFF CACHE BOOL "Remove runtime type information in the binaries" FORCE)
  endif()
  add_subdirectory(${PROJECT_SOURCE_DIR}/external/protobuf/cmake EXCLUDE_FROM_ALL)
  
  if(TARGET libprotoc)
    set_target_properties(libprotoc PROPERTIES FOLDER "External/Protobuf")
    add_executable(protobuf::protoc ALIAS protoc)
  endif()
  if(TARGET protoc)
    set_target_properties(protoc PROPERTIES FOLDER "External/Protobuf")
  endif()
  if (onnxruntime_USE_FULL_PROTOBUF)
    set(PROTOBUF_LIB libprotobuf)
  else()
    set(PROTOBUF_LIB libprotobuf-lite)
  endif()
endif()

if(UNIX AND onnxruntime_ENABLE_LTO AND NOT onnxruntime_PREFER_SYSTEM_LIB)
  #https://github.com/protocolbuffers/protobuf/issues/5923
  target_link_options(protoc PRIVATE "-Wl,--no-as-needed")
endif()

if(MSVC)
  target_compile_options(${PROTOBUF_LIB} PRIVATE "/wd5054")
endif()

include(protobuf_function.cmake)
#protobuf end

if(onnxruntime_USE_FEATURIZERS)
  add_definitions(-DML_FEATURIZERS)
  # Fetch and build featurizers
  include(external/featurizers.cmake)
endif()

if (onnxruntime_DISABLE_CONTRIB_OPS)
  add_definitions(-DDISABLE_CONTRIB_OPS)
endif()

if (onnxruntime_DISABLE_ML_OPS)
  add_definitions(-DDISABLE_ML_OPS)
endif()

if (onnxruntime_USE_CUDA AND "${onnxruntime_CUDNN_HOME}" STREQUAL "")
  message(FATAL_ERROR "onnxruntime_CUDNN_HOME required for onnxruntime_USE_CUDA")
endif()


get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
get_filename_component(ORTTRAINING_ROOT "${ORTTRAINING_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)

add_subdirectory(external/date EXCLUDE_FROM_ALL)

set(SAFEINT_INCLUDE_DIR ${REPO_ROOT}/cmake/external/SafeInt)
add_library(safeint_interface INTERFACE)
target_include_directories(safeint_interface INTERFACE ${SAFEINT_INCLUDE_DIR})

set(OPTIONAL_LITE_INCLUDE_DIR ${REPO_ROOT}/cmake/external/optional-lite/include)
if(onnxruntime_DISABLE_EXCEPTIONS)
  add_compile_definitions(optional_CONFIG_NO_EXCEPTIONS=1)
endif()

add_subdirectory(external/mp11 EXCLUDE_FROM_ALL)

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(external/json EXCLUDE_FROM_ALL)

if(onnxruntime_PREFER_SYSTEM_LIB)
  find_package(re2)
endif()
if(NOT TARGET re2::re2)
  add_subdirectory(external/re2 EXCLUDE_FROM_ALL)
  set_target_properties(re2 PROPERTIES FOLDER "External/re2")
  set(RE2_INCLUDE_DIR ${REPO_ROOT}/cmake/external/re2)
endif()


# Adding pytorch CPU info library
# TODO  do we have to add target_include_directories to each project that uses this?
if(MSVC AND (( CMAKE_SYSTEM_PROCESSOR  MATCHES "^(ARM.*|arm.*)$" ) OR (CMAKE_GENERATOR_PLATFORM  MATCHES "^(ARM.*|arm.*)$" ) ))
  # cpuinfo fail to compile with windows arm.
else()
set(PYTORCH_CPUINFO_DIR external/pytorch_cpuinfo)
set(PYTORCH_CPUINFO_INCLUDE_DIR ${PYTORCH_CPUINFO_DIR}/include)
set(CPUINFO_BUILD_TOOLS OFF CACHE INTERNAL "")
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE INTERNAL "")
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE INTERNAL "")
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE INTERNAL "")

if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
 set(IOS ON CACHE INTERNAL "")
 set(IOS_ARCH "${CMAKE_OSX_ARCHITECTURES}" CACHE INTERNAL "")
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_SYSTEM_NAME = ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_OSX_ARCHITECTURES = ${CMAKE_OSX_ARCHITECTURES}")
message(STATUS "IOS_ARCH = ${IOS_ARCH}")

add_subdirectory(external/pytorch_cpuinfo EXCLUDE_FROM_ALL)

endif()
# bounds checking behavior.
# throw instead of calling terminate if there's a bounds checking violation.
# we make it through via a handler so CUDA does not complain
# The following -DGSL macros are recognized by gsl-lite along with -Dgsl macros
# no bounds checking in release build so no perf cost
# if we enable onnxruntime_DISABLE_EXCEPTIONS, gsl will terminate
if (onnxruntime_DISABLE_EXCEPTIONS)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGSL_TERMINATE_ON_CONTRACT_VIOLATION")
else()
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DGSL_THROW_ON_CONTRACT_VIOLATION")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")
set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -DGSL_UNENFORCED_ON_CONTRACT_VIOLATION")

include(eigen)

#onnxruntime_EXTERNAL_LIBRARIES could contain onnx, onnx_proto,libprotobuf, cuda/cudnn,
# dnnl/mklml, onnxruntime_codegen_tvm, tvm, nnvm_compiler and pthread
# pthread is always at the last
set(onnxruntime_EXTERNAL_LIBRARIES onnx onnx_proto ${PROTOBUF_LIB} re2::re2)


set(onnxruntime_LINK_DIRS )
if(onnxruntime_USE_CUDA)
      #TODO: combine onnxruntime_CUDNN_HOME and onnxruntime_CUDA_HOME, assume they are the same
      if (WIN32)
        list(APPEND onnxruntime_LINK_DIRS ${onnxruntime_CUDNN_HOME}/lib/x64 ${onnxruntime_CUDA_HOME}/x64/lib64)
      else()
        list(APPEND onnxruntime_LINK_DIRS ${onnxruntime_CUDNN_HOME}/lib64 ${onnxruntime_CUDA_HOME}/lib64)
      endif()
endif()

FILE(TO_NATIVE_PATH ${CMAKE_BINARY_DIR}  ORT_BINARY_DIR)
FILE(TO_NATIVE_PATH ${PROJECT_SOURCE_DIR}  ORT_SOURCE_DIR)



set(ORT_WARNING_FLAGS)
if (WIN32)
    # parallel build
    # These compiler opitions cannot be forwarded to NVCC, so cannot use add_compiler_options
    string(APPEND CMAKE_CXX_FLAGS " /MP")
    # Several external libraries, like protobuf and eigen, directly compare constants with a normal 'if'.
    list(APPEND ORT_WARNING_FLAGS "/wd4127") # conditional expression is constant
    # class needs to have dll-interface to be used by clients
    list(APPEND ORT_WARNING_FLAGS "/wd4251")
    # issued by thrust nonstandard extension used: nameless struct/union
    list(APPEND ORT_WARNING_FLAGS "/wd4201")
    if(onnxruntime_USE_OPENMP)
      list(APPEND ORT_WARNING_FLAGS "/wd6993") # Code analysis ignores OpenMP constructs
    endif()
    # operator 'operator-name': deprecated between enumerations of different types
    list(APPEND ORT_WARNING_FLAGS "/wd5054")
    # Treat warning as error if onnxruntime_DEV_MODE is ON
    # For cross-compiled ARM64 binaries, there are too many warnings to fix, hence ignore warnings for now
    if (onnxruntime_DEV_MODE AND NOT CMAKE_CROSSCOMPILING)
      # treat warnings as errors
      list(APPEND ORT_WARNING_FLAGS "/WX")
      foreach(type EXE STATIC SHARED)
        set(CMAKE_${type}_LINKER_FLAGS "${CMAKE_${type}_LINKER_FLAGS} /WX")
      endforeach()
    endif()

    # set linker flags to minimize the binary size.
    if (MSVC)
      foreach(type EXE STATIC SHARED)
        if (NOT type MATCHES STATIC)
          # The WinML internal toolchain does not allow link's "additional options" to contain optimization
          # flags (/OPT#); these are already specified using msbuild properties.
          if (NOT DEFINED onnxruntime_DISABLE_LINKER_OPT_FLAGS)
            set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /INCREMENTAL:NO")
            set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /INCREMENTAL:NO")
            set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /OPT:REF,ICF,LBR")
            set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /INCREMENTAL:NO")
          endif()
        endif()
        if (onnxruntime_ENABLE_LTO AND NOT onnxruntime_USE_CUDA)
          set(CMAKE_${type}_LINKER_FLAGS_RELEASE "${CMAKE_${type}_LINKER_FLAGS_RELEASE} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_${type}_LINKER_FLAGS_RELWITHDEBINFO} /LTCG")
          set(CMAKE_${type}_LINKER_FLAGS_MINSIZEREL "${CMAKE_${type}_LINKER_FLAGS_MINSIZEREL} /LTCG")
        endif()
      endforeach()
    endif()

else()
  add_definitions(-DPLATFORM_POSIX)
  check_cxx_compiler_flag(-Wunused-but-set-variable HAS_UNUSED_BUT_SET_VARIABLE)
  check_cxx_compiler_flag(-Wunused-parameter HAS_UNUSED_PARAMETER)
  check_cxx_compiler_flag(-Wunused-variable HAS_UNUSED_VARIABLE)
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  check_cxx_compiler_flag(-Wmissing-braces HAS_MISSING_BRACES)
  check_cxx_compiler_flag(-Wignored-attributes HAS_IGNORED_ATTRIBUTES)
  check_cxx_compiler_flag(-Wdeprecated-copy HAS_DEPRECATED_COPY)
  check_cxx_compiler_flag(-Wdeprecated-declarations HAS_DEPRECATED_DECLARATIONS)
  check_cxx_compiler_flag(-Wclass-memaccess HAS_CLASS_MEMACCESS)
  check_cxx_compiler_flag(-Wmaybe-uninitialized HAS_MAYBE_UNINITIALIZED)
  check_cxx_compiler_flag(-Wstrict-aliasing HAS_STRICT_ALIASING)
  check_nvcc_compiler_flag(-Wstrict-aliasing NVCC_HAS_STRICT_ALIASING)
  check_cxx_compiler_flag(-Wambiguous-reversed-operator HAS_AMBIGUOUS_REVERSED_OPERATOR)
  check_cxx_compiler_flag(-Wdeprecated-anon-enum-enum-conversion HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
  check_cxx_compiler_flag(-Wundefined-var-template HAS_UNDEFINED_VAR_TEMPLATE)

  if(HAS_TAUTOLOGICAL_POINTER_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-tautological-pointer-compare)
  endif()
  if(HAS_NONNULL_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-nonnull-compare)
  endif()
  if(HAS_PARENTHESES)
    list(APPEND ORT_WARNING_FLAGS -Wno-parentheses)
  endif()
  if(HAS_AMBIGUOUS_REVERSED_OPERATOR)
    list(APPEND ORT_WARNING_FLAGS -Wno-ambiguous-reversed-operator)
  endif()
  if(HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-anon-enum-enum-conversion)
  endif()
  if(HAS_UNDEFINED_VAR_TEMPLATE)
    list(APPEND ORT_WARNING_FLAGS -Wno-undefined-var-template)
  endif()
endif()

#names in this var must match the directory names under onnxruntime/core/providers
#ONNXRUNTIME_PROVIDER_NAMES is the list of providers that needs to export additional symbols in the global namespace. 
#For example CUDA EP exports "OrtSessionOptionsAppendExecutionProvider_CUDA", which is a global function.
#However, all these things are legacy and deprecated and should be replaced with functions in onnxruntime_c_api.h.
set(ONNXRUNTIME_PROVIDER_NAMES cpu)

set(ORT_PROVIDER_FLAGS)
set(ORT_PROVIDER_CMAKE_FLAGS)

if (onnxruntime_USE_CUDA)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_CUDA=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_CUDA=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES cuda)
endif()
if (onnxruntime_USE_VITISAI)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_VITISAI=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_VITISAI=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES vitisai)
endif()
if (onnxruntime_USE_DNNL)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_DNNL=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES dnnl)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_DNNL=1)
endif()
if (onnxruntime_USE_OPENVINO)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_OPENVINO=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES openvino)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_OPENVINO=1)
endif()
if (onnxruntime_USE_TENSORRT)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_TENSORRT=1)
    #TODO: remove the following line and change the test code in onnxruntime_shared_lib_test to use the new EP API.
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES tensorrt)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_TENSORRT=1)
endif()
if (onnxruntime_USE_RKNPU)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_RKNPU=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_RKNPU=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES rknpu)
endif()
if (onnxruntime_USE_NNAPI_BUILTIN)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_NNAPI=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_NNAPI_BUILTIN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES nnapi)
endif()
if (onnxruntime_USE_NUPHAR)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_NUPHAR=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_NUPHAR=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES nuphar)
endif()
if(onnxruntime_USE_WINML)
  list(APPEND ORT_PROVIDER_FLAGS  -DUSE_WINML=1)
  list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_WINML=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES winml)
endif()
if (onnxruntime_USE_ACL)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ACL=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_ACL=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES acl)
endif()
if (onnxruntime_USE_DML)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_DML=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_DML=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES dml)
endif()
if(onnxruntime_USE_MIGRAPHX)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_MIGRAPHX=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_MIGRAPHX=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES migraphx)
endif()

if (onnxruntime_USE_ARMNN)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ARMNN=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_ARMNN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES armnn)
endif()
if (onnxruntime_USE_ROCM)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ROCM=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_ROCM=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES rocm)
endif()
if (onnxruntime_USE_COREML)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_COREML=1)
    list(APPEND ORT_PROVIDER_CMAKE_FLAGS -Donnxruntime_USE_COREML=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES coreml)
endif()
function(onnxruntime_set_compile_flags target_name)
    if (MSVC)
      target_compile_definitions(${target_name} PUBLIC -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
      if(onnxruntime_ENABLE_MEMLEAK_CHECKER)
        target_compile_definitions(${target_name} PUBLIC -DONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
      endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/sdl>")
      set_target_properties(${target_name}
                      PROPERTIES VS_GLOBAL_CAExcludePath "${ORT_BINARY_DIR};${ORT_SOURCE_DIR}")
      if (onnxruntime_ENABLE_STATIC_ANALYSIS)
        target_compile_options(${target_name} PRIVATE  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/analyze:stacksize 131072>")
        target_compile_options(${target_name} PRIVATE  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/analyze:external->")
      endif()
    else()
      # Enable warning
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall>")
      target_compile_options(${target_name} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wextra>")
      if(onnxruntime_DEV_MODE)
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Werror>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Werror>")
      endif()

      target_compile_definitions(${target_name} PUBLIC -DNSYNC_ATOMIC_CPP11)
      if(APPLE AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "iOSCross" AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "iOS")
        target_compile_definitions(${target_name} PUBLIC -Doptional_CONFIG_SELECT_OPTIONAL=optional_OPTIONAL_NONSTD)
      endif()
      target_include_directories(${target_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/external/nsync/public")
    endif()
    foreach(ORT_FLAG ${ORT_PROVIDER_FLAGS})
      target_compile_definitions(${target_name} PRIVATE ${ORT_FLAG})
    endforeach()
    if(HAS_DEPRECATED_COPY)
      #too many such errors in eigen
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-deprecated-copy>" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
    endif()
    if(onnxruntime_USE_CUDA)
      if((NVCC_HAS_STRICT_ALIASING AND "${target_name}" MATCHES "cuda") OR (HAS_STRICT_ALIASING AND NOT "${target_name}" MATCHES "cuda"))
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-strict-aliasing>")
      endif()
    endif()
    foreach(ORT_FLAG ${ORT_WARNING_FLAGS})
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options ${ORT_FLAG}>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:${ORT_FLAG}>")
    endforeach()
endfunction()

function(onnxruntime_set_source_file_properties target_name)
  get_target_property(srcs ${target_name} SOURCES)

  # enable ARC for Objective-C/C++
  set(objective_c_cc_srcs ${srcs})
  list(FILTER objective_c_cc_srcs INCLUDE REGEX "\\.mm?$")
  set_property(SOURCE ${objective_c_cc_srcs} APPEND PROPERTY COMPILE_OPTIONS "-fobjc-arc")
endfunction()

function(onnxruntime_configure_target target_name)
  target_link_directories(${target_name} PRIVATE ${onnxruntime_LINK_DIRS})
  onnxruntime_set_compile_flags(${target_name})
  onnxruntime_set_source_file_properties(${target_name})
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  if(onnxruntime_ENABLE_LTO)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
  endif()
endfunction()

function(onnxruntime_add_shared_library target_name)
  add_library(${target_name} SHARED ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

function(onnxruntime_add_static_library target_name)
  add_library(${target_name} ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

#For plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality.
function(onnxruntime_add_shared_library_module target_name)
  if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOSCross") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOS"))
    add_library(${target_name} SHARED ${ARGN})
  else()
    #On Windows, this target shouldn't generate an import lib, but I don't know how to disable it.
    add_library(${target_name} MODULE ${ARGN})
  endif()

  onnxruntime_configure_target(${target_name})
endfunction()

function(onnxruntime_add_executable target_name)
  if(${CMAKE_SYSTEM_NAME} MATCHES "iOSCross")
    message(FATAL_ERROR "iOS doesn't support commmand line tool")
  endif()
  add_executable(${target_name} ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
    endforeach()
endfunction()

set(onnxruntime_EXTERNAL_DEPENDENCIES onnx_proto)

# ACL
if (onnxruntime_USE_ACL OR onnxruntime_USE_ACL_1902 OR onnxruntime_USE_ACL_1905 OR onnxruntime_USE_ACL_1908 OR onnxruntime_USE_ACL_2002)
  set(onnxruntime_USE_ACL ON)
  if(onnxruntime_USE_ACL_1902)
    add_definitions(-DACL_1902=1)
  else()
    if(onnxruntime_USE_ACL_1908)
      add_definitions(-DACL_1908=1)
    else()
      if(onnxruntime_USE_ACL_2002)
        add_definitions(-DACL_2002=1)
      else()
        add_definitions(-DACL_1905=1)
      endif()
    endif()
  endif()

  if (NOT ${onnxruntime_ACL_LIBS} STREQUAL "")
    add_library(arm_compute SHARED IMPORTED)
    set_target_properties(arm_compute PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

    add_library(arm_compute_core SHARED IMPORTED)
    set_target_properties(arm_compute_core PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_core.so")

    add_library(arm_compute_graph SHARED IMPORTED)
    set_target_properties(arm_compute_graph PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_graph.so")
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_core arm_compute_graph)

endif()

# ArmNN
if (onnxruntime_USE_ARMNN)
  if (NOT onnxruntime_ARMNN_RELU_USE_CPU)
    add_definitions(-DRELU_ARMNN=1)
  endif()
  if (NOT onnxruntime_ARMNN_BN_USE_CPU)
    add_definitions(-DBN_ARMNN=1)
  endif()

  if (NOT onnxruntime_USE_ACL AND NOT ${onnxruntime_ACL_LIBS} STREQUAL "")
    add_library(arm_compute SHARED IMPORTED)
    set_target_properties(arm_compute PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

    add_library(arm_compute_core SHARED IMPORTED)
    set_target_properties(arm_compute_core PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_core.so")

    add_library(arm_compute_graph SHARED IMPORTED)
    set_target_properties(arm_compute_graph PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_graph.so")
  endif()

  if (NOT ${onnxruntime_ARMNN_LIBS} STREQUAL "")
    add_library(armnn SHARED IMPORTED)
    set_target_properties(armnn PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ARMNN_LIBS}/libarmnn.so")
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES armnn arm_compute arm_compute_core arm_compute_graph)
endif()

if (onnxruntime_USE_DNNL)
  include(dnnl)
endif()

# TVM
if (onnxruntime_USE_TVM)
  if (onnxruntime_USE_CUDA)
    set(USE_CUDA ON)
  endif()
  if (onnxruntime_USE_LLVM)
    set(USE_LLVM ON)
    add_definitions(-DUSE_TVM_WITH_LLVM)
  endif()
  if (onnxruntime_USE_OPENMP)
    set(USE_OPENMP "gnu")
  endif()

  add_subdirectory(${PROJECT_SOURCE_DIR}/external/tvm EXCLUDE_FROM_ALL)
  set_target_properties(tvm PROPERTIES FOLDER "External/tvm")
  set_target_properties(tvm_topi PROPERTIES FOLDER "External/tvm")
  set_target_properties(tvm_runtime PROPERTIES FOLDER "External/tvm")
  set_target_properties(nnvm_compiler PROPERTIES FOLDER "External/tvm")

  set(TVM_INCLUDES ${PROJECT_SOURCE_DIR}/external/tvm/include
    ${PROJECT_SOURCE_DIR}/external/tvm/3rdparty/dmlc-core/include
    ${PROJECT_SOURCE_DIR}/external/tvm/3rdparty/dlpack/include
    $<TARGET_PROPERTY:tvm,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:tvm_topi,INTERFACE_INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:nnvm_compiler,INTERFACE_INCLUDE_DIRECTORIES>)
  add_definitions(-DUSE_TVM)

  set(onnxruntime_tvm_libs onnxruntime_codegen_tvm)
  # needs to link with stdc++fs in Linux
  if(UNIX)
    if (NOT APPLE)
      set(FS_STDLIB stdc++fs)
    endif()
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tvm nnvm_compiler ${FS_STDLIB})
  list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES tvm nnvm_compiler)
endif()

if (onnxruntime_ENABLE_EXTENSION_CUSTOM_OPS)
  # add compile definition to enable custom operators in onnxruntime-extensions
  add_compile_definitions(ENABLE_EXTENSION_CUSTOM_OPS)

  # set options for onnxruntime-extensions
  set(OCOS_ENABLE_CTEST OFF CACHE INTERNAL "")
  set(OCOS_ENABLE_STATIC_LIB ON CACHE INTERNAL "")
  set(OCOS_ENABLE_SPM_TOKENIZER OFF CACHE INTERNAL "")
  add_subdirectory(external/onnxruntime-extensions EXCLUDE_FROM_ALL)
  # target library or executable are defined in CMakeLists.txt of onnxruntime-extensions
  target_include_directories(ocos_operators PRIVATE ${RE2_INCLUDE_DIR} external/json/include)
  target_include_directories(ortcustomops PUBLIC external/onnxruntime-extensions/shared)
endif()

if (APPLE OR CMAKE_SYSTEM_NAME STREQUAL "Android")
  #onnx/onnx/proto_utils.h:34:16: error: 'SetTotalBytesLimit' is deprecated: Please use the single
  #parameter version of SetTotalBytesLimit(). The second parameter is ignored.
  #  coded_stream.SetTotalBytesLimit((2048LL << 20) - 1, 512LL << 20);
  #TODO: fix the warning in ONNX and re-enable this flag
  string(APPEND CMAKE_CXX_FLAGS " -Wno-deprecated")
  string(APPEND CMAKE_C_FLAGS " -Wno-deprecated")
endif()

# ONNX
if(NOT onnxruntime_USE_FULL_PROTOBUF)
  set(ONNX_USE_LITE_PROTO ON CACHE BOOL "" FORCE)
else()
  set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)
endif()

if (NOT onnxruntime_MINIMAL_BUILD)
  add_subdirectory(external/onnx EXCLUDE_FROM_ALL)
else()
  include(onnx_minimal)
endif()

target_compile_definitions(onnx PUBLIC $<TARGET_PROPERTY:onnx_proto,INTERFACE_COMPILE_DEFINITIONS> PRIVATE "__ONNX_DISABLE_STATIC_REGISTRATION")
if (NOT onnxruntime_USE_FULL_PROTOBUF)
  target_compile_definitions(onnx PUBLIC "__ONNX_NO_DOC_STRINGS")
endif()
set_target_properties(onnx PROPERTIES FOLDER "External/ONNX")
set_target_properties(onnx_proto PROPERTIES FOLDER "External/ONNX")


# fix a warning in onnx code we can't do anything about
if (MSVC)
   string(APPEND CMAKE_CXX_FLAGS " -DEIGEN_HAS_C99_MATH") # required to be set explicitly to enable Eigen-Unsupported SpecialFunctions
endif()

#Dependencies end. In the next we'll enable "treat warning as error"

if (onnxruntime_RUN_ONNX_TESTS)
  add_definitions(-DORT_RUN_EXTERNAL_ONNX_TESTS)
endif()

# Flatbuffers
# We do not need to build flatc for iOS or Android Cross Compile
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR onnxruntime_BUILD_WEBASSEMBLY)
  set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATC" FORCE)
endif()
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "FLATBUFFERS_BUILD_TESTS" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "FLATBUFFERS_INSTALL" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATHASH" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "FLATBUFFERS_BUILD_FLATLIB" FORCE)
set_msvc_c_cpp_compiler_warning_level(4)
add_subdirectory(external/flatbuffers EXCLUDE_FROM_ALL)
set_msvc_c_cpp_compiler_warning_level(3)
list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES flatbuffers)
list(APPEND onnxruntime_EXTERNAL_LIBRARIES flatbuffers)

if (CMAKE_SYSTEM_NAME STREQUAL "Android" AND Onnxruntime_GCOV_COVERAGE)
  string(APPEND CMAKE_CXX_FLAGS " -g -O0 --coverage ")
  string(APPEND CMAKE_C_FLAGS   " -g -O0 --coverage ")
endif()

#Adjust warning flags
if(onnxruntime_USE_CUDA)
  set_msvc_c_cpp_compiler_warning_level(3)
else()
  set_msvc_c_cpp_compiler_warning_level(4)
endif()

set(onnxruntime_DELAYLOAD_FLAGS "")

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${REPO_ROOT}/include/onnxruntime/core/session
)

if(onnxruntime_USE_OPENVINO)

  add_definitions(-DUSE_OPENVINO=1)

  if (EXISTS "$ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/version.txt")
    file(READ $ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/version.txt VER)
  endif()

  if (NOT DEFINED ENV{INTEL_OPENVINO_DIR})
    message(FATAL_ERROR "[Couldn't locate OpenVINO] OpenVINO may not have been initialized")
  endif()

  # Check OpenVINO version for support
  if (${VER} MATCHES "2020.3" OR $ENV{INTEL_OPENVINO_DIR} MATCHES "2020.3")
    set(OPENVINO_VERSION "2020.3")
    add_definitions(-DOPENVINO_2020_3=1)
  elseif (${VER} MATCHES "2021.1" OR $ENV{INTEL_OPENVINO_DIR} MATCHES "2021.1")
    set(OPENVINO_VERSION "2021.1")
    add_definitions(-DOPENVINO_2021_1=1)
  elseif (${VER} MATCHES "2021.2" OR $ENV{INTEL_OPENVINO_DIR} MATCHES "2021.2")
    set(OPENVINO_VERSION "2021.2")
    add_definitions(-DOPENVINO_2021_2=1)
  elseif (${VER} MATCHES "2021.3" OR $ENV{INTEL_OPENVINO_DIR} MATCHES "2021.3")
    set(OPENVINO_VERSION "2021.3")
    add_definitions(-DOPENVINO_2021_3=1)
  elseif (${VER} MATCHES "2021.4" OR $ENV{INTEL_OPENVINO_DIR} MATCHES "2021.4")
    set(OPENVINO_VERSION "2021.4")
    add_definitions(-DOPENVINO_2021_4=1)
  elseif ($ENV{INTEL_OPENVINO_DIR} MATCHES "openvino")
    set(OPENVINO_VERSION "2021.3")
    add_definitions(-DOPENVINO_2021_3=1)
    add_definitions(-DOPENVINO_2021_4=1)
  else()
    message(FATAL_ERROR "Unsupported OpenVINO version: ${INTEL_OPENVINO_DIR}")
  endif()

  if(onnxruntime_USE_OPENVINO_MYRIAD)
    add_definitions(-DOPENVINO_CONFIG_MYRIAD=1)
  endif()

  if(onnxruntime_USE_OPENVINO_GPU_FP32)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP32=1)
  endif()

  if(onnxruntime_USE_OPENVINO_GPU_FP16)
    add_definitions(-DOPENVINO_CONFIG_GPU_FP16=1)
  endif()

  if(onnxruntime_USE_OPENVINO_CPU_FP32)
    add_definitions(-DOPENVINO_CONFIG_CPU_FP32=1)
  endif()

  if(onnxruntime_USE_OPENVINO_VAD_M)
    add_definitions(-DOPENVINO_CONFIG_VAD_M=1)
  endif()

  if(onnxruntime_USE_OPENVINO_VAD_F)
    add_definitions(-DOPENVINO_CONFIG_VAD_F=1)
  endif()

  if(onnxruntime_USE_OPENVINO_HETERO)
    add_definitions(-DOPENVINO_CONFIG_HETERO=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if(onnxruntime_USE_OPENVINO_MULTI)
    add_definitions(-DOPENVINO_CONFIG_MULTI=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

endif()

if (onnxruntime_USE_VITISAI)
  if(WIN32)
    message(FATAL_ERROR "Vitis-AI execution provider is not supported on Windows.")
  else()
    include(pyxir)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES pyxir)
    list(APPEND onnxruntime_EXTERNAL_DEPENDENCIES pyxir)
  endif()
endif()

configure_file(onnxruntime_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h)
if(WIN32)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/Debug/requirements.txt)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/Release/requirements.txt)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/RelWithDebInfo/requirements.txt)
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/MinSizeRel/requirements.txt)
else()
  configure_file(../requirements.txt.in ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt)
endif()

if (onnxruntime_USE_CUDA)
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  enable_language(CUDA)
  message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")
  if (WIN32)
    set(CMAKE_CUDA_STANDARD 17)
    foreach(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
      string(APPEND CMAKE_CXX_FLAGS " /external:I\"${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY}\"")
    endforeach()
  else()
    #CUDA 10.2 on Linux doesn't support C++17
    set(CMAKE_CUDA_STANDARD 14)
  endif()
  file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
  set(ONNXRUNTIME_CUDA_LIBRARIES ${CUDA_LIBRARIES})

  if (onnxruntime_ENABLE_NVTX_PROFILE)
    list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cublas cudnn curand cufft nvToolsExt)
  else()
    list(APPEND ONNXRUNTIME_CUDA_LIBRARIES cublas cudnn curand cufft)
  endif()

  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
      # Support for Jetson/Tegra ARM devices
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_53,code=sm_53") # TX1, Nano
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_62,code=sm_62") # TX2
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_72,code=sm_72") # AGX Xavier, NX Xavier
    else()
      # the following compute capabilities are removed in CUDA 11 Toolkit
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_30,code=sm_30") # K series
      endif()
      # 37, 50 still work in CUDA 11 but are marked deprecated and will be removed in future CUDA version.
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_37,code=sm_37") # K80
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_50,code=sm_50") # M series

      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_52,code=sm_52") # M60
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_60,code=sm_60") # P series
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_70,code=sm_70") # V series
      if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_80,code=sm_80") # A series
      endif()
    endif()
  endif()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror default-stream-launch")
  endif()
  if (NOT WIN32)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --compiler-options -fPIC")
  endif()
  # Options passed to cudafe
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=bad_friend_decl\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=unsigned_compare_with_zero\"")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe \"--diag_suppress=expr_has_no_effect\"")

  if (onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>")
  endif()
endif()

if (onnxruntime_USE_MIGRAPHX)
  if (WIN32)
    message(FATAL_ERROR "MIGraphX does not support build in Windows!")
  endif()
  set(AMD_MIGRAPHX_HOME ${onnxruntime_MIGRAPHX_HOME})
endif()

if (onnxruntime_USE_ROCM)
  if (WIN32)
    message(FATAL_ERROR "ROCM does not support build in Windows!")
  endif()
  if (onnxruntime_USE_CUDA)
    message(FATAL_ERROR "ROCM does not support build with CUDA!")
  endif()
endif()

if (onnxruntime_USE_TVM)
  if (WIN32 AND MSVC)
    # wd4100: identifier' : unreferenced formal parameter
    # wd4244: conversion from 'int' to 'char', possible loss of data
    # wd4251: class X needs to have dll-interface to be used by clients of class Y
    # wd4267: 'initializing': conversion from 'size_t' to 'int', possible loss of data
    # wd4275: non dll-interface class X used as base for dll-interface class Y
    # wd4389: signed/unsigned mismatch
    # wd4456: declaration of X hides previous local declaration
    set(DISABLED_WARNINGS_FOR_TVM "/wd4100" "/wd4244" "/wd4251" "/wd4267" "/wd4275" "/wd4389" "/wd4456")
  else()
    set(DISABLED_WARNINGS_FOR_TVM "-Wno-error=ignored-qualifiers" "-Wno-unused-parameter" "-Wno-ignored-qualifiers")
    if(HAS_UNUSED_PARAMETER)
      list(APPEND DISABLED_WARNINGS_FOR_TVM "-Wno-error=unused-parameter")
    endif()
    if(HAS_CATCH_VALUE)
      #TODO: send a PR to TVM and fix it
      list(APPEND DISABLED_WARNINGS_FOR_TVM "-Wno-error=catch-value")
    endif()
  endif()
  include(onnxruntime_codegen.cmake)
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  add_definitions(-DMICROSOFT_INTERNAL)
endif()

if (onnxruntime_USE_DML)
  if(NOT WIN32)
    message(FATAL_ERROR "The DirectML execution provider is only supported when building for Windows.")
  endif()

  include(dml)
endif()

if (onnxruntime_ENABLE_TRAINING_OPS)
  add_compile_definitions(ENABLE_TRAINING_OPS)
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_compile_definitions(ENABLE_TRAINING)
  add_compile_definitions(ENABLE_TRAINING_OPS)

  if (UNIX)
    if (EXISTS "${onnxruntime_MPI_HOME}")
      set(MPI_HOME "${onnxruntime_MPI_HOME}")
    elseif (EXISTS "/bert_ort/openmpi")
      set(MPI_HOME "/bert_ort/openmpi")
    endif()

    find_package(MPI)

    if (MPI_CXX_FOUND)
      message( STATUS "MPI Version: ${MPI_CXX_VERSION}")
      message( STATUS "MPI (include: ${MPI_CXX_INCLUDE_DIRS}, library: ${MPI_CXX_LIBRARIES})" )
      mark_as_advanced(MPI_CXX_INCLUDE_DIRS MPI_CXX_LIBRARIES)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${MPI_CXX_LIBRARIES} ${MPI_CXX_LINK_FLAGS})
    else ()
      set(onnxruntime_USE_NCCL OFF)
      set(onnxruntime_USE_MPI OFF)
      message( WARNING "MPI is not found. Please define onnxruntime_MPI_HOME to specify the path of MPI. Otherwise, NCCL will be disabled." )
    endif()

    # Find NCCL and MPI
    if (onnxruntime_USE_NCCL AND MPI_CXX_FOUND)
      if (onnxruntime_USE_CUDA)
        set(NCCL_LIBNAME "nccl")
      elseif (onnxruntime_USE_ROCM)
        set(NCCL_LIBNAME "rccl")
      endif()

      find_path(NCCL_INCLUDE_DIR
        NAMES ${NCCL_LIBNAME}.h
        HINTS
        ${onnxruntime_NCCL_HOME}/include
        $ENV{CUDA_ROOT}/include)

      find_library(NCCL_LIBRARY
        NAMES ${NCCL_LIBNAME}
        HINTS
        ${onnxruntime_NCCL_HOME}/lib/x86_64-linux-gnu
        ${onnxruntime_NCCL_HOME}/lib
        $ENV{CUDA_ROOT}/lib64)

      include(FindPackageHandleStandardArgs)
      find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

      if (NCCL_FOUND)
        set(NCCL_HEADER_FILE "${NCCL_INCLUDE_DIR}/${NCCL_LIBNAME}.h")
        message( STATUS "Determining NCCL version from the header file: ${NCCL_HEADER_FILE}" )
        file (STRINGS ${NCCL_HEADER_FILE} NCCL_MAJOR_VERSION_DEFINED
              REGEX "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
        if (NCCL_MAJOR_VERSION_DEFINED)
          string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MAJOR[ \t]+" ""
                  NCCL_MAJOR_VERSION ${NCCL_MAJOR_VERSION_DEFINED})
          message( STATUS "NCCL_MAJOR_VERSION: ${NCCL_MAJOR_VERSION}" )
        endif()
        file (STRINGS ${NCCL_HEADER_FILE} NCCL_MINOR_VERSION_DEFINED
              REGEX "^[ \t]*#define[ \t]+NCCL_MINOR[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
        if (NCCL_MINOR_VERSION_DEFINED)
          string (REGEX REPLACE "^[ \t]*#define[ \t]+NCCL_MINOR[ \t]+" ""
                  NCCL_MINOR_VERSION ${NCCL_MINOR_VERSION_DEFINED})
          message(STATUS "NCCL_MINOR_VERSION: ${NCCL_MINOR_VERSION}")
        endif()

        if (NCCL_MAJOR_VERSION_DEFINED AND NCCL_MINOR_VERSION_DEFINED)
          if ("${NCCL_MAJOR_VERSION}.${NCCL_MINOR_VERSION}" VERSION_GREATER_EQUAL "2.7")
            add_definitions(-DUSE_NCCL_P2P=1)
            message( STATUS "NCCL P2P is enabled for supporting ncclSend and ncclRecv." )
          endif()
        endif()

        set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
        set(NCCL_LIBRARIES ${NCCL_LIBRARY})
        message( STATUS "NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})" )
        mark_as_advanced(NCCL_INCLUDE_DIRS NCCL_LIBRARIES)

        list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${NCCL_LIBRARIES})

        add_definitions(-DORT_USE_NCCL=1)
        message( STATUS "NCCL is enabled in Linux GPU Build." )
      else ()
        set(onnxruntime_USE_NCCL OFF)
        message( WARNING "NCCL is not found. Please use --nccl_home to specify the path of NCCL. Otherwise, NCCL is disabled." )
      endif()
    endif()
  endif()

  if (onnxruntime_USE_MPI AND MPI_CXX_FOUND)
    add_definitions(-DUSE_MPI=1)
  endif()

  add_subdirectory(tensorboard EXCLUDE_FROM_ALL)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tensorboard)
endif()

#names in this var must match the directory names under onnxruntime/core/providers
set(ONNXRUNTIME_TARGETS onnxruntime_common onnxruntime_graph onnxruntime_framework onnxruntime_util onnxruntime_providers onnxruntime_optimizer onnxruntime_session onnxruntime_mlas onnxruntime_flatbuffers)
if(onnxruntime_ENABLE_EAGER_MODE)
  list(APPEND ONNXRUNTIME_TARGETS onnxruntime_eager)
endif()
foreach(target_name ${ONNXRUNTIME_TARGETS})
  include(${target_name}.cmake)
endforeach()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES log)
endif()

if (onnxruntime_USE_OPENMP)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES OpenMP::OpenMP_CXX)
endif()

if(WIN32)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SYS_PATH_LIB} Shlwapi)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
else()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES nsync_cpp)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (WIN32)
  if (WINDOWS_STORE)
    # Setting WINAPI_FAMILY, WINVER and _WIN32_WINNT restrict the APIs exposed in Windows headers to those available
    # in store or desktop, and that support at least the version of Windows specified
    # add_compile_definitions(WINAPI_FAMILY=2)
    add_compile_definitions(WINVER=0x0A00 _WIN32_WINNT=0x0A00)  # Support Windows 10
    # /ZW adds /FUplatform.winmd and /FUwindows.winmd. windows.winmd, in turn, overrides the include path for
    # the cppwinrt headers, which we set to use the WinML built ones.
    # Instead, we use /ZW:nostdlib and force include platform.winml only.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:SHELL:/ZW:nostdlib /FUplatform.winmd>")
    get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
    link_directories("${msvc_path}/lib/${onnxruntime_target_platform}/store")
    add_link_options(/APPCONTAINER)
  else()
    add_compile_definitions(WINVER=0x0601)  # Support Windows 7 and newer
    # add_compile_definitions(WINAPI_FAMILY=3)
    # FIXME adding _WIN32_WINNT=0x0601 and WINAPI_FAMILY is desirable. However, it hides some symbols that are used in WRL
    # headers and breaks the build of WinML. We should have separate WINVER/_WIN32_WINNT definitions for WinML
  endif()
endif()

# Default version parts for Microsoft.AI.MachineLearning.dll and onnxruntime.dll in non-ADO pipeline local builds
set(VERSION_MAJOR_PART   0 CACHE STRING "First part of numeric file/product version.")
set(VERSION_MINOR_PART   0 CACHE STRING "Second part of numeric file/product version.")
set(VERSION_BUILD_PART       0 CACHE STRING "Third part of numeric file/product version.")
set(VERSION_PRIVATE_PART     0 CACHE STRING "Fourth part of numeric file/product version.")
set(VERSION_STRING       "Internal Build" CACHE STRING "String representation of file/product version.")
include(wil.cmake)

if (onnxruntime_USE_WINML)
  # WINML uses and depends on the shared lib.  Note:  You can build WINML without DML and you will get a
  # CPU only WINML
  if (NOT onnxruntime_BUILD_SHARED_LIB)
    message(
        FATAL_ERROR
        "Option onnxruntime_USE_WINML can only be used when onnxruntime_BUILD_SHARED_LIB is also enabled")
  endif()
  include(winml.cmake)
endif() # if(onnxruntime_USE_WINML)


#The following files may use the 'onnxruntime_libs' and 'onnxruntime_EXTERNAL_LIBRARIES' vars

if (onnxruntime_BUILD_SHARED_LIB OR onnxruntime_BUILD_APPLE_FRAMEWORK)
  if (onnxruntime_BUILD_APPLE_FRAMEWORK AND NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin|iOS")
    message(FATAL_ERROR "onnxruntime_BUILD_APPLE_FRAMEWORK can only be enabled for macOS or iOS.")
  endif()
  include(onnxruntime.cmake)
endif()

if (onnxruntime_BUILD_JAVA)
  message(STATUS "Java Build is enabled")
  include(onnxruntime_java.cmake)
endif()

if (onnxruntime_BUILD_NODEJS)
  message(STATUS "Node.js Build is enabled")
  include(onnxruntime_nodejs.cmake)
endif()

if (onnxruntime_ENABLE_PYTHON)
  include(onnxruntime_python.cmake)
endif()

if (onnxruntime_BUILD_OBJC)
  message(STATUS "Objective-C Build is enabled")
  include(onnxruntime_objectivec.cmake)
endif()

if (onnxruntime_BUILD_UNIT_TESTS)
  include(onnxruntime_unittests.cmake)
endif()

if (onnxruntime_ENABLE_TRAINING)
  include(onnxruntime_training.cmake)
  if (onnxruntime_ENABLE_TRAINING_E2E_TESTS)
    include(onnxruntime_training_e2e_tests.cmake)
  endif()
endif()

if (onnxruntime_BUILD_CSHARP)
  message(STATUS "CSharp Build is enabled")
#  set_property(GLOBAL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "netstandard2.0")
  include(onnxruntime_csharp.cmake)
endif()

if (onnxruntime_BUILD_WEBASSEMBLY)
  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    string(APPEND CMAKE_CXX_FLAGS " -msimd128")
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING)
    string(APPEND CMAKE_CXX_FLAGS " -s DISABLE_EXCEPTION_CATCHING=0")
  endif()

  if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
    string(APPEND CMAKE_CXX_FLAGS " -s USE_PTHREADS=1 -Wno-pthreads-mem-growth")
  endif()

  message(STATUS "WebAssembly Build is enabled")
  include(onnxruntime_webassembly.cmake)
endif()

if (WINDOWS_STORE)
  target_link_options(onnxruntime PRIVATE /DYNAMICBASE /NXCOMPAT /APPCONTAINER)
  target_link_options(winml_dll PRIVATE /DYNAMICBASE /NXCOMPAT /APPCONTAINER)

  if (onnxruntime_target_platform STREQUAL "x86")
    target_link_options(onnxruntime PRIVATE /SAFESEH)
    target_link_options(winml_dll PRIVATE /SAFESEH)
  endif()
endif()

include(flake8.cmake)

if(UNIX)
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" ON)
else()
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" OFF)
endif()
if(BUILD_PKGCONFIG_FILES)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/pkgconfig )
endif()

if (onnxruntime_BUILD_OPSCHEMA_LIB AND onnxruntime_ENABLE_TRAINING)
  # opschema library requires training ops as well
  include(onnxruntime_opschema_lib.cmake)
endif()

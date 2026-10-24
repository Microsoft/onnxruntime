# Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Licensed under the MIT License.

# Minimum CMake required
cmake_minimum_required(VERSION 3.28)

# Don't let cmake set a default value for CMAKE_CUDA_ARCHITECTURES
cmake_policy(SET CMP0104 OLD)
# Project
project(onnxruntime C CXX ASM)

include(onnxruntime_language_standard_versions.cmake)

# We don't use C++20 modules yet.
# There are some known issues to address first:
# - Android builds from Linux Docker containers have trouble finding clang-scan-deps.
# - The MSVC /permissive option is needed for compiling some of the CUDA EP code which uses CUTLASS.
#   This option is not compatible with C++20 modules.
# So we will skip module scanning for now.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Disable fast-math for Intel oneAPI compiler
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "IntelLLVM")
  if("${CMAKE_CXX_COMPILER_ID}" MATCHES "MSVC-like")
    # Using icx-cl compiler driver with MSVC-like arguments
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:precise")
  else()
    # Using icpx compiler driver
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")
  endif()
endif()

include(CheckCXXCompilerFlag)
include(CheckLanguage)
include(CMakeDependentOption)
include(FetchContent)
include(CheckFunctionExists)
include(CheckSymbolExists)
include(GNUInstallDirs) # onnxruntime_providers_* require CMAKE_INSTALL_* variables

if (MSVC)
  #  Make sure Visual Studio sets __cplusplus macro correctly: https://learn.microsoft.com/en-us/cpp/build/reference/zc-cplusplus
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")

  # Prevents CMake from injecting '#pragma system_header', which results in warnings being disabled in projects that
  # use precompiled headers.
  set(CMAKE_PCH_PROLOGUE "")
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
# NOTE: POSITION INDEPENDENT CODE hurts performance, and it only make sense on POSIX systems
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()
if (NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - using RelWithDebInfo")
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Choose build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()

# Options
option(onnxruntime_USE_VCPKG "Build with the vcpkg package manager" OFF)
option(onnxruntime_RUN_ONNX_TESTS "Enable ONNX Compatibility Testing" OFF)
option(onnxruntime_GENERATE_TEST_REPORTS "Enable test report generation" OFF)
option(onnxruntime_ENABLE_STATIC_ANALYSIS "Enable static analysis" OFF)
option(onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE "Use a custom SDL Rule. It is mainly for our CI build" OFF)
option(onnxruntime_ENABLE_PYTHON "Enable python bindings" OFF)
# Enable it may cause LNK1169 error
option(onnxruntime_ENABLE_MEMLEAK_CHECKER "Experimental: Enable memory leak checker in Windows debug build" OFF)
option(onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER "Experimental: Enable ConvSymKernelAvx2 assembly saturation checker in build" OFF)
option(onnxruntime_USE_CUDA "Build with CUDA support" OFF)
cmake_dependent_option(onnxruntime_USE_CUDA_NHWC_OPS "Build CUDA with NHWC op support" ON "onnxruntime_USE_CUDA" OFF)
cmake_dependent_option(onnxruntime_BUILD_CUDA_EP_AS_PLUGIN "Build CUDA EP as a separate plugin shared library instead of the legacy in-tree provider" OFF "onnxruntime_USE_CUDA" OFF)
option(onnxruntime_BUILD_CUDA_QUANT_PREPROCESS "Build CUDA weight-packing module onnxruntime_cuda_quant_preprocess.so" OFF)
option(onnxruntime_CUDA_MINIMAL "Build CUDA without any operations apart from memcpy ops. Useful for a very minimal TRT build" OFF)
option(onnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO "When building with CUDA support, generate device code line number information." OFF)
option(onnxruntime_USE_OPENVINO "Build with OpenVINO support" OFF)
option(onnxruntime_USE_COREML "Build with CoreML support" OFF)
option(onnxruntime_USE_NNAPI_BUILTIN "Build with builtin NNAPI lib for Android NNAPI support" OFF)
option(onnxruntime_USE_QNN "Build with QNN support" OFF)
option(onnxruntime_BUILD_QNN_EP_STATIC_LIB "Build with QNN EP as a static library" OFF)
option(onnxruntime_USE_SNPE "Build with SNPE support" OFF)
option(onnxruntime_USE_RKNPU "Build with RKNPU support" OFF)
option(onnxruntime_USE_DNNL "Build with DNNL support" OFF)
option(onnxruntime_USE_JSEP "Build with JavaScript implemented kernels support" OFF)
option(onnxruntime_USE_SVE "Build with SVE support in MLAS" OFF)
option(onnxruntime_USE_RVV "Build with RISC-V Vector support in MLAS" OFF)
option(onnxruntime_USE_RVV_ZVFH "Build with RISC-V Zvfh (FP16 vector) support in MLAS" OFF)
option(onnxruntime_USE_ARM_NEON_NCHWC "Build with ARM Neon NCHWc kernels in MLAS" OFF)

option(onnxruntime_USE_KLEIDIAI "Build with KleidiAI integration in MLAS" OFF)
option(onnxruntime_USE_QMX_KLEIDIAI_COEXIST "Build with QMX and Arm KLEIDIAI libraries" OFF)
option(onnxruntime_BUILD_UNIT_TESTS "Build ONNXRuntime unit tests" ON)
option(onnxruntime_RUN_MLAS_TESTS "Run a targeted subset of MLAS unit tests with CTest" OFF)
# Declare both prerequisites before evaluating this dependent option on a fresh configure.
# Enable ONNX Runtime CUDA EP's internal unit tests that directly access the EP's internal functions instead of through
# OpKernels. When the option is ON, we will have two copies of GTest library in the same process. It is not a typical
# use. If you hit any problem with that, please do not report it to GTest. Turn OFF the following build option instead.
cmake_dependent_option(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS "Build with CUDA unit tests" OFF "onnxruntime_USE_CUDA;onnxruntime_BUILD_UNIT_TESTS" OFF)

# Materialize the ONNX node-test corpus from ONNX's Python generators into the build tree at
# configure/build time, instead of depending on the on-disk corpus shipped in the ONNX source
# archive. This detaches ORT from ONNX PR #7959 (which deletes onnx/backend/test/data/node).
#
# Defaults OFF -- it is an explicit opt-in, enabled only on CI legs that (a) install the pinned
# onnx + numpy pre-build and (b) actually consume the materialized node corpus (the QNN legs and
# one Linux CPU leg that runs the equivalence + materialized-node ctests), via
# `--cmake_extra_defines onnxruntime_MATERIALIZE_ONNX_NODE_TESTS=ON`. Legs that leave it OFF keep
# their pre-existing behavior (node tests are read from the FetchContent'd ONNX source tree, which
# still ships data/node on the pinned version), so the default is a zero-regression no-op today.
#
# When explicitly set ON, the configure step FAILS LOUDLY if onnx/numpy are missing or off-pin
# (it does NOT silently skip) -- the loud contract is preserved exactly where materialization is
# requested. On reduced/minimal builds the dependent condition forces it OFF regardless.
cmake_dependent_option(onnxruntime_MATERIALIZE_ONNX_NODE_TESTS
  "Materialize the ONNX node-test corpus into the build tree (opt-in; needs onnx+numpy)." OFF
  "onnxruntime_BUILD_UNIT_TESTS;NOT onnxruntime_REDUCED_OPS_BUILD;NOT onnxruntime_MINIMAL_BUILD" OFF)
option(onnxruntime_BUILD_CSHARP "Build C# library" OFF)
option(onnxruntime_BUILD_OBJC "Build Objective-C library" OFF)
option(onnxruntime_USE_PREINSTALLED_EIGEN "Use pre-installed EIGEN. Need to provide eigen_SOURCE_PATH if turn this on." OFF)
option(onnxruntime_BUILD_BENCHMARKS "Build ONNXRuntime micro-benchmarks" OFF)
option(onnxruntime_USE_VSINPU "Build with VSINPU support" OFF)

cmake_dependent_option(onnxruntime_USE_FLASH_ATTENTION "Build flash attention kernel for scaled dot product attention" ON "onnxruntime_USE_CUDA" OFF)
option(onnxruntime_USE_LEAN_ATTENTION "Build lean attention kernel for scaled dot product attention" OFF)
cmake_dependent_option(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION "Build memory efficient attention kernel for scaled dot product attention" ON "onnxruntime_USE_CUDA" OFF)
# The TensorRT fused MHA kernels are prebuilt cubins for sm70-sm89 that mainly benefit BERT-style
# encoder models. Turning this OFF drops ~14MB of embedded cubin data; attention falls back to
# flash / memory efficient / cuDNN / unfused kernels.
cmake_dependent_option(onnxruntime_USE_TRT_FUSED_ATTENTION "Build TensorRT fused multi-head attention cubin kernels" ON "onnxruntime_USE_CUDA" OFF)
cmake_dependent_option(onnxruntime_USE_FP4_QMOE "Build CUDA QMoE FP4 kernels" ON "onnxruntime_USE_CUDA" OFF)
option(onnxruntime_USE_FP8_QMOE "Build CUDA QMoE FP8 kernels" OFF)
cmake_dependent_option(onnxruntime_USE_FPA_INTB_GEMM "Build FpA IntB GEMM CUDA kernels" ON "onnxruntime_USE_CUDA" OFF)
cmake_dependent_option(onnxruntime_USE_FPA_INTB_GEMM_FULL
  "Build all FpA IntB GEMM CUDA kernel variants instead of the compact FP16/BF16 INT2/INT4/INT8 set" OFF
  "onnxruntime_USE_CUDA;onnxruntime_USE_FPA_INTB_GEMM" OFF)
option(onnxruntime_USE_INT4_KV_CACHE "Build cuda kernels for int4 kv cache" ON)
option(onnxruntime_USE_FP8_KV_CACHE "Build cuda kernels for fp8 kv cache" ON)
option(onnxruntime_QUICK_BUILD "Speed up build by skipping some kernels for faster development" OFF)
# Raises the minimum driver to the CUDA 12.4 level (Linux >= 550.54.14, Windows >= 551.61); always on for CUDA >= 13.0.
cmake_dependent_option(onnxruntime_CUDA_FATBIN_COMPRESS_SIZE "Compress CUDA fatbins with -compress-mode=size" OFF "onnxruntime_USE_CUDA" OFF)

option(onnxruntime_BUILD_FOR_NATIVE_MACHINE "Enable this option for turning on optimization specific to this machine" OFF)
option(onnxruntime_USE_AVX "Use AVX instructions" OFF)
option(onnxruntime_USE_AVX2 "Use AVX2 instructions" OFF)
option(onnxruntime_USE_AVX512 "Use AVX512 instructions" OFF)

option(onnxruntime_BUILD_SHARED_LIB "Build a shared library" OFF)
option(onnxruntime_BUILD_APPLE_FRAMEWORK "Build a macOS/iOS framework" OFF)
option(onnxruntime_ENABLE_MICROSOFT_INTERNAL "Use this option to enable/disable microsoft internal only code" OFF)

option(onnxruntime_USE_VITISAI "Build with Vitis-AI" OFF)
option(onnxruntime_USE_TENSORRT "Build with TensorRT support" OFF)
option(onnxruntime_USE_TENSORRT_BUILTIN_PARSER "Use TensorRT builtin parser" OFF)
option(onnxruntime_ENABLE_LTO "Enable link time optimization" OFF)
option(onnxruntime_CROSS_COMPILING "Cross compiling onnx runtime" OFF)
option(onnxruntime_GCOV_COVERAGE "Compile with options necessary to run code coverage" OFF)
option(onnxruntime_DONT_VECTORIZE "Do not vectorize operations in Eigen" OFF)

option(onnxruntime_USE_FULL_PROTOBUF "Link to libprotobuf instead of libprotobuf-lite when this option is ON" OFF)
option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS "Dump debug information about node inputs and outputs when executing the model." OFF)
option(onnxruntime_DUMP_TENSOR "Dump tensor inside kernel." OFF)
cmake_dependent_option(onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS_ENABLE_DUMP_TO_SQLDB "Build dump debug information about node inputs and outputs with support for sql database." OFF "onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS" OFF)

# When loading a delay loaded DLL, Windows searches the main EXE's folder first.
cmake_dependent_option(onnxruntime_ENABLE_DELAY_LOADING_WIN_DLLS "Delay load some of the dependent DLls that are part of the OS" ON "WIN32;NOT GDK_PLATFORM" OFF)
option(onnxruntime_USE_DML "Build with DirectML support" OFF)
option(onnxruntime_USE_MIGRAPHX "Build with AMDMIGraphX support" OFF)
option(onnxruntime_USE_WINML "Build with WinML support" OFF)
option(onnxruntime_USE_ACL "Build with ACL support" OFF)
option(onnxruntime_ENABLE_INSTRUMENT "Enable Instrument with Event Tracing for Windows (ETW)" OFF)
option(onnxruntime_USE_TELEMETRY "Build with Telemetry" OFF)
# AUTO selects 1DS on supported native platforms; TraceLogging requires an explicit Windows opt-in.
set(onnxruntime_TELEMETRY_BACKEND "AUTO" CACHE STRING "Telemetry backend: AUTO, 1DS, or WINDOWS")
set_property(CACHE onnxruntime_TELEMETRY_BACKEND PROPERTY STRINGS AUTO 1DS WINDOWS)
string(TOUPPER "${onnxruntime_TELEMETRY_BACKEND}" onnxruntime_TELEMETRY_BACKEND)
if(NOT onnxruntime_TELEMETRY_BACKEND MATCHES "^(AUTO|1DS|WINDOWS)$")
  message(FATAL_ERROR
    "Invalid onnxruntime_TELEMETRY_BACKEND='${onnxruntime_TELEMETRY_BACKEND}'. "
    "Expected AUTO, 1DS, or WINDOWS.")
endif()

# Deprecated compatibility alias for callers that adopted this option before the backend selector.
option(onnxruntime_USE_WINDOWS_TELEMETRY "Use Windows TraceLogging instead of 1DS telemetry" OFF)
if(onnxruntime_USE_WINDOWS_TELEMETRY)
  set(onnxruntime_TELEMETRY_BACKEND "WINDOWS")
endif()

if(onnxruntime_TELEMETRY_BACKEND STREQUAL "WINDOWS")
  if(NOT WIN32)
    message(FATAL_ERROR "The WINDOWS telemetry backend is only supported on Windows")
  endif()
  set(onnxruntime_USE_TELEMETRY ON CACHE BOOL "Build with Telemetry" FORCE)
endif()

set(onnxruntime_USE_1DS_TELEMETRY OFF)
if(onnxruntime_USE_TELEMETRY)
  if(onnxruntime_TELEMETRY_BACKEND STREQUAL "1DS" OR
     onnxruntime_TELEMETRY_BACKEND STREQUAL "AUTO")
    set(onnxruntime_USE_1DS_TELEMETRY ON)
  endif()
endif()
if(CMAKE_SYSTEM_NAME STREQUAL "WindowsStore" AND onnxruntime_USE_1DS_TELEMETRY)
  message(WARNING "1DS telemetry is not supported for UWP builds. Telemetry will be disabled.")
  set(onnxruntime_USE_TELEMETRY OFF CACHE BOOL "Build with Telemetry" FORCE)
  set(onnxruntime_USE_1DS_TELEMETRY OFF)
endif()
# Optional 1DS ingestion token. Leave empty to compile in the encoded
# in-repo platform default; callers can override it at configure time.
set(ONNXRUNTIME_TELEMETRY_TENANT_TOKEN "" CACHE STRING "Override the compiled-in 1DS telemetry ingestion token")
cmake_dependent_option(onnxruntime_USE_MIMALLOC "Override new/delete and arena allocator with mimalloc" OFF "WIN32;NOT onnxruntime_USE_CUDA;NOT onnxruntime_USE_OPENVINO" OFF)
option(onnxruntime_USE_CANN "Build with CANN support" OFF)
option(onnxruntime_USE_XNNPACK "Build with XNNPACK support. Provides an alternative math library on ARM, WebAssembly and x86." OFF)
option(onnxruntime_USE_WEBNN "Build with WebNN support. Enable hardware acceleration in web browsers." OFF)
option(onnxruntime_USE_WEBGPU "Build with WebGPU support. Enable WebGPU via C/C++ interface." OFF)
option(onnxruntime_USE_EP_API_ADAPTERS "Build EP (e.g. WebGPU) as a plugin EP library using EP API adapters" OFF)
option(onnxruntime_WEBGPU_STATIC_PLUGIN "Build the WebGPU plugin EP into the ORT binary instead of a separate shared library. Requires onnxruntime_USE_EP_API_ADAPTERS." OFF)
option(onnxruntime_USE_EXTERNAL_DAWN "Build with treating Dawn as external dependency. Will not link Dawn at build time." OFF)
option(onnxruntime_CUSTOM_DAWN_SRC_PATH "Path to custom Dawn src dir.")
set(onnxruntime_DAWN_PREBUILT_DIR "" CACHE PATH "Directory containing Dawn headers and standalone proc sources for external Dawn")
if(onnxruntime_DAWN_PREBUILT_DIR)
  if(NOT onnxruntime_USE_WEBGPU OR NOT onnxruntime_USE_EXTERNAL_DAWN)
    message(FATAL_ERROR "onnxruntime_DAWN_PREBUILT_DIR requires onnxruntime_USE_WEBGPU and onnxruntime_USE_EXTERNAL_DAWN")
  endif()
  if(NOT EXISTS "${onnxruntime_DAWN_PREBUILT_DIR}/include")
    message(FATAL_ERROR "onnxruntime_DAWN_PREBUILT_DIR is missing include/: ${onnxruntime_DAWN_PREBUILT_DIR}")
  endif()
endif()
option(onnxruntime_BUILD_DAWN_SHARED_LIBRARY "Build Dawn as a shared library" OFF)
option(onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP "Adding frame present for PIX to capture a frame" OFF)
option(onnxruntime_ENABLE_D3D12_FILE_LOADING
       "Enable accelerated file-to-D3D12-buffer loading on Windows" OFF)
if(onnxruntime_ENABLE_D3D12_FILE_LOADING)
  add_compile_definitions(ENABLE_D3D12_FILE_LOADING=1)
endif()
# The following 2 options are only for Windows
option(onnxruntime_ENABLE_DAWN_BACKEND_VULKAN "Enable Vulkan backend for Dawn (on Windows)" OFF)
option(onnxruntime_ENABLE_DAWN_BACKEND_D3D12 "Enable D3D12 backend for Dawn (on Windows)" ON)

# Options related to reducing the binary size produced by the build
# XNNPACK EP requires the internal NHWC contrib ops to be available, so this option must be OFF when onnxruntime_USE_XNNPACK is ON
cmake_dependent_option(onnxruntime_DISABLE_CONTRIB_OPS "Disable contrib ops" OFF "NOT onnxruntime_USE_XNNPACK" OFF)
option(onnxruntime_DISABLE_ML_OPS "Disable traditional ML ops" OFF)
option(onnxruntime_DISABLE_GENERATION_OPS "Disable generation contrib ops (BeamSearch, WhisperBeamSearch, GreedySearch, Sampling)" OFF)
option(onnxruntime_DISABLE_SPARSE_TENSORS "Disable sparse tensors data types" OFF)
option(onnxruntime_DISABLE_OPTIONAL_TYPE "Disable optional type" OFF)
option(onnxruntime_DISABLE_FLOAT8_TYPES "Disable float 8 types" OFF)
option(onnxruntime_DISABLE_FLOAT4_TYPES "Disable float 4 types" OFF)
option(onnxruntime_DISABLE_STRING_TYPE "Disable string type" OFF)
option(onnxruntime_DISABLE_DEVICE_DISCOVERY "Do not probe the platform for hardware devices; only the CPU device is reported." OFF)
option(onnxruntime_DISABLE_STREAM "Disable multi-stream execution support. Execution providers that use device streams (e.g. CUDA, TensorRT) need it." OFF)
option(onnxruntime_MINIMAL_BUILD "Exclude as much as possible from the build. Support ORT format models. No support for ONNX format models." OFF)
option(onnxruntime_CLIENT_PACKAGE_BUILD "Enables default settings that are more appropriate for client/on-device workloads." OFF)
option(onnxruntime_ENABLE_SESSION_THREADPOOL_CALLBACKS "Enable per-session thread pool work callbacks." OFF)
cmake_dependent_option(onnxruntime_DISABLE_RTTI "Disable RTTI" ON "NOT onnxruntime_ENABLE_PYTHON;NOT onnxruntime_USE_CUDA" OFF)
cmake_dependent_option(onnxruntime_DISABLE_EXCEPTIONS "Disable exception handling." ${onnxruntime_MINIMAL_BUILD} "NOT onnxruntime_ENABLE_PYTHON" OFF)
# Even when onnxruntime_DISABLE_ABSEIL is ON, ONNX Runtime still needs to link to abseil.
option(onnxruntime_DISABLE_ABSEIL "Do not use Abseil data structures in ONNX Runtime source code. Redefine Inlined containers to STD containers." OFF)

option(onnxruntime_EXTENDED_MINIMAL_BUILD "onnxruntime_MINIMAL_BUILD with support for execution providers that compile kernels." OFF)
cmake_dependent_option(onnxruntime_ENABLE_GQA_VALUE_LAYOUT "Enable GroupQueryAttention Value-cache layout conversion and validation" ON
  "NOT onnxruntime_MINIMAL_BUILD;NOT onnxruntime_EXTENDED_MINIMAL_BUILD;NOT onnxruntime_DISABLE_CONTRIB_OPS" OFF)
option(onnxruntime_MINIMAL_BUILD_CUSTOM_OPS "Add custom operator kernels support to a minimal build." OFF)
option(onnxruntime_REDUCED_OPS_BUILD "Reduced set of kernels are registered in build via modification of the kernel registration source files." OFF)
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
option(onnxruntime_ENABLE_TRAINING "Enable full training functionality. Includes ORTModule and ORT Training APIs" OFF)
option(onnxruntime_ENABLE_TRAINING_APIS "Enable ort training apis." OFF)
option(onnxruntime_ENABLE_TRAINING_OPS "Include training operators but no training session support." OFF)
option(onnxruntime_ENABLE_TRAINING_E2E_TESTS "Enable training end-to-end tests." OFF)
option(onnxruntime_ENABLE_CPU_FP16_OPS "Build with advanced instruction sets" ON)
option(onnxruntime_USE_NCCL "Build with NCCL support" OFF)

# WebAssembly options
option(onnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB "Enable this option to create WebAssembly static library" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_JSPI "Enable WebAssembly JavaScript Promise Integration" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_THREADS "Enable this option to create WebAssembly byte codes with multi-threads support" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING "Enable this option to turn on exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING "Enable this option to turn on api exception catching" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING "Enable this option to turn on exception throwing even if the build disabled exceptions support" OFF)
option(onnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER "Enable this option to run tests in browser instead of Node.js" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO "Enable this option to turn on DWARF format debug info" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_PROFILING "Enable this option to turn on WebAssembly profiling and preserve function names" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL "Enable this option to allow WebAssembly to output optimized model" OFF)
option(onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD "Enable WebAssembly Relaxed SIMD" OFF)

# Enable bitcode for iOS
option(onnxruntime_ENABLE_BITCODE "Enable bitcode for iOS only" OFF)

# build Pytorch's LazyTensor support
cmake_dependent_option(onnxruntime_ENABLE_LAZY_TENSOR "Enable ORT as a LazyTensor backend in Pytorch." ON "onnxruntime_ENABLE_TRAINING" OFF)

# build separate library of schemas of (custom) ops used by ORT (for ONNX to MLIR translation)
option(onnxruntime_BUILD_OPSCHEMA_LIB "Build op schema library" ON)

# option to enable custom operators in onnxruntime-extensions
option(onnxruntime_USE_EXTENSIONS "Build with onnxruntime-extensions for more features" OFF)

# option to enable custom operators in onnxruntime-extensions with a custom path
option(onnxruntime_EXTENSIONS_OVERRIDDEN, "Enable onnxruntime-extensions overridden" OFF)

# Enable registering custom op schemas from shared libraries in python environment.
option(onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS "Enable registering user defined custom op schemas dynamically" OFF)

set(ONNX_CUSTOM_PROTOC_EXECUTABLE "" CACHE STRING "Specify custom protoc executable to build ONNX")

# pre-build python path
option(onnxruntime_PREBUILT_PYTORCH_PATH "Path to pytorch installation dir")

# external transformer src path
option(onnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH "Path to external transformer src dir")

option(onnxruntime_ENABLE_CUDA_PROFILING "Enable CUDA kernel profiling" OFF)

option(onnxruntime_ENABLE_CPUINFO "Enable cpuinfo" ON)

# ATen fallback support
option(onnxruntime_ENABLE_ATEN "Enable ATen fallback" OFF)

# dlpack support
cmake_dependent_option(onnxruntime_ENABLE_DLPACK "Enable dlpack" ON "onnxruntime_ENABLE_TRAINING OR onnxruntime_ENABLE_ATEN OR onnxruntime_ENABLE_PYTHON" OFF)
option(onnxruntime_USE_AZURE "Build with azure inferencing support" OFF)
option(onnxruntime_USE_LOCK_FREE_QUEUE "Build with lock-free task queue for threadpool." OFF)
option(onnxruntime_FORCE_GENERIC_ALGORITHMS "Disable optimized arch-specific algorithms. Use only for testing and debugging generic algorithms." OFF)

# DX interop feature option
option(onnxruntime_USE_DX_INTEROP "Build with the DX Interop feature for graphics API synchronization." OFF)

if (onnxruntime_USE_DX_INTEROP)
  add_compile_definitions(USE_DX_INTEROP=1)
else()
  add_compile_definitions(USE_DX_INTEROP=0)
endif()

option(onnxruntime_USE_TENSORRT_INTERFACE "Build ONNXRuntime shared lib which is compatible with TensorRT EP interface" OFF)
option(onnxruntime_USE_CUDA_INTERFACE "Build ONNXRuntime shared lib which is compatible with Cuda EP interface" OFF)
option(onnxruntime_USE_OPENVINO_INTERFACE "Build ONNXRuntime shared lib which is compatible with OpenVINO EP interface" OFF)
option(onnxruntime_USE_VITISAI_INTERFACE "Build ONNXRuntime shared lib which is compatible with Vitis-AI EP interface" OFF)
option(onnxruntime_USE_QNN_INTERFACE "Build ONNXRuntime shared lib which is compatible with QNN EP interface" OFF)

if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 11.1)
  message(FATAL_ERROR  "GCC version must be greater than or equal to 11.1")
endif()

include(detect_onnxruntime_target_platform.cmake)

# ENABLE_TRAINING includes all training functionality
# The following 2 entry points
# 1. ORTModule
# 2. ORT Training APIs
# It includes all the feature additions as well like
# 1. Python OP
# 2. Aten Fallback
# 3. Strided Tensors
# 4. All training ops including communication and  collectives ops
# 5. ONNXBlock (Front end for training preparation when using training apis)
# Some features are only enabled when onnxruntime_ENABLE_PYTHON is ON as they are only relevant
# when using python env
if (onnxruntime_ENABLE_TRAINING)
  set(onnxruntime_ENABLE_TRAINING_APIS ON)
  set(onnxruntime_ENABLE_TRAINING_OPS ON)
  set(onnxruntime_ENABLE_ATEN ON)
  set(onnxruntime_ENABLE_TRITON ON)
  if (NOT APPLE)
    set(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP ON)
  endif()
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
  set(onnxruntime_ENABLE_TRAINING_OPS ON)
  if (onnxruntime_ENABLE_PYTHON AND NOT onnxruntime_ENABLE_TRAINING)
    message(FATAL_ERROR "Standalone On-Device Training build is not supported with Python bindings! "
    "Please use the --enable_training flag instead of the --enable_training_apis flag.")
  endif()
endif()

if (onnxruntime_DISABLE_STREAM AND onnxruntime_ENABLE_TRAINING)
  message(FATAL_ERROR "onnxruntime_DISABLE_STREAM=ON is not supported with onnxruntime_ENABLE_TRAINING=ON.")
endif()


# Single output directory for all binaries
set(RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin CACHE PATH "Single output directory for all binaries.")

# Local mirror directory of cmake dependencies
set(REPO_ROOT ${PROJECT_SOURCE_DIR}/..)
set(onnxruntime_CMAKE_DEPS_MIRROR_DIR ${REPO_ROOT}/mirror CACHE PATH "Path to the local mirror of cmake dependencies")


include(FetchContent)

function(set_msvc_c_cpp_compiler_warning_level warning_level)
  if (NOT "${warning_level}" MATCHES "^[0-4]$")
    message(FATAL_ERROR "Expected warning_level value of 0-4, got '${warning_level}'.")
  endif()

  if (MSVC)
    set(warning_flag "/W${warning_level}")
    get_property(opts DIRECTORY PROPERTY COMPILE_OPTIONS)
    # only match the generator expression added by this function
    list(FILTER opts
         EXCLUDE REGEX "^\\$<\\$<COMPILE_LANGUAGE:CXX,C>:/W[0-4]>$")
    list(APPEND opts "$<$<COMPILE_LANGUAGE:CXX,C>:${warning_flag}>")
    set_property(DIRECTORY PROPERTY COMPILE_OPTIONS "${opts}")
  endif()
endfunction()


if(MSVC_Z7_OVERRIDE)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>")
endif(MSVC_Z7_OVERRIDE)

# set default MSVC warning level to 3 for external dependencies
set_msvc_c_cpp_compiler_warning_level(3)

# Fuzz test has only been tested with BUILD_SHARED_LIB option,
# using the MSVC compiler and on windows OS and clang/gcc compiler on Linux.
if (onnxruntime_FUZZ_TEST AND onnxruntime_BUILD_SHARED_LIB AND onnxruntime_USE_FULL_PROTOBUF)
  # Fuzz test library dependency, protobuf-mutator,
  # needs the onnx message to be compiled using "non-lite protobuf version"
  set(onnxruntime_FUZZ_ENABLED ON)
endif()

if (onnxruntime_USE_VALGRIND AND NOT WIN32)
  add_definitions(-DRE2_ON_VALGRIND=1)
endif()

if (onnxruntime_ENABLE_NVTX_PROFILE)
  add_definitions(-DENABLE_NVTX_PROFILE=1)
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

set(ONNX_ML 1)

if (NOT (UNIX AND onnxruntime_ENABLE_PYTHON AND onnxruntime_ENABLE_TRAINING AND (NOT onnxruntime_BUILD_SHARED_LIB)))
  if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
    message(WARNING "onnxruntime_ENABLE_TRAINING_TORCH_INTEROP is turned OFF due to incompatible build combinations.")
  endif()
  set(onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OFF)
endif()

if (NOT (UNIX AND onnxruntime_USE_CUDA AND onnxruntime_ENABLE_PYTHON AND onnxruntime_ENABLE_TRAINING AND (NOT onnxruntime_BUILD_SHARED_LIB)))
  if (onnxruntime_ENABLE_TRITON)
    message(WARNING "onnxruntime_ENABLE_TRITON is turned OFF because it's designed to support CUDA training on Linux only currently.")
  endif()
  set(onnxruntime_ENABLE_TRITON OFF)
endif()

set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB OFF)
if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP)
  add_compile_definitions(ENABLE_TRAINING_TORCH_INTEROP)

  # Python::Python is required for building unit test executables.
  if (onnxruntime_BUILD_UNIT_TESTS)
    set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB ON)
  endif()
endif()

if (onnxruntime_ENABLE_TRITON)
  # Need SetOutputMLValue.
  set(onnxruntime_ENABLE_ATEN ON)
  add_compile_definitions(ENABLE_TRITON)

  # Python::Python is required for building unit test executables.
  if (onnxruntime_BUILD_UNIT_TESTS)
    set(onnxruntime_REQUIRE_PYTHON_EMBED_LIB ON)
  endif()
endif()

# Building plugin EPs that depend on EP API adapters.
if(onnxruntime_USE_EP_API_ADAPTERS)
  add_compile_definitions(ORT_USE_EP_API_ADAPTERS=1)
endif()

# The WebGPU plugin EP can either be a separate shared library or be linked into the ORT binary. In the latter case
# ORT core registers it at OrtEnv creation time instead of the application loading it by path.
if(onnxruntime_WEBGPU_STATIC_PLUGIN)
  if(NOT onnxruntime_USE_WEBGPU)
    message(FATAL_ERROR "onnxruntime_WEBGPU_STATIC_PLUGIN=ON requires onnxruntime_USE_WEBGPU=ON.")
  endif()
  if(NOT onnxruntime_USE_EP_API_ADAPTERS)
    message(FATAL_ERROR "onnxruntime_WEBGPU_STATIC_PLUGIN=ON requires onnxruntime_USE_EP_API_ADAPTERS=ON.")
  endif()
  # A minimal build excludes core/session/plugin_ep, so the registration that a static plugin EP depends on is not
  # compiled in. Without this check the EP would be linked into the binary but never registered, with no diagnostic
  # at configure, build or run time. Supporting the combination is possible but requires making the plugin EP
  # registration available to minimal builds first.
  if(onnxruntime_MINIMAL_BUILD)
    message(FATAL_ERROR "onnxruntime_WEBGPU_STATIC_PLUGIN=ON is not supported in a minimal build: the plugin EP "
                        "registration in core/session/plugin_ep is excluded, so the EP would be linked in but "
                        "never registered.")
  endif()
  add_compile_definitions(ORT_WEBGPU_STATIC_PLUGIN=1)
endif()

# True when the WebGPU EP is compiled into the ORT binary rather than built as a separate shared library.
if(onnxruntime_USE_WEBGPU AND (NOT onnxruntime_USE_EP_API_ADAPTERS OR onnxruntime_WEBGPU_STATIC_PLUGIN))
  set(onnxruntime_WEBGPU_LINKED_INTO_HOST ON)
else()
  set(onnxruntime_WEBGPU_LINKED_INTO_HOST OFF)
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

if (onnxruntime_BUILD_OBJC)
  check_language(OBJC)
  if(CMAKE_OBJC_COMPILER)
      enable_language(OBJC)
  else()
      message(FATAL_ERROR "Objective-C is not supported.")
  endif()

  check_language(OBJCXX)
  if(CMAKE_OBJCXX_COMPILER)
      enable_language(OBJCXX)
  else()
      message(FATAL_ERROR "Objective-C++ is not supported.")
  endif()
endif()

if (NOT WIN32)
  #TODO: On Linux we may try https://github.com/microsoft/TraceLogging.git
  if (onnxruntime_ENABLE_INSTRUMENT)
    message(WARNING "Instrument is only supported on Windows now")
    set(onnxruntime_ENABLE_INSTRUMENT OFF)
  endif()
endif()

# 'extended' implies minimal.
if (onnxruntime_EXTENDED_MINIMAL_BUILD AND NOT onnxruntime_MINIMAL_BUILD)
  set(onnxruntime_MINIMAL_BUILD ON)
endif()

set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/../onnxruntime)
set(ORTTRAINING_ROOT ${PROJECT_SOURCE_DIR}/../orttraining)
set(ORTTRAINING_SOURCE_DIR ${ORTTRAINING_ROOT}/orttraining)

include(adjust_global_compile_flags.cmake)

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
  endif()
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
endif()


file (STRINGS "${REPO_ROOT}/VERSION_NUMBER" ORT_VERSION)

find_package(Threads)
# On Windows we directly use Windows APIs to do the job
# Android NDK doesn't provide the iconv lib.
if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Android")
  find_package(Iconv REQUIRED)
  set(ICONV_LIB Iconv::Iconv)
endif()

find_package(Patch)
if (CMAKE_HOST_WIN32 AND NOT Patch_FOUND)
    # work around CI machines missing patch from the git install by falling back to the binary in this repo.
    # replicate what happens in https://github.com/Kitware/CMake/blob/master/Modules/FindPatch.cmake but without
    # the hardcoded suffixes in the path to the patch binary.
    find_program(Patch_EXECUTABLE NAMES patch PATHS ${PROJECT_SOURCE_DIR}/external/git.Win32.2.41.03.patch)
    if(Patch_EXECUTABLE)
      set(Patch_FOUND 1)
      if (NOT TARGET Patch::patch)
        add_executable(Patch::patch IMPORTED)
        set_property(TARGET Patch::patch PROPERTY IMPORTED_LOCATION ${Patch_EXECUTABLE})
      endif()
    endif()
endif()
if(Patch_FOUND)
  message("Patch found: ${Patch_EXECUTABLE}")
endif()

if (CMAKE_CROSSCOMPILING)
  message("Doing crosscompiling")
endif()

#Need python to generate def file
if (onnxruntime_BUILD_SHARED_LIB OR onnxruntime_ENABLE_PYTHON)
  if (onnxruntime_ENABLE_PYTHON)
    if (onnxruntime_REQUIRE_PYTHON_EMBED_LIB)
      find_package(Python 3.10 COMPONENTS Interpreter Development NumPy)
    else()
      find_package(Python 3.10 COMPONENTS Interpreter Development.Module NumPy)
    endif()
    message("Numpy version: ${Python_NumPy_VERSION}")
    if(Python_NumPy_VERSION VERSION_LESS "2.0.0")
      message(WARNING "The build binary will not be compatible with NumPy 2.0 because the NumPy installed on this machine is too low.")
    endif()
  else()
    find_package(Python 3.10 COMPONENTS Interpreter)
  endif()
endif()

if (onnxruntime_USE_ARM_NEON_NCHWC)
  message(STATUS "Building MLAS with ARM Neon NCHWc kernels")
endif()

if(onnxruntime_USE_SVE)
  # ABI flags that EVERY SVE translation unit must carry.
  #
  # x18 is the AArch64 platform register: Windows ARM64 reserves it for the TEB,
  # Darwin reserves it, and Linux shadow-call-stack builds use it. MLAS freezes
  # SVE kernels into machine code that ships unchanged on every platform
  # (aarch64/*_sve_asm.S), so an SVE TU that allocates x18 corrupts the TEB once
  # those bytes run on Windows (observed: 0xC0000005 writing 0x250 inside ntdll).
  # Saving/restoring x18 around the kernel is NOT a fix -- the OS may rewrite it
  # at any context switch.
  #
  # Keep in sync with sve/gen_sve_asm.py, which applies the same flag when
  # regenerating and hard-fails if x18 survives into a frozen section. The flag
  # must also be on the non-frozen build of the same sources, or the ASM=OFF and
  # ASM=ON paths would be built from differently-compiled code.
  set(ORT_SVE_ABI_FLAGS "-ffixed-x18")

  if(LINUX AND onnxruntime_target_platform STREQUAL "aarch64")
    check_cxx_compiler_flag("-march=armv8.2-a+sve" HAS_ARM64_SVE)
    if(HAS_ARM64_SVE)
      message(STATUS "Compiler supports SVE!")
    else()
      message(WARNING "onnxruntime_USE_SVE was set but compiler does not support SVE. It will be disabled.")
      set(onnxruntime_USE_SVE OFF)
    endif()
  elseif(WIN32 AND onnxruntime_target_platform STREQUAL "ARM64")
    # The SVE elementwise kernels ship as portable machine-code assembly
    # (hex-encoded instruction words assembled by armasm64), so no compiler
    # SVE support is needed; the kernels are runtime-gated on HasArmSve().
    message(STATUS "SVE enabled via portable machine-code kernels.")
  else()
    message(WARNING "onnxruntime_USE_SVE was set but it is not supported on this platform. It will be disabled.")
    set(onnxruntime_USE_SVE OFF)
  endif()
endif()

if(onnxruntime_USE_KLEIDIAI)
  function(is_kleidiai_supported is_supported_var)
    # check for supported target platforms
    if(NOT (onnxruntime_target_platform STREQUAL "aarch64" OR
            onnxruntime_target_platform STREQUAL "ARM64" OR
            onnxruntime_target_platform STREQUAL "arm64"))
      message(WARNING "KleidiAI is not supported on this platform.")

      set(${is_supported_var} FALSE PARENT_SCOPE)
      return()
    endif()

    if(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND MSVC_VERSION VERSION_LESS 1940)
      message(WARNING "KleidiAI requires MSVC compiler version 19.40 or newer, KleidiAI will be disabled in this build.")

      set(${is_supported_var} FALSE PARENT_SCOPE)
      return()
    endif()

    set(${is_supported_var} TRUE PARENT_SCOPE)
  endfunction()

  is_kleidiai_supported(is_kleidiai_supported_result)

  if(NOT is_kleidiai_supported_result)
    message(WARNING "onnxruntime_USE_KLEIDIAI was set but it is not supported. It will be disabled.")
    set(onnxruntime_USE_KLEIDIAI OFF)
  endif()
endif()

#Dependencies begin
get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
get_filename_component(ORTTRAINING_ROOT "${ORTTRAINING_ROOT}" ABSOLUTE)
get_filename_component(REPO_ROOT "${REPO_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR ${REPO_ROOT}/include/onnxruntime)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external)
include(external/onnxruntime_external_deps.cmake)

set(ORT_WARNING_FLAGS)
if (WIN32)
    # class needs to have dll-interface to be used by clients
    list(APPEND ORT_WARNING_FLAGS "/wd4251")
    # issued by thrust nonstandard extension used: nameless struct/union
    list(APPEND ORT_WARNING_FLAGS "/wd4201")
    # structure was padded due to __declspec(align())
    list(APPEND ORT_WARNING_FLAGS "/wd4324")
    # warning C4800: Implicit conversion from 'X' to bool. Possible information loss
    if (onnxruntime_USE_OPENVINO OR onnxruntime_USE_OPENVINO_INTERFACE)
       list(APPEND ORT_WARNING_FLAGS "/wd4800")
    endif()
    # operator 'operator-name': deprecated between enumerations of different types
    list(APPEND ORT_WARNING_FLAGS "/wd5054")
    # Enable warning: data member 'member' will be initialized after data member 'member2' / base class 'base_class'
    list(APPEND ORT_WARNING_FLAGS "/w15038")

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
  # Check for compiler flags
  check_cxx_compiler_flag(-Wambiguous-reversed-operator HAS_AMBIGUOUS_REVERSED_OPERATOR)
  # -Winterference-size was added in GCC 13
  check_cxx_compiler_flag(-Winterference-size HAS_INTERFERENCE_SIZE)
  check_cxx_compiler_flag(-Warray-bounds HAS_ARRAY_BOUNDS)
  check_cxx_compiler_flag(-Wbitwise-instead-of-logical HAS_BITWISE_INSTEAD_OF_LOGICAL)
  check_cxx_compiler_flag(-Wcast-function-type HAS_CAST_FUNCTION_TYPE)
  check_cxx_compiler_flag(-Wcatch-value HAS_CATCH_VALUE)
  check_cxx_compiler_flag(-Wclass-memaccess HAS_CLASS_MEMACCESS)
  check_cxx_compiler_flag(-Wcharacter-conversion HAS_CHARACTER_CONVERSION)
  check_cxx_compiler_flag(-Wno-error=character-conversion HAS_NO_ERROR_CHARACTER_CONVERSION)
  check_cxx_compiler_flag(-Wdangling-reference HAS_DANGLING_REFERENCE)
  check_cxx_compiler_flag(-Wdeprecated-anon-enum-enum-conversion HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
  check_cxx_compiler_flag(-Wdeprecated-builtins HAS_DEPRECATED_BUILTINS)
  check_cxx_compiler_flag(-Wdeprecated-copy HAS_DEPRECATED_COPY)
  check_cxx_compiler_flag(-Wdeprecated-declarations HAS_DEPRECATED_DECLARATIONS)
  check_cxx_compiler_flag(-Wdeprecated-literal-operator HAS_DEPRECATED_LITERAL_OPERATOR)
  check_cxx_compiler_flag(-Wdeprecated-this-capture HAS_DEPRECATED_THIS_CAPTURE)
  check_cxx_compiler_flag(-Wenum-constexpr-conversion HAS_ENUM_CONSTEXPR_CONVERSION)
  check_cxx_compiler_flag(-Wformat-truncation HAS_FORMAT_TRUNCATION)
  check_cxx_compiler_flag(-Wignored-attributes HAS_IGNORED_ATTRIBUTES)
  check_cxx_compiler_flag(-Wmaybe-uninitialized HAS_MAYBE_UNINITIALIZED)
  check_cxx_compiler_flag(-Wmissing-braces HAS_MISSING_BRACES)
  check_cxx_compiler_flag(-Wnonnull-compare HAS_NONNULL_COMPARE)
  check_cxx_compiler_flag(-Wparentheses HAS_PARENTHESES)
  check_cxx_compiler_flag(-Wshorten-64-to-32 HAS_SHORTEN_64_TO_32)
  check_cxx_compiler_flag(-Wstrict-aliasing HAS_STRICT_ALIASING)
  check_cxx_compiler_flag(-Wstringop-overflow HAS_STRINGOP_OVERFLOW)
  check_cxx_compiler_flag(-Wtautological-pointer-compare HAS_TAUTOLOGICAL_POINTER_COMPARE)
  check_cxx_compiler_flag(-Wundefined-var-template HAS_UNDEFINED_VAR_TEMPLATE)
  check_cxx_compiler_flag(-Wunused-but-set-parameter HAS_UNUSED_BUT_SET_PARAMETER)
  check_cxx_compiler_flag(-Wunused-but-set-variable HAS_UNUSED_BUT_SET_VARIABLE)
  check_cxx_compiler_flag(-Wunused-variable HAS_UNUSED_VARIABLE)
  check_cxx_compiler_flag(-Wuseless-cast HAS_USELESS_CAST)

  if(onnxruntime_ENABLE_TRAINING_APIS)
    if(HAS_DANGLING_REFERENCE)
      list(APPEND ORT_WARNING_FLAGS -Wno-dangling-reference)
    endif()
  endif()
  check_function_exists(reallocarray HAS_REALLOCARRAY)
  if (NOT APPLE AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_target_platform STREQUAL "aarch64")
   check_cxx_compiler_flag(-march=armv8.2-a+bf16 HAS_ARM64_BFLOAT16)
   if(NOT HAS_ARM64_BFLOAT16)
     message(FATAL_ERROR "The compiler/assembler does not support the aarch64 +bf16 ISA extension. "
       "On aarch64 this typically requires GCC >= 12 paired with a sufficiently recent binutils "
       "(the capability is gated by the assembler, not the compiler alone). "
       "See https://github.com/microsoft/onnxruntime/issues/22837 for details.")
   endif()
   check_cxx_compiler_flag(-march=armv8.2-a+fp16 HAS_ARM64_FLOAT16)
   if(NOT HAS_ARM64_FLOAT16)
     message(FATAL_ERROR "The compiler/assembler does not support the aarch64 +fp16 ISA extension. "
       "On aarch64 this typically requires GCC >= 12 paired with a sufficiently recent binutils "
       "(the capability is gated by the assembler, not the compiler alone). "
       "See https://github.com/microsoft/onnxruntime/issues/22837 for details.")
   endif()
  endif()

  if (HAS_TAUTOLOGICAL_POINTER_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-tautological-pointer-compare)
  endif()
  if (HAS_NONNULL_COMPARE)
    #we may have extra null pointer checkings in debug build, it's not an issue
    list(APPEND ORT_WARNING_FLAGS -Wno-nonnull-compare)
  endif()

  if (HAS_AMBIGUOUS_REVERSED_OPERATOR)
    list(APPEND ORT_WARNING_FLAGS -Wno-ambiguous-reversed-operator)
  endif()
  if (HAS_DEPRECATED_ANON_ENUM_ENUM_CONVERSION)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-anon-enum-enum-conversion)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-enum-enum-conversion)
  endif()
  if (HAS_UNDEFINED_VAR_TEMPLATE)
    list(APPEND ORT_WARNING_FLAGS -Wno-undefined-var-template)
  endif()
  if (HAS_DEPRECATED_BUILTINS)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-builtins)
  endif()
  if (HAS_DEPRECATED_LITERAL_OPERATOR)
    list(APPEND ORT_WARNING_FLAGS -Wno-deprecated-literal-operator)
  endif()
  #see:https://reviews.llvm.org/D131307
  #It was intended that the 'enum-constexpr-conversion' type warnings can not be silenced by -w
  if(HAS_ENUM_CONSTEXPR_CONVERSION AND NOT Protobuf_FOUND)
    if (TARGET libprotobuf)
      target_compile_options(libprotobuf PRIVATE "-Wno-enum-constexpr-conversion")
    endif()
    if (TARGET libprotobuf-lite)
      target_compile_options(libprotobuf-lite PRIVATE "-Wno-enum-constexpr-conversion")
    endif()
  endif()

  if(HAS_INTERFERENCE_SIZE)
    # The warning was from Eigen. Since we do not use Eigen in the public API, we can ignore it.
    # However, it also means that part of our code was compiled based on the build machine's hardware spec.
    # Specifically, the code was tuned based on the build machine's CPU cache line size, which can vary a lot.
    list(APPEND ORT_WARNING_FLAGS -Wno-interference-size)
  endif()
  # enable warning(s) that may not be on by default
  if (HAS_SHORTEN_64_TO_32)
    list(APPEND ORT_WARNING_FLAGS -Wshorten-64-to-32)
  endif()
endif()

#names in this var must match the directory names under onnxruntime/core/providers
#ONNXRUNTIME_PROVIDER_NAMES is the list of providers that needs to export additional symbols in the global namespace.
#For example CUDA EP exports "OrtSessionOptionsAppendExecutionProvider_CUDA", which is a global function.
#However, all these things are legacy and deprecated and should be replaced with functions in onnxruntime_c_api.h.
set(ONNXRUNTIME_PROVIDER_NAMES cpu)

set(ORT_PROVIDER_FLAGS)

if (onnxruntime_USE_CUDA)
  include(cuda_configuration)
  setup_cuda_compiler()
  setup_cuda_architectures()

  enable_language(CUDA)
  message( STATUS "CMAKE_CUDA_COMPILER_VERSION: ${CMAKE_CUDA_COMPILER_VERSION}")

  if (onnxruntime_DISABLE_CONTRIB_OPS)
    # Flash and memory efficient attention are also used by the ONNX domain Attention operator.
    set(onnxruntime_USE_LEAN_ATTENTION OFF)
    set(onnxruntime_USE_FPA_INTB_GEMM OFF)
    set(onnxruntime_USE_FPA_INTB_GEMM_FULL OFF)
  endif()

  if (WIN32)
    message( STATUS "Lean Attention unsupported in Windows")
    set(onnxruntime_USE_LEAN_ATTENTION OFF)
  endif()
else()
  set(onnxruntime_USE_FLASH_ATTENTION OFF)
  set(onnxruntime_USE_LEAN_ATTENTION OFF)
  set(onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION OFF)
  set(onnxruntime_USE_TRT_FUSED_ATTENTION OFF)
endif()

if (onnxruntime_USE_CUDA)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_CUDA=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES cuda)

    if (onnxruntime_USE_FLASH_ATTENTION)
      message( STATUS "Enable flash attention for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_FLASH_ATTENTION=1)
    endif()

    if (onnxruntime_USE_LEAN_ATTENTION)
      message( STATUS "Enable lean attention for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_LEAN_ATTENTION=1)
    endif()

    if (onnxruntime_USE_MEMORY_EFFICIENT_ATTENTION)
      message( STATUS "Enable memory efficient attention for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_MEMORY_EFFICIENT_ATTENTION=1)
    endif()

    if (onnxruntime_USE_TRT_FUSED_ATTENTION)
      message( STATUS "Enable TensorRT fused multi-head attention for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_TRT_FUSED_ATTENTION=1)
    endif()

    if (onnxruntime_USE_FPA_INTB_GEMM)
      if (onnxruntime_USE_FPA_INTB_GEMM_FULL)
        message( STATUS "Enable full FpA IntB GEMM kernel set for CUDA EP")
        list(APPEND ORT_PROVIDER_FLAGS -DUSE_FPA_INTB_GEMM=1)
      else()
        message( STATUS "Enable compact non-Hopper FP16/BF16 INT2/INT4/INT8 FpA IntB GEMM kernel set for CUDA EP")
        list(APPEND ORT_PROVIDER_FLAGS -DUSE_FPA_INTB_GEMM=1 -DUSE_COMPACT_FPA_INTB_GEMM=1)
      endif()
    endif()

    if (onnxruntime_QUICK_BUILD)
      message( STATUS "Quick build mode: reducing selected CUDA/CUTLASS kernel instantiations")
      list(APPEND ORT_PROVIDER_FLAGS -DORT_QUICK_BUILD=1)
    endif()

    if (onnxruntime_USE_INT4_KV_CACHE)
      message( STATUS "Enable int4 kv cache for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_INT4_KV_CACHE=1)
    endif()

    if (onnxruntime_USE_FP8_KV_CACHE)
      message( STATUS "Enable fp8 kv cache for CUDA EP")
      list(APPEND ORT_PROVIDER_FLAGS -DUSE_FP8_KV_CACHE=1)
    endif()
endif()

if (onnxruntime_USE_CUDA_INTERFACE AND (NOT onnxruntime_USE_CUDA))
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_CUDA_PROVIDER_INTERFACE=1)
endif()

if (onnxruntime_USE_VITISAI)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_VITISAI=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES vitisai)
endif()

if (onnxruntime_USE_VITISAI_INTERFACE AND (NOT onnxruntime_USE_VITISAI))
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_VITISAI_PROVIDER_INTERFACE=1)
endif()

if (onnxruntime_USE_DNNL)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_DNNL=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES dnnl)
endif()

if (onnxruntime_USE_OPENVINO)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_OPENVINO=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES openvino)
endif()

if (onnxruntime_USE_OPENVINO_INTERFACE AND (NOT onnxruntime_USE_OPENVINO))
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_OPENVINO_PROVIDER_INTERFACE=1)
endif()

if (onnxruntime_USE_TENSORRT)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_TENSORRT=1)
    #TODO: remove the following line and change the test code in onnxruntime_shared_lib_test to use the new EP API.
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES tensorrt)
endif()

if (onnxruntime_USE_TENSORRT_INTERFACE AND (NOT onnxruntime_USE_TENSORRT))
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_TENSORRT_PROVIDER_INTERFACE=1)
endif()

if (onnxruntime_USE_RKNPU)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_RKNPU=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES rknpu)
endif()
if (onnxruntime_USE_VSINPU)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_VSINPU=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES vsinpu)
endif()
if (onnxruntime_USE_NNAPI_BUILTIN)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_NNAPI=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES nnapi)
endif()
if (onnxruntime_USE_JSEP)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_JSEP=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES js)
endif()
if (onnxruntime_USE_QNN OR onnxruntime_USE_QNN_INTERFACE)

    if(onnxruntime_USE_QNN)
        list(APPEND ORT_PROVIDER_FLAGS -DUSE_QNN=1)
    else()
        list(APPEND ORT_PROVIDER_FLAGS -DUSE_QNN_PROVIDER_INTERFACE=1)
    endif()

    list(APPEND ONNXRUNTIME_PROVIDER_NAMES qnn)
    if (NOT QNN_ARCH_ABI)
      string(TOLOWER ${onnxruntime_target_platform} GEN_PLATFORM)
      if(MSVC)
        message(STATUS "Building MSVC for architecture ${CMAKE_SYSTEM_PROCESSOR} with CMAKE_GENERATOR_PLATFORM as ${GEN_PLATFORM}")
        if (${GEN_PLATFORM} STREQUAL "arm64")
          set(QNN_ARCH_ABI aarch64-windows-msvc)
        elseif (${GEN_PLATFORM} STREQUAL "arm64ec")
          set(QNN_ARCH_ABI arm64x-windows-msvc)
        else()
          set(QNN_ARCH_ABI x86_64-windows-msvc)
        endif()
      else()
        if (${CMAKE_SYSTEM_NAME} STREQUAL "Android")
          set(QNN_ARCH_ABI aarch64-android-clang6.0)
        elseif (${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
          if (${GEN_PLATFORM} STREQUAL "x86_64")
            set(QNN_ARCH_ABI x86_64-linux-clang)
          else()
            set(QNN_ARCH_ABI aarch64-android)
          endif()
        endif()
      endif()
    endif()

    if (onnxruntime_USE_QNN AND (MSVC OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"))
      file(GLOB QNN_LIB_FILES LIST_DIRECTORIES false "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/libQnn*.so"
                                             "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/Qnn*.dll"
                                             "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/libHtpPrepare.so"
                                             "${onnxruntime_QNN_HOME}/lib/${QNN_ARCH_ABI}/HtpPrepare.dll")
      if (${QNN_ARCH_ABI} STREQUAL "aarch64-windows-msvc" OR ${QNN_ARCH_ABI} STREQUAL "arm64x-windows-msvc")
        file(GLOB EXTRA_HTP_LIB LIST_DIRECTORIES false "${onnxruntime_QNN_HOME}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so"
                                  "${onnxruntime_QNN_HOME}/lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so"
                                  "${onnxruntime_QNN_HOME}/lib/hexagon-v73/unsigned/libqnnhtpv73.cat"
                                  "${onnxruntime_QNN_HOME}/lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so"
                                  "${onnxruntime_QNN_HOME}/lib/hexagon-v81/unsigned/libqnnhtpv81.cat")
        list(APPEND QNN_LIB_FILES ${EXTRA_HTP_LIB})
      elseif(${QNN_ARCH_ABI} STREQUAL "aarch64-ubuntu-gcc9.4")
        list(APPEND QNN_LIB_FILES "${onnxruntime_QNN_HOME}/lib/hexagon-v68/unsigned/libQnnHtpV68Skel.so")
      endif()
      message(STATUS "QNN lib files: " ${QNN_LIB_FILES})
    endif()
endif()
if (onnxruntime_USE_SNPE)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_SNPE=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES snpe)
endif()
if (onnxruntime_USE_WINML)
  list(APPEND ORT_PROVIDER_FLAGS  -DUSE_WINML=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES winml)
endif()
if (onnxruntime_USE_ACL)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_ACL=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES acl)
endif()
if (onnxruntime_USE_DML)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_DML=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES dml)
    if(onnxruntime_ENABLE_NPU_ADAPTER_ENUMERATION)
      list(APPEND ORT_PROVIDER_FLAGS -DENABLE_NPU_ADAPTER_ENUMERATION=1)
    endif()
endif()
if (onnxruntime_USE_MIGRAPHX)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_MIGRAPHX=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES migraphx)
endif()

if (onnxruntime_USE_MIGRAPHX_INTERFACE AND (NOT onnxruntime_USE_MIGRAPHX))
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_MIGRAPHX_PROVIDER_INTERFACE=1)
endif()

if (onnxruntime_USE_COREML)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_COREML=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES coreml)
endif()
if (onnxruntime_USE_XNNPACK)
  list(APPEND ORT_PROVIDER_FLAGS -DUSE_XNNPACK=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES xnnpack)
endif()
if (onnxruntime_USE_WEBNN)
  list(APPEND ORT_PROVIDER_FLAGS -DUSE_WEBNN=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES webnn)
endif()
if (onnxruntime_USE_WEBGPU)
  list(APPEND ORT_PROVIDER_FLAGS -DUSE_WEBGPU=1)
  list(APPEND ONNXRUNTIME_PROVIDER_NAMES webgpu)

  # TODO: the following code is used to disable building Dawn using vcpkg temporarily
  # until we figure out how to resolve the packaging pipeline failures
  #
  # if (onnxruntime_USE_VCPKG AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (FALSE)
    if (onnxruntime_BUILD_DAWN_SHARED_LIBRARY)
      message(FATAL_ERROR "onnxruntime_USE_VCPKG is not supported with onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON")
    endif()
    if (onnxruntime_USE_EXTERNAL_DAWN)
      message(FATAL_ERROR "onnxruntime_USE_VCPKG is not supported with onnxruntime_USE_EXTERNAL_DAWN=ON")
    endif()
    if (onnxruntime_CUSTOM_DAWN_SRC_PATH)
      message(FATAL_ERROR "onnxruntime_USE_VCPKG is not supported with a custom dawn source path")
    endif()
    if (WIN32)
      if (onnxruntime_ENABLE_DAWN_BACKEND_VULKAN)
        message(FATAL_ERROR "onnxruntime_USE_VCPKG is not supported with onnxruntime_ENABLE_DAWN_BACKEND_VULKAN=ON")
      endif()
      if (NOT onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
        message(FATAL_ERROR "onnxruntime_USE_VCPKG is not supported with onnxruntime_ENABLE_DAWN_BACKEND_D3D12=OFF")
      endif()
      if (onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP)
        message(FATAL_ERROR "onnxruntime_USE_VCPKG is not supported with onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP=ON")
      endif()
    endif()
  endif()

  if (onnxruntime_BUILD_DAWN_SHARED_LIBRARY)
    if (NOT onnxruntime_USE_EP_API_ADAPTERS)
      list(APPEND ORT_PROVIDER_FLAGS -DBUILD_DAWN_SHARED_LIBRARY=1)
    else()
      message(FATAL_ERROR "onnxruntime_USE_EP_API_ADAPTERS=ON is not supported with onnxruntime_BUILD_DAWN_SHARED_LIBRARY=ON")
    endif()
  endif()

  if (onnxruntime_USE_EXTERNAL_DAWN)
    list(APPEND ORT_PROVIDER_FLAGS -DUSE_EXTERNAL_DAWN=1)
  endif()
  if ((onnxruntime_ENABLE_DAWN_BACKEND_VULKAN AND WIN32) OR LINUX OR ANDROID)
    list(APPEND ORT_PROVIDER_FLAGS -DDAWN_ENABLE_VULKAN=1)
  endif()
  if (onnxruntime_ENABLE_DAWN_BACKEND_D3D12 AND WIN32)
    list(APPEND ORT_PROVIDER_FLAGS -DDAWN_ENABLE_D3D12=1)
  endif()
  if (onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP)
    if (NOT onnxruntime_ENABLE_DAWN_BACKEND_D3D12 OR NOT WIN32)
      message(
        FATAL_ERROR
        "Option onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP can only be set on windows with onnxruntime_ENABLE_DAWN_BACKEND_D3D12 is enabled.")
    endif()
    add_compile_definitions(ENABLE_PIX_FOR_WEBGPU_EP)
  endif()
endif()
if (onnxruntime_USE_CANN)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_CANN=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES cann)
endif()
if (onnxruntime_USE_AZURE)
    list(APPEND ORT_PROVIDER_FLAGS  -DUSE_AZURE=1)
    list(APPEND ONNXRUNTIME_PROVIDER_NAMES azure)
endif()
if (onnxruntime_USE_LOCK_FREE_QUEUE)
    add_compile_definitions(USE_LOCK_FREE_QUEUE)
endif()

if (onnxruntime_FORCE_GENERIC_ALGORITHMS)
    add_compile_definitions(FORCE_GENERIC_ALGORITHMS)
endif()

if (onnxruntime_ENABLE_LAZY_TENSOR)
  # To support LazyTensor, ORT needs to call Python function from C/C++.
  # so onnxruntime_ENABLE_PYTHON is required.
  if (NOT onnxruntime_ENABLE_TRAINING OR NOT onnxruntime_ENABLE_PYTHON)
    message(
        FATAL_ERROR
        "Option onnxruntime_ENABLE_LAZY_TENSOR can only be set when onnxruntime_ENABLE_TRAINING and onnxruntime_ENABLE_PYTHON are enabled")
  endif()
  # TODO: In the future, we can compile LazyTensor into a standalone
  # library target, onnxruntime_lazy_tensor, to make the buid
  # cleaner.
endif()

function(onnxruntime_set_compile_flags target_name)
    if (CPUINFO_SUPPORTED)
      onnxruntime_add_include_to_target(${target_name} cpuinfo::cpuinfo)
    endif()
    if(onnxruntime_ENABLE_LAZY_TENSOR)
      target_compile_definitions(${target_name} PRIVATE ENABLE_LAZY_TENSOR)
    endif()
	# Enable stream for all the non-minimal build unless disabled
    if (NOT onnxruntime_MINIMAL_BUILD AND NOT onnxruntime_DISABLE_STREAM)
      target_compile_definitions(${target_name} PRIVATE ORT_ENABLE_STREAM)
    endif()
	if (onnxruntime_REDUCED_OPS_BUILD)
      target_compile_definitions(${target_name} PRIVATE REDUCED_OPS_BUILD)
    endif()
	if (onnxruntime_DISABLE_EXTERNAL_INITIALIZERS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_EXTERNAL_INITIALIZERS)
    endif()
    if (onnxruntime_ENABLE_CPU_FP16_OPS)
      target_compile_definitions(${target_name} PRIVATE ENABLE_CPU_FP16_TRAINING_OPS)
    endif()
    if(onnxruntime_DISABLE_ABSEIL)
      target_compile_definitions(${target_name} PRIVATE DISABLE_ABSEIL)
    endif()
    if(UNIX)
      target_compile_definitions(${target_name} PRIVATE PLATFORM_POSIX)
    endif()
    target_compile_definitions(${target_name} PRIVATE EIGEN_USE_THREADS)
    if (onnxruntime_DISABLE_CONTRIB_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_CONTRIB_OPS)
    endif()
    if (onnxruntime_ENABLE_GQA_VALUE_LAYOUT)
      target_compile_definitions(${target_name} PRIVATE ORT_ENABLE_GQA_VALUE_LAYOUT)
    endif()

    if (onnxruntime_DISABLE_ML_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_ML_OPS)
    endif()

    if (onnxruntime_DISABLE_GENERATION_OPS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_GENERATION_OPS)
    endif()

    if (onnxruntime_DISABLE_SPARSE_TENSORS)
      target_compile_definitions(${target_name} PRIVATE DISABLE_SPARSE_TENSORS)
    endif()

    if (onnxruntime_DISABLE_OPTIONAL_TYPE)
      target_compile_definitions(${target_name} PRIVATE DISABLE_OPTIONAL_TYPE)
    endif()

    if (onnxruntime_DISABLE_FLOAT8_TYPES)
      target_compile_definitions(${target_name} PRIVATE DISABLE_FLOAT8_TYPES)
    endif()

    if (onnxruntime_DISABLE_FLOAT4_TYPES)
      target_compile_definitions(${target_name} PRIVATE DISABLE_FLOAT4_TYPES)
    endif()

    if (onnxruntime_DISABLE_STRING_TYPE)
      target_compile_definitions(${target_name} PRIVATE DISABLE_STRING_TYPE)
    endif()

    if (onnxruntime_ENABLE_ATEN)
      target_compile_definitions(${target_name} PRIVATE ENABLE_ATEN)
    endif()
    # TODO: Narrow scope for Kleidiai compile
    if (onnxruntime_USE_KLEIDIAI)
      target_compile_definitions(${target_name} PRIVATE USE_KLEIDIAI)
    endif()

    set_target_properties(${target_name} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    if (onnxruntime_USE_CUDA)
      # Suppress a "conversion_function_not_usable" warning in gsl/span
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe \"--diag_suppress=conversion_function_not_usable\">")
    endif()
    if (MSVC)
      foreach(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORY}>")
      endforeach()
      if(VCPKG_INSTALLED_DIR)
        cmake_path(NATIVE_PATH VCPKG_INSTALLED_DIR NORMALIZE VCPKG_INSTALLED_DIR_native)
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:SHELL:/external:I ${VCPKG_INSTALLED_DIR_native}>")
      endif()
      foreach(onnxruntime_external_lib IN LISTS onnxruntime_EXTERNAL_LIBRARIES)
        #TODO: the list contains cmake keywords like "debug". We should exclude them.
        if(TARGET ${onnxruntime_external_lib})
          get_target_property(onnxruntime_external_lib_include_dirs ${onnxruntime_external_lib} INTERFACE_INCLUDE_DIRECTORIES)
          foreach(onnxruntime_external_lib_include_dir IN LISTS onnxruntime_external_lib_include_dirs)
            if(onnxruntime_external_lib_include_dir MATCHES "^\\$")
              if(onnxruntime_external_lib_include_dir MATCHES "^\\$<BUILD_INTERFACE:([^>]+)>$")
                string(REGEX REPLACE "^\\$<BUILD_INTERFACE:([^>]+)>$" "\\1" onnxruntime_external_lib_include_dir_cmake "${onnxruntime_external_lib_include_dir}")
                cmake_path(NATIVE_PATH onnxruntime_external_lib_include_dir_cmake NORMALIZE onnxruntime_external_lib_include_dir_native)
                target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${onnxruntime_external_lib_include_dir_native}>")
              endif()
            else()
              cmake_path(NATIVE_PATH onnxruntime_external_lib_include_dir NORMALIZE onnxruntime_external_lib_include_dir_native)
              target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${onnxruntime_external_lib_include_dir_native}>")
            endif()
          endforeach()
        endif()
      endforeach()
      target_compile_definitions(${target_name} PRIVATE -DPLATFORM_WINDOWS -DNOGDI -DNOMINMAX -D_USE_MATH_DEFINES -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
      if (onnxruntime_ENABLE_MEMLEAK_CHECKER)
        target_compile_definitions(${target_name} PRIVATE -DONNXRUNTIME_ENABLE_MEMLEAK_CHECK)
      endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<COMPILE_LANGUAGE:CXX,C>:/utf-8>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /sdl>" "$<$<COMPILE_LANGUAGE:CXX,C>:/sdl>")
      set_target_properties(${target_name}
                      PROPERTIES VS_GLOBAL_CAExcludePath "${ORT_BINARY_DIR};${ORT_SOURCE_DIR}")
      # We do not treat warnings from 3rd-party libraries as errors. In order to do that, we need to add their header files locations to /external:I.
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/experimental:external>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /experimental:external>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:W0>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:W0>")
      #if(NOT onnxruntime_USE_VCPKG)
      #  target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:templates->" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:templates->")
      #endif()
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CURRENT_SOURCE_DIR}>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:I${CMAKE_CURRENT_SOURCE_DIR}>")
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX,C>:/external:I${CMAKE_CURRENT_BINARY_DIR}>" "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /external:I${CMAKE_CURRENT_BINARY_DIR}>")
      if (onnxruntime_ENABLE_STATIC_ANALYSIS)
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze>" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze>")
        if (onnxruntime_REDIRECT_STATIC_ANALYSIS_OUTPUTS_TO_FILE)
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze:autolog:ext.sarif>" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze:autolog:ext.sarif>")
        endif()
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /analyze:external->" "$<$<COMPILE_LANGUAGE:CXX,C>:/analyze:external->")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6385>" )
        # There are many such warnings from STL:
        # include\list(148): warning C6011: Dereferencing NULL pointer '_Mycont'. : Lines: 146, 147, 148
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /wd6011>" )
      endif()
    else()
      # Enable warning
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wall>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wall>")
      target_compile_options(${target_name} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wextra>")
      if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND onnxruntime_target_platform STREQUAL "aarch64")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-psabi>")
      endif()
      if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IBMClang")
        #external/protobuf/src/google/protobuf/arena.h:445:18: error: unused parameter 'p'
        target_compile_options(${target_name} PRIVATE "-Wno-unused-parameter")
      endif()
      if (CMAKE_CXX_COMPILER_ID STREQUAL "IBMClang")
        target_compile_options(${target_name} PRIVATE "-Wno-unused-function")
      endif()
    endif()
    foreach(ORT_FLAG ${ORT_PROVIDER_FLAGS})
      target_compile_definitions(${target_name} PRIVATE ${ORT_FLAG})
    endforeach()

    if (HAS_DEPRECATED_COPY)
      #too many such errors in eigen
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options -Wno-deprecated-copy>" "$<$<COMPILE_LANGUAGE:CXX>:-Wno-deprecated-copy>")
    endif()
    foreach(FLAG ${ORT_WARNING_FLAGS})
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${FLAG}>")
    endforeach()
    if("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_C_COMPILER_VERSION VERSION_LESS 13 AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
      target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>")
    endif()
    if (onnxruntime_USE_CUDA)
      foreach(FLAG ${ORT_WARNING_FLAGS})
        if (NOT "${FLAG}" STREQUAL "-Wshorten-64-to-32")
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options ${FLAG}>")
        endif()
      endforeach()
      if (NVCC_HAS_STRICT_ALIASING AND "${target_name}" MATCHES "cuda")
        target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Wno-strict-aliasing>")
      endif()
      if (HAS_STRICT_ALIASING AND NOT "${target_name}" MATCHES "cuda")
          target_compile_options(${target_name} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:-Wno-strict-aliasing>")
      endif()
    endif()
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
  if(WIN32 AND onnxruntime_ENABLE_STATIC_ANALYSIS AND onnxruntime_USE_CUSTOM_STATIC_ANALYSIS_RULES)
    set_target_properties(${target_name} PROPERTIES VS_USER_PROPS ${PROJECT_SOURCE_DIR}/EnableVisualStudioCodeAnalysis.props)
  endif()
  target_include_directories(${target_name} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${ONNXRUNTIME_ROOT})
  if (onnxruntime_ENABLE_TRAINING_OPS)
    target_include_directories(${target_name} PRIVATE ${ORTTRAINING_ROOT})
  endif()
  if (onnxruntime_ENABLE_LTO)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
    set_target_properties(${target_name} PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
  endif()

  # Keep BinSkim happy
  if(MSVC AND NOT onnxruntime_target_platform MATCHES "ARM")
    target_link_options(${target_name} PRIVATE "$<$<LINK_LANGUAGE:CXX,C>:/CETCOMPAT>")
  endif()

  # Add compile features for minimum required language standard versions.
  # Wrap in $<BUILD_INTERFACE:...> so the requirement is not exported to consumers of installed ORT targets.
  target_compile_features(${target_name} PUBLIC
                          $<BUILD_INTERFACE:${onnxruntime_c_std_compile_feature}>
                          $<BUILD_INTERFACE:${onnxruntime_cxx_std_compile_feature}>)
endfunction()

function(onnxruntime_add_shared_library target_name)
  add_library(${target_name} SHARED ${ARGN})
  onnxruntime_configure_target(${target_name})
  if(WIN32)
        target_compile_definitions(${target_name} PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_MINOR=${VERSION_MINOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_BUILD=${VERSION_BUILD_PART})
        target_compile_definitions(${target_name} PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
        target_compile_definitions(${target_name} PRIVATE VER_STRING=\"${VERSION_STRING}\")
  endif()
endfunction()

function(onnxruntime_add_static_library target_name)
  add_library(${target_name} STATIC ${ARGN})
  onnxruntime_configure_target(${target_name})
endfunction()

#For plugins that are not linked into other targets but may be loaded dynamically at runtime using dlopen-like functionality.
function(onnxruntime_add_shared_library_module target_name)
  if ((${CMAKE_SYSTEM_NAME} MATCHES "Darwin") OR (${CMAKE_SYSTEM_NAME} MATCHES "iOS"))
    add_library(${target_name} SHARED ${ARGN})
  else()
    #On Windows, this target shouldn't generate an import lib, but I don't know how to disable it.
    add_library(${target_name} MODULE ${ARGN})
    if(WIN32)
        target_compile_definitions(${target_name} PRIVATE VER_MAJOR=${VERSION_MAJOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_MINOR=${VERSION_MINOR_PART})
        target_compile_definitions(${target_name} PRIVATE VER_BUILD=${VERSION_BUILD_PART})
        target_compile_definitions(${target_name} PRIVATE VER_PRIVATE=${VERSION_PRIVATE_PART})
        target_compile_definitions(${target_name} PRIVATE VER_STRING=\"${VERSION_STRING}\")
    endif()
  endif()

  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_object_library target_name)
  add_library(${target_name} OBJECT ${ARGN})

  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_executable target_name)
  add_executable(${target_name} ${ARGN})
  onnxruntime_configure_target(${target_name})
  if (MSVC AND onnxruntime_target_platform STREQUAL "x86")
    target_link_options(${target_name} PRIVATE /SAFESEH)
  endif()
endfunction()

function(onnxruntime_add_include_to_target dst_target)
    foreach(src_target ${ARGN})
      if(TARGET ${src_target})
        target_include_directories(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_INCLUDE_DIRECTORIES>)
        target_compile_definitions(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_COMPILE_DEFINITIONS>)
        target_sources(${dst_target} PRIVATE $<TARGET_PROPERTY:${src_target},INTERFACE_SOURCES>)
      endif()
    endforeach()
endfunction()

# ACL
if (onnxruntime_USE_ACL)
  set(onnxruntime_USE_ACL ON)

  if (NOT ${onnxruntime_ACL_LIBS} STREQUAL "")
    add_library(arm_compute SHARED IMPORTED)
    set_target_properties(arm_compute PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute.so")

    add_library(arm_compute_graph SHARED IMPORTED)
    set_target_properties(arm_compute_graph PROPERTIES
        IMPORTED_NO_SONAME 1
        IMPORTED_LOCATION "${onnxruntime_ACL_LIBS}/libarm_compute_graph.so")
  endif()

  list(APPEND onnxruntime_EXTERNAL_LIBRARIES arm_compute arm_compute_graph)

endif()

if (onnxruntime_USE_DNNL)
  include(dnnl)
  add_compile_definitions(DNNL_OPENMP)
endif()

# onnxruntime-extensions
if (onnxruntime_USE_EXTENSIONS)
  include(extensions)
endif()

#Dependencies end. In the next we'll enable "treat warning as error"

#Adjust warning flags
set_msvc_c_cpp_compiler_warning_level(4)

set(onnxruntime_DELAYLOAD_FLAGS )

include_directories(
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${REPO_ROOT}/include/onnxruntime/core/session
)

if (onnxruntime_ENABLE_TRAINING_APIS)
  include_directories(
    ${REPO_ROOT}/orttraining/orttraining/training_api/include/
  )
endif()

if (onnxruntime_USE_OPENVINO)

  add_definitions(-DUSE_OPENVINO=1)

  if(onnxruntime_NPU_NO_FALLBACK)
    add_definitions(-DOPENVINO_CONFIG_NPU=1)
    add_definitions(-DOPENVINO_DISABLE_NPU_FALLBACK=1)
  endif()

  if (onnxruntime_USE_OPENVINO_GPU)
    add_definitions(-DOPENVINO_CONFIG_GPU=1)
  endif()

  if (onnxruntime_USE_OPENVINO_CPU)
    add_definitions(-DOPENVINO_CONFIG_CPU=1)
  endif()

  if (onnxruntime_USE_OPENVINO_NPU)
    add_definitions(-DOPENVINO_CONFIG_NPU=1)
  endif()

  if (onnxruntime_USE_OPENVINO_GPU_NP)
    add_definitions(-DOPENVINO_CONFIG_GPU=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_CPU_NP)
    add_definitions(-DOPENVINO_CONFIG_CPU=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_NPU_NP)
    add_definitions(-DOPENVINO_CONFIG_NPU=1)
    add_definitions(-DOPENVINO_DISABLE_GRAPH_PARTITION=1)
  endif()

  if (onnxruntime_USE_OPENVINO_HETERO)
    add_definitions(-DOPENVINO_CONFIG_HETERO=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if (onnxruntime_USE_OPENVINO_MULTI)
    add_definitions(-DOPENVINO_CONFIG_MULTI=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if(onnxruntime_USE_OPENVINO_AUTO)
    add_definitions(-DOPENVINO_CONFIG_AUTO=1)
    add_definitions(-DDEVICE_NAME="${onnxruntime_USE_OPENVINO_DEVICE}")
  endif()

  if($ENV{FIL_ENABLED})
    add_definitions(-DOPENVINO_FIL_ENABLED=1)
  endif()

endif()

if (onnxruntime_USE_VITISAI)
    set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_LIST_DIR}")
endif()

set(ORT_BUILD_INFO "ORT Build Info: ")
find_package(Git)
if (Git_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ORT_GIT_COMMIT)
  string(STRIP "${ORT_GIT_COMMIT}" ORT_GIT_COMMIT)
  execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --abbrev-ref HEAD
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      OUTPUT_VARIABLE ORT_GIT_BRANCH)
  string(STRIP "${ORT_GIT_BRANCH}" ORT_GIT_BRANCH)
  string(APPEND ORT_BUILD_INFO "git-branch=${ORT_GIT_BRANCH}, git-commit-id=${ORT_GIT_COMMIT}, ")
  if (onnxruntime_QUICK_BUILD)
    string(APPEND ORT_BUILD_INFO "quick-build=1, ")
  endif()
  if (onnxruntime_USE_CUDA AND onnxruntime_USE_INT4_KV_CACHE)
    string(APPEND ORT_BUILD_INFO "int4-kv-cache=1, ")
  endif()
  if (onnxruntime_USE_FP8_KV_CACHE)
    string(APPEND ORT_BUILD_INFO "fp8-kv-cache=1, ")
  endif()
  if (onnxruntime_USE_FP4_QMOE)
    string(APPEND ORT_BUILD_INFO "fp4-qmoe=1, ")
  endif()
  if (onnxruntime_USE_FP8_QMOE)
    string(APPEND ORT_BUILD_INFO "fp8-qmoe=1, ")
  endif()
  if (onnxruntime_USE_CUDA AND onnxruntime_BUILD_CUDA_EP_AS_PLUGIN)
    string(APPEND ORT_BUILD_INFO "cuda-plugin-ep=1, ")
  endif()
  if (onnxruntime_DUMP_TENSOR)
    string(APPEND ORT_BUILD_INFO "dump-tensor=1, ")
  endif()
  if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
    string(APPEND ORT_BUILD_INFO "dump-node=1, ")
  endif()
endif()
string(APPEND ORT_BUILD_INFO "build type=${CMAKE_BUILD_TYPE}")
configure_file(onnxruntime_config.h.in ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_config.h)
get_property(onnxruntime_GENERATOR_IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)

if (onnxruntime_USE_CUDA)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  if(onnxruntime_CUDA_HOME)
    file(TO_CMAKE_PATH ${onnxruntime_CUDA_HOME} CUDAToolkit_ROOT)
  endif()
  find_package(CUDAToolkit REQUIRED)

  if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.9
          AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    foreach(_cuda_include_dir IN LISTS CUDAToolkit_INCLUDE_DIRS)
      set(_clusterlaunchcontrol_header
          "${_cuda_include_dir}/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h")
      if(EXISTS "${_clusterlaunchcontrol_header}")
        # CUDA 12.9 uses Windows's 32-bit long for PTX 64-bit operands. Shadow the toolkit
        # header with the longlong2 correction shipped by newer CCCL releases.
        set(_cuda_12_9_fix_dir "${CMAKE_CURRENT_BINARY_DIR}/cuda_12_9_fix")
        ort_patch_cuda_12_9_clusterlaunchcontrol_header(
          "${_clusterlaunchcontrol_header}"
          "${_cuda_12_9_fix_dir}/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h")
        if(EXISTS "${_cuda_12_9_fix_dir}/cuda/__ptx/instructions/generated/clusterlaunchcontrol.h")
          include_directories(BEFORE "${_cuda_12_9_fix_dir}")
        endif()
        break()
      endif()
    endforeach()
  endif()

  # Note: The minimum required CUDA version is greater than 11.8.
  add_definitions("-DENABLE_BF16")
  message(STATUS "CUDA Toolkit version is greater or equal than 11.8, enable -DENABLE_BF16 flag")
  add_definitions("-DENABLE_FP8")
  message(STATUS "CUDA Toolkit version is greater or equal than 11.8, enable -DENABLE_FP8 flag")
  if(onnxruntime_USE_FP8_QMOE)
    add_definitions("-DUSE_FP8_QMOE")
    message(STATUS "CUDA FP8 QMoE kernels enabled")
  endif()

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    add_definitions("-DENABLE_FP4")
    message(STATUS "CUDA Toolkit version is greater or equal than 12.8, enable -DENABLE_FP4 flag")
    if(onnxruntime_USE_FP4_QMOE)
      add_definitions("-DUSE_FP4_QMOE")
      message(STATUS "CUDA FP4 QMoE kernels enabled")
    endif()
  elseif(onnxruntime_USE_FP4_QMOE)
    message(FATAL_ERROR "onnxruntime_USE_FP4_QMOE requires CUDA Toolkit version 12.8 or newer")
  endif()

  if(onnxruntime_CUDA_FATBIN_COMPRESS_SIZE AND CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "12.8")
    message(FATAL_ERROR "onnxruntime_CUDA_FATBIN_COMPRESS_SIZE requires CUDA Toolkit version 12.8 or newer")
  endif()

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL "13.0" OR onnxruntime_CUDA_FATBIN_COMPRESS_SIZE)
    message(STATUS "Compressing CUDA fatbins with -compress-mode=size (requires a CUDA 12.4 or newer driver)")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin=-compress-all -compress-mode=size")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xfatbin=-compress-all")
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Werror default-stream-launch")

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    if (UNIX)
      # Suppress deprecation errors (e.g., long4 in CUDA 13)
      add_compile_options(-Wno-deprecated-declarations)
    endif()
  endif()

  if (NOT WIN32)
    list(APPEND CUDA_NVCC_FLAGS --compiler-options -fPIC)
  endif()
  if(MSVC)
    if(CUDA_NVCC_FLAGS MATCHES "Zi")
      list(APPEND CUDA_NVCC_FLAGS "-Xcompiler" "-FS")
    endif()
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
  set(AMD_MIGRAPHX_HOME ${onnxruntime_MIGRAPHX_HOME})
endif()

if (onnxruntime_ENABLE_MICROSOFT_INTERNAL)
  add_definitions(-DMICROSOFT_INTERNAL)
endif()

if (onnxruntime_USE_DML)
  if (NOT WIN32)
    message(FATAL_ERROR "The DirectML execution provider is only supported when building for Windows.")
  endif()

  include(dml)
endif()

if (onnxruntime_ENABLE_TRAINING_APIS)
  add_compile_definitions(ENABLE_TRAINING_CORE)
  add_compile_definitions(ENABLE_TRAINING_APIS)
endif()

if (onnxruntime_ENABLE_TRAINING_OPS)
  add_compile_definitions(ENABLE_TRAINING_OPS)
endif()

if (onnxruntime_ENABLE_CUDA_PROFILING)
  add_compile_definitions(ENABLE_CUDA_PROFILING)
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_compile_definitions(ENABLE_TRAINING_CORE)
  add_compile_definitions(ENABLE_STRIDED_TENSORS)
  add_compile_definitions(ENABLE_TRAINING)

  add_subdirectory(tensorboard EXCLUDE_FROM_ALL)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES tensorboard)
endif()

if (onnxruntime_ENABLE_DLPACK)
  add_compile_definitions(ENABLE_DLPACK)
endif()

if (onnxruntime_CALLER_FRAMEWORK)
  add_definitions(-DORT_CALLER_FRAMEWORK="${onnxruntime_CALLER_FRAMEWORK}")
endif()

if (UNIX OR onnxruntime_USE_NCCL)
  # Find NCCL
  if (onnxruntime_USE_NCCL)
    if (onnxruntime_USE_CUDA)
      set(NCCL_LIBNAME "nccl")
    elseif (onnxruntime_USE_MIGRAPHX)
      set(NCCL_LIBNAME "rccl")
    endif()
    find_path(NCCL_INCLUDE_DIR
      NAMES ${NCCL_LIBNAME}.h
      HINTS
      ${onnxruntime_NCCL_HOME}/include/rccl
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
      message(
          FATAL_ERROR
          "NCCL is not found. Please use --nccl_home to specify the path of NCCL. Otherwise, NCCL is disabled."
        )
    endif()
  endif()
else()
  set(onnxruntime_USE_NCCL OFF)
  message( WARNING "NCCL is disabled because build is on Windows or USE_NCCL is set to OFF." )
endif()


# Default version parts for Microsoft.AI.MachineLearning.dll, onnxruntime.dll, onnxruntime_providers_openvino.dll and onnxruntime_providers_shared.dll in non-ADO pipeline local builds
set(VERSION_MAJOR_PART   0 CACHE STRING "First part of numeric file/product version.")
set(VERSION_MINOR_PART   0 CACHE STRING "Second part of numeric file/product version.")
set(VERSION_BUILD_PART       0 CACHE STRING "Third part of numeric file/product version.")
set(VERSION_PRIVATE_PART     0 CACHE STRING "Fourth part of numeric file/product version.")
set(VERSION_STRING       "Internal Build" CACHE STRING "String representation of file/product version.")
if(VERSION_MAJOR_PART STREQUAL "0" AND VERSION_MINOR_PART STREQUAL "0" AND VERSION_BUILD_PART STREQUAL "0" AND VERSION_PRIVATE_PART STREQUAL "0")
    string(REPLACE "." ";"  ORT_VERSION_STRING_LIST ${ORT_VERSION})
    list(GET ORT_VERSION_STRING_LIST 0 VERSION_MAJOR_PART)
    list(GET ORT_VERSION_STRING_LIST 1 VERSION_MINOR_PART)
    list(GET ORT_VERSION_STRING_LIST 2 VERSION_BUILD_PART)
    set(VERSION_STRING ${ORT_VERSION})
endif()


if (WIN32)
  if (onnxruntime_USE_MIMALLOC)
   list(APPEND onnxruntime_EXTERNAL_LIBRARIES mimalloc-static)
  endif()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SYS_PATH_LIB})
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES debug Dbghelp)
  # In a onecore build the umbrella libs already contains references to the APIs in advapi32, so in onecore build we do not need to link to advapi32
  # In a non-onecore build, usually we also do not need to link to advapi32 because VC++ by default should have provide everything we need, except when the build target is Windows ARM32.
  # In the future we will add a build option to allow users disabling all API uses from advapi32 because some Windows environments do not have these APIs. For example, some Windows do not have
  # Windows Registry so we cannot query Registry values.
  if(onnxruntime_target_platform STREQUAL "ARM" AND CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES advapi32)
  endif()
else()
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${ICONV_LIB} ${CMAKE_DL_LIBS} Threads::Threads)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES log)
endif()

# check if we need to link against librt on Linux
include(CheckLibraryExists)
include(CheckFunctionExists)
if ("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)

  if (NOT HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES time.h)
    check_function_exists(clock_gettime HAVE_CLOCK_GETTIME)
    set(CMAKE_EXTRA_INCLUDE_FILES)
  else()
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES rt)
  endif()
endif()

# From this point on, onnxruntime_EXTERNAL_LIBRARIES should be no longer be modified. It will be used in
# onnxruntime.cmake which will be included next.

# The order of the CMake file names (which exclude the ".cmake" suffix) in ONNXRUNTIME_CMAKE_FILES matters. Later CMake
# files may depend on earlier ones but earlier ones cannot depend on later ones. For example, if A appears before B,
# then A.cmake cannot use variables defined in B.cmake.
set(ONNXRUNTIME_CMAKE_FILES
    onnxruntime_flatbuffers
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_graph
    onnxruntime_lora
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_providers
    onnxruntime_optimizer
    onnxruntime_session
    ${ONNXRUNTIME_EAGER_CMAKE_FILE_NAME}
    )

if (onnxruntime_USE_WINML)
  # WINML uses and depends on the shared lib.  Note:  You can build WINML without DML and you will get a
  # CPU only WINML
  if (NOT onnxruntime_BUILD_SHARED_LIB)
    message(
        FATAL_ERROR
        "Option onnxruntime_USE_WINML can only be used when onnxruntime_BUILD_SHARED_LIB is also enabled")
  endif()
  list(APPEND ONNXRUNTIME_CMAKE_FILES winml)
endif() # if (onnxruntime_USE_WINML)

list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime)

if (onnxruntime_BUILD_JAVA)
  message(STATUS "Java Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_java)
endif()

if (onnxruntime_BUILD_NODEJS)
  message(STATUS "Node.js Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_nodejs)
endif()

if (onnxruntime_ENABLE_PYTHON)
  message(STATUS "Python Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_python)
endif()

if (onnxruntime_BUILD_OBJC)
  message(STATUS "Objective-C Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_objectivec)
endif()

if (onnxruntime_BUILD_UNIT_TESTS)
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_unittests)
endif()

if (onnxruntime_BUILD_WINML_TESTS)
  list(APPEND ONNXRUNTIME_CMAKE_FILES winml_unittests)
endif()

# onnxruntime_training depends on onnxruntime_unittests since onnxruntime_training.cmake uses a variable `TEST_SRC_DIR`
# that is defined in onnxruntime_unittests.cmake
if (onnxruntime_ENABLE_TRAINING)
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_training)
  if (onnxruntime_ENABLE_TRAINING_E2E_TESTS)
    list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_training_e2e_tests)
  endif()
endif()

if (onnxruntime_BUILD_CSHARP)
  message(STATUS "CSharp Build is enabled")
#  set_property(GLOBAL PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "netstandard2.0")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_csharp)
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  message(STATUS "WebAssembly Build is enabled")
  list(APPEND ONNXRUNTIME_CMAKE_FILES onnxruntime_webassembly)

  if (onnxruntime_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL)
    add_compile_definitions(ORT_ENABLE_WEBASSEMBLY_OUTPUT_OPTIMIZED_MODEL)
  endif()
endif()

# When GDK_PLATFORM is set then WINAPI_FAMILY is defined in gdk_toolchain.cmake (along with other relevant flags/definitions).
if (WIN32 AND NOT GDK_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
  if (NOT CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
    # On onecore, link to the onecore build of the MSVC runtime
    get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
    link_directories(BEFORE "${msvc_path}/lib/onecore/${onnxruntime_target_platform}")
    # The .lib files in the MSVC runtime have a DEFAULITLIB entry for onecore.lib, but it shold not cause any conflict with onecoreuap.lib
  endif()
endif()

foreach(onnxruntime_cmake_file ${ONNXRUNTIME_CMAKE_FILES})
  include(${onnxruntime_cmake_file}.cmake)
endforeach()

if (UNIX)
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" ON)
else()
  option(BUILD_PKGCONFIG_FILES "Build and install pkg-config files" OFF)
endif()
if (BUILD_PKGCONFIG_FILES)
   configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libonnxruntime.pc.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc @ONLY)
  install( FILES  ${CMAKE_CURRENT_BINARY_DIR}/libonnxruntime.pc DESTINATION
    ${CMAKE_INSTALL_LIBDIR}/pkgconfig )
endif()

if (onnxruntime_BUILD_OPSCHEMA_LIB AND onnxruntime_ENABLE_TRAINING)
  # opschema library requires training ops as well
  include(onnxruntime_opschema_lib.cmake)
endif()

if (onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS)
  add_compile_definitions(DEBUG_NODE_INPUTS_OUTPUTS)
endif()

if (onnxruntime_DUMP_TENSOR)
  add_compile_definitions(DUMP_TENSOR_LEVEL=1)
endif()

if (onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS)
  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "External custom operator schemas feature is only supported on Linux")
  endif()

  if (NOT ${ONNX_CUSTOM_PROTOC_EXECUTABLE} STREQUAL "")
    message(FATAL_ERROR "External custom operator schemas is not supported with the user specified protoc executable")
  endif()

  if (NOT onnxruntime_ENABLE_TRAINING)
    message(FATAL_ERROR "External custom operator schemas is supported only with --enable-training option")
  endif()

  add_custom_target(install_protobuf ALL DEPENDS ${PROTOBUF_LIB} protobuf::protoc)
  add_custom_command(
    TARGET install_protobuf
    COMMAND ${CMAKE_COMMAND} -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR} -P cmake_install.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/external/protobuf/cmake
    COMMENT "Installing protobuf"
  )
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB AND
    (onnxruntime_USE_WEBGPU OR (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_USE_XNNPACK)))
  message(WARNING
    "CMake target files will not be generated for static onnxruntime builds "
    "with WebGPU or Emscripten+XNNPACK support")
else()
  # Install
  include(CMakePackageConfigHelpers)

  set(PROJECT_CONFIG_CONTENT "@PACKAGE_INIT@\n")

  if (NOT onnxruntime_BUILD_SHARED_LIB)
  string(APPEND PROJECT_CONFIG_CONTENT
        "include(CMakeFindDependencyMacro)\n\
        find_dependency(absl)\n\
        find_dependency(date)\n\
        find_dependency(Eigen3)\n\
        find_dependency(nlohmann_json)\n\
        find_dependency(ONNX)\n\
        find_dependency(re2)\n\
        find_dependency(flatbuffers)\n\
        find_dependency(cpuinfo)\n\
        find_dependency(protobuf)\n\
        find_dependency(Boost COMPONENTS mp11)\n\
        find_dependency(Microsoft.GSL 4.0)\n\
        if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL \"Android\")\n\
        find_dependency(Iconv)\n\
        endif()\n\
        if(WIN32)\n\
        find_dependency(wil)\n\
        endif()\n\
        find_path(safeint_SOURCE_DIR NAMES \"SafeInt.hpp\" REQUIRED)\n\
        add_library(safeint_interface IMPORTED INTERFACE)\n\
        target_include_directories(safeint_interface INTERFACE ${safeint_SOURCE_DIR})\n\
        ")
    include("${CMAKE_CURRENT_LIST_DIR}/onnxruntime_telemetry_config.cmake")
  endif()

  string(APPEND PROJECT_CONFIG_CONTENT
        "include(\"\${CMAKE_CURRENT_LIST_DIR}/${PROJECT_NAME}Targets.cmake\")\n")


  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/PROJECT_CONFIG_FILE" ${PROJECT_CONFIG_CONTENT})
  install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE ${PROJECT_NAME}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME})

  # Create config for find_package()
  configure_package_config_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PROJECT_CONFIG_FILE" ${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

  write_basic_package_version_file(
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${ORT_VERSION}
    COMPATIBILITY SameMajorVersion)

  install(
    FILES
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}")

  if(DEFINED BUILD_AS_ARM64X)
    set(ARM64X_TARGETS onnxruntime)

    # Add additional ARM64X build targets
    if (onnxruntime_USE_QNN AND NOT onnxruntime_BUILD_QNN_EP_STATIC_LIB)
      list(APPEND ARM64X_TARGETS onnxruntime_providers_shared)
      list(APPEND ARM64X_TARGETS onnxruntime_providers_qnn)
    endif()

    include("${CMAKE_CURRENT_SOURCE_DIR}/arm64x.cmake")
  endif()
endif()

if(onnxruntime_BUILD_UNIT_TESTS)
  include("${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_test_pch.cmake")
endif()

# Include precompiled header configuration for providers
if(TARGET ${onnxruntime_providers_target})
  include("${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime_providers_pch.cmake")
endif()

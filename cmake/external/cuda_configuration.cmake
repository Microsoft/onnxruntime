#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
#

macro(setup_cuda_compiler)
  # Determine CUDA version before enabling the language extension check_language(CUDA) clears CMAKE_CUDA_HOST_COMPILER
  # if CMAKE_CUDA_COMPILER is not set
  include(CheckLanguage)
  if(NOT CMAKE_CUDA_COMPILER AND CMAKE_CUDA_HOST_COMPILER)
    set(CMAKE_CUDA_HOST_COMPILER_BACKUP ${CMAKE_CUDA_HOST_COMPILER})
  endif()
  check_language(CUDA)
  if(CMAKE_CUDA_HOST_COMPILER_BACKUP)
    set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER_BACKUP})
    check_language(CUDA)
  endif()
  if(CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA compiler: ${CMAKE_CUDA_COMPILER}")
    if(NOT WIN32) # Linux
      execute_process(
        COMMAND "bash" "-c" "${CMAKE_CUDA_COMPILER} --version | grep -E -o 'V[0-9]+.[0-9]+.[0-9]+' | cut -c2-"
        RESULT_VARIABLE _BASH_SUCCESS
        OUTPUT_VARIABLE CMAKE_CUDA_COMPILER_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)

      if(NOT _BASH_SUCCESS EQUAL 0)
        message(FATAL_ERROR "Failed to determine CUDA version")
      endif()

    else() # Windows
      execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} --version
        OUTPUT_VARIABLE versionString
        RESULT_VARIABLE versionResult)

      if(versionResult EQUAL 0 AND versionString MATCHES "V[0-9]+\\.[0-9]+\\.[0-9]+")
        string(REGEX REPLACE "V" "" version ${CMAKE_MATCH_0})
        set(CMAKE_CUDA_COMPILER_VERSION "${version}")
      else()
        message(FATAL_ERROR "Failed to determine CUDA version")
      endif()
    endif()
  else()
    message(FATAL_ERROR "No CUDA compiler found")
  endif()

  set(CUDA_REQUIRED_VERSION 12.0)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS CUDA_REQUIRED_VERSION)
    message(FATAL_ERROR "CUDA version ${CMAKE_CUDA_COMPILER_VERSION} must be at least ${CUDA_REQUIRED_VERSION}")
  endif()

  # For CUDA 13+, explicitly set the compiler front-end to Clang to handle
  # MSVC-specific pragmas correctly in device code.
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0 AND NOT DEFINED CMAKE_CUDA_COMPILER_FRONTEND_VARIANT)
    message(STATUS "Setting CUDA compiler front-end to Clang by default for CUDA 13+.")
    set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "CLANG")
  endif()

  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Hybrid")
  else()
    set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
  endif()
endmacro()

function(ort_patch_cuda_12_9_clusterlaunchcontrol_header src dst)
  if(NOT EXISTS "${src}")
    return()
  endif()

  file(READ "${src}" _content)
  set(_orig "${_content}")
  string(REPLACE "reinterpret_cast<long2*>" "reinterpret_cast<longlong2*>" _content "${_content}")
  if(NOT _content STREQUAL _orig)
    get_filename_component(_dst_dir "${dst}" DIRECTORY)
    file(MAKE_DIRECTORY "${_dst_dir}")
    file(WRITE "${dst}" "${_content}")
  elseif(EXISTS "${dst}")
    file(REMOVE "${dst}")
  endif()
endfunction()

# Parse one CUDA architecture entry of the form <cc>[a|f][-real|-virtual] (for example "121a",
# "120f-real" or "90-virtual") and set, in the caller's scope:
#   <PREFIX>_CC       the numeric compute capability (e.g. "121"), or "" if ARCH is not of that form
#   <PREFIX>_SUFFIX   "a" (arch-specific), "f" (family-specific) or ""
#   <PREFIX>_VIRTUAL  TRUE for a -virtual entry (PTX only, no SASS), FALSE otherwise
function(onnxruntime_parse_cuda_arch ARCH PREFIX)
  if(ARCH MATCHES "^([1-9][0-9]+)([af]?)(-real|-virtual)?$")
    set(${PREFIX}_CC "${CMAKE_MATCH_1}" PARENT_SCOPE)
    set(${PREFIX}_SUFFIX "${CMAKE_MATCH_2}" PARENT_SCOPE)
    if(CMAKE_MATCH_3 STREQUAL "-virtual")
      set(${PREFIX}_VIRTUAL TRUE PARENT_SCOPE)
    else()
      set(${PREFIX}_VIRTUAL FALSE PARENT_SCOPE)
    endif()
  else()
    set(${PREFIX}_CC "" PARENT_SCOPE)
    set(${PREFIX}_SUFFIX "" PARENT_SCOPE)
    set(${PREFIX}_VIRTUAL FALSE PARENT_SCOPE)
  endif()
endfunction()

# Set OUTPUT_VAR to ON if any architecture in ARGN generates SASS for a compute capability in
# [MIN_CC, MAX_CC], and to OFF otherwise. Architectures are compared by compute capability, so
# every spelling of one matches: "121", "121a", "121f", "121-real", "121a-real" and "121f-real".
# -virtual entries never match: they embed only PTX, which provides no native kernels, and the
# features gated on this (TMA/WGMMA, block-scaled MMA) need arch- or family-specific SASS.
#
# Usage:
#   onnxruntime_cuda_archs_have_sass(<output_var> <min_cc> <max_cc> <arch>...)
function(onnxruntime_cuda_archs_have_sass OUTPUT_VAR MIN_CC MAX_CC)
  set(_result OFF)
  foreach(_arch IN LISTS ARGN)
    onnxruntime_parse_cuda_arch("${_arch}" _parsed)
    if(NOT _parsed_CC STREQUAL "" AND NOT _parsed_VIRTUAL AND
       _parsed_CC GREATER_EQUAL MIN_CC AND _parsed_CC LESS_EQUAL MAX_CC)
      set(_result ON)
      break()
    endif()
  endforeach()
  set(${OUTPUT_VAR} ${_result} PARENT_SCOPE)
endfunction()

# Set OUTPUT_VAR to ON if any architecture in ARGN generates code, SASS or PTX, for a compute
# capability of at least MIN_CC, and to OFF otherwise. Unlike onnxruntime_cuda_archs_have_sass,
# -virtual entries count: the driver JIT-compiles PTX for compute_<cc> on GPUs of compute
# capability <cc> or later (within the architecture or family for an "a" or "f" suffix), so
# kernels that use no architecture-specific features are reachable from PTX as well.
#
# Usage:
#   onnxruntime_cuda_archs_have_code(<output_var> <min_cc> <arch>...)
function(onnxruntime_cuda_archs_have_code OUTPUT_VAR MIN_CC)
  set(_result OFF)
  foreach(_arch IN LISTS ARGN)
    onnxruntime_parse_cuda_arch("${_arch}" _parsed)
    if(NOT _parsed_CC STREQUAL "" AND _parsed_CC GREATER_EQUAL MIN_CC)
      set(_result ON)
      break()
    endif()
  endforeach()
  set(${OUTPUT_VAR} ${_result} PARENT_SCOPE)
endfunction()

macro(setup_cuda_architectures)
  # cmake-format: off
  # Initialize and normalize CMAKE_CUDA_ARCHITECTURES before enabling CUDA.
  # Special values:
  # (1) `native` is resolved to HIGHEST available architecture. Fallback to `all` if detection failed.
  # (2) `all` / `all-major` / unset is resolved to a default set of architectures we optimized and compiler supports.
  # Numerical architectures:
  #  * For `-virtual` architectures, the last one is kept as it is, and the others are ignored.
  #  * `-real` suffix is automatically added for other cases.
  #  * Always use accelerated (`-a` suffix) target for supported real architectures.
  #  * An explicit `a` or `f` suffix is kept. Family-specific (`f`) entries are kept as given:
  #    `<cc>f` generates SASS and PTX, `<cc>f-real` SASS only.
  # cmake-format: on

  # Allow override via CUDAARCHS environment variable (standard CMake variable)
  if(NOT CMAKE_CUDA_ARCHITECTURES AND DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}")
  endif()

  if(CMAKE_CUDA_ARCHITECTURES STREQUAL "native")
    # Detect highest available compute capability
    set(OUTPUTFILE ${PROJECT_BINARY_DIR}/detect_cuda_arch)
    set(CUDAFILE ${CMAKE_SOURCE_DIR}/utils/detect_cuda_arch.cu)
    execute_process(COMMAND ${CMAKE_CUDA_COMPILER} -lcuda ${CUDAFILE} -o ${OUTPUTFILE})
    message(VERBOSE "Detecting native CUDA compute capability")
    execute_process(
      COMMAND ${OUTPUTFILE}
      RESULT_VARIABLE CUDA_RETURN_CODE
      OUTPUT_VARIABLE CUDA_ARCH_OUTPUT)
    if(NOT ${CUDA_RETURN_CODE} EQUAL 0)
      message(WARNING "Detecting native CUDA compute capability - fail")
      message(WARNING "CUDA compute capability detection failed, compiling for all optimized architectures")
      unset(CMAKE_CUDA_ARCHITECTURES)
    else()
      message(STATUS "Detecting native CUDA compute capability - done")
      set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH_OUTPUT}")
    endif()
  elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all")
    unset(CMAKE_CUDA_ARCHITECTURES)
    message(STATUS "Setting CMAKE_CUDA_ARCHITECTURES to all enables a list of architectures OnnxRuntime optimized for, "
                   "not all architectures CUDA compiler supports.")
  elseif(CMAKE_CUDA_ARCHITECTURES STREQUAL "all-major")
    unset(CMAKE_CUDA_ARCHITECTURES)
    message(
      STATUS "Setting CMAKE_CUDA_ARCHITECTURES to all-major enables a list of architectures OnnxRuntime optimized for, "
             "not all major architectures CUDA compiler supports.")
  else()
    message(STATUS "Original CMAKE_CUDA_ARCHITECTURES : ${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_LIBRARY_ARCHITECTURE STREQUAL "aarch64-linux-gnu")
      # Support for Jetson/Tegra ARM devices
      set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87") # TX1/Nano, TX2, Xavier, Orin
    else()
      if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 12.8)
        set(CMAKE_CUDA_ARCHITECTURES "52;60;70;75;80;86;89;90")
      elseif(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
        set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90;100;120")
      else()
        # 13.x drops support for 60 and 70
        set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90;100;120")
      endif()
    endif()
  endif()

  unset(CMAKE_CUDA_ARCHITECTURES_CLEAN)
  unset(CMAKE_CUDA_ARCHITECTURES_LAST_VIRTUAL)
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_ARCH STREQUAL "")
      continue()
    endif()

    if(CUDA_ARCH MATCHES "^([1-9])([0-9])+[af]?-virtual$")
      set(CMAKE_CUDA_ARCHITECTURES_LAST_VIRTUAL ${CUDA_ARCH})
    elseif(CUDA_ARCH MATCHES "^[1-9][0-9]+f-real$")
      # Family-specific SASS only. Keep it verbatim: without the "f" it would become "<cc>a-real"
      # below, and sm_120a code runs only on 12.0, while sm_120f code runs on 12.0 and 12.1.
      list(APPEND CMAKE_CUDA_ARCHITECTURES_CLEAN ${CUDA_ARCH})
    elseif(CUDA_ARCH MATCHES "^(([1-9])([0-9])+[af]?)-real$")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_CLEAN ${CMAKE_MATCH_1})
    elseif(CUDA_ARCH MATCHES "^(([1-9])([0-9])+)([af]?)$")
      list(APPEND CMAKE_CUDA_ARCHITECTURES_CLEAN ${CMAKE_MATCH_1}${CMAKE_MATCH_4})
    else()
      message(FATAL_ERROR "Unrecognized CUDA architecture: ${CUDA_ARCH}")
    endif()
  endforeach()
  list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES_CLEAN)
  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_CLEAN})

  # CMAKE_CUDA_ARCHITECTURES_ORIG contains all architectures enabled, without automatically added -real or -a suffix.
  # A "<cc>f-real" entry keeps its -real suffix to tell it apart from "<cc>f" (SASS and PTX).
  set(CMAKE_CUDA_ARCHITECTURES_ORIG "${CMAKE_CUDA_ARCHITECTURES}")
  message(STATUS "GPU architectures: ${CMAKE_CUDA_ARCHITECTURES_ORIG}")

  unset(ORT_HAS_SM80_OR_LATER)
  unset(ORT_HAS_SM90_OR_LATER)
  unset(ORT_HAS_SM100_OR_LATER)
  unset(ORT_HAS_SM120_OR_LATER)
  # Include the -virtual entry, which _ORIG omits: PTX for compute_<cc> is device code for <cc> too,
  # and e.g. the XQA kernels rely on HAS_SM80_OR_LATER to emit their host stubs (xqa_impl_gen.cuh).
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES_ORIG CMAKE_CUDA_ARCHITECTURES_LAST_VIRTUAL)
    if(CUDA_ARCH MATCHES "^([0-9]+)")
      if(CMAKE_MATCH_1 GREATER_EQUAL 80)
        set(ORT_HAS_SM80_OR_LATER ON)
      endif()
      if(CMAKE_MATCH_1 GREATER_EQUAL 90)
        set(ORT_HAS_SM90_OR_LATER ON)
      endif()
      if(CMAKE_MATCH_1 GREATER_EQUAL 100)
        set(ORT_HAS_SM100_OR_LATER ON)
      endif()
      if(CMAKE_MATCH_1 GREATER_EQUAL 120)
        set(ORT_HAS_SM120_OR_LATER ON)
      endif()
    endif()
  endforeach()
  if(ORT_HAS_SM80_OR_LATER)
    add_definitions("-DHAS_SM80_OR_LATER")
  endif()
  if(ORT_HAS_SM90_OR_LATER)
    add_definitions("-DHAS_SM90_OR_LATER")
  endif()
  if(ORT_HAS_SM100_OR_LATER)
    add_definitions("-DHAS_SM100_OR_LATER")
  endif()
  if(ORT_HAS_SM120_OR_LATER)
    add_definitions("-DHAS_SM120_OR_LATER")
  endif()

  # Enable accelerated features (like WGMMA, TMA and setmaxnreg) for SM >= 90.
  set(ARCHITECTURES_WITH_ACCEL "90" "100" "103" "110" "120" "121")
  unset(CMAKE_CUDA_ARCHITECTURES_NORMALIZED)
  foreach(CUDA_ARCH IN LISTS CMAKE_CUDA_ARCHITECTURES)
    if(CUDA_ARCH MATCHES "^([0-9]+)f(-real)?$")
      # Family code, kept as given: "<cc>f" (SASS and PTX) or "<cc>f-real" (SASS only)
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}")
    elseif("${CUDA_ARCH}" IN_LIST ARCHITECTURES_WITH_ACCEL)
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}a-real")
    else()
      list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CUDA_ARCH}-real")
    endif()
  endforeach()

  if(DEFINED CMAKE_CUDA_ARCHITECTURES_LAST_VIRTUAL)
    list(APPEND CMAKE_CUDA_ARCHITECTURES_NORMALIZED "${CMAKE_CUDA_ARCHITECTURES_LAST_VIRTUAL}")
  endif()

  list(REMOVE_DUPLICATES CMAKE_CUDA_ARCHITECTURES_NORMALIZED)

  set(CMAKE_CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES_NORMALIZED})

  # The gates below are derived from the normalized list, in which every entry states what nvcc
  # generates ("<cc>[a|f]-real": SASS, "<cc>[a|f]-virtual": PTX, "<cc>f": both). Unlike
  # CMAKE_CUDA_ARCHITECTURES_ORIG, it also contains the -virtual entry.

  # ORT_HAS_SM12X: native code for SM120 or SM121 (compute capability 12.0/12.1) is requested,
  # in any spelling ("120", "121a", "120f", "121a-real", ...). Gates the SM120 TMA grouped GEMM
  # and block-scaled FP4 kernels, which are compiled for the requested 12.x architectures.
  onnxruntime_cuda_archs_have_sass(ORT_HAS_SM12X 120 121 ${CMAKE_CUDA_ARCHITECTURES})

  # ORT_HAS_SM90: native code for SM90 (Hopper) itself is requested. The Hopper TMA/WGMMA kernels
  # and DeepGEMM are compiled only for sm_90a, which runs on compute capability 9.0 and nothing
  # else (not SM100/SM120), so they are dead code unless SM90 is targeted.
  onnxruntime_cuda_archs_have_sass(ORT_HAS_SM90 90 90 ${CMAKE_CUDA_ARCHITECTURES})

  # EXCLUDE_SM_<sm> compiles out the kernels instantiated for <sm> (the #ifndef EXCLUDE_SM_<sm>
  # guards under contrib_ops/cuda/llm). It may only be defined when no GPU that this build has
  # code for can dispatch to those kernels; otherwise the dispatcher throws at run time.
  #  * 80, 86, 89: CUTLASS 2.x / CuTe SM80-style kernels (mma.sync, cp.async). They use no
  #    architecture-specific features, are compiled for every requested architecture, and are
  #    dispatched on every GPU of at least that compute capability: MoeGemmRunner::dispatchToArch
  #    uses the SM80 fused MoE GEMM on SM80-SM89 and as the fallback on SM90 and later, which is
  #    the only path for unquantized FP16/BF16 MoE on SM100 and SM12x. Keep them if any requested
  #    architecture, SASS or PTX, has at least that compute capability.
  #  * 90: sm_90a kernels, which run on compute capability 9.0 only.
  #  * 100, 110, 120: architecture- or family-specific kernels, compiled for the requested
  #    architectures of that family. Keep them if SASS for any member of the family is requested.
  #    Members per the family-specific compatibility table of the CUDA C++ Programming Guide:
  #    compute_100f covers 10.0, 10.3 and 10.7, compute_110f covers 11.0, and compute_120f covers
  #    12.0 and 12.1.
  onnxruntime_cuda_archs_have_code(_ort_keep_sm80 80 ${CMAKE_CUDA_ARCHITECTURES})
  onnxruntime_cuda_archs_have_code(_ort_keep_sm86 86 ${CMAKE_CUDA_ARCHITECTURES})
  onnxruntime_cuda_archs_have_code(_ort_keep_sm89 89 ${CMAKE_CUDA_ARCHITECTURES})
  set(_ort_keep_sm90 ${ORT_HAS_SM90})
  onnxruntime_cuda_archs_have_sass(_ort_keep_sm100 100 107 ${CMAKE_CUDA_ARCHITECTURES})
  onnxruntime_cuda_archs_have_sass(_ort_keep_sm110 110 110 ${CMAKE_CUDA_ARCHITECTURES})
  set(_ort_keep_sm120 ${ORT_HAS_SM12X})
  foreach(_ort_sm IN ITEMS 80 86 89 90 100 110 120)
    if(NOT _ort_keep_sm${_ort_sm})
      add_definitions("-DEXCLUDE_SM_${_ort_sm}")
      message(STATUS "Excluding SM ${_ort_sm}")
    endif()
    unset(_ort_keep_sm${_ort_sm})
  endforeach()
  unset(_ort_sm)

  message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endmacro()

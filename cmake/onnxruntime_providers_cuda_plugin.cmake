# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Build the CUDA Execution Provider as a plugin shared library.
# This file is included from onnxruntime_providers.cmake when onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON.

message(STATUS "Building CUDA EP as plugin shared library")



set(CUDA_PLUGIN_EP_DIR "${ONNXRUNTIME_ROOT}/core/providers/cuda/plugin")

# --- Collect standard CUDA EP sources ---
file(GLOB_RECURSE CUDA_EP_CC_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
)

file(GLOB_RECURSE CUDA_EP_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
)

# --- Collect contrib ops sources ---
file(GLOB_RECURSE CUDA_CONTRIB_OPS_CC_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cc"
)
if(NOT onnxruntime_ENABLE_MLAS_QMOE_OFFLOAD)
  list(APPEND CUDA_CONTRIB_OPS_CC_SRCS
     "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/moe/qmoe_shared_layout.cc")
endif()

file(GLOB_RECURSE CUDA_CONTRIB_OPS_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cu"
)

list(APPEND CUDA_PLUGIN_EP_CC_SRCS
     ${CUDA_EP_CC_SRCS}
     ${CUDA_CONTRIB_OPS_CC_SRCS}
)

list(APPEND CUDA_PLUGIN_EP_CU_SRCS
     ${CUDA_EP_CU_SRCS}
     ${CUDA_CONTRIB_OPS_CU_SRCS}
)

list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/aten_ops/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/collective/.*")

list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/aten_ops/.*")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/collective/.*")

if (NOT onnxruntime_USE_TRT_FUSED_ATTENTION)
  # Drop the prebuilt TensorRT fused MHA cubin blobs. cudaDriverWrapper.cc is kept because
  # sparse attention depends on it.
  list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX
    ".*/bert/tensorrt_fused_multihead_attention/.*(\\.cubin\\.cc|_kernel\\.sm[0-9]+\\.cc)$")
endif()

# Exclude files that include cuda_execution_provider.h (directly or transitively),
# which conflicts with the adapter shim CUDAExecutionProvider class.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_execution_provider\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_provider_factory\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_provider_interface\\.cc$")

# Exclude the framework controlflow/ subdirectory — these inherit from CPU base
# classes (If, Loop, Scan). The plugin has its own control flow wrappers in
# plugin/cuda_controlflow_plugin.cc that delegate to OrtEpApi.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/core/providers/cuda/controlflow/.*")

# Exclude the entire tunable/ subdirectory — it depends on the real CudaTuningContext
# and CUDAExecutionProvider which are not available in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tunable/.*")

# Exclude real EP infrastructure files (replaced by plugin/ equivalents).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_stream_handle\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_execution_provider_info\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_graph\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_mempool_arena\\.cc$")

# Exclude cuda_common.cc — its HalfGemmOptions definitions conflict with the
# adapter's inline shim. Utility functions are replaced or not needed.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_common\\.cc$")

# Exclude cuda_nhwc_kernels.cc and cuda_contrib_kernels.cc — these files contain
# explicit BuildKernelCreateInfo<> registration tables that reference ALL kernel
# classes (including those in excluded source files like space_depth_ops.cc,
# controlflow/, transformers/, etc.), causing undefined symbols at link time.
# With PluginKernelCollector, individual kernel files self-register via macro
# overrides, so these centralized tables are not needed in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_nhwc_kernels\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_contrib_kernels\\.cc$")

# Exclude sequence_op.cc — uses TensorSeq (incomplete type in plugin build).
# identity_op.cc is now included: TensorSeq code path is guarded by
# BUILD_CUDA_EP_AS_PLUGIN and opset 14+ registrations use Tensor-only types.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/sequence_op\\.cc$")

# Permanently excluded — pure CPU ops, handled by GetCpuPreferredNodes.
# size.cc registers onnxruntime::Size (CPU op) whose Compute() body lives
# in the CPU provider and is not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/size\\.cc$")

# shape_op.cc is INCLUDED in the plugin build. It provides an adapter-based
# Shape kernel under #ifdef BUILD_CUDA_EP_AS_PLUGIN (the CPU onnxruntime::Shape
# class, which derives from the framework OpKernel, is only used in the
# non-plugin build). Registering Shape on the EP keeps it off the CPU EP and
# avoids Memcpy nodes that would otherwise break CUDA Graph capture.

# Exclude contrib training ops (shrunken_gather depends on provider_api.h in header).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/tensor/shrunken_gather\\.cc$")


# Exclude contrib transformers/ (beam search, greedy search, sampling). Those need subgraph inference.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/transformers/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/transformers/.*")

# Apply shared CUDA .cu source filtering (flash attention quick build, MoE GEMM FP4/FP8).
include(onnxruntime_cuda_source_filters.cmake)
include(onnxruntime_cuda_cccl.cmake)
onnxruntime_filter_cuda_cu_sources(CUDA_PLUGIN_EP_CU_SRCS)
onnxruntime_extract_sm_specific_cuda_sources(CUDA_PLUGIN_EP_CU_SRCS
  SM90_SOURCES _cuda_plugin_sm90_tma_srcs
  SM120_SOURCES _cuda_plugin_sm120_tma_srcs
)
onnxruntime_extract_flash_attention_sources(CUDA_PLUGIN_EP_CU_SRCS
  FLASH_SOURCES _cuda_plugin_flash_attention_srcs
)
onnxruntime_extract_xqa_sources(CUDA_PLUGIN_EP_CU_SRCS
  XQA_SOURCES _cuda_plugin_xqa_srcs
)
onnxruntime_extract_llm_sources(CUDA_PLUGIN_EP_CU_SRCS
  LLM_SOURCES _cuda_plugin_llm_srcs
  LLM_SM90_SOURCES _cuda_plugin_llm_sm90_srcs
  LLM_FP4_SOURCES _cuda_plugin_llm_fp4_srcs
)
if(MSVC OR UNIX)
  foreach(_src IN LISTS _cuda_plugin_llm_sm90_srcs)
    if(_src MATCHES "/(moe_gemm/deep_gemm_sm90|deep_gemm_matmul_sm90)\\.cu$")
      if(MSVC)
        set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "-Xcompiler=/wd4068")
      else()
        set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "-Xcompiler=-Wno-unknown-pragmas")
      endif()
    endif()
  endforeach()
endif()

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)

if(WIN32)
  # Add version information to the packaged plugin DLL.
  target_sources(onnxruntime_providers_cuda_plugin PRIVATE
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/onnxruntime_providers_cuda.rc")
  target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE
      FILE_NAME=\"onnxruntime_providers_cuda.dll\")
elseif(UNIX AND NOT APPLE)
  # The build output is packaged directly, so do not embed the build machine's CUDA path.
  set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES SKIP_BUILD_RPATH TRUE)
endif()

# Mirror directory structure in the Visual Studio solution tree under "onnxruntime".
source_group(TREE ${ONNXRUNTIME_ROOT} PREFIX "onnxruntime" FILES ${CUDA_EP_CC_SRCS} ${CUDA_EP_CU_SRCS})
source_group(TREE ${ONNXRUNTIME_ROOT} PREFIX "onnxruntime" FILES ${CUDA_CONTRIB_OPS_CC_SRCS} ${CUDA_CONTRIB_OPS_CU_SRCS})
# Keep the plugin CUDA target aligned with the repo-wide C++20 baseline.
# Forcing CUDA C++17 here breaks newer protobuf/absl headers used by the plugin
# build, as absl::compare expects standard ordering support in this configuration.
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

# Suppress -Werror=maybe-uninitialized for local variables written by
# adapter OpKernelInfo::GetAttr<> (GCC falsely warns about variables that are
# initialized inside GetAttr’s output parameter path).
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wno-maybe-uninitialized>
)
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    # Flash-attention, XQA, MoE, and other pure CUDA kernel .cu files must NOT
    # receive the ORT-framework force-include (it conflicts with cute::Tensor etc.).
    # cuda_plugin_kernels.cu already #include "cuda_kernel_adapter.h" directly.
    # Op-registration .cc files do not include it directly, so they need it here.
    #
    # IMPORTANT: The CXX force-include order matters — adapters.h MUST precede
    # cuda_kernel_adapter.h because the adapter establishes type aliases that the
    # kernel adapter header depends on.
    #
    # Force NVCC onto C++20 explicitly. With the VS generator the CUDA standard
    # property alone still leaves `-std=c++17` in AdditionalOptions.
    # Suppress NVCC cudafe warnings:
    #   550  - variable set but never used (in adapter headers)
    #   2810 - [[nodiscard]] false positive on Status assignments in op_kernel.h / kernel_registry.h
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--std c++20>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--diag_suppress=550>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=2810>"
    # Force-include adapters.h and cuda_kernel_adapter.h for CXX sources.
    # GCC/Clang use -include, MSVC uses /FI.
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-include;${REPO_ROOT}/include/onnxruntime/ep/adapters.h>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:SHELL:-include ${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/FI${REPO_ROOT}/include/onnxruntime/ep/adapters.h>"
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/FI${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h>"
)

if (MSVC)
    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /permissive>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4211>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:preprocessor>"
        # Pass /bigobj to the CUDA host compiler using dash spelling. Raw /bigobj is excluded
        # from global ARM64 CUDA options in onnxruntime_common.cmake because nvcc parses it as input.
        "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-bigobj>"
    )

    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
        # /permissive is required for CUTLASS cute headers (cute::stride.hpp, cute::Layout etc.)
        "$<$<COMPILE_LANGUAGE:CXX>:/permissive>"
        # /permissive disables C++ alternative tokens (or, and, not, etc.).
        # Force-include iso646.h to restore them as macros.
        "$<$<COMPILE_LANGUAGE:CXX>:/FIiso646.h>"
        "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>"
    )
endif()

# Mirror the core CUDA provider's CUDA 12.8+ NVCC workarounds so the plugin
# target handles stricter cudafe diagnostics consistently.
if (DEFINED onnxruntime_NVCC_THREADS)
    set(onnxruntime_plugin_nvcc_threads "${onnxruntime_NVCC_THREADS}")
else()
    set(onnxruntime_plugin_nvcc_threads "4")
endif()
# Shared CUDA compile options (excluding --threads, which is set per-target so that
# flash attention can use a lower thread count without duplicate-flag nvcc warnings).
# These mirror the options from the parent plugin target and config_cuda_provider_shared_module
# so that OBJECT libraries compiled separately receive the same flags.
set(_cuda_plugin_shared_compile_options
    # Force NVCC onto C++20 explicitly. With the VS generator the CUDA_STANDARD
    # property alone still leaves `-std=c++17` in AdditionalOptions.
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--std c++20>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>"
    # Suppress cudafe front-end diagnostic 550 (variable set but never used) from third-party headers.
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=550>"
    # Suppress cudafe [[nodiscard]] false positive on Status assignments.
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=2810>"
)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    list(APPEND _cuda_plugin_shared_compile_options
            "$<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>"
            "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=221>"
      # Protobuf uses offsetof on MessageLite, which is intentionally non-standard-layout.
      "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1427>"
            "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2908>"
    )

    if (MSVC)
        list(APPEND _cuda_plugin_shared_compile_options
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4505>"
        )
    endif()
endif()

if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
  # CUDA 13 diagnoses Protobuf's intentional signed enum pointer-mask conversion as 68-D,
  # qualified friend declarations in Abseil and Protobuf as 970-D, and Protobuf's
  # always_inline template redeclaration as 2189-D.
  list(APPEND _cuda_plugin_shared_compile_options
      "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=68>"
      "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=970>"
      "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2189>"
  )

  if (MSVC)
    # Suppress diagnostics from CUDA 13 headers: unrecognized __pragma in device code,
    # cuda_fp4.hpp's unused parameter, and CCCL's unmatched #pragma warning(pop)
    # in the MSVC host compiler.
    list(APPEND _cuda_plugin_shared_compile_options
        "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20199>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4100>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4193>"
    )
  endif()
endif()

if (MSVC)
    list(APPEND _cuda_plugin_shared_compile_options
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /permissive>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4211>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus>"
            "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:preprocessor>"
            # Unlike the options explicitly paired with -Xcompiler above, the raw /bigobj inherited
            # from global compile options is parsed by nvcc as an input file on ARM64. Exclude that raw
            # option in onnxruntime_common.cmake and forward its dash-spelled equivalent explicitly.
            "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-bigobj>"
    )
endif()

include(cudnn_frontend)
include(cutlass)
if(ORT_HAS_SM90_OR_LATER AND NOT WIN32 AND NOT onnxruntime_DISABLE_CONTRIB_OPS)
  include(deep_gemm)
  target_include_directories(onnxruntime_providers_cuda_plugin PRIVATE ${deep_gemm_SOURCE_DIR}/deep_gemm/include)
  target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE USE_DEEP_GEMM)
endif()

# TMA compile definitions — mirror config_cuda_provider_shared_module in onnxruntime_providers_cuda.cmake
if(ORT_HAS_SM90_OR_LATER)
  list(APPEND _cuda_plugin_shared_compile_options
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-w>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-DCUTLASS_ENABLE_GDC_FOR_SM90=1>")
  if(NOT MSVC)
    # The native SM90 (Hopper) TMA/WGMMA launchers pass CUTLASS TMA descriptor types through
    # NVCC-generated host stubs. With CUDA 13 + MSVC those stubs contain 128-byte over-aligned
    # by-value formal parameters, which triggers MSVC C2719 ("formal parameter with requested
    # alignment of 128 won't be aligned"). Disable the native SM90 fpA_intB (COMPILE_HOPPER_TMA_GEMMS)
    # and grouped MoE (COMPILE_HOPPER_TMA_GROUPED_GEMMS) TMA kernels on MSVC; the launcher bodies
    # become throwing stubs and the SM80 compatibility path still runs on Hopper at runtime.
    # See docs/contrib_ops/cuda/moe_qmoe.md section 14.1.
    target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE COMPILE_HOPPER_TMA_GEMMS)
    target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE COMPILE_HOPPER_TMA_GROUPED_GEMMS)
  endif()
endif()
if(("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG OR "121" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG) AND
   (NOT MSVC OR onnxruntime_USE_FP4_QMOE))
  target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS)
endif()

# Apply shared options + --threads to the parent plugin target.
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
  ${_cuda_plugin_shared_compile_options}
  "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_plugin_nvcc_threads}\">"
)

if(MSVC)
  # NVCC emits CodeView references to a host compiler PDB that it does not retain.
  target_link_options(onnxruntime_providers_cuda_plugin PRIVATE "/IGNORE:4099")
endif()

# SM-specific OBJECT libraries — compiled with restricted CUDA architectures.
# SM90/SM120 TMA and LLM contain MoE and MatMulNBits kernels (contrib ops only).

# Flash Attention OBJECT library: SM80+ only, with independent nvcc_threads.
# Flash Attention V2 kernels require SM80 and are memory-intensive to compile.
# _cuda_plugin_flash_attention_srcs is only populated when onnxruntime_USE_FLASH_ATTENTION
# is ON; otherwise the .cu sources are excluded from the build entirely (see extraction above).
if(NOT DEFINED onnxruntime_FLASH_NVCC_THREADS)
  set(onnxruntime_FLASH_NVCC_THREADS "1")
endif()
if(_cuda_plugin_flash_attention_srcs)
  onnxruntime_filter_cuda_archs(_plugin_flash_cuda_architectures MIN_SM 80)
  if(_plugin_flash_cuda_architectures)
    onnxruntime_add_cuda_plugin_object_library(
      NAME onnxruntime_providers_cuda_plugin_flash_attention
      PARENT onnxruntime_providers_cuda_plugin
      CUDA_ARCHITECTURES "${_plugin_flash_cuda_architectures}"
      NVCC_THREADS "${onnxruntime_FLASH_NVCC_THREADS}"
      COMPILE_OPTIONS ${_cuda_plugin_shared_compile_options}
      SOURCES ${_cuda_plugin_flash_attention_srcs})
  else()
    # No SM80+ architectures available: compile flash sources in parent target so the
    # linker can find the host-side symbols referenced by flash_api.cc. The kernels
    # themselves will be empty stubs due to __CUDA_ARCH__ >= 800 guards.
    target_sources(onnxruntime_providers_cuda_plugin PRIVATE ${_cuda_plugin_flash_attention_srcs})
  endif()
endif()

if(_cuda_plugin_xqa_srcs)
  onnxruntime_filter_cuda_archs(_plugin_xqa_cuda_architectures MIN_SM 80)
  if(_plugin_xqa_cuda_architectures)
    onnxruntime_add_cuda_plugin_object_library(
      NAME onnxruntime_providers_cuda_plugin_xqa
      PARENT onnxruntime_providers_cuda_plugin
      CUDA_ARCHITECTURES "${_plugin_xqa_cuda_architectures}"
      NVCC_THREADS "${onnxruntime_plugin_nvcc_threads}"
      COMPILE_OPTIONS ${_cuda_plugin_shared_compile_options}
      SOURCES ${_cuda_plugin_xqa_srcs})
  else()
    target_sources(onnxruntime_providers_cuda_plugin PRIVATE ${_cuda_plugin_xqa_srcs})
  endif()
endif()

if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
  # SM90 TMA warp-specialized files use SM90-specific collective operations.
  # Also includes fpA_intB SM90 launchers (guarded by #ifndef EXCLUDE_SM_90).
  if(_cuda_plugin_sm90_tma_srcs OR _cuda_plugin_llm_sm90_srcs)
    set(_plugin_sm90_all_srcs ${_cuda_plugin_sm90_tma_srcs} ${_cuda_plugin_llm_sm90_srcs})
    onnxruntime_filter_cuda_archs(_plugin_sm90_check MIN_SM 90)
    if(_plugin_sm90_check)
      onnxruntime_add_cuda_plugin_object_library(
        NAME onnxruntime_providers_cuda_plugin_sm90_tma
        PARENT onnxruntime_providers_cuda_plugin
        CUDA_ARCHITECTURES "90a-real"
        NVCC_THREADS "${onnxruntime_plugin_nvcc_threads}"
        COMPILE_OPTIONS ${_cuda_plugin_shared_compile_options}
        SOURCES ${_plugin_sm90_all_srcs})
    endif()
  endif()

  # CUDA 13 gives CUtensorMap 128-byte host alignment when MSVC reports an accurate
  # __cplusplus value. The target-specific host flag below avoids C2719 for FP4 QMoE.
  if(_cuda_plugin_sm120_tma_srcs AND
     (NOT MSVC OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0 OR onnxruntime_USE_FP4_QMOE))
    onnxruntime_filter_cuda_archs(_plugin_sm120_cuda_architectures MIN_SM 120)
    if(_plugin_sm120_cuda_architectures)
      onnxruntime_add_cuda_plugin_object_library(
        NAME onnxruntime_providers_cuda_plugin_sm120_tma
        PARENT onnxruntime_providers_cuda_plugin
        CUDA_ARCHITECTURES "${_plugin_sm120_cuda_architectures}"
        NVCC_THREADS "${onnxruntime_plugin_nvcc_threads}"
        COMPILE_OPTIONS ${_cuda_plugin_shared_compile_options}
        SOURCES ${_cuda_plugin_sm120_tma_srcs})
      if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
        # CUDA 13's cuda.h gives CUtensorMap alignas(128) when MSVC reports the
        # accurate C++ language level. MSVC cannot pass that type by value in
        # NVCC-generated host stubs (C2719). Keep NVCC at C++20, but use the
        # legacy host __cplusplus value so CUtensorMap retains its 128-byte size
        # with ordinary host alignment.
        target_compile_options(onnxruntime_providers_cuda_plugin_sm120_tma PRIVATE
          "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus->")
      endif()
      target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_ENABLE_BLOCKQUANT_SM120)
    endif()
  endif()

  # LLM OBJECT library: SM75+ (backward compatible with fpA_intB_gemv/gemm which support SM75).
  # FP4 QMoE sources are compiled separately at native SM120 below. Under MSVC, keep virtual
  # SM120 PTX and add non-"a" native SASS to avoid CCCL tcgen05 host-compile failures.
  if(_cuda_plugin_llm_srcs)
    set(_plugin_llm_extra_compile_options)
    if(MSVC)
      onnxruntime_filter_cuda_archs(_plugin_llm_cuda_architectures MIN_SM 75 REPLACE_SM120_REAL_WITH_VIRTUAL)
      if("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG OR "120a" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG)
        # Virtual compute_120 PTX alone needs a driver as new as the toolkit to JIT; add native non-"a" SASS
        # (no tcgen05 path) so the ordinary LLM kernels (fpA_intB GEMV/GEMM) run without JIT.
        list(APPEND _plugin_llm_extra_compile_options
             "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-gencode=arch=compute_120,code=sm_120>")
      endif()
    else()
      onnxruntime_filter_cuda_archs(_plugin_llm_cuda_architectures MIN_SM 75)
    endif()
    if(_plugin_llm_cuda_architectures)
      onnxruntime_add_cuda_plugin_object_library(
        NAME onnxruntime_providers_cuda_plugin_llm
        PARENT onnxruntime_providers_cuda_plugin
        CUDA_ARCHITECTURES "${_plugin_llm_cuda_architectures}"
        NVCC_THREADS "${onnxruntime_plugin_nvcc_threads}"
        COMPILE_OPTIONS ${_cuda_plugin_shared_compile_options} ${_plugin_llm_extra_compile_options}
        SOURCES ${_cuda_plugin_llm_srcs})
    endif()
  endif()

  if(_cuda_plugin_llm_fp4_srcs)
    onnxruntime_filter_cuda_archs(_plugin_llm_fp4_cuda_architectures MIN_SM 75)
    if(_plugin_llm_fp4_cuda_architectures)
      onnxruntime_add_cuda_plugin_object_library(
        NAME onnxruntime_providers_cuda_plugin_llm_fp4
        PARENT onnxruntime_providers_cuda_plugin
        CUDA_ARCHITECTURES "${_plugin_llm_fp4_cuda_architectures}"
        NVCC_THREADS "${onnxruntime_plugin_nvcc_threads}"
        COMPILE_OPTIONS ${_cuda_plugin_shared_compile_options}
        SOURCES ${_cuda_plugin_llm_fp4_srcs})
    endif()
  endif()
endif()

# --- Find cuDNN (may be at a custom path via onnxruntime_CUDNN_HOME) ---
set(_CUDNN_SEARCH_PATHS "")
if(onnxruntime_CUDNN_HOME)
  list(APPEND _CUDNN_SEARCH_PATHS "${onnxruntime_CUDNN_HOME}")
endif()
if(DEFINED ENV{CUDNN_HOME})
  list(APPEND _CUDNN_SEARCH_PATHS "$ENV{CUDNN_HOME}")
endif()

set(CUDA_PLUGIN_CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR})
set(CUDA_PLUGIN_CUDNN_LIBRARY ${cudnn_LIBRARY})

if(NOT CUDA_PLUGIN_CUDNN_INCLUDE_DIR)
  message(FATAL_ERROR "cuDNN headers not found (from main ORT search) for CUDA Plugin EP.")
endif()

message(STATUS "CUDA Plugin EP: cuDNN include: ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Plugin EP: cuDNN runtime library: ${CUDA_PLUGIN_CUDNN_LIBRARY}")

# Include directories — only public ORT headers + CUDA toolkit + cuDNN + internal headers for adapter
target_include_directories(onnxruntime_providers_cuda_plugin PRIVATE
    ${REPO_ROOT}/include
    ${REPO_ROOT}/include/onnxruntime/core/session
    ${ONNXRUNTIME_ROOT}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}
    ${Eigen3_SOURCE_DIR}
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/examples
    ${cutlass_SOURCE_DIR}/tools/util/include
)

# The host .cc files globbed into this target (contrib_ops/cuda/llm/*.cc and friends) include
# CUTLASS headers, which reach <cuda/std/...>. In the non-plugin build the same files are part
# of onnxruntime_providers_cuda, which gets this from config_cuda_provider_shared_module.
#
# The SM-specific OBJECT libraries created above are covered by this call even though they
# already exist: onnxruntime_add_cuda_plugin_object_library gives them this target's includes
# as $<TARGET_PROPERTY:onnxruntime_providers_cuda_plugin,INCLUDE_DIRECTORIES>, which is
# evaluated after configuration and so picks up whatever is added here - order included, so
# the CUDA 13.3 patched-header directory keeps shadowing the toolkit CCCL headers for their
# .cu sources. Keep that indirection in mind before making the inheritance eager.
ort_configure_cuda_cccl(onnxruntime_providers_cuda_plugin)

onnxruntime_add_include_to_target(
    onnxruntime_providers_cuda_plugin
    onnxruntime_common
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
    flatbuffers::flatbuffers
)

# Ensure generated headers (e.g. onnx-ml.pb.h) are available before compiling.
add_dependencies(onnxruntime_providers_cuda_plugin ${onnxruntime_EXTERNAL_DEPENDENCIES})

# Link libraries
# cuDNN and cuFFT are loaded dynamically at runtime (see cudnn_stub.cc / cufft_stub.cc, which are
# picked up by the GLOB above), so they are intentionally not linked here to avoid a hard runtime
# dependency when the related ops are not used.
target_link_libraries(onnxruntime_providers_cuda_plugin PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
    cudnn_frontend
    Boost::mp11
    safeint_interface
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_mlas
    onnxruntime_flatbuffers
    onnxruntime_common
    cpuinfo::cpuinfo
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
)

  target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

if (onnxruntime_ENABLE_CUDA_PROFILING)
    target_link_libraries(onnxruntime_providers_cuda_plugin PRIVATE CUDA::cupti)
    target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ENABLE_CUDA_PROFILING)
endif()

# Default plugin EP version to ORT_VERSION with "-dev" suffix if not explicitly provided.
if(NOT DEFINED onnxruntime_PLUGIN_EP_VERSION)
  set(onnxruntime_PLUGIN_EP_VERSION "${ORT_VERSION}-dev")
endif()

# Bake the minimum compatible ORT version (the single source of truth lives in
# plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION) into the EP DLL so it can be enforced at runtime by
# onnxruntime::ep::ApiInit(). Format is strict "MAJOR.MINOR.PATCH".
set(_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION_FILE "${REPO_ROOT}/plugin-ep-cuda/MIN_ONNXRUNTIME_VERSION")
# Re-run CMake configure when the version file changes so the baked-in value stays in sync.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION_FILE}")
file(STRINGS "${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION_FILE}" _ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION LIMIT_COUNT 1)
string(STRIP "${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION}" _ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION)
if(NOT _ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION)
  message(FATAL_ERROR "CUDA plugin EP minimum ORT version file is missing or empty: "
                      "${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION_FILE}")
endif()
# ApiInit() strictly parses "MAJOR.MINOR.PATCH"; fail fast on any malformed value.
if(NOT _ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
  message(FATAL_ERROR "CUDA plugin EP minimum ORT version must be \"MAJOR.MINOR.PATCH\", got "
                      "\"${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION}\" from "
                      "${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION_FILE}")
endif()

# Symbol visibility — only export CreateEpFactories and ReleaseEpFactory
target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_API_MANUAL_INIT BUILD_CUDA_EP_AS_PLUGIN ORT_USE_EP_API_ADAPTERS=1 ONNX_ML=1 ONNX_NAMESPACE=onnx ONNX_USE_LITE_PROTO=1 ORT_PLUGIN_EP_VERSION="${onnxruntime_PLUGIN_EP_VERSION}" ORT_PLUGIN_EP_MIN_ORT_VERSION="${_ORT_PLUGIN_EP_CUDA_MIN_ORT_VERSION}")

if (onnxruntime_USE_CUDA_NHWC_OPS)
    target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ENABLE_CUDA_NHWC_OPS)
endif()

if(WIN32)
  # Windows: use .def file for symbol exports
  set(CUDA_PLUGIN_DEF_FILE ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_ep_symbols.def)
  if(EXISTS ${CUDA_PLUGIN_DEF_FILE})
    target_sources(onnxruntime_providers_cuda_plugin PRIVATE ${CUDA_PLUGIN_DEF_FILE})
  endif()
else()
  # Linux/macOS: hide all symbols by default, explicitly export via __attribute__((visibility("default")))
  set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
  )
endif()



# Set output name and solution folder
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
  OUTPUT_NAME "onnxruntime_providers_cuda"
    FOLDER "ONNXRuntime"
)

# Install
install(TARGETS onnxruntime_providers_cuda_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

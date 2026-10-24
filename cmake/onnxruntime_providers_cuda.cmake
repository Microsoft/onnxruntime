# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


  if (onnxruntime_CUDA_MINIMAL)
    file(GLOB onnxruntime_providers_cuda_cc_srcs CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.h"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/ml/*.h"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/ml/*.cc"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/tunable/*.h"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/tunable/*.cc"
    )
    # Remove pch files
    list(REMOVE_ITEM onnxruntime_providers_cuda_cc_srcs
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/integer_gemm.cc"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/triton_kernel.h"
    )
  else()
    file(GLOB_RECURSE onnxruntime_providers_cuda_cc_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.h"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
    )
  endif()
  # Exclude plugin directory if it was picked up by GLOB_RECURSE
  list(FILTER onnxruntime_providers_cuda_cc_srcs EXCLUDE REGEX "core/providers/cuda/plugin/.*")

  # Remove pch files
  list(REMOVE_ITEM onnxruntime_providers_cuda_cc_srcs
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.h"
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
  )

  # The shared_library files are in a separate list since they use precompiled headers, and the above files have them disabled.
  file(GLOB_RECURSE onnxruntime_providers_cuda_shared_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.h"
    "${ONNXRUNTIME_ROOT}/core/providers/shared_library/*.cc"
  )


  if (NOT onnxruntime_CUDA_MINIMAL)
    file(GLOB_RECURSE onnxruntime_providers_cuda_cu_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
      "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cuh"
    )
  else()
    set(onnxruntime_providers_cuda_cu_srcs
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops_impl.cu"
        "${ONNXRUNTIME_ROOT}/core/providers/cuda/ml/label_encoder_impl.cu"
        )
  endif()
  # Exclude plugin directory if it was picked up by GLOB_RECURSE
  list(FILTER onnxruntime_providers_cuda_cu_srcs EXCLUDE REGEX "core/providers/cuda/plugin/.*")
  source_group(TREE ${ONNXRUNTIME_ROOT}/core FILES ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_shared_srcs} ${onnxruntime_providers_cuda_cu_srcs})
  set(onnxruntime_providers_cuda_src ${onnxruntime_providers_cuda_cc_srcs} ${onnxruntime_providers_cuda_shared_srcs} ${onnxruntime_providers_cuda_cu_srcs})

  # Collect CUDA contrib ops sources
  file(GLOB_RECURSE onnxruntime_cuda_contrib_ops_cc_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.h"
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cc"
  )
  if(NOT onnxruntime_ENABLE_MLAS_QMOE_OFFLOAD)
    list(APPEND onnxruntime_cuda_contrib_ops_cc_srcs
      "${ONNXRUNTIME_ROOT}/contrib_ops/cpu/moe/qmoe_shared_layout.cc")
  endif()

  if(onnxruntime_MINIMAL_BUILD)
    list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cc_srcs
      "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.h"
      "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.cc"
    )
  endif()

  file(GLOB_RECURSE onnxruntime_cuda_contrib_ops_cu_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cu"
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cuh"
  )

  include(onnxruntime_cuda_source_filters.cmake)
  include(onnxruntime_cuda_cccl.cmake)
  onnxruntime_filter_cuda_cu_sources(onnxruntime_cuda_contrib_ops_cu_srcs)

  if (NOT onnxruntime_USE_TRT_FUSED_ATTENTION)
    # Drop the prebuilt TensorRT fused MHA cubin blobs. cudaDriverWrapper is kept because
    # sparse attention depends on it.
    list(FILTER onnxruntime_cuda_contrib_ops_cc_srcs EXCLUDE REGEX
      ".*/bert/tensorrt_fused_multihead_attention/.*(\\.cubin\\.cc|_kernel\\.sm[0-9]+\\.cc)$")
  endif()

  onnxruntime_extract_sm_specific_cuda_sources(onnxruntime_cuda_contrib_ops_cu_srcs
    SM90_SOURCES onnxruntime_cuda_sm90_tma_srcs
    SM120_SOURCES onnxruntime_cuda_sm120_tma_srcs
  )
  # Flash Attention sources are dropped entirely (not just excluded from this OBJECT library)
  # when onnxruntime_USE_FLASH_ATTENTION is OFF; see onnxruntime_extract_flash_attention_sources.
  onnxruntime_extract_flash_attention_sources(onnxruntime_cuda_contrib_ops_cu_srcs
    FLASH_SOURCES onnxruntime_cuda_flash_attention_srcs
  )
  onnxruntime_extract_xqa_sources(onnxruntime_cuda_contrib_ops_cu_srcs
    XQA_SOURCES onnxruntime_cuda_xqa_srcs
  )
  onnxruntime_extract_llm_sources(onnxruntime_cuda_contrib_ops_cu_srcs
    LLM_SOURCES onnxruntime_cuda_llm_srcs
    LLM_SM90_SOURCES onnxruntime_cuda_llm_sm90_srcs
    LLM_FP4_SOURCES onnxruntime_cuda_llm_fp4_srcs
  )
  if(MSVC OR UNIX)
    foreach(_src IN LISTS onnxruntime_cuda_llm_sm90_srcs)
      if(_src MATCHES "/(moe_gemm/deep_gemm_sm90|deep_gemm_matmul_sm90)\\.cu$")
        if(MSVC)
          set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "-Xcompiler=/wd4068")
        else()
          set_source_files_properties(${_src} PROPERTIES COMPILE_OPTIONS "-Xcompiler=-Wno-unknown-pragmas")
        endif()
      endif()
    endforeach()
  endif()

  # disable contrib ops conditionally
  if(NOT onnxruntime_DISABLE_CONTRIB_OPS AND NOT onnxruntime_CUDA_MINIMAL)
    if (NOT onnxruntime_ENABLE_ATEN)
      list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cc_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/aten_ops/aten_op.cc"
      )
    endif()
    if (NOT onnxruntime_USE_NCCL)
      list(REMOVE_ITEM onnxruntime_cuda_contrib_ops_cc_srcs
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/nccl_kernels.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/sharded_moe.h"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/sharded_moe.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/sharding_spec.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/sharding.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_matmul.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_slice.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_reshape.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_expand.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_reduce.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_unsqueeze.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/collective/distributed_squeeze.cc"
      )
    endif()
    # add using ONNXRUNTIME_ROOT so they show up under the 'contrib_ops' folder in Visual Studio
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_contrib_ops_cc_srcs} ${onnxruntime_cuda_contrib_ops_cu_srcs})
  elseif(onnxruntime_DISABLE_CONTRIB_OPS AND NOT onnxruntime_CUDA_MINIMAL)
    # The ONNX domain CUDA Attention kernel (core/providers/cuda/llm/attention.cc) depends on
    # attention infrastructure in contrib_ops/cuda/bert/ (flash attention, memory efficient
    # attention, unfused attention helpers, etc.). Include the bert attention infrastructure
    # even when contrib ops are disabled so that the ONNX Attention kernel can compile and link.
    set(onnxruntime_cuda_bert_cc_srcs ${onnxruntime_cuda_contrib_ops_cc_srcs})
    list(FILTER onnxruntime_cuda_bert_cc_srcs INCLUDE REGEX ".*/contrib_ops/cuda/bert/.*")
    set(onnxruntime_cuda_bert_cu_srcs ${onnxruntime_cuda_contrib_ops_cu_srcs})
    list(FILTER onnxruntime_cuda_bert_cu_srcs INCLUDE REGEX ".*/contrib_ops/cuda/bert/.*")
    source_group(TREE ${ONNXRUNTIME_ROOT} FILES ${onnxruntime_cuda_bert_cc_srcs} ${onnxruntime_cuda_bert_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_bert_cc_srcs} ${onnxruntime_cuda_bert_cu_srcs})
  endif()

  if (onnxruntime_ENABLE_TRAINING_OPS)
    file(GLOB_RECURSE onnxruntime_cuda_training_ops_cc_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.h"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cc"
    )

    file(GLOB_RECURSE onnxruntime_cuda_training_ops_cu_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cu"
      "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/*.cuh"
    )

    source_group(TREE ${ORTTRAINING_ROOT} FILES ${onnxruntime_cuda_training_ops_cc_srcs} ${onnxruntime_cuda_training_ops_cu_srcs})
    list(APPEND onnxruntime_providers_cuda_src ${onnxruntime_cuda_training_ops_cc_srcs} ${onnxruntime_cuda_training_ops_cu_srcs})

    if(NOT onnxruntime_ENABLE_TRAINING)
      file(GLOB_RECURSE onnxruntime_cuda_full_training_only_srcs
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/communication/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/record.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/record.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/wait.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/wait.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/controlflow/yield.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/gist/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/gist/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/gist/*.cu"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/torch/*.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/torch/*.h"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/triton/triton_op.cc"
      )

      list(REMOVE_ITEM onnxruntime_providers_cuda_src ${onnxruntime_cuda_full_training_only_srcs})
    elseif(WIN32 OR NOT onnxruntime_USE_NCCL)
      # NCCL is not support in Windows build
      file(GLOB_RECURSE onnxruntime_cuda_nccl_op_srcs
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_common.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/nccl_kernels.cc"
        "${ORTTRAINING_SOURCE_DIR}/training_ops/cuda/collective/megatron.cc"
      )
      list(REMOVE_ITEM onnxruntime_providers_cuda_src ${onnxruntime_cuda_nccl_op_srcs})
    endif()
  endif()

  if (onnxruntime_REDUCED_OPS_BUILD)
    substitute_op_reduction_srcs(onnxruntime_providers_cuda_src)
  endif()

  if(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
    # cuda_provider_interface.cc is removed from the object target: onnxruntime_providers_cuda_obj and
    # added to the lib onnxruntime_providers_cuda separately.
    # onnxruntime_providers_cuda_ut can share all the object files with onnxruntime_providers_cuda except cuda_provider_interface.cc.
    set(cuda_provider_interface_src ${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_provider_interface.cc)
    list(REMOVE_ITEM onnxruntime_providers_cuda_src ${cuda_provider_interface_src})
    onnxruntime_add_object_library(onnxruntime_providers_cuda_obj ${onnxruntime_providers_cuda_src})

    set(onnxruntime_providers_cuda_all_srcs ${cuda_provider_interface_src})
    if(WIN32)
      # Sets the DLL version info on Windows: https://learn.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource
      list(APPEND onnxruntime_providers_cuda_all_srcs "${ONNXRUNTIME_ROOT}/core/providers/cuda/onnxruntime_providers_cuda.rc")
    endif()

    onnxruntime_add_shared_library_module(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_all_srcs}
                                                                     $<TARGET_OBJECTS:onnxruntime_providers_cuda_obj>)
  else()
    set(onnxruntime_providers_cuda_all_srcs ${onnxruntime_providers_cuda_src})
    if(WIN32)
      # Sets the DLL version info on Windows: https://learn.microsoft.com/en-us/windows/win32/menurc/versioninfo-resource
      list(APPEND onnxruntime_providers_cuda_all_srcs "${ONNXRUNTIME_ROOT}/core/providers/cuda/onnxruntime_providers_cuda.rc")
    endif()

    onnxruntime_add_shared_library_module(onnxruntime_providers_cuda ${onnxruntime_providers_cuda_all_srcs})
  endif()

  if (MSVC)
    # Use /permissive to work around compilation error from CUTLASS header cute/tensor.hpp:
    #   cutlass-src\include\cute\stride.hpp(299,46): error C3545: 'Ints': parameter pack expects a non-type
    #     template argument
    # See https://github.com/NVIDIA/cutlass/issues/3065
    target_compile_options(onnxruntime_providers_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CXX>:/permissive>"
      "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /permissive>"
    )
  endif()

  if(WIN32)
    # FILE_NAME preprocessor definition is used in onnxruntime_providers_cuda.rc
    target_compile_definitions(onnxruntime_providers_cuda PRIVATE FILE_NAME=\"onnxruntime_providers_cuda.dll\")
  endif()

  # config_cuda_provider_shared_module can be used to config onnxruntime_providers_cuda_obj, onnxruntime_providers_cuda & onnxruntime_providers_cuda_ut.
  # This function guarantees that all 3 targets have the same configurations.
  function(config_cuda_provider_shared_module target)
    if (onnxruntime_REDUCED_OPS_BUILD)
      add_op_reduction_include_dirs(${target})
    endif()

    if (HAS_GUARD_CF)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /guard:cf>")
    endif()

    if (HAS_QSPECTRE)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Qspectre>")
    endif()

    foreach(ORT_FLAG ${ORT_WARNING_FLAGS})
      if (NOT "${ORT_FLAG}" STREQUAL "-Wshorten-64-to-32")
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler \"${ORT_FLAG}\">")
      endif()
    endforeach()

    # Note: CUDA 11.3+ supports parallel compilation
    # https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#options-for-guiding-compiler-driver-threads
    # --threads is NOT set here; it is applied per-target after calling this function
    # so that flash attention can use a different (lower) thread count.
    set(onnxruntime_NVCC_THREADS "4" CACHE STRING "Number of threads that NVCC can use for compilation.")

    # suppress warnings like this:
    #   cutlass-src\include\cute/arch/mma_sm120.hpp(3128): error #177-D: variable "tidA" was declared but never
    #     referenced
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>")
    # suppress cudafe "variable was set but never used" (#550-D) from flatbuffers/adapter headers
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=550>")

    # Since CUDA 12.8, compiling diagnostics become stricter
    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>")

      if (MSVC)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4505>")
      endif()
      # skip diagnosis error caused by cuda header files
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=221>")
      # NVCC false positive: assigning a [[nodiscard]] Status via operator= is flagged as discarding the value.
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2810>")
      # CUDA 12.8 also reports deprecated implicit by-copy 'this' captures from CUTLASS headers.
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=2908>")
    endif()

    if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
      if (UNIX)
        # Suppress -Wattributes warning from protobuf headers with nvcc on Linux
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-attributes>")
      endif()

      if (MSVC)
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20199>")
      endif()
    endif()

    if (UNIX)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-reorder>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-reorder>")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=sign-compare>"
                  "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=sign-compare>")
      if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IBMClang")
        foreach(CLANG_WARNING
          braced-scalar-init
          defaulted-function-deleted
          inconsistent-missing-override
          instantiation-after-specialization
          logical-op-parentheses
          mismatched-tags
          shorten-64-to-32
          unneeded-internal-declaration
          unknown-warning-option
          unused-private-field
          unused-variable)
          target_compile_options(${target} PRIVATE "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:-Wno-error=${CLANG_WARNING}>")
        endforeach()
        if (CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "Clang" OR CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "AppleClang" OR CMAKE_CUDA_HOST_COMPILER_ID STREQUAL "IBMClang")
          foreach(CLANG_WARNING
            braced-scalar-init
            defaulted-function-deleted
            inconsistent-missing-override
            instantiation-after-specialization
            logical-op-parentheses
            mismatched-tags
            shorten-64-to-32
            unneeded-internal-declaration
            unknown-warning-option
            unused-private-field
            unused-variable)
            target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler -Wno-error=${CLANG_WARNING}>")
          endforeach()
        endif()
      endif()
    else()
      #mutex.cuh(91): warning C4834: discarding return value of function with 'nodiscard' attribute
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>")
      if (MSVC)
        # the VS warnings for 'Conditional Expression is Constant' are spurious as they don't handle multiple conditions
        # e.g. `if (std::is_same_v<T, float> && not_a_const)` will generate the warning even though constexpr cannot
        # be used due to `&& not_a_const`. This affects too many places for it to be reasonable to disable at a finer
        # granularity.
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>")

        # Warning C4211: nonstandard extension used: redefined extern to static
        # non_max_suppression_impl.cu
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4211>")
      endif()
    endif()

    if(MSVC)
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus>")
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:preprocessor>")
      # Pass /bigobj to the CUDA host compiler using dash spelling. Raw /bigobj is excluded
      # from global ARM64 CUDA options in onnxruntime_common.cmake because nvcc parses it as input.
      target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-bigobj>")
      # /permissive is required for CUTLASS cute headers and to work around MSVC template resolution
      # issues with abseil headers when compiled through nvcc.
      # See https://github.com/NVIDIA/cutlass/issues/3065
      target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/permissive>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /permissive>"
      )
    endif()

    onnxruntime_add_include_to_target(${target} onnxruntime_common onnxruntime_framework onnx onnx_proto ${PROTOBUF_LIB} flatbuffers::flatbuffers)
    if (onnxruntime_ENABLE_TRAINING_OPS)
      onnxruntime_add_include_to_target(${target} onnxruntime_training)
      if (onnxruntime_ENABLE_TRAINING)
        target_link_libraries(${target} PRIVATE onnxruntime_training)
      endif()
      if (onnxruntime_ENABLE_TRAINING_TORCH_INTEROP OR onnxruntime_ENABLE_TRITON)
        onnxruntime_add_include_to_target(${target} Python::Module)
      endif()
    endif()

    add_dependencies(${target} onnxruntime_providers_shared ${onnxruntime_EXTERNAL_DEPENDENCIES})
    if(onnxruntime_CUDA_MINIMAL)
      target_compile_definitions(${target} PRIVATE USE_CUDA_MINIMAL)
      target_link_libraries(${target} PRIVATE ${ABSEIL_LIBS} ${ONNXRUNTIME_PROVIDERS_SHARED} Boost::mp11 safeint_interface CUDA::cudart)
    else()
      include(cudnn_frontend) # also defines CUDNN::*
      if (onnxruntime_USE_CUDA_NHWC_OPS)
        if(CUDNN_MAJOR_VERSION GREATER 8)
          add_compile_definitions(ENABLE_CUDA_NHWC_OPS)
        else()
          message( WARNING "To compile with NHWC ops enabled please compile against cuDNN 9 or newer." )
        endif()
      endif()
      target_compile_definitions(${target} PRIVATE NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)
      target_include_directories(${target} PRIVATE ${CUDNN_INCLUDE_DIR})
      # cuDNN and cuFFT are loaded dynamically at runtime (see cudnn_stub.cc / cufft_stub.cc), so they are
      # intentionally not linked here to avoid a hard runtime dependency when the related ops are not used.
      target_link_libraries(${target} PRIVATE CUDA::cublasLt CUDA::cublas cudnn_frontend CUDA::curand CUDA::cudart CUDA::cuda_driver
              ${ABSEIL_LIBS} ${ONNXRUNTIME_PROVIDERS_SHARED} onnxruntime_mlas Boost::mp11 safeint_interface)
    endif()

    include(cutlass)
    target_include_directories(${target} PRIVATE ${cutlass_SOURCE_DIR}/include ${cutlass_SOURCE_DIR}/examples ${cutlass_SOURCE_DIR}/tools/util/include)
    if(ORT_HAS_SM90_OR_LATER AND NOT WIN32 AND NOT onnxruntime_CUDA_MINIMAL AND NOT onnxruntime_DISABLE_CONTRIB_OPS)
      include(deep_gemm)
      target_include_directories(${target} PRIVATE ${deep_gemm_SOURCE_DIR}/deep_gemm/include)
      target_compile_definitions(${target} PRIVATE USE_DEEP_GEMM)
    endif()
    target_link_libraries(${target} PRIVATE Eigen3::Eigen)
    target_include_directories(${target} PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

    ort_configure_cuda_cccl(${target})

    # ${CMAKE_CURRENT_BINARY_DIR} is so that #include "onnxruntime_config.h" inside tensor_shape.h is found
    set_target_properties(${target} PROPERTIES LINKER_LANGUAGE CUDA)
    set_target_properties(${target} PROPERTIES FOLDER "ONNXRuntime")

    if(ORT_HAS_SM90_OR_LATER)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-w>)
      target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-DCUTLASS_ENABLE_GDC_FOR_SM90=1>)
      if(NOT MSVC)
        # The native SM90 (Hopper) TMA/WGMMA launchers pass CUTLASS TMA descriptor types through
        # NVCC-generated host stubs. With CUDA 13 + MSVC those stubs contain 128-byte over-aligned
        # by-value formal parameters, which triggers MSVC C2719 ("formal parameter with requested
        # alignment of 128 won't be aligned"). Disable the native SM90 fpA_intB (COMPILE_HOPPER_TMA_GEMMS)
        # and grouped MoE (COMPILE_HOPPER_TMA_GROUPED_GEMMS) TMA kernels on MSVC; the launcher bodies
        # become throwing stubs and the SM80 compatibility path still runs on Hopper at runtime.
        # See docs/contrib_ops/cuda/moe_qmoe.md section 14.1.
        target_compile_definitions(${target} PRIVATE COMPILE_HOPPER_TMA_GEMMS)
        target_compile_definitions(${target} PRIVATE COMPILE_HOPPER_TMA_GROUPED_GEMMS)
      endif()
      if (MSVC)
        # Do NOT add another /bigobj here: the MSVC block above already forwards it to cl.
        target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4172>")
      endif()
    endif()

    if(("120" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG OR "121" IN_LIST CMAKE_CUDA_ARCHITECTURES_ORIG) AND
       (NOT MSVC OR onnxruntime_USE_FP4_QMOE))
      target_compile_definitions(${target} PRIVATE COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS)
    endif()

    if (onnxruntime_ENABLE_CUDA_PROFILING) # configure cupti for cuda profiling
      target_link_libraries(${target} PRIVATE CUDA::cupti)
    endif()

    if (onnxruntime_ENABLE_NVTX_PROFILE)
      target_link_libraries(${target} PRIVATE CUDA::nvtx3)
    endif()

    if (onnxruntime_ENABLE_TRAINING_OPS)
      target_include_directories(${target} PRIVATE ${ORTTRAINING_ROOT} ${MPI_CXX_INCLUDE_DIRS})
    endif()

    if (onnxruntime_USE_NCCL)
      target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIRS})
      target_link_libraries(${target} PRIVATE ${NCCL_LIBRARIES})
    endif()

    if (WIN32)
      # *.cu cannot use PCH
      if (NOT onnxruntime_BUILD_CACHE)
        target_precompile_headers(${target} PUBLIC
          "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.h"
          "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_pch.cc"
        )
      endif()

      # minimize the Windows includes.
      # this avoids an issue with CUDA 11.6 where 'small' is defined in the windows and cuda headers.
      target_compile_definitions(${target} PRIVATE "WIN32_LEAN_AND_MEAN")

      # disable a warning from the CUDA headers about unreferenced local functions
      #target_compile_options(${target} PRIVATE /wd4505)
      set(onnxruntime_providers_cuda_static_library_flags
          -IGNORE:4221 # LNK4221: This object file does not define any previously undefined public symbols, so it will not be used by any link operation that consumes this library
      )
      set_target_properties(${target} PROPERTIES
          STATIC_LIBRARY_FLAGS "${onnxruntime_providers_cuda_static_library_flags}")
    endif()

    if(APPLE)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/core/providers/cuda/exported_symbols.lst")
    elseif(UNIX)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/core/providers/cuda/version_script.lds -Xlinker --gc-sections")
    elseif(WIN32)
      set_property(TARGET ${target} APPEND_STRING PROPERTY LINK_FLAGS "-DEF:${ONNXRUNTIME_ROOT}/core/providers/cuda/symbols.def")
    else()
      message(FATAL_ERROR "${target} unknown platform, need to specify shared library exports for it")
    endif()

    if (onnxruntime_ENABLE_ATEN)
      target_compile_definitions(${target} PRIVATE ENABLE_ATEN)
    endif()
  endfunction()
  if(onnxruntime_ENABLE_CUDA_EP_INTERNAL_TESTS)
    config_cuda_provider_shared_module(onnxruntime_providers_cuda_obj)
    target_compile_options(onnxruntime_providers_cuda_obj PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_NVCC_THREADS}\">")
  endif()
  config_cuda_provider_shared_module(onnxruntime_providers_cuda)
  target_compile_options(onnxruntime_providers_cuda PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_NVCC_THREADS}\">")

  # Create OBJECT libraries for SM-specific contrib CUDA sources that must be compiled
  # with restricted CUDA architectures. These files use CUTLASS 3.x SM90+/SM120+ features
  # (GMMA, TMA) that cannot produce useful device code for older architectures.
  #
  # SM90/SM120 TMA and LLM OBJECT libraries contain MoE and MatMulNBits kernels (contrib ops).
  # Flash Attention is also used by the ONNX domain Attention op, so when
  # onnxruntime_USE_FLASH_ATTENTION is ON it is included even when contrib ops are disabled.
  if(NOT onnxruntime_CUDA_MINIMAL)
    # Flash Attention OBJECT library: SM80+ only, with independent nvcc_threads.
    # Flash Attention V2 kernels require SM80 (Ampere) and are memory-intensive to compile.
    # Isolating them allows the rest of the build to use higher --threads without OOM.
    # Included even with onnxruntime_DISABLE_CONTRIB_OPS (when onnxruntime_USE_FLASH_ATTENTION
    # is ON) because the ONNX domain Attention kernel depends on flash attention infrastructure
    # in contrib_ops/cuda/bert/. onnxruntime_cuda_flash_attention_srcs is only populated when
    # onnxruntime_USE_FLASH_ATTENTION is ON; otherwise the .cu sources are excluded from the
    # build entirely (see extraction above), and onnxruntime_USE_FLASH_ATTENTION always takes
    # precedence over onnxruntime_DISABLE_CONTRIB_OPS.
    set(onnxruntime_FLASH_NVCC_THREADS "1" CACHE STRING
        "Number of NVCC threads for Flash Attention compilation (memory-intensive, keep low).")
    if(onnxruntime_cuda_flash_attention_srcs)
      onnxruntime_filter_cuda_archs(_ort_flash_cuda_architectures MIN_SM 80)
      if(_ort_flash_cuda_architectures)
        onnxruntime_add_cuda_object_library(
          NAME onnxruntime_providers_cuda_flash_attention
          PARENT onnxruntime_providers_cuda
          CUDA_ARCHITECTURES "${_ort_flash_cuda_architectures}"
          NVCC_THREADS "${onnxruntime_FLASH_NVCC_THREADS}"
          SOURCES ${onnxruntime_cuda_flash_attention_srcs})
      else()
        # No SM80+ architectures available: compile flash sources in parent target so the
        # linker can find the host-side symbols referenced by flash_api.cc. The kernels
        # themselves will be empty stubs due to __CUDA_ARCH__ >= 800 guards.
        target_sources(onnxruntime_providers_cuda PRIVATE ${onnxruntime_cuda_flash_attention_srcs})
      endif()
    endif()

    # XQA kernels require SM80+. Compiling them with mixed SM75/SM80+ architectures
    # causes CUDA 13.3 to emit host references to kernels omitted from the SM75 pass.
    if(onnxruntime_cuda_xqa_srcs)
      onnxruntime_filter_cuda_archs(_ort_xqa_cuda_architectures MIN_SM 80)
      if(_ort_xqa_cuda_architectures)
        onnxruntime_add_cuda_object_library(
          NAME onnxruntime_providers_cuda_xqa
          PARENT onnxruntime_providers_cuda
          CUDA_ARCHITECTURES "${_ort_xqa_cuda_architectures}"
          NVCC_THREADS "${onnxruntime_NVCC_THREADS}"
          SOURCES ${onnxruntime_cuda_xqa_srcs})
      else()
        if(TARGET onnxruntime_providers_cuda_obj)
          target_sources(onnxruntime_providers_cuda_obj PRIVATE ${onnxruntime_cuda_xqa_srcs})
        else()
          target_sources(onnxruntime_providers_cuda PRIVATE ${onnxruntime_cuda_xqa_srcs})
        endif()
      endif()
    endif()

    if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
      # SM90 TMA warp-specialized files use SM90-specific collective operations.
      # Compile at exactly 90a-real: sm_90a WGMMA/TMA code runs only on SM90, not SM100/SM120.
      # Also includes fpA_intB SM90 launchers (guarded by #ifndef EXCLUDE_SM_90).
      if(onnxruntime_cuda_sm90_tma_srcs OR onnxruntime_cuda_llm_sm90_srcs)
        set(_ort_sm90_all_srcs ${onnxruntime_cuda_sm90_tma_srcs} ${onnxruntime_cuda_llm_sm90_srcs})
        onnxruntime_filter_cuda_archs(_ort_sm90_check MIN_SM 90)
        if(_ort_sm90_check)
          onnxruntime_add_cuda_object_library(
            NAME onnxruntime_providers_cuda_sm90_tma
            PARENT onnxruntime_providers_cuda
            CUDA_ARCHITECTURES "90a-real"
            NVCC_THREADS "${onnxruntime_NVCC_THREADS}"
            SOURCES ${_ort_sm90_all_srcs})
        endif()
      endif()

      # CUDA 13 gives CUtensorMap 128-byte host alignment when MSVC reports an accurate
      # __cplusplus value. The target-specific host flag below avoids C2719 for FP4 QMoE.
      if(onnxruntime_cuda_sm120_tma_srcs AND
         (NOT MSVC OR CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0 OR onnxruntime_USE_FP4_QMOE))
        onnxruntime_filter_cuda_archs(_ort_sm120_cuda_architectures MIN_SM 120)
        if(_ort_sm120_cuda_architectures)
          onnxruntime_add_cuda_object_library(
            NAME onnxruntime_providers_cuda_sm120_tma
            PARENT onnxruntime_providers_cuda
            CUDA_ARCHITECTURES "${_ort_sm120_cuda_architectures}"
            NVCC_THREADS "${onnxruntime_NVCC_THREADS}"
            SOURCES ${onnxruntime_cuda_sm120_tma_srcs})
          if(MSVC AND CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13.0)
            # Keep CUDA's CUtensorMap payload unchanged while avoiding an
            # alignas(128) by-value parameter that MSVC cannot represent in
            # NVCC-generated host stubs (C2719).
            target_compile_options(onnxruntime_providers_cuda_sm120_tma PRIVATE
              "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus->")
          endif()
          target_compile_definitions(onnxruntime_providers_cuda PRIVATE ORT_ENABLE_BLOCKQUANT_SM120)
          if(TARGET onnxruntime_providers_cuda_obj)
            target_compile_definitions(onnxruntime_providers_cuda_obj PRIVATE ORT_ENABLE_BLOCKQUANT_SM120)
          endif()
        endif()
      endif()

      # LLM OBJECT library: SM75+ (backward compatible with fpA_intB_gemv/gemm which support SM75).
      # Restricts CUDA_ARCHITECTURES to avoid compiling heavy CUTLASS templates for pre-Turing GPUs.
      #
      # SM120+ real (native SASS) is included by default: the general LLM sources contain no
      # SM120-only device code of their own (the SM120 TMA warp-specialized kernels live in the
      # separate onnxruntime_providers_cuda_sm120_tma OBJECT library, and their bodies are gated
      # behind COMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS), so building this library at 120-real
      # simply produces native SASS for the ordinary LLM kernels (moe expand/finalize, fpA_intB
      # gemv/gemm, NVFP4 dequant, ...) instead of leaving them as compute_120 PTX that must be
      # JIT-compiled on first use. Native SASS avoids JIT warm-up and, crucially, avoids a
      # "no kernel image is available" (CUDA 209) failure when the arch list is real-only
      # (e.g. 86-real;120-real) and therefore carries no virtual compute_120 PTX fallback.
      #
      # Native sm_120a pulls CCCL tcgen05 PTX headers that fail with the MSVC host compiler, so
      # the broad LLM target uses virtual compute_120 PTX on Windows. FP4 QMoE is isolated below:
      # its activation conversion requires real sm_120a for `cvt.e2m1x2` and its smaller source
      # set does not pull in the problematic tcgen05 path.
      if(onnxruntime_cuda_llm_srcs)
        if(MSVC)
          onnxruntime_filter_cuda_archs(_ort_llm_cuda_architectures MIN_SM 75 REPLACE_SM120_REAL_WITH_VIRTUAL)
        else()
          onnxruntime_filter_cuda_archs(_ort_llm_cuda_architectures MIN_SM 75)
        endif()
        if(_ort_llm_cuda_architectures)
          onnxruntime_add_cuda_object_library(
            NAME onnxruntime_providers_cuda_llm
            PARENT onnxruntime_providers_cuda
            CUDA_ARCHITECTURES "${_ort_llm_cuda_architectures}"
            NVCC_THREADS "${onnxruntime_NVCC_THREADS}"
            SOURCES ${onnxruntime_cuda_llm_srcs})
        endif()
      endif()

      if(onnxruntime_cuda_llm_fp4_srcs)
        onnxruntime_filter_cuda_archs(_ort_llm_fp4_cuda_architectures MIN_SM 75)
        if(_ort_llm_fp4_cuda_architectures)
          onnxruntime_add_cuda_object_library(
            NAME onnxruntime_providers_cuda_llm_fp4
            PARENT onnxruntime_providers_cuda
            CUDA_ARCHITECTURES "${_ort_llm_fp4_cuda_architectures}"
            NVCC_THREADS "${onnxruntime_NVCC_THREADS}"
            SOURCES ${onnxruntime_cuda_llm_fp4_srcs})
        endif()
      endif()
    endif()
  endif()

  # Cannot use glob because the file cuda_provider_options.h should not be exposed out.
  set(ONNXRUNTIME_CUDA_PROVIDER_PUBLIC_HEADERS
        "${REPO_ROOT}/include/onnxruntime/core/providers/cuda/cuda_context.h"
        "${REPO_ROOT}/include/onnxruntime/core/providers/cuda/cuda_resource.h"
      )
  set_target_properties(onnxruntime_providers_cuda PROPERTIES
    PUBLIC_HEADER "${ONNXRUNTIME_CUDA_PROVIDER_PUBLIC_HEADERS}")
  if(WIN32 AND NOT onnxruntime_CUDA_MINIMAL)
    set(ORT_CUDNN_DLL_PATH "")
    if(onnxruntime_CUDNN_HOME)
      set(ORT_CUDNN_DLL_SEARCH_PATHS
        "${onnxruntime_CUDNN_HOME}/bin/cudnn64_*.dll"
        "${onnxruntime_CUDNN_HOME}/bin/x64/cudnn64_*.dll"
        "${onnxruntime_CUDNN_HOME}/bin/${onnxruntime_CUDA_VERSION}/cudnn64_*.dll"
        "${onnxruntime_CUDNN_HOME}/bin/${onnxruntime_CUDA_VERSION}/x64/cudnn64_*.dll"
      )
    else()
      set(ORT_CUDNN_DLL_SEARCH_PATHS "${onnxruntime_CUDA_HOME}/bin/cudnn64_*.dll")
    endif()
    foreach(search_path ${ORT_CUDNN_DLL_SEARCH_PATHS})
      file(GLOB ORT_CUDNN_DLL_PATH "${search_path}")
      if(ORT_CUDNN_DLL_PATH)
        break()
      endif()
    endforeach()
    if(ORT_CUDNN_DLL_PATH)
      add_custom_command(TARGET onnxruntime_providers_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ORT_CUDNN_DLL_PATH} $<TARGET_FILE_DIR:onnxruntime_providers_cuda>
      )
    endif()
  endif()
  install(TARGETS onnxruntime_providers_cuda
          PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/onnxruntime/core/providers/cuda
          ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
          RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})

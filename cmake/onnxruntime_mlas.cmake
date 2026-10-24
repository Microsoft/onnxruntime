# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(MLAS_ROOT ${ONNXRUNTIME_ROOT}/core/mlas)
set(MLAS_SRC_DIR ${MLAS_ROOT}/lib)
set(MLAS_INC_DIR ${MLAS_ROOT}/inc)


# mlas_private_compile_definitions contains compile definitions that are private to onnxruntime_mlas and targets which
# use internal MLAS headers like mlasi.h.
set(mlas_private_compile_definitions)
if(onnxruntime_BUILD_UNIT_TESTS)
  list(APPEND mlas_private_compile_definitions MLAS_ENABLE_TEST_HOOKS)
endif()

# Adds compile definitions to mlas_private_compile_definitions from any scope, including from within nested function
# calls. set(... PARENT_SCOPE) propagates a value exactly one scope up, so it silently drops the definition when the
# caller is itself a function (e.g. setup_arm_neon_nchwc() called from setup_mlas_source_for_windows()). Directory
# properties are not affected by function scopes, so accumulate the definitions in one and merge them back into
# mlas_private_compile_definitions before it is consumed.
function(mlas_add_private_compile_definitions)
  set_property(DIRECTORY APPEND PROPERTY mlas_private_compile_definitions_from_functions ${ARGN})
endfunction()
#
# All hardware agnostic source files here
# hardware specific files would cause trouble in
# multi-target build
#
onnxruntime_add_static_library(onnxruntime_mlas
  ${MLAS_SRC_DIR}/mlasi.h
  ${MLAS_SRC_DIR}/platform.cpp
  ${MLAS_SRC_DIR}/threading.cpp
  ${MLAS_SRC_DIR}/sgemm.cpp
  ${MLAS_SRC_DIR}/halfgemm.cpp
  ${MLAS_SRC_DIR}/halfconv.cpp
  ${MLAS_SRC_DIR}/qgemm.cpp
  ${MLAS_SRC_DIR}/qdwconv.cpp
  ${MLAS_SRC_DIR}/convolve.cpp
  ${MLAS_SRC_DIR}/sconv_nchw_depthwise_multiplier_greater_than_1.cpp
  ${MLAS_SRC_DIR}/convsym.cpp
  ${MLAS_SRC_DIR}/pooling.cpp
  ${MLAS_SRC_DIR}/transpose.cpp
  ${MLAS_SRC_DIR}/reorder.cpp
  ${MLAS_SRC_DIR}/snchwc.cpp
  ${MLAS_SRC_DIR}/activate.cpp
  ${MLAS_SRC_DIR}/logistic.cpp
  ${MLAS_SRC_DIR}/tanh.cpp
  ${MLAS_SRC_DIR}/eltwise.h
  ${MLAS_SRC_DIR}/eltwise.cpp
  ${MLAS_SRC_DIR}/erf.cpp
  ${MLAS_SRC_DIR}/silu.cpp
  ${MLAS_SRC_DIR}/gelu.cpp
  ${MLAS_SRC_DIR}/compute.cpp
  ${MLAS_SRC_DIR}/dequantize.cpp
  ${MLAS_SRC_DIR}/quantize.cpp
  ${MLAS_SRC_DIR}/qgemm_kernel_default.cpp
  ${MLAS_SRC_DIR}/qladd.cpp
  ${MLAS_SRC_DIR}/qlmul.cpp
  ${MLAS_SRC_DIR}/qpostprocessor.cpp
  ${MLAS_SRC_DIR}/qlgavgpool.cpp
  ${MLAS_SRC_DIR}/qdwconv_kernelsize.cpp
  ${MLAS_SRC_DIR}/qnbitgemm.h
  ${MLAS_SRC_DIR}/qnbitgemm.cpp
  ${MLAS_SRC_DIR}/qlutgemm.h
  ${MLAS_SRC_DIR}/qlutgemm.cpp
  ${MLAS_SRC_DIR}/sqnbitgemm_q8_block.h
  ${MLAS_SRC_DIR}/flashattn.cpp
  ${MLAS_SRC_DIR}/flashattn_qkv.cpp
  ${MLAS_SRC_DIR}/flashattn_gqa.cpp
  ${MLAS_SRC_DIR}/qkv_quant.cpp
  ${MLAS_SRC_DIR}/linear_attention.h
  ${MLAS_SRC_DIR}/linear_attention.cpp
  ${MLAS_SRC_DIR}/cast.cpp
  ${MLAS_SRC_DIR}/layernorm.cpp
  ${MLAS_SRC_DIR}/rotary_embedding.h
  ${MLAS_SRC_DIR}/rotary_embedding.cpp
  ${MLAS_SRC_DIR}/softmax.h
  ${MLAS_SRC_DIR}/saturation_check.cpp
)

target_sources(onnxruntime_mlas PRIVATE
  ${MLAS_INC_DIR}/mlas_float16.h
  ${MLAS_INC_DIR}/mlas_gemm_postprocessor.h
  ${MLAS_INC_DIR}/mlas_q4.h
  ${MLAS_INC_DIR}/mlas_qnbit.h
  ${MLAS_INC_DIR}/mlas_qkv_quant.h
  ${MLAS_INC_DIR}/mlas.h
)

if (NOT onnxruntime_ORT_MINIMAL_BUILD)
  target_sources(onnxruntime_mlas PRIVATE
    ${MLAS_SRC_DIR}/q4_dq.cpp
    ${MLAS_SRC_DIR}/q4gemm.cpp
  )
endif()

set(ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas)

#TODO: set MASM flags properly
function(setup_mlas_source_for_windows)

  #
  # Sources common for all platforms.
  #
  target_sources(onnxruntime_mlas PRIVATE
    ${MLAS_SRC_DIR}/activate_fp16.cpp
    ${MLAS_SRC_DIR}/dwconv.cpp
    ${MLAS_SRC_DIR}/pooling_fp16.cpp
  )

  #The onnxruntime_target_platform variable was added by Windows AI team in onnxruntime_common.cmake
  #Don't use it for other platforms.
  if((onnxruntime_target_platform STREQUAL "ARM64") OR (onnxruntime_target_platform STREQUAL "ARM64EC"))
    set(PREPROCESS_ARMASM_FLAGS "")
    set(ARMASM_FLAGS "")

    if(onnxruntime_target_platform STREQUAL "ARM64")
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/halfgemm_kernel_neon.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_udot.cpp
        ${MLAS_SRC_DIR}/qgemm_kernel_sdot.cpp
        ${MLAS_SRC_DIR}/qnbitgemm_kernel_neon.h
        ${MLAS_SRC_DIR}/qnbitgemm_kernel_neon.cpp
        ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_fp32.cpp
        ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8.cpp
        # Portable W2 pack helpers + scalar reference kernel. Misleadingly
        # Avx512-named because the AVX-512 W2 path was the first consumer, but
        # the TU contains no x86 intrinsics (see sqnbitgemm_kernel_avx512_2bit.cpp).
        # ARM64 W2 dispatch reuses these for pack-size / pack / layout; the
        # native compute kernel is the NEON DotProd TU listed just below.
        ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.h
        ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.cpp
        # W2 CompInt8 DotProd kernel (NEON FEAT_DotProd backend).
        ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8_2bit.cpp
        ${MLAS_SRC_DIR}/cast_kernel_neon.cpp
        ${MLAS_SRC_DIR}/hqnbitgemm_kernel_neon_fp16.cpp
        ${MLAS_SRC_DIR}/hqnbitgemm_kernel_neon_fp16_8bit.cpp
        ${MLAS_SRC_DIR}/rotary_embedding_kernel_neon.h
        ${MLAS_SRC_DIR}/rotary_embedding_kernel_neon.cpp
        ${MLAS_SRC_DIR}/rotary_embedding_kernel_neon_fp16.cpp
        ${MLAS_SRC_DIR}/qkv_quant_kernel.h
        ${MLAS_SRC_DIR}/qkv_quant_common.h
        ${MLAS_SRC_DIR}/qkv_quant_kernel_neon.cpp
        ${MLAS_SRC_DIR}/hgemm_kernel_neon.cpp
        ${MLAS_SRC_DIR}/halfgemm_kernel_neon_fp16.cpp
        ${MLAS_SRC_DIR}/softmax_kernel_neon.h
        ${MLAS_SRC_DIR}/softmax_kernel_neon.cpp
        ${MLAS_SRC_DIR}/softmax_kernel_neon_fp16.cpp
        ${MLAS_SRC_DIR}/eltwise_kernel_neon.h
        ${MLAS_SRC_DIR}/eltwise_kernel_neon.cpp
        ${MLAS_SRC_DIR}/eltwise_kernel_neon_fp16.cpp
        ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8_i8mm.cpp
        ${MLAS_SRC_DIR}/sconv_nchw_depthwise_multiplier_1.cpp
        ${MLAS_SRC_DIR}/linear_attention_kernel_neon.cpp
      )

      set(mlas_platform_preprocess_srcs
        ${MLAS_SRC_DIR}/arm64/ConvSymS8KernelDot.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymS8KernelDotLd64.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymU8KernelDot.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymS8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/ConvSymU8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/DepthwiseQConvSymS8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/DepthwiseQConvSymU8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/DepthwiseQConvKernelSize9Neon.asm
        ${MLAS_SRC_DIR}/arm64/HalfGemmKernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/QgemmU8X8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/QgemmS8S8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/QgemmU8X8KernelUdot.asm
        ${MLAS_SRC_DIR}/arm64/QgemmS8S8KernelSdot.asm
        ${MLAS_SRC_DIR}/arm64/SgemmKernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/SgemvKernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/SymQgemmS8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64/SymQgemmS8KernelSDot.asm
        ${MLAS_SRC_DIR}/arm64/SymQgemmS8KernelSDotLd64.asm
      )

      if (onnxruntime_USE_ARM_NEON_NCHWC)
		setup_arm_neon_nchwc()
	  endif()

	  if (onnxruntime_USE_KLEIDIAI)
        setup_kleidiai()
      endif()

      if (onnxruntime_USE_SVE)
        # Portable machine-code SVE kernels: plain C++ dispatch/driver layers
        # plus hex-encoded assembly. kai_asm_macros.h resolves to armasm64
        # directives (AREA/PROC/DCD) during the cl.exe preprocessing step
        # applied to mlas_platform_preprocess_srcs below, so no SVE toolchain
        # support is required for either the elementwise or the QGEMM kernels.
        target_sources(onnxruntime_mlas PRIVATE
          ${MLAS_SRC_DIR}/sve/mlasi_sve.h
          ${MLAS_SRC_DIR}/sve/elementwise_sve_dispatch.cpp
          ${MLAS_SRC_DIR}/sve/qgemm_mmla_sve.h
          ${MLAS_SRC_DIR}/sve/qgemm_kernel_smmla_sve.cpp
          ${MLAS_SRC_DIR}/sve/qgemm_kernel_ummla_sve.cpp
          ${MLAS_SRC_DIR}/sve/linear_attention_sve.h
          ${MLAS_SRC_DIR}/sve/linear_attention_kernel_sve.cpp
        )
        list(APPEND mlas_platform_preprocess_srcs ${MLAS_SRC_DIR}/aarch64/elementwise_sve_asm.S)
        list(APPEND mlas_platform_preprocess_srcs ${MLAS_SRC_DIR}/aarch64/qgemm_mmla_sve_asm.S)
        list(APPEND mlas_platform_preprocess_srcs ${MLAS_SRC_DIR}/aarch64/linear_attention_sve_asm.S)
        mlas_add_private_compile_definitions(MLAS_USE_SVE)
      endif()
    else()
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
      )

      set(mlas_platform_preprocess_srcs
        ${MLAS_SRC_DIR}/arm64ec/QgemmU8X8KernelNeon.asm
        ${MLAS_SRC_DIR}/arm64ec/SgemmKernelNeon.asm
      )

      string(APPEND PREPROCESS_ARMASM_FLAGS " /arm64EC")
      string(APPEND ARMASM_FLAGS " -machine ARM64EC")
    endif()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      string(APPEND ARMASM_FLAGS " -g")
    endif()

    # Remove double quotes from flag strings.
    separate_arguments(PREPROCESS_ARMASM_FLAGS NATIVE_COMMAND "${PREPROCESS_ARMASM_FLAGS}")
    separate_arguments(ARMASM_FLAGS NATIVE_COMMAND "${ARMASM_FLAGS}")

    # Run the C precompiler on each input before the assembler.
    foreach(asm_filename ${mlas_platform_preprocess_srcs})
      get_filename_component(asm_filename_base ${asm_filename} NAME_WLE)
      set(preprocess_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.i)
      set(obj_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.obj)
      add_custom_command(
        OUTPUT ${obj_filename}
          COMMAND
              cl.exe ${PREPROCESS_ARMASM_FLAGS} /P ${asm_filename} /Fi${preprocess_filename}
          COMMAND
              armasm64.exe ${ARMASM_FLAGS} ${preprocess_filename} ${obj_filename}
        DEPENDS ${asm_filename}
        BYPRODUCTS ${preprocess_filename}
      )
      target_sources(onnxruntime_mlas PRIVATE ${obj_filename})
    endforeach()
  elseif(onnxruntime_target_platform STREQUAL "ARM")
    target_sources(onnxruntime_mlas PRIVATE
      ${MLAS_SRC_DIR}/arm/sgemmc.cpp
    )
  elseif(onnxruntime_target_platform STREQUAL "x64")

    file(GLOB_RECURSE mlas_platform_srcs_avx CONFIGURE_DEPENDS
      "${MLAS_SRC_DIR}/intrinsics/avx/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "/arch:AVX")

    file(GLOB_RECURSE mlas_platform_srcs_avx2 CONFIGURE_DEPENDS
      "${MLAS_SRC_DIR}/intrinsics/avx2/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "/arch:AVX2")

    set_source_files_properties(${MLAS_SRC_DIR}/layernorm_kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")

    set(mlas_platform_srcs_avx512
      ${MLAS_SRC_DIR}/intrinsics/avx512/gelu_avx512f.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/silu_avx512f.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/sconv_nchw_depthwise_multiplier_greater_than_1_avx512f.cpp
      ${MLAS_SRC_DIR}/linear_attention_kernel_avx512f.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/reorder_avx512f.cpp
      ${MLAS_SRC_DIR}/intrinsics/avx512/sconv_depthwise_avx512f.cpp
    )

    set_source_files_properties(${mlas_platform_srcs_avx512} PROPERTIES COMPILE_FLAGS "/arch:AVX512")

    target_sources(onnxruntime_mlas PRIVATE
      ${MLAS_SRC_DIR}/dgemm.cpp
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${MLAS_SRC_DIR}/layernorm_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.h
      ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/qkv_quant_kernel.h
      ${MLAS_SRC_DIR}/qkv_quant_common.h
      ${MLAS_SRC_DIR}/qkv_quant_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
      ${mlas_platform_srcs_avx512}
      ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.h
      ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2vnni.cpp
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.h
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.cpp
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit_blklen64.h
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit_blklen128.h
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit_blklen32.h
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512.cpp
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni.cpp
      ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni_2bit.cpp
      ${MLAS_SRC_DIR}/qkv_quant_kernel_avx512vnni.cpp
      ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAmx.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8S8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8U8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemmU8X8KernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvx512Vnni.asm
      ${MLAS_SRC_DIR}/amd64/QgemvU8S8KernelAvxVnni.asm
      ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx2.asm
      ${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx512Core.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/DgemmKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelM1Avx.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/SgemmKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/SconvKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelSse2.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SpoolKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/sgemma.asm
      ${MLAS_SRC_DIR}/amd64/cvtfp16a.asm
      ${MLAS_SRC_DIR}/amd64/SoftmaxKernelAvx.asm
      ${MLAS_SRC_DIR}/amd64/SoftmaxKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/TransKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/TransKernelAvx512F.asm
      ${MLAS_SRC_DIR}/amd64/LogisticKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/TanhKernelFma3.asm
      ${MLAS_SRC_DIR}/amd64/ErfKernelFma3.asm
    )

    if(onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER)
      set_source_files_properties(${MLAS_SRC_DIR}/amd64/ConvSymKernelAvx2.asm PROPERTIES COMPILE_FLAGS "-DENABLE_CONVSYMKERNELAVX2_SAT_CHECKER")
    endif()

    if(MSVC_VERSION GREATER_EQUAL 1933)
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/amd64/cvtfp16Avx.asm
      )
    endif()

    if (NOT onnxruntime_ORT_MINIMAL_BUILD)
      target_sources(onnxruntime_mlas PRIVATE
        ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
      )
    endif()
  else()
    set_source_files_properties(${MLAS_SRC_DIR}/layernorm_kernel_avx2.cpp PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    target_sources(onnxruntime_mlas PRIVATE
      ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
      ${MLAS_SRC_DIR}/qgemm_kernel_sse41.cpp
      ${MLAS_SRC_DIR}/layernorm_kernel_avx2.cpp
      ${MLAS_SRC_DIR}/i386/SgemmKernelSse2.asm
      ${MLAS_SRC_DIR}/i386/SgemmKernelAvx.asm
    )
  endif()
endfunction()

function(setup_kleidiai)
  target_sources(onnxruntime_mlas PRIVATE
    ${MLAS_SRC_DIR}/kleidiai/kai_ukernel_interface.cpp
    ${MLAS_SRC_DIR}/kleidiai/sgemm_kleidiai.cpp
    ${MLAS_SRC_DIR}/kleidiai/halfgemm_kleidiai.cpp
    ${MLAS_SRC_DIR}/kleidiai/halfconv_kleidiai.cpp
    ${MLAS_SRC_DIR}/kleidiai/sbgemm_kleidiai.cpp
    ${MLAS_SRC_DIR}/kleidiai/convolve_kleidiai.cpp
    ${MLAS_SRC_DIR}/kleidiai/qgemm_kleidiai.cpp
    ${MLAS_SRC_DIR}/kleidiai/qnbitgemm_kleidiai.cpp
  )
  target_link_libraries(onnxruntime_mlas PRIVATE kleidiai)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES kleidiai)
  if(onnxruntime_USE_QMX_KLEIDIAI_COEXIST)
          target_link_libraries(onnxruntime_mlas PRIVATE  kleidiai-qmx)
          target_compile_definitions(onnxruntime_mlas PRIVATE ENABLE_QMX_KERNELS=1)
          list(APPEND onnxruntime_EXTERNAL_LIBRARIES kleidiai-qmx)
  endif()
  set(onnxruntime_EXTERNAL_LIBRARIES ${onnxruntime_EXTERNAL_LIBRARIES} PARENT_SCOPE)

  # If KLEIDIAI_DEBUG_LOGGING is enabled that implies both DEBUG and KERNEL messages.
  if(onnxruntime_KLEIDIAI_DEBUG_LOGGING)
    target_compile_definitions(onnxruntime_mlas PRIVATE KLEIDIAI_DEBUG_LOGGING=1)
    target_compile_definitions(onnxruntime_mlas PRIVATE KLEIDIAI_KERNEL_LOGGING=1)
  endif()
  if(onnxruntime_KLEIDIAI_KERNEL_LOGGING)
    target_compile_definitions(onnxruntime_mlas PRIVATE KLEIDIAI_KERNEL_LOGGING=1)
  endif()

  if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS kleidiai EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()

  if(onnxruntime_USE_QMX_KLEIDIAI_COEXIST)
    install(TARGETS kleidiai-qmx EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endfunction()

function (setup_arm_neon_nchwc)
  target_sources(onnxruntime_mlas PRIVATE
   ${MLAS_SRC_DIR}/sconv_nchwc_kernel_neon.h
   ${MLAS_SRC_DIR}/sconv_nchwc_kernel_neon.cpp
   ${MLAS_SRC_DIR}/spool_nchwc_kernel_neon.cpp
  )
  if(NOT WIN32)
    target_sources(onnxruntime_mlas PRIVATE
     # Hand written AArch64 micro-kernel for NCHW convolution.  Using a
     # separate assembly file allows tighter control over register allocation
     # and avoids the overhead of C++/intrinsics based code generation.
     ${MLAS_SRC_DIR}/aarch64/SconvKernelNeon.S
     ${MLAS_SRC_DIR}/aarch64/SconvNchwcKernelNeon.S
     ${MLAS_SRC_DIR}/aarch64/SconvDepthwiseKernelNeon.S
     ${MLAS_SRC_DIR}/aarch64/SconvPointwiseKernelNeon.S
     )
  endif()
  mlas_add_private_compile_definitions(MLAS_USE_ARM_NEON_NCHWC)
endfunction ()

if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    file(GLOB_RECURSE mlas_platform_srcs
      "${MLAS_SRC_DIR}/wasm_simd/*.cpp"
    )
    set(mlas_platform_srcs
      ${mlas_platform_srcs}
      ${MLAS_SRC_DIR}/qgemm_kernel_wasmsimd.cpp
    )
    if (onnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD)
      set(mlas_platform_srcs
        ${mlas_platform_srcs}
        ${MLAS_SRC_DIR}/qgemm_kernel_wasmrelaxedsimd.cpp
      )
    endif()
  else()
    file(GLOB_RECURSE mlas_platform_srcs
      "${MLAS_SRC_DIR}/scalar/*.cpp"
    )
  endif()
  target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})
elseif(MSVC)
  setup_mlas_source_for_windows()
else()
    if(APPLE)
        get_target_property(ONNXRUNTIME_MLAS_OSX_ARCH onnxruntime_mlas OSX_ARCHITECTURES)

        if(NOT ONNXRUNTIME_MLAS_OSX_ARCH)
         set(ONNXRUNTIME_MLAS_OSX_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
        endif()
        foreach(OSX_ARCH ${ONNXRUNTIME_MLAS_OSX_ARCH})
        if (OSX_ARCH STREQUAL "arm64")
            set(ARM64 TRUE)
        elseif (OSX_ARCH STREQUAL "arm64e")
            set(ARM64 TRUE)
        elseif (OSX_ARCH STREQUAL "arm")
            set(ARM TRUE)
        elseif (OSX_ARCH STREQUAL "x86_64")
            set(X86_64 TRUE)
        elseif (OSX_ARCH STREQUAL "i386")
            set(X86 TRUE)
        endif()
        endforeach()
    elseif(ANDROID)
        if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
          set(ARM TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
          set(ARM64 TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
          set(X86_64 TRUE)
        elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
          set(X86 TRUE)
        endif()
    else()
        #Linux/FreeBSD/PowerPC/...
        #The value of CMAKE_SYSTEM_PROCESSOR should be from `uname -m`
        #Example values:
        #arm64v8/ubuntu -> aarch64
        #arm32v6/alpine -> armv7l
        #arm32v7/centos -> armv7l
        #ppc64le/debian -> ppc64le
        #s390x/ubuntu -> s390x
        #ppc64le/busybox -> ppc64le
        #arm64v8/ubuntu -> aarch64
        #Android: armv7-a aarch64 i686 x86_64
        #chasun: I don't think anyone uses 'arm64'
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm64.*")
          set(ARM64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm.*")
          set(ARM TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
          set(ARM64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(powerpc.*|ppc.*)")
          set(POWER TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
          set(X86 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
          set(X86_64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^riscv64.*")
          set(RISCV64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^loongarch64.*")
          set(LOONGARCH64 TRUE)
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^s390x$")
          set(S390X TRUE)
        endif()
    endif()

    if(APPLE)
      get_target_property(ONNXRUNTIME_MLAS_MACOSX_ARCH onnxruntime_mlas OSX_ARCHITECTURES)
    endif()
    list(LENGTH ONNXRUNTIME_MLAS_MACOSX_ARCH ONNXRUNTIME_MLAS_MACOSX_ARCH_LENGTH)
    if(ONNXRUNTIME_MLAS_MACOSX_ARCH_LENGTH GREATER 1)
        set(ONNXRUNTIME_MLAS_MULTI_ARCH TRUE)
    endif()
    #If ONNXRUNTIME_MLAS_MULTI_ARCH is true, we need to go through every if branch below
    #and split MLAS to multiple static libraries.
    #Otherwise, it works like if(...) elseif(...) elseif(...) endif()
    set(MLAS_SOURCE_IS_NOT_SET 1)
    if(ARM)
        enable_language(ASM)

        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mfpu=neon")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/aarch32/QgemmU8X8KernelNeon.S
          ${MLAS_SRC_DIR}/arm/sgemmc.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
        )
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(ARM64 AND MLAS_SOURCE_IS_NOT_SET )
        enable_language(ASM)
        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/aarch64/ConvSymS8KernelDot.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymS8KernelDotLd64.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymU8KernelDot.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymS8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/ConvSymU8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/DepthwiseQConvSymS8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/DepthwiseQConvSymU8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/DepthwiseQConvKernelSize9Neon.S
          ${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelUdot.S
          ${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelSdot.S
          ${MLAS_SRC_DIR}/aarch64/SgemmKernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/SgemvKernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/SymQgemmS8KernelNeon.S
          ${MLAS_SRC_DIR}/aarch64/SymQgemmS8KernelSdot.S
          ${MLAS_SRC_DIR}/aarch64/SymQgemmS8KernelSdotLd64.S
          ${MLAS_SRC_DIR}/qgemm_kernel_neon.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_udot.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_sdot.cpp
          ${MLAS_SRC_DIR}/qnbitgemm_kernel_neon.h
          ${MLAS_SRC_DIR}/qnbitgemm_kernel_neon.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_fp32.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8.cpp
          # Portable W2 scalar pack / reference kernel. See ARM64 (Windows) branch
          # above for the rationale; this is the matching entry for non-Windows
          # ARM64 builds (Linux, macOS).
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.h
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.cpp
          # W2 CompInt8 DotProd kernel (NEON FEAT_DotProd backend). Compiled
          # with -march=armv8.2-a+dotprod on this branch -- see flag override
          # further below alongside the W4/W8 dotprod TU.
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8_2bit.cpp
          ${MLAS_SRC_DIR}/rotary_embedding_kernel_neon.h
          ${MLAS_SRC_DIR}/rotary_embedding_kernel_neon.cpp
          ${MLAS_SRC_DIR}/qkv_quant_kernel.h
          ${MLAS_SRC_DIR}/qkv_quant_common.h
          ${MLAS_SRC_DIR}/qkv_quant_kernel_neon.cpp
          ${MLAS_SRC_DIR}/hgemm_kernel_neon.cpp
          ${MLAS_SRC_DIR}/softmax_kernel_neon.h
          ${MLAS_SRC_DIR}/softmax_kernel_neon.cpp
          ${MLAS_SRC_DIR}/eltwise_kernel_neon.h
          ${MLAS_SRC_DIR}/eltwise_kernel_neon.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8_i8mm.cpp
          ${MLAS_SRC_DIR}/sconv_nchw_depthwise_multiplier_1.cpp
          ${MLAS_SRC_DIR}/linear_attention_kernel_neon.cpp
        )

        # Conditionally add the SVE implementation if compiler supports it
        if (onnxruntime_USE_SVE)
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/mlasi_sve.h)
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/elementwise_sve_dispatch.cpp)
          # The portable machine-code variant is the production default;
          # configure with -Donnxruntime_SVE_ELEMENTWISE_ASM=OFF to build the
          # SVE intrinsics reference implementation instead (the regeneration
          # source for elementwise_sve_asm.S).
          option(onnxruntime_SVE_ELEMENTWISE_ASM
                 "Build the portable machine-code SVE elementwise kernels instead of the intrinsics reference" ON)
          if (onnxruntime_SVE_ELEMENTWISE_ASM)
            # Portable machine-code variant (same symbols as the intrinsics TUs).
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/aarch64/elementwise_sve_asm.S)
          else()
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/elementwise_sve.cpp)
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/elementwise_sve_fp16.cpp)
            set_source_files_properties(${MLAS_SRC_DIR}/sve/elementwise_sve.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+sve+fp16 -DMLAS_SVE_SUMEXP_FEXPA=1 ${ORT_SVE_ABI_FLAGS} ")
            set_source_files_properties(${MLAS_SRC_DIR}/sve/elementwise_sve_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+sve+fp16 ${ORT_SVE_ABI_FLAGS} ")
          endif()
          # SVE i8mm QGEMM: the driver/pack/dispatch TUs are plain C++ (no
          # SVE compiler support required); the svmmla compute kernels come
          # from either the generated KleidiAI-style machine code (portable,
          # production default) or the SVE intrinsics reference TU (the
          # regeneration source for aarch64/qgemm_mmla_sve_asm.S, script
          # sve/gen_sve_asm.py).
          option(onnxruntime_SVE_QGEMM_ASM
                 "Build the portable machine-code SVE QGEMM kernels instead of the intrinsics reference" ON)
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/qgemm_kernel_smmla_sve.cpp)
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/qgemm_kernel_ummla_sve.cpp)
          if (onnxruntime_SVE_QGEMM_ASM)
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/aarch64/qgemm_mmla_sve_asm.S)
          else()
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/qgemm_mmla_sve_impl.cpp)
            set_source_files_properties(${MLAS_SRC_DIR}/sve/qgemm_mmla_sve_impl.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+sve+i8mm -fno-stack-protector ${ORT_SVE_ABI_FLAGS} ")
          endif()
          # SVE LinearAttention: the driver is plain C++ (no SVE compiler
          # support required); the compute kernel comes from either the
          # generated KleidiAI-style machine code (portable, production
          # default) or the SVE intrinsics reference TU (the regeneration
          # source for aarch64/linear_attention_sve_asm.S).
          option(onnxruntime_SVE_LINEAR_ATTENTION_ASM
                 "Build the portable machine-code SVE LinearAttention kernel instead of the intrinsics reference" ON)
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/linear_attention_sve.h)
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/linear_attention_kernel_sve.cpp)
          if (onnxruntime_SVE_LINEAR_ATTENTION_ASM)
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/aarch64/linear_attention_sve_asm.S)
          else()
            list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/sve/linear_attention_sve_impl.cpp)
            set_source_files_properties(${MLAS_SRC_DIR}/sve/linear_attention_sve_impl.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+sve -fno-stack-protector -fno-jump-tables -fstack-clash-protection --param=stack-clash-protection-guard-size=12 ${ORT_SVE_ABI_FLAGS} ")
          endif()
          list(APPEND mlas_private_compile_definitions MLAS_USE_SVE)
        endif()

        if (onnxruntime_USE_ARM_NEON_NCHWC)
		  setup_arm_neon_nchwc()
		endif()

		if (onnxruntime_USE_KLEIDIAI)
          setup_kleidiai()
        endif()
        set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8.cpp
                                    PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+dotprod")
        set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8_2bit.cpp
                                    PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+dotprod")
        set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_neon_int8_i8mm.cpp
				    PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+i8mm ")

        if ((NOT APPLE) OR (CMAKE_SYSTEM_NAME STREQUAL "Darwin"))
          list(APPEND mlas_platform_srcs
            ${MLAS_SRC_DIR}/aarch64/SbgemmKernelNeon.S
            ${MLAS_SRC_DIR}/sbgemm_kernel_neon.cpp
          )
          set_source_files_properties(
            ${MLAS_SRC_DIR}/aarch64/SbgemmKernelNeon.S
            ${MLAS_SRC_DIR}/sbgemm_kernel_neon.cpp
            PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+bf16 "
          )
        endif()

        if (NOT APPLE)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/aarch64/HalfGemmKernelNeon.S
            ${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelSmmla.S
            ${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelUmmla.S
            ${MLAS_SRC_DIR}/activate_fp16.cpp
            ${MLAS_SRC_DIR}/dwconv.cpp
            ${MLAS_SRC_DIR}/halfgemm_kernel_neon.cpp
            ${MLAS_SRC_DIR}/pooling_fp16.cpp
            ${MLAS_SRC_DIR}/qgemm_kernel_smmla.cpp
            ${MLAS_SRC_DIR}/qgemm_kernel_ummla.cpp
            ${MLAS_SRC_DIR}/sbconv_kernel_neon.cpp
            ${MLAS_SRC_DIR}/cast_kernel_neon.cpp
            ${MLAS_SRC_DIR}/hqnbitgemm_kernel_neon_fp16.cpp
            ${MLAS_SRC_DIR}/hqnbitgemm_kernel_neon_fp16_8bit.cpp
            ${MLAS_SRC_DIR}/rotary_embedding_kernel_neon_fp16.cpp
            ${MLAS_SRC_DIR}/halfgemm_kernel_neon_fp16.cpp
            ${MLAS_SRC_DIR}/softmax_kernel_neon_fp16.cpp
            ${MLAS_SRC_DIR}/eltwise_kernel_neon_fp16.cpp
            ${MLAS_SRC_DIR}/erf_neon_fp16.h
            ${MLAS_SRC_DIR}/erf_neon_fp16.cpp
            ${MLAS_SRC_DIR}/gelu_neon_fp16.h
            ${MLAS_SRC_DIR}/gelu_neon_fp16.cpp
          )
          if (onnxruntime_USE_ARM_NEON_NCHWC)
            list(APPEND mlas_platform_srcs
              ${MLAS_SRC_DIR}/aarch64/SconvDepthwiseKernelNeonBf16.S
              ${MLAS_SRC_DIR}/aarch64/SconvKernelNeonBf16.S
              ${MLAS_SRC_DIR}/aarch64/SconvPointwiseKernelNeonBf16.S
            )
          endif()
          set_source_files_properties(${MLAS_SRC_DIR}/aarch64/HalfGemmKernelNeon.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/aarch64/QgemmS8S8KernelSmmla.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+i8mm ")
          set_source_files_properties(${MLAS_SRC_DIR}/aarch64/QgemmU8X8KernelUmmla.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+i8mm ")
          set_source_files_properties(${MLAS_SRC_DIR}/activate_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/dwconv.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/pooling_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/sbconv_kernel_neon.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+bf16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/cast_kernel_neon.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/hqnbitgemm_kernel_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/hqnbitgemm_kernel_neon_fp16_8bit.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/rotary_embedding_kernel_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/halfgemm_kernel_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/softmax_kernel_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/eltwise_kernel_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          if (onnxruntime_USE_ARM_NEON_NCHWC)
            set_source_files_properties(${MLAS_SRC_DIR}/aarch64/SconvDepthwiseKernelNeonBf16.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+bf16 ")
            set_source_files_properties(${MLAS_SRC_DIR}/aarch64/SconvKernelNeonBf16.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+bf16 ")
            set_source_files_properties(${MLAS_SRC_DIR}/aarch64/SconvPointwiseKernelNeonBf16.S PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+bf16 ")
          endif()
          set_source_files_properties(${MLAS_SRC_DIR}/erf_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
          set_source_files_properties(${MLAS_SRC_DIR}/gelu_neon_fp16.cpp PROPERTIES COMPILE_FLAGS " -march=armv8.2-a+fp16 ")
        endif()

        # Enable the f16<->f32 cast kernel on macOS ARM64.  The conversion
        # intrinsics (vcvt_f32_f16 / vcvt_f16_f32) are baseline ARMv8-A and
        # do not require -march=armv8.2-a+fp16 (that flag is only needed for
        # fp16 *arithmetic* on float16x8_t).  Guard on Darwin (not APPLE) so
        # iOS/tvOS/visionOS are excluded.
        if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
          list(APPEND mlas_platform_srcs ${MLAS_SRC_DIR}/cast_kernel_neon.cpp)
        endif()

        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
            onnxruntime_add_static_library(onnxruntime_mlas_arm64 ${mlas_platform_srcs})
            set_target_properties(onnxruntime_mlas_arm64 PROPERTIES OSX_ARCHITECTURES "arm64")
            list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_arm64)
            set(mlas_platform_srcs )
        else()
            set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(POWER AND MLAS_SOURCE_IS_NOT_SET)
        check_cxx_source_compiles("
          #ifndef __VSX__
          #error VSX is not supported
          #endif
          int main() { return 0; }"
          MLAS_HAS_VSX
        )
    endif()
    if(POWER AND MLAS_HAS_VSX AND MLAS_SOURCE_IS_NOT_SET)
        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/power/SgemmKernelPower.cpp
          ${MLAS_SRC_DIR}/dgemm.cpp
          ${MLAS_SRC_DIR}/power/DgemmKernelPower.cpp
          ${MLAS_SRC_DIR}/power/QuantizePower.cpp
        )
        set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPower.cpp PROPERTIES COMPILE_FLAGS "-DSINGLE")

        check_cxx_compiler_flag("-mcpu=power9" HAS_POWER9)
        if (HAS_POWER9)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/power/QuantizePowerVSX.cpp
          )
          set_source_files_properties(${MLAS_SRC_DIR}/power/QuantizePowerVSX.cpp PROPERTIES COMPILE_FLAGS "-mcpu=power9")
        endif()

        check_cxx_compiler_flag("-mcpu=power10" HAS_POWER10)
        if(HAS_POWER10)
          set(CMAKE_REQUIRED_FLAGS "-mcpu=power10")
          check_cxx_source_compiles("
            #include <altivec.h>
            int main() {
              __vector_quad acc0;
              __builtin_mma_xxsetaccz (&acc0);
              return 0;
            }"
            COMPILES_P10
          )
          if(COMPILES_P10)
            enable_language(ASM)
            check_cxx_source_compiles("
              #ifdef _AIX
              #include <sys/systemcfg.h>
              #if !defined(POWER_10)
              #define POWER_10       0x40000
              #endif
              #if !defined(POWER_10_ANDUP)
              #define POWER_10_ANDUP (POWER_10)
              #endif
              #define __power_10_andup() (_system_configuration.implementation & POWER_10_ANDUP)
              int main() {
                bool HasP10 = (__power_10_andup() && __power_mma_version() == MMA_V31);
                return 0;
              }
              #else
              #include <sys/auxv.h>
              int main() {
                unsigned long hwcap2 = getauxval(AT_HWCAP2);
                bool HasP10 = ((hwcap2 & PPC_FEATURE2_MMA) && (hwcap2 & PPC_FEATURE2_ARCH_3_1));
                return 0;
              }
              #endif"
              HAS_P10_RUNTIME
            )
            if (HAS_P10_RUNTIME)
              set_source_files_properties(${MLAS_SRC_DIR}/platform.cpp PROPERTIES COMPILE_FLAGS "-DPOWER10")
              set_source_files_properties(${MLAS_SRC_DIR}/qgemm.cpp PROPERTIES COMPILE_FLAGS "-DPOWER10")
            endif()
            set(mlas_platform_srcs_power10
              ${MLAS_SRC_DIR}/power/SgemmKernelPOWER10.cpp
              ${MLAS_SRC_DIR}/power/DgemmKernelPOWER10.cpp
              ${MLAS_SRC_DIR}/power/qgemm_kernel_power10.cpp
            )
	    # Only compile assembly on non-AIX systems
            if (NOT AIX)
              list(APPEND mlas_platform_srcs_power10 ${MLAS_SRC_DIR}/power/SgemmKernelPackA.S)
              set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPackA.S PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10")
            endif()
            set_source_files_properties(${MLAS_SRC_DIR}/power/SgemmKernelPOWER10.cpp PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10 -DSINGLE")
            set_source_files_properties(${MLAS_SRC_DIR}/power/DgemmKernelPOWER10.cpp PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10")
            set_source_files_properties(${MLAS_SRC_DIR}/power/qgemm_kernel_power10.cpp PROPERTIES COMPILE_FLAGS "-O3 -mcpu=power10")
            set(mlas_platform_srcs
              ${mlas_platform_srcs}
              ${mlas_platform_srcs_power10}
            )
          endif()
        endif()
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        set(mlas_platform_srcs_sse2
          ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
          ${MLAS_SRC_DIR}/x86/SgemmKernelSse2.S
        )
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
          ${MLAS_SRC_DIR}/x86/SgemmKernelAvx.S
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs_avx2
          ${MLAS_SRC_DIR}/layernorm_kernel_avx2.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")

        set(mlas_platform_srcs
          ${mlas_platform_srcs_sse2}
          ${mlas_platform_srcs_avx}
          ${mlas_platform_srcs_avx2}
        )

        # In r23, NDK remove __x86.get_pc_thunk.* from libatomic. Add our own
        # implementation to avoid external dependency.
        if(ANDROID)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/x86/x86.get_pc_thunk.S
          )
        endif()

        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(X86_64 AND MLAS_SOURCE_IS_NOT_SET)
        enable_language(ASM)

        # Forward the flags for the minimum target platform version from the C
        # compiler to the assembler. This works around CMakeASMCompiler.cmake.in
        # not including the logic to set this flag for the assembler.
        set(CMAKE_ASM${ASM_DIALECT}_OSX_DEPLOYMENT_TARGET_FLAG "${CMAKE_C_OSX_DEPLOYMENT_TARGET_FLAG}")

        # The LLVM assembler does not support the .arch directive to enable instruction
        # set extensions and also doesn't support AVX-512F instructions without
        # turning on support via command-line option. Group the sources by the
        # instruction set extension and explicitly set the compiler flag as appropriate.

        set(mlas_platform_srcs_sse2
          ${MLAS_SRC_DIR}/qgemm_kernel_sse.cpp
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelSse2.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelSse2.S
          ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Sse2.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelSse2.S
          ${MLAS_SRC_DIR}/x86_64/SpoolKernelSse2.S
        )
        if(NOT APPLE)
          set(mlas_platform_srcs_sse2
            ${mlas_platform_srcs_sse2}
            ${MLAS_SRC_DIR}/x86_64/cvtfp16a.S
          )
        endif()
        set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

        set(mlas_platform_srcs_avx
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1Avx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelM1TransposeBAvx.S
          ${MLAS_SRC_DIR}/x86_64/SgemmTransposePackB16x4Avx.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx.S
          ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx.S
          ${MLAS_SRC_DIR}/intrinsics/avx/min_max_elements.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

        set(mlas_platform_srcs_avx2
          ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/QgemmU8U8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvxVnni.S
          ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx2.S
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/TransKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/LogisticKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/TanhKernelFma3.S
          ${MLAS_SRC_DIR}/x86_64/ErfKernelFma3.S
          ${MLAS_SRC_DIR}/intrinsics/avx2/qladd_avx2.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx2/qdwconv_avx2.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx2/saturation_check_avx2.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx2/q2_dq_avx2.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.h
          ${MLAS_SRC_DIR}/sqnbitgemm_lut_kernel_avx2.cpp
          ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.h
          ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
          ${MLAS_SRC_DIR}/rotary_embedding_kernel_avx2.cpp
          ${MLAS_SRC_DIR}/qkv_quant_kernel.h
          ${MLAS_SRC_DIR}/qkv_quant_common.h
          ${MLAS_SRC_DIR}/qkv_quant_kernel_avx2.cpp
          ${MLAS_SRC_DIR}/layernorm_kernel_avx2.cpp
        )
        if(CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 13.1 AND NOT(APPLE))
          set(mlas_platform_srcs_avx2
            ${mlas_platform_srcs_avx2}
            ${MLAS_SRC_DIR}/x86_64/cvtfp16Avx.S
          )
        endif()

        include(CheckCXXSourceCompiles)
        set(OLD_CMAKE_REQUIRED_FLAGS_AVXVNNI "${CMAKE_REQUIRED_FLAGS}")
        set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS_AVXVNNI} -mavx2 -mfma -mf16c -mavxvnni")
        check_cxx_source_compiles("
          #include <immintrin.h>
          int main() {
            __m256i a = _mm256_setzero_si256();
            __m256i b = _mm256_setzero_si256();
            (void)_mm256_dpbusds_avx_epi32(a, b, b);
            return 0;
          }"
          MLAS_COMPILER_SUPPORTS_AVXVNNI
        )
        set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS_AVXVNNI}")
        unset(OLD_CMAKE_REQUIRED_FLAGS_AVXVNNI)

        # Apply the base AVX2 flags to all AVX2 sources.
        # sqnbitgemm_kernel_avx2.cpp is now in this set and compiled without
        # -mavxvnni, preventing the auto-vectorizer from emitting AVX-VNNI
        # instructions in the pure-AVX2 kernels.
        set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")

        # sqnbitgemm_kernel_avx2vnni.cpp is always compiled so that
        # MlasSQNBitGemmDispatchAvx2vnni (referenced unconditionally in
        # platform.cpp, mlasi.h, and tests) is always defined.  Only the
        # -mavxvnni flag is conditional: without it __AVXVNNI__ is not defined
        # and the VNNI intrinsic paths compile out via the #if guards, leaving
        # only the AVX2 fallback kernels in the dispatch table.
        set(mlas_platform_srcs_avx2vnni
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2vnni.cpp
        )
        if(MLAS_COMPILER_SUPPORTS_AVXVNNI)
          message(STATUS "AVX-VNNI supported: compiling sqnbitgemm_kernel_avx2vnni.cpp with -mavx2 -mfma -mf16c -mavxvnni")
          # AVX-VNNI kernels live in a dedicated TU compiled with -mavxvnni.
          # Keeping them separate prevents the auto-vectorizer from emitting
          # VNNI instructions in the pure-AVX2 kernels in sqnbitgemm_kernel_avx2.cpp.
          set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2vnni.cpp
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c -mavxvnni")
        else()
          message(STATUS "AVX-VNNI not supported by compiler/assembler: compiling sqnbitgemm_kernel_avx2vnni.cpp with -mavx2 -mfma -mf16c only (VNNI paths compiled out)")
          # Still compile the TU so MlasSQNBitGemmDispatchAvx2vnni is defined.
          # __AVXVNNI__ is absent, so all VNNI intrinsic blocks are #if-excluded
          # and the dispatch table falls back to pure-AVX2 kernels.
          set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx2vnni.cpp
            PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c")
        endif()

        # The 2-bit fp-zero-point dequant kernel relies on separate multiply
        # and add rounding to stay bit-identical to the scalar kernel, so keep
        # the compiler from contracting them into an FMA.
        set_property(SOURCE ${MLAS_SRC_DIR}/intrinsics/avx2/q2_dq_avx2.cpp
          APPEND PROPERTY COMPILE_OPTIONS "-ffp-contract=off")
        set(mlas_platform_srcs_avx512f
          ${MLAS_SRC_DIR}/x86_64/DgemmKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SgemmKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SconvKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SoftmaxKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/SpoolKernelAvx512F.S
          ${MLAS_SRC_DIR}/x86_64/TransKernelAvx512F.S
          ${MLAS_SRC_DIR}/intrinsics/avx512/gelu_avx512f.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx512/silu_avx512f.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx512/quantize_avx512f.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx512/sconv_nchw_depthwise_multiplier_greater_than_1_avx512f.cpp
          ${MLAS_SRC_DIR}/linear_attention_kernel_avx512f.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx512/reorder_avx512f.cpp
          ${MLAS_SRC_DIR}/intrinsics/avx512/sconv_depthwise_avx512f.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")

        set(mlas_platform_srcs_avx512core
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Core.S
          ${MLAS_SRC_DIR}/x86_64/QgemvU8S8KernelAvx512Vnni.S
          ${MLAS_SRC_DIR}/x86_64/QgemmU8X8KernelAvx512Core.S
          ${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx512Core.S
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.h
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit_blklen64.h
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit_blklen128.h
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit_blklen32.h
        )
        set_source_files_properties(${mlas_platform_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl")

        # NOTE: this TU is intrinsic-free scalar helpers reached from both the
        # AVX2 and AVX-512 W2 dispatch tables at model load. Flags must not
        # enable any ISA the AVX2-only host lacks, or the compiler may
        # autovectorize the scalar loops into EVEX instructions and SIGILL.
        # If the AVX-512 W2 pack ever becomes a perf hot spot, split this
        # file: keep the scalar pack in a flag-less TU and put an optimized
        # variant in a separate one only used by the AVX-512 tables.
        set_source_files_properties(${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512_2bit.cpp PROPERTIES
          COMPILE_FLAGS "")

        set(mlas_platform_srcs_avx512vnni
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_avx512vnni_2bit.cpp
          ${MLAS_SRC_DIR}/qkv_quant_kernel_avx512vnni.cpp
        )
        set_source_files_properties(
          ${mlas_platform_srcs_avx512vnni} PROPERTIES
          COMPILE_FLAGS "-mfma -mf16c -mavx512vnni -mavx512bw -mavx512dq -mavx512vl -mavx512f")

        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/activate_fp16.cpp
          ${MLAS_SRC_DIR}/dwconv.cpp
          ${MLAS_SRC_DIR}/dgemm.cpp
          ${MLAS_SRC_DIR}/pooling_fp16.cpp
          ${MLAS_SRC_DIR}/qgemm_kernel_avx2.cpp
          ${mlas_platform_srcs_sse2}
          ${mlas_platform_srcs_avx}
          ${mlas_platform_srcs_avx2}
          ${mlas_platform_srcs_avx2vnni}
          ${mlas_platform_srcs_avx512f}
          ${mlas_platform_srcs_avx512core}
          ${mlas_platform_srcs_avx512vnni}
        )

        if (NOT onnxruntime_ORT_MINIMAL_BUILD)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${MLAS_SRC_DIR}/q4gemm_avx512.cpp
          )
          set_source_files_properties(${MLAS_SRC_DIR}/q4gemm_avx512.cpp PROPERTIES COMPILE_FLAGS "-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl -mavx512f")
        endif()
        if(NOT APPLE)
          set(mlas_platform_srcs
            ${mlas_platform_srcs}
	        ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmxCommon.S
            ${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp
            ${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmx.S
            )
          set_source_files_properties(${MLAS_SRC_DIR}/qgemm_kernel_amx.cpp PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f")
          set_source_files_properties(${MLAS_SRC_DIR}/x86_64/QgemmU8S8KernelAmx.S PROPERTIES COMPILE_FLAGS "-mavx2 -mavx512bw -mavx512dq -mavx512vl -mavx512f")
        endif()

        if(onnxruntime_ENABLE_CONVSYMKERNELAVX2_SAT_CHECKER)
          set_source_files_properties(${MLAS_SRC_DIR}/x86_64/ConvSymKernelAvx2.S PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -mf16c -DENABLE_CONVSYMKERNELAVX2_SAT_CHECKER")
        endif()

        if(ONNXRUNTIME_MLAS_MULTI_ARCH)
          onnxruntime_add_static_library(onnxruntime_mlas_x86_64 ${mlas_platform_srcs})
          set_target_properties(onnxruntime_mlas_x86_64 PROPERTIES OSX_ARCHITECTURES "x86_64")
          list(APPEND ONNXRUNTIME_MLAS_LIBS onnxruntime_mlas_x86_64)
          set(mlas_platform_srcs )
        else()
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(LOONGARCH64 AND MLAS_SOURCE_IS_NOT_SET)
        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/qgemm_kernel_lsx.cpp
          ${MLAS_SRC_DIR}/sqnbitgemm_kernel_lasx.cpp
          ${MLAS_SRC_DIR}/loongarch64/SgemmKernelLasx.S
          ${MLAS_SRC_DIR}/loongarch64/DgemmKernelLsx.S
          ${MLAS_SRC_DIR}/loongarch64/DgemmKernelLasx.S
          ${MLAS_SRC_DIR}/loongarch64/SgemmKernelLsx.S
          ${MLAS_SRC_DIR}/loongarch64/SconvKernelLsx.S
          ${MLAS_SRC_DIR}/loongarch64/SconvKernelLasx.S
          ${MLAS_SRC_DIR}/loongarch64/SpoolKernelLSX.S
          ${MLAS_SRC_DIR}/loongarch64/SpoolKernelLasx.S
          ${MLAS_SRC_DIR}/loongarch64/SgemmTransposePackB16x4LSX.S
          ${MLAS_SRC_DIR}/loongarch64/SgemmTransposePackB16x4Lasx.S
          ${MLAS_SRC_DIR}/loongarch64/SoftmaxKernelLasx.S
            )
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mlsx -mlasx")
        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(S390X AND MLAS_SOURCE_IS_NOT_SET)
        set(mlas_platform_srcs
          ${MLAS_SRC_DIR}/s390x/SgemmKernel.cpp
          ${MLAS_SRC_DIR}/s390x/SgemmKernelZVECTOR.cpp
          ${MLAS_SRC_DIR}/dgemm.cpp
          ${MLAS_SRC_DIR}/s390x/DgemmKernel.cpp
          ${MLAS_SRC_DIR}/s390x/Quantize.cpp
          ${MLAS_SRC_DIR}/s390x/QuantizeZVECTOR.cpp
          ${MLAS_SRC_DIR}/s390x/qgemm_kernel_zvector.cpp
        )
        set_source_files_properties(${MLAS_SRC_DIR}/s390x/SgemmKernel.cpp PROPERTIES COMPILE_FLAGS "-DSINGLE")
        set_source_files_properties(${MLAS_SRC_DIR}/s390x/SgemmKernelZVECTOR.cpp PROPERTIES COMPILE_FLAGS "-DSINGLE")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mvx -mzvector -march=z15")

        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(RISCV64 AND MLAS_SOURCE_IS_NOT_SET)
        file(GLOB_RECURSE mlas_platform_srcs CONFIGURE_DEPENDS
          "${MLAS_SRC_DIR}/scalar/*.cpp")
        # Remove scalar depthwise kernel; replaced by the vectorized version
        list(REMOVE_ITEM mlas_platform_srcs
          "${MLAS_SRC_DIR}/scalar/SconvDepthwiseKernelScalar.cpp")
        list(APPEND mlas_platform_srcs
          ${MLAS_SRC_DIR}/sconv_nchw_depthwise_multiplier_1.cpp)

        if(onnxruntime_USE_RVV)
          set(OLD_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
          set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS} -march=rv64gcv -mabi=lp64d")
          check_cxx_source_compiles("
            #include <stddef.h>
            #include <riscv_vector.h>
            int main() {
              size_t vl = __riscv_vsetvl_e32m1(4);
              return static_cast<int>(vl == 0);
            }"
            HAS_RISCV64_RVV
          )
          set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS}")
          unset(OLD_CMAKE_REQUIRED_FLAGS)

          if(HAS_RISCV64_RVV)
            list(APPEND mlas_platform_srcs
              ${MLAS_SRC_DIR}/riscv64/sgemm_pack_b_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/sgemm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/softmax_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/sconv_depthwise_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/sconv_nchwc_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/rotary_embedding_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/layernorm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/qgemm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/activation_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/conv_activation_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/qnbitgemm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/linear_attention_kernel_rvv.cpp
            )
            list(REMOVE_ITEM mlas_platform_srcs
              "${MLAS_SRC_DIR}/sconv_nchw_depthwise_multiplier_1.cpp")
            set_source_files_properties(
              ${MLAS_SRC_DIR}/riscv64/sgemm_pack_b_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/sgemm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/softmax_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/sconv_depthwise_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/sconv_nchwc_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/rotary_embedding_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/layernorm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/qgemm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/activation_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/conv_activation_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/qnbitgemm_kernel_rvv.cpp
              ${MLAS_SRC_DIR}/riscv64/linear_attention_kernel_rvv.cpp
              PROPERTIES COMPILE_FLAGS "-march=rv64gcv -mabi=lp64d")
            list(APPEND mlas_private_compile_definitions MLAS_USE_RVV=1)

            if(onnxruntime_USE_RVV_ZVFH)
              list(APPEND mlas_platform_srcs
                ${MLAS_SRC_DIR}/riscv64/halfgemm_kernel_rvv.cpp
                ${MLAS_SRC_DIR}/riscv64/cast_kernel_rvv.cpp
                ${MLAS_SRC_DIR}/riscv64/hqnbitgemm_kernel_rvv.cpp
              )
              set_source_files_properties(
                ${MLAS_SRC_DIR}/riscv64/halfgemm_kernel_rvv.cpp
                ${MLAS_SRC_DIR}/riscv64/cast_kernel_rvv.cpp
                ${MLAS_SRC_DIR}/riscv64/hqnbitgemm_kernel_rvv.cpp
                PROPERTIES COMPILE_FLAGS "-march=rv64gcv_zvfh -mabi=lp64d")
              list(APPEND mlas_private_compile_definitions MLAS_USE_RVV_ZVFH=1)
            endif()
          else()
            message(
              WARNING
              "onnxruntime_USE_RVV was requested, but the compiler does not support rv64gcv RVV intrinsics. Falling back to scalar MLAS kernels.")
          endif()
        endif()

        if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH)
          set(MLAS_SOURCE_IS_NOT_SET 0)
        endif()
    endif()
    if(NOT ONNXRUNTIME_MLAS_MULTI_ARCH AND MLAS_SOURCE_IS_NOT_SET)
        file(GLOB_RECURSE mlas_platform_srcs
          "${MLAS_SRC_DIR}/scalar/*.cpp")
    elseif (onnxruntime_FORCE_GENERIC_ALGORITHMS)
        file(GLOB_RECURSE mlas_platform_srcs_generic
          "${MLAS_SRC_DIR}/scalar/*.cpp")
        set(mlas_platform_srcs
            ${mlas_platform_srcs}
            ${mlas_platform_srcs_generic}
            )
    endif()
    target_sources(onnxruntime_mlas PRIVATE ${mlas_platform_srcs})
endif()

# Merge in the definitions that were added from within functions.
get_property(mlas_private_compile_definitions_from_functions
             DIRECTORY PROPERTY mlas_private_compile_definitions_from_functions)
list(APPEND mlas_private_compile_definitions ${mlas_private_compile_definitions_from_functions})
list(REMOVE_DUPLICATES mlas_private_compile_definitions)

foreach(mlas_target ${ONNXRUNTIME_MLAS_LIBS})
    target_include_directories(${mlas_target} PRIVATE ${MLAS_INC_DIR} ${MLAS_SRC_DIR})
    onnxruntime_add_include_to_target(${mlas_target} ${GSL_TARGET} safeint_interface)

    target_compile_definitions(${mlas_target} PRIVATE ${mlas_private_compile_definitions})

    set_target_properties(${mlas_target} PROPERTIES FOLDER "ONNXRuntime")
endforeach()

if (WIN32)
  target_compile_options(onnxruntime_mlas PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:/wd6385>" "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>")
  if (onnxruntime_ENABLE_STATIC_ANALYSIS)
    target_compile_options(onnxruntime_mlas PRIVATE  "$<$<COMPILE_LANGUAGE:CXX>:/analyze:stacksize 131072>")
  endif()
endif()

if (NOT onnxruntime_BUILD_SHARED_LIB)
    install(TARGETS onnxruntime_mlas EXPORT ${PROJECT_NAME}Targets
            ARCHIVE   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY   DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME   DESTINATION ${CMAKE_INSTALL_BINDIR}
            FRAMEWORK DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# set up source group for MLAS source files
block()
  set(source_group_srcs)
  foreach(mlas_target ${ONNXRUNTIME_MLAS_LIBS})
    get_target_property(mlas_target_srcs ${mlas_target} SOURCES)
    foreach(mlas_target_src ${mlas_target_srcs})
      cmake_path(IS_PREFIX MLAS_ROOT ${mlas_target_src} in_mlas_root)
      if(in_mlas_root)
        list(APPEND source_group_srcs ${mlas_target_src})
      endif()
    endforeach()
  endforeach()
  source_group(TREE ${MLAS_ROOT} FILES ${source_group_srcs})
endblock()


if (NOT onnxruntime_ORT_MINIMAL_BUILD)

  #
  # Command line tool for quantization and de-quantization of 2-D fp32 tensors
  # based on block-wise quantization of int4
  #

  onnxruntime_add_executable(onnxruntime_mlas_q4dq
    ${MLAS_SRC_DIR}/q4_dq_cli.cpp
  )
  target_include_directories(onnxruntime_mlas_q4dq PRIVATE ${MLAS_INC_DIR} ${MLAS_SRC_DIR})
  set_target_properties(onnxruntime_mlas_q4dq PROPERTIES FOLDER "ONNXRuntimeTest")

  target_link_libraries(onnxruntime_mlas_q4dq PRIVATE ${ONNXRUNTIME_MLAS_LIBS} onnxruntime_common)
  if (CPUINFO_SUPPORTED AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE cpuinfo)
  endif()
  if(NOT WIN32)
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE  ${CMAKE_DL_LIBS})
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE ${android_shared_libs})
  endif()

  if(WIN32)
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE debug Dbghelp Advapi32)
  endif()
  if (onnxruntime_LINK_LIBATOMIC)
    target_link_libraries(onnxruntime_mlas_q4dq PRIVATE atomic)
  endif()
  target_link_libraries(onnxruntime_mlas_q4dq PRIVATE Threads::Threads)

  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if (onnxruntime_ENABLE_WEBASSEMBLY_THREADS)
      set_target_properties(onnxruntime_mlas_q4dq PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1 -s PROXY_TO_PTHREAD=1 -s EXIT_RUNTIME=1")
    else()
      set_target_properties(onnxruntime_mlas_q4dq PROPERTIES LINK_FLAGS "-s ALLOW_MEMORY_GROWTH=1")
    endif()
  endif()

endif()

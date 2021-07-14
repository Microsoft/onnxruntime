# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set(mlas_common_srcs
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/platform.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/threading.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/sgemm.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qdwconv.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/convolve.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/pooling.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/transpose.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/reorder.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/snchwc.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/activate.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/logistic.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/tanh.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/erf.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/compute.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/quantize.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_default.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qladd.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qlmul.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qpostprocessor.cpp
  ${ONNXRUNTIME_ROOT}/core/mlas/lib/qlgavgpool.cpp
)

if (onnxruntime_BUILD_WEBASSEMBLY)
  if (onnxruntime_ENABLE_WEBASSEMBLY_SIMD)
    file(GLOB_RECURSE mlas_platform_srcs
      "${ONNXRUNTIME_ROOT}/core/mlas/lib/wasm_simd/*.cpp"
    )
  else()
    file(GLOB_RECURSE mlas_platform_srcs
      "${ONNXRUNTIME_ROOT}/core/mlas/lib/wasm/*.cpp"
    )
  endif()
elseif(MSVC)
  if(onnxruntime_target_platform STREQUAL "ARM64")
    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_neon.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_udot.cpp
    )

    set(mlas_platform_preprocess_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm64/QgemmU8X8KernelNeon.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm64/QgemmU8X8KernelUdot.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm64/SgemmKernelNeon.asm
    )

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(ARMASM_FLAGS "-g")
    else()
      set(ARMASM_FLAGS "")
    endif()

    # Run the C precompiler on each input before the assembler.
    foreach(asm_filename ${mlas_platform_preprocess_srcs})
      get_filename_component(asm_filename_base ${asm_filename} NAME_WLE)
      set(preprocess_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.i)
      set(obj_filename ${CMAKE_CURRENT_BINARY_DIR}/${asm_filename_base}.obj)
      add_custom_command(
        OUTPUT ${obj_filename}
          COMMAND
              cl.exe /P ${asm_filename} /Fi${preprocess_filename}
          COMMAND
              armasm64.exe ${ARMASM_FLAGS} ${preprocess_filename} ${obj_filename}
        DEPENDS ${asm_filename}
        BYPRODUCTS ${preprocess_filename}
      )
      list(APPEND mlas_platform_srcs ${obj_filename})
    endforeach()
  elseif((onnxruntime_target_platform STREQUAL "ARM") OR (onnxruntime_target_platform STREQUAL "ARM64EC"))
    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
    )
  elseif(onnxruntime_target_platform STREQUAL "x64")
    enable_language(ASM_MASM)

    file(GLOB_RECURSE mlas_platform_srcs_avx CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "/arch:AVX")

    file(GLOB_RECURSE mlas_platform_srcs_avx2 CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx2/*.cpp"
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "/arch:AVX2")

    if (onnxruntime_MINIMAL_BUILD)
      # exclude AVX512 in minimal build
      set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DMLAS_AVX512F_UNSUPPORTED")
    endif()

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/dgemm.cpp
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_avx2.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_sse.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_sse41.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx512/quantize_avx512f.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8S8KernelAvx2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8U8KernelAvx2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8X8KernelAvx2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemmU8X8KernelAvx512Core.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemvU8S8KernelAvx2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemvU8S8KernelAvx512Core.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemvU8S8KernelAvx512Vnni.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/QgemvU8S8KernelAvxVnni.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/DgemmKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/DgemmKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/DgemmKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/DgemmKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelM1Avx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SgemmKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SconvKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SpoolKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SpoolKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SpoolKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/sgemma.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/cvtfp16a.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/SoftmaxKernelAvx.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/TransKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/TransKernelAvx512F.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/LogisticKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/TanhKernelFma3.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/amd64/ErfKernelFma3.asm
    )
  else()
    enable_language(ASM_MASM)

    set(CMAKE_ASM_MASM_FLAGS "${CMAKE_ASM_MASM_FLAGS} /safeseh")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_sse.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_sse41.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/i386/SgemmKernelSse2.asm
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/i386/SgemmKernelAvx.asm
    )
  endif()
else()
  if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
    set(ARM64 TRUE)
  elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64e")
    set(ARM64 TRUE)
  elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "arm")
    set(ARM TRUE)
  elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(X86_64 TRUE)
  elseif (CMAKE_OSX_ARCHITECTURES STREQUAL "i386")
    set(X86 TRUE)
  endif()
  if (CMAKE_SYSTEM_NAME STREQUAL "Android")
    if (CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
      set(ARM TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
      set(ARM64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86_64")
      set(X86_64 TRUE)
    elseif (CMAKE_ANDROID_ARCH_ABI STREQUAL "x86")
      set(X86 TRUE)
    endif()
  elseif(CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "iOSCross")
    set(IOS TRUE)
  else()
    execute_process(
      COMMAND ${CMAKE_C_COMPILER} -dumpmachine
      OUTPUT_VARIABLE dumpmachine_output
      ERROR_QUIET
    )
    if(dumpmachine_output MATCHES "^arm64.*")
      set(ARM64 TRUE)
    elseif(dumpmachine_output MATCHES "^arm.*")
      set(ARM TRUE)
    elseif(dumpmachine_output MATCHES "^aarch64.*")
      set(ARM64 TRUE)
    elseif(dumpmachine_output MATCHES "^(powerpc.*|ppc.*)")
      set(POWER TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
      set(X86 TRUE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
      set(X86_64 TRUE)
    endif()
  endif()

  if(ARM)
    enable_language(ASM)

    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -mfpu=neon")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch32/QgemmU8X8KernelNeon.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/arm/sgemmc.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_neon.cpp
    )
  elseif(ARM64)
    enable_language(ASM)

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch64/QgemmU8X8KernelNeon.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch64/QgemmU8X8KernelUdot.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch64/SgemmKernelNeon.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/aarch64/SgemvKernelNeon.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_neon.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_udot.cpp
    )
  elseif(POWER)
    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/power/SgemmKernelPower.cpp
    )
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
        check_cxx_source_compiles("
          #include <sys/auxv.h>
          int main() {
            unsigned long hwcap2 = getauxval(AT_HWCAP2);
            bool HasP10 = ((hwcap2 & PPC_FEATURE2_MMA) && (hwcap2 & PPC_FEATURE2_ARCH_3_1));
            return 0;
          }"
          HAS_P10_RUNTIME
        )
        if (HAS_P10_RUNTIME)
          set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DPOWER10")
        endif()
        set(mlas_platform_srcs_power10
          ${ONNXRUNTIME_ROOT}/core/mlas/lib/power/SgemmKernelPOWER10.cpp
        )
        set_source_files_properties(${mlas_platform_srcs_power10} PROPERTIES COMPILE_FLAGS "-O2 -mcpu=power10")
        set(mlas_platform_srcs
          ${mlas_platform_srcs}
          ${mlas_platform_srcs_power10}
        )
      endif()
    endif()
  elseif(X86)
    enable_language(ASM)

    set(mlas_platform_srcs_sse2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_sse.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86/SgemmKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86/SgemmKernelAvx.S
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
    )
  elseif(X86_64)
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
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_sse.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/DgemmKernelSse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelSse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmTransposePackB16x4Sse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelSse2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SpoolKernelSse2.S
    )
    set_source_files_properties(${mlas_platform_srcs_sse2} PROPERTIES COMPILE_FLAGS "-msse2")

    set(mlas_platform_srcs_avx
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/DgemmKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelM1Avx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelM1TransposeBAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmTransposePackB16x4Avx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SpoolKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SoftmaxKernelAvx.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx/min_max_elements.cpp
    )
    set_source_files_properties(${mlas_platform_srcs_avx} PROPERTIES COMPILE_FLAGS "-mavx")

    set(mlas_platform_srcs_avx2
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8S8KernelAvx2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemvU8S8KernelAvx2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8U8KernelAvx2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemvU8S8KernelAvxVnni.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8X8KernelAvx2.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/DgemmKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/TransKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/LogisticKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/TanhKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/ErfKernelFma3.S
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx2/qladd_avx2.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx2/qdwconv_avx2.cpp
    )
    set_source_files_properties(${mlas_platform_srcs_avx2} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma")

    # Some toolchains do not support AVX512 compiler flags but are still able
    # to build the sources. Other toolchains require the AVX512 compiler flags
    # to be specified.
    check_cxx_compiler_flag("-mavx512f" HAS_AVX512F)
    if(HAS_AVX512F)
      set(CMAKE_REQUIRED_FLAGS "-mavx512f")
    else()
      set(CMAKE_REQUIRED_FLAGS "")
    endif()
    check_cxx_source_compiles("
      int main() {
        asm(\"vpxord %zmm0,%zmm0,%zmm0\");
        return 0;
      }"
      COMPILES_AVX512F
    )

    if(COMPILES_AVX512F AND NOT onnxruntime_MINIMAL_BUILD)
      set(mlas_platform_srcs_avx512f
        ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/DgemmKernelAvx512F.S
        ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SgemmKernelAvx512F.S
        ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SconvKernelAvx512F.S
        ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/SpoolKernelAvx512F.S
        ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/TransKernelAvx512F.S
      )

      check_cxx_source_compiles("
        #include <immintrin.h>
        int main() {
          __m512 zeros = _mm512_set1_ps(0.f);
          (void)zeros;
          return 0;
        }"
        COMPILES_AVX512F_INTRINSICS
      )
      if(COMPILES_AVX512F_INTRINSICS)
        set(mlas_platform_srcs_avx512f
          ${ONNXRUNTIME_ROOT}/core/mlas/lib/intrinsics/avx512/quantize_avx512f.cpp
          ${mlas_platform_srcs_avx512f}
        )
      else()
        set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DMLAS_AVX512F_INTRINSICS_UNSUPPORTED")
      endif()
      if(HAS_AVX512F)
        set_source_files_properties(${mlas_platform_srcs_avx512f} PROPERTIES COMPILE_FLAGS "-mavx512f")
      endif()

      check_cxx_compiler_flag("-mavx512bw -mavx512dq -mavx512vl" HAS_AVX512CORE)
      if(HAS_AVX512CORE)
        set(CMAKE_REQUIRED_FLAGS "-mavx512bw -mavx512dq -mavx512vl")
      endif()
      check_cxx_source_compiles("
        int main() {
          asm(\"vpmaddwd %zmm0,%zmm0,%zmm0\"); // AVX512BW feature
          asm(\"vandnps %xmm31,%xmm31,%xmm31\"); // AVX512DQ/AVX512VL feature
          return 0;
        }"
        COMPILES_AVX512CORE
      )

      if(COMPILES_AVX512CORE)
        set(mlas_platform_srcs_avx512core
          ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemvU8S8KernelAvx512Core.S
          ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemvU8S8KernelAvx512Vnni.S
          ${ONNXRUNTIME_ROOT}/core/mlas/lib/x86_64/QgemmU8X8KernelAvx512Core.S
        )
        if(HAS_AVX512CORE)
          set_source_files_properties(${mlas_platform_srcs_avx512core} PROPERTIES COMPILE_FLAGS "-mavx512bw -mavx512dq -mavx512vl")
        endif()
      else()
        set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DMLAS_AVX512CORE_UNSUPPORTED")
      endif()
    else()
      set_source_files_properties(${mlas_common_srcs} PROPERTIES COMPILE_FLAGS "-DMLAS_AVX512F_UNSUPPORTED")
    endif()

    set(mlas_platform_srcs
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/dgemm.cpp
      ${ONNXRUNTIME_ROOT}/core/mlas/lib/qgemm_kernel_avx2.cpp
      ${mlas_platform_srcs_sse2}
      ${mlas_platform_srcs_avx}
      ${mlas_platform_srcs_avx2}
      ${mlas_platform_srcs_avx512f}
      ${mlas_platform_srcs_avx512core}
    )
  endif()
endif()

onnxruntime_add_static_library(onnxruntime_mlas ${mlas_common_srcs} ${mlas_platform_srcs})
target_include_directories(onnxruntime_mlas PRIVATE ${ONNXRUNTIME_ROOT}/core/mlas/inc ${ONNXRUNTIME_ROOT}/core/mlas/lib)
set_target_properties(onnxruntime_mlas PROPERTIES FOLDER "ONNXRuntime")
if (WIN32)
  target_compile_options(onnxruntime_mlas PRIVATE "/wd6385" "/wd4127")
endif()

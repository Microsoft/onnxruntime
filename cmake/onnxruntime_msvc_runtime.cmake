# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

function(onnxruntime_msvc_spectre_enabled flags result_var)
  separate_arguments(msvc_flags WINDOWS_COMMAND "${flags}")
  set(spectre_enabled OFF)
  foreach(flag IN LISTS msvc_flags)
    if (flag STREQUAL "/Qspectre")
      set(spectre_enabled ON)
    elseif(flag STREQUAL "/Qspectre-")
      set(spectre_enabled OFF)
    endif()
  endforeach()
  set(${result_var} ${spectre_enabled} PARENT_SCOPE)
endfunction()

function(onnxruntime_configure_msvc_onecore_runtime)
  if (WIN32 AND NOT GDK_PLATFORM AND NOT CMAKE_CROSSCOMPILING)
    if (NOT CMAKE_CXX_STANDARD_LIBRARIES MATCHES kernel32.lib)
      # On OneCore, link to the OneCore build of the MSVC runtime.
      get_filename_component(msvc_path "${CMAKE_C_COMPILER}/../../../.." ABSOLUTE)
      set(msvc_onecore_platform "${onnxruntime_target_platform}")
      set(msvc_onecore_lib_dir "${msvc_path}/lib/onecore/${msvc_onecore_platform}")

      onnxruntime_msvc_spectre_enabled("${CMAKE_C_FLAGS}" msvc_c_spectre_enabled)
      onnxruntime_msvc_spectre_enabled("${CMAKE_CXX_FLAGS}" msvc_cxx_spectre_enabled)
      if (msvc_c_spectre_enabled OR msvc_cxx_spectre_enabled)
        if (msvc_onecore_platform STREQUAL "ARM64EC")
          # ARM64EC uses the ARM64 hybrid CRT libraries.
          set(msvc_onecore_platform "ARM64")
        endif()

        set(msvc_onecore_lib_dir "${msvc_path}/lib/spectre/onecore/${msvc_onecore_platform}")
        foreach(msvc_runtime_lib
                libcmt.lib
                libcmtd.lib
                libcpmt.lib
                libcpmtd.lib
                libvcruntime.lib
                libvcruntimed.lib)
          if (NOT EXISTS "${msvc_onecore_lib_dir}/${msvc_runtime_lib}")
            message(FATAL_ERROR
                    "Required Spectre-mitigated OneCore runtime library is missing: "
                    "${msvc_onecore_lib_dir}/${msvc_runtime_lib}")
          endif()
        endforeach()
      endif()

      message(STATUS "MSVC OneCore runtime library directory: ${msvc_onecore_lib_dir}")
      link_directories(BEFORE "${msvc_onecore_lib_dir}")
      # The MSVC runtime libraries contain a DEFAULTLIB entry for onecore.lib, but it does not conflict with
      # onecoreuap.lib.
    endif()
  endif()
endfunction()

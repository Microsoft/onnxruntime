# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

message(STATUS "Loading Dependencies URLs ...")

include(external/helper_functions.cmake)

file(STRINGS deps.txt ONNXRUNTIME_DEPS_LIST)
foreach(ONNXRUNTIME_DEP IN LISTS ONNXRUNTIME_DEPS_LIST)
  # Lines start with "#" are comments, so skip them.
  # cpp_client_telemetry is needed only when the 1DS backend is selected.
  if((NOT ONNXRUNTIME_DEP MATCHES "^#") AND
     ((NOT ONNXRUNTIME_DEP MATCHES "^cpp_client_telemetry") OR
      onnxruntime_USE_1DS_TELEMETRY))
    # The first column is name
    list(POP_FRONT ONNXRUNTIME_DEP ONNXRUNTIME_DEP_NAME)
    # The second column is URL
    # The URL below may be a local file path or an HTTPS URL
    list(POP_FRONT ONNXRUNTIME_DEP ONNXRUNTIME_DEP_URL)
    set(DEP_URL_${ONNXRUNTIME_DEP_NAME} ${ONNXRUNTIME_DEP_URL})
    # The third column is SHA1 hash value
    set(DEP_SHA1_${ONNXRUNTIME_DEP_NAME} ${ONNXRUNTIME_DEP})

    if(ONNXRUNTIME_DEP_URL MATCHES "^https://")
      # Search a local mirror folder
      string(REGEX REPLACE "^https://" "${onnxruntime_CMAKE_DEPS_MIRROR_DIR}/" LOCAL_URL "${ONNXRUNTIME_DEP_URL}")

      if(EXISTS "${LOCAL_URL}")
        cmake_path(ABSOLUTE_PATH LOCAL_URL)
        set(DEP_URL_${ONNXRUNTIME_DEP_NAME} "${LOCAL_URL}")
      endif()
    endif()
  endif()
endforeach()

message(STATUS "Loading Dependencies ...")
include(FetchContent)

# ABSL should be included before protobuf because protobuf may use absl
include(external/abseil-cpp.cmake)

set(RE2_BUILD_TESTING OFF CACHE BOOL "" FORCE)

onnxruntime_fetchcontent_declare(
    re2
    URL ${DEP_URL_re2}
    URL_HASH SHA1=${DEP_SHA1_re2}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS NAMES re2
)
onnxruntime_fetchcontent_makeavailable(re2)

if (onnxruntime_BUILD_UNIT_TESTS)
  # WebAssembly threading support in Node.js is still an experimental feature and
  # not working properly with googletest suite.
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    set(gtest_disable_pthreads ON)
  endif()
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  if (IOS OR ANDROID)
    # on mobile platforms the absl flags class dumps the flag names (assumably for binary size), which breaks passing
    # any args to gtest executables, such as using --gtest_filter to debug a specific test.
    # Processing of compile definitions:
    # https://github.com/abseil/abseil-cpp/blob/8dc90ff07402cd027daec520bb77f46e51855889/absl/flags/config.h#L21
    # If set, this code throws away the flag and does nothing on registration, which results in no flags being known:
    # https://github.com/abseil/abseil-cpp/blob/8dc90ff07402cd027daec520bb77f46e51855889/absl/flags/flag.h#L205-L217
    set(GTEST_HAS_ABSL OFF CACHE BOOL "" FORCE)
  else()
    set(GTEST_HAS_ABSL ON CACHE BOOL "" FORCE)
  endif()
  # gtest and gmock
  onnxruntime_fetchcontent_declare(
    googletest
    URL ${DEP_URL_googletest}
    URL_HASH SHA1=${DEP_SHA1_googletest}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 1.14.0...<2.0.0 NAMES GTest
  )
  FetchContent_MakeAvailable(googletest)
endif()

if (onnxruntime_BUILD_BENCHMARKS)
  # We will not need to test benchmark lib itself.
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing as we don't need it.")
  # We will not need to install benchmark since we link it statically.
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install to avoid overwriting vendor install.")

  onnxruntime_fetchcontent_declare(
    google_benchmark
    URL ${DEP_URL_google_benchmark}
    URL_HASH SHA1=${DEP_SHA1_google_benchmark}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 1.9.5 NAMES benchmark
  )
  onnxruntime_fetchcontent_makeavailable(google_benchmark)
endif()


if(onnxruntime_USE_MIMALLOC)
  add_definitions(-DUSE_MIMALLOC)

  set(MI_OVERRIDE OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MI_DEBUG_FULL OFF CACHE BOOL "" FORCE)
  set(MI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
  onnxruntime_fetchcontent_declare(
    mimalloc
    URL ${DEP_URL_mimalloc}
    URL_HASH SHA1=${DEP_SHA1_mimalloc}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS NAMES mimalloc
  )
  FetchContent_MakeAvailable(mimalloc)
endif()

# Download a protoc binary from Internet if needed
if(NOT ONNX_CUSTOM_PROTOC_EXECUTABLE AND NOT onnxruntime_USE_VCPKG)
  # This part of code is only for users' convenience. The code couldn't handle all cases. Users always can manually
  # download protoc from Protobuf's Github release page and pass the local path to the ONNX_CUSTOM_PROTOC_EXECUTABLE
  # variable.
  if (CMAKE_HOST_APPLE)
    # Using CMAKE_CROSSCOMPILING is not recommended for Apple target devices.
    # https://cmake.org/cmake/help/v3.26/variable/CMAKE_CROSSCOMPILING.html
    # To keep it simple, just download and use the universal protoc binary for all Apple host builds.
    onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_mac_universal} URL_HASH SHA1=${DEP_SHA1_protoc_mac_universal} EXCLUDE_FROM_ALL)
    FetchContent_Populate(protoc_binary)
    if(protoc_binary_SOURCE_DIR)
      message(STATUS "Use prebuilt protoc")
      set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${protoc_binary_SOURCE_DIR}/bin/protoc)
      set(PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
    endif()
  elseif (CMAKE_CROSSCOMPILING)
    message(STATUS "CMAKE_HOST_SYSTEM_NAME: ${CMAKE_HOST_SYSTEM_NAME}")
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
      if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
        onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_win64} URL_HASH SHA1=${DEP_SHA1_protoc_win64} EXCLUDE_FROM_ALL)
        FetchContent_Populate(protoc_binary)
      elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86")
        onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_win32} URL_HASH SHA1=${DEP_SHA1_protoc_win32} EXCLUDE_FROM_ALL)
        FetchContent_Populate(protoc_binary)
      elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ARM64")
        onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_win64} URL_HASH SHA1=${DEP_SHA1_protoc_win64} EXCLUDE_FROM_ALL)
        FetchContent_Populate(protoc_binary)
      endif()

      if(protoc_binary_SOURCE_DIR)
        message(STATUS "Use prebuilt protoc")
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${protoc_binary_SOURCE_DIR}/bin/protoc.exe)
        set(PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
      endif()
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
      if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64)$")
        onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_linux_x64} URL_HASH SHA1=${DEP_SHA1_protoc_linux_x64} EXCLUDE_FROM_ALL)
        FetchContent_Populate(protoc_binary)
      elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(i.86|x86?)$")
        onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_linux_x86} URL_HASH SHA1=${DEP_SHA1_protoc_linux_x86} EXCLUDE_FROM_ALL)
        FetchContent_Populate(protoc_binary)
      elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^aarch64.*")
        onnxruntime_fetchcontent_declare(protoc_binary URL ${DEP_URL_protoc_linux_aarch64} URL_HASH SHA1=${DEP_SHA1_protoc_linux_aarch64} EXCLUDE_FROM_ALL)
        FetchContent_Populate(protoc_binary)
      endif()

      if(protoc_binary_SOURCE_DIR)
        message(STATUS "Use prebuilt protoc")
        set(ONNX_CUSTOM_PROTOC_EXECUTABLE ${protoc_binary_SOURCE_DIR}/bin/protoc)
        set(PROTOC_EXECUTABLE ${ONNX_CUSTOM_PROTOC_EXECUTABLE})
      endif()
    endif()

    if(NOT ONNX_CUSTOM_PROTOC_EXECUTABLE)
      message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE must be set to cross-compile.")
    endif()
  endif()
endif()

# if ONNX_CUSTOM_PROTOC_EXECUTABLE is set we don't need to build the protoc binary
if (ONNX_CUSTOM_PROTOC_EXECUTABLE)
  if (NOT EXISTS "${ONNX_CUSTOM_PROTOC_EXECUTABLE}")
    message(FATAL_ERROR "ONNX_CUSTOM_PROTOC_EXECUTABLE is set to '${ONNX_CUSTOM_PROTOC_EXECUTABLE}' "
                        "but protoc executable was not found there.")
  endif()

  set(protobuf_BUILD_PROTOC_BINARIES OFF CACHE BOOL "Build protoc" FORCE)
endif()

#Here we support two build mode:
#1. if ONNX_CUSTOM_PROTOC_EXECUTABLE is set, build Protobuf from source, except protoc.exe. This mode is mainly
#   for cross-compiling
#2. if ONNX_CUSTOM_PROTOC_EXECUTABLE is not set, Compile everything(including protoc) from source code.
if(Patch_FOUND)
  set(ONNXRUNTIME_PROTOBUF_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/protobuf/protobuf_android_log.patch &&
                                         ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/protobuf/protobuf_msvc_unreachable_code.patch &&
                                         ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/protobuf/protobuf_msvc_map_unreachable_code.patch &&
                                         ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/protobuf/protobuf_compiler_incomplete_type.patch)
else()
 set(ONNXRUNTIME_PROTOBUF_PATCH_COMMAND "")
endif()

#Protobuf depends on absl and utf8_range
onnxruntime_fetchcontent_declare(
  Protobuf
  URL ${DEP_URL_protobuf}
  URL_HASH SHA1=${DEP_SHA1_protobuf}
  PATCH_COMMAND ${ONNXRUNTIME_PROTOBUF_PATCH_COMMAND}
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS NAMES Protobuf protobuf
)

set(protobuf_BUILD_TESTS OFF CACHE BOOL "Build protobuf tests" FORCE)
#TODO: we'd better to turn the following option off. However, it will cause
# ".\build.bat --config Debug --parallel --skip_submodule_sync --update" fail with an error message:
# install(EXPORT "ONNXTargets" ...) includes target "onnx_proto" which requires target "libprotobuf-lite" that is
# not in any export set.
#set(protobuf_INSTALL OFF CACHE BOOL "Install protobuf binaries and files" FORCE)
set(protobuf_USE_EXTERNAL_GTEST ON CACHE BOOL "" FORCE)

if (ANDROID)
  set(protobuf_WITH_ZLIB OFF CACHE BOOL "Build protobuf with zlib support" FORCE)
endif()

if (onnxruntime_DISABLE_RTTI)
  set(protobuf_DISABLE_RTTI ON CACHE BOOL "Remove runtime type information in the binaries" FORCE)
endif()

include(protobuf_function)
#protobuf end

onnxruntime_fetchcontent_makeavailable(Protobuf)
if(Protobuf_FOUND)
  message(STATUS "Using protobuf from find_package(or vcpkg). Protobuf version: ${Protobuf_VERSION}")
else()
  # Adjust warning flags
  if (TARGET libprotoc)
    if (NOT MSVC)
      target_compile_options(libprotoc PRIVATE "-w")
    endif()
  endif()
  if (TARGET protoc)
    add_executable(protobuf::protoc ALIAS protoc)
    if (UNIX AND onnxruntime_ENABLE_LTO)
      #https://github.com/protocolbuffers/protobuf/issues/5923
      target_link_options(protoc PRIVATE "-Wl,--no-as-needed")
    endif()
    if (NOT MSVC)
      target_compile_options(protoc PRIVATE "-w")
    endif()
    get_target_property(PROTOC_OSX_ARCH protoc OSX_ARCHITECTURES)
    if (PROTOC_OSX_ARCH)
      if (${CMAKE_HOST_SYSTEM_PROCESSOR} IN_LIST PROTOC_OSX_ARCH)
        message(STATUS "protoc can run")
      else()
        list(APPEND PROTOC_OSX_ARCH ${CMAKE_HOST_SYSTEM_PROCESSOR})
        set_target_properties(protoc PROPERTIES OSX_ARCHITECTURES "${CMAKE_HOST_SYSTEM_PROCESSOR}")
        set_target_properties(libprotoc PROPERTIES OSX_ARCHITECTURES "${PROTOC_OSX_ARCH}")
        set_target_properties(libprotobuf PROPERTIES OSX_ARCHITECTURES "${PROTOC_OSX_ARCH}")
      endif()
    endif()
   endif()
  if (TARGET libprotobuf AND NOT MSVC)
    target_compile_options(libprotobuf PRIVATE "-w")
  endif()
  if (TARGET libprotobuf-lite AND NOT MSVC)
    target_compile_options(libprotobuf-lite PRIVATE "-w")
  endif()
endif()
if (onnxruntime_USE_FULL_PROTOBUF)
  set(PROTOBUF_LIB protobuf::libprotobuf)
else()
  set(PROTOBUF_LIB protobuf::libprotobuf-lite)
endif()

# date
set(ENABLE_DATE_TESTING  OFF CACHE BOOL "" FORCE)
set(USE_SYSTEM_TZ_DB  ON CACHE BOOL "" FORCE)

onnxruntime_fetchcontent_declare(
  date
  URL ${DEP_URL_date}
  URL_HASH SHA1=${DEP_SHA1_date}
  EXCLUDE_FROM_ALL
  PATCH_COMMAND
    ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/date/date.patch
  FIND_PACKAGE_ARGS 3...<4 NAMES date
)
onnxruntime_fetchcontent_makeavailable(date)

if(NOT TARGET Boost::mp11)
  if(onnxruntime_USE_VCPKG)
     find_package(Boost REQUIRED)
     message(STATUS "Aliasing Boost::headers to Boost::mp11")
     add_library(Boost::mp11 ALIAS Boost::headers)
  else()
    onnxruntime_fetchcontent_declare(
     mp11
     URL ${DEP_URL_mp11}
     EXCLUDE_FROM_ALL
     FIND_PACKAGE_ARGS NAMES Boost
    )
    FetchContent_Populate(mp11)
    if(NOT TARGET Boost::mp11)
      add_library(Boost::mp11 IMPORTED INTERFACE)
      target_include_directories(Boost::mp11 INTERFACE $<BUILD_INTERFACE:${mp11_SOURCE_DIR}/include>)
    endif()
  endif()
endif()

set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install ON CACHE INTERNAL "")

onnxruntime_fetchcontent_declare(
    nlohmann_json
    URL ${DEP_URL_json}
    URL_HASH SHA1=${DEP_SHA1_json}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 3.10 NAMES nlohmann_json
)
onnxruntime_fetchcontent_makeavailable(nlohmann_json)

#TODO: include clog first
if (onnxruntime_ENABLE_CPUINFO)
  # Adding pytorch CPU info library
  # TODO!! need a better way to find out the supported architectures
  set(CPUINFO_SUPPORTED FALSE)
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # if xnnpack is enabled in a wasm build it needs clog from cpuinfo, but we won't internally use cpuinfo.
    if (onnxruntime_USE_XNNPACK)
      set(CPUINFO_SUPPORTED TRUE)
    endif()
  elseif (APPLE)
    list(LENGTH CMAKE_OSX_ARCHITECTURES CMAKE_OSX_ARCHITECTURES_LEN)
    if (CMAKE_OSX_ARCHITECTURES_LEN LESS_EQUAL 1)
      set(CPUINFO_SUPPORTED TRUE)
    else()
      message(WARNING "cpuinfo is not supported when CMAKE_OSX_ARCHITECTURES has more than one value.")
    endif()
  elseif (WIN32)
    set(CPUINFO_SUPPORTED TRUE)
  else()
    if (onnxruntime_target_platform MATCHES "^(i[3-6]86|AMD64|x86(_64)?|armv[5-8].*|aarch64|arm64)$")
      set(CPUINFO_SUPPORTED TRUE)
    else()
      message(WARNING "Target processor architecture \"${onnxruntime_target_platform}\" is not supported in cpuinfo.")
    endif()
  endif()

  if(NOT CPUINFO_SUPPORTED)
    message(WARNING "onnxruntime_ENABLE_CPUINFO was set but cpuinfo is not supported.")
  endif()
endif()

if (CPUINFO_SUPPORTED)
  if (CMAKE_SYSTEM_NAME STREQUAL "iOS")
    set(IOS ON CACHE INTERNAL "")
    set(IOS_ARCH "${CMAKE_OSX_ARCHITECTURES}" CACHE INTERNAL "")
  endif()

  # if this is a wasm build with xnnpack (only type of wasm build where cpuinfo is involved)
  # we do not use cpuinfo in ORT code, so don't define CPUINFO_SUPPORTED.
  if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND onnxruntime_USE_XNNPACK)
  else()
    add_compile_definitions(CPUINFO_SUPPORTED)
  endif()

  set(CPUINFO_BUILD_TOOLS OFF CACHE INTERNAL "")
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE INTERNAL "")
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE INTERNAL "")
  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
  if(onnxruntime_USE_VCPKG AND NOT APPLE)
    find_package(cpuinfo CONFIG REQUIRED)
  else()
    if (onnxruntime_target_platform STREQUAL "ARM64EC" OR onnxruntime_target_platform STREQUAL "ARM64")
      message(STATUS "Applying patches for Windows ARM64/ARM64EC in cpuinfo")
      onnxruntime_fetchcontent_declare(
        pytorch_cpuinfo
        URL ${DEP_URL_pytorch_cpuinfo}
        URL_HASH SHA1=${DEP_SHA1_pytorch_cpuinfo}
        EXCLUDE_FROM_ALL
        PATCH_COMMAND
          ${Patch_EXECUTABLE} -p1 < ${PROJECT_SOURCE_DIR}/patches/cpuinfo/patch_cpuinfo_h_for_arm64ec.patch &&
          # https://github.com/pytorch/cpuinfo/pull/324
          ${Patch_EXECUTABLE} -p1 < ${PROJECT_SOURCE_DIR}/patches/cpuinfo/patch_vcpkg_arm64ec_support.patch &&
          # https://github.com/pytorch/cpuinfo/pull/400
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 <
          ${PROJECT_SOURCE_DIR}/patches/cpuinfo/enable_deinit_refcounting.patch
      )
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      message(STATUS "Applying sysfs fallback patch for cpuinfo on Linux")
      onnxruntime_fetchcontent_declare(
        pytorch_cpuinfo
        URL ${DEP_URL_pytorch_cpuinfo}
        URL_HASH SHA1=${DEP_SHA1_pytorch_cpuinfo}
        EXCLUDE_FROM_ALL
        PATCH_COMMAND
          # https://github.com/microsoft/onnxruntime/issues/10038
          ${Patch_EXECUTABLE} -p1 < ${PROJECT_SOURCE_DIR}/patches/cpuinfo/fix_missing_sysfs_fallback.patch &&
          # https://github.com/pytorch/cpuinfo/pull/400
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 <
          ${PROJECT_SOURCE_DIR}/patches/cpuinfo/enable_deinit_refcounting.patch
      )
    else()
      onnxruntime_fetchcontent_declare(
        pytorch_cpuinfo
        URL ${DEP_URL_pytorch_cpuinfo}
        URL_HASH SHA1=${DEP_SHA1_pytorch_cpuinfo}
        EXCLUDE_FROM_ALL
        PATCH_COMMAND
          # https://github.com/pytorch/cpuinfo/pull/400
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 <
          ${PROJECT_SOURCE_DIR}/patches/cpuinfo/enable_deinit_refcounting.patch
      )
    endif()
    onnxruntime_fetchcontent_makeavailable(pytorch_cpuinfo)
  endif()
  if(TARGET cpuinfo::cpuinfo AND NOT TARGET cpuinfo)
    message(STATUS "Aliasing cpuinfo::cpuinfo to cpuinfo")
    add_library(cpuinfo ALIAS cpuinfo::cpuinfo)
  endif()
endif()

onnxruntime_fetchcontent_declare(
  GSL
  URL ${DEP_URL_microsoft_gsl}
  URL_HASH SHA1=${DEP_SHA1_microsoft_gsl}
  # Stringify fix for GSL_SUPPRESS on MSVC (C4875). Remove when GSL ships a release
  # containing microsoft/GSL#1213 (commit 543d0dd).
  PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/gsl/1213.patch
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS 4.0 NAMES Microsoft.GSL
)
set(GSL_TARGET "Microsoft.GSL::GSL")
set(GSL_INCLUDE_DIR "$<TARGET_PROPERTY:${GSL_TARGET},INTERFACE_INCLUDE_DIRECTORIES>")
onnxruntime_fetchcontent_makeavailable(GSL)

if (NOT GSL_FOUND AND NOT onnxruntime_BUILD_SHARED_LIB)
  install(TARGETS GSL EXPORT ${PROJECT_NAME}Targets
  ARCHIVE  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY  DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME  DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

find_path(safeint_SOURCE_DIR NAMES "SafeInt.hpp")
if(NOT safeint_SOURCE_DIR)
  unset(safeint_SOURCE_DIR)
  onnxruntime_fetchcontent_declare(
      safeint
      URL ${DEP_URL_safeint}
      URL_HASH SHA1=${DEP_SHA1_safeint}
      EXCLUDE_FROM_ALL
  )

  # use fetch content rather than makeavailable because safeint only includes unconditional test targets
  FetchContent_Populate(safeint)
endif()
add_library(safeint_interface IMPORTED INTERFACE)
target_include_directories(safeint_interface INTERFACE ${safeint_SOURCE_DIR})


# Flatbuffers
if(onnxruntime_USE_VCPKG)
  find_package(flatbuffers REQUIRED)
else()
# We do not need to build flatc for iOS or Android Cross Compile
if (CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "tvOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS" OR CMAKE_SYSTEM_NAME STREQUAL "Android" OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  set(FLATBUFFERS_BUILD_FLATC OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATC" FORCE)
endif()
set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "FLATBUFFERS_BUILD_TESTS" FORCE)
set(FLATBUFFERS_INSTALL ON CACHE BOOL "FLATBUFFERS_INSTALL" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "FLATBUFFERS_BUILD_FLATHASH" FORCE)
set(FLATBUFFERS_BUILD_FLATLIB ON CACHE BOOL "FLATBUFFERS_BUILD_FLATLIB" FORCE)
if(Patch_FOUND)
  set(ONNXRUNTIME_FLATBUFFERS_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/flatbuffers/flatbuffers.patch)
else()
 set(ONNXRUNTIME_FLATBUFFERS_PATCH_COMMAND "")
endif()

#flatbuffers 1.11.0 does not have flatbuffers::IsOutRange, therefore we require 1.12.0+
onnxruntime_fetchcontent_declare(
    flatbuffers
    URL ${DEP_URL_flatbuffers}
    URL_HASH SHA1=${DEP_SHA1_flatbuffers}
    PATCH_COMMAND ${ONNXRUNTIME_FLATBUFFERS_PATCH_COMMAND}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS 23.5.9 NAMES Flatbuffers flatbuffers
)

onnxruntime_fetchcontent_makeavailable(flatbuffers)
if(NOT flatbuffers_FOUND)
  if(NOT TARGET flatbuffers::flatbuffers)
    add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
  endif()
  if(TARGET flatc AND NOT TARGET flatbuffers::flatc)
    add_executable(flatbuffers::flatc ALIAS flatc)
  endif()
  if (GDK_PLATFORM)
    # cstdlib only defines std::getenv when _CRT_USE_WINAPI_FAMILY_DESKTOP_APP is defined, which
    # is probably an oversight for GDK/Xbox builds (::getenv exists and works).
    file(WRITE ${CMAKE_BINARY_DIR}/gdk_cstdlib_wrapper.h [[
#pragma once
#ifdef __cplusplus
#include <cstdlib>
namespace std { using ::getenv; }
#endif
]])
    if(TARGET flatbuffers)
      target_compile_options(flatbuffers PRIVATE /FI${CMAKE_BINARY_DIR}/gdk_cstdlib_wrapper.h)
    endif()
    if(TARGET flatc)
      target_compile_options(flatc PRIVATE /FI${CMAKE_BINARY_DIR}/gdk_cstdlib_wrapper.h)
    endif()
  endif()
endif()
endif()

# ONNX
if (NOT onnxruntime_USE_FULL_PROTOBUF)
  set(ONNX_USE_LITE_PROTO ON CACHE BOOL "" FORCE)
else()
  set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)
endif()

if(Patch_FOUND)
  set(ONNXRUNTIME_ONNX_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/onnx/onnx.patch)
else()
  set(ONNXRUNTIME_ONNX_PATCH_COMMAND "")
endif()

if(onnxruntime_ENABLE_PYTHON)
  if(onnxruntime_USE_VCPKG)
    find_package(pybind11 CONFIG REQUIRED)
  else()
    include(pybind11)
  endif()
if(TARGET pybind11::module)
  message("Setting pybind11_lib")
  set(pybind11_lib pybind11::module)
else()
  message("Setting pybind11_dep")
  set(pybind11_dep pybind11::pybind11)
endif()

endif()
onnxruntime_fetchcontent_declare(
  onnx
  URL ${DEP_URL_onnx}
  URL_HASH SHA1=${DEP_SHA1_onnx}
  PATCH_COMMAND ${ONNXRUNTIME_ONNX_PATCH_COMMAND}
  EXCLUDE_FROM_ALL
  FIND_PACKAGE_ARGS NAMES ONNX onnx
)

onnxruntime_fetchcontent_makeavailable(onnx)

if(TARGET ONNX::onnx AND NOT TARGET onnx)
  message(STATUS "Aliasing ONNX::onnx to onnx")
  add_library(onnx ALIAS ONNX::onnx)
endif()
if(TARGET ONNX::onnx_proto AND NOT TARGET onnx_proto)
  message(STATUS "Aliasing ONNX::onnx_proto to onnx_proto")
  add_library(onnx_proto ALIAS ONNX::onnx_proto)
endif()

# An installed ONNX package ships pre-generated protobuf headers/sources that ONNX Runtime compiles into its own
# objects, so both must resolve to a single protobuf runtime with a single message representation. If they do not,
# the mismatch is not caught by the linker: it shows up as heap corruption while copying onnx::AttributeProto (see
# https://github.com/microsoft/onnxruntime/issues/28664). Validate the two ways this can happen.
if(onnx_FOUND OR ONNX_FOUND)
  if(NOT Protobuf_FOUND)
    message(FATAL_ERROR
            "ONNX was resolved to an installed package but protobuf is being built from source by ONNX Runtime. "
            "The installed ONNX links a different protobuf runtime than the one ONNX Runtime would use. "
            "Install a protobuf CMake config package and make it discoverable via CMAKE_PREFIX_PATH, or force ONNX "
            "to be built from source with -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER.")
  endif()

  if(TARGET ONNX::onnx_proto)
    get_target_property(onnx_package_definitions ONNX::onnx_proto INTERFACE_COMPILE_DEFINITIONS)
  else()
    get_target_property(onnx_package_definitions onnx_proto INTERFACE_COMPILE_DEFINITIONS)
  endif()
  if(onnx_package_definitions)
    if("ONNX_USE_LITE_PROTO=1" IN_LIST onnx_package_definitions)
      set(onnx_package_protobuf_flavor "lite")
    else()
      set(onnx_package_protobuf_flavor "full")
    endif()
    if(onnxruntime_USE_FULL_PROTOBUF)
      set(onnxruntime_protobuf_flavor "full")
    else()
      set(onnxruntime_protobuf_flavor "lite")
    endif()
    if(NOT onnx_package_protobuf_flavor STREQUAL onnxruntime_protobuf_flavor)
      message(FATAL_ERROR
              "The installed ONNX package links the ${onnx_package_protobuf_flavor} protobuf runtime but ONNX "
              "Runtime is configured for the ${onnxruntime_protobuf_flavor} one "
              "(onnxruntime_USE_FULL_PROTOBUF=${onnxruntime_USE_FULL_PROTOBUF}). Rebuild ONNX with a matching "
              "ONNX_USE_LITE_PROTO, or reconfigure ONNX Runtime to use the ${onnx_package_protobuf_flavor} runtime.")
    endif()
  endif()
  message(STATUS "Using ONNX from find_package(or vcpkg). ONNX version: ${ONNX_VERSION}, "
                 "protobuf version: ${Protobuf_VERSION}")
endif()

if(onnxruntime_USE_VCPKG)
  find_package(Eigen3 CONFIG REQUIRED)
else()
  include(external/eigen.cmake)
endif()

if(WIN32)
  if(onnxruntime_USE_VCPKG)
    find_package(wil CONFIG REQUIRED)
    set(WIL_TARGET "WIL::WIL")
  else()
    include(wil) # FetchContent
  endif()
endif()

# XNNPACK EP
if (onnxruntime_USE_XNNPACK)
  if (onnxruntime_DISABLE_CONTRIB_OPS)
    message(FATAL_ERROR "XNNPACK EP requires the internal NHWC contrib ops to be available "
                         "but onnxruntime_DISABLE_CONTRIB_OPS is ON")
  endif()
  if(onnxruntime_USE_VCPKG)
     FIND_PATH(XNNPACK_HDR xnnpack.h PATH_SUFFIXES include)
     IF(NOT XNNPACK_HDR)
       MESSAGE(FATAL_ERROR "Cannot find xnnpack")
     ENDIF()
     ADD_LIBRARY(xnnpack STATIC IMPORTED)
     find_library(xnnpack_LIBRARY NAMES XNNPACK)
     find_library(microkernels_prod_LIBRARY NAMES xnnpack-microkernels-prod)
     find_package(unofficial-pthreadpool CONFIG REQUIRED)

     target_include_directories(xnnpack INTERFACE "${XNNPACK_HDR}")
     set(XNNPACK_INCLUDE_DIR ${XNNPACK_DIR}/include)
     set(onnxruntime_EXTERNAL_LIBRARIES_XNNPACK ${xnnpack_LIBRARY} ${microkernels_prod_LIBRARY} unofficial::pthreadpool unofficial::pthreadpool_interface)
  else()
    include(xnnpack)
  endif()
endif()

set(onnxruntime_EXTERNAL_LIBRARIES ${onnxruntime_EXTERNAL_LIBRARIES_XNNPACK} ${WIL_TARGET} nlohmann_json::nlohmann_json
                                   onnx onnx_proto ${PROTOBUF_LIB} re2::re2 Boost::mp11 safeint_interface
                                   flatbuffers::flatbuffers ${GSL_TARGET} ${ABSEIL_LIBS} date::date Eigen3::Eigen)

# The source code of onnx_proto is generated, we must build this lib first before starting to compile the other source code that uses ONNX protobuf types.
# The other libs do not have the problem. All the sources are already there. We can compile them in any order.
set(onnxruntime_EXTERNAL_DEPENDENCIES onnx_proto flatbuffers::flatbuffers)

if(NOT (onnx_FOUND OR ONNX_FOUND)) # building ONNX from source
  target_compile_definitions(onnx PUBLIC $<TARGET_PROPERTY:onnx_proto,INTERFACE_COMPILE_DEFINITIONS> PRIVATE "__ONNX_DISABLE_STATIC_REGISTRATION")
  if (NOT onnxruntime_USE_FULL_PROTOBUF)
    target_compile_definitions(onnx PUBLIC "__ONNX_NO_DOC_STRINGS")
  endif()
endif()

if(onnxruntime_ENABLE_DLPACK)
  message(STATUS "dlpack is enabled.")

  onnxruntime_fetchcontent_declare(
    dlpack
    URL ${DEP_URL_dlpack}
    URL_HASH SHA1=${DEP_SHA1_dlpack}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS NAMES dlpack
  )
  onnxruntime_fetchcontent_makeavailable(dlpack)
endif()

if(onnxruntime_ENABLE_TRAINING OR (onnxruntime_ENABLE_TRAINING_APIS AND onnxruntime_BUILD_UNIT_TESTS))
  # Once code under orttraining/orttraining/models dir is removed "onnxruntime_ENABLE_TRAINING" should be removed from
  # this conditional
  if(Patch_FOUND)
    set(ONNXRUNTIME_CXXOPTS_PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/cxxopts/gcc-15-compat.patch)
  else()
    set(ONNXRUNTIME_CXXOPTS_PATCH_COMMAND "")
  endif()

  onnxruntime_fetchcontent_declare(
    cxxopts
    URL ${DEP_URL_cxxopts}
    URL_HASH SHA1=${DEP_SHA1_cxxopts}
    PATCH_COMMAND ${ONNXRUNTIME_CXXOPTS_PATCH_COMMAND}
    EXCLUDE_FROM_ALL
    FIND_PACKAGE_ARGS NAMES cxxopts
  )
  set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  onnxruntime_fetchcontent_makeavailable(cxxopts)
endif()


if (onnxruntime_USE_WEBGPU)
  if (onnxruntime_DAWN_PREBUILT_DIR)
    if (CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR
        onnxruntime_BUILD_DAWN_SHARED_LIBRARY OR
        onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP OR DAWN_USE_AGILITY_SDK)
      message(FATAL_ERROR "The Dawn API package supports native external Dawn without shared Dawn, PIX, or Agility SDK")
    endif()
    if (onnxruntime_CUSTOM_DAWN_SRC_PATH)
      message(FATAL_ERROR "onnxruntime_DAWN_PREBUILT_DIR and onnxruntime_CUSTOM_DAWN_SRC_PATH are mutually exclusive")
    endif()

    set(ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR "${onnxruntime_DAWN_PREBUILT_DIR}/include")
    set(ONNXRUNTIME_DAWN_PROC_SRC
      "${onnxruntime_DAWN_PREBUILT_DIR}/src/dawn_proc.cpp"
      "${onnxruntime_DAWN_PREBUILT_DIR}/src/dawn_thread_dispatch_proc.cpp")
    foreach(_dawn_required_file IN ITEMS
        "${ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR}/webgpu/webgpu_cpp.h"
        "${ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR}/dawn/dawn_proc.h"
        "${ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR}/dawn/dawn_version.h"
        "${ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR}/dawn/dawn_thread_dispatch_proc.h"
        ${ONNXRUNTIME_DAWN_PROC_SRC})
      if (NOT EXISTS "${_dawn_required_file}")
        message(FATAL_ERROR "Dawn API package file not found: ${_dawn_required_file}")
      endif()
    endforeach()

    foreach(_dawn_header_target IN ITEMS dawn::dawncpp_headers dawn::dawn_headers)
      if (NOT TARGET ${_dawn_header_target})
        add_library(${_dawn_header_target} INTERFACE IMPORTED)
        set_target_properties(${_dawn_header_target} PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR}")
      endif()
    endforeach()
    if (NOT TARGET dawn::dawn_proc)
      add_library(onnxruntime_dawn_proc STATIC ${ONNXRUNTIME_DAWN_PROC_SRC})
      target_include_directories(onnxruntime_dawn_proc PUBLIC "${ONNXRUNTIME_DAWN_PREBUILT_INCLUDE_DIR}")
      if (MSVC)
        target_compile_options(onnxruntime_dawn_proc PRIVATE /W0)
      else()
        target_compile_options(onnxruntime_dawn_proc PRIVATE -w)
      endif()
      add_library(dawn::dawn_proc ALIAS onnxruntime_dawn_proc)
    endif()
    message(STATUS "Using Dawn API package: ${onnxruntime_DAWN_PREBUILT_DIR}")
  else()
  if (DAWN_USE_AGILITY_SDK)
    if (NOT WIN32)
      message(FATAL_ERROR "DAWN_USE_AGILITY_SDK is only supported on Windows.")
    endif()
    if (onnxruntime_CUSTOM_DAWN_SRC_PATH)
      message(FATAL_ERROR "DAWN_USE_AGILITY_SDK is not supported with onnxruntime_CUSTOM_DAWN_SRC_PATH.")
    endif()
    if (CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
      message(FATAL_ERROR "DAWN_USE_AGILITY_SDK is not supported for WindowsStore/UWP.")
    endif()
    if (onnxruntime_target_platform STREQUAL "ARM" OR
        onnxruntime_target_platform STREQUAL "ARM64EC")
      message(FATAL_ERROR
              "DAWN_USE_AGILITY_SDK does not support Windows ARM32 or ARM64EC. "
              "Use an x86, x64, or ARM64 target.")
    endif()
    if (NOT onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
      message(FATAL_ERROR "DAWN_USE_AGILITY_SDK requires the Dawn D3D12 backend.")
    endif()
  endif()

  # TODO: the following code is used to disable building Dawn using vcpkg temporarily
  # until we figure out how to resolve the packaging pipeline failures
  #
  # if (onnxruntime_USE_VCPKG AND NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
  if (FALSE)
    # vcpkg does not support Emscripten yet
    find_package(dawn REQUIRED)
  else()
    #
    # Please keep the following in sync with cmake/vcpkg-ports/dawn/portfile.cmake
    #
    set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DAWN_SUPPORTS_CXX_MODULES OFF CACHE BOOL "" FORCE)
    if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
      if (NOT onnxruntime_USE_EXTERNAL_DAWN AND NOT onnxruntime_DISABLE_RTTI)
        # ORT derives from dawn::platform::Platform to configure Dawn's worker pool.
        # Match ORT's RTTI setting so Dawn emits the base class typeinfo required by the subclass.
        set(DAWN_ENABLE_RTTI ON CACHE BOOL "" FORCE)
      endif()

      if (onnxruntime_BUILD_DAWN_SHARED_LIBRARY)
        set(DAWN_BUILD_MONOLITHIC_LIBRARY SHARED CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_INSTALL ON CACHE BOOL "" FORCE)

        if (onnxruntime_USE_EXTERNAL_DAWN)
          message(FATAL_ERROR "onnxruntime_USE_EXTERNAL_DAWN and onnxruntime_BUILD_DAWN_SHARED_LIBRARY cannot be enabled at the same time.")
        endif()
      else()
        # use dawn::dawn_native and dawn::dawn_proc instead of the monolithic dawn::webgpu_dawn to minimize binary size
        set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

        # use the same protobuf/abseil for ORT and Dawn when static linking
        if(abseil_cpp_SOURCE_DIR)
          set(DAWN_ABSEIL_DIR ${abseil_cpp_SOURCE_DIR})
        endif()
        if(protobuf_SOURCE_DIR)
          set(DAWN_PROTOBUF_DIR ${protobuf_SOURCE_DIR})
        endif()
      endif()

      if (onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP)
        set(DAWN_ENABLE_DESKTOP_GL ON CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_OPENGLES ON CACHE BOOL "" FORCE)
        set(DAWN_SUPPORTS_GLFW_FOR_WINDOWING ON CACHE BOOL "" FORCE)
        set(DAWN_USE_GLFW ON CACHE BOOL "" FORCE)
        set(DAWN_USE_WINDOWS_UI ON CACHE BOOL "" FORCE)
        set(TINT_BUILD_GLSL_WRITER ON CACHE BOOL "" FORCE)
        set(TINT_BUILD_GLSL_VALIDATOR ON CACHE BOOL "" FORCE)
      else()
        set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
        set(DAWN_ENABLE_OPENGLES OFF CACHE BOOL "" FORCE)
        set(DAWN_SUPPORTS_GLFW_FOR_WINDOWING OFF CACHE BOOL "" FORCE)
        set(DAWN_USE_GLFW OFF CACHE BOOL "" FORCE)
        set(DAWN_USE_WINDOWS_UI OFF CACHE BOOL "" FORCE)
        set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
        set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "" FORCE)
      endif()

      # disable things we don't use
      set(DAWN_DXC_ENABLE_ASSERTS_IN_NDEBUG OFF)
      set(DAWN_USE_X11 OFF CACHE BOOL "" FORCE)

      set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
      set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
      set(TINT_BUILD_IR_BINARY OFF CACHE BOOL "" FORCE)
      set(TINT_BUILD_SPV_READER OFF CACHE BOOL "" FORCE)  # don't need. disabling is a large binary size saving
      set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)  # needed to create cache key. runtime error if not enabled.

      # SPIR-V validation shouldn't be required given we're using Tint to create the SPIR-V.
      set(DAWN_ENABLE_SPIRV_VALIDATION OFF CACHE BOOL "" FORCE)

      if (WIN32)
        # building this requires the HLSL writer to be enabled in Tint. TBD if that we need either of these to be ON.
        set(DAWN_USE_BUILT_DXC ON CACHE BOOL "" FORCE)
        set(TINT_BUILD_HLSL_WRITER ON CACHE BOOL "" FORCE)

        if ((NOT onnxruntime_ENABLE_DAWN_BACKEND_VULKAN) AND (NOT onnxruntime_ENABLE_DAWN_BACKEND_D3D12))
          message(FATAL_ERROR "At least one of onnxruntime_ENABLE_DAWN_BACKEND_VULKAN or onnxruntime_ENABLE_DAWN_BACKEND_D3D12 must be enabled when using Dawn on Windows.")
        endif()
        if (onnxruntime_ENABLE_DAWN_BACKEND_VULKAN)
          set(DAWN_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
          set(TINT_BUILD_SPV_WRITER ON CACHE BOOL "" FORCE)
        else()
          set(DAWN_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
        endif()
        if (onnxruntime_ENABLE_DAWN_BACKEND_D3D12)
          set(DAWN_ENABLE_D3D12 ON CACHE BOOL "" FORCE)
        else()
          set(DAWN_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
        endif()
        # We are currently always using the D3D12 backend.
        set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
      endif()
    endif()

    if (onnxruntime_CUSTOM_DAWN_SRC_PATH)
      set(DAWN_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)
      # use the custom dawn source path if provided
      #
      # specified as:
      # build.py --use_webgpu --cmake_extra_defines "onnxruntime_CUSTOM_DAWN_SRC_PATH=<PATH_TO_DAWN_SRC_ROOT>"
      onnxruntime_fetchcontent_declare(
        dawn
        SOURCE_DIR ${onnxruntime_CUSTOM_DAWN_SRC_PATH}
        EXCLUDE_FROM_ALL
      )
    else()
      set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
      set(ONNXRUNTIME_Dawn_PATCH_COMMAND
          # The dawn_destroy_buffer_on_destructor.patch contains the following changes:
          #
          # - (private) Allow WGPUBufferImpl class to destroy the buffer in the destructor
          #   In native implementation, wgpuBufferRelease will trigger the buffer destroy (if refcount decreased to 0). But
          #   in emwgpu implementation, the buffer destroy won't happen. This change adds a destructor to the buffer class
          #   to destroy the buffer when the refcount is 0 for non-external buffers.
          #
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/dawn/dawn_destroy_buffer_on_destructor.patch &&

          # The dawn_binskim.patch contains the following changes:
          #
          # - (private) Fulfill the BinSkim requirements
          #   Some build warnings are not allowed to be disabled in project level.
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/dawn/dawn_binskim.patch &&

          # The safari_polyfill.patch contains the following changes:
          #
          # - (private) Fix compatibility issues with Safari. Contains the following changes:
          #   - Polyfill for `device.AdapterInfo` (returns `undefined` in Safari v26.0)
          #
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/dawn/safari_polyfill.patch &&

          # The dawn_device_lost_keepalive.patch contains the following changes:
          #
          # - (private) Fix premature ABORT when device.lost fires in callUserCallback
          #   The device.lost handler was wrapped in callUserCallback without runtimeKeepalivePush/Pop,
          #   causing maybeExit() to trigger _exit(0) and set ABORT=true when runtimeKeepaliveCounter
          #   was 0. This silently dropped all subsequent WebGPU callbacks (e.g. requestAdapter),
          #   breaking session re-creation after device destruction.
          #
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/dawn/dawn_device_lost_keepalive.patch &&

          # The dawn_dxc_output_dir.patch contains the following changes:
          #
          # - (private) Fix DXC output directory for RelWithDebInfo and MinSizeRel configs
          #   Dawn only overrides the DXC output directory for Debug and Release configs. This causes
          #   build failures when using multi-config generators (like Visual Studio) with RelWithDebInfo
          #   because dxcompiler.dll ends up in the default output path instead of CMAKE_BINARY_DIR/$<CONFIG>.
          #
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/dawn/dawn_dxc_output_dir.patch &&

          # The dawn_parallel_build_fix.patch contains the following changes:
          #
          # - (private) Fix parallel build race condition in emdawnwebgpu header copy
          #   Two separate fixes address this race:
          #
          #   1. The emdawnwebgpu_headers_gen_add macro's add_custom_command uses cmake -E copy
          #      without ensuring the destination directory exists first. When building with
          #      parallel jobs (-j32), the copy commands for webgpu_glfw.h and
          #      webgpu_enum_class_bitmasks.h can run before any DawnJSONGenerator command
          #      has created gen/src/emdawnwebgpu/include/webgpu/, causing the copy to fail.
          #      This patch adds cmake -E make_directory before the copy so the directory is
          #      always present regardless of parallel build ordering.
          #
          #   2. webgpu_enum_class_bitmasks.h is listed in emdawnwebgpu_cpp's HEADERS, which
          #      causes CMake to add it to INTERFACE_SOURCES. When ORT targets link to
          #      emdawnwebgpu_cpp, CMake propagates this generated file to ORT's directory scope
          #      and generates a second copy of the cmake -E copy recipe for that file.
          #      With parallel make (-jN), both the Dawn-directory recipe and the ORT-directory
          #      recipe run concurrently for the same output file, causing the copy to fail.
          #      Removing webgpu_enum_class_bitmasks.h from emdawnwebgpu_cpp's HEADERS
          #      eliminates the duplicate recipe. The header remains accessible via the include
          #      directory set on emdawnwebgpu_c_include (${EM_BUILD_GEN_DIR}/include), and
          #      build ordering is preserved through the emdawnwebgpu_c -> emdawnwebgpu_c_include
          #      -> emdawnwebgpu_headers_gen dependency chain.
          #
          ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/dawn/dawn_parallel_build_fix.patch &&

          # Remove the test folder to speed up potential file scan operations (70k+ files not needed for build).
          # Using <SOURCE_DIR> token ensures the correct absolute path regardless of working directory.
          ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>/test)

      onnxruntime_fetchcontent_declare(
        dawn
        URL ${DEP_URL_dawn}
        URL_HASH SHA1=${DEP_SHA1_dawn}
        PATCH_COMMAND ${ONNXRUNTIME_Dawn_PATCH_COMMAND}
        EXCLUDE_FROM_ALL
      )
    endif()

    if (DAWN_USE_AGILITY_SDK)
      # Set Dawn's expected Chromium CIPD layout before configuring Dawn. The SDK
      # package is populated below, before any Dawn target is compiled.
      if (FETCHCONTENT_BASE_DIR)
        set(ONNXRUNTIME_DAWN_SRC_DIR "${FETCHCONTENT_BASE_DIR}/dawn-src")
      else()
        set(ONNXRUNTIME_DAWN_SRC_DIR "${CMAKE_BINARY_DIR}/_deps/dawn-src")
      endif()
      set(DAWN_AGILITY_SDK_DIR "${ONNXRUNTIME_DAWN_SRC_DIR}/third_party/agility-sdk"
          CACHE PATH "Directory containing the D3D12 Agility SDK" FORCE)
      message(STATUS "Dawn Agility SDK directory: ${DAWN_AGILITY_SDK_DIR}")
    endif()

    onnxruntime_fetchcontent_makeavailable(dawn)

    if (DAWN_USE_AGILITY_SDK)
      onnxruntime_fetchcontent_declare(
        dawn_agility_sdk
        URL ${DEP_URL_dawn_agility_sdk}
        URL_HASH SHA1=${DEP_SHA1_dawn_agility_sdk}
        SOURCE_DIR "${DAWN_AGILITY_SDK_DIR}/src"
        EXCLUDE_FROM_ALL
      )
      onnxruntime_fetchcontent_makeavailable(dawn_agility_sdk)
      if (NOT EXISTS "${DAWN_AGILITY_SDK_DIR}/src/build/native/include/d3d12.h")
        message(FATAL_ERROR
                "The Agility SDK package does not contain build/native/include/d3d12.h: "
                "${DAWN_AGILITY_SDK_DIR}/src")
      endif()
    endif()
  endif()
  endif()

  if (NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if (onnxruntime_BUILD_DAWN_SHARED_LIBRARY)
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES dawn::webgpu_dawn)
    else()
      if (NOT onnxruntime_USE_EXTERNAL_DAWN)
        list(APPEND onnxruntime_EXTERNAL_LIBRARIES dawn::dawn_native)
      endif()
      list(APPEND onnxruntime_EXTERNAL_LIBRARIES dawn::dawn_proc)
    endif()
  endif()

  if (onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP)
    list(APPEND onnxruntime_EXTERNAL_LIBRARIES webgpu_glfw glfw)
  endif()
endif()

if(onnxruntime_USE_COREML)
  # Setup coremltools fp16 and json dependencies for creating an mlpackage.
  #
  # fp16 depends on psimd
  onnxruntime_fetchcontent_declare(psimd URL ${DEP_URL_psimd} URL_HASH SHA1=${DEP_SHA1_psimd} EXCLUDE_FROM_ALL)
  onnxruntime_fetchcontent_makeavailable(psimd)
  set(PSIMD_SOURCE_DIR ${psimd_SOURCE_DIR})
  onnxruntime_fetchcontent_declare(fp16 URL ${DEP_URL_fp16} URL_HASH SHA1=${DEP_SHA1_fp16} EXCLUDE_FROM_ALL)
  set(FP16_BUILD_TESTS OFF CACHE INTERNAL "")
  set(FP16_BUILD_BENCHMARKS OFF CACHE INTERNAL "")
  onnxruntime_fetchcontent_makeavailable(fp16)

  onnxruntime_fetchcontent_declare(
    coremltools
    URL ${DEP_URL_coremltools}
    URL_HASH SHA1=${DEP_SHA1_coremltools}
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 < ${PROJECT_SOURCE_DIR}/patches/coremltools/crossplatformbuild.patch
    EXCLUDE_FROM_ALL
  )
  # we don't build directly so use Populate. selected files are built from onnxruntime_providers_coreml.cmake
  FetchContent_Populate(coremltools)

endif()

if(onnxruntime_USE_KLEIDIAI)
  # Disable the KleidiAI tests
  set(KLEIDIAI_BUILD_TESTS  OFF)

  onnxruntime_fetchcontent_declare(kleidiai URL ${DEP_URL_kleidiai} URL_HASH SHA1=${DEP_SHA1_kleidiai} EXCLUDE_FROM_ALL)
  onnxruntime_fetchcontent_makeavailable(kleidiai)
  # Fetch Qualcomm's kleidiai library
  if(onnxruntime_USE_QMX_KLEIDIAI_COEXIST)
          onnxruntime_fetchcontent_declare(kleidiai-qmx URL ${DEP_URL_kleidiai-qmx} URL_HASH SHA1=${DEP_SHA1_kleidiai-qmx}
                  EXCLUDE_FROM_ALL)
          onnxruntime_fetchcontent_makeavailable(kleidiai-qmx)
  endif()
endif()

set(onnxruntime_LINK_DIRS)
if (onnxruntime_USE_CUDA)
  # Work around a CMake limitation (present through at least CMake 3.31 and current
  # upstream master) when building natively on a Windows-on-ARM64 host. FindCUDAToolkit
  # only sets the Windows import-library search suffix when the host is x64:
  #
  #   if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows")
  #     if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "AMD64")
  #       set(_CUDAToolkit_win_search_dirs lib/x64)
  #       set(_CUDAToolkit_win_stub_search_dirs lib/x64/stubs)
  #
  # On an ARM64 host the suffix is left empty, so find_library() for cudart only looks in
  # "lib64" and never finds <cuda_home>/lib/.../cudart.lib. find_package(CUDAToolkit) then
  # fails with: Could NOT find CUDAToolkit (missing: CUDA_CUDART). Pre-seed the (internal)
  # search-suffix variables with win-arm64 import-library locations (lib/arm64 and
  # lib/arm64/stubs) so the toolkit's cudart.lib can be found. FindCUDAToolkit unsets
  # these at the end, so this only affects the search below and is a no-op once CMake
  # gains native WoA support.
  if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ARM64")
    set(_CUDAToolkit_win_search_dirs lib/arm64)
    set(_CUDAToolkit_win_stub_search_dirs lib/arm64/stubs)
  endif()

  find_package(CUDAToolkit REQUIRED)

  # cuDNN is not needed for minimal CUDA builds (e.g., TensorRT-only builds)
  if(NOT onnxruntime_CUDA_MINIMAL)
    if(onnxruntime_CUDNN_HOME)
      file(TO_CMAKE_PATH ${onnxruntime_CUDNN_HOME} onnxruntime_CUDNN_HOME)
      set(CUDNN_PATH ${onnxruntime_CUDNN_HOME})
    endif()

    include(cuDNN)
  endif()
endif()

if(onnxruntime_USE_SNPE)
  include(external/find_snpe.cmake)
  list(APPEND onnxruntime_EXTERNAL_LIBRARIES ${SNPE_NN_LIBS})
endif()

# 1DS SDK (cpp_client_telemetry) is the default cross-platform telemetry backend.
if(onnxruntime_USE_1DS_TELEMETRY)
  if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    message(FATAL_ERROR "onnxruntime_USE_1DS_TELEMETRY / telemetry is not supported for WebAssembly/Emscripten builds: "
                        "the 1DS telemetry SDK is excluded on Emscripten. Disable telemetry for WASM builds.")
  endif()
  # Use the pinned GitHub SDK on every platform, independently of the vcpkg registry version.
  set(_ort_requested_apple_architectures "${CMAKE_OSX_ARCHITECTURES}")

  # Android AARs require HttpClient_Android and its Java bridge.
  # Use cpp_client_telemetry's canonical build options. The SDK keeps its
  # build policy and dependency selection local to 1DS.
  set(MATSDK_BUILD_HEADERS ON CACHE BOOL "Build 1DS SDK headers" FORCE)
  set(MATSDK_BUILD_LIBRARY ON CACHE BOOL "Build 1DS SDK library" FORCE)
  set(MATSDK_BUILD_TEST_TOOL OFF CACHE BOOL "Disable 1DS SDK test tool" FORCE)
  set(MATSDK_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable 1DS SDK unit tests" FORCE)
  set(MATSDK_BUILD_FUNC_TESTS OFF CACHE BOOL "Disable 1DS SDK functional tests" FORCE)
  set(MATSDK_BUILD_PRIVACYGUARD OFF CACHE BOOL "Disable 1DS privacy guard module" FORCE)
  set(MATSDK_BUILD_CDS OFF CACHE BOOL "Disable 1DS CDS module" FORCE)
  set(MATSDK_BUILD_LIVEEVENTINSPECTOR OFF CACHE BOOL "Disable 1DS live event inspector" FORCE)
  set(MATSDK_BUILD_SIGNALS OFF CACHE BOOL "Disable 1DS signals module" FORCE)
  set(MATSDK_BUILD_SANITIZER OFF CACHE BOOL "Disable 1DS sanitizer module" FORCE)
  set(MATSDK_BUILD_AZMON OFF CACHE BOOL "Disable 1DS Azure Monitor module" FORCE)
  set(MATSDK_BUILD_OBJC_WRAPPER OFF CACHE BOOL "Disable 1DS ObjC wrapper" FORCE)
  set(MATSDK_BUILD_SWIFT_WRAPPER OFF CACHE BOOL "Disable 1DS Swift wrapper" FORCE)
  set(MATSDK_BUILD_JNI_WRAPPER OFF CACHE BOOL "Disable 1DS JNI wrapper" FORCE)
  set(MATSDK_BUILD_PACKAGE OFF CACHE BOOL "Disable 1DS package generation" FORCE)
  set(MATSDK_BUILD_APPLE_HTTP ${APPLE} CACHE BOOL "Build the 1DS Apple HTTP client" FORCE)
  set(MATSDK_DISABLE_LOGGING ON CACHE BOOL "Compile internal 1DS logging out" FORCE)
  set(MATSDK_DISABLE_EXCEPTIONS OFF CACHE BOOL "Retain 1DS SDK exception support" FORCE)
  set(MATSDK_USE_WININET OFF CACHE BOOL "Use WinHTTP for desktop Windows telemetry" FORCE)
  # Desktop ORT supplies a hashed, generated device ID; mobile builds use the SDK's platform ID.
  if(ANDROID OR CMAKE_SYSTEM_NAME STREQUAL "iOS" OR CMAKE_SYSTEM_NAME STREQUAL "visionOS")
    set(MATSDK_ENABLE_DEVICE_ID ON CACHE BOOL "Use the 1DS platform device ID on mobile" FORCE)
  else()
    set(MATSDK_ENABLE_DEVICE_ID OFF CACHE BOOL "Disable native 1DS device ID collection on desktop" FORCE)
  endif()
  set(MATSDK_ANDROID_USE_ROOM OFF CACHE BOOL "Use native SQLite for Android telemetry storage" FORCE)
  set(MATSDK_ENABLE_CAPI_HTTP_CLIENT OFF CACHE BOOL "Disable the alternative Android C API HTTP client" FORCE)
  set(MATSDK_CURL_TLS_BACKEND MBEDTLS CACHE STRING "Use mbedTLS for 1DS curl" FORCE)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND NOT onnxruntime_USE_VCPKG)
    set(MATSDK_CURL_PROVIDER FETCH CACHE STRING "Build the SDK's pinned curl with mbedTLS" FORCE)
  else()
    set(MATSDK_CURL_PROVIDER SYSTEM CACHE STRING "Use the platform HTTP transport" FORCE)
  endif()
  if(APPLE)
    set(MATSDK_SQLITE_PROVIDER SYSTEM CACHE STRING "Use Apple's system SQLite" FORCE)
    set(MATSDK_ZLIB_PROVIDER SYSTEM CACHE STRING "Use Apple's system libz" FORCE)
    # Supply the canonical system targets even when the SDK detects ORT's vcpkg toolchain.
    if(NOT TARGET SQLite3::SQLite3)
      add_library(SQLite3::SQLite3 INTERFACE IMPORTED)
      target_link_libraries(SQLite3::SQLite3 INTERFACE sqlite3)
    endif()
    if(NOT TARGET ZLIB::ZLIB)
      add_library(ZLIB::ZLIB INTERFACE IMPORTED)
      target_link_libraries(ZLIB::ZLIB INTERFACE z)
    endif()
  else()
    set(MATSDK_SQLITE_PROVIDER MINIMAL CACHE STRING "Build the SDK's minimal private SQLite" FORCE)
    set(MATSDK_ZLIB_PROVIDER VENDORED CACHE STRING "Build the SDK's private zlib" FORCE)
  endif()
  # BUILD_SHARED_LIBS is a global that ORT's own targets read after this block, and the SDK selects
  # mat's library type from it (lib/CMakeLists.txt). Save it, force static for the SDK, restore below.
  set(BUILD_SHARED_LIBS_SAVED "${BUILD_SHARED_LIBS}")
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build 1DS SDK as static library" FORCE)
  set(BUILD_SHARED_LIBS OFF)

  # Android uses the Java transport; all other source builds use the SDK's
  # canonical Apple/system or fetched mbedTLS transport selection.
  set(MATSDK_ANDROID_HTTP_CLIENT JAVA CACHE STRING "Use the 1DS Java HTTP bridge on Android" FORCE)

  onnxruntime_fetchcontent_declare(
    cpp_client_telemetry
    URL ${DEP_URL_cpp_client_telemetry}
    URL_HASH SHA1=${DEP_SHA1_cpp_client_telemetry}
    PATCH_COMMAND ${Patch_EXECUTABLE} --binary --ignore-whitespace -p1 <
      ${PROJECT_SOURCE_DIR}/patches/cpp_client_telemetry/factory-static-storage.patch
    EXCLUDE_FROM_ALL
  )
  block(PROPAGATE cpp_client_telemetry_SOURCE_DIR cpp_client_telemetry_BINARY_DIR)
    # The SDK otherwise derives its version from the build date, not the release tag.
    set(BUILD_VERSION "3.10.279.1")
    onnxruntime_fetchcontent_makeavailable(cpp_client_telemetry)
  endblock()
  if(WIN32 AND TARGET mat)
    # CMake's Visual Studio generator otherwise forwards mat's BUILD_INTERFACE system include
    # expression verbatim to MASM. ORT adds dependency includes to C/C++ targets separately.
    set_property(TARGET mat PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "")
  endif()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND TARGET libcurl_static AND NOT onnxruntime_BUILD_SHARED_LIB)
    # curl's imported mbedTLS helper is not exported with ORT's static package.
    get_target_property(_ort_curl_link_libraries libcurl_static INTERFACE_LINK_LIBRARIES)
    list(FILTER _ort_curl_link_libraries EXCLUDE REGEX "CURL::mbedtls")
    set_target_properties(libcurl_static PROPERTIES
      INTERFACE_LINK_LIBRARIES "${_ort_curl_link_libraries}")
    target_link_libraries(libcurl_static PRIVATE mbedtls)
  endif()
  if(ANDROID)
    target_compile_definitions(mat PRIVATE ANDROID_SUPPRESS_LOGCAT)
  endif()

  if(ANDROID)
    string(CONCAT _ort_android_telemetry_java_source_dir
        "${cpp_client_telemetry_SOURCE_DIR}/lib/android_build/maesdk/src/main/java/"
        "com/microsoft/applications/events")
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/android/telemetry-java")
    set(_ort_android_telemetry_java_dir
        "${CMAKE_BINARY_DIR}/android/telemetry-java/ai/onnxruntime/telemetry")
    file(MAKE_DIRECTORY "${_ort_android_telemetry_java_dir}")
    foreach(_ort_android_telemetry_java_file HttpClient.java HttpClientRequest.java)
      file(READ
           "${_ort_android_telemetry_java_source_dir}/${_ort_android_telemetry_java_file}"
           _ort_android_telemetry_java_source)
      string(REPLACE
             "package com.microsoft.applications.events;"
             "package ai.onnxruntime.telemetry;"
             _ort_android_telemetry_java_source
             "${_ort_android_telemetry_java_source}")
      file(WRITE
           "${_ort_android_telemetry_java_dir}/${_ort_android_telemetry_java_file}"
           "${_ort_android_telemetry_java_source}")
    endforeach()

    set(_ort_android_telemetry_resource_dir "${CMAKE_BINARY_DIR}/android/telemetry-resources/META-INF")
    file(MAKE_DIRECTORY "${_ort_android_telemetry_resource_dir}")
    configure_file(
      "${cpp_client_telemetry_SOURCE_DIR}/LICENSE"
      "${_ort_android_telemetry_resource_dir}/LICENSE-1DS"
      COPYONLY)
  endif()

  if(TARGET mat)
    if(TARGET sqlite3_bundled)
      # 1DS uses sqlite only for its narrow offline-event store. Keep the previous vcpkg
      # size reductions and extension-loading hardening on the bundled replacement.
      # Memory accounting must remain enabled for the SDK's SQLite soft heap limit.
      target_compile_definitions(sqlite3_bundled PRIVATE
        SQLITE_OMIT_LOAD_EXTENSION
        SQLITE_OMIT_DEPRECATED
        SQLITE_OMIT_UTF16
        SQLITE_OMIT_PROGRESS_CALLBACK
        SQLITE_OMIT_SHARED_CACHE
        SQLITE_OMIT_GET_TABLE
        SQLITE_OMIT_COMPLETE
        SQLITE_OMIT_TCL_VARIABLE
        SQLITE_DQS=0
        SQLITE_DEFAULT_FOREIGN_KEYS=0
      )
    endif()
    foreach(_ort_apple_dep mat sqlite3_bundled zlib_bundled)
      if(TARGET ${_ort_apple_dep})
        if(APPLE AND _ort_requested_apple_architectures)
          set_target_properties(${_ort_apple_dep} PROPERTIES
            OSX_ARCHITECTURES "${_ort_requested_apple_architectures}"
            XCODE_ATTRIBUTE_ARCHS "${_ort_requested_apple_architectures}")
        endif()
        get_target_property(_ort_apple_inc
          ${_ort_apple_dep} INTERFACE_INCLUDE_DIRECTORIES)
        if(_ort_apple_inc)
          # Preserve individual build expressions; wrapping a semicolon-separated list nests
          # expressions that ORT's MSVC /external:I handling cannot flatten.
          set(_ort_build_includes "")
          foreach(_ort_include IN LISTS _ort_apple_inc)
            if(_ort_include MATCHES "^\\$<BUILD_INTERFACE:")
              list(APPEND _ort_build_includes "${_ort_include}")
            elseif(NOT _ort_include MATCHES "^\\$<INSTALL_INTERFACE:")
              list(APPEND _ort_build_includes "$<BUILD_INTERFACE:${_ort_include}>")
            endif()
          endforeach()
          set_target_properties(${_ort_apple_dep} PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${_ort_build_includes}")
        endif()
      endif()
    endforeach()
    # ORT enables -ffast-math globally, which conflicts with
    # std::numeric_limits<double>::infinity() in the 1DS SDK's bundled nlohmann/json.hpp.
    # Also suppress warnings in the 1DS SDK code that ORT treats as errors.
    if(NOT MSVC)
      target_compile_options(mat PRIVATE
        -fno-finite-math-only
        -Wno-unused-const-variable
        $<$<CXX_COMPILER_ID:GNU>:-Wno-reorder>
        $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wno-reorder-ctor>
      )
    endif()
    # Vendored 1DS dependencies emit unavoidable narrowing warnings under Apple's warning policy.
    # Keep the warning enabled for ORT sources while suppressing it only for third-party targets.
    if(APPLE)
      foreach(_ort_mat_tgt mat sqlite3_bundled zlib_bundled)
        if(TARGET ${_ort_mat_tgt})
          target_compile_options(${_ort_mat_tgt} PRIVATE -Wno-shorten-64-to-32)
        endif()
      endforeach()
      if(TARGET sqlite3_bundled)
        target_compile_options(sqlite3_bundled PRIVATE -Wno-ambiguous-macro)
      endif()
    endif()
    if(TARGET sqlite3_bundled
       AND CMAKE_C_COMPILER_ID STREQUAL "GNU"
       AND CMAKE_C_COMPILER_VERSION VERSION_GREATER_EQUAL 11)
      target_compile_options(sqlite3_bundled PRIVATE -Wno-error=stringop-overread)
    endif()
  endif()

  foreach(_ort_1ds_cache_var
      MATSDK_BUILD_HEADERS
      MATSDK_BUILD_LIBRARY
      MATSDK_BUILD_TEST_TOOL
      MATSDK_BUILD_UNIT_TESTS
      MATSDK_BUILD_FUNC_TESTS
      MATSDK_BUILD_PRIVACYGUARD
      MATSDK_BUILD_CDS
      MATSDK_BUILD_LIVEEVENTINSPECTOR
      MATSDK_BUILD_SIGNALS
      MATSDK_BUILD_SANITIZER
      MATSDK_BUILD_AZMON
      MATSDK_BUILD_OBJC_WRAPPER
      MATSDK_BUILD_SWIFT_WRAPPER
      MATSDK_BUILD_JNI_WRAPPER
      MATSDK_BUILD_PACKAGE
      MATSDK_BUILD_APPLE_HTTP
      MATSDK_DISABLE_LOGGING
      MATSDK_DISABLE_EXCEPTIONS
      MATSDK_USE_WININET
      MATSDK_ENABLE_DEVICE_ID
      MATSDK_ANDROID_USE_ROOM
      MATSDK_ENABLE_CAPI_HTTP_CLIENT
      MATSDK_ANDROID_HTTP_CLIENT
      MATSDK_CURL_PROVIDER
      MATSDK_CURL_TLS_BACKEND
      MATSDK_SQLITE_PROVIDER
      MATSDK_ZLIB_PROVIDER)
    unset(${_ort_1ds_cache_var} CACHE)
  endforeach()

  set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}" CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS "${BUILD_SHARED_LIBS_SAVED}")
endif()

FILE(TO_NATIVE_PATH ${CMAKE_BINARY_DIR} ORT_BINARY_DIR)
FILE(TO_NATIVE_PATH ${PROJECT_SOURCE_DIR} ORT_SOURCE_DIR)

message(STATUS "Finished fetching external dependencies")

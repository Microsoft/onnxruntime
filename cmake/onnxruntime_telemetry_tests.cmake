# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

if(NOT onnxruntime_USE_1DS_TELEMETRY OR CMAKE_CROSSCOMPILING OR ANDROID OR IOS
    OR GDK_PLATFORM OR CMAKE_SYSTEM_NAME STREQUAL "WindowsStore")
  return()
endif()

# Recompile only the SDK's debug source and worker with test-only observers. Explicit
# objects resolve these symbols before libmat's originals; production targets are unchanged.
# The mutex observer makes libc++'s nontrivial mutex lifetime visible on GCC/MSVC too.
set(telemetry_test_dir "${TEST_SRC_DIR}/platform/telemetry_1ds")
set(telemetry_sdk_test_dir "${CMAKE_CURRENT_BINARY_DIR}/telemetry_shutdown_sdk")
file(MAKE_DIRECTORY "${telemetry_sdk_test_dir}")
foreach(sdk_source callbacks/DebugSource.cpp pal/WorkerThread.cpp)
  set(sdk_source_path "${cpp_client_telemetry_SOURCE_DIR}/lib/${sdk_source}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${sdk_source_path}")
  file(READ "${sdk_source_path}" sdk_test_source)
  if(sdk_source STREQUAL "callbacks/DebugSource.cpp")
    string(REGEX MATCHALL "DE_LOCKGUARD\\(stateLock\\(\\)\\)" lock_sites "${sdk_test_source}")
    list(LENGTH lock_sites lock_site_count)
    if(NOT lock_site_count EQUAL 5)
      message(FATAL_ERROR "Update the telemetry regression's SDK debug-lock instrumentation")
    endif()
    string(REPLACE "DE_LOCKGUARD(stateLock());"
      "OrtTelemetryBeforeDebugLock();\n        (void)stateLock();\n        OrtTelemetryDebugLockInitialized();\n        DE_LOCKGUARD(stateLock());"
      sdk_test_source "${sdk_test_source}")
    string(PREPEND sdk_test_source
      "extern \"C\" void OrtTelemetryBeforeDebugLock();\nextern \"C\" void OrtTelemetryDebugLockInitialized();\n")
  else()
    string(REGEX MATCHALL "threadToJoin\\.join\\(\\)" join_sites "${sdk_test_source}")
    list(LENGTH join_sites join_site_count)
    if(NOT join_site_count EQUAL 2)
      message(FATAL_ERROR "Update the telemetry regression's SDK worker-join instrumentation")
    endif()
    string(REPLACE "threadToJoin.join();"
      "threadToJoin.join();\n                    OrtTelemetryWorkerJoined(this);"
      sdk_test_source "${sdk_test_source}")
    string(PREPEND sdk_test_source "extern \"C\" void OrtTelemetryWorkerJoined(void*);\n")
  endif()
  get_filename_component(sdk_test_name "${sdk_source}" NAME)
  file(WRITE "${telemetry_sdk_test_dir}/${sdk_test_name}" "${sdk_test_source}")
endforeach()

add_library(onnxruntime_telemetry_sdk_test OBJECT
  "${telemetry_sdk_test_dir}/DebugSource.cpp"
  "${telemetry_sdk_test_dir}/WorkerThread.cpp")
target_include_directories(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES>)
target_compile_definitions(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
target_compile_options(onnxruntime_telemetry_sdk_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_OPTIONS>)
target_link_libraries(onnxruntime_telemetry_sdk_test PRIVATE mat)

onnxruntime_add_executable(onnxruntime_telemetry_mutex_lifetime_test
  "${telemetry_test_dir}/telemetry_shutdown_test.cc"
  $<TARGET_OBJECTS:onnxruntime_telemetry_sdk_test>)
target_include_directories(onnxruntime_telemetry_mutex_lifetime_test PRIVATE ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR}
  $<TARGET_PROPERTY:onnxruntime_common,INCLUDE_DIRECTORIES>
  "${cpp_client_telemetry_SOURCE_DIR}/lib" "${cpp_client_telemetry_SOURCE_DIR}/lib/include"
  "${cpp_client_telemetry_SOURCE_DIR}/lib/include/mat")
target_compile_definitions(onnxruntime_telemetry_mutex_lifetime_test PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
target_link_libraries(onnxruntime_telemetry_mutex_lifetime_test PRIVATE onnxruntime_common mat ${CMAKE_DL_LIBS} Threads::Threads)
if(CPUINFO_SUPPORTED)
  target_link_libraries(onnxruntime_telemetry_mutex_lifetime_test PRIVATE cpuinfo)
endif()
set_target_properties(onnxruntime_telemetry_sdk_test onnxruntime_telemetry_mutex_lifetime_test
  PROPERTIES FOLDER "ONNXRuntimeTest")

# Ordinary test mains suppress telemetry. Each mode gets a private cache,
# clears CI suppression, and dispatches only an internal diagnostic, never an upload event.
foreach(mode enabled threaded disabled disabled-threaded)
  set(test_home "${CMAKE_CURRENT_BINARY_DIR}/telemetry_shutdown_test_${mode}")
  file(MAKE_DIRECTORY "${test_home}")
  set(disable_telemetry 0)
  if(mode MATCHES "^disabled")
    set(disable_telemetry 1)
  endif()
  add_test(NAME onnxruntime_telemetry_shutdown_${mode}_test
    COMMAND ${CMAKE_COMMAND} -E env
      "HOME=${test_home}" "XDG_CACHE_HOME=${test_home}" "LOCALAPPDATA=${test_home}" "APPDATA=${test_home}"
      CI=0 TF_BUILD=0 GITHUB_ACTIONS=0 GITLAB_CI=0 CIRCLECI=0 TRAVIS=0
      JENKINS_URL=0 CODEBUILD_BUILD_ID=0 BUILDKITE=0 TEAMCITY_VERSION=0
      APPVEYOR=0 BITBUCKET_BUILD_NUMBER=0 SYSTEM_TEAMFOUNDATIONCOLLECTIONURI=0
      ORT_RUNNING_UNIT_TESTS=0 "ORT_DISABLE_TELEMETRY=${disable_telemetry}"
      $<TARGET_FILE:onnxruntime_telemetry_mutex_lifetime_test> ${mode})
  set_tests_properties(onnxruntime_telemetry_shutdown_${mode}_test PROPERTIES TIMEOUT 30)
endforeach()

# Only the fixture's provider gets configuration/observation hooks. The shipped
# provider has no collector override or way to bypass CI suppression.
set(provider_source "${ONNXRUNTIME_ROOT}/core/platform/telemetry_1ds.cc")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${provider_source}")
file(READ "${provider_source}" transport_provider)
foreach(hook
    "  status_t status;"
    "  pending_log_manager.Commit(config_, log_manager_);")
  string(FIND "${transport_provider}" "${hook}" hook_position)
  string(FIND "${transport_provider}" "${hook}" last_hook_position REVERSE)
  if(hook_position EQUAL -1 OR NOT hook_position EQUAL last_hook_position)
    message(FATAL_ERROR "Update the telemetry transport fixture's provider instrumentation")
  endif()
endforeach()
string(REPLACE "  status_t status;"
  "  OrtTelemetryTestConfigure(config);\n  status_t status;"
  transport_provider "${transport_provider}")
string(REPLACE "  pending_log_manager.Commit(config_, log_manager_);"
  "  pending_log_manager.Commit(config_, log_manager_);\n  OrtTelemetryTestReady(*log_manager_);"
  transport_provider "${transport_provider}")
string(PREPEND transport_provider
  "#include <ILogConfiguration.hpp>\n#include <ILogManager.hpp>\nnamespace onnxruntime {\nvoid OrtTelemetryTestConfigure(Microsoft::Applications::Events::ILogConfiguration&);\nvoid OrtTelemetryTestReady(Microsoft::Applications::Events::ILogManager&);\n}\n")
file(WRITE "${telemetry_sdk_test_dir}/telemetry_transport_provider.cc" "${transport_provider}")
target_sources(onnxruntime_telemetry_mutex_lifetime_test PRIVATE
  "${telemetry_sdk_test_dir}/telemetry_transport_provider.cc")

set(transport_sources
  "${telemetry_test_dir}/telemetry_transport_test.cc"
  "${telemetry_sdk_test_dir}/telemetry_transport_provider.cc")
if(WIN32)
  # Automatic proxy discovery could send even a loopback test outside the machine.
  # Keep the real WinHTTP implementation, changing only its fixture proxy policy.
  set(winhttp_source "${cpp_client_telemetry_SOURCE_DIR}/lib/http/HttpClient_WinHttp.cpp")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${winhttp_source}")
  file(READ "${winhttp_source}" winhttp_test_source)
  string(FIND "${winhttp_test_source}" "WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY" proxy_position)
  string(FIND "${winhttp_test_source}" "WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY" last_proxy_position REVERSE)
  if(proxy_position EQUAL -1 OR NOT proxy_position EQUAL last_proxy_position)
    message(FATAL_ERROR "Update the telemetry transport fixture's WinHTTP proxy policy")
  endif()
  string(REPLACE "WINHTTP_ACCESS_TYPE_AUTOMATIC_PROXY" "WINHTTP_ACCESS_TYPE_NO_PROXY"
    winhttp_test_source "${winhttp_test_source}")
  file(WRITE "${telemetry_sdk_test_dir}/HttpClient_WinHttp.cpp" "${winhttp_test_source}")
  list(APPEND transport_sources "${telemetry_sdk_test_dir}/HttpClient_WinHttp.cpp")
endif()

onnxruntime_add_executable(onnxruntime_telemetry_transport_test ${transport_sources})
onnxruntime_add_shared_library(onnxruntime_telemetry_transport_fixture ${transport_sources})
target_compile_definitions(onnxruntime_telemetry_transport_fixture PRIVATE ORT_TELEMETRY_TRANSPORT_LIBRARY)
foreach(target onnxruntime_telemetry_transport_test onnxruntime_telemetry_transport_fixture)
  target_include_directories(${target} PRIVATE
    ${ONNXRUNTIME_ROOT} ${CMAKE_CURRENT_BINARY_DIR}
    $<TARGET_PROPERTY:onnxruntime_common,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:mat,INCLUDE_DIRECTORIES>
    "${cpp_client_telemetry_SOURCE_DIR}/lib"
    "${cpp_client_telemetry_SOURCE_DIR}/lib/http"
    "${cpp_client_telemetry_SOURCE_DIR}/lib/include"
    "${cpp_client_telemetry_SOURCE_DIR}/lib/include/mat")
  target_compile_definitions(${target} PRIVATE $<TARGET_PROPERTY:mat,COMPILE_DEFINITIONS>)
  target_compile_options(${target} PRIVATE $<TARGET_PROPERTY:mat,COMPILE_OPTIONS>)
  target_link_libraries(${target} PRIVATE onnxruntime_common mat ${CMAKE_DL_LIBS} Threads::Threads)
  if(CPUINFO_SUPPORTED)
    target_link_libraries(${target} PRIVATE cpuinfo)
  endif()
  set_target_properties(${target} PROPERTIES FOLDER "ONNXRuntimeTest")
endforeach()
if(WIN32)
  target_link_libraries(onnxruntime_telemetry_transport_test PRIVATE winhttp)
  target_link_libraries(onnxruntime_telemetry_transport_fixture PRIVATE winhttp)
elseif(UNIX AND NOT APPLE)
  # Keep the fixture's SDK/Env state private and unloadable (GNU-unique symbols
  # otherwise keep a dlopen'ed image resident after dlclose).
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(onnxruntime_telemetry_transport_fixture PRIVATE -fno-gnu-unique)
  endif()
  target_link_options(onnxruntime_telemetry_transport_fixture PRIVATE "-Wl,--exclude-libs,ALL" "-Wl,-Bsymbolic")
endif()

add_test(NAME onnxruntime_telemetry_transport_runner_test
  COMMAND "${Python_EXECUTABLE}" "${telemetry_test_dir}/telemetry-transport-runner-test.py")
set_tests_properties(onnxruntime_telemetry_transport_runner_test PROPERTIES TIMEOUT 10)

set(telemetry_transport_modes upload threaded retry pending disabled unit-disabled ci-disabled)
# The Objective-C runtime retains the Apple transport's image after dlclose, so
# loading it again cannot provide the fresh static state required by this test.
if(NOT APPLE)
  list(APPEND telemetry_transport_modes reload)
endif()
foreach(mode IN LISTS telemetry_transport_modes)
  add_test(NAME onnxruntime_telemetry_transport_${mode}_test
    COMMAND "${Python_EXECUTABLE}" "${telemetry_test_dir}/telemetry-transport-test.py"
      --executable $<TARGET_FILE:onnxruntime_telemetry_transport_test>
      --library $<TARGET_FILE:onnxruntime_telemetry_transport_fixture> --mode ${mode})
  set_tests_properties(onnxruntime_telemetry_transport_${mode}_test PROPERTIES TIMEOUT 45)
endforeach()

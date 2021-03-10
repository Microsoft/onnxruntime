# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

include(pybind11)
FIND_PACKAGE(NumPy)

if(NOT PYTHON_INCLUDE_DIR)
  set(PYTHON_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import distutils.sysconfig; print(distutils.sysconfig.get_python_inc())\""
    OUTPUT_VARIABLE PYTHON_INCLUDE_DIR
    RETURN_VALUE PYTHON_NOT_FOUND)
  if(${PYTHON_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get Python include directory. Is distutils installed?")
  endif(${PYTHON_NOT_FOUND})
endif(NOT PYTHON_INCLUDE_DIR)

# 2. Resolve the installed version of NumPy (for numpy/arrayobject.h).
if(NOT NUMPY_INCLUDE_DIR)
  set(NUMPY_NOT_FOUND false)
  exec_program("${PYTHON_EXECUTABLE}"
    ARGS "-c \"import numpy; print(numpy.get_include())\""
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    RETURN_VALUE NUMPY_NOT_FOUND)
  if(${NUMPY_NOT_FOUND})
    message(FATAL_ERROR
            "Cannot get NumPy include directory: Is NumPy installed?")
  endif(${NUMPY_NOT_FOUND})
endif(NOT NUMPY_INCLUDE_DIR)


# ---[ Python + Numpy
set(onnxruntime_pybind_srcs_pattern
    "${ONNXRUNTIME_ROOT}/python/*.cc"
    "${ONNXRUNTIME_ROOT}/python/*.h"
)

if (onnxruntime_ENABLE_TRAINING)
  list(APPEND onnxruntime_pybind_srcs_pattern
    "${ORTTRAINING_ROOT}/orttraining/python/*.cc"
    "${ORTTRAINING_ROOT}/orttraining/python/*.h"
  )
endif()

file(GLOB onnxruntime_pybind_srcs CONFIGURE_DEPENDS
  ${onnxruntime_pybind_srcs_pattern}
  )

onnxruntime_add_shared_library_module(onnxruntime_pybind11_state ${onnxruntime_pybind_srcs})
if(MSVC)
  target_compile_options(onnxruntime_pybind11_state PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--compiler-options /utf-8>" "$<$<NOT:$<COMPILE_LANGUAGE:CUDA>>:/utf-8>")
endif()
if(HAS_CAST_FUNCTION_TYPE)
  target_compile_options(onnxruntime_pybind11_state PRIVATE "-Wno-cast-function-type")
endif()

if(onnxruntime_PYBIND_EXPORT_OPSCHEMA)
  target_compile_definitions(onnxruntime_pybind11_state PRIVATE onnxruntime_PYBIND_EXPORT_OPSCHEMA)
endif()

if (onnxruntime_USE_DNNL)
  target_compile_definitions(onnxruntime_pybind11_state PRIVATE USE_DNNL=1)
endif()
if (MSVC AND NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    #TODO: fix the warnings
    target_compile_options(onnxruntime_pybind11_state PRIVATE "/wd4244")
endif()
target_include_directories(onnxruntime_pybind11_state PRIVATE ${ONNXRUNTIME_ROOT} ${PYTHON_INCLUDE_DIR} ${NUMPY_INCLUDE_DIR} ${pybind11_INCLUDE_DIRS})
if(onnxruntime_USE_CUDA)
    target_include_directories(onnxruntime_pybind11_state PRIVATE ${onnxruntime_CUDNN_HOME}/include)
endif()
if (onnxruntime_ENABLE_TRAINING)
  target_include_directories(onnxruntime_pybind11_state PRIVATE ${ORTTRAINING_ROOT})
endif()

if(APPLE)
  set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker -exported_symbols_list ${ONNXRUNTIME_ROOT}/python/exported_symbols.lst")
elseif(UNIX)
  set(ONNXRUNTIME_SO_LINK_FLAG "-Xlinker --version-script=${ONNXRUNTIME_ROOT}/python/version_script.lds -Xlinker --gc-sections")
else()
  set(ONNXRUNTIME_SO_LINK_FLAG "-DEF:${ONNXRUNTIME_ROOT}/python/pybind.def")
endif()

set(onnxruntime_pybind11_state_libs
    onnxruntime_session
    ${onnxruntime_libs}
    ${PROVIDERS_CUDA}
    ${PROVIDERS_MIGRAPHX}
    ${PROVIDERS_NUPHAR}
    ${PROVIDERS_VITISAI}
    ${PROVIDERS_NNAPI}
    ${PROVIDERS_RKNPU}
    ${PROVIDERS_DML}
    ${PROVIDERS_ACL}
    ${PROVIDERS_ARMNN}
    ${PROVIDERS_ROCM}
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_util
    ${onnxruntime_tvm_libs}
    onnxruntime_framework
    onnxruntime_util
    onnxruntime_graph
    onnxruntime_common
    onnxruntime_mlas
    onnxruntime_flatbuffers
    ${pybind11_lib}
)

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  list(APPEND onnxruntime_pybind11_state_libs onnxruntime_language_interop onnxruntime_pyop)
endif()

if (onnxruntime_ENABLE_TRAINING)
  list(INSERT onnxruntime_pybind11_state_libs 1 onnxruntime_training)
endif()

set(onnxruntime_pybind11_state_dependencies
    ${onnxruntime_EXTERNAL_DEPENDENCIES}
    ${pybind11_dep}
)
set_property(TARGET onnxruntime_pybind11_state APPEND_STRING PROPERTY LINK_FLAGS ${ONNXRUNTIME_SO_LINK_FLAG} ${onnxruntime_DELAYLOAD_FLAGS})
add_dependencies(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_dependencies})

if (MSVC)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES LINK_FLAGS "${ONNXRUNTIME_SO_LINK_FLAG}")
  # if MSVC, pybind11 looks for release version of python lib (pybind11/detail/common.h undefs _DEBUG)
  target_link_libraries(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_libs}
          ${PYTHON_LIBRARY_RELEASE} ${onnxruntime_EXTERNAL_LIBRARIES})
elseif (APPLE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES LINK_FLAGS "${ONNXRUNTIME_SO_LINK_FLAG} -undefined dynamic_lookup")
  target_link_libraries(onnxruntime_pybind11_state ${onnxruntime_pybind11_state_libs} ${onnxruntime_EXTERNAL_LIBRARIES})
  set_target_properties(onnxruntime_pybind11_state PROPERTIES
    INSTALL_RPATH "@loader_path"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH FALSE)
else()
  target_link_libraries(onnxruntime_pybind11_state PRIVATE ${onnxruntime_pybind11_state_libs} ${onnxruntime_EXTERNAL_LIBRARIES})
  set_property(TARGET onnxruntime_pybind11_state APPEND_STRING PROPERTY LINK_FLAGS " -Xlinker -rpath=\$ORIGIN")
endif()

set_target_properties(onnxruntime_pybind11_state PROPERTIES PREFIX "")
set_target_properties(onnxruntime_pybind11_state PROPERTIES FOLDER "ONNXRuntime")
if(onnxruntime_ENABLE_LTO)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO TRUE)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL TRUE)
endif()
if (MSVC)
  set_target_properties(onnxruntime_pybind11_state PROPERTIES SUFFIX ".pyd")
else()
  set_target_properties(onnxruntime_pybind11_state PROPERTIES SUFFIX ".so")
endif()

# Generate version_info.py in Windows build.
# Has to be done before onnxruntime_python_srcs is set.
if (WIN32)
  set(VERSION_INFO_FILE "${ONNXRUNTIME_ROOT}/python/version_info.py")

  if (onnxruntime_USE_CUDA)
    file(WRITE "${VERSION_INFO_FILE}" "use_cuda = True\n")

    file(GLOB CUDNN_DLL_PATH "${onnxruntime_CUDNN_HOME}/bin/cudnn64_*.dll")
    if (NOT CUDNN_DLL_PATH)
      message(FATAL_ERROR "cuDNN not found in ${onnxruntime_CUDNN_HOME}")
    endif()
    get_filename_component(CUDNN_DLL_NAME ${CUDNN_DLL_PATH} NAME_WE)
    string(REPLACE "cudnn64_" "" CUDNN_VERSION "${CUDNN_DLL_NAME}")

    file(APPEND "${VERSION_INFO_FILE}"
      "cuda_version = \"${onnxruntime_CUDA_VERSION}\"\n"
      "cudnn_version = \"${CUDNN_VERSION}\"\n"
    )
  else()
    file(WRITE "${VERSION_INFO_FILE}" "use_cuda = False\n")
  endif()

  if ("${MSVC_TOOLSET_VERSION}" STREQUAL "142")
    file(APPEND "${VERSION_INFO_FILE}" "vs2019 = True\n")
  else()
    file(APPEND "${VERSION_INFO_FILE}" "vs2019 = False\n")
  endif()
endif()

file(GLOB onnxruntime_backend_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/backend/*.py"
)

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB onnxruntime_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/*.py"
    "${ORTTRAINING_SOURCE_DIR}/python/*.py"
  )
else()
  file(GLOB onnxruntime_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/*.py"
  )
endif()

if (onnxruntime_ENABLE_TRAINING)
  file(GLOB onnxruntime_python_capi_training_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/deprecated/*.py"
  )
  file(GLOB onnxruntime_python_root_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/*.py"
  )
  file(GLOB onnxruntime_python_amp_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/amp/*.py"
  )
  file(GLOB onnxruntime_python_optim_srcs CONFIGURE_DEPENDS
    "${ORTTRAINING_SOURCE_DIR}/python/training/optim/*.py"
  )
  file(GLOB onnxruntime_python_train_tools_srcs CONFIGURE_DEPENDS
    "${REPO_ROOT}/tools/python/register_custom_ops_pytorch_exporter.py"
  )
else()
  file(GLOB onnxruntime_python_capi_training_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/training/*.py"
  )
endif()

if (onnxruntime_BUILD_UNIT_TESTS)
  file(GLOB onnxruntime_python_test_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/*.py"
      "${ORTTRAINING_SOURCE_DIR}/test/python/*.py"
      "${ORTTRAINING_SOURCE_DIR}/test/python/*.json"
  )
  file(GLOB onnxruntime_python_quantization_test_srcs CONFIGURE_DEPENDS
      "${ONNXRUNTIME_ROOT}/test/python/quantization/*.py"
  )
  file(GLOB onnxruntime_python_checkpoint_test_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/python/checkpoint/*.py"
  )
  file(GLOB onnxruntime_python_dhp_parallel_test_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/python/dhp_parallel/*.py"
  )
  file(GLOB onnxruntime_python_perf_log_srcs CONFIGURE_DEPENDS
      "${ORTTRAINING_SOURCE_DIR}/test/python/perf_log/*.py"
  )
endif()

file(GLOB onnxruntime_python_tools_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/*.py"
)
file(GLOB onnxruntime_python_tools_featurizers_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/featurizer_ops/*.py"
)
file(GLOB onnxruntime_python_quantization_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/quantization/*.py"
)
file(GLOB onnxruntime_python_quantization_operators_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/quantization/operators/*.py"
)
file(GLOB onnxruntime_python_transformers_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/*.py"
)
file(GLOB onnxruntime_python_transformers_longformer_src CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/tools/transformers/longformer/*.py"
)
file(GLOB onnxruntime_python_datasets_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/datasets/*.py"
)
file(GLOB onnxruntime_python_datasets_data CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/python/datasets/*.pb"
    "${ONNXRUNTIME_ROOT}/python/datasets/*.onnx"
)

set(build_output_target onnxruntime_common)

add_custom_command(
  TARGET onnxruntime_pybind11_state POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/backend
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/training
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/datasets
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/featurizer_ops
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/longformer
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/operators
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/checkpoint
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/dhp_parallel
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/perf_log
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/quantization
  COMMAND ${CMAKE_COMMAND} -E copy
      ${ONNXRUNTIME_ROOT}/__init__.py
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/ThirdPartyNotices.txt
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/docs/Privacy.md
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/LICENSE
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_backend_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/backend/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_capi_training_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/training/
  COMMAND ${CMAKE_COMMAND} -E copy
      $<TARGET_FILE:onnxruntime_pybind11_state>
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_datasets_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_datasets_data}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/datasets/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_tools_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_tools_featurizers_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/tools/featurizer_ops/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_quantization_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_quantization_operators_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/quantization/operators/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_transformers_longformer_src}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/transformers/longformer/
  COMMAND ${CMAKE_COMMAND} -E copy
      ${REPO_ROOT}/VERSION_NUMBER
      $<TARGET_FILE_DIR:${build_output_target}>
)

if (onnxruntime_BUILD_UNIT_TESTS)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_quantization_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/quantization/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_checkpoint_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/checkpoint/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_dhp_parallel_test_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/dhp_parallel/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_perf_log_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/perf_log/
  )
endif()

if (onnxruntime_ENABLE_TRAINING)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/amp
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/optim
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_capi_training_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/training/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_root_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_amp_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/amp/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_optim_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/optim/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${onnxruntime_python_train_tools_srcs}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/training/
  )
endif()

if (onnxruntime_USE_DNNL)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${DNNL_DLL_PATH} $<TARGET_FILE:onnxruntime_providers_dnnl>
        $<TARGET_FILE:onnxruntime_providers_shared>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_TENSORRT)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_tensorrt>
        $<TARGET_FILE:onnxruntime_providers_shared>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_OPENVINO)
    add_custom_command(
      TARGET onnxruntime_pybind11_state POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy
          $<TARGET_FILE:onnxruntime_providers_openvino>
          $<TARGET_FILE:onnxruntime_providers_shared>
          $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
    )
endif()

if (onnxruntime_USE_TVM)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:tvm> $<TARGET_FILE:nnvm_compiler>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_NUPHAR)
  file(GLOB onnxruntime_python_nuphar_python_srcs CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/nuphar/scripts/*"
  )
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/nuphar
    COMMAND ${CMAKE_COMMAND} -E copy
      ${onnxruntime_python_nuphar_python_srcs}
      $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/nuphar/
  )
endif()

if (onnxruntime_USE_DML)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${DML_PACKAGE_DIR}/bin/${onnxruntime_target_platform}-win/${DML_SHARED_LIB}
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_USE_NNAPI_BUILTIN)
  add_custom_command(
    TARGET onnxruntime_pybind11_state POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:onnxruntime_providers_nnapi>
        $<TARGET_FILE_DIR:${build_output_target}>/onnxruntime/capi/
  )
endif()

if (onnxruntime_ENABLE_LANGUAGE_INTEROP_OPS)
  include(onnxruntime_language_interop_ops.cmake)
endif()

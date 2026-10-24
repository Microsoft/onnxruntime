// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <absl/base/config.h>
#include <gsl/gsl>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/graph/constants.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/plugin_ep_stream.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_lite_custom_op.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/util/thread_utils.h"

#include "test/shared_lib/custom_op_utils.h"
#include "test/shared_lib/test_fixture.h"
#include "test/shared_lib/utils.h"
#include "test/util/include/providers.h"
#include "test/util/include/test_allocator.h"

#include "onnxruntime_config.h"  // generated file in build output dir

#ifdef _WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#include <cuda_runtime.h>
#endif

#ifdef USE_DML
#include <wrl/client.h>
#include <wil/result.h>
#include <DirectML.h>
#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"

using Microsoft::WRL::ComPtr;
#endif

// Once we use C++17 this could be replaced with std::size
template <typename T, size_t N>
constexpr size_t countof(T (&)[N]) { return N; }

extern std::unique_ptr<Ort::Env> ort_env;

#ifdef USE_DML
struct DmlObjects {
  ComPtr<ID3D12Device> d3d12_device;
  ComPtr<ID3D12CommandQueue> command_queue;
  ComPtr<ID3D12CommandAllocator> command_allocator;
  ComPtr<ID3D12GraphicsCommandList> command_list;
  ComPtr<ID3D12Resource> upload_buffer;
  ComPtr<IDMLDevice> dml_device;
};

static DmlObjects CreateDmlObjects() {
  D3D12_COMMAND_QUEUE_DESC command_queue_description =
      {
          D3D12_COMMAND_LIST_TYPE_DIRECT,
          0,
          D3D12_COMMAND_QUEUE_FLAG_NONE,
          0,
      };

  DmlObjects dml_objects;

  THROW_IF_FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dml_objects.d3d12_device)));
  THROW_IF_FAILED(DMLCreateDevice(dml_objects.d3d12_device.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dml_objects.dml_device)));
  THROW_IF_FAILED(dml_objects.d3d12_device->CreateCommandQueue(&command_queue_description, IID_PPV_ARGS(&dml_objects.command_queue)));
  THROW_IF_FAILED(dml_objects.d3d12_device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&dml_objects.command_allocator)));
  THROW_IF_FAILED(dml_objects.d3d12_device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, dml_objects.command_allocator.Get(), nullptr, IID_PPV_ARGS(&dml_objects.command_list)));

  return dml_objects;
}

static ComPtr<ID3D12Resource> CreateD3D12ResourceOfByteSize(
    ID3D12Device* d3d_device,
    size_t resource_byte_size,
    D3D12_HEAP_TYPE heap_type = D3D12_HEAP_TYPE_DEFAULT,
    D3D12_RESOURCE_STATES resource_state = D3D12_RESOURCE_STATE_COMMON,
    D3D12_RESOURCE_FLAGS resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
  resource_byte_size = std::max(resource_byte_size, size_t(DML_MINIMUM_BUFFER_TENSOR_ALIGNMENT));

  // DML needs the resources' sizes to be a multiple of 4 bytes
  (resource_byte_size += 3) &= ~3;

  D3D12_HEAP_PROPERTIES heap_properties = {};
  heap_properties.Type = heap_type;
  heap_properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heap_properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heap_properties.CreationNodeMask = 1;
  heap_properties.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC resource_desc = {};
  resource_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  resource_desc.Alignment = 0;
  resource_desc.Width = static_cast<uint64_t>(resource_byte_size);
  resource_desc.Height = 1;
  resource_desc.DepthOrArraySize = 1;
  resource_desc.MipLevels = 1;
  resource_desc.Format = DXGI_FORMAT_UNKNOWN;
  resource_desc.SampleDesc = {1, 0};
  resource_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  resource_desc.Flags = resource_flags;

  ComPtr<ID3D12Resource> gpu_resource;
  THROW_IF_FAILED(d3d_device->CreateCommittedResource(
      &heap_properties,
      D3D12_HEAP_FLAG_NONE,
      &resource_desc,
      resource_state,
      nullptr,
      IID_PPV_ARGS(&gpu_resource)));

  return gpu_resource;
}

static void WaitForQueueToComplete(ID3D12CommandQueue* queue) {
  ComPtr<ID3D12Device> device;
  THROW_IF_FAILED(queue->GetDevice(IID_PPV_ARGS(device.GetAddressOf())));
  ComPtr<ID3D12Fence> fence;
  THROW_IF_FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(fence.GetAddressOf())));
  THROW_IF_FAILED(queue->Signal(fence.Get(), 1));
  THROW_IF_FAILED(fence->SetEventOnCompletion(1, nullptr));
}

static void UploadDataToDml(
    DmlObjects& dml_objects,
    ID3D12Resource* destination_resource,
    gsl::span<const std::byte> source_data) {
  // Get the size of the resource.
  D3D12_RESOURCE_DESC resource_desc = destination_resource->GetDesc();
  assert(resource_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
  const size_t data_size_in_bytes = static_cast<size_t>(resource_desc.Width);

  // Create intermediate upload resource visible to both CPU and GPU.
  dml_objects.upload_buffer = CreateD3D12ResourceOfByteSize(dml_objects.d3d12_device.Get(), data_size_in_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, D3D12_RESOURCE_FLAG_NONE);

  // Copy CPU-side data to shared memory that is both CPU and GPU visible.
  size_t clamped_data_byte_size = std::min(data_size_in_bytes, source_data.size());
  std::byte* upload_buffer_data = nullptr;
  THROW_IF_FAILED(dml_objects.upload_buffer->Map(0, nullptr, reinterpret_cast<void**>(&upload_buffer_data)));
  memcpy(upload_buffer_data, source_data.data(), clamped_data_byte_size);
  dml_objects.upload_buffer->Unmap(0, nullptr);

  D3D12_RESOURCE_BARRIER resource_barrier{};
  resource_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  resource_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  resource_barrier.Transition = {};
  resource_barrier.Transition.pResource = destination_resource;
  resource_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  resource_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
  resource_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

  // Issue deferred command to copy from the intermediate shared resource to the final GPU resource,
  // and then execute the commands.
  dml_objects.command_list->CopyResource(destination_resource, dml_objects.upload_buffer.Get());
  dml_objects.command_list->ResourceBarrier(1, &resource_barrier);
  THROW_IF_FAILED(dml_objects.command_list->Close());
  ID3D12CommandList* command_lists[] = {dml_objects.command_list.Get()};
  dml_objects.command_queue->ExecuteCommandLists(ARRAYSIZE(command_lists), command_lists);

  THROW_IF_FAILED(dml_objects.command_allocator->Reset());
  THROW_IF_FAILED(dml_objects.command_list->Reset(dml_objects.command_allocator.Get(), nullptr));
}

static void DownloadDataFromDml(
    DmlObjects& dml_objects,
    ID3D12Resource* source_resource,
    gsl::span<std::byte> destination_data) {
  // Get the size of the resource.
  D3D12_RESOURCE_DESC resource_desc = source_resource->GetDesc();
  assert(resource_desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER);
  const size_t data_size_in_bytes = static_cast<size_t>(resource_desc.Width);

  // Create intermediate upload resource visible to both CPU and GPU.
  ComPtr<ID3D12Resource> download_buffer = CreateD3D12ResourceOfByteSize(dml_objects.d3d12_device.Get(), data_size_in_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_NONE);

  D3D12_RESOURCE_BARRIER resource_barrier = {};
  resource_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  resource_barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  resource_barrier.Transition = {};
  resource_barrier.Transition.pResource = source_resource;
  resource_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  resource_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  resource_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;

  // Copy GPU data into the download buffer.
  dml_objects.command_list->ResourceBarrier(1, &resource_barrier);
  dml_objects.command_list->CopyResource(download_buffer.Get(), source_resource);
  THROW_IF_FAILED(dml_objects.command_list->Close());
  ID3D12CommandList* command_lists[] = {dml_objects.command_list.Get()};
  dml_objects.command_queue->ExecuteCommandLists(static_cast<uint32_t>(std::size(command_lists)), command_lists);
  WaitForQueueToComplete(dml_objects.command_queue.Get());
  THROW_IF_FAILED(dml_objects.command_allocator->Reset());
  THROW_IF_FAILED(dml_objects.command_list->Reset(dml_objects.command_allocator.Get(), nullptr));

  // Copy from shared GPU/CPU memory to ordinary system RAM.
  size_t clamped_data_byte_size = std::min(data_size_in_bytes, destination_data.size());
  std::byte* sourceData = nullptr;
  D3D12_RANGE range = {0, clamped_data_byte_size};
  THROW_IF_FAILED(download_buffer->Map(0, &range, reinterpret_cast<void**>(&sourceData)));
  memcpy(destination_data.data(), sourceData, clamped_data_byte_size);
  download_buffer->Unmap(0, nullptr);
}

Ort::Value CreateTensorValueFromExistingD3DResource(
    const OrtDmlApi& ort_dml_api,
    Ort::MemoryInfo const& memory_information,
    ID3D12Resource* d3d_resource,
    gsl::span<const int64_t> tensor_dimensions,
    ONNXTensorElementDataType element_data_type,
    /*out*/ void** dml_ep_resource_wrapper  // Must stay alive with Ort::Value.
) {
  *dml_ep_resource_wrapper = nullptr;

  void* dml_allocator_resource;
  Ort::ThrowOnError(ort_dml_api.CreateGPUAllocationFromD3DResource(d3d_resource, &dml_allocator_resource));
  auto deleter = [&](void*) { ort_dml_api.FreeGPUAllocation(dml_allocator_resource); };
  std::unique_ptr<void, std::function<void(void*)>> dml_allocator_resource_cleanup(dml_allocator_resource, deleter);

  size_t tensor_byte_size = static_cast<size_t>(d3d_resource->GetDesc().Width);
  Ort::Value new_value(
      Ort::Value::CreateTensor(
          memory_information,
          dml_allocator_resource,
          tensor_byte_size,
          tensor_dimensions.data(),
          tensor_dimensions.size(),
          element_data_type));

  // Return values and the wrapped resource.
  *dml_ep_resource_wrapper = dml_allocator_resource;
  dml_allocator_resource_cleanup.release();

  return new_value;
}

#endif

template <typename ModelOutputT, typename ModelInputT = float, typename InputT = Input<float>>
static void TestInference(Ort::Env& env, const std::basic_string<ORTCHAR_T>& model_uri,
                          const std::vector<InputT>& inputs,
                          const char* output_name,
                          const std::vector<int64_t>& expected_dims_y,
                          const std::vector<ModelOutputT>& expected_values_y,
                          int provider_type,
                          OrtCustomOpDomain* custom_op_domain_ptr,
                          const ORTCHAR_T* custom_op_library_filename,
                          bool test_session_creation_only = false,
                          void* cuda_compute_stream = nullptr,
                          Ort::SessionOptions* predefined_session_options = nullptr) {
  Ort::SessionOptions default_session_options;
  Ort::SessionOptions& session_options = predefined_session_options ? *predefined_session_options
                                                                    : default_session_options;

  if (provider_type == 1) {
#ifdef USE_CUDA
    std::cout << "Running simple inference with cuda provider" << std::endl;
    auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(cuda_compute_stream);
    session_options.AppendExecutionProvider_CUDA(cuda_options);
#else
    ORT_UNUSED_PARAMETER(cuda_compute_stream);
    return;
#endif
  } else if (provider_type == 2) {
#ifdef USE_DNNL
    OrtDnnlProviderOptions dnnl_options;
    dnnl_options.use_arena = 1;
    dnnl_options.threadpool_args = nullptr;
    session_options.AppendExecutionProvider_Dnnl(dnnl_options);
#else
    return;
#endif
  } else {
    std::cout << "Running simple inference with default provider" << std::endl;
  }
  if (custom_op_domain_ptr) {
    session_options.Add(custom_op_domain_ptr);
  }

  if (custom_op_library_filename) {
    session_options.RegisterCustomOpsLibrary(custom_op_library_filename);
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri.c_str(), session_options);

  // caller wants to test running the model (not just loading the model)
  if (!test_session_creation_only) {
    // Now run
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    // without preallocated output tensor
    RunSession<ModelOutputT, ModelInputT, InputT>(default_allocator.get(),
                                                  session,
                                                  inputs,
                                                  output_name,
                                                  expected_dims_y,
                                                  expected_values_y,
                                                  nullptr);
    // with preallocated output tensor
    Ort::Value value_y = Ort::Value::CreateTensor<ModelOutputT>(default_allocator.get(),
                                                                expected_dims_y.data(), expected_dims_y.size());

    // test it twice
    for (int i = 0; i != 2; ++i)
      RunSession<ModelOutputT, ModelInputT, InputT>(default_allocator.get(),
                                                    session,
                                                    inputs,
                                                    output_name,
                                                    expected_dims_y,
                                                    expected_values_y,
                                                    &value_y);
  }
}

static constexpr PATH_TYPE MODEL_URI = TSTR("testdata/mul_1.onnx");
#if defined(USE_CUDA) || defined(USE_DML)
static constexpr PATH_TYPE CUDA_GRAPH_ANNOTATION_MODEL_URI = TSTR("testdata/mul_1_dynamic.onnx");
#endif
static constexpr PATH_TYPE MATMUL_MODEL_URI = TSTR("testdata/matmul_1.onnx");
#ifndef ORT_NO_RTTI
static constexpr PATH_TYPE SEQUENCE_MODEL_URI = TSTR("testdata/sequence_length.onnx");
#endif
#if !defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
static constexpr PATH_TYPE SEQUENCE_MODEL_URI_2 = TSTR("testdata/optional_sequence_tensor.onnx");
#endif
static constexpr PATH_TYPE CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_1.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_ATTR_TESTER_URI = TSTR("testdata/custom_op_library/attr_tester.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_TEST_MODEL_URI = TSTR("testdata/custom_op_library/custom_op_test.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_2 = TSTR("testdata/custom_op_library/copy_2_inputs_2_outputs.onnx");
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_3 = TSTR("testdata/custom_op_library/copy_3_inputs_3_outputs.onnx");
#if !defined(DISABLE_FLOAT8_TYPES)
static constexpr PATH_TYPE CUSTOM_OP_LIBRARY_TEST_MODEL_FLOAT8_URI = TSTR("testdata/custom_op_library/custom_op_test_float8.onnx");
#endif
#if defined(USE_OPENVINO) && (!defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))
static constexpr PATH_TYPE CUSTOM_OP_OPENVINO_WRAPPER_LIB_TEST_MODEL_URI = TSTR(
    "testdata/custom_op_openvino_wrapper_library/custom_op_mnist_ov_wrapper.onnx");
#endif
static constexpr PATH_TYPE OVERRIDABLE_INITIALIZER_MODEL_URI = TSTR("testdata/overridable_initializer.onnx");
static constexpr PATH_TYPE NAMED_AND_ANON_DIM_PARAM_URI = TSTR("testdata/capi_symbolic_dims.onnx");
static constexpr PATH_TYPE MODEL_WITH_CUSTOM_MODEL_METADATA = TSTR("testdata/model_with_valid_ort_config_json.onnx");
static constexpr PATH_TYPE VARIED_INPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/VariedInputCustomOp.onnx");
static constexpr PATH_TYPE VARIED_INPUT_CUSTOM_OP_MODEL_URI_2 = TSTR("testdata/foo_3.onnx");
static constexpr PATH_TYPE OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/foo_bar_1.onnx");
static constexpr PATH_TYPE OPTIONAL_INPUT_CUSTOM_OP_MODEL_URI_2 = TSTR("testdata/foo_bar_2.onnx");
static constexpr PATH_TYPE VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR("testdata/custom_op_variadic_io.onnx");
static constexpr PATH_TYPE VARIADIC_UNDEF_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI = TSTR(
    "testdata/custom_op_variadic_undef_io.onnx");
static constexpr PATH_TYPE CUSTOM_OP_MODEL_WITH_ATTRIBUTES_URI = TSTR("testdata/foo_bar_3.onnx");
static constexpr PATH_TYPE CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL = TSTR("testdata/custom_op_single_schema_multi_kernel.onnx");
#if !defined(DISABLE_SPARSE_TENSORS)
static constexpr PATH_TYPE SPARSE_OUTPUT_MODEL_URI = TSTR("testdata/sparse_initializer_as_output.onnx");
#ifndef DISABLE_CONTRIB_OPS
static constexpr PATH_TYPE SPARSE_INPUT_MATMUL_MODEL_URI = TSTR("testdata/sparse_to_dense_matmul.onnx");
#endif
#endif  // !defined(DISABLE_SPARSE_TENSORS)

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
static constexpr PATH_TYPE ORT_CUSTOM_OPS_MODEL_URI = TSTR("testdata/custom_op_string_lower.onnx");
static constexpr PATH_TYPE ORT_CUSTOM_OPS_MODEL_URI_2 = TSTR("testdata/custom_op_negpos.onnx");
#endif

#ifdef ENABLE_LANGUAGE_INTEROP_OPS
static constexpr PATH_TYPE PYOP_FLOAT_MODEL_URI = TSTR("testdata/pyop_1.onnx");
static constexpr PATH_TYPE PYOP_MULTI_MODEL_URI = TSTR("testdata/pyop_2.onnx");
static constexpr PATH_TYPE PYOP_KWARG_MODEL_URI = TSTR("testdata/pyop_3.onnx");
#endif

#ifndef REDUCED_OPS_BUILD
static constexpr PATH_TYPE RESIZE_AND_CROP_MODEL_URI = TSTR("testdata/crop_and_resize.onnx");
#endif

static constexpr PATH_TYPE SIMPLIFIED_SSD_MODEL_URI = TSTR("testdata/multi_stream_models/simplified_ssd.onnx");

class CApiTestWithProvider : public testing::Test, public ::testing::WithParamInterface<int> {
};

TEST_P(CApiTestWithProvider, simple) {
  // simple inference test
  // prepare inputs
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  TestInference<float>(*ort_env, MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, GetParam(),
                       nullptr, nullptr);
}

template <class T, size_t element_count_to_create>
void TestGetTensorSizeInBytes(Ort::ConstMemoryInfo cpu_meminfo) {
  constexpr const size_t expected_size_in_bytes = sizeof(T) * element_count_to_create;
  constexpr const std::array<int64_t, 2> dims = {1, static_cast<int64_t>(element_count_to_create)};
  std::array<T, element_count_to_create> data;
  std::fill(data.begin(), data.end(), T{1});

  auto value = Ort::Value::CreateTensor<T>(cpu_meminfo, data.data(),
                                           data.size(), dims.data(), dims.size());

  auto type_info = value.GetTypeInfo();
  ASSERT_EQ(type_info.GetONNXType(), ONNX_TYPE_TENSOR);
  auto tensor_type_info = type_info.GetTensorTypeAndShapeInfo();
  const auto element_count = tensor_type_info.GetElementCount();
  ASSERT_EQ(expected_size_in_bytes / sizeof(T), element_count);
  ASSERT_EQ(expected_size_in_bytes, value.GetTensorSizeInBytes());
}

TEST(CApiTest, TestGetTensorSizeInBytes) {
  Ort::MemoryInfo cpu_meminfo("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  TestGetTensorSizeInBytes<float, 1>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<float, 2>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<float, 3>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<float, 4>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<float, 5>(cpu_meminfo.GetConst());

  TestGetTensorSizeInBytes<int64_t, 1>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<int64_t, 2>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<int64_t, 3>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<int64_t, 4>(cpu_meminfo.GetConst());
  TestGetTensorSizeInBytes<int64_t, 5>(cpu_meminfo.GetConst());
}

TEST(CApiTest, dim_param) {
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, NAMED_AND_ANON_DIM_PARAM_URI, session_options);

  auto in0 = session.GetInputTypeInfo(0);
  auto in0_ttsi = in0.GetTensorTypeAndShapeInfo();

  auto num_input_dims = in0_ttsi.GetDimensionsCount();
  ASSERT_GE(num_input_dims, 1u);
  // reading 1st dimension only so don't need to malloc int64_t* or const char** values for the Get*Dimensions calls
  int64_t dim_value = 0;
  const char* dim_param = nullptr;
  auto dims = in0_ttsi.GetShape();
  if (!dims.empty()) dim_value = dims[0];
  in0_ttsi.GetSymbolicDimensions(&dim_param, 1);
  ASSERT_EQ(dim_value, -1) << "symbolic dimension should be -1";
  ASSERT_EQ(strcmp(dim_param, "n"), 0) << "Expected 'n'. Got: " << dim_param;

  auto out0 = session.GetOutputTypeInfo(0);
  auto out0_ttsi = out0.GetTensorTypeAndShapeInfo();
  auto num_output_dims = out0_ttsi.GetDimensionsCount();
  ASSERT_EQ(num_output_dims, 1u);

  dim_value = 0;
  dims = out0_ttsi.GetShape();
  if (!dims.empty()) dim_value = dims[0];

  out0_ttsi.GetSymbolicDimensions(&dim_param, 1);
  ASSERT_EQ(dim_value, -1) << "symbolic dimension should be -1";
  ASSERT_EQ(strcmp(dim_param, ""), 0);
}

// Tests calling OrtApi::GetTensorElementTypeAndShapeDataReference for a dense OrtValue tensor.
TEST(CApiTest, Value_GetTensorElementTypeAndShapeDataReference_DenseTensor) {
  Ort::MemoryInfo info_cpu = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Ort::Value x_value = Ort::Value::CreateTensor(info_cpu, x_values.data(), x_values.size(),
                                                x_shape.data(), x_shape.size());
  Ort::TensorTypeAndShapeInfo type_shape_info = x_value.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Value::Shape shape{};
  x_value.GetTensorElementTypeAndShapeDataReference(elem_type, shape);

  ASSERT_EQ(elem_type, type_shape_info.GetElementType());

  std::vector<int64_t> expected_shape = type_shape_info.GetShape();
  gsl::span<const int64_t> actual_shape(shape.shape, shape.shape_len);
  ASSERT_EQ(actual_shape, gsl::span<const int64_t>(expected_shape));
}

// Tests calling OrtApi::GetTensorElementTypeAndShapeDataReference for a scalar OrtValue tensor.
TEST(CApiTest, Value_GetTensorElementTypeAndShapeDataReference_Scalar) {
  Ort::MemoryInfo info_cpu = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<int64_t> x_shape = {};  // Scalar (no shape)
  std::array<float, 1> x_values = {1.0f};
  Ort::Value x_value = Ort::Value::CreateTensor(info_cpu, x_values.data(), x_values.size(),
                                                x_shape.data(), x_shape.size());
  Ort::TensorTypeAndShapeInfo type_shape_info = x_value.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Value::Shape shape{};
  x_value.GetTensorElementTypeAndShapeDataReference(elem_type, shape);

  ASSERT_EQ(elem_type, type_shape_info.GetElementType());

  std::vector<int64_t> expected_shape = type_shape_info.GetShape();
  gsl::span<const int64_t> actual_shape(shape.shape, shape.shape_len);
  ASSERT_EQ(actual_shape, gsl::span<const int64_t>(expected_shape));
  ASSERT_EQ(shape.shape, nullptr);
  ASSERT_EQ(shape.shape_len, 0);
}

#if !defined(DISABLE_SPARSE_TENSORS)
// Tests calling OrtApi::GetTensorElementTypeAndShapeDataReference for a sparse OrtValue tensor.
TEST(CApiTest, Value_GetTensorElementTypeAndShapeDataReference_SparseTensor) {
  std::vector<int64_t> common_shape{9, 9};
  std::vector<float> A_values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                              10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                              18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                              26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
                              34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0,
                              42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
                              50.0, 51.0, 52.0, 53.0};

  // 2 - D index
  std::vector<int64_t> indices_shape{gsl::narrow<int64_t>(A_values.size()), 2};
  std::vector<int64_t> A_indices{0, 1, 0, 2, 0, 6, 0, 7, 0, 8, 1, 0, 1,
                                 1, 1, 2, 1, 6, 1, 7, 1, 8, 2, 0, 2, 1,
                                 2, 2, 2, 6, 2, 7, 2, 8, 3, 3, 3, 4, 3,
                                 5, 3, 6, 3, 7, 3, 8, 4, 3, 4, 4, 4, 5,
                                 4, 6, 4, 7, 4, 8, 5, 3, 5, 4, 5, 5, 5,
                                 6, 5, 7, 5, 8, 6, 0, 6, 1, 6, 2, 6, 3,
                                 6, 4, 6, 5, 7, 0, 7, 1, 7, 2, 7, 3, 7,
                                 4, 7, 5, 8, 0, 8, 1, 8, 2, 8, 3, 8, 4,
                                 8, 5};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Value::Shape ort_dense_shape{common_shape.data(), common_shape.size()};
  Ort::Value::Shape ort_values_shape{&indices_shape[0], 1U};
  auto value_sparse = Ort::Value::CreateSparseTensor(info, A_values.data(), ort_dense_shape, ort_values_shape);
  value_sparse.UseCooIndices(A_indices.data(), A_indices.size());

  Ort::TensorTypeAndShapeInfo type_shape_info = value_sparse.GetTensorTypeAndShapeInfo();

  ONNXTensorElementDataType elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  Ort::Value::Shape shape{};
  value_sparse.GetTensorElementTypeAndShapeDataReference(elem_type, shape);

  ASSERT_EQ(elem_type, type_shape_info.GetElementType());

  std::vector<int64_t> expected_shape = type_shape_info.GetShape();
  gsl::span<const int64_t> actual_shape(shape.shape, shape.shape_len);
  ASSERT_EQ(actual_shape, gsl::span<const int64_t>(expected_shape));
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

static std::pair<bool, bool> LoadAndGetInputShapePresent(const ORTCHAR_T* const model_url) {
  Ort::Session session(*ort_env, model_url, Ort::SessionOptions{});
  const auto input_num = session.GetInputCount();
  EXPECT_EQ(input_num, 1U);
  const bool input_shape_present = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().HasShape();
  const auto output_num = session.GetOutputCount();
  EXPECT_EQ(output_num, 1U);
  const bool output_shape_present = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().HasShape();
  return {input_shape_present, output_shape_present};
}

TEST(CApiTest, OptionalShape) {
  const ORTCHAR_T* const input_shape_model = TSTR("testdata/abs_0d_input.onnx");
  auto result = LoadAndGetInputShapePresent(input_shape_model);
  ASSERT_TRUE(result.first);
  ASSERT_TRUE(result.second);

  const ORTCHAR_T* const no_shape_model = TSTR("testdata/abs_0d_lostdim.onnx");
  result = LoadAndGetInputShapePresent(no_shape_model);
  ASSERT_FALSE(result.first);
  ASSERT_FALSE(result.second);
}

INSTANTIATE_TEST_SUITE_P(CApiTestWithProviders,
                         CApiTestWithProvider,
                         ::testing::Values(0, 1, 2, 3, 4));

TEST(CApiTest, TestInputPassThroughToOutput) {
  const ORTCHAR_T* model_uri = TSTR("testdata/input_propagated_to_output.onnx");
  Ort::Session session(*ort_env, model_uri, Ort::SessionOptions{});
  auto inputs_meminfos = session.GetMemoryInfoForInputs();
  ASSERT_EQ(1U, inputs_meminfos.size());
  auto inputs_epdevices = session.GetEpDeviceForInputs();
  ASSERT_EQ(1U, inputs_epdevices.size());
  auto outputs_meminfos = session.GetMemoryInfoForOutputs();
  ASSERT_EQ(7U, outputs_meminfos.size());
}

TEST(CApiTest, TestDanglingInput) {
  // Here we test an issue with segments_ids that is an input not consumed by anything
  // This kind of model is unlikely to be used in practice but we want to make sure it works
  const ORTCHAR_T* model_uri = TSTR("testdata/test_dangling_input_segment_ids.onnx");
  Ort::Session session(*ort_env, model_uri, Ort::SessionOptions{});
  auto inputs_meminfos = session.GetMemoryInfoForInputs();
  ASSERT_EQ(2U, inputs_meminfos.size());
  auto outputs_meminfos = session.GetMemoryInfoForOutputs();
  ASSERT_EQ(2U, outputs_meminfos.size());
  auto inputs_epdevices = session.GetEpDeviceForInputs();
  ASSERT_EQ(2U, inputs_epdevices.size());
  // One of the devices returning is null since the input is not consumed
  // there is not a device for it.
  const bool null_present = std::any_of(inputs_epdevices.begin(), inputs_epdevices.end(),
                                        [](const auto& device) { return device == nullptr; });
  ASSERT_TRUE(null_present);
}

#if !defined(DISABLE_SPARSE_TENSORS)
TEST(CApiTest, SparseOutputModel) {
  std::vector<int64_t> dense_shape{3, 3};
  std::vector<float> values{1.764052391052246, 0.40015721321105957, 0.978738009929657};
  std::vector<int64_t> values_shape{3};
  std::vector<int64_t> coo_indices{2, 3, 5};
  std::vector<int64_t> indices_shape{3};

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  const char* const output_names[] = {"values"};
  // This model produces a sparse output from a constant sparse initializer
  Ort::Session session(*ort_env, SPARSE_OUTPUT_MODEL_URI, Ort::SessionOptions{});
  auto inputs_meminfos = session.GetMemoryInfoForInputs();
  ASSERT_TRUE(inputs_meminfos.empty());
  auto outputs_meminfos = session.GetMemoryInfoForOutputs();
  ASSERT_EQ(1U, outputs_meminfos.size());
  auto inputs_epdevices = session.GetEpDeviceForInputs();
  ASSERT_TRUE(inputs_epdevices.empty());

  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);
  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& sparse_output = ort_outputs[0];
  auto ti = sparse_output.GetTypeInfo();
  ASSERT_EQ(ONNX_TYPE_SPARSETENSOR, ti.GetONNXType());
  auto tensor_type_shape = ti.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(dense_shape, tensor_type_shape.GetShape());
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, tensor_type_shape.GetElementType());

  ASSERT_EQ(ORT_SPARSE_COO, sparse_output.GetSparseFormat());
  auto values_ts = sparse_output.GetSparseTensorValuesTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, values_ts.GetElementType());
  ASSERT_EQ(values_shape, values_ts.GetShape());

  const auto* values_fetch = sparse_output.GetSparseTensorValues<float>();
  auto val_span = gsl::make_span(values_fetch, values.size());
  ASSERT_TRUE(std::equal(values.cbegin(), values.cend(), val_span.begin(), val_span.end()));

  auto indices_ts = sparse_output.GetSparseTensorIndicesTypeShapeInfo(ORT_SPARSE_COO_INDICES);
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_ts.GetElementType());
  ASSERT_EQ(indices_shape, indices_ts.GetShape());

  size_t num_indices = 0;
  const int64_t* indices = sparse_output.GetSparseTensorIndicesData<int64_t>(ORT_SPARSE_COO_INDICES, num_indices);
  ASSERT_EQ(num_indices, static_cast<size_t>(indices_shape[0]));
  auto ind_span = gsl::make_span(indices, num_indices);
  ASSERT_TRUE(std::equal(coo_indices.cbegin(), coo_indices.cend(), ind_span.begin(), ind_span.end()));
}

#ifndef DISABLE_CONTRIB_OPS
TEST(CApiTest, SparseInputModel) {
  std::vector<int64_t> common_shape{9, 9};  // inputs and outputs same shape
  std::vector<float> A_values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                              10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                              18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                              26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
                              34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0,
                              42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
                              50.0, 51.0, 52.0, 53.0};

  // 2 - D index
  std::vector<int64_t> indices_shape{gsl::narrow<int64_t>(A_values.size()), 2};
  std::vector<int64_t> A_indices{0, 1, 0, 2, 0, 6, 0, 7, 0, 8, 1, 0, 1,
                                 1, 1, 2, 1, 6, 1, 7, 1, 8, 2, 0, 2, 1,
                                 2, 2, 2, 6, 2, 7, 2, 8, 3, 3, 3, 4, 3,
                                 5, 3, 6, 3, 7, 3, 8, 4, 3, 4, 4, 4, 5,
                                 4, 6, 4, 7, 4, 8, 5, 3, 5, 4, 5, 5, 5,
                                 6, 5, 7, 5, 8, 6, 0, 6, 1, 6, 2, 6, 3,
                                 6, 4, 6, 5, 7, 0, 7, 1, 7, 2, 7, 3, 7,
                                 4, 7, 5, 8, 0, 8, 1, 8, 2, 8, 3, 8, 4,
                                 8, 5};

  std::vector<float> B_data{0, 1, 2, 0, 0, 0, 3, 4, 5,
                            6, 7, 8, 0, 0, 0, 9, 10, 11,
                            12, 13, 14, 0, 0, 0, 15, 16, 17,
                            0, 0, 0, 18, 19, 20, 21, 22, 23,
                            0, 0, 0, 24, 25, 26, 27, 28, 29,
                            0, 0, 0, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 0, 0, 0,
                            42, 43, 44, 45, 46, 47, 0, 0, 0,
                            48, 49, 50, 51, 52, 53, 0, 0, 0};

  std::vector<float> Y_result{546, 561, 576, 552, 564, 576, 39, 42, 45,
                              1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
                              2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
                              2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
                              3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
                              4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
                              678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
                              786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
                              894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Value::Shape ort_dense_shape{common_shape.data(), common_shape.size()};
  Ort::Value::Shape ort_values_shape{&indices_shape[0], 1U};
  auto a_st = Ort::Value::CreateSparseTensor(info, A_values.data(), ort_dense_shape, ort_values_shape);
  a_st.UseCooIndices(A_indices.data(), A_indices.size());

  auto b_tensor = Ort::Value::CreateTensor(info, B_data.data(), B_data.size(), common_shape.data(), common_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(a_st));
  ort_inputs.push_back(std::move(b_tensor));
  const char* input_names[] = {"sparse_A", "dense_B"};
  const char* const output_names[] = {"dense_Y"};
  Ort::Session session(*ort_env, SPARSE_INPUT_MATMUL_MODEL_URI, Ort::SessionOptions{});
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
                                 output_names, 1);
  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& dense_Y = ort_outputs[0];
  ASSERT_TRUE(dense_Y.IsTensor());

  auto result_ts = dense_Y.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, result_ts.GetElementType());
  ASSERT_EQ(common_shape, result_ts.GetShape());

  const auto* result_vals = dense_Y.GetTensorData<float>();
  auto result_span = gsl::make_span(result_vals, Y_result.size());
  ASSERT_TRUE(std::equal(Y_result.cbegin(), Y_result.cend(), result_span.begin(), result_span.end()));
}
#endif  // DISABLE_CONTRIB_OPS
#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Test that a custom op compiled against a newer ORT version (higher OrtCustomOp::version)
// can still be loaded on this ORT runtime. This simulates the forward compatibility
// scenario where an IHV EP compiled against ORT v(N+1) is loaded into ORT v(N).
// We test session creation only (which covers schema registration, kernel def building, and
// CustomOpKernel construction with the capped API version).
TEST(CApiTest, custom_op_forward_version_compat) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider};

  // Simulate an EP compiled against a newer ORT version by setting version higher than ORT_API_VERSION.
  // The OrtCustomOp struct's function pointers are still valid for the current version, so this should work.
  custom_op.version = ORT_API_VERSION + 1;

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  // test_session_creation_only=true: exercises schema registration, kernel def building, and
  // CustomOpKernel construction (which was previously blocked by the blanket version reject).
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr, /*test_session_creation_only=*/true);
}

// Memory leak
#ifndef ABSL_HAVE_ADDRESS_SANITIZER
TEST(CApiTest, custom_op_handler) {
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

#ifdef USE_CUDA
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOp custom_op{onnxruntime::kCudaExecutionProvider};
#else
  MyCustomOp custom_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr, false, compute_stream);
  cudaStreamDestroy(compute_stream);
#else
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}
#endif

#ifdef USE_CUDA
TEST(CApiTest, custom_op_set_input_memory_type) {
  std::cout << "Running custom op inference" << std::endl;

  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOpSecondInputOnCpu custom_op{onnxruntime::kCudaExecutionProvider, compute_stream};

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  auto x_mem_type = custom_op.GetInputMemoryType(0);
  auto y_mem_type = custom_op.GetInputMemoryType(1);
  ASSERT_EQ(x_mem_type, OrtMemType::OrtMemTypeDefault);
  ASSERT_EQ(y_mem_type, OrtMemType::OrtMemTypeCPUInput);

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr, false, compute_stream);
  cudaStreamDestroy(compute_stream);
}
#endif

#if !defined(ORT_MINIMAL_BUILD)
TEST(CApiTest, StandaloneOpHandler) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

#ifdef USE_CUDA
  StandaloneCustomOp standalone_op{onnxruntime::kCudaExecutionProvider};
#else
  StandaloneCustomOp standalone_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&standalone_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 1,
                       custom_op_domain, nullptr);
#else
  Ort::SessionOptions session_options;
  const std::basic_string<ORTCHAR_T> ort_file = ORT_TSTR("testdata/foo_1.onnx.test_output.ort");
  session_options.SetOptimizedModelFilePath(ort_file.c_str());

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr, false, nullptr, &session_options);

  TestInference<float>(*ort_env, ort_file, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       custom_op_domain, nullptr);
#endif
}
#endif

#ifdef ENABLE_EXTENSION_CUSTOM_OPS
// test enabled ort-customops negpos
TEST(CApiTest, test_enable_ort_customops_negpos) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto allocator = std::make_unique<MockedOrtAllocator>();

  // Create Inputs
  std::vector<Ort::Value> ort_inputs;
  std::vector<float> input_data = {-1.1f, 2.2f, 4.4f, -5.5f};
  std::vector<int64_t> input_dims = {2, 2};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_data.data()), input_data.size(), input_dims.data(), input_dims.size()));

  // Create Session with ORT CustomOps
  Ort::SessionOptions session_options;
  session_options.EnableOrtCustomOps();
  Ort::Session session(*ort_env, ORT_CUSTOM_OPS_MODEL_URI_2, session_options);

  // Create Input and Output Names
  std::vector<const char*> input_names = {"X"};
  const char* output_names[] = {"out0", "out1"};

  // Run Session
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));

  // Validate Results
  ASSERT_EQ(ort_outputs.size(), 2u);

  std::vector<int64_t> out_dims = {2, 2};
  std::vector<float> values_out0 = {-1.1f, 0.0f, 0.0f, -5.5f};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), out_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_out0.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_out0[i], f[i]);
  }
}

// test enabled ort-customops stringlower
TEST(CApiTest, test_enable_ort_customops_stringlower) {
  auto allocator = std::make_unique<MockedOrtAllocator>();

  // Create Inputs
  std::vector<Ort::Value> ort_inputs;
  std::string input_data{"HI, This is ENGINEER from Microsoft."};
  const char* const input_strings[] = {input_data.c_str()};
  std::vector<int64_t> input_dims = {1, 1};

  Ort::Value input_tensor = Ort::Value::CreateTensor(allocator.get(), input_dims.data(), input_dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  input_tensor.FillStringTensor(input_strings, 1U);
  ort_inputs.push_back(std::move(input_tensor));

  // Create Session with ORT CustomOps
  Ort::SessionOptions session_options;
  session_options.EnableOrtCustomOps();
  Ort::Session session(*ort_env, ORT_CUSTOM_OPS_MODEL_URI, session_options);

  // Create Input and Output Names
  std::vector<const char*> input_names = {"input_1"};
  const char* output_names[] = {"customout"};

  // Run Session
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));

  // Validate Results
  ASSERT_EQ(ort_outputs.size(), 1u);

  std::vector<int64_t> out_dims = {1, 1};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), out_dims);
  ASSERT_EQ(type_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  std::string output_data{"hi, this is engineer from microsoft."};
  auto expected_string = output_data.c_str();
  size_t expected_string_len = strlen(expected_string);
  auto data_length = ort_outputs[0].GetStringTensorDataLength();
  ASSERT_EQ(expected_string_len, data_length);

  std::string result(data_length, '\0');
  std::vector<size_t> offsets(type_info.GetElementCount());
  ort_outputs[0].GetStringTensorContent((void*)result.data(), data_length, offsets.data(), offsets.size());
  ASSERT_STREQ(result.c_str(), expected_string);
}
#endif

// test custom op which accepts float and double as inputs
TEST(CApiTest, varied_input_custom_op_handler) {
  std::vector<Input<float>> inputs(2);
  inputs[0].name = "X";
  inputs[0].dims = {3};
  inputs[0].values = {2.0f, 3.0f, 4.0f};
  inputs[1].name = "Y";
  inputs[1].dims = {3};
  inputs[1].values = {5.0f, 6.0f, 7.0f};
  std::vector<int64_t> expected_dims_z = {1};
  std::vector<float> expected_values_z = {10.0f};

#ifdef USE_CUDA
  SliceCustomOp slice_custom_op{onnxruntime::kCudaExecutionProvider};
#else
  SliceCustomOp slice_custom_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("abc");
  custom_op_domain.Add(&slice_custom_op);

#ifdef USE_CUDA
  TestInference<float>(*ort_env, VARIED_INPUT_CUSTOM_OP_MODEL_URI, inputs, "Z",
                       expected_dims_z, expected_values_z, 1, custom_op_domain, nullptr);
#else
  TestInference<float>(*ort_env, VARIED_INPUT_CUSTOM_OP_MODEL_URI, inputs, "Z",
                       expected_dims_z, expected_values_z, 0, custom_op_domain, nullptr);
#endif
}

TEST(CApiTest, multiple_varied_input_custom_op_handler) {
#ifdef USE_CUDA
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  MyCustomOpMultipleDynamicInputs custom_op{onnxruntime::kCudaExecutionProvider};
#else
  MyCustomOpMultipleDynamicInputs custom_op{onnxruntime::kCpuExecutionProvider};
#endif

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;

#ifdef USE_CUDA
  auto cuda_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(compute_stream);
  session_options.AppendExecutionProvider_CUDA(cuda_options);
#endif

  session_options.Add(custom_op_domain);
  Ort::Session session(*ort_env, VARIED_INPUT_CUSTOM_OP_MODEL_URI_2, session_options);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("X");
  std::vector<float> input_0_data = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<int64_t> input_0_dims = {3, 2};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));

  // input 1 (double type)
  input_names.emplace_back("W");
  std::vector<double> input_1_data = {2, 3, 4, 5, 6, 7};
  std::vector<int64_t> input_1_dims = {3, 2};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<double>(info, const_cast<double*>(input_1_data.data()),
                                       input_1_data.size(), input_1_dims.data(), input_1_dims.size()));

  // Run
  const char* output_name = "Y";
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 &output_name, 1);
  ASSERT_EQ(ort_outputs.size(), 1u);

  // Validate results
  std::vector<int64_t> y_dims = {3, 2};
  std::vector<float> values_y = {3.f, 5.f, 7.f, 9.f, 11.f, 13.f};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), y_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }

#ifdef USE_CUDA
  cudaStreamDestroy(compute_stream);
#endif
}

TEST(CApiTest, variadic_input_output_custom_op) {
  // Create a custom op with 1 variadic input and 1 variadic output.
  // The model passes in 3 string inputs and expects 3 int64_t outputs.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::vector<int64_t>> expected_dims;
  std::vector<std::vector<int64_t>> expected_lens;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Create inputs.
  AddInputForCustomStringLengthsKernel("hello", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);
  AddInputForCustomStringLengthsKernel("", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);
  AddInputForCustomStringLengthsKernel("123", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);

  // Create arrays of c-strings for input and output names.
  auto get_c_str = [](const std::string& str) { return str.c_str(); };
  std::vector<const char*> input_name_cstrs(input_names.size());
  std::transform(input_names.begin(), input_names.end(), input_name_cstrs.begin(), get_c_str);
  std::vector<const char*> output_name_cstrs(output_names.size());
  std::transform(output_names.begin(), output_names.end(), output_name_cstrs.begin(), get_c_str);

  Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_name_cstrs.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_name_cstrs.data(), output_name_cstrs.size());
  ASSERT_EQ(ort_outputs.size(), 3u);

  // Validate outputs.
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), expected_dims[i]);
    ASSERT_EQ(type_info.GetElementCount(), 1u);

    int64_t* lens_data = ort_outputs[i].GetTensorMutableData<int64_t>();
    ASSERT_EQ(lens_data[0], expected_lens[i][0]);
  }
}

TEST(CApiTest, mixed_variadic_input_output_custom_op) {
  // Create a custom op with 2 inputs (required, variadic) and 2 outputs (required, variadic).
  // The model passes in 3 string inputs and expects 3 int64_t outputs.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<std::vector<int64_t>> expected_dims;
  std::vector<std::vector<int64_t>> expected_lens;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  // Create inputs.
  AddInputForCustomStringLengthsKernel("mixed variadic", allocator, ort_inputs, input_names, output_names,
                                       expected_dims, expected_lens);
  AddInputForCustomStringLengthsKernel("", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);
  AddInputForCustomStringLengthsKernel("abcd", allocator, ort_inputs, input_names, output_names, expected_dims,
                                       expected_lens);

  // Create arrays of c-strings for input and output names.
  auto get_c_str = [](const std::string& str) { return str.c_str(); };
  std::vector<const char*> input_name_cstrs(input_names.size());
  std::transform(input_names.begin(), input_names.end(), input_name_cstrs.begin(), get_c_str);
  std::vector<const char*> output_name_cstrs(output_names.size());
  std::transform(output_names.begin(), output_names.end(), output_name_cstrs.begin(), get_c_str);

  Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_name_cstrs.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_name_cstrs.data(), output_name_cstrs.size());
  ASSERT_EQ(ort_outputs.size(), 3u);

  // Validate outputs.
  for (size_t i = 0; i < ort_outputs.size(); ++i) {
    auto type_info = ort_outputs[i].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), expected_dims[i]);
    ASSERT_EQ(type_info.GetElementCount(), 1u);

    int64_t* lens_data = ort_outputs[i].GetTensorMutableData<int64_t>();
    ASSERT_EQ(lens_data[0], expected_lens[i][0]);
  }
}

TEST(CApiTest, variadic_undef_input_output_custom_op) {
  // Create a custom op with 1 variadic input and 1 variadic output.
  // Both the input and output are of undefined element type and allowed to differ in type (hetergeneous).
  // The model passes in inputs (string, int64_t, and float) which are then echoed in
  // reversed order (float, int64_t, string).
  TemplatedCustomOp<MyCustomEchoReversedArgsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      false,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      false);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ConstMemoryInfo mem_info = allocator.GetInfo();
  std::vector<int64_t> input_dims = {1};

  // Set string input.
  std::string str_input("hello_ort");
  Ort::Value& str_input_val = ort_inputs.emplace_back(
      Ort::Value::CreateTensor(allocator, input_dims.data(), input_dims.size(),
                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
  str_input_val.FillStringTensorElement(str_input.c_str(), 0);

  // Set int64_t input.
  std::array<int64_t, 1> int_inps = {23};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(mem_info, int_inps.data(), int_inps.size(),
                                                            input_dims.data(), input_dims.size()));

  // Set float input.
  std::array<float, 1> float_inps = {10.0f};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(mem_info, float_inps.data(), float_inps.size(),
                                                          input_dims.data(), input_dims.size()));

  constexpr std::array<const char*, 3> input_names = {"input_0", "input_1", "input_2"};
  constexpr std::array<const char*, 3> output_names = {"output_0", "output_1", "output_2"};

  Ort::Session session(*ort_env, VARIADIC_UNDEF_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 output_names.data(), output_names.size());
  ASSERT_EQ(ort_outputs.size(), 3u);

  // Validate outputs.

  // First output should be a float.
  {
    auto& ort_output = ort_outputs[0];
    auto type_info = ort_output.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), input_dims);
    ASSERT_EQ(type_info.GetElementCount(), 1u);
    ASSERT_EQ(type_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    const float* out_ptr = ort_output.GetTensorData<float>();
    ASSERT_EQ(out_ptr[0], float_inps[0]);
  }

  // Second output should be a int64_t.
  {
    auto& ort_output = ort_outputs[1];
    auto type_info = ort_output.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), input_dims);
    ASSERT_EQ(type_info.GetElementCount(), 1u);
    ASSERT_EQ(type_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);

    const int64_t* out_ptr = ort_output.GetTensorData<int64_t>();
    ASSERT_EQ(out_ptr[0], int_inps[0]);
  }

  // Last output should be a string.
  {
    auto& ort_output = ort_outputs[2];
    auto type_info = ort_output.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), input_dims);
    ASSERT_EQ(type_info.GetElementCount(), 1u);
    ASSERT_EQ(type_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    const size_t str_len = ort_output.GetStringTensorElementLength(0);
    ASSERT_EQ(str_len, str_input.size());

    std::string str;
    str.resize(str_len);

    ort_output.GetStringTensorElement(str_len, 0, str.data());
    ASSERT_EQ(str, str_input);
  }
}

TEST(CApiTest, invalid_variadic_input_not_last_custom_op) {
  // Create an invalid custom op with 2 inputs. The first input is variadic and the last is not.
  // Expect an error because only the last input may be marked as variadic.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Only the last input to a custom op may be marked variadic."));
  }
}

TEST(CApiTest, invalid_variadic_output_not_last_custom_op) {
  // Create an invalid custom op with 2 outputs. The first output is variadic and the last is not.
  // Expect an error because only the last output may be marked as variadic.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
       ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC,
       OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Only the last output to a custom op may be marked variadic."));
  }
}

TEST(CApiTest, invalid_variadic_input_min_arity_custom_op) {
  // Create a custom op with a variadic input with a minimum arity of 4.
  // Expect an error because the model passes in less than 4 inputs to the op.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      4,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Error Node(VariadicNode0) with schema(test::VariadicNode:1) has input size 3 not in range [min=4,"));
  }
}

TEST(CApiTest, invalid_variadic_output_min_arity_custom_op) {
  // Create a custom op with a variadic output with a minimum arity of 4.
  // Expect an error because the model instantiates the op with less than 4 outputs.
  TemplatedCustomOp<MyCustomStringLengthsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      4,
      true);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Error Node(VariadicNode0) with schema(test::VariadicNode:1) has output size 3 not in range [min=4"));
  }
}

TEST(CApiTest, invalid_variadic_input_homogeneity_custom_op) {
  // Create a custom op with a homogeneous variadic input. The model has heterogeneous inputs,
  // so we expect an error.
  TemplatedCustomOp<MyCustomEchoReversedArgsKernel> custom_op(
      "VariadicNode",
      // Input config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      true,  // Input homogeneity requirement will cause error!
      // Output config
      {ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED},
      {OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC},
      1,
      false);

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  try {
    Ort::Session session(*ort_env, VARIADIC_UNDEF_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    FAIL();
  } catch (const Ort::Exception& excpt) {
    ASSERT_THAT(excpt.what(), testing::HasSubstr("Type Error: Type parameter (Input0) of Optype (VariadicNode) bound "
                                                 "to different types"));
  }
}

TEST(CApiTest, optional_input_custom_op_handler) {
  MyCustomOpWithOptionalInput custom_op{onnxruntime::kCpuExecutionProvider};

  // `MyCustomOpFooBar` defines a custom op with atmost 3 inputs and the second input is optional.
  // In this test, we are going to try and run 2 models - one with the optional input and one without
  // the optional input.
  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  std::vector<Ort::Value> ort_inputs;

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  // input 0
  std::vector<float> input_0_data = {1.f};
  std::vector<int64_t> input_0_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));

  // input 1
  std::vector<float> input_1_data = {1.f};
  std::vector<int64_t> input_1_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_1_data.data()),
                                      input_1_data.size(), input_1_dims.data(), input_1_dims.size()));

  // input 2
  std::vector<float> input_2_data = {1.f};
  std::vector<int64_t> input_2_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_2_data.data()),
                                      input_2_data.size(), input_2_dims.data(), input_2_dims.size()));

  const char* output_name = "Y";

  // Part 1: Model with optional input present
  {
    std::vector<const char*> input_names = {"X1", "X2", "X3"};
    Ort::Session session(*ort_env, OPTIONAL_INPUT_OUTPUT_CUSTOM_OP_MODEL_URI, session_options);
    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                   &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);

    // Validate results
    std::vector<int64_t> y_dims = {1};
    std::vector<float> values_y = {3.f};
    auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), y_dims);
    size_t total_len = type_info.GetElementCount();
    ASSERT_EQ(values_y.size(), total_len);

    float* f = ort_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i != total_len; ++i) {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }

  // Part 2: Model with optional input absent
  {
    std::vector<const char*> input_names = {"X1", "X2"};
    ort_inputs.erase(ort_inputs.begin() + 2);  // remove the last input in the container
    Ort::Session session(*ort_env, OPTIONAL_INPUT_CUSTOM_OP_MODEL_URI_2, session_options);
    auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                   &output_name, 1);
    ASSERT_EQ(ort_outputs.size(), 1u);

    // Validate results
    std::vector<int64_t> y_dims = {1};
    std::vector<float> values_y = {2.f};
    auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), y_dims);
    size_t total_len = type_info.GetElementCount();
    ASSERT_EQ(values_y.size(), total_len);

    float* f = ort_outputs[0].GetTensorMutableData<float>();
    for (size_t i = 0; i != total_len; ++i) {
      ASSERT_EQ(values_y[i], f[i]);
    }
  }
}

TEST(CApiTest, custom_op_with_attributes_handler) {
  MyCustomOpWithAttributes custom_op{onnxruntime::kCpuExecutionProvider};

  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op);

  Ort::SessionOptions session_options;
  session_options.Add(custom_op_domain);

  Ort::Session session(*ort_env, CUSTOM_OP_MODEL_WITH_ATTRIBUTES_URI, session_options);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("X");
  std::vector<float> input_0_data = {1.f};
  std::vector<int64_t> input_0_dims = {1};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));

  // Run
  const char* output_name = "Y";
  auto ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                 &output_name, 1);
  ASSERT_EQ(ort_outputs.size(), 1u);

  // Validate results
  std::vector<int64_t> y_dims = {1};
  std::vector<float> values_y = {15.f};
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), y_dims);
  size_t total_len = type_info.GetElementCount();
  ASSERT_EQ(values_y.size(), total_len);

  float* f = ort_outputs[0].GetTensorMutableData<float>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(values_y[i], f[i]);
  }
}

// Tests registration of a custom op of the same name for both CPU and CUDA EPs
#ifdef USE_CUDA
TEST(CApiTest, RegisterCustomOpForCPUAndCUDA) {
  std::cout << "Tests registration of a custom op of the same name for both CPU and CUDA EPs" << std::endl;

  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f};

  MyCustomOp custom_op_cpu{onnxruntime::kCpuExecutionProvider};
  // We are going to test session creation only - hence it is not a problem to use the default stream as the compute stream for the custom op
  MyCustomOp custom_op_cuda{onnxruntime::kCudaExecutionProvider};
  Ort::CustomOpDomain custom_op_domain("test");
  custom_op_domain.Add(&custom_op_cpu);
  custom_op_domain.Add(&custom_op_cuda);

  TestInference<float>(*ort_env, CUSTOM_OP_MODEL_URI, inputs, "Y", expected_dims_y,
                       expected_values_y, 1, custom_op_domain, nullptr, true);
}
#endif

#if (!defined(ORT_MINIMAL_BUILD)) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
TEST(CApiTest, test_custom_op_get_const_input) {
  const auto* model_path = TSTR("testdata/test_kernel_info_get_const_input.onnx");

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("input1");
  std::vector<float> input_0_data = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<int64_t> input_0_dims = {1, 4};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));
  const char* output_name = "output";

  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_get_const_input_test_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_get_const_input_test_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_get_const_input_test_library.so");
#endif

  Ort::SessionOptions session_opts;

  session_opts.RegisterCustomOpsLibrary(lib_name);

  Ort::Session session(*ort_env, model_path, session_opts);
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
              &output_name, 1);
}
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#if defined(__ANDROID__)
// Disable on android because custom op libraries are not copied to the emulator.
TEST(CApiTest, DISABLED_test_custom_op_local_function) {
#else
TEST(CApiTest, test_custom_op_local_function) {
#endif  // defined(__ANDROID__)
  const auto* model_path = TSTR("testdata/custom_op_local_function/custom_ops_type_inference_fails_0.onnx");

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;

  // input 0 (float type)
  input_names.emplace_back("X");
  std::vector<float> input_0_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> input_0_dims = {2, 2};
  ort_inputs.emplace_back(
      Ort::Value::CreateTensor<float>(info, const_cast<float*>(input_0_data.data()),
                                      input_0_data.size(), input_0_dims.data(), input_0_dims.size()));
  const char* output_name = "Y";

  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_local_function.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_local_function.dylib");
#else
lib_name = ORT_TSTR("./libcustom_op_local_function.so");
#endif

  Ort::SessionOptions session_opts;

  session_opts.RegisterCustomOpsLibrary(lib_name);

  Ort::Session session(*ort_env, model_path, session_opts);
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(),
              &output_name, 1);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#if defined(USE_OPENVINO) && (!defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))
TEST(CApiTest, test_custom_op_openvino_wrapper_library) {
  // Tests a custom operator that wraps an OpenVINO MNIST model (.xml and .bin files serialized into node attributes).
  // The custom op extracts the serialized .xml/.bin bytes and creates an in-memory OpenVINO model
  // during kernel creation. The custom op is passed an image of a hand-drawn "1" as an input during computation, which
  // is then inferenced using OpenVINO C++ APIs.
  std::vector<Input<float>> inputs(1);
  inputs[0].name = "Input3";
  inputs[0].dims = {1, 1, 28, 28};

  // Float image with the digit "1".
  inputs[0].values = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.75f, 1.0f, 0.75f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.85f, 0.99f, 0.85f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 1.0f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 0.99f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.98f, 1.0f, 0.98f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.99f, 1.0f, 0.99f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.94f, 0.99f, 0.94f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.1f, 0.75f, 0.75f, 0.75f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  // prepare expected outputs
  std::vector<int64_t> expected_output_dims = {1, 10};

  // Digit 1 (index 1) has the highest probability (before applying softmax)
  std::vector<float> expected_vals = {-5.34957457f, 13.1904755f, -4.79670954f, -3.59232116f, 2.31260920f,
                                      -4.27866220f, -4.31867933f, 0.587718308f, -2.33952785f, -3.88515306f};

  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_openvino_wrapper_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_openvino_wrapper_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_openvino_wrapper_library.so");
#endif

  // Run with custom op session configurations.
  {
    Ort::SessionOptions session_opts;
    Ort::CustomOpConfigs custom_op_configs;

    custom_op_configs.AddConfig("OpenVINO_Wrapper", "device_type", "CPU");
    session_opts.RegisterCustomOpsLibrary(lib_name, custom_op_configs);

    Ort::Session session(*ort_env, CUSTOM_OP_OPENVINO_WRAPPER_LIB_TEST_MODEL_URI, session_opts);
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    RunSession(default_allocator.get(), session,
               inputs,
               "Plus214_Output_0",
               expected_output_dims,
               expected_vals,
               nullptr);
  }

  // Run without specifying any custom op session configurations.
  // Expect custom op to use "CPU" as OpenVINO's default backend.
  {
    Ort::SessionOptions session_opts;
    session_opts.RegisterCustomOpsLibrary(lib_name);

    Ort::Session session(*ort_env, CUSTOM_OP_OPENVINO_WRAPPER_LIB_TEST_MODEL_URI, session_opts);
    auto default_allocator = std::make_unique<MockedOrtAllocator>();

    RunSession(default_allocator.get(), session,
               inputs,
               "Plus214_Output_0",
               expected_output_dims,
               expected_vals,
               nullptr);
  }
}
#endif  // defined(USE_OPENVINO) && (!defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS))

// It has memory leak. The OrtCustomOpDomain created in custom_op_library.cc:RegisterCustomOps function was not freed
#if defined(__ANDROID__)
TEST(CApiTest, DISABLED_test_custom_op_library) {
// To accommodate a reduced op build pipeline
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, DISABLED_test_custom_op_library) {
#else
TEST(CApiTest, test_custom_op_library) {
#endif
  std::cout << "Running inference using custom op shared library" << std::endl;

  std::vector<Input<float>> inputs(2);
  inputs[0].name = "input_1";
  inputs[0].dims = {3, 5};
  inputs[0].values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  inputs[1].name = "input_2";
  inputs[1].dims = {3, 5};
  inputs[1].values = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                      10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                      5.5f, 4.4f, 3.3f, 2.2f, 1.1f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 5};
  std::vector<int32_t> expected_values_y =
      {17, 17, 17, 17, 17,
       17, 18, 18, 18, 17,
       17, 17, 17, 17, 17};

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

#ifdef USE_CUDA
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 1, nullptr, lib_name.c_str());
#elif USE_DML
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 4, nullptr, lib_name.c_str());
#else
  TestInference<int32_t>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_URI, inputs, "output", expected_dims_y,
                         expected_values_y, 0, nullptr, lib_name.c_str());
#endif
}

// Has memory leak
#if defined(__ANDROID__) || defined(ABSL_HAVE_ADDRESS_SANITIZER)
TEST(CApiTest, DISABLED_test_custom_op_shape_infer_attr) {
// To accommodate a reduced op build pipeline
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, DISABLED_test_custom_op_shape_infer_attr) {
#else
TEST(CApiTest, test_custom_op_shape_infer_attr) {
#endif
  std::vector<Input<float>> inputs(1);
  inputs[0].name = "input_0";
  inputs[0].dims = {5};
  inputs[0].values = {1.f, 2.f, 3.f, 4.f, 5.f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {5};
  std::vector<float> expected_values_y = {6.f, 12.f, 18.f, 24.f, 30.f};

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

  TestInference<float>(*ort_env, CUSTOM_OP_LIBRARY_ATTR_TESTER_URI, inputs, "output_0", expected_dims_y,
                       expected_values_y, 0, nullptr, lib_name.c_str());
}

// It has memory leak. The OrtCustomOpDomain created in custom_op_library.cc:RegisterCustomOps function was not freed
#if defined(__ANDROID__)
TEST(CApiTest, test_custom_op_library_copy_variadic) {
// To accommodate a reduced op build pipeline
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, test_custom_op_library_copy_variadic) {
#else
TEST(CApiTest, test_custom_op_library_copy_variadic) {
#endif
  std::cout << "Running inference using custom op shared library" << std::endl;

  std::vector<Input<float>> inputs(2);
  inputs[0].name = "input_0";
  inputs[0].dims = {15};
  inputs[0].values = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  inputs[1].name = "input_1";
  inputs[1].dims = {15};
  inputs[1].values = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                      10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                      5.5f, 4.4f, 3.3f, 2.2f, 1.1f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {15};
  std::vector<float> expected_values_y = inputs[1].values;

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

  TestInference<float>(*ort_env, CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_2,
                       inputs, "output_1", expected_dims_y,
                       expected_values_y, 0, nullptr, lib_name.c_str());

  inputs.push_back({});
  inputs[2].name = "input_2";
  inputs[2].dims = {15};
  inputs[2].values = {6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                      1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                      11.1f, 12.2f, 13.3f, 14.4f, 15.5f};

  expected_values_y = inputs[2].values;
  TestInference<float>(*ort_env, CUSTOM_OP_LIBRARY_COPY_TENSOR_ARRAY_3,
                       inputs, "output_2", expected_dims_y,
                       expected_values_y, 0, nullptr, lib_name.c_str());
}

#if !defined(DISABLE_FLOAT8_TYPES)

struct InputF8 {
  const char* name = nullptr;
  std::vector<int64_t> dims;
  std::vector<Ort::Float8E4M3FN_t> values;
};

// See test test_custom_op_library_float8.
#if defined(__ANDROID__)
TEST(CApiTest, DISABLED_test_custom_op_library_float8) {
#elif defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, DISABLED_test_custom_op_library_float8) {
#else
TEST(CApiTest, test_custom_op_library_float8) {
#endif
  std::cout << "Running inference using custom op shared library" << std::endl;

  std::vector<InputF8> inputs(2);
  inputs[0].name = "X";
  inputs[0].dims = {2};
  inputs[0].values = {0, 1};
  inputs[1].name = "Y";
  inputs[1].dims = {2};
  inputs[1].values = {3, 4};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {2};
  std::vector<Ort::Float8E4M3FN_t> expected_values_y = {0, 1};

  onnxruntime::PathString lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_library.dylib");
#else
  lib_name = ORT_TSTR("./libcustom_op_library.so");
#endif

  TestInference<Ort::Float8E4M3FN_t, Ort::Float8E4M3FN_t, InputF8>(*ort_env, CUSTOM_OP_LIBRARY_TEST_MODEL_FLOAT8_URI, inputs,
                                                                   "Z", expected_dims_y,
                                                                   expected_values_y, 0, nullptr, lib_name.c_str());
}

#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
#if defined(__ANDROID__)
// Disable on android because custom op libraries are not copied to the emulator.
TEST(CApiTest, DISABLED_test_custom_op_library_registration_error) {
#else
TEST(CApiTest, test_custom_op_library_registration_error) {
#endif  // defined(__ANDROID__)
  // Loads a custom op library with a RegisterCustomOps function that returns an error status.
  // This test tries to register the library with the session options and expects an error.
  const ORTCHAR_T* lib_name;
#if defined(_WIN32)
  lib_name = ORT_TSTR("custom_op_invalid_library.dll");
#elif defined(__APPLE__)
  lib_name = ORT_TSTR("libcustom_op_invalid_library.dylib");
#else
lib_name = ORT_TSTR("./libcustom_op_invalid_library.so");
#endif

  Ort::SessionOptions session_options;

  try {
    session_options.RegisterCustomOpsLibrary(lib_name);
    FAIL();
  } catch (const Ort::Exception& exception) {
    ASSERT_THAT(exception.what(), testing::HasSubstr("Failure from custom op library's RegisterCustomOps()"));
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

#if defined(ENABLE_LANGUAGE_INTEROP_OPS)
std::once_flag my_module_flag;

void PrepareModule() {
  std::ofstream module("mymodule.py");
  module << "class MyKernel:" << std::endl;
  module << "\t"
         << "def __init__(self,A,B,C):" << std::endl;
  module << "\t\t"
         << "self.a,self.b,self.c = A,B,C" << std::endl;
  module << "\t"
         << "def compute(self,x):" << std::endl;
  module << "\t\t"
         << "return x*2" << std::endl;
  module << "class MyKernel_2:" << std::endl;
  module << "\t"
         << "def __init__(self,A,B):" << std::endl;
  module << "\t\t"
         << "self.a,self.b = A,B" << std::endl;
  module << "\t"
         << "def compute(self,x):" << std::endl;
  module << "\t\t"
         << "return x*4" << std::endl;
  module << "class MyKernel_3:" << std::endl;
  module << "\t"
         << "def __init__(self,A,B):" << std::endl;
  module << "\t\t"
         << "self.a,self.b = A,B" << std::endl;
  module << "\t"
         << "def compute(self,*kwargs):" << std::endl;
  module << "\t\t"
         << "return kwargs[0]*5" << std::endl;
  module.close();
}

TEST(CApiTest, test_pyop) {
  std::call_once(my_module_flag, PrepareModule);
  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {2, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> expected_dims_y = {2, 2};
  std::vector<float> expected_values_y = {2.0f, 4.0f, 6.0f, 8.0f};
  TestInference<float>(*ort_env, PYOP_FLOAT_MODEL_URI, inputs, "Y", expected_dims_y, expected_values_y, 0,
                       nullptr, nullptr);
}

TEST(CApiTest, test_pyop_multi) {
  std::call_once(my_module_flag, PrepareModule);
  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {2, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> expected_dims_y = {2, 2};
  std::vector<float> expected_values_y = {8.0f, 16.0f, 24.0f, 32.0f};
  TestInference<float>(*ort_env, PYOP_MULTI_MODEL_URI, inputs, "Z", expected_dims_y, expected_values_y, 0,
                       nullptr, nullptr);
}

TEST(CApiTest, test_pyop_kwarg) {
  std::call_once(my_module_flag, PrepareModule);
  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "X";
  input.dims = {2, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> expected_dims_y = {2, 2};
  std::vector<float> expected_values_y = {25.0f, 50.0f, 75.0f, 100.0f};
  TestInference<float>(*ort_env, PYOP_KWARG_MODEL_URI, inputs, "Z", expected_dims_y, expected_values_y, 0,
                       nullptr, nullptr);
}
#endif

#ifdef REDUCED_OPS_BUILD
TEST(ReducedOpsBuildTest, test_excluded_ops) {
  // In reduced ops build, test a model containing ops not included in required_ops.config cannot be loaded.
  // See onnxruntime/test/testdata/reduced_build_test.readme.txt for more details of the setup
  constexpr PATH_TYPE model_uri = TSTR("testdata/reduced_build_test.onnx_model_with_excluded_ops");
  std::vector<Input<float>> inputs = {{"X", {3}, {-1.0f, 2.0f, -3.0f}}};
  std::vector<int64_t> expected_dims_y = {3};
  std::vector<float> expected_values_y = {0.1f, 0.1f, 0.1f};
  bool failed = false;
  try {
    // only test model loading, exception expected
    TestInference<float>(*ort_env, model_uri, inputs, "Y", expected_dims_y, expected_values_y, 0,
                         nullptr, nullptr, true);
  } catch (const Ort::Exception& e) {
    failed = e.GetOrtErrorCode() == ORT_NOT_IMPLEMENTED;
  }
  ASSERT_EQ(failed, true);
}
#endif

#if defined(USE_QNN)

// Returns true if QNN EP was created and QNN HTP shared memory allocator is available, false otherwise.
static bool CreateSessionWithQnnEpAndQnnHtpSharedMemoryAllocator(PATH_TYPE model_path, Ort::Session& session) {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
  constexpr bool use_htp_backend = true;
#else
  constexpr bool use_htp_backend = false;
#endif

#if defined(_WIN32)
  const char* backend_path = use_htp_backend ? "QnnHtp.dll" : "QnnCpu.dll";
#else
  const char* backend_path = use_htp_backend ? "libQnnHtp.so" : "libQnnCpu.so";
#endif

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider("QNN",
                                          {{"enable_htp_shared_memory_allocator", "1"},
                                           {"backend_path", backend_path}});

  try {
    session = Ort::Session{*ort_env, model_path, session_options};
    return true;
  } catch (const Ort::Exception& e) {
    // handle exception that indicates that the libcdsprpc.so / dll can't be loaded
    std::string_view error_message = e.what();
    std::string_view expected_error_message = "Failed to initialize RPCMEM dynamic library handle";

    if (e.GetOrtErrorCode() == ORT_FAIL &&
        error_message.find(expected_error_message) != std::string_view::npos) {
      session = Ort::Session{nullptr};
      return false;
    }

    // propagate other exceptions
    throw;
  }
}

#endif  // defined(USE_QNN)

TEST(CApiTest, get_allocator_cpu) {
  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_CPU(1);
  Ort::Session session(*ort_env, NAMED_AND_ANON_DIM_PARAM_URI, session_options);
  Ort::MemoryInfo info_cpu = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Allocator cpu_allocator(session, info_cpu);

  // CPU OrtMemoryInfo does not return OrtArenaAllocator on x86 but rather a device allocator
  // which causes MemoryInfo that is used to request the allocator and the actual instance
  // of MemoryInfo returned from the allocator exactly match, although they are functionally equivalent.
  auto allocator_info = cpu_allocator.GetInfo();
  ASSERT_EQ(info_cpu.GetAllocatorName(), allocator_info.GetAllocatorName());
  ASSERT_EQ(info_cpu.GetDeviceId(), allocator_info.GetDeviceId());
  ASSERT_EQ(info_cpu.GetMemoryType(), allocator_info.GetDeviceId());
  void* p = cpu_allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  cpu_allocator.Free(p);

  auto mem_allocation = cpu_allocator.GetAllocation(1024);
  ASSERT_NE(nullptr, mem_allocation.get());
  ASSERT_EQ(1024U, mem_allocation.size());

  Ort::KeyValuePairs stats = cpu_allocator.GetStats();

  // CPU allocator may not support arena usage.
  // See func DoesCpuAllocatorSupportArenaUsage() in allocator_utils.cc.
  if (allocator_info.GetAllocatorType() == OrtAllocatorType::OrtArenaAllocator) {
    ASSERT_EQ("-1", std::string(stats.GetValue("Limit")));
    ASSERT_EQ("1024", std::string(stats.GetValue("InUse")));
    ASSERT_EQ("1024", std::string(stats.GetValue("MaxInUse")));
    ASSERT_EQ("1024", std::string(stats.GetValue("MaxAllocSize")));
    ASSERT_EQ("2", std::string(stats.GetValue("NumAllocs")));
    ASSERT_EQ("0", std::string(stats.GetValue("NumReserves")));

    // We don't check values of the following stats
    ASSERT_NE(nullptr, stats.GetValue("TotalAllocated"));
    ASSERT_NE(nullptr, stats.GetValue("NumArenaExtensions"));
    ASSERT_NE(nullptr, stats.GetValue("NumArenaShrinkages"));
  } else {
    // If the allocator is not an arena allocator, we expect the stats to be empty.
    ASSERT_EQ(0, stats.GetKeyValuePairs().size());
  }
}

#ifdef USE_CUDA
TEST(CApiTest, get_allocator_cuda) {
  Ort::SessionOptions session_options;
  Ort::CUDAProviderOptions options;
  session_options.AppendExecutionProvider_CUDA_V2(*options);
  Ort::Session session(*ort_env, NAMED_AND_ANON_DIM_PARAM_URI, session_options);

  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator cuda_allocator(session, info_cuda);

  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);
  void* p = cuda_allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  cuda_allocator.Free(p);

  auto mem_allocation = cuda_allocator.GetAllocation(1024);
  ASSERT_NE(nullptr, mem_allocation.get());
  ASSERT_EQ(1024U, mem_allocation.size());

  Ort::KeyValuePairs stats = cuda_allocator.GetStats();

  ASSERT_EQ("-1", std::string(stats.GetValue("Limit")));
  ASSERT_EQ("1024", std::string(stats.GetValue("InUse")));
  ASSERT_EQ("1024", std::string(stats.GetValue("MaxInUse")));
  ASSERT_EQ("1024", std::string(stats.GetValue("MaxAllocSize")));
  ASSERT_EQ("2", std::string(stats.GetValue("NumAllocs")));
  ASSERT_EQ("0", std::string(stats.GetValue("NumReserves")));

  // We don't check values of the following stats
  ASSERT_NE(nullptr, stats.GetValue("TotalAllocated"));
  ASSERT_NE(nullptr, stats.GetValue("NumArenaExtensions"));
  ASSERT_NE(nullptr, stats.GetValue("NumArenaShrinkages"));
}
#endif

#if defined(USE_QNN)

TEST(CApiTest, get_allocator_qnn_htp_shared) {
  Ort::Session session{nullptr};

  if (!CreateSessionWithQnnEpAndQnnHtpSharedMemoryAllocator(NAMED_AND_ANON_DIM_PARAM_URI, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_qnn_htp_shared("QnnHtpShared", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::Allocator qnn_htp_shared_allocator(session, info_qnn_htp_shared);

  auto allocator_info = qnn_htp_shared_allocator.GetInfo();
  ASSERT_EQ(allocator_info, info_qnn_htp_shared);

  void* p = qnn_htp_shared_allocator.Alloc(1024);
  ASSERT_NE(p, nullptr);
  qnn_htp_shared_allocator.Free(p);

  auto mem_allocation = qnn_htp_shared_allocator.GetAllocation(1024);
  ASSERT_NE(mem_allocation.get(), nullptr);
  ASSERT_EQ(mem_allocation.size(), size_t{1024});
}

#endif  // defined(USE_QNN)

TEST(CApiTest, io_binding) {
  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_CPU(1);
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  Ort::MemoryInfo info_cpu = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemTypeDefault);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cpu, x_values.data(), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  const std::array<int64_t, 2> y_shape = {3, 2};
  std::array<float, 3 * 2> y_values;
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cpu, y_values.data(), y_values.size(),
                                                y_shape.data(), y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  session.Run(Ort::RunOptions(), binding);
  // Check the values against the bound raw memory
  ASSERT_TRUE(std::equal(std::begin(y_values), std::end(y_values), std::begin(expected_y)));

  // Now compare values via GetOutputValues
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();
    ASSERT_TRUE(std::equal(values, values + count, std::begin(expected_y)));
  }

  {
    std::vector<std::string> output_names = binding.GetOutputNames();
    ASSERT_EQ(1U, output_names.size());
    ASSERT_EQ(output_names[0].compare("Y"), 0);
  }

  // Now replace binding of Y with an on device binding instead of pre-allocated memory.
  // This is when we can not allocate an OrtValue due to unknown dimensions
  {
    Ort::MemoryInfo info_cpu_dev("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    binding.BindOutput("Y", info_cpu_dev);
    session.Run(Ort::RunOptions(), binding);
  }

  // Check the output value allocated based on the device binding.
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();
    ASSERT_TRUE(std::equal(values, values + count, std::begin(expected_y)));
  }

  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

#if defined(USE_CUDA) || defined(USE_TENSORRT)
TEST(CApiTest, io_binding_cuda) {
  Ort::SessionOptions session_options;
#ifdef USE_TENSORRT
  session_options.AppendExecutionProvider_TensorRT({});
#else
  Ort::CUDAProviderOptions cuda_options;
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#endif
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  Ort::Allocator cuda_allocator(session, info_cuda);
  auto allocator_info = cuda_allocator.GetInfo();
  ASSERT_TRUE(info_cuda == allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = cuda_allocator.GetAllocation(x_values.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  const std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = cuda_allocator.GetAllocation(expected_y.size() * sizeof(float));
  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  // Sychronize to make sure the copy on default stream is done since TensorRT isn't using default stream.
  binding.SynchronizeInputs();

  session.Run(Ort::RunOptions(), binding);

  binding.SynchronizeOutputs();

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values_0;
  cudaMemcpy(y_values_0.data(), output_data.get(), sizeof(float) * y_values_0.size(), cudaMemcpyDeviceToHost);
  ASSERT_TRUE(std::equal(std::begin(y_values_0), std::end(y_values_0), std::begin(expected_y)));

  // Now compare values via GetOutputValues
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();

    std::array<float, 3 * 2> y_values_1;
    cudaMemcpy(y_values_1.data(), values, sizeof(float) * y_values_1.size(), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(std::equal(std::begin(y_values_1), std::end(y_values_1), std::begin(expected_y)));
  }

  {
    std::vector<std::string> output_names = binding.GetOutputNames();
    ASSERT_EQ(1U, output_names.size());
    ASSERT_EQ(output_names[0].compare("Y"), 0);
  }

  // Now replace binding of Y with an on device binding instead of pre-allocated memory.
  // This is when we can not allocate an OrtValue due to unknown dimensions
  {
    binding.BindOutput("Y", info_cuda);
    session.Run(Ort::RunOptions(), binding);
  }

  // Check the output value allocated based on the device binding.
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);
    const float* values = Y_value.GetTensorData<float>();

    std::array<float, 3 * 2> y_values_2;
    cudaMemcpy(y_values_2.data(), values, sizeof(float) * y_values_2.size(), cudaMemcpyDeviceToHost);
    ASSERT_TRUE(std::equal(std::begin(y_values_2), std::end(y_values_2), std::begin(expected_y)));
  }

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}
#endif

#if defined(USE_QNN)

TEST(CApiTest, io_binding_qnn_htp_shared) {
  Ort::Session session{nullptr};
  if (!CreateSessionWithQnnEpAndQnnHtpSharedMemoryAllocator(MODEL_URI, session)) {
    GTEST_SKIP() << "HTP shared memory allocator is unavailable.";
  }

  Ort::MemoryInfo info_qnn_htp_shared("QnnHtpShared", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Allocator qnn_htp_shared_allocator(session, info_qnn_htp_shared);
  auto allocator_info = qnn_htp_shared_allocator.GetInfo();
  ASSERT_EQ(info_qnn_htp_shared, allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = qnn_htp_shared_allocator.GetAllocation(x_values.size() * sizeof(float));
  ASSERT_NE(input_data.get(), nullptr);
  memcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size());

  // Create an OrtValue tensor backed by data on QNN HTP shared memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_qnn_htp_shared, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  // Setup expected output (y) from model. Note that QNN EP runs float32 operators as float16,
  // so the output will not be exactly equal.
  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  const std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  constexpr float y_max_abs_err = 1e-5f;
  auto output_data = qnn_htp_shared_allocator.GetAllocation(expected_y.size() * sizeof(float));
  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on QNN HTP shared memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_qnn_htp_shared, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  session.Run(Ort::RunOptions(), binding);

  // Check the values against the bound raw memory
  {
    gsl::span y{reinterpret_cast<const float*>(output_data.get()), expected_y.size()};
    EXPECT_THAT(expected_y, ::testing::Pointwise(::testing::FloatNear(y_max_abs_err), y));
  }

  // Now compare values via GetOutputValues
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);

    gsl::span y{Y_value.GetTensorData<float>(), count};
    EXPECT_THAT(expected_y, ::testing::Pointwise(::testing::FloatNear(y_max_abs_err), y));
  }

  {
    std::vector<std::string> output_names = binding.GetOutputNames();
    ASSERT_EQ(1U, output_names.size());
    ASSERT_EQ(output_names[0].compare("Y"), 0);
  }

  // Now replace binding of Y with an on device binding instead of pre-allocated memory.
  // This is when we can not allocate an OrtValue due to unknown dimensions
  {
    binding.BindOutput("Y", info_qnn_htp_shared);
    session.Run(Ort::RunOptions(), binding);
  }

  // Check the output value allocated based on the device binding.
  {
    std::vector<Ort::Value> output_values = binding.GetOutputValues();
    ASSERT_EQ(output_values.size(), 1U);
    const Ort::Value& Y_value = output_values[0];
    ASSERT_TRUE(Y_value.IsTensor());
    Ort::TensorTypeAndShapeInfo type_info = Y_value.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, type_info.GetElementType());
    auto count = type_info.GetElementCount();
    ASSERT_EQ(expected_y.size(), count);

    gsl::span y{Y_value.GetTensorData<float>(), count};
    EXPECT_THAT(expected_y, ::testing::Pointwise(::testing::FloatNear(y_max_abs_err), y));
  }

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

#endif  // defined(USE_QNN)

#if defined(USE_CUDA) || defined(USE_TENSORRT) || defined(USE_DML)
TEST(CApiTest, basic_cuda_graph) {
  [[maybe_unused]] const auto& api = Ort::GetApi();
  Ort::SessionOptions session_options;

#if defined(USE_TENSORRT)
  // Enable cuda graph in TRT provider option.
  Ort::TensorRTProviderOptions trt_options;
  std::unordered_map<std::string, std::string> trt_options_map = {{"trt_cuda_graph_enable",
                                                                   "1"}};
  trt_options.Update(trt_options_map);
  session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);

#elif defined(USE_CUDA)
  // Enable cuda graph in cuda provider option.
  Ort::CUDAProviderOptions cuda_options;
  std::unordered_map<std::string, std::string> options_map = {{"enable_cuda_graph",
                                                               "1"}};
  cuda_options.Update(options_map);
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

#elif defined(USE_DML)
  // Enable dynamic DML graph in DML provider option.
  session_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
  const OrtDmlApi* ort_dml_api;
  Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));

  auto dml_objects = CreateDmlObjects();
  Ort::ThrowOnError(ort_dml_api->SessionOptionsAppendExecutionProvider_DML1(
      session_options, dml_objects.dml_device.Get(), dml_objects.command_queue.Get()));
#endif

  Ort::Session session(*ort_env, MODEL_URI, session_options);
#if defined(USE_CUDA) || defined(USE_TENSORRT)
  Ort::MemoryInfo info_mem("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
#elif defined(USE_DML)
  Ort::MemoryInfo info_mem("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
#endif

  Ort::Allocator allocator(session, info_mem);
  auto allocator_info = allocator.GetInfo();
  ASSERT_TRUE(info_mem == allocator_info);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto input_data = allocator.GetAllocation(x_values.size() * sizeof(float));

  ASSERT_NE(input_data.get(), nullptr);

#if defined(USE_CUDA) || defined(USE_TENSORRT)
  ASSERT_EQ(cudaSuccess, cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice));
#elif defined(USE_DML)
  ComPtr<ID3D12Resource> input_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(allocator, input_data.get(), &input_resource));
  UploadDataToDml(dml_objects, input_resource.Get(), gsl::make_span(reinterpret_cast<const std::byte*>(x_values.data()), sizeof(float) * x_values.size()));
#endif

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_mem, reinterpret_cast<float*>(input_data.get()), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  auto output_data = allocator.GetAllocation(expected_y.size() * sizeof(float));

  ASSERT_NE(output_data.get(), nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_mem, reinterpret_cast<float*>(output_data.get()),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);
  // Synchronize to make sure the input upload is complete, since it may be issued on a different stream/queue than the EP uses.
  binding.SynchronizeInputs();

  // One regular run for necessary memory allocation and graph capturing
  session.Run(Ort::RunOptions(), binding);

  // Synchronize to make sure the EP computation is complete before reading the output back to the host.
  binding.SynchronizeOutputs();

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values;

#if defined(USE_CUDA) || defined(USE_TENSORRT)
  ASSERT_EQ(cudaSuccess, cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost));
#elif defined(USE_DML)
  ComPtr<ID3D12Resource> output_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(allocator, output_data.get(), &output_resource));
  auto output_cpu_bytes = reinterpret_cast<std::byte*>(y_values.data());
  DownloadDataFromDml(dml_objects, output_resource.Get(), gsl::make_span(output_cpu_bytes, sizeof(float) * y_values.size()));
#endif

  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Replay the captured CUDA graph
  session.Run(Ort::RunOptions(), binding);

  binding.SynchronizeOutputs();

#if defined(USE_CUDA) || defined(USE_TENSORRT)
  ASSERT_EQ(cudaSuccess, cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost));
#elif defined(USE_DML)
  DownloadDataFromDml(dml_objects, output_resource.Get(), gsl::make_span(output_cpu_bytes, sizeof(float) * y_values.size()));
#endif

  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Change the input and replay the CUDA graph again.
  x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};

#if defined(USE_CUDA) || defined(USE_TENSORRT)
  ASSERT_EQ(cudaSuccess, cudaMemcpy(input_data.get(), x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice));
#elif defined(USE_DML)
  UploadDataToDml(dml_objects, input_resource.Get(), gsl::make_span(reinterpret_cast<const std::byte*>(x_values.data()), sizeof(float) * x_values.size()));
#endif

  binding.SynchronizeInputs();

  session.Run(Ort::RunOptions(), binding);

  binding.SynchronizeOutputs();

#if defined(USE_CUDA) || defined(USE_TENSORRT)
  ASSERT_EQ(cudaSuccess, cudaMemcpy(y_values.data(), output_data.get(), sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost));
#elif defined(USE_DML)
  DownloadDataFromDml(dml_objects, output_resource.Get(), gsl::make_span(output_cpu_bytes, sizeof(float) * y_values.size()));
#endif

  expected_y = {10.0f, 40.0f, 90.0f, 160.0f, 250.0f, 360.0f};
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

#if defined(USE_DML)
// Regression test for graph capture on a model that folds to an *empty* graph.
// The model below is a single Constant node, which constant-folding removes,
// leaving zero compute nodes. GenAI creates such an allocator-initialization
// session with DML graph capture enabled. Previously the graph-capture selection
// logic rejected an empty graph (because no node was assigned to the EP) and threw
// "all compute graph nodes have not been partitioned to the DmlExecutionProvider".
// An empty graph has nothing that violates the requirement, so it must succeed.
TEST(CApiTest, DmlGraphCaptureEmptyGraph) {
  // A minimal model consisting of a single Constant node producing "values".
  // Constant folding removes the node, leaving an empty graph.
  static const uint8_t constant_only_model[] = {
      0x08, 0x0a, 0x3a, 0x5b, 0x0a, 0x31, 0x12, 0x06, 0x76, 0x61, 0x6c, 0x75,
      0x65, 0x73, 0x22, 0x08, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e, 0x74,
      0x2a, 0x1d, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x2a, 0x11, 0x08,
      0x01, 0x10, 0x01, 0x22, 0x04, 0x00, 0x00, 0x80, 0x3f, 0x42, 0x05, 0x76,
      0x61, 0x6c, 0x75, 0x65, 0xa0, 0x01, 0x04, 0x12, 0x10, 0x74, 0x72, 0x69,
      0x76, 0x69, 0x61, 0x6c, 0x5f, 0x63, 0x6f, 0x6e, 0x73, 0x74, 0x61, 0x6e,
      0x74, 0x62, 0x14, 0x0a, 0x06, 0x76, 0x61, 0x6c, 0x75, 0x65, 0x73, 0x12,
      0x0a, 0x0a, 0x08, 0x08, 0x01, 0x12, 0x04, 0x0a, 0x02, 0x08, 0x01, 0x42,
      0x04, 0x0a, 0x00, 0x10, 0x0d};

  const auto& api = Ort::GetApi();
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
  const OrtDmlApi* ort_dml_api;
  Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));

  auto dml_objects = CreateDmlObjects();
  Ort::ThrowOnError(ort_dml_api->SessionOptionsAppendExecutionProvider_DML1(
      session_options, dml_objects.dml_device.Get(), dml_objects.command_queue.Get()));

  EXPECT_NO_THROW({
    Ort::Session session(*ort_env, constant_only_model, sizeof(constant_only_model), session_options);
  });
}
#endif  // defined(USE_DML)

#if defined(USE_CUDA) || defined(USE_DML)
struct CudaGraphInputOutputData_0 {
  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  std::array<float, 3 * 2> y_values;
  std::array<float, 3 * 2> new_x_values = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  std::array<float, 3 * 2> new_expected_y = {10.0f, 40.0f, 90.0f, 160.0f, 250.0f, 360.0f};
} cg_data_0;

struct CudaGraphInputOutputData_1 {
  const std::array<int64_t, 2> x_shape = {3, 1};
  std::array<float, 3> x_values = {1.0f, 3.0f, 5.0f};
  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 2.0f, 9.0f, 12.0f, 25.0f, 30.0f};

  std::array<float, 3 * 2> y_values;
  std::array<float, 3> new_x_values = {10.0f, 30.0f, 50.0f};
  std::array<float, 3 * 2> new_expected_y = {10.0f, 20.0f, 90.0f, 120.0f, 250.0f, 300.0f};
} cg_data_1;

struct CudaGraphInputOutputData_2 {
  const std::array<int64_t, 2> x_shape = {1, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f};
  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 3.0f, 8.0f, 5.0f, 12.0f};

  std::array<float, 3 * 2> y_values;
  std::array<float, 3 * 2> new_x_values = {10.0f, 20.0f};
  std::array<float, 3 * 2> new_expected_y = {10.0f, 40.0f, 30.0f, 80.0f, 50.0f, 120.0f};
} cg_data_2;

template <typename T>
static void RunWithCudaGraphAnnotation(T& cg_data,
                                       Ort::Session& session,
                                       Ort::MemoryInfo& info_mem,
#ifdef USE_DML
                                       DmlObjects& dml_objects,
#endif
                                       Ort::MemoryAllocation& input_data,
                                       Ort::MemoryAllocation& output_data,
                                       const char* cuda_graph_annotation) {
#ifdef USE_DML
  Ort::SessionOptions session_options;
  Ort::Allocator allocator(session, info_mem);
  const auto& api = Ort::GetApi();
  const OrtDmlApi* ort_dml_api;
  Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));

  ComPtr<ID3D12Resource> input_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(allocator, input_data.get(), &input_resource));
  UploadDataToDml(dml_objects, input_resource.Get(), gsl::make_span(reinterpret_cast<const std::byte*>(cg_data.x_values.data()), sizeof(float) * cg_data.x_values.size()));
#else
  (void)cudaMemcpy(input_data.get(),
                   cg_data.x_values.data(),
                   sizeof(float) * cg_data.x_values.size(),
                   cudaMemcpyHostToDevice);
#endif

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_mem,
                                                reinterpret_cast<float*>(input_data.get()),
                                                cg_data.x_values.size(),
                                                cg_data.x_shape.data(),
                                                cg_data.x_shape.size());

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_mem,
                                                reinterpret_cast<float*>(output_data.get()),
                                                cg_data.expected_y.size(),
                                                cg_data.expected_y_shape.data(),
                                                cg_data.expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  Ort::RunOptions run_option;
  if (cuda_graph_annotation != nullptr) {
    run_option.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, cuda_graph_annotation);
  }

  // Synchronize to make sure the input upload is complete, since it may be issued on a different stream/queue than the EP uses.
  binding.SynchronizeInputs();

  // One regular run for necessary memory allocation and graph capturing
  session.Run(run_option, binding);

  // Synchronize to make sure the EP computation is complete before reading the output back to the host.
  binding.SynchronizeOutputs();

  // Check the values against the bound raw memory (needs copying from device to host first)
#ifdef USE_DML
  ComPtr<ID3D12Resource> output_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(allocator, output_data.get(), &output_resource));
  auto output_cpu_bytes = reinterpret_cast<std::byte*>(cg_data.y_values.data());
  DownloadDataFromDml(dml_objects, output_resource.Get(), gsl::make_span(output_cpu_bytes, sizeof(float) * cg_data.y_values.size()));
#else
  (void)cudaMemcpy(cg_data.y_values.data(),
                   output_data.get(),
                   sizeof(float) * cg_data.y_values.size(),
                   cudaMemcpyDeviceToHost);
#endif

  ASSERT_THAT(cg_data.y_values, ::testing::ContainerEq(cg_data.expected_y));

  // Replay the captured CUDA graph
  session.Run(run_option, binding);

  binding.SynchronizeOutputs();

#ifdef USE_DML
  DownloadDataFromDml(dml_objects, output_resource.Get(), gsl::make_span(output_cpu_bytes, sizeof(float) * cg_data.y_values.size()));
#else
  (void)cudaMemcpy(cg_data.y_values.data(),
                   output_data.get(),
                   sizeof(float) * cg_data.y_values.size(),
                   cudaMemcpyDeviceToHost);
#endif
  ASSERT_THAT(cg_data.y_values, ::testing::ContainerEq(cg_data.expected_y));

  // Change the input and replay the CUDA graph again.
#ifdef USE_DML
  UploadDataToDml(dml_objects,
                  input_resource.Get(),
                  gsl::make_span(reinterpret_cast<const std::byte*>(cg_data.new_x_values.data()),
                                 sizeof(float) * cg_data.new_x_values.size()));
#else
  (void)cudaMemcpy(input_data.get(),
                   cg_data.new_x_values.data(),
                   sizeof(float) * cg_data.new_x_values.size(),
                   cudaMemcpyHostToDevice);
#endif

  binding.SynchronizeInputs();

  session.Run(run_option, binding);

  binding.SynchronizeOutputs();

#ifdef USE_DML
  DownloadDataFromDml(dml_objects, output_resource.Get(), gsl::make_span(output_cpu_bytes, sizeof(float) * cg_data.y_values.size()));
#else
  (void)cudaMemcpy(cg_data.y_values.data(),
                   output_data.get(),
                   sizeof(float) * cg_data.y_values.size(),
                   cudaMemcpyDeviceToHost);
#endif

  ASSERT_THAT(cg_data.y_values, ::testing::ContainerEq(cg_data.new_expected_y));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
}

TEST(CApiTest, basic_cuda_graph_with_annotation) {
  [[maybe_unused]] const auto& api = Ort::GetApi();
  Ort::SessionOptions session_options;

#ifdef USE_DML
  session_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
  const OrtDmlApi* ort_dml_api;
  Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));
  auto dml_objects = CreateDmlObjects();
  Ort::ThrowOnError(ort_dml_api->SessionOptionsAppendExecutionProvider_DML1(
      session_options, dml_objects.dml_device.Get(), dml_objects.command_queue.Get()));

  Ort::MemoryInfo info_mem("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
#elif defined(USE_CUDA)
  // Enable cuda graph in cuda provider option.
  Ort::CUDAProviderOptions cuda_options;
  std::unordered_map<std::string, std::string> options_map = {{"enable_cuda_graph", "1"}};
  cuda_options.Update(options_map);
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

  Ort::MemoryInfo info_mem("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
#endif

  Ort::Session session(*ort_env, CUDA_GRAPH_ANNOTATION_MODEL_URI, session_options);

  Ort::Allocator allocator(session, info_mem);
  auto allocator_info = allocator.GetInfo();
  ASSERT_TRUE(info_mem == allocator_info);

  size_t max_input_size = 6;
  size_t max_output_size = 6;

  auto input_data = allocator.GetAllocation(max_input_size * sizeof(float));
  auto output_data = allocator.GetAllocation(max_output_size * sizeof(float));

  ASSERT_NE(input_data.get(), nullptr);
  ASSERT_NE(output_data.get(), nullptr);

#ifdef USE_DML
  RunWithCudaGraphAnnotation(cg_data_0, session, info_mem, dml_objects, input_data, output_data, nullptr);
  RunWithCudaGraphAnnotation(cg_data_1, session, info_mem, dml_objects, input_data, output_data, "1");
  RunWithCudaGraphAnnotation(cg_data_2, session, info_mem, dml_objects, input_data, output_data, "2");
#else
  RunWithCudaGraphAnnotation(cg_data_0, session, info_mem, input_data, output_data, nullptr);
  RunWithCudaGraphAnnotation(cg_data_1, session, info_mem, input_data, output_data, "1");
  RunWithCudaGraphAnnotation(cg_data_2, session, info_mem, input_data, output_data, "2");
#endif
}
#endif

// The following test uses some ops not supported in the reduced ops build
#ifndef REDUCED_OPS_BUILD
#if defined(USE_CUDA) || defined(USE_TENSORRT)
TEST(CApiTest, cuda_graph_with_shape_nodes) {
  [[maybe_unused]] const auto& api = Ort::GetApi();

  // Enable cuda graph in cuda provider option.
  Ort::CUDAProviderOptions cuda_options;
  const std::unordered_map<std::string, std::string> options_map = {{"enable_cuda_graph", "1"}};
  cuda_options.Update(options_map);

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

  // Successful loading of the ONNX model with shape nodes with cuda graph feature enabled
  Ort::Session session(*ort_env, TSTR("testdata/cuda_graph_with_shape_nodes.onnx"), session_options);
}
#endif  // defined(USE_CUDA) || defined(USE_TENSORRT)

#if defined(USE_DML)
TEST(CApiTest, dml_graph_with_shape_nodes) {
  const auto& api = Ort::GetApi();

  const OrtDmlApi* ort_dml_api;
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry("ep.dml.enable_graph_capture", "1");
  Ort::ThrowOnError(api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));
  ort_dml_api->SessionOptionsAppendExecutionProvider_DML(session_options, 0);

  // Successful loading of the ONNX model with shape nodes with dml graph feature enabled
  Ort::Session session(*ort_env, TSTR("testdata/cuda_graph_with_shape_nodes.onnx"), session_options);
}
#endif  // defined(USE_DML)

#endif  // REDUCED_OPS_BUILD

#endif  // defined(USE_CUDA) || defined(USE_TENSORRT)

TEST(CApiTest, create_tensor) {
  const char* s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  Ort::ThrowOnError(Ort::GetApi().FillStringTensor(tensor, s, expected_len));
  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  const auto len = shape_info.GetElementCount();
  ASSERT_EQ(len, static_cast<size_t>(expected_len));
  std::vector<int64_t> shape_array(len);

  size_t data_len = tensor.GetStringTensorDataLength();
  std::string result(data_len, '\0');
  std::vector<size_t> offsets(len);
  tensor.GetStringTensorContent((void*)result.data(), data_len, offsets.data(), offsets.size());
}

TEST(CApiTest, fill_string_tensor) {
  const char* s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  for (size_t i = 0; i < expected_len; i++) {
    tensor.FillStringTensorElement(s[i], i);
  }

  auto shape_info = tensor.GetTensorTypeAndShapeInfo();

  const auto len = shape_info.GetElementCount();
  ASSERT_EQ(len, static_cast<size_t>(expected_len));
}

TEST(CApiTest, fill_string_tensor_directly) {
  constexpr std::string_view s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;

  MockedOrtAllocator default_allocator;
  Ort::Value tensor = Ort::Value::CreateTensor(&default_allocator, &expected_len, 1U,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  for (size_t i = 0; i < expected_len; i++) {
    auto* buffer = tensor.GetResizedStringTensorElementBuffer(i, s[i].size());
    memcpy(buffer, s[i].data(), s[i].size());
  }

  auto shape_info = tensor.GetTensorTypeAndShapeInfo();
  const auto len = shape_info.GetElementCount();
  ASSERT_EQ(len, static_cast<size_t>(expected_len));

  for (size_t i = 0; i < expected_len; i++) {
    auto element = tensor.GetStringTensorElement(i);
    ASSERT_EQ(s[i], element);
  }
}

TEST(CApiTest, get_string_tensor_element) {
  const char* s[] = {"abc", "kmp"};
  constexpr int64_t expected_len = 2;
  constexpr int64_t element_index = 0;
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  Ort::Value tensor = Ort::Value::CreateTensor(default_allocator.get(), &expected_len, 1,
                                               ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  tensor.FillStringTensor(s, expected_len);

  auto expected_string = s[element_index];
  size_t expected_string_len = strnlen(expected_string, onnxruntime::kMaxStrLen);

  std::string result(expected_string_len, '\0');
  tensor.GetStringTensorElement(expected_string_len, element_index, (void*)result.data());
  ASSERT_STREQ(result.c_str(), expected_string);

  auto string_len = tensor.GetStringTensorElementLength(element_index);
  ASSERT_EQ(expected_string_len, string_len);
}

TEST(CApiTest, create_tensor_with_data) {
  float values[] = {3.0f, 1.0f, 2.f, 0.f};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<int64_t> dims = {4};
  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values, values_length, dims.data(), dims.size());

  const float* new_pointer = tensor.GetTensorData<float>();
  ASSERT_EQ(new_pointer, values);

  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
}

TEST(CApiTest, create_tensor_with_data_float16) {
  // Example with C++. However, what we are feeding underneath is really
  // a continuous buffer of uint16_t
  // Use 3rd party libraries such as Eigen to convert floats and doubles to float16 types.
  constexpr float values[] = {1.f, 2.f, 3.f, 4.f, 5.f};
  constexpr size_t values_length = std::size(values);
  constexpr uint16_t expected_values[values_length] = {15360, 16384, 16896, 17408, 17664};

  std::vector<Ort::Float16_t> fp16_values;
  fp16_values.reserve(values_length);
  std::transform(std::begin(values), std::end(values), std::back_inserter(fp16_values),
                 [](float fl) { return Ort::Float16_t(fl); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_EQ(expected_values[i], fp16_values[i].val);
  }

  constexpr int64_t dims = static_cast<int64_t>(values_length);
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::Float16_t>(info, fp16_values.data(), values_length, &dims, 1u);
  const auto* new_pointer = tensor.GetTensorData<Ort::Float16_t>();

  ASSERT_EQ(new_pointer, fp16_values.data());
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  const auto element_count = tensor_info.GetElementCount();
  ASSERT_EQ(values_length, element_count);
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);

  const Ort::Float16_t& value_at_1 = tensor.At<Ort::Float16_t>({1});
  ASSERT_EQ(expected_values[1], value_at_1.val);

  std::vector<float> output_values;
  output_values.reserve(values_length);
  const auto data_span = gsl::make_span(tensor.GetTensorData<Ort::Float16_t>(), element_count);
  std::transform(data_span.begin(), data_span.end(), std::back_inserter(output_values),
                 [](const Ort::Float16_t& fp16) { return static_cast<float>(fp16); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_NEAR(values[i], output_values[i], 1e-3);
  }
}

TEST(CApiTest, create_tensor_with_data_bfloat16) {
  // Example with C++. However, what we are feeding underneath is really
  // a continuous buffer of uint16_t
  // Conversion from float to bfloat16 is simple. Strip off half of the bytes from float.
  constexpr float values[] = {1.f, 2.f, 3.f, 4.f, 5.f};
  constexpr size_t values_length = std::size(values);
  constexpr uint16_t expected_values[] = {16256, 16384, 16448, 16512, 16544};  // 1.f, 2.f, 3.f, 4.f, 5.f

  constexpr int64_t dims = static_cast<int64_t>(values_length);

  std::vector<Ort::BFloat16_t> b16_values;
  b16_values.reserve(values_length);
  std::transform(std::begin(values), std::end(values), std::back_inserter(b16_values),
                 [](float fl) { return Ort::BFloat16_t(fl); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_EQ(expected_values[i], b16_values[i].val);
  }

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::BFloat16_t>(info, b16_values.data(), values_length, &dims, 1u);
  const auto* new_pointer = tensor.GetTensorData<Ort::BFloat16_t>();
  ASSERT_EQ(new_pointer, b16_values.data());
  auto type_info = tensor.GetTypeInfo();
  const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  const auto element_count = tensor_info.GetElementCount();
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);

  const Ort::BFloat16_t& value_at_1 = tensor.At<Ort::BFloat16_t>({1});
  ASSERT_EQ(expected_values[1], value_at_1.val);

  std::vector<float> output_values;
  output_values.reserve(values_length);
  const auto data_span = gsl::make_span(tensor.GetTensorData<Ort::BFloat16_t>(), element_count);
  std::transform(data_span.begin(), data_span.end(), std::back_inserter(output_values),
                 [](const Ort::BFloat16_t& b16) { return static_cast<float>(b16); });

  for (size_t i = 0; i < values_length; ++i) {
    ASSERT_NEAR(values[i], output_values[i], 1e-3);
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)

TEST(CApiTest, create_tensor_with_data_float8) {
  Ort::Float8E4M3FN_t values[] = {0, 1, 2, 3, 4};
  constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
  std::vector<int64_t> dims = {static_cast<int64_t>(values_length)};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<Ort::Float8E4M3FN_t>(info, values, values_length, dims.data(), dims.size());
  const auto* new_pointer = tensor.GetTensorData<Ort::Float8E4M3FN_t>();
  ASSERT_EQ(new_pointer, values);
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(tensor_info, nullptr);
  ASSERT_EQ(1u, tensor_info.GetDimensionsCount());
  ASSERT_EQ(tensor_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN);

  Ort::Float8E4M3FN_t value_at_1 = tensor.At<Ort::Float8E4M3FN_t>({1});
  ASSERT_EQ(values[1], value_at_1);
}

#endif

// Test creating an Ort::Value with INT4 data.
TEST(CApiTest, create_tensor_with_data_int4) {
  std::array<uint8_t, 4> values = {0x10, 0x32, 0x78, 0x06};  // {0, 1, 2, 3, -8, 7, 6, pad_0}
  std::vector<int64_t> dims = {7};                           // 7 4-bit elements take up 4 bytes.

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor(info, values.data(), values.size(), dims.data(), dims.size(),
                                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4);
  const auto* new_pointer = tensor.GetTensorData<uint8_t>();
  ASSERT_EQ(new_pointer, values.data());
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(tensor_info, nullptr);
  auto query_dims = tensor_info.GetShape();
  ASSERT_EQ(query_dims, dims);
  ASSERT_EQ(tensor_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4);
}

// Test creating an Ort::Value with UINT4 data.
TEST(CApiTest, create_tensor_with_data_uint4) {
  std::array<uint8_t, 4> values = {0x10, 0x32, 0x54, 0x0F};  // {0, 1, 2, 3, 4, 5, 15, pad_0}
  std::vector<int64_t> dims = {7};                           // 7 4-bit elements take up 4 bytes.

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor(info, values.data(), values.size(), dims.data(), dims.size(),
                                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
  const auto* new_pointer = tensor.GetTensorData<uint8_t>();
  ASSERT_EQ(new_pointer, values.data());
  auto type_info = tensor.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  ASSERT_NE(tensor_info, nullptr);
  auto query_dims = tensor_info.GetShape();
  ASSERT_EQ(query_dims, dims);
  ASSERT_EQ(tensor_info.GetElementType(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
}

// Test that TensorAt rejects sub-byte packed types to prevent OOB pointer returns.
TEST(CApiTest, tensor_at_rejects_subbyte_types) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  // Int4: 7 logical elements packed into 4 bytes
  {
    std::array<uint8_t, 4> values = {0x10, 0x32, 0x78, 0x06};
    std::vector<int64_t> dims = {7};
    Ort::Value tensor = Ort::Value::CreateTensor(info, values.data(), values.size(), dims.data(), dims.size(),
                                                 ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4);
    try {
      tensor.At<uint8_t>({0});
      FAIL() << "Expected TensorAt to reject int4 sub-byte packed type";
    } catch (const Ort::Exception& excpt) {
      EXPECT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
      EXPECT_THAT(excpt.what(), testing::HasSubstr("does not support sub-byte packed types"));
    }
  }

  // UInt4: 7 logical elements packed into 4 bytes
  {
    std::array<uint8_t, 4> values = {0x10, 0x32, 0x54, 0x0F};
    std::vector<int64_t> dims = {7};
    Ort::Value tensor = Ort::Value::CreateTensor(info, values.data(), values.size(), dims.data(), dims.size(),
                                                 ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4);
    try {
      tensor.At<uint8_t>({0});
      FAIL() << "Expected TensorAt to reject uint4 sub-byte packed type";
    } catch (const Ort::Exception& excpt) {
      EXPECT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
      EXPECT_THAT(excpt.what(), testing::HasSubstr("does not support sub-byte packed types"));
    }
  }
}

// TensorAt provides direct element access for normal (non-sub-byte) types.
TEST(CApiTest, tensor_at_with_data_float) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> dims = {4};
  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values.data(), values.size(), dims.data(), dims.size());
  ASSERT_EQ(1.0f, tensor.At<float>({0}));
  ASSERT_EQ(4.0f, tensor.At<float>({3}));
}

TEST(CApiTest, tensor_at_with_data_int32) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<int32_t> values = {10, 20, 30};
  std::vector<int64_t> dims = {3};
  Ort::Value tensor = Ort::Value::CreateTensor<int32_t>(info, values.data(), values.size(), dims.data(), dims.size());
  ASSERT_EQ(10, tensor.At<int32_t>({0}));
  ASSERT_EQ(30, tensor.At<int32_t>({2}));
}

TEST(CApiTest, tensor_at_with_data_int8) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<int8_t> values = {-1, 0, 1, 127};
  std::vector<int64_t> dims = {4};
  Ort::Value tensor = Ort::Value::CreateTensor<int8_t>(info, values.data(), values.size(), dims.data(), dims.size());
  ASSERT_EQ(-1, tensor.At<int8_t>({0}));
  ASSERT_EQ(127, tensor.At<int8_t>({3}));
}

// Test that TensorAt bounds checking rejects out-of-range indices for normal types.
TEST(CApiTest, tensor_at_bounds_check) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::vector<int64_t> dims = {2, 3};
  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values.data(), values.size(), dims.data(), dims.size());

  // Valid access works
  ASSERT_EQ(1.0f, tensor.At<float>({0, 0}));
  ASSERT_EQ(6.0f, tensor.At<float>({1, 2}));

  auto expect_at_throws = [&tensor](const std::vector<int64_t>& location, const std::string& expected_message) {
    try {
      tensor.At<float>(location);
      FAIL() << "Expected TensorAt to throw for the given location";
    } catch (const Ort::Exception& excpt) {
      EXPECT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
      EXPECT_THAT(excpt.what(), testing::HasSubstr(expected_message));
    }
  };

  // Out-of-range indices throw
  expect_at_throws({2, 0}, "invalid location range");   // row out of range
  expect_at_throws({0, 3}, "invalid location range");   // col out of range
  expect_at_throws({-1, 0}, "invalid location range");  // negative index

  // Wrong number of dimensions throws
  expect_at_throws({0}, "location dimensions do not match shape size");
  expect_at_throws({0, 0, 0}, "location dimensions do not match shape size");
}

TEST(CApiTest, access_tensor_data_elements) {
  /**
   * Create a 2x3 data blob that looks like:
   *
   *  0 1 2
   *  3 4 5
   */
  std::vector<int64_t> shape = {2, 3};
  int element_count = 6;  // 2*3
  std::vector<float> values(element_count);
  for (int i = 0; i < element_count; i++)
    values[i] = static_cast<float>(i);

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  Ort::Value tensor = Ort::Value::CreateTensor<float>(info, values.data(), values.size(), shape.data(), shape.size());

  float expected_value = 0;
  for (int64_t row = 0; row < shape[0]; row++) {
    for (int64_t col = 0; col < shape[1]; col++) {
      ASSERT_EQ(expected_value++, tensor.At<float>({row, col}));
    }
  }
}

TEST(CApiTest, override_initializer) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto allocator = std::make_unique<MockedOrtAllocator>();
  // CreateTensor which is not owning this ptr
  bool Label_input[] = {true};
  std::vector<int64_t> dims = {1, 1};
  Ort::Value label_input_tensor = Ort::Value::CreateTensor<bool>(info, Label_input, 1U, dims.data(), dims.size());

  std::string f2_data{"f2_string"};
  // Place a string into Tensor OrtValue and assign to the
  Ort::Value f2_input_tensor = Ort::Value::CreateTensor(allocator.get(), dims.data(), dims.size(),
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  const char* const input_char_string[] = {f2_data.c_str()};
  f2_input_tensor.FillStringTensor(input_char_string, 1U);

  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, OVERRIDABLE_INITIALIZER_MODEL_URI, session_options);

  // Get Overrideable initializers
  size_t init_count = session.GetOverridableInitializerCount();
  ASSERT_EQ(init_count, 1U);

  {
    auto f1_init_name = session.GetOverridableInitializerNameAllocated(0, allocator.get());
    ASSERT_TRUE(strcmp("F1", f1_init_name.get()) == 0);
  }

  Ort::TypeInfo init_type_info = session.GetOverridableInitializerTypeInfo(0);
  ASSERT_EQ(ONNX_TYPE_TENSOR, init_type_info.GetONNXType());

  // Let's override the initializer
  float f11_input_data[] = {2.0f};
  Ort::Value f11_input_tensor = Ort::Value::CreateTensor<float>(info, f11_input_data, 1U, dims.data(), dims.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(label_input_tensor));
  ort_inputs.push_back(std::move(f2_input_tensor));
  ort_inputs.push_back(std::move(f11_input_tensor));

  std::vector<const char*> input_names = {"Label", "F2", "F1"};
  const char* output_names[] = {"Label0", "F20", "F11"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                    ort_inputs.data(), ort_inputs.size(),
                                                    output_names, countof(output_names));

  ASSERT_EQ(ort_outputs.size(), 3U);
  // Expecting the last output would be the overridden value of the initializer
  auto type_info = ort_outputs[2].GetTensorTypeAndShapeInfo();
  ASSERT_EQ(type_info.GetShape(), dims);
  ASSERT_EQ(type_info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(type_info.GetElementCount(), 1U);
  float* output_data = ort_outputs[2].GetTensorMutableData<float>();
  ASSERT_EQ(*output_data, f11_input_data[0]);
}

TEST(CApiTest, end_profiling) {
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto allocator = std::make_unique<MockedOrtAllocator>();

  // Create session with profiling enabled (profiling is automatically turned on)
  Ort::SessionOptions session_options_1;
#ifdef _WIN32
  session_options_1.EnableProfiling(L"profile_prefix");
#else
  session_options_1.EnableProfiling("profile_prefix");
#endif
  Ort::Session session_1(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options_1);
  {
    auto profile_file = session_1.EndProfilingAllocated(allocator.get());
    ASSERT_TRUE(std::string(profile_file.get()).find("profile_prefix") != std::string::npos);
  }
  // Create session with profiling disabled
  Ort::SessionOptions session_options_2;
#ifdef _WIN32
  session_options_2.DisableProfiling();
#else
  session_options_2.DisableProfiling();
#endif
  Ort::Session session_2(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options_2);
  {
    auto profile_file = session_2.EndProfilingAllocated(allocator.get());
    ASSERT_TRUE(std::string(profile_file.get()) == std::string());
  }
}

TEST(CApiTest, get_profiling_start_time) {
  // Test whether the C_API can access the profiler's start time
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Ort::SessionOptions session_options;
#ifdef _WIN32
  session_options.EnableProfiling(L"profile_prefix");
#else
  session_options.EnableProfiling("profile_prefix");
#endif

  uint64_t before_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::high_resolution_clock::now().time_since_epoch())
                                   .count();  // get current time
  Ort::Session session_1(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options);
  uint64_t profiling_start_time = session_1.GetProfilingStartTimeNs();
  uint64_t after_start_time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                  std::chrono::high_resolution_clock::now().time_since_epoch())
                                  .count();

  // the profiler's start time needs to be between before_time and after_time
  ASSERT_TRUE(before_start_time <= profiling_start_time && profiling_start_time <= after_start_time);
}

// Test that profiling events include memory stats (mem_bytes_in_use, mem_arena_held, etc.)
// when an arena allocator is in use.
TEST(CApiTest, profiling_memory_stats) {
  // Use add_mul_add.onnx: 3 nodes (Add, Mul, Add), inputs A[3,2] and B[3,2], output C[3,2]
  constexpr PATH_TYPE model_uri = TSTR("testdata/add_mul_add.onnx");

  Ort::SessionOptions session_options;
#ifdef _WIN32
  session_options.EnableProfiling(L"mem_profile_test");
#else
  session_options.EnableProfiling("mem_profile_test");
#endif

  Ort::Session session(*ort_env, model_uri, session_options);

  // Prepare inputs
  Ort::MemoryInfo mem_info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<float> input_data(3 * 2, 1.0f);
  std::array<int64_t, 2> input_shape = {3, 2};

  auto input_a = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(),
                                                 input_shape.data(), input_shape.size());
  auto input_b = Ort::Value::CreateTensor<float>(mem_info, input_data.data(), input_data.size(),
                                                 input_shape.data(), input_shape.size());

  // Run inference
  const char* input_names[] = {"A", "B"};
  const char* output_names[] = {"C"};
  std::array<Ort::Value, 2> inputs = {std::move(input_a), std::move(input_b)};
  auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs.data(), 2, output_names, 1);

  // End profiling and get the profile file path
  auto allocator = std::make_unique<MockedOrtAllocator>();
  auto profile_file = session.EndProfilingAllocated(allocator.get());
  std::string profile_path(profile_file.get());
  ASSERT_FALSE(profile_path.empty());

  // RAII cleanup: remove profile file regardless of test outcome
  auto cleanup = [&profile_path]() noexcept {
    if (!profile_path.empty()) {
      std::error_code ec;
      std::filesystem::remove(profile_path, ec);
    }
  };
  struct ScopeGuard {
    std::function<void()> fn;
    ~ScopeGuard() { fn(); }
  } guard{cleanup};

  // Read the profile JSON file
  std::ifstream file(profile_path);
  ASSERT_TRUE(file.is_open()) << "Could not open profile file: " << profile_path;
  std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();
  ASSERT_FALSE(content.empty()) << "Profile file is empty";

  // Memory stats are only emitted when an arena allocator is in use.
  // Arena is deterministically unavailable on 32-bit, jemalloc/mimalloc, or ASan builds.
#if defined(USE_JEMALLOC) || defined(USE_MIMALLOC)
  GTEST_SKIP() << "Arena allocator disabled (jemalloc/mimalloc build)";
#elif defined(ABSL_HAVE_ADDRESS_SANITIZER)
  GTEST_SKIP() << "Arena allocator disabled (ASan build)";
#else
  if constexpr (sizeof(void*) < 8) {
    GTEST_SKIP() << "Arena allocator disabled (32-bit build)";
  }
#endif

  // Verify memory stat keys are present in the profiling output
  EXPECT_NE(content.find("\"mem_bytes_in_use\""), std::string::npos) << "Expected mem_bytes_in_use in profiling output";
  EXPECT_NE(content.find("\"mem_bytes_requested_in_use\""), std::string::npos) << "Expected mem_bytes_requested_in_use in profiling output";
  EXPECT_NE(content.find("\"mem_requested_in_use_delta\""), std::string::npos) << "Expected mem_requested_in_use_delta in profiling output";
  EXPECT_NE(content.find("\"mem_arena_held\""), std::string::npos) << "Expected mem_arena_held in profiling output";
  EXPECT_NE(content.find("\"mem_in_use_delta\""), std::string::npos) << "Expected mem_in_use_delta in profiling output";
  EXPECT_NE(content.find("\"mem_in_use_peak\""), std::string::npos) << "Expected mem_in_use_peak in profiling output";
  EXPECT_NE(content.find("\"mem_arena_held_delta\""), std::string::npos) << "Expected mem_arena_held_delta in profiling output";
}

TEST(CApiTest, model_metadata) {
  auto allocator = std::make_unique<MockedOrtAllocator>();
  // The following all tap into the c++ APIs which internally wrap over C APIs

  // The following section tests a model containing all metadata supported via the APIs
  {
    Ort::SessionOptions session_options;
    Ort::Session session(*ort_env, MODEL_WITH_CUSTOM_MODEL_METADATA, session_options);

    // Fetch model metadata
    auto model_metadata = session.GetModelMetadata();

    {
      auto producer_name = model_metadata.GetProducerNameAllocated(allocator.get());
      ASSERT_TRUE(strcmp("Hari", producer_name.get()) == 0);
    }

    {
      auto graph_name = model_metadata.GetGraphNameAllocated(allocator.get());
      ASSERT_TRUE(strcmp("matmul test", graph_name.get()) == 0);
    }

    {
      auto domain = model_metadata.GetDomainAllocated(allocator.get());
      ASSERT_TRUE(strcmp("", domain.get()) == 0);
    }

    {
      auto description = model_metadata.GetDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("This is a test model with a valid ORT config Json", description.get()) == 0);
    }

    {
      auto graph_description = model_metadata.GetGraphDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("graph description", graph_description.get()) == 0);
    }

    int64_t version = model_metadata.GetVersion();
    ASSERT_TRUE(version == 1);

    {
      auto custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator.get());
      ASSERT_EQ(custom_metadata_map_keys.size(), 2U);
    }

    auto lookup_value_1 = model_metadata.LookupCustomMetadataMapAllocated("ort_config", allocator.get());
    ASSERT_TRUE(strcmp(lookup_value_1.get(),
                       "{\"session_options\": {\"inter_op_num_threads\": 5, \"intra_op_num_threads\": 2, "
                       "\"graph_optimization_level\": 99, \"enable_profiling\": 1}}") == 0);

    auto lookup_value_2 = model_metadata.LookupCustomMetadataMapAllocated("dummy_key", allocator.get());
    ASSERT_TRUE(strcmp(lookup_value_2.get(), "dummy_value") == 0);

    // key doesn't exist in custom metadata map
    auto lookup_value_3 = model_metadata.LookupCustomMetadataMapAllocated("key_doesnt_exist", allocator.get());
    ASSERT_TRUE(lookup_value_3 == nullptr);
  }

  // The following section tests a model with some missing metadata info
  // Adding this just to make sure the API implementation is able to handle empty/missing info
  {
    Ort::SessionOptions session_options;
    Ort::Session session(*ort_env, MODEL_URI, session_options);

    // Fetch model metadata
    auto model_metadata = session.GetModelMetadata();

    // Model description is empty
    {
      auto description = model_metadata.GetDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("", description.get()) == 0);
    }

    // Graph description is empty
    {
      auto graph_description = model_metadata.GetGraphDescriptionAllocated(allocator.get());
      ASSERT_TRUE(strcmp("", graph_description.get()) == 0);
    }

    // Model does not contain custom metadata map
    auto custom_metadata_map_keys = model_metadata.GetCustomMetadataMapKeysAllocated(allocator.get());
    ASSERT_TRUE(custom_metadata_map_keys.empty());
  }
}

TEST(CApiTest, get_available_providers) {
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  ASSERT_GT(providers.size(), 0);
  ASSERT_STREQ(providers.back().c_str(), "CPUExecutionProvider");
}

TEST(CApiTest, get_available_providers_cpp) {
  std::vector<std::string> providers = Ort::GetAvailableProviders();
  ASSERT_FALSE(providers.empty());
  ASSERT_EQ(providers.back(), "CPUExecutionProvider");

#ifdef USE_CUDA
  // CUDA EP will exist in the list but its position may vary based on other EPs included in the build
  ASSERT_TRUE(std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") != providers.end());
#endif
}

TEST(CApiTest, get_version_string_cpp) {
  auto version_string = Ort::GetVersionString();
  ASSERT_FALSE(version_string.empty());
  ASSERT_EQ(version_string, std::string(ORT_VERSION));
}

TEST(CApiTest, get_build_info_string) {
  auto build_info_string = Ort::GetBuildInfoString();
  ASSERT_FALSE(build_info_string.empty());
}

TEST(CApiTest, TestSharedAllocators) {
  // prepare inputs
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  auto allocator_for_input_memory_allocation = std::make_unique<MockedOrtAllocator>();

  // prepare expected outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Create session options and configure it appropriately
  Ort::SessionOptions session_options;
  // Turn on sharing of the allocator between sessions
  session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");

  // CASE 1: We test creating and registering an ORT-internal allocator implementation instance
  // for sharing between sessions
  {
    auto mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::ArenaCfg arena_cfg(0, -1, -1, -1);
    // This creates an ORT-internal allocator instance and registers it in the environment for sharing
    // NOTE: On x86 builds arenas are not supported and will default to using non-arena based allocator
    ort_env->CreateAndRegisterAllocator(mem_info, arena_cfg);
    // Registration is always a replace operation
    ort_env->CreateAndRegisterAllocator(mem_info, arena_cfg);

    {
      // create session 1
      Ort::Session session1(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session1,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 2
      Ort::Session session2(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session2,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);
    }

    // Remove the registered shared allocator for part 2 of this test
    // where-in we will register a custom allocator for the same device.
    ort_env->UnregisterAllocator(mem_info);
  }

  // CASE 2: We test registering a custom allocator implementation
  // for sharing between sessions
  {
    // This creates a custom  allocator instance and registers it in the environment for sharing
    // NOTE: This is a very basic allocator implementation. For optimal performance, allocations
    // need to be aligned for certain devices/build configurations/math libraries.
    // See docs/C_API.md for details.
    MockedOrtAllocator custom_allocator;
    ort_env->RegisterAllocator(&custom_allocator);
    // Registration is always a replace operation
    ort_env->RegisterAllocator(&custom_allocator);
    {
      // Keep this scoped to destroy the underlying sessions after use
      // This should trigger frees in our custom allocator

      // create session 1
      Ort::Session session1(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session1,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 2
      Ort::Session session2(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session2,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 3 to test separate allocation for initializers
      session_options.AddConfigEntry("session.use_device_allocator_for_initializers", "1");
      Ort::Session session3(*ort_env, MODEL_URI, session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session3,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);
    }

    // Remove the registered shared allocator from the global environment
    // (common to all tests) to prevent its accidental usage elsewhere
    ort_env->UnregisterAllocator(custom_allocator.Info());

    // Ensure that the registered custom allocator was indeed used for both sessions
    // We should have seen 2 allocations per session (one for the sole initializer
    // and one for the output). So, for two sessions, we should have seen 4 allocations.
    size_t num_allocations = custom_allocator.NumAllocations();
    ASSERT_TRUE(num_allocations == 6);

    size_t num_reserve_allocations = custom_allocator.NumReserveAllocations();
    ASSERT_EQ(3, num_reserve_allocations);

    // Ensure that there was no leak
    custom_allocator.LeakCheck();
  }
#ifdef USE_CUDA
  {
    auto cuda_meminfo = Ort::MemoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);

    Ort::ArenaCfg arena_cfg(0, -1, -1, -1);
    ort_env->CreateAndRegisterAllocatorV2(onnxruntime::kCudaExecutionProvider,
                                          cuda_meminfo, {}, arena_cfg);

    // Registration is always a replace operation
    ort_env->CreateAndRegisterAllocatorV2(onnxruntime::kCudaExecutionProvider,
                                          cuda_meminfo, {}, arena_cfg);

    {
      // create session 1
      Ort::SessionOptions cuda_session_options;
      cuda_session_options.AddConfigEntry(kOrtSessionOptionsConfigUseEnvAllocators, "1");
      cuda_session_options.AppendExecutionProvider_CUDA(OrtCUDAProviderOptions{});
      Ort::Session session1(*ort_env, MODEL_URI, cuda_session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session1,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);

      // create session 2
      Ort::Session session2(*ort_env, MODEL_URI, cuda_session_options);
      RunSession<float>(allocator_for_input_memory_allocation.get(),
                        session2,
                        inputs,
                        "Y",
                        expected_dims_y,
                        expected_values_y,
                        nullptr);
    }

    ort_env->UnregisterAllocator(cuda_meminfo);
  }
#endif
}

TEST(CApiTest, TestSharingOfInitializerAndItsPrepackedVersion) {
  // simple inference test
  // prepare inputs
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 1};
  std::vector<float> expected_values_y = {4.0f, 10.0f, 16.0f};

  Ort::SessionOptions session_options;
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  // These values are different from the actual initializer values in the model
  float data[] = {2.0f, 1.0f};
  constexpr int data_len = sizeof(data) / sizeof(data[0]);
  const int64_t shape[] = {2, 1};
  constexpr size_t shape_len = sizeof(shape) / sizeof(shape[0]);
  Ort::Value val = Ort::Value::CreateTensor<float>(mem_info, data, data_len, shape, shape_len);
  session_options.AddInitializer("W", val);

  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  // create session 1 (using model path)
  Ort::PrepackedWeightsContainer prepacked_weights_container;
  Ort::Session session1(*ort_env, MATMUL_MODEL_URI, session_options, prepacked_weights_container);
  RunSession<float>(default_allocator.get(),
                    session1,
                    inputs,
                    "Y",
                    expected_dims_y,
                    expected_values_y,
                    nullptr);

  // create session 2 (using model bytes)
  std::ifstream model_file_stream(MATMUL_MODEL_URI, std::ios::in | std::ios::binary);
  ASSERT_TRUE(model_file_stream.good());

  model_file_stream.seekg(0, std::ios::end);
  const auto size = onnxruntime::narrow<size_t>(model_file_stream.tellg());
  model_file_stream.seekg(0, std::ios::beg);
  std::vector<char> file_contents(size, 0);
  model_file_stream.read(&file_contents[0], size);
  model_file_stream.close();

  Ort::Session session2(*ort_env, file_contents.data(), size, session_options, prepacked_weights_container);
  RunSession<float>(default_allocator.get(),
                    session2,
                    inputs,
                    "Y",
                    expected_dims_y,
                    expected_values_y,
                    nullptr);
}

#ifndef ORT_NO_RTTI
TEST(CApiTest, TestIncorrectInputTypeToModel_Tensors) {
  // simple inference test
  // prepare inputs (incorrect type)
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  double data[] = {2., 1., 4., 3., 6., 5.};
  constexpr int data_len = sizeof(data) / sizeof(data[0]);
  const int64_t shape[] = {3, 2};
  constexpr size_t shape_len = sizeof(shape) / sizeof(shape[0]);
  Ort::Value val = Ort::Value::CreateTensor<double>(mem_info, data, data_len, shape, shape_len);

  std::vector<const char*> input_names{"X"};
  const char* output_names[] = {"Y"};
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, MODEL_URI, session_options);
  bool exception_thrown = false;
  try {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &val, 1, output_names, 1);
  } catch (const Ort::Exception& ex) {
    exception_thrown = true;
    const char* exception_string = ex.what();
    ASSERT_TRUE(strcmp(exception_string,
                       "Unexpected input data type. Actual: (tensor(double)) , expected: (tensor(float))") == 0);
  }

  ASSERT_TRUE(exception_thrown);
}
TEST(CApiTest, TestIncorrectInputTypeToModel_SequenceTensors) {
  // simple inference test
  // prepare inputs (incorrect type)
  Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  double data[] = {2., 1., 4., 3., 6., 5.};
  constexpr int data_len = sizeof(data) / sizeof(data[0]);
  const int64_t shape[] = {2, 3};
  constexpr size_t shape_len = sizeof(shape) / sizeof(shape[0]);
  Ort::Value val = Ort::Value::CreateTensor<double>(mem_info, data, data_len, shape, shape_len);

  std::vector<Ort::Value> seq;
  seq.push_back(std::move(val));

  Ort::Value seq_value = Ort::Value::CreateSequence(seq);

  std::vector<const char*> input_names{"X"};
  const char* output_names[] = {"Y"};
  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, SEQUENCE_MODEL_URI, session_options);
  bool exception_thrown = false;
  try {
    auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &seq_value, 1, output_names, 1);
  } catch (const Ort::Exception& ex) {
    exception_thrown = true;
    const char* exception_string = ex.what();
    ASSERT_TRUE(strcmp(exception_string,
                       "Unexpected input data type. Actual: (seq(double)) , expected: (seq(float))") == 0);
  }

  ASSERT_TRUE(exception_thrown);
}
#endif

TEST(CApiTest, AllocateInitializersFromNonArenaMemory) {
  Ort::SessionOptions session_options;

#ifdef USE_CUDA
  Ort::CUDAProviderOptions options;
  session_options.AppendExecutionProvider_CUDA_V2(*options);
#else
  // arena is enabled but the sole initializer will still be allocated from non-arena memory
  session_options.AppendExecutionProvider_CPU(1);
#endif

  // disable using arena for the sole initializer in the model
  session_options.AddConfigEntry(kOrtSessionOptionsUseDeviceAllocatorForInitializers, "1");

  // This is mostly an usage example - if the logging level for the default logger is made INFO (by default it is at WARNING)
  // when the Ort::Env instance is instantiated, logs pertaining to initializer memory being allocated from non-arena memory
  // can be confirmed by seeing logs like "Reserving memory in BFCArena...".
  Ort::Session session(*ort_env, MODEL_URI, session_options);
}

#ifdef USE_CUDA

// Usage example showing how to use CreateArenaCfgV2() API to configure the default memory CUDA arena allocator
TEST(CApiTest, ConfigureCudaArenaAndDemonstrateMemoryArenaShrinkage) {
  Ort::SessionOptions session_options;

  const std::unordered_map<std::string, size_t> config_map = {
      {"max_mem", 0},                              // let ort pick default max memory
      {"arena_extend_strategy", 0},                // use default extend strategy
      {"initial_chunk_size_bytes", 1024},          // initial chunk size in bytes
      {"max_dead_bytes_per_chunk", 0},             // no dead bytes per chunk
      {"initial_growth_chunk_size_bytes", 256},    // initial growth chunk size in bytes
      {"max_power_of_two_extend_bytes", 1L << 24}  // max power of two extend bytes
  };

  Ort::ArenaCfg arena_cfg(config_map);

  OrtCUDAProviderOptions cuda_provider_options = CreateDefaultOrtCudaProviderOptionsWithCustomStream(nullptr);
  cuda_provider_options.default_memory_arena_cfg = arena_cfg;

  session_options.AppendExecutionProvider_CUDA(cuda_provider_options);

  Ort::Session session(*ort_env, MODEL_URI, session_options);

  // Use a run option like this while invoking Run() to trigger a memory arena shrinkage post Run()
  // This will shrink memory allocations left unused at the end of Run() and cap the arena growth
  // This does come with associated costs as there are costs to cudaFree() but the goodness it offers
  // is that the memory held by the arena (memory pool) is kept checked.
  Ort::RunOptions run_option;
  run_option.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "gpu:0");

  // To also trigger a cpu memory arena shrinkage along with the gpu arena shrinkage, use the following-
  // (Memory arena for the CPU should not have been disabled)
  //  run_option.AddConfigEntry(kOrtRunOptionsConfigEnableMemoryArenaShrinkage, "cpu:0;gpu:0");
}
#endif

#ifdef USE_TENSORRT
TEST(TensorrtExecutionProviderTest, ShapeTensorTest) {
  // Test input tensor which is shape tensor with explicit trt profile shapes
  Ort::SessionOptions session_options;
  Ort::TensorRTProviderOptions trt_options;

  std::unordered_map<std::string, std::string> trt_options_map = {
      {"trt_profile_min_shapes", "data:2x2,shape:4x1"},
      {"trt_profile_max_shapes", "data:2x2,shape:4x1"},
      {"trt_profile_opt_shapes", "data:2x2,shape:4x1"}};
  trt_options.Update(trt_options_map);
  session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);

  auto model_path = ORT_TSTR("testdata/trt_reshape.onnx");

  std::vector<float> input_value_0{1.1f, 1.2f, 1.3f, 1.4f};
  std::vector<int64_t> input_shape_0{2, 2};
  std::vector<int64_t> input_value_1{4, 1};
  std::vector<int64_t> input_shape_1{2};

  std::vector<const char*> input_names{"data", "shape"};
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, input_value_0.data(), input_value_0.size(), input_shape_0.data(), input_shape_0.size()));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(info, input_value_1.data(), input_value_1.size(), input_shape_1.data(), input_shape_1.size()));

  const char* output_names[] = {"reshaped"};

  Ort::Session session(*ort_env, model_path, session_options);
  session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));

  // Test input tensor which is shape tensor with implicit trt profile shapes
  Ort::SessionOptions session_options_2;
  Ort::TensorRTProviderOptions trt_options_2;
  session_options_2.AppendExecutionProvider_TensorRT_V2(*trt_options_2);
  Ort::Session session_2(*ort_env, model_path, session_options_2);
  session_2.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, std::size(output_names));
}

TEST(CApiTest, TestExternalCUDAStreamWithIOBinding) {
  Ort::SessionOptions session_options;
  Ort::TensorRTProviderOptions trt_options;

  // updating provider option with user provided compute stream
  cudaStream_t compute_stream = nullptr;
  cudaStreamCreate(&compute_stream);
  trt_options.UpdateWithValue("user_compute_stream", compute_stream);
  void* user_compute_stream = trt_options.GetOptionByName("user_compute_stream");
  ASSERT_TRUE(user_compute_stream == (void*)compute_stream);

  session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);

  Ort::Session session(*ort_env, MODEL_URI, session_options);
  Ort::MemoryInfo info_cuda("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

  const std::array<int64_t, 2> x_shape = {3, 2};
  std::array<float, 3 * 2> x_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  /*
   * Use cudaMallocHost() (pinned memory allocation) to create input/output tensors
   */
  float* input_data;
  cudaMallocHost(&input_data, 3 * 2 * sizeof(float));
  ASSERT_NE(input_data, nullptr);
  cudaMemcpy(input_data, x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  std::cout << "pinned memory allocation" << std::endl;
  std::cout << "input tesnor:" << std::endl;
  for (int i = 0; i < 6; i++) {
    std::cout << input_data[i] << std::endl;
  }

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data), x_values.size(),
                                                x_shape.data(), x_shape.size());

  const std::array<int64_t, 2> expected_y_shape = {3, 2};
  std::array<float, 3 * 2> expected_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  float* output_data;
  cudaMallocHost(&output_data, 3 * 2 * sizeof(float));
  ASSERT_NE(output_data, nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data),
                                                expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding(session);
  binding.BindInput("X", bound_x);
  binding.BindOutput("Y", bound_y);

  /*
   * Use cudaMalloc() (pageable memory allocation first and then implicit pinned memory allocation) to create input/output tensors
   */
  float* input_data_2;
  cudaMalloc(&input_data_2, 3 * 2 * sizeof(float));
  ASSERT_NE(input_data_2, nullptr);
  cudaMemcpy(input_data_2, x_values.data(), sizeof(float) * x_values.size(), cudaMemcpyHostToDevice);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_x_2 = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(input_data_2), x_values.size(),
                                                  x_shape.data(), x_shape.size());

  float* output_data_2;
  cudaMalloc(&output_data_2, 3 * 2 * sizeof(float));
  ASSERT_NE(output_data_2, nullptr);

  // Create an OrtValue tensor backed by data on CUDA memory
  Ort::Value bound_y_2 = Ort::Value::CreateTensor(info_cuda, reinterpret_cast<float*>(output_data_2),
                                                  expected_y.size(), expected_y_shape.data(), expected_y_shape.size());

  // Create IoBinding for inputs and outputs.
  Ort::IoBinding binding_2(session);
  binding_2.BindInput("X", bound_x_2);
  binding_2.BindOutput("Y", bound_y_2);

  // Run with first iobindings
  session.Run(Ort::RunOptions(), binding);

  // Check the values against the bound raw memory (needs copying from device to host first)
  std::array<float, 3 * 2> y_values;
  cudaMemcpy(y_values.data(), output_data, sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);

  std::cout << "pinned memory allocation" << std::endl;
  std::cout << "output: " << std::endl;
  for (auto y : y_values) {
    std::cout << y << std::endl;
  }
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Run with second iobindings
  session.Run(Ort::RunOptions(), binding_2);

  // Check the values against the bound raw memory (needs copying from device to host first)
  cudaMemcpy(y_values.data(), output_data_2, sizeof(float) * y_values.size(), cudaMemcpyDeviceToHost);

  std::cout << "pageable memory allocation" << std::endl;
  std::cout << "output: " << std::endl;
  for (auto y : y_values) {
    std::cout << y << std::endl;
  }
  ASSERT_THAT(y_values, ::testing::ContainerEq(expected_y));

  // Clean up
  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
  binding_2.ClearBoundInputs();
  binding_2.ClearBoundOutputs();

  cudaFreeHost(input_data);
  cudaFreeHost(output_data);
  cudaFree(input_data_2);
  cudaFree(output_data_2);
  cudaStreamDestroy(compute_stream);
}

class CApiTensorRTTest : public testing::Test, public ::testing::WithParamInterface<std::string> {};

// This test uses CreateTensorRTProviderOptions/UpdateTensorRTProviderOptions APIs to configure and create a TensorRT Execution Provider
TEST_P(CApiTensorRTTest, TestConfigureTensorRTProviderOptions) {
  std::string param = GetParam();
  size_t pos = param.find("=");
  const std::string option_name = param.substr(0, pos);
  const std::string option_value = param.substr(pos + 1);
  ASSERT_NE(pos, std::string::npos);

  const char* engine_cache_path = "./trt_engine_folder";

  Ort::TensorRTProviderOptions trt_options;
  std::unordered_map<std::string, std::string> trt_options_map = {
      {"device_id", "0"},
      {"has_user_compute_stream", "0"},
      {"trt_fp16_enable", "1"},
      {"trt_int8_enable", "0"},
      {"trt_engine_cache_enable", "1"},
      {"trt_engine_cache_path", engine_cache_path},
      {option_name, option_value}};

  trt_options.Update(trt_options_map);

  std::string trt_options_str = trt_options.GetTensorRTProviderOptionsAsString();
  ASSERT_NE(trt_options_str.find(engine_cache_path), std::string::npos);
  ASSERT_NE(trt_options_str.find(param), std::string::npos);

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_TensorRT_V2(*trt_options);

  // simple inference test
  // prepare inputs
  std::vector<Input<float>> inputs(1);
  auto& input = inputs.back();
  input.name = "X";
  input.dims = {3, 2};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 2};
  std::vector<float> expected_values_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  std::basic_string<ORTCHAR_T> model_uri = MODEL_URI;

  // if session creation passes, model loads fine
  Ort::Session session(*ort_env, model_uri.c_str(), session_options);
  auto default_allocator = std::make_unique<MockedOrtAllocator>();

  // without preallocated output tensor
  RunSession(default_allocator.get(),
             session,
             inputs,
             "Y",
             expected_dims_y,
             expected_values_y,
             nullptr);

  struct stat buffer;
  ASSERT_TRUE(stat(engine_cache_path, &buffer) == 0);
}

/*
 * The TensorrtExecutionProviderOptionsTest can be used to test TRT options
 */
INSTANTIATE_TEST_SUITE_P(CApiTensorRTTest, CApiTensorRTTest,
                         ::testing::Values("trt_build_heuristics_enable=1", "trt_sparsity_enable=1", "trt_builder_optimization_level=0", "trt_tactic_sources=-CUDNN,+CUBLAS", "trt_auxiliary_streams=2", "trt_bf16_enable=1"));
#endif

#ifdef USE_CUDA

// This test uses CreateCUDAProviderOptions/UpdateCUDAProviderOptions/UpdateCUDAProviderOptionsWithValue APIs to configure and create a CUDA Execution Provider instance
TEST(CApiTest, TestConfigureCUDAProviderOptions) {
  Ort::CUDAProviderOptions cuda_options;

  // Only test updating OrtCUDAProviderOptionsV2 instance with user provided compute stream not running the inference
  cudaStream_t compute_stream = nullptr;
  void* user_compute_stream = nullptr;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  cuda_options.UpdateWithValue("user_compute_stream", compute_stream);
  user_compute_stream = cuda_options.GetOptionByName("user_compute_stream");
  ASSERT_TRUE(user_compute_stream == (void*)compute_stream);
  cudaStreamDestroy(compute_stream);

  std::unordered_map<std::string, std::string> cuda_options_map = {
      {"device_id", "0"},
      {"has_user_compute_stream", "0"},
      {"gpu_mem_limit", "1024"},
      {"arena_extend_strategy", "kSameAsRequested"},
      {"cudnn_conv_algo_search", "DEFAULT"},
      {"do_copy_in_default_stream", "1"},
      {"cudnn_conv_use_max_workspace", "1"},
      {"cudnn_conv1d_pad_to_nc1d", "1"}};

  cuda_options.Update(cuda_options_map);

  std::string s = cuda_options.GetCUDAProviderOptionsAsString();
  ASSERT_TRUE(s.find("device_id=0") != std::string::npos);
  ASSERT_TRUE(s.find("gpu_mem_limit=1024") != std::string::npos);
  ASSERT_TRUE(s.find("arena_extend_strategy=kSameAsRequested") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv_algo_search=DEFAULT") != std::string::npos);
  ASSERT_TRUE(s.find("do_copy_in_default_stream=1") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv_use_max_workspace=1") != std::string::npos);
  ASSERT_TRUE(s.find("cudnn_conv1d_pad_to_nc1d") != std::string::npos);

  Ort::SessionOptions session_options;
  session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);

  // if session creation passes, model loads fine
  std::basic_string<ORTCHAR_T> model_uri = MODEL_URI;
  Ort::Session session(*ort_env, model_uri.c_str(), session_options);
}

#endif

namespace TestPerSessionCustomThreadHooks {

std::vector<std::thread> threads;
int32_t custom_thread_creation_options = 5;
int32_t custom_creation_hook_called = 0;
int32_t custom_join_hook_called = 0;

OrtCustomThreadHandle CreateThreadCustomized(void* options, OrtThreadWorkerFn work_loop, void* param) {
  if (*((int32_t*)options) == 5) {
    custom_creation_hook_called += 1;
  }
  threads.push_back(std::thread(work_loop, param));
  return reinterpret_cast<OrtCustomThreadHandle>(threads.back().native_handle());
}

void JoinThreadCustomized(OrtCustomThreadHandle handle) {
  for (auto& t : threads) {
    if (reinterpret_cast<OrtCustomThreadHandle>(t.native_handle()) == handle) {
      custom_join_hook_called += 1;
      t.join();
    }
  }
}

TEST(CApiTest, TestPerSessionCustomThreadPoolHooks) {
  constexpr int32_t thread_count = 3;
  Ort::SessionOptions session_options;
  // test both intra and inter op thread pool
  session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
  session_options.SetIntraOpNumThreads(thread_count);
  session_options.SetInterOpNumThreads(thread_count);
  session_options.SetCustomCreateThreadFn(CreateThreadCustomized);
  session_options.SetCustomThreadCreationOptions(&custom_thread_creation_options);
  session_options.SetCustomJoinThreadFn(JoinThreadCustomized);
  {
    Ort::Session session(*ort_env, MODEL_URI, session_options);
  }
  ASSERT_TRUE(custom_creation_hook_called == (thread_count - 1) << 1);
  ASSERT_TRUE(custom_join_hook_called == (thread_count - 1) << 1);
}

// Preventing resize transformer issue:
// https://github.com/microsoft/onnxruntime/issues/9857
#ifndef REDUCED_OPS_BUILD
TEST(CApiTest, crop_and_resize) {
  std::vector<float> input_value_0;
  input_value_0.resize(2 * 36 * 36 * 3);
  for (ptrdiff_t i = 0; i < 36 * 36 * 3; ++i) {
    input_value_0[i] = 1.f;
    input_value_0[i + 36 * 36 * 3] = 2.f;
  }
  std::vector<int64_t> input_shape_0{2, 36, 36, 3};

  std::vector<int32_t> input_value_1{1, 0};
  std::vector<int64_t> input_shape_1{2};

  std::vector<const char*> input_names{"input:0", "input2:0"};
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, input_value_0.data(), input_value_0.size(), input_shape_0.data(), input_shape_0.size()));
  ort_inputs.emplace_back(Ort::Value::CreateTensor<int32_t>(info, input_value_1.data(), input_value_1.size(), input_shape_1.data(), input_shape_1.size()));

  Ort::SessionOptions session_options;
  Ort::Session session(*ort_env, RESIZE_AND_CROP_MODEL_URI, session_options);

  const char* output_names[] = {"output:0"};
  std::vector<int64_t> output_shape{2, 20, 20, 3};

  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{}, input_names.data(), ort_inputs.data(), ort_inputs.size(), output_names, countof(output_names));
  ASSERT_EQ(ort_outputs.size(), 1U);
  const auto& output_0 = ort_outputs[0];
  ASSERT_TRUE(output_0.IsTensor());

  auto output_type_shape = output_0.GetTensorTypeAndShapeInfo();
  ASSERT_EQ(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, output_type_shape.GetElementType());
  ASSERT_EQ(output_shape, output_type_shape.GetShape());
}
#endif

}  // namespace TestPerSessionCustomThreadHooks

#ifdef USE_CUDA
TEST(CApiTest, GitHubIssue10179) {
  // https://github.com/microsoft/onnxruntime/issues/10179
  // the issue was caused by a race condition in CUDAExecutionProvider::GetKernelRegistry()
  // if the test runs to completion, consider that run successful
  auto load_model_thread_fn = []() {
    try {
      const auto* model_path = MODEL_URI;
      Ort::SessionOptions session_options;
      Ort::CUDAProviderOptions options;
      session_options.AppendExecutionProvider_CUDA_V2(*options);
      Ort::Session session{*ort_env, model_path, session_options};
    } catch (const std::exception& e) {
      std::cerr << "exception: " << e.what() << "\n";
      throw e;
    }
  };

  constexpr int num_threads = 4;
  constexpr int num_iterations = 10;

  for (int i = 0; i < num_iterations; ++i) {
    std::vector<std::thread> threads(num_threads);
    for (auto& thread : threads) {
      thread = std::thread{load_model_thread_fn};
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }
}

#endif

// Reduced Ops build doesn't support If (16) yet
#if !defined(REDUCED_OPS_BUILD) && defined(USE_CUDA)
TEST(CApiTest, TestCudaMemcpyToHostWithSequenceTensors) {
  const auto* model_path = SEQUENCE_MODEL_URI_2;
  Ort::SessionOptions session_options{};
  Ort::CUDAProviderOptions options;
  session_options.AppendExecutionProvider_CUDA_V2(*options);
  Ort::Session session{*ort_env, model_path, session_options};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names{"cond"};
  bool input_data[] = {false};
  std::vector<int64_t> input_dims{};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<bool>(info, input_data, 1U, input_dims.data(), 0));
  const char* output_names[] = {"sequence"};

  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                                                    ort_inputs.data(), ort_inputs.size(),
                                                    output_names, countof(output_names));

  // There is no need to check the contents of the output, we are just checking to see if the
  // model runs without crashing
}

#endif

// Reduced Ops build doesn't support OptionalHasElement (16) yet
#if !defined(REDUCED_OPS_BUILD) && !defined(DISABLE_OPTIONAL_TYPE)
TEST(CApiTest, GH_11717) {
  const auto* model_path = TSTR("testdata/gh_issue_11717.onnx");

  Ort::SessionOptions session_options{};
  // Just check if the model loads fine without a segmentation fault
  // in the default CPU EP
  EXPECT_NO_THROW(Ort::Session session(*ort_env, model_path, session_options));
}
#endif

#ifndef REDUCED_OPS_BUILD
TEST(CApiTest, TestMultiStreamInferenceSimpleSSD) {
  Ort::SessionOptions session_options{};
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
  session_options.AddConfigEntry("session.node_partition_config_file",
                                 "./testdata/multi_stream_models/simplified_ssd_cpu.csv");
  Ort::Session session{*ort_env, SIMPLIFIED_SSD_MODEL_URI, session_options};
  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  std::vector<Ort::Value> ort_inputs;
  const char* input_names[] = {"graph_in"};
  std::unique_ptr<float[]> input_data = std::make_unique<float[]>(3 * 3 * 300 * 300);
  for (int i = 0; i < 3 * 3 * 300 * 300; ++i) {
    input_data[i] = 1.f;
  }
  int64_t input_dims[] = {3, 3, 300, 300};
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(info, input_data.get(), 3 * 3 * 300 * 300, input_dims, 4U));
  const char* output_names[] = {"graph_out"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names,
                                                    ort_inputs.data(), ort_inputs.size(),
                                                    output_names, countof(output_names));
  ASSERT_TRUE(ort_outputs.size() == 1);
  ASSERT_TRUE(ort_outputs[0].IsTensor());
  const auto& type_shape_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> output_dims = type_shape_info.GetShape();
  std::vector<int64_t> expected_output_dims = {3, 256, 150, 150};
  ASSERT_TRUE(output_dims == expected_output_dims);
}
#endif

#if !defined(REDUCED_OPS_BUILD) && !defined(DISABLE_OPTIONAL_TYPE)

TEST(LiteCustomOpTest, CustomFunc) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session{*ort_env, TSTR("testdata/fuse_select_filter.onnx"), session_options};

  const char* input_names[] = {"vector_1", "vector_2", "alpha", "indices"};
  const char* output_names[] = {"vector_filtered"};

  float vector_1_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
  int64_t vector_1_dim[] = {10};

  float vector_2_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
  int64_t vector_2_dim[] = {6};

  int32_t alpha_value[] = {2};
  int64_t alpha_dim[] = {1};

  int32_t indices_value[] = {0, 1, 2, 3, 4, 5};
  int64_t indices_dim[] = {6};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, 10, vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, 6, vector_2_dim, 1),
      Ort::Value::CreateTensor<int32_t>(memory_info, alpha_value, 1, alpha_dim, 1),
      Ort::Value::CreateTensor<int32_t>(memory_info, indices_value, 6, indices_dim, 1)};

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 4, output_names, 1);
  const auto& vector_filterred = output_tensors.at(0);
  auto type_shape_info = vector_filterred.GetTensorTypeAndShapeInfo();
  const float* floats_output = static_cast<const float*>(vector_filterred.GetTensorRawData());
  ASSERT_TRUE(floats_output[0] == 8);
  ASSERT_TRUE(floats_output[1] == 16);
}

TEST(LiteCustomOpTest, CustomFuncOpsetMismatch) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  EXPECT_THROW(Ort::Session(*ort_env, TSTR("testdata/fuse_select_filter_opset_8.onnx"), session_options), std::exception);
}

struct Merge {
  Merge(const OrtApi* ort_api, const OrtKernelInfo* info) {
    int64_t reverse;
    ORT_ENFORCE(ort_api->KernelInfoGetAttribute_int64(info, "reverse", &reverse) == nullptr);
    reverse_ = reverse != 0;
  }
  Ort::Status Compute(const Ort::Custom::Tensor<std::string_view>& strings_in,
                      std::string_view string_in,
                      Ort::Custom::Tensor<std::string>* strings_out) {
    if (strings_in.NumberOfElement() == 0) {
      return Ort::Status("the 1st input must have more than one string!", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    std::vector<std::string> string_pool;
    for (const auto& s : strings_in.Data()) {
      string_pool.emplace_back(s.data(), s.size());
    }
    string_pool.emplace_back(string_in.data(), string_in.size());
    if (reverse_) {
      for (auto& str : string_pool) {
        std::reverse(str.begin(), str.end());
      }
      std::reverse(string_pool.begin(), string_pool.end());
    }
    strings_out->SetStringOutput(string_pool, {static_cast<int64_t>(string_pool.size())});
    return Ort::Status(nullptr);
  }
  static Ort::Status InferOutputShape(Ort::ShapeInferContext& ctx) {
    auto input_count = ctx.GetInputCount();
    if (input_count != 2) {
      return Ort::Status("input count should be 2", OrtErrorCode::ORT_INVALID_ARGUMENT);
    }
    Ort::ShapeInferContext::Shape shape_1 = {{-1}};
    ctx.SetOutputShape(0, shape_1);
    return Ort::Status(nullptr);
  }
  bool reverse_ = false;
};

TEST(LiteCustomOpTest, CustomStruct) {
  const auto& ortApi = Ort::GetApi();

  Ort::CustomOpDomain v2_domain{"v2"};
  std::unique_ptr<Ort::Custom::OrtLiteCustomOp> mrg_op_ptr{Ort::Custom::CreateLiteCustomOp<Merge>("Merge", "CPUExecutionProvider")};
  v2_domain.Add(mrg_op_ptr.get());

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.Add(v2_domain);
  session_options.SetLogSeverityLevel(0);

  Ort::Session session{*ort_env, TSTR("testdata/merge.onnx"), session_options};

  const char* input_names[] = {"str_in_1", "str_in_2"};
  const char* output_names[] = {"str_out"};

  OrtAllocator* allocator = nullptr;
  ASSERT_TRUE(!ortApi.GetAllocatorWithDefaultOptions(&allocator));
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  int64_t str_1_dims[] = {2};
  int64_t str_2_dims[] = {1};

  Ort::Value input_tensors[] = {Ort::Value::CreateTensor(allocator, str_1_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING),
                                Ort::Value::CreateTensor(allocator, str_2_dims, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)};

  const char* str_1_raw[] = {"abc", "de"};
  const char* str_2_raw[] = {"fg"};

  input_tensors[0].FillStringTensor(str_1_raw, 2);
  input_tensors[1].FillStringTensor(str_2_raw, 1);

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 2, output_names, 1);
  const auto& str_out_tensor = output_tensors.at(0);
  auto num_chars = str_out_tensor.GetStringTensorDataLength();
  std::vector<char> chars(num_chars + 1, '\0');
  std::vector<size_t> offsets(3);
  str_out_tensor.GetStringTensorContent(static_cast<void*>(chars.data()), num_chars, offsets.data(), offsets.size());
  ASSERT_TRUE(strncmp(chars.data(), "gfedcba", 7) == 0);
}

TEST(LiteCustomOpTest, MissingOptional) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session(*ort_env, TSTR("testdata/optional_2.onnx"), session_options);

  const char* input_names[] = {"float_in_1", "float_in_2"};
  const char* output_names[] = {"float_out_1"};

  float vector_1_value[] = {0.f, 1.f, 2.f};
  int64_t vector_1_dim[] = {3};

  float vector_2_value[] = {4.f, 5.f, 6.f, 7.f};
  int64_t vector_2_dim[] = {4};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, gsl::narrow_cast<size_t>(vector_1_dim[0]),
                                      vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, gsl::narrow_cast<size_t>(vector_2_dim[0]),
                                      vector_2_dim, 1)};

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 2, output_names, 1);
  ASSERT_TRUE(output_tensors.size() == 1);
}

TEST(LiteCustomOpTest, HasOptional) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session(*ort_env, TSTR("testdata/optional_3.onnx"), session_options);

  const char* input_names[] = {"float_in_1", "float_in_2", "float_in_3"};
  const char* output_names[] = {"float_out_1", "float_out_2"};

  float vector_1_value[] = {0.f, 1.f, 2.f};
  int64_t vector_1_dim[] = {3};

  float vector_2_value[] = {4.f, 5.f, 6.f, 7.f};
  int64_t vector_2_dim[] = {4};

  float vector_3_value[] = {8.f, 9.f};
  int64_t vector_3_dim[] = {2};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[] = {
      Ort::Value::CreateTensor<float>(memory_info, vector_1_value, gsl::narrow_cast<size_t>(vector_1_dim[0]),
                                      vector_1_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_2_value, gsl::narrow_cast<size_t>(vector_2_dim[0]),
                                      vector_2_dim, 1),
      Ort::Value::CreateTensor<float>(memory_info, vector_3_value, gsl::narrow_cast<size_t>(vector_3_dim[0]),
                                      vector_3_dim, 1),
  };

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 3, output_names, 2);
  ASSERT_TRUE(output_tensors.size() == 2);
}

TEST(MultiKernelSingleSchemaTest, valid) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
#if defined(_WIN32)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("custom_op_library.dll"));
#elif defined(__APPLE__)
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("libcustom_op_library.dylib"));
#else
  session_options.RegisterCustomOpsLibrary(ORT_TSTR("./libcustom_op_library.so"));
#endif

  Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options);

  const char* input_names[] = {"X"};
  const char* output_names[] = {"Y", "Z"};
  float x_value[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f};
  int64_t x_dim[] = {10};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[1] = {
      Ort::Value::CreateTensor<float>(memory_info, x_value, 10, x_dim, 1),
  };

  Ort::RunOptions run_options;
  auto output_tensors = session.Run(run_options, input_names, input_tensors, 1, output_names, 2);
  ASSERT_TRUE(*output_tensors[1].GetTensorData<int32_t>() == 72);
}

// expect input count mismatch exception
TEST(MultiKernelSingleSchemaTest, InputCountMismatch) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32;
  MulTopOpDouble mul_top_double;

  v2_domain.Add(&mul_top_f32);
  v2_domain.Add(&mul_top_double);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options), std::exception);
}

// expect output count mismatch exception
TEST(MultiKernelSingleSchemaTest, OutputMismatch) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32;
  MulTopOpInt16 mul_top_int64;

  v2_domain.Add(&mul_top_f32);
  v2_domain.Add(&mul_top_int64);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options), std::exception);
}

// expect characteristic mismatch exception
TEST(MultiKernelSingleSchemaTest, CharacterMismatch) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32;
  MulTopOpFloat16 mul_top_f16;

  v2_domain.Add(&mul_top_f32);
  v2_domain.Add(&mul_top_f16);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options), std::exception);
}

TEST(MultiKernelSingleSchemaTest, DuplicateKernel) {
  Ort::CustomOpDomain v2_domain("v2");
  MulTopOpFloat mul_top_f32_1;
  MulTopOpFloat mul_top_f32_2;
  MulTopOpInt32 mul_top_i32;

  v2_domain.Add(&mul_top_f32_1);
  v2_domain.Add(&mul_top_f32_2);
  v2_domain.Add(&mul_top_i32);

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_options.SetLogSeverityLevel(0);
  session_options.Add(v2_domain);

  EXPECT_NO_THROW(Ort::Session session(*ort_env, CUSTOM_OP_SINGLE_SCHEMA_MULTI_KERNEL, session_options));
}

#endif

static std::thread::id caller_tid = std::this_thread::get_id();
static std::atomic_bool atomic_wait{false};

void CallbackSucceed(void* user_data, OrtValue** outputs, size_t num_outputs, OrtStatusPtr status_ptr) {
  auto callee_tid = std::this_thread::get_id();
  EXPECT_NE(*(reinterpret_cast<std::thread::id*>(user_data)), callee_tid);
  Ort::Status status(status_ptr);
  EXPECT_TRUE(status.IsOK());
  EXPECT_EQ(num_outputs, 1UL);
  Ort::Value output_value(outputs[0]);
  EXPECT_EQ(output_value.At<float>({1, 0}), 9.f);
  output_value.release();
  atomic_wait.store(true);
}

TEST(CApiTest, RunAsync) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(2);
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  const char* input_names[] = {"X"};
  float x_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t x_dim[] = {3, 2};
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[1] = {
      Ort::Value::CreateTensor<float>(memory_info, x_value, 6, x_dim, 2),
  };

  const char* output_names[] = {"Y"};
  Ort::RunOptions run_options;
  Ort::Value output_values[1] = {Ort::Value{nullptr}};

  EXPECT_NO_THROW(session.RunAsync(run_options,
                                   input_names,
                                   input_tensors,
                                   1,
                                   output_names,
                                   output_values,
                                   1,
                                   CallbackSucceed,
                                   &caller_tid));

  std::chrono::duration<double, std::milli> dur{100};
  // timeout in about 10 secs
  for (int i = 0; i < 100 && !atomic_wait.load(); ++i) {
    std::this_thread::sleep_for(dur);
  }

  EXPECT_EQ(atomic_wait.load(), true);
}

void CallbackFail(void*, OrtValue**, size_t, OrtStatusPtr) {
  EXPECT_TRUE(false);  // the callback is not supposed to be invoked
}

TEST(CApiTest, RunAsyncFail) {
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);  // This will cause RunAsync fail
  Ort::Session session(*ort_env, MODEL_URI, session_options);

  const char* input_names[] = {"X"};
  float x_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int64_t x_dim[] = {3, 2};

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensors[1] = {
      Ort::Value::CreateTensor<float>(memory_info, x_value, 6, x_dim, 2),
  };
  Ort::Value output_values[1] = {Ort::Value{nullptr}};
  const char* output_names[] = {"Y"};

  Ort::RunOptions run_options;
  EXPECT_THROW(session.RunAsync(run_options, input_names, input_tensors, 1, output_names, output_values, 1, CallbackFail, nullptr), std::exception);
}

static void TestRunWithLoraAdapter(const Ort::LoraAdapter& adapter) {
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/lora/two_params_lora_model.onnx");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);

  Ort::RunOptions run_options;
  run_options.AddActiveLoraAdapter(adapter);

  // Single input
  constexpr const std::array<int64_t, 2> input_shape = {4, 4};
  std::vector<float> input_x(16);
  std::fill(input_x.begin(), input_x.end(), 1.0f);
  constexpr const char* input_names[] = {"input_x"};
  constexpr const char* output_names[] = {"output"};

  auto cpu_meminfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  auto input_x_val = Ort::Value::CreateTensor(
      cpu_meminfo, input_x.data(), input_x.size(), input_shape.data(), input_shape.size());

  Ort::Value inputs[] = {std::move(input_x_val)};

  Ort::SessionOptions default_session_options;

  constexpr const std::array<float, 16> expected_output = {
      154.f, 176.f, 198.f, 220.f,
      154.f, 176.f, 198.f, 220.f,
      154.f, 176.f, 198.f, 220.f,
      154.f, 176.f, 198.f, 220.f};

  Ort::Session session(env, model_path, default_session_options);

  auto outputs = session.Run(run_options, input_names, inputs, std::size(input_names), output_names, std::size(output_names));
  ASSERT_EQ(1U, outputs.size());

  auto tensor_type_shape = outputs[0].GetTensorTypeAndShapeInfo();
  const auto elements = tensor_type_shape.GetElementCount();
  ASSERT_EQ(expected_output.size(), elements);
  const float* data = outputs[0].GetTensorData<float>();
  for (size_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected_output[i], data[i], 0.06);
  }
}

static Ort::LoraAdapter CreateAdapterFromFile() {
  constexpr const ORTCHAR_T* adapter_path = TSTR("testdata/lora/two_params_lora_model.onnx_adapter");
  return Ort::LoraAdapter::CreateLoraAdapter(adapter_path, nullptr);
}

static Ort::LoraAdapter CreateAdapterFromArray() {
  constexpr const ORTCHAR_T* adapter_path = TSTR("testdata/lora/two_params_lora_model.onnx_adapter");
  std::ifstream adapter_file(adapter_path, std::ios::binary);

  EXPECT_TRUE(adapter_file.is_open());
  adapter_file.seekg(0, std::ios::end);
  const size_t adapter_size = onnxruntime::narrow<size_t>(adapter_file.tellg());

  std::vector<uint8_t> buffer(adapter_size);
  adapter_file.seekg(0, std::ios::beg);
  adapter_file.read(reinterpret_cast<char*>(buffer.data()), adapter_size);
  adapter_file.close();

  return Ort::LoraAdapter::CreateLoraAdapterFromArray(buffer.data(), buffer.size(), nullptr);
}

TEST(CApiTest, RunWithLoraAdapterFromFile) {
  auto adapter = CreateAdapterFromFile();
  TestRunWithLoraAdapter(adapter);
}

TEST(CApiTest, RunWithLoraAdapterFromArray) {
  auto adapter = CreateAdapterFromArray();
  TestRunWithLoraAdapter(adapter);
}

TEST(CApiTest, RunBaseLoraModel) {
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/lora/two_params_lora_model.onnx");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  constexpr const std::array<int64_t, 2> input_shape = {4, 4};
  std::vector<float> input_x(16);
  std::fill(input_x.begin(), input_x.end(), 1.0f);
  constexpr const char* input_names[] = {"input_x"};
  constexpr const char* output_names[] = {"output"};

  auto cpu_meminfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  auto input_x_val = Ort::Value::CreateTensor(
      cpu_meminfo, input_x.data(), input_x.size(), input_shape.data(), input_shape.size());

  Ort::Value inputs[] = {std::move(input_x_val)};

  Ort::SessionOptions default_session_options;

  constexpr const std::array<float, 16> expected_output = {
      28.f, 32.f, 36.f, 40.f,
      28.f, 32.f, 36.f, 40.f,
      28.f, 32.f, 36.f, 40.f,
      28.f, 32.f, 36.f, 40.f};

  Ort::Session session(env, model_path, default_session_options);

  Ort::RunOptions run_options;
  auto outputs = session.Run(run_options, input_names, inputs, std::size(input_names), output_names, std::size(output_names));
  ASSERT_EQ(1U, outputs.size());

  auto tensor_type_shape = outputs[0].GetTensorTypeAndShapeInfo();
  const auto elements = tensor_type_shape.GetElementCount();
  ASSERT_EQ(expected_output.size(), elements);
  const float* data = outputs[0].GetTensorData<float>();
  for (size_t i = 0; i < elements; ++i) {
    EXPECT_NEAR(expected_output[i], data[i], 0.06);
  }
}

TEST(CApiTest, RequestLoadCancellation) {
  constexpr const ORTCHAR_T* model_path = ORT_TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING);
  Ort::SessionOptions session_options;
  session_options.SetLoadCancellationFlag(true);

  bool terminated = false;
  try {
    Ort::Session session(env, model_path, session_options);
  } catch (const Ort::Exception& ex) {
    terminated = OrtErrorCode::ORT_MODEL_LOAD_CANCELED == ex.GetOrtErrorCode();
  }
  ASSERT_TRUE(terminated);
}

struct MockGQA : public OrtCustomOp {
  MockGQA() {
    OrtCustomOp::GetMayInplace = [](int** input_index, int** output_index) {
      size_t ret = 2;
      *input_index = static_cast<int*>(malloc(ret * sizeof(int)));
      (*input_index)[0] = 3;
      (*input_index)[1] = 4;
      *output_index = static_cast<int*>(malloc(ret * sizeof(int)));
      (*output_index)[0] = 1;
      (*output_index)[1] = 2;
      return ret;
    };
    OrtCustomOp::ReleaseMayInplace = [](int* input_index, int* output_index) {
      free(input_index);
      free(output_index);
    };
    OrtCustomOp::GetAliasMap = [](int** input_index, int** output_index) {
      size_t ret = 2;
      *input_index = static_cast<int*>(malloc(ret * sizeof(int)));
      (*input_index)[0] = 5;
      (*input_index)[1] = 6;
      *output_index = static_cast<int*>(malloc(ret * sizeof(int)));
      (*output_index)[0] = 7;
      (*output_index)[1] = 8;
      return ret;
    };
    OrtCustomOp::ReleaseAliasMap = [](int* input_index, int* output_index) {
      free(input_index);
      free(output_index);
    };
  }
};

TEST(CApiTest, OrtCustomOp_GetInPlace) {
  MockGQA mock_gqa;
  int* input_index = nullptr;
  int* output_index = nullptr;
  size_t len = mock_gqa.GetMayInplace(&input_index, &output_index);
  ASSERT_NE(input_index, nullptr);
  ASSERT_NE(output_index, nullptr);
  ASSERT_EQ(input_index[0], 3);
  ASSERT_EQ(input_index[1], 4);
  ASSERT_EQ(output_index[0], 1);
  ASSERT_EQ(output_index[1], 2);
  ASSERT_EQ(len, static_cast<size_t>(2));
  mock_gqa.ReleaseMayInplace(input_index, output_index);

  input_index = output_index = nullptr;
  len = mock_gqa.GetAliasMap(&input_index, &output_index);
  ASSERT_NE(input_index, nullptr);
  ASSERT_NE(output_index, nullptr);
  ASSERT_EQ(input_index[0], 5);
  ASSERT_EQ(input_index[1], 6);
  ASSERT_EQ(output_index[0], 7);
  ASSERT_EQ(output_index[1], 8);
  ASSERT_EQ(len, static_cast<size_t>(2));
  mock_gqa.ReleaseAliasMap(input_index, output_index);
}

#if !defined(ORT_MINIMAL_BUILD) && defined(USE_CUDA)

TEST(CApiTest, GenerateNodeStatsFile) {
  Ort::Env env(ORT_LOGGING_LEVEL_INFO);
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/transformers/tiny_gpt2_beamsearch.onnx");

  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsCollectNodeMemoryStatsToFile,
                                 "tiny_gpt2_beamsearch_node_stats.txt");

  OrtCUDAProviderOptionsV2 cuda_options;
  cuda_options.use_tf32 = false;
  session_options.AppendExecutionProvider_CUDA_V2(cuda_options);

  std::vector<int64_t> input_ids_shape{3, 12};
  std::vector<int32_t> input_ids{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328};

  std::vector<int64_t> parameter_shape{1};
  std::vector<int32_t> max_length{20};
  std::vector<int32_t> min_length{1};
  std::vector<int32_t> num_beams{4};
  std::vector<int32_t> num_return_sequences{1};
  std::vector<float> length_penalty{1.0f};
  std::vector<float> repetition_penalty{1.0f};

  std::vector<int64_t> expected_output_shape{input_ids_shape[0], num_return_sequences[0], max_length[0]};
  std::vector<int32_t> expected_output{
      0, 0, 0, 0, 0, 52, 195, 731, 321, 301, 734, 620, 131, 131, 131, 181, 638, 638, 638, 638,
      41, 554, 74, 622, 206, 222, 75, 223, 221, 198, 224, 572, 292, 292, 292, 292, 292, 292, 292, 292,
      0, 0, 0, 52, 328, 219, 328, 206, 288, 227, 896, 328, 328, 669, 669, 669, 669, 669, 669, 669};

  Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  auto input_ids_tensor = Ort::Value::CreateTensor(
      info, input_ids.data(), input_ids.size(), input_ids_shape.data(), input_ids_shape.size());

  auto max_length_tensor = Ort::Value::CreateTensor(
      info, max_length.data(), max_length.size(), parameter_shape.data(), parameter_shape.size());

  auto min_length_tensor = Ort::Value::CreateTensor(
      info, min_length.data(), min_length.size(), parameter_shape.data(), parameter_shape.size());

  auto num_beams_tensor = Ort::Value::CreateTensor(
      info, num_beams.data(), num_beams.size(), parameter_shape.data(), parameter_shape.size());

  auto num_return_sequences_tensor = Ort::Value::CreateTensor(
      info, num_return_sequences.data(), num_return_sequences.size(), parameter_shape.data(), parameter_shape.size());

  auto length_penalty_tensor = Ort::Value::CreateTensor(
      info, length_penalty.data(), length_penalty.size(), parameter_shape.data(), parameter_shape.size());

  auto repetition_penalty_tensor = Ort::Value::CreateTensor(
      info, repetition_penalty.data(), repetition_penalty.size(), parameter_shape.data(), parameter_shape.size());

  std::vector<Ort::Value> ort_inputs;
  ort_inputs.push_back(std::move(input_ids_tensor));
  ort_inputs.push_back(std::move(max_length_tensor));
  ort_inputs.push_back(std::move(min_length_tensor));
  ort_inputs.push_back(std::move(num_beams_tensor));
  ort_inputs.push_back(std::move(num_return_sequences_tensor));
  ort_inputs.push_back(std::move(length_penalty_tensor));
  ort_inputs.push_back(std::move(repetition_penalty_tensor));
  const char* input_names[] = {"input_ids", "max_length", "min_length", "num_beams", "num_return_sequences",
                               "length_penalty", "repetition_penalty"};
  const char* const output_names[] = {"sequences"};

  // The ONNX model is generated like the following:
  // python convert_generation.py --model_type gpt2 -m hf-internal-testing/tiny-random-gpt2
  //        --output tiny_gpt2_beamsearch_fp16.onnx --use_gpu --max_length 20
  // (with separate_gpt2_decoder_for_init_run set to False as it is now set to True by default)
  Ort::Session session(env, model_path, session_options);
  session.Run(Ort::RunOptions{}, input_names, ort_inputs.data(), ort_inputs.size(),
              output_names, 1);
}

#endif

// Test that creates a custom Cast kernel which requires type inference of the output type to work.
// Also demonstrates overriding an ONNX operator as we register the custom op in the ONNX domain.
TEST(CApiTest, custom_cast) {
  std::vector<Input<float>> inputs(1);
  auto& input = inputs[0];
  input.name = "input";
  input.dims = {3, 4};
  input.values = {1.0f, 2.0f, 3.0f, 4.0f,
                  -1.0f, -2.0f, -3.0f, -4.0f,
                  1.0f, 2.0f, 3.0f, 4.0f};

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_y = {3, 4};
  std::vector<double> expected_values_y = {1.0, 2.0, 3.0, 4.0,
                                           -1.0, -2.0, -3.0, -4.0,
                                           1.0, 2.0, 3.0, 4.0};

  CustomCast custom_op{onnxruntime::kCpuExecutionProvider};

  Ort::CustomOpDomain custom_op_domain("");  // onnx domain is empty string
  custom_op_domain.Add(&custom_op);

  // model with Cast from ONNX test data
  TestInference<double, float>(*ort_env, TSTR("testdata/cast_float_to_double.onnx"),
                               inputs, "output", expected_dims_y, expected_values_y, 0,
                               custom_op_domain, nullptr);
}

TEST(CApiTest, ModelWithMaliciousExternalDataInMemoryShouldFailToLoad) {
  // Attempt to create an ORT session with the malicious model
  // This should fail due to the invalid external in-memory reference
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/test_evil_weights.onnx");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  bool exception_thrown = false;
  std::string exception_message;

  try {
    // This should throw an exception due to malicious external data
    Ort::Session session(env, model_path, session_options);
  } catch (const Ort::Exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  // Verify that loading the model failed
  EXPECT_TRUE(exception_thrown) << "Expected model loading to fail due to malicious in-memory data";

  // Verify that the exception message indicates security or external data issues
  EXPECT_TRUE(exception_message.find("in-memory") != std::string::npos ||
              exception_message.find("references") != std::string::npos ||
              exception_message.find("invalid") != std::string::npos ||
              exception_message.find("model") != std::string::npos)
      << "Exception message should indicate external data or security issue. Got: " << exception_message;
}

TEST(CApiTest, ModelWithExternalDataOutsideModelDirectoryShouldFailToLoad) {
  // Attempt to create an ORT session with the malicious model
  // This should fail due to the external file that is not under model directory structure
  // i.e. ../../../../etc/passwd
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/test_arbitrary_external_file.onnx");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  bool exception_thrown = false;
  std::string exception_message;

  try {
    // This should throw an exception due to malicious external data
    Ort::Session session(env, model_path, session_options);
  } catch (const Ort::Exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  // Verify that loading the model failed
  EXPECT_TRUE(exception_thrown) << "Expected model loading to fail due to malicious external data";

  // Verify that the exception message indicates security or external data issues
  EXPECT_TRUE(exception_message.find("External data path escapes model directory") != std::string::npos ||
              exception_message.find("invalid") != std::string::npos ||
              exception_message.find("model") != std::string::npos)
      << "Exception message should indicate external data or security issue. Got: " << exception_message;
}

TEST(CApiTest, InMemoryModel_ExternalDataOutsideWorkingDirectory_FailToLoad) {
  // Attempt to create an ORT session with the malicious model (loaded from bytes).
  // This should fail due to the use of an external file path that is not under current working directory.
  // i.e. ../../../../etc/passwd
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/test_arbitrary_external_file.onnx");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;

  // Load model contents into array
  std::ifstream model_file_stream(model_path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(model_file_stream.good());
  model_file_stream.seekg(0, std::ios::end);
  const auto file_contents_size = onnxruntime::narrow<size_t>(model_file_stream.tellg());
  model_file_stream.seekg(0, std::ios::beg);
  std::vector<char> file_contents(file_contents_size, 0);
  model_file_stream.read(&file_contents[0], file_contents_size);
  model_file_stream.close();

  bool exception_thrown = false;
  std::string exception_message;

  try {
    // This should throw an exception due to malicious external data
    Ort::Session session(env, file_contents.data(), file_contents_size, session_options);
  } catch (const Ort::Exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  // Verify that loading the model failed
  EXPECT_TRUE(exception_thrown) << "Expected model loading to fail due to malicious external data path";

  // Verify that the exception message indicates security or external data issues
  EXPECT_TRUE(exception_message.find("External data path") != std::string::npos &&
              exception_message.find("escapes working directory") != std::string::npos)
      << "Exception message should indicate external data or security issue. Got: " << exception_message;
}

TEST(CApiTest, InMemoryModel_SessionConfigExternalFileFolder_ExternalDataOutsideModelDirectory_FailToLoad) {
  // Attempt to create an ORT session with the malicious model (loaded from bytes).
  // A valid external file folder path is explicitly set via session options.
  // However, this should still fail due to the use of an external file path that escapes the set directory.
  // i.e. ../../../../etc/passwd
  constexpr const ORTCHAR_T* model_path = TSTR("testdata/test_arbitrary_external_file.onnx");

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  Ort::SessionOptions session_options;
  session_options.AddConfigEntry(kOrtSessionOptionsModelExternalInitializersFileFolderPath, "testdata");

  // Load model contents into array
  std::ifstream model_file_stream(model_path, std::ios::in | std::ios::binary);
  ASSERT_TRUE(model_file_stream.good());
  model_file_stream.seekg(0, std::ios::end);
  const auto file_contents_size = onnxruntime::narrow<size_t>(model_file_stream.tellg());
  model_file_stream.seekg(0, std::ios::beg);
  std::vector<char> file_contents(file_contents_size, 0);
  model_file_stream.read(&file_contents[0], file_contents_size);
  model_file_stream.close();

  bool exception_thrown = false;
  std::string exception_message;

  try {
    // This should throw an exception due to malicious external data
    Ort::Session session(env, file_contents.data(), file_contents_size, session_options);
  } catch (const Ort::Exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  } catch (const std::exception& e) {
    exception_thrown = true;
    exception_message = e.what();
  }

  // Verify that loading the model failed
  EXPECT_TRUE(exception_thrown) << "Expected model loading to fail due to malicious external data path";

  // Verify that the exception message indicates security or external data issues
  EXPECT_TRUE(exception_message.find("External data path") != std::string::npos &&
              exception_message.find("escapes model directory") != std::string::npos)
      << "Exception message should indicate external data or security issue. Got: " << exception_message;
}

#ifdef ORT_ENABLE_STREAM
#if USE_CUDA

namespace {
struct TestCudaStreamOverrideUsed : onnxruntime::Stream {
  TestCudaStreamOverrideUsed(onnxruntime::Stream* stream)
      : onnxruntime::Stream(stream->GetHandle(), stream->GetDevice()), real_stream(stream) {}

  std::unique_ptr<onnxruntime::synchronize::Notification> CreateNotification(size_t num_consumers) override {
    return real_stream->CreateNotification(num_consumers);
  }

  TestCudaStreamOverrideUsed(const TestCudaStreamOverrideUsed&) = delete;
  TestCudaStreamOverrideUsed& operator=(const TestCudaStreamOverrideUsed&) = delete;

  void Flush() override {
    flush_count++;
    real_stream->Flush();
  }

  onnxruntime::Status CleanUpOnRunEnd() override { return real_stream->CleanUpOnRunEnd(); }

  onnxruntime::Stream* real_stream;
  size_t flush_count{0};
};
}  // namespace

TEST(CApiTest, TestSyncStreamOverride) {
#ifdef _WIN32
  auto cuda_lib = ORT_TSTR("onnxruntime_providers_cuda.dll");
#else
  auto cuda_lib = ORT_TSTR("onnxruntime_providers_cuda.so");
#endif

  if (!std::filesystem::exists(cuda_lib)) {
    GTEST_SKIP() << "CUDA library was not found";
  }

  constexpr const char* cuda_ep_name = "ORT Cuda";
  ort_env->RegisterExecutionProviderLibrary(cuda_ep_name, cuda_lib);
  auto ep_devices = ort_env->GetEpDevices();

  Ort::ConstEpDevice cuda_device;
  for (const auto& device : ep_devices) {
    if (device.Device().Type() == OrtHardwareDeviceType_GPU &&
        device.Device().VendorId() == 0x10DE) {  // NVIDIA vendor ID
      cuda_device = device;
      break;
    }
  }

  if (!cuda_device) {
    GTEST_SKIP() << "No CUDA device found, skipping test.";
  }

  // Create session with CUDA EP using C++ public API in Ort:: namespace
  {
    // Create a stream on CUDA Device
    const auto sync_stream = cuda_device.CreateSyncStream();
    TestCudaStreamOverrideUsed cuda_override_stream(sync_stream);

    Ort::SessionOptions session_options;
    session_options.AppendExecutionProvider_V2(*ort_env, {cuda_device}, Ort::KeyValuePairs{});

    Ort::Session session(*ort_env, MODEL_URI, session_options);

    constexpr const std::array<const char*, 1U> input_names = {"X"};
    constexpr const std::array<const char*, 1U> output_names = {"Y"};
    constexpr const std::array<int64_t, 2U> input_shape = {3LL, 2LL};
    float x_value[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto input_value = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        x_value, std::size(x_value), input_shape.data(), input_shape.size());
    Ort::Value ort_inputs[] = {std::move(input_value)};

    Ort::RunOptions run_options;
    run_options.SetSyncStream(reinterpret_cast<OrtSyncStream*>(&cuda_override_stream));

    auto output_values = session.Run(run_options,
                                     input_names.data(), ort_inputs, std::size(ort_inputs),
                                     output_names.data(), output_names.size());

    ASSERT_GT(cuda_override_stream.flush_count, 0U)
        << "Expected the custom CUDA stream override to be used during session run.";
  }

  ort_env->UnregisterExecutionProviderLibrary(cuda_ep_name);
}
#endif
#endif

#if !defined(ORT_MINIMAL_BUILD)
namespace {
struct ElementTypeTestCase {
  ONNXTensorElementDataType ort_type;
  int proto_type;
  size_t storage_bytes;
  bool supported;
};

// Five logical elements exercise rounding of packed four-bit and two-bit storage.
constexpr ElementTypeTestCase element_type_test_cases[] = {
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT, 20, true},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4, ONNX_NAMESPACE::TensorProto_DataType_UINT4, 3, true},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4, ONNX_NAMESPACE::TensorProto_DataType_INT4, 3, true},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2, ONNX_NAMESPACE::TensorProto_DataType_UINT2, 2, true},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2, ONNX_NAMESPACE::TensorProto_DataType_INT2, 2, true},
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1, ONNX_NAMESPACE::TensorProto_DataType_FLOAT4E2M1, 3,
#if !defined(DISABLE_FLOAT4_TYPES)
     true
#else
     false
#endif
    },
    {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0, ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0, 5,
#if !defined(DISABLE_FLOAT8_TYPES)
     true
#else
     false
#endif
    },
};

struct ElementTypeIdentityKernel {
  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    const auto input = ctx.GetInput(0);
    const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    auto output = ctx.GetOutput(0, shape);
    std::memcpy(output.GetTensorMutableRawData(), input.GetTensorRawData(), input.GetTensorSizeInBytes());
  }
};

struct ElementTypeIdentityOp : Ort::CustomOpBase<ElementTypeIdentityOp, ElementTypeIdentityKernel> {
  explicit ElementTypeIdentityOp(ONNXTensorElementDataType type, bool with_shape_inference) : type_(type) {
    if (with_shape_inference) {
      OrtCustomOp::InferOutputShapeFn = [](const OrtCustomOp* op, OrtShapeInferContext* context) -> OrtStatusPtr {
        const auto* self = static_cast<const ElementTypeIdentityOp*>(op);
        self->shape_inference_called_ = true;
        Ort::ShapeInferContext shape_context(&Ort::GetApi(), context);
        return shape_context.SetOutputShape(0, shape_context.GetInputShape(0), self->type_).release();
      };
    }
  }
  void* CreateKernel(const OrtApi&, const OrtKernelInfo*) const { return new ElementTypeIdentityKernel(); }
  const char* GetName() const { return "ElementTypeIdentity"; }
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return type_; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return type_; }
  bool ShapeInferenceCalled() const { return shape_inference_called_; }

 private:
  ONNXTensorElementDataType type_;
  mutable bool shape_inference_called_{false};
};
}  // namespace

TEST(CApiTest, TensorCreationUsesOrtElementTypes) {
  const auto& api = Ort::GetApi();
  Ort::AllocatorWithDefaultOptions allocator;
  const auto info = allocator.GetInfo();
  const int64_t shape[] = {5};
  auto free_buffer = [&](void* buffer) { allocator.Free(buffer); };

  for (const auto& test_case : element_type_test_cases) {
    SCOPED_TRACE(test_case.ort_type);
    for (int creation_api = 0; creation_api < 3; ++creation_api) {
      SCOPED_TRACE(creation_api);
      std::unique_ptr<void, decltype(free_buffer)> buffer(allocator.Alloc(test_case.storage_bytes), free_buffer);
      std::memset(buffer.get(), 0x5a, test_case.storage_bytes);
      OrtValue* raw_value = nullptr;
      OrtStatus* raw_status = nullptr;
      switch (creation_api) {
        case 0:
          raw_status = api.CreateTensorWithDataAsOrtValue(info, buffer.get(), test_case.storage_bytes,
                                                          shape, 1, test_case.ort_type, &raw_value);
          break;
        case 1:
          raw_status = api.CreateTensorWithDataAndDeleterAsOrtValue(allocator, buffer.get(), test_case.storage_bytes,
                                                                    shape, 1, test_case.ort_type, &raw_value);
          if (raw_status == nullptr) {
            buffer.release();
          }
          break;
        case 2:
          raw_status = api.CreateTensorAsOrtValue(allocator, shape, 1, test_case.ort_type, &raw_value);
          break;
      }
      Ort::Status status(raw_status);
      Ort::Value value(raw_value);
      if (!test_case.supported) {
        ASSERT_FALSE(status.IsOK());
        EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
        EXPECT_EQ(raw_value, nullptr);
        continue;
      }
      ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
      EXPECT_EQ(value.GetTensorTypeAndShapeInfo().GetElementType(), test_case.ort_type);
      EXPECT_EQ(value.GetTensorTypeAndShapeInfo().GetElementCount(), 5u);
      EXPECT_EQ(value.GetTensorSizeInBytes(), test_case.storage_bytes);
      if (creation_api != 2) {
        const auto* bytes = static_cast<const uint8_t*>(value.GetTensorRawData());
        for (size_t i = 0; i < test_case.storage_bytes; ++i) {
          EXPECT_EQ(bytes[i], 0x5a);
        }
      }
    }

    if (test_case.supported) {
      for (bool with_deleter : {false, true}) {
        std::unique_ptr<void, decltype(free_buffer)> buffer(allocator.Alloc(test_case.storage_bytes), free_buffer);
        OrtValue* raw_value = nullptr;
        Ort::Status status(with_deleter
                               ? api.CreateTensorWithDataAndDeleterAsOrtValue(
                                     allocator, buffer.get(), test_case.storage_bytes - 1, shape, 1,
                                     test_case.ort_type, &raw_value)
                               : api.CreateTensorWithDataAsOrtValue(
                                     info, buffer.get(), test_case.storage_bytes - 1, shape, 1,
                                     test_case.ort_type, &raw_value));
        if (with_deleter && status.IsOK()) {
          buffer.release();
        }
        Ort::Value value(raw_value);
        EXPECT_FALSE(status.IsOK());
        EXPECT_EQ(status.GetErrorCode(), ORT_INVALID_ARGUMENT);
        EXPECT_EQ(raw_value, nullptr);
      }
    }
  }
}

TEST(CApiTest, SparseTensorCreationUsesOrtElementTypes) {
  const auto& api = Ort::GetApi();
  Ort::AllocatorWithDefaultOptions allocator;
  const auto info = allocator.GetInfo();
  const int64_t dense_shape[] = {9};
  const int64_t values_shape[] = {5};
  uint8_t values[] = {1, 2, 3, 4, 5};

  for (auto type : {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2,
                    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0}) {
    SCOPED_TRACE(type);
    for (bool with_data : {false, true}) {
      SCOPED_TRACE(with_data);
      OrtValue* raw_value = nullptr;
      Ort::Status status(with_data
                             ? api.CreateSparseTensorWithValuesAsOrtValue(
                                   info, values, dense_shape, 1, values_shape, 1, type, &raw_value)
                             : api.CreateSparseTensorAsOrtValue(allocator, dense_shape, 1, type, &raw_value));
      Ort::Value value(raw_value);
#if !defined(DISABLE_SPARSE_TENSORS) && !defined(DISABLE_FLOAT8_TYPES)
      if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0) {
        ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
        int64_t indices[] = {0, 2, 4, 6, 8};
        if (with_data) {
          value.UseCooIndices(indices, 5);
        } else {
          value.FillSparseTensorCoo(info, {values_shape, 1, {values}}, indices, 5);
        }
        EXPECT_EQ(value.GetTensorTypeAndShapeInfo().GetElementType(), type);
        EXPECT_EQ(value.GetSparseTensorValuesTypeAndShapeInfo().GetElementType(), type);
        EXPECT_EQ(value.GetSparseTensorValuesTypeAndShapeInfo().GetElementCount(), 5u);
        const void* sparse_values = nullptr;
        Ort::ThrowOnError(api.GetSparseTensorValues(value, &sparse_values));
        EXPECT_EQ(std::memcmp(sparse_values, values, sizeof(values)), 0);
        continue;
      }
#endif
      // Packed two-bit types are unsupported; UINT2 must not be interpreted as FLOAT8E8M0.
      EXPECT_FALSE(status.IsOK());
#if defined(DISABLE_SPARSE_TENSORS)
      EXPECT_EQ(status.GetErrorCode(), ORT_FAIL);
#else
      EXPECT_EQ(status.GetErrorCode(), ORT_NOT_IMPLEMENTED);
#endif
      EXPECT_EQ(raw_value, nullptr);
    }
  }
}

static void TestCustomOpElementTypes(bool with_shape_inference) {
  for (const auto& test_case : element_type_test_cases) {
    if (!test_case.supported) {
      continue;
    }
    SCOPED_TRACE(test_case.ort_type);
    ElementTypeIdentityOp op(test_case.ort_type, with_shape_inference);
    ElementTypeIdentityOp alternate_op(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, false);
    Ort::CustomOpDomain domain("test.element_types");
    domain.Add(&op);
    if (with_shape_inference) {
      // A unique schema output type would hide the callback's inferred element type in Graph::Resolve.
      domain.Add(&alternate_op);
    }
    Ort::SessionOptions options;
    options.Add(domain);
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
    model.add_opset_import()->set_version(26);
    auto* opset = model.add_opset_import();
    opset->set_domain("test.element_types");
    opset->set_version(1);
    auto* graph = model.mutable_graph();
    graph->set_name("element_type_identity");
    auto* input = graph->add_input();
    input->set_name("X");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(test_case.proto_type);
    input_type->mutable_shape()->add_dim()->set_dim_value(5);
    auto* output = graph->add_output();
    output->set_name("Y");
    *output->mutable_type() = input->type();
    auto* node = graph->add_node();
    node->set_op_type(op.GetName());
    node->set_domain("test.element_types");
    node->add_input("X");
    node->add_output("Y");

    const auto bytes = model.SerializeAsString();
    Ort::Session session(*ort_env, bytes.data(), bytes.size(), options);
    EXPECT_EQ(op.ShapeInferenceCalled(), with_shape_inference);
    EXPECT_EQ(session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(), test_case.ort_type);
    EXPECT_EQ(session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetElementType(), test_case.ort_type);

    Ort::AllocatorWithDefaultOptions allocator;
    const int64_t shape[] = {5};
    auto value = Ort::Value::CreateTensor(allocator, shape, 1, test_case.ort_type);
    std::memset(value.GetTensorMutableRawData(), 0x5a, test_case.storage_bytes);
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions(), input_names, &value, 1, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    EXPECT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetElementType(), test_case.ort_type);
    ASSERT_EQ(outputs[0].GetTensorSizeInBytes(), test_case.storage_bytes);
    EXPECT_EQ(std::memcmp(outputs[0].GetTensorRawData(), value.GetTensorRawData(), test_case.storage_bytes), 0);
  }
}

TEST(CApiTest, CustomOpSchemaAndKernelUseOrtElementTypes) {
  TestCustomOpElementTypes(false);
}

TEST(CApiTest, CustomOpShapeInferenceUsesOrtElementTypes) {
  TestCustomOpElementTypes(true);
}

#if !defined(REDUCED_OPS_BUILD)
namespace {
struct StandaloneDequantizeKernel {
  StandaloneDequantizeKernel(const OrtKernelInfo* info, ONNXTensorElementDataType type)
      : info_(Ort::ConstKernelInfo(info).Copy()) {
    const char* names[] = {"T1", "T2", "T3"};
    const ONNXTensorElementDataType types[] = {
        type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT};
    op_ = Ort::Op::Create(info_, "DequantizeLinear", "", 25, names, types, 3, nullptr, 0, 2, 1);
  }

  void Compute(OrtKernelContext* context) {
    Ort::KernelContext ctx(context);
    const auto input = ctx.GetInput(0);
    auto output = ctx.GetOutput(0, input.GetTensorTypeAndShapeInfo().GetShape());
    float scale_data = 2.0f;
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto scale = Ort::Value::CreateTensor<float>(memory_info, &scale_data, 1, nullptr, 0);
    const OrtValue* inputs[] = {input, scale};
    OrtValue* outputs[] = {output};
    op_.Invoke(context, inputs, 2, outputs, 1);
  }

 private:
  Ort::KernelInfo info_;
  Ort::Op op_{nullptr};
};

struct StandaloneDequantizeOp : Ort::CustomOpBase<StandaloneDequantizeOp, StandaloneDequantizeKernel> {
  explicit StandaloneDequantizeOp(ONNXTensorElementDataType type) : type_(type) {}
  void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
    return new StandaloneDequantizeKernel(info, type_);
  }
  const char* GetName() const { return "StandaloneDequantize"; }
  size_t GetInputTypeCount() const { return 1; }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const { return type_; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }

 private:
  ONNXTensorElementDataType type_;
};
}  // namespace

TEST(CApiTest, StandaloneCreateOpUsesOrtElementTypes) {
  const struct {
    ONNXTensorElementDataType ort_type;
    int proto_type;
    bool supported;
    std::array<float, 5> expected;
  } cases[] = {
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT2, ONNX_NAMESPACE::TensorProto_DataType_UINT2, true, {4, 2, 0, 6, 2}},
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_INT2, ONNX_NAMESPACE::TensorProto_DataType_INT2, true, {-4, 2, 0, -2, 2}},
#if !defined(DISABLE_FLOAT8_TYPES)
      // DequantizeLinear has no E8M0 kernel; it must not silently select the INT2 kernel.
      {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0, ONNX_NAMESPACE::TensorProto_DataType_FLOAT8E8M0, false, {}},
#endif
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.ort_type);
    StandaloneDequantizeOp op(test_case.ort_type);
    Ort::CustomOpDomain domain("test.element_types");
    domain.Add(&op);
    Ort::SessionOptions options;
    options.Add(domain);
    options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
    model.add_opset_import()->set_version(26);
    auto* opset = model.add_opset_import();
    opset->set_domain("test.element_types");
    opset->set_version(1);
    auto* graph = model.mutable_graph();
    graph->set_name("standalone_dequantize");
    auto* input = graph->add_input();
    input->set_name("X");
    auto* input_type = input->mutable_type()->mutable_tensor_type();
    input_type->set_elem_type(test_case.proto_type);
    input_type->mutable_shape()->add_dim()->set_dim_value(5);
    auto* output = graph->add_output();
    output->set_name("Y");
    *output->mutable_type() = input->type();
    output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* node = graph->add_node();
    node->set_op_type(op.GetName());
    node->set_domain("test.element_types");
    node->add_input("X");
    node->add_output("Y");
    const auto bytes = model.SerializeAsString();

    if (!test_case.supported) {
      EXPECT_THROW(Ort::Session(*ort_env, bytes.data(), bytes.size(), options), Ort::Exception);
      continue;
    }
    Ort::Session session(*ort_env, bytes.data(), bytes.size(), options);
    uint8_t packed[] = {0xc6, 0x01};
    const int64_t shape[] = {5};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    auto value = Ort::Value::CreateTensor(memory_info, packed, sizeof(packed), shape, 1, test_case.ort_type);
    const char* input_names[] = {"X"};
    const char* output_names[] = {"Y"};
    auto outputs = session.Run(Ort::RunOptions(), input_names, &value, 1, output_names, 1);
    ASSERT_EQ(outputs.size(), 1u);
    ASSERT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ASSERT_EQ(outputs[0].GetTensorTypeAndShapeInfo().GetElementCount(), test_case.expected.size());
    const auto* actual = outputs[0].GetTensorData<float>();
    for (size_t i = 0; i < test_case.expected.size(); ++i) {
      EXPECT_EQ(actual[i], test_case.expected[i]);
    }
  }
}

#endif  // !defined(REDUCED_OPS_BUILD)

TEST(CApiTest, GetEpGraphAssignmentInfo_NotEnabledError) {
  // Test that calling OrtApi::Session_GetEpGraphAssignmentInfo() without enabling the appropriate
  // session configuration option returns an error.

  Ort::SessionOptions options;
  // Do not set:
  // options.AddConfigEntry(kOrtSessionOptionsRecordEpGraphAssignmentInfo, "1");

  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), options);
  try {
    session.GetEpGraphAssignmentInfo();
    ASSERT_TRUE(false) << "Call to Session_GetEpGraphAssignmentInfo should have failed";
  } catch (const Ort::Exception& ex) {
    ASSERT_EQ(ex.GetOrtErrorCode(), ORT_FAIL);

    std::ostringstream oss;
    oss << "Session configuration entry '" << kOrtSessionOptionsRecordEpGraphAssignmentInfo << "' must be set to \"1\"";
    ASSERT_THAT(ex.what(), testing::HasSubstr(oss.str()));
  }
}
#endif

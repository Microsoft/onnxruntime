// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>

#include <core/common/status.h>

#include "core/framework/ortdevice.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/session_state_utils.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/data_transfer_manager.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/ort_value.h"
#include "core/framework/ort_value_pattern_planner.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/framework/sequential_execution_plan.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/framework/bfc_arena.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/framework/mem_buffer.h"
#include "core/framework/tensor_allocator.h"
#if !defined(ORT_MINIMAL_BUILD) && defined(ORT_MEMORY_PROFILE)
#include "core/framework/memory_info.h"
#endif

namespace onnxruntime {
namespace session_state_utils {

// The following method will allocate memory directly using the device allocator.
// It can handle arena-based allocators and non-arena based allocators.
static common::Status AllocateBufferUsingDeviceAllocatorFromShapeAndType(const TensorShape& tensor_shape,
                                                                         const DataTypeImpl* type,
                                                                         const AllocatorPtr& alloc,
                                                                         /*out*/ void*& p_data) {
  size_t mem_size = 0;
  ORT_RETURN_IF_ERROR(Tensor::CalculateTensorStorageSize(type, tensor_shape, /*alignment*/ 0, mem_size));

  p_data = alloc->Reserve(mem_size);

  return Status::OK();
}

/**
 * @brief Deserializes a TensorProto into an OrtValue.
 *
 * This function handles the complexities of deserializing a tensor, including
 * managing memory allocation, handling external data, and transferring data
 * between different devices (e.g., CPU to GPU). It can use a pre-allocated
 * memory buffer or an allocator to manage the tensor's memory.
 *
 * @param env The environment object, providing access to logging and other services.
 * @param proto_path The file path of the ONNX model, used for resolving external data paths.
 * @param tensor_proto The TensorProto message to deserialize.
 * @param memory_buffer Optional. A raw memory buffer that is pre-allocated for the tensor.
 *                      If provided, `alloc` must be null.
 * @param alloc Optional. An allocator to use for allocating the tensor's memory.
 *              If provided, `memory_buffer` must be null.
 * @param default_cpu_alloc The default CPU allocator, used for intermediate buffers if needed
 *                          (e.g., when copying from CPU to another device).
 * @param[out] ort_value The OrtValue to be populated with the deserialized tensor data.
 * @param data_transfer_mgr The manager responsible for copying tensor data between different memory locations/devices.
 * @param external_data_loader_mgr The manager for handling custom external data loaders.
 * @param prepacked_for_graph Reference to an object managing prepacked weights for the graph.
 * @param use_device_allocator_for_initializers A flag indicating whether to use the device-specific allocator
 *                                              directly for initializers, potentially bypassing arenas.
 * @return common::Status indicating success or failure of the deserialization process.
 *         Returns an error status if both `memory_buffer` and `alloc` are provided or if both are null (unless external data on CPU allows mmap),
 *         if string tensors are attempted to be copied to non-CPU devices, or if any underlying
 *         data loading, allocation, or copying operation fails.
 */
static common::Status DeserializeTensorProto(const Env& env, const std::basic_string<PATH_CHAR_TYPE>& proto_path,
                                             const ONNX_NAMESPACE::TensorProto& tensor_proto,
                                             const MemBuffer* memory_buffer,
                                             const AllocatorPtr& alloc, const AllocatorPtr& default_cpu_alloc,
                                             OrtValue& ort_value, const DataTransferManager& data_transfer_mgr,
                                             const ExternalDataLoaderManager& external_data_loader_mgr,
                                             PrepackedWeightsForGraph& prepacked_for_graph,
                                             bool use_device_allocator_for_initializers = false) {
  if (alloc != nullptr && memory_buffer != nullptr) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "DeserializeTensorProto() takes either pre-allocated buffer or an allocator!");
  }

  TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
  const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type())->GetElementType();
  Tensor tensor;

  // Get shape and type of the tensor, and allocate the empty tensor
  static const auto default_cpu_device = OrtDevice();
  const auto& memory_info = (alloc != nullptr) ? alloc->Info() : memory_buffer->GetAllocInfo();
  const auto device = memory_info.device;

  if (utils::HasExternalData(tensor_proto)) {
    auto external_data_loader =
        external_data_loader_mgr.GetExternalDataLoader(memory_info, tensor_proto.data_type());
#if defined(ENABLE_D3D12_FILE_LOADING)
    // Tensor-creating file loaders cannot consume memory-backed external data.
    // Other loaders, including WebAssembly loaders, must still receive it.
    if (utils::HasExternalDataInMemory(tensor_proto) &&
        external_data_loader != nullptr &&
        external_data_loader->CreatesTensorForDevice(device)) {
      external_data_loader = nullptr;
    }
#endif
    if (external_data_loader) {
#if defined(ENABLE_D3D12_FILE_LOADING)
      if (external_data_loader->CreatesTensorForDevice(device)) {
        ORT_RETURN_IF(memory_buffer != nullptr,
                      "An external data loader that creates tensors cannot use a preallocated buffer.");
        ORT_RETURN_IF(alloc == nullptr,
                      "An external data loader that creates tensors requires a device allocator.");
        tensor = Tensor{type, tensor_shape, nullptr, alloc};
      } else
#endif
      {
        // if custom external data loader is used, always allocate memory on device
        ORT_RETURN_IF_ERROR(
            AllocateTensor(memory_buffer, tensor, type, tensor_shape, use_device_allocator_for_initializers, alloc));
      }
      ORT_RETURN_IF_ERROR(utils::LoadExtDataToTensorFromTensorProto(env, proto_path, tensor_proto,
                                                                    *external_data_loader, alloc, tensor));
      ORT_RETURN_IF_ERROR(utils::LoadPrepackedWeightsFromExternalData(env, proto_path, tensor_proto,
                                                                      prepacked_for_graph));

      Tensor::InitOrtValue(std::move(tensor), ort_value);
      return common::Status::OK();
    } else if (device == default_cpu_device) {
      // for external initializer on CPU we will use mmap for large initializers so don't need to allocate memory in advance

      // NB: The file containing external data for the tensor is mmap'd. If the tensor will be used on CPU we can
      // utilize the mmap'd buffer directly.
      ORT_RETURN_IF_ERROR(utils::GetExtDataFromTensorProto(env, proto_path, tensor_proto,
                                                           ort_value,
                                                           &prepacked_for_graph));
      return common::Status::OK();
    } else {  // non-cpu tensor or tensor in a cpu accessible memory
      if (utils::HasString(tensor_proto)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "string tensor is not supported for copying between allocators");
      }

      // deserialize to CPU first for non-CPU allocator, then copy to device
      // for external initializer load on non-CPU device:
      // 1. allocate memory on device - tensor
      // 2. load initializer into CPU memory - deserialized_value,
      //    we will use mmap so no need to allocate memory on CPU in advance
      // 3. copy tensor from CPU to device - deserialized_value -> tensor -> ort_value
      ORT_RETURN_IF_ERROR(AllocateTensor(memory_buffer, tensor, type, tensor_shape, use_device_allocator_for_initializers, alloc));

      OrtValue deserialized_value;
      ORT_RETURN_IF_ERROR(utils::GetExtDataFromTensorProto(env, proto_path, tensor_proto,
                                                           deserialized_value,
                                                           &prepacked_for_graph));

      const Tensor& cpu_staging_tensor = deserialized_value.Get<Tensor>();
      // Bool external initializers are copied verbatim and may carry bytes outside the canonical
      // {0, 1} set. The CPU staging tensor above can be backed by a read-only mmap, so normalize into
      // a writable CPU copy before copying to the device (see utils::NormalizeBoolTensorIfNeeded).
      if (cpu_staging_tensor.IsDataType<bool>()) {
        Tensor normalized_cpu_tensor;
        ORT_RETURN_IF_ERROR(AllocateTensorOnDeviceOrMemory(/* use_device_allocator_for_initializers =*/true,
                                                           tensor_shape, type,
                                                           default_cpu_alloc, normalized_cpu_tensor));
        utils::MakeCpuTensorCopy(cpu_staging_tensor, normalized_cpu_tensor);
        utils::NormalizeBoolTensorIfNeeded(normalized_cpu_tensor);
        return CopyTensorFromCPUToDevice(data_transfer_mgr, normalized_cpu_tensor, std::move(tensor), ort_value);
      }

      return CopyTensorFromCPUToDevice(data_transfer_mgr, deserialized_value.Get<Tensor>(),
                                       std::move(tensor), ort_value);
    }
  } else {
    if (device == default_cpu_device) {
      // deserialize directly to CPU tensor
      // Do not use arena for internal initializer, just like we do for OrtValue initializers
      ORT_RETURN_IF_ERROR(AllocateTensorOnDeviceOrMemory(/* use_device_allocator_for_initializers =*/true,
                                                         tensor_shape, type,
                                                         default_cpu_alloc, tensor));
      ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(env, proto_path.c_str(), tensor_proto, tensor));
      Tensor::InitOrtValue(std::move(tensor), ort_value);
      return common::Status::OK();
    } else {  // non-cpu tensor
      if (utils::HasString(tensor_proto)) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "string tensor is not supported for copying between allocators");
      }

      // Allocate according to the plan on the device or directly on the device according to
      // use_device_allocator_for_initializers
      ORT_RETURN_IF_ERROR(AllocateTensor(memory_buffer, tensor, type, tensor_shape,
                                         use_device_allocator_for_initializers, alloc));

      // deserialize to CPU first for non-CPU allocator, then copy
      // for internal initializer
      // 1. allocate memory on CPU - deserialized_tensor. Do not use arena not to waste space for temporary buffers.
      // 2. deserialize tensor_proto into a pre-allocated tensor (deserialized_tensor)
      // 3. copy tensor from CPU to device - deserialized_tensor -> tensor (allocated above) -> ort_value
      Tensor deserialized_tensor;
      ORT_RETURN_IF_ERROR(AllocateTensorOnDeviceOrMemory(/* use_device_allocator_for_initializers =*/true,
                                                         tensor_shape, type,
                                                         default_cpu_alloc, deserialized_tensor));

      ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(env, proto_path.c_str(), tensor_proto, deserialized_tensor));
      return CopyTensorFromCPUToDevice(data_transfer_mgr, deserialized_tensor, std::move(tensor), ort_value);
    }
  }
}

common::Status AllocateTensor(const onnxruntime::MemBuffer* memory_buffer,
                              Tensor& tensor,
                              const onnxruntime::DataTypeImpl* const& type,
                              onnxruntime::TensorShape& tensor_shape,
                              bool use_device_allocator_for_initializers,
                              const onnxruntime::AllocatorPtr& alloc) {
  if (memory_buffer != nullptr) {
    tensor = Tensor{type, tensor_shape, memory_buffer->GetBuffer(), memory_buffer->GetAllocInfo()};
    if (memory_buffer->GetLen() < tensor.SizeInBytes()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Internal error. The preallocated buffer is too small. Requires ",
                             tensor.SizeInBytes(), ", Got ", memory_buffer->GetLen());
    }
  } else {
    return AllocateTensorOnDeviceOrMemory(use_device_allocator_for_initializers, tensor_shape, type, alloc, tensor);
  }
  return common::Status::OK();
}

common::Status AllocateTensorOnDeviceOrMemory(
    bool use_device_allocator_for_initializers,
    onnxruntime::TensorShape& tensor_shape,
    const onnxruntime::DataTypeImpl* const& type,
    const onnxruntime::AllocatorPtr& alloc,
    Tensor& tensor) {
  if (use_device_allocator_for_initializers) {
    void* tensor_buffer = nullptr;
    ORT_RETURN_IF_ERROR(AllocateBufferUsingDeviceAllocatorFromShapeAndType(tensor_shape, type, alloc, tensor_buffer));
    tensor = Tensor{type, tensor_shape, tensor_buffer, alloc};
  } else {
    // If the provided allocator is an arena-based allocator, the call to Alloc() will tap into memory from the arena
    // (may expand it if there isn't a chunk that can be allotted to the memory request).
    // If the provided allocator is non-arena based, the device specific Alloc() call will be used to allocate the necessary memory.
    tensor = Tensor{type, tensor_shape, alloc};
  }
  return common::Status::OK();
}

common::Status CopyTensorFromCPUToDevice(
    const onnxruntime::DataTransferManager& data_transfer_mgr,
    const Tensor& deserialized_tensor,
    Tensor&& tensor,
    OrtValue& ort_value) {
  Status copy_status = data_transfer_mgr.CopyTensor(deserialized_tensor, tensor);
  if (!copy_status.IsOK()) {
    if (copy_status.ErrorMessage().empty()) {
      // The windows execution provider does not return any error message today for CopyTensor since it is
      // not implemented yet. That's the reason we're adding our own error message so that we can debug better.
      return Status(copy_status.Category(), copy_status.Code(),
                    "Failed to copy tensor to " + tensor.Location().ToString());
    }
    return copy_status;
  } else {
    Tensor::InitOrtValue(std::move(tensor), ort_value);
    return common::Status::OK();
  }
}

common::Status SaveInitializedTensors(
    const Env& env, const std::basic_string<PATH_CHAR_TYPE>& graph_loc,
    const GraphViewer& graph, const AllocatorPtr& default_cpu_alloc,
    const OrtValueNameIdxMap& ort_value_name_idx_map,
    const std::vector<OrtValueIndex>& initializer_allocation_order,
    ITensorAllocator& planner,
    const SaveTensorFunction& save_tensor_func,
    const logging::Logger& logger,
    const DataTransferManager& data_transfer_mgr,
    const ExternalDataLoaderManager& external_data_loader_mgr,
    const ExecutionPlanBase& exec_plan,
    const SessionOptions& session_options,
    const MemoryProfileFunction& memory_profile_func,
    PrepackedWeightsForGraph& prepacked_for_graph) {
  LOGS(logger, INFO) << "Saving initialized tensors.";
  ORT_ENFORCE(ort_value_name_idx_map.MaxIdx() > -1, "OrtValue indexes should have been populated.");

  // Determine if an intializer was supplied by the user for the purpose of sharing and if it requires a cross-device
  // copy. In case a cross-device copy is required, sharing cannot be accomplished since we allocate our own buffer
  // for the destination device which cannot be shared between sessions.
  auto use_user_supplied_initializer =
      [&session_options, &exec_plan, &logger, &ort_value_name_idx_map](const std::string& name) -> bool {
    bool retval = false;
    auto it = session_options.initializers_to_share_map.find(name);
    if (it == session_options.initializers_to_share_map.end()) {
      retval = false;
    } else {
      int ort_value_index = -1;
      if (!ort_value_name_idx_map.GetIdx(name, ort_value_index).IsOK()) {
        retval = false;
      } else {
        const auto& planned_mem_device = exec_plan.GetLocation(ort_value_index);
        const auto& user_mem_info = it->second->Get<Tensor>().Location();
        retval = user_mem_info.device == planned_mem_device;
        if (!retval) {
          LOGS(logger, WARNING) << "Cannot use user supplied initializer with name: ("
                                << name << ") because the ORT planned memory location device "
                                << planned_mem_device.ToString()
                                << " ) is different from what is supplied (" << user_mem_info.ToString() << ")";
        }
      }
    }

    return retval;
  };

  // 1. first plan the memory
  const InitializedTensorSet& initialized_tensor_set = graph.GetAllInitializedTensors();
  InlinedHashMap<int, const ONNX_NAMESPACE::TensorProto*> id_to_initialized_tensor;
  InlinedHashSet<int> user_supplied_initializer_ids;  // set containing the ort value ids of all user supplied initializers

  id_to_initialized_tensor.reserve(initialized_tensor_set.size());
  user_supplied_initializer_ids.reserve(session_options.initializers_to_share_map.size());

  for (const auto& entry : initialized_tensor_set) {
    int ort_value_index;
    ORT_RETURN_IF_ERROR(ort_value_name_idx_map.GetIdx(entry.first, ort_value_index));
    if (use_user_supplied_initializer(entry.first)) {
      user_supplied_initializer_ids.insert(ort_value_index);
    }
    id_to_initialized_tensor[ort_value_index] = entry.second;
  }

  static const auto default_cpu_device = OrtDevice();

  // tensors requiring a specific allocation order are traced first, to ensure they are allocated in order
  // NB1: vector with init allocation order may contain a subset of all tensors (or none at all)
  // NB2: only skip tracing and planning memory when data is external (i.e mmap) and on CPU.
  //    when data is external and on GPU, need to copy first to cpu memory, then to gpu memory.
  auto initialized_tensors_to_allocate = id_to_initialized_tensor;
  for (int ort_value_index : initializer_allocation_order) {
    const auto entry = initialized_tensors_to_allocate.find(ort_value_index);
    ORT_ENFORCE(entry != initialized_tensors_to_allocate.end(),
                "OrtValue index: ", ort_value_index, " from initializer_allocation_order not found among initialized tensors");
    const auto* tensor_proto = entry->second;

    // We trace to allocate a single buffer using the planner. This reduces fragmentation.
    // We do not trace the following values because it would add to the memory consumption.
    // - Values that are on OrtDevice() (default CPU).
    // - Values that are external and mapped from disk. We let the OS manage the memory.
    // - we do not trace values that are in memory because they may be sitting on top of the user allocated
    //   memory.
#if defined(ENABLE_D3D12_FILE_LOADING)
    const bool loader_creates_tensor =
        utils::HasExternalData(*tensor_proto) &&
        !utils::HasExternalDataInMemory(*tensor_proto) &&
        external_data_loader_mgr.GetTensorCreator(
            exec_plan.GetLocation(ort_value_index), tensor_proto->data_type()) != nullptr;
#else
    constexpr bool loader_creates_tensor = false;
#endif
    const bool trace_allocation = !loader_creates_tensor &&
                                  ((exec_plan.GetLocation(ort_value_index) != default_cpu_device) ||
                                   !utils::HasExternalData(*tensor_proto));

    if (trace_allocation) {
      // can not trace string tensor, and they exist only on CPU
      ORT_ENFORCE(!utils::HasString(*tensor_proto), "Can not trace string tensor");
      ORT_RETURN_IF_ERROR(planner.Trace(ort_value_index, tensor_proto));
    }
    initialized_tensors_to_allocate.erase(entry);
  }

  for (const auto& entry : initialized_tensors_to_allocate) {
    // We don't want to trace shared initializers since their memory is provided by the user
    if (user_supplied_initializer_ids.find(entry.first) != user_supplied_initializer_ids.end()) {
      continue;
    }
    if (utils::HasString(*entry.second)) {
      // do not trace string tensor
      continue;
    }
#if defined(ENABLE_D3D12_FILE_LOADING)
    if (utils::HasExternalData(*entry.second) &&
        !utils::HasExternalDataInMemory(*entry.second) &&
        external_data_loader_mgr.GetTensorCreator(
            exec_plan.GetLocation(entry.first), entry.second->data_type()) != nullptr) {
      continue;
    }
#endif
    if (exec_plan.GetLocation(entry.first) == default_cpu_device && utils::HasExternalData(*entry.second)) {
      continue;
    }
    ORT_RETURN_IF_ERROR(planner.Trace(entry.first, entry.second));
  }

  // 2. allocate weight buffer on different locations
  //  planned_initializers_memory_size_in_byte is not actual physical size.
  //  It's the virtual size computed by planner.
  InlinedHashMap<OrtDevice, size_t> planned_initializers_memory_sizes_in_byte;
  ORT_RETURN_IF_ERROR(
      planner.FinalizePlan(planned_initializers_memory_sizes_in_byte));

  if (memory_profile_func)
    memory_profile_func(planner);

  for (const auto& i : planned_initializers_memory_sizes_in_byte) {
    LOGS(logger, INFO) << "[Memory] SessionStateInitializer statically allocates "
                       << i.second << " bytes for " << i.first.ToString() << std::endl;
  }

  // ??? Should we ignore this session option if the EP is explicitly providing the read only allocator?
  // bool have_readonly_initializer_allocator = alloc->Info().alloc_type == OrtReadOnlyAllocator;
  // This option also means to ignore arena if present and use Reserve().
  const bool use_device_allocator_for_initializers =
      session_options.config_options.GetConfigOrDefault(
          kOrtSessionOptionsUseDeviceAllocatorForInitializers, "0") == "1";

#if defined(ENABLE_D3D12_FILE_LOADING)
  ORT_RETURN_IF_ERROR(external_data_loader_mgr.BeginLoad());
  bool external_data_load_ended = false;
  auto end_external_data_load = gsl::finally([&]() {
    if (!external_data_load_ended) {
      external_data_loader_mgr.EndLoad();
    }
  });

  for (const auto& entry : id_to_initialized_tensor) {
    if (user_supplied_initializer_ids.contains(entry.first) ||
        !utils::HasExternalData(*entry.second) ||
        utils::HasExternalDataInMemory(*entry.second)) {
      continue;
    }

    if (session_options.IsLoadCancellationFlagSet()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                             "Preparing session state weights is canceled due to user request.");
    }

    const auto* tensor_creator = external_data_loader_mgr.GetTensorCreator(
        exec_plan.GetLocation(entry.first), entry.second->data_type());
    if (tensor_creator != nullptr) {
      ORT_RETURN_IF_ERROR(
          utils::RegisterExternalDataLoadCandidateFromTensorProto(
              env, graph_loc, *entry.second, *tensor_creator));
    }
  }

  ORT_RETURN_IF_ERROR(external_data_loader_mgr.CommitLoadCandidates(
      [&session_options]() { return session_options.IsLoadCancellationFlagSet(); }));
#endif

  // 3. create weight tensors based on weights buffer
  for (const auto& entry : id_to_initialized_tensor) {
    // We check for cancellation for every initializer since mapping from disk can be costly
    if (session_options.IsLoadCancellationFlagSet()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED,
                             "Saving session state weights is canceled due to user request.");
    }

    int ort_value_index = entry.first;
    const std::string& name = entry.second->name();

    if (name.empty()) {
      LOGS(logger, INFO) << "Skipping entry for missing optional value at idx " << ort_value_index;
      continue;
    }

    OrtValue ort_value;

    if (user_supplied_initializer_ids.find(entry.first) != user_supplied_initializer_ids.end()) {
      ort_value = *(session_options.initializers_to_share_map.at(name));
      LOGS(logger, INFO) << "Using user supplied initializer with name (" << name << ").";
    } else {
      const ONNX_NAMESPACE::TensorProto& tensor_proto = *(entry.second);

      std::optional<MemBuffer> memory_buffer;
      AllocatorPtr alloc;
      // TODO: if the tensor need be copied, does it have enough room?
      ORT_RETURN_IF_ERROR(planner.GetPreallocatedBuffer(ort_value_index, name, memory_buffer, alloc));

      // Check if we already have an OrtValue for this initializer on CPU
      if (OrtValue ort_value_from_graph;
          graph.GetOrtValueInitializer(name, ort_value_from_graph)) {
        const auto& memory_info = (alloc != nullptr) ? alloc->Info() : memory_buffer->GetAllocInfo();
        const auto& graph_value_device = ort_value_from_graph.Get<Tensor>().Location().device;
        if (memory_info.device == default_cpu_device || graph_value_device == memory_info.device) {
          // Either the initializer is planned on CPU, or the user-supplied initializer (e.g. from
          // AddExternalInitializers) already lives on the planned device. In both cases use it in
          // place: no per-session device allocation or copy, and the underlying buffer can be
          // shared across sessions that supply the same OrtValue (matching the AddInitializer path).
          ort_value = std::move(ort_value_from_graph);
        } else {
          TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(tensor_proto);
          const DataTypeImpl* const type = DataTypeImpl::TensorTypeFromONNXEnum(
                                               tensor_proto.data_type())
                                               ->GetElementType();
          Tensor tensor;
          ORT_RETURN_IF_ERROR(AllocateTensor((memory_buffer) ? &*memory_buffer : nullptr, tensor, type,
                                             tensor_shape, use_device_allocator_for_initializers,
                                             alloc));
          ORT_RETURN_IF_ERROR(CopyTensorFromCPUToDevice(data_transfer_mgr,
                                                        ort_value_from_graph.Get<Tensor>(),
                                                        std::move(tensor), ort_value));
        }
      } else {
        // if in memory we were expecting to find it above.
        ORT_ENFORCE(!utils::HasExternalDataInMemory(tensor_proto));

        // We need to deserialize the tensor proto into an OrtValue
        // using the preallocated buffer or allocator.
        Status st = DeserializeTensorProto(env, graph_loc, tensor_proto,
                                           (memory_buffer.has_value()) ? &*memory_buffer : nullptr,
                                           alloc, default_cpu_alloc, ort_value, data_transfer_mgr,
                                           external_data_loader_mgr, prepacked_for_graph,
                                           use_device_allocator_for_initializers);
        if (!st.IsOK()) {
          std::ostringstream oss;
          oss << "Deserialize tensor " << name << " failed." << st.ErrorMessage();
          return Status(st.Category(), st.Code(), oss.str());
        }
      }
    }

    // 'name' is a reference to a string within the TensorProto that save_tensor_func may free
    // so we need to output this message prior to calling save_tensor_func
    VLOGS(logger, 1) << "Adding weight with name : " << name << " with index: " << ort_value_index;

    // any outer scope value is shadowed by a local value and can't override it.
    // due to that check_outer_scope is false
    const bool constant = graph.IsConstantInitializer(name, /* check_outer_scope */ false);
#if !defined(DISABLE_SPARSE_TENSORS)
    const bool sparse = graph.GetGraph().IsSparseInitializer(name);
    ORT_RETURN_IF_ERROR(save_tensor_func(name, ort_value_index, ort_value, constant, sparse));
#else
    ORT_RETURN_IF_ERROR(save_tensor_func(name, ort_value_index, ort_value, constant, false));
#endif
  }

#if defined(ENABLE_D3D12_FILE_LOADING)
  // EndLoad is the idempotent end-of-batch cleanup hook. All claimed
  // allocations have transferred ownership, so this only releases metadata
  // and any unclaimed resources left by a successful batch.
  external_data_loader_mgr.EndLoad();
  external_data_load_ended = true;
#endif
  LOGS(logger, INFO) << "Done saving initialized tensors";
  return common::Status::OK();
}

template <typename T>  // T is container of const NodeArg* or NodeArg*
static bool IsArgNameInInputsOutputs(const std::string& name,
                                     const T& graph_args) {
  auto it = std::find_if(graph_args.begin(), graph_args.end(),
                         [&name](const onnxruntime::NodeArg* arg) {
                           return arg->Name() == name;
                         });
  return it != graph_args.end();
}

common::Status SaveInputOutputNamesToNodeMapping(const onnxruntime::GraphViewer& graph,
                                                 SessionState& session_state,
                                                 gsl::span<const NodeArg* const> implicit_inputs) {
  auto& graph_inputs = graph.GetInputsIncludingInitializers();
  auto& graph_outputs = graph.GetOutputs();

  const auto* exec_plan = session_state.GetExecutionPlan();
  const auto& name_to_id = session_state.GetOrtValueNameIdxMap();

  for (auto& node : graph.Nodes()) {
    const KernelCreateInfo& kci = session_state.GetNodeKernelCreateInfo(node.Index());
    int stream_index = static_cast<int>(exec_plan->node_stream_map_[node.Index()]);

    ORT_RETURN_IF_ERROR(
        onnxruntime::Node::ForEachWithIndex(
            node.InputDefs(),
            [&](const onnxruntime::NodeArg& arg, size_t index) {
              if (arg.Name().empty()) {
                return Status::OK();
              }

              int arg_index;
              ORT_RETURN_IF_ERROR(name_to_id.GetIdx(arg.Name(), arg_index));
              const auto& device = exec_plan->GetLocation(arg_index);
              SessionState::NodeInfo node_info(index, &node, &kci, device, stream_index);

              if (IsArgNameInInputsOutputs(arg.Name(), graph_inputs)) {
                ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info));
                return Status::OK();
              }

              if (!implicit_inputs.empty()) {
                if (IsArgNameInInputsOutputs(arg.Name(), implicit_inputs)) {
                  ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(arg.Name(), node_info));
                  return Status::OK();
                }
              }

              return Status::OK();
            }));

    // implicit inputs to a node could come directly from a feed, so we need to make sure they have an entry too
    const auto& node_implicit_inputs = node.ImplicitInputDefs();
    if (!node_implicit_inputs.empty()) {
      // In the main graph, the location of the implicit input(s) is the location it
      // is consumed in the main graph if there is an explicit consumer.
      // If the only consumer(s) are implicit consumers (i.e.) other control flow nodes and
      // all of them have been partitioned to the same EP, its location is the
      // location of the non-CPU device corresponding to the EP.
      // If multiple EPs are involved, then the planned location for such implicit inputs
      // just default to CPU (as there is ambiguity involved as to which non-CPU device is
      // most optimal)

      // In nested subgraphs, the location of the implicit input(s) is the location it
      // is consumed in the subgraph if there is an explicit consumer.
      // If the only consumer(s) are implicit consumers (i.e.) other control flow nodes, its
      // location is the location of the value in the enclosing outer scope.

      // All this is setup in the planner, we just use the location from the plan here.
      for (const auto& input_def : node_implicit_inputs) {
        int arg_index;
        ORT_RETURN_IF_ERROR(name_to_id.GetIdx(input_def->Name(), arg_index));
        auto& device = exec_plan->GetLocation(arg_index);
        SessionState::NodeInfo node_info(std::numeric_limits<size_t>::max(), &node, &kci, device, stream_index);
        ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(input_def->Name(), node_info));
      }
    }

    ORT_RETURN_IF_ERROR(
        onnxruntime::Node::ForEachWithIndex(
            node.OutputDefs(),
            [&](const onnxruntime::NodeArg& arg, size_t index) {
              if (arg.Name().empty()) {
                return Status::OK();
              }

              int arg_index;
              ORT_RETURN_IF_ERROR(name_to_id.GetIdx(arg.Name(), arg_index));
              const auto& device = exec_plan->GetLocation(arg_index);

              SessionState::NodeInfo node_info(index, &node, &kci, device);

              if (IsArgNameInInputsOutputs(arg.Name(), graph_outputs)) {
                session_state.AddOutputNameToNodeInfoMapping(arg.Name(), node_info);
                return Status::OK();
              }

              return Status::OK();
            }));
  }

  // It's possible (although assumably rare) for a graph to have inputs that aren't used. one reasonable occurrence
  // is in the Loop subgraph where the value of the condition used to decide whether to continue looping is passed in.
  // The condition evaluated to 'true' given the subgraph is being executed, so it's of dubious value as an input.
  // Similar is the current iteration number which may or may not be needed by the Loop subgraph.
  // In order to handle those, create a dummy entry in the input name to node info mapping so that
  // utils::CopyOneInputAcrossDevices is happy.

  auto& input_map = session_state.GetInputNodeInfoMap();

  for (const auto& graph_input : graph_inputs) {
    const auto& name = graph_input->Name();
    if (input_map.find(name) == input_map.cend()) {
      // dummy entry for an input that we didn't find a use of in the graph. log it in case that's a bug.
      // utils::CopyOneInputAcrossDevices will use the input OrtValue as is given we don't believe it's used anywhere.
      LOGS(session_state.Logger(), INFO) << (graph.IsSubgraph() ? "Subgraph" : "Graph") << " input with name "
                                         << name << " is not used by any node.";
      int arg_index;
      ORT_RETURN_IF_ERROR(name_to_id.GetIdx(name, arg_index));
      auto& device = exec_plan->GetLocation(arg_index);
      SessionState::NodeInfo empty_node_info(std::numeric_limits<size_t>::max(), nullptr, nullptr, device);
      ORT_RETURN_IF_ERROR(session_state.AddInputNameToNodeInfoMapping(name, empty_node_info));
    }
  }

  return Status::OK();
}

}  // namespace session_state_utils
}  // namespace onnxruntime

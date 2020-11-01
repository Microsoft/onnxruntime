#include "core/framework/memory_info.h"
#include "core/framework/mem_pattern.h"
#include "core/framework/ml_value.h"

#include <fstream>
#include <numeric>
#include <queue>

namespace onnxruntime {

void MemoryInfo::GenerateTensorMap(const SequentialExecutionPlan* execution_plan, const OrtValueNameIdxMap& value_name_idx_map) {
  if (!tensor_alloc_info_map_.empty()) {
    return;
  }
  num_node_size_ = execution_plan->execution_plan.size();
  for (OrtValueIndex value_idx = 0; value_idx < OrtValueIndex(execution_plan->allocation_plan.size()); ++value_idx) {
    //Only store tensor information
    if (!(execution_plan->allocation_plan[value_idx].value_type) || !(execution_plan->allocation_plan[value_idx].value_type->IsTensorType()))
      continue;
    AllocInfoPerTensor mem_info;
    mem_info.mlvalue_index = value_idx;
    value_name_idx_map.GetName(mem_info.mlvalue_index, mem_info.mlvalue_name);
    mem_info.lifetime_interval = execution_plan->allocation_plan[value_idx].life_interval;
    mem_info.alloctime_interval = execution_plan->allocation_plan[value_idx].allocate_interval;
    mem_info.reused_buffer = (execution_plan->allocation_plan[value_idx].alloc_kind != AllocKind::kReuse) ? value_idx : execution_plan->allocation_plan[value_idx].reused_buffer;
    //If the tensor is using memory outside of the scope, do not store it
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kPreExisting) continue;
    if (execution_plan->allocation_plan[mem_info.reused_buffer].alloc_kind == AllocKind::kAllocateOutput) continue;
    mem_info.inplace_reuse = (execution_plan->allocation_plan[value_idx].inplace_reuse != -1 && execution_plan->allocation_plan[value_idx].inplace_reuse != value_idx);
    mem_info.alloc_kind = execution_plan->allocation_plan[value_idx].alloc_kind;
    mem_info.location = execution_plan->allocation_plan[value_idx].location;

    ORT_ENFORCE(mem_info.lifetime_interval.first <= mem_info.lifetime_interval.second);
    ORT_ENFORCE(mem_info.lifetime_interval.first >= mem_info.alloctime_interval.first &&
                mem_info.lifetime_interval.second <= mem_info.alloctime_interval.second);
    tensor_alloc_info_map_[value_idx] = std::move(mem_info);
    tensors_memory_info_map_[mem_info.location];
  }
  return;
}

//Record the planned memory information
void MemoryInfo::RecordMemoryPatternInfo(const MemoryPatternGroup& mem_patterns, MapType type) {
  for (const auto& location : mem_patterns.locations) {
    for (const auto& p : mem_patterns.GetPatterns(location)->GetPatternsMap()) {
      ORT_ENFORCE(AllocPlan(p.first));
      tensors_memory_info_map_.at(location)[type].AddPlannedMemory(p.first, p.second);
    }
  }
}

void MemoryInfo::RecordActivationPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns, MapType::StaticActivation);
  for (auto& item : tensor_alloc_info_map_) {
    if (item.second.alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = AllocPlan(item.first)->reused_buffer;
      auto& map = tensors_memory_info_map_.at(AllocPlan(item.first)->location);
      if (map[MapType::StaticActivation].Contain(reuse_buffer)) {
        auto& reused_memory = map[MapType::StaticActivation].GetPlannedMemory(reuse_buffer);
        map[MapType::StaticActivation].AddPlannedMemory(item.first, reused_memory);
      }
    }
  }
}

void MemoryInfo::RecordInitializerPatternInfo(const MemoryPatternGroup& mem_patterns) {
  RecordMemoryPatternInfo(mem_patterns, MapType::Initializer);
  for (auto& item : tensor_alloc_info_map_) {
    if (item.second.alloc_kind == AllocKind::kReuse) {
      auto reuse_buffer = AllocPlan(item.first)->reused_buffer;
      auto& map = tensors_memory_info_map_.at(AllocPlan(item.first)->location);
      if (map[MapType::Initializer].Contain(reuse_buffer)) {
        auto& reused_memory = map[MapType::Initializer].GetPlannedMemory(reuse_buffer);
        map[MapType::Initializer].AddPlannedMemory(item.first, reused_memory);
      }
    }
  }
}

//Record the actual allocated tensor in the device
void MemoryInfo::RecordTensorDeviceAllocInfo(const OrtValueIndex idx, const OrtValue& value, const MapType& type) {
  if (tensor_alloc_info_map_.find(idx) == tensor_alloc_info_map_.end()) return;
  auto& tensor = value.Get<Tensor>();
  auto sizeinbytes = tensor.SizeInBytes() % alignment ? tensor.SizeInBytes() + alignment - tensor.SizeInBytes() % alignment : tensor.SizeInBytes();
  MemoryBlock mb(size_t(tensor.DataRaw()), sizeinbytes);
  if (type == MapType::StaticActivation) {
    bool valid = tensors_memory_info_map_.at(AllocPlan(idx)->location).at(type).Contain(idx);
    ORT_ENFORCE(valid);
  }
  tensors_memory_info_map_.at(AllocPlan(idx)->location)[type].AddAllocMemory(idx, mb);
}

void MemoryInfo::RecordInitializerAllocInfo(const std::unordered_map<int, OrtValue>& tensor_map) {
  for (const auto& item : tensor_map) {
    ORT_ENFORCE(AllocPlan(item.first));
    RecordTensorDeviceAllocInfo(item.first, item.second, MapType::Initializer);
  }
}

void MemoryInfo::RecordActivationAllocInfo(const OrtValueIndex idx, const OrtValue& value) {
  if (!AllocPlan(idx)) return;
  auto reuse_buffer = AllocPlan(idx)->reused_buffer;
  MapType map_type;
  auto& map = tensors_memory_info_map_.at(AllocPlan(reuse_buffer)->location);
  if (map[MapType::DynamicActivation].Contain(reuse_buffer))
    map_type = MapType::DynamicActivation;
  else if (map[MapType::StaticActivation].Contain(reuse_buffer))
    map_type = MapType::StaticActivation;
  else if (map[MapType::Initializer].Contain(reuse_buffer))
    map_type = MapType::Initializer;
  else
    ORT_ENFORCE(false, "Unsupported tensor type.");

  RecordTensorDeviceAllocInfo(idx, value, map_type);
}

void MemoryInfo::SetDynamicAllocation(const OrtValueIndex idx) {
  if (!AllocPlan(idx)) return;
  if (!tensors_memory_info_map_.at(AllocPlan(idx)->location)[MapType::DynamicActivation].Contain(idx)) {
    tensors_memory_info_map_.at(AllocPlan(idx)->location)[MapType::DynamicActivation][idx];
  }
}

void PrintInforPerTensor(const MemoryInfo::AllocInfoPerTensor& alloc_info, const MemoryInfoPerTensor& mem_info, const size_t& rel_addr) {
  std::cout << "Tensor name: " << alloc_info.mlvalue_name << ", ";
  std::cout << "Index: " << alloc_info.mlvalue_index << ", ";
  std::cout << "Type: " << alloc_info.tensor_type << ", ";
  std::cout << "Reuse inplace: " << alloc_info.inplace_reuse << ", ";
  std::cout << "Alloc type " << alloc_info.alloc_kind << ", ";
  std::cout << "Location: " << alloc_info.location.name << ", ";
  std::cout << "lifetime: (" << alloc_info.lifetime_interval.first << ", " << alloc_info.lifetime_interval.second << "), ";
  std::cout << "alloc time: (" << alloc_info.alloctime_interval.first << ", " << alloc_info.alloctime_interval.second << "), ";
  std::cout << "planned block: (" << mem_info.planned_block.offset_ << ", "
            << (mem_info.planned_block.offset_ + mem_info.planned_block.size_) << "), ";
  std::cout << "planned Size: " << mem_info.planned_block.size_ << ", ";
  std::cout << "allocated block: (" << rel_addr << ", "
            << (rel_addr + mem_info.alloced_block.size_) << "), ";
  std::cout << "allocated Size: " << mem_info.alloced_block.size_ << "\n";
}

void MemoryInfo::PrintMemoryInfoForLocation(const logging::Logger& /*logger*/, const OrtDevice::DeviceType location) {
  for (const auto& map : tensors_memory_info_map_) {
    std::cout << "Initializer in " << map.first.name << "\n";
    const auto& initailizer_map = map.second.at(MapType::Initializer);
    for (const auto& item : initailizer_map) {
      if (AllocPlan(item.first)->location.device.Type() != location) continue;
      PrintInforPerTensor(*AllocPlan(item.first), item.second, initailizer_map.GetAllocAddress(item.first));
    }
    std::cout << "\nStatic Activation in " << map.first.name << "\n";
    const auto& static_activation_map = map.second.at(MapType::StaticActivation);
    for (const auto& item : static_activation_map) {
      if (AllocPlan(item.first)->location.device.Type() != location) continue;
      PrintInforPerTensor(*AllocPlan(item.first), item.second, static_activation_map.GetAllocAddress(item.first));
    }

    std::cout << "\nDynamic Activation in " << map.first.name << "\n";
    const auto& dynamic_activation_map = map.second.at(MapType::DynamicActivation);
    for (const auto& item : dynamic_activation_map) {
      if (AllocPlan(item.first)->location.device.Type() != location) continue;
      PrintInforPerTensor(*AllocPlan(item.first), item.second, dynamic_activation_map.GetAllocAddress(item.first));
    }
  }
}

std::string MemoryInfo::MemoryInfoProfile::CreateMetadataEvent(const std::string& process_name, size_t process_id) {
  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"M\",";
  evt << "\"name\":\"process_name\",";
  evt << "\"pid\":" << process_id << ",";
  evt << "\"args\":{\"name\":\"" << process_name << "\"}";
  evt << "}";
  evt << "," << std::endl;
  evt << "{";
  evt << "\"ph\":\"M\",";
  evt << "\"name\":\"process_sort_index\",";
  evt << "\"pid\":" << process_id << ",";
  evt << "\"args\":{\"sort_index\":\"" << process_id << "\"}";
  evt << "}";
  return evt.str();
}

std::string MemoryInfo::MemoryInfoProfile::CreateMemoryEvent(size_t pid, size_t tid, const std::string& name, size_t offset, size_t size, const std::string& color_name) {
  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"X\",";
  evt << "\"pid\":" << pid << ",";
  evt << "\"tid\":" << tid << ",";
  evt << "\"ts\":" << offset << ",";
  evt << "\"dur\":" << size << ",";
  evt << "\"name\":\"" << name << "\",";
  evt << "\"cname\":\"" << color_name << "\",";
  evt << "\"args\":{";
  evt << "\"name\":\"" << name << "\",";
  evt << "\"offset\":" << offset << ",";
  evt << "\"size\":" << size;
  evt << "}";
  evt << "}";
  return evt.str();
}

std::string MemoryInfo::MemoryInfoProfile::CreateSummaryEvent(size_t pid, size_t tid, const AllocationSummary& summary, size_t size, size_t bytes_for_pattern) {
  const size_t total_bytes = summary.total_size;
  const size_t used_bytes = summary.used_size;
  const size_t free_bytes = total_bytes - used_bytes;
  const float used_percent = (float)used_bytes / (float)total_bytes;
  const float free_percent = (float)free_bytes / (float)total_bytes;

  std::stringstream evt;
  evt << "{";
  evt << "\"ph\":\"X\",";
  evt << "\"pid\":" << pid << ",";
  evt << "\"tid\":" << tid << ",";
  evt << "\"ts\":" << -(int64_t)size << ",";
  evt << "\"dur\":" << size << ",";
  evt << "\"name\":\"Summary\",";
  evt << "\"cname\":\"black\",";
  evt << "\"args\":{";
  evt << "\"total_bytes\":" << total_bytes << ",";
  evt << "\"used_bytes\":" << used_bytes << ",";
  evt << "\"free_bytes\":" << free_bytes << ",";
  evt << "\"used_percent\":" << used_percent << ",";
  evt << "\"free_percent\":" << free_percent << ",";
  evt << "\"bytes for pattern\":" << bytes_for_pattern;
  evt << "}";
  evt << "}";
  return evt.str();
}

static void UpdateSummary(MemoryInfo::AllocationSummary& summary, size_t alloc_offset, size_t alloc_size, const OrtValueIndex idx, const MemoryInfo::MapType& map_type) {
  summary.total_size = std::max(summary.total_size, alloc_offset + alloc_size);
  summary.used_size += alloc_size;
  if (map_type == MemoryInfo::MapType::DynamicActivation) {
    summary.total_size = summary.used_size;
  } else {
    summary.total_size = std::max(summary.total_size, alloc_offset + alloc_size);
  }
  summary.life_tensosrs.push_back(idx);
}

const std::vector<std::string> MemoryInfo::MemoryInfoProfile::color_names = {
    "good",
    "bad",
    "terrible",
    "yellow",
    "olive",
    "generic_work",
    "background_memory_dump",
    "light_memory_dump",
    "detailed_memory_dump",
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_unknown",
    "cq_build_running",
    "cq_build_passed",
    "cq_build_failed",
    "cq_build_abandoned",
    "cq_build_attempt_runnig",
    "cq_build_attempt_passed",
    "cq_build_attempt_failed",
};

void MemoryInfo::MemoryInfoProfile::CreateEvents(const std::string& p_name, const size_t pid, const MemoryInfo::MapType& map_type, const std::string& name_pattern, const size_t top_k) {
  // Metadata.
  std::string pid_name_internal = p_name + name_pattern;
  events.push_back(CreateMetadataEvent(pid_name_internal, pid));
  size_t summary_size = 10;
  std::hash<std::string> str_hash;

  //Create Event for each tensor
  for (const auto& location_map : mem_info_.tensors_memory_info_map_) {
    if (location_map.first.device.Type() != OrtDevice::GPU) continue;
    auto summary_key = str_hash(location_map.first.ToString() + std::to_string(map_type));
    //Preprocessing
    if (mem_info_.time_step_trace_[map_type].empty()) {
      for (const auto& item : location_map.second.at(map_type)) {
        mem_info_.time_step_trace_[map_type].insert(mem_info_.AllocPlan(item.first)->lifetime_interval.first);
        mem_info_.time_step_trace_[map_type].insert(mem_info_.AllocPlan(item.first)->lifetime_interval.second);
      }
    }

    const auto& map = location_map.second.at(map_type);
    //Update summary
    if (summary_.find(summary_key) == summary_.end()) {
      for (const auto& item : map) {
        const auto info = mem_info_.AllocPlan(item.first);
        if (info->inplace_reuse) continue;
        size_t offset = map_type == MemoryInfo::MapType::StaticActivation ? map.GetPlannedAddress(item.first) : map.GetAllocAddress(item.first);
        size_t size = map_type == MemoryInfo::MapType::StaticActivation ? map.GetPlannedSize(item.first) : map.GetAllocSize(item.first);
        size_t alloc_step = mem_info_.AllocPlan(item.first)->lifetime_interval.first;
        size_t dealloc_step = mem_info_.AllocPlan(item.first)->lifetime_interval.second;
        const auto& ts_map = mem_info_.time_step_trace_[map_type];
        const auto& start_itr = ts_map.find(alloc_step);
        const auto& end_itr = ts_map.find(dealloc_step);
        for (auto itr = start_itr; itr != end_itr; ++itr) {
          UpdateSummary(summary_[summary_key][*itr], offset, size, item.first, map_type);
        }
        UpdateSummary(summary_[summary_key][*end_itr], offset, size, item.first, map_type);
      }
    }

    //extract top_k total size
    size_t top_kth_total_size = 0;
    if (top_k != 0) {
      std::set<size_t> top_k_set;
      for (const auto& item : summary_[summary_key]) {
        if (top_k_set.size() < top_k)
          top_k_set.insert(item.second.total_size);
        else if (*top_k_set.cbegin() < item.second.total_size) {
          top_k_set.erase(top_k_set.begin());
          top_k_set.insert(item.second.total_size);
        }
      }
      top_kth_total_size = *top_k_set.cbegin();
    }

    for (const auto& item : summary_[summary_key]) {
      if (top_k != 0 && item.second.total_size < top_kth_total_size) continue;
      size_t alloc_size_for_pattern = 0;
      for (const auto& live_tensor : item.second.life_tensosrs) {
        const auto info = mem_info_.AllocPlan(live_tensor);
        if (info->inplace_reuse) continue;
        const std::string& name = info->mlvalue_name;
        //Filter out string without a certain name
        if (!name_pattern.empty() && name.find(name_pattern) == std::string::npos) continue;
        const std::string cname = color_names[str_hash(name) % color_names.size()];
        //Sometimes a tensor can be both statically planned and dynamically allocated, so we need to use planned address/size in static_activation type
        size_t offset = map_type == MemoryInfo::MapType::StaticActivation ? map.GetPlannedAddress(live_tensor) : map.GetAllocAddress(live_tensor);
        size_t size = map_type == MemoryInfo::MapType::StaticActivation ? map.GetPlannedSize(live_tensor) : map.GetAllocSize(live_tensor);
        alloc_size_for_pattern += size;
        events.push_back(CreateMemoryEvent(pid, item.first, name, offset, size, cname));
      }
      // Add summary events.  These will show up visually as 10% of the max width in a section.
      summary_size = std::max(summary_size, item.second.total_size / 10);
      events.push_back(CreateSummaryEvent(pid, item.first, item.second, summary_size, alloc_size_for_pattern));
    }
  }
}

void MemoryInfo::GenerateMemoryProfile() {
  profiler.CreateEvents("GPU (initializer)", profiler.GetAndIncreasePid(), MapType::Initializer, "", 1);
  profiler.CreateEvents("GPU (static activations)", profiler.GetAndIncreasePid(), MapType::StaticActivation, "", 1);
  profiler.CreateEvents("GPU (dynamic activations)", profiler.GetAndIncreasePid(), MapType::DynamicActivation, "", 1);
  profiler.CreateEvents("GPU (static activations)", profiler.GetAndIncreasePid(), MapType::StaticActivation, "_grad", 0);
  profiler.CreateEvents("GPU (dynamic activations)", profiler.GetAndIncreasePid(), MapType::DynamicActivation, "_grad", 0);
  profiler.CreateEvents("GPU (static activations)", profiler.GetAndIncreasePid(), MapType::StaticActivation, "_partition_", 0);
  profiler.CreateEvents("GPU (dynamic activations)", profiler.GetAndIncreasePid(), MapType::DynamicActivation, "_partition_", 0);

  // Write memory profile .json
  std::ofstream memory_profile("memory_profile_" + std::to_string(local_rank_) + ".json", std::ios::trunc);
  memory_profile << "[" << std::endl;
  for (size_t i = 0; i < profiler.events.size(); i++) {
    memory_profile << "  " << profiler.events[i];
    if (i < profiler.events.size() - 1) memory_profile << ",";
    memory_profile << std::endl;
  }
  memory_profile << "]" << std::endl;
  memory_profile.close();
}

}  // namespace onnxruntime

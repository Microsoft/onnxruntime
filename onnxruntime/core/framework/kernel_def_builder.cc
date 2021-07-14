// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_def_builder.h"

#include <algorithm>
#include <unordered_set>
#include <string>

#include "gsl/gsl"

#include "core/framework/murmurhash3.h"

namespace onnxruntime {
namespace {

//assume start1 <= end1, start2 <= end2
inline bool AreIntervalsOverlap(int start1, int end1, int start2, int end2) {
  return start1 <= end2 && start2 <= end1;
}

template <typename T>
inline bool AreVectorsOverlap(const std::vector<T>& v1, const std::vector<T>& v2) {
  for (const T& type : v1) {
    if (std::find(v2.begin(), v2.end(), type) != v2.end()) {
      return true;
    }
  }
  return false;
}

}  // namespace

// special node attribute name for fused node to specify input/output MemType
// the attribute would be an array of ints
const std::string KernelDefBuilder::NodeAttr_InputMemType = "InputMemType";
const std::string KernelDefBuilder::NodeAttr_OutputMemType = "OutputMemType";

void KernelDef::CalculateHash() {
  uint32_t hash[4] = {0, 0, 0, 0};

  auto hash_int = [&hash](int i) { MurmurHash3::x86_128(&i, sizeof(i), hash[0], &hash); };
  auto hash_str = [&hash](const std::string& str) {
    MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
  };

  // use name, start/end, domain, provider and the type constraints.
  // we wouldn't have two kernels that only differed by the inplace or alias info or memory types.
  // currently nothing sets exec_queue_id either (and would assumably be a runtime thing and not part of the base
  // kernel definition)

  hash_str(op_name_);
  hash_int(op_since_version_start_);

  // If we include op_since_version_end_ the hash of an existing op changes when it's superseded.
  // e.g. Unsqueeze 11 had no end version until Unsqueeze 13, at which point the existing op is changed to have
  // an end version of 12. That would result in a new ORT build having a different hash for Unsqueeze 11 and a
  // previously serialized ORT format model wouldn't find the kernel. In order to select the kernel to include
  // in the ORT model the full OpSchema info is used, so it's safe to exclude op_since_version_end_ from the hash.

  hash_str(op_domain_);
  hash_str(provider_type_);

  // use the hash_type_constraints_ or default_type_constraints_ list for the hash so the value in an ORT format model
  // is stable.
  const auto& hash_type_constraints =
      hash_type_constraints_.has_value() ? *hash_type_constraints_ : default_type_constraints_;
  for (const auto& key_value : hash_type_constraints) {
    hash_str(key_value.first);
    auto data_type_strings = DataTypeImpl::ToString(key_value.second);
    // sort type constraint data type strings so that order does not matter
    std::sort(data_type_strings.begin(), data_type_strings.end());
    for (const auto& data_type_string : data_type_strings) {
      hash_str(data_type_string);
    }
  }

  hash_ = hash[0] & 0xfffffff8;  // save low 3 bits for hash version info in case we need it in the future
  hash_ |= uint64_t(hash[1]) << 32;
}

// TODO: Tell user why it has conflicts
// TODO: Investigate why IsConflict() was not triggered when there were duplicate Tile CUDA
// kernels registered. Removing `InputMemoryType(OrtMemTypeCPUInput, 1)` in the kernel definition
// triggered the conflict.
bool KernelDef::IsConflict(const KernelDef& other) const {
  if (op_name_ != other.OpName() || provider_type_ != other.Provider())
    return false;
  int other_since_version_start = 0;
  int other_since_version_end = 0;
  other.SinceVersion(&other_since_version_start, &other_since_version_end);

  //When max version is INT_MAX, it means that it should be determined based on the
  //SinceVersion of schema from a higher version.  Since this sometimes isn't known until
  //all custom schema are available, make a conservative assumption here that the operator
  //is valid for only one version.
  int op_since_version_conservative_end = (op_since_version_end_ == INT_MAX) ? op_since_version_start_ : op_since_version_end_;
  int other_conservative_since_version_end = (other_since_version_end == INT_MAX) ? other_since_version_start : other_since_version_end;

  if (!AreIntervalsOverlap(op_since_version_start_, op_since_version_conservative_end, other_since_version_start, other_conservative_since_version_end))
    return false;
  //only one case they don't conflict:
  //There is a type_constraint, it exists in both hands, but they don't overlap
  //check types
  const auto& other_types = other.default_type_constraints_;
  bool type_has_conflict = true;
  for (const auto& it : default_type_constraints_) {
    auto iter = other_types.find(it.first);
    if (iter != other_types.end()) {
      if (!AreVectorsOverlap(it.second, iter->second)) {
        type_has_conflict = false;
        break;
      }
    }
  }
  if (!type_has_conflict)
    return false;
  //if has type conflict, check if any other field has different
  //for example, we register two kernel with float type, but one is inplace, another is not.
  //check in-place
  if (inplace_map_.empty() && !other.MayInplace().empty())
    return false;
  for (auto& it : inplace_map_) {
    if (std::find(other.MayInplace().begin(), other.MayInplace().end(), it) == other.MayInplace().end())
      return false;
  }

  //check alias
  for (auto& it : alias_map_) {
    if (std::find(other.Alias().begin(), other.Alias().end(), it) == other.Alias().end())
      return false;
  }
  if (alias_map_.empty() && !other.Alias().empty())
    return false;

  //check memory type
  auto& other_input_mem_types = other.input_memory_type_args_;
  for (auto it : input_memory_type_args_) {
    if (other_input_mem_types.count(it.first) && other_input_mem_types.find(it.first)->second == it.second)
      return false;
  }
  if (input_memory_type_args_.empty() && !other.input_memory_type_args_.empty())
    return false;

  auto& other_output_mem_types = other.output_memory_type_args_;
  for (auto it : output_memory_type_args_) {
    if (other_output_mem_types.count(it.first) && other_output_mem_types.find(it.second)->second == it.second)
      return false;
  }
  return !(output_memory_type_args_.empty() && !other.output_memory_type_args_.empty());
}

KernelDefBuilder& KernelDefBuilder::SetName(const std::string& op_name) {
  kernel_def_->op_name_ = op_name;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetName(const char* op_name) {
  kernel_def_->op_name_ = std::string(op_name);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetDomain(const std::string& domain) {
  kernel_def_->op_domain_ = domain;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::SetDomain(const char* domain) {
  kernel_def_->op_domain_ = std::string(domain);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Provider(onnxruntime::ProviderType provider_type) {
  kernel_def_->provider_type_ = provider_type;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Provider(const char* provider_type) {
  kernel_def_->provider_type_ = std::string(provider_type);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraintImpl(const std::string& arg_name,
                                                       const std::vector<MLDataType>& default_types,
                                                       const std::vector<MLDataType>* enabled_types) {
  // use the enabled types list if provided
  kernel_def_->enabled_type_constraints_[arg_name] = enabled_types ? *enabled_types : default_types;
  kernel_def_->default_type_constraints_[arg_name] = default_types;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const std::string& arg_name,
                                                   const std::vector<MLDataType>& default_types) {
  return TypeConstraintImpl(arg_name, default_types, nullptr);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* arg_name,
                                                   const std::vector<MLDataType>& default_types) {
  return TypeConstraintImpl(arg_name, default_types, nullptr);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const std::string& arg_name,
                                                   const std::vector<MLDataType>& default_types,
                                                   const std::vector<MLDataType>& enabled_types) {
  return TypeConstraintImpl(arg_name, default_types, &enabled_types);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* arg_name,
                                                   const std::vector<MLDataType>& default_types,
                                                   const std::vector<MLDataType>& enabled_types) {
  return TypeConstraintImpl(arg_name, default_types, &enabled_types);
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const std::string& arg_name,
                                                   MLDataType default_type) {
  kernel_def_->enabled_type_constraints_[arg_name] = std::vector<MLDataType>{default_type};
  kernel_def_->default_type_constraints_[arg_name] = std::vector<MLDataType>{default_type};
  return *this;
}

KernelDefBuilder& KernelDefBuilder::TypeConstraint(const char* arg_name,
                                                   MLDataType default_type) {
  return TypeConstraint(std::string(arg_name), default_type);
}

KernelDefBuilder& KernelDefBuilder::FixedTypeConstraintForHash(
    const std::string& arg_name,
    const std::vector<MLDataType>& default_types_for_hash) {
  auto& hash_type_constraints = kernel_def_->hash_type_constraints_;
  if (!hash_type_constraints.has_value()) {
    hash_type_constraints.emplace();
  }
  (*hash_type_constraints)[arg_name] = default_types_for_hash;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::FixedTypeConstraintForHash(
    const char* arg_name,
    const std::vector<MLDataType>& default_types_for_hash) {
  return FixedTypeConstraintForHash(std::string{arg_name}, default_types_for_hash);
}

KernelDefBuilder& KernelDefBuilder::MayInplace(const std::vector<std::pair<int, int>>& inplaces) {
  kernel_def_->inplace_map_ = inplaces;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::MayInplace(int input_index, int output_index) {
  // TODO: validate inputs.
  kernel_def_->inplace_map_.emplace_back(input_index, output_index);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Alias(const std::vector<std::pair<int, int>>& aliases) {
  kernel_def_->alias_map_ = aliases;
  return *this;
}

KernelDefBuilder& KernelDefBuilder::Alias(int input_index, int output_index) {
  kernel_def_->alias_map_.emplace_back(input_index, output_index);
  return *this;
}

KernelDefBuilder& KernelDefBuilder::VariadicAlias(int input_offset, int output_offset) {
  ORT_ENFORCE(input_offset >= 0 && output_offset >= 0);
  kernel_def_->variadic_alias_offsets_ = std::make_pair(input_offset, output_offset);
  return *this;
}

}  // namespace onnxruntime

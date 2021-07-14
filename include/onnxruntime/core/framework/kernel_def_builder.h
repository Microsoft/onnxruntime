// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <limits.h>

#include "core/common/common.h"
#include "core/common/optional.h"
#include "core/graph/basic_types.h"
#include "core/framework/data_types.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
class KernelDefBuilder;

typedef std::map<size_t, OrtMemType> MemTypeMap;

class KernelDef {
 private:
  // note that input/output might be on CPU implicitly when the node is from CPU execution provider
  static inline bool MemTypeOnCpuExplicitly(OrtMemType mem_type) {
    return mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput;
  }

 public:
  explicit KernelDef() = default;

  const std::string& OpName() const {
    return op_name_;
  }

  const std::string& Domain() const {
    return op_domain_;
  }

  void SinceVersion(/*out*/ int* start, /*out*/ int* end) const {
    *start = op_since_version_start_;
    *end = op_since_version_end_;
  }

#ifdef onnxruntime_PYBIND_EXPORT_OPSCHEMA
  const std::pair<int, int> SinceVersion() const {
    return std::pair<int, int>(op_since_version_start_, op_since_version_end_);
  }
#endif

  onnxruntime::ProviderType Provider() const {
    return provider_type_;
  }

  // type constraints with types supported by default
  const std::map<std::string, std::vector<MLDataType>>& TypeConstraints() const {
    return default_type_constraints_;
  }

  // type constraints with types supported in this build
  const std::map<std::string, std::vector<MLDataType>>& EnabledTypeConstraints() const {
    return enabled_type_constraints_;
  }

  const std::vector<std::pair<int, int>>& MayInplace() const {
    return inplace_map_;
  }

  const std::vector<std::pair<int, int>>& Alias() const {
    return alias_map_;
  }

  const optional<std::pair<int, int>>& VariadicAlias() const {
    return variadic_alias_offsets_;
  }

  OrtMemType InputMemoryType(size_t input_index) const {
    auto it = input_memory_type_args_.find(input_index);
    if (it == input_memory_type_args_.end())
      return default_inputs_mem_type_;
    return it->second;
  }

  bool IsInputOnCpu(size_t input_index) const { return MemTypeOnCpuExplicitly(InputMemoryType(input_index)); }

  bool IsOutputOnCpu(size_t output_index) const { return MemTypeOnCpuExplicitly(OutputMemoryType(output_index)); }

  bool AllocateInputsContiguously() const { return allocate_inputs_contiguously_; }

  bool HasExternalOutputs() const { return external_outputs_; }

  OrtMemType OutputMemoryType(size_t output_index) const {
    auto it = output_memory_type_args_.find(output_index);
    if (it == output_memory_type_args_.end())
      return default_outputs_mem_type_;
    return it->second;
  }

  int ExecQueueId() const {
    return exec_queue_id_;
  }

  bool IsConflict(const KernelDef& other) const;

  uint64_t GetHash() const noexcept {
    // if we need to support different hash versions we can update CalculateHash to take a version number
    // and calculate any non-default versions dynamically. we only use this during kernel lookup so
    // it's not performance critical
    return hash_;
  }

 private:
  friend class KernelDefBuilder;

  // called once by KernelDefBuilder::Build
  void CalculateHash();

  // The operator name supported by <*this> kernel..
  std::string op_name_;

  // The operator since_version range supported by <*this> kernel.
  // A kernel could support an operator definition between <op_since_version_start>
  // and <op_since_version_end> (inclusive).
  int op_since_version_start_ = 1;
  int op_since_version_end_ = INT_MAX;

  // The operator domain supported by <*this> kernel.
  // Default to 'onnxruntime::kOnnxDomain'.
  // Please note the behavior of std::string("") and std::string() are different
  std::string op_domain_;

  // The type of the execution provider.
  std::string provider_type_;

  // The data types that are supported by default for inputs/outputs.
  // Key is input/output name defined in op schema, Value are supported types.
  // note: std::map as we need the order to be deterministic for the hash
  // Note: default_type_constraints_ are used to calculate the kernel hash so that the hash is
  // stable across builds with and without kernel type reduction enabled.
  std::map<std::string, std::vector<MLDataType>> default_type_constraints_;

  // the type constraints that are supported in this build (enabled) for the kernel
  std::map<std::string, std::vector<MLDataType>> enabled_type_constraints_;

  // optional alternate type constraints to use to calculate the hash instead of default_type_constraints_
  // note: this provides a way to update the default type constraints while preserving the hash value
  optional<std::map<std::string, std::vector<MLDataType>>> hash_type_constraints_;

  // An element <i, j> means that output j reuses the memory of input i.
  std::vector<std::pair<int, int>> inplace_map_;

  // An element <i, j> means that output j is an alias of input i.
  std::vector<std::pair<int, int>> alias_map_;

  // This variable stores <input_offset, output_offset> for the variadic alias mapping
  // output 'i + output_offset' is an alias of input 'i + input_offset' for all i >= 0
  optional<std::pair<int, int>> variadic_alias_offsets_;

  // Require input tensors to be allocated contiguously.
  bool allocate_inputs_contiguously_ = false;

  // Whether the outputs are from external.
  bool external_outputs_ = false;

  // The memory types of inputs/outputs of this kernel
  MemTypeMap input_memory_type_args_;
  MemTypeMap output_memory_type_args_;

  // execution command queue id, 0 for default queue in execution provider
  int exec_queue_id_ = 0;
  // Default memory type for all inputs
  OrtMemType default_inputs_mem_type_{OrtMemTypeDefault};
  // Default memory type for all outputs
  OrtMemType default_outputs_mem_type_{OrtMemTypeDefault};

  // hash of kernel definition for lookup in minimal build
  uint64_t hash_ = 0;
};

class KernelDefBuilder {
 public:
  static std::unique_ptr<KernelDefBuilder> Create() { return std::make_unique<KernelDefBuilder>(); }

  explicit KernelDefBuilder()
      : kernel_def_(new KernelDef()) {}

  KernelDefBuilder& SetName(const std::string& op_name);
  KernelDefBuilder& SetName(const char* op_name);

  KernelDefBuilder& SetDomain(const std::string& domain);
  KernelDefBuilder& SetDomain(const char* domain);

  /**
     This kernel supports operator definition since <since_version> (to latest).
  */
  KernelDefBuilder& SinceVersion(int since_version) {
    kernel_def_->op_since_version_start_ = since_version;
    return *this;
  }

  /**
     The start and end version should be set accordingly per version range for
     each domain registered in OpSchemaRegistry::DomainToVersionRange in
     \onnxruntime\onnxruntime\core\graph\op.h as below.
     Key: domain. Value: <lowest version, highest version> pair.
     std::unordered_map<std::string, std::pair<int, int>> map_;
  */
  KernelDefBuilder& SinceVersion(int since_version_start, int since_version_end) {
    kernel_def_->op_since_version_start_ = since_version_start;
    kernel_def_->op_since_version_end_ = since_version_end;
    return *this;
  }

  /**
     The execution provider type of the kernel.
  */
  KernelDefBuilder& Provider(onnxruntime::ProviderType provider_type);
  KernelDefBuilder& Provider(const char* provider_type);

  /**
     Specify the set of types that this kernel supports. A further restriction
     of the set of types specified in the op schema.

     @param arg_name The arg name can be either op formal parameter name, say "X", or type
                     argument name specified in op schema, say "T".
     @param default_types The types that are supported by default.
     @param enabled_types The types that are supported in this build.
                          Possibly different from default_types when type reduction is enabled.
  */
  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                                   const std::vector<MLDataType>& default_types);
  KernelDefBuilder& TypeConstraint(const char* arg_name,
                                   const std::vector<MLDataType>& default_types);

  KernelDefBuilder& TypeConstraint(const std::string& arg_name,
                                   const std::vector<MLDataType>& default_types,
                                   const std::vector<MLDataType>& enabled_types);
  KernelDefBuilder& TypeConstraint(const char* arg_name,
                                   const std::vector<MLDataType>& default_types,
                                   const std::vector<MLDataType>& enabled_types);

  /**
     Like TypeConstraint but supports just a single type.
  */
  KernelDefBuilder& TypeConstraint(const std::string& arg_name, MLDataType default_type);
  KernelDefBuilder& TypeConstraint(const char* arg_name, MLDataType default_type);

  /**
     Specify the original set of types that this kernel supports by default to use when computing the kernel def hash.
     The set of types supported by default may change over time, but the hash should stay the same.
  */
  KernelDefBuilder& FixedTypeConstraintForHash(
      const std::string& arg_name,
      const std::vector<MLDataType>& default_types_for_hash);
  KernelDefBuilder& FixedTypeConstraintForHash(
      const char* arg_name,
      const std::vector<MLDataType>& default_types_for_hash);

  /**
     Inplace mapping from inputs to outputs allowed.
     It means that uplayer runtime could do memory in-place optimization
     as it will not impact the correctness of this kernel.
  */
  KernelDefBuilder& MayInplace(const std::vector<std::pair<int, int>>& inplaces);
  KernelDefBuilder& MayInplace(int input_index, int output_index);

  /**
     Alias mapping from inputs to outputs. Different from Inplace that the
     content of the tensor is not changed. This is to take care of operators
     such as Identity and Reshape.
  */
  KernelDefBuilder& Alias(const std::vector<std::pair<int, int>>& aliases);
  KernelDefBuilder& Alias(int input_index, int output_index);

  /**
     Apply variadic number of alias mapping from inputs to outputs. 
     This is effectively applying Alias(i + input_offset, i + output_offset) for i >= 0
  */
  KernelDefBuilder& VariadicAlias(int input_offset, int output_offset);

  /**
     Specify that this kernel requires input tensors to be allocated
     contiguously. This allows kernels to execute as a single large
     computation, rather than numerous smaller computations.
  */
  KernelDefBuilder& AllocateInputsContiguously() {
    kernel_def_->allocate_inputs_contiguously_ = true;
    return *this;
  }

  /**
     Specify that this kernel's output buffers are passed from external, 
     i.e. not created or managed by ORT's memory allocator.
  */
  KernelDefBuilder& ExternalOutputs() {
    kernel_def_->external_outputs_ = true;
    return *this;
  }

  /**
     Specify that this kernel requires an input arg
     in certain memory type (instead of the default, device memory).
  */
  KernelDefBuilder& InputMemoryType(OrtMemType type, int input_index) {
    kernel_def_->input_memory_type_args_.insert(std::make_pair(input_index, type));
    return *this;
  }

  /**
     Specify that this kernel requires input arguments
     in certain memory type (instead of the default, device memory).
  */
  KernelDefBuilder& InputMemoryType(OrtMemType type, const std::vector<int>& input_indexes) {
    for (auto input_index : input_indexes) {
      kernel_def_->input_memory_type_args_.insert(std::make_pair(input_index, type));
    }
    return *this;
  }

  /**
     Specify that this kernel provides an output arg
     in certain memory type (instead of the default, device memory).
  */
  KernelDefBuilder& OutputMemoryType(OrtMemType type, int output_index) {
    kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, type));
    return *this;
  }

  /**
     Specify that this kernel provides an output arguments
     in certain memory type (instead of the default, device memory).
  */
  KernelDefBuilder& OutputMemoryType(OrtMemType type, const std::vector<int>& output_indexes) {
    for (auto output_index : output_indexes) {
      kernel_def_->output_memory_type_args_.insert(std::make_pair(output_index, type));
    }
    return *this;
  }

  /**
     Specify that this kernel runs on which execution queue in the provider
  */
  KernelDefBuilder& ExecQueueId(int queue_id) {
    kernel_def_->exec_queue_id_ = queue_id;
    return *this;
  }

  /**
  Specify the default inputs memory type, if not specified, it is DefaultMemory
  */
  KernelDefBuilder& SetDefaultInputsMemoryType(OrtMemType mem_type) {
    kernel_def_->default_inputs_mem_type_ = mem_type;
    return *this;
  }

  /**
  Specify the default outputs memory type, if not specified, it is DefaultMemory
  */
  KernelDefBuilder& SetDefaultOutputMemoryType(OrtMemType mem_type) {
    kernel_def_->default_outputs_mem_type_ = mem_type;
    return *this;
  }

  /**
     Return the kernel definition, passing ownership of the KernelDef to the caller
  */
  std::unique_ptr<KernelDef> Build() {
    kernel_def_->CalculateHash();
    return std::move(kernel_def_);
  }

  // special node attribute name for fused node to specify input/output MemType
  // the attribute would be an array of ints
  static const std::string NodeAttr_InputMemType;
  static const std::string NodeAttr_OutputMemType;

 private:
  KernelDefBuilder& TypeConstraintImpl(const std::string& arg_name,
                                       const std::vector<MLDataType>& default_types,
                                       const std::vector<MLDataType>* enabled_types = nullptr);

  // we own the KernelDef until Build() is called.
  std::unique_ptr<KernelDef> kernel_def_;
};

}  // namespace onnxruntime

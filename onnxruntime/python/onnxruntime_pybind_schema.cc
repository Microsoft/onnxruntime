// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_state_common.h"
#include "core/framework/kernel_registry.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/utils.h"
#include "core/session/abi_devices.h"
#endif
#include <pybind11/stl.h>

namespace py = pybind11;

namespace onnxruntime {
namespace python {

void addGlobalSchemaFunctions(pybind11::module& m) {
  m.def(
      "get_all_operator_schema", []() -> const std::vector<ONNX_NAMESPACE::OpSchema> {
        return ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas_with_history();
      },
      "Return a vector of OpSchema all registered operators");
  m.def(
      "get_all_opkernel_def", []() -> const std::vector<onnxruntime::KernelDef> {
        std::vector<onnxruntime::KernelDef> result;

        std::vector<std::shared_ptr<onnxruntime::IExecutionProviderFactory>> factories = {
            onnxruntime::CPUProviderFactoryCreator::Create(0),
#ifdef USE_CUDA
            []() {
              OrtCUDAProviderOptions provider_options{};
              return CudaProviderFactoryCreator::Create(&provider_options);
            }(),
#endif
#ifdef USE_DNNL
            onnxruntime::DnnlProviderFactoryCreator::Create(1),
#endif
#ifdef USE_OPENVINO
            []() {
              ProviderOptions provider_options_map;
              SessionOptions session_options;
              return onnxruntime::OpenVINOProviderFactoryCreator::Create(&provider_options_map, &session_options);
            }(),
#endif
#ifdef USE_TENSORRT
            onnxruntime::TensorrtProviderFactoryCreator::Create(0),
#endif
#ifdef USE_MIGRAPHX
            onnxruntime::MIGraphXProviderFactoryCreator::Create(0),
#endif
#ifdef USE_VITISAI
            onnxruntime::VitisAIProviderFactoryCreator::Create(ProviderOptions{}),
#endif
#ifdef USE_ACL
            onnxruntime::ACLProviderFactoryCreator::Create(false),
#endif
#ifdef USE_DML
            []() {
              ConfigOptions config_options{};
              return onnxruntime::DMLProviderFactoryCreator::Create(config_options, 0, false, false, false);
            }(),
#endif
#ifdef USE_NNAPI
            onnxruntime::NnapiProviderFactoryCreator::Create(0, std::optional<std::string>()),
#endif
#ifdef USE_VSINPU
            onnxruntime::VSINPUProviderFactoryCreator::Create(),
#endif
#ifdef USE_RKNPU
            onnxruntime::RknpuProviderFactoryCreator::Create(),
#endif
#ifdef USE_COREML
            onnxruntime::CoreMLProviderFactoryCreator::Create(ProviderOptions{}),
#endif
#ifdef USE_XNNPACK
            onnxruntime::XnnpackProviderFactoryCreator::Create(ProviderOptions{}, nullptr),
#endif
#ifdef USE_CANN
            []() {
              OrtCANNProviderOptions provider_options{};
              return CannProviderFactoryCreator::Create(&provider_options);
            }(),
#endif
        };

        for (const auto& f : factories) {
          auto kernel_registry = f->CreateProvider()->GetKernelRegistry();
          for (const auto& m : kernel_registry->GetKernelCreateMap()) {
            result.emplace_back(*(m.second.kernel_def));
          }
        }

        return result;
      },
      "Return a vector of KernelDef for all registered OpKernels");

#if !defined(ORT_MINIMAL_BUILD)
  m.def(
      "get_registered_ep_kernel_defs",
      [](const std::string& ep_name) -> std::vector<onnxruntime::KernelDef> {
        std::vector<onnxruntime::KernelDef> result;
        auto& env = GetEnv();

        // Collect all OrtEpDevice pointers matching the requested EP name.
        std::vector<const OrtEpDevice*> selected_devices;
        for (const OrtEpDevice* device : env.GetOrtEpDevices()) {
          if (device && device->ep_name == ep_name) {
            selected_devices.push_back(device);
            break;  // one device is sufficient to create the factory and query kernels
          }
        }

        if (selected_devices.empty()) {
          throw std::runtime_error(
              "No devices found for EP '" + ep_name +
              "'. Ensure the plugin EP library is registered via register_execution_provider_library().");
        }

        // Create a factory for the plugin EP.
        std::unique_ptr<IExecutionProviderFactory> factory;
        auto status = CreateIExecutionProviderFactoryForEpDevices(env, selected_devices, factory);
        if (!status.IsOK()) {
          throw std::runtime_error("Failed to create EP factory for '" + ep_name + "': " + status.ToString());
        }

        // Create an EP instance with default session options.
        OrtSessionOptions ort_session_options{};
        const auto& logger = *env.GetLoggingManager()->DefaultLogger().ToExternal();
        auto provider = factory->CreateProvider(ort_session_options, logger);
        if (!provider) {
          throw std::runtime_error("Failed to create EP instance for '" + ep_name + "'.");
        }

        // Extract kernel defs from the EP's kernel registry.
        auto kernel_registry = provider->GetKernelRegistry();
        if (kernel_registry) {
          for (const auto& entry : kernel_registry->GetKernelCreateMap()) {
            result.emplace_back(*(entry.second.kernel_def));
          }
        }

        return result;
      },
      py::arg("ep_name"),
      "Return a vector of KernelDef for a dynamically registered plugin EP.\n"
      "The EP must be loaded first via register_execution_provider_library().");
#endif
}

void addOpKernelSubmodule(py::module& m) {
  auto opkernel = m.def_submodule("opkernel");
  opkernel.doc() = "OpKernel submodule";
  py::class_<onnxruntime::KernelDef> kernel_def(opkernel, "KernelDef");
  kernel_def.def_property_readonly("op_name", &onnxruntime::KernelDef::OpName)
      .def_property_readonly("domain", &onnxruntime::KernelDef::Domain)
      .def_property_readonly("provider", &onnxruntime::KernelDef::Provider)
      .def_property_readonly("version_range",
                             [](const onnxruntime::KernelDef& kernelDef) -> std::pair<int, int> {
                               return kernelDef.onnxruntime::KernelDef::SinceVersion();
                             })
      .def_property_readonly("type_constraints",
                             [](const onnxruntime::KernelDef& kernelDef) -> std::unordered_map<std::string, std::vector<std::string>> {
                               std::unordered_map<std::string, std::vector<std::string>> result;
                               const auto& tempResult = kernelDef.TypeConstraints();
                               for (const auto& tc : tempResult) {
                                 result[tc.first] = std::vector<std::string>();
                                 for (const auto& dt : tc.second) {
                                   result[tc.first].emplace_back(onnxruntime::DataTypeImpl::ToString(dt));
                                 }
                               }
                               return result;
                             });
}

void addOpSchemaSubmodule(py::module& m) {
  auto schemadef = m.def_submodule("schemadef");
  schemadef.doc() = "Schema submodule";

  // Keep this binding local to this module
  py::class_<ONNX_NAMESPACE::OpSchema> op_schema(schemadef, "OpSchema", py::module_local());
  op_schema.def_property_readonly("file", &ONNX_NAMESPACE::OpSchema::file)
      .def_property_readonly("line", &ONNX_NAMESPACE::OpSchema::line)
      .def_property_readonly("support_level", &ONNX_NAMESPACE::OpSchema::support_level)
      .def_property_readonly(
          "doc", &ONNX_NAMESPACE::OpSchema::doc, py::return_value_policy::reference)
      .def_property_readonly("since_version", &ONNX_NAMESPACE::OpSchema::since_version)
      .def_property_readonly("deprecated", &ONNX_NAMESPACE::OpSchema::deprecated)
      .def_property_readonly("domain", &ONNX_NAMESPACE::OpSchema::domain)
      .def_property_readonly("name", &ONNX_NAMESPACE::OpSchema::Name)
      .def_property_readonly("min_input", &ONNX_NAMESPACE::OpSchema::min_input)
      .def_property_readonly("max_input", &ONNX_NAMESPACE::OpSchema::max_input)
      .def_property_readonly("min_output", &ONNX_NAMESPACE::OpSchema::min_output)
      .def_property_readonly("max_output", &ONNX_NAMESPACE::OpSchema::max_output)
      .def_property_readonly("attributes", &ONNX_NAMESPACE::OpSchema::attributes)
      .def_property_readonly("inputs", &ONNX_NAMESPACE::OpSchema::inputs)
      .def_property_readonly("outputs", &ONNX_NAMESPACE::OpSchema::outputs)
      .def_property_readonly(
          "has_type_and_shape_inference_function",
          &ONNX_NAMESPACE::OpSchema::has_type_and_shape_inference_function)
      .def_property_readonly(
          "type_constraints", &ONNX_NAMESPACE::OpSchema::typeConstraintParams)
      .def_static("is_infinite", [](int v) {
        return v == std::numeric_limits<int>::max();
      });

  // Keep this binding local to this module
  py::class_<ONNX_NAMESPACE::OpSchema::Attribute>(op_schema, "Attribute", py::module_local())
      .def_readonly("name", &ONNX_NAMESPACE::OpSchema::Attribute::name)
      .def_readonly("description", &ONNX_NAMESPACE::OpSchema::Attribute::description)
      .def_readonly("type", &ONNX_NAMESPACE::OpSchema::Attribute::type)
      .def_property_readonly(
          "_default_value",
          [](ONNX_NAMESPACE::OpSchema::Attribute* attr) -> py::bytes {
            std::string out;
            attr->default_value.SerializeToString(&out);
            return out;
          })
      .def_readonly("required", &ONNX_NAMESPACE::OpSchema::Attribute::required);

  // Keep this binding local to this module
  py::class_<ONNX_NAMESPACE::OpSchema::TypeConstraintParam>(op_schema, "TypeConstraintParam", py::module_local())
      .def_readonly(
          "type_param_str", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::type_param_str)
      .def_readonly("description", &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::description)
      .def_readonly(
          "allowed_type_strs",
          &ONNX_NAMESPACE::OpSchema::TypeConstraintParam::allowed_type_strs);

  // Keep this binding local to this module
  py::enum_<ONNX_NAMESPACE::OpSchema::FormalParameterOption>(op_schema, "FormalParameterOption", py::module_local())
      .value("Single", ONNX_NAMESPACE::OpSchema::Single)
      .value("Optional", ONNX_NAMESPACE::OpSchema::Optional)
      .value("Variadic", ONNX_NAMESPACE::OpSchema::Variadic);

  // Keep this binding local to this module
  py::class_<ONNX_NAMESPACE::OpSchema::FormalParameter>(op_schema, "FormalParameter", py::module_local())
      .def_property_readonly("name", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetName)
      .def_property_readonly("types", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypes)
      .def_property_readonly("typeStr", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetTypeStr)
      .def_property_readonly(
          "description", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetDescription)
      .def_property_readonly("option", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetOption)
      .def_property_readonly(
          "isHomogeneous", &ONNX_NAMESPACE::OpSchema::FormalParameter::GetIsHomogeneous);

  // Keep this binding local to this module
  py::enum_<ONNX_NAMESPACE::AttributeProto::AttributeType>(op_schema, "AttrType", py::module_local())
      .value("FLOAT", ONNX_NAMESPACE::AttributeProto::FLOAT)
      .value("INT", ONNX_NAMESPACE::AttributeProto::INT)
      .value("STRING", ONNX_NAMESPACE::AttributeProto::STRING)
      .value("TENSOR", ONNX_NAMESPACE::AttributeProto::TENSOR)
      .value("SPARSE_TENSOR", ONNX_NAMESPACE::AttributeProto::SPARSE_TENSOR)
      .value("GRAPH", ONNX_NAMESPACE::AttributeProto::GRAPH)
      .value("FLOATS", ONNX_NAMESPACE::AttributeProto::FLOATS)
      .value("INTS", ONNX_NAMESPACE::AttributeProto::INTS)
      .value("STRINGS", ONNX_NAMESPACE::AttributeProto::STRINGS)
      .value("TENSORS", ONNX_NAMESPACE::AttributeProto::TENSORS)
      .value("SPARSE_TENSORS", ONNX_NAMESPACE::AttributeProto::SPARSE_TENSORS)
      .value("GRAPHS", ONNX_NAMESPACE::AttributeProto::GRAPHS);

  // Keep this binding local to this module
  py::enum_<ONNX_NAMESPACE::OpSchema::SupportType>(op_schema, "SupportType", py::module_local())
      .value("COMMON", ONNX_NAMESPACE::OpSchema::SupportType::COMMON)
      .value("EXPERIMENTAL", ONNX_NAMESPACE::OpSchema::SupportType::EXPERIMENTAL);
}
}  // namespace python
}  // namespace onnxruntime

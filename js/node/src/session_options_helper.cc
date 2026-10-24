// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_cxx_api.h"
#include <napi.h>

#include <cmath>
#include <unordered_map>
#include <filesystem>

#include "common.h"
#include "session_options_helper.h"
#include "tensor_helper.h"
#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#endif
#ifdef USE_DML
#include "core/providers/dml/dml_provider_factory.h"
#endif
#ifdef USE_WEBGPU
#include "core/providers/webgpu/webgpu_provider_factory.h"
#endif
#ifdef USE_TENSORRT
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#endif
#ifdef USE_COREML
#include "core/providers/coreml/coreml_provider_factory.h"
#endif

const std::unordered_map<std::string, GraphOptimizationLevel> GRAPH_OPT_LEVEL_NAME_TO_ID_MAP = {
    {"disabled", ORT_DISABLE_ALL},
    {"basic", ORT_ENABLE_BASIC},
    {"extended", ORT_ENABLE_EXTENDED},
    {"layout", ORT_ENABLE_LAYOUT},
    {"all", ORT_ENABLE_ALL}};

const std::unordered_map<std::string, ExecutionMode> EXECUTION_MODE_NAME_TO_ID_MAP = {{"sequential", ORT_SEQUENTIAL},
                                                                                      {"parallel", ORT_PARALLEL}};

void ParseExecutionProviders(const Napi::Array epList, Ort::SessionOptions& sessionOptions) {
  for (uint32_t i = 0; i < epList.Length(); i++) {
    Napi::Value epValue = epList[i];
    std::string name;
    int deviceId = 0;
#ifdef USE_COREML
    int coreMlFlags = 0;
#endif
#ifdef USE_WEBGPU
    std::unordered_map<std::string, std::string> webgpu_options;
#endif
#ifdef USE_QNN
    std::unordered_map<std::string, std::string> qnn_options;
#endif
    if (epValue.IsString()) {
      name = epValue.As<Napi::String>().Utf8Value();
    } else if (!epValue.IsObject() || epValue.IsNull() || !epValue.As<Napi::Object>().Has("name") ||
               !epValue.As<Napi::Object>().Get("name").IsString()) {
      ORT_NAPI_THROW_TYPEERROR(epList.Env(), "Invalid argument: sessionOptions.executionProviders[", i,
                               "] must be either a string or an object with property 'name'.");
    } else {
      auto obj = epValue.As<Napi::Object>();
      name = obj.Get("name").As<Napi::String>().Utf8Value();
      if (obj.Has("deviceId")) {
        deviceId = obj.Get("deviceId").As<Napi::Number>();
      }
#ifdef USE_COREML
      if (name == "coreml" && obj.Has("coreMlFlags")) {
        coreMlFlags = obj.Get("coreMlFlags").As<Napi::Number>();
      }
#endif
#ifdef USE_WEBGPU
      if (name == "webgpu") {
        for (const auto& nameIter : obj.GetPropertyNames()) {
          Napi::Value nameVar = nameIter.second;
          std::string name = nameVar.As<Napi::String>().Utf8Value();
          Napi::Value valueVar = obj.Get(nameVar);
          std::string value;
          if (name == "preferredLayout" ||
              name == "validationMode" ||
              name == "weightLoadAcceleration" ||
              name == "storageBufferCacheMode" ||
              name == "uniformBufferCacheMode" ||
              name == "queryResolveBufferCacheMode" ||
              name == "defaultBufferCacheMode") {
            ORT_NAPI_THROW_TYPEERROR_IF(!valueVar.IsString(), epList.Env(),
                                        "Invalid argument: \"", name, "\" must be a string.");
            value = valueVar.As<Napi::String>().Utf8Value();
          } else if (name == "enableRobustness") {
            ORT_NAPI_THROW_TYPEERROR_IF(!valueVar.IsBoolean(), epList.Env(),
                                        "Invalid argument: \"enableRobustness\" must be a boolean.");
            value = valueVar.As<Napi::Boolean>().Value() ? "1" : "0";
          } else if (name == "disableSubgroupMatrix") {
            ORT_NAPI_THROW_TYPEERROR_IF(!valueVar.IsBoolean(), epList.Env(),
                                        "Invalid argument: \"disableSubgroupMatrix\" must be a boolean.");
            value = valueVar.As<Napi::Boolean>().Value() ? "1" : "0";
          } else if (name == "enableMatmulFp32Accumulation") {
            ORT_NAPI_THROW_TYPEERROR_IF(!valueVar.IsBoolean(), epList.Env(),
                                        "Invalid argument: \"enableMatmulFp32Accumulation\" must be a boolean.");
            value = valueVar.As<Napi::Boolean>().Value() ? "1" : "0";
          } else if (name == "forceCpuNodeNames") {
            ORT_NAPI_THROW_TYPEERROR_IF(!valueVar.IsArray(), epList.Env(),
                                        "Invalid argument: \"forceCpuNodeNames\" must be a string array.");
            auto arr = valueVar.As<Napi::Array>();
            for (uint32_t i = 0; i < arr.Length(); i++) {
              Napi::Value v = arr[i];
              ORT_NAPI_THROW_TYPEERROR_IF(!v.IsString(), epList.Env(),
                                          "Invalid argument: elements of \"forceCpuNodeNames\" must be strings.");
              if (i > 0) {
                value += '\n';
              }
              value += v.As<Napi::String>().Utf8Value();
            }
          } else {
            // unrecognized option
            ORT_NAPI_THROW_TYPEERROR_IF(name != "name", epList.Env(),
                                        "Invalid argument: WebGPU EP has an unrecognized option: '", name, "'.");
            continue;
          }
          webgpu_options[name] = value;
        }
      }
#endif
#ifdef USE_QNN
      if (name == "qnn") {
        Napi::Value backend_type = obj.Get("backendType");
        if (!backend_type.IsUndefined()) {
          if (backend_type.IsString()) {
            qnn_options["backend_type"] = backend_type.As<Napi::String>().Utf8Value();
          } else {
            ORT_NAPI_THROW_TYPEERROR(epList.Env(), "Invalid argument: backendType must be a string.");
          }
        }
        Napi::Value backend_path = obj.Get("backendPath");
        if (!backend_path.IsUndefined()) {
          if (backend_path.IsString()) {
            qnn_options["backend_path"] = backend_path.As<Napi::String>().Utf8Value();
          } else {
            ORT_NAPI_THROW_TYPEERROR(epList.Env(), "Invalid argument: backendPath must be a string.");
          }
        }
        Napi::Value enable_htp_fp16_precision = obj.Get("enableFp16Precision");
        if (!enable_htp_fp16_precision.IsUndefined()) {
          if (enable_htp_fp16_precision.IsBoolean()) {
            qnn_options["enable_htp_fp16_precision"] = enable_htp_fp16_precision.As<Napi::Boolean>().Value() ? "1" : "0";
          } else {
            ORT_NAPI_THROW_TYPEERROR(epList.Env(), "Invalid argument: enableFp16Precision must be a boolean.");
          }
        }
      }
#endif
    }

    // CPU execution provider
    if (name == "cpu") {
      // TODO: handling CPU EP options
#ifdef USE_CUDA
    } else if (name == "cuda") {
      OrtCUDAProviderOptionsV2* options;
      Ort::ThrowOnError(Ort::GetApi().CreateCUDAProviderOptions(&options));
      options->device_id = deviceId;
      sessionOptions.AppendExecutionProvider_CUDA_V2(*options);
      Ort::GetApi().ReleaseCUDAProviderOptions(options);
#endif
#ifdef USE_TENSORRT
    } else if (name == "tensorrt") {
      OrtTensorRTProviderOptionsV2* options;
      Ort::ThrowOnError(Ort::GetApi().CreateTensorRTProviderOptions(&options));
      options->device_id = deviceId;
      sessionOptions.AppendExecutionProvider_TensorRT_V2(*options);
      Ort::GetApi().ReleaseTensorRTProviderOptions(options);
#endif
#ifdef USE_DML
    } else if (name == "dml") {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, deviceId));
#endif
#ifdef USE_WEBGPU
    } else if (name == "webgpu") {
      sessionOptions.AppendExecutionProvider("WebGPU", webgpu_options);
#endif
#ifdef USE_COREML
    } else if (name == "coreml") {
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sessionOptions, coreMlFlags));
#endif
#ifdef USE_QNN
    } else if (name == "qnn") {
      sessionOptions.AppendExecutionProvider("QNN", qnn_options);
#endif
    } else {
      ORT_NAPI_THROW_ERROR(epList.Env(), "Invalid argument: sessionOptions.executionProviders[", i,
                           "] is unsupported: '", name, "'.");
    }
  }
}

void IterateExtraOptions(const std::string& prefix, const Napi::Object& obj, Ort::SessionOptions& sessionOptions) {
  for (const auto& kvp : obj) {
    auto key = kvp.first.As<Napi::String>().Utf8Value();
    Napi::Value value = kvp.second;
    if (value.IsObject()) {
      IterateExtraOptions(prefix + key + ".", value.As<Napi::Object>(), sessionOptions);
    } else {
      ORT_NAPI_THROW_TYPEERROR_IF(!value.IsString(), obj.Env(),
                                  "Invalid argument: sessionOptions.extra value must be a string in Node.js binding.");
      std::string entry = prefix + key;
      auto val = value.As<Napi::String>().Utf8Value();
      sessionOptions.AddConfigEntry(entry.c_str(), val.c_str());
    }
  }
}

void ParseSessionOptions(const Napi::Object options, Ort::SessionOptions& sessionOptions,
                         std::vector<std::vector<char>>* externalDataBuffers) {
  // Execution provider
  if (options.Has("executionProviders")) {
    auto epsValue = options.Get("executionProviders");
    ORT_NAPI_THROW_TYPEERROR_IF(!epsValue.IsArray(), options.Env(),
                                "Invalid argument: sessionOptions.executionProviders must be an array.");
    ParseExecutionProviders(epsValue.As<Napi::Array>(), sessionOptions);
  }

  // Intra threads number
  if (options.Has("intraOpNumThreads")) {
    auto numValue = options.Get("intraOpNumThreads");
    ORT_NAPI_THROW_TYPEERROR_IF(!numValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.intraOpNumThreads must be a number.");
    double num = numValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(std::floor(num) != num || num < 0 || num > 4294967295, options.Env(),
                                 "'intraOpNumThreads' is invalid: ", num);
    sessionOptions.SetIntraOpNumThreads(static_cast<int>(num));
  }

  // Inter threads number
  if (options.Has("interOpNumThreads")) {
    auto numValue = options.Get("interOpNumThreads");
    ORT_NAPI_THROW_TYPEERROR_IF(!numValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.interOpNumThreads must be a number.");
    double num = numValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(std::floor(num) != num || num < 0 || num > 4294967295, options.Env(),
                                 "'interOpNumThreads' is invalid: ", num);
    sessionOptions.SetInterOpNumThreads(static_cast<int>(num));
  }

  // Optimization level
  if (options.Has("graphOptimizationLevel")) {
    auto optLevelValue = options.Get("graphOptimizationLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!optLevelValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.graphOptimizationLevel must be a string.");
    auto optLevelString = optLevelValue.As<Napi::String>().Utf8Value();
    auto v = GRAPH_OPT_LEVEL_NAME_TO_ID_MAP.find(optLevelString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == GRAPH_OPT_LEVEL_NAME_TO_ID_MAP.end(), options.Env(),
                                "'graphOptimizationLevel' is not supported: ", optLevelString);
    sessionOptions.SetGraphOptimizationLevel(v->second);
  }

  // CPU memory arena
  if (options.Has("enableCpuMemArena")) {
    auto enableCpuMemArenaValue = options.Get("enableCpuMemArena");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableCpuMemArenaValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableCpuMemArena must be a boolean value.");
    if (enableCpuMemArenaValue.As<Napi::Boolean>().Value()) {
      sessionOptions.EnableCpuMemArena();
    } else {
      sessionOptions.DisableCpuMemArena();
    }
  }

  // memory pattern
  if (options.Has("enableMemPattern")) {
    auto enableMemPatternValue = options.Get("enableMemPattern");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableMemPatternValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableMemPattern must be a boolean value.");
    if (enableMemPatternValue.As<Napi::Boolean>().Value()) {
      sessionOptions.EnableMemPattern();
    } else {
      sessionOptions.DisableMemPattern();
    }
  }

  // optimizedModelFilePath
  if (options.Has("optimizedModelFilePath")) {
    auto optimizedModelFilePathValue = options.Get("optimizedModelFilePath");
    ORT_NAPI_THROW_TYPEERROR_IF(!optimizedModelFilePathValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.optimizedModelFilePath must be a string.");
#ifdef _WIN32
    auto str = optimizedModelFilePathValue.As<Napi::String>().Utf16Value();
    std::filesystem::path optimizedModelFilePath{std::wstring{str.begin(), str.end()}};
#else
    std::filesystem::path optimizedModelFilePath{optimizedModelFilePathValue.As<Napi::String>().Utf8Value()};
#endif
    sessionOptions.SetOptimizedModelFilePath(optimizedModelFilePath.c_str());
  }

  // extra
  if (options.Has("extra")) {
    auto extraValue = options.Get("extra");
    ORT_NAPI_THROW_TYPEERROR_IF(!extraValue.IsObject(), options.Env(),
                                "Invalid argument: sessionOptions.extra must be an object.");
    IterateExtraOptions("", extraValue.As<Napi::Object>(), sessionOptions);
  }

  // execution mode
  if (options.Has("executionMode")) {
    auto executionModeValue = options.Get("executionMode");
    ORT_NAPI_THROW_TYPEERROR_IF(!executionModeValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.executionMode must be a string.");
    auto executionModeString = executionModeValue.As<Napi::String>().Utf8Value();
    auto v = EXECUTION_MODE_NAME_TO_ID_MAP.find(executionModeString);
    ORT_NAPI_THROW_TYPEERROR_IF(v == EXECUTION_MODE_NAME_TO_ID_MAP.end(), options.Env(),
                                "'executionMode' is not supported: ", executionModeString);
    sessionOptions.SetExecutionMode(v->second);
  }

  // log ID
  if (options.Has("logId")) {
    auto logIdValue = options.Get("logId");
    ORT_NAPI_THROW_TYPEERROR_IF(!logIdValue.IsString(), options.Env(),
                                "Invalid argument: sessionOptions.logId must be a string.");
    auto logIdString = logIdValue.As<Napi::String>().Utf8Value();
    sessionOptions.SetLogId(logIdString.c_str());
  }

  // Log severity level
  if (options.Has("logSeverityLevel")) {
    auto logLevelValue = options.Get("logSeverityLevel");
    ORT_NAPI_THROW_TYPEERROR_IF(!logLevelValue.IsNumber(), options.Env(),
                                "Invalid argument: sessionOptions.logSeverityLevel must be a number.");
    double logLevelNumber = logLevelValue.As<Napi::Number>().DoubleValue();
    ORT_NAPI_THROW_RANGEERROR_IF(
        std::floor(logLevelNumber) != logLevelNumber || logLevelNumber < 0 || logLevelNumber > 4, options.Env(),
        "Invalid argument: sessionOptions.logSeverityLevel must be one of the following: 0, 1, 2, 3, 4.");

    sessionOptions.SetLogSeverityLevel(static_cast<int>(logLevelNumber));
  }

  // Profiling
  if (options.Has("enableProfiling")) {
    auto enableProfilingValue = options.Get("enableProfiling");
    ORT_NAPI_THROW_TYPEERROR_IF(!enableProfilingValue.IsBoolean(), options.Env(),
                                "Invalid argument: sessionOptions.enableProfiling must be a boolean value.");

    if (enableProfilingValue.As<Napi::Boolean>().Value()) {
      ORT_NAPI_THROW_TYPEERROR_IF(!options.Has("profileFilePrefix"), options.Env(),
                                  "Invalid argument: sessionOptions.profileFilePrefix is required"
                                  " when sessionOptions.enableProfiling is set to true.");
      auto profileFilePrefixValue = options.Get("profileFilePrefix");
      ORT_NAPI_THROW_TYPEERROR_IF(!profileFilePrefixValue.IsString(), options.Env(),
                                  "Invalid argument: sessionOptions.profileFilePrefix must be a string."
                                  " when sessionOptions.enableProfiling is set to true.");
#ifdef _WIN32
      auto str = profileFilePrefixValue.As<Napi::String>().Utf16Value();
      std::basic_string<ORTCHAR_T> profileFilePrefix = std::wstring{str.begin(), str.end()};
#else
      std::basic_string<ORTCHAR_T> profileFilePrefix = profileFilePrefixValue.As<Napi::String>().Utf8Value();
#endif
      sessionOptions.EnableProfiling(profileFilePrefix.c_str());
    } else {
      sessionOptions.DisableProfiling();
    }
  }

  // external data
  if (options.Has("externalData")) {
    auto externalDataValue = options.Get("externalData");
    if (!externalDataValue.IsNull() && !externalDataValue.IsUndefined()) {
      ORT_NAPI_THROW_TYPEERROR_IF(!externalDataValue.IsArray(), options.Env(),
                                  "Invalid argument: sessionOptions.externalData must be an array.");
      auto externalData = externalDataValue.As<Napi::Array>();
      std::vector<std::basic_string<ORTCHAR_T>> paths;
      std::vector<char*> buffs;
      std::vector<size_t> sizes;

      for (const auto& kvp : externalData) {
        Napi::Value value = kvp.second;
        ORT_NAPI_THROW_TYPEERROR_IF(!value.IsObject(), options.Env(),
                                    "Invalid argument: sessionOptions.externalData value must be an object in Node.js binding.");
        Napi::Object obj = value.As<Napi::Object>();
        ORT_NAPI_THROW_TYPEERROR_IF(!obj.Has("path") || !obj.Get("path").IsString(), options.Env(),
                                    "Invalid argument: sessionOptions.externalData value must have a 'path' property of type string in Node.js binding.");
#ifdef _WIN32
        auto path = obj.Get("path").As<Napi::String>().Utf16Value();
        paths.push_back(std::wstring{path.begin(), path.end()});
#else
        auto path = obj.Get("path").As<Napi::String>().Utf8Value();
        paths.push_back(path);
#endif
        ORT_NAPI_THROW_TYPEERROR_IF(!obj.Has("data") ||
                                        (!obj.Get("data").IsBuffer() &&
                                         !(obj.Get("data").IsTypedArray() &&
                                           obj.Get("data").As<Napi::TypedArray>().TypedArrayType() == napi_uint8_array)),
                                    options.Env(),
                                    "Invalid argument: sessionOptions.externalData value must have an 'data' property of type buffer or typed array in Node.js binding.");

        auto data = obj.Get("data");
        char* source;
        size_t size;
        if (data.IsBuffer()) {
          source = data.As<Napi::Buffer<char>>().Data();
          size = data.As<Napi::Buffer<char>>().Length();
        } else {
          auto typedArray = data.As<Napi::TypedArray>();
          auto* arrayBufferData = reinterpret_cast<char*>(typedArray.ArrayBuffer().Data());
          source = typedArray.ByteOffset() == 0 ? arrayBufferData : arrayBufferData + typedArray.ByteOffset();
          size = typedArray.ByteLength();
        }
        if (externalDataBuffers != nullptr) {
          externalDataBuffers->emplace_back();
          if (size != 0) {
            externalDataBuffers->back().assign(source, source + size);
          } else {
            externalDataBuffers->back().push_back(0);
          }
          buffs.push_back(externalDataBuffers->back().data());
        } else {
          buffs.push_back(source);
        }
        sizes.push_back(size);
      }
      sessionOptions.AddExternalInitializersFromFilesInMemory(paths, buffs, sizes);
    }
  }
}

void ParsePreferredOutputLocations(const Napi::Object options, const std::vector<std::string>& outputNames, std::vector<int>& preferredOutputLocations) {
  if (options.Has("preferredOutputLocation")) {
    auto polValue = options.Get("preferredOutputLocation");
    if (polValue.IsNull() || polValue.IsUndefined()) {
      return;
    }
    if (polValue.IsString()) {
      DataLocation location = ParseDataLocation(polValue.As<Napi::String>().Utf8Value());
      ORT_NAPI_THROW_TYPEERROR_IF(location == DATA_LOCATION_NONE, options.Env(),
                                  "Invalid argument: preferredOutputLocation must be an array or a valid string.");

      if (location == DATA_LOCATION_GPU_BUFFER || location == DATA_LOCATION_ML_TENSOR) {
        preferredOutputLocations.resize(outputNames.size(), location);
      }
    } else if (polValue.IsObject()) {
      preferredOutputLocations.resize(outputNames.size(), DATA_LOCATION_CPU);

      auto pol = polValue.As<Napi::Object>();
      for (const auto& nameIter : pol.GetPropertyNames()) {
        Napi::Value nameVar = nameIter.second;
        std::string name = nameVar.As<Napi::String>().Utf8Value();
        // find the name in outputNames
        auto it = std::find(outputNames.begin(), outputNames.end(), name);
        ORT_NAPI_THROW_TYPEERROR_IF(it == outputNames.end(), options.Env(),
                                    "Invalid argument: \"", name, "\" is not a valid output name.");

        Napi::Value value = pol.Get(nameVar);
        DataLocation location = DATA_LOCATION_NONE;
        ORT_NAPI_THROW_TYPEERROR_IF(!value.IsString() || (location = ParseDataLocation(value.As<Napi::String>().Utf8Value())) == DATA_LOCATION_NONE,
                                    options.Env(),
                                    "Invalid argument: preferredOutputLocation[\"", name, "\"] must be a valid string.");

        size_t index = it - outputNames.begin();
        preferredOutputLocations[index] = location;
      }

      if (std::all_of(preferredOutputLocations.begin(), preferredOutputLocations.end(), [](int loc) { return loc == DATA_LOCATION_CPU; })) {
        preferredOutputLocations.clear();
      }
    } else {
      ORT_NAPI_THROW_TYPEERROR(options.Env(), "Invalid argument: preferredOutputLocation must be an array or a valid string.");
    }
  }
}

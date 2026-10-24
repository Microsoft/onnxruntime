// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "command_args_parser.h"
#include "utils.h"

#include <string.h>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unordered_map>

#include <core/graph/constants.h>
#include <core/platform/path_lib.h>
#include <core/optimizer/graph_transformer_level.h>
#include <core/session/onnxruntime_session_options_config_keys.h>
#include <core/session/onnxruntime_run_options_config_keys.h>

#include "test_configuration.h"
#include "strings_helper.h"

#ifdef _MSC_VER
#pragma warning(push)
// C4127: conditional expression is constant
#pragma warning(disable : 4127)
// C4324: structure was padded due to alignment specifier
// Usage of alignas causes some internal padding in places.
#pragma warning(disable : 4324)
// C4702: unreachable code
#pragma warning(disable : 4702)
#endif  // _MSC_VER

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/flags/usage.h"
#include "absl/flags/usage_config.h"
#include "absl/flags/reflection.h"
#include "absl/strings/str_split.h"

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

static const onnxruntime::perftest::PerformanceTestConfig& DefaultPerformanceTestConfig() {
  static onnxruntime::perftest::PerformanceTestConfig default_config{};
  return default_config;
}

ABSL_FLAG(std::string, f, "",
          "Specifies a free dimension by name to override to a specific value for performance optimization.\n"
          "[Usage]: -f \"dimension_name1:override_value1\" -f \"dimension_name2:override_value2\" ...  or"
          " -f \"dimension_name1:override_value1 dimension_name2:override_value2 ... \". Override value must > 0.");
ABSL_FLAG(std::string, F, "",
          "Specifies a free dimension by denotation to override to a specific value for performance optimization.\n"
          "[Usage]: -f \"dimension_denotation1:override_value1\" -f \"dimension_denotation2:override_value2\" ...  or"
          " -f \"dimension_denotation1:override_value1 dimension_denotation2 : override_value2... \". Override value must > 0.");
ABSL_FLAG(std::string, m, "duration", "Specifies the test mode. Value could be 'duration' or 'times'.");
ABSL_FLAG(std::string, e, "cpu",
          "Specifies the provider 'cpu','cuda','dnnl','tensorrt', 'openvino', 'dml', 'acl', 'nnapi', "
          "'coreml', 'qnn', 'snpe', 'migraphx', 'xnnpack', 'vitisai' or 'webgpu'.");
ABSL_FLAG(size_t, r, DefaultPerformanceTestConfig().run_config.repeated_times, "Specifies the repeated times if running in 'times' test mode.");
ABSL_FLAG(size_t, t, DefaultPerformanceTestConfig().run_config.duration_in_seconds, "Specifies the seconds to run for 'duration' mode.");
ABSL_FLAG(std::string, p, "", "Specifies the profile name to enable profiling and dump the profile data to the file.");
ABSL_FLAG(int, x, DefaultPerformanceTestConfig().run_config.intra_op_num_threads, "Sets the number of threads used to parallelize the execution within nodes, A value of 0 means ORT will pick a default. Must >=0.");
ABSL_FLAG(int, y, DefaultPerformanceTestConfig().run_config.inter_op_num_threads, "Sets the number of threads used to parallelize the execution of the graph (across nodes), A value of 0 means ORT will pick a default. Must >=0.");
ABSL_FLAG(size_t, c, DefaultPerformanceTestConfig().run_config.concurrent_session_runs, "Specifies the (max) number of runs to invoke simultaneously.");
ABSL_FLAG(int, d, DefaultPerformanceTestConfig().run_config.cudnn_conv_algo, "Specifies CUDNN convolution algorithms: 0(benchmark), 1(heuristic), 2(default).");
ABSL_FLAG(int, o, DefaultPerformanceTestConfig().run_config.optimization_level, "Specifies graph optimization level. Default is 99 (all). Valid values are 0 (disable), 1 (basic), 2 (extended), 3 (layout), 99 (all).");
ABSL_FLAG(std::string, u, "", "Specifies the optimized model path for saving.");
ABSL_FLAG(std::string, opt_data, "", "Specifies the data file path (relative to the opt model) for saving weights when -u is in effect");
ABSL_FLAG(int64_t, opt_weight_min_size, -1, "Min initializer size to save to --opt_data  when --opt_data is in effect");
ABSL_FLAG(bool, opt_save_prepacks, false, "Saves pre-packs to the file specified by --opt_data along with weights");
ABSL_FLAG(std::string, i, "",
          "Specifies EP specific runtime options as key-value pairs.\n Different runtime options available are: \n"
          "  [Usage]: -e <provider_name> -i '<key1>|<value1> <key2>|<value2>'\n"
          "\n"
          "  [ACL only] [enable_fast_math]: Options: 'true', 'false', default: 'false', \n"
          "\n"
          "  [DML only] [performance_preference]: DML device performance preference, options: 'default', 'minimum_power', 'high_performance', \n"
          "  [DML only] [device_filter]: DML device filter, options: 'any', 'gpu', 'npu', \n"
          "  [DML only] [disable_metacommands]: Options: 'true', 'false', \n"
          "  [DML only] [enable_graph_capture]: Options: 'true', 'false', \n"
          "  [DML only] [enable_graph_serialization]: Options: 'true', 'false', \n"
          "\n"
          "  [OpenVINO only] [device_type]: Overrides the accelerator hardware type and precision with these values at runtime.\n"
          "  [OpenVINO only] [device_id]: Selects a particular hardware device for inference.\n"
          "  [OpenVINO only] [num_of_threads]: Overrides the accelerator hardware type and precision with these values at runtime.\n"
          "  [OpenVINO only] [cache_dir]: Explicitly specify the path to dump and load the blobs(Model caching) or cl_cache (Kernel Caching) files feature. If blob files are already present, it will be directly loaded.\n"
          "  [OpenVINO only] [enable_opencl_throttling]: Enables OpenCL queue throttling for GPU device(Reduces the CPU Utilization while using GPU) \n"
          "  [OpenVINO only] [reshape_input]: Sets model input shapes with support for bounded dynamic dimensions using 'min..max' syntax (e.g., [1..10,3,224,224]) \n"
          "  [OpenVINO only] [layout]: Specifies the layout for inputs/outputs to interpret tensor dimensions correctly. \n"
          "  [Example] [For OpenVINO EP] -e openvino -i \"device_type|CPU num_of_threads|5 enable_opencl_throttling|true reshape_input|<input_name>[1,3,60,60..100] layout|<input_name>[NCHW] cache_dir|\"<path>\"\"\n"
          "\n"
          "  [QNN only] [backend_type]: QNN backend type. E.g., 'cpu', 'htp'. Mutually exclusive with 'backend_path'.\n"
          "  [QNN only] [backend_path]: QNN backend path. E.g., '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. Mutually exclusive with 'backend_type'.\n"
          "  [QNN only] [profiling_level]: QNN profiling level, options: 'basic', 'detailed', default 'off'.\n"
          "  [QNN only] [profiling_file_path] : QNN profiling file path if ETW not enabled.\n"
          "  [QNN only] [rpc_control_latency]: QNN rpc control latency. default to 10.\n"
          "  [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).\n"
          "  [QNN only] [htp_performance_mode]: QNN performance mode, options: 'burst', 'balanced', 'default', 'high_performance', \n"
          "  'high_power_saver', 'low_balanced', 'extreme_power_saver', 'low_power_saver', 'power_saver', 'sustained_high_performance'. Default to 'default'. \n"
          "  [QNN only] [op_packages]: QNN UDO package, allowed format: \n"
          "  op_packages|<op_type>:<op_package_path>:<interface_symbol_name>[:<target>],<op_type2>:<op_package_path2>:<interface_symbol_name2>[:<target2>]. \n"
          "  [QNN only] [qnn_context_priority]: QNN context priority, options: 'low', 'normal', 'normal_high', 'high'. Default to 'normal'. \n"
          "  [QNN only] [qnn_saver_path]: QNN Saver backend path. e.g '/folderpath/libQnnSaver.so'.\n"
          "  [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: \n"
          "  '0', '1', '2', '3', default is '0'.\n"
          "  [QNN only] [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown). \n"
          "  [QNN only] [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. \n"
          "  Options are '0', '68', '69', '73', '75'. Defaults to '0' (none). \n"
          "  [QNN only] [device_id]: The ID of the device to use when setting 'htp_arch'. Defaults to '0' (for single device). \n"
          "  [QNN only] [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision. \n"
          "  Otherwise, it will be fp32 precision. Works for float32 model for HTP backend. Defaults to '1' (with FP16 precision.). \n"
          "  [QNN only] [offload_graph_io_quantization]: Offload graph input quantization and graph output dequantization to another EP (typically CPU EP). \n"
          "  Defaults to '0' (QNN EP handles the graph I/O quantization and dequantization). \n"
          "  [QNN only] [enable_htp_spill_fill_buffer]: Enable HTP spill fill buffer, used while generating QNN context binary.\n"
          "  [QNN only] [enable_htp_shared_memory_allocator]: Enable the QNN HTP shared memory allocator and use it for inputs and outputs. Requires libcdsprpc.so/dll to be available.\n"
          "  Defaults to '0' (disabled).\n"
          "  [QNN only] [extended_udma]: Enable HTP extended UDMA mode for better performance on supported hardware, options: \n"
          "  '0' (disabled), '1' (enabled). Default: '0'. \n"
          "  [Example] [For QNN EP] -e qnn -i \"backend_type|cpu\" \n"
          "\n"
          "  [TensorRT only] [trt_max_partition_iterations]: Maximum iterations for TensorRT parser to get capability.\n"
          "  [TensorRT only] [trt_min_subgraph_size]: Minimum size of TensorRT subgraphs.\n"
          "  [TensorRT only] [trt_max_workspace_size]: Set TensorRT maximum workspace size in byte.\n"
          "  [TensorRT only] [trt_fp16_enable]: Enable TensorRT FP16 precision.\n"
          "  [TensorRT only] [trt_int8_enable]: Enable TensorRT INT8 precision.\n"
          "  [TensorRT only] [trt_int8_calibration_table_name]: Specify INT8 calibration table name.\n"
          "  [TensorRT only] [trt_int8_use_native_calibration_table]: Use Native TensorRT calibration table.\n"
          "  [TensorRT only] [trt_dla_enable]: Enable DLA in Jetson device.\n"
          "  [TensorRT only] [trt_dla_core]: DLA core number.\n"
          "  [TensorRT only] [trt_dump_subgraphs]: Dump TRT subgraph to onnx model.\n"
          "  [TensorRT only] [trt_engine_cache_enable]: Enable engine caching.\n"
          "  [TensorRT only] [trt_engine_cache_path]: Specify engine cache path.\n"
          "  [TensorRT only] [trt_engine_cache_prefix]: Customize engine cache prefix when trt_engine_cache_enable is true.\n"
          "  [TensorRT only] [trt_engine_hw_compatible]: Enable hardware compatibility. Engines ending with '_sm80+' can be re-used across all Ampere+ GPU (a hardware-compatible engine may have lower throughput and/or higher latency than its non-hardware-compatible counterpart).\n"
          "  [TensorRT only] [trt_weight_stripped_engine_enable]: Enable weight-stripped engine build.\n"
          "  [TensorRT only] [trt_onnx_model_folder_path]: Folder path for the ONNX model with weights.\n"
          "  [TensorRT only] [trt_force_sequential_engine_build]: Force TensorRT engines to be built sequentially.\n"
          "  [TensorRT only] [trt_context_memory_sharing_enable]: Enable TensorRT context memory sharing between subgraphs.\n"
          "  [TensorRT only] [trt_layer_norm_fp32_fallback]: Force Pow + Reduce ops in layer norm to run in FP32 to avoid overflow.\n"
          "  [Example] [For TensorRT EP] -e tensorrt -i 'trt_fp16_enable|true trt_int8_enable|true trt_int8_calibration_table_name|calibration.flatbuffers trt_int8_use_native_calibration_table|false trt_force_sequential_engine_build|false'\n"
          "\n"
          "  [NNAPI only] [NNAPI_FLAG_USE_FP16]: Use fp16 relaxation in NNAPI EP..\n"
          "  [NNAPI only] [NNAPI_FLAG_USE_NCHW]: Use the NCHW layout in NNAPI EP.\n"
          "  [NNAPI only] [NNAPI_FLAG_CPU_DISABLED]: Prevent NNAPI from using CPU devices.\n"
          "  [NNAPI only] [NNAPI_FLAG_CPU_ONLY]: Using CPU only in NNAPI EP.\n"
          "  [Example] [For NNAPI EP] -e nnapi -i \"NNAPI_FLAG_USE_FP16 NNAPI_FLAG_USE_NCHW NNAPI_FLAG_CPU_DISABLED\"\n"
          "\n"
          "  [CoreML only] [ModelFormat]:[MLProgram, NeuralNetwork] Create an ML Program model or Neural Network. Default is NeuralNetwork.\n"
          "  [CoreML only] [MLComputeUnits]:[CPUAndNeuralEngine CPUAndGPU ALL CPUOnly] Specify to limit the backend device used to run the model.\n"
          "  [CoreML only] [AllowStaticInputShapes]:[0 1].\n"
          "  [CoreML only] [EnableOnSubgraphs]:[0 1].\n"
          "  [CoreML only] [SpecializationStrategy]:[Default FastPrediction].\n"
          "  [CoreML only] [ProfileComputePlan]:[0 1].\n"
          "  [CoreML only] [AllowLowPrecisionAccumulationOnGPU]:[0 1].\n"
          "  [CoreML only] [ModelCacheDirectory]:[path../a/b/c].\n"
          "  [Example] [For CoreML EP] -e coreml -i \"ModelFormat|MLProgram MLComputeUnits|CPUAndGPU\"\n"
          "\n"
          "  [SNPE only] [runtime]: SNPE runtime, options: 'CPU', 'GPU', 'GPU_FLOAT16', 'DSP', 'AIP_FIXED_TF'. \n"
          "  [SNPE only] [priority]: execution priority, options: 'low', 'normal'. \n"
          "  [SNPE only] [buffer_type]: options: 'TF8', 'TF16', 'UINT8', 'FLOAT', 'ITENSOR'. default: ITENSOR'. \n"
          "  [SNPE only] [enable_init_cache]: enable SNPE init caching feature, set to 1 to enabled it. Disabled by default. \n"
          "  [Example] [For SNPE EP] -e snpe -i \"runtime|CPU priority|low\" \n");
ABSL_FLAG(int, S, DefaultPerformanceTestConfig().run_config.random_seed_for_input_data, "Given random seed, to produce the same input data. This defaults to -1(no initialize).");
ABSL_FLAG(std::string, T, "", "Specifies intra op thread affinity string.");
ABSL_FLAG(std::string, C, "",
          "Specifies session configuration entries as key-value pairs:\n -C \"<key1>|<value1> <key2>|<value2>\" \n"
          "Refer to onnxruntime_session_options_config_keys.h for valid keys and values. \n"
          "[Example] -C \"session.disable_cpu_ep_fallback|1 ep.context_enable|1\" \n");
ABSL_FLAG(std::string, R, "", "Allows user to register custom op by .so or .dll file.");
ABSL_FLAG(bool, A, !DefaultPerformanceTestConfig().run_config.enable_cpu_mem_arena, "Disables memory arena.");
ABSL_FLAG(std::string, shrink_arena_between_runs, "", "When arena is enabled call Shrink for specified devices 'cpu:0;gpu:0'");
ABSL_FLAG(std::string, enable_cuda_mempool, "", "When cuda is enabled use CudaMempoolArena with params 'pool_release_threshold;bytes_to_keep_on_shrink'");
ABSL_FLAG(bool, M, !DefaultPerformanceTestConfig().run_config.enable_memory_pattern, "Disables memory pattern.");
ABSL_FLAG(bool, s, DefaultPerformanceTestConfig().run_config.f_dump_statistics, "Shows statistics result, like P75, P90. If no result_file provided this defaults to on.");
ABSL_FLAG(bool, v, DefaultPerformanceTestConfig().run_config.f_verbose, "Shows verbose information.");
ABSL_FLAG(bool, I, DefaultPerformanceTestConfig().run_config.generate_model_input_binding, "Generates tensor input binding. Free dimensions are treated as 1 unless overridden using -f.");
ABSL_FLAG(bool, P, false, "Uses parallel executor instead of sequential executor.");
ABSL_FLAG(bool, q, DefaultPerformanceTestConfig().run_config.do_cuda_copy_in_separate_stream, "[CUDA only] Uses separate stream for copy.");
ABSL_FLAG(bool, z, DefaultPerformanceTestConfig().run_config.set_denormal_as_zero, "Sets denormal as zero. When turning on this option reduces latency dramatically, a model may have denormals.");
ABSL_FLAG(bool, D, DefaultPerformanceTestConfig().run_config.disable_spinning, "Disables spinning entirely for thread owned by onnxruntime intra-op thread pool.");
ABSL_FLAG(bool, Z, DefaultPerformanceTestConfig().run_config.disable_spinning_between_run, "Disallows thread from spinning during runs to reduce cpu usage.");
ABSL_FLAG(int, spin_duration_us, -1, "Sets the spin duration in microseconds for intra-op thread pool. Default (-1) uses iteration-count-based spinning. 0 disables spinning. Positive values enable time-based spinning.");
ABSL_FLAG(int, spin_backoff_max, 1,
          "Sets the exponential-backoff cap for the intra-op thread pool spin loop. 1 (default) keeps the "
          "legacy single-SpinPause behavior. Values >= 2 enable exp-backoff (typical: 4 or 8) to reduce "
          "CPU/power density during the spin window. Values above 64 are clamped to 64.");
ABSL_FLAG(bool, n, DefaultPerformanceTestConfig().run_config.exit_after_session_creation, "Allows user to measure session creation time to measure impact of enabling any initialization optimizations.");
ABSL_FLAG(uint32_t, hold_ms_after_session_creation, DefaultPerformanceTestConfig().run_config.hold_ms_after_session_creation,
          "When used with -n, keeps the process alive for the specified number of milliseconds after session creation.\n"
          "Prints 'SESSION_READY' to stdout before sleeping. Useful for multi-process memory measurements.");
ABSL_FLAG(bool, l, DefaultPerformanceTestConfig().model_info.load_via_path, "Provides file as binary in memory by using fopen before session creation.");
ABSL_FLAG(bool, g, DefaultPerformanceTestConfig().run_config.enable_cuda_io_binding,
          "[Plugin EP | TensorRT | CUDA] Enables tensor input and output bindings on the device before session run. "
          "For plugin EPs, binds to the selected EP device's allocator.");
ABSL_FLAG(bool, X, DefaultPerformanceTestConfig().run_config.use_extensions, "Registers custom ops from onnxruntime-extensions.");
ABSL_FLAG(std::string, plugin_ep_libs, "",
          "Specifies a list of plugin execution provider (EP) registration names and their corresponding shared libraries to register.\n"
          "[Usage]: --plugin_ep_libs \"plugin_ep_name_1|plugin_ep_1.dll plugin_ep_name_2|plugin_ep_2.dll ... \"");
ABSL_FLAG(std::string, plugin_eps, "", "Specifies a semicolon-separated list of plugin execution providers (EPs) to use.");
ABSL_FLAG(std::string, plugin_ep_options, "",
          "Specifies provider options for each EP listed in --plugin_eps. Options (key-value pairs) for each EP are separated by space and EPs are separated by semicolons.\n"
          "[Usage]: --plugin_ep_options \"ep_1_option_1_key|ep_1_option_1_value ...;ep_2_option_1_key|ep_2_option_1_value ...;... \" or \n"
          "--plugin_ep_options \";ep_2_option_1_key|ep_2_option_1_value ...;... \" or \n"
          "--plugin_ep_options \"ep_1_option_1_key|ep_1_option_1_value ...;;ep_3_option_1_key|ep_3_option_1_value ...;... \"");
ABSL_FLAG(bool, list_ep_devices, false, "Prints all available device indices and their properties (including metadata). This option makes the program exit early without performing inference.\n");
ABSL_FLAG(std::string, select_ep_devices, "", "Specifies a semicolon-separated list of device indices to add to the session and run with.");
ABSL_FLAG(std::string, filter_ep_devices, "",
          "Specifies EP or Device metadata entries as key-value pairs to filter ep devices passed to AppendExecutionProvider_V2.\n"
          "[Usage]: --filter_ep_devices \"<key1>|<value1> <key2>|<value2>\" \n"
          "Devices that match any of the key-value pair will be appended to the session. --select_ep_devices will take precedence over this option.\n"
          "[Example] --filter_ep_devices \"ov_device|NPU ov_device|CPU\" \n"
          "Above example will append npu device first if available, followed by cpu device.");
ABSL_FLAG(bool, plugin_ep_force_cpu_allocator, DefaultPerformanceTestConfig().plugin_ep_force_cpu_allocator,
          "Forces the use of the CPU allocator instead of a plugin EP's own allocator when running with plugin execution providers.\n");
ABSL_FLAG(bool, compile_ep_context, DefaultPerformanceTestConfig().run_config.compile_ep_context, "Generate an EP context model");
ABSL_FLAG(std::string, compile_model_path, "model_ctx.onnx", "The compiled model path for saving EP context model. Overwrites if already exists");
ABSL_FLAG(bool, compile_binary_embed, DefaultPerformanceTestConfig().run_config.compile_binary_embed, "Embed binary blob within EP context node");
ABSL_FLAG(bool, compile_only, DefaultPerformanceTestConfig().run_config.compile_only, "Only compile EP context model without running it");
ABSL_FLAG(std::string, data_shape, "",
          "Specifies input shapes for multi-shape profiling within a single session.\n"
          "The model is compiled once and run with each shape group in round-robin order.\n"
          "[Usage]: --data_shape \"input_name:[d0,d1,...][d0,d1,...] ...\"\n"
          "[Example]: --data_shape \"input:[1,3,224,224][1,3,448,448][1,3,112,112]\"\n"
          "With -I: generates random input of the specified shapes.\n"
          "Without -I: selects test data folders whose tensor shapes match.\n"
          "All inputs must have the same number of shape groups.\n"
          "All dimension values must be positive integers.");
ABSL_FLAG(bool, h, false, "Print program usage.");

namespace onnxruntime {
namespace perftest {

std::string CustomUsageMessage() {
  std::ostringstream oss;
  oss << "onnxruntime_perf_test [options...] model_path [result_file]\n\n";
  oss << "Note: Options may be specified with either a single dash(-option) or a double dash(--option). Both forms are accepted and treated identically.\n\n";
  oss << "Options:";

  return oss.str();
}

bool CommandLineParser::ParseArguments(PerformanceTestConfig& test_config, int argc, ORTCHAR_T* argv[]) {
  // Following callback is to make sure all the ABSL flags defined above will be showed up when running with "--help".
  // Note: By default abseil only wants flags in binary's main. It expects the main routine to reside in <program>.cc or <program>-main.cc or
  // <program>_main.cc, where the <program> is the name of the binary (without .exe on Windows). See usage_config.cc in abseil for more details.
  absl::FlagsUsageConfig config;
  config.contains_help_flags = [](absl::string_view filename) {
    return std::filesystem::path(filename).filename() == std::filesystem::path(__FILE__).filename();
  };

  config.normalize_filename = [](absl::string_view f) {
    return std::string(f);
  };
  absl::SetFlagsUsageConfig(config);
  absl::SetProgramUsageMessage(CustomUsageMessage());

  auto utf8_argv_strings = utils::ConvertArgvToUtf8Strings(argc, argv);
  auto utf8_argv = utils::CStringsFromStrings(utf8_argv_strings);
  auto positional = absl::ParseCommandLine(static_cast<int>(utf8_argv.size()), utf8_argv.data());

  // -f
  {
    const auto& dim_override_str = absl::GetFlag(FLAGS_f);
    if (!dim_override_str.empty()) {
      // Abseil doesn't support the same option being provided multiple times - only the last occurrence is applied.
      // To preserve the previous usage of '-f', where users may specify it multiple times to override different dimension names,
      // we need to manually parse argv.
      std::string option = "f";
      if (!ParseDimensionOverrideFromArgv(argc, utf8_argv_strings, option, test_config.run_config.free_dim_name_overrides)) {
        return false;
      }
    }
  }

  // -F
  {
    const auto& dim_override_str = absl::GetFlag(FLAGS_F);
    if (!dim_override_str.empty()) {
      // Same reason as '-f' above to manully parse argv.
      std::string option = "F";
      if (!ParseDimensionOverrideFromArgv(argc, utf8_argv_strings, option, test_config.run_config.free_dim_denotation_overrides)) {
        return false;
      }
    }
  }

  // -m
  {
    const auto& test_mode_str = absl::GetFlag(FLAGS_m);
    if (!test_mode_str.empty()) {
      if (test_mode_str == "duration") {
        test_config.run_config.test_mode = TestMode::kFixDurationMode;
      } else if (test_mode_str == "times") {
        test_config.run_config.test_mode = TestMode::KFixRepeatedTimesMode;
      } else {
        return false;
      }
    }
  }

  // -p
  {
    const auto& profile_file = absl::GetFlag(FLAGS_p);
    if (!profile_file.empty()) test_config.run_config.profile_file = ToPathString(profile_file);
  }

  // -M
  test_config.run_config.enable_memory_pattern = !absl::GetFlag(FLAGS_M);

  // -A
  test_config.run_config.enable_cpu_mem_arena = !absl::GetFlag(FLAGS_A);

  // --shrink_arena_between_runs
  if (test_config.run_config.enable_cpu_mem_arena) {
    auto shrink_spec = absl::GetFlag(FLAGS_shrink_arena_between_runs);
    test_config.run_config.run_config_entries.emplace(
        kOrtRunOptionsConfigEnableMemoryArenaShrinkage,
        std::move(shrink_spec));
  }

  // --enable_cuda_mempool
  {
    auto cuda_mempool_spec = absl::GetFlag(FLAGS_enable_cuda_mempool);
    if (!cuda_mempool_spec.empty()) {
      // Split the string with ';' separator in two parts
      std::vector<std::string> parts = absl::StrSplit(cuda_mempool_spec, ';');
      if (parts.size() == 2U) {
        test_config.run_config.cuda_mempool_arena_config = {
            std::move(parts[0]), std::move(parts[1])};
      } else {
        std::cerr << "Invalid format for --enable_cuda_mempool. "
                  << "Expected format : <pool_release_threshold;bytes_to_keep_on_shrink>" << std::endl;
      }
    }
  }

  // -e
  {
    auto const& ep = absl::GetFlag(FLAGS_e);
    if (!ep.empty()) {
      if (ep == "cpu") {
        test_config.machine_config.provider_type_name = onnxruntime::kCpuExecutionProvider;
      } else if (ep == "cuda") {
        test_config.machine_config.provider_type_name = onnxruntime::kCudaExecutionProvider;
      } else if (ep == "dnnl") {
        test_config.machine_config.provider_type_name = onnxruntime::kDnnlExecutionProvider;
      } else if (ep == "openvino") {
        test_config.machine_config.provider_type_name = onnxruntime::kOpenVINOExecutionProvider;
      } else if (ep == "tensorrt") {
        test_config.machine_config.provider_type_name = onnxruntime::kTensorrtExecutionProvider;
      } else if (ep == "qnn") {
        test_config.machine_config.provider_type_name = onnxruntime::kQnnExecutionProvider;
      } else if (ep == "snpe") {
        test_config.machine_config.provider_type_name = onnxruntime::kSnpeExecutionProvider;
      } else if (ep == "nnapi") {
        test_config.machine_config.provider_type_name = onnxruntime::kNnapiExecutionProvider;
      } else if (ep == "vsinpu") {
        test_config.machine_config.provider_type_name = onnxruntime::kVSINPUExecutionProvider;
      } else if (ep == "coreml") {
        test_config.machine_config.provider_type_name = onnxruntime::kCoreMLExecutionProvider;
      } else if (ep == "dml") {
        test_config.machine_config.provider_type_name = onnxruntime::kDmlExecutionProvider;
      } else if (ep == "acl") {
        test_config.machine_config.provider_type_name = onnxruntime::kAclExecutionProvider;
      } else if (ep == "migraphx") {
        test_config.machine_config.provider_type_name = onnxruntime::kMIGraphXExecutionProvider;
      } else if (ep == "xnnpack") {
        test_config.machine_config.provider_type_name = onnxruntime::kXnnpackExecutionProvider;
      } else if (ep == "vitisai") {
        test_config.machine_config.provider_type_name = onnxruntime::kVitisAIExecutionProvider;
      } else if (ep == "webgpu") {
        test_config.machine_config.provider_type_name = onnxruntime::kWebGpuExecutionProvider;
      } else {
        return false;
      }
    }
  }

  // Helper function to check if the option is explicitly specified.
  // Abseil Flags does not provide this capability by default.
  // It cannot distinguish between cases where:
  //   - The user typed `-r 1000` (explicitly passing the default value), and
  //   - The user omitted `-r` entirely.
  // To determine this accurately, we must inspect argv directly.
  auto is_option_specified = [&](std::string option) {
    for (int i = 1; i < argc; ++i) {
      auto utf8_arg = ToUTF8String(argv[i]);
      const auto short_option = "-" + option;
      const auto long_option = "--" + option;
      if (utf8_arg == short_option || utf8_arg == long_option ||
          utf8_arg.rfind(short_option + "=", 0) == 0 ||
          utf8_arg.rfind(long_option + "=", 0) == 0) {
        return true;
      }
    }
    return false;
  };

  // -r
  if (is_option_specified("r")) {
    if (absl::GetFlag(FLAGS_r) == static_cast<size_t>(0)) return false;
    test_config.run_config.repeated_times = absl::GetFlag(FLAGS_r);
    test_config.run_config.test_mode = TestMode::KFixRepeatedTimesMode;
  }

  // -t
  if (is_option_specified("t")) {
    if (absl::GetFlag(FLAGS_t) <= static_cast<size_t>(0)) return false;
    test_config.run_config.duration_in_seconds = absl::GetFlag(FLAGS_t);
    test_config.run_config.test_mode = TestMode::kFixDurationMode;
  }

  // -s
  test_config.run_config.f_dump_statistics = absl::GetFlag(FLAGS_s);

  // -S
  test_config.run_config.random_seed_for_input_data = absl::GetFlag(FLAGS_S);

  // -v
  test_config.run_config.f_verbose = absl::GetFlag(FLAGS_v);

  // -x
  if (absl::GetFlag(FLAGS_x) < 0) return false;
  test_config.run_config.intra_op_num_threads = absl::GetFlag(FLAGS_x);

  // -y
  if (absl::GetFlag(FLAGS_y) < 0) return false;
  test_config.run_config.inter_op_num_threads = absl::GetFlag(FLAGS_y);

  // -P
  if (absl::GetFlag(FLAGS_P)) test_config.run_config.execution_mode = ExecutionMode::ORT_PARALLEL;

  // -c
  if (absl::GetFlag(FLAGS_c) <= static_cast<size_t>(0)) return false;
  test_config.run_config.concurrent_session_runs = absl::GetFlag(FLAGS_c);

  // -o
  {
    const auto optimization_level = absl::GetFlag(FLAGS_o);
    if (optimization_level != test_config.run_config.optimization_level) {
      switch (optimization_level) {
        case ORT_DISABLE_ALL:
          test_config.run_config.optimization_level = ORT_DISABLE_ALL;
          break;
        case ORT_ENABLE_BASIC:
          test_config.run_config.optimization_level = ORT_ENABLE_BASIC;
          break;
        case ORT_ENABLE_EXTENDED:
          test_config.run_config.optimization_level = ORT_ENABLE_EXTENDED;
          break;
        case ORT_ENABLE_LAYOUT:
          test_config.run_config.optimization_level = ORT_ENABLE_LAYOUT;
          break;
        case ORT_ENABLE_ALL:
          test_config.run_config.optimization_level = ORT_ENABLE_ALL;
          break;
        default: {
          if (optimization_level > ORT_ENABLE_ALL) {  // relax constraint
            test_config.run_config.optimization_level = ORT_ENABLE_ALL;
          } else {
            return false;
          }
        }
      }
    }
  }

  // -u
  {
    const auto& optimized_model_path = absl::GetFlag(FLAGS_u);
    if (!optimized_model_path.empty()) {
      test_config.run_config.optimized_model_path = ToPathString(optimized_model_path);
      // --opt_data
      const auto& opt_data_path = absl::GetFlag(FLAGS_opt_data);
      if (!opt_data_path.empty()) {
        test_config.run_config.optimized_model_data_path = opt_data_path;
        // --opt_weight_min_size
        if (absl::GetFlag(FLAGS_opt_weight_min_size) >= 0) {
          test_config.run_config.optimized_model_weight_min_size =
              std::to_string(absl::GetFlag(FLAGS_opt_weight_min_size));
        }
        // --opt_save_prepacks
        if (absl::GetFlag(FLAGS_opt_save_prepacks)) {
          test_config.run_config.optimized_save_optimized_prepacks = true;
        }
      }
    }
  }

  // -I
  test_config.run_config.generate_model_input_binding = absl::GetFlag(FLAGS_I);

  // --data_shape
  {
    const auto& data_shape_str = absl::GetFlag(FLAGS_data_shape);
    if (!data_shape_str.empty()) {
      if (!ParseDataShapeGroups(data_shape_str, test_config.run_config.data_shape_groups)) {
        return false;
      }
    }
  }

  // -d
  if (absl::GetFlag(FLAGS_d) < 0) return false;
  test_config.run_config.cudnn_conv_algo = absl::GetFlag(FLAGS_d);

  // -q
  test_config.run_config.do_cuda_copy_in_separate_stream = absl::GetFlag(FLAGS_q);

  // -z
  test_config.run_config.set_denormal_as_zero = absl::GetFlag(FLAGS_z);

  // -i
  {
    const auto& ep_options = absl::GetFlag(FLAGS_i);
    if (!ep_options.empty()) test_config.run_config.ep_runtime_config_string = ToPathString(ep_options);
  }

  // -T
  if (!absl::GetFlag(FLAGS_T).empty()) test_config.run_config.intra_op_thread_affinities = absl::GetFlag(FLAGS_T);

  // -C
  {
    const auto& session_configs = absl::GetFlag(FLAGS_C);
    if (!session_configs.empty()) {
      ORT_TRY {
        ParseSessionConfigs(session_configs, test_config.run_config.session_config_entries);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          fprintf(stderr, "Error parsing session configuration entries: %s\n", ex.what());
        });
        return false;
      }
    }
  }

  // -D
  test_config.run_config.disable_spinning = absl::GetFlag(FLAGS_D);

  // -Z
  test_config.run_config.disable_spinning_between_run = absl::GetFlag(FLAGS_Z);

  // --spin_duration_us
  test_config.run_config.spin_duration_us = absl::GetFlag(FLAGS_spin_duration_us);

  // --spin_backoff_max
  test_config.run_config.spin_backoff_max = absl::GetFlag(FLAGS_spin_backoff_max);
  test_config.run_config.spin_backoff_max_set = is_option_specified("spin_backoff_max");

  // -n
  test_config.run_config.exit_after_session_creation = absl::GetFlag(FLAGS_n);

  // --hold_ms_after_session_creation
  test_config.run_config.hold_ms_after_session_creation = absl::GetFlag(FLAGS_hold_ms_after_session_creation);
  if (test_config.run_config.hold_ms_after_session_creation > 0 &&
      !test_config.run_config.exit_after_session_creation) {
    fprintf(stderr, "WARNING: --hold_ms_after_session_creation has no effect without -n.\n");
  }

  // -l
  test_config.model_info.load_via_path = absl::GetFlag(FLAGS_l);

  // -R
  {
    const auto& register_custom_op_path = absl::GetFlag(FLAGS_R);
    if (!register_custom_op_path.empty()) test_config.run_config.register_custom_op_path = ToPathString(register_custom_op_path);
  }

  // -g
  test_config.run_config.enable_cuda_io_binding = absl::GetFlag(FLAGS_g);

  // -X
  test_config.run_config.use_extensions = absl::GetFlag(FLAGS_X);

  // --plugin_ep_libs
  {
    const auto& plugin_ep_names_and_libs = absl::GetFlag(FLAGS_plugin_ep_libs);
    if (!plugin_ep_names_and_libs.empty()) test_config.plugin_ep_names_and_libs = ToPathString(plugin_ep_names_and_libs);
  }

  // --plugin_eps
  {
    const auto& plugin_eps = absl::GetFlag(FLAGS_plugin_eps);
    if (!plugin_eps.empty()) ParseEpList(plugin_eps, test_config.machine_config.plugin_provider_type_list);
  }

  // --plugin_ep_options
  {
    const auto& plugin_ep_options = absl::GetFlag(FLAGS_plugin_ep_options);
    if (!plugin_ep_options.empty()) test_config.run_config.ep_runtime_config_string = ToPathString(plugin_ep_options);
  }

  // --list_ep_devices
  if (absl::GetFlag(FLAGS_list_ep_devices)) {
    test_config.list_available_ep_devices = true;
    return true;
  }

  // --select_ep_devices
  {
    const auto& select_ep_devices = absl::GetFlag(FLAGS_select_ep_devices);
    if (!select_ep_devices.empty()) test_config.selected_ep_device_indices = select_ep_devices;
  }

  // --filter_ep_devices
  {
    const auto& filter_ep_devices = absl::GetFlag(FLAGS_filter_ep_devices);
    if (!filter_ep_devices.empty()) {
      ORT_TRY {
        ParseEpDeviceFilterKeyValuePairs(filter_ep_devices, test_config.filter_ep_device_kv_pairs);
      }
      ORT_CATCH(const std::exception& ex) {
        ORT_HANDLE_EXCEPTION([&]() {
          fprintf(stderr, "Error parsing filter_ep_devices: %s\n", ex.what());
        });
        return false;
      }
    }
  }

  // --plugin_ep_force_cpu_allocator
  test_config.plugin_ep_force_cpu_allocator = absl::GetFlag(FLAGS_plugin_ep_force_cpu_allocator);

  // --compile_ep_context
  test_config.run_config.compile_ep_context = absl::GetFlag(FLAGS_compile_ep_context);

  // --compile_model_path
  const auto& compile_model_path = absl::GetFlag(FLAGS_compile_model_path);
  test_config.run_config.compile_model_path = ToPathString(compile_model_path);

  // --compile_binary_embed
  test_config.run_config.compile_binary_embed = absl::GetFlag(FLAGS_compile_binary_embed);

  // --compile_only
  test_config.run_config.compile_only = absl::GetFlag(FLAGS_compile_only);

  if (positional.size() == 2) {
    test_config.model_info.model_file_path = ToPathString(positional[1]);
    test_config.run_config.f_dump_statistics = true;
  } else if (positional.size() == 3) {
    test_config.model_info.model_file_path = ToPathString(positional[1]);
    test_config.model_info.result_file_path = ToPathString(positional[2]);
  } else {
    return false;
  }

  return true;
}

}  // namespace perftest
}  // namespace onnxruntime

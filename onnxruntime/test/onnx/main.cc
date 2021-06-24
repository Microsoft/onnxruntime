// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>
#include <iostream>
#include <fstream>
#ifdef _WIN32
#include "getopt.h"
#else
#include <getopt.h>
#include <thread>
#endif
#include "TestResultStat.h"
#include "TestCase.h"
#include "testenv.h"
#include "providers.h"
#include <google/protobuf/stubs/common.h>
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/framework/session_options.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "core/session/inference_session.h"
#endif
#include "core/session/onnxruntime_session_options_config_keys.h"

using namespace onnxruntime;

namespace {
void usage() {
  printf(
      "onnx_test_runner [options...] <data_root>\n"
      "Options:\n"
      "\t-j [models]: Specifies the number of models to run simultaneously.\n"
      "\t-A : Disable memory arena\n"
      "\t-M : Disable memory pattern\n"
      "\t-c [runs]: Specifies the number of Session::Run() to invoke simultaneously for each model.\n"
      "\t-r [repeat]: Specifies the number of times to repeat\n"
      "\t-v: verbose\n"
      "\t-n [test_case_name]: Specifies a single test case to run.\n"
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'dnnl', 'tensorrt', "
      "'openvino', 'nuphar', 'rocm', 'migraphx', 'acl', 'armnn', 'nnapi' or 'coreml'. "
      "Default: 'cpu'.\n"
      "\t-p: Pause after launch, can attach debugger and continue\n"
      "\t-x: Use parallel executor, default (without -x): sequential executor.\n"
      "\t-d [device_id]: Specifies the device id for multi-device (e.g. GPU). The value should > 0\n"
      "\t-o [optimization level]: Default is 99. Valid values are 0 (disable), 1 (basic), 2 (extended), 99 (all).\n"
      "\t\tPlease see onnxruntime_c_api.h (enum GraphOptimizationLevel) for the full list of all optimization levels. "
      "\n"
      "\t-h: help\n"
      "\n"
      "onnxruntime version: %s\n",
      OrtGetApiBase()->GetVersionString());
}

#ifdef _WIN32
int GetNumCpuCores() {
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    // try GetSystemInfo
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (sysInfo.dwNumberOfProcessors <= 0) {
      ORT_THROW("Fatal error: 0 count processors from GetSystemInfo");
    }
    // This is the number of logical processors in the current group
    return sysInfo.dwNumberOfProcessors;
  }
  int processorCoreCount = 0;
  int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  for (int i = 0; i != count; ++i) {
    if (buffer[i].Relationship == RelationProcessorCore) {
      ++processorCoreCount;
    }
  }
  if (!processorCoreCount) ORT_THROW("Fatal error: 0 count processors from GetLogicalProcessorInformation");
  return processorCoreCount;
}
#else
int GetNumCpuCores() { return static_cast<int>(std::thread::hardware_concurrency()); }
#endif
}  // namespace

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[], Ort::Env& env) {
#else
int real_main(int argc, char* argv[], Ort::Env& env) {
#endif
  // if this var is not empty, only run the tests with name in this list
  std::vector<std::basic_string<PATH_CHAR_TYPE>> whitelisted_test_cases;
  int concurrent_session_runs = GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  int repeat_count = 1;
  int p_models = GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_dnnl = false;
  bool enable_openvino = false;
  bool enable_nuphar = false;
  bool enable_tensorrt = false;
  bool enable_mem_pattern = true;
  bool enable_nnapi = false;
  bool enable_coreml = false;
  bool enable_dml = false;
  bool enable_acl = false;
  bool enable_armnn = false;
  bool enable_rocm = false;
  bool enable_migraphx = false;
  int device_id = 0;
  GraphOptimizationLevel graph_optimization_level = ORT_ENABLE_ALL;
  bool user_graph_optimization_level_set = false;
  bool set_denormal_as_zero = false;

  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_ERROR;
  bool verbose_logging_required = false;

  bool pause = false;
#if !defined(ORT_MINIMAL_BUILD)
  InferenceSession::ORTRegisterONNXOpsetSchema(0);
#endif
  {
    int ch;
    while ((ch = getopt(argc, argv, ORT_TSTR("Ac:hj:Mn:r:e:xvo:d:pz"))) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'v':
          verbose_logging_required = true;
          break;
        case 'c':
          concurrent_session_runs = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (concurrent_session_runs <= 0) {
            usage();
            return -1;
          }
          break;
        case 'j':
          p_models = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (p_models <= 0) {
            usage();
            return -1;
          }
          break;
        case 'r':
          repeat_count = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (repeat_count <= 0) {
            usage();
            return -1;
          }
          break;
        case 'M':
          enable_mem_pattern = false;
          break;
        case 'n':
          // run only some whitelisted tests
          // TODO: parse name str to an array
          whitelisted_test_cases.emplace_back(optarg);
          break;
        case 'e':
          if (!CompareCString(optarg, ORT_TSTR("cpu"))) {
            // do nothing
          } else if (!CompareCString(optarg, ORT_TSTR("cuda"))) {
            enable_cuda = true;
          } else if (!CompareCString(optarg, ORT_TSTR("dnnl"))) {
            enable_dnnl = true;
          } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
            enable_openvino = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nuphar"))) {
            enable_nuphar = true;
          } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
            enable_tensorrt = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nnapi"))) {
            enable_nnapi = true;
          } else if (!CompareCString(optarg, ORT_TSTR("coreml"))) {
            enable_coreml = true;
          } else if (!CompareCString(optarg, ORT_TSTR("dml"))) {
            enable_dml = true;
          } else if (!CompareCString(optarg, ORT_TSTR("acl"))) {
            enable_acl = true;
          } else if (!CompareCString(optarg, ORT_TSTR("armnn"))) {
            enable_armnn = true;
          } else if (!CompareCString(optarg, ORT_TSTR("rocm"))) {
            enable_rocm = true;
          } else if (!CompareCString(optarg, ORT_TSTR("migraphx"))) {
            enable_migraphx = true;
          } else {
            usage();
            return -1;
          }
          break;
        case 'x':
          execution_mode = ExecutionMode::ORT_PARALLEL;
          break;
        case 'p':
          pause = true;
          break;
        case 'o': {
          int tmp = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          switch (tmp) {
            case ORT_DISABLE_ALL:
              graph_optimization_level = ORT_DISABLE_ALL;
              break;
            case ORT_ENABLE_BASIC:
              graph_optimization_level = ORT_ENABLE_BASIC;
              break;
            case ORT_ENABLE_EXTENDED:
              graph_optimization_level = ORT_ENABLE_EXTENDED;
              break;
            case ORT_ENABLE_ALL:
              graph_optimization_level = ORT_ENABLE_ALL;
              break;
            default: {
              if (tmp > ORT_ENABLE_ALL) {  // relax constraint
                graph_optimization_level = ORT_ENABLE_ALL;
              } else {
                fprintf(stderr, "See usage for valid values of graph optimization level\n");
                usage();
                return -1;
              }
            }
          }
          user_graph_optimization_level_set = true;
          break;
        }
        case 'd':
          device_id = static_cast<int>(OrtStrtol<PATH_CHAR_TYPE>(optarg, nullptr));
          if (device_id < 0) {
            usage();
            return -1;
          }
          break;
        case 'z':
          set_denormal_as_zero = true;
          break;
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }

  // TODO: Support specifying all valid levels of logging
  // Currently the logging level is ORT_LOGGING_LEVEL_ERROR by default and
  // if the user adds -v, the logging level is ORT_LOGGING_LEVEL_VERBOSE
  if (verbose_logging_required) {
    logging_level = ORT_LOGGING_LEVEL_VERBOSE;
  }

  if (concurrent_session_runs > 1 && repeat_count > 1) {
    fprintf(stderr, "when you use '-r [repeat]', please set '-c' to 1\n");
    usage();
    return -1;
  }
  argc -= optind;
  argv += optind;
  if (argc < 1) {
    fprintf(stderr, "please specify a test data dir\n");
    usage();
    return -1;
  }

  if (pause) {
    printf("Enter to continue...\n");
    fflush(stdout);
    (void)getchar();
  }

  {
    bool failed = false;
    ORT_TRY {
      env = Ort::Env{logging_level, "Default"};
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        fprintf(stderr, "Error creating environment: %s \n", ex.what());
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  std::vector<std::basic_string<PATH_CHAR_TYPE>> data_dirs;
  TestResultStat stat;

  for (int i = 0; i != argc; ++i) {
    data_dirs.emplace_back(argv[i]);
  }

  std::vector<std::unique_ptr<ITestCase>> owned_tests;
  {
    double per_sample_tolerance = 1e-3;
    // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
    // when openvino is enabled, set it to a larger value for resolving MNIST accuracy mismatch
    double relative_per_sample_tolerance = enable_cuda ? 0.017 : enable_openvino ? 0.009 : 1e-3;

    Ort::SessionOptions sf;

    if (enable_cpu_mem_arena)
      sf.EnableCpuMemArena();
    else
      sf.DisableCpuMemArena();
    if (enable_mem_pattern)
      sf.EnableMemPattern();
    else
      sf.DisableMemPattern();
    sf.SetExecutionMode(execution_mode);
    if (set_denormal_as_zero)
      sf.AddConfigEntry(kOrtSessionOptionsConfigSetDenormalAsZero, "1");

    if (enable_tensorrt) {
#ifdef USE_TENSORRT
      OrtCUDAProviderOptions cuda_options{
          device_id,
          OrtCudnnConvAlgoSearch::EXHAUSTIVE,
          std::numeric_limits<size_t>::max(),
          0,
          true,
          0,
          nullptr,
          nullptr};  // TODO: Support arena configuration for users of test runner

      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
      sf.AppendExecutionProvider_CUDA(cuda_options);
#else
      fprintf(stderr, "TensorRT is not supported in this build");
      return -1;
#endif
    }
    if (enable_openvino) {
#ifdef USE_OPENVINO
      //Setting default optimization level for OpenVINO can be overriden with -o option
      sf.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
      sf.AppendExecutionProvider_OpenVINO(OrtOpenVINOProviderOptions{});
#else
      fprintf(stderr, "OpenVINO is not supported in this build");
      return -1;
#endif
    }
    if (enable_cuda) {
#ifdef USE_CUDA
      OrtCUDAProviderOptions cuda_options{
          0,
          OrtCudnnConvAlgoSearch::EXHAUSTIVE,
          std::numeric_limits<size_t>::max(),
          0,
          true,
          0,
          nullptr,
          nullptr};  // TODO: Support arena configuration for users of test runner
      sf.AppendExecutionProvider_CUDA(cuda_options);
#else
      fprintf(stderr, "CUDA is not supported in this build");
      return -1;
#endif
    }
    if (enable_nuphar) {
#ifdef USE_NUPHAR
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, /*allow_unaligned_buffers*/ 1, ""));
#else
      fprintf(stderr, "Nuphar is not supported in this build");
      return -1;
#endif
    }
    if (enable_dnnl) {
#ifdef USE_DNNL
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "DNNL is not supported in this build");
      return -1;
#endif
    }
    if (enable_nnapi) {
#ifdef USE_NNAPI
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf, 0));
#else
      fprintf(stderr, "NNAPI is not supported in this build");
      return -1;
#endif
    }
    if (enable_coreml) {
#ifdef USE_COREML
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(sf, 0));
#else
      fprintf(stderr, "CoreML is not supported in this build");
      return -1;
#endif
    }
    if (enable_dml) {
#ifdef USE_DML
      fprintf(stderr, "Disabling mem pattern and forcing single-threaded execution since DML is used");
      sf.DisableMemPattern();
      sf.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      p_models = 1;
      concurrent_session_runs = 1;
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(sf, device_id));
#else
      fprintf(stderr, "DML is not supported in this build");
      return -1;
#endif
    }
    if (enable_acl) {
#ifdef USE_ACL
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "ACL is not supported in this build");
      return -1;
#endif
    }
    if (enable_armnn) {
#ifdef USE_ARMNN
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ArmNN(sf, enable_cpu_mem_arena ? 1 : 0));
#else
      fprintf(stderr, "ArmNN is not supported in this build\n");
      return -1;
#endif
    }
    if (enable_rocm) {
#ifdef USE_ROCM
      OrtROCMProviderOptions rocm_options{
          0,
          0,
          std::numeric_limits<size_t>::max(),
          0};
      sf.AppendExecutionProvider_ROCM(rocm_options);
#else
      fprintf(stderr, "ROCM is not supported in this build");
      return -1;
#endif
    }
    if (enable_migraphx) {
#ifdef USE_MIGRAPHX
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(sf, device_id));
#else
      fprintf(stderr, "MIGRAPHX is not supported in this build");
      return -1;
#endif
    }

    if (user_graph_optimization_level_set) {
      sf.SetGraphOptimizationLevel(graph_optimization_level);
    }

    // Permanently exclude following tests because ORT support only opset staring from 7,
    // Please make no more changes to the list
    static const ORTCHAR_T* immutable_broken_tests[] =
        {
            ORT_TSTR("AvgPool1d"),
            ORT_TSTR("AvgPool1d_stride"),
            ORT_TSTR("AvgPool2d"),
            ORT_TSTR("AvgPool2d_stride"),
            ORT_TSTR("AvgPool3d"),
            ORT_TSTR("AvgPool3d_stride"),
            ORT_TSTR("AvgPool3d_stride1_pad0_gpu_input"),
            ORT_TSTR("BatchNorm1d_3d_input_eval"),
            ORT_TSTR("BatchNorm2d_eval"),
            ORT_TSTR("BatchNorm2d_momentum_eval"),
            ORT_TSTR("BatchNorm3d_eval"),
            ORT_TSTR("BatchNorm3d_momentum_eval"),
            ORT_TSTR("GLU"),
            ORT_TSTR("GLU_dim"),
            ORT_TSTR("Linear"),
            ORT_TSTR("PReLU_1d"),
            ORT_TSTR("PReLU_1d_multiparam"),
            ORT_TSTR("PReLU_2d"),
            ORT_TSTR("PReLU_2d_multiparam"),
            ORT_TSTR("PReLU_3d"),
            ORT_TSTR("PReLU_3d_multiparam"),
            ORT_TSTR("PoissonNLLLLoss_no_reduce"),
            ORT_TSTR("Softsign"),
            ORT_TSTR("operator_add_broadcast"),
            ORT_TSTR("operator_add_size1_broadcast"),
            ORT_TSTR("operator_add_size1_right_broadcast"),
            ORT_TSTR("operator_add_size1_singleton_broadcast"),
            ORT_TSTR("operator_addconstant"),
            ORT_TSTR("operator_addmm"),
            ORT_TSTR("operator_basic"),
            ORT_TSTR("operator_mm"),
            ORT_TSTR("operator_non_float_params"),
            ORT_TSTR("operator_params"),
            ORT_TSTR("operator_pow"),
        };

    static const ORTCHAR_T* cuda_flaky_tests[] = {
        ORT_TSTR("fp16_inception_v1"),
        ORT_TSTR("fp16_shufflenet"), ORT_TSTR("fp16_tiny_yolov2")};
    static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mlperf_ssd_mobilenet_300"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("zfnet512"), ORT_TSTR("keras2coreml_Dense_ImageNet")};
    static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("test_densenet121"), ORT_TSTR("test_resnet18v2"), ORT_TSTR("test_resnet34v2"), ORT_TSTR("test_resnet50v2"), ORT_TSTR("test_resnet101v2"),
                                                     ORT_TSTR("test_resnet101v2"), ORT_TSTR("test_vgg19"), ORT_TSTR("tf_inception_resnet_v2"), ORT_TSTR("tf_inception_v1"), ORT_TSTR("tf_inception_v3"), ORT_TSTR("tf_inception_v4"), ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                     ORT_TSTR("tf_mobilenet_v2_1.0_224"), ORT_TSTR("tf_mobilenet_v2_1.4_224"), ORT_TSTR("tf_nasnet_large"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("tf_resnet_v1_50"), ORT_TSTR("tf_resnet_v1_101"), ORT_TSTR("tf_resnet_v1_101"),
                                                     ORT_TSTR("tf_resnet_v2_101"), ORT_TSTR("tf_resnet_v2_152"), ORT_TSTR("batchnorm_example_training_mode"), ORT_TSTR("batchnorm_epsilon_training_mode")};

    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests), std::end(immutable_broken_tests));
    if (enable_cuda) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    }
    if (enable_dml) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    }
    if (enable_dnnl) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
    }
#if !defined(__amd64__) && !defined(_M_AMD64)
    //out of memory
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mask_rcnn_keras"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("vgg19"), ORT_TSTR("coreml_VGG16_ImageNet")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

    std::vector<ITestCase*> tests;
    LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance,
              all_disabled_tests,
              [&owned_tests, &tests](std::unique_ptr<ITestCase> l) {
                tests.push_back(l.get());
                owned_tests.push_back(std::move(l));
              });

    TestEnv test_env(env, sf, TestEnv::GetDefaultThreadPool(Env::Default()), std::move(tests), stat);
    Status st = test_env.Run(p_models, concurrent_session_runs, repeat_count);
    if (!st.IsOK()) {
      fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
      return -1;
    }
    std::string res = stat.ToString();
    fwrite(res.c_str(), 1, res.size(), stdout);
  }

  struct BrokenTest {
    std::string test_name_;
    std::string reason_;
    std::set<std::string> broken_versions_ = {};  // apply to all versions if empty
    BrokenTest(std::string name, std::string reason) : test_name_(std::move(name)), reason_(std::move(reason)) {}
    BrokenTest(std::string name, std::string reason, const std::initializer_list<std::string>& versions) : test_name_(std::move(name)), reason_(std::move(reason)), broken_versions_(versions) {}
    bool operator<(const struct BrokenTest& test) const {
      return strcmp(test_name_.c_str(), test.test_name_.c_str()) < 0;
    }
  };

  std::set<BrokenTest> broken_tests = {
      {"BERT_Squad", "test data bug"},
      {"constantofshape_float_ones", "test data bug", {"onnx141", "onnx150"}},
      {"constantofshape_int_zeros", "test data bug", {"onnx141", "onnx150"}},
      {"cast_STRING_to_FLOAT", "Linux CI has old ONNX python package with bad test data", {"onnx141"}},
      // Numpy float to string has unexpected rounding for some results given numpy default precision is meant to be 8.
      // "e.g. 0.296140194 -> '0.2961402' not '0.29614019'. ORT produces the latter with precision set to 8,
      // which doesn't match the expected output that was generated with numpy.
      {"cast_FLOAT_to_STRING", "Numpy float to string has unexpected rounding for some results."},
      {"tf_nasnet_large", "disable temporarily"},
      {"tf_nasnet_mobile", "disable temporarily"},
      {"tf_pnasnet_large", "disable temporarily"},
      {"shrink", "test case is wrong", {"onnx141"}},
      {"maxpool_with_argmax_2d_precomputed_strides", "ShapeInferenceError"},
      {"tf_inception_v2", "result mismatch"},
      {"tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied"},
      {"tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied"},
      {"mxnet_arcface", "Model is an invalid ONNX model"},
      {"unique_not_sorted_without_axis", "Expected data for 'Y' is incorrect and in sorted order."},
      {"cumsum_1d_reverse_exclusive", "only failing linux GPU CI. Likely build error."},
      {"resize_downsample_scales_cubic_align_corners", "results mismatch with onnx tests"},
      {"resize_downsample_scales_linear_align_corners", "results mismatch with onnx tests"},
      {"resize_tf_crop_and_resize", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_ceil_half_pixel", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_floor_align_corners", "Bad onnx test output. Needs test fix."},
      {"resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric", "Bad onnx test output. Needs test fix."},
      {"bitshift_right_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"bitshift_left_uint16", "BitShift(11) uint16 support not enabled currently"},
      {"maxunpool_export_with_output_shape", "Invalid output in ONNX test. See https://github.com/onnx/onnx/issues/2398"},
      {"training_dropout", "result differs", {}},                       // Temporary, subsequent PR will remove this.
      {"training_dropout_default", "result differs", {}},               // Temporary, subsequent PR will remove this.
      {"training_dropout_default_mask", "result differs", {}},          // Temporary, subsequent PR will remove this.
      {"training_dropout_mask", "result differs", {}},                  // Temporary, subsequent PR will remove this.
      {"adagrad", "not a registered function/op", {}},                  // Op not registered.
      {"adagrad_multiple", "not a registered function/op", {}},         // Op not registered.
      {"adam", "not a registered function/op", {}},                     // Op not registered.
      {"adam_multiple", "not a registered function/op", {}},            // Op not registered.
      {"gradient_of_add", "not a registered function/op", {}},          // Op not registered.
      {"gradient_of_add_and_mul", "not a registered function/op", {}},  // Op not registered.
      {"momentum", "not a registered function/op", {}},                 // Op not registered.
      {"momentum_multiple", "not a registered function/op", {}},        // Op not registered.
      {"nesterov_momentum", "not a registered function/op", {}},        // Op not registered.
      {"cast_FLOAT_to_BFLOAT16", "onnx generate bfloat tensor as uint16 type", {}},
      {"cast_BFLOAT16_to_FLOAT", "onnx generate bfloat tensor as uint16 type", {}},
      {"sequence_insert_at_back", "onnx currently not supporting loading segment", {}},
      {"sequence_insert_at_front", "onnx currently not supporting loading segment", {}},
      {"loop13_seq", "ORT api does not currently support creating empty sequences (needed for this test)", {}},
  };

#ifdef DISABLE_ML_OPS
  auto starts_with = [](const std::string& find_in, const std::string& find_what) {
    return find_in.compare(0, find_what.size(), find_what) == 0;
  };
  for (const auto& test_ptr : owned_tests) {
    const std::string& test_name = test_ptr->GetTestCaseName();
    if (starts_with(test_name, "XGBoost_") ||
        starts_with(test_name, "coreml_") ||
        starts_with(test_name, "scikit_") ||
        starts_with(test_name, "libsvm_")) {
      broken_tests.insert({test_name, "Traditional ML ops are disabled in this build."});
    }
  }
#endif

  if (enable_openvino) {
    broken_tests.insert({"operator_permute2", "Disabled temporariliy"});
    broken_tests.insert({"operator_repeat", "Disabled temporariliy"});
    broken_tests.insert({"operator_repeat_dim_overflow", "Disabled temporariliy"});
    broken_tests.insert({"mlperf_ssd_resnet34_1200", "Disabled temporariliy"});
    broken_tests.insert({"candy", "Results mismatch: 1 of 150528"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "OpenVino does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "OpenVino does not support 5D+ tensors"});
  }

  if (enable_dnnl) {
    broken_tests.insert({"tf_mobilenet_v2_1.0_224", "result mismatch"});
    broken_tests.insert({"tf_mobilenet_v2_1.4_224", "result mismatch"});
    broken_tests.insert({"tf_mobilenet_v1_1.0_224", "result mismatch"});
    broken_tests.insert({"mobilenetv2-1.0", "result mismatch"});
    broken_tests.insert({"candy", "result mismatch"});
    broken_tests.insert({"range_float_type_positive_delta_expanded", "get unknown exception from DNNL EP"});
    broken_tests.insert({"range_int32_type_negative_delta_expanded", "get unknown exception from DNNL EP"});
    broken_tests.insert({"averagepool_2d_ceil", "maxpool ceiling not supported"});
    broken_tests.insert({"maxpool_2d_ceil", "maxpool ceiling not supported"});
    broken_tests.insert({"maxpool_2d_dilations", "maxpool dilations not supported"});
    broken_tests.insert({"mlperf_ssd_resnet34_1200", "test pass on dev box but fails on CI build"});
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests.insert({"convtranspose_3d", "3d convtranspose not supported yet"});
    broken_tests.insert({"maxpool_2d_uint8", "Does not work on DNNL, NNAPI"});
  }

  if (enable_nnapi) {
    broken_tests.insert({"scan9_sum", "Error with the extra graph"});
    broken_tests.insert({"scan_sum", "Error with the extra graph"});
    broken_tests.insert({"mvn_expanded", "Failed to find kernel for MemcpyFromHost(1) (node Memcpy_1)"});
    broken_tests.insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"gemm_transposeB", "Temporarily disabled pending investigation"});
    broken_tests.insert({"range_float_type_positive_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"range_int32_type_negative_delta_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests.insert({"convtranspose_3d", "3d convtranspose not supported yet"});
    broken_tests.insert({"maxpool_2d_uint8", "result mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NC_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
    // Disable based on George Wu's recommendation.
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NC", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_ignore_index_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_none_weights_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_sum_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"nllloss_NCd1_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ignore_index", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ignore_index_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_with_weight_reduction_mean", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_with_weight_reduction_mean_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2d3d4d5_mean_weight", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2d3d4d5_mean_weight_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_ii_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_mean_weight_negative_ii_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1_weight_ii_expanded", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ii", "wait for investigation"});
    broken_tests.insert({"nllloss_NCd1d2_no_weight_reduction_mean_ii_expanded", "wait for investigation"});
  }

  if (enable_tensorrt) {
    broken_tests.insert({"fp16_shufflenet", "TRT EP bug"});
    broken_tests.insert({"fp16_inception_v1", "TRT EP bug"});
    broken_tests.insert({"fp16_tiny_yolov2", "TRT EP bug"});
    broken_tests.insert({"tf_inception_v3", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_mobilenet_v1_1.0_224", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_mobilenet_v2_1.0_224", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_mobilenet_v2_1.4_224", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_resnet_v1_101", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_resnet_v1_152", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_resnet_v1_50", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_resnet_v2_101", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_resnet_v2_152", "TRT Engine couldn't be created"});
    broken_tests.insert({"tf_resnet_v2_50", "TRT Engine couldn't be created"});
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
    broken_tests.insert({"convtranspose_3d", "3d convtranspose not supported yet"});
  }

  if (enable_cuda) {
    broken_tests.insert({"candy", "result mismatch"});
    broken_tests.insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests.insert({"mlperf_ssd_mobilenet_300", "unknown error"});
    broken_tests.insert({"mlperf_ssd_resnet34_1200", "unknown error"});
    broken_tests.insert({"tf_inception_v1", "flaky test"});  //TODO: Investigate cause for flakiness
    broken_tests.insert({"faster_rcnn", "Linux: faster_rcnn:output=6383:shape mismatch, expect {77} got {57}"});
    broken_tests.insert({"split_zero_size_splits", "alloc failed"});
  }

  if (enable_dml) {
    broken_tests.insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests.insert({"PixelShuffle", "Test requires 6D Reshape, which isn't supported by DirectML"});
    broken_tests.insert({"operator_permute2", "Test requires 6D Transpose, which isn't supported by DirectML"});
    broken_tests.insert({"resize_downsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests.insert({"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests.insert({"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});

    // These tests are temporarily disabled pending investigation
    broken_tests.insert({"dynamicquantizelinear_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_max_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"dynamicquantizelinear_min_adjusted_expanded", "Temporarily disabled pending investigation"});
    broken_tests.insert({"mxnet_arcface", "Temporarily disabled pending investigation"});
    broken_tests.insert({"yolov3", "Temporarily disabled pending investigation"});
    broken_tests.insert({"tf_inception_v2", "Temporarily disabled pending investigation"});
    broken_tests.insert({"fp16_inception_v1", "Temporarily disabled pending investigation"});
    broken_tests.insert({"candy", "Temporarily disabled pending investigation"});
    broken_tests.insert({"BERT_Squad", "Temporarily disabled pending investigation"});
    broken_tests.insert({"LSTM_Seq_lens_unpacked", "The parameter is incorrect"});

    broken_tests.insert({"resize_downsample_scales_linear", "DML uses half_pixel and this test assumed \"asymmetric\" but does not include \"mode\""});
    broken_tests.insert({"resize_downsample_sizes_linear_pytorch_half_pixel", "DML does not support downsampling by such a large factor - skips input pixels"});
    broken_tests.insert({"resize_downsample_sizes_nearest", "DML uses pixel centers for nearest, rounding 1 value off for the middle column"});
    broken_tests.insert({"resize_upsample_sizes_nearest", "DML uses pixel centers for nearest, which makes more sense (the 3rd row mismatches)"});
    broken_tests.insert({"unsqueeze_three_axes", "DML does not support 6D tensors"});
    broken_tests.insert({"unsqueeze_unsorted_axes", "DMLdoes not support 6D tensors"});

    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "DML does not support 5D+ tensors"});
  }

#if defined(_WIN32) && !defined(_WIN64)
  broken_tests.insert({"vgg19", "failed: bad allocation"});
#endif

  // Disable mask_rcnn_keras as this model currently has an invalid contrib op version set to 10
  broken_tests.insert({"mask_rcnn_keras", "This model uses contrib ops."});

#ifdef DISABLE_CONTRIB_OPS
  broken_tests.insert({"coreml_SqueezeNet_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Permute_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_ReLU_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Padding-Upsampling-Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"fp16_tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Pooling_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Padding_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Normalizer_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet_small", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet_large", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_linear_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_leakyrelu_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_hard_sigmoid_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_elu_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Dense_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Conv2D_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_VGG16_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_Resnet50_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_Inceptionv3_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_FNS-Candy_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"coreml_AgeNet_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_ImageNet_large", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_ImageNet_small", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_thresholdedrelu_sklearn_load_breast_cancer", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu_default", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_default_axes", "This model uses contrib ops."});
  broken_tests.insert({"thresholdedrelu_example", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_neg failed", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_start_out_of_bounds", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_end_out_of_bounds", "This model uses contrib ops."});
  broken_tests.insert({"dynamic_slice_neg", "This model uses contrib ops."});
  broken_tests.insert({"mvn", "This model uses contrib ops.", {"onnx130"}});
  broken_tests.insert({"cdist_float32_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float32_sqeuclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_euclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_euclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_euclidean_1_1_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_sqeuclidean_1000_2000_1", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_sqeuclidean_1000_2000_500", "This model uses contrib ops."});
  broken_tests.insert({"cdist_float64_sqeuclidean_1_1_1", "This model uses contrib ops."});
#endif

  int result = 0;
  for (const auto& p : stat.GetFailedTest()) {
    BrokenTest t = {p.first, ""};
    auto iter = broken_tests.find(t);
    if (iter == broken_tests.end() || (p.second != TestModelInfo::unknown_version && !iter->broken_versions_.empty() &&
                                       iter->broken_versions_.find(p.second) == iter->broken_versions_.end())) {
      fprintf(stderr, "test %s failed, please fix it\n", p.first.c_str());
      result = -1;
    }
  }
  return result;
}
#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  Ort::Env env{nullptr};
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv, env);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();
  return retval;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/session/onnxruntime_cxx_api.h>
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
#include "testenv.h"
#include "runner.h"
#include "sync_api.h"
#include "providers.h"
#include <google/protobuf/stubs/common.h>
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/framework/session_options.h"

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
      "\t-e [EXECUTION_PROVIDER]: EXECUTION_PROVIDER could be 'cpu', 'cuda', 'dnnl', 'tensorrt', 'ngraph', "
      "'openvino', 'nuphar' or 'acl'. "
      "Default: 'cpu'.\n"
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
  std::vector<std::basic_string<PATH_CHAR_TYPE> > whitelisted_test_cases;
  int concurrent_session_runs = GetNumCpuCores();
  bool enable_cpu_mem_arena = true;
  ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL;
  int repeat_count = 1;
  int p_models = GetNumCpuCores();
  bool enable_cuda = false;
  bool enable_dnnl = false;
  bool enable_ngraph = false;
  bool enable_nuphar = false;
  bool enable_tensorrt = false;
  bool enable_mem_pattern = true;
  bool enable_openvino = false;
  bool enable_nnapi = false;
  bool enable_dml = false;
  bool enable_acl = false;
  int device_id = 0;
  GraphOptimizationLevel graph_optimization_level = ORT_DISABLE_ALL;
  bool user_graph_optimization_level_set = false;
  int verbosity_option_count = 0;

  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  {
    int ch;
    while ((ch = getopt(argc, argv, ORT_TSTR("Ac:hj:Mn:r:e:xvo:d:"))) != -1) {
      switch (ch) {
        case 'A':
          enable_cpu_mem_arena = false;
          break;
        case 'v':
          verbosity_option_count += 1;
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
          } else if (!CompareCString(optarg, ORT_TSTR("ngraph"))) {
            enable_ngraph = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nuphar"))) {
            enable_nuphar = true;
          } else if (!CompareCString(optarg, ORT_TSTR("tensorrt"))) {
            enable_tensorrt = true;
          } else if (!CompareCString(optarg, ORT_TSTR("openvino"))) {
            enable_openvino = true;
          } else if (!CompareCString(optarg, ORT_TSTR("nnapi"))) {
            enable_nnapi = true;
          } else if (!CompareCString(optarg, ORT_TSTR("dml"))) {
            enable_dml = true;
          } else if (!CompareCString(optarg, ORT_TSTR("acl"))) {
            enable_acl = true;
          } else {
            usage();
            return -1;
          }
          break;
        case 'x':
          execution_mode = ExecutionMode::ORT_PARALLEL;
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
        case '?':
        case 'h':
        default:
          usage();
          return -1;
      }
    }
  }

  // set log level based on number of verbosity options
  if (verbosity_option_count == 1) {
    logging_level = ORT_LOGGING_LEVEL_INFO;
  } else if (verbosity_option_count > 1) {
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

  try {
    env = Ort::Env{logging_level, "Default"};
  } catch (std::exception& ex) {
    fprintf(stderr, "Error creating environment: %s \n", ex.what());
    return -1;
  }

  std::vector<std::basic_string<PATH_CHAR_TYPE> > data_dirs;
  TestResultStat stat;

  for (int i = 0; i != argc; ++i) {
    data_dirs.emplace_back(argv[i]);
  }
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

    if (enable_tensorrt) {
#ifdef USE_TENSORRT
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf, device_id));
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, device_id));
#else
      fprintf(stderr, "TensorRT is not supported in this build");
      return -1;
#endif
    }

    if (enable_openvino) {
#ifdef USE_OPENVINO
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(sf, "CPU"));
#else
      fprintf(stderr, "OpenVINO is not supported in this build");
      return -1;
#endif
    }
    if (enable_cuda) {
#ifdef USE_CUDA
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, device_id));
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
    if (enable_ngraph) {  // TODO: Re-order the priority?
#ifdef USE_NGRAPH
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_NGraph(sf, "CPU"));
#else
      fprintf(stderr, "nGraph is not supported in this build");
      return -1;
#endif
    }
    if (enable_nnapi) {
#ifdef USE_NNAPI
      Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(sf));
#else
      fprintf(stderr, "DNNLibrary/NNAPI is not supported in this build");
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
            ORT_TSTR("operator_pow")};

    static const ORTCHAR_T* cuda_flaky_tests[] = {
        ORT_TSTR("fp16_inception_v1"),
        ORT_TSTR("fp16_shufflenet"), ORT_TSTR("fp16_tiny_yolov2")};
    static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mlperf_ssd_mobilenet_300"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("zfnet512")};
    static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("test_densenet121"), ORT_TSTR("test_resnet18v2"), ORT_TSTR("test_resnet34v2"), ORT_TSTR("test_resnet50v2"), ORT_TSTR("test_resnet101v2"),
                                                     ORT_TSTR("test_resnet101v2"), ORT_TSTR("test_vgg19"), ORT_TSTR("tf_inception_resnet_v2"), ORT_TSTR("tf_inception_v1"), ORT_TSTR("tf_inception_v3"), ORT_TSTR("tf_inception_v4"), ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                     ORT_TSTR("tf_mobilenet_v2_1.0_224"), ORT_TSTR("tf_mobilenet_v2_1.4_224"), ORT_TSTR("tf_nasnet_large"), ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("tf_resnet_v1_50"), ORT_TSTR("tf_resnet_v1_101"), ORT_TSTR("tf_resnet_v1_101"),
                                                     ORT_TSTR("tf_resnet_v2_101"), ORT_TSTR("tf_resnet_v2_152"), ORT_TSTR("batchnorm_example_training_mode"), ORT_TSTR("batchnorm_epsilon_training_mode")};

    std::unordered_set<std::basic_string<ORTCHAR_T> > all_disabled_tests(std::begin(immutable_broken_tests), std::end(immutable_broken_tests));
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
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mask_rcnn_keras"), ORT_TSTR("mask_rcnn"), ORT_TSTR("faster_rcnn"), ORT_TSTR("vgg19")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

    std::vector<ITestCase*> tests;
    LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance, all_disabled_tests,
              [&tests](ITestCase* l) { tests.push_back(l); });

    TestEnv args(tests, stat, env, sf);
    Status st = RunTests(args, p_models, concurrent_session_runs, static_cast<size_t>(repeat_count),
                         GetDefaultThreadPool(Env::Default()));
    if (!st.IsOK()) {
      fprintf(stderr, "%s\n", st.ErrorMessage().c_str());
      return -1;
    }
    for (ITestCase* l : tests) {
      delete l;
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
      {"convtranspose_3d", "3d convtranspose not supported yet"},
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
      {"maxunpool_export_with_output_shape", "Invalid output in ONNX test. See https://github.com/onnx/onnx/issues/2398"}
  };

  if (enable_ngraph) {
    broken_tests.insert({"qlinearconv", "ambiguity in scalar dimensions [] vs [1]"});
    broken_tests.insert({"clip_splitbounds", "not implemented yet for opset 11"});
    broken_tests.insert({"clip_outbounds", "not implemented yet for opset 11"});
    broken_tests.insert({"clip_example", "not implemented yet for opset 11"});
    broken_tests.insert({"clip_default_min", "not implemented yet for opset 11"});
    broken_tests.insert({"clip_default_max", "not implemented yet for opset 11"});
    broken_tests.insert({"clip", "not implemented yet for opset 11"});
    broken_tests.insert({"depthtospace_crd_mode_example", "NGraph does not support CRD mode"});
    broken_tests.insert({"depthtospace_crd_mode", "NGraph does not support CRD mode"});
    broken_tests.insert({"gemm_default_no_bias", "not implemented yet for opset 11"});
    broken_tests.insert({"quantizelinear", "ambiguity in scalar dimensions [] vs [1]", {"onnx150"}});
    broken_tests.insert({"dequantizelinear", "ambiguity in scalar dimensions [] vs [1]", {"onnx150"}});
    broken_tests.insert({"mlperf_ssd_resnet34_1200", "Results mismatch"});
    broken_tests.insert({"BERT_Squad", "Invalid Feed Input Name:input4"});
    broken_tests.insert({"candy", "Results mismatch: 2 of 150528"});
    broken_tests.insert({"tf_mobilenet_v1_1.0_224", "Results mismatch"});    
    broken_tests.insert({"tf_mobilenet_v2_1.0_224", "Results mismatch"});
    broken_tests.insert({"tf_mobilenet_v2_1.4_224", "Results mismatch"});
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
  }
  if (enable_nuphar) {
    broken_tests.insert({"cgan", "TVM exception during initialization"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded", "TVM exception during initialization"});
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
    broken_tests.insert({"maxpool_2d_uint8", "Does not work on DNNL, NNAPI"});
  }

  if (enable_openvino) {
    broken_tests.insert({"fp16_shufflenet", "accuracy mismatch with fp16 precision"});
    broken_tests.insert({"fp16_inception_v1", "accuracy mismatch with fp16 precision"});
    broken_tests.insert({"fp16_tiny_yolov2", "accuaracy mismatch with fp16 precision"});
    broken_tests.insert({"scan_sum", "disable temporarily"});
    broken_tests.insert({"scan9_sum", "disable temporarily"});
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
	broken_tests.insert({"bvlc_alexnet", "disable temporarily"});
	broken_tests.insert({"bvlc_googlenet", "disable temporarily"});
	broken_tests.insert({"bvlc_reference_caffenet", "disable temporarily"});
	broken_tests.insert({"bvlc_reference_rcnn_ilsvrc13", "disable temporarily"});
	broken_tests.insert({"inception_v1", "disable temporarily"});
	broken_tests.insert({"squeezenet", "disable temporarily"});
	broken_tests.insert({"vgg19", "disable temporarily"});
#ifdef OPENVINO_CONFIG_GPU_FP32
    broken_tests.insert({"tiny_yolov2", "accuracy mismatch"});
    broken_tests.insert({"div", "will be fixed in the next release"});
#ifdef OPENVINO_CONFIG_GPU_FP16
    broken_tests.insert({"div", "will be fixed in the next release"});
#endif
#endif
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
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NC_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
	broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded", "shape mismatch"});
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
  }

  if (enable_cuda) {
    broken_tests.insert({"candy", "result mismatch"});
    broken_tests.insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests.insert({"mlperf_ssd_mobilenet_300", "unknown error"});
    broken_tests.insert({"mlperf_ssd_resnet34_1200", "unknown error"});
    broken_tests.insert({"tf_inception_v1", "flaky test"});  //TODO: Investigate cause for flakiness
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});
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
    broken_tests.insert({"convtranspose_1d", "1d convtranspose not supported yet"});

    // These tests are temporarily disabled pending a fix to the DML EP for handling of the output_padding attribute
    broken_tests.insert({"ConvTranspose2d", "Temporarily disabled due to EP bug"});
    broken_tests.insert({"ConvTranspose2d_no_bias", "Temporarily disabled due to EP bug"});
    broken_tests.insert({"operator_convtranspose", "Temporarily disabled due to EP bug"});

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
  try {
    retval = real_main(argc, argv, env);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }
  ::google::protobuf::ShutdownProtobufLibrary();
  return retval;
}

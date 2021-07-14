// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iterator>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "asserts.h"
#include <core/platform/path_lib.h>
#include "default_providers.h"

// test infrastructure
#include "test/onnx/TestCase.h"
#include "test/compare_ortvalue.h"
#include "test/onnx/heap_buffer.h"
#include "test/onnx/onnx_model_info.h"
#include "test/onnx/callback.h"

extern std::unique_ptr<Ort::Env> ort_env;

using namespace onnxruntime::common;

namespace onnxruntime {
namespace test {
// parameter is provider_name + "_" + model_path
class ModelTest : public testing::TestWithParam<std::basic_string<ORTCHAR_T>> {};

namespace {
struct BrokenTest {
  std::string test_name_;
  std::string reason_;
  std::set<std::string> broken_versions_ = {};  // apply to all versions if empty
  BrokenTest(std::string name, std::string reason) : test_name_(std::move(name)), reason_(std::move(reason)) {
  }

  BrokenTest(std::string name, std::string reason, const std::initializer_list<std::string>& versions)
      : test_name_(std::move(name)), reason_(std::move(reason)), broken_versions_(versions) {
  }

  bool operator<(const struct BrokenTest& test) const {
    return strcmp(test_name_.c_str(), test.test_name_.c_str()) < 0;
  }
};
}  // namespace
#ifdef GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ModelTest);
#endif
TEST_P(ModelTest, Run) {
  std::basic_string<ORTCHAR_T> param = GetParam();
  size_t pos = param.find(ORT_TSTR("_"));
  ASSERT_NE(pos, std::string::npos);
  std::string provider_name = ToMBString(param.substr(0, pos));
  std::basic_string<ORTCHAR_T> model_path = param.substr(pos + 1);
  double per_sample_tolerance = 1e-3;
  // when cuda is enabled, set it to a larger value for resolving random MNIST test failure
  // when openvino is enabled, set it to a larger value for resolving MNIST accuracy mismatch
  double relative_per_sample_tolerance = 1e-3;
  if (provider_name == "openvino") {
    relative_per_sample_tolerance = 0.009;
  }

  std::unique_ptr<OnnxModelInfo> model_info = std::make_unique<OnnxModelInfo>(model_path.c_str());
  if (model_info->GetONNXOpSetVersion() != 8 && provider_name == "tensorrt") {
    // TensorRT can run most of the model tests, but only part of
    // them is enabled here to save CI build time.
    return;
  }
  if (model_info->GetONNXOpSetVersion() == 10 && provider_name == "dnnl") {
    // DNNL can run most of the model tests, but only part of
    // them is enabled here to save CI build time.
    return;
  }
#ifndef ENABLE_TRAINING
  if (model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_TRAINING_DOMAIN) ||
      model_info->HasDomain(ONNX_NAMESPACE::AI_ONNX_PREVIEW_TRAINING_DOMAIN)) {
    return;
  }
#endif
  // TODO: filter model based on opset
  std::set<BrokenTest> broken_tests = {
      {"slice_neg_steps", "Type parameter (Tind) bound to different types (tensor(int64) and tensor(int32) in node ()."},
      {"cast_BFLOAT16_to_FLOAT", "Unexpected input data type"},
      {"loop13_seq", "Creation of empty sequences is currently not supported in the test runner"},
      {"sequence_insert_at_front", "shape mismatch, expect {4} got {3}"},
      {"cast_FLOAT_to_BFLOAT16", "expect uint16 got bfloat16"},
      {"mnist", "Input data isn't in valid range"},
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
      {"maxunpool_export_with_output_shape",
       "Invalid output in ONNX test. See https://github.com/onnx/onnx/issues/2398"},
      {"cntk_simple_seg", "Bad onnx test output caused by wrong SAME_UPPER/SAME_LOWER for ConvTranspose"},
      {"training_dropout", "result differs", {}},               // Temporary, subsequent PR will remove this.
      {"training_dropout_default", "result differs", {}},       // Temporary, subsequent PR will remove this.
      {"training_dropout_default_mask", "result differs", {}},  // Temporary, subsequent PR will remove this.
      {"training_dropout_mask", "result differs", {}},          // Temporary, subsequent PR will remove this.
#ifdef ENABLE_TRAINING
      {"adagrad", "not a registered function/op", {}},                  // Op not registered.
      {"adagrad_multiple", "not a registered function/op", {}},         // Op not registered.
      {"adam", "not a registered function/op", {}},                     // Op not registered.
      {"adam_multiple", "not a registered function/op", {}},            // Op not registered.
      {"gradient_of_add", "not a registered function/op", {}},          // Op not registered.
      {"gradient_of_add_and_mul", "not a registered function/op", {}},  // Op not registered.
      {"momentum", "not a registered function/op", {}},                 // Op not registered.
      {"momentum_multiple", "not a registered function/op", {}},        // Op not registered.
      {"nesterov_momentum", "not a registered function/op", {}},        // Op not registered.
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_4d", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_3d_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_none_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_3d", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_ignore_index_3d_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_none_weights_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_sum_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight_ignore_index", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_3d", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "type error", {"onnx170"}},
      {"softmax_cross_entropy_sum", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_none_weights", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d_log_prob", "type error", {"onnx170"}},
      {"softmax_cross_entropy_none", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "type error", {"onnx170"}},
      {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_weight", "type error", {"onnx170"}},
      {"softmax_cross_entropy_mean_no_weight_ignore_index_4d", "type error", {"onnx170"}},
#endif
      {"mask_rcnn_keras", "this model currently has an invalid contrib op version set to 10", {}}};

  if (provider_name == "nuphar") {
    // https://msdata.visualstudio.com/Vienna/_workitems/edit/1000703
    broken_tests.insert({"fp16_test_tiny_yolov2", "Computed value is off by a bit more than tol."});
    broken_tests.insert({"keras2coreml_Repeat_ImageNet", "this test fails with Nuphar EP."});
    broken_tests.insert({"fp16_coreml_FNS-Candy", "this test fails with Nuphar EP."});
  }

  if (provider_name == "nnapi") {
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
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum_expanded", "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean_expanded", "shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_expanded", "shape mismatch"});
    // Disable based on George Wu's recommendation.
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index_expanded",
         "shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_iinput_shape_is_NCd1_weight_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NC", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_no_weight_reduction_mean_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_mean", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_reduction_sum", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_mean", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2_with_weight_reduction_sum_ignore_index",
                         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                         "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1_mean_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                         "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_sum_weight_high_ignore_index_log_prob_expanded",
                         "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded", "Shape mismatch"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob", "Shape mismatch"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded", "Shape mismatch"});
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
  }

  if (provider_name == "dml") {
    broken_tests.insert({"tinyyolov3", "The parameter is incorrect"});
    broken_tests.insert({"PixelShuffle", "Test requires 6D Reshape, which isn't supported by DirectML"});
    broken_tests.insert({"operator_permute2", "Test requires 6D Transpose, which isn't supported by DirectML"});
    broken_tests.insert({"resize_downsample_linear",
                         "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests.insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});
    broken_tests.insert(
        {"resize_upsample_linear", "ORT 0.4 uses asymmetric but will conform to half_pixel in the next ONNX version."});

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

    broken_tests.insert({"resize_downsample_scales_linear",
                         "DML uses half_pixel and this test assumed \"asymmetric\" but does not include \"mode\""});
    broken_tests.insert({"resize_downsample_sizes_linear_pytorch_half_pixel",
                         "DML does not support downsampling by such a large factor - skips input pixels"});
    broken_tests.insert({"resize_downsample_sizes_nearest",
                         "DML uses pixel centers for nearest, rounding 1 value off for the middle column"});
    broken_tests.insert({"resize_upsample_sizes_nearest",
                         "DML uses pixel centers for nearest, which makes more sense (the 3rd row mismatches)"});
    broken_tests.insert({"unsqueeze_three_axes", "DML does not support 6D tensors"});
    broken_tests.insert({"unsqueeze_unsorted_axes", "DMLdoes not support 6D tensors"});

    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3_none_no_weight_negative_ignore_index_log_prob_expanded",
         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded", "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert(
        {"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight", "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob",
                         "DML does not support 5D+ tensors"});
    broken_tests.insert({"softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded",
                         "DML does not support 5D+ tensors"});
  }

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
  broken_tests.insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"bidaf", "This model uses contrib ops."});
  broken_tests.insert({"fp16_test_tiny_yolov2", "This model uses contrib ops."});
  broken_tests.insert({"fp16_coreml_FNS-Candy", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Repeat_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_BiDirectional_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"fp16_coreml_LinearRegression_NYCTaxi", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Average_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_GRU_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_SimpleRNN_ImageNet", "This model uses contrib ops."});
  broken_tests.insert({"keras2coreml_Dot_imageNet", "This model uses contrib ops."});
#endif

  std::basic_string<ORTCHAR_T> model_dir;
  (void)GetDirNameFromFilePath(model_path, model_dir);
  std::basic_string<PATH_CHAR_TYPE> test_case_name = GetLastComponent(model_dir);
  if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
    test_case_name = test_case_name.substr(5);
  {
    BrokenTest t = {ToMBString(test_case_name), ""};
    auto iter = broken_tests.find(t);
    auto model_version = model_info->GetModelVersion();
    if (iter != broken_tests.end() &&
        (model_version == TestModelInfo::unknown_version || iter->broken_versions_.empty() ||
         iter->broken_versions_.find(model_version) != iter->broken_versions_.end())) {
      return;
    }
  }
  bool is_single_node = !model_info->GetNodeName().empty();
  std::vector<ExecutionMode> execution_modes = {ExecutionMode::ORT_SEQUENTIAL};
  if (provider_name == "cpu" && !is_single_node)
    execution_modes.push_back(ExecutionMode::ORT_PARALLEL);

#ifndef _OPENMP
  std::vector<bool> use_single_thread{false};
  // Test the model with intra op threadpool disabled
  if (provider_name == "cpu" && is_single_node)
    use_single_thread.push_back(true);
#endif

  std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToMBString(test_case_name), std::move(model_info),
                                                    per_sample_tolerance, relative_per_sample_tolerance);
#ifndef _OPENMP
  for (bool is_single_thread : use_single_thread) {
#endif
    for (ExecutionMode execution_mode : execution_modes) {
      SessionOptions so;
#ifndef _OPENMP
      if (!is_single_thread)
        so.use_per_session_threads = false;
      else
        so.intra_op_param.thread_pool_size = 1;  // Disable intra op thread pool
#endif
      so.execution_mode = execution_mode;
      so.session_logid = ToMBString(test_case_name);
      so.session_log_severity_level = (int)logging::Severity::kERROR;
      InferenceSession session_object(so, (**ort_env).GetEnvironment());
      if (provider_name == "cuda") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));
      } else if (provider_name == "rocm") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultRocmExecutionProvider()));
      } else if (provider_name == "dnnl") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultDnnlExecutionProvider()));
      } else if (provider_name == "nuphar") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultNupharExecutionProvider()));
      } else if (provider_name == "tensorrt") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultTensorrtExecutionProvider()));
      } else if (provider_name == "migraphx") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultMIGraphXExecutionProvider()));
      } else if (provider_name == "openvino") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultOpenVINOExecutionProvider()));
      } else if (provider_name == "nnapi") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultNnapiExecutionProvider()));
      } else if (provider_name == "rknpu") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultRknpuExecutionProvider()));
      } else if (provider_name == "acl") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultAclExecutionProvider()));
      } else if (provider_name == "armnn") {
        ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultArmNNExecutionProvider()));
      }

      ASSERT_STATUS_OK(session_object.Load(model_path));
      auto st = session_object.Initialize();
      if (st.Code() == NOT_IMPLEMENTED)
        return;
      ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();
      const size_t data_count = l->GetDataCount();
      for (size_t task_id = 0; task_id != data_count; ++task_id) {
        onnxruntime::test::HeapBuffer holder;
        std::unordered_map<std::string, Ort::Value> feeds;
        l->LoadTestData(task_id, holder, feeds, true);

        std::pair<common::Status, const OutputDefList*> output_meta_data = session_object.GetModelOutputs();
        ASSERT_STATUS_OK(output_meta_data.first);
        // Create output feed
        size_t output_count = output_meta_data.second->size();
        std::vector<std::string> output_names(output_count);
        for (size_t i = 0; i != output_count; ++i) {
          output_names[i] = (*output_meta_data.second)[i]->Name();
        }

        std::vector<OrtValue> output_values(output_count);
        {
          std::unordered_map<std::string, OrtValue> input;
          for (auto& p : feeds) {
            const OrtValue* v = p.second;
            input.emplace(p.first, *v);
          }
          ASSERT_STATUS_OK(session_object.Run(input, output_names, &output_values));
        }

        bool post_procesing = false;
        Status status;
        l->GetPerSampleTolerance(&per_sample_tolerance);
        l->GetRelativePerSampleTolerance(&relative_per_sample_tolerance);
        l->GetPostProcessing(&post_procesing);

        // TODO: if there are no output value files, just skip the validation
        std::unordered_map<std::string, Ort::Value> expected_output_values;
        l->LoadTestData(task_id, holder, expected_output_values, false);

        std::unordered_map<std::string, OrtValue*> name_fetch_output_map;
        std::unordered_map<std::string, const ONNX_NAMESPACE::ValueInfoProto*> name_output_value_info_proto;
        size_t i = 0;
        for (auto& output_name : output_names) {
          // p_fetches is filled in the order of output_names.
          name_fetch_output_map[output_name] = &output_values[i];
          const ONNX_NAMESPACE::ValueInfoProto* infoProto = l->GetOutputInfoFromModel(i);
          if (infoProto != nullptr)
            name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
          i++;
        }

        for (auto& output : expected_output_values) {
          const OrtValue* expected_output_value = output.second;
          const std::string& output_name = output.first;
          auto iter = name_fetch_output_map.find(output_name);
          ASSERT_NE(iter, name_fetch_output_map.end());

          OrtValue* actual_output_value = iter->second;
          std::pair<COMPARE_RESULT, std::string> ret =
              CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                              relative_per_sample_tolerance, post_procesing);
          COMPARE_RESULT compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
          if (v == nullptr)
            continue;
          ret = VerifyValueInfo(*v, Ort::Unowned<Ort::Value>{actual_output_value});
          compare_result = ret.first;
          ASSERT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;

          if (compare_result != COMPARE_RESULT::SUCCESS) {
            break;
          }
        }
      }
    }
#ifndef _OPENMP
  }
#endif
}

// TODO: all providers
::std::vector<::std::basic_string<ORTCHAR_T>> GetParameterStrings() {
  std::vector<const ORTCHAR_T*> provider_names;
  provider_names.push_back(ORT_TSTR("cpu"));
#ifdef USE_TENSORRT
  provider_names.push_back(ORT_TSTR("tensorrt"));
#endif
#ifdef USE_MIGRAPHX
  provider_names.push_back(ORT_TSTR("migraphx"));
#endif
#ifdef USE_OPENVINO
  provider_names.push_back(ORT_TSTR("openvino"));
#endif
#ifdef USE_CUDA
  provider_names.push_back(ORT_TSTR("cuda"));
#endif
#ifdef USE_ROCM
  provider_names.push_back(ORT_TSTR("rocm"));
#endif
#ifdef USE_DNNL
  provider_names.push_back(ORT_TSTR("dnnl"));
#endif
#ifdef USE_NUPHAR
  provider_names.push_back(ORT_TSTR("nuphar"));
#endif
// For any non-Android system, NNAPI will only be used for ort model converter
#if defined(USE_NNAPI) && defined(__ANDROID__)
  provider_names.push_back(ORT_TSTR("nnapi"));
#endif
#ifdef USE_RKNPU
  provider_names.push_back(ORT_TSTR("rknpu"));
#endif
#ifdef USE_ACL
  provider_names.push_back(ORT_TSTR("acl"));
#endif
#ifdef USE_ARMNN
  provider_names.push_back(ORT_TSTR("armnn"));
#endif
  std::vector<std::basic_string<ORTCHAR_T>> v;
  // Permanently exclude following tests because ORT support only opset starting from 7,
  // Please make no more changes to the list
  static const ORTCHAR_T* immutable_broken_tests[] = {
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
      ORT_TSTR("fp16_shufflenet"),
      ORT_TSTR("fp16_tiny_yolov2"),
      ORT_TSTR("candy"),
      ORT_TSTR("tinyyolov3"),
      ORT_TSTR("mlperf_ssd_mobilenet_300"),
      ORT_TSTR("mlperf_ssd_resnet34_1200"),
      ORT_TSTR("tf_inception_v1"),
      ORT_TSTR("faster_rcnn"),
      ORT_TSTR("split_zero_size_splits"),
      ORT_TSTR("convtranspose_3d"),
      ORT_TSTR("fp16_test_tiny_yolov2-Candy"),
      ORT_TSTR("fp16_coreml_FNS-Candy"),
      ORT_TSTR("fp16_test_tiny_yolov2"),
      ORT_TSTR("fp16_test_shufflenet"),
      ORT_TSTR("keras2coreml_SimpleRNN_ImageNet")};
  static const ORTCHAR_T* openvino_disabled_tests[] = {ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                       ORT_TSTR("bertsquad"),
                                                       ORT_TSTR("yolov3"),
                                                       ORT_TSTR("LSTM_Seq_lens_unpacked"),
                                                       ORT_TSTR("tinyyolov3"),
                                                       ORT_TSTR("faster_rcnn"),
                                                       ORT_TSTR("mask_rcnn"),
                                                       ORT_TSTR("coreml_FNS-Candy_ImageNet"),
                                                       ORT_TSTR("tf_mobilenet_v2_1.0_224"),
                                                       ORT_TSTR("tf_mobilenet_v2_1.4_224"),
                                                       ORT_TSTR("operator_permute2"),
                                                       ORT_TSTR("operator_repeat"),
                                                       ORT_TSTR("operator_repeat_dim_overflow"),
                                                       ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                       ORT_TSTR("candy"),
                                                       ORT_TSTR("cntk_simple_seg"),
                                                       ORT_TSTR("GPT2_LM_HEAD"),
                                                       ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight"),
                                                       ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded"),
                                                       ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight"),
                                                       ORT_TSTR("negative_log_likelihood_loss_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_expanded"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob_expanded"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_expanded"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob"),
                                                       ORT_TSTR("softmax_cross_entropy_input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob_expanded")};
  static const ORTCHAR_T* dml_disabled_tests[] = {ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                  ORT_TSTR("mlperf_ssd_mobilenet_300"), ORT_TSTR("mask_rcnn"),
                                                  ORT_TSTR("faster_rcnn"), ORT_TSTR("tf_pnasnet_large"),
                                                  ORT_TSTR("zfnet512"), ORT_TSTR("keras2coreml_Dense_ImageNet")};
  static const ORTCHAR_T* dnnl_disabled_tests[] = {ORT_TSTR("densenet121"), ORT_TSTR("resnet18v2"),
                                                   ORT_TSTR("resnet34v2"), ORT_TSTR("resnet50v2"),
                                                   ORT_TSTR("resnet101v2"),
                                                   ORT_TSTR("resnet101v2"), ORT_TSTR("vgg19"),
                                                   ORT_TSTR("tf_inception_resnet_v2"), ORT_TSTR("tf_inception_v1"),
                                                   ORT_TSTR("tf_inception_v3"), ORT_TSTR("tf_inception_v4"),
                                                   ORT_TSTR("tf_mobilenet_v1_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.0_224"),
                                                   ORT_TSTR("tf_mobilenet_v2_1.4_224"), ORT_TSTR("tf_nasnet_large"),
                                                   ORT_TSTR("tf_pnasnet_large"), ORT_TSTR("tf_resnet_v1_50"),
                                                   ORT_TSTR("tf_resnet_v1_101"), ORT_TSTR("tf_resnet_v1_101"),
                                                   ORT_TSTR("tf_resnet_v2_101"), ORT_TSTR("tf_resnet_v2_152"),
                                                   ORT_TSTR("batchnorm_example_training_mode"),
                                                   ORT_TSTR("batchnorm_epsilon_training_mode"),
                                                   ORT_TSTR("mobilenetv2-1.0"),
                                                   ORT_TSTR("candy"),
                                                   ORT_TSTR("range_float_type_positive_delta_expanded"),
                                                   ORT_TSTR("range_int32_type_negative_delta_expanded"),
                                                   ORT_TSTR("averagepool_2d_ceil"),
                                                   ORT_TSTR("maxpool_2d_ceil"),
                                                   ORT_TSTR("maxpool_2d_dilations"),
                                                   ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                   ORT_TSTR("convtranspose_1d"),
                                                   ORT_TSTR("convtranspose_3d"),
                                                   ORT_TSTR("maxpool_2d_uint8")};
  static const ORTCHAR_T* tensorrt_disabled_tests[] = {
      ORT_TSTR("udnie"), ORT_TSTR("rain_princess"),
      ORT_TSTR("pointilism"), ORT_TSTR("mosaic"),
      ORT_TSTR("LSTM_Seq_lens_unpacked"),
      ORT_TSTR("cgan"), ORT_TSTR("candy"),
      ORT_TSTR("tinyyolov3"), ORT_TSTR("yolov3"),
      ORT_TSTR("mlperf_ssd_resnet34_1200"), ORT_TSTR("mlperf_ssd_mobilenet_300"),
      ORT_TSTR("mask_rcnn"),
      ORT_TSTR("faster_rcnn"),
      ORT_TSTR("fp16_shufflenet"),
      ORT_TSTR("fp16_inception_v1"),
      ORT_TSTR("fp16_tiny_yolov2"),
      ORT_TSTR("tf_inception_v3"),
      ORT_TSTR("tf_mobilenet_v1_1.0_224"),
      ORT_TSTR("tf_mobilenet_v2_1.0_224"),
      ORT_TSTR("tf_mobilenet_v2_1.4_224"),
      ORT_TSTR("tf_resnet_v1_101"),
      ORT_TSTR("tf_resnet_v1_152"),
      ORT_TSTR("tf_resnet_v1_50"),
      ORT_TSTR("tf_resnet_v2_101"),
      ORT_TSTR("tf_resnet_v2_152"),
      ORT_TSTR("tf_resnet_v2_50"),
      ORT_TSTR("convtranspose_1d"),
      ORT_TSTR("convtranspose_3d"),
      ORT_TSTR("conv_with_strides_and_asymmetric_padding"),
      ORT_TSTR("conv_with_strides_padding"),
      ORT_TSTR("size")  //INVALID_ARGUMENT: Cannot find binding of given name: x
  };
  for (const ORTCHAR_T* provider_name : provider_names) {
    std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests(std::begin(immutable_broken_tests),
                                                                        std::end(immutable_broken_tests));
    if (CompareCString(provider_name, ORT_TSTR("cuda")) == 0) {
      all_disabled_tests.insert(std::begin(cuda_flaky_tests), std::end(cuda_flaky_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("dml")) == 0) {
      all_disabled_tests.insert(std::begin(dml_disabled_tests), std::end(dml_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("dnnl")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(dnnl_disabled_tests), std::end(dnnl_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("tensorrt")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(tensorrt_disabled_tests), std::end(tensorrt_disabled_tests));
    } else if (CompareCString(provider_name, ORT_TSTR("openvino")) == 0) {
      // these models run but disabled tests to keep memory utilization low
      // This will be removed after LRU implementation
      all_disabled_tests.insert(std::begin(openvino_disabled_tests), std::end(openvino_disabled_tests));
    }

#if !defined(__amd64__) && !defined(_M_AMD64)
    // out of memory
    static const ORTCHAR_T* x86_disabled_tests[] = {ORT_TSTR("BERT_Squad"),
                                                    ORT_TSTR("bvlc_alexnet"),
                                                    ORT_TSTR("bvlc_reference_caffenet"),
                                                    ORT_TSTR("coreml_VGG16_ImageNet"),
                                                    ORT_TSTR("faster_rcnn"),
                                                    ORT_TSTR("GPT2"),
                                                    ORT_TSTR("GPT2_LM_HEAD"),
                                                    ORT_TSTR("keras_lotus_resnet3D"),
                                                    ORT_TSTR("mlperf_ssd_resnet34_1200"),
                                                    ORT_TSTR("mask_rcnn_keras"),
                                                    ORT_TSTR("mask_rcnn"),
                                                    ORT_TSTR("ssd"),
                                                    ORT_TSTR("vgg19"),
                                                    ORT_TSTR("zfnet512")};
    all_disabled_tests.insert(std::begin(x86_disabled_tests), std::end(x86_disabled_tests));
#endif

    std::vector<std::basic_string<ORTCHAR_T>> paths;
#if defined(NDEBUG) || defined(RUN_MODELTEST_IN_DEBUG_MODE)
#ifdef _WIN32
    paths.push_back(ORT_TSTR("..\\models"));
#else
    paths.push_back(ORT_TSTR("../models"));
#endif
#endif

// TENSORRT/OpenVino has too many test failures in the single node tests
#if !defined(_WIN32) && !defined(USE_TENSORRT) && !defined(USE_OPENVINO)
    paths.push_back("/data/onnx");
#endif
    while (!paths.empty()) {
      std::basic_string<ORTCHAR_T> node_data_root_path = paths.back();
      paths.pop_back();
      std::basic_string<ORTCHAR_T> my_dir_name = GetLastComponent(node_data_root_path);
      ORT_TRY {
        LoopDir(node_data_root_path, [&](const ORTCHAR_T* filename, OrtFileType f_type) -> bool {
          if (filename[0] == ORT_TSTR('.'))
            return true;
          if (f_type == OrtFileType::TYPE_DIR) {
            std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename);
            paths.push_back(p);
            return true;
          }
          std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
          if (!HasExtensionOf(filename_str, ORT_TSTR("onnx")))
            return true;

          std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
          if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0)
            test_case_name = test_case_name.substr(5);
          if (all_disabled_tests.find(test_case_name) != all_disabled_tests.end())
            return true;

#ifdef DISABLE_ML_OPS
          auto starts_with = [](const std::basic_string<PATH_CHAR_TYPE>& find_in,
                                const std::basic_string<PATH_CHAR_TYPE>& find_what) {
            return find_in.compare(0, find_what.size(), find_what) == 0;
          };
          if (starts_with(test_case_name, ORT_TSTR("XGBoost_")) || starts_with(test_case_name, ORT_TSTR("coreml_")) ||
              starts_with(test_case_name, ORT_TSTR("scikit_")) || starts_with(test_case_name, ORT_TSTR("libsvm_"))) {
            return true;
          }
#endif
          std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename_str);
          std::basic_string<PATH_CHAR_TYPE> r = provider_name;
          r.append(ORT_TSTR("_")).append(p);
          v.emplace_back(r);
          return true;
        });
      }
      ORT_CATCH(const std::exception&) {
      }  // ignore non-exist dir
    }
  }
  return v;
}

INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTest, testing::ValuesIn(GetParameterStrings()));

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"

using namespace std;
namespace onnxruntime {
namespace test {

namespace {

struct ConvTransposeOpAttributes {
  vector<int64_t> kernel_shape;
  vector<int64_t> output_padding;
  vector<int64_t> output_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  vector<int64_t> dilations;
  int64_t group;
  string auto_pad;
};

void TestConvTransposeOpInitializer(const ConvTransposeOpAttributes& attributes,
                                    const vector<vector<float>>& inputs,
                                    const vector<vector<int64_t>>& input_shapes,
                                    const std::initializer_list<float>& expected_output,
                                    const vector<int64_t>& expected_output_shape,
                                    bool is_filter_initializer = false,
                                    OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                                    const std::string& err_str = "",
                                    const std::unordered_set<std::string>& excluded_provider_types = {kTensorrtExecutionProvider}) {
  OpTester test("ConvTranspose", 11);
  test.AddAttribute("kernel_shape", attributes.kernel_shape);
  test.AddAttribute("group", attributes.group);

  // Only one of pads / auto_pad can be present
  if (!attributes.pads.empty()) {
    test.AddAttribute("pads", attributes.pads);
  } else {
    test.AddAttribute("auto_pad", attributes.auto_pad);
  }

  if (!attributes.output_padding.empty()) {
    test.AddAttribute("output_padding", attributes.output_padding);
  }
  if (!attributes.output_shape.empty()) {
    test.AddAttribute("output_shape", attributes.output_shape);
  }

  if (!attributes.strides.empty()) {
    test.AddAttribute("strides", attributes.strides);
  }

  if (!attributes.dilations.empty()) {
    test.AddAttribute("dilations", attributes.dilations);
  }

  ORT_ENFORCE(inputs.size() <= 3, "Our name array is only setup to handle 3 inputs");
  const char* szNames[] = {"X", "W", "B"};
  bool isInitializers[] = {false, is_filter_initializer, false};
  for (size_t i = 0; i < inputs.size(); i++) {
    test.AddInput<float>(szNames[i], input_shapes[i], inputs[i], isInitializers[i]);
  }
  test.AddOutput<float>("Y", expected_output_shape, expected_output);

  test.Run(expect_result, err_str, excluded_provider_types);  // Disable TensorRT because weight as input is not supported
}

void TestConvTransposeOp(const ConvTransposeOpAttributes& attributes,
                         const vector<vector<float>>& inputs,
                         const vector<vector<int64_t>>& input_shapes,
                         const std::initializer_list<float>& expected_output,
                         const vector<int64_t>& expected_output_shape,
                         OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess,
                         const std::string& err_str = "",
                         const std::unordered_set<std::string>& excluded_provider_types = {kTensorrtExecutionProvider}) {
  std::unordered_set<std::string> extra_exclude_openvino_for_initializer_filter = excluded_provider_types;
  extra_exclude_openvino_for_initializer_filter.insert(kOpenVINOExecutionProvider);
  TestConvTransposeOpInitializer(attributes, inputs, input_shapes, expected_output, expected_output_shape,
                                 true, expect_result, err_str, extra_exclude_openvino_for_initializer_filter);
  TestConvTransposeOpInitializer(attributes, inputs, input_shapes, expected_output, expected_output_shape,
                                 false, expect_result, err_str, excluded_provider_types);
}

}  // namespace

TEST(ConvTransposeTest, ConvTranspose_1D) {
  ConvTransposeOpAttributes attrs{
      vector<int64_t>{3},     // kernel_shape
      {},                     // output_padding
      {},                     // output_shape
      vector<int64_t>{0, 0},  // pads
      vector<int64_t>{1},     // strides
      vector<int64_t>{1},     // dilations
      1,                      // group
      "NOTSET"                // auto_pad
  };
  vector<float> X = {0.0f, 1.0f, 2.0f};
  vector<int64_t> X_shape = {1, 1, 3};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 2, 3};
  vector<int64_t> Y_shape = {1, 2, 5};
  auto expected_vals = {0.0f, 1.0f, 3.0f, 3.0f, 2.0f, 0.0f, 1.0f, 3.0f, 3.0f, 2.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1},        // output_padding
      {},                           // output_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{2, 2},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0.16857791f, -0.15161794f, 0.08540368f,
                     0.1820628f, -0.21746576f, 0.08245695f,
                     0.1431433f, -0.43156421f, 0.30591947f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {-0.06230065f, 0.37932432f, -0.25388849f,
                     0.33878803f, 0.43709868f, -0.22477469f,
                     0.04118127f, -0.44696793f, 0.06373066f};
  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {1, 1, 6, 6};
  auto expected_vals = {0.07368518f, -0.08925839f, -0.06627201f, 0.06301362f, 0.03732984f, -0.01919658f,
                        -0.00628807f, -0.02817563f, -0.01472169f, 0.04392925f, -0.00689478f, -0.01549204f,
                        0.07957941f, -0.11459791f, -0.09505399f, 0.07681622f, 0.03604182f, -0.01853423f,
                        -0.0270785f, -0.00680824f, -0.06650258f, 0.08004665f, 0.07918708f, -0.0724144f,
                        0.06256775f, -0.17838378f, -0.18863615f, 0.20064656f, 0.133717f, -0.06876295f,
                        -0.06398046f, -0.00864975f, 0.19289537f, -0.01490572f, -0.13673618f, 0.01949645f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Bias_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{0, 0},        // output_padding
      {},                           // output_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0.22572887f, -0.07105902f, -0.40399021f, -0.14461157f, 0.05367219f,
                     -0.08353302f, 0.41023391f, 0.42745841f, -0.3769345f, -0.42057109f,
                     -0.1372498f, 0.05485916f, 0.34602994f, -0.06402895f, -0.06000063f,
                     0.07891446f, -0.09410021f, 0.26251942f, -0.11043271f, 0.47966552f,
                     0.34682763f, -0.04511502f, 0.22414422f, 0.24618894f, -0.21480265f};
  vector<int64_t> X_shape = {1, 1, 5, 5};
  vector<float> W = {-0.0962126f, 0.19827795f, 0.03667754f,
                     0.36756599f, -0.01076147f, -0.11781135f,
                     -0.11574665f, -0.38404959f, 0.44403327f};
  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<float> B = {0.04676145f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {-0.03781903f, -0.09041066f, 0.14239404f, 0.09704495f, -0.03399426f,
                        0.08749044f, 0.35613984f, 0.07240347f, -0.27841991f, -0.00337578f,
                        0.07770107f, -0.09561026f, 0.13388641f, 0.30945939f, 0.14015588f,
                        0.13079405f, -0.00488365f, -0.06758944f, 0.45621645f, 0.01566098f,
                        0.00703105f, 0.12956856f, 0.0103332f, 0.04221053f, -0.21318194f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Bias_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0},        // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0.01270282f, 0.09657472f, -0.36909008f, -0.08085269f,
                     0.0242992f, 0.40873009f, -0.46927932f, 0.34412372f,
                     -0.39574206f, 0.26234281f, 0.27352369f, -0.22265741f,
                     0.43270493f, -0.24710381f, -0.03418651f, -0.04413456f,
                     -0.16414353f, 0.3158558f, 0.1087395f, -0.38577938f,
                     -0.38986659f, -0.09614426f, 0.17591673f, 0.40140027f,
                     -0.0869683f, -0.47193506f, -0.05010766f, 0.29325962f,
                     0.22680271f, -0.0793834f, -0.36764491f, 0.20451134f,
                     0.46361887f, -0.12190259f, 0.03413916f, 0.12307656f,
                     0.28569579f, -0.392129f, 0.17179191f, 0.27161086f,
                     -0.12766263f, 0.1371125f, 0.28137422f, -0.39899838f,
                     0.23824286f, -0.19693244f, 0.32956779f, 0.46209556f,
                     -0.46913007f};
  vector<int64_t> X_shape = {1, 1, 7, 7};
  vector<float> W = {-0.34922412f, 0.1114341f, -0.01778314f, 0.46861196f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<float> B = {0.17402864f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 8, 8};
  auto expected_vals = {0.1695925f, 0.14171794f, 0.31368554f, 0.16113512f,
                        0.15653302f, 0.033998f, 0.38345876f, 0.12173492f,
                        0.05362644f, 0.35481372f, 0.09013268f, -0.06378071f,
                        0.24394518f, 0.00222442f, 0.50842237f, -0.07341707f,
                        0.17984779f, 0.35392997f, 0.03631867f, 0.16350585f,
                        0.30338728f, 0.2088346f, 0.47435546f, 0.0147884f,
                        0.20821247f, 0.08664516f, 0.03569011f, 0.16659322f,
                        0.47522858f, 0.19675478f, -0.10781619f, 0.02401161f,
                        0.0965334f, 0.1788421f, 0.36887163f, 0.2512877f,
                        0.00254938f, 0.04799958f, 0.11982619f, 0.31525785f,
                        0.12701407f, 0.19566584f, 0.31214368f, -0.10558143f,
                        0.18591091f, 0.46830338f, 0.05418756f, 0.20530567f,
                        0.07357728f, 0.39731777f, 0.1872202f, 0.08253923f,
                        0.11266428f, 0.17892915f, 0.32709083f, 0.1860041f,
                        0.16902491f, 0.3129794f, -0.01718347f, 0.28917417f,
                        0.07588299f, 0.32025051f, 0.39891475f, -0.04581133f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShape_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      {},                           // output_padding
      vector<int64_t>{1, 3, 4, 4},  // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  int image_size = 4 * 4;
  int input_channels = 3;
  int output_channels = 3;
  std::vector<float> X;
  for (int i = 0; i < input_channels * image_size; i++)
    X.push_back(1.0f);
  std::vector<float> W;
  int kernel_size = output_channels * input_channels * 3 * 3;
  for (int i = 0; i < kernel_size; i++)
    W.push_back(1.0f);

  vector<int64_t> X_shape = {1, 3, 4, 4};
  vector<int64_t> W_shape = {3, 3, 3, 3};

  vector<int64_t> Y_shape = {1, 3, 4, 4};
  auto expected_vals = {12.0f, 18.0f, 18.0f, 12.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        12.0f, 18.0f, 18.0f, 12.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        18.0f, 27.0f, 27.0f, 18.0f,
                        12.0f, 18.0f, 18.0f, 12.0f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShape_1_group_2_for_tranpose_path) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      {},                           // output_padding
      vector<int64_t>{1, 6, 4, 4},  // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      2,                            // group
      "NOTSET"                      // auto_pad
  };
  int image_size = 4 * 4;
  int input_channels = 3 * 2;
  int output_channels = 3;
  std::vector<float> X;
  for (int i = 0; i < input_channels * image_size; i++)
    X.push_back(1.0f);
  std::vector<float> W;
  int kernel_size = output_channels * input_channels * 3 * 3;
  for (int i = 0; i < kernel_size; i++)
    W.push_back(1.0f);

  vector<int64_t> X_shape = {1, 6, 4, 4};
  vector<int64_t> W_shape = {6, 3, 3, 3};

  vector<int64_t> Y_shape = {1, 6, 4, 4};
  auto expected_vals = {
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,  // duplicate below
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
  };
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShape_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 5},         // kernel_shape
      {},                            // output_padding
      vector<int64_t>{1, 1, 1, 14},  // output_shape
      vector<int64_t>{0, 0, 0, 0},   // pads
      vector<int64_t>{1, 1},         // strides
      vector<int64_t>{1, 1},         // dilations
      1,                             // group
      "NOTSET"                       // auto_pad
  };
  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 1, 1, 10};
  vector<float> W = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 5};
  vector<float> B = {1.0f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 1, 1, 14};
  auto expected_vals = {1.0f, 2.0f, 5.0f, 11.0f, 19.0f, 28.0f, 37.0f, 46.0f, 55.0f, 64.0f, 63.0f, 51.0f, 27.0f, 10.0f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_OutputShapeWithBatchSize) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 5},         // kernel_shape
      {},                            // output_padding
      vector<int64_t>{2, 1, 1, 14},  // output_shape
      vector<int64_t>{0, 0, 0, 0},   // pads
      vector<int64_t>{1, 1},         // strides
      vector<int64_t>{1, 1},         // dilations
      1,                             // group
      "NOTSET"                       // auto_pad
  };
  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                     10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
  vector<int64_t> X_shape = {2, 1, 1, 10};
  vector<float> W = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 5};
  vector<float> B = {1.0f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {2, 1, 1, 14};
  auto expected_vals = {1.0f, 2.0f, 5.0f, 11.0f, 19.0f, 28.0f, 37.0f, 46.0f, 55.0f, 64.0f, 63.0f, 51.0f, 27.0f, 10.0f,
                        11.0f, 32.0f, 65.0f, 91.0f, 109.0f, 118.0f, 127.0f, 136.0f, 145.0f, 154.0f, 143.0f, 111.0f, 57.0f, 20.0f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_InvalidKernelShape) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 1, 1, 5},   // invalid kernel_shape, should be [1, 5]
      {},                            // output_padding
      vector<int64_t>{2, 1, 1, 14},  // output_shape
      vector<int64_t>{0, 0, 0, 0},   // pads
      vector<int64_t>{1, 1},         // strides
      vector<int64_t>{1, 1},         // dilations
      1,                             // group
      "NOTSET"                       // auto_pad
  };
  vector<float> X = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                     10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f};
  vector<int64_t> X_shape = {2, 1, 1, 10};
  vector<float> W = {1.0f, 2.0f, 3.0f, 2.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 1, 5};
  vector<float> B = {1.0f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {2, 1, 1, 14};
  auto expected_vals = {1.0f, 2.0f, 5.0f, 11.0f, 19.0f, 28.0f, 37.0f, 46.0f, 55.0f, 64.0f, 63.0f, 51.0f, 27.0f, 10.0f,
                        11.0f, 32.0f, 65.0f, 91.0f, 109.0f, 118.0f, 127.0f, 136.0f, 145.0f, 154.0f, 143.0f, 111.0f, 57.0f, 20.0f};
  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectFailure,
                      "kernel_shape num_dims is not compatible with W num_dims. kernel_shape: {1,1,1,5} W: {1,1,1,5}");
}

TEST(ConvTransposeTest, ConvTranspose_onnx) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0., 1., 2., 3., 4., 5., 6., 7., 8.};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.};
  vector<int64_t> W_shape = {1, 2, 3, 3};
  vector<int64_t> Y_shape = {1, 2, 5, 5};
  auto expected_vals = {
      0.f, 0.f, 1.f, 4.f, 4.f,
      0.f, 6.f, 20.f, 26.f, 20.f,
      9.f, 36.f, 84.f, 84.f, 57.f,
      36.f, 90.f, 164.f, 134.f, 80.f,
      36.f, 84.f, 145.f, 112.f, 64.f,
      0.f, 9.f, 28.f, 31.f, 22.f,
      27.f, 78.f, 155.f, 134.f, 83.f,
      90.f, 225.f, 408.f, 327.f, 192.f,
      117.f, 270.f, 461.f, 350.f, 197.f,
      90.f, 201.f, 334.f, 247.f, 136.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_onnx2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.};
  vector<int64_t> W_shape = {2, 3, 2, 2};  // this requires weight transpose
  vector<int64_t> Y_shape = {1, 3, 4, 4};
  auto expected_vals = {
      108.f, 237.f, 263.f, 145.f,
      270.f, 592.f, 652.f, 358.f,
      354.f, 772.f, 832.f, 454.f,
      222.f, 481.f, 515.f, 279.f,
      144.f, 317.f, 359.f, 197.f,
      366.f, 800.f, 892.f, 486.f,
      498.f, 1076.f, 1168.f, 630.f,
      306.f, 657.f, 707.f, 379.f,
      180.f, 397.f, 455.f, 249.f,
      462.f, 1008.f, 1132.f, 614.f,
      642.f, 1380.f, 1504.f, 806.f,
      390.f, 833.f, 899.f, 479.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_onnx_group) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 1},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      vector<int64_t>{1, 1},        // dilations
      4,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  vector<int64_t> X_shape = {1, 16, 1, 1};
  vector<float> W = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f, 27.f, 28.f, 29.f, 30.f, 31.0f};
  vector<int64_t> W_shape = {16, 2, 1, 1};
  vector<int64_t> Y_shape = {1, 8, 1, 1};
  auto expected_vals = {28.f, 34.f, 252.f, 274.f, 732.f, 770.f, 1468.f, 1522.f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {2, 2},
      1,
      "NOTSET"};

  vector<float> X = {11.0f, 12.0f, 21.0f, 22.0f};
  vector<int64_t> X_shape = {1, 1, 2, 2};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 4, 4};
  auto expected_vals = {11.0f, 12.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 21.0f, 22.0f,
                        11.0f, 12.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 21.0f, 22.0f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {3, 3},
      1,
      "NOTSET"};

  vector<float> X = {11.0f, 12.0f, 21.0f, 22.0f};
  vector<int64_t> X_shape = {1, 1, 2, 2};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {11.0f, 12.0f, 0.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 0.0f, 21.0f, 22.0f,
                        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                        11.0f, 12.0f, 0.0f, 11.0f, 12.0f,
                        21.0f, 22.0f, 0.0f, 21.0f, 22.0f};
  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_3) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {2, 2},
      1,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 5, 5};
  auto expected_vals = {21.0f, 56.0f, 13.0f, 16.0f, 2.0f,
                        63.0f, 35.0f, 67.0f, 10.0f, 14.0f,
                        24.0f, 22.0f, 76.0f, 76.0f, 21.0f,
                        9.0f, 5.0f, 88.0f, 45.0f, 63.0f,
                        3.0f, 2.0f, 33.0f, 18.0f, 54.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_4) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {3, 3},
      1,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 6, 6};
  auto expected_vals = {21.0f, 56.0f, 7.0f, 6.0f, 16.0f, 2.0f,
                        63.0f, 35.0f, 49.0f, 18.0f, 10.0f, 14.0f,
                        21.0f, 14.0f, 42.0f, 6.0f, 4.0f, 12.0f,
                        3.0f, 8.0f, 1.0f, 27.0f, 72.0f, 9.0f,
                        9.0f, 5.0f, 7.0f, 81.0f, 45.0f, 63.0f,
                        3.0f, 2.0f, 6.0f, 27.0f, 18.0f, 54.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}


TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_AsymmetricPads_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{2, 2, 1, 1},
      vector<int64_t>{1, 1},
      {3, 3},
      1,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 3, 3};
  auto expected_vals = {42.0f, 6.0f, 4.0f,
                        1.0f, 27.0f, 72.0f,
                        7.0f, 81.0f, 45.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}


TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_AsymmetricPads_2) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{1, 1, 2, 2},
      vector<int64_t>{1, 1},
      {3, 3},
      1,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 3, 3};
  auto expected_vals = {35.0f, 49.0f, 18.0f,
                        14.0f, 42.0f, 6.0f,
                        8.0f, 1.0f, 27.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}


TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_AsymmetricPads_3) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{2, 2, 0, 0},
      vector<int64_t>{1, 1},
      {3, 3},
      1,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 4, 4};
  auto expected_vals = {42.0f, 6.0f, 4.0f, 12.0f,
                        1.0f, 27.0f, 72.0f, 9.0f,
                        7.0f, 81.0f, 45.0f, 63.0f,
                        6.0f, 27.0f, 18.0f, 54.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}


TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_AsymmetricPads_4) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 2, 2},
      vector<int64_t>{1, 1},
      {3, 3},
      1,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 6.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};
  vector<float> W = {7.0f, 2.0f, 1.0f, 9.0f};
  vector<int64_t> W_shape = {1, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 1, 4, 4};
  auto expected_vals = {21.0f, 56.0f, 7.0f, 6.0f,
                        63.0f, 35.0f, 49.0f, 18.0f,
                        21.0f, 14.0f, 42.0f, 6.0f,
                        3.0f, 8.0f, 1.0f, 27.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_Dilation_Group_1) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},
      {},
      {},
      vector<int64_t>{0, 0, 0, 0},
      vector<int64_t>{1, 1},
      {2, 2},
      2,
      "NOTSET"};

  vector<float> X = {3.0f, 8.0f, 1.0f, 9.0f, 5.0f, 7.0f, 3.0f, 2.0f, 3.0f, 7.0f, 9.0f, 1.0f, 5.0f, 2.0f, 3.0f, 9.0f, 0.0f, 2.0f};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {9.0f, 3.0f, 1.0f, 2.0f, 3.0f, 7.0f, 0.0f, 8.0f};
  vector<int64_t> W_shape = {2, 1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 5, 5};
  auto expected_vals = {27.0f, 72.0f, 18.0f, 24.0f, 3.0f,
                        81.0f, 45.0f, 90.0f, 15.0f, 21.0f,
                        30.0f, 26.0f, 43.0f, 22.0f, 11.0f,
                        9.0f, 5.0f, 25.0f, 10.0f, 14.0f,
                        3.0f, 2.0f, 9.0f, 4.0f, 6.0f,
                        21.0f, 27.0f, 52.0f, 63.0f, 7.0f,
                        15.0f, 6.0f, 44.0f, 14.0f, 21.0f,
                        27.0f, 0.0f, 125.0f, 72.0f, 22.0f,
                        0.0f, 0.0f, 40.0f, 16.0f, 24.0f,
                        0.0f, 0.0f, 72.0f, 0.0f, 16.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_DefaultStridesAndDilations) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2, 2},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{},            // strides
      vector<int64_t>{},            // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.};
  vector<int64_t> X_shape = {1, 2, 3, 3};
  vector<float> W = {0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23.};
  vector<int64_t> W_shape = {2, 3, 2, 2};  // this requires weight transpose
  vector<int64_t> Y_shape = {1, 3, 4, 4};
  auto expected_vals = {
      108.f, 237.f, 263.f, 145.f,
      270.f, 592.f, 652.f, 358.f,
      354.f, 772.f, 832.f, 454.f,
      222.f, 481.f, 515.f, 279.f,
      144.f, 317.f, 359.f, 197.f,
      366.f, 800.f, 892.f, 486.f,
      498.f, 1076.f, 1168.f, 630.f,
      306.f, 657.f, 707.f, 379.f,
      180.f, 397.f, 455.f, 249.f,
      462.f, 1008.f, 1132.f, 614.f,
      642.f, 1380.f, 1504.f, 806.f,
      390.f, 833.f, 899.f, 479.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_NonDefaultStridesAndDilations) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{1, 4},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 2},        // strides
      vector<int64_t>{1, 3},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {1., 2.};
  vector<int64_t> X_shape = {1, 1, 1, 2};
  vector<float> W = {1., 1., 1., 1.};
  vector<int64_t> W_shape = {1, 1, 1, 4};
  vector<int64_t> Y_shape = {1, 1, 1, 12};
  auto expected_vals = {1.f, 0.f, 2.f, 1.f, 0.f, 2.f, 1.f, 0.f, 2.f, 1.f, 0.f, 2.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, ConvTranspose_2D_NonDefaultStridesAndDilations_T) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{4, 1},        // kernel_shape
      {},                           // output_padding
      {},                           // output_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{2, 1},        // strides
      vector<int64_t>{3, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {1., 2.};
  vector<int64_t> X_shape = {1, 1, 2, 1};
  vector<float> W = {1., 1., 1., 1.};
  vector<int64_t> W_shape = {1, 1, 4, 1};
  vector<int64_t> Y_shape = {1, 1, 12, 1};
  auto expected_vals = {1.f, 0.f, 2.f, 1.f, 0.f, 2.f, 1.f, 0.f, 2.f, 1.f, 0.f, 2.f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(ConvTransposeTest, DimWithZero) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1},        // output_padding
      {},                           // output_shape
      vector<int64_t>{1, 1, 1, 1},  // pads
      vector<int64_t>{2, 2},        // strides
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      "NOTSET"                      // auto_pad
  };
  vector<float> X = {};
  vector<int64_t> X_shape = {0, 1, 3, 3};
  vector<float> W = {-0.06230065f, 0.37932432f, -0.25388849f,
                     0.33878803f, 0.43709868f, -0.22477469f,
                     0.04118127f, -0.44696793f, 0.06373066f};
  vector<int64_t> W_shape = {1, 1, 3, 3};
  vector<int64_t> Y_shape = {0, 1, 6, 6};
  initializer_list<float> expected_vals = {};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kAclExecutionProvider});
}

TEST(ConvTransposeTest, ConvTranspose_3D) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3, 3},           // kernel_shape
      {},                                 // output_padding
      {},                                 // output_shape
      vector<int64_t>{0, 0, 0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1, 1},           // strides
      vector<int64_t>{1, 1, 1},           // dilations
      1,                                  // group
      "NOTSET"                            // auto_pad
  };

  vector<float> X = {0.82670355f, -0.041401573f, 0.026631273f, -0.9765811f, -0.1628872f,
                     0.6781846f, 0.38049284f, 0.5573809f, -0.56348205f, 0.6192993f,
                     -1.3645133f, -0.025706587f, 0.3444407f, 1.6839422f, 0.44769225f,
                     -0.94038606f, 0.1865747f, 0.22024752f, -1.3399711f, 0.48898873f,

                     1.3871458f, -0.4798906f, -1.3498452f, 1.9507161f, -0.36717513f,
                     -1.3160661f, 0.04001215f, -1.6359671f, -0.27051282f, 1.3602601f,
                     -0.6915065f, -1.480801f, 0.008796313f, -0.42371505f, 0.2846156f,
                     -0.041113783f, -0.8274711f, 1.649845f, -1.4032182f, 0.9754836f,

                     -1.061012f, 1.9735539f, 1.5394408f, 0.46846536f, 1.5393354f,
                     -0.10323338f, -0.25534126f, 0.03429055f, 0.3142054f, -0.4348722f,
                     -1.399293f, 0.8268838f, 0.061832584f, 1.32346f, 1.326872f,
                     0.015338173f, -0.7772104f, 0.82150716f, -0.8285072f, -0.745792f};

  vector<int64_t> X_shape = {1, 1, 3, 4, 5};

  vector<float> W = {0.042848215f, 0.063926056f, -0.01786653f,
                     -0.007932588f, -0.06435914f, 0.045959294f,
                     -0.03683681f, 0.076584175f, -0.083441734f,

                     -0.08745442f, -0.053135775f, -0.07282642f,
                     -0.11123853f, -0.114605635f, 0.050257847f,
                     0.03769763f, 0.008607149f, -2.6613474e-05f,

                     -0.06418988f, 0.11692271f, 0.12203565f,
                     0.042627826f, 0.098034576f, -0.010402724f,
                     -0.12522504f, -0.10751359f, 0.12747335f,

                     -0.056666218f, -0.02816818f, -0.00018641353f,
                     -0.053967796f, 0.08958836f, -0.060138382f,
                     -0.108521f, -0.12912428f, 0.05260901f,

                     0.1330998f, -0.09916313f, -0.12123653f,
                     0.022630543f, -0.018046886f, 0.08967489f,
                     -0.033889048f, -0.006379664f, 0.059431687f,

                     0.04010451f, 0.103126734f, 0.0036035478f,
                     0.030677304f, 0.017750308f, 0.012351051f,
                     0.017721564f, -0.013308428f, -0.011259012f};

  vector<int64_t> W_shape = {1, 2, 3, 3, 3};

  vector<int64_t> B_shape = {2};
  vector<float> B = {-0.11784090101718903f, -0.060990236699581146f};

  vector<int64_t> Y_shape = {1, 2, 5, 6, 7};
  auto expected_vals = {-0.08241813f, -0.06676699f, -0.13411677f, -0.15724352f, -0.18772511f, -0.11080553f, -0.114930674f,
                        -0.0953398f, -0.111061305f, -0.0413035f, -0.10902196f, -0.071916685f, -0.102583766f, -0.13639182f,
                        -0.21214074f, -0.18799849f, -0.15122052f, 0.00434383f, -0.011207409f, -0.11604968f, -0.08378546f,
                        -0.1722928f, -0.044016793f, -0.1914465f, -0.16952308f, -0.39505655f, 0.080385f, -0.15767722f,
                        -0.060116887f, -0.16235165f, -0.075614765f, -0.14631891f, 0.05837299f, -0.31712085f, -0.13272354f,
                        -0.08320008f, -0.1967324f, -0.033198006f, -0.06718128f, -0.2568521f, 0.0314174f, -0.15864298f,

                        -0.13070306f, -0.09003539f, -0.29147533f, -0.024966106f, 0.079442084f, -0.096389435f, -0.09941827f,
                        -0.3365072f, -0.4451772f, -0.13154466f, -0.08992967f, -0.16572365f, 0.06494926f, -0.21230686f,
                        -0.11307171f, -0.056943115f, -0.35291147f, -0.317253f, -0.070464894f, -0.6300395f, -0.031246513f,
                        0.19395588f, 0.011135533f, 0.096916616f, -0.3942836f, -0.29872403f, 0.16881491f, -0.24881886f,
                        -0.038873613f, -0.032735735f, -0.21593677f, 0.088557824f, 0.13849314f, -0.30753696f, -0.07219358f,
                        -0.15177673f, -0.09156879f, -0.2286228f, 0.080623806f, -0.39201033f, 0.07819712f, -0.19924995f,

                        -0.3376814f, -0.033524483f, 0.230105f, -0.0377952f, -0.12315659f, -0.28858358f, -0.13848148f,
                        -0.16134796f, 0.012239918f, 0.27276647f, 0.020731017f, -0.4651906f, -0.14341736f, -0.07956973f,
                        0.1342433f, -0.16956037f, 0.310399f, 0.34338957f, -0.040192716f, 0.12504166f, -0.21490449f,
                        -0.15410437f, -0.1338158f, -0.39244395f, 0.29117042f, -0.26415867f, -0.4450379f, 0.0699404f,
                        0.042872816f, -0.14961651f, -0.17582522f, -0.6919577f, -0.13723494f, -0.0681901f, -0.16183335f,
                        -0.0021959245f, -0.0418434f, -0.32134426f, 0.16967098f, -0.08680786f, -0.32077473f, 0.0066963434f,

                        -0.114091426f, -0.041066267f, -0.080250874f, -0.72594404f, -0.30254412f, -0.03862554f, -0.27475363f,
                        0.15282185f, -0.22887689f, -0.72043663f, -0.47111863f, -0.3755179f, -0.20074406f, 0.16101281f,
                        -0.20939936f, -0.21245953f, 0.11726546f, -0.8030824f, -0.5866715f, 0.20001571f, -0.26259118f,
                        0.17054747f, 0.061063558f, -0.6348493f, 0.2620284f, -0.782919f, -0.31278569f, 0.2926497f,
                        -0.08745579f, 0.20646049f, -0.050303012f, -0.13460274f, 0.060659587f, -0.037006564f, -0.1292249f,
                        -0.11211421f, -0.038967483f, -0.21644044f, -0.24912538f, 0.08591288f, -0.40798867f, 0.006527111f,

                        -0.049734667f, -0.3685795f, -0.11538547f, 0.27292788f, 0.025990233f, 0.119311824f, 0.0700129f,
                        -0.156443f, -0.13340846f, 0.10764159f, -0.014803357f, 0.046525866f, 0.015691683f, -0.1869241f,
                        0.1004442f, -0.4885978f, -0.7585998f, -0.047841772f, -0.07570776f, 0.0471261f, 0.24483289f,
                        -0.16554686f, -0.1250152f, -0.15132052f, -0.08515984f, 0.14412321f, -0.1030291f, -0.2780918f,
                        0.05803944f, -0.10257156f, -0.4341917f, -0.13150966f, -0.53996617f, -0.15628646f, 0.059058204f,
                        -0.11976162f, -0.022163756f, -0.13519828f, -0.20148787f, 0.16934697f, -0.14327072f, -0.2129095f,

                        -0.107836396f, -0.0819309f, -0.06148723f, -0.0063935146f, -0.02425649f, -0.056219954f, -0.06095987f,
                        -0.14403576f, -0.025357183f, -0.15828207f, 0.012748428f, -0.16061643f, -0.03419252f, -0.05130991f,
                        -0.109983265f, -0.08312916f, -0.07035978f, -0.008285124f, -0.10610263f, -0.01489019f, -0.106886685f,
                        -0.007659614f, -0.2947925f, -0.09132287f, -0.040577132f, 0.089866154f, -0.24528673f, -0.055424154f,
                        0.13783869f, 0.023674607f, -0.10545369f, -0.20873478f, -0.4685722f, 0.09418375f, -0.06684458f,
                        0.0410614f, 0.04018917f, -0.15845582f, 0.06580096f, 0.070554025f, -0.19462511f, -0.03526502f,

                        -0.02956047f, -0.16035908f, -0.0638171f, -0.261022f, -0.022948403f, 0.08353848f, -0.041173913f,
                        0.04770004f, 0.091520615f, 0.006987013f, -0.39962748f, 0.23266485f, -0.32719564f, -0.12885109f,
                        -0.29559937f, -0.08031146f, 0.76168066f, 0.0009028502f, -0.4091536f, -0.14801738f, -0.17058557f,
                        -0.05754847f, 0.2955231f, -0.089874476f, 0.17254886f, -0.13203058f, -0.007648442f, 0.010943003f,
                        0.04123217f, 0.26074114f, -0.24313056f, 0.1008903f, -0.26472318f, 0.01998391f, -0.03422378f,
                        -0.024659738f, 0.033793047f, -0.1998924f, -0.110185415f, 0.10620246f, -0.3435271f, 0.019390412f,

                        0.21691665f, -0.26076952f, -0.5040901f, 0.28383943f, -0.34750903f, -0.32484284f, -0.01734912f,
                        -0.08909689f, -0.0466362f, 0.21648785f, 0.06733417f, 0.009496197f, 0.18728223f, -0.35110205f,
                        -0.04908372f, -0.36729553f, -0.346236f, -0.13589534f, -0.16435221f, -0.16853788f, 0.12264759f,
                        -0.019215636f, -0.38316554f, 0.35669535f, -0.56980205f, -0.059346225f, 0.15008381f, -0.1751053f,
                        0.059508912f, 0.116622455f, -0.32607535f, -0.22282779f, -0.29149055f, -0.3829086f, 0.15905643f,
                        -0.077926554f, 0.06549884f, -0.09004557f, -0.15897253f, 0.26810864f, -0.08931713f, -0.047756508f,

                        -0.14657992f, 0.43070868f, -0.021787114f, -0.4532621f, 0.092385404f, -0.30126676f, -0.24893704f,
                        -0.10896815f, -0.14514503f, -0.21353528f, 0.018723361f, 0.037694372f, 0.11514955f, 0.13013864f,
                        -0.25713888f, -0.056000195f, -0.3505367f, 0.0836427f, -0.032017898f, -0.26742116f, -0.14740711f,
                        -0.13330215f, -0.18958306f, -0.08968873f, 0.014723815f, -0.20343366f, 0.3098968f, 0.114284225f,
                        -0.026738256f, -0.14110464f, -0.054464605f, -0.17529932f, -0.0030034669f, -0.050670102f, -0.04016705f,
                        -0.062238634f, -0.04886609f, -0.042247344f, -0.12185234f, 0.0357792f, -0.10265522f, -0.116296895f,

                        -0.1035416f, -0.09126053f, 0.20045105f, 0.12366664f, 0.05460281f, 0.09944453f, -0.055443168f,
                        -0.09767935f, -0.040166672f, -0.01716708f, 0.020299219f, 0.02864775f, -0.07159522f, -0.04354491f,
                        -0.1390779f, -0.13270372f, 0.02992779f, -0.025869183f, 0.12530258f, 0.05101595f, -0.07891131f,
                        -0.1051311f, -0.093200594f, -0.10368025f, 0.047598884f, -0.12069465f, -0.098738566f, -0.042393237f,
                        -0.08531736f, -0.051284637f, -0.04354899f, -0.06810297f, -0.083224006f, -0.11702064f, -0.08514082f,
                        -0.06071842f, -0.07496775f, -0.03626109f, -0.07785503f, -0.07243007f, -0.041736744f, -0.052593358f};

  TestConvTransposeOp(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectSuccess, "",
                      {kTensorrtExecutionProvider, kCudaExecutionProvider});
}

TEST(ConvTransposeTest, ConvTranspose_1D_AsymmetricPads) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2},  // kernel_shape
      {},                  // output_padding
      {},                  // output_shape
      {1, 0},              // pads (asymmetric)
      vector<int64_t>{1},  // strides
      vector<int64_t>{1},  // dilations
      1,                   // group
      "NOTSET"             // auto_pad
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  vector<int64_t> X_shape = {1, 1, 4};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 4};
  auto expected_vals = {3.0f, 5.0f, 7.0f, 4.0f, 3.0f, 5.0f, 7.0f, 4.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(ConvTransposeTest, ConvTranspose_1D_AutoPad_SameUpper) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2},  // kernel_shape
      {},                  // output_padding
      {},                  // output_shape
      {},                  // pads
      vector<int64_t>{1},  // strides
      vector<int64_t>{1},  // dilations
      1,                   // group
      "SAME_UPPER"         // auto_pad
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  vector<int64_t> X_shape = {1, 1, 4};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 4};
  auto expected_vals = {3.0f, 5.0f, 7.0f, 4.0f, 3.0f, 5.0f, 7.0f, 4.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider}); //Accuracy Mismatch on OpenVINO-EP
}

TEST(ConvTransposeTest, ConvTranspose_1D_AutoPad_SameLower) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{2},  // kernel_shape
      {},                  // output_padding
      {},                  // output_shape
      {},                  // pads
      vector<int64_t>{1},  // strides
      vector<int64_t>{1},  // dilations
      1,                   // group
      "SAME_LOWER"         // auto_pad
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f};
  vector<int64_t> X_shape = {1, 1, 4};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 2, 2};
  vector<int64_t> Y_shape = {1, 2, 4};
  auto expected_vals = {1.0f, 3.0f, 5.0f, 7.0f, 1.0f, 3.0f, 5.0f, 7.0f};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider}); //Accuracy Mismatch on OpenVINO-EP
}

TEST(ConvTransposeTest, ConvTranspose_AutoPad_with_non_default_strides) {
  ConvTransposeOpAttributes attrs = {
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // output_padding
      {},                     // output_shape
      {},                     // pads
      vector<int64_t>{2, 2},  // strides
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      "SAME_LOWER"            // auto_pad
  };

  vector<float> X = {0.0f, 1.0f, 2.0f,
                     3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f};
  vector<int64_t> X_shape = {1, 1, 3, 3};

  vector<float> W = {1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,

                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f,
                     1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {1, 2, 3, 3};

  auto expected_vals = {0.0f, 0.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                        0.0f, 0.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                        3.0f, 3.0f, 8.0f, 5.0f, 12.0f, 7.0f,
                        3.0f, 3.0f, 7.0f, 4.0f, 9.0f, 5.0f,
                        9.0f, 9.0f, 20.0f, 11.0f, 24.0f, 13.0f,
                        6.0f, 6.0f, 13.0f, 7.0f, 15.0f, 8.0f,

                        0.0f, 0.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                        0.0f, 0.0f, 1.0f, 1.0f, 3.0f, 2.0f,
                        3.0f, 3.0f, 8.0f, 5.0f, 12.0f, 7.0f,
                        3.0f, 3.0f, 7.0f, 4.0f, 9.0f, 5.0f,
                        9.0f, 9.0f, 20.0f, 11.0f, 24.0f, 13.0f,
                        6.0f, 6.0f, 13.0f, 7.0f, 15.0f, 8.0f};
  vector<int64_t> Y_shape = {1, 2, 6, 6};

  TestConvTransposeOp(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape,
                      OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider}); //Accuracy Mismatch on OpenVINO-EP
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(ConvTransposeTest, SharedPrepackedWeights) {
  OpTester test("ConvTranspose", 11);
  test.AddAttribute("kernel_shape", vector<int64_t>{3, 3});
  test.AddAttribute("group", static_cast<int64_t>(2));
  test.AddAttribute("pads", vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("output_shape", vector<int64_t>{1, 6, 4, 4});

  int image_size = 4 * 4;
  int input_channels = 3 * 2;
  int output_channels = 3;
  std::vector<float> X;
  for (int i = 0; i < input_channels * image_size; i++)
    X.push_back(1.0f);
  test.AddInput<float>("X", {1, 6, 4, 4}, X, false);

  std::vector<float> W;
  int kernel_size = output_channels * input_channels * 3 * 3;
  for (int i = 0; i < kernel_size; i++)
    W.push_back(1.0f);
  test.AddInput<float>("W", {6, 3, 3, 3}, W, true);  // Trigger pre-packing

  auto expected_vals = {
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,  // duplicate below
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      18.0f,
      27.0f,
      27.0f,
      18.0f,
      12.0f,
      18.0f,
      18.0f,
      12.0f,
  };
  test.AddOutput<float>("Y", {1, 6, 4, 4}, expected_vals);

  auto p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<float>(), TensorShape({6, 3, 3, 3}),
                                           W.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue w;

  w.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(),
         DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  // Set up W as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("W", &w), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime

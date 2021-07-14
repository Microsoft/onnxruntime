// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "embed_layer_norm_test_vectors.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr float kEpsilon =  1e-12f;

class OpData {
 public:
  explicit OpData(
      const int64_t batch_size,
      const int64_t sequence_length,
      const int64_t hidden_size,
      const int64_t bias_size,
      const std::vector<float>& input_data,
      const std::vector<float>& skip_data,
      const std::vector<float>& gamma_data,
      const std::vector<float>& beta_data,
      const std::vector<float>& bias_data,
      const std::vector<float>& matmul_1_b_data,
      const std::vector<float>& bias_gelu_bias_data,
      const std::vector<float>& matmul_2_b_data,
      const std::vector<float>& output_data)
      : batch_size(batch_size)
      , sequence_length(sequence_length)
      , hidden_size(hidden_size)
      , bias_size(bias_size)
      , input_data(input_data)
      , skip_data(skip_data)
      , gamma_data(gamma_data)
      , beta_data(beta_data)
      , bias_data(bias_data)
      , matmul_1_b_data(matmul_1_b_data)
      , bias_gelu_bias_data(bias_gelu_bias_data)
      , matmul_2_b_data(matmul_2_b_data)
      , output_data(output_data) {}

  OpData() = delete;
  OpData(const OpData&) = delete;

  const int64_t batch_size;
  const int64_t sequence_length;
  const int64_t hidden_size;
  const int64_t bias_size;
  const std::vector<float>& input_data;
  const std::vector<float>& skip_data;
  const std::vector<float>& gamma_data;
  const std::vector<float>& beta_data;
  const std::vector<float>& bias_data;
  const std::vector<float>& matmul_1_b_data;
  const std::vector<float>& bias_gelu_bias_data;
  const std::vector<float>& matmul_2_b_data;
  const std::vector<float>& output_data;
};

void TestInvokeOp(const OpData& data) {
  std::vector<int64_t> input_dims =
      {data.batch_size, data.sequence_length, data.hidden_size};
  std::vector<int64_t> skip_dims =
      {data.batch_size, data.sequence_length, data.hidden_size};
  std::vector<int64_t> gamma_dims = {data.hidden_size};
  std::vector<int64_t> beta_dims = {data.hidden_size};
  std::vector<int64_t> bias_dims = {data.hidden_size};
  std::vector<int64_t> matmul_1_b_dims = {data.hidden_size, data.bias_size};
  std::vector<int64_t> bias_gelu_bias_dims = {data.bias_size};
  std::vector<int64_t> matmul_2_b_dims = {data.hidden_size, data.bias_size};
  std::vector<int64_t> output_dims =
      {data.batch_size, data.sequence_length, data.hidden_size};

  OpTester tester("EmbedLayerNormBiasGelu", 1, onnxruntime::kMSDomain);

  tester.AddInput<float>("input", input_dims, data.input_data);
  tester.AddInput<float>("skip", skip_dims, data.skip_data);
  tester.AddInput<float>(
      "gamma", gamma_dims, data.gamma_data, /*is_initializer=*/true);
  tester.AddInput<float>(
      "beta", beta_dims, data.beta_data, /*is_initializer=*/true);
  tester.AddInput<float>(
      "bias", bias_dims, data.bias_data, /*is_initializer=*/true);
  tester.AddInput<float>("matmul_1_b",
                         matmul_1_b_dims,
                         data.matmul_1_b_data,
                         /*is_initializer=*/true);
  tester.AddInput<float>("bias_gelu_bias",
                         bias_gelu_bias_dims,
                         data.bias_gelu_bias_data,
                         /*is_initializer=*/true);
  tester.AddInput<float>("matmul_2_b",
                         matmul_2_b_dims,
                         data.matmul_2_b_data,
                         /*is_initializer=*/true);
  tester.AddOutput<float>("output", output_dims, data.output_data);
  tester.AddAttribute("epsilon", kEpsilon);

  tester.Run();
}

}  // namespace

TEST(EmbedLayerNormBiasGelu, ShouldWork) {
  const int64_t batch_size = 1;
  const int64_t sequence_length = 8;
  const int64_t hidden_size = 4;
  const int64_t bias_size = 16;

  // clang-format off
  const std::vector<float> input_data = {
      0.877292f, 0.686702f, 0.26983f, 0.410746f,
      0.299725f, 0.314675f, 0.233526f, 0.0724602f,
      0.56409f, 0.30251f, 0.94781f, 0.706121f,
      0.0640292f, 0.301493f, 0.964735f, 0.636928f,
      0.608366f, 0.622913f, 0.395354f, 0.40979f,
      0.427133f, 0.920674f, 0.428131f, 0.626119f,
      0.73837f, 0.480352f, 0.421635f, 0.287868f,
      0.451956f, 0.167709f, 0.235804f, 0.873725f,
  };

  const std::vector<float> skip_data = {
      0.548171f, 0.993236f, 0.214446f, 0.359704f,
      0.686656f, 0.0831026f, 0.236436f, 0.434125f,
      0.685505f, 0.906413f, 0.547314f, 0.105903f,
      0.689571f, 0.0312862f, 0.696121f, 0.51158f,
      0.0757117f, 0.536431f, 0.68158f, 0.566632f,
      0.381293f, 0.823399f, 0.223761f, 0.861105f,
      0.5068f, 0.693589f, 0.831246f, 0.681481f,
      0.0792981f, 0.234749f, 0.920688f, 0.0283016f,
  };

  const std::vector<float> gamma_data = {
      0.785711f, 0.621543f, 0.696568f, 0.155369f,
  };

  const std::vector<float> beta_data = {
      0.705561f, 0.717018f, 0.881026f, 0.821076f,
  };

  const std::vector<float> bias_data = {
      0.213908f, 0.723697f, 0.746659f, 0.299502f,
  };

  const std::vector<float> matmul_1_b_data = {
      0.694677f, 0.808467f, 0.408159f, 0.320281f,
      0.507119f, 0.250854f, 0.138410f, 0.160306f,
      0.656907f, 0.199202f, 0.160970f, 0.773275f,
      0.247220f, 0.751201f, 0.326223f, 0.991398f,
      0.750411f, 0.991848f, 0.508204f, 0.493569f,
      0.449371f, 0.931187f, 0.858229f, 0.271200f,
      0.359458f, 0.121867f, 0.562401f, 0.651273f,
      0.785968f, 0.396731f, 0.752781f, 0.216864f,
      0.039308f, 0.008273f, 0.014030f, 0.121503f,
      0.796551f, 0.663447f, 0.492519f, 0.047497f,
      0.000834f, 0.095889f, 0.720413f, 0.076311f,
      0.777774f, 0.559522f, 0.883951f, 0.741611f,
      0.201977f, 0.458355f, 0.010255f, 0.241033f,
      0.469569f, 0.827956f, 0.375958f, 0.580942f,
      0.952894f, 0.154101f, 0.398278f, 0.126517f,
      0.063309f, 0.369749f, 0.837581f, 0.664672f,
  };

  const std::vector<float> bias_gelu_bias_data = {
      0.483700f, 0.546866f, 0.855935f, 0.337103f,
      0.286496f, 0.557222f, 0.830892f, 0.660373f,
      0.201642f, 0.665974f, 0.581277f, 0.110132f,
      0.193586f, 0.335197f, 0.885712f, 0.331057f,
  };

  const std::vector<float> matmul_2_b_data = {
      0.258151f, 0.565847f, 0.022721f, 0.632483f,
      0.806391f, 0.731046f, 0.767692f, 0.682241f,
      0.443943f, 0.283666f, 0.864421f, 0.957503f,
      0.311825f, 0.566500f, 0.263296f, 0.300785f,
      0.313493f, 0.008240f, 0.072697f, 0.066978f,
      0.885168f, 0.695196f, 0.718143f, 0.844434f,
      0.854290f, 0.919375f, 0.322758f, 0.332899f,
      0.700420f, 0.166575f, 0.179450f, 0.181747f,
      0.064637f, 0.016411f, 0.850099f, 0.430959f,
      0.478824f, 0.207435f, 0.281102f, 0.192032f,
      0.205712f, 0.939199f, 0.112358f, 0.720338f,
      0.813306f, 0.218414f, 0.314841f, 0.915983f,
      0.440634f, 0.270116f, 0.899378f, 0.259115f,
      0.868253f, 0.113262f, 0.703957f, 0.277704f,
      0.126201f, 0.321887f, 0.362347f, 0.781157f,
      0.166295f, 0.744470f, 0.210401f, 0.175204f,
  };

  const std::vector<float> output_data = {
      0.931759f, 0.338765f, -0.131476f, 1.25103f,
      0.998018f, 0.271481f, -0.0677575f, 1.2027f,
      0.609153f, 0.311766f, -0.0742768f, 1.39308f,
      0.912295f, 0.453042f, -0.203592f, 1.14993f,
      0.973502f, 0.484089f, -0.217615f, 1.05523f,
      0.822251f, 0.49487f, -0.21649f, 1.13241f,
      0.869146f, 0.611431f, -0.232114f, 0.797104f,
      0.148023f, 0.656236f, -0.182884f, 1.04682f,
  };
  // clang-format on

  OpData data(batch_size,
              sequence_length,
              hidden_size,
              bias_size,
              input_data,
              skip_data,
              gamma_data,
              beta_data,
              bias_data,
              matmul_1_b_data,
              bias_gelu_bias_data,
              matmul_2_b_data,
              output_data);

  TestInvokeOp(data);
}

}  // namespace test
}  // namespace onnxruntime

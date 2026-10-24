// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <type_traits>

#include "gtest/gtest.h"

#include "core/common/inlined_containers.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/test_environment.h"

#if !defined(DISABLE_CONTRIB_OPS)

namespace onnxruntime {
namespace test {
namespace {

template <typename T, typename V>
void RunLayerNormTest(bool simplified, bool has_bias) {
  // Cover scalar, vec2, vec4, multiple elements per thread, and split normalization.
  for (const auto& dims : {TensorShapeVector{2, 3}, {2, 6}, {2, 128}, {2, 260}, {2, 1024}, {1, 1024}}) {
    SCOPED_TRACE(MakeString("shape=", TensorShape(dims), ", simplified=", simplified, ", bias=", has_bias));
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (!webgpu_ep) {
      GTEST_SKIP() << "WebGPU execution provider is not available.";
    }

    constexpr float epsilon = 1e-5f;
    const int64_t norm_count = dims[0];
    const int64_t norm_size = dims[1];
    InlinedVector<T> input;
    InlinedVector<V> scale;
    InlinedVector<V> bias;
    InlinedVector<V> expected;
    InlinedVector<float> mean;
    InlinedVector<float> inv_std_dev;
    input.reserve(norm_count * norm_size);
    expected.reserve(norm_count * norm_size);
    scale.reserve(norm_size);
    bias.reserve(norm_size);
    mean.reserve(norm_count);
    inv_std_dev.reserve(norm_count);

    for (int64_t j = 0; j < norm_size; ++j) {
      scale.emplace_back(0.5f * static_cast<float>(1 + j % 3));
      bias.emplace_back(0.125f * static_cast<float>(j % 5 - 2));
    }

    for (int64_t n = 0; n < norm_count; ++n) {
      double sum = 0.0;
      for (int64_t j = 0; j < norm_size; ++j) {
        input.emplace_back(0.25f * static_cast<float>(j % 17 - 8) + 0.5f * static_cast<float>(n));
        sum += static_cast<float>(input.back());
      }
      const double row_mean = simplified ? 0.0 : sum / norm_size;
      double squared_deviation = 0.0;
      for (int64_t j = 0; j < norm_size; ++j) {
        const double deviation = static_cast<float>(input[n * norm_size + j]) - row_mean;
        squared_deviation += deviation * deviation;
      }
      const double row_inv_std_dev = 1.0 / std::sqrt(squared_deviation / norm_size + epsilon);
      mean.push_back(static_cast<float>(row_mean));
      inv_std_dev.push_back(static_cast<float>(row_inv_std_dev));
      for (int64_t j = 0; j < norm_size; ++j) {
        const V normalized(static_cast<float>((static_cast<float>(input[n * norm_size + j]) - row_mean) * row_inv_std_dev));
        expected.emplace_back(static_cast<float>(normalized) * static_cast<float>(scale[j]) +
                              (has_bias ? static_cast<float>(bias[j]) : 0.0f));
      }
    }

    OpTester test(simplified ? "SimplifiedLayerNormalization" : "LayerNormalization", 15);
    test.AddAttribute("epsilon", epsilon);
    test.AddInput<T>("X", dims, input.data(), input.size());
    test.AddInput<V>("Scale", {norm_size}, scale.data(), scale.size(), true);
    if (has_bias) {
      test.AddInput<V>("B", {norm_size}, bias.data(), bias.size(), true);
    }
    test.AddOutput<V>("Y", dims, expected.data(), expected.size());
    constexpr float tolerance = std::is_same_v<V, MLFloat16> ? 0.002f : 1e-5f;
    test.SetOutputAbsErr("Y", tolerance);
    test.SetOutputRelErr("Y", tolerance);
    if (!simplified) {
      test.AddOutput<float>("Mean", {norm_count, 1}, mean.data(), mean.size());
      test.AddOutput<float>("InvStdDev", {norm_count, 1}, inv_std_dev.data(), inv_std_dev.size());
    }
    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
    ASSERT_EQ(test.GetFetches().size(), simplified ? 1u : 3u);
  }
}

template <typename T, typename V>
void RunLayerNormTests() {
  RunLayerNormTest<T, V>(false, false);
  RunLayerNormTest<T, V>(false, true);
  RunLayerNormTest<T, V>(true, false);
}

template <typename T>
void RunLayerNormLargeOffsetTest(bool constant) {
  // Include the issue's shape, scalar/vec2/vec4, and both normalization dispatch paths.
  for (const auto& dims : {TensorShapeVector{1, 16}, {2, 3}, {2, 6}, {2, 16}, {2, 260}, {2, 1024}, {1, 512}, {1, 1024}}) {
    for (float input_offset : {1000.0f, 2396.814f, 10000.0f}) {
      for (bool has_bias : {false, true}) {
        SCOPED_TRACE(MakeString("shape=", TensorShape(dims), ", offset=", input_offset,
                                ", constant=", constant, ", bias=", has_bias));
        auto webgpu_ep = DefaultWebGpuExecutionProvider();
        if (!webgpu_ep) {
          GTEST_SKIP() << "WebGPU execution provider is not available.";
        }

        constexpr float epsilon = 1e-5f;
        const int64_t norm_count = dims[0];
        const int64_t norm_size = dims[1];
        InlinedVector<T> input, scale, bias, expected;
        InlinedVector<float> mean, inv_std_dev;
        input.reserve(norm_count * norm_size);
        expected.reserve(norm_count * norm_size);
        scale.reserve(norm_size);
        bias.reserve(norm_size);
        mean.reserve(norm_count);
        inv_std_dev.reserve(norm_count);
        for (int64_t j = 0; j < norm_size; ++j) {
          scale.emplace_back(0.5f * static_cast<float>(1 + j % 3));
          bias.emplace_back(0.125f * static_cast<float>(j % 5 - 2));
        }
        for (int64_t n = 0; n < norm_count; ++n) {
          double sum = 0.0;
          for (int64_t j = 0; j < norm_size; ++j) {
            input.emplace_back((n == 0 ? input_offset : -input_offset) +
                               (constant ? 0.0f : 0.5f * static_cast<float>(j % 17 - 8)));
            sum += static_cast<float>(input.back());
          }
          const double row_mean = sum / norm_size;
          double squared_deviation = 0.0;
          for (int64_t j = 0; j < norm_size; ++j) {
            const double deviation = static_cast<float>(input[n * norm_size + j]) - row_mean;
            squared_deviation += deviation * deviation;
          }
          const double row_inv_std_dev = 1.0 / std::sqrt(squared_deviation / norm_size + epsilon);
          mean.push_back(static_cast<float>(row_mean));
          inv_std_dev.push_back(static_cast<float>(row_inv_std_dev));
          for (int64_t j = 0; j < norm_size; ++j) {
            const T normalized(static_cast<float>((static_cast<float>(input[n * norm_size + j]) - row_mean) *
                                                  row_inv_std_dev));
            expected.emplace_back(static_cast<float>(normalized) * static_cast<float>(scale[j]) +
                                  (has_bias ? static_cast<float>(bias[j]) : 0.0f));
          }
        }

        OpTester test("LayerNormalization", 17);
        test.AddAttribute("epsilon", epsilon);
        test.AddInput<T>("X", dims, input.data(), input.size());
        test.AddInput<T>("Scale", {norm_size}, scale.data(), scale.size(), true);
        if (has_bias) {
          test.AddInput<T>("B", {norm_size}, bias.data(), bias.size(), true);
        }
        test.AddOutput<T>("Y", dims, expected.data(), expected.size());
        test.AddOutput<float>("Mean", {norm_count, 1}, mean.data(), mean.size());
        test.AddOutput<float>("InvStdDev", {norm_count, 1}, inv_std_dev.data(), inv_std_dev.size());
        // GPU arithmetic may round the large mean to FP32 before subtraction.
        const float tolerance = constant ? 1e-5f : (std::is_same_v<T, MLFloat16> ? 0.002f : 1e-4f);
        test.SetOutputAbsErr("Y", tolerance);
        test.SetOutputRelErr("Y", tolerance);
        SessionOptions session_options;
        ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
        test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
        ASSERT_EQ(test.GetFetches().size(), 3u);
      }
    }
  }
}

void RunLayerNormMixedSignMeanTest(int64_t norm_size) {
  for (int opset : {15, 17}) {
    for (float residual : {1.0f, -1.0f}) {
      for (bool has_inv_std_dev : {false, true}) {
        SCOPED_TRACE(MakeString("norm_size=", norm_size, ", opset=", opset, ", residual=", residual,
                                ", inv_std_dev=", has_inv_std_dev));
        auto webgpu_ep = DefaultWebGpuExecutionProvider();
        if (!webgpu_ep) {
          GTEST_SKIP() << "WebGPU execution provider is not available.";
        }

        constexpr float epsilon = 1e-5f;
        InlinedVector<float> input(static_cast<size_t>(norm_size), residual);
        input[0] = 1e8f;
        input[1] = -1e8f;
        const double mean = static_cast<double>(residual) * (norm_size - 2) / norm_size;
        double squared_deviation = 0.0;
        for (float value : input) {
          const double deviation = static_cast<double>(value) - mean;
          squared_deviation += deviation * deviation;
        }
        const double inv_std_dev = 1.0 / std::sqrt(squared_deviation / norm_size + epsilon);
        InlinedVector<float> expected;
        expected.reserve(norm_size);
        for (float value : input) {
          expected.push_back(static_cast<float>((static_cast<double>(value) - mean) * inv_std_dev));
        }

        OpTester test("LayerNormalization", opset);
        test.AddAttribute("epsilon", epsilon);
        test.AddInput<float>("X", {1, norm_size}, input.data(), input.size());
        const InlinedVector<float> scale(static_cast<size_t>(norm_size), 1.0f);
        test.AddInput<float>("Scale", {norm_size}, scale.data(), scale.size(), true);
        test.AddOutput<float>("Y", {1, norm_size}, expected.data(), expected.size());
        test.AddOutput<float>("Mean", {1, 1}, {static_cast<float>(mean)});
        test.SetOutputAbsErr("Mean", 1e-6f);
        test.SetOutputRelErr("Mean", 1e-6f);
        if (has_inv_std_dev) {
          test.AddOutput<float>("InvStdDev", {1, 1}, {static_cast<float>(inv_std_dev)});
        }
        SessionOptions session_options;
        ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
        test.Config(session_options).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
        ASSERT_EQ(test.GetFetches().size(), has_inv_std_dev ? 3u : 2u);
      }
    }
  }
}

template <typename V, typename U>
void CheckLayerNormKernelTypes(const KernelRegistry& kernel_registry, bool simplified, bool expect_supported) {
  SCOPED_TRACE(MakeString("simplified=", simplified, ", V=", DataTypeImpl::ToString(DataTypeImpl::GetTensorType<V>()),
                          ", U=", DataTypeImpl::ToString(DataTypeImpl::GetTensorType<U>())));
  // Kernel registration checks need a graph without an OpTester inference run.
  const auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("LayerNormKernelTypes", false, ModelMetaData{}, PathString{}, IOnnxRuntimeOpSchemaRegistryList{},
              {{kOnnxDomain, 15}}, {}, logger);
  auto& graph = model.MainGraph();
  auto& x = graph.GetOrCreateNodeArg("X", DataTypeImpl::GetTensorType<float>()->GetTypeProto());
  const auto* v_type = DataTypeImpl::GetTensorType<V>()->GetTypeProto();
  const auto* u_type = DataTypeImpl::GetTensorType<U>()->GetTypeProto();
  auto& scale = graph.GetOrCreateNodeArg("Scale", v_type);
  InlinedVector<NodeArg*> outputs{&graph.GetOrCreateNodeArg("Y", v_type)};
  if (!simplified) {
    outputs.push_back(&graph.GetOrCreateNodeArg("Mean", u_type));
  }
  outputs.push_back(&graph.GetOrCreateNodeArg("InvStdDev", u_type));
  auto& node = graph.AddNode("LayerNorm", simplified ? "SimplifiedLayerNormalization" : "LayerNormalization", "",
                             {&x, &scale}, outputs);
  node.AddAttribute("stash_type", static_cast<int64_t>(std::is_same_v<U, double>
                                                           ? ONNX_NAMESPACE::TensorProto_DataType_DOUBLE
                                                           : ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(graph.NumberOfNodes(), 1u);
  const KernelCreateInfo* kernel_create_info = nullptr;
  const auto status = kernel_registry.TryFindKernel(node, kWebGpuExecutionProvider,
                                                    OpSchemaKernelTypeStrResolver{}, logger, &kernel_create_info);
  EXPECT_EQ(status.IsOK(), expect_supported) << status.ErrorMessage();
  EXPECT_EQ(kernel_create_info != nullptr, expect_supported);
}

}  // namespace

TEST(LayerNorm_WebGPU, KernelTypeSupport) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
  auto kernel_registry = webgpu_ep->GetKernelRegistry();
  ASSERT_NE(kernel_registry, nullptr);
  for (bool simplified : {false, true}) {
    CheckLayerNormKernelTypes<float, float>(*kernel_registry, simplified, true);
    CheckLayerNormKernelTypes<MLFloat16, float>(*kernel_registry, simplified, true);
    CheckLayerNormKernelTypes<double, float>(*kernel_registry, simplified, false);
    CheckLayerNormKernelTypes<BFloat16, float>(*kernel_registry, simplified, false);
    CheckLayerNormKernelTypes<float, double>(*kernel_registry, simplified, false);
  }
}

TEST(LayerNorm_WebGPU, Float16InputFloat32Output) {
  RunLayerNormTests<MLFloat16, float>();
}

TEST(LayerNorm_WebGPU, Float32InputFloat16Output) {
  RunLayerNormTests<float, MLFloat16>();
}

TEST(LayerNorm_WebGPU, Float16) {
  RunLayerNormTests<MLFloat16, MLFloat16>();
}

TEST(LayerNorm_WebGPU, Float32) {
  RunLayerNormTests<float, float>();
}

TEST(LayerNorm_WebGPU, ConstantFloat32) {
  RunLayerNormLargeOffsetTest<float>(true);
}

TEST(LayerNorm_WebGPU, ConstantFloat16) {
  RunLayerNormLargeOffsetTest<MLFloat16>(true);
}

TEST(LayerNorm_WebGPU, NearlyConstantFloat32) {
  RunLayerNormLargeOffsetTest<float>(false);
}

TEST(LayerNorm_WebGPU, NearlyConstantFloat16) {
  RunLayerNormLargeOffsetTest<MLFloat16>(false);
}

TEST(LayerNorm_WebGPU, MixedSignMean) {
  RunLayerNormMixedSignMeanTest(4);
}

TEST(LayerNorm_WebGPU, MixedSignMeanSplit) {
  RunLayerNormMixedSignMeanTest(512);
  RunLayerNormMixedSignMeanTest(1024);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS)

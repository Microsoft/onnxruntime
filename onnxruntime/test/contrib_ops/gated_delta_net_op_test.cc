// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "core/common/logging/logging.h"
#include "core/graph/model.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "contrib_ops/cuda/bert/gated_delta_net_plan.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/test_environment.h"

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef USE_WEBGPU
#include "contrib_ops/webgpu/bert/gated_delta_net.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

namespace {

struct Geometry {
  int total_tokens;
  int batch;
  int hq;  // == hk
  int hv;
  int dk;
  int dv;
};

struct Options {
  std::string update_rule = "gated_delta";
  std::string gate_activation = "none";
  std::string beta_activation = "none";
  int qk_l2_norm = 0;
  int state_update_capacity = 0;
  int chunk_size = 0;
  float scale = 0.0f;
};

struct Inputs {
  std::vector<float> q, k, v, decay, beta, state0, a_log, dt_bias;
  std::vector<int32_t> cu_seqlens;  // empty means uniform packing
  std::vector<int32_t> capture_count;
  std::vector<int32_t> state_update_active;
};

void VerifyPackedQkvShapeInference(const std::vector<int64_t>& packed_qkv_shape,
                                   const std::vector<int64_t>& expected_output_shape,
                                   const std::string& expected_error = "",
                                   const std::vector<int64_t>& initial_state_shape = {2, 48, 128, 128},
                                   bool ragged = false) {
  std::unordered_map<std::string, int> domain_to_version = {{kMSDomain, 1}};
  Model model("gated_delta_net_packed_qkv", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto packed_qkv_type;
  packed_qkv_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (int64_t dim : packed_qkv_shape) {
    auto* dimension = packed_qkv_type.mutable_tensor_type()->mutable_shape()->add_dim();
    if (dim >= 0) dimension->set_dim_value(dim);
  }
  ONNX_NAMESPACE::TypeProto state_type;
  state_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t dim : initial_state_shape) {
    auto* dimension = state_type.mutable_tensor_type()->mutable_shape()->add_dim();
    if (dim >= 0) dimension->set_dim_value(dim);
  }
  ONNX_NAMESPACE::TypeProto cu_seqlens_type;
  cu_seqlens_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  cu_seqlens_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& packed_qkv = graph.GetOrCreateNodeArg("packed_qkv", &packed_qkv_type);
  auto& initial_state = graph.GetOrCreateNodeArg("initial_state", &state_type);
  auto& cu_seqlens = graph.GetOrCreateNodeArg("cu_seqlens", &cu_seqlens_type);
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  auto& output = graph.GetOrCreateNodeArg("output", nullptr);
  auto& final_state = graph.GetOrCreateNodeArg("final_state", nullptr);
  graph.AddNode("gdn", "GatedDeltaNet", "packed QKV shape inference",
                {&packed_qkv, &empty, &empty, ragged ? &cu_seqlens : &empty, &empty, &empty, &initial_state},
                {&output, &final_state}, nullptr, kMSDomain);
  graph.SetOutputs({&output, &final_state});

  if (!expected_error.empty()) {
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(graph.Resolve(), expected_error);
    return;
  }
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_EQ(output.Shape()->dim_size(), static_cast<int>(expected_output_shape.size()));
  for (int i = 0; i < output.Shape()->dim_size(); ++i) {
    const int64_t expected_dimension = expected_output_shape[static_cast<size_t>(i)];
    ASSERT_EQ(output.Shape()->dim(i).has_dim_value(), expected_dimension >= 0);
    if (expected_dimension >= 0) {
      EXPECT_EQ(output.Shape()->dim(i).dim_value(), expected_dimension);
    }
  }
  ASSERT_EQ(final_state.Shape()->dim_size(), state_type.tensor_type().shape().dim_size());
  for (int i = 0; i < final_state.Shape()->dim_size(); ++i) {
    const auto& expected_dimension = state_type.tensor_type().shape().dim(i);
    ASSERT_EQ(final_state.Shape()->dim(i).has_dim_value(), expected_dimension.has_dim_value());
    if (expected_dimension.has_dim_value()) {
      EXPECT_EQ(final_state.Shape()->dim(i).dim_value(), expected_dimension.dim_value());
    }
  }
}

TEST(GatedDeltaNetShapeInferenceTest, PackedQkv) {
  VerifyPackedQkvShapeInference({6, 10240}, {6, 48, 128});
  VerifyPackedQkvShapeInference({2, 5, 10240}, {2, 5, 48, 128});
}

TEST(GatedDeltaNetShapeInferenceTest, PackedQkvRejectsNonDivisibleUniformBatch) {
  VerifyPackedQkvShapeInference({5, 10240}, {}, "must be divisible by batch");
}

TEST(GatedDeltaNetShapeInferenceTest, PackedQkvSymbolicAndRaggedShapes) {
  VerifyPackedQkvShapeInference({6, -1}, {6, 48, 128});
  VerifyPackedQkvShapeInference({-1, 10240}, {-1, 48, 128});
  VerifyPackedQkvShapeInference({6, 10240}, {6, -1, 128}, "", {2, -1, 128, 128});
  VerifyPackedQkvShapeInference({5, 10240}, {5, 48, 128}, "", {2, 48, 128, 128}, true);
}

TEST(GatedDeltaNetShapeInferenceTest, PackedQkvRejectsInvalidDimensions) {
  VerifyPackedQkvShapeInference({6, 10241}, {}, "packed QKV last dimension");
  VerifyPackedQkvShapeInference({6, 6144}, {}, "packed QKV last dimension");
  VerifyPackedQkvShapeInference({6, 7936}, {}, "must be a positive multiple");
  VerifyPackedQkvShapeInference({6, 10240}, {}, "head counts and head sizes must be positive", {2, 48, 128, 0});
  VerifyPackedQkvShapeInference({6, 10240}, {}, "dimensions overflow int64",
                                {2, std::numeric_limits<int64_t>::max(), 128, 128});
}

TEST(GatedDeltaNetShapeInferenceTest, ShortOptionalInputList) {
  std::unordered_map<std::string, int> domain_to_version = {{kMSDomain, 1}};
  Model model("gated_delta_net_short_inputs", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  ONNX_NAMESPACE::TypeProto query_type;
  query_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  query_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(5);
  query_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(10240);
  auto& query = graph.GetOrCreateNodeArg("query", &query_type);
  auto& output = graph.GetOrCreateNodeArg("output", nullptr);
  graph.AddNode("gdn", "GatedDeltaNet", "short packed QKV inputs",
                {&query}, {&output}, nullptr, kMSDomain);
  graph.SetOutputs({&output});

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(graph.Resolve(), "initial_state is required for packed QKV");
}

TEST(GatedDeltaNetShapeInferenceTest, UnknownQueryShapeRequiresPairedKeyValue) {
  for (bool key_only : {true, false}) {
    std::unordered_map<std::string, int> domain_to_version = {{kMSDomain, 1}};
    Model model("gated_delta_net_unpaired_inputs", false, ModelMetaData(), PathString(),
                IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    ONNX_NAMESPACE::TypeProto input_type;
    input_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    auto& query = graph.GetOrCreateNodeArg("query", &input_type);
    auto& supplied = graph.GetOrCreateNodeArg("supplied", &input_type);
    auto& empty = graph.GetOrCreateNodeArg("", nullptr);
    auto& output = graph.GetOrCreateNodeArg("output", nullptr);
    graph.AddNode("gdn", "GatedDeltaNet", "unpaired key or value",
                  key_only ? std::vector<NodeArg*>{&query, &supplied}
                           : std::vector<NodeArg*>{&query, &empty, &supplied},
                  {&output}, nullptr, kMSDomain);
    graph.SetOutputs({&output});

    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(graph.Resolve(), "key and value must both be present");
  }
}

Inputs MakeInputs(const Geometry& g, uint32_t seed, bool with_state = true) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  Inputs in;
  in.q.resize(static_cast<size_t>(g.total_tokens) * g.hq * g.dk);
  in.k.resize(static_cast<size_t>(g.total_tokens) * g.hq * g.dk);
  in.v.resize(static_cast<size_t>(g.total_tokens) * g.hv * g.dv);
  in.decay.resize(static_cast<size_t>(g.total_tokens) * g.hv);
  in.beta.resize(static_cast<size_t>(g.total_tokens) * g.hv);
  in.a_log.resize(g.hv);
  in.dt_bias.resize(g.hv);
  for (auto& x : in.q) x = u(rng) * 0.5f;
  for (auto& x : in.k) x = u(rng) * 0.5f;
  // The delta family needs L2-normalized keys: without them (I + M) is arbitrarily
  // ill-conditioned and the recurrence diverges. Real callers either normalize upstream or
  // set qk_l2_norm=1.
  for (int t = 0; t < g.total_tokens; ++t) {
    for (int h = 0; h < g.hq; ++h) {
      float* kp = in.k.data() + (static_cast<size_t>(t) * g.hq + h) * g.dk;
      float n = 0.0f;
      for (int i = 0; i < g.dk; ++i) n += kp[i] * kp[i];
      n = 1.0f / std::sqrt(n + 1e-12f);
      for (int i = 0; i < g.dk; ++i) kp[i] *= n;
    }
  }
  for (auto& x : in.v) x = u(rng) * 0.5f;
  // Decay must be <= 0 in log space so the recurrence contracts.
  for (auto& x : in.decay) x = -0.05f * (u(rng) + 1.2f);
  for (auto& x : in.beta) x = 0.5f + 0.25f * u(rng);
  for (auto& x : in.a_log) x = 0.1f * u(rng);
  for (auto& x : in.dt_bias) x = 0.1f * u(rng);
  if (with_state) {
    in.state0.resize(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk);
    for (auto& x : in.state0) x = 0.05f * u(rng);
  }
  return in;
}

float SoftPlus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }
float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// float64 sequential reference. State is V-major [B, Hv, V, K] on the boundary and
// [K][V] internally, matching the operator contract.
void Reference(const Geometry& g, const Options& o, const Inputs& in, std::vector<float>* out,
               std::vector<float>* final_state, std::vector<float>* state_update = nullptr) {
  const bool gated = o.update_rule == "gated" || o.update_rule == "gated_delta";
  const bool delta = o.update_rule == "delta" || o.update_rule == "gated_delta";
  const float scale = o.scale != 0.0f ? o.scale : 1.0f / std::sqrt(static_cast<float>(g.dk));
  const int out_heads = std::max(g.hq, g.hv);

  out->assign(static_cast<size_t>(g.total_tokens) * out_heads * g.dv, 0.0f);
  final_state->assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);
  const int64_t decay_elements = static_cast<int64_t>(o.state_update_capacity) * g.hv;
  const int64_t key_elements = static_cast<int64_t>(o.state_update_capacity) * g.hq * g.dk;
  const int64_t row_width = decay_elements + key_elements +
                            static_cast<int64_t>(o.state_update_capacity) * g.hv * g.dv;
  if (state_update != nullptr) {
    state_update->assign(static_cast<size_t>(g.batch) * row_width, 0.0f);
  }

  std::vector<int32_t> cu = in.cu_seqlens;
  if (cu.empty()) {
    cu.resize(g.batch + 1);
    for (int b = 0; b <= g.batch; ++b) cu[b] = b * (g.total_tokens / g.batch);
  }

  for (int b = 0; b < g.batch; ++b) {
    for (int hv = 0; hv < g.hv; ++hv) {
      const int hq = hv * g.hq / g.hv;
      std::vector<double> S(static_cast<size_t>(g.dk) * g.dv, 0.0);
      if (!in.state0.empty()) {
        for (int r = 0; r < g.dk; ++r) {
          for (int c = 0; c < g.dv; ++c) {
            const size_t si = ((static_cast<size_t>(b) * g.hv + hv) * g.dv + c) * g.dk + r;
            S[static_cast<size_t>(r) * g.dv + c] = in.state0[si];
          }
        }
      }
      for (int t = cu[b]; t < cu[b + 1]; ++t) {
        const int local_token = t - cu[b];
        std::vector<double> qv(g.dk), kv(g.dk);
        for (int i = 0; i < g.dk; ++i) {
          const float q = in.q[(static_cast<size_t>(t) * g.hq + hq) * g.dk + i];
          const float k = in.k[(static_cast<size_t>(t) * g.hq + hq) * g.dk + i];
          qv[i] = q;
          kv[i] = k;
        }
        if (o.qk_l2_norm) {
          double nq = 0.0, nk = 0.0;
          for (int i = 0; i < g.dk; ++i) {
            nq += qv[i] * qv[i];
            nk += kv[i] * kv[i];
          }
          nq = 1.0 / std::sqrt(nq + 1e-12);
          nk = 1.0 / std::sqrt(nk + 1e-12);
          for (int i = 0; i < g.dk; ++i) {
            qv[i] *= nq;
            kv[i] *= nk;
          }
        }

        const size_t gi = static_cast<size_t>(t) * g.hv + hv;
        double decay = 1.0;
        if (gated) {
          float raw = in.decay[gi];
          if (o.gate_activation == "qwen") {
            raw = -std::exp(in.a_log[hv]) * SoftPlus(raw + in.dt_bias[hv]);
          }
          decay = std::exp(static_cast<double>(raw));
        }
        double beta = 1.0;
        if (delta && !in.beta.empty()) {
          float raw = in.beta[gi];
          if (o.beta_activation == "sigmoid") {
            raw = Sigmoid(raw);
          }
          beta = raw;
        }

        for (auto& s : S) s *= decay;

        std::vector<double> delta_v(g.dv);
        for (int c = 0; c < g.dv; ++c) {
          double acc = 0.0;
          if (delta) {
            for (int r = 0; r < g.dk; ++r) acc += S[static_cast<size_t>(r) * g.dv + c] * kv[r];
          }
          const double vv = in.v[(static_cast<size_t>(t) * g.hv + hv) * g.dv + c];
          delta_v[c] = beta * (vv - acc);
        }
        const int capture_count = in.capture_count.empty()
                                      ? 0
                                      : std::clamp(in.capture_count[b], 0,
                                                   std::min(o.state_update_capacity, cu[b + 1] - cu[b]));
        if (state_update != nullptr && local_token < capture_count) {
          const int64_t row_base = static_cast<int64_t>(b) * row_width;
          (*state_update)[row_base + static_cast<int64_t>(local_token) * g.hv + hv] =
              static_cast<float>(decay);
          const int first_value_head_for_query = hq * g.hv / g.hq;
          if (hv == first_value_head_for_query) {
            for (int r = 0; r < g.dk; ++r) {
              (*state_update)[row_base + decay_elements +
                              (static_cast<int64_t>(local_token) * g.hq + hq) * g.dk + r] =
                  static_cast<float>(kv[r]);
            }
          }
          for (int c = 0; c < g.dv; ++c) {
            (*state_update)[row_base + decay_elements + key_elements +
                            (static_cast<int64_t>(local_token) * g.hv + hv) * g.dv + c] =
                static_cast<float>(delta_v[c]);
          }
        }
        for (int r = 0; r < g.dk; ++r) {
          for (int c = 0; c < g.dv; ++c) {
            S[static_cast<size_t>(r) * g.dv + c] += kv[r] * delta_v[c];
          }
        }
        for (int c = 0; c < g.dv; ++c) {
          double acc = 0.0;
          for (int r = 0; r < g.dk; ++r) acc += S[static_cast<size_t>(r) * g.dv + c] * qv[r];
          (*out)[(static_cast<size_t>(t) * out_heads + hv) * g.dv + c] =
              static_cast<float>(scale * acc);
        }
      }
      for (int r = 0; r < g.dk; ++r) {
        for (int c = 0; c < g.dv; ++c) {
          const size_t si = ((static_cast<size_t>(b) * g.hv + hv) * g.dv + c) * g.dk + r;
          (*final_state)[si] = static_cast<float>(S[static_cast<size_t>(r) * g.dv + c]);
        }
      }
    }
  }
}

void AddCommonAttrs(OpTester& t, const Options& o) {
  t.AddAttribute("update_rule", o.update_rule);
  t.AddAttribute("gate_activation", o.gate_activation);
  t.AddAttribute("beta_activation", o.beta_activation);
  t.AddAttribute("qk_l2_norm", static_cast<int64_t>(o.qk_l2_norm));
  if (o.state_update_capacity > 0) {
    t.AddAttribute("state_update_capacity", static_cast<int64_t>(o.state_update_capacity));
  }
  if (o.chunk_size > 0) t.AddAttribute("chunk_size", static_cast<int64_t>(o.chunk_size));
  if (o.scale != 0.0f) t.AddAttribute("scale", o.scale);
}

template <typename T>
std::vector<T> ToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, MLFloat16>) {
    return ToFloat16(data);
  } else if constexpr (std::is_same_v<T, BFloat16>) {
    return ToBFloat16(data);
  } else {
    return data;
  }
}

// Snaps floats to the grid of the operator's input type, so the reference sees the exact values
// the kernel reads. Without it the tolerance has to absorb input rounding and stops measuring the
// kernel: bfloat16 needs 4e-2 against unrounded inputs where 6e-3 suffices against rounded ones.
template <typename T>
std::vector<float> RoundToTensorType(const std::vector<float>& data) {
  if constexpr (std::is_same_v<T, float>) {
    return data;
  } else {
    const std::vector<T> narrowed = ToTensorType<T>(data);
    std::vector<float> rounded(data.size());
    for (size_t i = 0; i < data.size(); ++i) rounded[i] = narrowed[i].ToFloat();
    return rounded;
  }
}

// Runs the operator and compares against the float64 reference.
// With rank4 the leading token axis is spelled [batch, sequence] instead of [total_tokens];
// the buffers are byte-identical, only the declared shapes differ.
template <typename T>
void RunTypedCase(const Geometry& g, const Options& o, const Inputs& in_raw, float out_tol,
                  float state_tol, bool rank4 = false, std::vector<OrtValue>* fetches = nullptr,
                  bool use_webgpu = false, bool omit_final_state = false,
                  [[maybe_unused]] const ConfigOptions* webgpu_config = nullptr,
                  [[maybe_unused]] uint64_t test_max_storage_buffer_binding_size = 0,
                  bool packed_qkv = false, bool packed_gates = false) {
  Inputs in = in_raw;
  in.q = RoundToTensorType<T>(in_raw.q);
  in.k = RoundToTensorType<T>(in_raw.k);
  in.v = RoundToTensorType<T>(in_raw.v);
  if (packed_gates) {
    in.decay = RoundToTensorType<T>(in_raw.decay);
    in.beta = RoundToTensorType<T>(in_raw.beta);
  }

  std::vector<float> ref_out, ref_state, ref_state_update;
  Reference(g, o, in, &ref_out, &ref_state, use_webgpu ? &ref_state_update : nullptr);

  std::unique_ptr<IExecutionProvider> webgpu_ep;
  if (use_webgpu) {
#ifdef USE_WEBGPU
    if (test_max_storage_buffer_binding_size != 0) {
      WebGpuExecutionProviderTestOptions test_options{};
      test_options.max_storage_buffer_binding_size = test_max_storage_buffer_binding_size;
      webgpu_ep = webgpu_config != nullptr
                      ? WebGpuExecutionProviderWithTestOptions(*webgpu_config, test_options)
                      : WebGpuExecutionProviderWithTestOptions(test_options);
    } else {
      webgpu_ep = webgpu_config != nullptr ? WebGpuExecutionProviderWithOptions(*webgpu_config)
                                           : DefaultWebGpuExecutionProvider();
    }
#else
    webgpu_ep = DefaultWebGpuExecutionProvider();
#endif
    if (webgpu_ep == nullptr) {
      GTEST_SKIP() << "WebGPU execution provider is not available";
    }
  }

  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  AddCommonAttrs(test, o);

  const int out_heads = std::max(g.hq, g.hv);
  // Leading token axes shared by q/k/v, decay, beta and the output.
  std::vector<int64_t> lead{g.total_tokens};
  if (rank4) lead = {g.batch, g.total_tokens / g.batch};
  auto shaped = [&lead](std::initializer_list<int64_t> tail) {
    std::vector<int64_t> s(lead);
    s.insert(s.end(), tail);
    return s;
  };

  if (packed_qkv) {
    const size_t query_size = static_cast<size_t>(g.hq) * g.dk;
    const size_t key_size = query_size;
    const size_t value_size = static_cast<size_t>(g.hv) * g.dv;
    std::vector<float> packed;
    packed.reserve(static_cast<size_t>(g.total_tokens) * (query_size + key_size + value_size));
    for (int token = 0; token < g.total_tokens; ++token) {
      packed.insert(packed.end(), in.q.begin() + token * query_size, in.q.begin() + (token + 1) * query_size);
      packed.insert(packed.end(), in.k.begin() + token * key_size, in.k.begin() + (token + 1) * key_size);
      packed.insert(packed.end(), in.v.begin() + token * value_size, in.v.begin() + (token + 1) * value_size);
    }
    test.AddInput<T>("query", shaped({static_cast<int64_t>(query_size + key_size + value_size)}),
                     ToTensorType<T>(packed));
    test.AddOptionalInputEdge<T>();
    test.AddOptionalInputEdge<T>();
  } else {
    test.AddInput<T>("query", shaped({g.hq, g.dk}), ToTensorType<T>(in.q));
    test.AddInput<T>("key", shaped({g.hq, g.dk}), ToTensorType<T>(in.k));
    test.AddInput<T>("value", shaped({g.hv, g.dv}), ToTensorType<T>(in.v));
  }
  if (in.cu_seqlens.empty()) {
    test.AddOptionalInputEdge<int32_t>();
  } else {
    test.AddInput<int32_t>("cu_seqlens", {static_cast<int64_t>(in.cu_seqlens.size())},
                           in.cu_seqlens);
  }
  const bool needs_decay =
      o.update_rule == "gated" || o.update_rule == "gated_delta";
  const bool needs_beta =
      o.update_rule == "delta" || o.update_rule == "gated_delta";
  if (needs_decay && !packed_gates) {
    test.AddInput<float>("decay", shaped({g.hv}), in.decay);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  if (needs_beta && !packed_gates) {
    test.AddInput<float>("beta", shaped({g.hv}), in.beta);
  } else {
    test.AddOptionalInputEdge<float>();
  }
  if (in.state0.empty()) {
    test.AddOptionalInputEdge<float>();
  } else {
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  }
  if (o.gate_activation == "qwen") {
    test.AddInput<float>("a_log", {g.hv}, in.a_log);
    test.AddInput<float>("dt_bias", {g.hv}, in.dt_bias);
  } else {
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
  }
  if (o.state_update_capacity > 0) {
    test.AddInput<int32_t>("capture_count", {g.batch}, in.capture_count);
  } else if (!in.state_update_active.empty() || packed_gates) {
    test.AddOptionalInputEdge<int32_t>();
  }
  if (!in.state_update_active.empty()) {
    test.AddInput<int32_t>("state_update_active", {1}, in.state_update_active);
  } else if (packed_gates) {
    test.AddOptionalInputEdge<int32_t>();
  }
  if (packed_gates) {
    std::vector<float> projections;
    projections.reserve(static_cast<size_t>(g.total_tokens) * g.hv * 2);
    for (int token = 0; token < g.total_tokens; ++token) {
      const size_t base = static_cast<size_t>(token) * g.hv;
      projections.insert(projections.end(), in.decay.begin() + base, in.decay.begin() + base + g.hv);
      projections.insert(projections.end(), in.beta.begin() + base, in.beta.begin() + base + g.hv);
    }
    test.AddInput<T>("gate_projections", shaped({2 * g.hv}), ToTensorType<T>(projections));
  }

  test.AddOutput<T>("output", shaped({out_heads, g.dv}), ToTensorType<T>(ref_out),
                    false, out_tol, out_tol);
  if (omit_final_state) {
    test.AddOptionalOutputEdge<float>();
  } else {
    test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk}, ref_state, false, state_tol,
                          state_tol);
  }
  const bool state_update_enabled = in.state_update_active.empty() || in.state_update_active[0] != 0;
  if (o.state_update_capacity > 0) {
    const int64_t width = static_cast<int64_t>(o.state_update_capacity) *
                          (g.hv + g.hq * g.dk + g.hv * g.dv);
    test.AddOutput<float>("state_update", {g.batch, width},
                          use_webgpu && state_update_enabled
                              ? ref_state_update
                              : std::vector<float>(static_cast<size_t>(g.batch) * width, 0.0f),
                          false, use_webgpu ? state_tol : 1e9f, use_webgpu ? state_tol : 1e9f);
  } else {
    test.AddOutput<float>("state_update", {g.batch, 0}, {});
  }

  if (use_webgpu && o.state_update_capacity > 0 && state_update_enabled) {
    test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& actual_outputs, const std::string&) {
      ASSERT_EQ(actual_outputs.size(), 3u);
      const Tensor& actual_output = actual_outputs[0].Get<Tensor>();
      const Tensor& actual_final_state = actual_outputs[1].Get<Tensor>();
      const Tensor& actual_state_update = actual_outputs[2].Get<Tensor>();
      const int64_t decay_elements = static_cast<int64_t>(o.state_update_capacity) * g.hv;
      const int64_t key_elements = static_cast<int64_t>(o.state_update_capacity) * g.hq * g.dk;
      const int64_t width = decay_elements + key_elements +
                            static_cast<int64_t>(o.state_update_capacity) * g.hv * g.dv;
      ASSERT_EQ(actual_output.Shape(), TensorShape(shaped({out_heads, g.dv})));
      ASSERT_EQ(actual_final_state.Shape(), TensorShape({g.batch, g.hv, g.dv, g.dk}));
      ASSERT_EQ(actual_state_update.Shape(), TensorShape({g.batch, width}));

      const T* actual_output_data = actual_output.Data<T>();
      const std::vector<T> expected_output = ToTensorType<T>(ref_out);
      for (size_t i = 0; i < expected_output.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) {
          EXPECT_NEAR(actual_output_data[i], expected_output[i], out_tol);
        } else {
          EXPECT_NEAR(actual_output_data[i].ToFloat(), expected_output[i].ToFloat(), out_tol);
        }
      }
      const float* actual_final_state_data = actual_final_state.Data<float>();
      for (size_t i = 0; i < ref_state.size(); ++i) {
        EXPECT_NEAR(actual_final_state_data[i], ref_state[i], state_tol);
      }

      const float* actual_data = actual_state_update.Data<float>();
      for (int b = 0; b < g.batch; ++b) {
        const int sequence_length = in.cu_seqlens.empty()
                                        ? g.total_tokens / g.batch
                                        : in.cu_seqlens[b + 1] - in.cu_seqlens[b];
        const int count = std::clamp(in.capture_count[b], 0,
                                     std::min(o.state_update_capacity, sequence_length));
        auto check_range = [&](int64_t offset, int64_t elements) {
          for (int64_t i = 0; i < elements; ++i) {
            const int64_t index = static_cast<int64_t>(b) * width + offset + i;
            EXPECT_NEAR(actual_data[index], ref_state_update[index], state_tol);
          }
        };
        check_range(0, static_cast<int64_t>(count) * g.hv);
        check_range(decay_elements, static_cast<int64_t>(count) * g.hq * g.dk);
        check_range(decay_elements + key_elements, static_cast<int64_t>(count) * g.hv * g.dv);
      }
    });
  }

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  if (use_webgpu) {
    eps.push_back(std::move(webgpu_ep));
  } else {
    eps.push_back(DefaultCudaExecutionProvider());
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  if (fetches != nullptr) *fetches = test.GetFetches();
}

void RunCase(const Geometry& g, const Options& o, const Inputs& in, float out_tol,
             float state_tol, bool rank4 = false, std::vector<OrtValue>* fetches = nullptr,
             bool use_webgpu = false) {
  RunTypedCase<MLFloat16>(g, o, in, out_tol, state_tol, rank4, fetches, use_webgpu);
}

constexpr int kDim = 128;  // the chunked engine is specialised for head_size 128

bool NeedSkipGatedDeltaNetTest() {
  return !HasCudaEnvironment(800);
}

bool NeedSkipGatedDeltaNetSplitTest() {
  if (NeedSkipGatedDeltaNetTest()) return true;

#ifdef USE_CUDA
  int device = 0;
  int max_shared_memory = 0;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&max_shared_memory, cudaDevAttrMaxSharedMemoryPerBlockOptin, device) != cudaSuccess) {
    return true;
  }

  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor desc{};
  desc.total_tokens = 64;
  desc.batch = 1;
  desc.num_heads_q = 1;
  desc.num_heads_k = 1;
  desc.num_heads_v = 2;
  desc.head_size_qk = kDim;
  desc.head_size_v = kDim;
  desc.io_type = gdn::IoType::kFloat16;
  desc.preferred_engine = gdn::Engine::kChunkedSplit;
  desc.sm_major = GetCudaArchitecture() / 100;
  return gdn::SelectPlan(desc, static_cast<size_t>(max_shared_memory)).engine !=
         gdn::Engine::kChunkedSplit;
#else
  return true;
#endif
}

bool NeedSkipGatedDeltaNetWebGpuTest() {
  return DefaultWebGpuExecutionProvider() == nullptr;
}

}  // namespace

TEST(GatedDeltaNetWebGpuTest, Rank4AllRulesAndInverseGqa) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{6, 2, 2, 6, 8, 5};  // Hv/Hq == 3
  for (const char* rule : {"linear", "gated", "delta", "gated_delta"}) {
    SCOPED_TRACE(rule);
    Options options;
    options.update_rule = rule;
    options.gate_activation = "qwen";
    options.beta_activation = "sigmoid";
    options.qk_l2_norm = 1;
    RunCase(g, options, MakeInputs(g, 201), 3e-3f, 3e-4f, /*rank4=*/true,
            /*fetches=*/nullptr, /*use_webgpu=*/true);
  }
}

TEST(GatedDeltaNetWebGpuTest, Rank3UniformFloat32) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{8, 2, 1, 3, 7, 4};
  RunTypedCase<float>(g, Options{}, MakeInputs(g, 211), 3e-4f, 3e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, RecurrentVectorizedValueIoAndSharedGate) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{7, 2, 1, 2, 8, 8};
  Inputs inputs = MakeInputs(g, 212, /*with_state=*/false);
  inputs.cu_seqlens = {0, 3, 7};
  Options options;
  options.update_rule = "gated";
  options.gate_activation = "qwen";
  RunTypedCase<MLFloat16>(g, options, inputs, 3e-3f, 3e-4f,
                          /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true,
                          /*omit_final_state=*/true);
}

TEST(GatedDeltaNetWebGpuTest, ParallelPrefillLinearUniformRank3AndRank4) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{128, 2, 2, 6, 8, 5};
  const Inputs inputs = MakeInputs(g, 213);
  Options options;
  options.update_rule = "linear";
  RunTypedCase<float>(g, options, inputs, 4e-4f, 4e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
  RunTypedCase<float>(g, options, inputs, 4e-4f, 4e-4f,
                      /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/true);
  RunTypedCase<float>(g, options, inputs, 4e-4f, 4e-4f,
                      /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/true);
}

TEST(GatedDeltaNetWebGpuTest, PackedQkvPrefillAndDecode) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  Geometry prefill{128, 2, 2, 6, 8, 5};
  Options prefill_options;
  prefill_options.update_rule = "linear";
  RunTypedCase<float>(prefill, prefill_options, MakeInputs(prefill, 215), 4e-4f, 4e-4f,
                      /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/false, /*webgpu_config=*/nullptr,
                      /*test_max_storage_buffer_binding_size=*/0, /*packed_qkv=*/true);

  Geometry decode{2, 2, 2, 6, 8, 5};
  RunTypedCase<float>(decode, Options{}, MakeInputs(decode, 216), 4e-4f, 4e-4f,
                      /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/false, /*webgpu_config=*/nullptr,
                      /*test_max_storage_buffer_binding_size=*/0, /*packed_qkv=*/true);
}

#ifdef USE_WEBGPU
TEST(GatedDeltaNetWebGpuTest, PackedQkvAndParamsWithSegmentedBacking) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  constexpr uint64_t max_binding_size = 256;
  constexpr uint32_t max_storage_buffers = 8;
  Geometry g{9, 2, 1, 1, 8, 4};
  Inputs inputs = MakeInputs(g, 228);
  inputs.cu_seqlens = {0, 4, 9};
  Options options;
  options.update_rule = "gated_delta";
  options.gate_activation = "qwen";

  const auto binding_count = [](uint64_t bytes) {
    return (bytes + max_binding_size - 1) / max_binding_size;
  };
  const uint64_t query_bytes = static_cast<uint64_t>(g.total_tokens) * g.hq * g.dk * sizeof(float);
  const uint64_t key_bytes = query_bytes;
  const uint64_t value_bytes = static_cast<uint64_t>(g.total_tokens) * g.hv * g.dv * sizeof(float);
  const uint64_t qkv_binding_count =
      binding_count(query_bytes) + binding_count(key_bytes) + binding_count(value_bytes);
  const uint64_t packed_qkv_binding_count = binding_count(query_bytes + key_bytes + value_bytes);
  constexpr uint64_t dynamic_param_binding_count = 4;
  const uint64_t packed_param_binding_count =
      binding_count(static_cast<uint64_t>(g.total_tokens) * g.hv * 2 * sizeof(float));
  constexpr uint64_t other_binding_count = 4;  // cu_seqlens, state, output, and final state.
  ASSERT_GT(qkv_binding_count + dynamic_param_binding_count + other_binding_count, max_storage_buffers);
  ASSERT_GT(qkv_binding_count + packed_param_binding_count + other_binding_count, max_storage_buffers);
  ASSERT_LE(packed_qkv_binding_count + packed_param_binding_count + other_binding_count, max_storage_buffers);
  ASSERT_GT(query_bytes + key_bytes + value_bytes, max_binding_size);
  const uint64_t key_view_offset = query_bytes / sizeof(float);
  const uint64_t value_view_offset = (query_bytes + key_bytes) / sizeof(float);
  ASSERT_GT(key_view_offset, 0u);
  ASSERT_GT(value_view_offset, key_view_offset);

  ConfigOptions config_options;
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kMaxStorageBuffersPerShaderStage, "8"));
  RunTypedCase<float>(g, options, inputs, 5e-4f, 5e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/false, &config_options, max_binding_size);
}

TEST(GatedDeltaNetWebGpuTest, SegmentedQueryAndKeyUseHelperIndexing) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }

  constexpr uint64_t max_binding_size = 256;
  // The small test-only binding limit forces Q and K into two storage-buffer
  // segments while keeping the test's memory footprint negligible.
  Geometry g{17, 1, 1, 1, 4, 1};
  Inputs inputs = MakeInputs(g, 227, /*with_state=*/false);
  inputs.cu_seqlens = {0, g.total_tokens};
  Options options;
  options.update_rule = "linear";
  RunTypedCase<float>(g, options, inputs, 5e-4f, 5e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/true, /*config_options=*/nullptr, max_binding_size);
}
#endif

TEST(GatedDeltaNetWebGpuTest, LongUniformNonLinearRulesUseRecurrentFallback) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{128, 2, 2, 6, 8, 5};
  const Inputs inputs = MakeInputs(g, 217);
  for (const char* rule : {"gated", "delta", "gated_delta"}) {
    SCOPED_TRACE(rule);
    Options options;
    options.update_rule = rule;
    RunTypedCase<float>(g, options, inputs, 4e-4f, 4e-4f,
                        /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/true);
  }
}

#ifdef USE_WEBGPU
TEST(GatedDeltaNetWebGpuPlanTest, RaggedChunkCapacityIsLinear) {
  using onnxruntime::contrib::webgpu::GatedDeltaNetChunkCapacity;

  EXPECT_EQ(GatedDeltaNetChunkCapacity(130, 3, 16, true), 11u);
  EXPECT_EQ(GatedDeltaNetChunkCapacity(130, 2, 16, false), 10u);
  for (uint64_t batch : {1u, 8u, 64u, 256u}) {
    const uint64_t capacity = GatedDeltaNetChunkCapacity(65 * batch, batch, 16, true);
    EXPECT_GE(capacity, 5 * batch);
    EXPECT_LE(capacity, 6 * batch);
    constexpr uint64_t chunk_elements = 16 * (3 * 128 + 128) + 16 * 16 + 1;
    const uint64_t workspace_bytes = (capacity * chunk_elements + 2 * 65 * batch * 128 +
                                      2 * 65 * batch) *
                                     sizeof(float);
    EXPECT_LE(workspace_bytes, 64ull << 20);
  }
}

TEST(GatedDeltaNetWebGpuPlanTest, ParallelPrefillWorkspaceIsBounded) {
  using onnxruntime::contrib::webgpu::SelectGatedDeltaNetParallelPrefillPlan;

  const auto short_plan = SelectGatedDeltaNetParallelPrefillPlan(1ull << 20, 4);
  const auto long_plan = SelectGatedDeltaNetParallelPrefillPlan(1ull << 20, 4096);
  ASSERT_TRUE(short_plan.has_value());
  ASSERT_TRUE(long_plan.has_value());
  EXPECT_LE(short_plan->workspace_bytes, 64ull << 20);
  EXPECT_LE(long_plan->workspace_bytes, 64ull << 20);
  EXPECT_LT(long_plan->chunks_per_pass, 4096u);
  EXPECT_LT(long_plan->workspace_bytes, 16 * short_plan->workspace_bytes);
  EXPECT_FALSE(SelectGatedDeltaNetParallelPrefillPlan(32ull << 20, 2).has_value());
  EXPECT_FALSE(SelectGatedDeltaNetParallelPrefillPlan(1ull << 20, 2, 8ull << 20).has_value());
}
#endif

TEST(GatedDeltaNetWebGpuTest, RaggedWithoutInitialState) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{7, 3, 1, 3, 8, 4};
  Inputs inputs = MakeInputs(g, 223, /*with_state=*/false);
  inputs.cu_seqlens = {0, 1, 5, 7};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.scale = 0.37f;
  RunTypedCase<float>(g, options, inputs, 3e-4f, 3e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, RaggedQwenWithInitialStateAndNonDivisibleDv) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{7, 3, 1, 3, 8, 5};
  Inputs inputs = MakeInputs(g, 225);
  inputs.cu_seqlens = {0, 1, 5, 7};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  RunTypedCase<float>(g, options, inputs, 3e-4f, 3e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, ChunkwiseQwenPrefillPartialChunk) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  for (const Geometry geometry : {Geometry{65, 1, 2, 4, 128, 128}, Geometry{65, 1, 16, 48, 128, 128}}) {
    SCOPED_TRACE(geometry.hv);
    Inputs inputs = MakeInputs(geometry, 231);
    inputs.cu_seqlens = {0, 65};
    RunTypedCase<MLFloat16>(geometry, options, inputs, 3e-3f, 5e-4f,
                            /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
  }
}

TEST(GatedDeltaNetWebGpuTest, PackedGateProjectionsDecodeAndChunkwisePrefill) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  for (int tokens : {1, 65}) {
    SCOPED_TRACE(tokens);
    const Geometry geometry{tokens, 1, 16, 48, 128, 128};
    Inputs inputs = MakeInputs(geometry, 241);
    inputs.cu_seqlens = {0, tokens};
    for (bool packed_qkv : {false, true}) {
      SCOPED_TRACE(packed_qkv);
      RunTypedCase<MLFloat16>(geometry, options, inputs, 3e-3f, 5e-4f,
                              false, nullptr, true, false, nullptr, 0, packed_qkv, true);
    }
  }
}

TEST(GatedDeltaNetWebGpuTest, PackedGateProjectionsRecurrentPrefillWithActiveCapture) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.state_update_capacity = 8;
  const Geometry geometry{65, 1, 16, 48, 128, 128};
  Inputs inputs = MakeInputs(geometry, 243);
  inputs.cu_seqlens = {0, geometry.total_tokens};
  inputs.capture_count = {3};
  inputs.state_update_active = {1};
  for (bool packed_qkv : {false, true}) {
    SCOPED_TRACE(packed_qkv);
    RunTypedCase<MLFloat16>(geometry, options, inputs, 3e-3f, 5e-4f,
                            false, nullptr, true, false, nullptr, 0, packed_qkv, true);
  }
}

TEST(GatedDeltaNetWebGpuTest, PackedGateProjectionsUniformAndRagged) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  const Geometry geometry{130, 2, 2, 4, 32, 32};
  for (bool rank4 : {false, true}) {
    SCOPED_TRACE(rank4);
    Inputs inputs = MakeInputs(geometry, 242);
    if (!rank4) inputs.cu_seqlens = {0, 0, 130};
    RunTypedCase<float>(geometry, options, inputs, 5e-4f, 5e-4f,
                        rank4, nullptr, true, false, nullptr, 0, false, true);
    RunTypedCase<MLFloat16>(geometry, options, inputs, 3e-3f, 5e-4f,
                            rank4, nullptr, true, false, nullptr, 0, false, true);
  }
}

TEST(GatedDeltaNetWebGpuTest, PackedGateProjectionsRejectInvalidWidth) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  test.AddAttribute("gate_activation", std::string("qwen"));
  test.AddAttribute("beta_activation", std::string("sigmoid"));
  test.AddInput<float>("query", {1, 1, 4}, std::vector<float>(4, 0.1f));
  test.AddInput<float>("key", {1, 1, 4}, std::vector<float>(4, 0.1f));
  test.AddInput<float>("value", {1, 2, 4}, std::vector<float>(8, 0.1f));
  test.AddInput<int32_t>("cu_seqlens", {2}, {0, 1});
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("a_log", {2}, {0.0f, 0.0f});
  test.AddInput<float>("dt_bias", {2}, {0.0f, 0.0f});
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("gate_projections", {1, 5}, std::vector<float>(5, 0.0f));
  test.AddOutput<float>("output", {1, 2, 4}, std::vector<float>(8, 0.0f));
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "gate_projections must match query type and have shape", {}, nullptr, &providers);
}

TEST(GatedDeltaNetWebGpuTest, ChunkwiseQwenPrefillSaturatedGates) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry geometry{32, 1, 1, 2, 16, 7};
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  for (bool isolated_resets : {false, true}) {
    SCOPED_TRACE(isolated_resets);
    Inputs inputs = MakeInputs(geometry, 235);
    if (isolated_resets) {
      inputs.a_log[0] = 1.0f;
      for (int token : {0, 7, 16, 22}) {
        inputs.decay[static_cast<size_t>(token) * geometry.hv] = std::numeric_limits<float>::max();
      }
    } else {
      inputs.a_log[0] = 90.0f;
      for (int token = 0; token < geometry.total_tokens; ++token) {
        inputs.decay[static_cast<size_t>(token) * geometry.hv] = 1.0f;
      }
    }
    RunTypedCase<float>(geometry, options, inputs, 5e-4f, 5e-4f,
                        /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
  }
}

TEST(GatedDeltaNetWebGpuTest, ChunkwiseQwenPrefillRaggedAndEmptySequence) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry geometry{130, 3, 1, 3, 16, 7};
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.scale = 0.37f;
  for (const std::vector<int32_t>& boundaries : {std::vector<int32_t>{0, 0, 33, 130},
                                                 std::vector<int32_t>{0, 33, 33, 130},
                                                 std::vector<int32_t>{0, 33, 130, 130},
                                                 std::vector<int32_t>{0, 1, 2, 130},
                                                 std::vector<int32_t>{0, 16, 48, 130}}) {
    SCOPED_TRACE(testing::PrintToString(boundaries));
    for (bool with_state : {false, true}) {
      SCOPED_TRACE(with_state);
      Inputs inputs = MakeInputs(geometry, 232, with_state);
      inputs.cu_seqlens = boundaries;
      RunTypedCase<float>(geometry, options, inputs, 5e-4f, 5e-4f,
                          /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
    }
  }
}

TEST(GatedDeltaNetWebGpuTest, ChunkwiseQwenPrefillLargeRaggedBatch) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry geometry{2080, 64, 1, 1, 128, 128};
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  Inputs inputs = MakeInputs(geometry, 234);
  inputs.cu_seqlens.assign(geometry.batch + 1, 0);
  inputs.cu_seqlens.back() = geometry.total_tokens;
  RunTypedCase<float>(geometry, options, inputs, 5e-4f, 5e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, ChunkwiseQwenPrefillPackedAndUniform) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry geometry{130, 2, 2, 4, 32, 32};
  Options options;
  options.chunk_size = 16;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  const Inputs inputs = MakeInputs(geometry, 233);
  RunTypedCase<MLFloat16>(geometry, options, inputs, 3e-3f, 5e-4f,
                          /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/true,
                          /*omit_final_state=*/false, /*webgpu_config=*/nullptr,
                          /*test_max_storage_buffer_binding_size=*/0, /*packed_qkv=*/true);
  RunTypedCase<float>(geometry, options, inputs, 5e-4f, 5e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/true);
}

TEST(GatedDeltaNetWebGpuTest, CompactStateUpdates) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{12, 3, 1, 2, 4, 5};
  Inputs inputs = MakeInputs(g, 227);
  inputs.cu_seqlens = {0, 1, 5, 12};
  inputs.capture_count = {-1, 2, 8};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.state_update_capacity = 7;
  RunTypedCase<float>(g, options, inputs, 4e-4f, 4e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, CompactStateUpdatesFp16QwenGeometry) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{12, 3, 2, 6, 128, 128};
  Inputs inputs = MakeInputs(g, 229);
  inputs.cu_seqlens = {0, 1, 5, 12};
  inputs.capture_count = {-1, 2, 8};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.state_update_capacity = 7;
  RunCase(g, options, inputs, 3e-2f, 3e-2f,
          /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

#ifdef USE_WEBGPU
TEST(GatedDeltaNetWebGpuTest, CapturedStateReplayMatchesIndependentPrefixes) {
  auto replay_ep = DefaultWebGpuExecutionProvider();
  if (replay_ep == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry geometry{5, 1, 2, 6, 16, 5};
  Inputs inputs = MakeInputs(geometry, 233);
  for (auto& value : inputs.state0) {
    value *= 8.0f;
  }
  inputs.capture_count = {geometry.total_tokens};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  // Capture capacity intentionally exceeds both the full run length and every partial prefix.
  options.state_update_capacity = 7;
  std::vector<OrtValue> producer_fetches;
  ASSERT_NO_FATAL_FAILURE(RunTypedCase<float>(geometry, options, inputs, 4e-4f, 4e-4f,
                                              /*rank4=*/false, &producer_fetches, /*use_webgpu=*/true));
  ASSERT_EQ(producer_fetches.size(), 3u);
  const auto& captured = producer_fetches[2].Get<Tensor>();
  const int64_t capacity = options.state_update_capacity;
  const int64_t decay_elements = capacity * geometry.hv;
  const int64_t key_elements = capacity * geometry.hq * geometry.dk;
  const int64_t capsule_elements = decay_elements + key_elements + capacity * geometry.hv * geometry.dv;
  ASSERT_EQ(captured.Shape(), TensorShape({1, capsule_elements}));
  const int64_t state_elements = static_cast<int64_t>(inputs.state0.size());
  constexpr int64_t kDestinationOffset = 5;
  std::vector<float> destination(static_cast<size_t>(state_elements + 11), -931.25f);
  std::array<int64_t, 11> descriptor{0, kDestinationOffset, 0, decay_elements, decay_elements + key_elements,
                                     geometry.hv, geometry.dv, geometry.dk, geometry.hq, capacity, 1};

  Model model("gated_delta_net_captured_replay", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kMSDomain, 1}}, {},
              DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  auto tensor_arg = [&](const char* name, int element_type, int64_t extent) -> NodeArg& {
    ONNX_NAMESPACE::TypeProto type;
    auto* tensor_type = type.mutable_tensor_type();
    tensor_type->set_elem_type(element_type);
    tensor_type->mutable_shape()->add_dim()->set_dim_value(extent);
    return graph.GetOrCreateNodeArg(name, &type);
  };
  auto& source_arg = tensor_arg("source", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, state_elements);
  auto& capsule_arg = tensor_arg("capsule", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, capsule_elements);
  auto& destination_arg = tensor_arg("destination", ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                     static_cast<int64_t>(destination.size()));
  auto& metadata_arg = tensor_arg("metadata", ONNX_NAMESPACE::TensorProto_DataType_INT64, 11);
  auto& output_arg = graph.GetOrCreateNodeArg("output", nullptr);
  auto& node = graph.AddNode("replay", "GatedDeltaNetStateReplay", "",
                             {&source_arg, &capsule_arg, &destination_arg, &metadata_arg},
                             {&output_arg}, nullptr, kMSDomain);
  node.SetExecutionProviderType(kWebGpuExecutionProvider);
  graph.SetOutputs({&output_arg});
  ASSERT_STATUS_OK(graph.Resolve());
  std::stringstream stream;
  ASSERT_TRUE(model.ToProto().SerializeToOstream(&stream));
  auto* ep = replay_ep.get();
  const auto allocators = ep->CreatePreferredAllocators();
  ASSERT_FALSE(allocators.empty());
  const auto memory_info = allocators[0]->Info();
  SessionOptions session_options;
  session_options.graph_optimization_level = TransformerLevel::Default;
  InferenceSession session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(replay_ep)));
  ASSERT_STATUS_OK(session.Load(stream));
  ASSERT_STATUS_OK(session.Initialize());
  const auto allocator = session.GetAllocator(memory_info);
  ASSERT_NE(allocator, nullptr);
  CPUAllocator cpu_allocator;
  auto upload = [&](const Tensor& cpu, OrtValue& gpu) -> Status {
    Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape{cpu.Shape().Size()}, allocator, gpu);
    return ep->GetDataTransfer()->CopyTensor(cpu, *gpu.GetMutable<Tensor>());
  };
  Tensor incoming(DataTypeImpl::GetType<float>(), TensorShape{state_elements},
                  inputs.state0.data(), cpu_allocator.Info());
  Tensor cpu_destination(DataTypeImpl::GetType<float>(), TensorShape{static_cast<int64_t>(destination.size())},
                         destination.data(), cpu_allocator.Info());
  OrtValue source_value, capsule_value, destination_value, metadata_value;
  ASSERT_STATUS_OK(upload(incoming, source_value));
  ASSERT_STATUS_OK(upload(captured, capsule_value));
  ASSERT_STATUS_OK(upload(cpu_destination, destination_value));
  Tensor::InitOrtValue(DataTypeImpl::GetType<int64_t>(), TensorShape{11}, descriptor.data(),
                       cpu_allocator.Info(), metadata_value);
  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("source", source_value));
  ASSERT_STATUS_OK(binding->BindInput("capsule", capsule_value));
  ASSERT_STATUS_OK(binding->BindInput("destination", destination_value));
  ASSERT_STATUS_OK(binding->BindInput("metadata", metadata_value));
  ASSERT_STATUS_OK(binding->BindOutput("output", destination_value));

  for (int kept = 1; kept <= geometry.total_tokens; ++kept) {
    SCOPED_TRACE(kept);
    Geometry prefix_geometry = geometry;
    prefix_geometry.total_tokens = kept;
    Inputs prefix_inputs = inputs;
    prefix_inputs.q.resize(static_cast<size_t>(kept * geometry.hq * geometry.dk));
    prefix_inputs.k.resize(static_cast<size_t>(kept * geometry.hq * geometry.dk));
    prefix_inputs.v.resize(static_cast<size_t>(kept * geometry.hv * geometry.dv));
    prefix_inputs.decay.resize(static_cast<size_t>(kept * geometry.hv));
    prefix_inputs.beta.resize(static_cast<size_t>(kept * geometry.hv));
    prefix_inputs.capture_count.clear();
    Options prefix_options = options;
    prefix_options.state_update_capacity = 0;
    std::vector<OrtValue> prefix_fetches;
    ASSERT_NO_FATAL_FAILURE(RunTypedCase<float>(prefix_geometry, prefix_options, prefix_inputs, 4e-4f, 4e-4f,
                                                /*rank4=*/false, &prefix_fetches, /*use_webgpu=*/true));
    ASSERT_EQ(prefix_fetches.size(), 3u);
    const auto expected = prefix_fetches[1].Get<Tensor>().DataAsSpan<float>();
    ASSERT_EQ(expected.size(), static_cast<size_t>(state_elements));

    descriptor[10] = kept;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, *binding));
    std::vector<float> actual(destination.size());
    Tensor cpu_actual(DataTypeImpl::GetType<float>(), cpu_destination.Shape(), actual.data(), cpu_allocator.Info());
    ASSERT_STATUS_OK(ep->GetDataTransfer()->CopyTensor(destination_value.Get<Tensor>(), cpu_actual));
    for (size_t index = 0; index < expected.size(); ++index) {
      const float value = actual[static_cast<size_t>(kDestinationOffset) + index];
      ASSERT_TRUE(std::isfinite(value)) << index;
      ASSERT_LE(std::abs(value - expected[index]), 1e-5f + 2e-5f * std::abs(expected[index]))
          << "state element " << index;
    }
    for (size_t index = 0; index < actual.size(); ++index) {
      if (index < kDestinationOffset || index >= static_cast<size_t>(kDestinationOffset + state_elements)) {
        EXPECT_EQ(actual[index], destination[index]) << "neighboring destination element " << index;
      }
    }
  }
}
#endif  // USE_WEBGPU

TEST(GatedDeltaNetWebGpuTest, InactiveCompactStateUpdatesAreZero) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{4, 1, 1, 2, 4, 3};
  Inputs inputs = MakeInputs(g, 228);
  inputs.capture_count = {4};
  inputs.state_update_active = {0};
  Options options;
  options.state_update_capacity = 7;
  RunTypedCase<float>(g, options, inputs, 1e-4f, 1e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, IgnoresStateUpdateActiveWithoutCapture) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{2, 1, 1, 1, 4, 3};
  Inputs inputs = MakeInputs(g, 228);
  inputs.state_update_active = {0};
  RunTypedCase<float>(g, Options{}, inputs, 1e-4f, 1e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true);
}

TEST(GatedDeltaNetWebGpuTest, FinalStateIsOptional) {
  if (NeedSkipGatedDeltaNetWebGpuTest()) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Geometry g{2, 1, 1, 1, 4, 3};
  RunTypedCase<float>(g, Options{}, MakeInputs(g, 229), 1e-4f, 1e-4f,
                      /*rank4=*/false, /*fetches=*/nullptr, /*use_webgpu=*/true,
                      /*omit_final_state=*/true);
}

// ---------------------------------------------------------------------------
// Chunked engine (prefill): T well above the 32-token plan threshold.
// ---------------------------------------------------------------------------
TEST(GatedDeltaNetTest, Chunked_GatedDelta_InverseGqa) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{128, 1, 2, 6, kDim, kDim};  // Hv/Hq == 3, the Qwen3.8 ratio
  RunCase(g, Options{}, MakeInputs(g, 11), 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_PartialChunkAndBoundaries) {
  if (NeedSkipGatedDeltaNetTest()) return;
  for (int total : {63, 64, 65, 130}) {
    Geometry g{total, 1, 1, 2, kDim, kDim};
    RunCase(g, Options{}, MakeInputs(g, static_cast<uint32_t>(total)), 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, Chunked_AllUpdateRules) {
  if (NeedSkipGatedDeltaNetTest()) return;
  for (const char* rule : {"linear", "gated", "delta", "gated_delta"}) {
    SCOPED_TRACE(rule);
    Geometry g{128, 1, 1, 2, kDim, kDim};
    Options o;
    o.update_rule = rule;
    RunCase(g, o, MakeInputs(g, 5), 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, Chunked_FusedActivations) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{128, 1, 1, 2, kDim, kDim};
  Options o;
  o.gate_activation = "qwen";
  o.beta_activation = "sigmoid";
  o.qk_l2_norm = 1;
  RunCase(g, o, MakeInputs(g, 7), 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_NoInitialState) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{128, 1, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 13, /*with_state=*/false);
  in.cu_seqlens = {0, 128};  // batch cannot be inferred from a missing state
  RunCase(g, Options{}, in, 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_RaggedUnequalLengths) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{200, 3, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 17);
  in.cu_seqlens = {0, 40, 150, 200};
  RunCase(g, Options{}, in, 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_UniformBatch) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{256, 2, 1, 2, kDim, kDim};  // no cu_seqlens: batch comes from initial_state
  RunCase(g, Options{}, MakeInputs(g, 19), 3e-2f, 3e-2f);
}

TEST(GatedDeltaNetTest, Chunked_PackedQkv) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 16, 48, kDim, kDim};
  RunTypedCase<MLFloat16>(g, Options{}, MakeInputs(g, 21), 3e-2f, 3e-2f,
                          /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/false,
                          /*omit_final_state=*/false, /*webgpu_config=*/nullptr,
                          /*test_max_storage_buffer_binding_size=*/0, /*packed_qkv=*/true);
}

// The rank-4 [batch, sequence, heads, head_size] spelling must match the packed one exactly.
TEST(GatedDeltaNetTest, Rank4BatchSequenceMatchesPacked) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry prefill{256, 2, 2, 6, kDim, kDim};
  RunCase(prefill, Options{}, MakeInputs(prefill, 23), 3e-2f, 3e-2f, /*rank4=*/true);
  Geometry decode{3, 3, 1, 2, kDim, kDim};  // one token per request
  RunCase(decode, Options{}, MakeInputs(decode, 29), 3e-2f, 3e-2f, /*rank4=*/true);
}

// chunk_size=32 pins the narrow shared-memory configuration (96 KB) that consumer
// Blackwell needs, since SM120 allows only 99 KB per block against SM90's 227 KB.
TEST(GatedDeltaNetTest, Chunked_NarrowChunkForSmallSharedMemory) {
  if (NeedSkipGatedDeltaNetTest()) return;
  for (int total : {128, 65, 200}) {
    Geometry g{total, 1, 2, 6, kDim, kDim};
    Options o;
    o.chunk_size = 32;
    RunCase(g, o, MakeInputs(g, static_cast<uint32_t>(total) + 1), 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, ChunkedSplit_DuplicateRowsMatchReference) {
  if (NeedSkipGatedDeltaNetSplitTest()) return;
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_GDN_PLAN", "chunked_split"}}};

  for (int total : {64, 65, 128, 130}) {
    SCOPED_TRACE(total);
    Geometry g{total, 1, 2, 6, kDim, kDim};
    Inputs in = MakeInputs(g, static_cast<uint32_t>(total) + 101);

    auto duplicate_token = [](auto& values, int token_width, int dst, int src) {
      std::copy_n(values.begin() + static_cast<ptrdiff_t>(src) * token_width,
                  token_width,
                  values.begin() + static_cast<ptrdiff_t>(dst) * token_width);
    };
    for (int dst = 1; dst < total; dst += 2) {
      duplicate_token(in.q, g.hq * g.dk, dst, dst - 1);
      duplicate_token(in.k, g.hq * g.dk, dst, dst - 1);
      duplicate_token(in.v, g.hv * g.dv, dst, dst - 1);
      duplicate_token(in.decay, g.hv, dst, dst - 1);
      duplicate_token(in.beta, g.hv, dst, dst - 1);
    }

    RunCase(g, Options{}, in, 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, ChunkedSplit_AllUpdateRulesMatchReference) {
  if (NeedSkipGatedDeltaNetSplitTest()) return;
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_GDN_PLAN", "chunked_split"}}};

  Geometry g{256, 2, 1, 2, kDim, kDim};
  for (const char* rule : {"linear", "gated", "delta", "gated_delta"}) {
    SCOPED_TRACE(rule);
    Options options;
    options.update_rule = rule;
    RunCase(g, options, MakeInputs(g, 131), 3e-2f, 3e-2f);
  }
}

TEST(GatedDeltaNetTest, ChunkedSplit_FusedActivationsMatchReference) {
  if (NeedSkipGatedDeltaNetSplitTest()) return;
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_GDN_PLAN", "chunked_split"}}};

  Geometry g{128, 1, 2, 6, kDim, kDim};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  RunCase(g, options, MakeInputs(g, 137), 3e-2f, 3e-2f);
}

// ---------------------------------------------------------------------------
// Recurrent engine (decode / MTP verify): T below the plan threshold.
// ---------------------------------------------------------------------------
TEST(GatedDeltaNetTest, Recurrent_SingleToken) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{1, 1, 1, 2, kDim, kDim};
  RunCase(g, Options{}, MakeInputs(g, 23), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Recurrent_PackedQkvDecode) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{2, 2, 16, 48, kDim, kDim};
  RunTypedCase<MLFloat16>(g, Options{}, MakeInputs(g, 27), 2e-2f, 2e-2f,
                          /*rank4=*/true, /*fetches=*/nullptr, /*use_webgpu=*/false,
                          /*omit_final_state=*/false, /*webgpu_config=*/nullptr,
                          /*test_max_storage_buffer_binding_size=*/0, /*packed_qkv=*/true);
}

TEST(GatedDeltaNetTest, Recurrent_VerifyBatch) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 2, kDim, kDim};
  RunCase(g, Options{}, MakeInputs(g, 29), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Recurrent_Float32) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 2, 64, 32};
  RunTypedCase<float>(g, Options{}, MakeInputs(g, 149), 1e-4f, 1e-4f);
}

TEST(GatedDeltaNetTest, BFloat16UsesRecurrentFallback) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  const Inputs in = MakeInputs(g, 151);
  RunTypedCase<BFloat16>(g, Options{}, in, 6e-3f, 2e-3f);
  {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_PLAN", "recurrent"}}};
    RunTypedCase<BFloat16>(g, Options{}, in, 6e-3f, 2e-3f);
  }
  if (!NeedSkipGatedDeltaNetSplitTest()) {
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_PLAN", "chunked_split"}}};
    RunTypedCase<BFloat16>(g, Options{}, in, 6e-3f, 2e-3f);
  }
}

TEST(GatedDeltaNetTest, BFloat16ValuesOutsideFp16Range) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};

  // Values above FP16 max finite (65504): 2^20 * 0.5 = 524288.0f would overflow to Inf if
  // narrowed through FP16.
  {
    Inputs in = MakeInputs(g, 159);
    const float large_scale = std::ldexp(1.0f, 20);
    for (float& x : in.v) x *= large_scale;
    for (float& x : in.state0) x *= large_scale;
    Options o;
    o.scale = std::ldexp(1.0f, -20) / std::sqrt(static_cast<float>(kDim));
    std::vector<OrtValue> fetches;
    RunTypedCase<BFloat16>(g, o, in, 6e-3f, 2e-3f * large_scale, false, &fetches);
    ASSERT_EQ(fetches.size(), 3u);
    const float* final_state = fetches[1].Get<Tensor>().Data<float>();
    float max_abs_state = 0.0f;
    for (int64_t i = 0; i < fetches[1].Get<Tensor>().Shape().Size(); ++i) {
      ASSERT_TRUE(std::isfinite(final_state[i]));
      max_abs_state = std::max(max_abs_state, std::abs(final_state[i]));
    }
    EXPECT_GT(max_abs_state, 65504.0f);
  }

  // Values below FP16 min subnormal (2^-24): 2^-30 would flush to 0 if narrowed through FP16.
  {
    Inputs in = MakeInputs(g, 163);
    const float small_scale = std::ldexp(1.0f, -30);
    for (float& x : in.v) x *= small_scale;
    for (float& x : in.state0) x *= small_scale;
    Options o;
    o.scale = std::ldexp(1.0f, 30) / std::sqrt(static_cast<float>(kDim));
    std::vector<OrtValue> fetches;
    RunTypedCase<BFloat16>(g, o, in, 6e-3f, 2e-3f * small_scale, false, &fetches);
    ASSERT_EQ(fetches.size(), 3u);
    const BFloat16* out = fetches[0].Get<Tensor>().Data<BFloat16>();
    float max_abs_out = 0.0f;
    for (int64_t i = 0; i < fetches[0].Get<Tensor>().Shape().Size(); ++i) {
      max_abs_out = std::max(max_abs_out, std::abs(out[i].ToFloat()));
    }
    EXPECT_GT(max_abs_out, 1e-2f);
  }
}

TEST(GatedDeltaNetTest, BFloat16OutputAvoidsFp16DoubleRounding) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  Inputs in;
  in.q.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.k.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.v.assign(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f);
  in.decay.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.beta.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.state0.assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);
  for (int t = 0; t < g.total_tokens; ++t) {
    in.q[static_cast<size_t>(t) * g.dk] = 1.0f;
  }
  for (int hv = 0; hv < g.hv; ++hv) {
    for (int c = 0; c < g.dv; ++c) {
      in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk] = 1.0f;
    }
  }

  // 1 + 23 * 2^-11 = 1.01123046875 rounds directly to BFloat16 1.0078125 (0x3f81),
  // whereas rounding to FP16 first (1.01171875) and then to BFloat16 double-rounds to
  // 1.015625 (0x3f82).
  Options o;
  o.update_rule = "linear";
  o.scale = 1.01123046875f;
  std::vector<OrtValue> fetches;
  RunTypedCase<BFloat16>(g, o, in, 0.0f, 0.0f, false, &fetches);
  ASSERT_EQ(fetches.size(), 3u);
  const BFloat16 expected(1.0078125f);
  const BFloat16* out = fetches[0].Get<Tensor>().Data<BFloat16>();
  for (int64_t i = 0; i < fetches[0].Get<Tensor>().Shape().Size(); ++i) {
    ASSERT_EQ(out[i].val, expected.val) << "output element " << i;
  }
}

TEST(GatedDeltaNetTest, BFloat16PreservesFp32StateCancellation) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  Inputs in;
  in.q.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.k.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.v.assign(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f);
  in.decay.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.beta.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.state0.assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);
  for (int t = 0; t < g.total_tokens; ++t) {
    in.q[static_cast<size_t>(t) * g.dk + 0] = 1.0f;
    in.q[static_cast<size_t>(t) * g.dk + 1] = -1.0f;
  }
  for (int hv = 0; hv < g.hv; ++hv) {
    for (int c = 0; c < g.dv; ++c) {
      in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + 0] = 256.0f;
      in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + 1] = 257.0f;
    }
  }

  Options o;
  o.update_rule = "linear";
  o.qk_l2_norm = 0;
  o.scale = 1.0f;
  const BFloat16 expected(-1.0f);

  for (const char* plan_env : {"", "chunked_split"}) {
    if (std::string(plan_env) == "chunked_split" && NeedSkipGatedDeltaNetSplitTest()) continue;
    SCOPED_TRACE(plan_env);
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_PLAN", plan_env}}};
    std::vector<OrtValue> fetches;
    RunTypedCase<BFloat16>(g, o, in, 0.0f, 0.0f, false, &fetches);
    ASSERT_EQ(fetches.size(), 3u);
    const BFloat16* out = fetches[0].Get<Tensor>().Data<BFloat16>();
    for (int64_t i = 0; i < fetches[0].Get<Tensor>().Shape().Size(); ++i) {
      ASSERT_EQ(out[i].val, expected.val) << "output element " << i;
    }
  }
}

TEST(GatedDeltaNetTest, BFloat16PreservesTinyFp32StateCancellation) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  const float big = std::ldexp(1.0f, 125);
  const float tiny = std::ldexp(1.0f, -125);
  const float tiny_plus = tiny * (1.0f + std::ldexp(1.0f, -10));

  Options o;
  o.update_rule = "linear";
  o.qk_l2_norm = 0;
  o.scale = 1.0f;

  // Test both uniform tiny pairs (-64 * 2^-10 = -0.0625) and a mixed-magnitude column where
  // the first pair is q=(0,0), state=(1,0) and the remaining 63 pairs are tiny (-63 * 2^-10 =
  // -0.0615234375).
  for (bool mixed_column : {false, true}) {
    SCOPED_TRACE(mixed_column ? "mixed_column" : "uniform_tiny");
    Inputs in;
    in.q.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
    in.k.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
    in.v.assign(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f);
    in.decay.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
    in.beta.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
    in.state0.assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);

    for (int t = 0; t < g.total_tokens; ++t) {
      for (int d = 0; d < g.dk; d += 2) {
        in.q[static_cast<size_t>(t) * g.dk + d] = big;
        in.q[static_cast<size_t>(t) * g.dk + d + 1] = -big;
      }
      if (mixed_column) {
        in.q[static_cast<size_t>(t) * g.dk + 0] = 0.0f;
        in.q[static_cast<size_t>(t) * g.dk + 1] = 0.0f;
      }
    }
    for (int hv = 0; hv < g.hv; ++hv) {
      for (int c = 0; c < g.dv; ++c) {
        for (int d = 0; d < g.dk; d += 2) {
          in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + d] = tiny;
          in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + d + 1] = tiny_plus;
        }
        if (mixed_column) {
          in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + 0] = 1.0f;
          in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + 1] = 0.0f;
        }
      }
    }

    const BFloat16 expected(mixed_column ? -0.0615234375f : -0.0625f);
    for (const char* plan_env : {"", "chunked_split"}) {
      if (std::string(plan_env) == "chunked_split" && NeedSkipGatedDeltaNetSplitTest()) continue;
      SCOPED_TRACE(plan_env);
      ScopedEnvironmentVariables scoped_env_vars{
          EnvVarMap{{"ORT_GDN_PLAN", plan_env}}};
      std::vector<OrtValue> fetches;
      RunTypedCase<BFloat16>(g, o, in, 0.0f, 0.0f, false, &fetches);
      ASSERT_EQ(fetches.size(), 3u);
      const BFloat16* out = fetches[0].Get<Tensor>().Data<BFloat16>();
      for (int64_t i = 0; i < fetches[0].Get<Tensor>().Shape().Size(); ++i) {
        ASSERT_EQ(out[i].val, expected.val) << "output element " << i;
      }
    }
  }
}

TEST(GatedDeltaNetTest, BFloat16NearFltMaxStateDoesNotOverflow) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  Inputs in;
  in.q.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.k.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.v.assign(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f);
  in.decay.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.beta.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.state0.assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);

  const float q0 = std::ldexp(1.0f, -120);
  const float max_f = std::numeric_limits<float>::max();
  for (int t = 0; t < g.total_tokens; ++t) {
    in.q[static_cast<size_t>(t) * g.dk + 0] = q0;
  }
  for (int hv = 0; hv < g.hv; ++hv) {
    for (int c = 0; c < g.dv; ++c) {
      in.state0[(static_cast<size_t>(hv) * g.dv + c) * g.dk + 0] = max_f;
    }
  }

  Options o;
  o.update_rule = "linear";
  o.qk_l2_norm = 0;
  o.scale = 1.0f;
  const BFloat16 expected(256.0f);

  for (const char* plan_env : {"", "chunked_split"}) {
    if (std::string(plan_env) == "chunked_split" && NeedSkipGatedDeltaNetSplitTest()) continue;
    SCOPED_TRACE(plan_env);
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_PLAN", plan_env}}};
    std::vector<OrtValue> fetches;
    RunTypedCase<BFloat16>(g, o, in, 0.0f, 0.0f, false, &fetches);
    ASSERT_EQ(fetches.size(), 3u);
    const BFloat16* out = fetches[0].Get<Tensor>().Data<BFloat16>();
    for (int64_t i = 0; i < fetches[0].Get<Tensor>().Shape().Size(); ++i) {
      ASSERT_EQ(out[i].val, expected.val) << "output element " << i;
    }
    const float* final_state = fetches[1].Get<Tensor>().Data<float>();
    for (int64_t i = 0; i < fetches[1].Get<Tensor>().Shape().Size(); ++i) {
      ASSERT_TRUE(std::isfinite(final_state[i])) << "state element " << i;
    }
  }
}

TEST(GatedDeltaNetTest, BFloat16PreservesDeltaUpdateCancellation) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  Inputs in;
  in.q.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.k.assign(static_cast<size_t>(g.total_tokens) * g.hq * g.dk, 0.0f);
  in.v.assign(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f);
  in.decay.assign(static_cast<size_t>(g.total_tokens) * g.hv, 0.0f);
  in.beta.assign(static_cast<size_t>(g.total_tokens) * g.hv, 1.0f);
  in.state0.assign(static_cast<size_t>(g.batch) * g.hv * g.dv * g.dk, 0.0f);

  for (int t = 0; t < g.total_tokens; ++t) {
    in.q[static_cast<size_t>(t) * g.dk + 0] = 1.0f;
    in.k[static_cast<size_t>(t) * g.dk + 0] = 1.0f;
    const float v_val = (t % 2 == 0) ? 1024.0f : 1.0f;
    for (int i = 0; i < g.hv * g.dv; ++i) {
      in.v[static_cast<size_t>(t) * g.hv * g.dv + i] = v_val;
    }
  }

  Options o;
  o.update_rule = "delta";
  o.qk_l2_norm = 0;
  o.scale = 1.0f;
  const BFloat16 expected_even(1024.0f);
  const BFloat16 expected_odd(1.0f);

  for (const char* plan_env : {"", "chunked_split"}) {
    if (std::string(plan_env) == "chunked_split" && NeedSkipGatedDeltaNetSplitTest()) continue;
    SCOPED_TRACE(plan_env);
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_PLAN", plan_env}}};
    std::vector<OrtValue> fetches;
    RunTypedCase<BFloat16>(g, o, in, 0.0f, 0.0f, false, &fetches);
    ASSERT_EQ(fetches.size(), 3u);
    const BFloat16* out = fetches[0].Get<Tensor>().Data<BFloat16>();
    for (int t = 0; t < g.total_tokens; ++t) {
      const BFloat16 expected = (t % 2 == 0) ? expected_even : expected_odd;
      for (int i = 0; i < g.hv * g.dv; ++i) {
        ASSERT_EQ(out[static_cast<size_t>(t) * g.hv * g.dv + i].val, expected.val)
            << "token " << t << " element " << i;
      }
    }
    const float* final_state = fetches[1].Get<Tensor>().Data<float>();
    for (int hv = 0; hv < g.hv; ++hv) {
      for (int c = 0; c < g.dv; ++c) {
        ASSERT_EQ(final_state[(static_cast<size_t>(hv) * g.dv + c) * g.dk + 0], 1.0f);
      }
    }
  }
}

TEST(GatedDeltaNetTest, BFloat16PrecisionModesAndSessionOption) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 1, 1, 2, kDim, kDim};
  const Inputs in = MakeInputs(g, 173);
  for (const char* mode : {"fp32", "fp16", "bf16"}) {
    SCOPED_TRACE(mode);
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_BF16_PRECISION", mode}}};
    RunTypedCase<BFloat16>(g, Options{}, in, 6e-3f, 2e-3f);
  }

  // Session option overrides the environment variable, and invalid values are rejected.
  Inputs in_rounded = in;
  in_rounded.q = RoundToTensorType<BFloat16>(in.q);
  in_rounded.k = RoundToTensorType<BFloat16>(in.k);
  in_rounded.v = RoundToTensorType<BFloat16>(in.v);
  std::vector<float> ref_out, ref_state;
  Reference(g, Options{}, in_rounded, &ref_out, &ref_state);

  auto populate_tester = [&](OpTester& test) {
    AddCommonAttrs(test, Options{});
    test.AddInput<BFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToTensorType<BFloat16>(in_rounded.q));
    test.AddInput<BFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToTensorType<BFloat16>(in_rounded.k));
    test.AddInput<BFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToTensorType<BFloat16>(in_rounded.v));
    test.AddOptionalInputEdge<int32_t>();
    test.AddInput<float>("decay", {g.total_tokens, g.hv}, in_rounded.decay);
    test.AddInput<float>("beta", {g.total_tokens, g.hv}, in_rounded.beta);
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in_rounded.state0);
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddOutput<BFloat16>("output", {g.total_tokens, g.hv, g.dv}, ToTensorType<BFloat16>(ref_out),
                             false, 6e-3f, 6e-3f);
    test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk}, ref_state, false, 2e-3f, 2e-3f);
    test.AddOutput<float>("state_update", {g.batch, 0}, {});
  };

  for (const char* mode : {"fp32", "fp16", "bf16"}) {
    SCOPED_TRACE(mode);
    ScopedEnvironmentVariables scoped_env_vars{
        EnvVarMap{{"ORT_GDN_BF16_PRECISION", "invalid_env_overridden_by_session"}}};
    SessionOptions session_options;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(
        contrib::cuda::gated_delta_net::kGatedDeltaNetBf16PrecisionConfig, mode));
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    populate_tester(test);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }

  for (const char* invalid_mode : {"invalid_mode", "high", "fast", "half", "1", "0", ""}) {
    SCOPED_TRACE(invalid_mode);
    SessionOptions bad_options;
    ASSERT_STATUS_OK(bad_options.config_options.AddConfigEntry(
        contrib::cuda::gated_delta_net::kGatedDeltaNetBf16PrecisionConfig, invalid_mode));
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    populate_tester(test);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(bad_options, OpTester::ExpectResult::kExpectFailure,
             "ep.cuda.gated_delta_net_bf16_precision must be one of: fp32, fp16, bf16", {}, nullptr,
             &eps);
  }
}

TEST(GatedDeltaNetTest, BFloat16SplitPreservesMultiChunkDecay) {
  if (NeedSkipGatedDeltaNetSplitTest()) return;
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_GDN_PLAN", "chunked_split"}}};

  constexpr int total_tokens = 65536;
  constexpr float raw_decay = -1e-5f;
  const float expected_state = std::exp(raw_decay * static_cast<float>(total_tokens));

  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  o.update_rule = "gated";
  o.qk_l2_norm = 0;
  o.scale = 1.0f;
  AddCommonAttrs(test, o);

  const std::vector<BFloat16> zero_qkv(static_cast<size_t>(total_tokens) * kDim, BFloat16(0.0f));
  const std::vector<float> decay(total_tokens, raw_decay);
  const std::vector<float> state0(static_cast<size_t>(kDim) * kDim, 1.0f);
  const std::vector<float> ref_state(static_cast<size_t>(kDim) * kDim, expected_state);

  test.AddInput<BFloat16>("query", {total_tokens, 1, kDim}, zero_qkv);
  test.AddInput<BFloat16>("key", {total_tokens, 1, kDim}, zero_qkv);
  test.AddInput<BFloat16>("value", {total_tokens, 1, kDim}, zero_qkv);
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {total_tokens, 1}, decay);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("initial_state", {1, 1, kDim, kDim}, state0);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOutput<BFloat16>("output", {total_tokens, 1, kDim}, zero_qkv, /*sort_output=*/false, 0.0f, 0.0f);
  test.AddOutput<float>("final_state", {1, 1, kDim, kDim}, ref_state, /*sort_output=*/false, 1e-3f, 1e-3f);
  test.AddOutput<float>("state_update", {1, 0}, {});

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

// One token per request drives the warp-specialised decode kernel, a separate instantiation
// from the chunk loop the case above exercises.
TEST(GatedDeltaNetTest, BFloat16SingleTokenDecode) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{3, 3, 1, 2, kDim, kDim};
  RunTypedCase<BFloat16>(g, Options{}, MakeInputs(g, 153), 6e-3f, 2e-3f);
}

TEST(GatedDeltaNetTest, BFloat16RaggedUnequalLengths) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{200, 3, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 157);
  in.cu_seqlens = {0, 40, 150, 200};
  RunTypedCase<BFloat16>(g, Options{}, in, 6e-3f, 2e-3f);
}

TEST(GatedDeltaNetTest, Recurrent_SmallHeadSizes) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 2, 64, 32};  // shapes the chunked engine rejects
  RunCase(g, Options{}, MakeInputs(g, 31), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Recurrent_SingleValueChannelWithQkNormalization) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 1, 64, 1};
  Options options;
  options.qk_l2_norm = 1;
  RunCase(g, options, MakeInputs(g, 37), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Recurrent_GenericSingleValueChannelWithQkNormalization) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 1, 96, 1};
  Options options;
  options.qk_l2_norm = 1;
  RunCase(g, options, MakeInputs(g, 139), 2e-2f, 2e-2f);
}

TEST(GatedDeltaNetTest, Ragged_MixedPrefillKeepsDecodeArithmeticStable) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry mixed_geometry{65, 2, 1, 2, kDim, kDim};
  Inputs mixed_inputs = MakeInputs(mixed_geometry, 60);
  mixed_inputs.cu_seqlens = {0, 64, 65};

  std::vector<OrtValue> mixed_fetches;
  RunCase(mixed_geometry, Options{}, mixed_inputs, 3e-2f, 3e-2f, false, &mixed_fetches);

  Geometry decode_geometry{1, 1, 1, 2, kDim, kDim};
  Inputs decode_inputs;
  const size_t qk_row = static_cast<size_t>(decode_geometry.hq) * kDim;
  const size_t v_row = static_cast<size_t>(decode_geometry.hv) * kDim;
  const size_t state_row = static_cast<size_t>(decode_geometry.hv) * kDim * kDim;
  decode_inputs.q.assign(mixed_inputs.q.end() - qk_row, mixed_inputs.q.end());
  decode_inputs.k.assign(mixed_inputs.k.end() - qk_row, mixed_inputs.k.end());
  decode_inputs.v.assign(mixed_inputs.v.end() - v_row, mixed_inputs.v.end());
  decode_inputs.decay.assign(mixed_inputs.decay.end() - decode_geometry.hv,
                             mixed_inputs.decay.end());
  decode_inputs.beta.assign(mixed_inputs.beta.end() - decode_geometry.hv,
                            mixed_inputs.beta.end());
  decode_inputs.state0.assign(mixed_inputs.state0.end() - state_row, mixed_inputs.state0.end());
  decode_inputs.a_log = mixed_inputs.a_log;
  decode_inputs.dt_bias = mixed_inputs.dt_bias;

  std::vector<OrtValue> decode_fetches;
  RunCase(decode_geometry, Options{}, decode_inputs, 2e-2f, 2e-2f, false, &decode_fetches);

  ASSERT_EQ(mixed_fetches.size(), 3u);
  ASSERT_EQ(decode_fetches.size(), 3u);
  const auto* mixed_output = mixed_fetches[0].Get<Tensor>().Data<MLFloat16>() +
                             static_cast<size_t>(64) * decode_geometry.hv * kDim;
  const auto* decode_output = decode_fetches[0].Get<Tensor>().Data<MLFloat16>();
  for (size_t i = 0; i < v_row; ++i) {
    ASSERT_EQ(mixed_output[i].val, decode_output[i].val) << "output element " << i;
  }
  const auto* mixed_state = mixed_fetches[1].Get<Tensor>().Data<float>() + state_row;
  const auto* decode_state = decode_fetches[1].Get<Tensor>().Data<float>();
  for (size_t i = 0; i < state_row; ++i) {
    ASSERT_EQ(mixed_state[i], decode_state[i]) << "state element " << i;
  }
}

// A mixed ragged call captures zero, partial, and full-capacity prefixes. Replaying the packed
// factors from the incoming state must exactly match an independent recurrent run for every
// prefix of the full-capacity row.
TEST(GatedDeltaNetTest, Ragged_CompactReplayMatchesSequentialPrefixes) {
  if (NeedSkipGatedDeltaNetTest()) return;
  constexpr int kPrefillTokens = 128;
  constexpr int kCapacity = 8;
  constexpr int kPartial = 4;
  Geometry g{kPrefillTokens + 5 + kCapacity, 4, 1, 2, kDim, kDim};
  Options options;
  options.gate_activation = "qwen";
  options.beta_activation = "sigmoid";
  options.qk_l2_norm = 1;
  options.state_update_capacity = kCapacity;

  Inputs in = MakeInputs(g, 79);
  in.cu_seqlens = {0, 0, kPrefillTokens, kPrefillTokens + 5, g.total_tokens};
  in.capture_count = {0, 0, kPartial, kCapacity};

  std::vector<OrtValue> fetches;
  RunCase(g, options, in, 3e-2f, 3e-2f, false, &fetches);
  ASSERT_EQ(fetches.size(), 3u);

  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{{"ORT_GDN_PLAN", "recurrent"}}};
  std::vector<OrtValue> recurrent_fetches;
  RunCase(g, options, in, 3e-2f, 3e-2f, false, &recurrent_fetches);
  ASSERT_EQ(recurrent_fetches.size(), 3u);

  const Tensor& output_tensor = fetches[0].Get<Tensor>();
  const Tensor& recurrent_output_tensor = recurrent_fetches[0].Get<Tensor>();
  ASSERT_EQ(output_tensor.Shape(), recurrent_output_tensor.Shape());
  const MLFloat16* output_data = output_tensor.Data<MLFloat16>();
  const MLFloat16* recurrent_output_data = recurrent_output_tensor.Data<MLFloat16>();
  for (int64_t i = 0; i < output_tensor.Shape().Size(); ++i) {
    ASSERT_NEAR(output_data[i].ToFloat(), recurrent_output_data[i].ToFloat(), 3e-2f)
        << "output element " << i;
  }

  const Tensor& final_state_tensor = fetches[1].Get<Tensor>();
  const Tensor& recurrent_final_state_tensor = recurrent_fetches[1].Get<Tensor>();
  ASSERT_EQ(final_state_tensor.Shape(), recurrent_final_state_tensor.Shape());
  const float* final_state_data = final_state_tensor.Data<float>();
  const float* recurrent_final_state_data = recurrent_final_state_tensor.Data<float>();
  for (int64_t i = 0; i < final_state_tensor.Shape().Size(); ++i) {
    ASSERT_NEAR(final_state_data[i], recurrent_final_state_data[i], 3e-2f)
        << "final_state element " << i;
  }

  const Tensor& state_update_tensor = fetches[2].Get<Tensor>();
  const Tensor& recurrent_state_update_tensor = recurrent_fetches[2].Get<Tensor>();
  const int64_t decay_elements = static_cast<int64_t>(kCapacity) * g.hv;
  const int64_t key_elements = static_cast<int64_t>(kCapacity) * g.hq * g.dk;
  const int64_t delta_elements = static_cast<int64_t>(kCapacity) * g.hv * g.dv;
  const int64_t row_width = decay_elements + key_elements + delta_elements;
  EXPECT_EQ(state_update_tensor.Shape(), TensorShape({g.batch, row_width}));
  EXPECT_EQ(state_update_tensor.DataType(), DataTypeImpl::GetType<float>());

  const size_t state_head_size = static_cast<size_t>(g.dv) * g.dk;
  const size_t state_row_size = static_cast<size_t>(g.hv) * state_head_size;
  Options prefix_options = options;
  prefix_options.state_update_capacity = 0;
  for (int verify_batch : {2, 3}) {
    const int capture_count = in.capture_count[verify_batch];
    const float* row = state_update_tensor.Data<float>() + verify_batch * row_width;
    const float* recurrent_row =
        recurrent_state_update_tensor.Data<float>() + verify_batch * row_width;
    for (int t = 0; t < capture_count; ++t) {
      for (int i = 0; i < g.hv; ++i) {
        EXPECT_NEAR(row[static_cast<int64_t>(t) * g.hv + i],
                    recurrent_row[static_cast<int64_t>(t) * g.hv + i], 1e-5f);
      }
      for (int i = 0; i < g.hq * g.dk; ++i) {
        EXPECT_NEAR(row[decay_elements + static_cast<int64_t>(t) * g.hq * g.dk + i],
                    recurrent_row[decay_elements + static_cast<int64_t>(t) * g.hq * g.dk + i],
                    1e-5f);
      }
      for (int i = 0; i < g.hv * g.dv; ++i) {
        EXPECT_NEAR(row[decay_elements + key_elements +
                        static_cast<int64_t>(t) * g.hv * g.dv + i],
                    recurrent_row[decay_elements + key_elements +
                                  static_cast<int64_t>(t) * g.hv * g.dv + i],
                    1e-5f);
      }
    }

    std::vector<float> replay(in.state0.begin() + verify_batch * state_row_size,
                              in.state0.begin() + (verify_batch + 1) * state_row_size);
    const int verify_start = in.cu_seqlens[verify_batch];
    auto copy_tokens = [verify_start](const std::vector<float>& source, int token_width, int count) {
      const auto first = source.begin() + static_cast<ptrdiff_t>(verify_start) * token_width;
      return std::vector<float>(first, first + static_cast<ptrdiff_t>(count) * token_width);
    };

    for (int t = 0; t < capture_count; ++t) {
      for (int hv = 0; hv < g.hv; ++hv) {
        const int hk = hv * g.hq / g.hv;
        const size_t decay_head = static_cast<size_t>(t) * g.hv + hv;
        const size_t key_head = static_cast<size_t>(t) * g.hq + hk;
        const size_t delta_head = static_cast<size_t>(t) * g.hv + hv;
        float* state = replay.data() + static_cast<size_t>(hv) * state_head_size;
        for (int c = 0; c < g.dv; ++c) {
          for (int r = 0; r < g.dk; ++r) {
            const size_t i = static_cast<size_t>(c) * g.dk + r;
            state[i] = std::fma(
                row[decay_elements + key_head * g.dk + r],
                row[decay_elements + key_elements + delta_head * g.dv + c],
                state[i] * row[decay_head]);
          }
        }
      }

      Geometry prefix_geometry{t + 1, 1, g.hq, g.hv, g.dk, g.dv};
      Inputs prefix_inputs;
      prefix_inputs.q = copy_tokens(in.q, g.hq * g.dk, t + 1);
      prefix_inputs.k = copy_tokens(in.k, g.hq * g.dk, t + 1);
      prefix_inputs.v = copy_tokens(in.v, g.hv * g.dv, t + 1);
      prefix_inputs.decay = copy_tokens(in.decay, g.hv, t + 1);
      prefix_inputs.beta = copy_tokens(in.beta, g.hv, t + 1);
      prefix_inputs.state0.assign(in.state0.begin() + verify_batch * state_row_size,
                                  in.state0.begin() + (verify_batch + 1) * state_row_size);
      prefix_inputs.a_log = in.a_log;
      prefix_inputs.dt_bias = in.dt_bias;

      std::vector<OrtValue> prefix_fetches;
      RunCase(prefix_geometry, prefix_options, prefix_inputs, 2e-2f, 2e-2f, false,
              &prefix_fetches);
      ASSERT_EQ(prefix_fetches.size(), 3u);
      const float* expected = prefix_fetches[1].Get<Tensor>().Data<float>();
      for (size_t i = 0; i < state_row_size; ++i) {
        ASSERT_EQ(replay[i], expected[i])
            << "batch " << verify_batch << " captured prefix " << t + 1
            << " state element " << i;
      }
    }
  }
}

// initial_state and final_state may be the same allocation. Feeding a run's state output
// back in as the next run's input must reproduce one long run.
TEST(GatedDeltaNetTest, TwoCallContinuationMatchesSingleRun) {
  if (NeedSkipGatedDeltaNetTest()) return;
  constexpr int kFirst = 64, kSecond = 64;
  Geometry g_all{kFirst + kSecond, 1, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g_all, 41);

  std::vector<float> out_all, state_all;
  Reference(g_all, Options{}, in, &out_all, &state_all);

  Geometry g1{kFirst, 1, 1, 2, kDim, kDim};
  Inputs in1 = in;
  in1.q.resize(static_cast<size_t>(kFirst) * g1.hq * g1.dk);
  in1.k.resize(static_cast<size_t>(kFirst) * g1.hq * g1.dk);
  in1.v.resize(static_cast<size_t>(kFirst) * g1.hv * g1.dv);
  in1.decay.resize(static_cast<size_t>(kFirst) * g1.hv);
  in1.beta.resize(static_cast<size_t>(kFirst) * g1.hv);
  std::vector<float> out1, state1;
  Reference(g1, Options{}, in1, &out1, &state1);

  Geometry g2{kSecond, 1, 1, 2, kDim, kDim};
  Inputs in2;
  in2.q.assign(in.q.begin() + static_cast<size_t>(kFirst) * g1.hq * g1.dk, in.q.end());
  in2.k.assign(in.k.begin() + static_cast<size_t>(kFirst) * g1.hq * g1.dk, in.k.end());
  in2.v.assign(in.v.begin() + static_cast<size_t>(kFirst) * g1.hv * g1.dv, in.v.end());
  in2.decay.assign(in.decay.begin() + static_cast<size_t>(kFirst) * g1.hv, in.decay.end());
  in2.beta.assign(in.beta.begin() + static_cast<size_t>(kFirst) * g1.hv, in.beta.end());
  in2.state0 = state1;  // the first call's output state
  std::vector<float> out2, state2;
  Reference(g2, Options{}, in2, &out2, &state2);

  for (size_t i = 0; i < state_all.size(); ++i) {
    ASSERT_NEAR(state_all[i], state2[i], 1e-4f) << "state element " << i;
  }

  // And the operator agrees with the continuation on the device.
  RunCase(g2, Options{}, in2, 3e-2f, 3e-2f);
}

void RunAliasedStateIoBindingCase(int total_tokens, std::unique_ptr<IExecutionProvider> ep,
                                  const char* execution_provider_type, const Options& options = Options{}) {
  ASSERT_NE(ep, nullptr);

  Geometry geometry{total_tokens, 1, 1, 2, kDim, kDim};
  Inputs inputs = MakeInputs(geometry, static_cast<uint32_t>(total_tokens) + 163);
  std::vector<float> first_output, first_state;
  Reference(geometry, options, inputs, &first_output, &first_state);
  Inputs second_inputs = inputs;
  second_inputs.state0 = first_state;
  std::vector<float> expected_output, expected_state;
  Reference(geometry, options, second_inputs, &expected_output, &expected_state);

  std::unordered_map<std::string, int> domain_to_version = {{kMSDomain, 1}};
  std::vector<ONNX_NAMESPACE::FunctionProto> functions;
  auto model = std::make_unique<Model>(
      "gated_delta_net_alias", true, ModelMetaData(), PathString(),
      IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, functions,
      DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  auto& graph = model->MainGraph();
  std::vector<ONNX_NAMESPACE::TypeProto> types;
  types.reserve(10);
  auto tensor_type = [&](int elem_type, std::initializer_list<int64_t> dims) {
    types.emplace_back();
    auto* type = &types.back();
    type->mutable_tensor_type()->set_elem_type(elem_type);
    for (int64_t dim : dims) {
      type->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
    }
    return type;
  };
  auto& query_arg = graph.GetOrCreateNodeArg(
      "query", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                           {total_tokens, geometry.hq, geometry.dk}));
  auto& key_arg = graph.GetOrCreateNodeArg(
      "key", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                         {total_tokens, geometry.hq, geometry.dk}));
  auto& value_arg = graph.GetOrCreateNodeArg(
      "value", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                           {total_tokens, geometry.hv, geometry.dv}));
  auto& empty = graph.GetOrCreateNodeArg("", nullptr);
  auto& decay_arg = graph.GetOrCreateNodeArg(
      "decay", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                           {total_tokens, geometry.hv}));
  auto& beta_arg = graph.GetOrCreateNodeArg(
      "beta", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                          {total_tokens, geometry.hv}));
  auto& state_arg = graph.GetOrCreateNodeArg(
      "initial_state", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                   {1, geometry.hv, geometry.dv, geometry.dk}));
  auto& output_arg = graph.GetOrCreateNodeArg(
      "output", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16,
                            {total_tokens, geometry.hv, geometry.dv}));
  auto& final_state_arg = graph.GetOrCreateNodeArg(
      "final_state", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
                                 {1, geometry.hv, geometry.dv, geometry.dk}));
  const bool needs_decay = options.update_rule == "gated" || options.update_rule == "gated_delta";
  const bool needs_beta = options.update_rule == "delta" || options.update_rule == "gated_delta";
  std::vector<NodeArg*> node_inputs = {
      &query_arg, &key_arg, &value_arg, &empty, needs_decay ? &decay_arg : &empty,
      needs_beta ? &beta_arg : &empty, &state_arg};
  std::vector<NodeArg*> node_outputs = {&output_arg, &final_state_arg};
  auto& node = graph.AddNode("gdn", "GatedDeltaNet", "aliased recurrent state",
                             node_inputs, node_outputs, nullptr, kMSDomain);
  node.AddAttribute("update_rule", options.update_rule);
  node.SetExecutionProviderType(execution_provider_type);
  ASSERT_STATUS_OK(graph.Resolve());

  std::string serialized;
  ASSERT_TRUE(model->ToProto().SerializeToString(&serialized));
  std::stringstream model_stream(serialized);
  SessionOptions session_options;
  InferenceSession session(session_options, GetEnvironment());
  IExecutionProvider* ep_ptr = ep.get();
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(ep)));
  auto allocators = ep_ptr->CreatePreferredAllocators();
  const OrtMemoryInfo* gpu_info = nullptr;
  for (const auto& allocator : allocators) {
    if (allocator->Info().device.Type() == OrtDevice::GPU &&
        allocator->Info().mem_type == OrtMemTypeDefault) {
      gpu_info = &allocator->Info();
    }
  }
  ASSERT_NE(gpu_info, nullptr);
  const OrtMemoryInfo copied_gpu_info = *gpu_info;
  ASSERT_STATUS_OK(session.Load(model_stream));
  ASSERT_STATUS_OK(session.Initialize());
  auto gpu_allocator = session.GetAllocator(copied_gpu_info);
  auto cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  auto make_gpu = [&](const auto& data, MLDataType data_type, const TensorShape& shape) {
    using Element = typename std::decay_t<decltype(data)>::value_type;
    Tensor cpu(data_type, shape, const_cast<Element*>(data.data()), cpu_allocator->Info());
    Tensor gpu(data_type, shape, gpu_allocator);
    ORT_THROW_IF_ERROR(ep_ptr->GetDataTransfer()->CopyTensor(cpu, gpu));
    OrtValue result;
    Tensor::InitOrtValue(std::move(gpu), result);
    return result;
  };

  auto query_value = make_gpu(ToFloat16(inputs.q), DataTypeImpl::GetType<MLFloat16>(),
                              TensorShape({total_tokens, geometry.hq, geometry.dk}));
  auto key_value = make_gpu(ToFloat16(inputs.k), DataTypeImpl::GetType<MLFloat16>(),
                            TensorShape({total_tokens, geometry.hq, geometry.dk}));
  auto value_value = make_gpu(ToFloat16(inputs.v), DataTypeImpl::GetType<MLFloat16>(),
                              TensorShape({total_tokens, geometry.hv, geometry.dv}));
  auto decay_value = make_gpu(inputs.decay, DataTypeImpl::GetType<float>(),
                              TensorShape({total_tokens, geometry.hv}));
  auto beta_value = make_gpu(inputs.beta, DataTypeImpl::GetType<float>(),
                             TensorShape({total_tokens, geometry.hv}));
  auto state_value = make_gpu(inputs.state0, DataTypeImpl::GetType<float>(),
                              TensorShape({1, geometry.hv, geometry.dv, geometry.dk}));
  auto output_value = make_gpu(
      std::vector<MLFloat16>(static_cast<size_t>(total_tokens) * geometry.hv * geometry.dv),
      DataTypeImpl::GetType<MLFloat16>(),
      TensorShape({total_tokens, geometry.hv, geometry.dv}));

  std::unique_ptr<IOBinding> binding;
  ASSERT_STATUS_OK(session.NewIOBinding(&binding));
  ASSERT_STATUS_OK(binding->BindInput("query", query_value));
  ASSERT_STATUS_OK(binding->BindInput("key", key_value));
  ASSERT_STATUS_OK(binding->BindInput("value", value_value));
  if (needs_decay) {
    ASSERT_STATUS_OK(binding->BindInput("decay", decay_value));
  }
  if (needs_beta) {
    ASSERT_STATUS_OK(binding->BindInput("beta", beta_value));
  }
  ASSERT_STATUS_OK(binding->BindInput("initial_state", state_value));
  ASSERT_STATUS_OK(binding->BindOutput("output", output_value));
  ASSERT_STATUS_OK(binding->BindOutput("final_state", state_value));
  RunOptions run_options;
  ASSERT_STATUS_OK(session.Run(run_options, *binding));
  ASSERT_STATUS_OK(session.Run(run_options, *binding));
  ASSERT_EQ(binding->GetOutputs().size(), 2u);
  EXPECT_EQ(binding->GetOutputs()[1].Get<Tensor>().Data<float>(),
            state_value.Get<Tensor>().Data<float>());

  std::vector<float> actual_state(expected_state.size());
  Tensor cpu_state(DataTypeImpl::GetType<float>(),
                   TensorShape({1, geometry.hv, geometry.dv, geometry.dk}),
                   actual_state.data(), cpu_allocator->Info());
  ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(state_value.Get<Tensor>(), cpu_state));
  for (size_t i = 0; i < actual_state.size(); ++i) {
    EXPECT_NEAR(actual_state[i], expected_state[i], 4e-2f) << "state element " << i;
  }

  std::vector<MLFloat16> actual_output(expected_output.size());
  Tensor cpu_output(DataTypeImpl::GetType<MLFloat16>(),
                    TensorShape({total_tokens, geometry.hv, geometry.dv}),
                    actual_output.data(), cpu_allocator->Info());
  ASSERT_STATUS_OK(ep_ptr->GetDataTransfer()->CopyTensor(output_value.Get<Tensor>(), cpu_output));
  for (size_t i = 0; i < actual_output.size(); ++i) {
    EXPECT_NEAR(actual_output[i].ToFloat(), expected_output[i], 4e-2f) << "output element " << i;
  }
}

TEST(GatedDeltaNetTest, AliasedStateIoBindingRecurrentAndChunked) {
  if (NeedSkipGatedDeltaNetTest()) return;
  RunAliasedStateIoBindingCase(/*total_tokens=*/4, DefaultCudaExecutionProvider(), kCudaExecutionProvider);
  RunAliasedStateIoBindingCase(/*total_tokens=*/64, DefaultCudaExecutionProvider(), kCudaExecutionProvider);
}

TEST(GatedDeltaNetWebGpuTest, AliasedStateIoBinding) {
  WebGpuExecutionProviderTestOptions test_options{};
  test_options.max_storage_buffer_binding_size = 64 * 1024;
  auto webgpu_ep = WebGpuExecutionProviderWithTestOptions(test_options);
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU execution provider is not available";
  }
  Options options;
  options.update_rule = "linear";
  RunAliasedStateIoBindingCase(/*total_tokens=*/1, std::move(webgpu_ep), kWebGpuExecutionProvider, options);
}

// Device-supplied offsets must not be able to steer an out-of-bounds access.
TEST(GatedDeltaNetTest, MalformedCuSeqlensIsClamped) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{64, 2, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 43);

  for (const std::vector<int32_t>& bad :
       {std::vector<int32_t>{0, -8, 64},       // negative
        std::vector<int32_t>{0, 48, 16},       // decreasing
        std::vector<int32_t>{0, 32, 4096}}) {  // end beyond total_tokens
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain, /*verify_output=*/false);
    Options o;
    AddCommonAttrs(test, o);
    test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
    test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
    test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
    test.AddInput<int32_t>("cu_seqlens", {3}, bad);
    test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
    test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
    // Values are unspecified for malformed offsets; the contract is only that the run is
    // memory safe and does not fault.
    test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                              ToFloat16(std::vector<float>(
                                  static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)),
                              false, 1e9f, 1e9f);
    test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk},
                          std::vector<float>(in.state0.size(), 0.0f), false, 1e9f, 1e9f);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(GatedDeltaNetTest, RejectsNonDivisiblePackedQkvUniformBatch) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{5, 2, 1, 2, kDim, kDim};
  Inputs in = MakeInputs(g, 46);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  o.update_rule = "linear";
  AddCommonAttrs(test, o);
  const int packed_size = 2 * g.hq * g.dk + g.hv * g.dv;
  test.AddInput<MLFloat16>("query", {g.total_tokens, packed_size},
                           ToFloat16(std::vector<float>(static_cast<size_t>(g.total_tokens) * packed_size, 0.0f)));
  test.AddOptionalInputEdge<MLFloat16>();
  test.AddOptionalInputEdge<MLFloat16>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                            ToFloat16(std::vector<float>(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "must be divisible by batch", {}, nullptr, &eps);
}

TEST(GatedDeltaNetTest, RejectsMismatchedHeadCounts) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 2, 5, 64, 64};  // hv=5 is not a multiple of hq=2
  Inputs in = MakeInputs(g, 47);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  AddCommonAttrs(test, o);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                            ToFloat16(std::vector<float>(
                                static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "must be a positive multiple", {}, nullptr,
           &eps);
}

TEST(GatedDeltaNetTest, RejectsMissingRequiredGateInput) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 1, 64, 64};
  Inputs in = MakeInputs(g, 51);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  AddCommonAttrs(test, o);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddOptionalInputEdge<float>();
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOutput<MLFloat16>(
      "output", {g.total_tokens, g.hv, g.dv},
      ToFloat16(std::vector<float>(
          static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "beta input presence must match update_rule", {}, nullptr, &eps);
}

TEST(GatedDeltaNetTest, RejectsPerKeyDtBias) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 1, 64, 64};
  Inputs in = MakeInputs(g, 53);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options o;
  o.gate_activation = "qwen";
  AddCommonAttrs(test, o);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddInput<float>("a_log", {g.hv}, in.a_log);
  test.AddInput<float>("dt_bias", {g.hv, g.dk},
                       std::vector<float>(static_cast<size_t>(g.hv) * g.dk, 0.0f));
  test.AddOutput<MLFloat16>(
      "output", {g.total_tokens, g.hv, g.dv},
      ToFloat16(std::vector<float>(
          static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "dt_bias must be [num_heads_v]",
           {}, nullptr, &eps);
}

// This test exercises shape inference which uses fail_shape_inference (throws InferenceError).
// In no-exception builds, fail_shape_inference calls abort(), so this test must be skipped.
#ifndef ORT_NO_EXCEPTIONS
TEST(GatedDeltaNetTest, RejectsStateUpdateWidthOverflow) {
  struct Case {
    std::array<int64_t, 3> query_dims;
    std::array<int64_t, 3> value_dims;
    int64_t state_update_capacity;
  };

  const int64_t max_dimension = std::numeric_limits<int64_t>::max();
  const int64_t large_head_size = 4000000000LL;
  const std::vector<Case> cases = {
      // num_heads_k * head_size_qk.
      {{1, large_head_size, large_head_size}, {1, 1, 1}, 1},
      // num_heads_v * head_size_v.
      {{1, 1, 1}, {1, large_head_size, large_head_size}, 1},
      // num_heads_v + (num_heads_k * head_size_qk).
      {{1, 1, max_dimension}, {1, 1, 1}, 1},
      // value_width + (num_heads_v + key_width).
      {{1, 1, 1}, {1, 1, max_dimension}, 1},
      // state_update_capacity * per_token_width.
      {{1, 1, 1}, {1, 1, max_dimension / 8}, 8},
  };

  for (size_t case_index = 0; case_index < cases.size(); ++case_index) {
    SCOPED_TRACE(case_index);
    const auto& test_case = cases[case_index];
    std::unordered_map<std::string, int> domain_to_version = {{kMSDomain, 1}};
    std::vector<ONNX_NAMESPACE::FunctionProto> functions;
    auto model = std::make_unique<Model>(
        "gated_delta_net_overflow", true, ModelMetaData(), PathString(),
        IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, functions,
        DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
    auto& graph = model->MainGraph();

    std::vector<ONNX_NAMESPACE::TypeProto> types;
    types.reserve(6);
    auto tensor_type = [&](int elem_type, const auto& dims) {
      types.emplace_back();
      auto* type = &types.back();
      type->mutable_tensor_type()->set_elem_type(elem_type);
      for (int64_t dim : dims) {
        type->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
      }
      return type;
    };

    auto& query_arg = graph.GetOrCreateNodeArg(
        "query", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, test_case.query_dims));
    auto& key_arg = graph.GetOrCreateNodeArg(
        "key", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, std::array<int64_t, 3>{1, 1, 1}));
    auto& value_arg = graph.GetOrCreateNodeArg(
        "value", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, test_case.value_dims));
    auto& output_arg = graph.GetOrCreateNodeArg(
        "output", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, std::array<int64_t, 0>{}));
    auto& final_state_arg = graph.GetOrCreateNodeArg(
        "final_state", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, std::array<int64_t, 0>{}));
    auto& state_update_arg = graph.GetOrCreateNodeArg(
        "state_update", tensor_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT, std::array<int64_t, 0>{}));

    auto& node = graph.AddNode("node", "GatedDeltaNet", "", {&query_arg, &key_arg, &value_arg},
                               {&output_arg, &final_state_arg, &state_update_arg}, nullptr, kMSDomain);
    node.AddAttribute("state_update_capacity", test_case.state_update_capacity);

    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(graph.Resolve(), "overflows int64");
  }
}
#endif  // !ORT_NO_EXCEPTIONS

TEST(GatedDeltaNetTest, RequiresCaptureCountExactlyWhenCapacityIsPositive) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 2, 64, 32};
  Inputs in = MakeInputs(g, 57);

  auto add_base = [&](OpTester& test, int capacity) {
    Options o;
    o.state_update_capacity = capacity;
    AddCommonAttrs(test, o);
    test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
    test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
    test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
    test.AddOptionalInputEdge<int32_t>();
    test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
    test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
    test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  };
  auto add_output = [&](OpTester& test) {
    test.AddOutput<MLFloat16>(
        "output", {g.total_tokens, g.hv, g.dv},
        ToFloat16(std::vector<float>(static_cast<size_t>(g.total_tokens) * g.hv * g.dv, 0.0f)));
  };

  {
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    add_base(test, /*capacity=*/4);
    add_output(test);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "capture_count must be present exactly when state_update_capacity is positive",
             {}, nullptr, &eps);
  }

  {
    OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
    add_base(test, /*capacity=*/0);
    test.AddOptionalInputEdge<float>();
    test.AddOptionalInputEdge<float>();
    test.AddInput<int32_t>("capture_count", {g.batch}, {1});
    add_output(test);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             "capture_count must be present exactly when state_update_capacity is positive",
             {}, nullptr, &eps);
  }
}

TEST(GatedDeltaNetTest, InactiveStateUpdateIgnoresCountsAndClearsOutput) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{4, 1, 1, 2, 64, 32};
  Options options;
  options.state_update_capacity = 4;

  for (int capture_count : {0, 2}) {
    SCOPED_TRACE(capture_count);
    Inputs inputs = MakeInputs(g, 151 + capture_count);
    inputs.capture_count = {capture_count};
    inputs.state_update_active = {0};
    std::vector<OrtValue> fetches;
    RunCase(g, options, inputs, 2e-2f, 2e-2f, false, &fetches);
    ASSERT_EQ(fetches.size(), 3u);
    const Tensor& state_update = fetches[2].Get<Tensor>();
    const float* data = state_update.Data<float>();
    for (int64_t i = 0; i < state_update.Shape().Size(); ++i) {
      EXPECT_EQ(data[i], 0.0f) << "state_update element " << i;
    }
  }
}

TEST(GatedDeltaNetTest, StateUpdateCapacityIsBounded) {
  if (NeedSkipGatedDeltaNetTest()) return;
  Geometry g{1, 1, 1, 1, 64, 32};
  Inputs in = MakeInputs(g, 81);
  OpTester test("GatedDeltaNet", 1, onnxruntime::kMSDomain);
  Options options;
  options.state_update_capacity = 9;
  AddCommonAttrs(test, options);
  test.AddInput<MLFloat16>("query", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.q));
  test.AddInput<MLFloat16>("key", {g.total_tokens, g.hq, g.dk}, ToFloat16(in.k));
  test.AddInput<MLFloat16>("value", {g.total_tokens, g.hv, g.dv}, ToFloat16(in.v));
  test.AddOptionalInputEdge<int32_t>();
  test.AddInput<float>("decay", {g.total_tokens, g.hv}, in.decay);
  test.AddInput<float>("beta", {g.total_tokens, g.hv}, in.beta);
  test.AddInput<float>("initial_state", {g.batch, g.hv, g.dv, g.dk}, in.state0);
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddInput<int32_t>("capture_count", {g.batch}, {1});
  test.AddOutput<MLFloat16>("output", {g.total_tokens, g.hv, g.dv},
                            std::vector<MLFloat16>(static_cast<size_t>(g.total_tokens) * g.hv * g.dv));
  test.AddOutput<float>("final_state", {g.batch, g.hv, g.dv, g.dk},
                        std::vector<float>(in.state0.size()));
  const int64_t width = 9 * (g.hv + g.hq * g.dk + g.hv * g.dv);
  test.AddOutput<float>("state_update", {g.batch, width},
                        std::vector<float>(static_cast<size_t>(g.batch) * width));
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "state_update_capacity must be in [0, 8]");
}

// The plan decision is pure host code, so the consumer-Blackwell shared-memory budget can
// be checked without an SM120 device. CUTLASS records sm120_smem_capacity_bytes = 101376
// against SM90/SM100's 232448.
TEST(GatedDeltaNetPlanTest, PicksNarrowChunkOnConsumerBlackwellSharedMemory) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor d{};
  d.total_tokens = 1024;
  d.batch = 1;
  d.num_heads_q = 16;
  d.num_heads_k = 16;
  d.num_heads_v = 48;
  d.head_size_qk = 128;
  d.head_size_v = 128;
  d.chunk_size = 64;
  d.io_type = gdn::IoType::kFloat16;
  d.sm_major = 12;
  // Pin the fused engine so this stays a test of chunk sizing. The automatic rule would
  // otherwise route this shape to the split engine; that choice is covered below.
  d.preferred_engine = gdn::Engine::kChunked;

  const gdn::Plan blackwell = gdn::SelectPlan(d, 101376);
  EXPECT_TRUE(blackwell.supported);
  EXPECT_EQ(blackwell.engine, gdn::Engine::kChunked) << "must not fall back to the scalar engine";
  EXPECT_EQ(blackwell.chunk_size, 32);
  EXPECT_LE(blackwell.smem_bytes, 101376u);

  const gdn::Plan large_smem = gdn::SelectPlan(d, 232448);
  EXPECT_EQ(large_smem.engine, gdn::Engine::kChunked);
  EXPECT_EQ(large_smem.chunk_size, 64) << "the wider chunk is faster where it fits";

  // A device too small for either chunk must degrade to the sequential engine, not fail.
  const gdn::Plan tiny = gdn::SelectPlan(d, 48 * 1024);
  EXPECT_TRUE(tiny.supported);
  EXPECT_EQ(tiny.engine, gdn::Engine::kRecurrent);
}

TEST(GatedDeltaNetPlanTest, BFloat16UsesRecurrentEngine) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor d{};
  d.total_tokens = 64;
  d.batch = 1;
  d.num_heads_q = 1;
  d.num_heads_k = 1;
  d.num_heads_v = 2;
  d.head_size_qk = 128;
  d.head_size_v = 128;
  d.io_type = gdn::IoType::kBFloat16;
  d.sm_major = 8;

  const auto plan = gdn::SelectPlan(d, 160 * 1024);
  EXPECT_TRUE(plan.supported);
  EXPECT_EQ(plan.engine, gdn::Engine::kChunked);

  d.sm_major = 7;
  const auto pre_ampere_plan = gdn::SelectPlan(d, 160 * 1024);
  EXPECT_TRUE(pre_ampere_plan.supported);
  EXPECT_EQ(pre_ampere_plan.engine, gdn::Engine::kRecurrent);

  d.sm_major = 8;
  d.io_type = gdn::IoType::kFloat;
  const auto float_plan = gdn::SelectPlan(d, 160 * 1024);
  EXPECT_TRUE(float_plan.supported);
  EXPECT_EQ(float_plan.engine, gdn::Engine::kRecurrent);
}

TEST(GatedDeltaNetPlanTest, RecurrentSharedMemoryIncludesBothNormalizationScalars) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  EXPECT_EQ(gdn::RecurrentSmemBytes(96, 1), sizeof(float) * (96 + 2 * 96 + 2 + 96));
}

TEST(GatedDeltaNetPlanTest, AutomaticSelectionKeepsTheFusedEngine) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor base{};
  base.num_heads_q = 16;
  base.num_heads_k = 16;
  base.num_heads_v = 48;
  base.head_size_qk = 128;
  base.head_size_v = 128;
  base.chunk_size = 64;
  base.io_type = gdn::IoType::kFloat16;
  base.sm_major = 9;

  auto engine_for = [&base](int batch, int64_t tokens_per_seq) {
    gdn::Descriptor d = base;
    d.batch = batch;
    d.total_tokens = tokens_per_seq * batch;
    return gdn::SelectPlan(d, 232448).engine;
  };

  EXPECT_EQ(engine_for(1, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(2, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(3, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(4, 1024), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(8, 1024), gdn::Engine::kChunked);

  EXPECT_EQ(engine_for(1, 64), gdn::Engine::kChunked);
  EXPECT_EQ(engine_for(1, 128), gdn::Engine::kChunked);

  gdn::Descriptor wide = base;
  wide.batch = 4;
  wide.total_tokens = 4096;
  EXPECT_EQ(gdn::SelectPlan(wide, 232448).engine, gdn::Engine::kChunked);

  // Explicit requests retain both engines for diagnostics and benchmarking.
  gdn::Descriptor forced = base;
  forced.batch = 8;
  forced.total_tokens = 8 * 1024;
  forced.preferred_engine = gdn::Engine::kChunkedSplit;
  EXPECT_EQ(gdn::SelectPlan(forced, 232448).engine, gdn::Engine::kChunkedSplit);
  forced.batch = 1;
  forced.total_tokens = 1024;
  forced.preferred_engine = gdn::Engine::kChunked;
  EXPECT_EQ(gdn::SelectPlan(forced, 232448).engine, gdn::Engine::kChunked);
}

// The split engine's whole point is that its scan can be narrow: the state-independent work
// has been hoisted into the prepare launch, so every one of the scan's GEMMs scales with the
// v-block. Two CTAs of it must therefore fit an SM, and the workspace must stay bounded no
// matter how long the sequence is.
TEST(GatedDeltaNetPlanTest, SplitEngineIsTwoCtasPerSmAndBoundedWorkspace) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor d{};
  d.total_tokens = 1024;
  d.batch = 1;
  d.num_heads_q = 16;
  d.num_heads_k = 16;
  d.num_heads_v = 48;
  d.head_size_qk = 128;
  d.head_size_v = 128;
  d.chunk_size = 64;
  d.io_type = gdn::IoType::kFloat16;
  d.sm_major = 9;
  d.preferred_engine = gdn::Engine::kChunkedSplit;

  const gdn::Plan hopper = gdn::SelectPlan(d, 232448);
  EXPECT_TRUE(hopper.supported);
  EXPECT_EQ(hopper.engine, gdn::Engine::kChunkedSplit);
  EXPECT_EQ(hopper.v_block, 32);
  EXPECT_LE(2 * hopper.smem_bytes, 232448u);
  EXPECT_LE(hopper.smem_bytes_prepare, 232448u);
  EXPECT_GT(hopper.workspace_bytes, 0u);

  // Sixteen times the tokens must not mean sixteen times the workspace.
  const size_t short_ws = hopper.workspace_bytes;
  d.total_tokens = 16384;
  const gdn::Plan lng = gdn::SelectPlan(d, 232448);
  EXPECT_EQ(lng.engine, gdn::Engine::kChunkedSplit);
  EXPECT_LE(lng.workspace_bytes, 64u << 20);
  EXPECT_LT(lng.workspace_bytes, 16 * short_ws);

  // Where the scan does not fit, the request degrades to the fused engine rather than failing.
  d.total_tokens = 1024;
  d.sm_major = 12;
  const gdn::Plan blackwell = gdn::SelectPlan(d, 101376);
  EXPECT_TRUE(blackwell.supported);
  EXPECT_EQ(blackwell.engine, gdn::Engine::kChunked);
}

TEST(GatedDeltaNetPlanTest, CompactCaptureUsesDeviceCountRecurrentTail) {
  namespace gdn = onnxruntime::contrib::cuda::gated_delta_net;
  gdn::Descriptor d{};
  d.total_tokens = 1024;
  d.batch = 2;
  d.num_heads_q = 1;
  d.num_heads_k = 1;
  d.num_heads_v = 2;
  d.head_size_qk = 128;
  d.head_size_v = 128;
  d.chunk_size = 64;
  d.state_update_capacity = 4;
  d.state_update_active = true;
  d.io_type = gdn::IoType::kFloat16;
  d.ragged = true;
  d.sm_major = 9;

  const gdn::Plan plan = gdn::SelectPlan(d, 232448);
  EXPECT_EQ(plan.engine, gdn::Engine::kChunked);
  EXPECT_TRUE(plan.state_update_tail_pass);

  d.state_update_active = false;
  d.num_heads_q = 16;
  d.num_heads_k = 16;
  d.num_heads_v = 48;
  d.batch = 1;
  const gdn::Plan inactive = gdn::SelectPlan(d, 232448);
  EXPECT_EQ(inactive.engine, gdn::Engine::kChunked);
  EXPECT_FALSE(inactive.state_update_tail_pass);
  EXPECT_TRUE(inactive.short_row_tail_pass);
}

}  // namespace test
}  // namespace onnxruntime

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/providers/cuda/cuda_common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace onnxruntime {
namespace test {
namespace {

struct CudaBuffer {
  void* data = nullptr;
  size_t bytes = 0;

  explicit CudaBuffer(size_t size_in_bytes) : bytes(size_in_bytes) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~CudaBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  template <typename T>
  T* As() {
    return reinterpret_cast<T*>(data);
  }

  void CopyFromHost(const void* src) {
    CUDA_CALL_THROW(cudaMemcpy(data, src, bytes, cudaMemcpyHostToDevice));
  }

  void CopyToHost(void* dst) const {
    CUDA_CALL_THROW(cudaMemcpy(dst, data, bytes, cudaMemcpyDeviceToHost));
  }
};

struct ExpectedTopK {
  std::vector<float> scales;
  std::vector<int> indices;
};

template <typename T>
float ToFloat(T value) {
  return static_cast<float>(value);
}

template <typename T>
T FromFloat(float value) {
  return static_cast<T>(value);
}

template <typename T>
ExpectedTopK ReferenceSoftmaxTopK(const std::vector<T>& logits, int num_rows, int num_experts, int k,
                                  bool normalize_scales) {
  ExpectedTopK expected;
  expected.scales.resize(static_cast<size_t>(num_rows) * k);
  expected.indices.resize(static_cast<size_t>(num_rows) * k);

  for (int row = 0; row < num_rows; ++row) {
    const T* row_logits = logits.data() + static_cast<size_t>(row) * num_experts;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int expert = 0; expert < num_experts; ++expert) {
      max_logit = std::max(max_logit, ToFloat(row_logits[expert]));
    }

    float sum_exp = 0.0f;
    for (int expert = 0; expert < num_experts; ++expert) {
      sum_exp += std::exp(ToFloat(row_logits[expert]) - max_logit);
    }

    std::vector<int> order(num_experts);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      const float lhs_logit = ToFloat(row_logits[lhs]);
      const float rhs_logit = ToFloat(row_logits[rhs]);
      return lhs_logit > rhs_logit || (lhs_logit == rhs_logit && lhs < rhs);
    });

    float topk_sum = 0.0f;
    for (int rank = 0; rank < k; ++rank) {
      const int expert = order[rank];
      const float scale = std::exp(ToFloat(row_logits[expert]) - max_logit) / sum_exp;
      expected.scales[static_cast<size_t>(row) * k + rank] = scale;
      expected.indices[static_cast<size_t>(row) * k + rank] = expert;
      topk_sum += scale;
    }

    if (normalize_scales && topk_sum > 1e-6f) {
      for (int rank = 0; rank < k; ++rank) {
        expected.scales[static_cast<size_t>(row) * k + rank] /= topk_sum;
      }
    }
  }

  return expected;
}

template <typename T>
void RunSoftmaxTopKTest(int num_rows, int num_experts, int k, bool normalize_scales,
                        const std::vector<float>& logits_float, float tolerance) {
  ASSERT_EQ(logits_float.size(), static_cast<size_t>(num_rows) * num_experts);

  std::vector<T> logits(logits_float.size());
  std::transform(logits_float.begin(), logits_float.end(), logits.begin(), [](float value) {
    return FromFloat<T>(value);
  });

  CudaBuffer d_logits(logits.size() * sizeof(T));
  CudaBuffer d_scales(static_cast<size_t>(num_rows) * k * sizeof(float));
  CudaBuffer d_indices(static_cast<size_t>(num_rows) * k * sizeof(int));
  d_logits.CopyFromHost(logits.data());

  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  onnxruntime::contrib::cuda::LaunchSoftmaxTopK(
      d_logits.As<T>(), d_scales.As<float>(), d_indices.As<int>(), num_rows, num_experts, k, normalize_scales, stream);
  CUDA_CALL_THROW(cudaGetLastError());
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));

  std::vector<float> actual_scales(static_cast<size_t>(num_rows) * k);
  std::vector<int> actual_indices(static_cast<size_t>(num_rows) * k);
  d_scales.CopyToHost(actual_scales.data());
  d_indices.CopyToHost(actual_indices.data());

  const ExpectedTopK expected = ReferenceSoftmaxTopK(logits, num_rows, num_experts, k, normalize_scales);
  ASSERT_EQ(actual_indices, expected.indices);
  ASSERT_EQ(actual_scales.size(), expected.scales.size());
  for (size_t i = 0; i < actual_scales.size(); ++i) {
    EXPECT_NEAR(actual_scales[i], expected.scales[i], tolerance) << "at flattened top-k index " << i;
  }
}

std::vector<float> MakeLogits(int num_rows, int num_experts) {
  std::vector<float> logits(static_cast<size_t>(num_rows) * num_experts);
  for (int row = 0; row < num_rows; ++row) {
    for (int expert = 0; expert < num_experts; ++expert) {
      const int v = (expert * 17 + row * 11) % 29;
      logits[static_cast<size_t>(row) * num_experts + expert] = 0.125f * static_cast<float>(v - 14);
    }
  }
  return logits;
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpBitonicStableTiesFloat) {
  constexpr int num_rows = 9;
  constexpr int num_experts = 8;
  constexpr int k = 4;
  std::vector<float> logits = {
      4.0f, 1.0f, 4.0f, 0.0f, 3.0f, 4.0f, -2.0f, 4.0f,
      -1.0f, 2.0f, 2.0f, 5.0f, 5.0f, 5.0f, 0.5f, -3.0f,
      0.0f, 0.0f, 0.0f, -1.0f, -1.0f, 2.0f, 2.0f, 2.0f,
      3.0f, 1.0f, 2.0f, 3.0f, 3.0f, -4.0f, -5.0f, 0.0f,
      1.5f, 1.5f, 1.0f, 2.0f, 0.0f, 2.0f, 2.0f, -2.0f,
      -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f, 0.0f, 1.0f,
      7.0f, 7.0f, 6.0f, 6.0f, 5.0f, 5.0f, 4.0f, 4.0f,
      -1.0f, 8.0f, -1.0f, 8.0f, 3.0f, 8.0f, 2.0f, 1.0f,
      0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, false, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpBitonicBoundaryNormalizeHalf) {
  constexpr int num_rows = 10;
  constexpr int num_experts = 32;
  constexpr int k = 8;
  auto logits = MakeLogits(num_rows, num_experts);
  logits[3 * num_experts + 0] = 6.0f;
  logits[3 * num_experts + 5] = 6.0f;
  logits[3 * num_experts + 17] = 6.0f;

  RunSoftmaxTopKTest<half>(num_rows, num_experts, k, true, logits, 2e-3f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpMergeStableTiesFloat) {
  constexpr int num_rows = 5;
  constexpr int num_experts = 64;
  constexpr int k = 8;
  auto logits = MakeLogits(num_rows, num_experts);
  for (int expert : {2, 13, 31, 63}) {
    logits[static_cast<size_t>(1) * num_experts + expert] = 5.0f;
  }
  for (int expert : {0, 16, 32, 48}) {
    logits[static_cast<size_t>(4) * num_experts + expert] = 4.5f;
  }

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, false, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpMergeNormalizeBFloat16) {
  constexpr int num_rows = 6;
  constexpr int num_experts = 64;
  constexpr int k = 16;
  auto logits = MakeLogits(num_rows, num_experts);
  logits[2 * num_experts + 4] = 7.0f;
  logits[2 * num_experts + 9] = 7.0f;
  logits[2 * num_experts + 47] = 7.0f;

  RunSoftmaxTopKTest<__nv_bfloat16>(num_rows, num_experts, k, true, logits, 2e-2f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_BlockMergeStillMatchesReference) {
  constexpr int num_rows = 4;
  constexpr int num_experts = 128;
  constexpr int k = 8;
  auto logits = MakeLogits(num_rows, num_experts);
  logits[0] = 9.0f;
  logits[7] = 9.0f;
  logits[65] = 9.0f;

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, true, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpBitonicHandlesNegativeInfinityPadding) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 8;
  constexpr int k = 8;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits = {
      3.0f, neg_inf, 2.0f, neg_inf, 1.0f, neg_inf, 0.0f, neg_inf,
      neg_inf, 4.0f, neg_inf, neg_inf, 3.0f, neg_inf, neg_inf, 2.0f};

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, false, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_WarpMergeHandlesNegativeInfinityPadding) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 40;
  constexpr int k = 40;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits(static_cast<size_t>(num_rows) * num_experts, neg_inf);
  for (int expert = 0; expert < 10; ++expert) {
    logits[expert * 3] = static_cast<float>(expert) * 0.25f;
    logits[static_cast<size_t>(1) * num_experts + expert * 2] = 3.0f - static_cast<float>(expert) * 0.125f;
  }

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, true, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_BlockMergeHandlesNegativeInfinityPadding) {
  constexpr int num_rows = 2;
  constexpr int num_experts = 80;
  constexpr int k = 64;
  constexpr float neg_inf = -std::numeric_limits<float>::infinity();
  std::vector<float> logits(static_cast<size_t>(num_rows) * num_experts, neg_inf);
  for (int expert = 0; expert < 20; ++expert) {
    logits[expert * 2] = static_cast<float>(expert) * 0.125f;
    logits[static_cast<size_t>(1) * num_experts + expert * 3] = 5.0f - static_cast<float>(expert) * 0.25f;
  }

  RunSoftmaxTopKTest<float>(num_rows, num_experts, k, true, logits, 1e-6f);
}

TEST(CUDA_EP_Unittest, SoftmaxTopK_RejectsKGreaterThanExperts) {
  EXPECT_THROW(onnxruntime::contrib::cuda::LaunchSoftmaxTopK(
                   static_cast<const float*>(nullptr), nullptr, nullptr, 1, 8, 9, false, nullptr),
               OnnxRuntimeException);
}

template <typename T>
void RunFusedRoutingMapsTest(int rows, int experts, int k, bool ties = false, bool nonfinite = false,
                             const std::vector<float>* custom_values = nullptr) {
  auto values = custom_values ? *custom_values : MakeLogits(rows, experts);
  ASSERT_EQ(values.size(), static_cast<size_t>(rows) * experts);
  if (ties) std::fill(values.begin(), values.end(), 0.0f);
  if (nonfinite) {
    std::fill(values.begin(), values.end(), -std::numeric_limits<float>::infinity());
    if (rows == 2) values[experts] = std::numeric_limits<float>::infinity();
  }
  std::vector<T> logits(values.size());
  std::transform(values.begin(), values.end(), logits.begin(), [](float v) { return FromFloat<T>(v); });
  const int expanded = rows * k;
  CudaBuffer d_logits(logits.size() * sizeof(T));
  CudaBuffer d_scales(expanded * sizeof(float)), d_reference_scales(expanded * sizeof(float));
  CudaBuffer d_indices(expanded * sizeof(int)), d_reference_indices(expanded * sizeof(int));
  CudaBuffer d_r2s(expanded * sizeof(int)), d_s2r(expanded * sizeof(int));
  CudaBuffer d_sorted(expanded * sizeof(int)), d_offsets((experts + 1) * sizeof(int64_t));
  d_logits.CopyFromHost(logits.data());
  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  contrib::cuda::LaunchSoftmaxTopK(d_logits.As<T>(), d_reference_scales.As<float>(),
                                   d_reference_indices.As<int>(), rows, experts, k, true, stream);
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  const bool launched = contrib::cuda::TryLaunchSoftmaxTopKExpertMaps(
      d_logits.As<T>(), d_scales.As<float>(), d_indices.As<int>(), d_r2s.As<int>(),
      d_s2r.As<int>(), d_sorted.As<int>(), d_offsets.As<int64_t>(), rows, experts, k, true, stream);
  CUDA_CALL_THROW(cudaStreamEndCapture(stream, &graph));
  ASSERT_TRUE(launched);
  size_t node_count = 0;
  CUDA_CALL_THROW(cudaGraphGetNodes(graph, nullptr, &node_count));
  EXPECT_EQ(node_count, 1u);
  CUDA_CALL_THROW(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  std::vector<int> actual_indices(expanded), reference_indices(expanded);
  std::vector<float> actual_scales(expanded), reference_scales(expanded);
  std::vector<int> r2s(expanded), s2r(expanded), sorted(expanded);
  std::vector<int64_t> offsets(experts + 1);
  d_reference_indices.CopyToHost(reference_indices.data());
  d_reference_scales.CopyToHost(reference_scales.data());
  std::vector<int> order(expanded);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return reference_indices[a] < reference_indices[b];
  });
  for (int replay = 0; replay < 3; ++replay) {
    CUDA_CALL_THROW(cudaGraphLaunch(executable, stream));
    CUDA_CALL_THROW(cudaStreamSynchronize(stream));
    d_indices.CopyToHost(actual_indices.data());
    d_scales.CopyToHost(actual_scales.data());
    d_r2s.CopyToHost(r2s.data());
    d_s2r.CopyToHost(s2r.data());
    d_sorted.CopyToHost(sorted.data());
    d_offsets.CopyToHost(offsets.data());
    EXPECT_EQ(actual_indices, reference_indices);
    for (int i = 0; i < expanded; ++i) {
      EXPECT_NEAR(actual_scales[i], reference_scales[i], 2e-6f);
      const int original = order[i];
      const int source = original % k * rows + original / k;
      EXPECT_EQ(r2s[i], source);
      EXPECT_EQ(s2r[source], i);
      EXPECT_EQ(sorted[i], reference_indices[original]);
    }
    for (int expert = 0; expert <= experts; ++expert) {
      EXPECT_EQ(offsets[expert], std::count_if(reference_indices.begin(), reference_indices.end(),
                                               [&](int selected) { return selected < expert; }));
    }
  }
  CUDA_CALL_THROW(cudaGraphExecDestroy(executable));
  CUDA_CALL_THROW(cudaGraphDestroy(graph));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
}

TEST(CUDA_EP_Unittest, FusedSoftmaxTopKExpertMaps_NormalizedAndGraphReplay) {
  for (int rows : {1, 2}) {
    for (int experts : {1, 7, 32, 65, 256, 511, 512}) {
      SCOPED_TRACE(::testing::Message() << "rows=" << rows << " experts=" << experts);
      const int k = std::min(10, experts);
      RunFusedRoutingMapsTest<float>(rows, experts, k);
      RunFusedRoutingMapsTest<half>(rows, experts, k);
      RunFusedRoutingMapsTest<__nv_bfloat16>(rows, experts, k);
    }
    RunFusedRoutingMapsTest<__nv_bfloat16>(rows, 512, 16, true);
    RunFusedRoutingMapsTest<float>(rows, 80, 16, false, true);
  }
}

TEST(CUDA_EP_Unittest, FusedSoftmaxTopKExpertMaps_UnsupportedUsesFallback) {
  auto launch = [](int rows, int experts, int k, bool normalize) {
    return contrib::cuda::TryLaunchSoftmaxTopKExpertMaps(
        static_cast<const float*>(nullptr), nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, rows, experts, k, normalize, nullptr);
  };
  EXPECT_FALSE(launch(0, 512, 10, true));
  EXPECT_FALSE(launch(3, 512, 10, true));
  EXPECT_FALSE(launch(2, 513, 10, true));
  EXPECT_FALSE(launch(2, 512, 17, true));
  EXPECT_FALSE(launch(2, 8, 9, true));
  EXPECT_FALSE(launch(2, 512, 0, true));
  EXPECT_FALSE(launch(2, 512, 10, false));
}

TEST(CUDA_EP_Unittest, FusedSoftmaxTopKExpertMaps_WarpSelectSpecialized) {
  for (int rows : {1, 2}) {
    for (bool ties : {false, true}) {
      RunFusedRoutingMapsTest<float>(rows, 512, 10, ties);
      RunFusedRoutingMapsTest<half>(rows, 512, 10, ties);
      RunFusedRoutingMapsTest<__nv_bfloat16>(rows, 512, 10, ties);
    }
    RunFusedRoutingMapsTest<float>(rows, 512, 10, false, true);
  }
}

TEST(CUDA_EP_Unittest, FusedSoftmaxTopKExpertMaps_WarpSelectAdversarialKeys) {
  for (int rows : {1, 2}) {
    for (int pattern = 0; pattern < 4; ++pattern) {
      SCOPED_TRACE(::testing::Message() << "rows=" << rows << " pattern=" << pattern);
      auto values = MakeLogits(rows, 512);
      for (int row = 0; row < rows; ++row) {
        for (int expert = 0; expert < 512; ++expert) {
          float& value = values[row * 512 + expert];
          if (pattern == 0) {
            value = expert % 32 == row ? 32.0f - expert / 32 : -32.0f;
          } else if (pattern == 1) {
            value = expert % 2 ? -0.0f : 0.0f;
          } else if (pattern == 2) {
            value = expert % 3 ? -std::numeric_limits<float>::infinity()
                               : std::numeric_limits<float>::quiet_NaN();
          } else {
            value = (expert + row) % 7 == 0 ? 1000.0f : -1000.0f;
          }
        }
      }
      RunFusedRoutingMapsTest<float>(rows, 512, 10, false, false, &values);
      RunFusedRoutingMapsTest<half>(rows, 512, 10, false, false, &values);
      RunFusedRoutingMapsTest<__nv_bfloat16>(rows, 512, 10, false, false, &values);
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mlas.h"
#include "bench_util.h"
#include "core/util/thread_utils.h"

#include <stdexcept>
#include <memory>
#include <numeric>
#include <algorithm>

static const std::vector<std::string> qgemm_arg_names = {"M", "N", "K", "Batch", "Threads"};

void QGEMM(benchmark::State& state, bool pack_b) {
  const bool b_is_signed = true;
  const uint8_t a_zero_point = 29;
  const uint8_t b_zero_point = 179;

  if (state.range(0) <= 0) throw std::invalid_argument("M must greater than 0!");
  if (state.range(1) <= 0) throw std::invalid_argument("N must greater than 0!");
  if (state.range(2) <= 0) throw std::invalid_argument("K must greater than 0!");
  if (state.range(3) <= 0) throw std::invalid_argument("Batch must greater than 0!");
  if (state.range(4) <= 0) throw std::invalid_argument("Threads must greater than 0!");

  const size_t M = static_cast<size_t>(state.range(0));
  const size_t N = static_cast<size_t>(state.range(1));
  const size_t K = static_cast<size_t>(state.range(2));

  const size_t batch = static_cast<size_t>(state.range(3));
  const size_t threads = static_cast<size_t>(state.range(4));
  
  OrtThreadPoolParams tpo;
  tpo.thread_pool_size = int(threads);
  tpo.auto_set_affinity = true;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> tp(
      onnxruntime::concurrency::CreateThreadPool(&onnxruntime::Env::Default(),
      tpo, onnxruntime::concurrency::ThreadPoolType::INTRA_OP));

  auto A_holder = RandomVectorUniform<uint8_t>(static_cast<size_t>(M * K * batch), uint8_t(-100), uint8_t(100));
  auto B_holder = RandomVectorUniform<uint8_t>(static_cast<size_t>(N * K * batch), uint8_t(-110), uint8_t(110));
  std::vector<int32_t> C_holder(static_cast<size_t>(M * N * batch));
  std::vector<uint8_t> pack_b_holder;

  size_t packed_b_size = 0;
  if (pack_b) {
    packed_b_size = MlasGemmPackBSize(N, K, b_is_signed);
    pack_b_holder.resize(packed_b_size * batch);
  }

  MLAS_GEMM_U8X8_SHAPE_PARAMS gemm_shape;

  gemm_shape.M = static_cast<size_t>(M);
  gemm_shape.N = static_cast<size_t>(N);
  gemm_shape.K = static_cast<size_t>(K);
  gemm_shape.BIsSigned = b_is_signed;


  std::vector<MLAS_GEMM_U8X8_DATA_PARAMS> gemm_data_vec(batch);
  for (size_t i = 0; i < batch; i++) {
    auto& gemm_params = gemm_data_vec[i];
    gemm_params.lda = gemm_shape.K;
    gemm_params.ZeroPointA = a_zero_point;
    gemm_params.ZeroPointB = &b_zero_point;
    gemm_params.ldc = gemm_shape.N;
    gemm_params.A = A_holder.data() + M * K * i;
    gemm_params.B = B_holder.data() + N * K * i;
    gemm_params.ldb = gemm_shape.N;
    gemm_params.C = C_holder.data() + M * N * i;
    if (pack_b) {
      MlasGemmPackB(N, K, (const uint8_t*)gemm_params.B, N, b_is_signed, (void*)(pack_b_holder.data() + packed_b_size * i));
      gemm_params.BIsPacked = true;
      gemm_params.B = (void*)(pack_b_holder.data() + packed_b_size * i);
    }
  }
  for (auto _ : state) {
    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), batch, tp.get());
  }
}

static void QGemmSize(benchmark::internal::Benchmark* b) {
  b->ArgNames(qgemm_arg_names);
  // Args for  "M", "N", "K", "Batch",

  b->Args({512, 32128, 768, 1, 1});
  b->Args({512, 32128, 768, 1, 4});
  b->Args({512, 32128, 768, 1, 6});

  b->Args({512, 3072, 768, 1, 1});
  b->Args({512, 3072, 768, 1, 4});
  b->Args({512, 3072, 768, 1, 6});

  b->Args({512, 768, 3072, 1, 1});
  b->Args({512, 768, 3072, 1, 4});
  b->Args({512, 768, 3072, 1, 6});

  b->Args({512, 768, 768, 1, 1});
  b->Args({512, 768, 768, 1, 4});
  b->Args({512, 768, 768, 1, 6});

  b->Args({512, 64, 512, 1, 1});
  b->Args({512, 64, 512, 1, 4});
  b->Args({512, 64, 512, 1, 6});

  b->Args({512, 512, 64, 12, 1});
  b->Args({512, 512, 64, 12, 4});
  b->Args({512, 512, 64, 12, 6});

  b->Args({512, 64, 512, 12, 1});
  b->Args({512, 64, 512, 12, 4});
  b->Args({512, 64, 512, 12, 6});

}


BENCHMARK_CAPTURE(QGEMM, PackB, true)->Apply(QGemmSize)->UseRealTime();
BENCHMARK_CAPTURE(QGEMM, NoPackB, false)->Apply(QGemmSize)->UseRealTime();

#include "common.h"
#include "core/util/thread_utils.h"

#include <core/providers/cpu/tensor/copy.h>
#include <benchmark/benchmark.h>
#include <core/platform/threadpool.h>

using namespace onnxruntime;
using namespace onnxruntime::concurrency;

static void BM_StridedCopy_Memcpy(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t feature_size = static_cast<size_t>(state.range(1));

  float* output = (float*)aligned_alloc(sizeof(float) * batch_size * feature_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size * feature_size, -1, 1);


  for (auto _ : state) {
	memcpy(output, data, batch_size * feature_size * sizeof(float)); 
  }
  aligned_free(data);
  aligned_free(output);
}

static void BM_StridedCopy_SingleThread(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t feature_size = static_cast<size_t>(state.range(1));

  float* output = (float*)aligned_alloc(sizeof(float) * batch_size * feature_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size * feature_size, -1, 1);

  int64_t ibatch_size = static_cast<int64_t> (batch_size);
  int64_t ifeature_size = static_cast<int64_t> (feature_size);
  for (auto _ : state) {
	  // use nullptr threadpool to make it run single threaded
	StridedCopy<float>(nullptr, output, {ibatch_size, ifeature_size}, {ifeature_size, 1}, data, {ifeature_size, 1});
  }
  aligned_free(data);
  aligned_free(output);
}

static void BM_StridedCopy_Parallel(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const size_t feature_size = static_cast<size_t>(state.range(1));

  float* output = (float*)aligned_alloc(sizeof(float) * batch_size * feature_size, 64);
  float* data = GenerateArrayWithRandomValue<float>(batch_size * feature_size, -1, 1);

  // setup threadpool
  OrtThreadPoolParams tpo;
  tpo.auto_set_affinity = true;
  std::unique_ptr<concurrency::ThreadPool> tp(
      concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP));

  int64_t ibatch_size = static_cast<int64_t> (batch_size);
  int64_t ifeature_size = static_cast<int64_t> (feature_size);
  for (auto _ : state) {
	StridedCopy<float>(tp.get(), output, {ibatch_size, ifeature_size}, {ifeature_size, 1}, data, {ifeature_size, 1});
  }
  aligned_free(data);
  aligned_free(output);
}

#define SC_BENCHMARK(name) \
	BENCHMARK(name) \
	    ->UseRealTime() \
	    ->UseRealTime() \
	    ->Unit(benchmark::TimeUnit::kNanosecond) \
	    ->Args({10000, 64}) \
	    ->Args({20000, 64}) \
	    ->Args({40000, 64}) \
	    ->Args({400000, 64}); \

SC_BENCHMARK(BM_StridedCopy_Memcpy);
SC_BENCHMARK(BM_StridedCopy_SingleThread);
SC_BENCHMARK(BM_StridedCopy_Parallel);


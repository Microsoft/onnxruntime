
#include <core/graph/onnx_protobuf.h>
#include <core/framework/tensor.h>
#include <core/util/math_cpuonly.h>
#include "core/platform/threadpool.h"
#include <benchmark/benchmark.h>
#include <random>

using namespace onnxruntime;

using namespace onnxruntime;

template <typename T>
void SetRandom(Tensor& input) {
  int64_t size = input.Shape().Size();
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_real_distribution<T> distr(0, 1);
  T* data = input.MutableData<T>();
  for (int64_t i = 0; i != size; ++i) {
    data[i] = distr(gen);
  }
}

static void BM_BatchNormOldEigen(benchmark::State& state) {
  std::shared_ptr<CPUAllocator> alloc = std::make_shared<CPUAllocator>();
  const int64_t batch_size = state.range(0);

  const TensorShape shape = {batch_size, 64, 75, 75};
  using T = float;

  Tensor* X = new Tensor(DataTypeImpl::GetType<float>(), shape, alloc);
  SetRandom<T>(*X);
  const TensorShape& x_shape = X->Shape();
  Tensor* Y = new Tensor(DataTypeImpl::GetType<float>(), shape, alloc);
  Tensor* scale = new Tensor(DataTypeImpl::GetType<float>(), {shape[1]}, alloc);
  SetRandom<T>(*scale);
  Tensor* mean = new Tensor(DataTypeImpl::GetType<float>(), {shape[1]}, alloc);
  SetRandom<T>(*mean);

  Tensor* B = new Tensor(DataTypeImpl::GetType<float>(), {shape[1]}, alloc);
  SetRandom<T>(*B);

  Tensor* var = new Tensor(DataTypeImpl::GetType<float>(), {shape[1]}, alloc);
  SetRandom<T>(*var);

  bool is_spatial_ = true;
  double epsilon_ = 1e-5;
  const auto& dims_vec = x_shape.GetDims();
  const size_t N = dims_vec[0];
  const size_t C = dims_vec[1];  // assume NCHW as per the spec

  // calculate sample_size (per individual channel)
  size_t sample_size = 1;
  for (size_t i = 2; i < dims_vec.size(); ++i) {
    sample_size *= dims_vec[i];
  }

  // calculate sample_size (including all channels)
  size_t sample_size_incl_all_channels = sample_size * C;
  for (auto _ : state) {
    ConstEigenVectorArrayMap<T> scale_arr(scale->Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
    ConstEigenVectorArrayMap<T> bias_arr(B->Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);

    // Regardless of training or testing, we will apply the estimated mean
    // and standard deviation to the input. For testing, they are
    // specified directly by the input, and for training, they are computed
    // by the op.
    Eigen::Array<T, Eigen::Dynamic, 1> inv_std(is_spatial_ ? C : sample_size_incl_all_channels);
    ConstEigenVectorArrayMap<T> var_arr(var->Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
    inv_std = (var_arr + epsilon_).sqrt().inverse();
    ConstEigenVectorArrayMap<T> mean_arr(mean->Data<T>(), is_spatial_ ? C : sample_size_incl_all_channels);
    // We can fuse the output computation as follows:
    //   ((x - est_mean) * (inv_std)) * scale + bias
    // to
    //   (x * inv_std * scale) + (bias - est_mean * inv_std * scale)
    Eigen::Array<T, Eigen::Dynamic, 1> new_scale = inv_std * scale_arr;
    Eigen::Array<T, Eigen::Dynamic, 1> new_bias = bias_arr - mean_arr * new_scale;
    EigenArrayMap<T> Y_arr(Y->template MutableData<T>(), is_spatial_ ? sample_size : sample_size_incl_all_channels,
                           is_spatial_ ? N * C : N);
    ConstEigenArrayMap<T> X_arr(X->Data<T>(), is_spatial_ ? sample_size : sample_size_incl_all_channels,
                                is_spatial_ ? N * C : N);
    if (is_spatial_) {  // spatial == 1
      for (size_t nc = 0; nc < N * C; ++nc) {
        Y_arr.col(nc) = X_arr.col(nc) * new_scale(nc % C) + new_bias(nc % C);
      }
    } else {  // spatial == 0
      for (size_t n = 0; n < N; ++n) {
        Y_arr.col(n) = X_arr.col(n) * new_scale.col(0) + new_bias.col(0);
      }
    }
  }
}
BENCHMARK(BM_BatchNormOldEigen)
    ->Arg(1)
    ->Arg(16)
    ->Arg(64)
    ->UseRealTime()
    ->Unit(benchmark::TimeUnit::kMicrosecond);

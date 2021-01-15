// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/core/framework/distributed_run_context.h"

namespace onnxruntime {
namespace test {

using onnxruntime::training::DistributedRunContext;
using onnxruntime::training::MPIContext;

class NcclKernelTest : public testing::Test {
 protected:
  NcclKernelTest() {}

  virtual void SetUp() override {
    world_size_ = MPIContext::GetInstance().GetWorldSize();
    world_rank_ = MPIContext::GetInstance().GetWorldRank();
    local_size_ = MPIContext::GetInstance().GetLocalSize();
    local_rank_ = MPIContext::GetInstance().GetLocalRank();
    data_parallel_size_ = world_size_;
    horizontal_parallel_size_ = 1;
    pipeline_parallel_size_ = 1;

    if (local_size_ > 1) {
      ORT_ENFORCE(local_size_ % 2 == 0 || local_size_ % 4 == 0);
    }

    DistributedRunContext::CreateInstance({world_rank_, world_size_,
                                           local_rank_, local_size_,
                                           data_parallel_size_,
                                           horizontal_parallel_size_,
                                           pipeline_parallel_size_});
  }

  int world_size_;
  int world_rank_;
  int local_size_;
  int local_rank_;
  int data_parallel_size_;
  int horizontal_parallel_size_;
  int pipeline_parallel_size_;
};

static void RunNcclAllReduceTest(bool use_fp16, int local_size, int local_rank) {
  if (local_size <= 1) return;

  RandomValueGenerator random{42};
  OpTester test("NcclAllReduce", 1, onnxruntime::kMSDomain);
  test.AddBufferedInputOutput();
  test.SetDeviceId(local_rank);

  const std::vector<std::vector<int64_t>> tensors_dims = {{2, 3}, {128}, {512}, {7, 13}, {1024, 3}};
  for (size_t input_id = 0; input_id < tensors_dims.size(); ++input_id) {
    const std::vector<int64_t>& tensor_dims = tensors_dims[input_id];

    std::vector<float> X_data = random.Uniform<float>(tensor_dims, -10.0f, 10.0f);
    std::vector<float> Y_data(X_data.size());
    for (size_t i = 0; i < X_data.size(); ++i) {
      Y_data[i] = local_size * X_data[i];
    }

    const std::string input_string = "input" + std::to_string(input_id);
    const std::string output_string = "output" + std::to_string(input_id);

    if (use_fp16) {
      test.AddInput<MLFloat16>(input_string.c_str(), tensor_dims, ToFloat16(X_data));
      test.AddOutput<MLFloat16>(output_string.c_str(), tensor_dims, ToFloat16(Y_data));
    } else {
      test.AddInput<float>(input_string.c_str(), tensor_dims, X_data);
      test.AddOutput<float>(output_string.c_str(), tensor_dims, Y_data);
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
  }
}

TEST_F(NcclKernelTest, NcclAllReduce_FP32) {
  RunNcclAllReduceTest(false /*use_fp16*/, local_size_, local_rank_);
}

TEST_F(NcclKernelTest, NcclAllReduce_FP16) {
  RunNcclAllReduceTest(true /*use_fp16*/, local_size_, local_rank_);
}

static void RunNcclReduceScatterTest(bool use_fp16, int local_size, int local_rank) {
  if (local_size <= 1) return;

  const size_t elem_size = use_fp16 ? sizeof(MLFloat16) : sizeof(float);

  RandomValueGenerator random{42};
  OpTester test("NcclReduceScatter", 1, onnxruntime::kMSDomain);
  test.AddBufferedInputOutput();
  test.SetDeviceId(local_rank);

  const std::vector<std::vector<int64_t>> tensors_dims = {{2, 3}, {5, 17}, {512}, {7, 13}, {256}};

  size_t total_bytes = 0;
  for (auto& tensor_dims : tensors_dims) {
    TensorShape shape = TensorShape(tensor_dims);
    size_t tensor_occupied_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(shape.Size(), elem_size, &tensor_occupied_bytes));
    total_bytes += tensor_occupied_bytes;
  }
  ORT_ENFORCE(total_bytes % (local_size * 32) == 0);

  const size_t bytes_per_rank = total_bytes / local_size;
  const size_t rank_address_start = local_rank * bytes_per_rank;
  const size_t rank_address_end = rank_address_start + bytes_per_rank;

  size_t tensor_bytes_offset = 0;
  for (size_t input_id = 0; input_id < tensors_dims.size(); ++input_id) {
    const std::vector<int64_t>& tensor_dims = tensors_dims[input_id];

    std::vector<float> X_data = random.Uniform<float>(tensor_dims, -10.0f, 10.0f);
    std::vector<float> Y_data(X_data);
    for (size_t i = 0; i < X_data.size(); ++i) {
      size_t offset = tensor_bytes_offset + i * elem_size;
      if (rank_address_start <= offset && offset < rank_address_end) {
        Y_data[i] = local_size * X_data[i];
      }
    }

    const std::string input_string = "input" + std::to_string(input_id);
    const std::string output_string = "output" + std::to_string(input_id);

    if (use_fp16) {
      test.AddInput<MLFloat16>(input_string.c_str(), tensor_dims, ToFloat16(X_data));
      test.AddOutput<MLFloat16>(output_string.c_str(), tensor_dims, ToFloat16(Y_data));
    } else {
      test.AddInput<float>(input_string.c_str(), tensor_dims, X_data);
      test.AddOutput<float>(output_string.c_str(), tensor_dims, Y_data);
    }

    size_t tensor_occupied_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(X_data.size(), elem_size, &tensor_occupied_bytes));
    tensor_bytes_offset += tensor_occupied_bytes;
  }

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

TEST_F(NcclKernelTest, NcclReduceScatter_FP32) {
  RunNcclReduceScatterTest(false /*use_fp16*/, local_size_, local_rank_);
}

TEST_F(NcclKernelTest, NcclReduceScatter_FP16) {
  RunNcclReduceScatterTest(true /*use_fp16*/, local_size_, local_rank_);
}

static void RunNcclAllGatherTest(bool use_fp16, int local_size, int local_rank) {
  if (local_size <= 1) return;

  const size_t elem_size = use_fp16 ? sizeof(MLFloat16) : sizeof(float);

  RandomValueGenerator random{42};
  OpTester test("NcclAllGather", 1, onnxruntime::kMSDomain);
  test.AddBufferedInputOutput();
  test.SetDeviceId(local_rank);

  const std::vector<std::vector<int64_t>> tensors_dims = {{2, 3}, {5, 17}, {512}, {7, 13}, {256}};

  size_t total_bytes = 0;
  for (auto& tensor_dims : tensors_dims) {
    TensorShape shape = TensorShape(tensor_dims);
    size_t tensor_occupied_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(shape.Size(), elem_size, &tensor_occupied_bytes));
    total_bytes += tensor_occupied_bytes;
  }
  ORT_ENFORCE(total_bytes % (local_size * 32) == 0);

  const size_t bytes_per_rank = total_bytes / local_size;
  const size_t rank_address_start = local_rank * bytes_per_rank;
  const size_t rank_address_end = rank_address_start + bytes_per_rank;

  size_t tensor_bytes_offset = 0;
  for (size_t input_id = 0; input_id < tensors_dims.size(); ++input_id) {
    const std::vector<int64_t>& tensor_dims = tensors_dims[input_id];

    std::vector<float> Y_data = random.Uniform<float>(tensor_dims, -10.0f, 10.0f);
    std::vector<float> X_data(Y_data);
    for (size_t i = 0; i < X_data.size(); ++i) {
      size_t offset = tensor_bytes_offset + i * elem_size;
      if (rank_address_start <= offset && offset < rank_address_end) {
        X_data[i] = Y_data[i];
      }
    }

    const std::string input_string = "input" + std::to_string(input_id);
    const std::string output_string = "output" + std::to_string(input_id);
    if (use_fp16) {
      test.AddInput<MLFloat16>(input_string.c_str(), tensor_dims, ToFloat16(X_data));
      test.AddOutput<MLFloat16>(output_string.c_str(), tensor_dims, ToFloat16(Y_data));
    } else {
      test.AddInput<float>(input_string.c_str(), tensor_dims, X_data);
      test.AddOutput<float>(output_string.c_str(), tensor_dims, Y_data);
    }

    size_t tensor_occupied_bytes = 0;
    ORT_ENFORCE(IAllocator::CalcMemSizeForArrayWithAlignment<kAllocAlignment>(X_data.size(), elem_size, &tensor_occupied_bytes));
    tensor_bytes_offset += tensor_occupied_bytes;
  }

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCpuExecutionProvider});
}

TEST_F(NcclKernelTest, NcclAllGather_FP32) {
  RunNcclAllGatherTest(false /*use_fp16*/, local_size_, local_rank_);
}

TEST_F(NcclKernelTest, NcclAllGather_FP16) {
  RunNcclAllGatherTest(true /*use_fp16*/, local_size_, local_rank_);
}

}  // namespace test
}  // namespace onnxruntime
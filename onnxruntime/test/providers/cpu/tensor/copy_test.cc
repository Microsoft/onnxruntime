// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/providers/cpu/tensor/copy.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace test {

class CopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    OrtThreadPoolParams tpo;
    tpo.auto_set_affinity = true;
    tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), tpo, concurrency::ThreadPoolType::INTRA_OP);
  }
  std::unique_ptr<concurrency::ThreadPool> tp;
};

TEST_F(CopyTest, Contiguous1D) {
  int src[10];
  for (int i = 0; i < 10; i++) {
    src[i] = i;
  }

  int dst[10];

  StridedCopy<int>(tp.get(), dst, {1}, {10}, src, {1});

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(src[i], dst[i]);
  }
}

TEST_F(CopyTest, Contiguous3D) {
  double src[3 * 4 * 5];
  for (int i = 0; i < 3 * 4 * 5; i++) {
    src[i] = static_cast<double>(i);
  }

  double dst[3 * 4 * 5];

  StridedCopy<double>(tp.get(), dst, {20, 5, 1}, {3, 4, 5}, src, {20, 5, 1});

  for (int i = 0; i < 3 * 4 * 5; i++) {
    EXPECT_EQ(src[i], dst[i]);
  }
}

TEST_F(CopyTest, Transpose4D) {
  // Test performing a transpose using a strided copy
  int64_t numel = 2 * 3 * 4 * 5;
  double* src = new double[numel];
  for (int i = 0; i < numel; i++) {
    src[i] = static_cast<double>(i);
  }

  double* dst = new double[numel];

  std::vector<int64_t> dst_strides = {60, 5, 15, 1};
  std::vector<int64_t> src_strides = {60, 20, 5, 1};
  StridedCopy<double>(tp.get(), dst, dst_strides, {2, 3, 4, 5}, src, src_strides);

  // stride to access the dst tensor as if it were contiguous
  std::vector<int64_t> contig_dst_strides = {60, 15, 5, 1};

  for (int i0 = 0; i0 < 2; i0++) {
    for (int i1 = 0; i1 < 3; i1++) {
      for (int i2 = 0; i2 < 4; i2++) {
        for (int i3 = 0; i3 < 5; i3++) {
          size_t src_access = src_strides[0] * i0 + src_strides[1] * i1 + src_strides[2] * i2 + src_strides[3] * i3;
          size_t dst_access = contig_dst_strides[0] * i0 + contig_dst_strides[1] * i2 + contig_dst_strides[2] * i1 + contig_dst_strides[3] * i3;

          EXPECT_EQ(src[src_access], dst[dst_access]);
        }
      }
    }
  }
}

TEST_F(CopyTest, Concat2D) {
  // test performing a concat using a strided copy
  double* src = new double[6 * 2];
  for (int i = 0; i < 6 * 2; i++) {
    src[i] = static_cast<double>(i);
  }

  double* dst = new double[10 * 5];
  for (int i = 0; i < 10 * 5; i++) {
    dst[i] = 0;
  }

  std::vector<int64_t> dst_strides = {5, 1};
  std::vector<int64_t> src_strides = {2, 1};
  std::ptrdiff_t offset = 3;
  StridedCopy<double>(tp.get(), dst + offset, dst_strides, {6, 2}, src, src_strides);

  for (int i0 = 0; i0 < 10; i0++) {
    for (int i1 = 0; i1 < 5; i1++) {
      size_t dst_access = dst_strides[0] * i0 + dst_strides[1] * i1;
      if (3 <= i1 && 0 <= i0 && i0 < 6) {
        size_t src_access = src_strides[0] * i0 + src_strides[1] * (i1 - 3);
        EXPECT_EQ(src[src_access], dst[dst_access]);
      } else {
        EXPECT_EQ(0, dst[dst_access]);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

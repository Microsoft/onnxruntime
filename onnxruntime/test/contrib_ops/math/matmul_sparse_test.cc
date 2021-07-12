// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/framework/sparse_utils.h"

namespace onnxruntime {
namespace test {

// Compilation only tests
//TEST(SparseTestInterfaces, Test) {
//
//  // Test initializer overload for booleans input
//  OpTester tester("DoesNotMatter", 1, onnxruntime::kMSDomain);
//  tester.AddSparseCooInput("A", {2, 3}, {true, true}, {0, 4});
//  tester.AddSparseCsrInput("B", {2, 3}, {true, true, true}, {0, 2, 5}, {0, 2, 3});
//
//  // std::vector<T> overloads input
//  const std::vector<float> float_values = {1.f, 2.f, 3.f};
//  tester.AddSparseCooInput("A", {2, 3}, float_values, {0, 4});
//  tester.AddSparseCsrInput("B", {2, 3}, float_values, {0, 2, 5}, {0, 2, 3});
//
//  // gsl::span<const T> overloads for input
//  tester.AddSparseCooInput("A", {2, 3}, gsl::make_span(float_values), {0, 4});
//  tester.AddSparseCsrInput("B", {2, 3}, gsl::make_span(float_values), {0, 2, 5}, {0, 2, 3});
//
//  // Output boolean
//  tester.AddSparseCooOutput("A", {2, 3}, {true, true}, {0, 4}, CheckParams());
//  tester.AddSparseCsrOutput("B", {2, 3}, {true, true, true}, {0, 2, 5}, {0, 2, 3}, CheckParams());
//
//  // Output vector
//  tester.AddSparseCooOutput("A", {2, 3}, float_values, {0, 4}, CheckParams());
//  tester.AddSparseCsrOutput("B", {2, 3}, float_values, {0, 2, 5}, {0, 2, 3}, CheckParams());
//
//  // Output span
//  tester.AddSparseCooOutput("A", {2, 3}, gsl::make_span(float_values), {0, 4}, CheckParams());
//  tester.AddSparseCsrOutput("B", {2, 3}, gsl::make_span(float_values), {0, 2, 5}, {0, 2, 3}, CheckParams());
//}

template<typename T>
void ConvertToCsr(gsl::span<const T> input_span,
  const std::vector<int64_t>& dims,
  std::vector<T>& values_out,
  std::vector<int64_t>& inner_indicies_out,
  std::vector<int64_t>& outer_indicies_out) {

  ASSERT_EQ(dims.size(), 2U);
  const auto rows = dims[0];
  const auto cols = dims[1];
  const auto dense_size = rows * cols;
  ASSERT_EQ(input_span.size(), static_cast<size_t>(dense_size));

  const int64_t nnz = std::count_if(input_span.cbegin(), input_span.cend(),
                                    [](T v) { return v != T(0); });

  std::vector<T> values;
  std::vector<int64_t> inner;
  std::vector<int64_t> outer;

  if (nnz > 0) {
    values.reserve(nnz);
    inner.reserve(nnz);
    outer.reserve(rows + 1);

    outer.push_back(0);
    int64_t col = 0;
    for (int64_t i = 0; i < dense_size; ++i, ++col) {
      if (col >= cols) {
        outer.push_back(static_cast<int64_t>(values.size()));
        col = 0;
      }
      if (input_span[i] != T(0)) {
        values.push_back(input_span[i]);
        inner.push_back(col);
      }
    }
    // Final entry
    outer.push_back(static_cast<int64_t>(values.size()));
  }

  values_out = std::move(values);
  inner_indicies_out = std::move(inner);
  outer_indicies_out = std::move(outer);
}

// Converts with 2-D indices
template<typename T>
void ConvertToCoo(gsl::span<const T> input_span,
  const std::vector<int64_t>& dims,
  std::vector<T>& values_out,
  std::vector<int64_t>& indices_out) {

  ASSERT_EQ(dims.size(), 2U);
  const auto rows = dims[0];
  const auto cols = dims[1];
  const auto dense_size = rows * cols;
  ASSERT_EQ(input_span.size(), static_cast<size_t>(dense_size));

  const int64_t nnz = std::count_if(input_span.cbegin(), input_span.cend(),
                                    [](T v) { return v != T(0); });

  std::vector<T> values;
  std::vector<int64_t> indices;
  if (nnz > 0) {
    values.reserve(nnz);
    indices.reserve(2 * nnz);

    int64_t row = 0;
    int64_t col = 0;
    for (int64_t i = 0; i < dense_size; ++i, ++col) {
      if (col >= cols) {
        col = 0;
        ++row;
      }
      if (input_span[i] != T(0)) {
        values.push_back(input_span[i]);
        indices.push_back(row);
        indices.push_back(col);
      }
    }
  }
  values_out = std::move(values);
  indices_out = std::move(indices);
}

/// This test must be disabled x86 build because of an apparent Eigen bug
/// However, things work on x64 builds
/// It manifests itself as Status Message: std::bad_alloc or Status Message: bad allocation
/// due to this piece of code in Eigen
/// The size is 19 but the allocated size is 18, after std::min realloc_size is -1 which causes it to throw.
/// wasm (nodejs) is built on 32-bit and thus is also affected.
/*
void resize(Index size, double reserveSizeFactor = 0) {
  if (m_allocatedSize < size) {
    Index realloc_size = (std::min<Index>)(NumTraits<StorageIndex>::highest(), size + Index(reserveSizeFactor * double(size)));
    if (realloc_size < size)
      internal::throw_std_bad_alloc();
    reallocate(realloc_size);
  }
  m_size = size;
}
*/

#if !defined(__i386__) && !defined(_M_IX86) && !defined(__wasm__)
TEST(SparseToDenseMatMul, TestCsr) {
  constexpr int64_t rows = 9;
  constexpr int64_t cols = 9;
  const std::vector<int64_t> A_shape = {rows, cols};
  const std::vector<float> input_data = {
      0, 1, 2, 0, 0, 0, 3, 4, 5,
      6, 7, 8, 0, 0, 0, 9, 10, 11,
      12, 13, 14, 0, 0, 0, 15, 16, 17,
      0, 0, 0, 18, 19, 20, 21, 22, 23,
      0, 0, 0, 24, 25, 26, 27, 28, 29,
      0, 0, 0, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 0, 0, 0,
      42, 43, 44, 45, 46, 47, 0, 0, 0,
      48, 49, 50, 51, 52, 53, 0, 0, 0};

  std::vector<float> A_values;
  std::vector<int64_t> A_inner_indices;
  std::vector<int64_t> A_outer_indices;
  ConvertToCsr(gsl::make_span(input_data), A_shape, A_values, A_inner_indices, A_outer_indices);

  ASSERT_EQ(A_values.size(), A_inner_indices.size());
  ASSERT_EQ(static_cast<int64_t>(A_outer_indices.size()), rows + 1);

  const std::vector<int64_t> B_shape = {9, 9};
  const std::vector<float> B_data = {
      0, 1, 2, 0, 0, 0, 3, 4, 5,
      6, 7, 8, 0, 0, 0, 9, 10, 11,
      12, 13, 14, 0, 0, 0, 15, 16, 17,
      0, 0, 0, 18, 19, 20, 21, 22, 23,
      0, 0, 0, 24, 25, 26, 27, 28, 29,
      0, 0, 0, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 0, 0, 0,
      42, 43, 44, 45, 46, 47, 0, 0, 0,
      48, 49, 50, 51, 52, 53, 0, 0, 0};

  const std::vector<int64_t> X_shape = {rows, cols};
  const std::vector<float> non_t_output = {
      546, 561, 576, 552, 564, 576, 39, 42, 45,
      1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
      2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
      2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
      3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
      4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
      678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
      786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
      894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165};

  // Testing CSR non-transpose
  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddSparseCsrInput("A", A_shape, A_values, A_inner_indices, A_outer_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, non_t_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }

  // Transpose A output
  const std::vector<float> t_a_output = {
        5544, 5688, 5832, 5742, 5868, 5994, 234, 252, 270,
        5688, 5838, 5988, 5877, 6006, 6135, 261, 282, 303,
        5832, 5988, 6144, 6012, 6144, 6276, 288, 312, 336,
        5742, 5877, 6012, 7947, 8154, 8361, 2016, 2088, 2160,
        5868, 6006, 6144, 8154, 8367, 8580, 2097, 2172, 2247,
        5994, 6135, 6276, 8361, 8580, 8799, 2178, 2256, 2334,
        234, 261, 288, 2016, 2097, 2178, 2574, 2682, 2790,
        252, 282, 312, 2088, 2172, 2256, 2682, 2796, 2910,
        270, 303, 336, 2160, 2247, 2334, 2790, 2910, 3030};

  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddAttribute("transA", int64_t{1});
    tester.AddSparseCsrInput("A", A_shape, A_values, A_inner_indices, A_outer_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, t_a_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }

  // Transpose B output
  const std::vector<float> t_b_output = {
      55,  145, 235,  266,  338,  410,  113,  131,  149,
      145, 451, 757,  662,  842,  1022, 779,  905,  1031,
      235, 757, 1279, 1058, 1346, 1634, 1445, 1679, 1913,
      266, 662, 1058, 2539, 3277, 4015, 2282, 2624, 2966,
      338, 842, 1346, 3277, 4231, 5185, 3002, 3452, 3902,
      410, 1022, 1634, 4015, 5185, 6355, 3722, 4280, 4838,
      113, 779, 1445, 2282, 3002, 3722, 8911, 10297, 11683,
      131, 905, 1679, 2624, 3452, 4280, 10297, 11899, 13501,
      149, 1031, 1913, 2966, 3902, 4838, 11683, 13501, 15319};

  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddAttribute("transB", int64_t{1});
    tester.AddSparseCsrInput("A", A_shape, A_values, A_inner_indices, A_outer_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, t_b_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }

  // Transpose both A and B
  const std::vector<float> t_a_b_output = {
      546, 1410, 2274, 2784, 3540, 4296, 678, 786, 894,
      561, 1461, 2361, 2850, 3624, 4398, 789, 915, 1041,
      576, 1512, 2448, 2916, 3708, 4500, 900, 1044, 1188,
      552, 1362, 2172, 4362, 5604, 6846, 2892, 3324, 3756,
      564, 1392, 2220, 4485, 5763, 7041, 3012, 3462, 3912,
      576, 1422, 2268, 4608, 5922, 7236, 3132, 3600, 4068,
      39,  201,  363,  1551, 2037, 2523, 4263, 4911, 5559,
      42,  222,  402,  1608, 2112, 2616, 4494, 5178, 5862,
      45,  243,  441,  1665, 2187, 2709, 4725, 5445, 6165
  };

  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddAttribute("transA", int64_t{1});
    tester.AddAttribute("transB", int64_t{1});
    tester.AddSparseCsrInput("A", A_shape, A_values, A_inner_indices, A_outer_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, t_a_b_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}
#endif // x86

TEST(SparseToDenseMatMul, TestCoo) {
  constexpr int64_t rows = 9;
  constexpr int64_t cols = 9;
  const std::vector<int64_t> A_shape = {rows, cols};
  const std::vector<float> input_data = {
      0, 1, 2, 0, 0, 0, 3, 4, 5,
      6, 7, 8, 0, 0, 0, 9, 10, 11,
      12, 13, 14, 0, 0, 0, 15, 16, 17,
      0, 0, 0, 18, 19, 20, 21, 22, 23,
      0, 0, 0, 24, 25, 26, 27, 28, 29,
      0, 0, 0, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 0, 0, 0,
      42, 43, 44, 45, 46, 47, 0, 0, 0,
      48, 49, 50, 51, 52, 53, 0, 0, 0};

  std::vector<float> A_values;
  std::vector<int64_t> A_indices;
  ConvertToCoo(gsl::make_span(input_data), A_shape, A_values, A_indices);
  ASSERT_FALSE(A_values.empty());
  ASSERT_EQ(A_values.size() * 2, A_indices.size());
  const std::vector<int64_t> A_indicies_shape{static_cast<int64_t>(A_values.size()), 2};

  const std::vector<int64_t> B_shape = {9, 9};
  const std::vector<float> B_data = {
      0, 1, 2, 0, 0, 0, 3, 4, 5,
      6, 7, 8, 0, 0, 0, 9, 10, 11,
      12, 13, 14, 0, 0, 0, 15, 16, 17,
      0, 0, 0, 18, 19, 20, 21, 22, 23,
      0, 0, 0, 24, 25, 26, 27, 28, 29,
      0, 0, 0, 30, 31, 32, 33, 34, 35,
      36, 37, 38, 39, 40, 41, 0, 0, 0,
      42, 43, 44, 45, 46, 47, 0, 0, 0,
      48, 49, 50, 51, 52, 53, 0, 0, 0};

  const std::vector<int64_t> X_shape = {rows, cols};
  const std::vector<float> non_t_output = {
      546, 561, 576, 552, 564, 576, 39, 42, 45,
      1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
      2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
      2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
      3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
      4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
      678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
      786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
      894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165};

  // Check no transpose case
  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddSparseCooInput("A", A_shape, A_values, A_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, non_t_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // Transpose A output
  const std::vector<float> t_a_output = {
      5544, 5688, 5832, 5742, 5868, 5994, 234, 252, 270,
      5688, 5838, 5988, 5877, 6006, 6135, 261, 282, 303,
      5832, 5988, 6144, 6012, 6144, 6276, 288, 312, 336,
      5742, 5877, 6012, 7947, 8154, 8361, 2016, 2088, 2160,
      5868, 6006, 6144, 8154, 8367, 8580, 2097, 2172, 2247,
      5994, 6135, 6276, 8361, 8580, 8799, 2178, 2256, 2334,
      234, 261, 288, 2016, 2097, 2178, 2574, 2682, 2790,
      252, 282, 312, 2088, 2172, 2256, 2682, 2796, 2910,
      270, 303, 336, 2160, 2247, 2334, 2790, 2910, 3030};

  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddAttribute("transA", int64_t{1});
    tester.AddSparseCooInput("A", A_shape, A_values, A_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, t_a_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }

    // Transpose B output
  const std::vector<float> t_b_output = {
      55, 145, 235, 266, 338, 410, 113, 131, 149,
      145, 451, 757, 662, 842, 1022, 779, 905, 1031,
      235, 757, 1279, 1058, 1346, 1634, 1445, 1679, 1913,
      266, 662, 1058, 2539, 3277, 4015, 2282, 2624, 2966,
      338, 842, 1346, 3277, 4231, 5185, 3002, 3452, 3902,
      410, 1022, 1634, 4015, 5185, 6355, 3722, 4280, 4838,
      113, 779, 1445, 2282, 3002, 3722, 8911, 10297, 11683,
      131, 905, 1679, 2624, 3452, 4280, 10297, 11899, 13501,
      149, 1031, 1913, 2966, 3902, 4838, 11683, 13501, 15319};

  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddAttribute("transB", int64_t{1});
    tester.AddSparseCooInput("A", A_shape, A_values, A_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, t_b_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
  // Transpose both A and B
  const std::vector<float> t_a_b_output = {
      546, 1410, 2274, 2784, 3540, 4296, 678, 786, 894,
      561, 1461, 2361, 2850, 3624, 4398, 789, 915, 1041,
      576, 1512, 2448, 2916, 3708, 4500, 900, 1044, 1188,
      552, 1362, 2172, 4362, 5604, 6846, 2892, 3324, 3756,
      564, 1392, 2220, 4485, 5763, 7041, 3012, 3462, 3912,
      576, 1422, 2268, 4608, 5922, 7236, 3132, 3600, 4068,
      39, 201, 363, 1551, 2037, 2523, 4263, 4911, 5559,
      42, 222, 402, 1608, 2112, 2616, 4494, 5178, 5862,
      45, 243, 441, 1665, 2187, 2709, 4725, 5445, 6165};

  {
    OpTester tester("SparseToDenseMatMul", 1, onnxruntime::kMSDomain);
    tester.AddAttribute("transA", int64_t{1});
    tester.AddAttribute("transB", int64_t{1});
    tester.AddSparseCooInput("A", A_shape, A_values, A_indices);
    tester.AddInput("B", B_shape, B_data);
    tester.AddOutput("X", X_shape, t_a_b_output);
    tester.Run(OpTester::ExpectResult::kExpectSuccess);
  }
}

}  // namespace test
}  // namespace onnxruntime
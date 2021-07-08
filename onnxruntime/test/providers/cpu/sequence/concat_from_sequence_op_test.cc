// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(SequenceOpsTest, ConcatFromSequence_Stack_Axis0) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<int64_t>("new_axis", 1);  // stack mode
  SeqTensors<float> input;
  input.AddTensor({1, 2}, {0.0f, 1.0f});
  input.AddTensor({1, 2}, {2.0f, 3.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<float>("I", {2, 1, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Stack_Axis1) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("new_axis", 1);  // stack mode
  SeqTensors<int32_t> input;
  input.AddTensor({1, 2}, {0, 1});
  input.AddTensor({1, 2}, {2, 3});
  test.AddSeqInput("S", input);
  test.AddOutput<int32_t>("I", {1, 2, 2}, {0, 1, 2, 3});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Stack_Axis2) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddAttribute<int64_t>("new_axis", 1);  // stack mode
  SeqTensors<int64_t> input;
  input.AddTensor({1, 2}, {0, 1});
  input.AddTensor({1, 2}, {2, 3});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {1, 2, 2}, {0, 2, 1, 3});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Stack_Axis1_WithEmptyInput) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("new_axis", 1);  // stack mode
  SeqTensors<int64_t> input;
  input.AddTensor({1, 0}, {});
  input.AddTensor({1, 0}, {});
  input.AddTensor({1, 0}, {});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {1, 3, 0}, {});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Stack_ScalarInputs) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<int64_t>("new_axis", 1);  // stack mode
  SeqTensors<int64_t> input;
  input.AddTensor({}, {1});
  input.AddTensor({}, {2});
  input.AddTensor({}, {3});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {3}, {1, 2, 3});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Concat_Axis0) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<int64_t>("new_axis", 0);  // concat mode
  SeqTensors<float> input;
  input.AddTensor({1, 2}, {0.0f, 1.0f});
  input.AddTensor({1, 2}, {2.0f, 3.0f});
  test.AddSeqInput("S", input);
  test.AddOutput<float>("I", {2, 2}, {0.0f, 1.0f, 2.0f, 3.0f});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Concat_Axis1) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("new_axis", 0);  // concat mode
  SeqTensors<int32_t> input;
  input.AddTensor({1, 2}, {0, 1});
  input.AddTensor({1, 2}, {2, 3});
  test.AddSeqInput("S", input);
  test.AddOutput<int32_t>("I", {1, 4}, {0, 1, 2, 3});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Concat_Axis2) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddAttribute<int64_t>("new_axis", 0);  // concat mode
  SeqTensors<int64_t> input;
  input.AddTensor({1, 2}, {0, 1});
  input.AddTensor({1, 2}, {2, 3});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {1, 2, 2}, {0, 2, 1, 3});
  test.Run(OpTester::ExpectResult::kExpectFailure, "axis 2 is not in valid range [-2,1]");
}

TEST(SequenceOpsTest, ConcatFromSequence_Concat_Axis1_WithEmptyInput) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<int64_t>("new_axis", 0);  // concat mode
  SeqTensors<int64_t> input;
  input.AddTensor({1, 0}, {});
  input.AddTensor({1, 0}, {});
  input.AddTensor({1, 0}, {});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {1, 0}, {});
  test.Run();
}

TEST(SequenceOpsTest, ConcatFromSequence_Concat_ScalarInputs) {
  OpTester test("ConcatFromSequence", 11);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<int64_t>("new_axis", 0);  // concat mode
  SeqTensors<int64_t> input;
  input.AddTensor({}, {1});
  input.AddTensor({}, {2});
  input.AddTensor({}, {3});
  test.AddSeqInput("S", input);
  test.AddOutput<int64_t>("I", {3}, {1, 2, 3});
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Cannot concatenate scalars");
}

}  // namespace test
}  // namespace onnxruntime

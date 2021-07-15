// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dataitem_request.h"

#include "heap_buffer.h"
#include "TestCase.h"
#include "test/compare_ortvalue.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/platform/threadpool.h"
#include <core/session/onnxruntime_cxx_api.h>

#include <memory>

namespace onnxruntime {
namespace test {

std::pair<EXECUTE_RESULT, TIME_SPEC> DataTaskRequestContext::Run(const ITestCase& c, ::Ort::Session& session,
                                                                 OrtAllocator* allocator, size_t task_id) {
  std::pair<EXECUTE_RESULT, TIME_SPEC> result;
  Callback empty_cb;
  DataTaskRequestContext ctx(empty_cb, c, session, allocator, task_id);
  ORT_TRY {
    result = ctx.RunImpl();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      result = std::make_pair(EXECUTE_RESULT::WITH_EXCEPTION, ctx.GetTimeSpent());
      LOGS_DEFAULT(ERROR) << ctx.test_case_.GetTestCaseName() << ":" << ex.what();
    });
  }
  return result;
}

void DataTaskRequestContext::Request(const Callback& cb, concurrency::ThreadPool* tp,
                                     const ITestCase& c, Ort::Session& session,
                                     OrtAllocator* allocator, size_t task_id) {
  assert(cb);
  std::unique_ptr<DataTaskRequestContext> self(new DataTaskRequestContext(cb, c, session, allocator, task_id));
  CallableFactory<DataTaskRequestContext, void> f(self.get());
  auto runnable = f.GetCallable<&DataTaskRequestContext::RunAsync>();
  onnxruntime::concurrency::ThreadPool::Schedule(tp, [runnable]() { runnable.Invoke(); });
  self.release();
}

void DataTaskRequestContext::RunAsync() {
  std::pair<EXECUTE_RESULT, TIME_SPEC> result;
  ORT_TRY {
    result = RunImpl();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      result = std::make_pair(EXECUTE_RESULT::WITH_EXCEPTION, spent_time_);
      LOGS_DEFAULT(ERROR) << test_case_.GetTestCaseName() << ":" << ex.what();
    });
  }

  assert(cb_);
  std::unique_ptr<DataTaskRequestContext> self(this);
  cb_.Invoke(task_id_, result.first, spent_time_);
}

std::pair<EXECUTE_RESULT, TIME_SPEC> DataTaskRequestContext::RunImpl() {
  onnxruntime::test::HeapBuffer holder;
  std::unordered_map<std::string, Ort::Value> feeds;
  test_case_.LoadTestData(task_id_, holder, feeds, true);

  std::vector<const char*> input_names(feeds.size());
  std::vector<OrtValue*> input_values;
  input_values.reserve(feeds.size());
  {
    size_t input_index = 0;
    for (auto& kvp : feeds) {
      input_names[input_index] = kvp.first.c_str();
      input_values.push_back(kvp.second);  // automatic cast
      ++input_index;
    }
  }

  // Create output feed
  size_t output_count = session_.GetOutputCount();
  std::vector<std::string> output_names(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name = session_.GetOutputName(i, default_allocator_);
    assert(output_name != nullptr);
    output_names[i] = output_name;
    Ort::ThrowOnError(Ort::GetApi().AllocatorFree(default_allocator_, output_name));
  }

  TIME_SPEC start_time;
  TIME_SPEC end_time;
  std::vector<const char*> output_names_raw_ptr(output_count);
  std::vector<Ort::Value> output_values;
  output_values.reserve(output_names.size());
  {
    for (size_t i = 0; i != output_count; ++i) {
      output_names_raw_ptr[i] = output_names[i].c_str();
      output_values.emplace_back(nullptr);
    }
  }

  GetMonotonicTimeCounter(&start_time);
  assert(input_names.size() == input_values.size());

  Ort::ThrowOnError(Ort::GetApi().Run(session_, nullptr, input_names.data(), input_values.data(),
                                      input_values.size(), output_names_raw_ptr.data(),
                                      output_count, reinterpret_cast<OrtValue**>(output_values.data())));
  GetMonotonicTimeCounter(&end_time);
  AccumulateTimeSpec(&spent_time_, &start_time, &end_time);

  double per_sample_tolerance;
  double relative_per_sample_tolerance;
  bool post_procesing;
  Status status;
  test_case_.GetPerSampleTolerance(&per_sample_tolerance);
  test_case_.GetRelativePerSampleTolerance(&relative_per_sample_tolerance);
  test_case_.GetPostProcessing(&post_procesing);

  std::unordered_map<std::string, Ort::Value> expected_output_values;
  test_case_.LoadTestData(task_id_, holder, expected_output_values, false);

  std::unordered_map<std::string, OrtValue*> name_fetch_output_map;
  std::unordered_map<std::string, const ONNX_NAMESPACE::ValueInfoProto*> name_output_value_info_proto;
  size_t i = 0;
  for (auto& output_name : output_names) {
    // p_fetches is filled in the order of output_names.
    name_fetch_output_map[output_name] = output_values[i];  // Automatic cast
    const ONNX_NAMESPACE::ValueInfoProto* infoProto = test_case_.GetOutputInfoFromModel(i);
    if (infoProto != nullptr) {
      name_output_value_info_proto.insert(std::make_pair(infoProto->name(), infoProto));
    }
    i++;
  }

  EXECUTE_RESULT res = EXECUTE_RESULT::SUCCESS;
  for (auto& output : expected_output_values) {
    const std::string& output_name = output.first;
    OrtValue* expected_output_value = output.second;  // Automatic cast
    auto iter = name_fetch_output_map.find(output_name);
    if (iter == name_fetch_output_map.end()) {
      res = EXECUTE_RESULT::INVALID_GRAPH;
      LOGF_DEFAULT(ERROR, "cannot find %s in the outputs", output_name.c_str());
      break;
    }
    OrtValue* actual_output_value = iter->second;

    std::pair<COMPARE_RESULT, std::string> ret{COMPARE_RESULT::SUCCESS, ""};

    // Expected output is not None
    if (expected_output_value != nullptr) {
      // Actual output is None
      if (!actual_output_value->HasElement()) {
        ret = std::pair<COMPARE_RESULT, std::string>{
            COMPARE_RESULT::RESULT_DIFFERS,
            "Expected non-None output but received an OrtValue that is None"};
      } else {  // Both expect and actual OrtValues are not None, proceed with data checking
        ret =
            CompareOrtValue(*actual_output_value, *expected_output_value, per_sample_tolerance,
                            relative_per_sample_tolerance, post_procesing);
      }
    } else {  // Expected output is None, ensure that the received output OrtValue is None as well
      if (actual_output_value->HasElement()) {
        ret = std::pair<COMPARE_RESULT, std::string>{
            COMPARE_RESULT::RESULT_DIFFERS,
            "Expected None output but received an OrtValue that is not None"};
      }
    }

    COMPARE_RESULT compare_result = ret.first;
    if (compare_result == COMPARE_RESULT::SUCCESS) {
      const ONNX_NAMESPACE::ValueInfoProto* v = name_output_value_info_proto[output_name];
      if (v == nullptr) continue;
      ret = VerifyValueInfo(*v, Ort::Unowned<Ort::Value>{actual_output_value});
      compare_result = ret.first;
      if (compare_result != COMPARE_RESULT::SUCCESS) {
        switch (compare_result) {
          case COMPARE_RESULT::NOT_SUPPORT:
            res = EXECUTE_RESULT::NOT_SUPPORT;
            break;
          case COMPARE_RESULT::SHAPE_MISMATCH:
            res = EXECUTE_RESULT::MODEL_SHAPE_MISMATCH;
            break;
          case COMPARE_RESULT::TYPE_MISMATCH:
            res = EXECUTE_RESULT::MODEL_TYPE_MISMATCH;
            break;
          default:
            res = EXECUTE_RESULT::UNKNOWN_ERROR;
        }
      }
    } else {
      switch (compare_result) {
        case COMPARE_RESULT::NOT_SUPPORT:
          res = EXECUTE_RESULT::NOT_SUPPORT;
          break;
        case COMPARE_RESULT::RESULT_DIFFERS:
          res = EXECUTE_RESULT::RESULT_DIFFERS;
          break;
        case COMPARE_RESULT::SHAPE_MISMATCH:
          res = EXECUTE_RESULT::SHAPE_MISMATCH;
          break;
        case COMPARE_RESULT::TYPE_MISMATCH:
          res = EXECUTE_RESULT::TYPE_MISMATCH;
          break;
        default:
          res = EXECUTE_RESULT::UNKNOWN_ERROR;
      }
    }

    if (compare_result != COMPARE_RESULT::SUCCESS && !ret.second.empty()) {
      LOGS_DEFAULT(ERROR) << test_case_.GetTestCaseName() << ":output=" << output_name << ":" << ret.second;
    }
    if (compare_result != COMPARE_RESULT::SUCCESS) {
      break;
    }
  }
  return std::make_pair(res, spent_time_);
}

}  // namespace test
}  // namespace onnxruntime

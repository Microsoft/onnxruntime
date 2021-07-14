// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include "TestCase.h"
#include "TestCaseResult.h"
#include "TestResultStat.h"
#include "testcase_driver.h"

#include "core/platform/env.h"
#include "core/platform/threadpool.h"
#include "core/platform/threadpoollite.h"

using onnxruntime::Status;

static std::unique_ptr<onnxruntime::concurrency::ThreadPool> default_pool;
static std::once_flag default_pool_init;

PThreadPool TestEnv::GetDefaultThreadPool(onnxruntime::Env& env) {
  std::call_once(default_pool_init, [&env] {
    using namespace onnxruntime::concurrency;
    int core_num = env.GetNumCpuCores();

    onnxruntime::ThreadOptions t_opts;
    default_pool = std::make_unique<ThreadPool>(&env, t_opts, ORT_TSTR("onnx_runner_tp"), core_num, false);
  });
  return default_pool.get();
}

TestEnv::TestEnv(Ort::Env& env, Ort::SessionOptions& so, PThreadPool tp,
                 std::vector<ITestCase*>&& tests, TestResultStat& stat)
    : env_(env),
      so_(so),
      tp_(tp),
      tests_(std::move(tests)),
      stat_(stat) {
}

TestEnv::~TestEnv() {
  // need dtor in .cc so 'finished' can be cleaned up as TestCaseResult only has a forward declaration in the header.
}

Status TestEnv::Run(size_t parallel_models, int concurrent_runs, size_t repeat_count) {
  std::vector<std::shared_ptr<TestCaseResult>> results;
  if (parallel_models > 1U && tests_.size() > 1U) {
    results = onnxruntime::test::TestCaseDriver::RunParallel(*this, parallel_models, concurrent_runs);
  } else {
    results = onnxruntime::test::TestCaseDriver::Run(*this, concurrent_runs, repeat_count);
  }

  CalculateStats(results);

  return Status::OK();
}

static inline void AddFailedName(const TestCaseResult& r, TestResultStat& stat) {
  const auto& name = r.GetName();
  if (!name.empty()) {
    stat.AddFailedKernels(name);
  } else {
    stat.AddFailedKernels("EMPTY FAILED TEST CASE NAME!");
  }
}

void TestEnv::CalculateStats(const std::vector<std::shared_ptr<TestCaseResult>>& results) {
  ORT_ENFORCE(tests_.size() == results.size(), "Should have received results for all the test cases");
  stat_.total_model_count = tests_.size();
  stat_.total_test_case_count = std::accumulate(std::begin(tests_), std::end(tests_), size_t{0},
                                                [](size_t v, const ITestCase* c) {
                                                  return c->GetDataCount() + v;
                                                });

  for (size_t i = 0; i != tests_.size(); ++i) {
    const auto& test = tests_[i];
    // Missing result
    if (!results[i]) {
      stat_.AddFailedTest(std::make_pair(test->GetTestCaseName(), test->GetTestCaseVersion()));
      continue;
    }

    const TestCaseResult& r = *results[i];

    for (auto res : r.GetExcutionResult()) {
      if (res != EXECUTE_RESULT::SUCCESS && res != EXECUTE_RESULT::NOT_SUPPORT) {
        stat_.AddFailedTest(std::make_pair(test->GetTestCaseName(), test->GetTestCaseVersion()));
      }
      switch (res) {
        case EXECUTE_RESULT::SUCCESS:
          stat_.succeeded++;
          break;
        case EXECUTE_RESULT::INVALID_ARGUMENT:
        case EXECUTE_RESULT::UNKNOWN_ERROR:
          AddFailedName(r, stat_);
          break;
        case EXECUTE_RESULT::INVALID_GRAPH:
          stat_.invalid_graph++;
          break;
        case EXECUTE_RESULT::WITH_EXCEPTION:
          stat_.throwed_exception++;
          AddFailedName(r, stat_);
          break;
        case EXECUTE_RESULT::RESULT_DIFFERS:
        case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
        case EXECUTE_RESULT::SHAPE_MISMATCH:
        case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
        case EXECUTE_RESULT::TYPE_MISMATCH:
          stat_.result_differs++;
          AddFailedName(r, stat_);
          break;
        case EXECUTE_RESULT::NOT_SUPPORT:
          stat_.not_implemented++;
          stat_.AddNotImplementedKernels(r.GetName());
          break;
        case EXECUTE_RESULT::LOAD_MODEL_FAILED:
          stat_.load_model_failed++;
          AddFailedName(r, stat_);
          break;
        default: {
          auto s = onnxruntime::MakeString(r.GetName().c_str(), " Unknown result enum: ", res);
          stat_.AddFailedKernels(s);
        } break;
      }
    }
  }
}

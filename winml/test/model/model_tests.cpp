#include "testPch.h"
#include "test/onnx/TestCase.h"
#include "test/onnx/heap_buffer.h"
#include "test/util/include/test/compare_ortvalue.h"
#include "ort_value_helper.h"
#include "onnxruntime_cxx_api.h"
#include "StringHelpers.h"
#include "skip_model_tests.h"
#include "compare_feature_value.h"
#include <regex>
#include "CommonDeviceHelpers.h"

#ifndef BUILD_GOOGLE_TEST
#error Must use googletest for value-parameterized tests
#endif

using namespace onnxruntime::test;
using namespace winml;
using namespace onnxruntime;
using namespace winrt::Windows::Foundation::Collections;

namespace WinML {
// Global needed to keep the actual ITestCase alive while the tests are going on. Only ITestCase* are used as test parameters.
std::vector<std::unique_ptr<ITestCase>> ownedTests;

class ModelTest : public testing::TestWithParam<std::tuple<ITestCase*, winml::LearningModelDeviceKind>> {
 protected:
  void SetUp() override {
#ifdef BUILD_INBOX
    winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
    std::tie(m_testCase, m_deviceKind) = GetParam();
    WINML_EXPECT_NO_THROW(m_testCase->GetPerSampleTolerance(&m_perSampleTolerance));
    WINML_EXPECT_NO_THROW(m_testCase->GetRelativePerSampleTolerance(&m_relativePerSampleTolerance));
    WINML_EXPECT_NO_THROW(m_testCase->GetPostProcessing(&m_postProcessing));

    // DirectML runs needs a higher relativePerSampleTolerance to handle GPU variability in results.
#ifdef USE_DML
    if (m_deviceKind == winml::LearningModelDeviceKind::DirectX) {
      m_relativePerSampleTolerance = 0.009;  // tolerate up to 0.9% difference of expected result.
      auto gpuSampleTolerancePerTestsItr = gpuSampleTolerancePerTests.find(m_testCase->GetTestCaseName());
      if (gpuSampleTolerancePerTestsItr != gpuSampleTolerancePerTests.end()) {
        m_perSampleTolerance = gpuSampleTolerancePerTestsItr->second;
      }
    }
#endif
  }
  // Called after the last test in this test suite.
  static void TearDownTestSuite() {
    ownedTests.clear();  // clear the global vector
  }
  winml::LearningModelDeviceKind m_deviceKind;
  ITestCase* m_testCase;
  double m_perSampleTolerance = 1e-3;
  double m_relativePerSampleTolerance = 1e-3;
  bool m_postProcessing = false;

  void BindInputsFromFeed(LearningModelBinding& binding, std::unordered_map<std::string, Ort::Value>& feed) {
    for (auto& [name, value] : feed) {
      ITensor bindingValue;
      WINML_EXPECT_NO_THROW(bindingValue = OrtValueHelpers::LoadTensorFromOrtValue(value));
      WINML_EXPECT_NO_THROW(binding.Bind(_winml::Strings::WStringFromString(name), bindingValue));
    }
  }

  void CompareEvaluationResults(LearningModelEvaluationResult& results,
                                std::unordered_map<std::string,
                                Ort::Value>& expectedOutputFeeds,
                                const IVectorView<ILearningModelFeatureDescriptor>& outputFeatureDescriptors) {
    for (const auto& [name, value] : expectedOutputFeeds) {
      // Extract the output buffer from the evaluation output
      std::wstring outputName = _winml::Strings::WStringFromString(name);
      
      // find the output descriptor
      ILearningModelFeatureDescriptor outputDescriptor = nullptr;
      for (const auto& descriptor : outputFeatureDescriptors) {
        if (descriptor.Name() == outputName) {
          outputDescriptor = descriptor;
          break;
        }
      }
      if (outputDescriptor == nullptr) {
        throw std::invalid_argument("Expected protobuf output name doesn't match the output names in the model.");
      }

      if (outputDescriptor.Kind() == LearningModelFeatureKind::Tensor) {
        auto actualOutputTensorValue = results.Outputs().Lookup(outputName).as<ITensor>();
        Ort::Value actualOutput = OrtValueHelpers::CreateOrtValueFromITensor(actualOutputTensorValue);
        // Use the expected and actual OrtValues to compare
        std::pair<COMPARE_RESULT, std::string> ret = CompareOrtValue(*actualOutput, *value, m_perSampleTolerance, m_relativePerSampleTolerance, m_postProcessing);
        WINML_EXPECT_EQUAL(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;
      } else if (outputDescriptor.Kind() == LearningModelFeatureKind::Sequence) {
        auto sequenceOfMapsStringToFloat = results.Outputs().Lookup(outputName).try_as<IVectorView<IMap<winrt::hstring, float>>>();
        if (sequenceOfMapsStringToFloat != nullptr) {
          WINML_EXPECT_TRUE(CompareFeatureValuesHelper::CompareSequenceOfMapsStringToFloat(
              sequenceOfMapsStringToFloat,
              value,
              m_perSampleTolerance,
              m_relativePerSampleTolerance));
        } else {
          throw winrt::hresult_not_implemented(L"This particular type of sequence output hasn't been handled yet.");
        }
      }
    }
  }
};

TEST_P(ModelTest, Run) {
  LearningModel model = nullptr;
  LearningModelDevice device = nullptr;
  LearningModelSession session = nullptr;
  LearningModelBinding binding = nullptr;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(m_testCase->GetModelUrl()));
  WINML_EXPECT_NO_THROW(device = LearningModelDevice(m_deviceKind));
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device));
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  for (size_t i = 0; i < m_testCase->GetDataCount(); i++) {
    // Load and bind inputs
    onnxruntime::test::HeapBuffer inputHolder;
    std::unordered_map<std::string, Ort::Value> inputFeeds;
    WINML_EXPECT_NO_THROW(m_testCase->LoadTestData(i, inputHolder, inputFeeds, true));
    WINML_EXPECT_NO_THROW(BindInputsFromFeed(binding, inputFeeds));

    // evaluate
    LearningModelEvaluationResult results = nullptr;
    WINML_EXPECT_NO_THROW(results = session.Evaluate(binding, L"Testing"));

    // Load expected outputs
    onnxruntime::test::HeapBuffer outputHolder;
    std::unordered_map<std::string, Ort::Value> outputFeeds;
    WINML_EXPECT_NO_THROW(m_testCase->LoadTestData(i, outputHolder, outputFeeds, false));

    // compare results
    CompareEvaluationResults(results, outputFeeds, model.OutputFeatures());
  }
}

// Get the path of the model test collateral. Will return empty string if it doesn't exist.
std::string GetTestDataPath() {
  std::string testDataPath(MAX_PATH, '\0');
  auto environmentVariableFetchSuceeded = GetEnvironmentVariableA("WINML_TEST_DATA_PATH", testDataPath.data(), MAX_PATH);
  if (environmentVariableFetchSuceeded == 0 && GetLastError() == ERROR_ENVVAR_NOT_FOUND || environmentVariableFetchSuceeded > MAX_PATH) {
    // if the WINML_TEST_DATA_PATH environment variable cannot be found, attempt to find the hardcoded models folder
    std::wstring modulePath = FileHelpers::GetModulePath();
    std::filesystem::path currPath = modulePath.substr(0, modulePath.find_last_of(L"\\"));
    std::filesystem::path parentPath = currPath.parent_path();
    auto hardcodedModelPath = parentPath.string() + "\\models";
    if (std::filesystem::exists(hardcodedModelPath) && hardcodedModelPath.length() <= MAX_PATH) {
      return hardcodedModelPath;
    } else {
      std::string errorStr = "WINML_TEST_DATA_PATH environment variable path not found and \"models\" folder not found in same directory as test exe.\n";
      std::cerr << errorStr;
      throw std::exception(errorStr.c_str());
    }
  }
  const std::string testDataPathFolderName = "\\testData\\";
  if (MAX_PATH - environmentVariableFetchSuceeded >= testDataPathFolderName.length()) {
    testDataPath.replace(environmentVariableFetchSuceeded,
        testDataPathFolderName.length(),
        testDataPathFolderName);
  } else {
    throw std::exception("WINML_TEST_DATA_PATH environment variable path needs to be shorter to accomodate the maximum path size of %d\n", MAX_PATH);
  }
  return testDataPath;
}

// This function returns the list of all test cases inside model test collateral
static std::vector<ITestCase*> GetAllTestCases() {
  std::vector<ITestCase*> tests;
  std::vector<std::basic_string<PATH_CHAR_TYPE>> whitelistedTestCases;
  double perSampleTolerance = 1e-3;
  double relativePerSampleTolerance = 1e-3;
  std::unordered_set<std::basic_string<ORTCHAR_T>> allDisabledTests;
  std::vector<std::basic_string<PATH_CHAR_TYPE>> dataDirs;
  auto testDataPath = GetTestDataPath();
  if (testDataPath == "") return tests;

  for (auto& p : std::filesystem::directory_iterator(testDataPath.c_str())) {
    if (p.is_directory()) {
      dataDirs.push_back(std::move(p.path()));
    }
  }

  #if !defined(__amd64__) && !defined(_M_AMD64)
  // Should match "x86_disabled_tests" in onnxruntime/test/providers/cpu/model_tests.cc
  // However there are more tests skipped. TODO: bugs must be filed for difference in models.
  static const ORTCHAR_T* x86DisabledTests[] = {
      ORT_TSTR("BERT_Squad"),
      ORT_TSTR("bvlc_reference_rcnn_ilsvrc13"),
      ORT_TSTR("bvlc_reference_caffenet"),
      ORT_TSTR("bvlc_alexnet"),
      ORT_TSTR("coreml_AgeNet_ImageNet"),
      ORT_TSTR("coreml_Resnet50"),
      ORT_TSTR("coreml_VGG16_ImageNet"),
      ORT_TSTR("faster_rcnn"),
      ORT_TSTR("fp16_test_tiny_yolov2"),
      ORT_TSTR("GPT2"),
      ORT_TSTR("GPT2_LM_HEAD"),
      ORT_TSTR("keras_lotus_resnet3D"),
      ORT_TSTR("keras2coreml_Dense_ImageNet"),
      ORT_TSTR("mask_rcnn_keras"),
      ORT_TSTR("mask_rcnn"),
      ORT_TSTR("mlperf_ssd_resnet34_1200"),
      ORT_TSTR("resnet50"),
      ORT_TSTR("resnet50v2"),
      ORT_TSTR("resnet152v2"),
      ORT_TSTR("resnet101v2"),
      ORT_TSTR("resnet34v2"),
      ORT_TSTR("roberta_sequence_classification"),
      ORT_TSTR("ssd"),
      ORT_TSTR("tf_inception_resnet_v2"),
      ORT_TSTR("tf_inception_v4"),
      ORT_TSTR("tf_nasnet_large"),
      ORT_TSTR("tf_pnasnet_large"),
      ORT_TSTR("tf_resnet_v1_50"),
      ORT_TSTR("tf_resnet_v1_101"),
      ORT_TSTR("tf_resnet_v1_152"),
      ORT_TSTR("tf_resnet_v2_50"),
      ORT_TSTR("tf_resnet_v2_101"),
      ORT_TSTR("tf_resnet_v2_152"),
      ORT_TSTR("vgg19"),
      ORT_TSTR("yolov3"),
      ORT_TSTR("zfnet512")
  };
  allDisabledTests.insert(std::begin(x86DisabledTests), std::end(x86DisabledTests));
#endif
allDisabledTests.insert(ORT_TSTR("cntk_simple_seg"));

  
  WINML_EXPECT_NO_THROW(LoadTests(dataDirs, whitelistedTestCases, perSampleTolerance, relativePerSampleTolerance,
                                  allDisabledTests,
                                  [&tests](std::unique_ptr<ITestCase> l) {
                                    tests.push_back(l.get());
                                    ownedTests.push_back(std::move(l));
                                  }));
  return tests;
}

bool ShouldSkipTestOnGpuAdapterDxgi(std::string& testName) {
  winrt::com_ptr<IDXGIFactory1> spFactory;
  winrt::com_ptr<IDXGIAdapter1> spAdapter;
  UINT i = 0;
  WINML_EXPECT_HRESULT_SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(spFactory.put())));
  while (spFactory->EnumAdapters1(i, spAdapter.put()) != DXGI_ERROR_NOT_FOUND) {
    DXGI_ADAPTER_DESC1 pDesc;
    WINML_EXPECT_HRESULT_SUCCEEDED(spAdapter->GetDesc1(&pDesc));
    
    // Check if WARP adapter
    // see here for documentation on filtering WARP adapter:
    // https://docs.microsoft.com/en-us/windows/desktop/direct3ddxgi/d3d10-graphics-programming-guide-dxgi#new-info-about-enumerating-adapters-for-windows-8
    auto isBasicRenderDriverVendorId = pDesc.VendorId == 0x1414;
    auto isBasicRenderDriverDeviceId = pDesc.DeviceId == 0x8c;
    auto isSoftwareAdapter = pDesc.Flags == DXGI_ADAPTER_FLAG_SOFTWARE;
    bool isWarpAdapter = isSoftwareAdapter || (isBasicRenderDriverVendorId && isBasicRenderDriverDeviceId);

    if (!isWarpAdapter) {
      // Found an adapter that is not WARP. This is the adapter that will be used by WinML.
      std::string regex = disabledGpuAdapterTests[testName].first;
      std::wstring adapterDescription = pDesc.Description;
      return std::regex_search(
          _winml::Strings::UTF8FromUnicode(adapterDescription.c_str(), adapterDescription.length()),
          std::regex(regex, std::regex_constants::icase | std::regex_constants::nosubs));
    }
    spAdapter = nullptr;
    i++;
  }
  // If no adapters can be enumerated or none of them are hardware, might as well skip this test
  return true;
}
#ifdef ENABLE_DXCORE
bool ShouldSkipTestOnGpuAdapterDxcore(std::string& testName) {
  winrt::com_ptr<IDXCoreAdapterFactory> spFactory;
  WINML_EXPECT_HRESULT_SUCCEEDED(DXCoreCreateAdapterFactory(IID_PPV_ARGS(spFactory.put())));

  winrt::com_ptr<IDXCoreAdapterList> spAdapterList;
  const GUID gpuFilter[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS};
  WINML_EXPECT_HRESULT_SUCCEEDED(spFactory->CreateAdapterList(1, gpuFilter, IID_PPV_ARGS(spAdapterList.put())));

  winrt::com_ptr<IDXCoreAdapter> firstHardwareAdapter;
  
  // select first hardware adapter
  for (uint32_t i = 0; i < spAdapterList->GetAdapterCount(); i++) {
    winrt::com_ptr<IDXCoreAdapter> spCurrAdapter;
    WINML_EXPECT_HRESULT_SUCCEEDED(spAdapterList->GetAdapter(i, IID_PPV_ARGS(spCurrAdapter.put())));

    bool isHardware = false;
    WINML_EXPECT_HRESULT_SUCCEEDED(spCurrAdapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware));

    if (isHardware) {
      // Found an adapter that is not WARP. This is the adapter that will be used by WinML.
      std::string regex = disabledGpuAdapterTests[testName].first;
      std::string adapterDescription;
      WINML_EXPECT_HRESULT_SUCCEEDED(spCurrAdapter->GetProperty(DXCoreAdapterProperty::DriverDescription, &adapterDescription));
      return std::regex_search(
          adapterDescription,
          std::regex(regex, std::regex_constants::icase | std::regex_constants::nosubs));
    }
  }
  // If no adapters can be enumerated or none of them are hardware, might as well skip this test
  return true;
}
#endif

bool ShouldSkipTestOnGpuAdapter(std::string& testName) {
  CommonDeviceHelpers::AdapterEnumerationSupport support;
  if (FAILED(CommonDeviceHelpers::GetAdapterEnumerationSupport(&support))) {
    WINML_LOG_ERROR("Unable to load DXGI or DXCore");
    // If cannot load DXGI or DXCore, then don't run the GPU test
    return true;
  }
  if (support.has_dxgi) {
    return ShouldSkipTestOnGpuAdapterDxgi(testName);
  }
#ifdef ENABLE_DXCORE
  if (support.has_dxcore) {
    return ShouldSkipTestOnGpuAdapterDxcore(testName);
  }
#endif
  // don't skip by default (shouldn't really hit this case)
  return false;
}

// determine if test should be disabled
void DetermineIfDisableTest(std::string& testName, winml::LearningModelDeviceKind deviceKind) {
  bool shouldSkip = false;
  std::string reason = "Reason not found.";
  if (disabledTests.find(testName) != disabledTests.end()) {
    reason = disabledTests.at(testName);
    shouldSkip = true;
  } else if (deviceKind == LearningModelDeviceKind::DirectX) {
    if (SkipGpuTests()) {
      reason = "GPU tests are not enabled for this build.";
      shouldSkip = true;
    } else if (disabledGpuTests.find(testName) != disabledGpuTests.end()) {
      reason = disabledGpuTests.at(testName);
      shouldSkip = true;
    } else if (disabledGpuAdapterTests.find(testName) != disabledGpuAdapterTests.end() && ShouldSkipTestOnGpuAdapter(testName)) {
      reason = disabledGpuAdapterTests[testName].second;
      shouldSkip = true;
    }
  }
  if (shouldSkip) {
    printf("Disabling %s test because : %s\n", testName.c_str(), reason.c_str());
    testName = "DISABLED_" + testName;
  }
}

// This function gets the name of the test
static std::string GetNameOfTest(const testing::TestParamInfo<ModelTest::ParamType>& info) {
  std::string name = "";
  auto modelPath = std::wstring(std::get<0>(info.param)->GetModelUrl());
  auto modelPathStr = _winml::Strings::UTF8FromUnicode(modelPath.c_str(), modelPath.length());
  std::vector<std::string> tokenizedModelPath;
  std::istringstream ss(modelPathStr);
  std::string token;
  while (std::getline(ss, token, '\\')) {
    tokenizedModelPath.push_back(std::move(token));
  }
  // The model path is structured like this "<opset>/<model_name>/model.onnx
  // The desired naming of the test is like this <model_name>_<opset>_<CPU/GPU>
  name += tokenizedModelPath[tokenizedModelPath.size() - 2] += "_";  // model name
  name += tokenizedModelPath[tokenizedModelPath.size() - 3];         // opset version

  std::replace_if(name.begin(), name.end(), [](char c) { return !google::protobuf::ascii_isalnum(c); }, '_');

  auto deviceKind = std::get<1>(info.param);
  // Determine if test should be skipped
  DetermineIfDisableTest(name, deviceKind);
  if (deviceKind == winml::LearningModelDeviceKind::Cpu) {
    name += "_CPU";
  } else {
    name += "_GPU";
  }

  return name;
}

INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTest, testing::Combine(testing::ValuesIn(GetAllTestCases()), testing::Values(winml::LearningModelDeviceKind::Cpu, winml::LearningModelDeviceKind::DirectX)),
                         GetNameOfTest);
}  // namespace WinML
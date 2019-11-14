#include "pch.h"
#include "APITest.h"
#include "SqueezeNetValidator.h"

#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Media.h>
#include "winrt/Windows.Storage.h"
#include "DeviceHelpers.h"

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Storage;

class LearningModelBindingAPITest : public APITest
{};

TEST_F(LearningModelBindingAPITest, CpuSqueezeNet)
{
    std::string cpuInstance("CPU");
    WinML::Engine::Test::ModelValidator::SqueezeNet(cpuInstance, LearningModelDeviceKind::Cpu, /*dataTolerance*/ 0.00001f, false);
}

TEST_F(LearningModelBindingAPITest, CpuSqueezeNetEmptyOutputs)
{
    std::string cpuInstance("CPU");
    WinML::Engine::Test::ModelValidator::SqueezeNet(
        cpuInstance,
        LearningModelDeviceKind::Cpu,
        /*dataTolerance*/ 0.00001f,
        false,
        OutputBindingStrategy::Empty);
}

TEST_F(LearningModelBindingAPITest, CpuSqueezeNetUnboundOutputs)
{
    std::string cpuInstance("CPU");
    WinML::Engine::Test::ModelValidator::SqueezeNet(
        cpuInstance,
        LearningModelDeviceKind::Cpu,
        /*dataTolerance*/ 0.00001f,
        false,
        OutputBindingStrategy::Unbound);
}

TEST_F(LearningModelBindingAPITest, CpuSqueezeNetBindInputTensorAsInspectable)
{
    std::string cpuInstance("CPU");
    WinML::Engine::Test::ModelValidator::SqueezeNet(
        cpuInstance,
        LearningModelDeviceKind::Cpu,
        /*dataTolerance*/ 0.00001f,
        false,
        OutputBindingStrategy::Bound /* empty outputs */,
        true /* bind inputs as inspectables */);
}

TEST_F(LearningModelBindingAPITest, CpuFnsCandy16)
{
    std::string cpuInstance("CPU");
    WinML::Engine::Test::ModelValidator::FnsCandy16(
        cpuInstance,
        LearningModelDeviceKind::Cpu,
        OutputBindingStrategy::Bound,
        true,
        /*dataTolerance*/ 0.00001f);
}

TEST_F(LearningModelBindingAPITest, CpuFnsCandy16UnboundOutputs)
{
    std::string cpuInstance("CPU");
    WinML::Engine::Test::ModelValidator::FnsCandy16(
        cpuInstance,
        LearningModelDeviceKind::Cpu,
        OutputBindingStrategy::Unbound,
        true,
        /*dataTolerance*/ 0.00001f);
}

TEST_F(LearningModelBindingAPITest, CastMapInt64)
{
    EXPECT_NO_THROW(LoadModel(L"castmap-int64.onnx"));
    // TODO: Check Descriptor
}

TEST_F(LearningModelBindingAPITest, DictionaryVectorizerMapInt64)
{
    EXPECT_NO_THROW(LoadModel(L"dictvectorizer-int64.onnx"));

    auto inputDescriptor = m_model.InputFeatures().First().Current();
    EXPECT_TRUE(inputDescriptor.Kind() == LearningModelFeatureKind::Map);
    auto mapDescriptor = inputDescriptor.as<MapFeatureDescriptor>();
    EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::Int64);
    EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);
    auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
    // empty size means tensor of scalar value
    EXPECT_TRUE(tensorDescriptor.Shape().Size() == 0);
    EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

    LearningModelSession modelSession(m_model);
    LearningModelBinding binding(modelSession);
    std::unordered_map<int64_t, float> map;
    map[1] = 1.f;
    map[10] = 10.f;
    map[3] = 3.f;


    auto mapInputName = inputDescriptor.Name();

    // Bind as IMap
    auto abiMap = winrt::single_threaded_map(std::move(map));
    binding.Bind(mapInputName, abiMap);
    auto mapInputInspectable = abiMap.as<winrt::Windows::Foundation::IInspectable>();
    auto first = binding.First();
    EXPECT_TRUE(first.Current().Key() == mapInputName);
    EXPECT_TRUE(first.Current().Value() == mapInputInspectable);
    EXPECT_TRUE(binding.Lookup(mapInputName) == mapInputInspectable);

    // Bind as IMapView
    auto mapView = abiMap.GetView();
    binding.Bind(mapInputName, mapView);
    mapInputInspectable = mapView.as<winrt::Windows::Foundation::IInspectable>();
    first = binding.First();
    EXPECT_TRUE(first.Current().Key() == mapInputName);
    EXPECT_TRUE(first.Current().Value() == mapView);
    EXPECT_TRUE(binding.Lookup(mapInputName) == mapView);

}

TEST_F(LearningModelBindingAPITest, DictionaryVectorizerMapString)
{
    EXPECT_NO_THROW(LoadModel(L"dictvectorizer-string.onnx"));

    auto inputDescriptor = m_model.InputFeatures().First().Current();
    EXPECT_TRUE(inputDescriptor.Kind() == LearningModelFeatureKind::Map);

    auto mapDescriptor = inputDescriptor.as<MapFeatureDescriptor>();
    EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::String);
    EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);

    auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
    // empty size means tensor of scalar value
    EXPECT_TRUE(tensorDescriptor.Shape().Size() == 0);
    EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

    LearningModelSession modelSession(m_model);
    LearningModelBinding binding(modelSession);
    std::unordered_map<winrt::hstring, float> map;
    map[L"1"] = 1.f;
    map[L"10"] = 10.f;
    map[L"2"] = 2.f;

    auto mapInputName = inputDescriptor.Name();
    auto abiMap = winrt::single_threaded_map(std::move(map));
    binding.Bind(mapInputName, abiMap);

    auto mapInputInspectable = abiMap.as<winrt::Windows::Foundation::IInspectable>();
    auto first = binding.First();
    EXPECT_TRUE(first.Current().Key() == mapInputName);
    EXPECT_TRUE(first.Current().Value() == mapInputInspectable);
    EXPECT_TRUE(binding.Lookup(mapInputName) == mapInputInspectable);
}

static void RunZipMapInt64(
    winrt::Windows::AI::MachineLearning::LearningModel model,
    OutputBindingStrategy bindingStrategy)
{
    auto outputFeatures = model.OutputFeatures();
    auto outputDescriptor = outputFeatures.First().Current();
    EXPECT_TRUE(outputDescriptor.Kind() == LearningModelFeatureKind::Sequence);

    auto seqDescriptor = outputDescriptor.as<SequenceFeatureDescriptor>();
    auto mapDescriptor = seqDescriptor.ElementDescriptor().as<MapFeatureDescriptor>();
    EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::Int64);

    EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);
    auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
    EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

    LearningModelSession session(model);
    LearningModelBinding binding(session);

    std::vector<float> inputs = { 0.5f, 0.25f, 0.125f };
    std::vector<int64_t> shape = { 1, 3 };

    // Bind inputs
    auto inputTensor =
        TensorFloat::CreateFromArray(
            shape,
            winrt::array_view<const float>(std::move(inputs)));
    binding.Bind(winrt::hstring(L"X"), inputTensor);

    typedef IMap<int64_t, float> ABIMap;
    typedef IVector<ABIMap> ABISequeneceOfMap;

    ABISequeneceOfMap abiOutput = nullptr;
    // Bind outputs
    if (bindingStrategy == OutputBindingStrategy::Bound)
    {
        abiOutput = winrt::single_threaded_vector<ABIMap>();
        binding.Bind(winrt::hstring(L"Y"), abiOutput);
    }

    // Evaluate
    auto result = session.Evaluate(binding, L"0").Outputs();

    if (bindingStrategy == OutputBindingStrategy::Bound)
    {
        // from output binding
        auto &out1 = abiOutput.GetAt(0);
        auto &out2 = result.Lookup(L"Y").as<IVectorView<ABIMap>>().GetAt(0);
        SCOPED_TRACE((std::ostringstream() << "size: " << out1.Size()).str());
        // check outputs
        auto iter1 = out1.First();
        auto iter2 = out2.First();
        for (uint32_t i = 0, size = (uint32_t)inputs.size(); i < size; ++i)
        {
            EXPECT_TRUE(iter1.HasCurrent());
            EXPECT_TRUE(iter2.HasCurrent());
            auto &pair1 = iter1.Current();
            auto &pair2 = iter2.Current();
            SCOPED_TRACE((std::ostringstream() << "key: " << pair1.Key() << ", value: " << pair2.Value()).str());
            EXPECT_TRUE(pair1.Key() == i && pair2.Key() == i);
            EXPECT_TRUE(pair1.Value() == inputs[i] && pair2.Value() == inputs[i]);
            iter1.MoveNext();
            iter2.MoveNext();
        }
        EXPECT_TRUE(!iter1.HasCurrent());
        EXPECT_TRUE(!iter2.HasCurrent());
    }
    else
    {
        abiOutput = result.Lookup(L"Y").as<ABISequeneceOfMap>();
        EXPECT_TRUE(abiOutput.Size() == 1);
        ABIMap map = abiOutput.GetAt(0);
        EXPECT_TRUE(map.Size() == 3);
        EXPECT_TRUE(map.Lookup(0) == 0.5);
        EXPECT_TRUE(map.Lookup(1) == .25);
        EXPECT_TRUE(map.Lookup(2) == .125);
    }
}

TEST_F(LearningModelBindingAPITest, ZipMapInt64)
{
    EXPECT_NO_THROW(LoadModel(L"zipmap-int64.onnx"));
    RunZipMapInt64(m_model, OutputBindingStrategy::Bound);
}

TEST_F(LearningModelBindingAPITest, ZipMapInt64Unbound)
{
    EXPECT_NO_THROW(LoadModel(L"zipmap-int64.onnx"));
    RunZipMapInt64(m_model, OutputBindingStrategy::Unbound);
}

TEST_F(LearningModelBindingAPITest, ZipMapString)
{
    // output constraint: "seq(map(string, float))" or "seq(map(int64, float))"
    EXPECT_NO_THROW(LoadModel(L"zipmap-string.onnx"));
    auto outputs = m_model.OutputFeatures();
    auto outputDescriptor = outputs.First().Current();
    EXPECT_TRUE(outputDescriptor.Kind() == LearningModelFeatureKind::Sequence);
    auto mapDescriptor = outputDescriptor.as<SequenceFeatureDescriptor>().ElementDescriptor().as<MapFeatureDescriptor>();
    EXPECT_TRUE(mapDescriptor.KeyKind() == TensorKind::String);
    EXPECT_TRUE(mapDescriptor.ValueDescriptor().Kind() == LearningModelFeatureKind::Tensor);
    auto tensorDescriptor = mapDescriptor.ValueDescriptor().as<TensorFeatureDescriptor>();
    EXPECT_TRUE(tensorDescriptor.TensorKind() == TensorKind::Float);

    LearningModelSession session(m_model);
    LearningModelBinding binding(session);

    std::vector<float> inputs = { 0.5f, 0.25f, 0.125f };
    std::vector<int64_t> shape = { 1, 3 };
    std::vector<winrt::hstring> labels = { L"cat", L"dog", L"lion" };
    std::map<winrt::hstring, float> mapData = { { L"cat", 0.0f }, { L"dog", 0.0f }, { L"lion", 0.0f } };
    typedef IMap<winrt::hstring, float> ABIMap;
    ABIMap abiMap = winrt::single_threaded_map<winrt::hstring, float>(std::move(mapData));
    std::vector<ABIMap> seqOutput = { abiMap };
    IVector<ABIMap> ABIOutput = winrt::single_threaded_vector<ABIMap>(std::move(seqOutput));

    TensorFloat inputTensor = TensorFloat::CreateFromArray(shape, winrt::array_view<const float>(std::move(inputs)));
    binding.Bind(winrt::hstring(L"X"), inputTensor);
    binding.Bind(winrt::hstring(L"Y"), ABIOutput);
    auto result = session.Evaluate(binding, L"0").Outputs();
    // from output binding
    auto &out1 = ABIOutput.GetAt(0);
    auto &out2 = result.Lookup(L"Y").as<IVectorView<ABIMap>>().GetAt(0);
    SCOPED_TRACE((std::ostringstream() << "size: " << out1.Size()).str());
    // single key,value pair for each map
    auto iter1 = out1.First();
    auto iter2 = out2.First();
    for (uint32_t i = 0, size = (uint32_t)inputs.size(); i < size; ++i)
    {
        EXPECT_TRUE(iter2.HasCurrent());
        auto &pair1 = iter1.Current();
        auto &pair2 = iter2.Current();
        SCOPED_TRACE((std::ostringstream() << "key: " << pair1.Key().c_str() << ", value " << pair2.Value()).str());
        EXPECT_TRUE(std::wstring(pair1.Key().c_str()).compare(labels[i]) == 0);
        EXPECT_TRUE(std::wstring(pair2.Key().c_str()).compare(labels[i]) == 0);
        EXPECT_TRUE(pair1.Value() == inputs[i] && pair2.Value() == inputs[i]);
        iter1.MoveNext();
        iter2.MoveNext();
    }
    EXPECT_TRUE(!iter1.HasCurrent());
    EXPECT_TRUE(!iter2.HasCurrent());
}

// TEST_F(LearningModelBindingAPITest, GpuSqueezeNet)
// {
//     GPUTEST;

//     std::string gpuInstance("GPU");
//     WinML::Engine::Test::ModelValidator::SqueezeNet(
//         gpuInstance,
//         LearningModelDeviceKind::DirectX,
//         /*dataTolerance*/ 0.00001f);
// }

// TEST_F(LearningModelBindingAPITest, GpuFnsCandy16)
// {
//     GPUTEST;
//     LearningModelDevice device(LearningModelDeviceKind::DirectX);
//     if (!DeviceHelpers::IsFloat16Supported(device))
//     {
//         GTEST_SKIP() << "The current device does not support FP16";
//     }

//     std::string gpuInstance("GPU");
//     WinML::Engine::Test::ModelValidator::FnsCandy16(
//         gpuInstance,
//         LearningModelDeviceKind::DirectX,
//         OutputBindingStrategy::Bound,
//         true,
//         /*dataTolerance*/ 0.00001f);
// }

// TEST_F(LearningModelBindingAPITest, GpuFnsCandy16UnboundOutputs)
// {
//     GPUTEST;

//     LearningModelDevice device(LearningModelDeviceKind::DirectX);
//     if (!DeviceHelpers::IsFloat16Supported(device))
//     {
//         GTEST_SKIP() << "The current device does not support FP16";
//     }

//     std::string gpuInstance("GPU");
//     WinML::Engine::Test::ModelValidator::FnsCandy16(
//         gpuInstance,
//         LearningModelDeviceKind::DirectX,
//         OutputBindingStrategy::Unbound,
//         true,
//         /*dataTolerance*/ 0.00001f);
// }

// TEST_F(LearningModelBindingAPITest, GpuSqueezeNetEmptyOutputs)
// {
//     GPUTEST;

//     std::string gpuInstance("GPU");
//     WinML::Engine::Test::ModelValidator::SqueezeNet(
//         gpuInstance,
//         LearningModelDeviceKind::DirectX,
//         /*dataTolerance*/ 0.00001f,
//         false,
//         OutputBindingStrategy::Empty);
// }

// TEST_F(LearningModelBindingAPITest, GpuSqueezeNetUnboundOutputs)
// {
//     GPUTEST;

//     std::string gpuInstance("GPU");
//     WinML::Engine::Test::ModelValidator::SqueezeNet(
//         gpuInstance,
//         LearningModelDeviceKind::DirectX,
//         /*dataTolerance*/ 0.00001f,
//         false,
//         OutputBindingStrategy::Unbound);
// }

// Validates that when the input image is the same as the model expects, the binding step is executed correctly.
// TEST_F(LearningModelBindingAPITest, ImageBindingDimensions)
// {
//     GPUTEST;
//     LearningModelBinding m_binding = nullptr;
//     std::wstring filePath = FileHelpers::GetModulePath() + L"model.onnx";
//     // load a model with expected input size: 224 x 224
//     EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::Default));
//     EXPECT_NO_THROW(m_model = LearningModel::LoadFromFilePath(filePath));
//     EXPECT_TRUE(m_model != nullptr);
//     EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
//     EXPECT_NO_THROW(m_binding = LearningModelBinding(m_session));

//     // Create input images and execute bind
//     // Test Case 1: both width and height are larger than model expects
//     VideoFrame inputImage1(BitmapPixelFormat::Rgba8, 1000, 1000);
//     ImageFeatureValue inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage1);
//     EXPECT_NO_THROW(m_binding.Bind(L"data_0", inputTensor));

//     // Test Case 2: only height is larger, while width is smaller
//     VideoFrame inputImage2(BitmapPixelFormat::Rgba8, 20, 1000);
//     inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage2);
//     EXPECT_NO_THROW(m_binding.Bind(L"data_0", inputTensor));

//     // Test Case 3: only width is larger, while height is smaller
//     VideoFrame inputImage3(BitmapPixelFormat::Rgba8, 1000, 20);
//     inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage3);
//     EXPECT_NO_THROW(m_binding.Bind(L"data_0", inputTensor));

//     // Test Case 4: both width and height are smaller than model expects
//     VideoFrame inputImage4(BitmapPixelFormat::Rgba8, 20, 20);
//     inputTensor = ImageFeatureValue::CreateFromVideoFrame(inputImage4);
//     EXPECT_NO_THROW(m_binding.Bind(L"data_0", inputTensor));
// }

// TEST_F(LearningModelBindingAPITest, VerifyInvalidBindExceptions)
// {
//     GPUTEST;

//     EXPECT_NO_THROW(LoadModel(L"zipmap-int64.onnx"));

//     LearningModelSession session(m_model);
//     LearningModelBinding binding(session);

//     std::vector<float> inputs = { 0.5f, 0.25f, 0.125f };
//     std::vector<int64_t> shape = { 1, 3 };

//     auto matchException =
//         [](const winrt::hresult_error& e, HRESULT hr) -> bool
//     {
//         return e.code() == hr;
//     };

//     auto ensureWinmlSizeMismatch = std::bind(matchException, std::placeholders::_1, WINML_ERR_SIZE_MISMATCH);
//     auto ensureWinmlInvalidBinding = std::bind(matchException, std::placeholders::_1, WINML_ERR_INVALID_BINDING);

//     /*
//         Verify tensor bindings throw correct bind exceptions
//     */

//     // Bind invalid image as tensorfloat input
//     auto image = FileHelpers::LoadImageFeatureValue(L"doritos_227.png");
//     EXPECT_THROW_SPECIFIC(binding.Bind(L"X", image), winrt::hresult_error, ensureWinmlSizeMismatch);

//     // Bind invalid map as tensorfloat input
//     std::unordered_map<float, float> map;
//     auto abiMap = winrt::single_threaded_map(std::move(map));
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"X", abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid sequence as tensorfloat input
//     std::vector<uint32_t> sequence;
//     auto abiSequence = winrt::single_threaded_vector(std::move(sequence));
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"X", abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor size as tensorfloat input
//     auto tensorBoolean = TensorBoolean::Create();
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"X", tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor shape as tensorfloat input
//     auto tensorInvalidShape = TensorFloat::Create(std::vector<int64_t> { 2, 3, 4 });
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"X", tensorInvalidShape), winrt::hresult_error, ensureWinmlInvalidBinding);

//     /*
//         Verify sequence bindings throw correct bind exceptions
//     */

//     // Bind invalid image as sequence<map<int, float> output
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"Y", image), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid map as sequence<map<int, float> output
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"Y", abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid sequence<int> as sequence<map<int, float> output
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"Y", abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor as sequence<map<int, float> output
//     VERIFY_THROWS_SPECIFIC(binding.Bind(L"Y", tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);

//     /*
//         Verify image bindings throw correct bind exceptions
//     */

//     EXPECT_NO_THROW(LoadModel(L"fns-candy.onnx"));

//     LearningModelSession imageSession(m_model);
//     LearningModelBinding imageBinding(imageSession);

//     auto inputName = m_model.InputFeatures().First().Current().Name();

//     // Bind invalid map as image input
//     VERIFY_THROWS_SPECIFIC(imageBinding.Bind(inputName, abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid sequence as image input
//     VERIFY_THROWS_SPECIFIC(imageBinding.Bind(inputName, abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor type as image input
//     VERIFY_THROWS_SPECIFIC(imageBinding.Bind(inputName, tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor size as image input
//     auto tensorFloat = TensorFloat::Create(std::vector<int64_t> { 1, 1, 100, 100 });
//     VERIFY_THROWS_SPECIFIC(imageBinding.Bind(inputName, tensorFloat), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor shape as image input
//     VERIFY_THROWS_SPECIFIC(imageBinding.Bind(inputName, tensorInvalidShape), winrt::hresult_error, ensureWinmlInvalidBinding);

//     /*
//         Verify map bindings throw correct bind exceptions
//     */
//     EXPECT_NO_THROW(LoadModel(L"dictvectorizer-int64.onnx"));

//     LearningModelSession mapSession(m_model);
//     LearningModelBinding mapBinding(mapSession);

//     inputName = m_model.InputFeatures().First().Current().Name();

//     // Bind invalid image as image input
//     auto smallImage = FileHelpers::LoadImageFeatureValue(L"doritos_100.png");
//     VERIFY_THROWS_SPECIFIC(mapBinding.Bind(inputName, smallImage), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid map as image input
//     VERIFY_THROWS_SPECIFIC(mapBinding.Bind(inputName, abiMap), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid sequence as image input
//     VERIFY_THROWS_SPECIFIC(mapBinding.Bind(inputName, abiSequence), winrt::hresult_error, ensureWinmlInvalidBinding);

//     // Bind invalid tensor type as image input
//     VERIFY_THROWS_SPECIFIC(mapBinding.Bind(inputName, tensorBoolean), winrt::hresult_error, ensureWinmlInvalidBinding);
// }

// // Verify that it throws an error when binding an invalid name.
// TEST_F(LearningModelBindingAPITest, BindInvalidInputName)
// {
//     GPUTEST;
//     LearningModelBinding m_binding = nullptr;
//     std::wstring modelPath = FileHelpers::GetModulePath() + L"Add_ImageNet1920.onnx";
//     EXPECT_NO_THROW(m_model = LearningModel::LoadFromFilePath(modelPath));
//     EXPECT_TRUE(m_model != nullptr);
//     EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::Default));
//     EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
//     EXPECT_NO_THROW(m_binding = LearningModelBinding(m_session));

//     VideoFrame iuputImage(BitmapPixelFormat::Rgba8, 1920, 1080);
//     ImageFeatureValue inputTensor = ImageFeatureValue::CreateFromVideoFrame(iuputImage);

//     auto first = m_model.InputFeatures().First();
//     std::wstring testInvalidName = L"0";

//     // Verify that testInvalidName is not in model's InputFeatures
//     while (first.HasCurrent())
//     {
//         EXPECT_NE(testInvalidName, first.Current().Name());
//         first.MoveNext();
//     }

//     // Bind inputTensor to a valid input name
//     EXPECT_NO_THROW(m_binding.Bind(L"input_39:0", inputTensor));

//     // Bind inputTensor to an invalid input name
//     VERIFY_THROWS_SPECIFIC(m_binding.Bind(testInvalidName, inputTensor),
//         winrt::hresult_error,
//         [](const winrt::hresult_error& e) -> bool
//     {
//         return e.code() == WINML_ERR_INVALID_BINDING;
//     });
// }

TEST_F(LearningModelBindingAPITest, VerifyOutputAfterEvaluateAsyncCalledTwice)
{
    LearningModelBinding m_binding = nullptr;
    std::wstring filePath = FileHelpers::GetModulePath() + L"relu.onnx";
    EXPECT_NO_THROW(m_device = LearningModelDevice(LearningModelDeviceKind::Default));
    EXPECT_NO_THROW(m_model = LearningModel::LoadFromFilePath(filePath));
    EXPECT_TRUE(m_model != nullptr);
    EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
    EXPECT_NO_THROW(m_binding = LearningModelBinding(m_session));

    auto inputShape = std::vector<int64_t>{ 5 };
    auto inputData1 = std::vector<float>{ -50.f, -25.f, 0.f, 25.f, 50.f };
    auto inputValue1 =
        TensorFloat::CreateFromIterable(
            inputShape,
            single_threaded_vector<float>(std::move(inputData1)).GetView());

    auto inputData2 = std::vector<float>{ 50.f, 25.f, 0.f, -25.f, -50.f };
    auto inputValue2 =
        TensorFloat::CreateFromIterable(
            inputShape,
            single_threaded_vector<float>(std::move(inputData2)).GetView());

    EXPECT_NO_THROW(m_binding.Bind(L"X", inputValue1));

    auto outputValue = TensorFloat::Create();
    EXPECT_NO_THROW(m_binding.Bind(L"Y", outputValue));

    EXPECT_NO_THROW(m_session.Evaluate(m_binding, L""));

    auto buffer1 = outputValue.GetAsVectorView();
    EXPECT_TRUE(buffer1 != nullptr);

    // The second evaluation
    // If we don't bind output again, the output value will not change
    EXPECT_NO_THROW(m_binding.Bind(L"X", inputValue2));
    EXPECT_NO_THROW(m_session.Evaluate(m_binding, L""));
    auto buffer2 = outputValue.GetAsVectorView();
    EXPECT_EQ(buffer1.Size(), buffer2.Size());
    bool isSame = true;
    for (uint32_t i = 0; i < buffer1.Size(); ++i)
    {
        if (buffer1.GetAt(i) != buffer2.GetAt(i))
        {
            isSame = false;
            break;
        }
    }
    EXPECT_FALSE(isSame);
}

static VideoFrame CreateVideoFrame(const wchar_t* path)
{
    auto imagefile = StorageFile::GetFileFromPathAsync(path).get();
    auto stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    auto decoder = BitmapDecoder::CreateAsync(stream).get();
    auto softwareBitmap = decoder.GetSoftwareBitmapAsync().get();
    return VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
}

TEST_F(LearningModelBindingAPITest, VerifyOutputAfterImageBindCalledTwice)
{
    std::wstring fullModelPath = FileHelpers::GetModulePath() + L"model.onnx";
    std::wstring fullImagePath1 = FileHelpers::GetModulePath() + L"kitten_224.png";
    std::wstring fullImagePath2 = FileHelpers::GetModulePath() + L"fish.png";

    // winml model creation
    LearningModel model = nullptr;
    EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(fullModelPath));
    LearningModelSession modelSession = nullptr;
    EXPECT_NO_THROW(modelSession = LearningModelSession(model, LearningModelDevice(LearningModelDeviceKind::Default)));
    LearningModelBinding modelBinding(modelSession);

    // create the tensor for the actual output
    auto output = TensorFloat::Create();
    modelBinding.Bind(L"softmaxout_1", output);

    // Bind image 1 and evaluate
    auto frame = CreateVideoFrame(fullImagePath1.c_str());
    auto imageTensor = ImageFeatureValue::CreateFromVideoFrame(frame);
    EXPECT_NO_THROW(modelBinding.Bind(L"data_0", imageTensor));
    EXPECT_NO_THROW(modelSession.Evaluate(modelBinding, L""));

    // Store 1st result
    auto outputVectorView1 = output.GetAsVectorView();

    // Bind image 2 and evaluate
    // In this scenario, the backing videoframe is updated, and the imagefeaturevalue is rebound.
    // The expected result is that the videoframe will be re-tensorized at bind
    auto frame2 = CreateVideoFrame(fullImagePath2.c_str());
    frame2.CopyToAsync(frame).get();
    EXPECT_NO_THROW(modelBinding.Bind(L"data_0", imageTensor));
    EXPECT_NO_THROW(modelSession.Evaluate(modelBinding, L""));

    // Store 2nd result
    auto outputVectorView2 = output.GetAsVectorView();

    EXPECT_EQ(outputVectorView1.Size(), outputVectorView2.Size());
    bool isSame = true;
    for (uint32_t i = 0; i < outputVectorView1.Size(); ++i)
    {
        if (outputVectorView1.GetAt(i) != outputVectorView2.GetAt(i))
        {
            isSame = false;
            break;
        }
    }
    EXPECT_FALSE(isSame);
}

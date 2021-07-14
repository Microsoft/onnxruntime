// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC.  All rights reserved.
// Licensed under the MIT License.

namespace OrtTrainingApis {

ORT_API(const OrtTrainingApi*, GetApi, uint32_t version);
ORT_API(const char*, GetVersionString);

ORT_API(void, ReleaseTrainingParameters, _Frees_ptr_opt_ OrtTrainingParameters*);

ORT_API_STATUS_IMPL(CreateTrainingParameters, _Outptr_ OrtTrainingParameters** out);
ORT_API_STATUS_IMPL(CloneTrainingParameters, _In_ const OrtTrainingParameters* input, _Outptr_ OrtTrainingParameters** out);

ORT_API_STATUS_IMPL(SetTrainingParameter_string, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingStringParameter key, _In_ const ORTCHAR_T* value);
ORT_API_STATUS_IMPL(GetTrainingParameter_string, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingStringParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue);

ORT_API_STATUS_IMPL(SetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingBooleanParameter key, _In_ const bool value);
ORT_API_STATUS_IMPL(GetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingBooleanParameter key, _Out_ bool* pvalue);

ORT_API_STATUS_IMPL(SetTrainingParameter_long, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingLongParameter key, _In_ const long value);
ORT_API_STATUS_IMPL(GetTrainingParameter_long, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingLongParameter key, _Out_ long* pvalue);

ORT_API_STATUS_IMPL(SetTrainingParameter_double, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingNumericParameter key, _In_ const double value);
ORT_API_STATUS_IMPL(GetTrainingParameter_double, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingNumericParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue);

ORT_API_STATUS_IMPL(SetTrainingOptimizer, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingOptimizer opt);
ORT_API_STATUS_IMPL(GetTrainingOptimizer, _In_ OrtTrainingParameters* pParam, _Out_ OrtTrainingOptimizer* popt);

ORT_API_STATUS_IMPL(SetTrainingLossFunction, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingLossFunction loss);
ORT_API_STATUS_IMPL(GetTrainingLossFunction, _In_ OrtTrainingParameters* pParam, _Out_ OrtTrainingLossFunction* ploss);

ORT_API_STATUS_IMPL(SetupTrainingParameters, _In_ OrtTrainingParameters* pParam, OrtErrorFunctionCallback errorFn, OrtEvaluationFunctionCallback evalFn);
ORT_API_STATUS_IMPL(SetupTrainingData, _In_ OrtTrainingParameters* pParam, OrtDataGetBatchCallback trainingdataqueryFn, OrtDataGetBatchCallback testingdataqueryFn, _In_ const ORTCHAR_T* szFeedNames);

ORT_API_STATUS_IMPL(InitializeTraining, _In_ OrtEnv* pEnv, _In_ OrtTrainingParameters* pParam);
ORT_API_STATUS_IMPL(RunTraining, _In_ OrtTrainingParameters* pParam);
ORT_API_STATUS_IMPL(EndTraining, _In_ OrtTrainingParameters* pParam);

ORT_API_STATUS_IMPL(GetCount, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount);
ORT_API_STATUS_IMPL(GetCapacity, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount);
ORT_API_STATUS_IMPL(GetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _Outptr_ OrtValue** output, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppName);
ORT_API_STATUS_IMPL(SetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _In_ OrtValue* output, _In_ const ORTCHAR_T* name);

ORT_API_STATUS_IMPL(GetDimCount, _In_ OrtShape* pShape, _Out_ size_t* pnCount);
ORT_API_STATUS_IMPL(GetDimAt, _In_ OrtShape* pShape, _In_ size_t nIdx, _Out_ size_t* output);
}  // namespace OrtTrainingApis

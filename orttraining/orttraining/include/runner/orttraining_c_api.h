// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum OrtTrainingOptimizer {
    ORT_TRAINING_OPTIMIZER_SGD = 0
};

typedef enum OrtTrainingLossFunction {
    ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY = 0
};

typedef enum OrtTrainingStringParameter {
    ORT_TRAINING_MODEL_PATH = 0,
    ORT_TRAINING_LOG_PATH = 1,
    ORT_TRAINING_INPUT_LABELS = 2,
    ORT_TRAINING_OUTPUT_PREDICTIONS = 3,
    ORT_TRAINING_OUTPUT_LOSS = 4
};

typedef enum OrtTrainingLongParameter {
  ORT_TRAINING_NUM_TRAIN_STEPS = 0,
  ORT_TRAINING_TRAIN_BATCH_SIZE = 1,
  ORT_TRAINING_EVAL_BATCH_SIZE = 2,
  ORT_TRAINING_EVAL_PERIOD = 3,
  ORT_TRAINING_DISPLAY_LOSS_STEPS = 4
};

typedef enum OrtTrainingNumericParameter {
    ORT_TRAINING_LEARNING_RATE = 0,
};

typedef enum OrtTrainingBooleanParameter {
  ORT_TRAINING_USE_GIST = 0,
  ORT_TRAINING_USE_CUDA = 1,
  ORT_TRAINING_USE_PROFILER = 2,
  ORT_TRAINING_USE_TENSORBOARD = 3,
  ORT_TRAINING_IS_PERFTEST = 4,
  ORT_TRAINING_SHUFFLE_DATA = 5
};

typedef enum OrtDataUse {
  ORT_DATAUSE_TRAINING = 0,
  ORT_DATAUSE_TESTING = 1
};

struct OrtTrainingParameters;
struct OrtValueCollection;
struct OrtShape;

typedef void(__stdcall* OrtErrorFunctionCallback)(_In_ OrtValueCollection* output);  // Array of outputs (e.g. label, predictions, loss).
typedef void(__stdcall* OrtEvaluationFunctionCallback)(size_t num_samples, size_t step);
typedef void(__stdcall* OrtDataGetBatchCallback)(_In_ const size_t ulBatchSize, _In_ OrtValueCollection* data, _In_ OrtShape* input_shape, _In_ OrtShape* output_shape); 


struct OrtTrainingApi;
typedef struct OrtTrainingApi OrtTrainingApi;

struct OrtTrainingApiBase {
  const OrtTrainingApi*(ORT_API_CALL* GetApi)(uint32_t version)NO_EXCEPTION;  // Pass in ORT_API_VERSION
  // nullptr will be returned if the version is unsupported, for example when using a runtime older than this header file

  const char*(ORT_API_CALL* GetVersionString)() NO_EXCEPTION;
};
typedef struct OrtTrainingApiBase OrtTrainingApiBase;

ORT_EXPORT const OrtTrainingApiBase* ORT_API_CALL OrtTrainingGetApiBase(void) NO_EXCEPTION;


struct OrtTrainingApi {
  /**
    * \return A pointer of the newly created object. The pointer should be freed by OrtReleaseTrainingParameters after use
    */
  ORT_API2_STATUS(CreateTrainingParameters, _Outptr_ OrtTrainingParameters** options);

  // create a copy of an existing OrtTrainingParameters
  ORT_API2_STATUS(CloneTrainingParameters, _In_ const OrtTrainingParameters* in_options,
                  _Outptr_ OrtTrainingParameters** out_options);

  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the string parameter to set.
     * \param value specifies the string value to set.
     */
  ORT_API2_STATUS(SetTrainingParameter_string, _Inout_ OrtTrainingParameters* pParam, _In_ const OrtTrainingStringParameter key, _In_ const ORTCHAR_T* value);
  /**
     * \param pParam is set to null terminated string allocated using 'allocator'.  The caller is responsible for freeing it.
     * \param key specifies the string parameter to set.
     * \param ppvalue specifies the returned string value.
     */
  ORT_API2_STATUS(GetTrainingParameter_string, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingStringParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue);
  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the boolean parameter to set.
     * \param value specifies the boolean value to set.
     */
  ORT_API2_STATUS(SetTrainingParameter_bool, _Inout_ OrtTrainingParameters* pParam, _In_ const OrtTrainingBooleanParameter key, _In_ const bool value);
  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the boolean parameter to set.
     * \param value specifies the boolean value to get.
     */
  ORT_API2_STATUS(GetTrainingParameter_bool, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingBooleanParameter key, _Out_ bool* pvalue);
  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the numeric parameter to set.
     * \param value specifies the numeric value to set.
     */
  ORT_API2_STATUS(SetTrainingParameter_long, _Inout_ OrtTrainingParameters* pParam, _In_ const OrtTrainingLongParameter key, _In_ const long value);
  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the numeric parameter to set.
     * \param value specifies the numeric value to get.
     */
  ORT_API2_STATUS(GetTrainingParameter_long, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingLongParameter key, _Out_ long* pvalue);
  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the numeric parameter to set.
     * \param value specifies the numeric value to get.
     */
  ORT_API2_STATUS(SetTrainingParameter_double, _Inout_ OrtTrainingParameters* pParam, _In_ const OrtTrainingNumericParameter key, _In_ const double value);
  /**
     * \param pParam points to the training parameters to configure.
     * \param key specifies the numeric parameter to set.
     * \param value specifies the numeric value to get.
     */
  ORT_API2_STATUS(GetTrainingParameter_double, _In_ OrtTrainingParameters* pParam, _In_ const OrtTrainingNumericParameter key, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppvalue);
  /**
     * \param pParam points to the training parameters to configure.
     * \param opt specifies the optimizer to use.
     */
  ORT_API2_STATUS(SetTrainingOptimizer, _Inout_ OrtTrainingParameters* pParam, _In_ const OrtTrainingOptimizer opt);
  /**
     * \param pParam points to the training parameters to configure.
     * \param opt get the optimizer used.
     */
  ORT_API2_STATUS(GetTrainingOptimizer, _In_ OrtTrainingParameters* pParam, _Out_ OrtTrainingOptimizer* opt);
  /**
     * \param pParam points to the training parameters to configure.
     * \param loss specifies the loss function to use.
     */
  ORT_API2_STATUS(SetTrainingLossFunction, _Inout_ OrtTrainingParameters* pParam, _In_ const OrtTrainingLossFunction loss);
  /**
     * \param pParam points to the training parameters to configure.
     * \param loss get the loss function used.
     */
  ORT_API2_STATUS(GetTrainingLossFunction, _In_ OrtTrainingParameters* pParam, _Out_ OrtTrainingLossFunction* loss);

  /**
     * \param pParam points to the training parameters to setup (configures all SetTrainingParameters have been set).
     * \param errorFn callback called when the error function is called.
     * \param evalFn callback called when the evaluation function is called.
     */
  ORT_API2_STATUS(SetupTrainingParameters, _In_ OrtTrainingParameters* pParam, _In_ OrtErrorFunctionCallback errorFn, _In_ OrtEvaluationFunctionCallback evalFn);
  /**
     * \param pParam points to the training parameters to setup (configures all SetTrainingParameters have been set).
     * \param trainingdataqueryFn callback called to get each training data batch.
     * \param testingdataqueryFn callback called to get each testing data batch.
     * \param szFeedNames null terminated string containing semicolon separated feed names (e.g. "X;labels"); 
     */
  ORT_API2_STATUS(SetupTrainingData, _In_ OrtTrainingParameters* pParam, _In_ OrtDataGetBatchCallback trainingdataqueryFn, _In_ OrtDataGetBatchCallback testingdataqueryFn, _In_ const ORTCHAR_T* szFeedNames);

  /**
     * \param pEnv points to the environment.
     * \param pParam points to the training parameters to setup (configures all SetTrainingParameters have been set).
     */
  ORT_API2_STATUS(InitializeTraining, _In_ OrtEnv* pEnv, _In_ OrtTrainingParameters* pParam);
  /**
     * \param pParam points to the training parameters to setup (configures all SetTrainingParameters have been set).
     */
  ORT_API2_STATUS(RunTraining, _In_ OrtTrainingParameters* pParam);
  /**
     * \param pParam points to the training parameters to setup (configures all SetTrainingParameters have been set).
     */
  ORT_API2_STATUS(EndTraining, _In_ OrtTrainingParameters* pParam);


  /**
     * \param pCol points to the OrtValue collection.
     * \param pnCount the number of items in the collection are returned in pnCount.
     */
  ORT_API2_STATUS(GetCount, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount);
  /**
     * \param pCol points to the OrtValue collection.
     * \param pnCount the number of items the collection is capable of holding.
     */
  ORT_API2_STATUS(GetCapacity, _In_ OrtValueCollection* pCol, _Out_ size_t* pnCount);
  /**
     * \param pCol points to the OrtValue collection.
     * \param nIdx specifies the index of the OrtValue to retrieve.
     * \param output specifies the OrtValue is returned in output.
     * \param allocator specifies the allocator used to allocate the string name.
     * \param ppszName specifies where the name of the OrtValue is returned.
     */
  ORT_API2_STATUS(GetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _Outptr_ OrtValue** output, _Inout_ OrtAllocator* allocator, _Outptr_ char** ppszName);
  /**
     * \param pCol points to the OrtValue collection.
     * \param nIdx specifies the index of the OrtValue to set (up to Capacity).
     * \param input specifies the OrtValue that is set in the collection.
     * \param szName specifies the name of the OrtValue (if any).
     */
  ORT_API2_STATUS(SetAt, _In_ OrtValueCollection* pCol, _In_ size_t nIdx, _Outptr_ OrtValue* input, _In_ const ORTCHAR_T* szName);


  /**
     * \param pCol points to the OrtShape.
     * \param pnCount the number of items in the collection are returned in pnCount.
     */
  ORT_API2_STATUS(GetDimCount, _In_ OrtShape* pShape, _Out_ size_t* pnCount);
  /**
     * \param pCol points to the OrtShape.
     * \param nIdx specifies the index of the OrtValue to retrieve.
     * \param output specifies the OrtValue is returned in output.
     */
  ORT_API2_STATUS(GetDimAt, _In_ OrtShape* pShape, _In_ size_t nIdx, _Out_ size_t* output);


  ORT_CLASS_RELEASE(TrainingParameters);
};

#ifdef __cplusplus
}
#endif

#pragma once
#include "common.h"

//Need to file bugs for failing tests and add to reason. Before that happens, default reasons will be used.
static const std::string disabledTestDefaultReason = "Model not working on CPU and GPU. Please file bug and replace this reason message.";
static const std::string disabledGpuTestDefaultReason = "Model not working on GPU. Please file bug and replace this reason message.";

// {"model test name", "reason for why it is happening and bug filed for it."}
std::unordered_map<std::string, std::string> disabledTests(
    {
     // Tier 3 models
     {"mxnet_arcface_opset8", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"XGBoost_XGClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_RandomForestClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_RandomForestClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_Nu_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_Normalizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_LabelEncoder_OpenML_3_chess_opset7", disabledTestDefaultReason},
     {"scikit_LabelEncoder_BikeSharing_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_sklearn_load_Iris_missing_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_sklearn_load_digits_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_sklearn_load_diabetes_missing_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"scikit_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"scikit_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"scikit_Binarization_DecisionTreeClassifier_OpenML_1492_plants_opset7", disabledTestDefaultReason},
     {"scikit_Binarization_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"libsvm_Nu_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_VGG16_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_SVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_SVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_SVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_SVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_SqueezeNet_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_Scaler_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_Scaler_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_Resnet50_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_RandomForestClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_Normalizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_Normalizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_LogisticRegression_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_LinearSVC_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_Inceptionv3_ImageNet_opset7", disabledTestDefaultReason},
     {"coreml_Imputer_LogisticRegression_OpenML_1464_blood_transfusion_missing_opset7", disabledTestDefaultReason},
     {"coreml_Imputer_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_Imputer_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_Criteo_opset7", disabledTestDefaultReason},
     {"coreml_GradientBoostingClassifier_BingClick_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_sklearn_load_Iris_missing_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_sklearn_load_digits_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_sklearn_load_diabetes_missing_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_OpenML_31_credit_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_RandomForestRegressor_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_RandomForestClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_LinearSVC_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_LinearRegression_sklearn_load_diabetes_opset7", disabledTestDefaultReason},
     {"coreml_DictVectorizer_GradientBoostingRegressor_sklearn_load_boston_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_sklearn_load_wine_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_sklearn_load_breast_cancer_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_OpenML_312_scene_opset7", disabledTestDefaultReason},
     {"coreml_DecisionTreeClassifier_OpenML_1464_blood_transfusion_opset7", disabledTestDefaultReason},
     {"coreml_AgeNet_ImageNet_opset7", disabledTestDefaultReason}
    });

std::unordered_map<std::string, std::string> disabledGpuTests(
    {
     // Onnx zoo models
     {"mask_rcnn_opset10", "Bug 31005388: mask_rcnn opset 10 onnx zoo model fails to evaluate on DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/31005388"},
     {"faster_rcnn_opset10", "Bug 31005511: Failed to extract tensor data from evaluate result of faster_rcnn opset 10 model in DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/31005511"},

     // Tier 2 models
     {"fp16_test_tiny_yolov2_opset7", "Bug 31005780: Result of fp16_test_tiny_yolov2_opset7 and fp16_coreml_FNS_Candy_opset7 models on DirectML aren't as accurate as on CPU https://microsoft.visualstudio.com/OS/_workitems/edit/31005780"},
     {"fp16_tiny_yolov2_opset8", "Bug 31005780: Result of fp16_test_tiny_yolov2_opset7 and fp16_coreml_FNS_Candy_opset7 models on DirectML aren't as accurate as on CPU https://microsoft.visualstudio.com/OS/_workitems/edit/31005780"},
     {"fp16_coreml_FNS_Candy_opset7", "Bug 31005780: Result of fp16_test_tiny_yolov2_opset7 and fp16_coreml_FNS_Candy_opset7 models on DirectML aren't as accurate as on CPU https://microsoft.visualstudio.com/OS/_workitems/edit/31005780"},
     {"mlperf_ssd_mobilenet_300_opset10", "Bug 31005624: mlperf_ssd_mobilenet_300 opset 10 model fails to evaluate in DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/31005624"},
     {"fp16_inception_v1_opset7","Bug 32966366: fp16_inception_v1_opset7 and fp16_inception_v1_opset8 failing because of accuracy in DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/32966366"},
     {"fp16_inception_v1_opset8","Bug 32966366: fp16_inception_v1_opset7 and fp16_inception_v1_opset8 failing because of accuracy in DirectML https://microsoft.visualstudio.com/OS/_workitems/edit/32966366"}
    });

/*
    model name -> (adapter name regex, skipped test reason)
*/
std::unordered_map<std::string, std::pair<std::string, std::string>> disabledGpuAdapterTests(
    {
      {"fp16_inception_v1_opset7", std::make_pair("NVIDIA", "Bug 31144419: Results of fp16_inception_v1 opset7 and opset8 aren't accurate enough on AMD Radeon VII & Intel(R) UHD Graphics 630 & NVIDIA https://microsoft.visualstudio.com/OS/_workitems/edit/31144419")},
      {"fp16_inception_v1_opset8", std::make_pair("NVIDIA", "Bug 31144419: Results of fp16_inception_v1 opset7 and opset8 aren't accurate enough on AMD Radeon VII & Intel(R) UHD Graphics 630 & NVIDIA https://microsoft.visualstudio.com/OS/_workitems/edit/31144419")},
      {"candy_opset9", std::make_pair("Intel\\(R\\) (UHD )?Graphics", "Bug 31652854: Results of candy_opset9 aren't accurate enough on Intel Graphics https://microsoft.visualstudio.com/OS/_workitems/edit/31652854")},
    });

/*
    test name -> sampleTolerance
*/
std::unordered_map<std::string, double> gpuSampleTolerancePerTests(
    {{"fp16_inception_v1", 0.005}});

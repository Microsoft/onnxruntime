// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/*
 * This file defines SessionOptions Config Keys and format of the Config Values.
 *
 * The Naming Convention for a SessionOptions Config Key,
 * "[Area][.[SubArea1].[SubArea2]...].[Keyname]"
 * Such as "ep.cuda.use_arena"
 * The Config Key cannot be empty
 * The maximum length of the Config Key is 128
 *
 * The string format of a SessionOptions Config Value is defined individually for each Config.
 * The maximum length of the Config Value is 1024
 */

// Key for disable PrePacking,
// If the config value is set to "1" then the prepacking is disabled, otherwise prepacking is enabled (default value)
static const char* const kOrtSessionOptionsConfigDisablePrepacking = "session.disable_prepacking";

// A value of "1" means allocators registered in the env will be used. "0" means the allocators created in the session
// will be used. Use this to override the usage of env allocators on a per session level.
static const char* const kOrtSessionOptionsConfigUseEnvAllocators = "session.use_env_allocators";

// Set to 'ORT' (case sensitive) to load an ORT format model.
// If unset, model type will default to ONNX unless inferred from filename ('.ort' == ORT format) or bytes to be ORT
static const char* const kOrtSessionOptionsConfigLoadModelFormat = "session.load_model_format";

// Set to 'ORT' (case sensitive) to save optimized model in ORT format when SessionOptions.optimized_model_path is set.
// If unset, format will default to ONNX unless optimized_model_filepath ends in '.ort'.
static const char* const kOrtSessionOptionsConfigSaveModelFormat = "session.save_model_format";

// If a value is "1", flush-to-zero and denormal-as-zero are applied. The default is "0".
// When multiple sessions are created, a main thread doesn't override changes from succeeding session options,
// but threads in session thread pools follow option changes.
// When ORT runs with OpenMP, the same rule is applied, i.e. the first session option to flush-to-zero and
// denormal-as-zero is only applied to global OpenMP thread pool, which doesn't support per-session thread pool.
// Note that an alternative way not using this option at runtime is to train and export a model without denormals
// and that's recommended because turning this option on may hurt model accuracy.
static const char* const kOrtSessionOptionsConfigSetDenormalAsZero = "session.set_denormal_as_zero";

// It controls to run quantization model in QDQ (QuantizelinearDeQuantizelinear) format or not.
// "0": enable. ORT does fusion logic for QDQ format.
// "1": disable. ORT doesn't do fusion logic for QDQ format.
// Its default value is "0"
static const char* const kOrtSessionOptionsDisableQuantQDQ = "session.disable_quant_qdq";

// Enable or disable gelu approximation in graph optimization. "0": disable; "1": enable. The default is "0".
// GeluApproximation has side effects which may change the inference results. It is disabled by default due to this.
static const char* const kOrtSessionOptionsEnableGeluApproximation = "optimization.enable_gelu_approximation";

// Enable or disable using device allocator for allocating initialized tensor memory. "1": enable; "0": disable. The default is "0".
// Using device allocators means the memory allocation is made using malloc/new.
static const char* const kOrtSessionOptionsUseDeviceAllocatorForInitializers = "session.use_device_allocator_for_initializers";

// Configure whether to allow the inter_op/intra_op threads spinning a number of times before blocking
// "0": thread will block if found no job to run
// "1": default, thread will spin a number of times before blocking
static const char* const kOrtSessionOptionsConfigAllowInterOpSpinning = "session.inter_op.allow_spinning";
static const char* const kOrtSessionOptionsConfigAllowIntraOpSpinning = "session.intra_op.allow_spinning";

// Configure whether use threadpoollite for intra-op
// "0": default, not using tplite for intro-op
// "1": use tplite for intra-op
static const char* const kOrtSessionOptionsConfigUseTpLite = "session.intra_op.use_tplite";

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/common/status.h"
#include "core/framework/execution_providers.h"
#include "core/framework/framework_common.h"
#include "core/framework/iexecutor.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/session_state.h"
#include "core/graph/basic_types.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/insert_cast_transformer.h"
#include "core/framework/session_options.h"
#include "core/framework/allocatormgr.h"
#if !defined(ORT_MINIMAL_BUILD)
#include "onnx/defs/operator_sets.h"
#include "onnx/defs/schema.h"
#endif
#ifdef ENABLE_LANGUAGE_INTEROP_OPS
#include "core/language_interop_ops/language_interop_ops.h"
#endif
#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
#include "core/platform/tracing.h"
#include <TraceLoggingActivity.h>
#endif

#ifdef ENABLE_TRAINING
#include "core/framework/partial_graph_execution_state.h"
#endif
namespace onnxruntime {  // forward declarations
class GraphTransformer;
class Environment;
}  // namespace onnxruntime

namespace ONNX_NAMESPACE {
class ModelProto;
}  // namespace ONNX_NAMESPACE

struct OrtCustomOpDomain {
  std::string domain_;
  std::vector<const OrtCustomOp*> custom_ops_;
};

namespace onnxruntime {
class IExecutionProvider;  // forward decl
class IOBinding;
class CustomRegistry;
struct Notification;

namespace logging {
class LoggingManager;
}

/**
  * Pre-defined and custom metadata about the model.
  */
struct ModelMetadata {
  ModelMetadata() = default;
  ModelMetadata(const ModelMetadata&) = default;
  ~ModelMetadata() = default;
  ModelMetadata& operator=(const ModelMetadata&) = delete;

  std::string producer_name;
  std::string graph_name;
  std::string domain;
  std::string description;
  std::string graph_description;
  int64_t version = 0;
  std::unordered_map<std::string, std::string> custom_metadata_map;
};

/**
 * @brief This is the main class used to Run a model.
 * Sample simple usage:
 *  CPUExecutionProviderInfo epi;
 *  ProviderOption po{"CPUExecutionProvider", epi};
 *  SessionOptions so(vector<ProviderOption>{po});
 *  string log_id = "Foo";
 *  auto logging_manager = std::make_unique<LoggingManager>
                (std::unique_ptr<ISink>{new CLogSink{}},
                                  static_cast<Severity>(lm_info.default_warning_level),
                                  false,
                                  LoggingManager::InstanceType::Default,
                                  &log_id)
 *  Environment::Create(std::move(logging_manager), env)
 *  InferenceSession session_object{so,env};
 *  common::Status status = session_object.Load(MODEL_URI);
 *  common::Status status = session_object.Initialize();
 *
 *  NameMLValMap feeds;
 *  feeds.insert({});
 *  ...
 *  std::vector<std::string> output_names;
 *  output_names.insert(...);
 *  ...
 *  std::vector<OrtValue> fetches;
 *  common::Status status = session_object.Run(run_options, feeds, output_names, &fetches);
 *  process the output here...
 */

class InferenceSession {
 public:
  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    */
  explicit InferenceSession(const SessionOptions& session_options,
                            const Environment& session_env);

#if !defined(ORT_MINIMAL_BUILD)

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param model_uri absolute path of the model file.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   const std::string& model_uri);
#ifdef _WIN32
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   const std::wstring& model_uri);
#endif

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param istream object of the model.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   std::istream& model_istream);

  /**
    Create a new InferenceSession
    @param session_options Session options.
    @param model_data Model data buffer.
    @param model_data_len Model data buffer size.
    @param session_env This represents the context for the session and contains the logger and the global threadpools.
    This ctor will throw on encountering model parsing issues.
    */
  InferenceSession(const SessionOptions& session_options,
                   const Environment& session_env,
                   const void* model_data,
                   int model_data_len);

#endif  // !defined(ORT_MINIMAL_BUILD)

  virtual ~InferenceSession();

  /**
    * Register an execution provider. If you've one to register, call this before invoking Initialize().
    * The order of invocation indicates the preference order as well. In other words call this method
    * on your most preferred execution provider first followed by the less preferred ones.
    * Calling this API is optional in which case onnxruntime will use its internal CPU execution provider.
    * @return OK if success.
    */
  common::Status RegisterExecutionProvider(std::unique_ptr<IExecutionProvider> p_exec_provider) ORT_MUST_USE_RESULT;

#if !defined(ORT_MINIMAL_BUILD)
  /**
    * Register a graph transformer. If you've one to register, call this before invoking Initialize().
    * Calling this API is optional.
    * @param[in] - providers Optional. If providers is non-empty this transformer will only to
      applied to nodes which are assigned to given providers.
    * @param[in] - level Optional. Level to which this transformer should be registered. Default is set to 2.
    * @return OK if success.
    */
  common::Status RegisterGraphTransformer(std::unique_ptr<onnxruntime::GraphTransformer> p_graph_transformer,
                                          TransformerLevel level = TransformerLevel::Level2) ORT_MUST_USE_RESULT;

  /**
    * Filter the enabled optimizers (either transformer or rewrite rule) using optimizers_to_disable.
    * For an optimizer to be enabled, it must be allowed at the current optimization level (as specified in
    * session options), and NOT in optimizers_to_disable. 
    * This allows finer grained control of the enabled/disabled optimizations.
    * Must be called before Initialize() to take effect.
    * 
    * Calling this API is optional.
    * @return OK if success.
    */
  common::Status FilterEnabledOptimizers(const std::unordered_set<std::string>& optimizers_to_disable)
      ORT_MUST_USE_RESULT;

  common::Status AddONNXOpDomain(int onnx_opset_version) ORT_MUST_USE_RESULT;

  static common::Status ORTRegisterONNXOpsetSchema(int onnx_opset_version) {
    if (ONNX_NAMESPACE::OpSchemaRegistry::Instance()->GetLoadedSchemaVersion() == -1) {
      if (onnx_opset_version == 0) {
        // By default if session_onnx_opset_version=0, it registers all ONNX opset schema for all opset versions
        ONNX_NAMESPACE::RegisterOnnxOperatorSetSchema();
      } else {
        // If giving session_onnx_opset_version, only load the latest ones for each opset before specified onnx_opset_version
        ONNX_NAMESPACE::RegisterOnnxOperatorSetSchema(onnx_opset_version);
      }
    }
    return Status::OK();
  }

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  /**
    * Add custom ops. This API is not thread safe.
    */
  common::Status AddCustomOpDomains(const std::vector<OrtCustomOpDomain*>& ops) ORT_MUST_USE_RESULT;

  /**
    * Register a custom registry for operator schema and kernels.  If you've one to register,
    * call this before invoking Initialize().
    * The order of invocation indicates the reversed preference order: Register your most
    * preferred registry at the end.
    * Calling this API is optional.
    * This API is not thread safe.
    * @return OK if success.
    */
  common::Status RegisterCustomRegistry(std::shared_ptr<CustomRegistry> custom_registry) ORT_MUST_USE_RESULT;
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

  /**
    * Load an ONNX or ORT format model.
    *
    * Set SessionOptions session config value ORT_SESSION_OPTIONS_CONFIG_LOAD_MODEL_FORMAT to 'ORT' or 'ONNX' to 
    * explicitly choose model format.
    *
    * If format is not explicitly specified and filename ends in '.ort' it will be inferred to be an ORT format model.
	* All other files are assumed to be in ONNX format.
    * 
    * @param model_uri absolute path of the model file.
    * @return OK if success.
    */
  common::Status Load(const std::string& model_uri) ORT_MUST_USE_RESULT;
#ifdef _WIN32
  common::Status Load(const std::wstring& model_uri) ORT_MUST_USE_RESULT;
#endif
  /**
    * Load an ONNX or ORT format model.
    *
    * Set SessionOptions session config value ORT_SESSION_OPTIONS_CONFIG_LOAD_MODEL_FORMAT to 'ORT' or 'ONNX' to 
    * explicitly choose model format.
    *
    * If format is not explicitly specified the model format will be inferred from the bytes, defaulting to ONNX.
    * 
    * @param model_data Model data buffer
    * @param model_data_len Model data buffer size
    * @return OK if success.
    */
  common::Status Load(const void* model_data, int model_data_len) ORT_MUST_USE_RESULT;

#if !defined(ORT_MINIMAL_BUILD)
  /**
    * Load an ONNX model.
    * @param istream object of the model.
    * @return OK if success.
    */
  common::Status Load(std::istream& model_istream) ORT_MUST_USE_RESULT;

  /**
    * Load an ONNX model from the member model_proto_.
    * To be called only in conjunction with a ctor that takes in a model path/ model stream/ model array
    * @return OK if success.
    */
  common::Status Load() ORT_MUST_USE_RESULT;
#endif  // !defined(ORT_MINIMAL_BUILD)

  /**
    * Initializes a previously loaded ONNX model. Initialization includes but is not
    * limited to graph transformations, construction of kernels, etc.
    * This method assumes that a method has been loaded previously.
    * This API is thread-safe.
    * @return OK if success
    */
  common::Status Initialize() ORT_MUST_USE_RESULT;

  common::Status Run(const RunOptions& run_options, const std::vector<std::string>& feed_names,
                     const std::vector<OrtValue>& feeds, const std::vector<std::string>& output_names,
                     std::vector<OrtValue>* p_fetches,
                     const std::vector<OrtDevice>* p_fetches_device_info = nullptr) ORT_MUST_USE_RESULT;

  /**
    * Run a pre-loaded and pre-intialized model.
    * Multiple threads are allowed to run this function; hence its thread-safe.
    * @param feeds named inputs owned by client code and should not be changed during
    *        execution of this function.
    * @param output_names output names
    * @param p_fetches output values in the order specified by output_names.
    *        This should not be changed during execution of this function.
    * @return OK if success.
    */
  common::Status Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names,
                     std::vector<OrtValue>* p_fetches) ORT_MUST_USE_RESULT;

  /**
   * See Run(const NameMLValMap& feeds, const std::vector<std::string>& output_names, std::vector<OrtValue>* p_fetches)
   * for details.
   * @param run_options use this to tune the Run call to your needs.
   */
  common::Status Run(const RunOptions& run_options, const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<OrtValue>* p_fetches) ORT_MUST_USE_RESULT;

  /**
  * Creates a new binding object for binding inputs and outputs.
  * @param provider_type specifies the location where the inputs need to be potentially copied.
  * See IOBinding class for more info.
  */
  common::Status NewIOBinding(std::unique_ptr<IOBinding>* io_binding) ORT_MUST_USE_RESULT;

  virtual common::Status Run(const RunOptions& run_options, IOBinding& io_binding) ORT_MUST_USE_RESULT;
  common::Status Run(IOBinding& io_binding) ORT_MUST_USE_RESULT;

#ifdef ENABLE_TRAINING
  /**
  * Partially run a pre-loaded and pre-intialized model.
    * @param run_options run options. 
    * @param feeds inputs owned by client code and should not be changed during
    *        execution of this function.
    * @param fetches outputs produced after the executin of this function.
    * @param state State of the graph needed to resume partial graph run.
    * @param feeds_fetches_manager Contains feed/fetches name to internal indices mapping and information for device
    *                              copy/checks.
  */
  common::Status PartialRun(onnxruntime::RunOptions& run_options,
                            const std::vector<OrtValue>& feeds,
                            std::vector<OrtValue>& fetches,
                            PartialGraphExecutionState& state,
                            FeedsFetchesManager& feeds_fetches_manager);
#endif

  /**
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<common::Status, const ModelMetadata*> GetModelMetadata() const;

  /**
    * Get all input definitions of the model. This does not include weights. Use this
    * to get the name/type/shapes of the inputs.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<common::Status, const InputDefList*> GetModelInputs() const;

  /**
    * Get all definitions of the model for overridable initializers.
    * This does not include weights. Use this to get the name/type/shapes of the overridable initializers.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    * @note for IR < 4 returned list will always be empty.
    */
  std::pair<common::Status, const InputDefList*> GetOverridableInitializers() const;

  /**
    * Get all output definitions of the model. Use this to get the name/type/shapes of the outputs.
    * @return pair.first = OK; FAIL otherwise. pair.second is non-NULL when pair.first = OK.
    * @note lifetime of the returned pointer is valid as long as the Session object is live.
    */
  std::pair<common::Status, const OutputDefList*> GetModelOutputs() const;

  /**
    * Get the current number of in-progress concurrent Run calls.
    */
  int GetCurrentNumRuns() const;

  /**
    * Get the names of registered Execution Providers. The returned vector is ordered by Execution Provider
    * priority. The first provider in the vector has the highest priority.
    */
  const std::vector<std::string>& GetRegisteredProviderTypes() const;

  /*
   * Get the options this session was initialized with.
   */
  const SessionOptions& GetSessionOptions() const;

  /*
   * Get the DataTransferManager associated with this session
   */
  const DataTransferManager& GetDataTransferManager() const;

  /*
   * Get all the providers' options this session was initialized with.
   */
  const ProviderOptionsMap& GetAllProviderOptions() const;

  /**
    * Start profiling on this inference session. This simply turns on profiling events to be
    * recorded. A corresponding EndProfiling has to follow to write profiling data to a file.
    *@param file_prefix is the prefix of the profile file. It can include a directory path.
    */
  void StartProfiling(const std::string& file_prefix);
#ifdef _WIN32
  void StartProfiling(const std::wstring& file_prefix);
#endif
  /**
    * Start profiling on this inference session. This simply turns on profiling events to be
    * recorded. A corresponding EndProfiling has to follow to send profiling events through the logger's ISink.
    *@param logger_ptr is pointer to the logger where profiling events will be sent to.
    */
  void StartProfiling(const logging::Logger* logger_ptr);

  /**
    * Write captured profile events in chromium format.
    @return the name of the profile file.
    */
  std::string EndProfiling();
  /**
    * Return the profiler to access its attributes
    @return the profiler object
    */
  const profiling::Profiler& GetProfiling() const;

  /**
    * Search registered execution providers for an allocator that has characteristics
    * specified within mem_info
    * @param mem_info is a reference to OrtMemoryInfo that contains required specs
    * @return a ptr to the allocator or nullptr if not available
    */
  AllocatorPtr GetAllocator(const OrtMemoryInfo& mem_info) const;

  std::shared_ptr<onnxruntime::AllocatorManager> GetAllocatorManager() {
    return allocator_manager_;
  }

  /**
    *Get InferenceSession logger.
    */
  const logging::Logger* GetLogger() const { return session_logger_; };

  const SessionState& GetSessionState() const {
    ORT_ENFORCE(session_state_ != nullptr, "Session must be initialized to create session state.");
    return *session_state_;
  }

  /**
    * Add a PrepackedWeightsContainer instance to the session so as to store the pre-packed weights 
    *  of shared initializers to be shared across sessions.
    * @param prepacked_weights_container PrepackedWeightsContainer instance 
    */
  Status AddPrePackedWeightsContainer(PrepackedWeightsContainer* prepacked_weights_container);

 protected:
#if !defined(ORT_MINIMAL_BUILD)
  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. model_proto will be copied by the API.
    * @return OK if success.
    */
  common::Status Load(const ONNX_NAMESPACE::ModelProto& model_proto) ORT_MUST_USE_RESULT;

  /**
    * Load an ONNX model.
    * @param protobuf object corresponding to the model file. This is primarily to support large models.
    * @return OK if success.
    */
  common::Status Load(std::unique_ptr<ONNX_NAMESPACE::ModelProto> p_model_proto) ORT_MUST_USE_RESULT;

  common::Status DoPostLoadProcessing(onnxruntime::Model& model) ORT_MUST_USE_RESULT;

#endif  // !defined(ORT_MINIMAL_BUILD)

  bool IsInitialized() const;

  // Use these 2 threadpool methods to get access to the threadpools since they rely on
  // specific flags in session options
  // These methods assume that session options have been finalized before the call.
  onnxruntime::concurrency::ThreadPool* GetIntraOpThreadPoolToUse() const {
    return session_options_.use_per_session_threads ? thread_pool_.get() : intra_op_thread_pool_from_env_;
  }

  onnxruntime::concurrency::ThreadPool* GetInterOpThreadPoolToUse() const {
    return session_options_.use_per_session_threads ? inter_op_thread_pool_.get() : inter_op_thread_pool_from_env_;
  }

  /// convenience pointer to logger. should always be the same as session_state_.Logger();
  const logging::Logger* session_logger_;

  // The model served by this inference session instance.
  // Currently this has to be a shared ptr because the Model::Load method
  // returns a shared_ptr only. Ideally factory functions should always return
  // unique_ptr for maximum flexibility. Client can always upgrade it to shared_ptr
  // if they need.
  std::shared_ptr<onnxruntime::Model> model_;

  // names of model outputs used for quick validation.
  std::unordered_set<std::string> model_output_names_;

  // The file path of where the model was loaded. e.g. /tmp/test_squeezenet/model.onnx
  std::basic_string<ORTCHAR_T> model_location_;

  // The list of execution providers.
  ExecutionProviders execution_providers_;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(InferenceSession);

  void ConstructorCommon(const SessionOptions& session_options,
                         const Environment& session_env);

  common::Status SaveModelMetadata(const onnxruntime::Model& model) ORT_MUST_USE_RESULT;

#if !defined(ORT_MINIMAL_BUILD)
  common::Status Load(std::function<common::Status(std::shared_ptr<Model>&)> loader,
                      const std::string& event_name) ORT_MUST_USE_RESULT;

  template <typename T>
  common::Status Load(const std::basic_string<T>& model_uri) ORT_MUST_USE_RESULT;

  bool HasLocalSchema() const {
    return !custom_schema_registries_.empty();
  }

  common::Status SaveToOrtFormat(const std::basic_string<ORTCHAR_T>& filepath) const;
#endif

#if defined(ENABLE_ORT_FORMAT_LOAD)
  /**
    * Load an ORT format model.
    * @param model_uri absolute path of the model file.
    * @return OK if success.
    */
  common::Status LoadOrtModel(const std::string& model_uri) ORT_MUST_USE_RESULT;
#ifdef _WIN32
  common::Status LoadOrtModel(const std::wstring& model_uri) ORT_MUST_USE_RESULT;
#endif

  /**
    * Load an ORT format model.
    * @param model_data Model data buffer
    * @param model_data_len Model data buffer size
    * @return OK if success.
    * @remarks TODO: Provide way to load from in-memory bytes without copying. InferenceSession would need to
    *                take ownership of the buffer passed in.
    */
  common::Status LoadOrtModel(const void* model_data, int model_data_len) ORT_MUST_USE_RESULT;

  common::Status LoadOrtModel(std::function<Status()> load_ort_format_model_bytes) ORT_MUST_USE_RESULT;

#endif  // defined(ENABLE_ORT_FORMAT_LOAD)

  // Create a Logger for a single execution if possible. Otherwise use the default logger.
  // If a new logger is created, it will also be stored in new_run_logger,
  // which must remain valid for the duration of the execution.
  // If the default logger is used, new_run_logger will remain empty.
  // The returned value should be used in the execution.
  const logging::Logger& CreateLoggerForRun(const RunOptions& run_options,
                                            std::unique_ptr<logging::Logger>& new_run_logger);

  void InitLogger(logging::LoggingManager* logging_manager);

  common::Status CheckShapes(const std::string& input_name, const TensorShape& input_shape,
                             const TensorShape& expected_shape) const ORT_MUST_USE_RESULT;

  common::Status ValidateInputs(const std::vector<std::string>& feed_names,
                                const std::vector<OrtValue>& feeds) const ORT_MUST_USE_RESULT;

  common::Status ValidateOutputs(const std::vector<std::string>& output_names,
                                 const std::vector<OrtValue>* p_fetches) const ORT_MUST_USE_RESULT;

  common::Status WaitForNotification(Notification* p_executor_done, int64_t timeout_in_ms) ORT_MUST_USE_RESULT;

  template <typename T>
  void StartProfiling(const std::basic_string<T>& file_prefix);

  // Updates all providers with the allocators from the env based on OrtMemoryInfo
  void UpdateProvidersWithSharedAllocators();

  /*
   * Validate and parses the shrink arena request string from the user
   * List format: "device_0:device_id_0;device_1:device_id_1"
   * If we encounter an invalid request, we return an error
   * back to the user.
   */

  common::Status ValidateAndParseShrinkArenaString(const std::string& ort_device_list,
                                                   /*out*/ std::vector<AllocatorPtr>& arenas_to_shrink) const ORT_MUST_USE_RESULT;

  /*
   * Performs the shrinkage of arenas requested to be shrunk by the user
   * The `arenas_to_shrink` parameter is got from ValidateAndParseShrinkArenaString()
   */
  void ShrinkMemoryArenas(const std::vector<AllocatorPtr>& arenas_to_shrink);

#if !defined(ORT_MINIMAL_BUILD)
  virtual void AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                         TransformerLevel graph_optimization_level);

  common::Status TransformGraph(onnxruntime::Graph& graph,
                                const onnxruntime::GraphTransformerManager& graph_transformer_mgr,
                                const ExecutionProviders& providers, KernelRegistryManager& kernel_registry_manager,
                                const InsertCastTransformer& insert_cast_transformer,
                                SessionState& session_state,
                                bool saving_model_in_ort_format) ORT_MUST_USE_RESULT;

  onnxruntime::GraphTransformerManager graph_transformation_mgr_;

  InsertCastTransformer insert_cast_transformer_;

  // Any GraphTransformer/RewriteRule name in this set will not be enabled.
  std::unordered_set<std::string> optimizers_to_disable_;
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  Status PartitionOrtFormatModel(onnxruntime::Graph& graph, const ExecutionProviders& providers,
                                 KernelRegistryManager& kernel_registry_manager, SessionState& session_state) const;
#endif

  SessionOptions session_options_;

  /// Logging manager if provided.
  logging::LoggingManager* const logging_manager_;

  /// Logger for this session. WARNING: Will contain nullptr if logging_manager_ is nullptr.
  std::unique_ptr<logging::Logger> owned_session_logger_ = nullptr;

  // Profiler for this session.
  profiling::Profiler session_profiler_;

  // Immutable state for each op in the model. Shared by all executors.
  // It has a dependency on execution_providers_.
  std::unique_ptr<SessionState> session_state_;

  // Threadpools per session. These are initialized and used for the entire duration of the session
  // when use_per_session_threads is true.
  std::basic_string<ORTCHAR_T> thread_pool_name_;
  std::basic_string<ORTCHAR_T> inter_thread_pool_name_;

  std::unique_ptr<onnxruntime::concurrency::ThreadPool> thread_pool_;
  std::unique_ptr<onnxruntime::concurrency::ThreadPool> inter_op_thread_pool_;

  // Global threadpools. These are intialized and used when use_per_session_threads is false *and*
  // the environment is created with create_global_thread_pools = true.
  onnxruntime::concurrency::ThreadPool* intra_op_thread_pool_from_env_{};
  onnxruntime::concurrency::ThreadPool* inter_op_thread_pool_from_env_{};

  // initialized from session options
  // Determines which threadpools will be intialized and used for the duration of this session.
  // If true, use the per session ones, or else the global threadpools.
  bool use_per_session_threads_;

  KernelRegistryManager kernel_registry_manager_;

#if !defined(ORT_MINIMAL_BUILD)
  std::list<std::shared_ptr<onnxruntime::IOnnxRuntimeOpSchemaCollection>> custom_schema_registries_;
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)

  //CustomRegistry objects own the corresponding KernelRegistry and OnnxRuntimeOpSchemaRegistry objects.
  //So its lifetime should be same as its constituents. This vector is to extend the lifetime of the owner.
  std::vector<std::shared_ptr<CustomRegistry>> custom_registries_;
#endif

  ModelMetadata model_metadata_;
  std::unordered_set<std::string> required_inputs_;

  struct InputDefMetaData {
    InputDefMetaData(const NodeArg* node_arg0, MLDataType ml_data_type0, TensorShape&& tensor_shape0)
        : node_arg(node_arg0), ml_data_type(ml_data_type0), tensor_shape(std::move(tensor_shape0)) {
    }
    const NodeArg* node_arg;
    MLDataType ml_data_type;
    TensorShape tensor_shape;  // not applicable if the input is non-tensor type
  };

  std::unordered_map<std::string, InputDefMetaData> input_def_map_;
  OutputDefList output_def_list_;

  // Data transfer manager.
  DataTransferManager data_transfer_mgr_;

  // Number of concurrently running executors
  std::atomic<int> current_num_runs_;

  mutable onnxruntime::OrtMutex session_mutex_;  // to ensure only one thread can invoke Load/Initialize
  bool is_model_loaded_ = false;                 // GUARDED_BY(session_mutex_)
  bool is_inited_ = false;                       // GUARDED_BY(session_mutex_)

#ifdef ENABLE_LANGUAGE_INTEROP_OPS
  InterOpDomains interop_domains_;
#endif
  // used to support platform telemetry
  static std::atomic<uint32_t> global_session_id_;  // a monotonically increasing session id
  uint32_t session_id_;                             // the current session's id

  struct Telemetry {
    Telemetry() : time_sent_last_() {}
    uint32_t total_runs_since_last_ = 0;           // the total number of Run() calls since the last report
    long long total_run_duration_since_last_ = 0;  // the total duration (us) of Run() calls since the last report
    std::string event_name_;                       // where the model is loaded from: ["model_loading_uri", "model_loading_proto", "model_loading_istream"]

    TimePoint time_sent_last_;  // the TimePoint of the last report
    // Event Rate per provider < 20 peak events per second
    constexpr static long long kDurationBetweenSending = 1000 * 1000 * 60 * 10;  // duration in (us).  send a report every 10 mins
  } telemetry_;

#ifdef ONNXRUNTIME_ENABLE_INSTRUMENT
  bool session_activity_started_ = false;
  TraceLoggingActivity<telemetry_provider_handle> session_activity;
#endif

  // used to hold the ModelProto parsed in an applicable ctor to be used while calling parameter-less Load()
  ONNX_NAMESPACE::ModelProto model_proto_;

  // Flag indicating if ModelProto has been parsed in an applicable ctor
  bool is_model_proto_parsed_ = false;
  const Environment& environment_;

  // Bytes from an ORT format model.
  // We store them currently to make the Load + Initialize behave the same way as for an ONNX model
  // as we need some of the bytes for the Load (create the Model) and some for the Initialize (create SessionState).
  // Short term we free them after Initialize.
  // Longer term we may want to directly refer to offsets in this buffer for initializers so we don't need to copy
  // those into new OrtValue instances, at which point we won't free them until the InferenceSession goes away.
  std::vector<uint8_t> ort_format_model_bytes_;

  std::shared_ptr<onnxruntime::AllocatorManager> allocator_manager_;

  // Container to store pre-packed weights to share between sessions.
  // The life-cycle of the cache itself is maintained by the user and the user will ensure
  // the cache is valid until any session reliant on it is still in scope.
  PrepackedWeightsContainer* prepacked_weights_container_ = nullptr;
};

struct SessionIOBinding {
 public:
  SessionIOBinding(InferenceSession* session);

  const IOBinding* Get() const;
  IOBinding* Get();
  const InferenceSession* GetInferenceSession() const;
  InferenceSession* GetInferenceSession();

 private:
  InferenceSession* sess_;
  std::unique_ptr<IOBinding> binding_;
};

}  // namespace onnxruntime

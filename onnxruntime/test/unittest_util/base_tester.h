// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <variant>
#include <vector>
#include <type_traits>

#include "core/framework/customregistry.h"
#include "core/framework/float4.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/framework/TensorSeq.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"

#include "test/unittest_util/checkers.h"
#include "test/unittest_util/test_allocator_manager.h"
#include "test/unittest_util/tester_types.h"

namespace onnxruntime {
class InferenceSession;
struct SessionOptions;

namespace test {

/// <summary>
/// Base class for testing operators and models.
/// </summary>
class BaseTester {
 protected:
  explicit BaseTester(std::string_view test_name, int opset_version, std::string_view domain,
                      bool verify_output = true)
      : test_name_(test_name), domain_(domain), opset_version_(opset_version), verify_output_(verify_output) {
    if (opset_version_ < 0) {
      static int latest_onnx_version =
          ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(ONNX_NAMESPACE::ONNX_DOMAIN).second;
      opset_version_ = latest_onnx_version;
    }
    number_of_nodes_ = 0;
  }

  // Derived class to implement to provide the model to test.
  // Return is void so the GTEST ASSERT/EXPECT macros can be used in the implementation
  // a pointer to allow testing scenarios where the expected result is ExpectResult::kExpectFailure.
  virtual void CreateModelToTest(const ModelOptions& model_options, Model*& model) = 0;

 public:
  virtual ~BaseTester();

  // We have an initializer_list and vector version of the Add functions because std::vector is specialized for
  // bool and we can't get the raw data out. So those cases must use an initializer_list

  // Dims variant is needed to reduce the number of overloads
  // MS compiler refuses to create a gsl::span from initializer_list especially if it contains a single element
  using DimsVariant = std::variant<std::vector<int64_t>, TensorShapeVector>;

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values.begin(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, const std::vector<T>& values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values.data(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, const T* p_values,
                const size_t size, bool is_initializer = false,
                const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, p_values, size, is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, std::initializer_list<int64_t> dims, gsl::span<const T> values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values.data(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, const DimsVariant& dims, std::initializer_list<T> values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, values.begin(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, const DimsVariant& dims, const std::vector<T>& values,
                bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, values.data(), values.size(), is_initializer, false, dim_params);
  }

  template <typename T>
  void AddInput(const char* name, const DimsVariant& dims, const T* p_values,
                const size_t size, bool is_initializer = false,
                const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, p_values, size, is_initializer, false, dim_params);
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  // Useful to add boolean data
  template <typename T>
  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         const std::initializer_list<T>& values, const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(input_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(indices),
                           ValidateOutputParams(), dim_params);
  }

  template <typename T>
  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<T>& values, const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(input_data_, ml_type, name, dims,
                           gsl::as_bytes(gsl::make_span(values)),
                           gsl::make_span(indices),
                           ValidateOutputParams(), dim_params);
  }

  template <typename T>
  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         gsl::span<const T> values_span,
                         const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(input_data_, ml_type, name, dims,
                           values_span.as_bytes(),
                           gsl::make_span(indices),
                           ValidateOutputParams(), dim_params);
  }

  void AddSparseCooInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<std::string>& values,
                         const std::vector<int64_t>& indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    AddSparseCooTensorStrings(input_data_, name, dims,
                              gsl::make_span(values),
                              gsl::make_span(indices),
                              dim_params);
  }

  // Useful to add boolean data
  template <typename T>
  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         const std::initializer_list<T>& values,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           ValidateOutputParams(), dim_params);
  }

  template <typename T>
  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<T>& values,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                           gsl::as_bytes(gsl::make_span(values)),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           ValidateOutputParams(), dim_params);
  }

  template <typename T>
  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         gsl::span<const T> values_span,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(input_data_, ml_type, name, dims,
                           values_span.as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           ValidateOutputParams(), dim_params);
  }

  void AddSparseCsrInput(const char* name, const std::vector<int64_t>& dims,
                         const std::vector<std::string>& values,
                         const std::vector<int64_t>& inner_indices,
                         const std::vector<int64_t>& outer_indices,
                         const std::vector<std::string>* dim_params = nullptr) {
    AddSparseCsrTensorStrings(input_data_, name, dims,
                              gsl::make_span(values),
                              gsl::make_span(inner_indices),
                              gsl::make_span(outer_indices),
                              dim_params);
  }
#endif

  // Add other registered types, possibly experimental
  template <typename T>
  void AddInput(const char* name, const T& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(val);
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    input_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                               optional<float>()));
  }

  template <typename T>
  void AddInput(const char* name, T&& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(std::move(val));
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    input_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                               optional<float>()));
  }

  template <typename T>
  void AddSeqInput(const char* name, const SeqTensors<T>& seq_tensors) {
    AddSeqData<T>(input_data_, name, &seq_tensors);
  }

  template <typename T>
  void AddSeqOutput(const char* name, const SeqTensors<T>& seq_tensors,
                    float rel_error = 0.0f, float abs_error = 0.0f) {
    AddSeqData<T>(output_data_, name, &seq_tensors, false, rel_error, abs_error);
  }

#if !defined(DISABLE_OPTIONAL_TYPE)
  template <typename T>
  void AddOptionalTypeTensorInput(const char* name, const DimsVariant& dims,
                                  const std::initializer_list<T>* values = nullptr,
                                  bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    AddData(input_data_, name, dims, values ? values->begin() : nullptr,
            values ? values->size() : 0, is_initializer, false, dim_params, 0.0f, 0.0f, true);
  }

  template <typename T>
  void AddOptionalTypeTensorInput(const char* name, std::initializer_list<int64_t> dims,
                                  const std::initializer_list<T>* values = nullptr,
                                  bool is_initializer = false, const std::vector<std::string>* dim_params = nullptr) {
    DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(input_data_, name, dims_var, values ? values->begin() : nullptr,
            values ? values->size() : 0, is_initializer, false, dim_params, 0.0f, 0.0f, true);
  }

  template <typename T>
  void AddOptionalTypeTensorOutput(const char* name, const DimsVariant& dims,
                                   const std::initializer_list<T>* expected_values = nullptr,
                                   bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values ? expected_values->begin() : nullptr,
            expected_values ? expected_values->size() : 0, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error, true);
  }

  template <typename T>
  void AddOptionalTypeTensorOutput(const char* name, std::initializer_list<int64_t> dims,
                                   const std::initializer_list<T>* expected_values = nullptr,
                                   bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, expected_values ? expected_values->begin() : nullptr,
            expected_values ? expected_values->size() : 0, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error, true);
  }

  template <typename T>
  void AddOptionalTypeSeqInput(const char* name,
                               const SeqTensors<T>* seq_tensors) {
    AddSeqData<T>(input_data_, name, seq_tensors, true);
  }

  template <typename T>
  void AddOptionalTypeSeqOutput(const char* name,
                                const SeqTensors<T>* seq_tensors,
                                float rel_error = 0.0f, float abs_error = 0.0f) {
    AddSeqData<T>(output_data_, name, seq_tensors, true, rel_error, abs_error);
  }
#endif

  template <typename TKey, typename TVal>
  void AddInput(const char* name, const std::map<TKey, TVal>& val) {
    std::unique_ptr<std::map<TKey, TVal>> ptr = std::make_unique<std::map<TKey, TVal>>(val);
    OrtValue value;
    value.Init(ptr.release(), DataTypeImpl::GetType<std::map<TKey, TVal>>(),
               DataTypeImpl::GetType<std::map<TKey, TVal>>()->GetDeleteFunc());
    input_data_.push_back(Data(NodeArg(name, &MMapType<TKey, TVal>::s_map_type_proto.proto), std::move(value),
                               optional<float>(), optional<float>()));
  }

  /*
   * Use this API to add an input *edge* to the node/op being tested that won't
   * have any data passed into.
   * Such an edge will have the qualifier OpSchema::Optional in the schema.
   * This is exposed to ensure the op kernel implementations can be tested to handle
   * presence/absence of such optional input edges.
   */
  template <typename T>
  void AddOptionalInputEdge() {
    std::string name;  // empty == input doesn't exist
    input_data_.push_back(Data(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue(), optional<float>(),
                               optional<float>()));
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, std::initializer_list<T> expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, expected_values.begin(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, const std::vector<T>& expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, expected_values.data(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, std::initializer_list<int64_t> dims, const T* p_values, const size_t size,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    const DimsVariant dims_var = std::vector<int64_t>(dims);
    AddData(output_data_, name, dims_var, p_values, size, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, std::initializer_list<T> expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values.begin(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  // This function doesn't work for vector<bool> because const vector<bool> cannot invoke its data().
  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, const std::vector<T>& expected_values,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, expected_values.data(), expected_values.size(), false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

  template <typename T>
  void AddOutput(const char* name, const DimsVariant& dims, const T* p_values, const size_t size,
                 bool sort_output = false, float rel_error = 0.0f, float abs_error = 0.0f) {
    AddData(output_data_, name, dims, p_values, size, false,
            sort_output, nullptr /* dim_params */, rel_error, abs_error);
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  template <typename T>
  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::initializer_list<T>& expected_values,
                          const std::vector<int64_t>& expected_indices,
                          const ValidateOutputParams& check_params = {}) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(expected_values).as_bytes(),
                           gsl::make_span(expected_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<T>& expected_values,
                          const std::vector<int64_t>& expected_indices,
                          const ValidateOutputParams& check_params = {}) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(expected_values).as_bytes(),
                           gsl::make_span(expected_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          gsl::span<const T> expected_values_span,
                          const std::vector<int64_t>& expected_indices,
                          const ValidateOutputParams& check_params = {}) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCooTensorData(output_data_, ml_type, name, dims,
                           expected_values_span.as_bytes(),
                           gsl::make_span(expected_indices),
                           check_params, nullptr /*dim_params*/);
  }

  void AddSparseCooOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<std::string>& expected_values,
                          const std::vector<int64_t>& expected_indices) {
    AddSparseCooTensorStrings(output_data_, name, dims,
                              gsl::make_span(expected_values),
                              gsl::make_span(expected_indices));
  }

  template <typename T>
  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::initializer_list<T>& values,
                          const std::vector<int64_t>& inner_indices,
                          const std::vector<int64_t>& outer_indices,
                          const ValidateOutputParams& check_params = {}) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<T>& values,
                          const std::vector<int64_t>& inner_indices,
                          const std::vector<int64_t>& outer_indices,
                          const ValidateOutputParams& check_params = {}) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                           gsl::make_span(values).as_bytes(),
                           gsl::make_span(inner_indices),
                           gsl::make_span(outer_indices),
                           check_params, nullptr /*dim_params*/);
  }

  template <typename T>
  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          gsl::span<const T> expected_values_span,
                          const std::vector<int64_t>& expected_inner_indices,
                          const std::vector<int64_t>& expected_outer_indices,
                          const ValidateOutputParams& check_params = {}) {
    auto ml_type = DataTypeImpl::GetType<T>();
    AddSparseCsrTensorData(output_data_, ml_type, name, dims,
                           expected_values_span.as_bytes(),
                           gsl::make_span(expected_inner_indices),
                           gsl::make_span(expected_outer_indices),
                           check_params, nullptr /*dim_params*/);
  }

  void AddSparseCsrOutput(const char* name, const std::vector<int64_t>& dims,
                          const std::vector<std::string>& expected_values,
                          const std::vector<int64_t>& expected_inner_indices,
                          const std::vector<int64_t>& expected_outer_indices) {
    AddSparseCsrTensorStrings(output_data_, name, dims,
                              gsl::make_span(expected_values),
                              gsl::make_span(expected_inner_indices),
                              gsl::make_span(expected_outer_indices));
  }
#endif

  /*
   * Use this API to add an output *edge* to the node/op being tested that shouldn't have any
   * data produced into.
   * Such an edge will have the qualifier OpSchema::Optional in the schema.
   * This is exposed to ensure the op kernel implementations can be tested to handle
   * presence/absence of such optional output edges.
   */
  template <typename T>
  void AddOptionalOutputEdge() {
    std::string name;  // empty == output doesn't exist
    output_data_.push_back(Data(NodeArg(name, &TTensorType<T>::s_type_proto.proto), OrtValue(), optional<float>(),
                                optional<float>()));
  }

  // Add other registered types, possibly experimental
  template <typename T>
  void AddOutput(const char* name, const T& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(val);
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    output_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                                optional<float>()));
  }

  template <typename T>
  void AddOutput(const char* name, T&& val) {
    auto mltype = DataTypeImpl::GetType<T>();
    ORT_ENFORCE(mltype != nullptr, "T must be a registered cpp type");
    auto ptr = std::make_unique<T>(std::move(val));
    OrtValue value;
    value.Init(ptr.get(), mltype, mltype->GetDeleteFunc());
    ptr.release();
    output_data_.push_back(Data(NodeArg(name, mltype->GetTypeProto()), std::move(value), optional<float>(),
                                optional<float>()));
  }

  // Add non tensor output
  template <typename TKey, typename TVal>
  void AddOutput(const char* name, const std::vector<std::map<TKey, TVal>>& val) {
    auto ptr = std::make_unique<std::vector<std::map<TKey, TVal>>>(val);
    OrtValue ml_value;
    ml_value.Init(ptr.release(), DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>(),
                  DataTypeImpl::GetType<std::vector<std::map<TKey, TVal>>>()->GetDeleteFunc());
    output_data_.push_back(Data(NodeArg(name, &VectorOfMapType<TKey, TVal>::s_vec_map_type_proto.proto), std::move(ml_value),
                                optional<float>(), optional<float>()));
  }

  // Generate the reference outputs by running the provided the model
  void AddReferenceOutputs(const std::string& model_path, float abs_error = 0.0f,
                           std::unique_ptr<IExecutionProvider> ep = nullptr);

  void AddCustomOpRegistry(std::shared_ptr<CustomRegistry> registry) {
    custom_schema_registries_.push_back(registry->GetOpschemaRegistry());
    custom_session_registries_.push_back(registry);
  }

  // For floating types (double/float/half/bfloat16), tolerance is similar to numpy.isclose:
  //   absolute(expected_value - actual_value) <= abs_error + rel_error * absolute(expected_value)
  // For integer types, tolerance parameters are ignored except the following cases:
  //   For uint8, tolerance is only applied to NNAPI/XNNPACK/DML providers.
  //   For int8, only abs_error is used, and rel_error is ignored. See checkers.cc for detail.
  // If abs_error or rel_error is not set, a default value is used (search DefaultTolerance for detail).
  void SetOutputAbsErr(const char* name, float v);
  void SetOutputRelErr(const char* name, float v);

  // Set absolute and relative tolerance for all existed outputs.
  // Negative value will be ignored.
  // Note that it will not set tolerance for new outputs added after this call.
  void SetOutputTolerance(float abs_error, float rel_error = -1.0f);

  // Number of times to call InferenceSession::Run. The same feeds are used each time.
  // e.g. used to verify the generator ops behave as expected
  void SetNumRunCalls(int n) {
    ORT_ENFORCE(n > 0);
    num_run_calls_ = n;
  }

  using CustomOutputVerifierFn = std::function<void(const std::vector<OrtValue>& /*fetches*/,
                                                    const std::string& /*provider_type*/)>;

  void SetCustomOutputVerifier(CustomOutputVerifierFn custom_output_verifier) {
    custom_output_verifier_ = custom_output_verifier;
  }

  enum class ExpectResult {
    kExpectSuccess,
    kExpectFailure
  };

  BaseTester& Config(const SessionOptions& sess_options);
  BaseTester& Config(ExpectResult expect_result, const std::string& expected_failure_string);
  BaseTester& ConfigExcludeEps(const std::unordered_set<std::string>& excluded_provider_types);
  // Continue with other EPs if WebGPU reports that ShaderF16 is unavailable.
  // Report a skipped test only if no EP could execute it.
  BaseTester& ConfigSkipUnsupportedWebGpuFp16() {
    skip_unsupported_webgpu_fp16_ = true;
    return *this;
  }
  BaseTester& Config(const RunOptions* run_options);
  BaseTester& ConfigEps(std::vector<std::unique_ptr<IExecutionProvider>>&& execution_providers);
  // Configure a single EP to run.
  BaseTester& ConfigEp(std::unique_ptr<IExecutionProvider>&& execution_provider) {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.emplace_back(std::move(execution_provider));
    return ConfigEps(std::move(execution_providers));
  }
  BaseTester& Config(const Graph::ResolveOptions& resolve_options);

  void RunWithConfig(size_t* number_of_pre_packed_weights_counter = nullptr,
                     size_t* number_of_shared_pre_packed_weights_counter = nullptr);

  // [[deprecated("Use builder pattern Config* and RunWithConfig")]]
  void Run(ExpectResult expect_result = ExpectResult::kExpectSuccess, const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
           ExecutionMode execution_mode = ExecutionMode::ORT_SEQUENTIAL,
           const Graph::ResolveOptions& resolve_options = {});

  // [[deprecated("Use builder pattern Config* and RunWithConfig")]]
  // Take SessionOptions by value (i.e. make a copy) because we may need to modify it
  void Run(SessionOptions session_options,
           ExpectResult expect_result = ExpectResult::kExpectSuccess,
           const std::string& expected_failure_string = "",
           const std::unordered_set<std::string>& excluded_provider_types = {},
           const RunOptions* run_options = nullptr,
           std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr,
           const Graph::ResolveOptions& resolve_options = {},
           /*out*/ size_t* number_of_pre_packed_weights_counter = nullptr,
           /*out*/ size_t* number_of_shared_pre_packed_weights_counter = nullptr);

  std::vector<OrtValue> GetFetches() { return fetches_; }

  struct Data {
    Data(onnxruntime::NodeArg&& def, OrtValue&& data, optional<float>&& rel, optional<float>&& abs,
         bool sort_output = false)
        : def(std::move(def)),
          data(std::move(data)),
          validation_params{std::move(rel), std::move(abs), sort_output} {}
    Data(Data&&) = default;
    Data& operator=(Data&&) = default;

    onnxruntime::NodeArg def;
    OrtValue data;
    ValidateOutputParams validation_params;
  };

  void SetDeterminism(bool use_determinism) {
    use_determinism_ = use_determinism;
  }

  void EnableSharingOfPrePackedWeightsAcrossSessions() {
    add_prepacked_shared_container_to_sessions_ = true;
  }

  size_t GetNumPrePackedWeightsShared() const {
    return prepacked_weights_container_.GetNumberOfElements();
  }

  void SetAllowUnreleasedOnnxOpset() {
    test_allow_released_onnx_opset_only_ = false;
  }

  int GetNumberOfNodesAfterRun() const;

 protected:
  //// if the derived class is caching the model this helper can be called in CreateModelToTest to reset the nodes
  // static void ClearEpsForAllNodes(Graph& graph);

  const std::string& Domain() const { return domain_; }
  int Opset() const { return opset_version_; }

  // std::vector<std::shared_ptr<CustomRegistry>> custom_session_registries_;

  const IOnnxRuntimeOpSchemaRegistryList& CustomSchemaRegistries() const {
    return custom_schema_registries_;
  }

  bool GetAddShapeToTensorData() const { return add_shape_to_tensor_data_; }
  void SetAddShapeToTensorData(bool enable) { add_shape_to_tensor_data_ = enable; }
  void SetAddSymbolicDimToTensorData(int symbolic_dim) { add_symbolic_dim_to_tensor_data_ = symbolic_dim; }
  void SetTestFunctionCalled() { testing_function_called_ = true; }

  struct RunContext {
    SessionOptions session_options{};
    ExpectResult expect_result{ExpectResult::kExpectSuccess};
    std::string expected_failure_string{};
    std::unordered_set<std::string> excluded_provider_types = {};
    const RunOptions* run_options{};
    bool run_with_specified_eps{false};
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers{};
    Graph::ResolveOptions resolve_options{};
  };

  std::vector<Data>& GetInputData() { return input_data_; }
  std::vector<Data>& GetOutputData() { return output_data_; }
  std::vector<size_t>& GetInitializerIndexes() { return initializer_indexes_; }

  void AddInitializers(onnxruntime::Graph& graph);

  void FillFeedsAndOutputNames(std::unordered_map<std::string, OrtValue>& feeds,
                               std::vector<std::string>& output_names);

  const RunContext& GetRunContext() const { return ctx_; }

  template <class SessionType>
  void ExecuteModel(Model& model,
                    SessionType& session_object,
                    ExpectResult expect_result,
                    const std::string& expected_failure_string,
                    const RunOptions* run_options,
                    const std::unordered_map<std::string, OrtValue>& feeds,
                    const std::vector<std::string>& output_names,
                    const std::string& provider_type,
                    bool allow_released_onnx_opset_only = true);

  template <typename T>
  void AddData(std::vector<Data>& data, const char* name, const DimsVariant& dims_var, const T* values,
               int64_t values_count, bool is_initializer = false, bool sort_output = false,
               const std::vector<std::string>* dim_params = nullptr,
               float rel_error = 0.0f, float abs_error = 0.0f, bool is_optional_type_tensor = false) {
    auto dims = ToDimsSpan(dims_var);
#if defined(DISABLE_OPTIONAL_TYPE)
    if (is_optional_type_tensor) {
      ORT_THROW("Optional type is not supported in this build");
    }
#endif

    ORT_TRY {
      TensorShape shape{dims};
      OrtValue value;

      if (!is_optional_type_tensor || (is_optional_type_tensor && values != nullptr)) {
        // In case values is nullptr for optional type tensor, it means we are creating
        // an optional type tensor which is None and we hence skip values count validation
        if constexpr (std::is_same_v<T, Int4x2> || std::is_same_v<T, UInt4x2>) {
          const int64_t expected_values_count = T::CalcNumInt4Pairs(shape.Size());
          ORT_ENFORCE(expected_values_count == values_count, values_count,
                      " input values doesn't match tensor size of ", expected_values_count);
        } else if constexpr (std::is_same_v<T, Int2x4> || std::is_same_v<T, UInt2x4>) {
          const int64_t expected_values_count = T::CalcNumInt2Quads(shape.Size());
          ORT_ENFORCE(expected_values_count == values_count, values_count,
                      " input values doesn't match tensor size of ", expected_values_count);
        }
#if !defined(DISABLE_FLOAT4_TYPES)
        else if constexpr (std::is_same_v<T, Float4E2M1x2>) {
          const int64_t expected_values_count = T::CalcNumFloat4Pairs(shape.Size());
          ORT_ENFORCE(expected_values_count == values_count, values_count,
                      " input values doesn't match tensor size of ", expected_values_count);
        }
#endif
        else {
          ORT_ENFORCE(shape.Size() == values_count, values_count, " input values doesn't match tensor size of ",
                      shape.Size());
        }

        // If it is an optional tensor type with no values (i.e.) None,
        // we won't even pass it in to Run() as part of the feeds,
        // so we don't even have to create a Tensor.
        // Conversely, if it is an optional tensor type with values,
        // we pass it in as a regular tensor.
        auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);
        Tensor::InitOrtValue(DataTypeImpl::GetType<T>(), shape, std::move(allocator), value);

        // values *could* be nullptr for a non-optional tensor if it is empty.
        // Update the data buffer of the input only if values if non-nullptr.
        if (values != nullptr) {
          auto* data_ptr = value.GetMutable<Tensor>()->MutableData<T>();
          for (int64_t i = 0; i < values_count; i++) {
            data_ptr[i] = values[i];
          }
        }
      } else {  // "None" Tensor OrtValue. Initialize appropriately.
        auto ml_tensor = DataTypeImpl::GetType<Tensor>();
        value.Init(nullptr, ml_tensor, ml_tensor->GetDeleteFunc());
      }

      std::vector<int64_t> dims_for_proto = GetDimsForProto(dims);
      TTypeProto<T> tensor_type_proto(add_shape_to_tensor_data_ ? &dims_for_proto : nullptr);

#if !defined(DISABLE_OPTIONAL_TYPE)
      OptionalTypeProto optional_type_proto(tensor_type_proto.proto);
      auto node_arg = NodeArg(name, !is_optional_type_tensor ? &tensor_type_proto.proto : &optional_type_proto.proto);
#else
      auto node_arg = NodeArg(name, &tensor_type_proto.proto);
#endif

      AddShapeToTensorData(node_arg, dims, dim_params);

      optional<float> rel;
      optional<float> abs;

      if (rel_error != 0.0f) {
        rel = rel_error;
      }

      if (abs_error != 0.0f) {
        abs = abs_error;
      }

      data.push_back(Data(std::move(node_arg), std::move(value), std::move(rel), std::move(abs), sort_output));

      // Optional values cannot be initializers
      if (is_initializer && !is_optional_type_tensor) {
        initializer_indexes_.push_back(data.size() - 1);
      }
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::cerr << "AddData for '" << name << "' threw: " << ex.what();
      });
      ORT_RETHROW;
    }
  }

 private:
  void FillFeeds(std::unordered_map<std::string, OrtValue>& feeds);

  RunContext ctx_{};

  std::vector<Data> input_data_;
  std::vector<Data> output_data_;
  std::vector<OrtValue> fetches_;
  int number_of_nodes_;

  bool testing_function_called_{};  // has the function that performs the actual testing been called yet?
  bool has_executed_ep_ = false;
  bool skipped_webgpu_fp16_ = false;
  bool skip_unsupported_webgpu_fp16_ = false;

  bool SkipUnsupportedWebGpuFp16(const Status& status, ExpectResult expect_result,
                                 const std::string& provider_type);

  gsl::span<const int64_t> ToDimsSpan(const DimsVariant& dims_var) {
    return std::visit([](auto&& dims) { return gsl::span<const int64_t>(dims); }, dims_var);
  }

  template <typename T>
  void AddSeqData(std::vector<Data>& data, const char* name,
                  const SeqTensors<T>* seq_tensors,
                  bool is_optional_sequence_tensor_type = false,
                  float rel_error = 0.0f, float abs_error = 0.0f) {
#if defined(DISABLE_OPTIONAL_TYPE)
    if (is_optional_sequence_tensor_type) {
      ORT_THROW("Optional type is not supported in this build");
    }
#endif

    std::unique_ptr<TensorSeq> ptr;
    SequenceTensorTypeProto<T> sequence_tensor_proto;

    if (seq_tensors) {
      auto num_tensors = seq_tensors->tensors.size();
      auto elem_type = DataTypeImpl::GetType<T>();

      ptr = std::make_unique<TensorSeq>(elem_type);
      ptr->Reserve(num_tensors);
      for (size_t i = 0; i < num_tensors; ++i) {
        TensorShape shape{seq_tensors->tensors[i].shape};
        auto values_count = static_cast<int64_t>(seq_tensors->tensors[i].data.size());
        ORT_ENFORCE(shape.Size() == values_count, values_count,
                    " input values doesn't match tensor size of ", shape.Size());

        auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);
        Tensor tensor(elem_type, shape, allocator);

        auto* data_ptr = tensor.MutableData<T>();
        for (int64_t x = 0; x < values_count; ++x) {
          data_ptr[x] = seq_tensors->tensors[i].data[x];
        }

        ptr->Add(std::move(tensor));

        if (add_shape_to_tensor_data_) {
          auto* output_tensor_type = sequence_tensor_proto.proto.mutable_sequence_type()
                                         ->mutable_elem_type()
                                         ->mutable_tensor_type();
          if (i == 0) {
            ONNX_NAMESPACE::TensorShapeProto* seq_input_shape = output_tensor_type->mutable_shape();
            output_tensor_type->set_elem_type(utils::ToTensorProtoElementType<T>());
            for (size_t j = 0; j < shape.NumDimensions(); ++j) {
              auto dim = seq_input_shape->add_dim();
              dim->set_dim_value(shape[j]);
            }
          } else {
            ONNX_NAMESPACE::TensorShapeProto shape_proto;
            for (size_t j = 0; j < shape.NumDimensions(); ++j) {
              auto dim = shape_proto.add_dim();
              dim->set_dim_value(shape[j]);
            }

            ONNX_NAMESPACE::UnionShapeInfo(shape_proto, *output_tensor_type);
          }
        }
      }
    }

    OrtValue value;
    auto mltype = DataTypeImpl::GetType<TensorSeq>();

    // nullptr means None OrtValue which we will skip inserting into the feeds
    value.Init(ptr ? ptr.release() : nullptr, mltype, mltype->GetDeleteFunc());

#if !defined(DISABLE_OPTIONAL_TYPE)
    OptionalTypeProto optional_type_proto(sequence_tensor_proto.proto);
    auto node_arg = NodeArg(name, !is_optional_sequence_tensor_type
                                      ? &sequence_tensor_proto.proto
                                      : &optional_type_proto.proto);
#else
    auto node_arg = NodeArg(name, &sequence_tensor_proto.proto);
#endif

    optional<float> rel;
    optional<float> abs;

    if (rel_error != 0.0f) {
      rel = rel_error;
    }

    if (abs_error != 0.0f) {
      abs = abs_error;
    }

    data.push_back(Data(std::move(node_arg), std::move(value), std::move(rel), std::move(abs)));
  }

  std::vector<int64_t> GetDimsForProto(gsl::span<const int64_t> dims);

  void AddShapeToTensorData(NodeArg& node_arg, gsl::span<const int64_t> dims,
                            const std::vector<std::string>* dim_params);

  void CopyDataToTensor(gsl::span<const std::byte> data, Tensor& dst);

#if !defined(DISABLE_SPARSE_TENSORS)
  NodeArg MakeSparseNodeArg(int32_t dtype, const char* name,
                            const gsl::span<const int64_t>& dims,
                            const std::vector<std::string>* dim_params);

  void AddSparseCooTensorData(std::vector<Data>& data,
                              MLDataType data_type,
                              const char* name,
                              gsl::span<const int64_t> dims,
                              gsl::span<const std::byte> values,
                              gsl::span<const int64_t> indices,
                              const ValidateOutputParams& check_params,
                              const std::vector<std::string>* dim_params = nullptr);

  void AddSparseCooTensorStrings(std::vector<Data>& data,
                                 const char* name,
                                 gsl::span<const int64_t> dims,
                                 gsl::span<const std::string> values,
                                 gsl::span<const int64_t> indices,
                                 const std::vector<std::string>* dim_params = nullptr);

  void AddSparseCsrTensorData(std::vector<Data>& data,
                              MLDataType data_type,
                              const char* name,
                              gsl::span<const int64_t> dims,
                              gsl::span<const std::byte> values,
                              gsl::span<const int64_t> inner_indices,
                              gsl::span<const int64_t> outer_indices,
                              const ValidateOutputParams& check_params,
                              const std::vector<std::string>* dim_params = nullptr);

  void AddSparseCsrTensorStrings(std::vector<Data>& data,
                                 const char* name,
                                 gsl::span<const int64_t> dims,
                                 gsl::span<const std::string> values,
                                 gsl::span<const int64_t> inner_indices,
                                 gsl::span<const int64_t> outer_indices,
                                 const std::vector<std::string>* dim_params = nullptr);

  void AddSparseTensorData(std::vector<Data>& data, NodeArg node_arg,
                           std::unique_ptr<SparseTensor> p_tensor,
                           const ValidateOutputParams& check_params);
#endif

  // Execute the model for a single execution providers combination
  void ExecuteModelForEps(std::vector<std::unique_ptr<IExecutionProvider>>&& execution_providers,
                          onnxruntime::Model& model,
                          SessionOptions sess_options,
                          ExpectResult expect_result,
                          const std::string& expected_failure_string,
                          const onnxruntime::RunOptions* run_options,
                          const std::unordered_map<std::string, OrtValue>& feeds,
                          const std::vector<std::string>& output_names,
                          const std::vector<std::shared_ptr<CustomRegistry>>* custom_registries,
                          bool try_assign_ep_for_nodes,
                          bool allow_released_onnx_opset_only,
                          size_t* number_of_pre_packed_weights_counter,
                          size_t* number_of_shared_pre_packed_weights_counter);

  const std::string test_name_;
  const std::string domain_;
  int opset_version_ = -1;

  bool test_allow_released_onnx_opset_only_ = true;
  bool add_shape_to_tensor_data_ = true;
  int add_symbolic_dim_to_tensor_data_ = -1;

  int num_run_calls_ = 1;

  IOnnxRuntimeOpSchemaRegistryList custom_schema_registries_;
  std::vector<std::shared_ptr<CustomRegistry>> custom_session_registries_;

  bool verify_output_ = true;
  bool use_determinism_ = false;
  CustomOutputVerifierFn custom_output_verifier_;

  std::vector<size_t> initializer_indexes_;

  bool add_prepacked_shared_container_to_sessions_ = false;
  onnxruntime::PrepackedWeightsContainer prepacked_weights_container_;
};

}  // namespace test
}  // namespace onnxruntime

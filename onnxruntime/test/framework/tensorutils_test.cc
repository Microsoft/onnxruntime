// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/common/parse_string.h"
#include "core/framework/endian_utils.h"
#include "core/framework/prepacked_weights.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/framework/tensor.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "test/util/include/asserts.h"
#include "file_util.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <fstream>
#include <sstream>
#include <utility>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#ifdef _WIN32
#include <Windows.h>
#include "core/platform/windows/env.h"
#endif

using namespace ::onnxruntime::utils;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

constexpr bool TensorProtoElementSizesAreConstexpr() {
  constexpr std::array<size_t, TensorProto_DataType_DataType_ARRAYSIZE> expected_sizes{
      0,                 // UNDEFINED
      sizeof(float),     // FLOAT
      sizeof(uint8_t),   // UINT8
      sizeof(int8_t),    // INT8
      sizeof(uint16_t),  // UINT16
      sizeof(int16_t),   // INT16
      sizeof(int32_t),   // INT32
      sizeof(int64_t),   // INT64
      0,                 // STRING
      sizeof(uint8_t),   // BOOL
      sizeof(uint16_t),  // FLOAT16
      sizeof(double),    // DOUBLE
      sizeof(uint32_t),  // UINT32
      sizeof(uint64_t),  // UINT64
      sizeof(float),     // COMPLEX64
      sizeof(double),    // COMPLEX128
      sizeof(uint16_t),  // BFLOAT16
      sizeof(uint8_t),   // FLOAT8E4M3FN
      sizeof(uint8_t),   // FLOAT8E4M3FNUZ
      sizeof(uint8_t),   // FLOAT8E5M2
      sizeof(uint8_t),   // FLOAT8E5M2FNUZ
      sizeof(uint8_t),   // UINT4
      sizeof(uint8_t),   // INT4
      sizeof(uint8_t),   // FLOAT4E2M1
      sizeof(uint8_t),   // FLOAT8E8M0
      sizeof(uint8_t),   // UINT2
      sizeof(uint8_t),   // INT2
      sizeof(uint8_t),   // FLOAT6E2M3
      sizeof(uint8_t),   // FLOAT6E3M2
  };

  for (size_t index = 0; index < expected_sizes.size(); ++index) {
    if (GetElementSizeOfTensor(static_cast<TensorProto_DataType>(index)) != expected_sizes[index]) {
      return false;
    }
  }

  return GetElementSizeOfTensor(static_cast<TensorProto_DataType>(-1)) == 0 &&
         GetElementSizeOfTensor(static_cast<TensorProto_DataType>(expected_sizes.size())) == 0;
}

static_assert(TensorProtoElementSizesAreConstexpr());

// if `expected_error_message_substring` is nullptr, parsing is expected to be successful
static void TestExternalDataInfoParsingOffsetAndLengthWithStrings(
    std::string_view offset_str,
    std::string_view length_str,
    const char* expected_error_message_substring = nullptr) {
  SCOPED_TRACE(MakeString("offset: \"", offset_str, "\", length: \"", length_str, "\""));

  ONNX_NAMESPACE::TensorProto tensor_proto;
  const std::filesystem::path kExternalDataPath("test.bin");

  tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);

  auto* location_entry = tensor_proto.add_external_data();
  location_entry->set_key("location");
  location_entry->set_value(ToUTF8String(kExternalDataPath.native()));

  auto* offset_entry = tensor_proto.add_external_data();
  offset_entry->set_key("offset");
  offset_entry->set_value(offset_str.data(), offset_str.size());

  auto* length_entry = tensor_proto.add_external_data();
  length_entry->set_key("length");
  length_entry->set_value(length_str.data(), length_str.size());

  std::unique_ptr<ExternalDataInfo> external_data_info{};
  const auto create_status = ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info);
  if (expected_error_message_substring) {
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(create_status, expected_error_message_substring);
    return;
  }
  ASSERT_STATUS_OK(create_status);

  // if we got this far, assume that offset_str and length_str are able to be parsed.
  const auto expected_offset = ParseStringWithClassicLocale<ExternalDataInfo::OFFSET_TYPE>(offset_str);
  const auto expected_length = ParseStringWithClassicLocale<size_t>(length_str);

  ASSERT_EQ(external_data_info->GetOffset(), expected_offset);
  ASSERT_EQ(external_data_info->GetLength(), expected_length);
}

// if `expected_error_message_substring` is nullptr, parsing is expected to be successful
static void TestExternalDataInfoParsingOffsetAndLength(intmax_t offset,
                                                       uintmax_t length,
                                                       const char* expected_error_message_substring = nullptr) {
  TestExternalDataInfoParsingOffsetAndLengthWithStrings(std::to_string(offset), std::to_string(length),
                                                        expected_error_message_substring);
}

TEST(TensorProtoUtilsTest, ParseExternalDataInfoOffsetAndLength) {
  TestExternalDataInfoParsingOffsetAndLength(0, 0);

  TestExternalDataInfoParsingOffsetAndLength(0, 1024);
  TestExternalDataInfoParsingOffsetAndLength(0, std::numeric_limits<size_t>::max());

  TestExternalDataInfoParsingOffsetAndLength(1024, 1024);
  TestExternalDataInfoParsingOffsetAndLength(std::numeric_limits<ExternalDataInfo::OFFSET_TYPE>::max(), 1024);

  {
    // assuming that this value is too large to fit in either size_t or ExternalDataInfo::OFFSET_TYPE
    const std::string_view two_to_the_65th_power = "36893488147419103232";
    const std::string_view zero = "0";
    TestExternalDataInfoParsingOffsetAndLengthWithStrings(two_to_the_65th_power, zero, "Failed to parse value");
    TestExternalDataInfoParsingOffsetAndLengthWithStrings(zero, two_to_the_65th_power, "Failed to parse value");
  }

  // TODO should ExternalDataInfo::Create() also reject negative offset values?
}

TEST(TensorProtoUtilsTest, ExternalDataInfoRejectsDuplicateLocations) {
  const std::string memory_tag = ToUTF8String(utils::kTensorProtoNativeEndianMemoryAddressTag);
  const std::array<std::pair<std::string, std::string>, 3> locations{{
      {"data.bin", "data.bin"},
      {"data.bin", memory_tag},
      {memory_tag, "data.bin"},
  }};

  for (const auto& [first_location, second_location] : locations) {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* first = tensor_proto.add_external_data();
    first->set_key("location");
    first->set_value(first_location);
    auto* second = tensor_proto.add_external_data();
    second->set_key("location");
    second->set_value(second_location);

    std::unique_ptr<ExternalDataInfo> external_data_info;
    const Status status = ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "duplicate 'location'");
  }
}

TEST(TensorProtoUtilsTest, HasExternalDataInMemoryChecksAllLocations) {
  auto make_tensor = [](const std::string& first_location, const std::string& second_location) {
    ONNX_NAMESPACE::TensorProto tensor_proto;
    tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
    auto* first = tensor_proto.add_external_data();
    first->set_key("location");
    first->set_value(first_location);
    auto* second = tensor_proto.add_external_data();
    second->set_key("location");
    second->set_value(second_location);
    return tensor_proto;
  };

  const std::string memory_tag = ToUTF8String(utils::kTensorProtoNativeEndianMemoryAddressTag);
  EXPECT_TRUE(utils::HasExternalDataInMemory(make_tensor("data.bin", memory_tag)));
  EXPECT_TRUE(utils::HasExternalDataInMemory(make_tensor(memory_tag, "data.bin")));
  EXPECT_TRUE(utils::HasExternalDataInMemory(make_tensor(memory_tag, "")));
}

// Test ExternalData functionality
TEST(TensorProtoUtilsTest, SetExternalDataInformation) {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  const std::filesystem::path kExternalDataPath("test.bin");
  constexpr const int64_t init_offset = 100;
  constexpr const size_t init_length = 200;

  ExternalDataInfo::SetExternalLocationToProto(kExternalDataPath, init_offset, init_length, tensor_proto);

  ASSERT_EQ(tensor_proto.data_location(), ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
  ASSERT_EQ(tensor_proto.external_data_size(), 3);
  ASSERT_EQ(tensor_proto.external_data(0).key(), "location");
  ASSERT_EQ(tensor_proto.external_data(0).value(), ToUTF8String(kExternalDataPath.native()));
  ASSERT_EQ(tensor_proto.external_data(1).key(), "offset");
  ASSERT_EQ(tensor_proto.external_data(1).value(), std::to_string(init_offset));
  ASSERT_EQ(tensor_proto.external_data(2).key(), "length");
  ASSERT_EQ(tensor_proto.external_data(2).value(), std::to_string(init_length));

  PrepackedKeyToBlobMap key_to_blob;
  constexpr bool save_mode_on = true;
  PrepackedWeightsForGraph prepacked_for_graph(key_to_blob, save_mode_on);
  PrePackedWeights prepacked_weights;
  const std::string init_name = "test_initializer";
  const std::string blob_key = "test_key";

  std::array<float, 2> kData = {1.2345f, 2.4690f};
  const size_t buffer_size = kData.size() * sizeof(float);

  prepacked_weights.buffers_.push_back(BufferUniquePtr(kData.data(), BufferDeleter(nullptr)));
  prepacked_weights.buffer_sizes_.push_back(buffer_size);
  // Write a second entry like this
  prepacked_weights.buffers_.push_back(BufferUniquePtr(kData.data(), BufferDeleter(nullptr)));
  prepacked_weights.buffer_sizes_.push_back(buffer_size);

  prepacked_for_graph.WritePackedMaybeForSave(init_name, blob_key, std::move(prepacked_weights));

  constexpr const int64_t starting_offset = 300;
  int64_t external_offset = starting_offset;
  std::stringstream ss;
  const auto* blobs_for_weight = prepacked_for_graph.GetKeysForWeightForSaving(init_name);
  ASSERT_TRUE(blobs_for_weight != nullptr);
  InlinedHashSet<std::string> blob_keys{blobs_for_weight->begin(), blobs_for_weight->end()};
  ASSERT_TRUE(ExternalDataInfo::WritePrepackedToFileAndAddToProto(prepacked_for_graph,
                                                                  blob_keys,
                                                                  true, 1024 * 1024, 0,
                                                                  ss, external_offset,
                                                                  tensor_proto));

  auto external_data_info = std::make_unique<ExternalDataInfo>();
  ASSERT_STATUS_OK(ExternalDataInfo::Create(tensor_proto.external_data(), external_data_info));

  // This should have prepacked_data entry with two blobs for a single key.
  ASSERT_TRUE(external_data_info->HasPrepackedInfo());
  auto prepacked_infos = external_data_info->TakePrepackedInfos();
  ASSERT_EQ(prepacked_infos.size(), 1U);
  ASSERT_TRUE(prepacked_infos.count(blob_key) > 0);

  int64_t final_offset = starting_offset;
  for (const auto& blob_info : prepacked_infos[blob_key]) {
    int64_t offset = std::get<0>(blob_info);
    ASSERT_EQ(offset, final_offset);
    size_t length = std::get<1>(blob_info);
    std::string checksum = std::get<2>(blob_info);  // currently "0"
    final_offset = offset + length;
    ASSERT_EQ(length, buffer_size);
    ASSERT_EQ(checksum, "0");
  }
  ASSERT_EQ(final_offset, external_offset);
}

TEST(TensorProtoUtilsTest, PrepackedExternalDataHasDeterministicAlignedLayout) {
  PrepackedKeyToBlobMap key_to_blob;
  PrepackedWeightsForGraph prepacked_for_graph{key_to_blob, true};
  std::array<std::string, 3> data{"aaa", "bbbbbbbbb", "ccccc"};
  const std::array<std::string, 3> keys{"a", "b", "c"};
  for (size_t i = 0; i < keys.size(); ++i) {
    PrePackedWeights weights;
    weights.buffers_.push_back(BufferUniquePtr(data[i].data(), BufferDeleter(nullptr)));
    weights.buffer_sizes_.push_back(data[i].size());
    prepacked_for_graph.WritePackedMaybeForSave("weight", keys[i], std::move(weights));
  }

  const std::string expected_bytes =
      std::string(3, '\0') + data[0] + std::string(2, '\0') + data[1] + std::string(7, '\0') + data[2];
  const std::array<std::string, 3> expected_entries{"a|3;3;0", "b|8;9;0", "c|24;5;0"};
  for (bool reverse : {false, true}) {
    for (size_t capacity : {3u, 32u}) {
      SCOPED_TRACE(reverse);
      SCOPED_TRACE(capacity);
      InlinedHashSet<std::string> blob_keys;
      blob_keys.reserve(capacity);
      for (size_t i = 0; i < keys.size(); ++i) {
        blob_keys.insert(keys[reverse ? keys.size() - 1 - i : i]);
      }
      std::stringstream stream;
      stream << std::string(3, '\0');
      int64_t offset = 3;
      TensorProto tensor_proto;
      ASSERT_TRUE(ExternalDataInfo::WritePrepackedToFileAndAddToProto(
          prepacked_for_graph, blob_keys, true, 4, 8, stream, offset, tensor_proto));
      EXPECT_EQ(offset, 29);
      EXPECT_EQ(stream.str(), expected_bytes);
      ASSERT_EQ(tensor_proto.external_data_size(), 3);
      for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_EQ(tensor_proto.external_data(static_cast<int>(i)).key(), "prepacked_" + std::to_string(i));
        EXPECT_EQ(tensor_proto.external_data(static_cast<int>(i)).value(), expected_entries[i]);
      }
    }
  }
}

TEST(PrepackedWeightsForGraphTest, DiscardReferencesProvidedWeightWhenSaving) {
  constexpr const char* weight_name = "weight";
  constexpr const char* key = "key";
  std::array<uint8_t, 1> discarded_data{1};
  std::array<uint8_t, 1> shared_data{2};

  PrePackedWeights discarded_weights;
  discarded_weights.buffers_.emplace_back(discarded_data.data(), BufferDeleter(nullptr));
  discarded_weights.buffer_sizes_.push_back(discarded_data.size());

  PrePackedWeights shared_weights;
  shared_weights.buffers_.emplace_back(shared_data.data(), BufferDeleter(nullptr));
  shared_weights.buffer_sizes_.push_back(shared_data.size());

  PrepackedKeyToBlobMap key_to_blob;
  PrepackedWeightsForGraph prepacked_for_graph(key_to_blob, true);
  prepacked_for_graph.InsertPrepackedWeights(key, std::move(discarded_weights));

  prepacked_for_graph.DiscardAndReplaceWithReferenceIfSaving(weight_name, key, shared_weights);

  const auto* retained_weights = prepacked_for_graph.GetPrepackedWeights(key);
  ASSERT_NE(retained_weights, nullptr);
  ASSERT_EQ(retained_weights->buffers_.size(), 1U);
  EXPECT_EQ(retained_weights->buffers_[0].get(), shared_data.data());
  const auto* keys_for_weight = prepacked_for_graph.GetKeysForWeightForSaving(weight_name);
  ASSERT_NE(keys_for_weight, nullptr);
  EXPECT_EQ(keys_for_weight->count(key), 1U);
}

TEST(PrepackedWeightsForGraphTest, DiscardRemovesWeightWhenNotSaving) {
  constexpr const char* key = "key";
  std::array<uint8_t, 1> discarded_data{1};
  std::array<uint8_t, 1> shared_data{2};

  PrePackedWeights discarded_weights;
  discarded_weights.buffers_.emplace_back(discarded_data.data(), BufferDeleter(nullptr));

  PrePackedWeights shared_weights;
  shared_weights.buffers_.emplace_back(shared_data.data(), BufferDeleter(nullptr));

  PrepackedKeyToBlobMap key_to_blob;
  PrepackedWeightsForGraph prepacked_for_graph(key_to_blob, false);
  prepacked_for_graph.InsertPrepackedWeights(key, std::move(discarded_weights));

  prepacked_for_graph.DiscardAndReplaceWithReferenceIfSaving("weight", key, shared_weights);

  EXPECT_EQ(prepacked_for_graph.GetPrepackedWeights(key), nullptr);
}

// T must be float for double, and it must match with the 'type' argument
template <typename T>
void TestUnpackFloatTensor(TensorProto_DataType type, const std::filesystem::path& model_path) {
  TensorProto float_tensor_proto;
  float_tensor_proto.set_data_type(type);
  T f[4] = {1.1f, 2.2f, 3.3f, 4.4f};
  constexpr size_t len = sizeof(T) * 4;
  char rawdata[len];
  for (int i = 0; i < 4; ++i) {
    memcpy(rawdata + i * sizeof(T), &(f[i]), sizeof(T));
  }
  utils::SetRawDataInTensorProto(float_tensor_proto, rawdata, len);
  T float_data2[4];
  auto status = UnpackTensor(float_tensor_proto, model_path, float_data2, 4);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(1.1f, float_data2[0]);
  EXPECT_EQ(2.2f, float_data2[1]);
  EXPECT_EQ(3.3f, float_data2[2]);
  EXPECT_EQ(4.4f, float_data2[3]);
}

TEST(TensorProtoUtilsTest, UnpackTensor) {
  TensorProto bool_tensor_proto;
  // Path is required for loading external data.
  // Using empty path here since this test does not test
  // external data utils
  std::filesystem::path model_path;
  bool_tensor_proto.set_data_type(TensorProto_DataType_BOOL);
  bool_tensor_proto.add_int32_data(1);

  bool bool_data[1];
  auto status = UnpackTensor(bool_tensor_proto, model_path, bool_data, 1);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(bool_data[0]);

  float float_data[1];
  status = UnpackTensor(bool_tensor_proto, model_path, float_data, 1);
  EXPECT_FALSE(status.IsOK());

  TestUnpackFloatTensor<float>(TensorProto_DataType_FLOAT, model_path);
  TestUnpackFloatTensor<double>(TensorProto_DataType_DOUBLE, model_path);

  TensorProto string_tensor_proto;
  string_tensor_proto.set_data_type(TensorProto_DataType_STRING);
  string_tensor_proto.add_string_data("a");
  string_tensor_proto.add_string_data("b");

  std::string string_data[2];
  status = UnpackTensor(string_tensor_proto, model_path, string_data, 2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ("a", string_data[0]);
  EXPECT_EQ("b", string_data[1]);

  status = UnpackTensor(bool_tensor_proto, model_path, string_data, 2);
  EXPECT_FALSE(status.IsOK());
}

namespace {
TensorProto CreateStringTensorProto(std::initializer_list<size_t> payload_sizes) {
  TensorProto tensor_proto;
  tensor_proto.set_name("string_initializer");
  tensor_proto.set_data_type(TensorProto_DataType_STRING);
  tensor_proto.add_dims(static_cast<int64_t>(payload_sizes.size()));

  for (size_t payload_size : payload_sizes) {
    tensor_proto.add_string_data(std::string(payload_size, 'a'));
  }

  return tensor_proto;
}
}  // namespace

TEST(TensorProtoUtilsTest, ValidateEmbeddedStringTensorProtoRejectsOversizedPayload) {
  constexpr size_t kPayloadBudgetBytes = 128;
  constexpr size_t kTestBudgetBytes = sizeof(std::string) + kPayloadBudgetBytes;
  const TensorProto tensor_proto = CreateStringTensorProto({kPayloadBudgetBytes + 1});

  const Status status = ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto, kTestBudgetBytes);

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "string_data shape bytes + payload exceeds");
}

TEST(TensorProtoUtilsTest, ValidateEmbeddedStringTensorProtoRejectsCombinedShapeAndPayloadOverflow) {
  constexpr size_t kFirstPayloadBytes = 64;
  constexpr size_t kTestBudgetBytes = 2 * sizeof(std::string) + kFirstPayloadBytes;
  const TensorProto tensor_proto = CreateStringTensorProto({kFirstPayloadBytes, 1});

  // STRING tensors account for the shape-declared bytes and the aggregate payload bytes.
  // The first payload exactly fills the remaining allowance after object storage; the second
  // payload proves that validation uses cumulative accounting.
  const Status status = ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto, kTestBudgetBytes);

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "string_data shape bytes + payload exceeds");
}

TEST(TensorProtoUtilsTest, ValidateEmbeddedStringTensorProtoAcceptsExactPayloadLimit) {
  constexpr size_t kPayloadBytes = 17;
  constexpr size_t kTestBudgetBytes = sizeof(std::string) + kPayloadBytes;
  const TensorProto tensor_proto = CreateStringTensorProto({kPayloadBytes});

  ASSERT_STATUS_OK(ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto, kTestBudgetBytes));
}

TEST(TensorProtoUtilsTest, ValidateEmbeddedStringTensorProtoAcceptsNormalPayload) {
  constexpr size_t kTestBudgetBytes = 3 * sizeof(std::string) + 32;
  const TensorProto tensor_proto = CreateStringTensorProto({7, 11, 13});

  ASSERT_STATUS_OK(ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto, kTestBudgetBytes));
}

TEST(TensorProtoUtilsTest, ValidateEmbeddedStringTensorProtoRejectsRawData) {
  TensorProto tensor_proto = CreateStringTensorProto({1});
  tensor_proto.set_raw_data(std::string(sizeof(std::string), '\0'));

  const Status status = ValidateEmbeddedTensorProtoDataSizeAndShape(
      tensor_proto, sizeof(std::string) + 1);

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "string tensor can not have raw data");
}

// A bool initializer supplied through raw_data is copied verbatim, so its bytes are not
// restricted to {0, 1}. UnpackTensor must normalize them so downstream consumers (which assume
// canonical bool values) all observe the same result regardless of how they read the byte.
TEST(TensorProtoUtilsTest, UnpackBoolTensorWithRawDataNormalizesToZeroOne) {
  std::filesystem::path model_path;
  TensorProto bool_tensor_proto;
  bool_tensor_proto.set_data_type(TensorProto_DataType_BOOL);
  bool_tensor_proto.add_dims(4);

  // Bytes outside {0, 1}: 0x00 -> 0, 0x01 -> 1, 0x02 -> 1, 0xFF -> 1.
  const unsigned char raw_bytes[] = {0x00, 0x01, 0x02, 0xFF};
  bool_tensor_proto.set_raw_data(std::string(reinterpret_cast<const char*>(raw_bytes), sizeof(raw_bytes)));

  bool bool_data[4];
  auto status = UnpackTensor(bool_tensor_proto, model_path, bool_data, 4);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const auto* bytes = reinterpret_cast<const unsigned char*>(bool_data);
  EXPECT_EQ(bytes[0], 0);
  EXPECT_EQ(bytes[1], 1);
  EXPECT_EQ(bytes[2], 1);
  EXPECT_EQ(bytes[3], 1);
}

// NormalizeBoolTensorIfNeeded normalizes a CPU bool tensor's bytes to {0, 1} in place. It backs the
// external-initializer device-copy path (session_state_utils.cc), where bool bytes that may be
// non-canonical are normalized in a writable CPU staging copy before being copied to the device.
TEST(TensorProtoUtilsTest, NormalizeBoolTensorIfNeededNormalizesToZeroOne) {
  auto cpu_allocator = std::make_shared<CPUAllocator>();

  // Bool tensor: write non-canonical bytes, then normalize.
  Tensor bool_tensor(DataTypeImpl::GetType<bool>(), TensorShape({4}), cpu_allocator);
  unsigned char* bool_bytes = reinterpret_cast<unsigned char*>(bool_tensor.MutableDataRaw());
  bool_bytes[0] = 0x00;
  bool_bytes[1] = 0x01;
  bool_bytes[2] = 0x02;
  bool_bytes[3] = 0xFF;

  NormalizeBoolTensorIfNeeded(bool_tensor);

  EXPECT_EQ(bool_bytes[0], 0);
  EXPECT_EQ(bool_bytes[1], 1);
  EXPECT_EQ(bool_bytes[2], 1);
  EXPECT_EQ(bool_bytes[3], 1);

  // Non-bool tensor: bytes must be left untouched.
  Tensor int32_tensor(DataTypeImpl::GetType<int32_t>(), TensorShape({3}), cpu_allocator);
  int32_t* int32_data = int32_tensor.MutableData<int32_t>();
  int32_data[0] = 0;
  int32_data[1] = 2;
  int32_data[2] = 255;

  NormalizeBoolTensorIfNeeded(int32_tensor);

  EXPECT_EQ(int32_data[0], 0);
  EXPECT_EQ(int32_data[1], 2);
  EXPECT_EQ(int32_data[2], 255);
}

namespace {
template <typename T>
std::vector<T> CreateValues() {
  return {1, 2, 3, 4};
}

template <>
std::vector<std::string> CreateValues<std::string>() {
  return {"one", "two", "three", "four"};
}

template <>
std::vector<bool> CreateValues() {
  return {true, false, false, true};
}

template <>
std::vector<MLFloat16> CreateValues<MLFloat16>() {
  return {MLFloat16(0.f), MLFloat16(1.f), MLFloat16(2.f), MLFloat16(3.f)};
}

template <>
std::vector<BFloat16> CreateValues<BFloat16>() {
  return {BFloat16(0.f), BFloat16(1.f), BFloat16(2.f), BFloat16(3.f)};
}

template <typename T>
void WriteDataToFile(FILE* fp, const std::vector<T>& test_data) {
  size_t size_in_bytes = test_data.size() * sizeof(T);

  std::vector<unsigned char> data;
  data.resize(size_in_bytes);

  auto src_span = gsl::make_span(test_data.data(), test_data.size());
  auto dst_span = gsl::make_span(data.data(), data.size());

  ASSERT_STATUS_OK(onnxruntime::utils::WriteLittleEndian(src_span, dst_span));
  ASSERT_EQ(size_in_bytes, fwrite(data.data(), 1, size_in_bytes, fp));
}

std::unique_ptr<bool[]> BoolDataFromVector(const std::vector<bool>& test_data) {
  auto arr = std::make_unique<bool[]>(test_data.size());
  std::copy(std::begin(test_data), std::end(test_data), arr.get());
  return arr;
}

// work around std::vector<bool> storing data in bits
template <>
void WriteDataToFile<bool>(FILE* fp, const std::vector<bool>& test_data) {
  auto arr = BoolDataFromVector(test_data);
  size_t size_in_bytes = test_data.size() * sizeof(bool);
  ASSERT_EQ(size_in_bytes, fwrite(arr.get(), 1, size_in_bytes, fp));
}

template <typename T>
void CreateTensorWithExternalData(TensorProto_DataType type, const std::vector<T>& test_data,
                                  std::basic_string<ORTCHAR_T>& filename,
                                  TensorProto& tensor_proto) {
  // Create external data
  FILE* fp;
  CreateTestFile(fp, filename);
  WriteDataToFile(fp, test_data);
  ASSERT_EQ(0, fclose(fp));

  // set the tensor_proto to reference this external data
  onnx::StringStringEntryProto* location = tensor_proto.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));
  tensor_proto.mutable_dims()->Add(test_data.size());
  tensor_proto.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  tensor_proto.set_data_type(type);
}

template <typename T>
void UnpackAndValidate(const TensorProto& tensor_proto, const std::filesystem::path& model_path, const std::vector<T>& test_data) {
  // Unpack tensor with external data
  std::vector<T> val(test_data.size());
  auto st = utils::UnpackTensor(tensor_proto, model_path, val.data(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  // Validate data
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_TRUE(val[i] == test_data[i]);  // need to use ASSERT_TRUE with '==' to handle MFLoat16 and BFloat16
  }
}

template <>
void UnpackAndValidate<bool>(const TensorProto& tensor_proto, const std::filesystem::path& model_path,
                             const std::vector<bool>& test_data) {
  // Unpack tensor with external data
  auto arr = std::make_unique<bool[]>(test_data.size());
  auto st = utils::UnpackTensor(tensor_proto, model_path, arr.get(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  // Validate data
  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_TRUE(arr[i] == test_data[i]);
  }
}

template <typename T>
void TestUnpackExternalTensor(TensorProto_DataType type, const std::filesystem::path& model_path) {
  // Create external data
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("tensor_XXXXXX"));
  TensorProto tensor_proto;
  auto test_data = CreateValues<T>();
  CreateTensorWithExternalData<T>(type, test_data, filename, tensor_proto);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(filename.c_str()),
                                                                         DeleteFileFromDisk);
  UnpackAndValidate(tensor_proto, model_path, test_data);
}
}  // namespace
TEST(TensorProtoUtilsTest, UnpackTensorWithExternalData) {
  std::filesystem::path model_path;
  TestUnpackExternalTensor<float>(TensorProto_DataType_FLOAT, model_path);
  TestUnpackExternalTensor<double>(TensorProto_DataType_DOUBLE, model_path);
  TestUnpackExternalTensor<int32_t>(TensorProto_DataType_INT32, model_path);
  TestUnpackExternalTensor<int8_t>(TensorProto_DataType_INT8, model_path);
  TestUnpackExternalTensor<MLFloat16>(TensorProto_DataType_FLOAT16, model_path);
  TestUnpackExternalTensor<BFloat16>(TensorProto_DataType_BFLOAT16, model_path);
  TestUnpackExternalTensor<bool>(TensorProto_DataType_BOOL, model_path);
}

// A bool initializer supplied through external data is copied verbatim, so its bytes are not
// restricted to {0, 1}. UnpackTensor must normalize them so downstream consumers (which assume
// canonical bool values) all observe the same result regardless of how they read the byte.
TEST(TensorProtoUtilsTest, UnpackBoolTensorWithExternalDataNormalizesToZeroOne) {
  std::filesystem::path model_path;

  // Bytes outside {0, 1}: 0x00 -> 0, 0x01 -> 1, 0x02 -> 1, 0xFF -> 1.
  const unsigned char raw_bytes[] = {0x00, 0x01, 0x02, 0xFF};

  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("bool_tensor_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  ASSERT_EQ(sizeof(raw_bytes), fwrite(raw_bytes, 1, sizeof(raw_bytes), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(filename.c_str()),
                                                                         DeleteFileFromDisk);

  TensorProto bool_tensor_proto;
  onnx::StringStringEntryProto* location = bool_tensor_proto.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));
  bool_tensor_proto.add_dims(4);
  bool_tensor_proto.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  bool_tensor_proto.set_data_type(TensorProto_DataType_BOOL);

  auto arr = std::make_unique<bool[]>(4);
  auto status = utils::UnpackTensor(bool_tensor_proto, model_path, arr.get(), 4);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  const auto* bytes = reinterpret_cast<const unsigned char*>(arr.get());
  EXPECT_EQ(bytes[0], 0);
  EXPECT_EQ(bytes[1], 1);
  EXPECT_EQ(bytes[2], 1);
  EXPECT_EQ(bytes[3], 1);
}

template <typename T>
static NodeProto CreateConstantNode(const std::string& attrib_name, AttributeProto_AttributeType type,
                                    std::function<void(AttributeProto&)> add_data) {
  NodeProto constant_node;
  constant_node.set_op_type("Constant");
  constant_node.add_output("Constant_output");

  AttributeProto& attrib = *constant_node.mutable_attribute()->Add();

  attrib.set_name(attrib_name);
  attrib.set_type(type);
  add_data(attrib);

  return constant_node;
}

template <typename T>
static void TestConstantNodeConversion(const std::string& attrib_name,
                                       AttributeProto_AttributeType type,
                                       std::function<void(AttributeProto&, const std::vector<T>& data)> add_data,
                                       std::function<std::vector<T>(const TensorProto&)> get_data,
                                       int64_t num_elements) {
  auto input = CreateValues<T>();
  if (num_elements == -1) {
    num_elements = static_cast<int64_t>(input.size());
  } else {
    input.resize(num_elements);
  }

  auto c = CreateConstantNode<T>(
      attrib_name, type,
      [&input, &add_data](AttributeProto& attrib) { add_data(attrib, input); });

  TensorProto tp;
  std::filesystem::path model_path;
  EXPECT_STATUS_OK(utils::ConstantNodeProtoToTensorProto(c, model_path, tp));

  EXPECT_THAT(get_data(tp), ::testing::ContainerEq(input));
}

TEST(TensorProtoUtilsTest, ConstantTensorProto) {
  TestConstantNodeConversion<float>(
      "value_float", AttributeProto_AttributeType_FLOAT,
      [](AttributeProto& attrib, const std::vector<float>& data) { attrib.set_f(data[0]); },
      [](const TensorProto& tp) {
        return std::vector<float>(tp.float_data().cbegin(), tp.float_data().cend());
      },
      1);

  TestConstantNodeConversion<float>(
      "value_floats", AttributeProto_AttributeType_FLOATS,
      [](AttributeProto& attrib, const std::vector<float>& data) {
        *attrib.mutable_floats() = {data.cbegin(), data.cend()};
      },
      [](const TensorProto& tp) {
        return std::vector<float>(tp.float_data().cbegin(), tp.float_data().cend());
      },
      -1);

  TestConstantNodeConversion<int64_t>(
      "value_int", AttributeProto_AttributeType_INT,
      [](AttributeProto& attrib, const std::vector<int64_t>& data) { attrib.set_i(data[0]); },
      [](const TensorProto& tp) {
        return std::vector<int64_t>(tp.int64_data().cbegin(), tp.int64_data().cend());
      },
      1);

  TestConstantNodeConversion<int64_t>(
      "value_ints", AttributeProto_AttributeType_INTS,
      [](AttributeProto& attrib, const std::vector<int64_t>& data) {
        *attrib.mutable_ints() = {data.cbegin(), data.cend()};
      },
      [](const TensorProto& tp) {
        return std::vector<int64_t>(tp.int64_data().cbegin(), tp.int64_data().cend());
      },
      -1);

  TestConstantNodeConversion<std::string>(
      "value_string", AttributeProto_AttributeType_STRING,
      [](AttributeProto& attrib, const std::vector<std::string>& data) { attrib.set_s(data[0]); },
      [](const TensorProto& tp) {
        return std::vector<std::string>(tp.string_data().cbegin(), tp.string_data().cend());
      },
      1);

  TestConstantNodeConversion<std::string>(
      "value_strings", AttributeProto_AttributeType_STRINGS,
      [](AttributeProto& attrib, const std::vector<std::string>& data) {
        // for (const auto& s : data)
        *attrib.mutable_strings() = {data.cbegin(), data.cend()};
      },
      [](const TensorProto& tp) {
        return std::vector<std::string>(tp.string_data().cbegin(), tp.string_data().cend());
      },
      -1);

  // sparse_tensor is covered by SparseTensorConversionTests.TestConstantNodeConversion
}

template <typename T>
static NodeProto CreateConstantNodeWithExternalData(TensorProto_DataType type, PathString& tensor_filename,
                                                    const std::vector<T>& test_data) {
  NodeProto constant_node;
  constant_node.set_op_type("Constant");
  constant_node.add_output("Constant_output");

  AttributeProto& attrib = *constant_node.mutable_attribute()->Add();
  attrib.set_name("attrib");
  attrib.set_type(AttributeProto_AttributeType_TENSOR);

  TensorProto& tp = *attrib.mutable_t();
  CreateTensorWithExternalData<T>(type, test_data, tensor_filename, tp);

  return constant_node;
}

template <typename T>
static void TestConstantNodeConversionWithExternalData(TensorProto_DataType type) {
  // Create a constant node with external data
  auto test_data = CreateValues<T>();
  std::filesystem::path model_path;
  PathString tensor_filename(ORT_TSTR("tensor_XXXXXX"));
  auto c = CreateConstantNodeWithExternalData<T>(type, tensor_filename, test_data);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(tensor_filename.c_str()),
                                                                         DeleteFileFromDisk);

  // Convert NodeProto to tensorproto (with external data)
  TensorProto tp;
  EXPECT_STATUS_OK(utils::ConstantNodeProtoToTensorProto(c, model_path, tp));

  // Unpack tensor and validate the data
  std::vector<T> val(test_data.size());
  auto st = utils::UnpackTensor(tp, model_path, val.data(), test_data.size());
  ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

  for (size_t i = 0; i < test_data.size(); i++) {
    ASSERT_EQ(val[i], test_data[i]);
  }
}

TEST(TensorProtoUtilsTest, ConstantTensorProtoWithExternalData) {
  TestConstantNodeConversionWithExternalData<float>(TensorProto_DataType_FLOAT);
  TestConstantNodeConversionWithExternalData<double>(TensorProto_DataType_DOUBLE);
}

// Test fixture for creating temporary directories and files for path validation tests.
class PathValidationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Create a temporary directory for the tests.
    base_dir_ = std::filesystem::temp_directory_path() / "PathValidationTest";
    outside_dir_ = std::filesystem::temp_directory_path() / "outside";
    std::filesystem::create_directories(base_dir_);
    std::filesystem::create_directories(outside_dir_);
  }

  void TearDown() override {
    // Clean up the temporary directory.
    std::filesystem::remove_all(base_dir_);
    std::filesystem::remove_all(outside_dir_);

    for (const auto& other_dir : other_dirs_) {
      std::filesystem::remove_all(other_dir);
    }

    for (const auto& other_file : other_files_) {
      std::filesystem::remove(other_file);
    }
  }

  // Create directory that will be removed during test teardown.
  void CreateDirectories(std::filesystem::path dir) {
    std::filesystem::create_directories(dir);
    other_dirs_.push_back(std::move(dir));
  }

  // Create empty file that will be removed during test teardown.
  void CreateEmptyFile(std::filesystem::path file_path) {
    std::ofstream{file_path};
    other_files_.push_back(std::move(file_path));
  }

  std::filesystem::path base_dir_;
  std::filesystem::path outside_dir_;
  std::vector<std::filesystem::path> other_dirs_;
  std::vector<std::filesystem::path> other_files_;
};

TEST_F(PathValidationTest, ValidateExternalDataPathRejectsMemoryTags) {
  for (const auto* memory_tag : {utils::kTensorProtoLittleEndianMemoryAddressTag,
                                 utils::kTensorProtoNativeEndianMemoryAddressTag}) {
    const Status status = utils::ValidateExternalDataPathFromDir(base_dir_, std::filesystem::path{memory_tag});
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "In-memory external data reference tag");

    const Status wrapper_status =
        utils::ValidateExternalDataPath(base_dir_ / "model.onnx", std::filesystem::path{memory_tag});
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(wrapper_status, "In-memory external data reference tag");
  }
}

#if !defined(__wasm__)
TEST_F(PathValidationTest, Float6ExternalInitializerRetainsPrepackedWeights) {
  constexpr std::array<uint8_t, 3> packed_data{0x81, 0x30, 0x10};
  std::array<uint8_t, 2> packed_weights_data{0xA5, 0x5A};
  const auto data_path = base_dir_ / "float6.bin";
  ONNX_NAMESPACE::TensorProto proto;
  proto.set_name("float6");
  proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3);
  proto.add_dims(4);
  ExternalDataInfo::SetExternalLocationToProto("float6.bin", 0, packed_data.size(), proto);

  PrepackedKeyToBlobMap key_to_blob;
  PrepackedWeightsForGraph saved_weights(key_to_blob, true);
  PrePackedWeights weights;
  weights.buffers_.emplace_back(packed_weights_data.data(), BufferDeleter(nullptr));
  weights.buffer_sizes_.push_back(packed_weights_data.size());
  saved_weights.WritePackedMaybeForSave(proto.name(), "float6_key", std::move(weights));

  std::stringstream blob_stream;
  int64_t offset = packed_data.size();
  ASSERT_TRUE(ExternalDataInfo::WritePrepackedToFileAndAddToProto(
      saved_weights, InlinedHashSet<std::string>{"float6_key"}, true, 1024 * 1024, 0,
      blob_stream, offset, proto));
  {
    std::ofstream data_file(data_path, std::ios::binary);
    data_file.write(reinterpret_cast<const char*>(packed_data.data()), packed_data.size());
    const auto blobs = blob_stream.str();
    data_file.write(blobs.data(), blobs.size());
  }

  PrepackedKeyToBlobMap loaded_key_to_blob;
  PrepackedWeightsForGraph loaded_weights(loaded_key_to_blob, false);
  OrtValue value;
  ASSERT_STATUS_OK(utils::GetExtDataFromTensorProto(
      Env::Default(), base_dir_ / "model.onnx", proto, value, &loaded_weights));
  const auto& tensor = value.Get<Tensor>();
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_EQ(tensor.Data<Float6E2M3>()[i].ToBits(), i + 1);
  }
  const auto* restored = loaded_weights.GetPrepackedWeights("float6_key");
  ASSERT_NE(restored, nullptr);
  ASSERT_EQ(restored->buffers_.size(), 1U);
  ASSERT_EQ(restored->buffer_sizes_[0], packed_weights_data.size());
  EXPECT_EQ(std::memcmp(restored->buffers_[0].get(), packed_weights_data.data(), packed_weights_data.size()), 0);
}

TEST_F(PathValidationTest, Float6ExternalInitializerRejectsShortDataBeforeAllocation) {
  const auto data_path = base_dir_ / "float6_short.bin";
  {
    std::ofstream data_file(data_path, std::ios::binary);
    data_file.put('\x01');
  }
  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    TensorProto proto;
    proto.set_name("float6_short");
    proto.set_data_type(data_type);
    proto.add_dims(sizeof(size_t) == 8 ? int64_t{1} << 40 : int64_t{1} << 28);
    ExternalDataInfo::SetExternalLocationToProto("float6_short.bin", 0, 0, proto);

    OrtValue value;
    const Status status = utils::GetExtDataFromTensorProto(Env::Default(), base_dir_ / "model.onnx", proto, value);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "out of bounds");
  }
}

TEST_F(PathValidationTest, Float6ExternalInitializerRejectsRedirectedFile) {
  const auto outside_file = outside_dir_ / "float6.bin";
  const auto redirected_file = base_dir_ / "float6_redirect.bin";
  {
    std::ofstream output(outside_file, std::ios::binary);
    output.write("\x01\x42\xc3", 3);
  }
  try {
    std::filesystem::create_symlink(outside_file, redirected_file);
  } catch (const std::filesystem::filesystem_error& e) {
    GTEST_SKIP() << "Symlink creation not supported: " << e.what();
  }

  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    TensorProto proto;
    proto.set_name("float6_redirect");
    proto.set_data_type(data_type);
    proto.add_dims(4);
    ExternalDataInfo::SetExternalLocationToProto("float6_redirect.bin", 0, 3, proto);
    OrtValue value;
    const Status status = utils::GetExtDataFromTensorProto(Env::Default(), base_dir_ / "model.onnx", proto, value);
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "escapes");
  }
}

TEST_F(PathValidationTest, Float6EmptyExternalInitializer) {
  CreateEmptyFile(base_dir_ / "float6_empty.bin");
  for (const auto data_type : {TensorProto_DataType_FLOAT6E2M3, TensorProto_DataType_FLOAT6E3M2}) {
    TensorProto proto;
    proto.set_name("float6_empty");
    proto.set_data_type(data_type);
    proto.add_dims(0);
    ExternalDataInfo::SetExternalLocationToProto("float6_empty.bin", 0, 0, proto);

    OrtValue value;
    ASSERT_STATUS_OK(utils::GetExtDataFromTensorProto(Env::Default(), base_dir_ / "model.onnx", proto, value));
    EXPECT_EQ(value.Get<Tensor>().Shape().Size(), 0);
  }
}
#endif  // !defined(__wasm__)

// Test cases for ValidateExternalDataPath.
TEST_F(PathValidationTest, ValidateExternalDataPath) {
  std::filesystem::path model_path = base_dir_ / "model.onnx";
  std::filesystem::path cwd = std::filesystem::current_path();
  const bool is_cwd_root = cwd == cwd.root_path();

  // Create empty external data files that we'll need for testing.
  CreateEmptyFile(base_dir_ / "data.bin");
  CreateDirectories(base_dir_ / "sub");
  CreateEmptyFile(base_dir_ / "sub" / "data.bin");
  CreateEmptyFile(cwd / "data.bin");
  CreateDirectories(cwd / "abc");
  CreateEmptyFile(cwd / "abc" / "data.bin");
  CreateEmptyFile(cwd / "data..bin");

  // Valid relative path.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "data.bin"));

  // Empty location not allowed.
  {
    Status status = utils::ValidateExternalDataPath(model_path, "");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Empty external data path"));
  }

  // Path with ".." that escapes the base directory.
  ASSERT_FALSE(utils::ValidateExternalDataPath(model_path, "../data.bin").IsOK());

  // Absolute path.
  {
    Status status;
#ifdef _WIN32
    status = utils::ValidateExternalDataPath(model_path, "C:\\data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));

    status = utils::ValidateExternalDataPath("", "C:\\data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
#endif  // Absolute path.

    // Paths starting from / should be rejected even on Windows.
    status = utils::ValidateExternalDataPath(model_path, "/data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));

    status = utils::ValidateExternalDataPath("", "/data.bin");
    ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
  }

  // Windows vs Unix path separators.
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "sub\\data.bin"));
#endif
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "sub/data.bin"));

  // Model in a directory that does not exist.
  ASSERT_FALSE(utils::ValidateExternalDataPath("non_existent_dir/model.onnx", "data.bin").IsOK());

  // Model path is a bare filename (no directory component).
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("model.onnx", "data.bin"));
  ASSERT_EQ(utils::ValidateExternalDataPath("model.onnx", "../data.bin").IsOK(), is_cwd_root);

  // Model relative path checks.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("./model.onnx", "data.bin"));
  ASSERT_EQ(utils::ValidateExternalDataPath("./model.onnx", "../data.bin").IsOK(), is_cwd_root);
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(".\\model.onnx", "data.bin"));
  ASSERT_EQ(utils::ValidateExternalDataPath(".\\model.onnx", "../data.bin").IsOK(), is_cwd_root);
#endif

  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("./abc/model.onnx", "data.bin"));
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(".\\abc\\model.onnx", "data.bin"));
#endif

  //
  // Tests for an empty model path (model loaded from bytes).
  // The model path would be empty when 1) the session loads a model from bytes and 2) the application does not
  // set an external file folder path via the session config option
  // kOrtSessionOptionsModelExternalInitializersFileFolderPath.
  //

  // A simple filename is ok (would not escape current working directory).
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "data.bin"));
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "./data.bin"));

  // A ".." that is not a path component (part of the filename) is ok
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "data..bin"));

  // A path that would escape the current working directory is invalid.
  ASSERT_EQ(utils::ValidateExternalDataPath("", "../data.bin").IsOK(), is_cwd_root);

  // A path that uses ".." but would not escape the current working directory should be fine.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath("", "a/../data.bin"));

  // A path with multiple internal ".." that would escape current working direction should fail.
  ASSERT_EQ(utils::ValidateExternalDataPath("", "a/../../data.bin").IsOK(), is_cwd_root);
}

TEST_F(PathValidationTest, ValidateExternalDataPathWithSymlinkInside) {
  // Symbolic link that points inside the base directory.
  auto model_path = base_dir_ / "model.onnx";
  try {
    auto target = base_dir_ / "target.bin";
    std::ofstream{target};
    auto link = base_dir_ / "link.bin";
    std::filesystem::create_symlink(target, link);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink tests since symlink creation is not supported in this environment. Exception: "
                 << e.what();
  }
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(model_path, "link.bin"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathWithSymlinkOutside) {
  // Symbolic link that points outside the base directory.
  auto model_path = base_dir_ / "model.onnx";
  auto outside_target = outside_dir_ / "outside.bin";
  try {
    {
      std::ofstream{outside_target};
      auto outside_link = base_dir_ / "outside_link.bin";
      std::filesystem::create_symlink(outside_target, outside_link);
    }
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink tests since symlink creation is not supported in this environment. Exception: " << e.what();
  }
  ASSERT_FALSE(utils::ValidateExternalDataPath(model_path, "outside_link.bin").IsOK());
}

TEST_F(PathValidationTest, ValidateExternalDataPathEmptyModelPathWithSymlinkInside) {
  // Test external data path validation when the model path is empty.
  // Specifically tests that the following scenario is valid:
  //   - A symbolic link within the current working directory pointing to a file still within CWD.
  try {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path sub_dir = cwd / "symlink_test_subdir";
    CreateDirectories(sub_dir);

    std::filesystem::path target = sub_dir / "target_inside.bin";
    std::filesystem::path symlink = sub_dir / "link_inside.bin";
    std::ofstream{target};
    std::filesystem::create_symlink(target, symlink);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping test due to failure setting up directory and symlink files: "
                 << e.what();
  }

  EXPECT_STATUS_OK(utils::ValidateExternalDataPath("", "./symlink_test_subdir/link_inside.bin"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathEmptyModelPathWithSymlinkOutside) {
  // Test external data path validation when the model path is empty.
  // Specifically tests that the following scenario is NOT valid:
  //  - A symbolic link within the current working directory pointing to a file outside CWD.
  try {
    std::filesystem::path cwd = std::filesystem::current_path();
    std::filesystem::path sub_dir = cwd / "symlink_test_subdir2";
    CreateDirectories(sub_dir);

    // Check if we can actually make a file outside of the current working directory (i.e., in a temp dir).
    // This is only possible if the current working directory is NOT the same as the temp directory.
    // Otherwise, we need to skip this test. This happens in Android CI.
    auto [cwd_end, outside_end] = std::mismatch(cwd.begin(), cwd.end(), outside_dir_.begin(), outside_dir_.end());
    if (cwd_end == cwd.end()) {
      GTEST_SKIP() << "Skipping test that needs to create a symlink outside of the cwd because the cwd is the same as "
                   << "the temp dir. cwd: " << cwd << " outside_dir_: " << outside_dir_;
    }

    std::filesystem::path outside_target = outside_dir_ / "outside_for_empty_basedir.bin";
    std::filesystem::path symlink = sub_dir / "outside_link.bin";
    std::ofstream{outside_target};
    std::filesystem::create_symlink(outside_target, symlink);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping test due to failure setting up directory and symlink files: "
                 << e.what();
  }

  Status status = utils::ValidateExternalDataPath("", "./symlink_test_subdir2/outside_link.bin");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("escapes working directory"));
}

// Test that symlinked model + symlinked data in same blob store passes (HuggingFace layout).
// Layout:
//   base_dir_/blobs/model_blob       (real model file)
//   base_dir_/blobs/data_blob        (real data file)
//   base_dir_/snapshots/v1/model.onnx -> ../../blobs/model_blob  (symlink)
//   base_dir_/snapshots/v1/data.bin   -> ../../blobs/data_blob   (symlink)
TEST_F(PathValidationTest, ValidateExternalDataPathSymlinkedModelAndData_HuggingFaceLayout) {
  auto blobs_dir = base_dir_ / "blobs";
  auto snapshots_dir = base_dir_ / "snapshots" / "v1";
  try {
    CreateDirectories(blobs_dir);
    CreateDirectories(snapshots_dir);

    // Create real files in blobs/
    std::ofstream{blobs_dir / "model_blob"};
    std::ofstream{blobs_dir / "data_blob"};

    // Create symlinks in snapshots/v1/
    std::filesystem::create_symlink(blobs_dir / "model_blob", snapshots_dir / "model.onnx");
    std::filesystem::create_symlink(blobs_dir / "data_blob", snapshots_dir / "data.bin");
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink test: " << e.what();
  }

  // model.onnx is a symlink; data.bin is also a symlink.
  // Both resolve to the same blobs/ directory — should pass.
  ASSERT_STATUS_OK(utils::ValidateExternalDataPath(snapshots_dir / "model.onnx", "data.bin"));
}

// Test that symlinked model + empty external data path is rejected (not silently accepted).
TEST_F(PathValidationTest, ValidateExternalDataPathSymlinkedModel_EmptyPathRejected) {
  auto blobs_dir = base_dir_ / "blobs";
  auto snapshots_dir = base_dir_ / "snapshots" / "v1";
  try {
    CreateDirectories(blobs_dir);
    CreateDirectories(snapshots_dir);
    std::ofstream{blobs_dir / "model_blob"};
    std::filesystem::create_symlink(blobs_dir / "model_blob", snapshots_dir / "model.onnx");
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink test: " << e.what();
  }

  Status status = utils::ValidateExternalDataPath(snapshots_dir / "model.onnx", "");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Empty external data path"));
}

// Test that symlinked model + absolute external data path is rejected without symlink fallback.
TEST_F(PathValidationTest, ValidateExternalDataPathSymlinkedModel_AbsolutePathRejected) {
  auto blobs_dir = base_dir_ / "blobs";
  auto snapshots_dir = base_dir_ / "snapshots" / "v1";
  try {
    CreateDirectories(blobs_dir);
    CreateDirectories(snapshots_dir);
    std::ofstream{blobs_dir / "model_blob"};
    std::filesystem::create_symlink(blobs_dir / "model_blob", snapshots_dir / "model.onnx");
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink test: " << e.what();
  }

  Status status = utils::ValidateExternalDataPath(snapshots_dir / "model.onnx", "/etc/passwd");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Absolute path not allowed"));
}

// Test that symlinked model + data symlink resolving OUTSIDE the real model dir is rejected.
TEST_F(PathValidationTest, ValidateExternalDataPathSymlinkedModel_DataEscapesRealDir) {
  auto blobs_dir = base_dir_ / "blobs";
  auto snapshots_dir = base_dir_ / "snapshots" / "v1";
  try {
    CreateDirectories(blobs_dir);
    CreateDirectories(snapshots_dir);
    std::ofstream{blobs_dir / "model_blob"};
    std::filesystem::create_symlink(blobs_dir / "model_blob", snapshots_dir / "model.onnx");

    // Create data symlink that resolves outside of blobs/ (the real model dir)
    auto outside_target = outside_dir_ / "evil.bin";
    std::ofstream{outside_target};
    std::filesystem::create_symlink(outside_target, snapshots_dir / "evil_data.bin");
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink test: " << e.what();
  }

  Status status = utils::ValidateExternalDataPath(snapshots_dir / "model.onnx", "evil_data.bin");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("escapes model directory"));
}

// Test that ValidateExternalDataPathFromDir rejects a non-existent file even if path is valid.
TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_NonExistentFileRejected) {
  // base_dir_ exists but "no_such_file.bin" does not.
  Status status = utils::ValidateExternalDataPathFromDir(base_dir_, "no_such_file.bin");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("does not exist"));
}

// Tests for ValidateExternalDataPathFromDir (directory-based overload used by EPs).
TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_ValidSubpath) {
  // A valid relative path under the given directory should pass.
  CreateEmptyFile(base_dir_ / "engine.cache");
  CreateDirectories(base_dir_ / "subdir");
  CreateEmptyFile(base_dir_ / "subdir" / "engine.cache");

  ASSERT_STATUS_OK(utils::ValidateExternalDataPathFromDir(base_dir_, "engine.cache"));
  ASSERT_STATUS_OK(utils::ValidateExternalDataPathFromDir(base_dir_, "subdir/engine.cache"));
#ifdef _WIN32
  ASSERT_STATUS_OK(utils::ValidateExternalDataPathFromDir(base_dir_, "subdir\\engine.cache"));
#endif
}

TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_EscapeViaDotDot) {
  // A path with ".." that escapes the directory must fail.
  CreateEmptyFile(outside_dir_ / "data.bin");

  Status status = utils::ValidateExternalDataPathFromDir(base_dir_, "../data.bin");
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("External data path escapes model directory"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_AbsolutePathRejected) {
  // Absolute paths must be rejected.
#ifdef _WIN32
  Status status = utils::ValidateExternalDataPathFromDir(base_dir_, "C:\\data.bin");
  ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
#endif
  Status status_unix = utils::ValidateExternalDataPathFromDir(base_dir_, "/data.bin");
  ASSERT_THAT(status_unix.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_EmptyPathRejected) {
  Status status = utils::ValidateExternalDataPathFromDir(base_dir_, "");
  ASSERT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Empty external data path"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_EmptyDirFallsToCwd) {
  // When directory is empty, should fall back to current working directory.
  std::filesystem::path cwd = std::filesystem::current_path();
  CreateEmptyFile(cwd / "fromdir_test_data.bin");
  other_files_.push_back(cwd / "fromdir_test_data.bin");

  ASSERT_STATUS_OK(utils::ValidateExternalDataPathFromDir("", "fromdir_test_data.bin"));
}

TEST_F(PathValidationTest, ValidateExternalDataPathFromDir_SymlinkOutsideRejected) {
  // A symlink that resolves outside the given directory must be rejected.
  auto outside_target = outside_dir_ / "outside_engine.bin";
  try {
    std::ofstream{outside_target};
    auto symlink_path = base_dir_ / "escape_link.bin";
    std::filesystem::create_symlink(outside_target, symlink_path);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Skipping symlink test: " << e.what();
  }

  Status status = utils::ValidateExternalDataPathFromDir(base_dir_, "escape_link.bin");
  ASSERT_FALSE(status.IsOK());
}

#if defined(_WIN32)
// Direct tests for the Windows AppContainer fallback used by
// WindowsEnv::GetWeaklyCanonicalPath. The AppContainer trigger itself can't be
// reproduced in a unit test environment; see microsoft/onnxruntime#28508.
TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_ExistingDirectory) {
  std::filesystem::path canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(base_dir_, canonical));

  EXPECT_THAT(canonical.wstring(), testing::StartsWith(L"\\\\?\\GLOBALROOT\\Device\\"));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::exists(canonical, ec)) << "ec=" << ec.message();
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_ExistingFile) {
  CreateEmptyFile(base_dir_ / "data.bin");

  std::filesystem::path canonical;
  ASSERT_TRUE(
      onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(base_dir_ / "data.bin", canonical));

  EXPECT_THAT(canonical.wstring(), testing::StartsWith(L"\\\\?\\GLOBALROOT\\Device\\"));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::exists(canonical, ec)) << "ec=" << ec.message();
  EXPECT_TRUE(std::filesystem::is_regular_file(canonical, ec)) << "ec=" << ec.message();
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_NonExistentLeafLexicallyAppended) {
  const std::filesystem::path leaf{L"does_not_exist.bin"};
  std::filesystem::path canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(base_dir_ / leaf, canonical));

  EXPECT_THAT(canonical.wstring(), testing::StartsWith(L"\\\\?\\GLOBALROOT\\Device\\"));
  EXPECT_EQ(canonical.filename(), leaf);

  // The canonicalized parent must be a path-component prefix of the result so that the
  // containment check in ValidateExternalDataPath still works.
  std::filesystem::path parent_canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(base_dir_, parent_canonical));
  auto [parent_end, full_it] = std::mismatch(parent_canonical.begin(), parent_canonical.end(),
                                             canonical.begin(), canonical.end());
  EXPECT_EQ(parent_end, parent_canonical.end())
      << "parent: " << parent_canonical << " full: " << canonical;
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_NonExistentMiddleAndLeaf) {
  std::filesystem::path canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(
      base_dir_ / L"missing_dir" / L"data.bin", canonical));

  EXPECT_THAT(canonical.wstring(), testing::StartsWith(L"\\\\?\\GLOBALROOT\\Device\\"));
  EXPECT_EQ(canonical.filename(), std::filesystem::path{L"data.bin"});
  EXPECT_EQ(canonical.parent_path().filename(), std::filesystem::path{L"missing_dir"});
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_AllNonExistentReturnsFalse) {
  // Synthetic absolute path on a non-existent volume. The fallback must return false so
  // the caller surfaces the original weakly_canonical error rather than substituting an
  // unverified path.
  const std::filesystem::path bogus{L"\\\\?\\Volume{00000000-0000-0000-0000-000000000000}\\nope\\data.bin"};
  std::filesystem::path canonical;
  EXPECT_FALSE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(bogus, canonical));
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_MatchesWeaklyCanonicalAtFile) {
  // Compare via std::filesystem::equivalent: the fallback returns the NT form
  // (\\?\GLOBALROOT\Device\HarddiskVolumeN\...) while weakly_canonical returns the DOS
  // form (C:\...), but both must point at the same file.
  CreateEmptyFile(base_dir_ / "compare.bin");
  const auto target = base_dir_ / "compare.bin";

  std::error_code ec;
  const auto reference = std::filesystem::weakly_canonical(target, ec);
  ASSERT_FALSE(ec) << ec.message();

  std::filesystem::path fallback;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(target, fallback));

  EXPECT_TRUE(std::filesystem::equivalent(reference, fallback, ec))
      << "reference=" << reference << " fallback=" << fallback << " ec=" << ec.message();
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_ResolvesSymlinks) {
  const auto target = base_dir_ / "symlink_target.bin";
  const auto link = base_dir_ / "symlink_link.bin";
  try {
    std::ofstream{target};
    std::filesystem::create_symlink(target, link);
  } catch (const std::exception& e) {
    GTEST_SKIP() << "Symlink creation not supported in this environment: " << e.what();
  }

  std::filesystem::path link_canonical;
  std::filesystem::path target_canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(link, link_canonical));
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(target, target_canonical));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::equivalent(link_canonical, target_canonical, ec))
      << "link=" << link_canonical << " target=" << target_canonical << " ec=" << ec.message();
}

TEST_F(PathValidationTest, WeaklyCanonicalPathNtVolumeFallback_ResolvesDotDot) {
  CreateDirectories(base_dir_ / "sub_for_dotdot");

  std::filesystem::path canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(
      base_dir_ / "sub_for_dotdot" / "..", canonical));

  std::filesystem::path base_canonical;
  ASSERT_TRUE(onnxruntime::internal::WeaklyCanonicalPathNtVolumeFallbackForTesting(base_dir_, base_canonical));

  std::error_code ec;
  EXPECT_TRUE(std::filesystem::equivalent(canonical, base_canonical, ec))
      << "canonical=" << canonical << " base=" << base_canonical << " ec=" << ec.message();
}
#endif  // defined(_WIN32)

#if !defined(DISABLE_SPARSE_TENSORS)
// Regression test: SparseTensorProtoToDenseTensorProto must reject external_data paths
// that escape the model directory (path traversal via "../" in location).
TEST_F(PathValidationTest, SparseTensorExternalDataPathTraversalBlocked_Values) {
  // Create model directory and a "secret" file outside it.
  auto model_dir = base_dir_ / "model_dir";
  std::error_code ec;
  std::filesystem::create_directories(model_dir, ec);
  ASSERT_FALSE(ec) << "Failed to create model_dir: " << ec.message();

  // Write known float data to a file outside the model directory.
  auto secret_file = base_dir_ / "secret.bin";
  {
    std::ofstream ofs(secret_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << secret_file;
    float secret_data[] = {42.0f, 99.0f};
    ofs.write(reinterpret_cast<const char*>(secret_data), sizeof(secret_data));
    ASSERT_TRUE(ofs.good()) << "Failed to write to " << secret_file;
  }

  // Construct a SparseTensorProto whose values use external data with a path-traversal location.
  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);  // dense shape: [4]

  // Values tensor: 2 non-zero float values stored in external file.
  auto* values = sparse.mutable_values();
  values->set_name("sparse_test");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(2);  // 2 non-zero elements
  values->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* loc = values->add_external_data();
  loc->set_key("location");
  loc->set_value("../secret.bin");  // path traversal!

  auto* len_entry = values->add_external_data();
  len_entry->set_key("length");
  len_entry->set_value(std::to_string(2 * sizeof(float)));

  // Indices: positions 0 and 1 in the dense tensor.
  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(2);
  indices->add_int64_data(0);
  indices->add_int64_data(1);

  // Attempt to convert — this should fail with a path validation error.
  ONNX_NAMESPACE::TensorProto dense;
  std::filesystem::path model_path = model_dir / "model.onnx";
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "SparseTensorProtoToDenseTensorProto should reject path-traversal "
                                 "in values external_data location, but it succeeded (reading "
                                 "arbitrary file outside model directory).";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("escapes"));
}

// Same as above but for path traversal in the indices external data.
TEST_F(PathValidationTest, SparseTensorExternalDataPathTraversalBlocked_Indices) {
  auto model_dir = base_dir_ / "model_dir";
  std::error_code ec;
  std::filesystem::create_directories(model_dir, ec);
  ASSERT_FALSE(ec) << "Failed to create model_dir: " << ec.message();

  // Write indices data (2 x int64) to a file outside the model directory.
  auto secret_file = base_dir_ / "indices_secret.bin";
  {
    std::ofstream ofs(secret_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << secret_file;
    int64_t idx_data[] = {0, 1};
    ofs.write(reinterpret_cast<const char*>(idx_data), sizeof(idx_data));
    ASSERT_TRUE(ofs.good()) << "Failed to write to " << secret_file;
  }

  // Also need a valid values file inside the model directory.
  auto values_file = model_dir / "values.bin";
  {
    std::ofstream ofs(values_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << values_file;
    float val_data[] = {1.0f, 2.0f};
    ofs.write(reinterpret_cast<const char*>(val_data), sizeof(val_data));
    ASSERT_TRUE(ofs.good()) << "Failed to write to " << values_file;
  }

  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);

  // Values: legitimate external data within model directory.
  auto* values = sparse.mutable_values();
  values->set_name("sparse_idx_test");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(2);
  values->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* val_loc = values->add_external_data();
  val_loc->set_key("location");
  val_loc->set_value("values.bin");

  auto* val_len = values->add_external_data();
  val_len->set_key("length");
  val_len->set_value(std::to_string(2 * sizeof(float)));

  // Indices: external data with path traversal.
  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(2);
  indices->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* idx_loc = indices->add_external_data();
  idx_loc->set_key("location");
  idx_loc->set_value("../indices_secret.bin");  // path traversal!

  auto* idx_len = indices->add_external_data();
  idx_len->set_key("length");
  idx_len->set_value(std::to_string(2 * sizeof(int64_t)));

  ONNX_NAMESPACE::TensorProto dense;
  std::filesystem::path model_path = model_dir / "model.onnx";
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "SparseTensorProtoToDenseTensorProto should reject path-traversal "
                                 "in indices external_data location, but it succeeded.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("escapes"));
}

// Regression test: SparseTensorProtoToDenseTensorProto must reject absolute paths
// in values external_data location.
TEST_F(PathValidationTest, SparseTensorExternalDataAbsolutePathBlocked_Values) {
  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);

  auto* values = sparse.mutable_values();
  values->set_name("abs_path_test");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(2);
  values->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* loc = values->add_external_data();
  loc->set_key("location");
  loc->set_value("/data.bin");  // absolute path

  auto* len_entry = values->add_external_data();
  len_entry->set_key("length");
  len_entry->set_value(std::to_string(2 * sizeof(float)));

  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(2);
  indices->add_int64_data(0);
  indices->add_int64_data(1);

  ONNX_NAMESPACE::TensorProto dense;
  std::filesystem::path model_path = base_dir_ / "model.onnx";
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "SparseTensorProtoToDenseTensorProto should reject absolute path "
                                 "in values external_data location.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));

#ifdef _WIN32
  // Also verify Windows-style absolute path.
  loc->set_value("C:\\data.bin");
  status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "SparseTensorProtoToDenseTensorProto should reject Windows absolute path "
                                 "in values external_data location.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
#endif
}

// Regression test: SparseTensorProtoToDenseTensorProto must reject absolute paths
// in indices external_data location.
TEST_F(PathValidationTest, SparseTensorExternalDataAbsolutePathBlocked_Indices) {
  // Create a valid values file inside base_dir_ so values validation passes.
  auto values_file = base_dir_ / "values.bin";
  {
    std::ofstream ofs(values_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << values_file;
    float val_data[] = {1.0f, 2.0f};
    ofs.write(reinterpret_cast<const char*>(val_data), sizeof(val_data));
    ASSERT_TRUE(ofs.good()) << "Failed to write to " << values_file;
  }

  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);

  // Values: legitimate external data within base_dir_.
  auto* values = sparse.mutable_values();
  values->set_name("abs_path_idx_test");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(2);
  values->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* val_loc = values->add_external_data();
  val_loc->set_key("location");
  val_loc->set_value("values.bin");

  auto* val_len = values->add_external_data();
  val_len->set_key("length");
  val_len->set_value(std::to_string(2 * sizeof(float)));

  // Indices: external data with absolute path.
  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(2);
  indices->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* idx_loc = indices->add_external_data();
  idx_loc->set_key("location");
  idx_loc->set_value("/data.bin");  // absolute path

  auto* idx_len = indices->add_external_data();
  idx_len->set_key("length");
  idx_len->set_value(std::to_string(2 * sizeof(int64_t)));

  ONNX_NAMESPACE::TensorProto dense;
  std::filesystem::path model_path = base_dir_ / "model.onnx";
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "SparseTensorProtoToDenseTensorProto should reject absolute path "
                                 "in indices external_data location.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));

#ifdef _WIN32
  idx_loc->set_value("C:\\data.bin");
  status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "SparseTensorProtoToDenseTensorProto should reject Windows absolute path "
                                 "in indices external_data location.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
#endif
}

// Regression test: validation must still reject escaping paths for zero-element dense tensors,
// which previously returned early before path validation ran.
TEST_F(PathValidationTest, SparseTensorExternalDataPathTraversalBlocked_ZeroDenseElements) {
  auto model_dir = base_dir_ / "model_dir";
  std::error_code ec;
  std::filesystem::create_directories(model_dir, ec);
  ASSERT_FALSE(ec) << "Failed to create model_dir: " << ec.message();

  // Create the escaping file so that a "file not found" error would NOT be raised.
  auto secret_file = base_dir_ / "secret.bin";
  {
    std::ofstream ofs(secret_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << secret_file;
    ofs.put('\0');
    ASSERT_TRUE(ofs.good()) << "Failed to write to " << secret_file;
  }

  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(0);  // dense shape [0] → dense_elements == 0

  auto* values = sparse.mutable_values();
  values->set_name("zero_dense_test");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(0);  // NNZ=0
  values->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* loc = values->add_external_data();
  loc->set_key("location");
  loc->set_value("../secret.bin");  // path traversal

  auto* len_entry = values->add_external_data();
  len_entry->set_key("length");
  len_entry->set_value("0");

  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(0);

  ONNX_NAMESPACE::TensorProto dense;
  std::filesystem::path model_path = model_dir / "model.onnx";
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "Should reject path-traversal in values even when dense_elements == 0.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("escapes"));
}

// Regression test: validation must reject escaping paths in indices even when NNZ == 0.
TEST_F(PathValidationTest, SparseTensorExternalDataPathTraversalBlocked_ZeroNNZ) {
  auto model_dir = base_dir_ / "model_dir";
  std::error_code ec;
  std::filesystem::create_directories(model_dir, ec);
  ASSERT_FALSE(ec) << "Failed to create model_dir: " << ec.message();

  // Create the escaping file so that a "file not found" error would NOT be raised.
  auto secret_file = base_dir_ / "indices_secret.bin";
  {
    std::ofstream ofs(secret_file, std::ios::binary);
    ASSERT_TRUE(ofs.is_open()) << "Failed to open " << secret_file;
    ofs.put('\0');
    ASSERT_TRUE(ofs.good()) << "Failed to write to " << secret_file;
  }

  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);  // dense shape [4] → non-zero dense_elements

  auto* values = sparse.mutable_values();
  values->set_name("zero_nnz_test");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(0);  // NNZ=0

  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(0);
  indices->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* idx_loc = indices->add_external_data();
  idx_loc->set_key("location");
  idx_loc->set_value("../indices_secret.bin");  // path traversal

  auto* idx_len = indices->add_external_data();
  idx_len->set_key("length");
  idx_len->set_value("0");

  ONNX_NAMESPACE::TensorProto dense;
  std::filesystem::path model_path = model_dir / "model.onnx";
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, model_path, dense);
  ASSERT_FALSE(status.IsOK()) << "Should reject path-traversal in indices even when NNZ == 0.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("escapes"));
}

// Defense-in-depth: SparseTensorProtoToDenseTensorProto must reject ORT's in-memory address
// marker on sparse sub-tensors unconditionally. The trusted .ort loader is required to
// materialize sparse sub-tensors as inline raw_data so they never carry markers. Without this
// self-check, a caller that bypasses the Graph-ctor chokepoint would dereference an
// attacker-controlled address.
TEST(SparseTensorProtoToDenseTensorProtoMarkerTest, RejectsInMemoryMarkerOnValuesByDefault) {
  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);

  auto* values = sparse.mutable_values();
  values->set_name("sparse_marker_values");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(2);
  values->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = values->add_external_data();
  loc->set_key("location");
  loc->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag));
  auto* off = values->add_external_data();
  off->set_key("offset");
  off->set_value("0");
  auto* len = values->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(2 * sizeof(float)));

  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(2);
  indices->add_int64_data(0);
  indices->add_int64_data(1);

  ONNX_NAMESPACE::TensorProto dense;
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, std::filesystem::path{}, dense);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("in-memory address marker"));
}

TEST(SparseTensorProtoToDenseTensorProtoMarkerTest, RejectsInMemoryMarkerOnIndicesByDefault) {
  ONNX_NAMESPACE::SparseTensorProto sparse;
  sparse.add_dims(4);

  auto* values = sparse.mutable_values();
  values->set_name("sparse_marker_indices");
  values->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  values->add_dims(2);
  values->add_float_data(1.0f);
  values->add_float_data(2.0f);

  auto* indices = sparse.mutable_indices();
  indices->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  indices->add_dims(2);
  indices->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto* loc = indices->add_external_data();
  loc->set_key("location");
  loc->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag));
  auto* off = indices->add_external_data();
  off->set_key("offset");
  off->set_value("0");
  auto* len = indices->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(2 * sizeof(int64_t)));

  ONNX_NAMESPACE::TensorProto dense;
  Status status = utils::SparseTensorProtoToDenseTensorProto(sparse, std::filesystem::path{}, dense);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("in-memory address marker"));
}

#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Defense-in-depth: ConstantNodeProtoToTensorProto must reject ORT's in-memory address marker
// on a Constant node's dense tensor attribute. This isolates the guard added in
// ConstantNodeProtoToTensorProto from the pre-existing dense-initializer guard in the Graph
// constructor: callers such as Graph::AddConstantProtoAsInitializer and the ORT-format build
// path emplace directly into name_to_initial_tensor_ and bypass that constructor-side check,
// so this test exercises the new chokepoint directly.
TEST(ConstantNodeProtoToTensorProtoMarkerTest, RejectsInMemoryMarkerOnDenseTensorAttribute) {
  ONNX_NAMESPACE::NodeProto node;
  node.set_op_type("Constant");
  node.set_name("malicious_constant");
  node.add_output("c");

  auto* attr = node.add_attribute();
  attr->set_name("value");
  attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  auto* t = attr->mutable_t();
  t->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  t->add_dims(4);
  t->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  // Backing buffer is irrelevant — the guard must reject before any dereference.
  static std::vector<uint8_t> backing(16, 0);

  auto* loc = t->add_external_data();
  loc->set_key("location");
  loc->set_value(ToUTF8String(onnxruntime::utils::kTensorProtoLittleEndianMemoryAddressTag));
  auto* off = t->add_external_data();
  off->set_key("offset");
  off->set_value(std::to_string(reinterpret_cast<intptr_t>(backing.data())));
  auto* len = t->add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(backing.size()));

  ONNX_NAMESPACE::TensorProto tensor_out;
  Status status = utils::ConstantNodeProtoToTensorProto(node, std::filesystem::path{}, tensor_out);
  ASSERT_FALSE(status.IsOK())
      << "Constant node tensor attribute with an in-memory address marker must be rejected.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("in-memory address marker"));
}

// A Constant node reaches this helper from model-local function bodies and subgraphs, where the
// output list and the attribute set are model controlled and need not match the op schema.
TEST(ConstantNodeProtoToTensorProtoTest, RejectsUnexpectedOutputCountAndAttributeType) {
  // No output: the tensor name is derived from output(0), which must not be indexed blindly.
  {
    ONNX_NAMESPACE::NodeProto node;
    node.set_op_type("Constant");
    node.set_name("no_output_constant");

    auto* attr = node.add_attribute();
    attr->set_name("value_int");
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr->set_i(1);

    ONNX_NAMESPACE::TensorProto tensor_out;
    Status status = utils::ConstantNodeProtoToTensorProto(node, std::filesystem::path{}, tensor_out);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("should have 1 output"));
  }

  // More than one output.
  {
    ONNX_NAMESPACE::NodeProto node;
    node.set_op_type("Constant");
    node.set_name("two_output_constant");
    node.add_output("c0");
    node.add_output("c1");

    auto* attr = node.add_attribute();
    attr->set_name("value_int");
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
    attr->set_i(1);

    ONNX_NAMESPACE::TensorProto tensor_out;
    Status status = utils::ConstantNodeProtoToTensorProto(node, std::filesystem::path{}, tensor_out);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("should have 1 output"));
  }

  // No attributes at all.
  {
    ONNX_NAMESPACE::NodeProto node;
    node.set_op_type("Constant");
    node.set_name("no_attribute_constant");
    node.add_output("c");

    ONNX_NAMESPACE::TensorProto tensor_out;
    Status status = utils::ConstantNodeProtoToTensorProto(node, std::filesystem::path{}, tensor_out);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("has no data attributes"));
  }

  // An attribute whose type carries no value must produce a Status, not an exception, so that
  // builds without exception support reject the model instead of terminating.
  {
    ONNX_NAMESPACE::NodeProto node;
    node.set_op_type("Constant");
    node.set_name("undefined_attribute_constant");
    node.add_output("c");

    auto* attr = node.add_attribute();
    attr->set_name("value");
    attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED);

    ONNX_NAMESPACE::TensorProto tensor_out;
    Status status = utils::ConstantNodeProtoToTensorProto(node, std::filesystem::path{}, tensor_out);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Unsupported attribute value type"));
  }
}

// Defense-in-depth: GetExtDataFromTensorProto must reject absolute external paths even when
// called with an empty model_path (e.g. from training checkpoint or custom-op init paths).
// Previously, ValidateExternalDataPath was only invoked from Graph::ConvertInitializersIntoOrtValues,
// so direct callers of GetExtDataFromTensorProto could load arbitrary files.
TEST(GetExtDataFromTensorProtoTest, RejectsAbsoluteExternalPathWithEmptyModelPath) {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name("abs_external");
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(2);
  tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* loc = tensor_proto.add_external_data();
  loc->set_key("location");
#ifdef _WIN32
  loc->set_value("C:\\data.bin");
#else
  loc->set_value("/etc/passwd");
#endif

  auto* off = tensor_proto.add_external_data();
  off->set_key("offset");
  off->set_value("0");

  auto* len = tensor_proto.add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(2 * sizeof(float)));

  OrtValue value;
  Status status = utils::GetExtDataFromTensorProto(Env::Default(), {}, tensor_proto, value);
  ASSERT_FALSE(status.IsOK()) << "Absolute external path must be rejected even with empty model_path.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("Absolute path not allowed"));
}

// Defense-in-depth: GetExtDataFromTensorProto must reject directory-escaping external paths even
// when the caller passes a non-empty model_path. This guards callers outside Graph::Resolve.
TEST(GetExtDataFromTensorProtoTest, RejectsEscapingExternalPath) {
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name("escape_external");
  tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(2);
  tensor_proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

  auto* loc = tensor_proto.add_external_data();
  loc->set_key("location");
  loc->set_value("../escape.bin");

  auto* off = tensor_proto.add_external_data();
  off->set_key("offset");
  off->set_value("0");

  auto* len = tensor_proto.add_external_data();
  len->set_key("length");
  len->set_value(std::to_string(2 * sizeof(float)));

  OrtValue value;
  // Pass a synthetic model_path so the validator has a model directory to compare against.
  std::filesystem::path model_path = std::filesystem::temp_directory_path() / "sub" / "model.onnx";
  Status status = utils::GetExtDataFromTensorProto(Env::Default(), model_path, tensor_proto, value);
  ASSERT_FALSE(status.IsOK()) << "Directory-escaping external path must be rejected.";
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("escapes"));
}

TEST(TensorProtoUtilsTest, GetNodeProtoLayeringAnnotation) {
  // Case 1: Annotation exists
  {
    ONNX_NAMESPACE::NodeProto node_proto;
    node_proto.set_name("test_node");
    auto* prop = node_proto.add_metadata_props();
    prop->set_key(utils::kNodeProtoLayerAnnotation);
    prop->set_value("foo");

    auto result = utils::GetNodeProtoLayeringAnnotation(node_proto);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "foo");
  }

  // Case 2: Annotation missing (empty metadata_props)
  {
    ONNX_NAMESPACE::NodeProto node_proto;
    node_proto.set_name("test_node");

    auto result = utils::GetNodeProtoLayeringAnnotation(node_proto);
    EXPECT_FALSE(result.has_value());
  }

  // Case 3: Other metadata exists, but not the annotation
  {
    ONNX_NAMESPACE::NodeProto node_proto;
    node_proto.set_name("test_node");
    auto* prop = node_proto.add_metadata_props();
    prop->set_key("some_other_key");
    prop->set_value("some_value");

    auto result = utils::GetNodeProtoLayeringAnnotation(node_proto);
    EXPECT_FALSE(result.has_value());
  }

  // Case 4: Multiple metadata, including the annotation
  {
    ONNX_NAMESPACE::NodeProto node_proto;
    node_proto.set_name("test_node");

    auto* prop1 = node_proto.add_metadata_props();
    prop1->set_key("some_other_key");
    prop1->set_value("some_value");

    auto* prop2 = node_proto.add_metadata_props();
    prop2->set_key(utils::kNodeProtoLayerAnnotation);
    prop2->set_value("bar");

    auto* prop3 = node_proto.add_metadata_props();
    prop3->set_key("yet_another_key");
    prop3->set_value("baz");

    auto result = utils::GetNodeProtoLayeringAnnotation(node_proto);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(result.value(), "bar");
  }
}

// Tests for ValidateEmbeddedTensorProtoDataSizeAndShape and embedded initializer size limits

TEST(TensorProtoDataSizeShapeValidationTest, ValidTensorProtoWithRawData) {
  // A valid float tensor with 4 elements and matching raw_data
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_raw");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(2);
  tensor_proto.add_dims(2);
  // 4 floats = 16 bytes
  std::string raw(16, '\0');
  tensor_proto.set_raw_data(raw);

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, ValidTensorProtoWithTypedData) {
  // A valid float tensor with typed float_data
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_typed");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(3);
  tensor_proto.add_float_data(1.0f);
  tensor_proto.add_float_data(2.0f);
  tensor_proto.add_float_data(3.0f);

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, ValidZeroElementTensor) {
  // A valid zero-element tensor (one dim is 0)
  TensorProto tensor_proto;
  tensor_proto.set_name("zero_elem");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(0);
  tensor_proto.add_dims(5);

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, LargeDimsNoDataRejected) {
  // Malicious: large dims but no data at all
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_no_data");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(10000);
  tensor_proto.add_dims(10000);
  // No raw_data or float_data set

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected count from shape"));
}

TEST(TensorProtoDataSizeShapeValidationTest, LargeDimsSmallRawDataRejected) {
  // Malicious: large dims with tiny raw_data
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_small_raw");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(10000);
  tensor_proto.add_dims(10000);
  // Only 4 bytes of raw data (1 float), but shape says 100M elements
  std::string raw(4, '\0');
  tensor_proto.set_raw_data(raw);

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected size from shape and data type"));
}

TEST(TensorProtoDataSizeShapeValidationTest, LargeDimsSmallTypedDataRejected) {
  // Malicious: large dims with just a few typed data elements
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_small_typed");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(10000);
  tensor_proto.add_dims(10000);
  tensor_proto.add_float_data(1.0f);

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected count from shape"));
}

TEST(TensorProtoDataSizeShapeValidationTest, EmbeddedInitializerExceeding2GiBRejected) {
  // A tensor whose declared shape exceeds 2 GiB should be rejected by TensorProtoToOrtValue and
  // CreateTensorFromTensorProto.
  TensorProto tensor_proto;
  tensor_proto.set_name("too_large");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  // 536870913 floats * 4 bytes = 2147483652 bytes > 2 GiB
  tensor_proto.add_dims(536870913);
  // No data — the 2 GiB check should trigger before the consistency check

  // Test call to TensorProtoToOrtValue
  {
    OrtValue ort_value;
    auto status = utils::TensorProtoToOrtValue(Env::Default(), std::filesystem::path{},
                                               tensor_proto, CPUAllocator::DefaultInstance(), ort_value);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("exceeds the 2147483648 byte limit"));
  }

  // Test call to CreateTensorFromTensorProto
  {
    Tensor tensor;
    auto status = utils::CreateTensorFromTensorProto(Env::Default(), std::filesystem::path{},
                                                     tensor_proto, tensor);
    ASSERT_FALSE(status.IsOK());
    EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("exceeds the 2147483648 byte limit"));
  }
}

TEST(TensorProtoDataSizeShapeValidationTest, ValidStringTensorProto) {
  // A valid string tensor with matching string_data
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_string");
  tensor_proto.set_data_type(TensorProto_DataType_STRING);
  tensor_proto.add_dims(2);
  tensor_proto.add_string_data("hello");
  tensor_proto.add_string_data("world");

  ASSERT_STATUS_OK(utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto));
}

TEST(TensorProtoDataSizeShapeValidationTest, StringTensorWithMismatchedCountRejected) {
  TensorProto tensor_proto;
  tensor_proto.set_name("bad_string");
  tensor_proto.set_data_type(TensorProto_DataType_STRING);
  tensor_proto.add_dims(100);
  tensor_proto.add_string_data("only_one");

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("does not match expected count from shape"));
}

TEST(TensorProtoDataSizeShapeValidationTest, NegativeDimsRejected) {
  TensorProto tensor_proto;
  tensor_proto.set_name("negative_dims");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(-1);
  tensor_proto.add_dims(10);

  auto status = utils::ValidateEmbeddedTensorProtoDataSizeAndShape(tensor_proto);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("negative dimensions"));
}

#if !defined(__wasm__)
// Tests for external data file size validation in ReadExternalDataForTensor.
// These verify that the file size is checked before allocating memory for the tensor.

TEST(TensorProtoDataSizeShapeValidationTest, ExternalDataFileTooSmallForDeclaredShape) {
  // Create a small external data file with 4 floats (16 bytes)
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("ext_small_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  const float small_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(sizeof(small_data), fwrite(small_data, 1, sizeof(small_data), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(
      const_cast<ORTCHAR_T*>(filename.c_str()), DeleteFileFromDisk);

  // Declare a tensor with 1000 floats (4000 bytes) but the file only has 16 bytes
  TensorProto tensor_proto;
  tensor_proto.set_name("malicious_external");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(1000);
  tensor_proto.set_data_location(TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor_proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));

  std::vector<uint8_t> unpacked_tensor;
  auto status = utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("out of bounds"));
}

TEST(TensorProtoDataSizeShapeValidationTest, ExternalDataOffsetPushesReadPastEndOfFile) {
  // Create an external data file with 4 floats (16 bytes)
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("ext_offset_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(sizeof(data), fwrite(data, 1, sizeof(data), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(
      const_cast<ORTCHAR_T*>(filename.c_str()), DeleteFileFromDisk);

  // Declare a tensor with 4 floats (16 bytes) but at offset 8, so read needs bytes [8..24) but file is only 16 bytes
  TensorProto tensor_proto;
  tensor_proto.set_name("offset_external");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(4);
  tensor_proto.set_data_location(TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor_proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));
  auto* offset = tensor_proto.add_external_data();
  offset->set_key("offset");
  offset->set_value("8");

  std::vector<uint8_t> unpacked_tensor;
  auto status = utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("out of bounds"));
}

TEST(TensorProtoDataSizeShapeValidationTest, ExternalDataValidFileSizeSucceeds) {
  // Create an external data file with exactly 4 floats (16 bytes)
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("ext_valid_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, filename);
  const float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_EQ(sizeof(data), fwrite(data, 1, sizeof(data), fp));
  ASSERT_EQ(0, fclose(fp));
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(
      const_cast<ORTCHAR_T*>(filename.c_str()), DeleteFileFromDisk);

  // Declare a tensor with matching shape (4 floats = 16 bytes)
  TensorProto tensor_proto;
  tensor_proto.set_name("valid_external");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(4);
  tensor_proto.set_data_location(TensorProto_DataLocation_EXTERNAL);
  auto* location = tensor_proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));

  std::vector<uint8_t> unpacked_tensor;
  ASSERT_STATUS_OK(utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor));
  ASSERT_EQ(unpacked_tensor.size(), sizeof(data));
}

TEST(TensorProtoDataSizeShapeValidationTest, UnpackInitializerDataRejectsInlineRawDataShapeMismatch) {
  TensorProto tensor_proto;
  tensor_proto.set_name("inline_raw_mismatch");
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_dims(2);

  const float raw_value = 1.0f;
  utils::SetRawDataInTensorProto(tensor_proto, &raw_value, sizeof(raw_value));

  std::vector<uint8_t> unpacked_tensor;
  auto status = utils::UnpackInitializerData(tensor_proto, std::filesystem::path{}, unpacked_tensor);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), ::testing::HasSubstr("raw_data size"));
}
#endif  // !defined(__wasm__)

}  // namespace test
}  // namespace onnxruntime

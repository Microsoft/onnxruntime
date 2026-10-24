// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/graph/graph_flatbuffers_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test/flatbuffers/flatbuffers_utils_test.fbs.h"

#include "test/util/include/asserts.h"

namespace onnxruntime {
using namespace fbs::utils;
namespace test {

namespace {
void CreateWriter(const std::string& filename, std::ofstream& external_data_stream, ExternalDataWriter& writer) {
  external_data_stream = std::ofstream(filename, std::ios::binary);

  ASSERT_FALSE(external_data_stream.fail());

  // setup the data writer to write aligned data to external_data_stream
  writer = [&external_data_stream](int32_t data_type, gsl::span<const uint8_t> bytes, uint64_t& offset) {
    // align everything to 4 or 8 bytes. we can optimize this later if needed.
    // aligning to data type or more is required on ARM platforms for the data to be used.
    int32_t alignment = 4;

    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_INT64 ||
        data_type == ONNX_NAMESPACE::TensorProto_DataType_DOUBLE) {
      alignment = 8;
    }

    int64_t pos = external_data_stream.tellp();

    if (pos % alignment != 0) {
      // 8 bytes of 0's so we can pad to alignment 8 in a single `write`
      constexpr static const uint64_t zeros = 0;
      int64_t padding = alignment - (pos % alignment);
      external_data_stream.write(reinterpret_cast<const char*>(&zeros), padding);
      ORT_ENFORCE(!external_data_stream.fail(), "Failed adding padding to external checkpoint data.");
      pos += padding;
    }

    external_data_stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    ORT_ENFORCE(!external_data_stream.fail(), "Failed writing external checkpoint data.");
    ORT_ENFORCE(pos + int64_t(bytes.size()) == external_data_stream.tellp());  // sanity check

    offset = pos;
    return Status::OK();
  };
}

void CreateReader(const std::string& filename, std::ifstream& external_data_stream, ExternalDataReader& reader) {
  external_data_stream = std::ifstream(filename, std::ios::binary);

  ASSERT_FALSE(external_data_stream.fail()) << "Failed to open data file.";

  reader = [&external_data_stream](uint64_t offset, gsl::span<uint8_t> output_buffer) {
    external_data_stream.seekg(offset);
    external_data_stream.read(reinterpret_cast<char*>(output_buffer.data()), output_buffer.size());

    ORT_ENFORCE(!external_data_stream.fail(),
                "Failed to read external checkpoint data. Offset:", offset, " Bytes:", output_buffer.size());

    return Status::OK();
  };
}

template <typename T>
ONNX_NAMESPACE::TensorProto CreateInitializer(const std::string& name,
                                              ONNX_NAMESPACE::TensorProto_DataType data_type,
                                              const std::vector<int64_t>& dims) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(data_type);

  int64_t num_elements = 1;
  for (auto dim : dims) {
    tp.add_dims(dim);
    num_elements *= dim;
  }

  std::vector<T> data(num_elements);
  std::iota(data.begin(), data.end(), T(1));  // fill with 1..num_elements

  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
      for (auto val : data) {
        tp.add_int64_data(val);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT: {
      for (auto val : data) {
        tp.add_float_data(val);
      }
      break;
    }
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32: {
      for (auto val : data) {
        tp.add_int32_data(static_cast<int32_t>(val));
      }
      break;
    }
    default:
      ORT_THROW("Unsupported data type: ", data_type);
  }

  return tp;
}

template <>
ONNX_NAMESPACE::TensorProto CreateInitializer<std::string>(const std::string& name,
                                                           ONNX_NAMESPACE::TensorProto_DataType data_type,
                                                           const std::vector<int64_t>& dims) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(name);
  tp.set_data_type(data_type);

  int64_t num_elements = 1;
  for (auto dim : dims) {
    tp.add_dims(dim);
    num_elements *= dim;
  }

  for (int i = 0; i < num_elements; ++i) {
    tp.add_string_data("string_" + std::to_string(i));
  }

  return tp;
}

std::vector<ONNX_NAMESPACE::TensorProto> CreateInitializers() {
  std::vector<ONNX_NAMESPACE::TensorProto> initializers;
  // create data of various sizes. order is chosen to require padding between most but not all.
  // our writer aligns to 4 bytes unless it's 64-bit data (which is aligned to 8 bytes).
  // buffer: <16-bit data><pad 2 bytes><32-bit data><8-bit data><pad 7 bytes><64-bit data>
  // 128 bytes is required to write to external data. each data type has enough elements to satisfy this.

  // 16-bit. 81 elements so we're 2 bytes past 4 byte alignment
  initializers.emplace_back(
      CreateInitializer<int16_t>("tensor_16", ONNX_NAMESPACE::TensorProto_DataType_INT16, {9, 9}));

  // string (should not use external)
  initializers.emplace_back(
      CreateInitializer<std::string>("tensor_string", ONNX_NAMESPACE::TensorProto_DataType_STRING, {2, 2}));

  // 32-bit, 64 elements
  initializers.emplace_back(
      CreateInitializer<float>("tensor_f32", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {8, 8}));

  // 8-bit. 129 elements so we're 1 byte past 4 or 8 byte alignment
  initializers.emplace_back(
      CreateInitializer<uint8_t>("tensor_8", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {3, 43}));

  // small (should not use external)
  initializers.emplace_back(
      CreateInitializer<int32_t>("tensor_32_small", ONNX_NAMESPACE::TensorProto_DataType_INT32, {2, 2}));

  // 64-bit, 36 elements
  initializers.emplace_back(
      CreateInitializer<int64_t>("tensor_64", ONNX_NAMESPACE::TensorProto_DataType_INT64, {6, 6}));

  return initializers;
}

#ifdef ENABLE_TRAINING_APIS
std::vector<ONNX_NAMESPACE::TensorProto> CreateInitializersNoString() {
  std::vector<ONNX_NAMESPACE::TensorProto> initializers;
  // create data of various sizes. order is chosen to require padding between most but not all
  // assuming our writer aligns to 4 bytes unless it's 64-bit data (which is aligned to 8 bytes)
  // buffer: <16-bit><pad 2><32-bit><8-bit><pad 7><64-bit>
  // need 128 bytes to write to external data

  // 16-bit. 81 elements so we're 2 bytes past 4 byte alignment
  initializers.emplace_back(
      CreateInitializer<int16_t>("tensor_16", ONNX_NAMESPACE::TensorProto_DataType_INT16, {9, 9}));

  // 32-bit, 64 elements
  initializers.emplace_back(
      CreateInitializer<float>("tensor_f32", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {8, 8}));

  // 8-bit. 129 elements so we're 1 byte past 4 or 8 byte alignment
  initializers.emplace_back(
      CreateInitializer<uint8_t>("tensor_8", ONNX_NAMESPACE::TensorProto_DataType_UINT8, {3, 43}));

  // 64-bit, 36 elements
  initializers.emplace_back(
      CreateInitializer<int64_t>("tensor_64", ONNX_NAMESPACE::TensorProto_DataType_INT64, {6, 6}));

  // small (should not use external)
  initializers.emplace_back(
      CreateInitializer<int32_t>("tensor_32_small", ONNX_NAMESPACE::TensorProto_DataType_INT32, {2, 2}));

  return initializers;
}
#endif  // ENABLE_TRAINING_APIS

#define ASSERT_EQ_TENSORPROTO_VECTORFIELD(EXPECTED, ACTUAL, FIELD) \
  ASSERT_EQ((EXPECTED).FIELD.size(), (ACTUAL).FIELD.size());       \
  for (int j = 0; j < (EXPECTED).FIELD.size(); ++j) {              \
    ASSERT_EQ((EXPECTED).FIELD[j], (ACTUAL).FIELD[j]);             \
  }

}  // namespace

// tests method that loads to tensorproto protobuf (used when loading a checkpoint into an inference model)
TEST(FlatbufferUtilsTest, ExternalWriteReadWithLoadInitializers) {
  // create data
  auto initializers = CreateInitializers();

  flatbuffers::FlatBufferBuilder builder(1024);

  // write
  std::ofstream output_stream;
  ExternalDataWriter writer;
  CreateWriter("ExternalWriteReadBasicTest.bin", output_stream, writer);

  std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors;
  for (const auto& initializer : initializers) {
    flatbuffers::Offset<fbs::Tensor> fbs_tensor;
    ASSERT_STATUS_OK(SaveInitializerOrtFormat(builder, initializer, std::filesystem::path(), fbs_tensor, writer));
    fbs_tensors.push_back(fbs_tensor);
  }

  ASSERT_EQ(output_stream.tellp(), 840) << "Data written to the external file is incorrect.";
  output_stream.close();
  ASSERT_TRUE(output_stream.good()) << "Failed to close data file.";

  auto fbs_tensors_offset = builder.CreateVector(fbs_tensors);
  fbs::test::TestDataBuilder tdb(builder);
  tdb.add_initializers(fbs_tensors_offset);

  builder.Finish(tdb.Finish());
  auto fb_data = builder.GetBufferSpan();

  const auto* test_data = fbs::test::GetTestData(fb_data.data());
  const auto* fbs_tensors2 = test_data->initializers();

  // read
  std::ifstream input_stream;
  ExternalDataReader reader;
  CreateReader("ExternalWriteReadBasicTest.bin", input_stream, reader);

  std::vector<ONNX_NAMESPACE::TensorProto> loaded_initializers;
  OrtFormatLoadOptions options;

  for (const auto* fbs_tensor : *fbs_tensors2) {
    ONNX_NAMESPACE::TensorProto initializer;
    ASSERT_STATUS_OK(LoadInitializerOrtFormat(*fbs_tensor, initializer, options, reader));
    loaded_initializers.emplace_back(std::move(initializer));
    // also check that the loaded flatbuffer tensors have accurately written to the external_data_offset field
    if (fbs_tensor->data_type() != fbs::TensorDataType::STRING && fbs_tensor->name()->str() != "tensor_32_small") {
      ASSERT_GE(fbs_tensor->external_data_offset(), 0)
          << "external_data_offset is not set when we expect it to be set for tensor " << fbs_tensor->name()->str();
    } else {
      ASSERT_EQ(fbs_tensor->external_data_offset(), -1)
          << "external_data_offset is set for string data when we expect it to not be set for tensor "
          << fbs_tensor->name()->str();
      ASSERT_TRUE(fbs_tensor->raw_data() || fbs_tensor->string_data())
          << "tensor has no data attached to it" << fbs_tensor->name()->str();
    }
  }

  // initializers = expected, in the form of tensorproto
  // loaded_initializers = actual, in the form of tensorproto
  ASSERT_EQ(initializers.size(), loaded_initializers.size());

  for (int i = 0; i < narrow<int>(initializers.size()); i++) {
    const auto& expected_initializer = initializers[i];
    const auto& loaded_initializer = loaded_initializers[i];
    // validate the loaded initializer
    ASSERT_EQ(expected_initializer.name(), loaded_initializer.name());
    ASSERT_EQ(expected_initializer.data_type(), loaded_initializer.data_type());
    ASSERT_EQ_TENSORPROTO_VECTORFIELD(expected_initializer, loaded_initializer, dims());

    if (loaded_initializer.data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING) {
      std::vector<uint8_t> expected_data, loaded_data;
      ASSERT_STATUS_OK(onnxruntime::utils::UnpackInitializerData(expected_initializer, expected_data));
      ASSERT_STATUS_OK(onnxruntime::utils::UnpackInitializerData(loaded_initializer, loaded_data));
      ASSERT_EQ(expected_data, loaded_data) << loaded_initializer.name();
    } else {
      // string type tensor
      ASSERT_EQ_TENSORPROTO_VECTORFIELD(expected_initializer, loaded_initializer, string_data());
    }
  }
}

TEST(FlatbufferUtilsTest, Float6InitializerOrtFormatRoundTrip) {
  for (auto type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3,
                    ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2}) {
    for (int64_t count : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4},
                          int64_t{5}, int64_t{6}, int64_t{7},
                          int64_t{89}, int64_t{90}, int64_t{91}}) {
      SCOPED_TRACE(::testing::Message() << "type=" << type << " count=" << count);
      ONNX_NAMESPACE::TensorProto initializer;
      initializer.set_name("float6");
      initializer.set_data_type(type);
      initializer.add_dims(count);
      for (int64_t i = 0; i < count; ++i) {
        initializer.add_int32_data(static_cast<int32_t>((i * 7 + 1) % 64));
      }

      std::vector<uint8_t> external_data;
      ExternalDataWriter writer = [&external_data](int32_t, gsl::span<const uint8_t> bytes, uint64_t& offset) {
        offset = 0;
        external_data.assign(bytes.begin(), bytes.end());
        return Status::OK();
      };
      flatbuffers::FlatBufferBuilder builder;
      flatbuffers::Offset<fbs::Tensor> tensor_offset;
      ASSERT_STATUS_OK(SaveInitializerOrtFormat(builder, initializer, {}, tensor_offset, writer));
      builder.Finish(tensor_offset);

      const auto* tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());
      const size_t packed_size = static_cast<size_t>((count * 6 + 7) / 8);
      if (count < 89) {
        ASSERT_NE(tensor->raw_data(), nullptr);
        EXPECT_EQ(tensor->raw_data()->size(), packed_size);
        EXPECT_EQ(tensor->external_data_offset(), -1);
        EXPECT_TRUE(external_data.empty());
      } else {
        EXPECT_EQ(tensor->external_data_offset(), 0);
        EXPECT_EQ(tensor->raw_data(), nullptr);
        EXPECT_EQ(external_data.size(), packed_size);
      }

      ExternalDataReader reader = [&external_data](uint64_t offset, gsl::span<uint8_t> bytes) {
        ORT_ENFORCE(offset == 0 && bytes.size() == external_data.size());
        std::copy(external_data.begin(), external_data.end(), bytes.begin());
        return Status::OK();
      };
      ONNX_NAMESPACE::TensorProto loaded;
      OrtFormatLoadOptions options;
      ASSERT_STATUS_OK(LoadInitializerOrtFormat(*tensor, loaded, options, reader));
      EXPECT_EQ(loaded.name(), initializer.name());
      EXPECT_EQ(loaded.data_type(), type);
      ASSERT_EQ(loaded.dims_size(), 1);
      EXPECT_EQ(loaded.dims(0), count);
      std::vector<uint8_t> loaded_data;
      ASSERT_STATUS_OK(onnxruntime::utils::UnpackInitializerData(loaded, loaded_data));
      ASSERT_EQ(loaded_data.size(), static_cast<size_t>(count));
      for (int64_t i = 0; i < count; ++i) {
        EXPECT_EQ(loaded_data[static_cast<size_t>(i)], static_cast<uint8_t>((i * 7 + 1) % 64))
            << "element " << i;
      }
    }
  }
}

TEST(FlatbufferUtilsTest, Float6NativeMemoryInitializerOrtFormatRoundTrip) {
  for (const auto type : {ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E2M3,
                          ONNX_NAMESPACE::TensorProto_DataType_FLOAT6E3M2}) {
    constexpr size_t count = 132;
    std::vector<uint8_t> unpacked(count);
    for (size_t i = 0; i < count; ++i) {
      unpacked[i] = static_cast<uint8_t>((i * 7 + 1) % 64);
    }

    ONNX_NAMESPACE::TensorProto initializer;
    initializer.set_name("float6");
    initializer.set_data_type(type);
    initializer.add_dims(count);
    ExternalDataInfo::SetExternalLocationToProto(
        utils::kTensorProtoNativeEndianMemoryAddressTag,
        narrow<ExternalDataInfo::OFFSET_TYPE>(reinterpret_cast<intptr_t>(unpacked.data())),
        unpacked.size(), initializer);

    std::vector<uint8_t> external_data;
    ExternalDataWriter writer = [&external_data](int32_t, gsl::span<const uint8_t> bytes, uint64_t& offset) {
      offset = 0;
      external_data.assign(bytes.begin(), bytes.end());
      return Status::OK();
    };
    flatbuffers::FlatBufferBuilder builder;
    flatbuffers::Offset<fbs::Tensor> tensor_offset;
    ASSERT_STATUS_OK(SaveInitializerOrtFormat(builder, initializer, {}, tensor_offset, writer));
    builder.Finish(tensor_offset);

    const auto* tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());
    EXPECT_EQ(tensor->external_data_offset(), 0);
    EXPECT_EQ(external_data.size(), count * 6 / 8);
    EXPECT_EQ(tensor->raw_data(), nullptr);

    ExternalDataReader reader = [&external_data](uint64_t offset, gsl::span<uint8_t> bytes) {
      ORT_ENFORCE(offset == 0 && bytes.size() == external_data.size());
      std::copy(external_data.begin(), external_data.end(), bytes.begin());
      return Status::OK();
    };
    ONNX_NAMESPACE::TensorProto loaded;
    OrtFormatLoadOptions options;
    ASSERT_STATUS_OK(LoadInitializerOrtFormat(*tensor, loaded, options, reader));
    std::vector<uint8_t> decoded;
    ASSERT_STATUS_OK(utils::UnpackInitializerData(loaded, decoded));
    EXPECT_EQ(decoded, unpacked);
  }
}

TEST(FlatbufferUtilsTest, LoadInitializerRejectsReservedTensorDataTypes) {
  for (int32_t value = 21; value <= 26; ++value) {
    flatbuffers::FlatBufferBuilder builder;
    const auto name = builder.CreateString("reserved_type");
    const auto dims = builder.CreateVector(std::vector<int64_t>{1});
    const auto raw_data = builder.CreateVector(std::vector<uint8_t>{0});

    fbs::TensorBuilder tensor_builder(builder);
    tensor_builder.add_name(name);
    tensor_builder.add_dims(dims);
    tensor_builder.add_data_type(static_cast<fbs::TensorDataType>(value));
    tensor_builder.add_raw_data(raw_data);
    builder.Finish(tensor_builder.Finish());

    const auto* tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());
    ONNX_NAMESPACE::TensorProto initializer;
    OrtFormatLoadOptions options;
    ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*tensor, initializer, options),
                                        "Unsupported tensor data type '<unknown>'");
  }
}

TEST(FlatbufferUtilsTest, LoadInitializerRejectsNullStringDataEntry) {
  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("tensor_string");
  auto dims = builder.CreateVector(std::vector<int64_t>{2});
  auto string_data = builder.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{
      builder.CreateString("string_0"),
      builder.CreateString("string_1"),
  });

  fbs::TensorBuilder tensor_builder(builder);
  tensor_builder.add_name(name);
  tensor_builder.add_dims(dims);
  tensor_builder.add_data_type(fbs::TensorDataType::STRING);
  tensor_builder.add_string_data(string_data);
  builder.Finish(tensor_builder.Finish());

  auto* fbs_tensor = flatbuffers::GetMutableRoot<fbs::Tensor>(builder.GetBufferPointer());
  ASSERT_NE(fbs_tensor, nullptr);

  const auto* fbs_string_data = fbs_tensor->string_data();
  ASSERT_NE(fbs_string_data, nullptr);
  auto* mutable_string_data =
      const_cast<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>*>(fbs_string_data);
  auto* string_offsets = mutable_string_data->data();
  const flatbuffers::Offset<flatbuffers::String> null_string_offset;
  std::memcpy(&string_offsets[1], &null_string_offset, sizeof(null_string_offset));

  ONNX_NAMESPACE::TensorProto initializer;
  OrtFormatLoadOptions options;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*fbs_tensor, initializer, options),
                                      "Null string data entry for initializer");
}

TEST(FlatbufferUtilsTest, LoadInitializerRejectsMismatchedStringDataSize) {
  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("tensor_string");
  auto dims = builder.CreateVector(std::vector<int64_t>{1});
  auto string_data = builder.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{
      builder.CreateString("string_0"),
      builder.CreateString("string_1"),
  });

  fbs::TensorBuilder tensor_builder(builder);
  tensor_builder.add_name(name);
  tensor_builder.add_dims(dims);
  tensor_builder.add_data_type(fbs::TensorDataType::STRING);
  tensor_builder.add_string_data(string_data);
  builder.Finish(tensor_builder.Finish());

  const auto* fbs_tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());
  ONNX_NAMESPACE::TensorProto initializer;
  OrtFormatLoadOptions options;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*fbs_tensor, initializer, options),
                                      "Initializer string data size mismatch");
}

#if !defined(DISABLE_SPARSE_TENSORS)
TEST(FlatbufferUtilsTest, LoadSparseInitializerRejectsStringValues) {
  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("sparse_string");
  auto dims = builder.CreateVector(std::vector<int64_t>{0});
  auto string_data = builder.CreateVector(std::vector<flatbuffers::Offset<flatbuffers::String>>{
      builder.CreateString("unexpected"),
  });

  fbs::TensorBuilder values_builder(builder);
  values_builder.add_name(name);
  values_builder.add_dims(dims);
  values_builder.add_data_type(fbs::TensorDataType::STRING);
  values_builder.add_string_data(string_data);
  auto values = values_builder.Finish();

  auto raw_data = builder.CreateVector(std::vector<uint8_t>{});
  fbs::TensorBuilder indices_builder(builder);
  indices_builder.add_dims(dims);
  indices_builder.add_data_type(fbs::TensorDataType::INT64);
  indices_builder.add_raw_data(raw_data);
  auto indices = indices_builder.Finish();

  fbs::SparseTensorBuilder sparse_builder(builder);
  sparse_builder.add_values(values);
  sparse_builder.add_indices(indices);
  sparse_builder.add_dims(dims);
  builder.Finish(sparse_builder.Finish());

  const auto* fbs_sparse_tensor = flatbuffers::GetRoot<fbs::SparseTensor>(builder.GetBufferPointer());
  ONNX_NAMESPACE::SparseTensorProto initializer;
  OrtFormatLoadOptions options;
  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(
      LoadSparseInitializerOrtFormat(*fbs_sparse_tensor, initializer, options),
      "Unsupported sparse tensor data type");
}
#endif  // !defined(DISABLE_SPARSE_TENSORS)

TEST(FlatbufferUtilsTest, LoadInitializerRejectsExternalTensorWithoutDims) {
  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("tensor_external");

  fbs::TensorBuilder tensor_builder(builder);
  tensor_builder.add_name(name);
  tensor_builder.add_data_type(fbs::TensorDataType::FLOAT);
  tensor_builder.add_external_data_offset(0);
  builder.Finish(tensor_builder.Finish());

  const auto* fbs_tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());

  ONNX_NAMESPACE::TensorProto initializer;
  OrtFormatLoadOptions options;
  ExternalDataReader reader = [](uint64_t, gsl::span<uint8_t>) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Reader should not be called for invalid tensor.");
  };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*fbs_tensor, initializer, options, reader),
                                      "Missing dimensions for initializer");
}

TEST(FlatbufferUtilsTest, LoadInitializerRejectsExternalTensorWithNegativeDim) {
  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("tensor_negative_dim");
  auto dims = builder.CreateVector(std::vector<int64_t>{2, -1});

  fbs::TensorBuilder tensor_builder(builder);
  tensor_builder.add_name(name);
  tensor_builder.add_dims(dims);
  tensor_builder.add_data_type(fbs::TensorDataType::FLOAT);
  tensor_builder.add_external_data_offset(0);
  builder.Finish(tensor_builder.Finish());

  const auto* fbs_tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());

  ONNX_NAMESPACE::TensorProto initializer;
  OrtFormatLoadOptions options;
  ExternalDataReader reader = [](uint64_t, gsl::span<uint8_t>) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Reader should not be called for invalid tensor.");
  };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*fbs_tensor, initializer, options, reader),
                                      "Invalid negative dimension -1");
}

TEST(FlatbufferUtilsTest, LoadInitializerRejectsExternalTensorWithOverflowingDims) {
  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("tensor_overflow_dims");
  std::vector<int64_t> overflow_dims;
  if (sizeof(size_t) < sizeof(int64_t)) {
    overflow_dims = {static_cast<int64_t>(std::numeric_limits<size_t>::max()), 2};
  } else {
    overflow_dims = {std::numeric_limits<int64_t>::max(), 3};
  }
  auto dims = builder.CreateVector(overflow_dims);

  fbs::TensorBuilder tensor_builder(builder);
  tensor_builder.add_name(name);
  tensor_builder.add_dims(dims);
  tensor_builder.add_data_type(fbs::TensorDataType::FLOAT);
  tensor_builder.add_external_data_offset(0);
  builder.Finish(tensor_builder.Finish());

  const auto* fbs_tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());

  ONNX_NAMESPACE::TensorProto initializer;
  OrtFormatLoadOptions options;
  ExternalDataReader reader = [](uint64_t, gsl::span<uint8_t>) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Reader should not be called for invalid tensor.");
  };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*fbs_tensor, initializer, options, reader),
                                      "overflows size_t");
}

TEST(FlatbufferUtilsTest, LoadInitializerRejectsExternalTensorWithDimTooLargeForSizeT) {
  if (sizeof(size_t) >= sizeof(int64_t)) {
    GTEST_SKIP() << "This platform does not have a narrower size_t than int64_t.";
  }

  flatbuffers::FlatBufferBuilder builder(256);

  auto name = builder.CreateString("tensor_dim_too_large_for_size_t");
  auto dims = builder.CreateVector(std::vector<int64_t>{std::numeric_limits<int64_t>::max()});

  fbs::TensorBuilder tensor_builder(builder);
  tensor_builder.add_name(name);
  tensor_builder.add_dims(dims);
  tensor_builder.add_data_type(fbs::TensorDataType::FLOAT);
  tensor_builder.add_external_data_offset(0);
  builder.Finish(tensor_builder.Finish());

  const auto* fbs_tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());

  ONNX_NAMESPACE::TensorProto initializer;
  OrtFormatLoadOptions options;
  ExternalDataReader reader = [](uint64_t, gsl::span<uint8_t>) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Reader should not be called for invalid tensor.");
  };

  ASSERT_STATUS_NOT_OK_AND_HAS_SUBSTR(LoadInitializerOrtFormat(*fbs_tensor, initializer, options, reader),
                                      "does not fit in size_t");
}

#ifdef ENABLE_TRAINING_APIS
template <typename F6>
void TestFloat6OrtTensorRoundTrip(bool use_external_data, int64_t count) {
  SCOPED_TRACE(::testing::Message() << "external=" << use_external_data << " count=" << count);
  static onnxruntime::CPUExecutionProviderInfo info;
  static onnxruntime::CPUExecutionProvider cpu_provider(info);
  AllocatorPtr cpu_allocator = cpu_provider.CreatePreferredAllocators()[0];

  Tensor input{DataTypeImpl::GetType<F6>(), TensorShape({count}), cpu_allocator};
  auto* input_data = input.MutableData<F6>();
  for (int64_t i = 0; i < count; ++i) {
    input_data[i] = F6(static_cast<uint8_t>((i * 7 + 1) % 64), F6::FromBits());
  }

  std::vector<uint8_t> external_data;
  ExternalDataWriter writer;
  if (use_external_data) {
    writer = [&external_data](int32_t, gsl::span<const uint8_t> bytes, uint64_t& offset) {
      offset = 0;
      external_data.assign(bytes.begin(), bytes.end());
      return Status::OK();
    };
  }

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<fbs::Tensor> tensor_offset;
  ASSERT_STATUS_OK(SaveOrtTensorOrtFormat("float6", input, builder, tensor_offset, writer));
  builder.Finish(tensor_offset);

  const auto* fbs_tensor = flatbuffers::GetRoot<fbs::Tensor>(builder.GetBufferPointer());
  const size_t packed_size = static_cast<size_t>((count * 6 + 7) / 8);
  if (use_external_data) {
    ASSERT_EQ(fbs_tensor->external_data_offset(), 0);
    ASSERT_EQ(fbs_tensor->raw_data(), nullptr);
    ASSERT_EQ(external_data.size(), packed_size);
  } else {
    ASSERT_NE(fbs_tensor->raw_data(), nullptr);
    ASSERT_EQ(fbs_tensor->external_data_offset(), -1);
    ASSERT_EQ(fbs_tensor->raw_data()->size(), packed_size);
    ASSERT_TRUE(external_data.empty());
  }

  ExternalDataReader reader = [&external_data](uint64_t offset, gsl::span<uint8_t> bytes) {
    ORT_ENFORCE(offset == 0 && bytes.size() == external_data.size());
    std::copy(external_data.begin(), external_data.end(), bytes.begin());
    return Status::OK();
  };
  Tensor output;
  std::string tensor_name;
  ASSERT_STATUS_OK(LoadOrtTensorOrtFormat(*fbs_tensor, cpu_allocator, tensor_name, output,
                                          use_external_data ? reader : ExternalDataReader{}));
  ASSERT_EQ(tensor_name, "float6");
  ASSERT_EQ(output.Shape().NumDimensions(), 1);
  ASSERT_EQ(output.Shape()[0], count);
  const auto* output_data = output.Data<F6>();
  for (int64_t i = 0; i < count; ++i) {
    EXPECT_EQ(output_data[i].ToBits(), static_cast<uint8_t>((i * 7 + 1) % 64)) << "element " << i;
  }
}

TEST(FlatbufferUtilsTest, Float6E2M3OrtTensorInlineRoundTrip) {
  for (int64_t count : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4},
                        int64_t{5}, int64_t{6}, int64_t{7}}) {
    TestFloat6OrtTensorRoundTrip<Float6E2M3>(false, count);
  }
}

TEST(FlatbufferUtilsTest, Float6E3M2OrtTensorInlineRoundTrip) {
  for (int64_t count : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{4},
                        int64_t{5}, int64_t{6}, int64_t{7}}) {
    TestFloat6OrtTensorRoundTrip<Float6E3M2>(false, count);
  }
}

TEST(FlatbufferUtilsTest, Float6E2M3OrtTensorExternalRoundTrip) {
  for (int64_t count : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                        int64_t{6}, int64_t{7}, int64_t{89}, int64_t{90}, int64_t{91}}) {
    TestFloat6OrtTensorRoundTrip<Float6E2M3>(true, count);
  }
}

TEST(FlatbufferUtilsTest, Float6E3M2OrtTensorExternalRoundTrip) {
  for (int64_t count : {int64_t{1}, int64_t{2}, int64_t{3}, int64_t{5},
                        int64_t{6}, int64_t{7}, int64_t{89}, int64_t{90}, int64_t{91}}) {
    TestFloat6OrtTensorRoundTrip<Float6E3M2>(true, count);
  }
}

// tests method that loads to OrtTensor (used when loading a checkpoint into a checkpoint state)
TEST(FlatbufferUtilsTest, ExternalWriteReadWithLoadOrtTensor) {
  // create data
  auto initializers = CreateInitializersNoString();

  flatbuffers::FlatBufferBuilder builder(1024);

  // write
  std::ofstream output_stream;
  ExternalDataWriter writer;
  CreateWriter("ExternalWriteReadBasicTest.bin", output_stream, writer);

  std::vector<flatbuffers::Offset<fbs::Tensor>> fbs_tensors;
  for (const auto& initializer : initializers) {
    flatbuffers::Offset<fbs::Tensor> fbs_tensor;
    ASSERT_STATUS_OK(SaveInitializerOrtFormat(builder, initializer, std::filesystem::path(), fbs_tensor, writer));
    fbs_tensors.push_back(fbs_tensor);
  }

  ASSERT_EQ(output_stream.tellp(), 840) << "Data written to the external file is incorrect.";
  output_stream.close();
  ASSERT_TRUE(output_stream.good()) << "Failed to close data file.";

  auto fbs_tensors_offset = builder.CreateVector(fbs_tensors);
  fbs::test::TestDataBuilder tdb(builder);
  tdb.add_initializers(fbs_tensors_offset);
  builder.Finish(tdb.Finish());
  auto fb_data = builder.GetBufferSpan();

  auto test_data = fbs::test::GetTestData(fb_data.data());
  auto fbs_tensors2 = test_data->initializers();

  // read
  std::ifstream input_stream;
  ExternalDataReader reader;
  CreateReader("ExternalWriteReadBasicTest.bin", input_stream, reader);
  static onnxruntime::CPUExecutionProviderInfo info;
  static onnxruntime::CPUExecutionProvider cpu_provider(info);
  AllocatorPtr cpu_allocator = cpu_provider.CreatePreferredAllocators()[0];

  std::vector<Tensor> loaded_tensors;

  for (const auto* fbs_tensor : *fbs_tensors2) {
    Tensor ort_tensor;
    std::string fbs_tensor_name = fbs_tensor->name()->str();
    ASSERT_STATUS_OK(LoadOrtTensorOrtFormat(*fbs_tensor, cpu_allocator, fbs_tensor_name, ort_tensor, reader));
    loaded_tensors.push_back(std::move(ort_tensor));
  }

  ASSERT_EQ(initializers.size(), loaded_tensors.size());

  // convert expected initializers (TensorProtos) to Tensors for easier comparison
  std::vector<Tensor> expected_tensors;
  const Env& env = Env::Default();
  const auto* placeholder_model_path = ORT_TSTR("placeholder_model_path");

  for (int i = 0; i < narrow<int>(initializers.size()); i++) {
    auto expected_proto = initializers[i];
    TensorShape tensor_shape = utils::GetTensorShapeFromTensorProto(expected_proto);
    const auto* type = DataTypeImpl::TensorTypeFromONNXEnum(expected_proto.data_type())->GetElementType();
    Tensor expected_tensor(type, tensor_shape, cpu_allocator);
    ASSERT_STATUS_OK(utils::TensorProtoToTensor(env, placeholder_model_path, initializers[i], expected_tensor));
    expected_tensors.push_back(std::move(expected_tensor));
  }

  // validate data
  for (int i = 0; i < narrow<int>(expected_tensors.size()); i++) {
    auto& expected_tensor = expected_tensors[i];
    auto& loaded_tensor = loaded_tensors[i];
    ASSERT_EQ(expected_tensor.DataType(), loaded_tensor.DataType());
    ASSERT_EQ(expected_tensor.Shape(), loaded_tensor.Shape());
    ASSERT_EQ(expected_tensor.SizeInBytes(), loaded_tensor.SizeInBytes());
    gsl::span<const uint8_t> expected_data(static_cast<const uint8_t*>(expected_tensor.DataRaw()),
                                           expected_tensor.SizeInBytes());
    gsl::span<const uint8_t> loaded_data(static_cast<const uint8_t*>(loaded_tensor.DataRaw()),
                                         loaded_tensor.SizeInBytes());

    ASSERT_EQ(expected_data, loaded_data);
  }
}
#endif  // ENABLE_TRAINING_APIS
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD

// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_bin_manager.h"
#include "ov_shared_context.h"
#include <limits>
#include <nlohmann/json.hpp>
#include "core/providers/shared_library/provider_api.h"  // for ORT_VERSION and kOpenVINOExecutionProvider

namespace onnxruntime {
namespace openvino_ep {

static inline uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Custom streambuf that wraps an ov::Tensor's memory
// Provides us a std::istream interface over the tensor data without copying.
// Only supports input operations.
class TensorStreamBuf : public std::streambuf {
 public:
  explicit TensorStreamBuf(const ov::Tensor& tensor) {
    // Suppress warning for tensor.data() returning const in 2026.0. Should be removable after 2026.0 is min supported ov version.
    OPENVINO_SUPPRESS_DEPRECATED_START
    char* data = const_cast<char*>(tensor.data<const char>());  // setg requires non-const char* but we won't modify data
    OPENVINO_SUPPRESS_DEPRECATED_END
    size_t size = tensor.get_byte_size();
    setg(data, data, data + size);
  }

 protected:
  // Override seekoff for proper seeking support
  std::streampos seekoff(std::streamoff off, std::ios_base::seekdir dir, std::ios_base::openmode which) override {
    if (which & std::ios_base::in) {
      char* new_pos = nullptr;
      switch (dir) {
        case std::ios_base::beg:
          new_pos = eback() + off;
          break;
        case std::ios_base::cur:
          new_pos = gptr() + off;
          break;
        case std::ios_base::end:
          new_pos = egptr() + off;
          break;
        default:
          return std::streampos(std::streamoff(-1));
      }

      if (new_pos >= eback() && new_pos <= egptr()) {
        setg(eback(), new_pos, egptr());
        return std::streampos(new_pos - eback());
      }
    }
    return std::streampos(std::streamoff(-1));
  }

  // Override seekpos for proper seeking support
  std::streampos seekpos(std::streampos pos, std::ios_base::openmode which) override {
    return seekoff(std::streamoff(pos), std::ios_base::beg, which);
  }
};

// Custom istream that owns the tensor to ensure proper lifetime management
class TensorStream : public std::istream {
 public:
  explicit TensorStream(ov::Tensor tensor)
      : std::istream(&buf_),
        tensor_(std::move(tensor)),
        buf_(tensor_) {}

 private:
  const ov::Tensor tensor_;  // Keep tensor alive
  TensorStreamBuf buf_;      // Buffer wrapping tensor data
};

/*
    Logical layout of the single binary file:
    [Header]
    [BSON Metadata]              ← Contains blob_metadata_map with data_offset and size for each blob
    [Padding to 64K alignment]   ← Blob section starts here (64K aligned)
    [Blob 1]                     ← BSON blob_metadata_map["blob_name"].data_offset points here
    [Padding to 64K alignment]   ← Each blob end is 64K aligned
    [Blob 2]                     ← BSON blob_metadata_map["blob_name2"].data_offset points here
    [Padding to 64K alignment]
    [Blob 3]                     ← BSON blob_metadata_map["blob_name3"].data_offset points here
    ...

    BSON Schema:
    {
      "version": <string>,                    // BSON schema version (semver format)
      "producer": <string>,                   // Producer identifier (e.g., "onnxruntime-openvino-ep-plugin")
      "weights_metadata_map": {               // Map of ONNX tensor names to external weight file metadata
        "<tensor_name>": {
          "location": <string>,               // Relative path to external weights file
          "data_offset": <int64>,            // Offset within external weights file
          "size": <int64>                    // Size of weight data in bytes
        },
        ...
      },
      "blob_metadata_map": {                  // Map of blob names to compiled model blob metadata
        "<blob_name>": {
          "data_offset": <int64>,            // Absolute file offset to blob data (64K aligned)
          "size": <int64>                    // Actual blob data size (excluding padding)
        },
        ...
      }
    }

    Note: data_offset values in blob_metadata_map are absolute file positions.
          size values exclude alignment padding bytes.
*/

// "OVEP_BIN" in little-endian (memory will read as 'O','V','E','P','_','B','I','N')
constexpr uint64_t kMagicNumber = 0x4E49425F5045564FULL;

enum class BinVersion : uint64_t {
  v1 = 1,
  current = v1
};

struct header_t {
  uint64_t magic;
  uint64_t version;
  uint64_t header_size;
  uint64_t bson_start_offset;
  uint64_t bson_size;
};

constexpr uint64_t kBlobAlignment = 64 * 1024;

// BSON field names
namespace BSONFields {
constexpr const char* kVersion = "version";
constexpr const char* kProducer = "producer";
constexpr const char* kWeightsMetadata = "weights_metadata_map";
constexpr const char* kBlobMetadata = "blob_metadata_map";
constexpr const char* kLocation = "location";
constexpr const char* kDataOffset = "data_offset";
constexpr const char* kSize = "size";
constexpr const char* kCurrentBsonVersion = "1.0.0";
constexpr const char* kProducerName = "onnxruntime-openvino-ep-" ORT_VERSION;
}  // namespace BSONFields

template <typename E>
constexpr std::underlying_type_t<E> to_underlying(E e) noexcept {
  static_assert(std::is_enum_v<E>, "to_underlying requires an enum type");
  return static_cast<std::underlying_type_t<E>>(e);
}

void BinManager::AddNativeBlob(const std::string& name, const ov::CompiledModel& compiled_model) {
  std::unique_lock lock(mutex_);
  native_blobs_[name] = BlobContainer{.compiled_model = compiled_model, .tensor = {}, .data = {}, .serialized_info = {0, 0}};
}

ov::Tensor BinManager::GetNativeBlob(const std::string& blob_name) {
  std::unique_lock lock(mutex_);

  auto it = native_blobs_.find(blob_name);
  ORT_ENFORCE(it != native_blobs_.end(), "Blob not found for ", blob_name);

  auto& blob_container = it->second;
  if (blob_container.tensor) {
    return blob_container.tensor;
  }

  ORT_ENFORCE(blob_container.serialized_info.size > 0 || !blob_container.data.empty(),
              "Blob has no serialization info or embedded data for ", blob_name);

  if (!external_bin_path_.value_or("").empty() && !mapped_bin_) {
    // Use ov::read_tensor_data to create a memory-mapped tensor from external file
    mapped_bin_ = ov::read_tensor_data(external_bin_path_.value());
  }

  if (mapped_bin_) {
    auto blob_offset = blob_container.serialized_info.file_offset;
    auto blob_size = blob_container.serialized_info.size;
    auto bin_size = mapped_bin_.get_byte_size();
    ORT_ENFORCE(blob_offset < bin_size && blob_size <= bin_size && blob_offset <= bin_size - blob_size,
                "Blob offset+size out of bounds for ", blob_name,
                ". Offset: ", blob_offset, " Size: ", blob_size, " File size: ", bin_size);
    const uint8_t* src = mapped_bin_.data<const uint8_t>() + blob_offset;

    blob_container.tensor = ov::Tensor(
        ov::element::u8,
        ov::Shape{blob_size},
        const_cast<uint8_t*>(src));
  } else {
    // Create a tensor from embedded data vector
    blob_container.tensor = ov::Tensor(
        ov::element::u8,
        ov::Shape{blob_container.data.size()},
        blob_container.data.data());
  }

  return blob_container.tensor;
}

std::unique_ptr<std::istream> BinManager::GetNativeBlobAsStream(const std::string& blob_name) {
  return std::make_unique<TensorStream>(GetNativeBlob(blob_name));
}

std::filesystem::path BinManager::GetBinPathForModel(const std::filesystem::path& model_path) {
  ORT_ENFORCE(!model_path.empty());
  return model_path.parent_path() / (model_path.stem().string() + "_" + kOpenVINOExecutionProvider + ".bin");
}

void BinManager::Serialize(std::shared_ptr<SharedContext> shared_context) {
  auto path = GetExternalBinPath();
  std::ofstream stream(path, std::ios::out | std::ios::binary);
  ORT_ENFORCE(stream.is_open(), "Failed to open file for serialization: " + path.string());
  Serialize(stream, shared_context);
}

void BinManager::Deserialize(std::shared_ptr<SharedContext> shared_context) {
  auto path = GetExternalBinPath();
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  ORT_ENFORCE(stream.is_open(), "Failed to open file for deserialization: " + path.string());
  Deserialize(stream, shared_context);
}

void BinManager::Serialize(std::ostream& stream, std::shared_ptr<SharedContext> shared_context) {
  std::shared_lock ul(mutex_);

  auto metadata = shared_context ? shared_context->GetMetadataCopy() : SharedContext::Metadata::Map{};
  if (metadata.empty() && native_blobs_.empty()) {
    return;  // Nothing to serialize
  }

  const auto stream_start = stream.tellp();

  auto write_alignment_padding = [&stream](uint64_t current_pos, uint64_t alignment) {
    uint64_t aligned_position = AlignUp(current_pos, alignment);
    uint64_t padding_size = aligned_position - current_pos;
    if (padding_size > 0) {
      std::vector<char> padding(padding_size, 0);
      stream.write(padding.data(), padding.size());
      ORT_ENFORCE(stream.good(), "Error: Failed to write alignment padding.");
    }
  };

  // Reserve space for header (will be updated later)
  header_t header{};
  header.magic = kMagicNumber;
  header.version = to_underlying(BinVersion::current);
  header.header_size = sizeof(header_t);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  ORT_ENFORCE(stream.good(), "Error: Failed to write header.");

  // Build JSON metadata
  nlohmann::json j;
  j[BSONFields::kVersion] = BSONFields::kCurrentBsonVersion;
  j[BSONFields::kProducer] = BSONFields::kProducerName;

  // Add weights metadata as a map (from SharedContext if available)
  if (!metadata.empty()) {
    nlohmann::json weights_map = nlohmann::json::object();
    for (const auto& [key, value] : metadata) {
      nlohmann::json weight_entry;
      weight_entry[BSONFields::kLocation] = value.serialized.location.string();
      weight_entry[BSONFields::kDataOffset] = value.serialized.data_offset;
      weight_entry[BSONFields::kSize] = value.serialized.size;
      weights_map[key] = weight_entry;
    }
    j[BSONFields::kWeightsMetadata] = weights_map;
  }

  // Add blob metadata with placeholder values as a map (will be updated after writing blobs)
  nlohmann::json blob_map = nlohmann::json::object();
  for (const auto& [key, value] : native_blobs_) {
    nlohmann::json blob_entry;
    auto max_val = std::numeric_limits<int64_t>::max();
    // Placehold max size since we don't know actual offsets/sizes yet, and if they aren't max they might serialize smaller.
    blob_entry[BSONFields::kDataOffset] = max_val;
    blob_entry[BSONFields::kSize] = max_val;
    blob_map[key] = blob_entry;
  }
  j[BSONFields::kBlobMetadata] = blob_map;

  // Write BSON metadata (will be rewritten later with correct blob info)
  header.bson_start_offset = stream.tellp();

  size_t orig_bson_size;
  {
    std::vector<uint8_t> bson_data = nlohmann::json::to_bson(j);
    orig_bson_size = bson_data.size();
    stream.write(reinterpret_cast<const char*>(bson_data.data()), bson_data.size());
    ORT_ENFORCE(stream.good(), "Error: Failed to write BSON data.");
  }
  uint64_t bson_end = stream.tellp();

  write_alignment_padding(bson_end, kBlobAlignment);

  // Write blob data and capture actual offsets/sizes
  for (auto& [blob_name, value] : native_blobs_) {
    uint64_t blob_start = stream.tellp();
    value.compiled_model.export_model(stream);
    ORT_ENFORCE(stream.good(), "Error: Failed to write blob data for ", blob_name);
    // Seek to end of stream after writing in case export model didn't leave us there
    stream.seekp(0, std::ios::end);
    uint64_t blob_end = stream.tellp();
    uint64_t blob_size = blob_end - blob_start;

    // Update the BlobContainer + BSON with serialization info
    value.serialized_info.file_offset = blob_start;
    value.serialized_info.size = blob_size;
    j[BSONFields::kBlobMetadata][blob_name][BSONFields::kDataOffset] = blob_start;
    j[BSONFields::kBlobMetadata][blob_name][BSONFields::kSize] = blob_size;

    write_alignment_padding(blob_end, kBlobAlignment);
  }

  // Rewrite BSON metadata with correct blob info
  std::vector<uint8_t> updated_bson_data = nlohmann::json::to_bson(j);
  ORT_ENFORCE(updated_bson_data.size() <= orig_bson_size,
              "Error: BSON size larger after updating blob info. Original: ", orig_bson_size,
              " Updated: ", updated_bson_data.size());

  stream.seekp(header.bson_start_offset);
  stream.write(reinterpret_cast<const char*>(updated_bson_data.data()), updated_bson_data.size());
  ORT_ENFORCE(stream.good(), "Error: Failed to rewrite BSON data.");
  bson_end = stream.tellp();
  header.bson_size = bson_end - header.bson_start_offset;

  // Update header with BSON offsets
  stream.seekp(stream_start);
  stream.write(reinterpret_cast<const char*>(&header), sizeof(header));
  ORT_ENFORCE(stream.good(), "Error: Failed to update header.");

  stream.seekp(0, std::ios::end);  // Move to end after writing.
}

void BinManager::Deserialize(std::istream& stream, std::shared_ptr<SharedContext> shared_context) {
  try {
    DeserializeImpl(stream, shared_context);
  } catch (const std::exception& e) {
    ORT_THROW(e.what(), "\nCould not deserialize binary data. This could mean the bin is corrupted or incompatible. Try re-generating ep context cache.");
  }
}

void BinManager::DeserializeImpl(std::istream& stream, const std::shared_ptr<SharedContext>& shared_context) {
  const auto stream_start = stream.tellg();
  ORT_ENFORCE(stream_start >= std::streampos{0}, "Error: Failed to determine binary stream position.");
  stream.seekg(0, std::ios::end);
  const auto stream_end = stream.tellg();
  ORT_ENFORCE(stream.good() && stream_end >= stream_start, "Error: Failed to determine binary stream size.");
  const auto stream_size = static_cast<uint64_t>(static_cast<std::streamoff>(stream_end));
  stream.seekg(stream_start);

  const auto validate_range = [stream_size](uint64_t offset, uint64_t size) {
    ORT_ENFORCE(offset <= stream_size && size <= stream_size - offset,
                "Error: Binary data range exceeds stream size.");
    ORT_ENFORCE(size <= std::numeric_limits<size_t>::max() &&
                    size <= static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()),
                "Error: Binary data size exceeds allocation or stream limits.");
  };
  validate_range(static_cast<uint64_t>(static_cast<std::streamoff>(stream_start)), sizeof(header_t));

  // Read and validate header
  header_t header{};

  stream.read(reinterpret_cast<char*>(&header), sizeof(header));
  ORT_ENFORCE(stream.good(), "Error: Failed to read header.");
  ORT_ENFORCE(header.magic == kMagicNumber, "Error: Invalid magic number. Expected: 0x", std::hex, kMagicNumber, " Got: 0x", header.magic);
  ORT_ENFORCE(header.version == to_underlying(BinVersion::current), "Error: Unsupported file version: ", header.version);
  ORT_ENFORCE(header.header_size == sizeof(header_t), "Error: Header size mismatch.");
  validate_range(header.bson_start_offset, header.bson_size);

  // Seek to BSON metadata and read it
  stream.seekg(static_cast<std::streamoff>(header.bson_start_offset));
  ORT_ENFORCE(stream.good(), "Error: Failed to seek to BSON metadata.");

  // Parse BSON
  nlohmann::json j;
  {
    std::vector<uint8_t> bson_data(static_cast<size_t>(header.bson_size));
    stream.read(reinterpret_cast<char*>(bson_data.data()), static_cast<std::streamsize>(header.bson_size));
    ORT_ENFORCE(stream.good(), "Error: Failed to read BSON metadata.");
    j = nlohmann::json::from_bson(bson_data);
  }

  // Validate BSON version (check major version compatibility)
  ORT_ENFORCE(j.contains(BSONFields::kVersion), "Error: Missing version in BSON metadata.");
  auto bson_version = j[BSONFields::kVersion].get<std::string>();

  // Extract major version from semver strings (format: "major.minor.patch")
  auto get_major_version = [](const std::string& version) -> int {
    size_t dot_pos = version.find('.');
    if (dot_pos == std::string::npos) return -1;
    try {
      return std::stoi(version.substr(0, dot_pos));
    } catch (...) {
      return -1;
    }
  };

  int file_major = get_major_version(bson_version);
  int current_major = get_major_version(BSONFields::kCurrentBsonVersion);

  ORT_ENFORCE(file_major >= 0 && current_major >= 0,
              "Error: Invalid BSON version format. Expected: ", BSONFields::kCurrentBsonVersion,
              " Got: ", bson_version);
  ORT_ENFORCE(file_major == current_major,
              "Error: Incompatible BSON schema major version. Expected: ", current_major,
              " Got: ", file_major, " (full version: ", bson_version, ")");

  // Parse weights metadata and populate SharedContext if available
  if (j.contains(BSONFields::kWeightsMetadata)) {
    ORT_ENFORCE(shared_context, "Error: Bin contains shared weights metadata but no SharedContext was provided during deserialization.");
    const auto& weights_map = j[BSONFields::kWeightsMetadata];
    if (weights_map.is_object()) {
      for (const auto& [weight_name, weight_entry] : weights_map.items()) {
        auto location = weight_entry[BSONFields::kLocation].get<std::string>();
        auto data_offset = weight_entry[BSONFields::kDataOffset].get<size_t>();
        auto size = weight_entry[BSONFields::kSize].get<size_t>();
        shared_context->AddExternalWeight(weight_name, data_offset, size, location);
      }
    }
  }

  // Parse blob metadata
  ORT_ENFORCE(j.contains(BSONFields::kBlobMetadata), "Error: Missing blob metadata in BSON.");
  const auto& blob_map = j[BSONFields::kBlobMetadata];
  ORT_ENFORCE(blob_map.is_object(), "Error: Blob metadata must be an object.");

  const auto read_size = [](const nlohmann::json& entry, const char* field) {
    const auto& value = entry.at(field);
    ORT_ENFORCE(value.is_number_integer() &&
                    (value.is_number_unsigned() || value.get<int64_t>() >= 0),
                "Error: Binary metadata offsets and sizes must be non-negative integers.");
    return value.get<uint64_t>();
  };
  // Validate every blob before any embedded blob allocation, including later map entries.
  uint64_t remaining_blob_bytes = stream_size;
  for (const auto& blob_entry : blob_map) {
    const auto blob_size = read_size(blob_entry, BSONFields::kSize);
    validate_range(read_size(blob_entry, BSONFields::kDataOffset), blob_size);
    ORT_ENFORCE(blob_size <= remaining_blob_bytes, "Error: Total blob data size exceeds stream size.");
    remaining_blob_bytes -= blob_size;
  }

  // Determine if we're deserializing from an external file or embedded stream
  const bool has_external_file = !external_bin_path_.value_or("").empty();

  std::unique_lock lock(mutex_);
  for (const auto& [blob_name, blob_entry] : blob_map.items()) {
    uint64_t blob_offset = read_size(blob_entry, BSONFields::kDataOffset);
    uint64_t blob_size = read_size(blob_entry, BSONFields::kSize);

    BlobContainer container;
    container.serialized_info.file_offset = blob_offset;
    container.serialized_info.size = blob_size;

    // If no external file, extract blob data into vector
    if (!has_external_file) {
      // Seek to blob offset and read data into vector
      auto current_pos = stream.tellg();
      stream.seekg(static_cast<std::streamoff>(blob_offset));
      ORT_ENFORCE(stream.good(), "Error: Failed to seek to blob data for ", blob_name);

      container.data.resize(static_cast<size_t>(blob_size));
      stream.read(reinterpret_cast<char*>(container.data.data()), static_cast<std::streamsize>(blob_size));
      ORT_ENFORCE(stream.good(), "Error: Failed to read blob data for ", blob_name);

      // Restore stream position
      stream.seekg(current_pos);
    }

    native_blobs_[blob_name] = std::move(container);
  }
}

}  // namespace openvino_ep
}  // namespace onnxruntime

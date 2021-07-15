// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// needs to be included first to get around onnxruntime\cmake\external\onnx\onnx/common/constants.h(14): error C2513: 'bool': no variable declared before '='

#include "TestCase.h"

#include "callback.h"
#include "heap_buffer.h"
#include "mem_buffer.h"
#include "onnx_model_info.h"
#include "pb_helper.h"
#include "tensorprotoutils.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/platform/ort_mutex.h"
#include "core/platform/path_lib.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/framework/allocator.h"
#include "core/framework/TensorSeq.h"
#include "re2/re2.h"

#include <cctype>
#include <fstream>
#include <memory>
#include <sstream>
#include <map>
#include <regex>

using namespace onnxruntime;
using namespace onnxruntime::common;
using google::protobuf::RepeatedPtrField;

static constexpr int protobuf_block_size_in_bytes = 4 * 1024 * 1024;

const std::string TestModelInfo::unknown_version = "unknown version";

namespace {

template <typename T>
inline Ort::Value CreateTensorWithDataAsOrtValue(const Ort::MemoryInfo& info,
                                                 OrtAllocator*,
                                                 const std::vector<int64_t>& dims,
                                                 std::vector<T>& input) {
  return Ort::Value::CreateTensor<T>(static_cast<const OrtMemoryInfo*>(info), input.data(), input.size() * sizeof(T),
                                     dims.data(), dims.size());
}

template <>
inline Ort::Value CreateTensorWithDataAsOrtValue(const Ort::MemoryInfo&,
                                                 OrtAllocator* allocator,
                                                 const std::vector<int64_t>& dims,
                                                 std::vector<std::string>& input) {
  auto tensor_value = Ort::Value::CreateTensor(allocator, dims.data(), dims.size(),
                                               ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

  std::vector<const char*> p_str;
  for (const auto& s : input) {
    p_str.push_back(s.c_str());
  }

  tensor_value.FillStringTensor(p_str.data(), p_str.size());
  return tensor_value;
}

template <typename key_type, typename value_type>
Ort::Value PbMapToOrtValue(const google::protobuf::Map<key_type, value_type>& map) {
  auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t ele_count = map.size();
  std::vector<int64_t> dims(1, static_cast<int64_t>(ele_count));
  std::vector<key_type> keys(ele_count);
  std::vector<value_type> values(ele_count);
  size_t i = 0;
  for (auto& kvp : map) {
    keys[i] = kvp.first;
    values[i] = kvp.second;
    ++i;
  }

  //// See helper above
  auto ort_keys = CreateTensorWithDataAsOrtValue(info, allocator, dims, keys);
  auto ort_values = CreateTensorWithDataAsOrtValue(info, allocator, dims, values);
  return Ort::Value::CreateMap(ort_keys, ort_values);
}

template <typename T>
Ort::Value VectorProtoToOrtValue(const RepeatedPtrField<T>& input) {
  auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::Value> seq;
  seq.reserve(input.size());
  for (const T& v : input) {
    // create key tensor
    const auto& map = v.v();
    size_t ele_count = map.size();
    using key_type = typename std::remove_reference<decltype(v.v())>::type::key_type;
    using value_type = typename std::remove_reference<decltype(v.v())>::type::mapped_type;
    std::vector<int64_t> dims(1, static_cast<int64_t>(ele_count));
    std::vector<key_type> keys(ele_count);
    std::vector<value_type> values(ele_count);
    size_t i = 0;
    for (auto& kvp : map) {
      keys[i] = kvp.first;
      values[i] = kvp.second;
      ++i;
    }

    auto ort_keys = CreateTensorWithDataAsOrtValue(info, allocator, dims, keys);
    auto ort_values = CreateTensorWithDataAsOrtValue(info, allocator, dims, values);
    auto ort_map = Ort::Value::CreateMap(ort_keys, ort_values);
    seq.push_back(std::move(ort_map));
  }
  return Ort::Value::CreateSequence(seq);
}

template <typename CHAR_T>
static int ExtractFileNo(const std::basic_string<CHAR_T>& name) {
  size_t p1 = name.rfind('.');
  size_t p2 = name.rfind('_', p1);
  ++p2;
  std::basic_string<CHAR_T> number_str = name.substr(p2, p1 - p2);
  const CHAR_T* start = number_str.c_str();
  const CHAR_T* end = number_str.c_str();
  long ret = OrtStrtol(start, const_cast<CHAR_T**>(&end));
  if (end == start) {
    ORT_THROW("parse file name failed");
  }
  return static_cast<int>(ret);
}
using PATH_STRING_TYPE = std::basic_string<PATH_CHAR_TYPE>;

static void SortFileNames(std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_pb_files) {
  if (input_pb_files.size() <= 1) return;
  std::sort(input_pb_files.begin(), input_pb_files.end(),
            [](const std::basic_string<PATH_CHAR_TYPE>& left, const std::basic_string<PATH_CHAR_TYPE>& right) -> bool {
              std::basic_string<PATH_CHAR_TYPE> leftname = GetLastComponent(left);
              std::basic_string<PATH_CHAR_TYPE> rightname = GetLastComponent(right);
              int left1 = ExtractFileNo(leftname);
              int right1 = ExtractFileNo(rightname);
              return left1 < right1;
            });

  for (size_t i = 0; i != input_pb_files.size(); ++i) {
    int fileno = ExtractFileNo(GetLastComponent(input_pb_files[i]));
    if (static_cast<size_t>(fileno) != i) {
      std::basic_ostringstream<PATH_CHAR_TYPE> oss;
      oss << input_pb_files[0];
      for (size_t j = 1; j != input_pb_files.size(); ++j)
        oss << ORT_TSTR(" ") << input_pb_files[j];
      ORT_THROW("illegal input file name:", ToMBString(oss.str()));
    }
  }
}

Ort::Value TensorToOrtValue(const ONNX_NAMESPACE::TensorProto& t, onnxruntime::test::HeapBuffer& b) {
  size_t len = 0;
  auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(t, &len);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  Ort::Value temp_value{nullptr};
  onnxruntime::test::OrtCallback d;
  auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  status = onnxruntime::test::TensorProtoToMLValue(t, onnxruntime::test::MemBuffer(p, len, *static_cast<OrtMemoryInfo*>(cpu_memory_info)),
                                                   temp_value, d);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  if (d.f) {
    b.AddDeleter(d);
  }
  return temp_value;
}

void LoopDataFile(int test_data_pb_fd, bool is_input, const TestModelInfo& modelinfo,
                  std::unordered_map<std::string, Ort::Value>& name_data_map, onnxruntime::test::HeapBuffer& b,
                  std::ostringstream& oss) {
  google::protobuf::io::FileInputStream f(test_data_pb_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  google::protobuf::io::CodedInputStream coded_input(&f);
  bool clean_eof = false;
  int item_id = 1;
  for (proto::TraditionalMLData data;
       ParseDelimitedFromCodedStream(&data, &coded_input, &clean_eof);
       ++item_id, data.Clear()) {
    ORT_TRY {
      Ort::Value gvalue{nullptr};
      switch (data.values_case()) {
        case proto::TraditionalMLData::kVectorMapStringToFloat:
          gvalue = VectorProtoToOrtValue(data.vector_map_string_to_float().v());
          break;
        case proto::TraditionalMLData::kVectorMapInt64ToFloat:
          gvalue = VectorProtoToOrtValue(data.vector_map_int64_to_float().v());
          break;
        case proto::TraditionalMLData::kMapStringToString:
          gvalue = PbMapToOrtValue(data.map_string_to_string().v());
          break;
        case proto::TraditionalMLData::kMapStringToInt64:
          gvalue = PbMapToOrtValue(data.map_string_to_int64().v());
          break;
        case proto::TraditionalMLData::kMapStringToFloat:
          gvalue = PbMapToOrtValue(data.map_string_to_float().v());
          break;
        case proto::TraditionalMLData::kMapStringToDouble:
          gvalue = PbMapToOrtValue(data.map_string_to_double().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToString:
          gvalue = PbMapToOrtValue(data.map_int64_to_string().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToInt64:
          gvalue = PbMapToOrtValue(data.map_int64_to_int64().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToFloat:
          gvalue = PbMapToOrtValue(data.map_int64_to_float().v());
          break;
        case proto::TraditionalMLData::kMapInt64ToDouble:
          gvalue = PbMapToOrtValue(data.map_int64_to_double().v());
          break;
        case proto::TraditionalMLData::kTensor: {
          gvalue = TensorToOrtValue(data.tensor(), b);
        } break;
        default:
          ORT_NOT_IMPLEMENTED("unknown data type inside TraditionalMLData");
      }
      if (!data.debug_info().empty()) {
        oss << ":" << data.debug_info();
      }
      std::string value_name = data.name();
      if (value_name.empty()) {
        const size_t c = name_data_map.size();
        value_name = is_input ? modelinfo.GetInputName(c) : modelinfo.GetOutputName(c);
      }

      auto p = name_data_map.emplace(value_name, std::move(gvalue));
      if (!p.second) {
        ORT_THROW("duplicated test data name");
        break;
      }
    }
    ORT_CATCH(onnxruntime::NotImplementedException & ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::ostringstream oss2;
        oss2 << "load the " << item_id << "-th item failed," << ex.what();
        ORT_NOT_IMPLEMENTED(oss2.str());
      });
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::ostringstream oss2;
        oss2 << "load the " << item_id << "-th item failed," << ex.what();
        ORT_THROW(oss2.str());
      });
    }
  }
  if (!clean_eof) {
    ORT_THROW("parse input file failed, has extra unparsed data");
  }
}

}  // namespace

#if !defined(ORT_MINIMAL_BUILD)
std::unique_ptr<TestModelInfo> TestModelInfo::LoadOnnxModel(_In_ const PATH_CHAR_TYPE* model_url) {
  return std::unique_ptr<TestModelInfo>(new OnnxModelInfo(model_url));
}
#endif

#if defined(ENABLE_ORT_FORMAT_LOAD)
std::unique_ptr<TestModelInfo> TestModelInfo::LoadOrtModel(_In_ const PATH_CHAR_TYPE* model_url) {
  return std::unique_ptr<TestModelInfo>(new OnnxModelInfo(model_url, true));
}
#endif

/**
   * test_case_dir must have contents of:
   * model.onnx
   * ???/input_??.pb
   * ???/output_??.pb
   * ???/input_??.pb
   * ???/output_??.pb
   */
class OnnxTestCase : public ITestCase {
 private:
  std::string test_case_name_;
  mutable std::vector<std::string> debuginfo_strings_;
  mutable onnxruntime::OrtMutex m_;

  std::vector<std::basic_string<PATH_CHAR_TYPE>> test_data_dirs_;

  std::string GetDatasetDebugInfoString(size_t dataset_id) const override {
    std::lock_guard<OrtMutex> l(m_);
    if (dataset_id < debuginfo_strings_.size()) {
      return debuginfo_strings_[dataset_id];
    }
    // return empty string
    return std::string();
  }

  void ConvertTestData(const ONNX_NAMESPACE::TensorProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

  void ConvertTestData(const ONNX_NAMESPACE::SequenceProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

  void ConvertTestData(const ONNX_NAMESPACE::OptionalProto& test_data_pb,
                       onnxruntime::test::HeapBuffer& b,
                       bool is_input, size_t i,
                       std::unordered_map<std::string, Ort::Value>& out) const;

  std::once_flag model_parsed_;
  std::once_flag config_parsed_;
  double per_sample_tolerance_;
  double relative_per_sample_tolerance_;
  bool post_processing_;
  std::unique_ptr<TestModelInfo> model_info_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OnnxTestCase);

 public:
  OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
               double default_per_sample_tolerance, double default_relative_per_sample_tolerance);
  void GetPerSampleTolerance(double* value) const override;
  void GetRelativePerSampleTolerance(double* value) const override;
  void GetPostProcessing(bool* value) const override;

  const ONNX_NAMESPACE::ValueInfoProto* GetInputInfoFromModel(size_t i) const override {
    return model_info_->GetInputInfoFromModel(i);
  }

  const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const override {
    return model_info_->GetOutputInfoFromModel(i);
  }

  size_t GetDataCount() const override { return test_data_dirs_.size(); }
  const std::string& GetNodeName() const override { return model_info_->GetNodeName(); }
  const PATH_CHAR_TYPE* GetModelUrl() const override { return model_info_->GetModelUrl(); }
  const std::string& GetTestCaseName() const override { return test_case_name_; }
  std::string GetTestCaseVersion() const override { return model_info_->GetModelVersion(); }

  void LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b, std::unordered_map<std::string, Ort::Value>&,
                    bool is_input) const override;
};

std::unique_ptr<ITestCase> CreateOnnxTestCase(const std::string& test_case_name,
                                              std::unique_ptr<TestModelInfo> model,
                                              double default_per_sample_tolerance,
                                              double default_relative_per_sample_tolerance) {
  return std::unique_ptr<ITestCase>(new OnnxTestCase(test_case_name, std::move(model),
                                                     default_per_sample_tolerance,
                                                     default_relative_per_sample_tolerance));
}

void OnnxTestCase::GetPerSampleTolerance(double* value) const {
  *value = per_sample_tolerance_;
}

void OnnxTestCase::GetRelativePerSampleTolerance(double* value) const {
  *value = relative_per_sample_tolerance_;
}

void OnnxTestCase::GetPostProcessing(bool* value) const {
  *value = post_processing_;
}

// CentOS lacks find_if
template <class Iter, class Pred>
inline Iter find_with_pred(Iter first, Iter last, Pred p) {
  while (first != last) {
    if (p(*first)) {
      break;
    }
    ++first;
  }
  return first;
}

static std::string trim_str(const std::string& in) {
  std::string s = in;
  s.erase(s.begin(), find_with_pred(s.begin(), s.end(), [](int ch) {
            return !std::isspace(ch);
          }));
  s.erase(find_with_pred(s.rbegin(), s.rend(), [](int ch) {
            return !std::isspace(ch);
          }).base(),
          s.end());
  return s;
}

static bool read_config_file(const std::basic_string<PATH_CHAR_TYPE>& path, std::map<std::string, std::string>& fc) {
  std::ifstream infile(path);
  if (!infile.good()) {
    return false;
  }

  for (std::string line; std::getline(infile, line);) {
    std::istringstream ss(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> tokens;
    for (std::string token; std::getline(ss, token, ':');) {
      std::string trimmed_token = trim_str(token);
      if (trimmed_token.empty()) {
        continue;
      }
      tokens.push_back(trimmed_token);
    }
    fc[tokens[0]] = tokens[1];
  }
  return true;
}

//load tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadTensor(const PATH_STRING_TYPE& pb_file, ONNX_NAMESPACE::TensorProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToMBString(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToMBString(pb_file), "' failed");
  }
}

//load sequence tensors from disk
template <typename PATH_STRING_TYPE>
static void LoadSequenceTensor(const PATH_STRING_TYPE& pb_file, ONNX_NAMESPACE::SequenceProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToMBString(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToMBString(pb_file), "' failed");
  }
}

template <typename PATH_STRING_TYPE>
static void LoadOptional(const PATH_STRING_TYPE& pb_file,
                         ONNX_NAMESPACE::OptionalProto& input_pb) {
  int tensor_fd;
  auto st = Env::Default().FileOpenRd(pb_file, tensor_fd);
  if (!st.IsOK()) {
    ORT_THROW("open file '", ToMBString(pb_file), "' failed:", st.ErrorMessage());
  }
  google::protobuf::io::FileInputStream f(tensor_fd, protobuf_block_size_in_bytes);
  f.SetCloseOnDelete(true);
  if (!input_pb.ParseFromZeroCopyStream(&f)) {
    ORT_THROW("parse file '", ToMBString(pb_file), "' failed");
  }
}
void OnnxTestCase::LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b,
                                std::unordered_map<std::string, Ort::Value>& name_data_map,
                                bool is_input) const {
  if (id >= test_data_dirs_.size()) {
    ORT_THROW("index out of bound");
  }

  PATH_STRING_TYPE test_data_pb = ConcatPathComponent<PATH_CHAR_TYPE>(
      test_data_dirs_[id], (is_input ? ORT_TSTR("inputs.pb") : ORT_TSTR("outputs.pb")));
  int test_data_pb_fd;
  auto st = Env::Default().FileOpenRd(test_data_pb, test_data_pb_fd);
  if (st.IsOK()) {  //has an all-in-one input file
    std::ostringstream oss;
    {
      std::lock_guard<OrtMutex> l(m_);
      oss << debuginfo_strings_[id];
    }
    ORT_TRY {
      LoopDataFile(test_data_pb_fd, is_input, *model_info_, name_data_map, b, oss);
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        std::ostringstream oss2;
        oss2 << "parse data file \"" << ToMBString(test_data_pb) << "\" failed:" << ex.what();
        ORT_THROW(oss.str());
      });
    }

    {
      std::lock_guard<OrtMutex> l(m_);
      debuginfo_strings_[id] = oss.str();
    }
    return;
  }

  std::vector<PATH_STRING_TYPE> test_data_pb_files;

  const PATH_STRING_TYPE& dir_path = test_data_dirs_[id];
  LoopDir(dir_path,
          [&test_data_pb_files, &dir_path, is_input](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
            if (filename[0] == '.') return true;
            if (f_type != OrtFileType::TYPE_REG) return true;
            std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
            if (!HasExtensionOf(filename_str, ORT_TSTR("pb"))) return true;
            const std::basic_string<PATH_CHAR_TYPE> file_prefix =
                is_input ? ORT_TSTR("input_") : ORT_TSTR("output_");
            if (!filename_str.compare(0, file_prefix.length(), file_prefix)) {
              std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(dir_path, filename_str);
              test_data_pb_files.push_back(p);
            }
            return true;
          });

  SortFileNames(test_data_pb_files);

  for (size_t i = 0; i < test_data_pb_files.size(); ++i) {
    const ONNX_NAMESPACE::ValueInfoProto* value_info_proto = is_input ? model_info_->GetInputInfoFromModel(i) : model_info_->GetOutputInfoFromModel(i);
    if (!value_info_proto->has_type()) {
      ORT_THROW("Model ", is_input ? "input " : "output ", i, " is missing type info");
    }

    if (value_info_proto->type().has_tensor_type()) {
      ONNX_NAMESPACE::TensorProto test_pb;
      LoadTensor(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    } else if (value_info_proto->type().has_sequence_type()) {
      ONNX_NAMESPACE::SequenceProto test_pb;
      LoadSequenceTensor(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    } else if (value_info_proto->type().has_optional_type()) {
      ONNX_NAMESPACE::OptionalProto test_pb;
      LoadOptional(test_data_pb_files[i], test_pb);
      ConvertTestData(test_pb, b, is_input, i, name_data_map);
    } else {
      ORT_THROW("Unsupported type for the ", is_input ? "input " : "output ", i, " in the test runner");
    }
  }
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::TensorProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  const std::string& name = test_data_pb.name();
  const std::string& name_finalized = !name.empty()
                                          ? name
                                          : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

  size_t len = 0;

  auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(test_data_pb, &len);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  void* p = len == 0 ? nullptr : b.AllocMemory(len);
  Ort::Value v1{nullptr};
  onnxruntime::test::OrtCallback d;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  status = onnxruntime::test::TensorProtoToMLValue(test_data_pb, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                   v1, d);
  if (!status.IsOK()) {
    ORT_THROW(status.ToString());
  }
  if (d.f) {
    b.AddDeleter(d);
  }
  out.emplace(name_finalized, std::move(v1));
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::SequenceProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  const std::string& name = test_data_pb.name();
  const std::string& name_finalized = !name.empty()
                                          ? name
                                          : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

  size_t len = 0;

  std::vector<Ort::Value> seq;
  if (test_data_pb.elem_type() != ONNX_NAMESPACE::SequenceProto_DataType_TENSOR) {
    ORT_THROW("Only parsing a sequence of tensors is currently supported");
  }
  const auto& tensors = test_data_pb.tensor_values();
  const size_t val = tensors.size();
  seq.reserve(val);

  for (auto it = tensors.cbegin(); it != tensors.cend(); ++it) {
    auto status = onnxruntime::test::GetSizeInBytesFromTensorProto<0>(*it, &len);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    void* p = len == 0 ? nullptr : b.AllocMemory(len);
    Ort::Value v1{nullptr};
    onnxruntime::test::OrtCallback d;
    OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
    status = onnxruntime::test::TensorProtoToMLValue(*it, onnxruntime::test::MemBuffer(p, len, cpu_memory_info),
                                                     v1, d);
    if (!status.IsOK()) {
      ORT_THROW(status.ToString());
    }
    if (d.f) {
      b.AddDeleter(d);
    }

    seq.push_back(std::move(v1));
  }

  if (seq.size() == 0) {
    // TODO: implement support for creating empty sequences. Not urgent yet since we don't have real world models.
    // For now, only the single node ONNX test - `test_loop13_seq` requires it (will keep it disabled for now).
    ORT_THROW("Creation of empty sequences is currently not supported in the test runner");
  } else {
    out.emplace(name_finalized, Ort::Value::CreateSequence(seq));
  }
}

void OnnxTestCase::ConvertTestData(const ONNX_NAMESPACE::OptionalProto& test_data_pb,
                                   onnxruntime::test::HeapBuffer& b,
                                   bool is_input, size_t i,
                                   std::unordered_map<std::string, Ort::Value>& out) const {
  // Optional Tensor
  if (test_data_pb.elem_type() ==
      ONNX_NAMESPACE::OptionalProto_DataType::OptionalProto_DataType_TENSOR) {
    // The optional tensor is not "None", deal with it as a regular tensor
    if (test_data_pb.has_tensor_value()) {
      ConvertTestData(test_data_pb.tensor_value(), b, is_input, i, out);
    } else {
      // Process None
      // If is_input is true, don't include the None in the feeds
      // If is_input is false, include it in the fetches, so that we can validate
      // whether we received a None output from ORT.

      if (!is_input) {
        const std::string& name = test_data_pb.name();
        const std::string& name_finalized = !name.empty()
                                                ? name
                                                : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

        // Our API doesn't support creating None OrtValue,
        // so we place an nullptr into the expected values.
        Ort::Value value{nullptr};
        out.emplace(name_finalized, std::move(value));
      }
    }
  }  // Optional Sequence Tensor
  else if (test_data_pb.elem_type() ==
           ONNX_NAMESPACE::OptionalProto_DataType::OptionalProto_DataType_SEQUENCE) {
    // The optional sequence tensor is not "None", deal with it as a regular tensor
    if (test_data_pb.has_sequence_value()) {
      // ConvertTestData() ensures that sequence contains only tensors - we do no need
      // a redundant check here
      ConvertTestData(test_data_pb.sequence_value(), b, is_input, i, out);
    } else {
      // Process None
      // If is_input is true, don't include the None in the feeds
      // If is_input is false, include it in the fetches, so that we can validate
      // whether we received a None output from ORT.

      if (!is_input) {
        const std::string& name = test_data_pb.name();
        const std::string& name_finalized = !name.empty()
                                                ? name
                                                : (is_input ? model_info_->GetInputName(i) : model_info_->GetOutputName(i));

        // Our API doesn't support creating None OrtValue,
        // so we place an nullptr into the expected values.
        Ort::Value value{nullptr};
        out.emplace(name_finalized, std::move(value));
      }
    }
  }
}

OnnxTestCase::OnnxTestCase(const std::string& test_case_name, _In_ std::unique_ptr<TestModelInfo> model,
                           double default_per_sample_tolerance, double default_relative_per_sample_tolerance)
    : test_case_name_(test_case_name), model_info_(std::move(model)) {
  std::basic_string<PATH_CHAR_TYPE> test_case_dir = model_info_->GetDir();

  // parse config
  std::basic_string<PATH_CHAR_TYPE> config_path =
      ConcatPathComponent<PATH_CHAR_TYPE>(test_case_dir, ORT_TSTR("config.txt"));
  /* Note: protobuf-lite doesn't support reading protobuf files as text-format. Config.txt is exactly that.
     That's the reason I've to parse the file in a different way to read the configs. Currently
     this affects 2 tests - fp16_tiny_yolov2 and fp16_inception_v1. It's not clear why we've to use protobuf
     to represent simple config files that have only key-value pairs.
   */
  std::map<std::string, std::string> fc;
  per_sample_tolerance_ = default_per_sample_tolerance;
  relative_per_sample_tolerance_ = default_relative_per_sample_tolerance;
  post_processing_ = false;
  if (read_config_file(config_path, fc)) {
    if (fc.count("per_sample_tolerance") > 0) {
      per_sample_tolerance_ = stod(fc["per_sample_tolerance"]);
    }
    if (fc.count("relative_per_sample_tolerance") > 0) {
      relative_per_sample_tolerance_ = stod(fc["relative_per_sample_tolerance"]);
    }
    if (fc.count("post_processing") > 0) {
      post_processing_ = fc["post_processing"] == "true";
    }
  }

  LoopDir(test_case_dir, [&test_case_dir, this](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
    if (filename[0] == '.') return true;
    if (f_type == OrtFileType::TYPE_DIR) {
      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(test_case_dir, filename);
      test_data_dirs_.push_back(p);
      debuginfo_strings_.push_back(ToMBString(p));
    }
    return true;
  });
}

void LoadTests(const std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_paths,
               const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases,
               double default_per_sample_tolerance, double default_relative_per_sample_tolerance,
               const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests,
               const std::function<void(std::unique_ptr<ITestCase>)>& process_function) {
  std::vector<std::basic_string<PATH_CHAR_TYPE>> paths(input_paths);
  while (!paths.empty()) {
    std::basic_string<PATH_CHAR_TYPE> node_data_root_path = paths.back();
    paths.pop_back();
    std::basic_string<PATH_CHAR_TYPE> my_dir_name = GetLastComponent(node_data_root_path);
    LoopDir(node_data_root_path, [&](const PATH_CHAR_TYPE* filename, OrtFileType f_type) -> bool {
      if (filename[0] == '.') return true;
      if (f_type == OrtFileType::TYPE_DIR) {
        std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename);
        paths.push_back(p);
        return true;
      }

      std::basic_string<PATH_CHAR_TYPE> filename_str = filename;
      bool is_onnx_format = HasExtensionOf(filename_str, ORT_TSTR("onnx"));
      bool is_ort_format = HasExtensionOf(filename_str, ORT_TSTR("ort"));
      bool is_valid_model = false;

#if !defined(ORT_MINIMAL_BUILD)
      is_valid_model = is_onnx_format;
#endif

#if defined(ENABLE_ORT_FORMAT_LOAD)
      is_valid_model = is_valid_model || is_ort_format;
#endif
      if (!is_valid_model)
        return true;

      std::basic_string<PATH_CHAR_TYPE> test_case_name = my_dir_name;
      if (test_case_name.compare(0, 5, ORT_TSTR("test_")) == 0) test_case_name = test_case_name.substr(5);

      if (!whitelisted_test_cases.empty() && std::find(whitelisted_test_cases.begin(), whitelisted_test_cases.end(),
                                                       test_case_name) == whitelisted_test_cases.end()) {
        return true;
      }
      if (disabled_tests.find(test_case_name) != disabled_tests.end()) return true;

      std::basic_string<PATH_CHAR_TYPE> p = ConcatPathComponent<PATH_CHAR_TYPE>(node_data_root_path, filename_str);

      std::unique_ptr<TestModelInfo> model_info;

      if (is_onnx_format) {
#if !defined(ORT_MINIMAL_BUILD)
        model_info = TestModelInfo::LoadOnnxModel(p.c_str());
#else
        ORT_THROW("onnx model is not supported in this build");
#endif
      } else if (is_ort_format) {
#if defined(ENABLE_ORT_FORMAT_LOAD)
        model_info = TestModelInfo::LoadOrtModel(p.c_str());
#else
        ORT_THROW("ort model is not supported in this build");
#endif
      } else {
        ORT_NOT_IMPLEMENTED(ToMBString(filename_str), " is not supported");
      }

      std::unique_ptr<ITestCase> l = CreateOnnxTestCase(ToMBString(test_case_name), std::move(model_info),
                                                        default_per_sample_tolerance,
                                                        default_relative_per_sample_tolerance);
      process_function(std::move(l));
      return true;
    });
  }
}

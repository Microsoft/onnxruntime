// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"

#include <filesystem>
#include <fstream>
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <thread>

#ifdef _WIN32
#include <Windows.h>
#include <winioctl.h>
#else
#include <sys/stat.h>
#endif

#include <gsl/gsl>
#include "gtest/gtest.h"

#include "core/common/path_string.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {

TEST(PlatformEnvTest, DirectoryCreationAndDeletion) {
  const auto& env = Env::Default();
  const PathString root_dir = ORT_TSTR("tmp_platform_env_test_dir");
  const PathString sub_dir = root_dir + ORT_TSTR("/some/test/directory");

  ASSERT_FALSE(env.FolderExists(root_dir));

  ASSERT_STATUS_OK(env.CreateFolder(sub_dir));
  ASSERT_TRUE(env.FolderExists(sub_dir));

  // create a file in the subdirectory
  {
    std::ofstream outfile{sub_dir + ORT_TSTR("/file")};
    outfile << "hello!";
  }

  ASSERT_STATUS_OK(env.DeleteFolder(root_dir));
  ASSERT_FALSE(env.FolderExists(root_dir));
}

TEST(PlatformEnvTest, GetErrnoInfo) {
  // command that should generate an errno error
  std::ifstream file("non_existent_file");
  ASSERT_TRUE(file.fail());
  auto [err, msg] = GetErrnoInfo();
  ASSERT_EQ(err, ENOENT);

#if defined(_WIN32)
#pragma warning(push)
#pragma warning(disable : 4996)
#endif

  // GetErrnoInfo uses strerror_r or strerror_s depending on the platform. use the unsafe std::sterror to get the
  // expected value given this is a unit test so doesn't have to be as robust.
  ASSERT_EQ(msg, std::strerror(ENOENT));

#if defined(_WIN32)
#pragma warning(pop)
#endif
}

namespace {

void WriteRandomAccessTestFile(const std::string& contents, PathString& path, ScopedFileDeleter& deleter) {
  path = ORT_TSTR("random_access_file_XXXXXX");
  FILE* file = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateTestFile(file, path));
  deleter = ScopedFileDeleter(path);
  std::unique_ptr<FILE, int (*)(FILE*)> owner(file, fclose);
  ASSERT_EQ(contents.size(), fwrite(contents.data(), 1, contents.size(), file));
  ASSERT_EQ(0, fclose(owner.release()));
}

class RandomAccessFileTest : public testing::Test {
 protected:
  void SetUp() override {
    contents_.resize(64 * 1024);
    for (size_t i = 0; i < contents_.size(); ++i) {
      contents_[i] = static_cast<char>((i * 31 + i / 257) & 0xff);
    }
    ASSERT_NO_FATAL_FAILURE(WriteRandomAccessTestFile(contents_, path_, deleter_));
    ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(path_.c_str(), file_));
    ASSERT_NE(file_, nullptr);
  }

  std::string contents_;
  PathString path_;
  ScopedFileDeleter deleter_;
  std::unique_ptr<RandomAccessFile> file_;
};

TEST_F(RandomAccessFileTest, ReadsRangesAndLengthFromOneOpenFile) {
  uint64_t length = 0;
  ASSERT_STATUS_OK(file_->GetLength(length));
  EXPECT_EQ(length, contents_.size());
  size_t legacy_length = 0;
  ASSERT_STATUS_OK(Env::Default().GetFileLength(path_.c_str(), legacy_length));
  EXPECT_EQ(length, legacy_length);
  std::string output(193, '\0');
  ASSERT_STATUS_OK(file_->Read(271, gsl::span<char>(output)));
  EXPECT_EQ(output, contents_.substr(271, output.size()));
  ASSERT_STATUS_OK(file_->Read(7, gsl::span<char>(output)));
  EXPECT_EQ(output, contents_.substr(7, output.size()));

  std::string legacy_output(output.size(), '\0');
  ASSERT_STATUS_OK(Env::Default().ReadFileIntoBuffer(path_.c_str(), 7, legacy_output.size(),
                                                     gsl::span<char>(legacy_output)));
  EXPECT_EQ(output, legacy_output);
}

TEST_F(RandomAccessFileTest, MappingUsesOpenedFileAfterPathReplacement) {
  PathString replacement_path;
  ScopedFileDeleter replacement_deleter;
  const std::string replacement(contents_.size(), 'x');
  ASSERT_NO_FATAL_FAILURE(WriteRandomAccessTestFile(replacement, replacement_path, replacement_deleter));

  const PathString opened_path = path_ + ORT_TSTR(".opened");
  ScopedFileDeleter opened_deleter(opened_path);
  std::error_code ec;
  std::filesystem::rename(path_, opened_path, ec);
  ASSERT_FALSE(ec) << ec.message();
  std::filesystem::rename(replacement_path, path_, ec);
  ASSERT_FALSE(ec) << ec.message();

  RandomAccessFile::MappedMemoryPtr mapped_memory;
  ASSERT_STATUS_OK(file_->Map(0, contents_.size(), mapped_memory));
  ASSERT_NE(mapped_memory, nullptr);
  EXPECT_EQ(std::memcmp(mapped_memory.get(), contents_.data(), contents_.size()), 0);

#ifndef __wasm__
  PathString canonical_path;
  ASSERT_STATUS_OK(file_->GetCanonicalPath(canonical_path));
  EXPECT_TRUE(std::filesystem::equivalent(canonical_path, opened_path, ec)) << ec.message();
#endif
}

#if !defined(_WIN32) && !defined(__wasm__)
TEST_F(RandomAccessFileTest, MappingIsWritableWithoutChangingTheFile) {
  RandomAccessFile::MappedMemoryPtr mapped_memory;
  ASSERT_STATUS_OK(file_->Map(0, contents_.size(), mapped_memory));
  mapped_memory[0] = static_cast<char>(contents_[0] ^ 1);
  std::array<char, 1> output{};
  ASSERT_STATUS_OK(file_->Read(0, output));
  EXPECT_EQ(output[0], contents_[0]);
  RandomAccessFile::MappedMemoryPtr second_mapping;
  ASSERT_STATUS_OK(file_->Map(0, contents_.size(), second_mapping));
  EXPECT_EQ(second_mapping[0], contents_[0]);
}
#endif

#ifndef __wasm__
TEST_F(RandomAccessFileTest, ConcurrentReadsDoNotShareAFilePosition) {
  constexpr size_t kReaderCount = 4;
  std::array<Status, kReaderCount> statuses;
  std::array<bool, kReaderCount> matched;
  matched.fill(true);
  {
    InlinedVector<std::thread> readers;
    readers.reserve(kReaderCount);
    auto join_readers = gsl::finally([&] {
      for (auto& reader : readers) {
        reader.join();
      }
    });
    for (size_t reader = 0; reader < kReaderCount; ++reader) {
      readers.emplace_back([&, reader] {
        std::string output(4096, '\0');
        for (size_t iteration = 0; iteration < 100; ++iteration) {
          const auto offset = (reader * 1009 + iteration * 3277) % (contents_.size() - output.size());
          statuses[reader] = file_->Read(static_cast<FileOffsetType>(offset), gsl::span<char>(output));
          if (!statuses[reader].IsOK()) {
            return;
          }
          if (output != contents_.substr(offset, output.size())) {
            matched[reader] = false;
            return;
          }
        }
      });
    }
  }
  for (size_t reader = 0; reader < kReaderCount; ++reader) {
    ASSERT_STATUS_OK(statuses[reader]);
    EXPECT_TRUE(matched[reader]) << "Reader " << reader;
  }
}
#endif

TEST_F(RandomAccessFileTest, RejectsInvalidRangesAndUnexpectedEof) {
  std::array<char, 4> output{};
  EXPECT_EQ(file_->Read(-1, output).Code(), common::INVALID_ARGUMENT);
  constexpr auto kMaxOffset = std::numeric_limits<FileOffsetType>::max();
  EXPECT_EQ(file_->Read(kMaxOffset, output).Code(), common::INVALID_ARGUMENT);
  ASSERT_STATUS_OK(file_->Read(kMaxOffset, {}));
  ASSERT_STATUS_OK(file_->Read(static_cast<FileOffsetType>(contents_.size()), {}));
  EXPECT_FALSE(file_->Read(static_cast<FileOffsetType>(contents_.size()), output).IsOK());
  // This request can read a prefix, but must fail rather than accept a short read.
  EXPECT_FALSE(file_->Read(static_cast<FileOffsetType>(contents_.size() - 2), output).IsOK());
  ASSERT_STATUS_OK(file_->Read(0, output));
  EXPECT_EQ(std::string(output.data(), output.size()), contents_.substr(0, output.size()));
}

TEST_F(RandomAccessFileTest, EmptyFileHasZeroLength) {
  PathString empty_path;
  ScopedFileDeleter empty_deleter;
  ASSERT_NO_FATAL_FAILURE(WriteRandomAccessTestFile({}, empty_path, empty_deleter));
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(empty_path.c_str(), file_));
  uint64_t length = 123;
  ASSERT_STATUS_OK(file_->GetLength(length));
  EXPECT_EQ(length, 0U);
  ASSERT_STATUS_OK(file_->Read(0, {}));
  char byte;
  EXPECT_FALSE(file_->Read(0, gsl::span<char>(&byte, 1)).IsOK());
}

TEST_F(RandomAccessFileTest, FailedOpenDoesNotReplaceAnExistingFile) {
  const auto* original = file_.get();
  const auto missing_path = path_ + ORT_TSTR(".missing");
  ASSERT_FALSE(Env::Default().FileExists(missing_path));
  EXPECT_FALSE(Env::Default().OpenRandomAccessFile(missing_path.c_str(), file_).IsOK());
  EXPECT_EQ(file_.get(), original);
  EXPECT_EQ(Env::Default().OpenRandomAccessFile(nullptr, file_).Code(), common::INVALID_ARGUMENT);
  EXPECT_EQ(file_.get(), original);
  EXPECT_FALSE(Env::Default().OpenRandomAccessFile(ORT_TSTR("."), file_).IsOK());
  EXPECT_EQ(file_.get(), original);
  char byte;
  ASSERT_STATUS_OK(file_->Read(1, gsl::span<char>(&byte, 1)));
  EXPECT_EQ(byte, contents_[1]);
}

TEST_F(RandomAccessFileTest, DefaultImplementationReportsUnsupportedWithoutReplacingFile) {
  const auto* original = file_.get();
  EXPECT_EQ(Env::Default().Env::OpenRandomAccessFile(path_.c_str(), file_).Code(), common::NOT_IMPLEMENTED);
  EXPECT_EQ(file_.get(), original);
}

TEST_F(RandomAccessFileTest, PathReplacementDoesNotChangeTheOpenFile) {
  const std::string replacement_contents = "replacement file";
  PathString replacement_path;
  ScopedFileDeleter replacement_deleter;
  ASSERT_NO_FATAL_FAILURE(WriteRandomAccessTestFile(replacement_contents, replacement_path, replacement_deleter));
#ifdef _WIN32
  // Ordinary Windows rename cannot replace an open destination, even with delete sharing.
  const HANDLE replacement_handle =
      CreateFile2(replacement_path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  OPEN_EXISTING, nullptr);
  ASSERT_NE(replacement_handle, INVALID_HANDLE_VALUE) << GetLastError();
  auto close_replacement = gsl::finally([&] { CloseHandle(replacement_handle); });

  const auto target_path = std::filesystem::absolute(path_).native();
  const size_t name_bytes = SafeInt<size_t>(target_path.size()) * sizeof(wchar_t);
  const size_t rename_info_bytes = SafeInt<size_t>(sizeof(FILE_RENAME_INFO)) + name_bytes;
  const auto rename_info_size = gsl::narrow<DWORD>(rename_info_bytes);
  auto rename_buffer = std::make_unique<char[]>(rename_info_size);
  auto* rename_info = reinterpret_cast<FILE_RENAME_INFO*>(rename_buffer.get());
  rename_info->Flags = FILE_RENAME_FLAG_REPLACE_IF_EXISTS | FILE_RENAME_FLAG_POSIX_SEMANTICS;
  rename_info->RootDirectory = nullptr;
  rename_info->FileNameLength = gsl::narrow<DWORD>(name_bytes);
  std::memcpy(rename_info->FileName, target_path.c_str(), name_bytes);
  // Some SDK headers omit FileRenameInfoEx. Its documented FILE_INFO_BY_HANDLE_CLASS value is 22.
  constexpr auto kFileRenameInfoEx = static_cast<FILE_INFO_BY_HANDLE_CLASS>(22);
  const BOOL renamed =
      SetFileInformationByHandle(replacement_handle, kFileRenameInfoEx, rename_info, rename_info_size);
  const DWORD rename_error = GetLastError();
  ASSERT_NE(renamed, FALSE) << rename_error;
#else
  std::error_code error;
  std::filesystem::rename(replacement_path, path_, error);
  ASSERT_FALSE(error) << error.message();
#endif

  uint64_t length = 0;
  ASSERT_STATUS_OK(file_->GetLength(length));
  EXPECT_EQ(length, contents_.size());
  std::string original_output(contents_.size(), '\0');
  ASSERT_STATUS_OK(file_->Read(0, gsl::span<char>(original_output)));
  EXPECT_EQ(original_output, contents_);

  std::unique_ptr<RandomAccessFile> replacement;
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(path_.c_str(), replacement));
  ASSERT_STATUS_OK(replacement->GetLength(length));
  EXPECT_EQ(length, replacement_contents.size());
  std::string replacement_output(length, '\0');
  ASSERT_STATUS_OK(replacement->Read(0, gsl::span<char>(replacement_output)));
  EXPECT_EQ(replacement_output, replacement_contents);
}

TEST_F(RandomAccessFileTest, HandlesInPlaceTruncationAccordingToPlatformSharingRules) {
#ifdef _WIN32
  // Windows denies write sharing while the file is open; destruction must release that restriction.
  {
    std::ofstream writer(path_, std::ios::binary | std::ios::trunc);
    EXPECT_FALSE(writer.is_open());
  }
  file_.reset();
  std::ofstream writer(path_, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(writer.is_open());
#else
  std::error_code error;
  std::filesystem::resize_file(path_, 3, error);
  ASSERT_FALSE(error) << error.message();
  uint64_t length = 0;
  ASSERT_STATUS_OK(file_->GetLength(length));
  EXPECT_EQ(length, 3U);
  std::array<char, 4> output{};
  EXPECT_FALSE(file_->Read(0, output).IsOK());
  ASSERT_STATUS_OK(file_->Read(0, gsl::span<char>(output.data(), 3)));
  EXPECT_EQ(std::string(output.data(), 3), contents_.substr(0, 3));
#endif
}

#if !defined(_WIN32) && !defined(__wasm__)
TEST_F(RandomAccessFileTest, RejectsFifosWithoutWaitingForAWriter) {
  PathString fifo_path;
  ScopedFileDeleter fifo_deleter;
  ASSERT_NO_FATAL_FAILURE(WriteRandomAccessTestFile({}, fifo_path, fifo_deleter));
  ASSERT_EQ(std::remove(fifo_path.c_str()), 0);
  const int create_result = mkfifo(fifo_path.c_str(), 0600);
  const int create_error = errno;
#ifdef __ANDROID__
  if (create_result != 0 && (create_error == EACCES || create_error == EPERM)) {
    GTEST_SKIP() << "Android SELinux policy denies FIFO creation: " << std::strerror(create_error);
  }
#endif
  ASSERT_EQ(create_result, 0) << std::strerror(create_error);
  std::unique_ptr<RandomAccessFile> fifo;
  EXPECT_FALSE(Env::Default().OpenRandomAccessFile(fifo_path.c_str(), fifo).IsOK());
  EXPECT_EQ(fifo, nullptr);
}
#endif

#ifndef __wasm__
TEST_F(RandomAccessFileTest, ReadsSparseFileBeyondFourGiB) {
  if (sizeof(FileOffsetType) < 8 || sizeof(size_t) < 8) {
    GTEST_SKIP() << "Requires 64-bit file offsets and sizes.";
  }
  constexpr int64_t kOffset = (int64_t{1} << 32) + 123;
  file_.reset();
#ifdef _WIN32
  {
    const HANDLE sparse_handle = CreateFile2(path_.c_str(), GENERIC_WRITE, 0, OPEN_EXISTING, nullptr);
    ASSERT_NE(sparse_handle, INVALID_HANDLE_VALUE) << GetLastError();
    auto close_sparse = gsl::finally([&] { CloseHandle(sparse_handle); });
    DWORD bytes_returned = 0;
    const BOOL marked_sparse =
        DeviceIoControl(sparse_handle, FSCTL_SET_SPARSE, nullptr, 0, nullptr, 0, &bytes_returned, nullptr);
    const DWORD sparse_error = GetLastError();
    ASSERT_NE(marked_sparse, FALSE) << sparse_error;
  }
#endif
  {
    std::fstream writer(path_, std::ios::binary | std::ios::in | std::ios::out);
    ASSERT_TRUE(writer.is_open());
    writer.seekp(kOffset);
    writer.put('Z');
    writer.close();
    ASSERT_FALSE(writer.fail());
  }
  ASSERT_STATUS_OK(Env::Default().OpenRandomAccessFile(path_.c_str(), file_));
  uint64_t length = 0;
  ASSERT_STATUS_OK(file_->GetLength(length));
  EXPECT_EQ(length, static_cast<size_t>(kOffset + 1));
  std::array<char, 2> output{};
  ASSERT_STATUS_OK(file_->Read(static_cast<FileOffsetType>(kOffset - 1), output));
  EXPECT_EQ(output[0], '\0');
  EXPECT_EQ(output[1], 'Z');
}
#endif

}  // namespace

}  // namespace test
}  // namespace onnxruntime

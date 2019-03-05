#include "file_util.h"

#include <gtest/gtest.h>
#include <stdio.h>

#ifdef _WIN32
#include <io.h>
#endif

namespace onnxruntime {
namespace test {

void DeleteFileFromDisk(const ORTCHAR_T* path) {
#ifdef _WIN32
  ASSERT_EQ(TRUE, DeleteFileW(path));
#else
  ASSERT_EQ(0, unlink(path));
#endif
}
void CreateTestFile(FILE** out, std::basic_string<ORTCHAR_T>& filename_template) {
  if (filename_template.empty()) throw std::runtime_error("file name template can't be empty");
  ORTCHAR_T* filename = const_cast<ORTCHAR_T*>(filename_template.c_str());
#ifdef _WIN32
  ASSERT_EQ(0, _wmktemp_s(filename, filename_template.length() + 1));
  FILE* fp = nullptr;
  ASSERT_EQ(0, _wfopen_s(&fp, filename, ORT_TSTR("w")));
#else
  int fd = mkstemp(filename);
  if (fd < 0) {
    throw std::runtime_error("open temp file failed");
  }
  FILE* fp = fdopen(fd, "w");
#endif
  *out = fp;
}
}  // namespace test
}  // namespace onnxruntime
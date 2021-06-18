#include "core/providers/common.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {

template <typename T>
void StridedCopy(onnxruntime::concurrency::ThreadPool* thread_pool,
                 T* dst,
                 std::vector<int64_t> dst_shape,
                 std::vector<int64_t> dst_strides,
                 T* src,
                 std::vector<int64_t> src_strides);

}  // namespace onnxruntime

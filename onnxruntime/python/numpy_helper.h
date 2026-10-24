// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif
#include <numpy/arrayobject.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
namespace onnxruntime {
namespace python {

// Returns the NumPy type_num for `ml_dtypes.bfloat16`, or -1 if ml_dtypes is
// not installed / cannot be imported. Cached after the first call.
int GetMlDtypesBfloat16TypeNum();

inline bool IsNumericNumpyType(int npy_type) {
  if (npy_type < NPY_OBJECT || npy_type == NPY_HALF) {
    return true;
  }
  const int bf16 = GetMlDtypesBfloat16TypeNum();
  return bf16 >= 0 && npy_type == bf16;
}
}  // namespace python
}  // namespace onnxruntime
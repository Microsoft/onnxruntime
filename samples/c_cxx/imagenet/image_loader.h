// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <string>
#include "cached_interpolation.h"
#include "sync_api.h"
#include "data_processing.h"
#include <onnxruntime/core/session/onnxruntime_c_api.h>

#ifndef HAVE_JPEG
#include <wincodec.h>
#include <wincodecsdk.h>
#include <atlbase.h>
#endif

template <typename T>
void ResizeImageInMemory(const T* input_data, float* output_data, int in_height, int in_width, int out_height,
                         int out_width, int channels);




template <typename InputType>
class OutputCollector {
 public:
  virtual void operator()(const std::vector<InputType>& task_id_list, const OrtValue* tensor) = 0;
  // Release the internal cache. It need be called whenever batchsize is changed
  virtual void ResetCache() = 0;
};

class InceptionPreprocessing : public DataProcessing {
 private:
  const int out_height_;
  const int out_width_;
  const int channels_;
  const double central_fraction_=0.875;
#ifndef HAVE_JPEG
  IWICImagingFactory *piFactory = nullptr;
#endif
 public:
  InceptionPreprocessing(int out_height, int out_width, int channels);
  

  void operator()(const void* input_data, void* output_data) const override;
 
  //output data from this class is in NWHC format
  std::vector<int64_t> GetOutputShape(size_t batch_size) const override {
    return {(int64_t)batch_size, out_height_, out_width_, channels_};
  }
};

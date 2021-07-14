// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/cpu/nn/pool_base.h"
namespace onnxruntime {

template <typename T, typename PoolType>
struct Pool1DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }
  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      T Yh = PoolType::Initialize();
      for (int64_t h = hstart; h < hend; ++h) {
        PoolType::Process(x_d[h], Yh, pool_context_);
      }
      if (pool_attrs_.count_include_pad) {
        PoolType::Finalize(kernel_shape[0], Yh, pool_context_);
      } else {
        PoolType::Finalize(hend - hstart, Yh, pool_context_);
      }
      y_d[ph] = Yh;
    }
  }
};

template <typename T, typename PoolType>
struct Pool2DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = std::min(wstart + kernel_shape[1], width);
        wstart = std::max(wstart, static_cast<int64_t>(0));
        const int64_t pool_index = ph * pooled_width + pw;
        T Yh = PoolType::Initialize();
        for (int64_t h = hstart; h < hend; ++h) {
          for (int64_t w = wstart; w < wend; ++w) {
            const int64_t input_index = h * width + w;
            PoolType::Process(x_d[input_index], Yh, pool_context_);
          }
        }
        if (pool_attrs_.count_include_pad) {
          PoolType::Finalize(kernel_shape[0] * kernel_shape[1], Yh, pool_context_);
        } else {
          PoolType::Finalize((hend - hstart) * (wend - wstart), Yh, pool_context_);
        }
        y_d[pool_index] = Yh;
      }
    }
  }
};

template <typename T, typename PoolType>
struct Pool3DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = std::min(wstart + kernel_shape[1], width);
        wstart = std::max(wstart, static_cast<int64_t>(0));
        for (int64_t pd = 0; pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = std::min(dstart + kernel_shape[2], depth);
          dstart = std::max(dstart, static_cast<int64_t>(0));
          const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          T Yh = PoolType::Initialize();
          for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
              for (int64_t d = dstart; d < dend; ++d) {
                const int64_t input_index = h * width * depth + w * depth + d;
                PoolType::Process(x_d[input_index], Yh, pool_context_);
              }
            }
          }
          if (pool_attrs_.count_include_pad) {
            PoolType::Finalize(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh, pool_context_);
          } else {
            PoolType::Finalize((hend - hstart) * (wend - wstart) * (dend - dstart), Yh, pool_context_);
          }
          y_d[pool_index] = Yh;
        }
      }
    }
  }
};

template <typename T>
struct MaxPool1DTask final {
  const T* X_data;
  T* Y_data;
  int64_t* I_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }
  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      T Yh = std::numeric_limits<T>::lowest();
      int64_t h_index = -1;
      for (int64_t h = hstart; h < hend; h += dilation_h) {
        if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
          if (x_d[h] > Yh) {
            Yh = x_d[h];
            h_index = h;
          }
        }
      }
      y_d[ph] = Yh;
      if (i_d != nullptr)
        i_d[ph] = c * x_step + h_index;
    }
  }
};

template <typename T>
struct MaxPool2DTask final {
  const T* X_data;
  T* Y_data;
  int64_t* I_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  int64_t storage_order;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        const int64_t pool_index = ph * pooled_width + pw;
        T Yh = std::numeric_limits<T>::lowest();
        int64_t h_index = -1;
        int64_t w_index = -1;
        for (int64_t h = hstart; h < hend; h += dilation_h) {
          if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
            for (int64_t w = wstart; w < wend; w += dilation_w) {
              if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                const int64_t input_index = h * width + w;
                if (x_d[input_index] > Yh) {
                  Yh = x_d[input_index];
                  h_index = h;
                  w_index = w;
                }
              }
            }
          }
        }
        y_d[pool_index] = Yh;
        if (i_d != nullptr)
          i_d[pool_index] =
              storage_order == 0 ? c * x_step + h_index * width + w_index : c * x_step + h_index + w_index * height;
      }
    }
  }
};

template <typename T>
struct MaxPool3DTask {
  const T* X_data;
  T* Y_data;
  int64_t* I_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t dilation_d;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  int64_t storage_order;

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        for (int64_t pd = 0; pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = dstart + kernel_shape[2] * dilation_d;
          const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          T Yh = std::numeric_limits<T>::lowest();
          int64_t h_index = -1;
          int64_t w_index = -1;
          int64_t d_index = -1;
          for (int64_t h = hstart; h < hend; h += dilation_h) {
            if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
              for (int64_t w = wstart; w < wend; w += dilation_w) {
                if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                  for (int64_t d = dstart; d < dend; d += dilation_d) {
                    if (math::is_a_ge_zero_and_a_lt_b(d, depth)) {
                      const int64_t input_index = h * width * depth + w * depth + d;
                      if (x_d[input_index] > Yh) {
                        Yh = x_d[input_index];
                        h_index = h;
                        w_index = w;
                        d_index = d;
                      }
                    }
                  }
                }
              }
            }
          }
          y_d[pool_index] = Yh;
          if (i_d != nullptr)
            i_d[pool_index] = storage_order == 0 ? c * x_step + h_index * width * depth + w_index * depth + d_index :
                                                   c * x_step + h_index + w_index * height + d_index * height * width;
        }
      }
    }
  }
};

template<typename T>
T* FindLastLargerEqual(T* head, T* tail, const T* key) {
  //find first elem smaller than key in a descending vector
  while (head < tail) {
    auto mid = head + ((tail-head)>>1);
    if (*mid >= *key) {
      head = mid + 1;
    } else {
      tail = mid;
    }
  }
  return head - 1;
}

template<typename T>
void MaxPoolOpt(const T* x,
                const T* x_end,
                int64_t x_gap, //x_next = x + x_gap
                int64_t dilation,
                int64_t pad_front,
                int64_t pad_back,
                int64_t pool_size,
                T* y, int64_t y_gap, //y_next = y + y_gap 
                T* que) { //que of size pool_size

  T padding = std::numeric_limits<T>::lowest();
  T* head = que;
  T* tail = head;
  int64_t cnt = 0; //num of elems in que
  int64_t x_step = x_gap * dilation;
  T* back = que + pool_size - 1;
  const T* x_i = x - pad_front * x_gap;
  const T* last_x_i = x_i;
  const T* x_end_with_pads = x_end + pad_back * x_gap;

  //init queue with first pool_size elems
  for (int64_t i = 0; i < pool_size; i++, x_i += x_step) {
    if (x_i < x) {
      *tail = padding;
      tail = tail == back ? que : tail + 1;
      ++cnt;
    } else {
      if (cnt == 0) {
        *tail = *x_i;
        tail = tail == back ? que : tail + 1;
        ++cnt;
      } else {
        T* last_le = nullptr;
        if (head < tail) {
          last_le = FindLastLargerEqual(head, tail, x_i);
          cnt = last_le - head + 2;
        } else {
          if (*back >= *x_i) {
            last_le = FindLastLargerEqual(que, tail, x_i);
            cnt = back - head + last_le - que + 3;
          } else {
            last_le = FindLastLargerEqual(head, back + 1, x_i);
            cnt = last_le - head + 2;
          }
        }
        *++last_le = *x_i;
        tail = last_le == back ? que : last_le + 1; 
      }
    }
  }
  
  //fill max in current window
  for (;; x_i += x_step, last_x_i += x_step, y += y_gap) {
    //save max
    *y = *head;
    //dequeue
    if (((last_x_i < x || last_x_i >= x_end) && *head == padding) ||
        (last_x_i >= x && last_x_i < x_end && *head == *last_x_i)) {
      head = head == back ? que : head + 1;
      --cnt;
    }
    if (x_i >= x_end_with_pads) {
      break;
    }
    //enqueue
    if (x_i < x || x_i >= x_end) {
      *tail = padding;
      tail = tail == back ? que : tail + 1;
      ++cnt;
    } else {
      if (cnt == 0) {
        *tail = *x_i;
        tail = tail == back ? que : tail + 1;
        ++cnt;
      } else {
        T* last_le = nullptr;
        if (head < tail) {
          last_le = FindLastLargerEqual(head, tail, x_i);
          cnt = last_le - head + 2;
        } else {
          if (*back >= *x_i) {
            last_le = FindLastLargerEqual(que, tail, x_i);
            cnt = back - head + last_le - que + 3;
          } else {
            last_le = FindLastLargerEqual(head, back + 1, x_i);
            cnt = last_le - head + 2;
          }
        }
        *++last_le = *x_i;
        tail = last_le == back ? que : last_le + 1; 
      }
    }
  }
}//MaxPoolOpt

template <typename T>
struct MaxPool1DTaskOpt {

  const T* X_data;
  T* Y_data;
  int64_t x_h;
  int64_t y_h;
  int64_t pad_front;
  int64_t pad_back;
  int64_t dilation;
  int64_t pool_size;

  TensorOpCost Cost() const {
    auto loop_count = static_cast<double>(x_h);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    std::unique_ptr<T[]> que_ptr{new T[pool_size]};
    for (int64_t c = begin; c < end; ++c) {
      operator()(c, que_ptr);
    }//for
  }//operator

  void operator()(std::ptrdiff_t c, std::unique_ptr<T[]>& que_ptr) const {
    MaxPoolOpt(X_data + c * x_h,
               X_data + (c + 1) * x_h, 1,
               dilation, pad_front, pad_back, pool_size,
               Y_data + c * y_h, 1,
               que_ptr.get());
  }
};

template <typename T>
struct MaxPool2DTaskOpt {

  const T* X_data;
  T* Y_data;
  int64_t x_h;
  int64_t x_w;
  int64_t y_h;
  int64_t y_w;
  int64_t pad_h_front;
  int64_t pad_h_back;
  int64_t pad_w_front;
  int64_t pad_w_back;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pool_h;
  int64_t pool_w;

  TensorOpCost Cost() const {
    auto loop_count = static_cast<double>(x_h * x_w);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    std::unique_ptr<T[]> que_ptr{new T[std::max(pool_h, pool_w)]};
    std::unique_ptr<T[]> y_temp{new T[x_h * y_w]};
    for (int64_t c = begin; c < end; ++c) {
      operator()(c, que_ptr, y_temp);
    }//for
  }//operator

  void operator()(std::ptrdiff_t c, std::unique_ptr<T[]>& que_ptr, std::unique_ptr<T[]>& y_temp) const {

    auto x = X_data + c * x_h * x_w;
    auto y = Y_data + c * y_h * y_w;
    //reduce width
    for (int h_i = 0; h_i < x_h; ++h_i) {
      MaxPoolOpt(x + h_i * x_w,
                 x + (h_i + 1) * x_w,
                 1, dilation_w, pad_w_front, pad_w_back, pool_w,
                 y_temp.get() + h_i * y_w, 1, que_ptr.get());
    }
    //reduce height
    for (int w_i = 0; w_i < y_w; ++w_i) {
      MaxPoolOpt(y_temp.get() + w_i,
                 y_temp.get() + w_i + x_h * y_w,
                 y_w, dilation_h, pad_h_front, pad_h_back,
                 pool_h, y + w_i, y_w, que_ptr.get());
    }
  }
};

template <typename T>
struct MaxPool3DTaskOpt {

  const T* X_data;
  T* Y_data;
  int64_t x_h;
  int64_t x_w;
  int64_t x_d;
  int64_t y_h;
  int64_t y_w;
  int64_t y_d;
  int64_t pad_h_front;
  int64_t pad_h_back;
  int64_t pad_w_front;
  int64_t pad_w_back;
  int64_t pad_d_front;
  int64_t pad_d_back;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t dilation_d;
  int64_t pool_h;
  int64_t pool_w;
  int64_t pool_d;

  TensorOpCost Cost() const {
    auto loop_count = static_cast<double>(x_h * x_w * x_d);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    std::unique_ptr<T[]> que_ptr{new T[std::max(pool_h, std::max(pool_w, pool_d))]};
    std::unique_ptr<T[]> y_temp_1{new T[x_h * x_w * y_d]};
    std::unique_ptr<T[]> y_temp_2{new T[x_h * y_w * y_d]};
    for (int64_t c = begin; c < end; ++c) {
      operator()(c, que_ptr, y_temp_1, y_temp_2);
    }//for
  }//operator

  void operator()(std::ptrdiff_t c,
                  std::unique_ptr<T[]>& que_ptr,
                  std::unique_ptr<T[]>& y_temp_1,
                  std::unique_ptr<T[]>& y_temp_2) const {

    auto x = X_data + c * x_h * x_w * x_d;
    auto y = Y_data + c * y_h * y_w * y_d;
    //reduce depth
    for (int h_i = 0; h_i < x_h; h_i++) {
      for (int w_i = 0; w_i < x_w; w_i++) {
        MaxPoolOpt(x + h_i * x_w * x_d + w_i * x_d,
                   x + h_i * x_w * x_d + w_i * x_d + x_d,
                   1, dilation_d, pad_d_front, pad_d_back, pool_d,
                   y_temp_1.get() + h_i * x_w * y_d + w_i * y_d, 1, que_ptr.get());
      }
    }
    //reduce width
    for (int h_i = 0; h_i < x_h; h_i++) {
      for (int d_i = 0; d_i < y_d; d_i++) {
        MaxPoolOpt(y_temp_1.get() + h_i * x_w * y_d + d_i,
                   y_temp_1.get() + (h_i + 1) * x_w * y_d + d_i,
                   y_d, dilation_w, pad_w_front, pad_w_back, pool_w,
                   y_temp_2.get() + h_i * y_w * y_d + d_i, y_d, que_ptr.get());
      }
    }
    //reduce height
    for (int w_i = 0; w_i < y_w; w_i++) {
      for (int d_i = 0; d_i < y_d; d_i++) {
        MaxPoolOpt(y_temp_2.get() + w_i * y_d + d_i,
                   y_temp_2.get() + x_h * y_w * y_d + w_i * y_d + d_i,
                   y_w * y_d, dilation_h, pad_h_front, pad_h_back, pool_h,
                   y + w_i * y_d + d_i, y_w * y_d, que_ptr.get());
      }
    }//for
  }
};

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <algorithm>
#include <limits>
#include <stdint.h>

extern "C" {
void average_pool_f32(void *);
void max_pool_f32(void *);
}

// Core operator implementations
// Pool1D implementation
// isGlobalPool - true if GlobalMaxPool or GlobalAveragePool, false otherwise
template <typename PoolType>
void pool1D_f32(bool isGlobalPool, float *X, int *X_shape, float *Y,
                int *Y_shape, int *kernel_shape, int *pads, int *strides,
                bool count_include_pad) {
  int batch_size = X_shape[0];
  int channels = X_shape[1];
  int height = X_shape[2];
  int pooled_height = Y_shape[2];
  int stride_h = isGlobalPool ? 1 : strides[0];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + kernel_shape[0], height);
        hstart = std::max(hstart, 0);
        float Yh = PoolType::Initialize();
        for (int h = hstart; h < hend; ++h) {
          PoolType::Process(X[h], Yh);
        }
        if (count_include_pad) {
          PoolType::Finalize(kernel_shape[0], Yh);
        } else {
          PoolType::Finalize(hend - hstart, Yh);
        }
        Y[ph] = Yh;
      }
      // Do offset.
      X += height;
      Y += pooled_height;
    }
  }
}

// Pool2D implementation
// isGlobalPool - true if GlobalMaxPool or GlobalAveragePool, false otherwise
template <typename PoolType>
void pool2D_f32(bool isGlobalPool, float *X, int *X_shape, float *Y,
                int *Y_shape, int *kernel_shape, int *pads, int *strides,
                bool count_include_pad) {
  int batch_size = X_shape[0];
  int channels = X_shape[1];
  int height = X_shape[2];
  int width = X_shape[3];
  int pooled_height = Y_shape[2];
  int pooled_width = Y_shape[3];
  int stride_h = isGlobalPool ? 1 : strides[0];
  int stride_w = isGlobalPool ? 1 : strides[1];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + kernel_shape[0], height);
        hstart = std::max(hstart, 0);
        for (int pw = 0; pw < pooled_width; ++pw) {
          int wstart = pw * stride_w - pads[1];
          int wend = std::min(wstart + kernel_shape[1], width);
          wstart = std::max(wstart, 0);
          const int pool_index = ph * pooled_width + pw;
          float Yh = PoolType::Initialize();
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              PoolType::Process(X[input_index], Yh);
            }
          }
          if (count_include_pad) {
            PoolType::Finalize(kernel_shape[0] * kernel_shape[1], Yh);
          } else {
            PoolType::Finalize((hend - hstart) * (wend - wstart), Yh);
          }
          Y[pool_index] = Yh;
        }
      }
      // Do offset.
      X += height * width;
      Y += pooled_height * pooled_width;
    }
  }
}

// Pool3D - implementation
// isGlobalPool - true if GlobalMaxPool or GlobalAveragePool, false otherwise
template <typename PoolType>
void pool3D_f32(bool isGlobalPool, float *X, int *X_shape, float *Y,
                int *Y_shape, int *kernel_shape, int *pads, int *strides,
                bool count_include_pad) {
  int batch_size = X_shape[0];
  int channels = X_shape[1];
  int height = X_shape[2];
  int width = X_shape[3];
  int depth = X_shape[4];
  int pooled_height = Y_shape[2];
  int pooled_width = Y_shape[3];
  int pooled_depth = Y_shape[4];
  int stride_h = isGlobalPool ? 1 : strides[0];
  int stride_w = isGlobalPool ? 1 : strides[1];
  int stride_d = isGlobalPool ? 1 : strides[2];

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + kernel_shape[0], height);
        hstart = std::max(hstart, 0);
        for (int pw = 0; pw < pooled_width; ++pw) {
          int wstart = pw * stride_w - pads[1];
          int wend = std::min(wstart + kernel_shape[1], width);
          wstart = std::max(wstart, 0);
          for (int pd = 0; pd < pooled_depth; ++pd) {
            int dstart = pd * stride_d - pads[2];
            int dend = std::min(dstart + kernel_shape[2], depth);
            dstart = std::max(dstart, 0);
            const int pool_index =
                ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
            float Yh = PoolType::Initialize();
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                for (int d = dstart; d < dend; ++d) {
                  const int input_index = h * width * depth + w * depth + d;
                  PoolType::Process(X[input_index], Yh);
                }
              }
            }
            if (count_include_pad) {
              PoolType::Finalize(
                  kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh);
            } else {
              PoolType::Finalize(
                  (hend - hstart) * (wend - wstart) * (dend - dstart), Yh);
            }
            Y[pool_index] = Yh;
          }
        }
      }
      // Do offset.
      X += height * width * depth;
      Y += pooled_height * pooled_width * pooled_depth;
    }
  }
}

// Core pool classes
class AveragePool {
public:
  static float Initialize() { return 0; }

  template <typename T> static void Process(const T &x_data, T &y_data) {
    y_data += x_data;
  }

  template <typename T> static void Finalize(const int32_t size, T &y_data) {
    y_data /= size;
  }
};

class MaxPool {
public:
  static float Initialize() { return std::numeric_limits<float>::lowest(); }

  template <typename T> static void Process(const T &x_data, T &y_data) {
    if (x_data > y_data) {
      y_data = x_data;
    }
  }

  template <typename T>
  static void Finalize(const int32_t /*size*/, T & /*y_data*/) {}
};

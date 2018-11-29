// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "pool.h"
#include "common.h"
#include <algorithm>

// Wasm interop method
void pool_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  uint32_t pool_size = PARAM_INT32(data, dataIndex[1]);

  switch (pool_size) {
  case 1:
    pool1D_f32_imp(
        PARAM_BOOL(data, dataIndex[2]), PARAM_INT32(data, dataIndex[3]),
        PARAM_FLOAT_PTR(data, dataIndex[4]),
        PARAM_INT32_PTR(data, dataIndex[5]),
        PARAM_FLOAT_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]),
        PARAM_INT32_PTR(data, dataIndex[10]), PARAM_BOOL(data, dataIndex[11]));
    break;
  case 2:
    pool2D_f32_imp(
        PARAM_BOOL(data, dataIndex[2]), PARAM_INT32(data, dataIndex[3]),
        PARAM_FLOAT_PTR(data, dataIndex[4]),
        PARAM_INT32_PTR(data, dataIndex[5]),
        PARAM_FLOAT_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]),
        PARAM_INT32_PTR(data, dataIndex[10]), PARAM_BOOL(data, dataIndex[11]));
    break;
  case 3:
    pool3D_f32_imp(
        PARAM_BOOL(data, dataIndex[2]), PARAM_INT32(data, dataIndex[3]),
        PARAM_FLOAT_PTR(data, dataIndex[4]),
        PARAM_INT32_PTR(data, dataIndex[5]),
        PARAM_FLOAT_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]),
        PARAM_INT32_PTR(data, dataIndex[10]), PARAM_BOOL(data, dataIndex[11]));
    break;
  default:
    throw "Unsupported pooling size";
  }
}

// Core operator implementations
// Pool1D - implementation
// isGlobalPool - true if GlobalMaxPool or GlobalAveragePool, false otherwise
// poolType - 1 if average pool, 2 if max pool
void pool1D_f32_imp(bool isGlobalPool, int poolType, float *X, int *X_shape,
                    float *Y, int *Y_shape, int *kernel_shape, int *pads,
                    int *strides, bool count_include_pad) {
  int batch_size = X_shape[0];
  int channels = X_shape[1];
  int height = X_shape[2];
  int pooled_height = Y_shape[2];
  int stride_h = isGlobalPool ? 1 : strides[0];

  float (*PoolInitialize)();
  void (*PoolProcess)(const float &, float &);
  void (*PoolFinalize)(const int, float &);

  switch (poolType) {
  case 1:
    PoolInitialize = AveragePoolInitialize;
    PoolProcess = AveragePoolProcess;
    PoolFinalize = AveragePoolFinalize;
    break;
  case 2:
    PoolInitialize = MaxPoolInitialize;
    PoolProcess = MaxPoolProcess;
    PoolFinalize = MaxPoolFinalize;
    break;
  default:
    throw "Unsupported pool type";
  }

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + kernel_shape[0], height);
        hstart =
            std::max(static_cast<int64_t>(hstart), static_cast<int64_t>(0));
        float Yh = (*PoolInitialize)();
        for (int h = hstart; h < hend; ++h) {
          (*PoolProcess)(X[h], Yh);
        }
        if (count_include_pad) {
          (*PoolFinalize)(kernel_shape[0], Yh);
        } else {
          (*PoolFinalize)(hend - hstart, Yh);
        }
        Y[ph] = Yh;
      }
      // Do offset.
      X += height;
      Y += pooled_height;
    }
  }
}

// Pool2D - implementation
// isGlobalPool - true if GlobalMaxPool or GlobalAveragePool, false otherwise
// poolType - 1 if average pool, 2 if max pool
void pool2D_f32_imp(bool isGlobalPool, int poolType, float *X, int *X_shape,
                    float *Y, int *Y_shape, int *kernel_shape, int *pads,
                    int *strides, bool count_include_pad) {
  int batch_size = X_shape[0];
  int channels = X_shape[1];
  int height = X_shape[2];
  int width = X_shape[3];
  int pooled_height = Y_shape[2];
  int pooled_width = Y_shape[3];
  int stride_h = isGlobalPool ? 1 : strides[0];
  int stride_w = isGlobalPool ? 1 : strides[1];

  float (*PoolInitialize)();
  void (*PoolProcess)(const float &, float &);
  void (*PoolFinalize)(const int, float &);

  switch (poolType) {
  case 1:
    PoolInitialize = AveragePoolInitialize;
    PoolProcess = AveragePoolProcess;
    PoolFinalize = AveragePoolFinalize;
    break;
  case 2:
    PoolInitialize = MaxPoolInitialize;
    PoolProcess = MaxPoolProcess;
    PoolFinalize = MaxPoolFinalize;
    break;
  default:
    throw "Unsupported pool type";
  }

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + kernel_shape[0], height);
        hstart =
            std::max(static_cast<int64_t>(hstart), static_cast<int64_t>(0));
        for (int pw = 0; pw < pooled_width; ++pw) {
          int wstart = pw * stride_w - pads[1];
          int wend = std::min(wstart + kernel_shape[1], width);
          wstart =
              std::max(static_cast<int64_t>(wstart), static_cast<int64_t>(0));
          const int pool_index = ph * pooled_width + pw;
          float Yh = (*PoolInitialize)();
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int input_index = h * width + w;
              (*PoolProcess)(X[input_index], Yh);
            }
          }
          if (count_include_pad) {
            (*PoolFinalize)(kernel_shape[0] * kernel_shape[1], Yh);
          } else {
            (*PoolFinalize)((hend - hstart) * (wend - wstart), Yh);
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
// poolType - 1 if average pool, 2 if max pool
void pool3D_f32_imp(bool isGlobalPool, int poolType, float *X, int *X_shape,
                    float *Y, int *Y_shape, int *kernel_shape, int *pads,
                    int *strides, bool count_include_pad) {
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

  float (*PoolInitialize)();
  void (*PoolProcess)(const float &, float &);
  void (*PoolFinalize)(const int, float &);

  switch (poolType) {
  case 1:
    PoolInitialize = AveragePoolInitialize;
    PoolProcess = AveragePoolProcess;
    PoolFinalize = AveragePoolFinalize;
    break;
  case 2:
    PoolInitialize = MaxPoolInitialize;
    PoolProcess = MaxPoolProcess;
    PoolFinalize = MaxPoolFinalize;
    break;
  default:
    throw "Unsupported pool type";
  }

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < pooled_height; ++ph) {
        int hstart = ph * stride_h - pads[0];
        int hend = std::min(hstart + kernel_shape[0], height);
        hstart =
            std::max(static_cast<int64_t>(hstart), static_cast<int64_t>(0));
        for (int pw = 0; pw < pooled_width; ++pw) {
          int wstart = pw * stride_w - pads[1];
          int wend = std::min(wstart + kernel_shape[1], width);
          wstart =
              std::max(static_cast<int64_t>(wstart), static_cast<int64_t>(0));
          for (int pd = 0; pd < pooled_depth; ++pd) {
            int dstart = pd * stride_d - pads[2];
            int dend = std::min(dstart + kernel_shape[2], depth);
            dstart =
                std::max(static_cast<int64_t>(dstart), static_cast<int64_t>(0));
            const int pool_index =
                ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
            float Yh = (*PoolInitialize)();
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                for (int d = dstart; d < dend; ++d) {
                  const int input_index = h * width * depth + w * depth + d;
                  (*PoolProcess)(X[input_index], Yh);
                }
              }
            }
            if (count_include_pad) {
              (*PoolFinalize)(
                  kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh);
            } else {
              (*PoolFinalize)(
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

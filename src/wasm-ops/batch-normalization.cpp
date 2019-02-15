// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "batch-normalization.h"
#include "common.h"
#include <math.h>

// Wasm interop method
void batch_normalization_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  batch_normalization_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_FLOAT_PTR(data, dataIndex[2]),
      PARAM_INT32(data, dataIndex[3]), PARAM_INT32(data, dataIndex[4]),
      PARAM_INT32(data, dataIndex[5]), PARAM_FLOAT_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_FLOAT_PTR(data, dataIndex[8]),
      PARAM_FLOAT_PTR(data, dataIndex[9]), PARAM_FLOAT(data, dataIndex[10]));
}

// Core operator implementation
void batch_normalization_f32_imp(float *X, float *Y, int32_t batch_size,
                                 int32_t num_channels, int32_t channel_size,
                                 float *scale, float *bias, float *mean,
                                 float *variance, float epsilon) {
  for (size_t nc = 0; nc < batch_size * num_channels; ++nc) {
    for (size_t i = 0; i < channel_size; ++i) {
      Y[nc * channel_size + i] =
          scale[nc % num_channels] *
              ((X[nc * channel_size + i] - mean[nc % num_channels]) /
               sqrt(variance[nc % num_channels] + epsilon)) +
          bias[nc % num_channels];
    }
  }
}

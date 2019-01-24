// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "instance-normalization.h"
#include "common.h"
#include <math.h>

// Wasm interop method
void instance_normalization_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  instance_normalization_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_FLOAT_PTR(data, dataIndex[2]),
      PARAM_INT32(data, dataIndex[3]), PARAM_INT32(data, dataIndex[4]),
      PARAM_INT32(data, dataIndex[5]), PARAM_FLOAT_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_FLOAT(data, dataIndex[8]));
}

// Core operator implementation
void instance_normalization_f32_imp(float *X, float *Y, int32_t batch_size,
                                 int32_t num_channels, int32_t channel_size,
                                 float *scale, float *bias, float epsilon) {
  float temp;
  float mean;
  float variance;
  size_t sampleOffset;
  size_t physicalOffset;
  size_t iterEnd;
  size_t sample_size = channel_size * num_channels;

  for (size_t n = 0; n < batch_size; ++n) {
    sampleOffset = n * sample_size;
    for (size_t c = 0; c < num_channels; ++c) {
      physicalOffset = sampleOffset + c * channel_size;
      iterEnd = physicalOffset + channel_size;

      // compute mean for this channel
      temp = 0;
      for (int32_t i = physicalOffset; i < iterEnd; ++i) {
        temp += X[i];
      }
      mean = temp / channel_size;

      // compute variance for this channel
      temp = 0;
      for (size_t i = physicalOffset; i < iterEnd; ++i) {
        temp += pow(X[i] - mean, 2);
      }
      variance = temp / channel_size;

      // compute normalized value for data in this channel
      for (size_t i = physicalOffset; i < iterEnd; ++i) {
        Y[i] = scale[c] * ((X[i] - mean) / sqrt(variance + epsilon)) + bias[c];
      }
    }
  }
}

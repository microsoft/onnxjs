// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "softmax.h"
#include "common.h"
#include <math.h>

// Wasm interop method
void softmax_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  softmax_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_FLOAT_PTR(data, dataIndex[2]),
      PARAM_INT32(data, dataIndex[3]), PARAM_INT32(data, dataIndex[4]));
}

// Core operator implementation
void softmax_f32_imp(float *X, float *Y, int32_t N, int32_t D) {
  for (size_t i = 0; i < N; i++) {
    // find row offset
    int offset = i * D;

    // find max of each logical row
    float max = std::numeric_limits<float>::lowest();
    for (size_t j = 0; j < D; j++) {
      if (X[offset + j] > max)
        max = X[offset + j];
    }

    // find normalization scale per row
    float scale = 0;
    for (size_t j = 0; j < D; j++) {
      Y[offset + j] = exp(X[offset + j] - max);
      scale += Y[offset + j];
    }

    // perform the softmax normalization
    for (size_t j = 0; j < D; j++) {
      // If scale is 0, then all elements in that row are 0, so no normalization
      // operation required
      if (scale != 0)
        Y[offset + j] /= scale;
    }
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "sum.h"
#include "common.h"

// Wasm interop method
void sum_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  sum_f32_imp(PARAM_INT32(data, dataIndex[1]), PARAM_INT32(data, dataIndex[2]),
              PARAM_FLOAT_PTR(data, dataIndex[3]),
              PARAM_FLOAT_PTR(data, dataIndex[4]));
}

// Core operator implementation
void sum_f32_imp(int num_tensors, int size, float *Y, float *X) {
  for (size_t i = 0; i < num_tensors; ++i) {
    for (size_t j = 0; j < size; ++j)
      Y[j] += X[i * size + j];
  }
}

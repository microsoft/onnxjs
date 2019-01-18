// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "pool.h"
#include "common.h"

// Wasm interop method
void average_pool_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  uint32_t pool_size = PARAM_INT32(data, dataIndex[1]);

  switch (pool_size) {
  case 1:
    pool1D_f32<AveragePool>(
        PARAM_BOOL(data, dataIndex[2]), PARAM_FLOAT_PTR(data, dataIndex[3]),
        PARAM_INT32_PTR(data, dataIndex[4]),
        PARAM_FLOAT_PTR(data, dataIndex[5]),
        PARAM_INT32_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]), PARAM_BOOL(data, dataIndex[10]));
    break;
  case 2:
    pool2D_f32<AveragePool>(
        PARAM_BOOL(data, dataIndex[2]), PARAM_FLOAT_PTR(data, dataIndex[3]),
        PARAM_INT32_PTR(data, dataIndex[4]),
        PARAM_FLOAT_PTR(data, dataIndex[5]),
        PARAM_INT32_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]), PARAM_BOOL(data, dataIndex[10]));
    break;
  case 3:
    pool3D_f32<AveragePool>(
        PARAM_BOOL(data, dataIndex[2]), PARAM_FLOAT_PTR(data, dataIndex[3]),
        PARAM_INT32_PTR(data, dataIndex[4]),
        PARAM_FLOAT_PTR(data, dataIndex[5]),
        PARAM_INT32_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]), PARAM_BOOL(data, dataIndex[10]));
    break;
  default:
    throw "Unsupported pooling size";
  }
}

void max_pool_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  uint32_t pool_size = PARAM_INT32(data, dataIndex[1]);

  switch (pool_size) {
  case 1:
    pool1D_f32<MaxPool>(
        PARAM_BOOL(data, dataIndex[2]), PARAM_FLOAT_PTR(data, dataIndex[3]),
        PARAM_INT32_PTR(data, dataIndex[4]),
        PARAM_FLOAT_PTR(data, dataIndex[5]),
        PARAM_INT32_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]), PARAM_BOOL(data, dataIndex[10]));
    break;
  case 2:
    pool2D_f32<MaxPool>(
        PARAM_BOOL(data, dataIndex[2]), PARAM_FLOAT_PTR(data, dataIndex[3]),
        PARAM_INT32_PTR(data, dataIndex[4]),
        PARAM_FLOAT_PTR(data, dataIndex[5]),
        PARAM_INT32_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]), PARAM_BOOL(data, dataIndex[10]));
    break;
  case 3:
    pool3D_f32<MaxPool>(
        PARAM_BOOL(data, dataIndex[2]), PARAM_FLOAT_PTR(data, dataIndex[3]),
        PARAM_INT32_PTR(data, dataIndex[4]),
        PARAM_FLOAT_PTR(data, dataIndex[5]),
        PARAM_INT32_PTR(data, dataIndex[6]),
        PARAM_INT32_PTR(data, dataIndex[7]),
        PARAM_INT32_PTR(data, dataIndex[8]),
        PARAM_INT32_PTR(data, dataIndex[9]), PARAM_BOOL(data, dataIndex[10]));
    break;
  default:
    throw "Unsupported pooling size";
  }
}

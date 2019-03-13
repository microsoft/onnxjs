// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "clip.h"

// Wasm interop methods
void clip_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input = PARAM_FLOAT_PTR(data, dataIndex[1]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[2]);
  const int32_t length = PARAM_INT32(data, dataIndex[3]);
  const float min = PARAM_FLOAT(data, dataIndex[4]);
  const float max = PARAM_FLOAT(data, dataIndex[5]);
  clip_imp<float>(input, output, length, min, max);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "binary-op.h"
#include "common.h"

// Wasm interop methods
void add_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_imp<float, Add>(data, input_1, input_2, output);
}
void sub_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_imp<float, Sub>(data, input_1, input_2, output);
}
void mul_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_imp<float, Mul>(data, input_1, input_2, output);
}
void div_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_imp<float, Div>(data, input_1, input_2, output);
}
void prelu_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_imp<float, PRelu>(data, input_1, input_2, output);
}
void xor_u8(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  binary_imp<uint8_t, Xor>(data, input_1, input_2, output);
}
void or_u8(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  binary_imp<uint8_t, Or>(data, input_1, input_2, output);
}
void and_u8(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  binary_imp<uint8_t, And>(data, input_1, input_2, output);
}

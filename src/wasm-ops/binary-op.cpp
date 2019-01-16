// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "binary-op.h"
#include "common.h"

// Core op
float add_core(const float &a, const float &b) { return a + b; }
float sub_core(const float &a, const float &b) { return a - b; }
float mul_core(const float &a, const float &b) { return a * b; }
float div_core(const float &a, const float &b) { return a / b; }
float prelu_core(const float &a, const float &b) { return a >= 0 ? a : a * b; }
uint8_t xor_core(const uint8_t &a, const uint8_t &b) { return a ^ b; }
uint8_t or_core(const uint8_t &a, const uint8_t &b) { return a || b; }
uint8_t and_core(const uint8_t &a, const uint8_t &b) { return a && b; }

// Wasm interop methods
void add_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_wrapper<float>(data, dataIndex, input_1, input_2, output, add_core);
}
void sub_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_wrapper<float>(data, dataIndex, input_1, input_2, output, sub_core);
}
void mul_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_wrapper<float>(data, dataIndex, input_1, input_2, output, mul_core);
}
void div_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_wrapper<float>(data, dataIndex, input_1, input_2, output, div_core);
}
void prelu_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  binary_wrapper<float>(data, dataIndex, input_1, input_2, output, prelu_core);
}
void xor_(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  binary_wrapper<uint8_t>(data, dataIndex, input_1, input_2, output, xor_core);
}
void or_(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  binary_wrapper<uint8_t>(data, dataIndex, input_1, input_2, output, or_core);
}
void and_(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  binary_wrapper<uint8_t>(data, dataIndex, input_1, input_2, output, and_core);
}

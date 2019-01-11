// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "binary-op.h"
#include "common.h"
#include "utils/broadcast_utils.h"
#include "utils/shape_utils.h"

// Core op
float add_core(const float &a, const float &b) { return a + b; }
float sub_core(const float &a, const float &b) { return a - b; }
float mul_core(const float &a, const float &b) { return a * b; }
float div_core(const float &a, const float &b) { return a / b; }
float prelu_core(const float &a, const float &b) { return a >= 0 ? a : a * b; }
uint8_t xor_core(const uint8_t &a, const uint8_t &b) { return a ^ b; }
uint8_t or_core(const uint8_t &a, const uint8_t &b) { return a || b; }
uint8_t and_core(const uint8_t &a, const uint8_t &b) { return a && b; }

// Core binary operator implementation
void binary_f32_input_f32_output_imp(
    const float *input_1, const int32_t &rank_1,
    const std::vector<int32_t> &dims_1, const float *input_2,
    const int32_t &rank2, const std::vector<int32_t> &dims_2, float *output,
    const int32_t &output_length, const int32_t &output_rank,
    const std::vector<int32_t> &output_dims,
    float (*core_op)(const float &, const float &)) {
  const std::vector<int32_t> strides1 = ShapeUtils::compute_strides(dims_1);
  const std::vector<int32_t> strides2 = ShapeUtils::compute_strides(dims_2);
  const std::vector<int32_t> output_strides =
      ShapeUtils::compute_strides(output_dims);

  for (size_t i = 0; i < output_length; ++i) {
    auto broadcasted_indices = ShapeUtils::offset_to_indices(output_strides, i);
    auto indices1 = BroadcastUtils::broadcasted_to_original_indices(
        broadcasted_indices, dims_1);
    auto offset1 = ShapeUtils::indices_to_offset(strides1, indices1);
    auto indices2 = BroadcastUtils::broadcasted_to_original_indices(
        broadcasted_indices, dims_2);
    auto offset2 = ShapeUtils::indices_to_offset(strides2, indices2);
    output[i] = core_op(input_1[offset1], input_2[offset2]);
  }
}

void binary_bool_input_bool_output_imp(
    const uint8_t *input_1, const int32_t &rank_1,
    const std::vector<int32_t> &dims_1, const uint8_t *input_2,
    const int32_t &rank2, const std::vector<int32_t> &dims_2, uint8_t *output,
    const int32_t &output_length, const int32_t &output_rank,
    const std::vector<int32_t> &output_dims,
    uint8_t (*core_op)(const uint8_t &, const uint8_t &)) {
  const std::vector<int32_t> strides1 = ShapeUtils::compute_strides(dims_1);
  const std::vector<int32_t> strides2 = ShapeUtils::compute_strides(dims_2);
  const std::vector<int32_t> output_strides =
      ShapeUtils::compute_strides(output_dims);

  for (size_t i = 0; i < output_length; ++i) {
    auto broadcasted_indices = ShapeUtils::offset_to_indices(output_strides, i);
    auto indices1 = BroadcastUtils::broadcasted_to_original_indices(
        broadcasted_indices, dims_1);
    auto offset1 = ShapeUtils::indices_to_offset(strides1, indices1);
    auto indices2 = BroadcastUtils::broadcasted_to_original_indices(
        broadcasted_indices, dims_2);
    auto offset2 = ShapeUtils::indices_to_offset(strides2, indices2);
    output[i] = core_op(input_1[offset1], input_2[offset2]);
  }
}

// Core binary operator wrapper (Does some pre-processing prior to calling the
// core function)
void binary_f32_input_f32_output_wrapper(void *data,
                                         float (*core_op)(const float &,
                                                          const float &)) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const int32_t rank_1 = PARAM_INT32(data, dataIndex[2]);
  const int32_t *dims_1 = PARAM_INT32_PTR(data, dataIndex[3]);
  std::vector<int32_t> dims1_vector;
  if (rank_1 > 0) {
    dims1_vector.resize(rank_1);
    for (int32_t i = 0; i < rank_1; ++i) {
      dims1_vector[i] = dims_1[i];
    }
  }
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  const int32_t rank2 = PARAM_INT32(data, dataIndex[5]);
  const int32_t *dims_2 = PARAM_INT32_PTR(data, dataIndex[6]);
  std::vector<int32_t> dims2_vector;
  if (rank2 > 0) {
    dims2_vector.resize(rank2);
    for (int32_t i = 0; i < rank2; ++i) {
      dims2_vector[i] = dims_2[i];
    }
  }
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  const int32_t output_length = PARAM_INT32(data, dataIndex[8]);
  const int32_t output_rank = PARAM_INT32(data, dataIndex[9]);
  const int32_t *output_dims = PARAM_INT32_PTR(data, dataIndex[10]);
  std::vector<int32_t> output_dims_vector;
  if (output_rank != 0) {
    output_dims_vector.resize(output_rank);
    for (int32_t i = 0; i < output_rank; ++i) {
      output_dims_vector[i] = output_dims[i];
    }
  }
  binary_f32_input_f32_output_imp(input_1, rank_1, dims1_vector, input_2, rank2,
                                  dims2_vector, output, output_length,
                                  output_rank, output_dims_vector, core_op);
}

void binary_bool_input_bool_output_wrapper(
    void *data, uint8_t (*core_op)(const uint8_t &, const uint8_t &)) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  const uint8_t *input_1 = PARAM_BOOL_PTR(data, dataIndex[1]);
  const int32_t rank_1 = PARAM_INT32(data, dataIndex[2]);
  const int32_t *dims_1 = PARAM_INT32_PTR(data, dataIndex[3]);
  std::vector<int32_t> dims1_vector;
  if (rank_1 > 0) {
    dims1_vector.resize(rank_1);
    for (int32_t i = 0; i < rank_1; ++i) {
      dims1_vector[i] = dims_1[i];
    }
  }
  const uint8_t *input_2 = PARAM_BOOL_PTR(data, dataIndex[4]);
  const int32_t rank2 = PARAM_INT32(data, dataIndex[5]);
  const int32_t *dims_2 = PARAM_INT32_PTR(data, dataIndex[6]);
  std::vector<int32_t> dims2_vector;
  if (rank2 > 0) {
    dims2_vector.resize(rank2);
    for (int32_t i = 0; i < rank2; ++i) {
      dims2_vector[i] = dims_2[i];
    }
  }
  uint8_t *output = PARAM_BOOL_PTR(data, dataIndex[7]);
  const int32_t output_length = PARAM_INT32(data, dataIndex[8]);
  const int32_t output_rank = PARAM_INT32(data, dataIndex[9]);
  const int32_t *output_dims = PARAM_INT32_PTR(data, dataIndex[10]);
  std::vector<int32_t> output_dims_vector;
  if (output_rank != 0) {
    output_dims_vector.resize(output_rank);
    for (int32_t i = 0; i < output_rank; ++i) {
      output_dims_vector[i] = output_dims[i];
    }
  }
  binary_bool_input_bool_output_imp(input_1, rank_1, dims1_vector, input_2,
                                    rank2, dims2_vector, output, output_length,
                                    output_rank, output_dims_vector, core_op);
}

// Wasm interop methods
void add_f32(void *data) {
  binary_f32_input_f32_output_wrapper(data, add_core);
}
void sub_f32(void *data) {
  binary_f32_input_f32_output_wrapper(data, sub_core);
}
void mul_f32(void *data) {
  binary_f32_input_f32_output_wrapper(data, mul_core);
}
void div_f32(void *data) {
  binary_f32_input_f32_output_wrapper(data, div_core);
}
void prelu_f32(void *data) {
  binary_f32_input_f32_output_wrapper(data, prelu_core);
}
void xor_(void *data) { binary_bool_input_bool_output_wrapper(data, xor_core); }
void or_(void *data) { binary_bool_input_bool_output_wrapper(data, or_core); }
void and_(void *data) { binary_bool_input_bool_output_wrapper(data, and_core); }

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common.h"
#include "utils/broadcast_utils.h"
#include "utils/shape_utils.h"
#include <stdint.h>
#include <vector>

extern "C" {
// Arithmetic ops
void add_f32(void *);
void sub_f32(void *);
void mul_f32(void *);
void div_f32(void *);
void prelu_f32(void *);

// Logical ops
void xor_(void *);
void or_(void *);
void and_(void *);
}

// Core binary operator implementation
template <class T>
void binary_imp(const T *input_1, const int32_t &rank_1,
                const std::vector<int32_t> &dims_1, const T *input_2,
                const int32_t &rank_2, const std::vector<int32_t> &dims_2,
                T *output, const int32_t &output_length,
                const int32_t &output_rank,
                const std::vector<int32_t> &output_dims,
                T (*core_op)(const T &, const T &)) {
  const std::vector<int32_t> strides_1 = ShapeUtils::compute_strides(dims_1);
  const std::vector<int32_t> strides_2 = ShapeUtils::compute_strides(dims_2);
  const std::vector<int32_t> output_strides =
      ShapeUtils::compute_strides(output_dims);
  std::vector<int32_t> indices_1(rank_1);
  std::vector<int32_t> indices_2(rank_2);
  std::vector<int32_t> broadcasted_indices(output_strides.size());

  for (size_t i = 0; i < output_length; ++i) {
    ShapeUtils::offset_to_indices(output_strides, i, broadcasted_indices);
    BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices, dims_1,
                                                    indices_1);
    auto offset1 = ShapeUtils::indices_to_offset(strides_1, indices_1);
    BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices, dims_2,
                                                    indices_2);
    auto offset2 = ShapeUtils::indices_to_offset(strides_2, indices_2);
    output[i] = core_op(input_1[offset1], input_2[offset2]);
  }
}

// Core binary operator wrapper (Does some pre-processing prior to calling the
// core implementation function)
template <class T>
void binary_wrapper(void *data, uint32_t *dataIndex, const T *input_1,
                    const T *input_2, T *output,
                    T (*core_op)(const T &, const T &)) {
  const int32_t rank_1 = PARAM_INT32(data, dataIndex[2]);
  const int32_t *dims_1 = PARAM_INT32_PTR(data, dataIndex[3]);
  std::vector<int32_t> dims1_vector;
  if (rank_1 > 0) {
    dims1_vector.resize(rank_1);
    for (int32_t i = 0; i < rank_1; ++i) {
      dims1_vector[i] = dims_1[i];
    }
  }

  const int32_t rank2 = PARAM_INT32(data, dataIndex[5]);
  const int32_t *dims_2 = PARAM_INT32_PTR(data, dataIndex[6]);
  std::vector<int32_t> dims2_vector;
  if (rank2 > 0) {
    dims2_vector.resize(rank2);
    for (int32_t i = 0; i < rank2; ++i) {
      dims2_vector[i] = dims_2[i];
    }
  }

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

  binary_imp<T>(input_1, rank_1, dims1_vector, input_2, rank2, dims2_vector,
                output, output_length, output_rank, output_dims_vector,
                core_op);
}

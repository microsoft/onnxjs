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
void xor_u8(void *);
void or_u8(void *);
void and_u8(void *);
}

// Binary operator (with broadcasting)
template <typename T, typename BinaryOp>
void binary_imp(void *data, const T *input_1, const T *input_2, T *output) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);

  // first input related
  const int32_t rank_1 = PARAM_INT32(data, dataIndex[2]);
  const int32_t *dims_1 = PARAM_INT32_PTR(data, dataIndex[3]);
  std::vector<int32_t> dims1_vector;
  if (rank_1 > 0) {
    dims1_vector.resize(rank_1);
    for (int32_t i = 0; i < rank_1; ++i) {
      dims1_vector[i] = dims_1[i];
    }
  }

  // second input related
  const int32_t rank_2 = PARAM_INT32(data, dataIndex[5]);
  const int32_t *dims_2 = PARAM_INT32_PTR(data, dataIndex[6]);
  std::vector<int32_t> dims2_vector;
  if (rank_2 > 0) {
    dims2_vector.resize(rank_2);
    for (int32_t i = 0; i < rank_2; ++i) {
      dims2_vector[i] = dims_2[i];
    }
  }

  // output related
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

  // compute strides and some preprocessing
  const std::vector<int32_t> strides_1 =
      ShapeUtils::compute_strides(dims1_vector);
  const std::vector<int32_t> strides_2 =
      ShapeUtils::compute_strides(dims2_vector);
  const std::vector<int32_t> output_strides =
      ShapeUtils::compute_strides(output_dims_vector);
  std::vector<int32_t> indices_1(rank_1);
  std::vector<int32_t> indices_2(rank_2);
  std::vector<int32_t> broadcasted_indices(output_strides.size());

  // core functionality (with broadcasting)
  for (size_t i = 0; i < output_length; ++i) {
    ShapeUtils::offset_to_indices(output_strides, i, broadcasted_indices);
    BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices,
                                                    dims1_vector, indices_1);
    auto offset1 = ShapeUtils::indices_to_offset(strides_1, indices_1);
    BroadcastUtils::broadcasted_to_original_indices(broadcasted_indices,
                                                    dims2_vector, indices_2);
    auto offset2 = ShapeUtils::indices_to_offset(strides_2, indices_2);
    output[i] = BinaryOp::calc(input_1[offset1], input_2[offset2]);
  }
}

// Core op classes
class Add {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a + b; }
};

class Sub {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a - b; }
};

class Mul {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a * b; }
};

class Div {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a / b; }
};

class PRelu {
public:
  template <typename T> static T calc(const T &a, const T &b) {
    return a >= 0 ? a : a * b;
  }
};

class Xor {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a ^ b; }
};

class Or {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a || b; }
};

class And {
public:
  template <typename T> static T calc(const T &a, const T &b) { return a && b; }
};

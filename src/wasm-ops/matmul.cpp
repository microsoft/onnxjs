// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "matmul.h"
#include "common.h"
#include "utils/broadcast_utils.h"
#include "utils/shape_utils.h"
#include <math.h>

// Wasm interop method
void matmul_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  const float *input_1 = PARAM_FLOAT_PTR(data, dataIndex[1]);
  const int32_t *dims_1 = PARAM_INT32_PTR(data, dataIndex[2]);
  const int32_t rank_1 = PARAM_INT32(data, dataIndex[3]);
  const float *input_2 = PARAM_FLOAT_PTR(data, dataIndex[4]);
  const int32_t *dims_2 = PARAM_INT32_PTR(data, dataIndex[5]);
  const int32_t rank_2 = PARAM_INT32(data, dataIndex[6]);
  float *output = PARAM_FLOAT_PTR(data, dataIndex[7]);
  const int32_t output_length = PARAM_INT32(data, dataIndex[8]);
  const int32_t *output_dims = PARAM_INT32_PTR(data, dataIndex[9]);
  const int32_t output_rank = PARAM_INT32(data, dataIndex[10]);
  matmul_f32_imp(input_1, dims_1, rank_1, input_2, dims_2, rank_2, output,
                 output_length, output_dims, output_rank);
}

// Core operator implementation
void matmul_f32_imp(const float *input_1, const int32_t *dims_1,
                    const int32_t rank_1, const float *input_2,
                    const int32_t *dims_2, const int32_t rank_2, float *output,
                    const int32_t output_length, const int32_t *output_dims,
                    const int32_t output_rank) {
  int32_t M = dims_1[rank_1 - 2];
  int32_t K = dims_1[rank_1 - 1];
  int32_t N = dims_2[rank_2 - 1];

  // 2D matrices only
  if (output_rank == 2) {
    matmul2D_f32(input_1, input_2, output, M, K, N);
    return;
  }

  // multi-D matrices
  else {
    const int32_t num_matrices = (output_length) / (M * N);

    const float *input_1_traverse;
    const float *input_2_traverse;

    std::vector<int32_t> dims_1_vector(rank_1);
    for (int32_t r = 0; r < rank_1; ++r) {
      dims_1_vector[r] = dims_1[r];
    }
    const std::vector<int32_t> strides_1 =
        ShapeUtils::compute_strides(dims_1_vector);

    std::vector<int32_t> dims_2_vector(rank_2);
    for (int32_t r = 0; r < rank_2; ++r) {
      dims_2_vector[r] = dims_2[r];
    }
    const std::vector<int32_t> strides_2 =
        ShapeUtils::compute_strides(dims_2_vector);

    std::vector<int32_t> broadcasted_indices =
        std::vector<int32_t>(output_rank);
    broadcasted_indices[output_rank - 1] = 0;
    broadcasted_indices[output_rank - 2] = 0;

    std::vector<int32_t> original_indices_1(rank_1);
    int32_t original_offset_1;
    std::vector<int32_t> original_indices_2(rank_2);
    int32_t original_offset_2;
    int32_t offset_remainder;

    for (int32_t i = 0; i < num_matrices; i++) {
      // Compute broadcasted_indices for this offset
      int32_t offset_remainder = i;
      for (int32_t j = output_rank - 3; j >= 0; j--) {
        broadcasted_indices[j] = offset_remainder % output_dims[j];
        offset_remainder = floor(offset_remainder / output_dims[j]);
      }

      // This matrix is 2D, so no need to find the start_offset
      if (rank_1 == 2) {
        input_1_traverse = input_1;
      }
      // This matrix is not 2D, so no need to find appropriate start_offset
      else {
        BroadcastUtils::broadcasted_to_original_indices(
            broadcasted_indices, dims_1_vector, original_indices_1);
        original_offset_1 =
            ShapeUtils::indices_to_offset(strides_1, original_indices_1);
        input_1_traverse = input_1 + original_offset_1;
      }

      // This matrix is 2D, so no need to find the start_offset
      if (rank_2 == 2) {
        input_2_traverse = input_2;
      }
      // This matrix is not 2D, so no need to find appropriate start_offset
      else {
        BroadcastUtils::broadcasted_to_original_indices(
            broadcasted_indices, dims_2_vector, original_indices_2);
        original_offset_2 =
            ShapeUtils::indices_to_offset(strides_2, original_indices_2);
        input_2_traverse = input_2 + original_offset_2;
      }

      // process this 2D component alone
      matmul2D_f32(input_1_traverse, input_2_traverse, output + (i * M * N), M,
                   K, N);
    }
  }
}

// Core functionality implementation
void matmul2D_f32(const float *input_1, const float *input_2, float *output,
                  const int32_t M, const int32_t K, const int32_t N) {
  for (int32_t row = 0; row < M; ++row) {
    for (int32_t col = 0; col < N; ++col) {
      float sum = 0;
      for (int32_t traverse = 0; traverse < K; ++traverse) {
        sum += input_1[row * K + traverse] * input_2[traverse * N + col];
      }
      output[row * N + col] = sum;
    }
  }
}

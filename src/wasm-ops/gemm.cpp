// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "gemm.h"
#include "common.h"
#include <Eigen/Core>
#include <Eigen/Dense>

// Wasm interop method
void gemm_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];

  gemm_f32_imp(
      PARAM_BOOL(data, dataIndex[1]), PARAM_BOOL(data, dataIndex[2]),
      PARAM_INT32(data, dataIndex[3]), PARAM_INT32(data, dataIndex[4]),
      PARAM_INT32(data, dataIndex[5]), PARAM_FLOAT(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_FLOAT_PTR(data, dataIndex[8]),
      PARAM_FLOAT(data, dataIndex[9]), PARAM_FLOAT_PTR(data, dataIndex[10]));
}

// Core operator implementation
void gemm_f32_imp(const bool TransA, const bool TransB, const int M,
                  const int N, const int K, const float alpha, const float *A,
                  const float *B, const float beta, float *C) {
  auto C_mat = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
      C, static_cast<int64_t>(N), static_cast<int64_t>(M));
  if (beta == 0) {
    C_mat.setZero();
  } else {
    C_mat *= beta;
  }

  if (!TransA) {
    if (!TransB) {
      C_mat.noalias() +=
          alpha *
          (Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               B, static_cast<int64_t>(N), static_cast<int64_t>(K)) *
           Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               A, static_cast<int64_t>(K), static_cast<int64_t>(M)));
    } else {
      C_mat.noalias() +=
          alpha *
          (Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               B, static_cast<int64_t>(K), static_cast<int64_t>(N))
               .transpose() *
           Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               A, static_cast<int64_t>(K), static_cast<int64_t>(M)));
    }
  } else {
    if (!TransB) {
      C_mat.noalias() +=
          alpha *
          (Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               B, static_cast<int64_t>(N), static_cast<int64_t>(K)) *
           Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               A, static_cast<int64_t>(M), static_cast<int64_t>(K))
               .transpose());
    } else {
      C_mat.noalias() +=
          alpha *
          (Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               B, static_cast<int64_t>(K), static_cast<int64_t>(N))
               .transpose() *
           Eigen::Map<
               const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
               A, static_cast<int64_t>(M), static_cast<int64_t>(K))
               .transpose());
    }
  }
}

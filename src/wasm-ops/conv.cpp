// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "conv.h"
#include "common.h"
#include "gemm.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

// Wasm interop method
void conv_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  // TODO: Support muti-dimensional convolution (1D and 3D atleast)
  conv2D_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_INT32_PTR(data, dataIndex[2]),
      PARAM_FLOAT_PTR(data, dataIndex[3]), PARAM_INT32_PTR(data, dataIndex[4]),
      PARAM_FLOAT_PTR(data, dataIndex[5]), PARAM_INT32_PTR(data, dataIndex[6]),
      PARAM_FLOAT_PTR(data, dataIndex[7]), PARAM_INT32_PTR(data, dataIndex[8]),
      PARAM_INT32(data, dataIndex[9]), PARAM_INT32_PTR(data, dataIndex[10]),
      PARAM_INT32_PTR(data, dataIndex[11]));
}

// Core operator implementation
void conv2D_f32_imp(float *X, int *X_shape, float *W, int *W_shape, float *Y,
                    int *Y_shape, float *bias, int *dilations, int group,
                    int *pads, int *strides) {
  const int input_num = X_shape[0];
  const int input_channels = X_shape[1];
  const int input_height = X_shape[2];
  const int input_width = X_shape[3];
  const int input_size =
      input_num * input_channels * input_height * input_width;

  const int filter_num = W_shape[0];
  const int filter_channels = W_shape[1];
  const int filter_height = W_shape[2];
  const int filter_width = W_shape[3];
  const int filter_size =
      filter_num * filter_channels * filter_height * filter_width;
  std::vector<int> kernel_shape;
  kernel_shape.push_back(filter_height);
  kernel_shape.push_back(filter_width);

  const int output_num = Y_shape[0];
  const int output_channels = Y_shape[1];
  const int output_height = Y_shape[2];
  const int output_width = Y_shape[3];
  const int output_size =
      output_num * output_channels * output_height * output_width;

  const int input_image_size = input_height * input_width;
  const int output_image_size = output_height * output_width;
  const int kernel_size = kernel_shape[0] * kernel_shape[1];
  const int X_offset = input_channels / group * input_image_size;
  const int Y_offset = output_size / output_num / group;
  const int W_offset = filter_size / group;
  const int kernel_dim = input_channels / group * kernel_size;
  const int col_buffer_size = kernel_dim * output_image_size;

  float *col_buffer_data = new float[col_buffer_size]();

  for (int image_id = 0; image_id < input_num; ++image_id) {
    for (int group_id = 0; group_id < group; ++group_id) {
      im2col_f32(X + group_id * X_offset, input_channels / group, input_height,
                 input_width, kernel_shape[0], kernel_shape[1], dilations[0],
                 dilations[1], pads[0], pads[1], pads[2], pads[3], strides[0],
                 strides[1], col_buffer_data);

      gemm_f32_imp(false, false, filter_num / group, output_image_size,
                   kernel_dim, 1, W + group_id * W_offset, col_buffer_data, 0,
                   Y + group_id * Y_offset);
    }

    if (bias != nullptr) {
      auto Ymatrix =
          Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>(
              Y, output_image_size, filter_num);
      auto Bvec = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 1>>(
          bias, filter_num);
      Ymatrix.rowwise() += Bvec.transpose();
    }

    X += X_offset * group;
    Y += Y_offset * group;
  }

  delete[] col_buffer_data;
}

// Some helpers specific to conv operator
void im2col_f32(const float *data_im, const int channels, const int height,
                const int width, const int kernel_h, const int kernel_w,
                const int dilation_h, const int dilation_w, const int pad_t,
                const int pad_l, const int pad_b, const int pad_r,
                const int stride_h, const int stride_w, float *data_col) {
  const int output_h =
      (height + pad_b + pad_t - (dilation_h * (kernel_h - 1) + 1)) / stride_h +
      1;
  const int output_w =
      (width + pad_l + pad_r - (dilation_w * (kernel_w - 1) + 1)) / stride_w +
      1;

  // Fast path for zero padding and no dilation
  // From Torch, THNN_(unfolded_copy)
  if (dilation_h == 1 && dilation_w == 1 && pad_l == 0 && pad_r == 0 &&
      pad_t == 0 && pad_b == 0) {
    for (auto k = 0; k < channels * kernel_h * kernel_w; k++) {
      const auto nip = k / (kernel_h * kernel_w);
      const auto rest = k % (kernel_h * kernel_w);
      const auto kh = rest / kernel_w;
      const auto kw = rest % kernel_w;
      auto *dst = data_col + nip * (kernel_h * kernel_w * output_h * output_w) +
                  kh * (kernel_w * output_h * output_w) +
                  kw * (output_h * output_w);
      const auto *src = data_im + nip * (height * width);
      for (auto y = 0; y < output_h; y++) {
        const auto iy = y * stride_h + kh;
        const auto ix = kw;
        if (stride_w == 1) {
          memcpy(dst + (y * output_w), src + (iy * width + ix),
                 sizeof(float) * output_w);
        } else {
          for (auto x = 0; x < output_w; x++) {
            memcpy(dst + (y * output_w + x),
                   src + (iy * width + ix + x * stride_w), sizeof(float));
          }
        }
      }
    }
    return;
  }

  // Fast path for equal padding
  if (pad_l == pad_r && pad_t == pad_b) {
    // From Intel, https://github.com/BVLC/caffe/pull/3536
    const int pad_h = pad_t;
    const int pad_w = pad_l;
    const int channel_size = height * width;
    for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
        for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
          int input_row = -pad_h + kernel_row * dilation_h;
          for (int output_rows = output_h; output_rows; output_rows--) {
            if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
              for (int output_cols = output_w; output_cols; output_cols--) {
                *(data_col++) = 0;
              }
            } else {
              int input_col = -pad_w + kernel_col * dilation_w;
              for (int output_col = output_w; output_col; output_col--) {
                if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                  *(data_col++) = data_im[input_row * width + input_col];
                } else {
                  *(data_col++) = 0;
                }
                input_col += stride_w;
              }
            }
            input_row += stride_h;
          }
        }
      }
    }
    return;
  }

  // Baseline
  const int dkernel_h = dilation_h * (kernel_h - 1) + 1;
  const int dkernel_w = dilation_w * (kernel_w - 1) + 1;

  int height_col = (height + pad_t + pad_b - dkernel_h) / stride_h + 1;
  int width_col = (width + pad_l + pad_r - dkernel_w) / stride_w + 1;

  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_t + h_offset * dilation_h;
        int w_pad = w * stride_w - pad_l + w_offset * dilation_w;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
              data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

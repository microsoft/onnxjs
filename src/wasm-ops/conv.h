// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void conv_f32(void *);

// TODO: Support muti-dimensional convolution (1D and 3D atleast)
void conv2D_f32_imp(float *, int32_t *, float *, int32_t *, float *, int32_t *,
                    float *, int32_t *, int32_t, int32_t *, int32_t *);
void im2col_f32(const float *, const int32_t, const int32_t, const int32_t,
                const int32_t, const int32_t, const int32_t, const int32_t,
                const int32_t, const int32_t, const int32_t, const int32_t,
                const int32_t, const int32_t, float *);

// Helper functions
bool is_a_ge_zero_and_a_lt_b(int32_t a, int32_t b) {
  return static_cast<uint32_t>(a) < static_cast<uint32_t>(b);
}
}

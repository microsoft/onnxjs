// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void matmul_f32(void *);
void matmul_f32_imp(const float *, const int32_t *, const int32_t,
                    const float *, const int32_t *, const int32_t, float *,
                    const int32_t, const int32_t *, const int32_t);
void matmul2D_f32(const float *, const float *, float *, const int32_t,
                  const int32_t, const int32_t);
}

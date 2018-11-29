// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void gemm_f32(void *);
void gemm_f32_imp(const bool, const bool, const int32_t, const int32_t,
                  const int32_t, const float, const float *, const float *,
                  const float, float *);
}

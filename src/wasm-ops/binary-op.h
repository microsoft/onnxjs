// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>
#include <vector>

extern "C" {
void add_f32(void *);
void sub_f32(void *);
void mul_f32(void *);
void div_f32(void *);
void xor_f32(void *);
void or_f32(void *);
void and_f32(void *);
void prelu_f32(void *);

void binary_f32_imp_wrapper(void *data,
                            float (*core_op)(const float &, const float &));
void binary_f32_imp(const float *, const int32_t &, const int32_t &,
                    const std::vector<int32_t> &, const float *,
                    const int32_t &, const int32_t &,
                    const std::vector<int32_t> &, float *, const int32_t &,
                    const int32_t &, const std::vector<int32_t> &,
                    float (*)(const float &, const float &));
}

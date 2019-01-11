// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

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

// Wrappers
void binary_f32_input_f32_output_wrapper(void *data,
                                         float (*core_op)(const float &,
                                                          const float &));
void binary_bool_input_bool_output_wrapper(void *data,
                                           bool (*core_op)(const bool &,
                                                           const bool &));

// Implementations
void binary_f32_input_f32_output_imp(const float *, const int32_t &,
                                     const std::vector<int32_t> &,
                                     const float *, const int32_t &,
                                     const std::vector<int32_t> &, float *,
                                     const int32_t &, const int32_t &,
                                     const std::vector<int32_t> &,
                                     float (*)(const float &, const float &));
void binary_bool_input_bool_output_imp(const bool *, const int32_t &,
                                       const std::vector<int32_t> &,
                                       const bool *, const int32_t &,
                                       const std::vector<int32_t> &, bool *,
                                       const int32_t &, const int32_t &,
                                       const std::vector<int32_t> &,
                                       bool (*)(const bool &, const bool &));
}

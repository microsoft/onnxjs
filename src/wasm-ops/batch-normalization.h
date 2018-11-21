// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void batch_normalization_f32(void *);
void batch_normalization_f32_imp(float *, float *, int32_t, int32_t, int32_t,
                                 float *, float *, float *, float *, float);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void sum_f32(void *);
void sum_f32_imp(int32_t, int32_t, float *, float *);
}

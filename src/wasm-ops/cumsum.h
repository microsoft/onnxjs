// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void cumsum_f32(void *);
void cumsum_f32_imp(float *X, int32_t *dims, int32_t rank, int32_t axis,
                    bool exclusive, bool reverse, float *Y);
}

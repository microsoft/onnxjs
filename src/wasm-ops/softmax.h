// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void softmax_f32(void *);
void softmax_f32_imp(float *, float *, int32_t, int32_t);
}

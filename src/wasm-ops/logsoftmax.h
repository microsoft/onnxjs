// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <stdint.h>

extern "C" {
void logsoftmax_f32(void *);
void logsoftmax_f32_imp(float *input, float *output, int sz1, int sz2);
void logsoftmax1D_f32_imp(float *input, float *output, int sz1);
}

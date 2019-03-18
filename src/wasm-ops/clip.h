// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include "common.h"

extern "C" {
void clip_f32(void *);
// Expand for other supported data types for `clip`
}

// Core implementation of the op
template <typename T>
void clip_imp(const T *input, T *output, const int32_t length, const float min,
              const float max) {
  for (size_t i = 0; i < length; ++i) {
    const auto &val = input[i];
    output[i] = (val < min) ? min : (val > max) ? max : val;
  }
}

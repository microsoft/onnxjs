// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <limits>
#include <stdint.h>

extern "C" {
void pool_f32(void *);
void pool1D_f32_imp(bool, int32_t, float *, int32_t *, float *, int32_t *,
                    int32_t *, int32_t *, int32_t *, bool);
void pool2D_f32_imp(bool, int32_t, float *, int32_t *, float *, int32_t *,
                    int32_t *, int32_t *, int32_t *, bool);
void pool3D_f32_imp(bool, int32_t, float *, int32_t *, float *, int32_t *,
                    int32_t *, int32_t *, int32_t *, bool);

// Core Pool operation functions
// Average Pool
float AveragePoolInitialize() { return 0; }

void AveragePoolProcess(const float &x_data, float &y_data) {
  y_data += x_data;
}

void AveragePoolFinalize(const int32_t size, float &y_data) { y_data /= size; }

// Max Pool
float MaxPoolInitialize() { return std::numeric_limits<float>::lowest(); }

void MaxPoolProcess(const float &x_data, float &y_data) {
  if (x_data > y_data) {
    y_data = x_data;
  }
}

void MaxPoolFinalize(const int32_t /*size*/, float & /*y_data*/) {}
}

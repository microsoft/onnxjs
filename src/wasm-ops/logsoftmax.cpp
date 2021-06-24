/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 * Licensed under the BSD license.
 */

#include "logsoftmax.h"
#include "common.h"
#include <math.h>

#include <cfloat>
#include <cmath>

double approxexpminus(const double x) {
  const double a0 = 1.0;
  const double a1 = 0.125;
  const double a2 = 0.0078125;
  const double a3 = 0.00032552083;
  const double a4 = 1.0172526e-5;

  if (x < 13.0) {
    double y;
    y = a0 + x * (a1 + x * (a2 + x * (a3 + x * a4)));
    y *= y;
    y *= y;
    y *= y;
    y = 1 / y;
    return y;
  }
  return 0;
}

void logsoftmax_f32(void *data) {
  uint32_t *dataIndex = static_cast<uint32_t *>(data);
  uint32_t const argc = dataIndex[0];
  logsoftmax_f32_imp(
      PARAM_FLOAT_PTR(data, dataIndex[1]), PARAM_FLOAT_PTR(data, dataIndex[2]),
      PARAM_INT32(data, dataIndex[3]), PARAM_INT32(data, dataIndex[4]));
}

void logsoftmax_f32_imp(float *input, float *output, int sz1, int sz2) {
#pragma omp parallel for
  for (int i = 0; i < sz1; i++) {
    logsoftmax1D_f32_imp(input + i * sz2, output + i * sz2, sz2);
  }
}

void logsoftmax1D_f32_imp(float *input, float *output, int sz1) {
  float max = -FLT_MAX;
  float *in = input;
  for (int i = 0; i < sz1; i++) {
    float v = *in++;
    if (max < v) {
      max = v;
    }
  }

  double logsum = 0;
  in = input;
  for (int i = 0; i < sz1; i++) {
    logsum += approxexpminus(max - *in++);
  }
  logsum = max + log(logsum);

  for (int i = 0; i < sz1; i++) {
    *output++ = *input++ - logsum;
  }
}

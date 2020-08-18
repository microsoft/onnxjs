// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Softmax} from '../../../ops/softmax';
import {Tensor} from '../../../tensor';
import * as util from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuSoftmax extends Softmax {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = softmax(inputs[0], this.axis);
    return [output];
  }
}

export function softmax(x: Tensor, axis: number): Tensor {
  const inputDimensions = x.dims;
  const inputRank = inputDimensions.length;

  axis = util.ShapeUtil.normalizeAxis(axis, inputRank);
  const N = util.ShapeUtil.sizeToDimension(inputDimensions, axis);
  const D = util.ShapeUtil.sizeFromDimension(inputDimensions, axis);

  const X = x.numberData;

  const output = new Tensor(x.dims, x.type);
  const Y = output.numberData;

  for (let i = 0; i < N; i++) {
    // find row offset
    const offset = i * D;

    // find max of each logical row
    let max = Number.MIN_VALUE;
    for (let j = 0; j < D; j++) {
      if (X[offset + j] > max) {
        max = X[offset + j];
      }
    }

    // find normalization scale per row
    let scale = 0;
    for (let j = 0; j < D; j++) {
      const value = X[offset + j] - max;
      Y[offset + j] = Math.exp(value);
      scale += Math.exp(value);
    }

    // perform the softmax normalization
    for (let j = 0; j < D; j++) {
      if (scale === 0) {
        Y[offset + j] = 0;
      } else {
        Y[offset + j] /= scale;
      }
    }
  }

  return output;
}

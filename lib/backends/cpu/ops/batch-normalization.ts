// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BatchNormalization} from '../../../ops/batch-normalization';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuBatchNormalization extends BatchNormalization {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = batchNormalization(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], this.epsilon, this.momentum, this.spatial);
    return [output];
  }
}

export function batchNormalization(
    x: Tensor, scale: Tensor, b: Tensor, mean: Tensor, variance: Tensor, epsilon: number, momentum: number,
    spatial: number) {
  const inputDimensions = x.dims;
  const N = inputDimensions[0];
  const C = inputDimensions[1];

  // calculate channel size (i.e.) data points per channel
  let channelSize = 1;
  for (let i = 2; i < inputDimensions.length; i++) {
    channelSize *= inputDimensions[i];
  }

  const output = new Tensor(x.dims, x.type);

  const X = x.floatData;
  const Y = output.floatData;
  const scaleData = scale.numberData;
  const bData = b.numberData;
  const meanData = mean.numberData;
  const varianceData = variance.numberData;

  for (let nc = 0; nc < N * C; nc++) {
    const offset = nc * channelSize;
    for (let i = 0; i < channelSize; i++) {
      Y[offset + i] =
          scaleData[nc % C] * ((X[offset + i] - meanData[nc % C]) / Math.sqrt(varianceData[nc % C] + epsilon)) +
          bData[nc % C];
    }
  }
  return output;
}

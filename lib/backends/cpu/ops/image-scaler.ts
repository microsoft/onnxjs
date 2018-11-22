// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {ImageScaler} from '../../../ops/image-scaler';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuImageScaler extends ImageScaler {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = imageScaler(inputs[0], this.bias, this.scale);
    return [output];
  }
}

export function imageScaler(x: Tensor, bias: number[], scale: number): Tensor {
  const [N, C, H, W] = x.dims;
  const output = new Tensor([N, C, H, W], x.type);
  const X = x.floatData;
  const Y = output.floatData;
  for (let nc = 0; nc < N * C; nc++) {
    for (let hw = 0; hw < H * W; hw++) {
      const index = nc * H * W + hw;
      Y[index] = X[index] * scale + bias[nc % C];
    }
  }

  return output;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Squeeze} from '../../../ops/squeeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuSqueeze extends Squeeze {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = squeeze(inputs[0], this.axes);
    return [output];
  }
}

export function squeeze(x: Tensor, axes: number[]): Tensor {
  const outputDims = ShapeUtil.squeezeShape(x.dims, axes);
  const output = new Tensor(outputDims, x.type);

  const X = x.numberData;
  const Y = output.numberData;

  Y.set(X);

  return output;
}

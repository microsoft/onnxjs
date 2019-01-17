// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Unsqueeze} from '../../../ops/unsqueeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuUnsqueeze extends Unsqueeze {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = unsqueeze(inputs[0], this.axes);
    return [output];
  }
}

export function unsqueeze(x: Tensor, axes: number[]): Tensor {
  const outputDims = ShapeUtil.unsqueezeShape(x.dims, axes);
  const output = new Tensor(outputDims, x.type);

  const X = x.numberData;
  const Y = output.numberData;

  Y.set(X);

  return output;
}

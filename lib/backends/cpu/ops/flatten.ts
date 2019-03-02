// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Flatten} from '../../../ops/flatten';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuFlatten extends Flatten {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = flatten(inputs[0], this.axis);
    return [output];
  }
}

export function flatten(x: Tensor, axis: number): Tensor {
  const outputDims = ShapeUtil.flattenShape(x.dims, axis);
  const output = new Tensor(outputDims, x.type);

  const X = x.numberData;
  const Y = output.numberData;

  Y.set(X);

  return output;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Reshape} from '../../../ops/reshape';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuReshape extends Reshape {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = reshape(inputs[0], inputs[1]);
    return [output];
  }
}

export function reshape(x: Tensor, shape: Tensor): Tensor {
  const reshapedDims = ShapeUtil.calculateReshapedDims(x.dims, shape.integerData);
  const output = new Tensor(reshapedDims, x.type);
  const Y = output.floatData;
  Y.set(x.floatData);
  return output;
}

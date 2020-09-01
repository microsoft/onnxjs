// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Expand} from '../../../ops/expand';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

// import { getLogger } from 'log4js';

export class CpuExpand extends Expand {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    return [expand(inputs[0], inputs[1])];
  }
}

export function expand(x: Tensor, shape: Tensor): Tensor {
  const shapeData = shape.integerData as Int32Array;
  const dimensions = [...shapeData];
  const originalDimensions = x.dims;

  for (let i = dimensions.length - 1; i >= dimensions.length - originalDimensions.length; --i) {
    // Replace -1 with the original dimension
    if (dimensions[i] === -1) {
      dimensions[i] = originalDimensions[i + dimensions.length - originalDimensions.length];
    }
  }

  const output = new Tensor(dimensions, x.type);

  const result = BroadcastUtil.calc(x, output, (a, b) => a, false);
  if (!result) {
    throw new Error('not broadcastable');
  }

  return result;
}

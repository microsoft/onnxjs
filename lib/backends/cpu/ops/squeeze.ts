// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Squeeze} from '../../../ops/squeeze';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuSqueeze extends Squeeze {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = squeeze(inputs[0], this.axes);
    return [output];
  }
}

export function squeeze(x: Tensor, axes: number[]): Tensor {
  const inputDimensions = x.dims;

  const outputDims = new Array<number>();

  for (let i = 0; i < inputDimensions.length; i++) {
    if (axes.length > 0) {
      if (axes.indexOf(i) === -1) {  // not in squeeze list
        outputDims.push(inputDimensions[i]);
      } else {  // in squeeze list
        if (inputDimensions[i] !== 1) {
          throw new Error(`squeeze an axis of size different than 1`);
        }
      }
    } else {  // any axis with size=1 is squeezed
      if (inputDimensions[i] > 1) {
        outputDims.push(inputDimensions[i]);
      }
    }
  }

  const output = new Tensor(outputDims, x.type);

  const X = x.numberData;
  const Y = output.numberData;

  Y.set(X);

  return output;
}

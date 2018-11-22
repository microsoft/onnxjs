// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Unsqueeze} from '../../../ops/unsqueeze';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuUnsqueeze extends Unsqueeze {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = unsqueeze(inputs[0], this.axes);
    return [output];
  }
}

export function unsqueeze(x: Tensor, axes: number[]): Tensor {
  const inputDimensions = x.dims;

  const outputDims = new Array<number>(x.dims.length + axes.length);

  // initialize the array elements to 0
  outputDims.fill(0);

  // set all axes indices to 1 in outputDims and check for duplicates
  for (let i = 0; i < axes.length; i++) {
    const axis = axes[i];
    if (axis >= outputDims.length) {
      throw new Error(`'axes' has an out of range axis`);
    }
    if (outputDims[axis] !== 0) {
      throw new Error(`'axes' has a duplicate axis`);
    }

    outputDims[axis] = 1;
  }

  // fill in the zero entries of outputDims with the input tensor's shape
  let inputDimsIterator = 0;
  for (let i = 0; i < outputDims.length; i++) {
    if (outputDims[i] === 0) {
      outputDims[i] = inputDimensions[inputDimsIterator++];
    }
  }

  // sanity check assertion. 'inputDimsIterator'
  // should be equal to the length of 'inputDimensions'
  if (inputDimsIterator !== inputDimensions.length) {
    throw new Error('the unsqueezed dimension could not be established');
  }

  const output = new Tensor(outputDims, x.type);

  const X = x.numberData;
  const Y = output.numberData;

  Y.set(X);

  return output;
}

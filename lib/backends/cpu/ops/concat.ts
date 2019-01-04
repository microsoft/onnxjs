// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Concat} from '../../../ops/concat';
import {Tensor} from '../../../tensor';
import * as Util from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuConcat extends Concat {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = concat(inputs, this.axis);
    return [output];
  }
}

export function concat(x: Tensor[], axis: number) {
  const input0 = x[0];
  const inputShape = input0.dims;

  if (axis >= inputShape.length || axis < (-1 * inputShape.length)) {
    throw new Error(`axis specified for concat doesn't match input dimensionality`);
  }

  if (axis < 0) {
    axis = inputShape.length + axis;
  }

  // ensure all of the non-concatenated axes match each other
  // along the way, calculate the shape of the output tensor
  let concatAxisSize = inputShape[axis];
  const outputShape = new Array<number>(inputShape.length);

  for (let i = 1; i < x.length; i++) {
    const dataN = x[i];
    const dataNShape = dataN.dims;

    for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
      // add to the placeholder for computing output shape
      if (axisIndex === axis) {
        concatAxisSize += dataNShape[axisIndex];
      }

      // ensure all non-cancatenated axes match each other
      else if (inputShape[axisIndex] !== dataNShape[axisIndex]) {
        throw new Error(`non concat dimensions must match`);
      }

      // fill the 'outputShape' array
      outputShape[axisIndex] = dataNShape[axisIndex];
    }
  }

  // complete the 'outputShape' array
  outputShape[axis] = concatAxisSize;

  // main logic
  const output = new Tensor(outputShape, input0.type);
  const Y = output.numberData;

  // the axisPitch is the number of elements to add to move
  // to the next split axis in the output
  let axisPitch = 1;
  for (let i = outputShape.length - 1; i >= axis; i--) {
    axisPitch *= outputShape[i];
  }

  let outputBase = 0;
  for (let inputIndex = 0; inputIndex < x.length; inputIndex++) {
    const dataN = x[inputIndex];

    // the inputAxisPitch is the number of elements to add
    // to move to the next split axis in the input
    let inputAxisPitch = 1;
    for (let i = dataN.dims.length - 1; i >= axis; i--) {
      inputAxisPitch *= dataN.dims[i];
    }

    const inputData = dataN.numberData;
    const inputSize = Util.ShapeUtil.size(dataN.dims);

    // copy the data across.
    // for every 'inputAxisPitch' values copied, we move over by
    // the 'axisPitch'

    let outputOffset = outputBase;

    for (let i = 0, j = 0; i < inputSize; i++) {
      Y[outputOffset + i] = inputData[i];
      if (++j === inputAxisPitch) {
        // subtract inputAxisPitch because output is being indexed by 'i'
        outputOffset += (axisPitch - inputAxisPitch);
        j = 0;
      }
    }
    outputBase += inputAxisPitch;
  }

  return output;
}

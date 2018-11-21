// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Sum} from '../../../ops/sum';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuSum extends Sum {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = sum(inputs);
    return [output];
  }
}

export function sum(x: Tensor[]): Tensor {
  const output = new Tensor(x[0].dims, x[0].type);
  const size = x[0].floatData.length;
  const Y = output.floatData;
  for (let i = 0; i < x.length; i++) {
    const arr = x[i].floatData;
    for (let j = 0; j < size; ++j) {
      Y[j] += arr[j];
    }
  }

  return output;
}

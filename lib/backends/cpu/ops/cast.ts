// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Cast} from '../../../ops/cast';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

// import { getLogger } from 'log4js';

export class CpuCast extends Cast {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    return [cast(inputs[0], this.to)];
  }
}

export function cast(x: Tensor, to: Tensor.DataType): Tensor {
  const output = new Tensor([...x.dims], to);
  const inputData = x.data;
  const outputData = output.data;

  for (let i = 0; i < outputData.length; ++i) {
    outputData[i] = inputData[i];
  }

  return output;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Shape} from '../../../ops/shape';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

// import { getLogger } from 'log4js';

export class CpuShape extends Shape {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    return [shape(inputs[0])];
  }
}

export function shape(x: Tensor): Tensor {
  const output = new Tensor([x.dims.length], 'int32');
  const data = output.data;
  for (let i = 0; i < data.length; ++i) {
    data[i] = x.dims[i];
  }

  return output;
}

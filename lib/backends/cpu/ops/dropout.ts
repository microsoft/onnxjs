// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Dropout} from '../../../ops/dropout';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuDropout extends Dropout {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = dropout(inputs[0], this.ratio, this.testMode);
    return [output];
  }
}

export function dropout(x: Tensor, ratio: number, isTestMode: boolean) {
  if (!isTestMode) {
    throw new Error('only test mode is supported');
  }

  const output = new Tensor(x.dims, x.type);
  const X = x.floatData;
  const Y = output.numberData;
  Y.set(X);
  return output;
}

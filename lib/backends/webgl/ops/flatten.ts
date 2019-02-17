// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Flatten} from '../../../ops/flatten';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {reshape} from './reshape';

export class WebGLFlatten extends Flatten {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const total = inputs[0].dims.reduce((x, y) => x * y, 1);
    const right = inputs[0].dims.slice(this.axis).reduce((x, y) => x * y, 1);
    const outputDims = [total / right, right];

    return [reshape(inferenceHandler, inputs[0], outputDims)];
  }
}

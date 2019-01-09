// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Unsqueeze} from '../../../ops/unsqueeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {WebGLReshape} from './reshape';

export class WebGLUnsqueeze extends Unsqueeze {
  reshapeOps: WebGLReshape = new WebGLReshape();

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputDims = new Int32Array(ShapeUtil.unsqueezeShape(inputs[0].dims, this.axes));
    return this.reshapeOps.run(
        inferenceHandler, [inputs[0], Tensor.fromData(outputDims, [outputDims.length], 'int32')]);
  }
}

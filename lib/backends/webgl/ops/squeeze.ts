// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Squeeze} from '../../../ops/squeeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {WebGLReshape} from './reshape';

export class WebGLSqueeze extends Squeeze {
  reshapeOps: WebGLReshape = new WebGLReshape();

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputDims = new Int32Array(ShapeUtil.squeezeShape(inputs[0].dims, this.axes));
    return this.reshapeOps.run(
        inferenceHandler, [inputs[0], Tensor.fromData(outputDims, [outputDims.length], 'int32')]);
  }
}

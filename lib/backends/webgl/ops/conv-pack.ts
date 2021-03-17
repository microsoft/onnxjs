// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger} from '../../../instrument';
import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {reshape} from '../webgl_utils';
import {WebGLConv} from './conv';
import {WebGLIm2ColPacked} from './im2col-pack';
import {WebGLMatMulPacked} from './matmul-pack';

export class WebGLConvPacked extends Conv {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const xshape = inputs[0].dims.slice();
    if (xshape.length !== 4 && xshape[0] !== 1 && this.group !== 1) {
      const conv = new WebGLConv();
      return conv.run(inferenceHandler, inputs);
    }

    // For single batch 2D conv
    const kshape = inputs[1].dims.slice();
    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      for (let i = 2; i < kshape.length; ++i) {
        this.kernelShape.push(kshape[i]);
      }
    }
    PoolConvUtil.adjustPadsBasedOnAutoPad(
        inputs[0].dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    Logger.verbose(
        'Conv',
        `autpPad:${this.autoPad}, dilations:${this.dilations}, group:${this.group}, kernelShape:${
            this.kernelShape}, pads:${this.pads}, strides:${this.strides}`);

    const outputShape = WebGLConv.calcOutputShape(xshape, kshape, this.dilations, this.pads, this.strides);
    const im2col = new WebGLIm2ColPacked(outputShape, kshape, this.dilations, this.pads, this.strides);
    const im2colOutput: Tensor[] = im2col.run(inferenceHandler, [inputs[0], inputs[1]]);
    const kernelReshaped = reshape(inferenceHandler, inputs[1], [kshape[0], kshape[1] * kshape[2] * kshape[3]]);
    const matmul = new WebGLMatMulPacked();
    const matmulOutput = inputs.length === 3 ?
        matmul.run(inferenceHandler, [kernelReshaped, im2colOutput[0], inputs[2]]) :
        matmul.run(inferenceHandler, [kernelReshaped, im2colOutput[0]]);
    return [reshape(inferenceHandler, matmulOutput[0], outputShape)];
  }
}

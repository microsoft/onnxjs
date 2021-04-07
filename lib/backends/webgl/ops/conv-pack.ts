// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
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
    if (xshape.length !== 4 || xshape[0] !== 1 || this.group !== 1 ||
        (this.kernelShape[0] === 1 && this.kernelShape[1] === 1)) {
      const conv = new WebGLConv();
      const attrs = new Attribute(undefined);
      attrs.set('autoPad', 'string', this.autoPad);
      attrs.set('dilation', 'ints', this.dilations);
      attrs.set('group', 'int', this.group);
      attrs.set('kernelShape', 'ints', this.kernelShape);
      attrs.set('pads', 'ints', this.pads);
      attrs.set('strides', 'ints', this.strides);
      conv.initialize(attrs);
      return conv.run(inferenceHandler, inputs);
    }

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
    // console.log('conv_pack input 0-2:', inputs[0].data[0], inputs[0].data[1], inputs[0].data[2]);
    //, inputs[0].data[3], inputs[0].data[4], inputs[0].data[5], inputs[0].data[6], inputs[0].data[7]);
    /*console.log(
        'im2col output 0-17:', im2colOutput[0].data[0], im2colOutput[0].data[1], im2colOutput[0].data[2],
        im2colOutput[0].data[3], im2colOutput[0].data[4], im2colOutput[0].data[5], im2colOutput[0].data[6],
        im2colOutput[0].data[7], im2colOutput[0].data[8], im2colOutput[0].data[9], im2colOutput[0].data[10],
        im2colOutput[0].data[11], im2colOutput[0].data[12], im2colOutput[0].data[13], im2colOutput[0].data[14],
        im2colOutput[0].data[15], im2colOutput[0].data[16], im2colOutput[0].data[17]);*/
    const kernelReshaped = reshape(inferenceHandler, inputs[1], [kshape[0], kshape[1] * kshape[2] * kshape[3]]);
    /*console.log(
        'kernel reshaped 0-7:', kernelReshaped.data[0], kernelReshaped.data[1], kernelReshaped.data[2],
        kernelReshaped.data[3], kernelReshaped.data[4], kernelReshaped.data[5], kernelReshaped.data[6],
        kernelReshaped.data[7]);*/
    const matmul = new WebGLMatMulPacked();
    const matmulOutput = inputs.length === 3 ?
        matmul.run(inferenceHandler, [kernelReshaped, im2colOutput[0], inputs[2]]) :
        matmul.run(inferenceHandler, [kernelReshaped, im2colOutput[0]]);
    const res = [reshape(inferenceHandler, matmulOutput[0], outputShape)];
    /*console.log(
        'ConvPack Output 0-7:', res[0].data[0], res[0].data[1], res[0].data[2], res[0].data[3], res[0].data[4],
        res[0].data[5], res[0].data[6], res[0].data[7]);*/
    // console.log('convPack output 61437-61439:', res[0].data[61437], res[0].data[61438], res[0].data[61439]);

    return res;
  }
}

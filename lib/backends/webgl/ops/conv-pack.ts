// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// import {assert} from 'chai';
import {Attribute} from '../../../attribute';
import {Logger} from '../../../instrument';
import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo} from '../types';
import {WebGLConv} from './conv';
import {WebGLIm2ColPacked} from './im2col-pack';
import {WebGLMatMulPacked} from './matmul-pack';
// import {WebGLMatMul} from './matmul';
import {WebGLReshapePacked} from './reshape-packed';

export class WebGLConvPacked extends Conv {
  protected artifacts: Artifact[];
  protected programInfo: ProgramInfo[];

  protected fallbackArtifact: Artifact[];
  protected fallbackProgramInfo: ProgramInfo[];

  protected fallbackConv: WebGLConv;

  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const programManager = inferenceHandler.session.programManager;
    const xshape = inputs[0].dims.slice();
    if (
        xshape.length !== 4 || xshape[0] !== 1 || this.group !== 1
        //|| (this.kernelShape[0] === 1 && this.kernelShape[1] === 1)) {
    ) {
      if (!this.fallbackConv) {
        this.fallbackConv = new WebGLConv();
        const attrs = new Attribute(undefined);
        attrs.set('autoPad', 'string', this.autoPad);
        attrs.set('dilation', 'ints', this.dilations);
        attrs.set('group', 'int', this.group);
        attrs.set('kernelShape', 'ints', this.kernelShape);
        attrs.set('pads', 'ints', this.pads);
        attrs.set('strides', 'ints', this.strides);
        this.fallbackConv.initialize(attrs);
      }
      return this.fallbackConv.run(inferenceHandler, inputs);

      // if (!this.fallbackArtifact) {
      //   this.fallbackArtifact = [];
      //   this.fallbackProgramInfo = this.fallbackConv.createProgramInfos(inferenceHandler, inputs);
      //   for (let i = 0; i < this.fallbackProgramInfo.length; ++i) {
      //     const artifact = inferenceHandler.session.programManager.build(this.fallbackProgramInfo[i]);
      //     this.artifacts.push(artifact);
      //   }
      // }

      // const runDatas =
      //     this.fallbackConv.createRunDatas(inferenceHandler, this.fallbackArtifact.map(a => a.programInfo), inputs);

      // inferenceHandler.checkAndUpdateTextureForm(this.artifacts[0], runDatas[0]);
      // programManager.run(this.fallbackArtifact[0], runDatas[0]);
      // programManager.run(this.fallbackArtifact[1], runDatas[1]);
      // return [runDatas[1].outputTextureData.tensor];
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
    const matmul = new WebGLMatMulPacked();
    // const matmul = new WebGLFusedMatMulPacked();
    const reshape = new WebGLReshapePacked();
    // shape for kernel reshape
    const shape = new Tensor([2], 'int32');
    shape.data[0] = kshape[0];
    shape.data[1] = kshape[1] * kshape[2] * kshape[3];

    // assert(this.artifacts.length < 5, 'ConvPacked kernel shouldn\'t have more than 4 artifacts.');
    if (!this.artifacts) {
      this.artifacts = [];
      this.programInfo = [];
      this.programInfo[0] = im2col.createProgramInfo(inferenceHandler, [inputs[0], inputs[1]]);
      this.artifacts[0] = programManager.build(this.programInfo[0]);

      this.programInfo[1] = reshape.createProgramInfo(inferenceHandler, [inputs[1], shape]);
      this.artifacts[1] = programManager.build(this.programInfo[1]);
    }

    // run im2col
    const runDataIm2col = im2col.createRunData(inferenceHandler, this.programInfo[0], [inputs[0], inputs[1]]);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[0], runDataIm2col);
    programManager.run(this.artifacts[0], runDataIm2col);
    const im2colOutput = runDataIm2col.outputTextureData.tensor;

    // reshape kernel
    const runDataKernelReshape = reshape.createRunData(inferenceHandler, this.programInfo[1], [inputs[1], shape]);
    // inferenceHandler.checkAndUpdateTextureForm(this.artifacts[1], runDataKernelReshape);
    programManager.run(this.artifacts[1], runDataKernelReshape);
    const kernelReshaped = runDataKernelReshape.outputTextureData.tensor;

    // run matmul
    const hasBias = (inputs.length === 3);
    if (this.artifacts.length === 2) {
      this.programInfo[2] = matmul.createProgramInfo(
          inferenceHandler, hasBias ? [kernelReshaped, im2colOutput, inputs[2]] : [kernelReshaped, im2colOutput]);
      this.artifacts[2] = programManager.build(this.programInfo[2]);
    }
    const runDataMatmul = matmul.createRunData(
        inferenceHandler, this.programInfo[2],
        hasBias ? [kernelReshaped, im2colOutput, inputs[2]] : [kernelReshaped, im2colOutput]);
    inferenceHandler.checkAndUpdateTextureForm(this.artifacts[2], runDataMatmul);
    programManager.run(this.artifacts[2], runDataMatmul);
    const matmulOutput = runDataMatmul.outputTextureData.tensor;

    // reshape output
    const outputShapeTensor = new Tensor([outputShape.length], 'int32');
    // outputShape.map((i) => outputShapeTensor.data[i] = outputShape[i]);

    for (let i = 0; i < outputShape.length; i++) {
      outputShapeTensor.data[i] = outputShape[i];
    }

    if (this.artifacts.length === 3) {
      this.programInfo[3] = reshape.createProgramInfo(inferenceHandler, [matmulOutput, outputShapeTensor]);
      this.artifacts[3] = programManager.build(this.programInfo[3]);
    }
    const runDataOutputReshape =
        reshape.createRunData(inferenceHandler, this.programInfo[3], [matmulOutput, outputShapeTensor]);
    // inferenceHandler.checkAndUpdateTextureForm(this.artifacts[3], runDataOutputReshape);
    programManager.run(this.artifacts[3], runDataOutputReshape);
    return [runDataOutputReshape.outputTextureData.tensor];
  }
}

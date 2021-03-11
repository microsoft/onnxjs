// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// import {InferenceHandler} from '../../../backend';
import {Logger} from '../../../instrument';
import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
// import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData} from '../types';
// import {WebGLContext} from '../webgl-context';

export class WebGLConvPacked extends Conv {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    if (!this.artifacts) {
      this.artifacts = [];
      const programInfo = this.createProgramInfo(inferenceHandler, inputs);
      for (let i = 0; i < programInfo.length; ++i) {
        const artifact = inferenceHandler.session.programManager.build(programInfo[i]);
        this.artifacts.push(artifact);
      }
    }
    const runDatas = this.createRunData(inferenceHandler, this.artifacts.map(a => a.programInfo), inputs);
    inferenceHandler.runProgram(this.artifacts[0], runDatas[0]);
    // programManager.run(this.artifacts[1], runDatas[1]);
    return [runDatas[0].outputTextureData.tensor];
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo[] {
    const xshape = inputs[0].dims.slice();
    const kshape = inputs[1].dims.slice();
    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      const wDims = inputs[1].dims;
      for (let i = 2; i < wDims.length; ++i) {
        this.kernelShape.push(wDims[i]);
      }
    }
    PoolConvUtil.adjustPadsBasedOnAutoPad(
        inputs[0].dims, this.strides, this.dilations, this.kernelShape, this.pads, this.autoPad);
    Logger.verbose(
        'Conv',
        `autpPad:${this.autoPad}, dilations:${this.dilations}, group:${this.group}, kernelShape:${
            this.kernelShape}, pads:${this.pads}, strides:${this.strides}`);
    const outputShape = this.calcOutputShape(xshape, kshape, this.dilations, this.pads, this.strides);
    const im2colProgramInfo = this.createIm2ColProgramInfo(inferenceHandler, inputs, outputShape);
    // const matmulProgramInfo =
    //    this.createMatmulProgramInfo(inferenceHandler, im2colProgramInfo.outputLayout, inputs, outputShape);
    // return [im2colProgramInfo, matmulProgramInfo];
    return [im2colProgramInfo];
  }
  createMatmulProgramInfo(
      inferenceHandler: WebGLInferenceHandler, outputLayout: any, inputs: Tensor[],
      outputShape: number[]): ProgramInfo {
    throw new Error('Method not implemented.');
  }
  createIm2ColProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[], outputShape: number[]):
      ProgramInfo {
    const xshape = inputs[0].dims.slice();
    const wshape = inputs[1].dims.slice();
    const rowDim = 2;
    const colDim = 3;
    const rank = outputShape.length;
    const im2colShape = [wshape[1] * wshape[2] * wshape[3], outputShape[2] * outputShape[3]];
    const itemsPerBlockRow = xshape[2] * wshape[3];
    const unpackChannel = unpackFromChannel();
    let unrolled = ``;

    for (let row = 0; row <= 1; row++) {
      for (let col = 0; col <= 1; col++) {
        unrolled += `
          blockIndex = rc.x + ${col};
          pos = rc.y + ${row};

          if(blockIndex < ${im2colShape[1]} && pos < ${im2colShape[0]}) {
            offsetY = int(blockIndex / (${outputShape[rank - 1]})) * ${this.strides[0]} - ${this.pads[1]};
            d0 = offsetY + ${this.dilations[0]} * (pos / ${itemsPerBlockRow});

            if(d0 < ${xshape[rowDim]} && d0 >= 0) {
              offsetX = int(mod(float(blockIndex), ${outputShape[rank - 1]}.) * ${this.strides[1]}. - ${
            this.pads[0]}.);
              d1 = offsetX + ${this.dilations[1]} * (int(mod(float(pos), ${itemsPerBlockRow}.) / ${xshape[1]}.));
              result[${row * 2 + col}]=float(d1);

              if(d1 < ${xshape[colDim]} && d1 >= 0) {

                ch = int(mod(float(pos), ${xshape[1]}.));
                  innerDims = vec2(d0, d1);
                  //d0=0, d1=2
                  result[${row * 2 + col}] = getChannel(
                    getA(0, ch, int(innerDims.x),
                    int(innerDims.y)), innerDims);
                    int p[4];
                    p[0]=0;
                    p[1]=0;
                    p[2]=0;
                    p[3]=0;
                    result[${row * 2 + col}] = getChannel(
                      _A_Pack(p), vec2(0.0, 0.0));
              }
            }
          }

        `;
      }
    }

    const shaderSource = `
    ${unpackChannel}

    void main() {
      ivec2 rc = getOutputCoords();
        vec4 result = vec4(0.0);
        int blockIndex, pos, offsetY, d0, offsetX, d1, ch;
        vec2 innerDims;
        //TexCoords.xy: [0.5, 0.25], [0.5, 0.75]
        //rc.xy: [0, 0], [0, 2]
        ${unrolled}
        //outputColor = result;
        outputColor = vec4(100.0,200.0,300.0,400.0);
      }
          `;
    return {
      inputLayouts: [inferenceHandler.getOrCreateTextureLayout(inputs[0], 4, true, xshape, true)],
      outputLayout:
          inferenceHandler.createTextureLayoutFromShape(im2colShape, 4, im2colShape, {isPacked: true, reverseWH: true}),
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      isInputsPacked: true,
      isOutputPacked: true,
    };
  }

  createRunData(inferenceHandler: WebGLInferenceHandler, programInfos: ProgramInfo[], inputs: Tensor[]): RunData[] {
    /*const w = inputs[1];
    // const b = inputs.length >= 3 ? inputs[2] : undefined;
    let wTextureData = inferenceHandler.getTextureData(w.dataId);
    if (!wTextureData) {
      Logger.verbose('Conv', 'Did not find the adjustedKernel texture in the cache. Creating rew.');
      const newKernelData = this.prepKernelForMatmul(w.dims.slice(), this.group, 4, w.floatData as Float32Array);
      // hack: should use graph transformer to rewrite initializer K
      wTextureData = inferenceHandler.createTextureDataFromLayoutBindTensor(
          programInfos[1].inputLayouts[1], w.type, newKernelData, w);
    }*/
    const runtDataIm2Col = {
      inputTextureDatas: [inferenceHandler.getOrCreateTextureData(
          inputs[0], inferenceHandler.getOrCreateTextureLayout(inputs[0], 1, false, [], true))],
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfos[0].outputLayout, inputs[0].type),
      uniformData: {}
    };
    return [runtDataIm2Col];
  }
  protected calcOutputShape(
      inputShape: number[], kernelShape: number[], dilations: number[], adjustPads: number[],
      strides: number[]): number[] {
    const batchSize = inputShape[0];
    const inputSpatialShape = inputShape.slice(2);
    const spatialRank = inputSpatialShape.length;
    const outChannels = kernelShape[0];
    const kernelSpatialShape = kernelShape.slice(2);
    const dilatedKernelShape = kernelSpatialShape.map((v, i) => v + (v - 1) * (dilations[i] - 1));
    const inputSpatialShapeWithPad = inputSpatialShape.map((v, i) => v + adjustPads[i] + adjustPads[i + spatialRank]);
    const outputSpatialShape =
        inputSpatialShapeWithPad.map((v, i) => Math.floor((v - dilatedKernelShape[i] + strides[i]) / strides[i]));
    const outputShape = [batchSize, outChannels].concat(...outputSpatialShape);
    return outputShape;
  }
  protected prepKernelForMatmul(shape: number[], group: number, channels: number, kernel: Float32Array): Float32Array {
    if (group === 1 && (channels === 1 || (shape[2] * shape[3]) % channels === 0)) {
      return kernel;
    }
    const numFeatureMaps = shape[0];
    const oldRowSize = shape[1] * shape[2] * shape[3];
    const newRowSize = Math.ceil(oldRowSize * group / channels) * channels;
    const newSize = numFeatureMaps * newRowSize;
    const buffer = new Float32Array(newSize);
    for (let f = 0; f < numFeatureMaps; ++f) {
      const oldOffset = f * oldRowSize;
      const newOffset = f * newRowSize + f % group * oldRowSize;
      buffer.set(kernel.subarray(oldOffset, oldOffset + oldRowSize), newOffset);
    }
    return buffer;
  }

  protected artifacts: Artifact[];
}

function unpackFromChannel(): string {
  return `
  float getChannel(vec4 frag, vec2 innerDims) {
    vec2 modCoord = mod(innerDims, 2.);
    return modCoord.x == 0. ?
      (modCoord.y == 0. ? frag.r : frag.g) :
      (modCoord.y == 0. ? frag.b : frag.a);
  }
  `;
}

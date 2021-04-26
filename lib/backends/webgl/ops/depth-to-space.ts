// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {DepthToSpace} from '../../../ops/depth-to-space';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData} from '../types';

export class WebGLDepthToSpace extends DepthToSpace {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0]);
    const outputShape = this.getOutShape(inputs[0]);
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const shaderSource = `
    void main() {
      ivec4 coords = getOutputCoords();
      int b = coords[0];
      int d = coords[1];
      int h = coords[2];
      int w = coords[3];

      int in_h = h / ${this.blocksize};
      int offset_h = imod(h, ${this.blocksize});
      int in_w = w / ${this.blocksize};
      int offset_w = imod(w, ${this.blocksize});
      int offset_d = (offset_h * ${this.blocksize} + offset_w) *
        ${outputShape[1]};
      int in_depth = d + offset_d;

      float result = getX(b, in_depth, in_h, in_w);
      ${glsl.output} = vec4(result, 0, 0, 0);
    }
      `;
    return {
      inputLayouts: [inputLayout],
      outputLayout: handler.createTextureLayoutFromShape(outputShape),
      samplers: ['X'],
      shaderSource,
      hasMain: true
    };
  }
  protected getOutShape(input: Tensor): number[] {
    const batchSize = input.dims[0];
    const inputDepth = input.dims[1];
    const inputHeight = input.dims[2];
    const inputWidth = input.dims[3];
    if (inputDepth % (this.blocksizeSqr) !== 0) {
      throw new Error('Input depth must be divisible by squared blocksize.');
    }
    const outputDepth = inputDepth / this.blocksizeSqr;
    const outputHeight = inputHeight * this.blocksize;
    const outputWidth = inputWidth * this.blocksize;
    return [batchSize, outputDepth, outputHeight, outputWidth];
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreateTextureData(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

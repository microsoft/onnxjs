// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

import {getChannels} from './packing_utils';

export class WebGLPack implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (inputs.length !== 1) {
      throw new Error(`Pack kernel should have input tensor count to 1.`);
    }

    const inputShape = inputs[0].dims;

    // TODO(Du): look into ways to simplify createTextureLayoutFromShape's signature
    const outputLayout = handler.createTextureLayoutFromShape(inputShape, 4, inputShape, {isPacked: true});
    const outputShape = outputLayout.shape;
    const rank = outputShape.length;

    const setup = getSetup(rank, inputShape[inputShape.length - 1], inputShape[inputShape.length - 2]);

    const channels = getChannels('rc', rank);
    const outOfBoundsCondition = getOutOfBoundsCondition(rank, inputShape, channels);
    const output = getOutput(outputShape, channels);
    const shaderSource = `
        void main() {
          // TODO(TJ): implement getOutputCoords() to map input uv to output xy.
          ivec2 rc = getOutputCoords();
          //ivec2 rc = ivec2(0, 0);

          if(${outOfBoundsCondition}) {
            outputColor = vec4(0);
          } else {
            ${setup}

            outputColor = vec4(${output});
          }
        }
      `;

    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout,
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      isInputsPacked: false,
      isOutputPacked: true,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

function getOutOfBoundsCondition(rank: number, shape: ReadonlyArray<number>, dims: string[]): string {
  if (rank === 1) {
    return `rc > ${shape[0]}`;
  }

  let cond = '';
  for (let i = rank - 2; i < rank; i++) {
    cond += `${dims[i]} >= ${shape[i]}`;
    if (i < rank - 1) {
      cond += '||';
    }
  }

  return cond;
}

function getOutput(shape: ReadonlyArray<number>, dims: string[]): string {
  const rank = shape.length;
  if (rank === 1) {
    return `getA(rc),
            rc + 1 >= ${shape[0]} ? 0. : getA(rc + 1),
            0, 0`;
  }

  return `getA(r, c),
          cEdge ? 0. : getA(r, cp1),
          rEdge ? 0. : getA(rp1, c),
          rEdge || cEdge ? 0. : getA(rp1, cp1)`;
}

function getSetup(rank: number, cols: number, rows: number): string {
  if (rank === 1) {
    return '';
  }
  // rank >= 2 for width+height pack.
  else {
    const setup = `
    int r = rc.x;
    int c = rc.y;
    int rp1 = rc.x + 1;
    int cp1 = rc.y + 1;
    bool cEdge = cp1 >= ${cols};
    bool rEdge = rp1 >= ${rows};
    `;
    return setup;
  }
}

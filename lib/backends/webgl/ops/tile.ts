// Licensed under the MIT license.

import {Tile} from '../../../ops/tile';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {WebGLOperator} from '../webgl-operator';

export class WebGLTile extends Tile implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }

  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputShape = inputs[0].dims.slice();
    const outputShape = new Array(inputShape.length);  // inputs[0].dims.slice();

    const tileOps: string[] = [];
    for (let i = 0; i < inputShape.length; i++) {
      outputShape[i] = inputShape[i] * inputs[1].numberData[i];
      tileOps.push(`inputIdx[${i}] = int(mod(float(outputIdx[${i}]), ${inputShape[i]}.));`);
    }

    const rank = outputShape.length;
    const shaderSource = `
    uniform sampler2D A;
    float process(int outputIdx[${rank}]) {
      int inputIdx[${rank}];
      ${tileOps.join('\n')}
      return _A(inputIdx);
    }`;
    return {
      hasMain: false,
      inputLayouts: inputs.map(t => handler.getOrCreateTextureLayout(t)),
      outputLayout: handler.createBasicTextureLayout(outputShape),
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => handler.getOrCreate(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType),
      uniformData: {}
    };
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {LeakyRelu} from '../../../ops/leaky-relu';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {WebGLOperator} from '../webgl-operator';

export class WebGLLeakyRelu extends LeakyRelu implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const shaderSource = `
      uniform sampler2D A;
      void main() {
        float v = texture2D(A, TexCoords).r;
        gl_FragColor = vec4(v < 0.0 ? v * float(${this.alpha}) : v);
      }
      `;
    return {
      hasMain: true,
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout: handler.createBasicTextureLayout(outputShape),
      shaderSource,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreate(inputs[0], programInfo.inputLayouts[0])];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType),
      uniformData: {}
    };
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BatchNormalization} from '../../../ops/batch-normalization';
import {Tensor} from '../../../tensor';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {WebGLOperatorHelper} from '../webgl-operator-utils';

export class WebGLBatchNormalization extends BatchNormalization {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayouts = inputs.map(t => handler.getOrCreateTextureLayout(t));
    const outputShape = inputs[0].dims.slice();
    const rank = outputShape.length;
    const scale = inputLayouts[1];
    const shaderSource = `
      uniform sampler2D A;
      uniform sampler2D Scale;
      uniform sampler2D B;
      uniform sampler2D Mean;
      uniform sampler2D Variance;

      float process(int[${rank}] indices) {
        vec2 position = offsetToCoords(indices[1], ${scale.width}, ${scale.height});
        float scale = getColorAsFloat(texture2D(Scale, position));
        float mean = getColorAsFloat(texture2D(Mean, position));
        float variance = getColorAsFloat(texture2D(Variance, position));
        float b = getColorAsFloat(texture2D(B, position));

        return scale * ( (_A(indices) - mean) / sqrt(variance + float(${this.epsilon})) ) + b;
      }`;
    return {hasMain: false, inputLayouts, outputLayout: handler.createBasicTextureLayout(outputShape), shaderSource};
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = [handler.getOrCreate(inputs[0], programInfo.inputLayouts[0])];
    inputs.slice(1).forEach(t => inputTDs.push(handler.getOrCreate(t)));
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType);
    return {inputTextureDatas: inputTDs, outputTextureData: outputTD, uniformData: {}};
  }
}

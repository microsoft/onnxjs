// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {Operator} from '../../../operators';
import {Tensor} from '../../../tensor';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Encoder} from '../texture-data-encoder';
import {ProgramInfo, RunData, WebGLOperator} from '../types';

/**
 * WebGLDownloadFloat is a custom operator that convert float32 data into a 4-channel encoded uint8 data.
 * The generated output will be written into a uint8 format texture, with its content binary equivalent to the
 * corresponding float32 data.
 * WebGL session handler is responsible to append this operator to the graph in environment where downloading from
 * float texture is unavailable. (typically on Apple devices)
 */
export class WebGLDownloadFloat implements Operator, WebGLOperator {
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const shaderSource = `
      const float FLOAT_MAX = 1.70141184e38;
      const float FLOAT_MIN = 1.17549435e-38;

      bool isNaN(float val) {
        return (val < 1.0 || 0.0 < val || val == 0.0) ? false : true;
      }

      highp vec4 encodeAsUint8(highp float v) {
        if (isNaN(v)) {
          return vec4(255, 255, 255, 255);
        }

        highp float av = abs(v);

        if(av < FLOAT_MIN) {
          return vec4(0.0, 0.0, 0.0, 0.0);
        } else if(v > FLOAT_MAX) {
          return vec4(0.0, 0.0, 128.0, 127.0) / 255.0;
        } else if(v < -FLOAT_MAX) {
          return vec4(0.0, 0.0,  128.0, 255.0) / 255.0;
        }

        highp vec4 c = vec4(0,0,0,0);

        highp float e = floor(log2(av));
        highp float m = exp2(fract(log2(av))) - 1.0;

        c[2] = floor(128.0 * m);
        m -= c[2] / 128.0;
        c[1] = floor(32768.0 * m);
        m -= c[1] / 32768.0;
        c[0] = floor(8388608.0 * m);

        highp float ebias = e + 127.0;
        c[3] = floor(ebias / 2.0);
        ebias -= c[3] * 2.0;
        c[2] += floor(ebias) * 128.0;

        c[3] += 128.0 * step(0.0, -v);

        return c / 255.0;
      }

      void main() {
        float value = ${glsl.texture2D}(X,TexCoords).r;
        ${glsl.output} = encodeAsUint8(value);
      }`;
    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0])],
      outputLayout: handler.createTextureLayoutFromShape(inputs[0].dims, 4, inputs[0].dims),
      samplers: ['X'],
      shaderSource,
      hasMain: true,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    return {
      inputTextureDatas: [handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0])],
      outputTextureData: handler.createTextureDataFromLayout(
          programInfo.outputLayout, 'float32', Encoder.Usage.Download4BytesAsFloat32),
      uniformData: {}
    };
  }
  initialize(attributes: Attribute): void {}
  checkInputs(inputs: Tensor[]): boolean {
    return inputs.length === 1 && inputs[0].type === 'float32';
  }
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
}

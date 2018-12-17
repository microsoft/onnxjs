// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {EncodingGlslLib} from '../glsl-encoding-lib.';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo} from '../program-info';
import {RunData} from '../program-manager';
import {TextureData, TextureLayout} from '../texture-data';
import {WebGLOperator} from '../webgl-operator';
import {WebGLOperatorHelper} from '../webgl-operator-utils';

export class WebGLUint8Encode implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return WebGLOperatorHelper.run(this, inferenceHandler, inputs);
  }
  createProgramInfo(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const outputShape = inputs[0].dims.slice();
    const [width, height] = inferenceHandler.session.layoutStrategy.computeTextureWH(inputs[0].dims);
    const outputLayout: TextureLayout = {
      width,
      height,
      channels: 4,
      shape: outputShape,
      strides: ShapeUtil.computeStrides(outputShape),
      unpackedShape: outputShape,
    };
    const shaderSource = `
     uniform sampler2D X;
      void main() {
        float value = texture2D(X,TexCoords).r;
        gl_FragColor = encode(value);
      }`;
    return {
      hasMain: true,
      inputLayouts: [inferenceHandler.getOrCreateTextureLayout(inputs[0])],
      outputLayout,
      shaderSource,
    };
  }
  createRunData(inferenceHandler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs = inputs.map((t, i) => inferenceHandler.getOrCreate(t, programInfo.inputLayouts[i]));
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: inferenceHandler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].dataType),
      uniformData: {}
    };
  }
  runInternal(inferenceHandler: WebGLInferenceHandler, input: TextureData): TextureData {
    const outputShape = input.shape;
    const [width, height] = inferenceHandler.session.layoutStrategy.computeTextureWH(input.shape);
    const outputLayout: TextureLayout = {
      width,
      height,
      channels: 4,
      shape: outputShape,
      strides: ShapeUtil.computeStrides(outputShape),
      unpackedShape: outputShape
    };
    const endianness = EncodingGlslLib.isLittleEndian() ? 'rgba.rgba=rgba.abgr;' : '';
    const shaderSource = `
       uniform sampler2D X;
       highp vec4 encodeAsUint8(highp float f) {
        highp float F = abs(f);
        highp float Sign = step(0.0,-f);
        highp float Exponent = floor(log2(F));
        highp float Mantissa = (exp2(- Exponent) * F);
        Exponent = floor(log2(F) + 127.0) + floor(log2(Mantissa));
        highp vec4 rgba;
        rgba[0] = 128.0 * Sign  + floor(Exponent*exp2(-1.0));
        rgba[1] = 128.0 * mod(Exponent,2.0) + mod(floor(Mantissa*128.0),128.0);
        rgba[2] = floor(mod(floor(Mantissa*exp2(23.0 -8.0)),exp2(8.0)));
        rgba[3] = floor(exp2(23.0)*mod(Mantissa,exp2(-15.0)));
        ${endianness}
        rgba = rgba / 255.0; // values need to be normalized to [0,1]
        return rgba;
      }

      void main() {
        float value = texture2D(X,TexCoords).r;
        gl_FragColor = encodeAsUint8(value);
      }`;
    const programInfo = {
      hasMain: true,
      inputLayouts: [input],
      outputLayout,
      shaderSource,
    };
    const artifact = inferenceHandler.programManager.build(programInfo);

    const texture =
        inferenceHandler.backend.glContext.allocateTexture(outputLayout.width, outputLayout.height, 'byte', 4);
    const outputTextureData: TextureData = {...outputLayout, dataType: 'uint8', texture};
    const runData = {inputTextureDatas: [input], outputTextureData, uniformData: {}};

    inferenceHandler.programManager.run(artifact, runData);
    return runData.outputTextureData;
  }
}

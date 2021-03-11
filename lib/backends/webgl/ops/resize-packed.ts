// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {assert} from 'chai';
import {Upsample} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {getGlsl, Glsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData, TextureLayout, WebGLOperator} from '../types';

import {unpackFromChannel} from './packing_utils';

export class WebGLResizePacked extends Upsample implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0], 4, true, inputs[0].dims, true);

    const [roi, scales, outputShape] = this.prepareInputs(inputs);

    const outputLayout =
        handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true});

    const glsl = getGlsl(handler.session.backend.glContext.version);
    return createResizeProgramInfo(
        glsl, this.mode, inputLayout, outputLayout, scales, roi, this.useExtrapolation, this.extrapolationValue,
        this.cubicCoefficientA, this.excludeOutside, this.coordinateTransformMode);
  }

  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTD =
        handler.getOrCreateTextureData(inputs[0], handler.getOrCreateTextureLayout(inputs[0], 1, false, [], true));
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTD.tensor.type);
    return {inputTextureDatas: [inputTD], outputTextureData: outputTD, uniformData: {}};
  }

  protected artifacts: Artifact[];
}

function createResizeProgramInfo(
    glsl: Glsl, mode: string, inputLayout: TextureLayout, outputLayout: TextureLayout, scales: ReadonlyArray<number>,
    roi: ReadonlyArray<number>, extrapolationEnabled: boolean, extrapolationValue: number, cubicCoefficientA: number,
    excludeOutside: boolean, coordinateTransformMode: string): ProgramInfo {
  const isSame = scales.every(s => s === 1) && coordinateTransformMode !== 'tf_crop_and_resize';
  if (isSame) {
    return {
      inputLayouts: [inputLayout],
      outputLayout,
      samplers: ['X'],
      hasMain: true,
      shaderSource: `void main() {
      vec4 v = ${glsl.texture2D}(X, TexCoords);
      ${glsl.output} = v;
    }`
    };
  }
  const outputShape = outputLayout.unpackedShape;
  const dim = outputShape.length;
  assert(dim >= 2);

  const outputHeight = outputShape[dim - 2];
  const outputWidth = outputShape[dim - 1];

  const inputShape = inputLayout.unpackedShape;
  assert(dim === inputShape.length);
  const inputHeight = inputShape[dim - 2];
  const inputWidth = inputShape[dim - 1];

  const scalesHeight = scales[dim - 2];
  const scalesWidth = scales[dim - 1];

  let getSourceFracIndex = '';

  if (mode !== 'linear') {
    // TODO: support other modes
    throw new Error(`resize (packed) does not support mode: '${mode}'`);
  }
  switch (coordinateTransformMode) {
    case 'asymmetric':
      getSourceFracIndex = `
        vec2 getSourceFracIndex(ivec2 coords){
          return vec2(coords) / scaleWH;
        }
        vec4 getSourceFracIndex(ivec4 coords){
          return vec4(coords) / scaleWHWH;
        }
    `;
      break;
    case 'half_pixel':
      getSourceFracIndex = `
        vec2 getSourceFracIndex(ivec2 coords){
          return (vec2(coords) + 0.5) / scaleWH - 0.5;
        }
        vec4 getSourceFracIndex(ivec4 coords){
          return (vec4(coords) + 0.5) / scaleWHWH - 0.5;
        }
    `;
      break;
    case 'align_corners':
      getSourceFracIndex = `
        vec2 getSourceFracIndex(ivec2 coords){
          vec2 resized = vec2(${outputWidth}.0 - 1.0, ${outputHeight}.0 - 1.0);
          vec2 original = vec2(${inputWidth}.0 - 1.0, ${inputHeight}.0 - 1.0);
          vec2 new_scale = original / resized;
          return vec2(coords) * new_scale;
        }
        vec4 getSourceFracIndex(ivec4 coords){
          vec4 resized = vec4(${outputWidth}.0 - 1.0, ${outputHeight}.0 - 1.0, ${outputWidth}.0 - 1.0, ${
          outputHeight}.0 - 1.0);
          vec4 original = vec4(${inputWidth}.0 - 1.0, ${inputHeight}.0 - 1.0, ${inputWidth}.0 - 1.0, ${
          inputHeight}.0 - 1.0);
          vec4 new_scale = original / resized;
          return vec4(coords) * new_scale;
        }
      `;
      break;
    default:
      // TODO:supporting other coordinateTransformModes
      throw new Error(`resize (packed) does not support coordinateTransformMode: '${coordinateTransformMode}'`);
  }

  const unpackChannel = unpackFromChannel(dim);
  const shader = `
        const vec2 inputWH = vec2(${inputHeight}.0, ${inputWidth}.0);
        const vec2 scaleWH = vec2(${scalesHeight}.0, ${scalesWidth}.0);
        const vec4 scaleWHWH = vec4(${scalesHeight}.0, ${scalesWidth}.0, ${scalesHeight}.0, ${scalesWidth}.0);
        ${unpackChannel}
        ${getSourceFracIndex}
        float getAValue(int x10, int r, int c, int d) {
          return getChannel(getA(x10, r, c, d), vec2(c, d));
        }
        void main() {
          ivec4 rc = getOutputCoords();

          int batch = rc[0];
          int depth = rc[1];

          // retrieve the 4 coordinates that will be packed in one texel in output texture.
          // ivec2 x00 = rc.wz;
          // ivec2 x01 = ivec2(rc.w, rc.z+1);
          // ivec2 x10 = ivec2(rc.w+1, rc.z);
          // ivec2 x11 = ivec2(rc.w+1, rc.z + 1);
          ivec4 coords = ivec4(rc.wz, rc.w + 1, rc.z + 1);
          vec4 sourceFrac = getSourceFracIndex(coords);

          // vec2 sourceFracR = getSourceFracIndex(x00);
          // vec2 sourceFracG = getSourceFracIndex(x01);
          // vec2 sourceFracB = getSourceFracIndex(x10);
          // vec2 sourceFracA = getSourceFracIndex(x11);

          ivec4 rr = ivec4(max(sourceFrac.xy, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.xy)));
          ivec4 gg = ivec4(max(sourceFrac.xw, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.xw)));
          ivec4 bb = ivec4(max(sourceFrac.zy, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.zy)));
          ivec4 aa = ivec4(max(sourceFrac.zw, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.zw)));

          bool hasNextRow = rc.w < ${outputHeight - 1};
          bool hasNextCol = rc.z < ${outputWidth - 1};

          vec4 topLeft = vec4(
            getAValue(batch, depth, rr.x, rr.y),
            hasNextCol ? getAValue(batch, depth, gg.x, gg.y)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, bb.x, bb.y)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, aa.x, aa.y) : 0.0);

          vec4 topRight = vec4(
            getAValue(batch, depth, rr.x, rr.w),
            hasNextCol ? getAValue(batch, depth, gg.x, gg.w)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, bb.x, bb.w)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, aa.x, aa.w) : 0.0);

          vec4 bottomLeft = vec4(
            getAValue(batch, depth, rr.z, rr.y),
            hasNextCol ? getAValue(batch, depth, gg.z, gg.y)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, bb.z, bb.y)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, aa.z, aa.y) : 0.0);



          vec4 bottomRight = vec4(
            getAValue(batch, depth, rr.z, rr.w),
            hasNextCol ? getAValue(batch, depth, gg.z, gg.w)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, bb.z, bb.w)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, aa.z, aa.w) : 0.0);


          // vec4 fracX = vec4(sourceFracR.x, sourceFracG.x, sourceFracB.x, sourceFracA.x) ;
          // fracX = fracX - floor(fracX);
          // vec4 fracY = vec4(sourceFracR.y, sourceFracG.y, sourceFracB.y, sourceFracA.y);
          // fracY = fracY - floor(fracY);

          // vec4 clampX = clamp(fracX, vec4(0.0), vec4(1.0));
          // vec4 clampY = clamp(fracY, vec4(0.0), vec4(1.0));
          vec4 frac = vec4(sourceFrac) - floor(sourceFrac);
          vec4 clampFrac = clamp(frac, vec4(0.0), vec4(1.0));

          // vec4 top = mix(topLeft, topRight, clampY);
          // vec4 bottom = mix(bottomLeft, bottomRight, clampY);
          // vec4 newValue = mix(top, bottom, clampX);
          vec4 top = mix(topLeft, topRight, clampFrac.ywyw);
          vec4 bottom = mix(bottomLeft, bottomRight, clampFrac.ywyw);
          vec4 newValue = mix(top, bottom, clampFrac.xxzz);

          outputColor = vec4(newValue);
        }
      `;
  return {
    inputLayouts: [inputLayout],
    outputLayout,
    samplers: ['A'],
    shaderSource: shader,
    hasMain: true,
    expectPackedInputs: true,
    expectPackedoutputs: true,
  };
}

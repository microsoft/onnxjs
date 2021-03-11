// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Upsample} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {getGlsl, Glsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {Artifact, ProgramInfo, RunData, TextureLayout, VariableInfo, WebGLOperator} from '../types';

export class WebGLUpsample extends Upsample implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    const inputLayout = handler.getOrCreateTextureLayout(inputs[0]);
    const [roi, scales, outputShape] = this.prepare(inputs);
    this.roiCache = roi;
    this.scalesCache = scales.map(x => Math.ceil(x));
    const outputLayout = handler.createTextureLayoutFromShape(outputShape);
    const dim = outputShape.length;

    const glsl = getGlsl(handler.session.backend.glContext.version);
    if (this.isResize) {
      this.mappingOriginCache = [];
      this.mappingWeightCache = [];
      this.mappingExtrapolateCache = [];
      this.mappingCoeffCache = [];
      return createResizeProgramInfo(
          glsl, this.mode, dim, inputLayout, outputLayout, scales, roi, this.useExtrapolation, this.extrapolationValue,
          this.cubicCoefficientA, this.excludeOutside, this.coordinateTransformMode, this.getOriginalCoordinate,
          this.getNearestPixel, this.mappingOriginCache, this.mappingWeightCache, this.mappingExtrapolateCache,
          this.mappingCoeffCache);
    } else {
      return createUpsampleProgramInfo(glsl, this.mode, dim, inputLayout, outputLayout, scales);
    }
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTD = handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0]);
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTD.tensor.type);
    return {
      inputTextureDatas: [inputTD],
      outputTextureData: outputTD,
      uniformData: {
        scales: this.scalesCache,
        mo: this.mappingOriginCache,
        me: this.mappingExtrapolateCache,
        mw: this.mappingWeightCache,
        mc: this.mappingCoeffCache
      }
    };
  }

  protected roiCache: number[];
  protected scalesCache: number[];
  protected mappingOriginCache: number[];
  protected mappingExtrapolateCache: number[];
  protected mappingWeightCache: number[];
  protected mappingCoeffCache: number[];

  protected artifacts: Artifact[];
}

function createResizeProgramInfo(
    glsl: Glsl, mode: string, dim: number, inputLayout: TextureLayout, outputLayout: TextureLayout,
    scales: ReadonlyArray<number>, roi: ReadonlyArray<number>, extrapolationEnabled: boolean,
    extrapolationValue: number, cubicCoefficientA: number, excludeOutside: boolean, coordinateTransformMode: string,
    getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc, getNearestPixel: Upsample.GetNearestPixelFunc,
    mappingOriginCache: number[], mappingWeightCache: number[], mappingExtrapolateCache: number[],
    mappingCoeffCache: number[]): ProgramInfo {
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

  const inputShape = inputLayout.shape;
  const inputHeight = inputShape[dim - 2];
  const inputWidth = inputShape[dim - 1];
  const outputShape = outputLayout.shape;
  const outputHeight = outputShape[dim - 2];
  const outputWidth = outputShape[dim - 1];
  const scalesHeight = scales[dim - 2];
  const scalesWidth = scales[dim - 1];
  const roiStartHeight = roi[dim - 2];
  const roiEndHeight = roi[dim - 2 + dim];
  const roiStartWidth = roi[dim - 1];
  const roiEndWidth = roi[dim - 1 + dim];

  const precalculatedPitches = shaderPrecalculatedPitches(dim, outputShape, inputShape);
  const getInputFloatFunction = shaderGetInputFloatFunction(inputLayout, glsl);

  if (mode === 'nearest') {
    const could2d =
        dim >= 2 && coordinateTransformMode !== 'tf_crop_and_resize' && scales.some((s, i) => s === 1 && i < dim - 2);
    if (could2d) {
      fillResizeNearestMapping2D(
          inputHeight, inputWidth, outputHeight, outputWidth, scalesHeight, scalesWidth, roiStartHeight, roiEndHeight,
          roiStartWidth, roiEndWidth, extrapolationEnabled, getOriginalCoordinate, getNearestPixel, mappingOriginCache,
          mappingExtrapolateCache);
      const variables: VariableInfo[] = [{name: 'mo', type: 'int', arrayLength: mappingOriginCache.length}];
      if (extrapolationEnabled) {
        variables.push({name: 'me', type: 'int', arrayLength: mappingExtrapolateCache.length});
      }
      return {
        inputLayouts: [inputLayout],
        outputLayout,
        samplers: ['X'],
        shaderSource: `
        ${getInputFloatFunction}
float process(int indices[${dim}]) {
  int input_index = 0;
  int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

  ${precalculatedPitches}

  ${
            dim === 2 ? 'int m = output_index; int imageid = 0;' :
                        `int imageid = output_index / output_pitches[${
                            dim - 3}]; int m = output_index - imageid * output_pitches[${dim - 3}];`}
  int h = m / output_pitches[${dim - 2}];
  int w = m - h * output_pitches[${dim - 2}];

  ${
            extrapolationEnabled ? `if (me[h] + me[${outputHeight}+w] > 0) {
    return float(${extrapolationValue});
  }` :
                                   ''}

  input_index = ${dim === 2 ? '0' : `imageid * input_pitches[${dim - 3}]`} + mo[h] * input_pitches[${dim - 2}] + mo[${
            outputHeight}+w];
  return getInputFloat(input_index);
}`,
        variables
      };
    }

    // could2d === false
    throw new Error('non-2D nearest mode is not implemented yet');

  } else if (mode === 'linear') {
    fillResizeBilinearCoordinateMapping(
        inputHeight, inputWidth, outputHeight, outputWidth, scalesHeight, scalesWidth, roiStartHeight, roiEndHeight,
        roiStartWidth, roiEndWidth, extrapolationEnabled, getOriginalCoordinate, mappingOriginCache, mappingWeightCache,
        mappingExtrapolateCache);
    const variables: VariableInfo[] = [
      {name: 'mo', type: 'int', arrayLength: mappingOriginCache.length},
      {name: 'mw', type: 'float', arrayLength: mappingWeightCache.length}
    ];
    if (extrapolationEnabled) {
      variables.push({name: 'me', type: 'int', arrayLength: mappingExtrapolateCache.length});
    }
    return {
      inputLayouts: [inputLayout],
      outputLayout,
      samplers: ['X'],
      shaderSource: `
    ${getInputFloatFunction}
float process(int indices[${dim}]) {
  int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

  ${precalculatedPitches}

  ${
          dim === 2 ? 'int m = output_index; int imageid = 0;' :
                      `int imageid = output_index / output_pitches[${
                          dim - 3}]; int m = output_index - imageid * output_pitches[${dim - 3}];`}
  int input_index = imageid * ${inputHeight * inputWidth};
  int output_y = m / output_pitches[${dim - 2}];
  int output_x = m - output_y * output_pitches[${dim - 2}];

  ${
          extrapolationEnabled ? `if (me[output_y] + me[${outputHeight}+output_x] > 0) {
      return float(${extrapolationValue});
    }` :
                                 ''}

  float y_offset_0 = mw[output_y];
  int y_int = mo[output_y];
  float x_offset_0 = mw[${outputHeight}+output_x];
  int x_int = mo[${outputHeight}+output_x];
  input_index += y_int * ${inputWidth} + x_int;

  float x00 = getInputFloat(input_index);
  bool end_of_h = (y_int >= ${inputHeight} - 1);
  bool end_of_w = (x_int >= ${inputWidth} - 1);
  float x10 = end_of_w ? x00 : getInputFloat(input_index + 1);
  float x01 = end_of_h ? x00 : getInputFloat(input_index + ${inputWidth});
  float x11 = end_of_w ? x01 : (end_of_h ? x10 : getInputFloat(input_index + ${inputWidth} + 1));

  float y_offset_1 = 1.0 - y_offset_0;
  float x_offset_1 = 1.0 - x_offset_0;

  return x00 * (y_offset_1 * x_offset_1) +
         x01 * (y_offset_0 * x_offset_1) +
         x10 * (y_offset_1 * x_offset_0) +
         x11 * (y_offset_0 * x_offset_0);
}`,
      variables
    };

  } else {  // cubic
    fillResizeCubicCoordinateMapping(
        inputHeight, inputWidth, outputHeight, outputWidth, scalesHeight, scalesWidth, roiStartHeight, roiEndHeight,
        roiStartWidth, roiEndWidth, extrapolationEnabled, cubicCoefficientA, excludeOutside, getOriginalCoordinate,
        mappingOriginCache, mappingExtrapolateCache, mappingCoeffCache);
    const variables: VariableInfo[] = [
      {name: 'mo', type: 'int', arrayLength: mappingOriginCache.length},
      {name: 'mc', type: 'float', arrayLength: mappingCoeffCache.length}
    ];
    if (extrapolationEnabled) {
      variables.push({name: 'me', type: 'int', arrayLength: mappingExtrapolateCache.length});
    }
    return {
      inputLayouts: [inputLayout],
      outputLayout,
      samplers: ['X'],
      shaderSource: `
    ${getInputFloatFunction}
float rowwise(int x, int y, int offset, float coeff0, float coeff1, float coeff2, float coeff3) {
  int row_index = max(0, min(y, ${inputHeight - 1})) * ${inputWidth};
  return coeff0 * getInputFloat(offset + row_index + max(0, min(x - 1, ${inputWidth} - 1))) +
         coeff1 * getInputFloat(offset + row_index + max(0, min(x, ${inputWidth} - 1))) +
         coeff2 * getInputFloat(offset + row_index + max(0, min(x + 1, ${inputWidth} - 1))) +
         coeff3 * getInputFloat(offset + row_index + max(0, min(x + 2, ${inputWidth} - 1)));
}
float process(int indices[${dim}]) {
  int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

  ${precalculatedPitches}

  ${
          dim === 2 ? 'int m = output_index; int imageid = 0;' :
                      `int imageid = output_index / output_pitches[${
                          dim - 3}]; int m = output_index - imageid * output_pitches[${dim - 3}];`}
  int input_index = imageid * ${inputHeight * inputWidth};
  int output_y = m / output_pitches[${dim - 2}];
  int output_x = m - output_y * output_pitches[${dim - 2}];

  ${
          extrapolationEnabled ? `if (me[output_y] + me[${outputHeight}+output_x] > 0) {
      return float(${extrapolationValue});
    }` :
                                 ''}

  float w0 = mc[(${outputHeight}+output_x)*4];
  float w1 = mc[(${outputHeight}+output_x)*4+1];
  float w2 = mc[(${outputHeight}+output_x)*4+2];
  float w3 = mc[(${outputHeight}+output_x)*4+3];
  int x_int = mo[${outputHeight}+output_x];
  float y0 = mc[output_y*4];
  float y1 = mc[output_y*4+1];
  float y2 = mc[output_y*4+2];
  float y3 = mc[output_y*4+3];
  int y_int = mo[output_y];

  return y0 * rowwise(x_int, y_int - 1, input_index, w0, w1, w2, w3) +
         y1 * rowwise(x_int, y_int, input_index, w0, w1, w2, w3) +
         y2 * rowwise(x_int, y_int + 1, input_index, w0, w1, w2, w3) +
         y3 * rowwise(x_int, y_int + 2, input_index, w0, w1, w2, w3);
}`,
      variables
    };
  }
}

function fillResizeNearestMapping2D(
    inputHeight: number, inputWidth: number, outputHeight: number, outputWidth: number, scalesHeight: number,
    scalesWidth: number, roiStartHeight: number, roiEndHeight: number, roiStartWidth: number, roiEndWidth: number,
    extrapolationEnabled: boolean, getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc,
    getNearestPixel: Upsample.GetNearestPixelFunc, mappingOrigin: number[], mappingExtrapolation: number[]): void {
  for (let i = 0; i < outputHeight; i++) {
    let dim = i;
    const originalCoord =
        getOriginalCoordinate(dim, scalesHeight, outputHeight, inputHeight, roiStartHeight, roiEndHeight);
    // extrapolate
    mappingExtrapolation.push((extrapolationEnabled && (originalCoord < 0 || originalCoord > inputHeight - 1)) ? 1 : 0);
    dim = Math.max(0, Math.min(inputHeight - 1, getNearestPixel(originalCoord, scalesHeight < 1)));
    // origin
    mappingOrigin.push(dim);
  }

  for (let i = 0; i < outputWidth; i++) {
    let dim = i;
    const originalCoord = getOriginalCoordinate(dim, scalesWidth, outputWidth, inputWidth, roiStartWidth, roiEndWidth);
    // extrapolate
    mappingExtrapolation.push((extrapolationEnabled && (originalCoord < 0 || originalCoord > inputWidth - 1)) ? 1 : 0);
    dim = Math.max(0, Math.min(inputWidth - 1, getNearestPixel(originalCoord, scalesWidth < 1)));
    // origin
    mappingOrigin.push(dim);
  }
}

function fillResizeBilinearCoordinateMapping(
    inputHeight: number, inputWidth: number, outputHeight: number, outputWidth: number, scalesHeight: number,
    scalesWidth: number, roiStartHeight: number, roiEndHeight: number, roiStartWidth: number, roiEndWidth: number,
    extrapolationEnabled: boolean, getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc, mappingOrigin: number[],
    mappingWeight: number[], mappingExtrapolation: number[]) {
  for (let i = 0; i < outputHeight; i++) {
    let inputY = getOriginalCoordinate(i, scalesHeight, outputHeight, inputHeight, roiStartHeight, roiEndHeight);
    mappingExtrapolation.push((extrapolationEnabled && (inputY < 0 || inputY > inputHeight - 1)) ? 1 : 0);
    inputY = Math.max(0, Math.min(inputY, inputHeight - 1));
    const intY = Math.floor(inputY);
    mappingOrigin.push(intY);
    mappingWeight.push(intY >= inputHeight - 1 ? 0.5 : inputY - intY);
  }
  for (let i = 0; i < outputWidth; i++) {
    let inputX = getOriginalCoordinate(i, scalesWidth, outputWidth, inputWidth, roiStartWidth, roiEndWidth);
    mappingExtrapolation.push((extrapolationEnabled && (inputX < 0 || inputX > inputWidth - 1)) ? 1 : 0);
    inputX = Math.max(0, Math.min(inputX, inputWidth - 1));
    const intX = Math.floor(inputX);
    mappingOrigin.push(intX);
    mappingWeight.push(intX >= inputWidth - 1 ? 0.5 : inputX - intX);
  }
}

function fillResizeCubicCoordinateMapping(
    inputHeight: number, inputWidth: number, outputHeight: number, outputWidth: number, scalesHeight: number,
    scalesWidth: number, roiStartHeight: number, roiEndHeight: number, roiStartWidth: number, roiEndWidth: number,
    extrapolationEnabled: boolean, cubicCoefficientA: number, excludeOutside: boolean,
    getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc, mappingOrigin: number[], mappingExtrapolation: number[],
    mappingCoeffCache: number[]) {
  for (let i = 0; i < outputHeight + outputWidth; i++) {
    const isY = i < outputHeight;
    const maxInputCoord = isY ? inputHeight : inputWidth;
    const inputCoord = getOriginalCoordinate(
        isY ? i : i - outputHeight, isY ? scalesHeight : scalesWidth, isY ? outputHeight : outputWidth, maxInputCoord,
        isY ? roiStartHeight : roiStartWidth, isY ? roiEndHeight : roiEndWidth);
    const intCoord = Math.floor(inputCoord);
    const sCoord = Math.abs(intCoord - inputCoord);
    let coeffSum = 1.0;
    let coeff0 = ((cubicCoefficientA * (sCoord + 1) - 5 * cubicCoefficientA) * (sCoord + 1) + 8 * cubicCoefficientA) *
            (sCoord + 1) -
        4 * cubicCoefficientA;
    let coeff1 = ((cubicCoefficientA + 2) * sCoord - (cubicCoefficientA + 3)) * sCoord * sCoord + 1;
    let coeff2 = ((cubicCoefficientA + 2) * (1 - sCoord) - (cubicCoefficientA + 3)) * (1 - sCoord) * (1 - sCoord) + 1;
    let coeff3 = ((cubicCoefficientA * (2 - sCoord) - 5 * cubicCoefficientA) * (2 - sCoord) + 8 * cubicCoefficientA) *
            (2 - sCoord) -
        4 * cubicCoefficientA;
    if (excludeOutside) {
      coeff0 = (intCoord - 1 < 0 || intCoord - 1 >= maxInputCoord) ? 0.0 : coeff0;
      coeff1 = (intCoord + 0 < 0 || intCoord + 0 >= maxInputCoord) ? 0.0 : coeff1;
      coeff2 = (intCoord + 1 < 0 || intCoord + 1 >= maxInputCoord) ? 0.0 : coeff2;
      coeff3 = (intCoord + 2 < 0 || intCoord + 2 >= maxInputCoord) ? 0.0 : coeff3;
      coeffSum = coeff0 + coeff1 + coeff2 + coeff3;
    }
    mappingOrigin.push(intCoord);
    mappingExtrapolation.push((extrapolationEnabled && (inputCoord < 0 || inputCoord > maxInputCoord - 1)) ? 1 : 0);
    mappingCoeffCache.push(coeff0 / coeffSum);
    mappingCoeffCache.push(coeff1 / coeffSum);
    mappingCoeffCache.push(coeff2 / coeffSum);
    mappingCoeffCache.push(coeff3 / coeffSum);
  }
}

function createUpsampleProgramInfo(
    glsl: Glsl, mode: string, dim: number, inputLayout: TextureLayout, outputLayout: TextureLayout,
    scales: ReadonlyArray<number>): ProgramInfo {
  const outputShape = outputLayout.shape;
  const inputShape = inputLayout.shape;
  const precalculatedPitches = shaderPrecalculatedPitches(dim, outputShape, inputShape);
  const getInputFloatFunction = shaderGetInputFloatFunction(inputLayout, glsl);

  const shaderSource = mode === 'nearest' ?
      // nearest
      `
        ${getInputFloatFunction}
        float process(int indices[${dim}]) {
          int input_index = 0;
          int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

          ${precalculatedPitches}

          int d, m;
          for (int dim = 0; dim < ${dim}; ++dim) {
            d = output_index / output_pitches[dim];
            m = output_index - d * output_pitches[dim];
            output_index = m;

            if (scales[dim] != 1 && d > 0) {
              int d2 = d / scales[dim];
              m = d - d2 * scales[dim];
              d = d2;
            }
            input_index += input_pitches[dim] * d;
          }

          return getInputFloat(input_index);
        }` :
      dim === 4 ?
      // bilinear 4D
          `
        ${getInputFloatFunction}
        float process(int indices[4]) {
          int input_index = 0;
          int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

          ${precalculatedPitches}

          int m;
          int index_of_dim0, index_of_dim1, index_of_dim2, index_of_dim3;
          index_of_dim0 = output_index / output_pitches[0];
          m = output_index - index_of_dim0 * output_pitches[0];
          index_of_dim1 = m / output_pitches[1];
          m = m - index_of_dim1 * output_pitches[1];
          index_of_dim2 = m / output_pitches[2];
          m = m - index_of_dim2 * output_pitches[2];
          index_of_dim3 = m;

          int index_of_input_dim2, index_of_input_dim3, x_offset, y_offset;
          index_of_input_dim2 = index_of_dim2 / scales[2];
          y_offset = index_of_dim2 - index_of_input_dim2 * scales[2];
          index_of_input_dim3 = index_of_dim3 / scales[3];
          x_offset = index_of_dim3 - index_of_input_dim3 * scales[3];

          input_index = index_of_dim0 * input_pitches[0] +
                        index_of_dim1 * input_pitches[1] +
                        index_of_input_dim2 * input_pitches[2] +
                        index_of_input_dim3;

          float x00 = getInputFloat(input_index);
          float x10, x01, x11;

          bool end_of_dim2 = false;
          if (index_of_input_dim2 == (${inputShape[2]} - 1)) {
            // It's the end in dimension 2
            x01 = x00;
            end_of_dim2 = true;
          } else {
            x01 = getInputFloat(input_index + input_pitches[2]);
          }

          if (index_of_input_dim3 == (input_pitches[2] - 1)) {
            // It's the end in dimension 3
            x10 = x00;
            x11 = x01;
          }
          else {
            x10 = getInputFloat(input_index + 1);
            x11 = end_of_dim2 ? x10 : getInputFloat(input_index + input_pitches[2] + 1);
          }

          float y0 = x00 + float(y_offset) * (x01 - x00) / float(scales[2]);
          float y1 = x10 + float(y_offset) * (x11 - x10) / float(scales[2]);
          return y0 + float(x_offset) * (y1 - y0) / float(scales[3]);
        }` :
          // bilinear 2D
          `
        ${getInputFloatFunction}
        float process(int indices[2]) {
          int input_index = 0;
          int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});

          ${precalculatedPitches}

          int m;
          int index_of_dim0, index_of_dim1;
          index_of_dim0 = output_index / output_pitches[0];
          m = output_index - index_of_dim0 * output_pitches[0];
          index_of_dim1 = m;

          int index_of_input_dim0, index_of_input_dim1, x_offset, y_offset;
          index_of_input_dim0 = index_of_dim0 / scales[0];
          y_offset = index_of_dim0 - index_of_input_dim0 * scales[0];
          index_of_input_dim1 = index_of_dim1 / scales[1];
          x_offset = index_of_dim1 - index_of_input_dim1 * scales[1];

          input_index = index_of_input_dim0 * input_pitches[0] + index_of_input_dim1;

          float x00 = getInputFloat(input_index);
          float x10, x01, x11;

          bool end_of_dim0 = false;
          if (index_of_input_dim0 == (${inputShape[0]} - 1)) {
            // It's the end in dimension 0
            x01 = x00;
            end_of_dim0 = true;
          } else {
            x01 = getInputFloat(input_index + input_pitches[0]);
          }

          if (index_of_input_dim1 == (input_pitches[0] - 1)) {
            // It's the end in dimension 1
            x10 = x00;
            x11 = x01;
          }
          else {
            x10 = getInputFloat(input_index + 1);
            x11 = end_of_dim0 ? x10 : getInputFloat(input_index + input_pitches[0] + 1);
          }

          float y0 = x00 + float(y_offset) * (x01 - x00) / float(scales[0]);
          float y1 = x10 + float(y_offset) * (x11 - x10) / float(scales[0]);
          return y0 + float(x_offset) * (y1 - y0) / float(scales[1]);
        }`;
  return {
    inputLayouts: [inputLayout],
    outputLayout,
    samplers: ['X'],
    shaderSource,
    variables: [{name: 'scales', type: 'int', arrayLength: scales.length}]
  };
}

function shaderPrecalculatedPitches(
    dim: number, outputShape: ReadonlyArray<number>, inputShape: ReadonlyArray<number>) {
  const outputPitches = new Array<number>(dim);
  const inputPitches = new Array<number>(dim);
  let precalculatedPitches = `
        int output_pitches[${dim}];
        int input_pitches[${dim}];
        `;
  for (let d = dim - 1; d >= 0; d--) {
    outputPitches[d] = (d === dim - 1) ? 1 : outputPitches[d + 1] * outputShape[d + 1];
    inputPitches[d] = (d === dim - 1) ? 1 : inputPitches[d + 1] * inputShape[d + 1];

    precalculatedPitches += `
        output_pitches[${d}] = ${outputPitches[d]};
        input_pitches[${d}] = ${inputPitches[d]};
        `;
  }
  return precalculatedPitches;
}

function shaderGetInputFloatFunction(inputLayout: TextureLayout, glsl: Glsl) {
  return `
float getInputFloat(int index) {
  vec2 coords = offsetToCoords(index, ${inputLayout.width}, ${inputLayout.height});
  float value = getColorAsFloat(${glsl.texture2D}(X, coords));
  return value;
}
`;
}

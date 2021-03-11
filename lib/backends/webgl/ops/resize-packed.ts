// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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
    const dim = outputShape.length;

    const glsl = getGlsl(handler.session.backend.glContext.version);
    return createResizeProgramInfo(
        glsl, this.mode, dim, inputLayout, outputLayout, scales, roi, this.useExtrapolation, this.extrapolationValue,
        this.cubicCoefficientA, this.excludeOutside, this.coordinateTransformMode);
    // if (this.isResize) {
    //   this.mappingOriginCache = [];
    //   this.mappingWeightCache = [];
    //   this.mappingExtrapolateCache = [];
    //   this.mappingCoeffCache = [];
    //   return createResizeProgramInfo(
    //       glsl, this.mode, dim, inputLayout, outputLayout, scales, roi, this.useExtrapolation,
    //       this.extrapolationValue, this.cubicCoefficientA, this.excludeOutside, this.coordinateTransformMode,
    //       this.mappingOriginCache, this.mappingWeightCache, this.mappingExtrapolateCache, this.mappingCoeffCache);
    // } else {
    //   return createUpsampleProgramInfo(glsl, this.mode, dim, inputLayout, outputLayout, scales);
    // }
  }
  // getAndAalidateInputs(inputs: Tensor[]): [number[], number[], ReadonlyArray<number>] {
  //   const x = inputs[0];
  //   const xDims = x.dims;

  //   // TODO: get roi data

  //   // Get scales and sizes
  //   let scales = this.scales;
  //   let outputSizes: number[]|undefined;
  //   if (!scales) {
  //     const scalesTensor = inputs[this.scalesInputIdx];
  //     if (scalesTensor && scalesTensor.size !== 0) {
  //       if (inputs[this.sizesInputIdx]) {
  //         throw new Error('Only one of scales or sizes must be provided as input.');
  //       }
  //       scales = parseScalesData(scalesTensor, this.mode, this.isResize);
  //     } else {
  //       const sizesTensor = inputs[this.sizesInputIdx];
  //       if (!sizesTensor || sizesTensor.size === 0) {
  //         throw new Error('Either scales or sizes MUST be provided as input.');
  //       }

  //       outputSizes = Array.from(sizesTensor.integerData);
  //       scales = parseScalesDataFromOutputSize(outputSizes, xDims, this.mode, this.isResize);
  //     }
  //   } else {
  //     if (inputs[this.sizesInputIdx]) {
  //       throw new Error('Only one of scales or sizes must be provided as input.');
  //     }
  //   }

  //   const yDims = outputSizes || computeOutputShape(scales, xDims);

  //   return [this.roi, scales, yDims];
  // }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    // const inputTD = handler.getOrCreateTextureData(inputs[0], programInfo.inputLayouts[0]);
    const inputTD =
        handler.getOrCreateTextureData(inputs[0], handler.getOrCreateTextureLayout(inputs[0], 1, false, [], true));
    const outputTD = handler.createTextureDataFromLayout(programInfo.outputLayout, inputTD.tensor.type);
    return {
      inputTextureDatas: [inputTD],
      outputTextureData: outputTD,
      uniformData: {}
      // uniformData: {
      //   scales: this.scalesCache,
      //   mo: this.mappingOriginCache,
      //   me: this.mappingExtrapolateCache,
      //   mw: this.mappingWeightCache,
      //   mc: this.mappingCoeffCache
      // }
    };
  }
  // protected roiCache: number[];
  // protected scalesCache: number[];
  // protected mappingOriginCache: number[];
  // protected mappingExtrapolateCache: number[];
  // protected mappingWeightCache: number[];
  // protected mappingCoeffCache: number[];

  protected artifacts: Artifact[];
}

function createResizeProgramInfo(
    glsl: Glsl, mode: string, dim: number, inputLayout: TextureLayout, outputLayout: TextureLayout,
    scales: ReadonlyArray<number>, roi: ReadonlyArray<number>, extrapolationEnabled: boolean,
    extrapolationValue: number, cubicCoefficientA: number, excludeOutside: boolean,
    coordinateTransformMode: string): ProgramInfo {
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

  const inputShape = inputLayout.unpackedShape;
  const inputHeight = inputShape[dim - 2];
  const inputWidth = inputShape[dim - 1];
  // const inputHeight = inputLayout.height;
  // const inputWidth = inputLayout.width;
  const outputShape = outputLayout.unpackedShape;

  // TODO: what if input has only 2 dims
  // const depth = outputShape[dim - 3];
  const outputHeight = outputShape[dim - 2];
  const outputWidth = outputShape[dim - 1];
  const scalesHeight = scales[dim - 2];
  const scalesWidth = scales[dim - 1];

  let shader = '';
  let getSourceFracIndex = '';
  const scaleWH = `const vec2 scaleWH = vec2(${scalesHeight}.0, ${scalesWidth}.0);`;

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
    `;
      break;
    case 'half_pixel':
      getSourceFracIndex = `
    vec2 getSourceFracIndex(ivec2 coords){
      return (vec2(coords) + 0.5)/ scaleWH - 0.5;
    }
    `;
      break;
    case 'align_corners':
      getSourceFracIndex = `
      vec2 getSourceFracIndex(ivec2 coords){
        vec2 resized = vec2(${outputWidth}.0 - 1.0, ${outputHeight}.0 - 1.0);
        vec2 original = vec2(${inputWidth}.0 - 1.0, ${inputHeight}.0 - 1.0);
        vec2 new_scale = resized / original;
        return vec2(coords) / new_scale;
      }
      `;
      break;
    default:
      throw new Error(`resize (packed) does not support coordinateTransformMode: '${coordinateTransformMode}'`);
  }

  // let sourceFracIndexRC: string;
  // sourceFracIndexRC = `(vec4(outputPacked) + 0.5)/ effectiveInputOverOutputRatioRC - 0.5`;

  const unpackChannel = unpackFromChannel(dim);
  // if (halfPixelCenters) {
  //   sourceFracIndexRC = `(vec3(yRC) + vec3(0.5)) * ` +
  //       `effectiveInputOverOutputRatioRC - vec3(0.5)`;
  // }
  // else {
  //   sourceFracIndexRC = `vec3(yRC) * effectiveInputOverOutputRatioRC`;
  // }
  shader = `
        // const vec3 effectiveInputOverOutputRatioRC = vec3(
        //   ${scalesHeight},
        //   ${scalesHeight},
        //   ${scalesWidth},
        //   ${scalesWidth});
        const vec3 inputShapeRC = vec3(${inputHeight}.0, ${inputWidth}.0,
                                      ${inputWidth}.0);
        ${scaleWH}
        const vec2 inputWH = vec2(${inputHeight}.0, ${inputWidth}.0);
        const vec2 outputWH = vec2(${outputHeight}.0, ${outputWidth}.0);
        ${unpackChannel}
        ${getSourceFracIndex}
        float getAValue(int b, int r, int c, int d) {
          return getChannel(getA(b, r, c, d), vec2(c, d));
        }
        void main() {
          ivec4 rc = getOutputCoords();

          int batch = rc[0];
          int depth = rc[1];

          ivec2 r = rc.wz;
          ivec2 g = ivec2(rc.w, rc.z+1);
          ivec2 b = ivec2(rc.w+1, rc.z);
          ivec2 a = ivec2(rc.w+1, rc.z + 1);

          vec2 sourceFracR = getSourceFracIndex(r);
          vec2 sourceFracG = getSourceFracIndex(g);
          vec2 sourceFracB = getSourceFracIndex(b);
          vec2 sourceFracA = getSourceFracIndex(a);

          ivec4 rr = ivec4(max(sourceFracR, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFracR)));
          ivec4 gg = ivec4(max(sourceFracG, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFracG)));
          ivec4 bb = ivec4(max(sourceFracB, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFracB)));
          ivec4 aa = ivec4(max(sourceFracA, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFracA)));

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
          // vec4 fracY = vec4(sourceFracR.y, sourceFracG.y, sourceFracB.y, sourceFracA.y);

          vec4 fracX = vec4(sourceFracR.x, sourceFracG.x, sourceFracB.x, sourceFracA.x) ;
          fracX = fracX - floor(fracX);
          vec4 fracY = vec4(sourceFracR.y, sourceFracG.y, sourceFracB.y, sourceFracA.y);
          fracY = fracY - floor(fracY);

          vec4 clampX = clamp(fracX, vec4(0.0), vec4(1.0));
          vec4 clampY = clamp(fracY, vec4(0.0), vec4(1.0));

          vec4 top = mix(topLeft, topRight, clampY);
          vec4 bottom = mix(bottomLeft, bottomRight, clampY);
          vec4 newValue = mix(top, bottom, clampX);

          //outputColor = vec4(float(hasNextCol), float(hasNextRow), 0, 0);
          outputColor = vec4(newValue);
          //outputColor = vec4(sourceFloorRCY, 0);
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

// function scalesValidataion(scales: number[], mode: string, isResize: boolean) {
//   if (!isResize) {
//     for (const scale of scales) {
//       if (scale < 1) {
//         throw new Error('Scale value should be greater than or equal to 1.');
//       }
//     }
//   } else {
//     for (const scale of scales) {
//       if (scale <= 0) {
//         throw new Error('Scale value should be greater than 0.');
//       }
//     }
//   }
//   if (mode === 'linear' || mode === 'cubic') {
//     if (scales.length !== 2 && (scales.length !== 4 || scales[0] !== 1 || scales[1] !== 1)) {
//       throw new Error(`'Linear' mode and 'Cubic' mode only support 2-D inputs ('Bilinear', 'Bicubic') or 4-D inputs\
// with the corresponding outermost 2 scale values being 1 in the ${isResize ? 'Resize' : 'Upsample'} opeartor.`);
//     }
//   }
// }

// function parseRoiData(roi: Tensor): number[] {
//   return roi.size > 0 ? Array.from(roi.floatData) : [];
// }

// function parseScalesData(scale: Tensor, mode: string, isResize: boolean): number[] {
//   const scales = Array.from(scale.floatData);
//   scalesValidataion(scales, mode, isResize);
//   return scales;
// }

// function parseScalesDataFromOutputSize(
//     yDims: ReadonlyArray<number>, xDims: ReadonlyArray<number>, mode: string, isResize: boolean): number[] {
//   const length = xDims.length;
//   const scales = new Array<number>(length);

//   for (let i = 0, end = length; i < end; i++) {
//     if (xDims[i] === 0) {
//       if (yDims[i] !== 0) {
//         throw new Error('Input dim is zero but required output dim is non-zero.');
//       }
//       scales[i] = 1;
//     } else {
//       scales[i] = yDims[i] / xDims[i];
//     }
//   }
//   scalesValidataion(scales, mode, isResize);
//   return scales;
// }

// function computeOutputShape(scales: ReadonlyArray<number>, inputDims: ReadonlyArray<number>): number[] {
//   return inputDims.map((dim, i) => Math.floor(dim * scales[i]));
// }

// function fillResizeNearestMapping2D(
//     inputHeight: number, inputWidth: number, outputHeight: number, outputWidth: number, scalesHeight: number,
//     scalesWidth: number, roiStartHeight: number, roiEndHeight: number, roiStartWidth: number, roiEndWidth: number,
//     extrapolationEnabled: boolean, getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc,
//     getNearestPixel: Upsample.GetNearestPixelFunc, mappingOrigin: number[], mappingExtrapolation: number[]): void {
//   for (let i = 0; i < outputHeight; i++) {
//     let dim = i;
//     const originalCoord =
//         getOriginalCoordinate(dim, scalesHeight, outputHeight, inputHeight, roiStartHeight, roiEndHeight);
//     // extrapolate
//     mappingExtrapolation.push((extrapolationEnabled && (originalCoord < 0 || originalCoord > inputHeight - 1)) ? 1 :
//     0); dim = Math.max(0, Math.min(inputHeight - 1, getNearestPixel(originalCoord, scalesHeight < 1)));
//     // origin
//     mappingOrigin.push(dim);
//   }

//   for (let i = 0; i < outputWidth; i++) {
//     let dim = i;
//     const originalCoord = getOriginalCoordinate(dim, scalesWidth, outputWidth, inputWidth, roiStartWidth,
//     roiEndWidth);
//     // extrapolate
//     mappingExtrapolation.push((extrapolationEnabled && (originalCoord < 0 || originalCoord > inputWidth - 1)) ? 1 :
//     0); dim = Math.max(0, Math.min(inputWidth - 1, getNearestPixel(originalCoord, scalesWidth < 1)));
//     // origin
//     mappingOrigin.push(dim);
//   }
// }

// function fillResizeBilinearCoordinateMapping(
//     inputHeight: number, inputWidth: number, outputHeight: number, outputWidth: number, scalesHeight: number,
//     scalesWidth: number, roiStartHeight: number, roiEndHeight: number, roiStartWidth: number, roiEndWidth: number,
//     extrapolationEnabled: boolean, getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc, mappingOrigin:
//     number[], mappingWeight: number[], mappingExtrapolation: number[]) {
//   for (let i = 0; i < outputHeight; i++) {
//     let inputY = getOriginalCoordinate(i, scalesHeight, outputHeight, inputHeight, roiStartHeight, roiEndHeight);
//     mappingExtrapolation.push((extrapolationEnabled && (inputY < 0 || inputY > inputHeight - 1)) ? 1 : 0);
//     inputY = Math.max(0, Math.min(inputY, inputHeight - 1));
//     const intY = Math.floor(inputY);
//     mappingOrigin.push(intY);
//     mappingWeight.push(intY >= inputHeight - 1 ? 0.5 : inputY - intY);
//   }
//   for (let i = 0; i < outputWidth; i++) {
//     let inputX = getOriginalCoordinate(i, scalesWidth, outputWidth, inputWidth, roiStartWidth, roiEndWidth);
//     mappingExtrapolation.push((extrapolationEnabled && (inputX < 0 || inputX > inputWidth - 1)) ? 1 : 0);
//     inputX = Math.max(0, Math.min(inputX, inputWidth - 1));
//     const intX = Math.floor(inputX);
//     mappingOrigin.push(intX);
//     mappingWeight.push(intX >= inputWidth - 1 ? 0.5 : inputX - intX);
//   }
// }

// function fillResizeCubicCoordinateMapping(
//     inputHeight: number, inputWidth: number, outputHeight: number, outputWidth: number, scalesHeight: number,
//     scalesWidth: number, roiStartHeight: number, roiEndHeight: number, roiStartWidth: number, roiEndWidth: number,
//     extrapolationEnabled: boolean, cubicCoefficientA: number, excludeOutside: boolean,
//     getOriginalCoordinate: Upsample.p, mappingOrigin: number[], mappingExtrapolation:
//     number[], mappingCoeffCache: number[]) {
//   for (let i = 0; i < outputHeight + outputWidth; i++) {
//     const isY = i < outputHeight;
//     const maxInputCoord = isY ? inputHeight : inputWidth;
//     const inputCoord = getOriginalCoordinate(
//         isY ? i : i - outputHeight, isY ? scalesHeight : scalesWidth, isY ? outputHeight : outputWidth,
//         maxInputCoord, isY ? roiStartHeight : roiStartWidth, isY ? roiEndHeight : roiEndWidth);
//     const intCoord = Math.floor(inputCoord);
//     const sCoord = Math.abs(intCoord - inputCoord);
//     let coeffSum = 1.0;
//     let coeff0 = ((cubicCoefficientA * (sCoord + 1) - 5 * cubicCoefficientA) * (sCoord + 1) + 8 * cubicCoefficientA)
//     *
//             (sCoord + 1) -
//         4 * cubicCoefficientA;
//     let coeff1 = ((cubicCoefficientA + 2) * sCoord - (cubicCoefficientA + 3)) * sCoord * sCoord + 1;
//     let coeff2 = ((cubicCoefficientA + 2) * (1 - sCoord) - (cubicCoefficientA + 3)) * (1 - sCoord) * (1 - sCoord) +
//     1; let coeff3 = ((cubicCoefficientA * (2 - sCoord) - 5 * cubicCoefficientA) * (2 - sCoord) + 8 *
//     cubicCoefficientA) *
//             (2 - sCoord) -
//         4 * cubicCoefficientA;
//     if (excludeOutside) {
//       coeff0 = (intCoord - 1 < 0 || intCoord - 1 >= maxInputCoord) ? 0.0 : coeff0;
//       coeff1 = (intCoord + 0 < 0 || intCoord + 0 >= maxInputCoord) ? 0.0 : coeff1;
//       coeff2 = (intCoord + 1 < 0 || intCoord + 1 >= maxInputCoord) ? 0.0 : coeff2;
//       coeff3 = (intCoord + 2 < 0 || intCoord + 2 >= maxInputCoord) ? 0.0 : coeff3;
//       coeffSum = coeff0 + coeff1 + coeff2 + coeff3;
//     }
//     mappingOrigin.push(intCoord);
//     mappingExtrapolation.push((extrapolationEnabled && (inputCoord < 0 || inputCoord > maxInputCoord - 1)) ? 1 : 0);
//     mappingCoeffCache.push(coeff0 / coeffSum);
//     mappingCoeffCache.push(coeff1 / coeffSum);
//     mappingCoeffCache.push(coeff2 / coeffSum);
//     mappingCoeffCache.push(coeff3 / coeffSum);
//   }
// }

// function createUpsampleProgramInfo(
//     glsl: Glsl, mode: string, dim: number, inputLayout: TextureLayout, outputLayout: TextureLayout,
//     scales: ReadonlyArray<number>): ProgramInfo {
//   const outputShape = outputLayout.shape;
//   const inputShape = inputLayout.shape;
//   const precalculatedPitches = shaderPrecalculatedPitches(dim, outputShape, inputShape);
//   const getInputFloatFunction = shaderGetInputFloatFunction(inputLayout, glsl);

//   const shaderSource = mode === 'nearest' ?
//       // nearest
//       `
//       ${getInputFloatFunction}
//       float process(int indices[${dim}]) {
//         int input_index = 0;
//         int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});
//         ${precalculatedPitches}
//         int d, m;
//         for (int dim = 0; dim < ${dim}; ++dim) {
//           d = output_index / output_pitches[dim];
//           m = output_index - d * output_pitches[dim];
//           output_index = m;
//           if (scales[dim] != 1 && d > 0) {
//             int d2 = d / scales[dim];
//             m = d - d2 * scales[dim];
//             d = d2;
//           }
//           input_index += input_pitches[dim] * d;
//         }
//         return getInputFloat(input_index);
//       }` :
//       dim === 4 ?
//       // bilinear 4D
//           `
//       ${getInputFloatFunction}
//       float process(int indices[4]) {
//         int input_index = 0;
//         int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});
//         ${precalculatedPitches}
//         int m;
//         int index_of_dim0, index_of_dim1, index_of_dim2, index_of_dim3;
//         index_of_dim0 = output_index / output_pitches[0];
//         m = output_index - index_of_dim0 * output_pitches[0];
//         index_of_dim1 = m / output_pitches[1];
//         m = m - index_of_dim1 * output_pitches[1];
//         index_of_dim2 = m / output_pitches[2];
//         m = m - index_of_dim2 * output_pitches[2];
//         index_of_dim3 = m;
//         int index_of_input_dim2, index_of_input_dim3, x_offset, y_offset;
//         index_of_input_dim2 = index_of_dim2 / scales[2];
//         y_offset = index_of_dim2 - index_of_input_dim2 * scales[2];
//         index_of_input_dim3 = index_of_dim3 / scales[3];
//         x_offset = index_of_dim3 - index_of_input_dim3 * scales[3];
//         input_index = index_of_dim0 * input_pitches[0] +
//                       index_of_dim1 * input_pitches[1] +
//                       index_of_input_dim2 * input_pitches[2] +
//                       index_of_input_dim3;
//         float x00 = getInputFloat(input_index);
//         float x10, x01, x11;
//         bool end_of_dim2 = false;
//         if (index_of_input_dim2 == (${inputShape[2]} - 1)) {
//           // It's the end in dimension 2
//           x01 = x00;
//           end_of_dim2 = true;
//         } else {
//           x01 = getInputFloat(input_index + input_pitches[2]);
//         }
//         if (index_of_input_dim3 == (input_pitches[2] - 1)) {
//           // It's the end in dimension 3
//           x10 = x00;
//           x11 = x01;
//         }
//         else {
//           x10 = getInputFloat(input_index + 1);
//           x11 = end_of_dim2 ? x10 : getInputFloat(input_index + input_pitches[2] + 1);
//         }
//         float y0 = x00 + float(y_offset) * (x01 - x00) / float(scales[2]);
//         float y1 = x10 + float(y_offset) * (x11 - x10) / float(scales[2]);
//         return y0 + float(x_offset) * (y1 - y0) / float(scales[3]);
//       }` :
//           // bilinear 2D
//           `
//       ${getInputFloatFunction}
//       float process(int indices[2]) {
//         int input_index = 0;
//         int output_index = coordsToOffset(TexCoords, ${outputLayout.width}, ${outputLayout.height});
//         ${precalculatedPitches}
//         int m;
//         int index_of_dim0, index_of_dim1;
//         index_of_dim0 = output_index / output_pitches[0];
//         m = output_index - index_of_dim0 * output_pitches[0];
//         index_of_dim1 = m;
//         int index_of_input_dim0, index_of_input_dim1, x_offset, y_offset;
//         index_of_input_dim0 = index_of_dim0 / scales[0];
//         y_offset = index_of_dim0 - index_of_input_dim0 * scales[0];
//         index_of_input_dim1 = index_of_dim1 / scales[1];
//         x_offset = index_of_dim1 - index_of_input_dim1 * scales[1];
//         input_index = index_of_input_dim0 * input_pitches[0] + index_of_input_dim1;
//         float x00 = getInputFloat(input_index);
//         float x10, x01, x11;
//         bool end_of_dim0 = false;
//         if (index_of_input_dim0 == (${inputShape[0]} - 1)) {
//           // It's the end in dimension 0
//           x01 = x00;
//           end_of_dim0 = true;
//         } else {
//           x01 = getInputFloat(input_index + input_pitches[0]);
//         }
//         if (index_of_input_dim1 == (input_pitches[0] - 1)) {
//           // It's the end in dimension 1
//           x10 = x00;
//           x11 = x01;
//         }
//         else {
//           x10 = getInputFloat(input_index + 1);
//           x11 = end_of_dim0 ? x10 : getInputFloat(input_index + input_pitches[0] + 1);
//         }
//         float y0 = x00 + float(y_offset) * (x01 - x00) / float(scales[0]);
//         float y1 = x10 + float(y_offset) * (x11 - x10) / float(scales[0]);
//         return y0 + float(x_offset) * (y1 - y0) / float(scales[1]);
//       }`;
//   return {
//     inputLayouts: [inputLayout],
//     outputLayout,
//     samplers: ['X'],
//     shaderSource,
//     variables: [{name: 'scales', type: 'int', arrayLength: scales.length}]
//   };
// }

// function shaderPrecalculatedPitches(
//     dim: number, outputShape: ReadonlyArray<number>, inputShape: ReadonlyArray<number>) {
//   const outputPitches = new Array<number>(dim);
//   const inputPitches = new Array<number>(dim);
//   let precalculatedPitches = `
//       int output_pitches[${dim}];
//       int input_pitches[${dim}];
//       `;
//   for (let d = dim - 1; d >= 0; d--) {
//     outputPitches[d] = (d === dim - 1) ? 1 : outputPitches[d + 1] * outputShape[d + 1];
//     inputPitches[d] = (d === dim - 1) ? 1 : inputPitches[d + 1] * inputShape[d + 1];

//     precalculatedPitches += `
//       output_pitches[${d}] = ${outputPitches[d]};
//       input_pitches[${d}] = ${inputPitches[d]};
//       `;
//   }
//   return precalculatedPitches;
// }

// function shaderGetInputFloatFunction(inputLayout: TextureLayout, glsl: Glsl) {
//   return `
// float getInputFloat(int index) {
// vec2 coords = offsetToCoords(index, ${inputLayout.width}, ${inputLayout.height});
// float value = getColorAsFloat(${glsl.texture2D}(X, coords));
// return value;
// }
// `;
// }

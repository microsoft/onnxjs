// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Upsample} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuUpsample extends Upsample {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const [roi, scales, yDims] = this.prepare(inputs);
    const y = new Tensor(yDims, inputs[0].type);
    this.compute(inputs[0], y, roi, scales);
    return [y];
  }

  compute(x: Tensor, y: Tensor, roi: ReadonlyArray<number>, scales: ReadonlyArray<number>): void {
    const xDims = x.dims;
    const yDims = y.dims;
    if (yDims.length !== xDims.length) {
      throw new Error('Rank of input and output tensor should be same.');
    }

    if (y.size === 0) {
      return;
    }

    if (xDims.length !== scales.length) {
      throw new Error('input tensor\'s dimension does not match the scales.');
    }

    if (roi.length !== 2 * xDims.length) {
      throw new Error('size of roi array should be 2 * N where N is the rank of input tensor X.');
    }

    const noScale = xDims.every((d, i) => yDims[i] === d);
    if (noScale) {
      y.numberData.set(x.numberData);
      return;
    }

    if (this.mode === 'nearest') {
      upsampleNearest(
          x.data, y.data, xDims, yDims, scales, roi, this.isResize, this.useExtrapolation, this.extrapolationValue,
          this.useNearest2xOptimization, this.getOriginalCoordinate, this.getNearestPixel);
    } else {
      if (xDims.length !== 2 && xDims.length !== 4) {
        throw new Error('\'Linear\' mode only support 2-D inputs or 4-D inputs');
      }
      const is2D = xDims.length === 2;
      const batchSize = is2D ? 1 : xDims[0];
      const numChannels = is2D ? 1 : xDims[1];
      const inputHeight = is2D ? xDims[0] : xDims[2];
      const inputWidth = is2D ? xDims[1] : xDims[3];
      const outputHeight = is2D ? yDims[0] : yDims[2];
      const outputWidth = is2D ? yDims[1] : yDims[3];

      if (this.mode === 'linear') {
        upsampleBilinear(
            batchSize, numChannels, inputHeight, inputWidth, outputHeight, outputWidth, is2D ? scales[0] : scales[2],
            is2D ? scales[1] : scales[3], roi, this.useExtrapolation, this.extrapolationValue, x.numberData,
            y.numberData, this.getOriginalCoordinate);
      } else {
        upsampleBiCubic(
            batchSize, numChannels, inputHeight, inputWidth, outputHeight, outputWidth, is2D ? scales[0] : scales[2],
            is2D ? scales[1] : scales[3], this.cubicCoefficientA, this.useExtrapolation, this.extrapolationValue,
            this.excludeOutside, roi, x.numberData, y.numberData, this.getOriginalCoordinate);
      }
    }
  }
}

function upsampleNearest(
    xData: Tensor.DataTypeMap[Tensor.DataType], yData: Tensor.DataTypeMap[Tensor.DataType],
    xDims: ReadonlyArray<number>, yDims: ReadonlyArray<number>, scales: ReadonlyArray<number>,
    roi: ReadonlyArray<number>, isResize: boolean, extrapolationEnabled: boolean, extrapolationValue: number,
    useNearest2xOptimization: boolean, getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc,
    getNearestPixel: Upsample.GetNearestPixelFunc) {
  const dim = xDims.length;

  if (useNearest2xOptimization && dim === 4 && scales[0] === 1 && scales[1] === 1 && scales[2] === 2 &&
      scales[3] === 2) {
    // TODO: 2x optimization
  }

  const inputDimCounter = new Array<number>(dim).fill(0);
  const inputDimFactor = new Array<number>(dim);
  const useExtrapolationValue = new Array<boolean>(dim);
  inputDimFactor[dim - 1] = 1;  // initialize dimension factor
  for (let i = dim - 2; i >= 0; i--) {
    inputDimFactor[i] = inputDimFactor[i + 1] * xDims[i + 1];
  }

  let yIdx = 0;
  let xIdx = 0;

  const oneDimensionProcessor = (dimIdx: number, yDim: number) => {
    useExtrapolationValue[dimIdx] = false;
    const originalIdx =
        getOriginalCoordinate(yDim, scales[dimIdx], yDims[dimIdx], xDims[dimIdx], roi[dimIdx], roi[dim + dimIdx]);
    if (extrapolationEnabled && (originalIdx < 0 || originalIdx > xDims[dimIdx] - 1)) {
      useExtrapolationValue[dimIdx] = true;
    }
    let currentInputDimCounter = getNearestPixel(originalIdx, scales[dimIdx] < 1);
    currentInputDimCounter = Math.max(0, Math.min(currentInputDimCounter, (xDims[dimIdx] - 1)));
    if (currentInputDimCounter !== inputDimCounter[dimIdx]) {
      xIdx += (currentInputDimCounter - inputDimCounter[dimIdx]) * inputDimFactor[dimIdx];
      inputDimCounter[dimIdx] = currentInputDimCounter;
    }
  };

  if (dim === 1) {
    for (let yDim0 = 0; yDim0 < yDims[0]; yDim0++) {
      oneDimensionProcessor(0, yDim0);
      yData[yIdx++] = useExtrapolationValue[0] ? extrapolationValue : xData[xIdx];
    }

  } else if (dim === 2) {
    for (let yDim0 = 0; yDim0 < yDims[0]; yDim0++) {
      oneDimensionProcessor(0, yDim0);
      for (let yDim1 = 0; yDim1 < yDims[1]; yDim1++) {
        oneDimensionProcessor(1, yDim1);
        yData[yIdx++] = useExtrapolationValue.some(i => i) ? extrapolationValue : xData[xIdx];
      }
    }

  } else if (dim === 3) {
    for (let yDim0 = 0; yDim0 < yDims[0]; yDim0++) {
      oneDimensionProcessor(0, yDim0);
      for (let yDim1 = 0; yDim1 < yDims[1]; yDim1++) {
        oneDimensionProcessor(1, yDim1);
        for (let yDim2 = 0; yDim2 < yDims[2]; yDim2++) {
          oneDimensionProcessor(2, yDim2);
          yData[yIdx++] = useExtrapolationValue.some(i => i) ? extrapolationValue : xData[xIdx];
        }
      }
    }

  } else if (dim === 4) {
    for (let yDim0 = 0; yDim0 < yDims[0]; yDim0++) {
      oneDimensionProcessor(0, yDim0);
      for (let yDim1 = 0; yDim1 < yDims[1]; yDim1++) {
        oneDimensionProcessor(1, yDim1);
        for (let yDim2 = 0; yDim2 < yDims[2]; yDim2++) {
          oneDimensionProcessor(2, yDim2);
          for (let yDim3 = 0; yDim3 < yDims[3]; yDim3++) {
            oneDimensionProcessor(3, yDim3);
            yData[yIdx++] = useExtrapolationValue.some(i => i) ? extrapolationValue : xData[xIdx];
          }
        }
      }
    }
  } else {
    const outputDimCounter = new Array<number>(dim).fill(0);
    outputDimCounter[dim - 1] = -1;

    for (; yIdx < yData.length; yIdx++) {
      for (let dimIdx = dim - 1; dimIdx >= 0; dimIdx--) {
        if (++outputDimCounter[dimIdx] < yDims[dimIdx]) {
          let currentInputDimCounter = 0;
          const originalIdx = getOriginalCoordinate(
              outputDimCounter[dimIdx], scales[dimIdx], yDims[dimIdx], xDims[dimIdx], roi[dimIdx], roi[dim + dimIdx]);
          currentInputDimCounter = getNearestPixel(originalIdx, scales[dimIdx] < 1);
          currentInputDimCounter = Math.max(0, Math.min(currentInputDimCounter, (xDims[dimIdx] - 1)));

          if (currentInputDimCounter !== inputDimCounter[dimIdx]) {
            xIdx += (currentInputDimCounter - inputDimCounter[dimIdx]) * inputDimFactor[dimIdx];
            inputDimCounter[dimIdx] = currentInputDimCounter;
          }
          break;
        } else {
          outputDimCounter[dimIdx] = 0;
          xIdx += (0 - inputDimCounter[dimIdx]) * inputDimFactor[dimIdx];
          inputDimCounter[dimIdx] = 0;
        }
      }
      yData[yIdx] = xData[xIdx];
    }
  }
}

function upsampleBilinear(
    batchSize: number, numChannels: number, inputHeight: number, inputWidth: number, outputHeight: number,
    outputWidth: number, heightScale: number, widthScale: number, roi: ReadonlyArray<number>, useExtrapolation: boolean,
    extrapolationValue: number, xData: Tensor.NumberType, yData: Tensor.NumberType,
    getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc) {
  const yOriginal: number[] = [];
  const xOriginal: number[] = [];

  const inputWidthMulY1 = new Array<number>(outputHeight);
  const inputWidthMulY2 = new Array<number>(outputHeight);
  const inX1 = new Array<number>(outputWidth);
  const inX2 = new Array<number>(outputWidth);
  const dy1 = new Array<number>(outputHeight);
  const dy2 = new Array<number>(outputHeight);
  const dx1 = new Array<number>(outputWidth);
  const dx2 = new Array<number>(outputWidth);

  const roiYStart = roi.length / 2 - 2;
  const roiYEnd = roi.length - 2;
  for (let y = 0; y < outputHeight; ++y) {
    let inY = getOriginalCoordinate(y, heightScale, outputHeight, inputHeight, roi[roiYStart], roi[roiYEnd]);
    yOriginal.push(inY);
    inY = Math.max(0, Math.min(inY, inputHeight - 1));

    const inY1 = Math.min(Math.floor(inY), inputHeight - 1);
    const inY2 = Math.min(inY1 + 1, inputHeight - 1);

    if (inY1 === inY2) {
      dy1[y] = 0.5;
      dy2[y] = 0.5;
    } else {
      dy1[y] = Math.abs(inY - inY1);
      dy2[y] = Math.abs(inY - inY2);
    }

    inputWidthMulY1[y] = inputWidth * inY1;
    inputWidthMulY2[y] = inputWidth * inY2;
  }

  const roiXStart = roi.length / 2 - 1;
  const roiXEnd = roi.length - 1;
  for (let x = 0; x < outputWidth; ++x) {
    let inX = getOriginalCoordinate(x, widthScale, outputWidth, inputWidth, roi[roiXStart], roi[roiXEnd]);
    xOriginal.push(inX);
    inX = Math.max(0, Math.min(inX, inputWidth - 1));

    inX1[x] = Math.min(Math.floor(inX), inputWidth - 1);
    inX2[x] = Math.min(inX1[x] + 1, inputWidth - 1);

    if (inX1[x] === inX2[x]) {
      dx1[x] = 0.5;
      dx2[x] = 0.5;
    } else {
      dx1[x] = Math.abs(inX - inX1[x]);
      dx2[x] = Math.abs(inX - inX2[x]);
    }
  }

  let xOffset = 0;
  let yOffset = 0;
  for (let n = 0; n < batchSize; ++n) {
    for (let c = 0; c < numChannels; ++c) {
      for (let y = 0; y < outputHeight; ++y) {
        for (let x = 0; x < outputWidth; ++x) {
          if (useExtrapolation &&
              ((yOriginal[y] < 0 || yOriginal[y] > inputHeight - 1) ||
               (xOriginal[x] < 0 || xOriginal[x] > inputWidth - 1))) {
            yData[outputWidth * y + x] = extrapolationValue;
            continue;
          }

          const x11 = xData[xOffset + inputWidthMulY1[y] + inX1[x]];
          const x21 = xData[xOffset + inputWidthMulY1[y] + inX2[x]];
          const x12 = xData[xOffset + inputWidthMulY2[y] + inX1[x]];
          const x22 = xData[xOffset + inputWidthMulY2[y] + inX2[x]];

          yData[yOffset + outputWidth * y + x] =
              (dx2[x] * dy2[y] * x11 + dx1[x] * dy2[y] * x21 + dx2[x] * dy1[y] * x12 + dx1[x] * dy1[y] * x22);
        }
      }
      xOffset += inputHeight * inputWidth;
      yOffset += outputWidth * outputHeight;
    }
  }
}

const CUBIC_MODE_GRID_LENGTH = 4;

function getCubicCoeffs(s: number, cubicCoeffA = -0.75): number[] {
  s = Math.abs(s);
  return [
    (((cubicCoeffA * (s + 1) - 5 * cubicCoeffA) * (s + 1) + 8 * cubicCoeffA) * (s + 1) - 4 * cubicCoeffA),
    (((cubicCoeffA + 2) * s - (cubicCoeffA + 3)) * s * s + 1),
    (((cubicCoeffA + 2) * (1 - s) - (cubicCoeffA + 3)) * (1 - s) * (1 - s) + 1),
    (((cubicCoeffA * (2 - s) - 5 * cubicCoeffA) * (2 - s) + 8 * cubicCoeffA) * (2 - s) - 4 * cubicCoeffA)
  ];
}

function getDataForCoordinate(
    xData: Tensor.NumberType, x: number, y: number, inputHeight: number, inputWidth: number): number {
  x = Math.max(0, Math.min(x, inputWidth - 1));
  y = Math.max(0, Math.min(y, inputHeight - 1));
  return xData[y * inputWidth + x];
}

function cubicInterpolation1D(
    xData: Tensor.NumberType, x: number, y: number, inputHeight: number, inputWidth: number, coeffArray: number[],
    coeffSum: number, cache: Map<number, number>): number {
  // When calculating cubic interpolation we move the 4*4 grid across the original data and therefore there is
  // opportunity to cache the results for previously seen combinations.
  // Check if the result is already available in the cache
  const gridStartPosition = (y) * inputWidth + (x - 1);
  let result = cache.get(gridStartPosition);
  if (result !== undefined) {
    return result;
  }

  // get the neighbors in 1D and find interpolation for this dimension
  // for 1D cubic interpolation 4 samples are used. 2 on the left and 2 on the right of x
  result = 0;
  for (let i = 0, j = -1; i < CUBIC_MODE_GRID_LENGTH; i++, j++) {
    const originalData = getDataForCoordinate(xData, x + j, y, inputHeight, inputWidth);
    result += coeffArray[i] / coeffSum * originalData;
  }
  cache.set(gridStartPosition, result);

  return result;
}

function upsampleBiCubic(
    batchSize: number, numChannels: number, inputHeight: number, inputWidth: number, outputHeight: number,
    outputWidth: number, heightScale: number, widthScale: number, cubicCoefficientA: number, useExtrapolation: boolean,
    extrapolationValue: number, excludeOutside: boolean, roi: ReadonlyArray<number>, xData: Tensor.NumberType,
    yData: Tensor.NumberType, getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc) {
  const yOriginal: number[] = [];
  const xOriginal: number[] = [];
  const cubicCoeffs = new Map<number, number[]>();
  const coeffTo1DinterpolationMap = new Map<number, Map<number, number>>();
  const roiYStart = roi.length / 2 - 2;
  const roiYEnd = roi.length - 2;
  const roiXStart = roi.length / 2 - 1;
  const roiXEnd = roi.length - 1;

  // generate coefficients in y direction
  for (let y = 0; y < outputHeight; ++y) {
    const inY = getOriginalCoordinate(y, heightScale, outputHeight, inputHeight, roi[roiYStart], roi[roiYEnd]);
    yOriginal.push(inY);
    const s = yOriginal[y] - Math.floor(yOriginal[y]);
    if (!cubicCoeffs.has(s)) {
      cubicCoeffs.set(s, getCubicCoeffs(s, cubicCoefficientA));
      coeffTo1DinterpolationMap.set(s, new Map());
    }
  }

  // generate coefficients in x direction
  for (let x = 0; x < outputWidth; ++x) {
    const inX = getOriginalCoordinate(x, widthScale, outputWidth, inputWidth, roi[roiXStart], roi[roiXEnd]);
    xOriginal.push(inX);
    const s = xOriginal[x] - Math.floor(xOriginal[x]);
    if (!cubicCoeffs.has(s)) {
      cubicCoeffs.set(s, getCubicCoeffs(s, cubicCoefficientA));
      coeffTo1DinterpolationMap.set(s, new Map());
    }
  }

  // setup up temp arrays to hold coefficients when exclude_outside is set to true
  const yCoeffHolder = new Array<number>(CUBIC_MODE_GRID_LENGTH);
  const xCoeffHolder = new Array<number>(CUBIC_MODE_GRID_LENGTH);
  let yCoeffSum = 1;
  let xCoeffSum = 1;

  for (let n = 0; n < batchSize; n++) {
    for (let c = 0; c < numChannels; c++) {
      for (let y = 0; y < outputHeight; y++) {
        const inY = yOriginal[y];

        // when use_extrapolation is set and original index is out of the dim range
        // then use extrapolation_value as the output value.
        if (useExtrapolation && (inY < 0 || inY > inputHeight - 1)) {
          for (let x = 0; x < outputWidth; x++) {
            yData[y * outputWidth + x] = extrapolationValue;
          }
          continue;
        }

        const yInt = Math.floor(inY);
        const sY = inY - yInt;
        const coeffY = excludeOutside ? yCoeffHolder : cubicCoeffs.get(sY)!;
        yCoeffSum = 1;

        if (excludeOutside) {
          // When true, the weight of sampling locations outside the grid will be set to 0
          // and the weight will be renormalized so that their sum is 1.0
          yCoeffSum = 0;
          const origYCoeffs = cubicCoeffs.get(sY)!;
          for (let i = 0, yVal = yInt - 1; yVal <= yInt + 2; yVal++, i++) {
            yCoeffHolder[i] = (yVal < 0 || yVal >= inputHeight) ? 0.0 : origYCoeffs[i];
            yCoeffSum += yCoeffHolder[i];
          }
        }

        for (let x = 0; x < outputWidth; x++) {
          const inX = xOriginal[x];

          // when use_extrapolation is set and original index is out of the dim range
          // then use extrapolation_value as the output value.
          if (useExtrapolation && (inX < 0 || inX > inputWidth - 1)) {
            yData[y * outputWidth + x] = extrapolationValue;
            continue;
          }

          const xInt = Math.floor(inX);
          const sX = inX - xInt;
          const coeffX = excludeOutside ? xCoeffHolder : cubicCoeffs.get(sX)!;
          xCoeffSum = 1;

          if (excludeOutside) {
            // When true, the weight of sampling locations outside the grid will be set to 0
            // and the weight will be renormalized so that their sum is 1.0
            xCoeffSum = 0;
            const origXCoeffs = cubicCoeffs.get(sX)!;
            for (let i = 0, xVal = xInt - 1; xVal <= xInt + 2; xVal++, i++) {
              xCoeffHolder[i] = (xVal < 0 || xVal >= inputWidth) ? 0.0 : origXCoeffs[i];
              xCoeffSum += xCoeffHolder[i];
            }
          }

          // Compute cubic interpolation in x dimension using the x coefficients.
          // From the result of cubic interpolation in x dim, compute cubic interpolation in y dimension
          const interpolationResultCache = coeffTo1DinterpolationMap.get(sX)!;
          let result = 0;
          for (let yVal = yInt - 1, i = 0; yVal <= yInt + 2; yVal++, i++) {
            const xResult = cubicInterpolation1D(
                xData, xInt, yVal, inputHeight, inputWidth, coeffX, xCoeffSum, interpolationResultCache);
            result += xResult * coeffY[i] / yCoeffSum;
          }

          yData[y * outputWidth + x] = result;
        }
      }

      xData = xData.subarray(inputHeight * inputWidth);
      yData = yData.subarray(outputHeight * outputWidth);

      // clear the cache when moving to the next channel
      coeffTo1DinterpolationMap.clear();
    }
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Upsample, UpsampleV9} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuUpsample extends Upsample {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const xDims = inputs[0].dims;
    const yDims = xDims.map((dim, i) => Math.floor(dim * this.scales[i]));
    const y = new Tensor(yDims, inputs[0].type);
    if (this.mode === 'nearest') {
      upsampleNearest(inputs[0].data, y.data, xDims, yDims, this.scales);
    } else {
      upsampleLinear(inputs[0].data, y.data, xDims, yDims, this.scales);
    }
    return [y];
  }
}

export class CpuUpsampleV9 extends UpsampleV9 {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const scales = inputs[1].floatData;

    if (this.mode === 'linear' && scales.length !== 2 && scales.length !== 4) {
      throw new Error(`only support 2-D or 4-D upsampling for linear mode`);
    }

    const xDims = inputs[0].dims;
    const yDims = xDims.map((dim, i) => Math.floor(dim * scales[i]));
    const y = new Tensor(yDims, inputs[0].type);
    if (this.mode === 'nearest') {
      upsampleNearest(inputs[0].data, y.data, xDims, yDims, [...scales]);
    } else {
      upsampleLinear(inputs[0].data, y.data, xDims, yDims, [...scales]);
    }
    return [y];
  }
}

function upsampleNearest(
    xData: Tensor.DataTypeMap[Tensor.DataType], yData: Tensor.DataTypeMap[Tensor.DataType],
    xDims: ReadonlyArray<number>, yDims: ReadonlyArray<number>, scales: number[]) {
  const dim = xDims.length;

  const inputDimCounter = new Array<number>(dim);
  inputDimCounter.fill(0);
  const inputDimFactor = new Array<number>(dim);
  inputDimFactor[dim - 1] = 1;  // initialize dimension factor
  for (let i = dim - 2; i >= 0; i--) {
    inputDimFactor[i] = inputDimFactor[i + 1] * xDims[i + 1];
  }
  const outputDimCounter = new Array<number>(dim);
  outputDimCounter.fill(0);
  outputDimCounter[dim - 1] = -1;

  let yIdx = 0;
  let xIdx = 0;
  for (; yIdx < yData.length; yIdx++) {
    for (let dimIdx = dim - 1; dimIdx >= 0; dimIdx--) {
      if (++outputDimCounter[dimIdx] < yDims[dimIdx]) {
        let currentInputDimCounter = 0;
        const originalIdx = getOriginalCoordinate(outputDimCounter[dimIdx], scales[dimIdx]);
        currentInputDimCounter = Math.floor(originalIdx);
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

function upsampleLinear(
    xData: Tensor.DataTypeMap[Tensor.DataType], yData: Tensor.DataTypeMap[Tensor.DataType],
    xDims: ReadonlyArray<number>, yDims: ReadonlyArray<number>, scales: number[]) {
  const is2D = xDims.length === 2;
  const batchSize = is2D ? 1 : xDims[0];
  const numChannels = is2D ? 1 : xDims[1];
  const inputHeight = is2D ? xDims[0] : xDims[2];
  const inputWidth = is2D ? xDims[1] : xDims[3];
  const outputHeight = is2D ? yDims[0] : yDims[2];
  const outputWidth = is2D ? yDims[1] : yDims[3];

  upsampleBilinear(
      xData as Tensor.NumberType, yData as Tensor.NumberType, batchSize, numChannels, inputHeight, inputWidth,
      outputHeight, outputWidth, is2D ? scales[0] : scales[2], is2D ? scales[1] : scales[3]);
}

function upsampleBilinear(
    xData: Tensor.NumberType, yData: Tensor.NumberType, batchSize: number, numChannels: number, inputHeight: number,
    inputWidth: number, outputHeight: number, outputWidth: number, heightScale: number, widthScale: number) {
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

  for (let y = 0; y < outputHeight; ++y) {
    let inY = getOriginalCoordinate(y, heightScale);
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

  for (let x = 0; x < outputWidth; ++x) {
    let inX = getOriginalCoordinate(x, widthScale);
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

function getOriginalCoordinate(xResized: number, xScale: number): number {
  // Coordinate transformation mode attr was introduced in version 11, before that asymmetric mode was the only
  // available transformation mode
  // return ((xResized + 0.5) / xScale) - 0.5;
  return xResized / xScale;
}

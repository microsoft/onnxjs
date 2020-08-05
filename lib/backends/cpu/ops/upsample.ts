// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Upsample} from '../../../ops/upsample';
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
      upsampleLinear(inputs[0].data, y.data, xDims, yDims, this.scales, this.roi);
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
    xDims: ReadonlyArray<number>, yDims: ReadonlyArray<number>, scales: number[], roi: number[]) {
  const is2D = xDims.length === 2;
  const batchSize = is2D ? 1 : xDims[0];
  const numChannels = is2D ? 1 : xDims[1];
  const inputHeight = is2D ? xDims[0] : xDims[2];
  const inputWidth = is2D ? xDims[1] : xDims[3];
  const outputHeight = is2D ? yDims[0] : yDims[2];
  const outputWidth = is2D ? yDims[1] : yDims[3];

  upsampleBilinear(
      xData as Tensor.NumberType, yData as Tensor.NumberType, batchSize, numChannels, inputHeight, inputWidth,
      outputHeight, outputWidth, is2D ? scales[0] : scales[2], is2D ? scales[1] : scales[3], roi);
}

function upsampleBilinear(
    xData: Tensor.NumberType, yData: Tensor.NumberType, batchSize: number, numChannels: number, inputHeight: number,
    inputWidth: number, outputHeight: number, outputWidth: number, heightScale: number, widthScale: number,
    roi: number[]) {
  const y_original: number[] = [];
  const x_original: number[] = [];

  const input_width_mul_y1 = new Array<number>(outputHeight);
  const input_width_mul_y2 = new Array<number>(outputHeight);
  const in_x1 = new Array<number>(outputWidth);
  const in_x2 = new Array<number>(outputWidth);
  const dy1 = new Array<number>(outputHeight);
  const dy2 = new Array<number>(outputHeight);
  const dx1 = new Array<number>(outputWidth);
  const dx2 = new Array<number>(outputWidth);

  for (let y = 0; y < outputHeight; ++y) {
    let in_y = getOriginalCoordinate(y, heightScale);
    y_original.push(in_y);
    in_y = Math.max(0, Math.min(in_y, inputHeight - 1));

    const in_y1 = Math.min(Math.floor(in_y), inputHeight - 1);
    const in_y2 = Math.min(in_y1 + 1, inputHeight - 1);
    dy1[y] = Math.abs(in_y - in_y1);
    dy2[y] = Math.abs(in_y - in_y2);

    if (in_y1 == in_y2) {
      dy1[y] = 0.5;
      dy2[y] = 0.5;
    }

    input_width_mul_y1[y] = inputWidth * in_y1;
    input_width_mul_y2[y] = inputWidth * in_y2;
  }

  for (let x = 0; x < outputWidth; ++x) {
    let in_x = getOriginalCoordinate(x, widthScale);
    x_original.push(in_x);
    in_x = Math.max(0, Math.min(in_x, inputWidth - 1));

    in_x1[x] = Math.min(Math.floor(in_x), inputWidth - 1);
    in_x2[x] = Math.min(in_x1[x] + 1, inputWidth - 1);

    dx1[x] = Math.abs(in_x - in_x1[x]);
    dx2[x] = Math.abs(in_x - in_x2[x]);
    if (in_x1[x] == in_x2[x]) {
      dx1[x] = 0.5;
      dx2[x] = 0.5;
    }
  }

  let xOffset = 0;
  let yOffset = 0;
  for (let n = 0; n < batchSize; ++n) {
    for (let c = 0; c < numChannels; ++c) {
      for (let y = 0; y < outputHeight; ++y) {
        for (let x = 0; x < outputWidth; ++x) {
          const X11 = xData[xOffset + input_width_mul_y1[y] + in_x1[x]];
          const X21 = xData[xOffset + input_width_mul_y1[y] + in_x2[x]];
          const X12 = xData[xOffset + input_width_mul_y2[y] + in_x1[x]];
          const X22 = xData[xOffset + input_width_mul_y2[y] + in_x2[x]];

          yData[yOffset + outputWidth * y + x] =
              (dx2[x] * dy2[y] * X11 + dx1[x] * dy2[y] * X21 + dx2[x] * dy1[y] * X12 + dx1[x] * dy1[y] * X22);
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

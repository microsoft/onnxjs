// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Lrn} from '../../../ops/lrn';
import {Tensor} from '../../../tensor';
import * as util from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuLrn extends Lrn {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = lrn(inputs[0], this.alpha, this.beta, this.bias, this.size);
    return [output];
  }
}

export function lrn(x: Tensor, alpha: number, beta: number, bias: number, size: number): Tensor {
  const N = x.dims[0];
  const C = x.dims[1];
  const X = x.floatData;
  let channelSize = 1;

  for (let i = 2; i < x.dims.length; ++i) {
    channelSize *= x.dims[i];
  }

  const tensorDataSize = channelSize * C;

  // create new tensor to hold the result
  const output = new Tensor(x.dims, x.type);
  const Y: number[] = new Array(util.ShapeUtil.size(x.dims));

  // update the output with just the bias to begin with
  for (let i = 0; i < Y.length; ++i) {
    Y[i] = bias;
  }

  // placeholder to store padded square (i.e.) intermediate data
  const paddedSquareSize = (C + size - 1) * channelSize;
  const paddedSquareData = new Float64Array(paddedSquareSize);

  const alphaOverSize = alpha / size;
  const prePad = (size - 1) / 2;

  // go through the images
  for (let n = 0; n < N; ++n) {
    // compute the padded square
    util.MathUtil.sqr(paddedSquareData, X, prePad * channelSize, tensorDataSize * n, tensorDataSize);

    // create the first channel
    for (let c = 0; c < size; ++c) {
      util.MathUtil.axpy(Y, paddedSquareData, tensorDataSize * n, c * channelSize, channelSize, alphaOverSize);
    }

    for (let c = 1; c < C; ++c) {
      const scaleSliceStart = n * tensorDataSize + c * channelSize;

      // copy previous scale
      util.arrayCopyHelper(Y, Y, scaleSliceStart, scaleSliceStart - channelSize, channelSize);

      // add head
      util.MathUtil.axpy(
          Y, paddedSquareData, scaleSliceStart, (c + size - 1) * channelSize, channelSize, alphaOverSize);

      // subtract tail
      util.MathUtil.axpy(Y, paddedSquareData, scaleSliceStart, (c - 1) * channelSize, channelSize, -alphaOverSize);
    }
  }

  util.MathUtil.powx(Y, Y, 0, 0, util.ShapeUtil.size(x.dims), -beta);

  util.MathUtil.mul(Y, X, 0, 0, util.ShapeUtil.size(x.dims));

  output.floatData.set(Y);

  return output;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import ndarray from 'ndarray';
import matrixProduct from 'ndarray-gemm';
import nd_ops from 'ndarray-ops';

import {Conv} from '../../../ops/conv';
import {Tensor} from '../../../tensor';
import {PoolConvUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuConv extends Conv {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    // if kernelShape is not specified in the attributes of this op, infer it from the weight tensor dims
    if (this.kernelShape.length === 0) {
      const wDims = inputs[1].dims;
      for (let i = 2; i < wDims.length; ++i) {
        this.kernelShape.push(wDims[i]);
      }
    }

    const output = conv(
        inputs[0], inputs[1], inputs.length === 3 ? inputs[2] : null, this.autoPad, this.dilations, this.group,
        this.kernelShape, this.pads, this.strides);
    return [output];
  }
}

export function conv(
    x: Tensor, w: Tensor, b: Tensor|null, autoPad: string, dilations: number[], group: number, kernelShape: number[],
    pads: number[], strides: number[]): Tensor {
  let ndx = ndarray(x.floatData as Float32Array, x.dims.slice(0)).transpose(0, 2, 3, 1);
  const ndk = ndarray(w.floatData as Float32Array, w.dims.slice(0)).transpose(2, 3, 1, 0);

  // adjusting pads based on 'autoPad' attribute
  PoolConvUtil.adjustPadsBasedOnAutoPad(x.dims, strides, dilations, kernelShape, pads, autoPad);

  // padding if needed
  const localPads: Array<[number, number]> = [[0, 0], [pads[0], pads[2]], [pads[1], pads[3]], [0, 0]];
  const padTotal = localPads.reduce((s, p) => s + p[0] + p[1], 0);
  if (padTotal !== 0) {
    const shape: number[] = ndx.shape;
    const newShape = shape.map((len, index) => len + localPads[index][0] + localPads[index][1]);
    const newSize = newShape.reduce((m, v) => m * v, 1);
    const ndp = ndarray(new Float32Array(newSize), newShape);
    const hiPoint = localPads.map((pair, index) => newShape[index] - pair[1]);
    const loPoint = localPads.map(pair => pair[0]);
    const originalSlice = ndp.hi(...hiPoint).lo(...loPoint);
    nd_ops.assign(originalSlice, ndx);
    ndx = ndp;
  }

  const [batchSize, xRows, xCols, xChannels] = ndx.shape;
  const [wRows, wCols, yChannels] = [ndk.shape[0], ndk.shape[1], ndk.shape[3]];

  // calculate the patch view in source image's size after dilations
  const pvRows = wRows + (wRows - 1) * (dilations[0] - 1);
  const pvCols = wCols + (wCols - 1) * (dilations[1] - 1);

  const yRows = Math.floor((xRows - pvRows + strides[0]) / strides[0]);
  const yCols = Math.floor((xCols - pvCols + strides[1]) / strides[1]);

  const ySize = batchSize * yRows * yCols * yChannels;
  const patchSize = wRows * wCols * xChannels;

  const ndf = ndarray(new Float64Array(ndk.size), [patchSize, yChannels]);
  const patch = ndarray(new Float64Array(patchSize), [wRows, wCols, xChannels]);
  for (let yChannel = 0; yChannel < yChannels; ++yChannel) {
    nd_ops.assign(patch, ndk.pick(null, null, null, yChannel));
    const reshapedPatch = ndarray(patch.data, [patchSize]);
    nd_ops.assign(ndf.pick(null, yChannel), reshapedPatch);
  }

  const yArray = new Float64Array(ySize);
  const pixelVec = ndarray(new Float64Array(yChannels), [1, yChannels]);
  let offset = 0;
  for (let b = 0; b < batchSize; ++b) {
    const image = ndx.pick(b, null, null, null);
    for (let yRow = 0; yRow < yRows; ++yRow) {
      const xRowStart = yRow * strides[0];
      for (let yCol = 0; yCol < yCols; ++yCol) {
        const xColStart = yCol * strides[1];

        const patchView = image.hi(xRowStart + pvRows, xColStart + pvCols, xChannels)
                              .lo(xRowStart, xColStart, 0)
                              .step(dilations[0], dilations[1], 1);
        nd_ops.assign(patch, patchView);
        const pvVec = ndarray(patch.data, [1, patchSize]);
        matrixProduct(pixelVec, pvVec, ndf);
        yArray.set(pixelVec.data, offset);
        offset += yChannels;
      }
    }
  }
  const ndy = ndarray(yArray, [batchSize, yRows, yCols, yChannels]);
  const ndyTransed = ndarray(new Float32Array(ySize), [batchSize, yChannels, yRows, yCols]);
  nd_ops.assign(ndyTransed, ndy.transpose(0, 3, 1, 2));
  const Y = new Tensor(ndyTransed.shape, 'float32');
  Y.floatData.set(ndyTransed.data);

  // Add bias if applicable
  if (b) {
    const biasData = b.numberData;
    const outputData = Y.floatData;
    const batchSize = Y.dims[0];
    const outputChannels = Y.dims[1];
    const channelSize = Y.dims[2] * Y.dims[3];
    const dataSize = outputChannels * channelSize;
    for (let batch = 0; batch < batchSize; ++batch) {
      for (let channel = 0; channel < outputChannels; ++channel) {
        const offset = batch * dataSize + channel * channelSize;
        for (let index = 0; index < channelSize; ++index) {
          outputData[offset + index] += biasData[channel];
        }
      }
    }
  }

  return Y;
}

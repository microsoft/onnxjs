// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {AveragePool, GlobalAveragePool, GlobalMaxPool, MaxPool} from '../../../ops/pool';
import {Tensor} from '../../../tensor';
import {PoolConvUtil, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuAveragePool extends AveragePool {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output =
        averagePool(inputs[0], this.autoPad, this.countIncludePad, this.kernelShape, this.pads, this.strides);
    return [output];
  }
}

export class CpuGlobalAveragePool extends GlobalAveragePool {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = globalAveragePool(inputs[0]);
    return [output];
  }
}

export class CpuMaxPool extends MaxPool {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = maxPool(inputs[0], this.autoPad, this.kernelShape, this.pads, this.strides);
    return [output];
  }
}

export class CpuGlobalMaxPool extends GlobalMaxPool {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]> {
    const output = globalMaxPool(inputs[0]);
    return [output];
  }
}

// Functions implementing specific pooling operations
export function averagePool(
    input: Tensor, autoPad: string, countIncludePad: boolean, kernelShape: number[], pads: number[],
    strides: number[]): Tensor {
  return pool(
      false, input, autoPad, countIncludePad, kernelShape, pads, strides, 0, (a, b) => (a + b), (a, b) => (a / b));
}

export function globalAveragePool(input: Tensor): Tensor {
  return pool(true, input, 'NOTSET', false, [], [], [], 0, (a, b) => (a + b), (a, b) => (a / b));
}

export function maxPool(
    input: Tensor, autoPad: string, kernelShape: number[], pads: number[], strides: number[]): Tensor {
  return pool(
      false, input, autoPad, false, kernelShape, pads, strides, Number.MIN_SAFE_INTEGER, (a, b) => (Math.max(a, b)),
      (a, b) => a);
}

export function globalMaxPool(input: Tensor): Tensor {
  return pool(
      true, input, 'NOTSET', false, [], [], [], Number.MIN_SAFE_INTEGER, (a, b) => (Math.max(a, b)), (a, b) => a);
}

/**
 * Perform pooling operations based on input
 * @param isGlobalOperator If true, perform global pooling.
 * @param input The input tensor.
 * @param autoPad DEPRECATED attribute supported for legacy models. Specifies how to implicitly calculate pads in each
 *     dimension. Can take values NOTSET, SAME_UPPER, SAME_LOWER, or VALID
 * @param countIncludePad Whether include pad pixels when calculating values for the edges.
 * @param kernelShape The size of the kernel along each axis.
 * @param pads Padding for the beginning and ending along each axis. `pads` format should be as follow [x1_begin,
 *       x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels added at the beginning of axis `i` and
 *       xi_end, the number of pixels added at the end of axis `i`.
 * @param strides Stride along each axis.
 * @param startVal The initial value to do pooling operations
 * @param processOp The operation to be performed on each element inside kernel
 * @param finalOp The operation to be performed over all elements inside kernel
 */
export function pool(
    isGlobalOperator: boolean, input: Tensor, autoPad: string, countIncludePad: boolean, kernelShape: number[],
    pads: number[], strides: number[], startVal: number, processOp: (a: number, b: number) => number,
    finalOp: (a: number, b: number) => number): Tensor {
  // adjust the shapes of input attributes
  PoolConvUtil.adjustPoolAttributes(isGlobalOperator, input.dims, kernelShape, strides, pads);

  // calculate output shape based on input attributes.
  const outputShape =
      PoolConvUtil.computePoolOutputShape(isGlobalOperator, input.dims, strides, kernelShape, pads, autoPad);

  const kernelSize = ShapeUtil.size(kernelShape);
  const kernelStrides = ShapeUtil.computeStrides(kernelShape);
  const stridesRank = kernelStrides.length;
  const rank = outputShape.length;

  const outputSize = ShapeUtil.size(outputShape);
  const output = new Tensor(outputShape, input.type);
  const outputStride = ShapeUtil.computeStrides(outputShape);

  for (let ind = 0; ind < outputSize; ind++) {
    const curInd = ShapeUtil.offsetToIndices(ind, outputStride);
    const startInd = curInd.slice(0);
    const x = curInd.slice(0);
    // calculate the start indices of kernel corresponding to current output indices
    for (let i = 0; i < stridesRank; i++) {
      startInd[rank - stridesRank + i] = curInd[rank - stridesRank + i] * strides[i];
    }
    let value = startVal;
    let pad = 0;
    let isPad = false;
    // loop through elements within kernel
    for (let i = 0; i < kernelSize; i++) {
      const offset = ShapeUtil.offsetToIndices(i, kernelStrides);
      isPad = false;
      // "Shift" the kernel by the kernel start indices to loop through the kernel mapped to current output indices
      for (let j = rank - stridesRank; j < rank; j++) {
        x[j] = startInd[j] + offset[j - rank + stridesRank] - pads[j - 2];
        // check if current indices fall in the padding area
        if (x[j] >= input.dims[j] || x[j] < 0) {
          pad++;
          isPad = true;
          break;
        }
      }
      value = isPad ? value : processOp(value, input.get(x) as number);
    }
    value = countIncludePad ? finalOp(value, kernelSize) : finalOp(value, kernelSize - pad);
    output.set(curInd, value);
  }

  return output;
}

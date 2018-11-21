// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import ndarray from 'ndarray';
import matrixProduct from 'ndarray-gemm';

import {Gemm} from '../../../ops/gemm';
import {Tensor} from '../../../tensor';
import * as util from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

import {binaryOp} from './binary-op';
import {transpose} from './transpose';

export class CpuGemm extends Gemm {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = gemm(inputs[0], inputs[1], inputs[2], this.alpha, this.beta, this.transA, this.transB);
    return [output];
  }
}

export function gemm(a: Tensor, b: Tensor, c: Tensor, alpha: number, beta: number, transA: boolean, transB: boolean) {
  let M: number;
  let N: number;

  [M, N] = util.GemmUtil.getShapeOfGemmResult(a.dims, transA, b.dims, transB, c.dims);

  // Transpose if needed
  let finalA: Tensor;
  let finalB: Tensor;

  if (transA) {
    finalA = transpose(a, [-1]);
  } else {
    finalA = a;
  }

  if (transB) {
    finalB = transpose(b, [-1]);
  } else {
    finalB = b;
  }

  // gemm using ndarray
  let finalNdA: ndarray;
  let finalNdB: ndarray;
  let finalNdC: ndarray;

  finalNdA = ndarray(finalA.floatData, finalA.dims as number[]);
  finalNdB = ndarray(finalB.floatData, finalB.dims as number[]);

  // The result will always be of the shape [M,N]
  finalNdC = ndarray(new Float64Array(M * N), [M, N]);

  matrixProduct(finalNdC, finalNdA, finalNdB, alpha, beta);

  // re-convert result to 'Tensor' type
  const output = new Tensor(finalNdC.shape, a.type);

  // scale the bias data (i.e.) 'c' tensor data by 'beta'
  for (let i = 0; i < c.floatData.length; ++i) {
    c.floatData[i] = beta * c.floatData[i];
  }

  // Add the ndarray's gemm result with the scaled bias data
  // Leverage exisitng binary op add with broadcast
  // Internally add will throw an exception when no broadcast is possible
  if (a.type === 'float32') {
    output.floatData.set(
        binaryOp(Tensor.fromNdarray(finalNdC, 'float32', true), c, (e1: number, e2: number) => (e1 + e2)).floatData);
  }

  // currently supports only 'float32' or 'float64'
  else {
    output.floatData.set(
        binaryOp(Tensor.fromNdarray(finalNdC, 'float64', false), c, (e1: number, e2: number) => (e1 + e2)).floatData);
  }

  return output;
}

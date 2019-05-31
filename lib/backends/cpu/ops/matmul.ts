// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, MatMulUtil, ShapeUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuMatMul extends MatMul {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = matMul(inputs[0], inputs[1]);
    return [output];
  }
}

export function matMul(a: Tensor, b: Tensor) {
  const [dimsA, dimsB] = MatMulUtil.preprocessInputShapes(a.dims, b.dims);
  const mat2dShape = [dimsA[dimsA.length - 2], dimsB[dimsB.length - 1]];
  const shape = BroadcastUtil.calcShape(dimsA, dimsB, true);
  if (!shape) {
    // the inputs cannot broadcast or cannot multiply
    throw new Error(`input dimensions do not match the requirement`);
  }
  const size = ShapeUtil.size(shape);
  const num2dMatrices = size / (mat2dShape[0] * mat2dShape[1]);

  const y = new Tensor(shape, a.type === 'float64' || b.type === 'float64' ? 'float64' : 'float32');
  let offsetY = 0;
  const indices = new Array<number>(shape.length);
  const indicesA = new Array<number>(a.dims.length);
  const indicesB = new Array<number>(b.dims.length);
  for (let i = 0; i < num2dMatrices; i++) {
    // traverse nd array at 2d level
    indices[shape.length - 2] = 0;
    indices[shape.length - 1] = 0;
    let rest = i;
    for (let j = shape.length - 3; j >= 0; j--) {
      indices[j] = rest % shape[j];
      rest = Math.floor(rest / shape[j]);
    }
    // map the "broadcasted" index to original index
    BroadcastUtil.fillIndex(indices, a.dims, indicesA);
    BroadcastUtil.fillIndex(indices, b.dims, indicesB);
    // calculate subarrays offset for A and B
    const offsetA = indicesA.length <= 2 ? 0 : ShapeUtil.indicesToOffset(indicesA, a.strides, shape.length - 2);
    const offsetB = indicesB.length <= 2 ? 0 : ShapeUtil.indicesToOffset(indicesB, b.strides, shape.length - 2);
    // multiply like conventional matrices
    matMul2d(
        a.floatData.subarray(offsetA), b.floatData.subarray(offsetB), y.floatData.subarray(offsetY), false, false, 1, 0,
        mat2dShape[0], mat2dShape[1], dimsA[dimsA.length - 1]);
    offsetY += mat2dShape[0] * mat2dShape[1];
  }
  return y;
}

/**
 * perform matrix multiply on C = alpha * A * B + beta * C
 * @param A data of tensor A, whose shape is [M,K] or [K,M] (if transA)
 * @param B data of tensor B, whose shape is [K,N] or [N,K] (if transB)
 * @param C data of tensor C, whose shape is [M,N]
 */
export function matMul2d(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, transA: boolean,
    transB: boolean, alpha: number, beta: number, M: number, N: number, K: number) {
  if (transA && transB) {
    return matMul2d_tAtB(A, B, C, alpha, beta, M, N, K);
  } else if (transA) {
    return matMul2d_tA(A, B, C, alpha, beta, M, N, K);
  } else if (transB) {
    return matMul2d_tB(A, B, C, alpha, beta, M, N, K);
  } else {
    return matMul2d_(A, B, C, alpha, beta, M, N, K);
  }
}

function matMul2d_(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += 1;
        offsetB += N;
      }
      offsetA -= K;
      offsetB -= N * K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB++;
    }
    offsetB -= N;
    offsetA += K;
  }
}

function matMul2d_tA(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += M;
        offsetB += N;
      }
      offsetA -= M * K;
      offsetB -= N * K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB++;
    }
    offsetB -= N;
    offsetA++;
  }
}

function matMul2d_tB(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += 1;
        offsetB += 1;
      }
      offsetA -= K;
      offsetB -= K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB += K;
    }
    offsetB -= N * K;
    offsetA += K;
  }
}

function matMul2d_tAtB(
    A: Float32Array|Float64Array, B: Float32Array|Float64Array, C: Float32Array|Float64Array, alpha: number,
    beta: number, M: number, N: number, K: number) {
  let offsetA = 0, offsetB = 0, offsetC = 0;
  for (let mm = 0; mm < M; mm++) {
    for (let nn = 0; nn < N; nn++) {
      let sum = 0;
      for (let kk = 0; kk < K; kk++) {
        sum += A[offsetA] * B[offsetB];
        offsetA += M;
        offsetB += 1;
      }
      offsetA -= M * K;
      offsetB -= K;
      C[offsetC] = alpha * sum + beta * C[offsetC];
      offsetC++;
      offsetB += K;
    }
    offsetB -= N * K;
    offsetA++;
  }
}

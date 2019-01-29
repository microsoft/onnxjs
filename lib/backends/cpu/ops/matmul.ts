// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import ndarray from 'ndarray';

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
  let dimsA: number[];
  let dimsB: number[];
  [dimsA, dimsB] = MatMulUtil.preprocessInputShapes(a.dims.slice(), b.dims.slice());
  const mat2dShape = [dimsA[dimsA.length - 2], dimsB[dimsB.length - 1]];
  let shape = BroadcastUtil.calcShape(dimsA, dimsB, true);
  if (!shape) {
    // the inputs cannot broadcast or cannot multiply
    throw new Error(`input dimensions do not match the requirement`);
  }
  // make a copy and re-assign because this can be modified later
  shape = shape.slice(0);
  const size = ShapeUtil.size(shape);
  const num2dMatrices = size / (mat2dShape[0] * mat2dShape[1]);

  let ndA: ndarray;
  let ndB: ndarray;
  let ndY: ndarray;
  let isFloat64 = false;
  ndA = ndarray(a.floatData, dimsA);
  ndB = ndarray(b.floatData, dimsB);
  if (a.type === 'float64' || b.type === 'float64') {
    ndY = ndarray(new Float64Array(size));
    isFloat64 = true;
  } else {
    ndY = ndarray(new Float32Array(size));
  }

  let curPos = 0;
  const indices = new Array<number>(shape.length);
  const indicesA = new Array(ndA.shape.length);
  const indicesB = new Array(ndB.shape.length);
  for (let i = 0; i < num2dMatrices; i++) {
    // traverse nd array at 2d level
    let rest = i;
    for (let j = shape.length - 3; j >= 0; j--) {
      indices[j] = rest % shape[j];
      rest = Math.floor(rest / shape[j]);
    }
    // map the "broadcasted" index to original ndarray index
    BroadcastUtil.fillIndex(indices, ndA.shape, indicesA);
    BroadcastUtil.fillIndex(indices, ndB.shape, indicesB);
    // slice and get 2d subarrays
    const subarrayA = shape.length === 2 ? ndA : ndA.pick(...indicesA);
    const subarrayB = shape.length === 2 ? ndB : ndB.pick(...indicesB);
    // multiply like conventional matrices
    MatMul2d(subarrayA, subarrayB, ndY, curPos);
    curPos += mat2dShape[0] * mat2dShape[1];
  }
  MatMulUtil.postprocessOutputShape(shape as number[], a.dims.length, b.dims.length);
  const tensorY = new Tensor(shape, isFloat64 ? 'float64' : 'float32');
  tensorY.floatData.set(ndY.data);
  return tensorY;
}

function MatMul2d(A: ndarray, B: ndarray, Y: ndarray, startPos: number) {
  // 2d matrix multiplication. Y[i,j] = sum(A[i, k] + B[k, j])
  let offset = 0;
  for (let i = 0; i < A.shape[0]; i++) {
    for (let j = 0; j < B.shape[1]; j++) {
      let sum = 0;
      for (let k = 0; k < A.shape[1]; k++) {
        sum += A.get(i, k) * B.get(k, j);
      }
      Y.set(startPos + offset, sum);
      offset++;
    }
  }
}

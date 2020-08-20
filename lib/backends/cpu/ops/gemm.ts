// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Gemm} from '../../../ops/gemm';
import {Tensor} from '../../../tensor';
import * as util from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

import {matMul2d} from './matmul';

export class CpuGemm extends Gemm {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = gemm(
        inputs[0], inputs[1], this.alpha, this.beta, this.transA, this.transB,
        inputs.length === 3 ? inputs[2] : undefined);
    return [output];
  }
}

export function gemm(a: Tensor, b: Tensor, alpha: number, beta: number, transA: boolean, transB: boolean, c?: Tensor) {
  const [M, N, K] = util.GemmUtil.getShapeOfGemmResult(a.dims, transA, b.dims, transB, c?.dims);

  // The result will always be of the shape [M,N]
  const output = new Tensor([M, N], a.type);
  // broadcast and assign value from C to output
  if (c && util.BroadcastUtil.calc(output, c, (a, b) => b, true) !== output) {
    throw new Error(`tensor C is not broadcastable to [M,N]`);
  }

  matMul2d(a.floatData, b.floatData, output.floatData, transA, transB, alpha, beta, M, N, K);

  return output;
}

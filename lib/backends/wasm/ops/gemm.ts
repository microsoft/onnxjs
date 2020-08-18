// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Gemm} from '../../../ops/gemm';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, GemmUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmGemm extends Gemm {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const a = inputs[0];
    const b = inputs[1];
    const c = inputs[2];

    const [M, N] = GemmUtil.getShapeOfGemmResult(a.dims, this.transA, b.dims, this.transB, c?.dims);
    const y = new Tensor([M, N], a.type);
    if (c && !BroadcastUtil.calc(y, c, (a, b) => (b), true)) {
      throw new Error(`c is not broadcastable to the shape of the result of the Gemm operator`);
    }
    WasmBinding.getInstance().ccall(
        '_gemm_f32', [this.transA, 'bool'], [this.transB, 'bool'], [this.transA ? a.dims[1] : a.dims[0], 'int32'],
        [this.transB ? b.dims[0] : b.dims[1], 'int32'], [this.transA ? a.dims[0] : a.dims[1], 'int32'],
        [this.alpha, 'float32'], [a.floatData, 'float32ptr'], [b.floatData, 'float32ptr'], [this.beta, 'float32'],
        [y.floatData, 'float32ptr', 'inout']);

    return [y];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32' || inputs[2].type !== 'float32') {
      return false;
    }

    if ((inputs[0].type !== inputs[1].type) || (inputs[0].type !== inputs[2].type)) {
      return false;
    }

    return true;
  }
}

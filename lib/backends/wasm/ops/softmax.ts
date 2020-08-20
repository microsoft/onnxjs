// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Softmax} from '../../../ops/softmax';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmSoftmax extends Softmax {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const x = inputs[0];
    const axis = ShapeUtil.normalizeAxis(this.axis, x.dims.length);
    const N = ShapeUtil.sizeToDimension(x.dims, axis);
    const D = ShapeUtil.sizeFromDimension(x.dims, axis);
    const y = new Tensor(x.dims, x.type);
    WasmBinding.getInstance().ccall(
        '_softmax_f32', [x.floatData, 'float32ptr'], [y.floatData, 'float32ptr', 'out'], [N, 'int32'], [D, 'int32']);

    return [y];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32') {
      return false;
    }

    return true;
  }
}

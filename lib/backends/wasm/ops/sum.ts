// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Sum} from '../../../ops/sum';
import {Tensor} from '../../../tensor';
import {WasmBinding, WasmCallArgument} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmSum extends Sum {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const y = new Tensor(inputs[0].dims, inputs[0].type);
    const size = inputs[0].floatData.length;
    const input = new Array<WasmCallArgument>(inputs.length);
    for (let i = 0; i < inputs.length; i++) {
      input[i] = [inputs[i].floatData, 'float32ptr'];
    }
    WasmBinding.getInstance().ccall(
        '_sum_f32', [inputs.length, 'int32'], [size, 'int32'], [y.floatData, 'float32ptr', 'inout'], ...input);

    return [y];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32') {
      return false;
    }
    for (let i = 1; i < inputs.length; i++) {
      if (inputs[0].type !== inputs[i].type) {
        return false;
      }
    }

    return true;
  }
}

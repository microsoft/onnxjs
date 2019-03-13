// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Clip} from '../../../ops/clip';
import {Tensor} from '../../../tensor';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmClip extends Clip {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const result = new Tensor(inputs[0].dims, inputs[0].type);
    const size = result.floatData.length;
    if (inputs[0].type === 'float32') {
      WasmBinding.getInstance().ccall(
          '_clip_f32', [inputs[0].floatData, 'float32ptr'], [result.floatData, 'float32ptr', 'out'], [size, 'int32'],
          [this.min, 'float32'], [this.max, 'float32']);
    }
    // Expand for differnt types supported for this specific kernel of Clip
    else {
      throw new Error(`Unsupported input type for Clip operator.`);
    }
    return [result];
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

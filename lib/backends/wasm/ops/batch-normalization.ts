// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BatchNormalization} from '../../../ops/batch-normalization';
import {Tensor} from '../../../tensor';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmBatchNormalization extends BatchNormalization {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const x = inputs[0];
    const scale = inputs[1];
    const b = inputs[2];
    const mean = inputs[3];
    const variance = inputs[4];

    // calculate channel size (i.e.) data points per channel
    let channelSize = 1;
    for (let i = 2; i < x.dims.length; i++) {
      channelSize *= x.dims[i];
    }

    // create output Tensor after determining output size
    const y = new Tensor(x.dims, x.type);
    WasmBinding.getInstance().ccall(
        '_batch_normalization_f32', [x.floatData, 'float32ptr'], [y.floatData, 'float32ptr', 'out'],
        [x.dims[0], 'int32'], [x.dims[1], 'int32'], [channelSize, 'int32'], [scale.floatData, 'float32ptr'],
        [b.floatData, 'float32ptr'], [mean.floatData, 'float32ptr'], [variance.floatData, 'float32ptr'],
        [this.epsilon, 'float32']);

    return [y];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    const X = inputs[0];
    const scale = inputs[1];
    const B = inputs[2];
    const mean = inputs[3];
    const var_ = inputs[4];

    // input should atleast have three dimensions - N,C,dim1,...,dimn
    // other inputs need to be one dimensional
    if (X.dims.length < 3 || scale.dims.length !== 1 || B.dims.length !== 1 || mean.dims.length !== 1 ||
        var_.dims.length !== 1) {
      return false;
    }
    if (scale.dims[0] !== X.dims[1] || B.dims[0] !== X.dims[1] || mean.dims[0] !== X.dims[1] ||
        var_.dims[0] !== X.dims[1]) {
      return false;
    }
    // currently Wasm backend only supports 'float32' input type
    if (X.type !== 'float32' || scale.type !== 'float32' || B.type !== 'float32' || mean.type !== 'float32' ||
        var_.type !== 'float32') {
      return false;
    }
    return true;
  }
}

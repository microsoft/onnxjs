// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {MatMul} from '../../../ops/matmul';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, MatMulUtil, ShapeUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmMatMul extends MatMul {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const [dimsA, dimsB] = MatMulUtil.preprocessInputShapes(inputs[0].dims, inputs[1].dims);
    const outputShape = BroadcastUtil.calcShape(dimsA, dimsB, true);
    if (!outputShape) {
      // the inputs cannot broadcast or cannot multiply
      throw new Error(`input dimensions do not match the requirement`);
    }

    const outputSize = ShapeUtil.size(outputShape);
    const resultData = new Float32Array(outputSize);
    WasmBinding.getInstance().ccall(
        '_matmul_f32', [inputs[0].floatData, 'float32ptr'], [inputs[0].dims, 'int32ptr'],
        [inputs[0].dims.length, 'int32'], [inputs[1].floatData, 'float32ptr'], [inputs[1].dims, 'int32ptr'],
        [inputs[1].dims.length, 'int32'], [resultData, 'float32ptr', 'out'], [resultData.length, 'int32'],
        [outputShape, 'int32ptr'], [outputShape.length, 'int32']);
    MatMulUtil.postprocessOutputShape(outputShape as number[], inputs[0].dims.length, inputs[1].dims.length);
    const result = new Tensor(outputShape, inputs[0].type);
    result.floatData.set(resultData);
    return [result];
  }

  // overriding the checkInputTypes() in the base class because Wasm backend has special type limitations
  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32' || inputs[1].type !== 'float32') {
      return false;
    }

    if (inputs[0].type !== inputs[1].type) {
      return false;
    }

    return true;
  }
}

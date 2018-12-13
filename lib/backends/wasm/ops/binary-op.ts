// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BinaryOp} from '../../../ops/binary-op';
import {Tensor} from '../../../tensor';
import {BroadcastUtil, ShapeUtil} from '../../../util';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmBinaryOp extends BinaryOp {
  constructor(typeConstraint: ReadonlyArray<Tensor.DataType>, opType: string, resultType?: Tensor.DataType) {
    super(typeConstraint, opType, resultType);
  }

  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const outputShape = BroadcastUtil.calcShape(inputs[0].dims, inputs[1].dims, false);
    if (!outputShape) {
      throw new Error('not broadcastable');
    }
    const resultDataLength = ShapeUtil.size(outputShape);
    const resultData = new Float32Array(resultDataLength);
    let fun = '';
    switch (this.opType) {
      case 'Add':
        fun = '_add_f32';
        break;
      case 'Sub':
        fun = '_sub_f32';
        break;
      case 'Mul':
        fun = '_mul_f32';
        break;
      case 'Div':
        fun = '_div_f32';
        break;
      case 'Xor':
        fun = '_xor_f32';
        break;
      case 'Or':
        fun = '_or_f32';
        break;
      case 'And':
        fun = '_and_f32';
        break;
      case 'PRelu':
        fun = '_prelu_f32';
        break;
      default:
        throw Error(`unsupported binary op by the Wasm backend`);
    }

    const inputType = inputs[0].type;
    const outputType = this.resultType ? this.resultType : inputs[0].type;
    type fourByteTypes = Float32Array|Int32Array|Uint32Array;

    if (inputType === 'float32' || inputType === 'int32' || inputType === 'uint32') {
      WasmBinding.getInstance().ccall(
          fun, [inputs[0].numberData as fourByteTypes, 'float32ptr'], [inputs[0].data.length, 'int32'],
          [inputs[0].dims.length, 'int32'], [inputs[0].dims, 'int32ptr'],
          [inputs[1].numberData as fourByteTypes, 'float32ptr'], [inputs[1].data.length, 'int32'],
          [inputs[1].dims.length, 'int32'], [inputs[1].dims, 'int32ptr'], [resultData, 'float32ptr', 'out'],
          [resultData.length, 'int32'], [outputShape.length, 'int32'], [outputShape, 'int32ptr']);
    } else {
      WasmBinding.getInstance().ccall(
          fun, [Float32Array.from(inputs[0].numberData), 'float32ptr'], [inputs[0].data.length, 'int32'],
          [inputs[0].dims.length, 'int32'], [inputs[0].dims, 'int32ptr'],
          [Float32Array.from(inputs[1].numberData), 'float32ptr'], [inputs[1].data.length, 'int32'],
          [inputs[1].dims.length, 'int32'], [inputs[1].dims, 'int32ptr'], [resultData, 'float32ptr', 'out'],
          [resultData.length, 'int32'], [outputShape.length, 'int32'], [outputShape, 'int32ptr']);
    }

    return [new Tensor(
        outputShape, outputType, undefined, undefined, createOutputTypedArrayBasedOnType(resultData, outputType))];
  }
}

function createOutputTypedArrayBasedOnType(result: Float32Array, requiredType: string) {
  switch (requiredType) {
    case 'bool':
    case 'uint8':
      return Uint8Array.from(result);
    case 'int8':
      return Int8Array.from(result);
    case 'int16':
      return Int16Array.from(result);
    case 'uint16':
      return Uint16Array.from(result);
    case 'int32':
      return Int32Array.from(result);
    case 'uint32':
      return Uint32Array.from(result);
    case 'float32':
      return result;
    case 'float64':
      return Float64Array.from(result);
    default:
      throw new Error('unsupported tensor type');
  }
}

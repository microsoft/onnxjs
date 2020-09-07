// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Einsum} from '../../../ops/einsum';
import {Tensor} from '../../../tensor';
import {WasmBinding} from '../../../wasm-binding';
import {WasmInferenceHandler} from '../inference-handler';

export class WasmEinsum extends Einsum {
  run(inferenceHandler: WasmInferenceHandler, inputs: Tensor[]): Tensor[] {
    const dimensionSizeMap: {[name: string]: number} = {};
    this.matchInputs(inputs, dimensionSizeMap);
    const outputShape = this.calculateOutputSize(dimensionSizeMap);

    let i = 0;
    const sizes = [];
    const nameToId: {[name: string]: number} = {};
    const idToName: {[id: number]: string} = {};

    for (const name in dimensionSizeMap) {
      sizes.push(dimensionSizeMap[name]);
      nameToId[name] = i;
      idToName[i] = name;
      i++;
    }

    const outputIndices: number[] = [];
    const input1Indices: number[] = [];
    const input2Indices: number[] = [];
    for (const outputName of this.outputNames) {
      outputIndices.push(nameToId[outputName]);
    }
    for (const inputName of this.input1Names) {
      input1Indices.push(nameToId[inputName]);
    }
    if (this.input2) {
      for (const inputName of this.input2Names) {
        input2Indices.push(nameToId[inputName]);
      }
    }

    const y = new Tensor(outputShape, inputs[0].type);

    if (inputs.length === 2) {
      WasmBinding.getInstance().ccall(
          '_einsum_f32',
          [inputs[0].floatData, 'float32ptr'],
          [inputs[1].floatData, 'float32ptr'],
          [y.floatData, 'float32ptr', 'inout'],
          [sizes, 'int32ptr'],
          [sizes.length, 'int32'],
          [outputIndices, 'int32ptr'],
          [outputIndices.length, 'int32'],
          [input1Indices, 'int32ptr'],
          [input1Indices.length, 'int32'],
          [input1Indices, 'int32ptr'],
          [input2Indices.length, 'int32'],
      );
    } else {
      WasmBinding.getInstance().ccall(
          '_einsum_single_f32',
          [inputs[0].floatData, 'float32ptr'],
          [y.floatData, 'float32ptr', 'inout'],
          [sizes, 'int32ptr'],
          [sizes.length, 'int32'],
          [outputIndices, 'int32ptr'],
          [outputIndices.length, 'int32'],
          [input1Indices, 'int32ptr'],
          [input1Indices.length, 'int32'],
      );
    }

    return [y];
  }

  checkInputTypes(inputs: Tensor[]): boolean {
    // currently Wasm backend only supports 'float32' input type
    if (inputs[0].type !== 'float32' || (inputs.length > 1 && inputs[1].type !== 'float32')) {
      return false;
    }

    return super.checkInputTypes(inputs);
  }
}

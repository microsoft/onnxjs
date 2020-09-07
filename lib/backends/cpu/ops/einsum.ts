// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Einsum} from '../../../ops/einsum';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

import {ShapeUtil} from './../../../util';

export class CpuEinsum extends Einsum {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
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
    const inputIndices: number[][] = [];
    for (const outputName of this.outputNames) {
      outputIndices.push(nameToId[outputName]);
    }
    for (let i = 0; i < this.inputs.length; i++) {
      const indices = [];
      for (const inputName of this.inputNames[i]) {
        indices.push(nameToId[inputName]);
      }
      inputIndices.push(indices);
    }

    const result = einsum(outputShape, inputs, sizes, outputIndices, inputIndices);

    return [result];
  }
}

export function einsum(
    outputShape: number[], inputs: Tensor[], sizes: number[], outputIndices: number[],
    inputIndices: number[][]): Tensor {
  const result = new Tensor(outputShape, inputs[0].type);
  const totalSize = ShapeUtil.size(sizes);
  let i = 0;
  const index = new Array(sizes.length).fill(0);

  while (i < totalSize) {
    const outputIx: number[] = [];
    for (const outputIndex of outputIndices) {
      outputIx.push(index[outputIndex]);
    }

    let value = 1;
    for (let i = 0; i < inputIndices.length; i++) {
      const inputIx: number[] = [];
      for (const inputIndex of inputIndices[i]) {
        inputIx.push(index[inputIndex]);
      }
      value *= inputs[i].get(inputIx) as number;
    }

    result.set(outputIx, result.get(outputIx) as number + value);

    i++;
    ShapeUtil.incrementIndex(index, sizes);
  }

  return result;
}

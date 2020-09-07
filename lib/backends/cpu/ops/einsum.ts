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

    const result =
        einsum(outputShape, inputs, sizes, outputIndices, input1Indices, this.input2 ? input2Indices : undefined);

    return [result];
  }
}

export function einsum(
    outputShape: number[], inputs: Tensor[], sizes: number[], outputIndices: number[], input1Indices: number[],
    input2Indices?: number[]): Tensor {
  const result = new Tensor(outputShape, inputs[0].type);
  const totalSize = ShapeUtil.size(sizes);
  let i = 0;
  const index = new Array(sizes.length).fill(0);

  while (i < totalSize) {
    const outputIx: number[] = [];
    for (const outputIndex of outputIndices) {
      outputIx.push(index[outputIndex]);
    }

    const input1Ix: number[] = [];
    for (const input1Index of input1Indices) {
      input1Ix.push(index[input1Index]);
    }
    let value = inputs[0].get(input1Ix) as number;
    if (input2Indices) {
      const input2Ix: number[] = [];
      for (const input2Index of input2Indices) {
        input2Ix.push(index[input2Index]);
      }
      value *= inputs[1].get(input2Ix) as number;
    }

    result.set(outputIx, result.get(outputIx) as number + value);

    i++;
    ShapeUtil.incrementIndex(index, sizes);
  }

  return result;
}

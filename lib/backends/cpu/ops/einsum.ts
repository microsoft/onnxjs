// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Einsum} from '../../../ops/einsum';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

import {ShapeUtil} from './../../../util';

export class CpuEinsum extends Einsum {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const dimensionSizeMap: {[id: string]: number} = {};
    this.matchInputs(inputs, dimensionSizeMap);
    const outputShape = this.calculateOutputSize(dimensionSizeMap);

    const result = new Tensor(outputShape, inputs[0].type);

    let i = 0;
    const index = [];
    const sizes = [];
    const nameToId: {[name: string]: number} = {};
    const idToName: {[id: number]: string} = {};
    for (const name in dimensionSizeMap) {
      index.push(0);
      sizes.push(dimensionSizeMap[name]);
      nameToId[name] = i;
      idToName[i] = name;
      i++;
    }

    const totalSize = ShapeUtil.size(sizes);
    i = 0;

    while (i < totalSize) {
      const outputIx: number[] = [];
      for (const outputName of this.outputIndices) {
        outputIx.push(index[nameToId[outputName]]);
      }

      const input1Ix: number[] = [];
      for (const input1Name of this.input1Indices) {
        input1Ix.push(index[nameToId[input1Name]]);
      }
      let value = inputs[0].get(input1Ix) as number;
      if (this.input2) {
        const input2Ix: number[] = [];
        for (const input2Name of this.input2Indices) {
          input2Ix.push(index[nameToId[input2Name]]);
        }
        value *= inputs[1].get(input2Ix) as number;
      }

      result.set(outputIx, result.get(outputIx) as number + value);

      i++;
      ShapeUtil.incrementIndex(index, sizes);
    }

    return [result];
  }
}

export function einsum(a: Tensor, b: Tensor) {
  return undefined;
}

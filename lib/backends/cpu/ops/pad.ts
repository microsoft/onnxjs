// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Pad} from '../../../ops/pad';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';
// import { getLogger } from 'log4js';

export class CpuPad extends Pad {
  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = pad(inputs[0], this.mode, this.value, this.pads);
    return [output];
  }
}

export function pad(x: Tensor, mode: string, value: number, pads: number[]) {
  const inputDimensions = x.dims;
  const outputDimensions = getPadDimension(inputDimensions, pads);

  const output = new Tensor(outputDimensions, x.type);

  switch (mode) {
    case 'constant':
      for (let ind = 0; ind < outputDimensions.reduce((acc, cur) => acc * cur, 1); ind++) {
        const inx = mapToArrayIndex(ind, outputDimensions);
        output.set(inx, value);
      }
      for (let ind = 0; ind < inputDimensions.reduce((acc, cur) => acc * cur, 1); ind++) {
        const inx = mapToArrayIndex(ind, inputDimensions);
        output.set(inx.map((v, i) => v + pads[i]), x.get(inx));
      }
      break;
    case 'reflect':
      for (let ind = 0; ind < outputDimensions.reduce((acc, cur) => acc * cur, 1); ind++) {
        const inx = mapToArrayIndex(ind, outputDimensions);
        output.set(inx, x.get(inx.map((v, i) => getReflectionIndex(v, pads[i], inputDimensions[i]))));
      }
      break;
    case 'edge':
      for (let ind = 0; ind < outputDimensions.reduce((acc, cur) => acc * cur, 1); ind++) {
        const inx = mapToArrayIndex(ind, outputDimensions);
        output.set(inx, x.get(inx.map((v, i) => getEdgeIndex(v, pads[i], inputDimensions[i]))));
      }
      break;
    default:
      throw Error('Illegal pad mode.');
  }

  return output;
}

function getReflectionIndex(index: number, offset: number, size: number) {
  if (index < offset) {
    const position = (offset - index - 1) % (size - 1);
    const direction = Math.floor((offset - index - 1) / (size - 1)) % 2;
    if (direction === 1) {
      return size - position - 2;
    } else {
      return position + 1;
    }
  } else if (index < offset + size) {
    return index - offset;
  } else {
    const position = (index - (offset + size)) % (size - 1);
    const direction = Math.floor((index - (offset + size)) / (size - 1)) % 2;
    if (direction === 0) {
      return size - position - 2;
    } else {
      return position + 1;
    }
  }
}

function getEdgeIndex(index: number, offset: number, size: number) {
  if (index < offset) {
    return 0;
  } else if (index < offset + size) {
    return index - offset;
  } else {
    return size - 1;
  }
}

function mapToArrayIndex(numberIndex: number, dimension: readonly number[]) {
  if (numberIndex < 0 || (dimension.some(val => val < 0))) {
    throw Error('Array index out of range');
  }
  const arrayIndex = [...dimension];
  arrayIndex.reverse();
  function product(array: readonly number[]) {
    return array.reduce((acc, cur) => acc * cur, 1);
  }
  return arrayIndex.map((value, index, array) => Math.floor(numberIndex / product(array.slice(0, index))) % value)
      .reverse();
}

function getPadDimension(inputDimensions: readonly number[], pads: number[]) {
  const outputDimensions = Array(inputDimensions.length);
  Object.assign(outputDimensions, inputDimensions);
  for (let i = 0; i < inputDimensions.length; i++) {
    outputDimensions[i] += pads[i] + pads[i + outputDimensions.length];
  }
  return outputDimensions;
}

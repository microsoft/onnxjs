// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export abstract class Einsum implements Operator {
  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute): void {
    this.equation = attributes.getString('equation');
    const split = this.equation.split('->');
    this.lhs = split[0].trim();
    if (split.length === 2) {
      this.rhs = split[1].trim();
      this.implicit = false;
    } else {
      this.implicit = true;
    }

    const lhsSplit = this.lhs.split(',');
    this.input1 = lhsSplit[0].trim();
    if (lhsSplit.length === 2) {
      this.input2 = lhsSplit[1].trim();
    }

    this.parseEquationPart(this.input1, this.input1Names);
    if (this.input2) {
      this.parseEquationPart(this.input2, this.input2Names);
    }
    if (this.rhs) {
      this.parseEquationPart(this.rhs, this.outputNames);
    }
  }

  private parseEquationPart(part: string, indices: string[]) {
    for (let i = 0; i < part.length; i++) {
      const char = part.charAt(i);

      if (char === '.') {
        throw new Error('Use of ellipsis (...) in einsum not yet supported');
      }

      indices.push(char);
    }
  }

  protected matchInputs(inputs: Tensor[], dimensionSizeMap: {[name: string]: number}) {
    this.matchDimensions(this.input1Names, inputs[0].dims, dimensionSizeMap);
    if (this.input2) {
      this.matchDimensions(this.input2Names, inputs[1].dims, dimensionSizeMap);
    }
  }

  protected calculateOutputSize(dimensionSizeMap: {[name: string]: number}): number[] {
    const result: number[] = [];
    for (let i = 0; i < this.outputNames.length; i++) {
      result.push(dimensionSizeMap[this.outputNames[i]]);
    }
    return result;
  }

  checkInputs(inputs: Tensor[]): boolean {
    const dimensionMap: {[id: string]: number} = {};

    if (inputs.length < 1 || inputs.length > 2) {
      return false;
    }

    if (!this.matchDimensions(this.input1Names, inputs[0].dims, dimensionMap)) {
      return false;
    }

    if (this.input2 && inputs.length < 2) {
      return false;
    } else if (this.input2 && !this.matchDimensions(this.input2Names, inputs[1].dims, dimensionMap)) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected matchDimensions(indices: string[], inputDims: readonly number[], dimensionMap: {[id: string]: number}):
      boolean {
    for (let j = 0; j < indices.length; j++) {
      const ix = indices[j];
      if (dimensionMap[ix] && dimensionMap[ix] !== inputDims[j]) {
        return false;
      } else if (!dimensionMap[ix]) {
        dimensionMap[ix] = inputDims[j];
      }
    }

    return true;
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    const allowedTypes = ['float32', 'float64', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32'];
    if (allowedTypes.indexOf(inputs[0].type) === -1 ||
        (inputs.length > 1 && allowedTypes.indexOf(inputs[1].type) === -1)) {
      return false;
    }

    if (inputs.length > 1 && inputs[0].type !== inputs[1].type) {
      return false;
    }
    return true;
  }

  protected equation: string;
  protected lhs: string;
  protected rhs?: string;

  protected input1: string;
  protected input2?: string;

  // Maps from input 1 axis to general axis id
  protected input1Names: string[] = [];
  // Maps from input 2 axis to general axis id
  protected input2Names: string[] = [];

  // Maps from output axis to general axis id
  protected outputNames: string[] = [];

  protected implicit: boolean;
}

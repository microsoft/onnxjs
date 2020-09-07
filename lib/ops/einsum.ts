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

    this.parseEquationPart(this.input1, this.input1Indices);
    if (this.input2) {
      this.parseEquationPart(this.input2, this.input2Indices);
    }
    if (this.rhs) {
      this.parseEquationPart(this.rhs, this.outputIndices);
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

  protected matchInputs(inputs: Tensor[], dimensionMap: {[id: string]: number}) {
    this.matchDimensions(this.input1Indices, inputs[0].dims, dimensionMap);
    if (this.input2) {
      this.matchDimensions(this.input2Indices, inputs[1].dims, dimensionMap);
    }
  }

  protected calculateOutputSize(dimensionMap: {[id: string]: number}): number[] {
    if (this.outputIndices.length === 0) {
      return [];
    }

    const result: number[] = [];
    for (let i = 0; i < this.outputIndices.length; i++) {
      result.push(dimensionMap[this.outputIndices[i]]);
    }
    return result;
  }

  checkInputs(inputs: Tensor[]): boolean {
    const dimensionMap: {[id: string]: number} = {};

    this.matchDimensions(this.input1Indices, inputs[0].dims, dimensionMap);
    if (this.input2) {
      this.matchDimensions(this.input2Indices, inputs[1].dims, dimensionMap);
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
    return true;
  }

  protected equation: string;
  protected lhs: string;
  protected rhs?: string;

  protected input1: string;
  protected input2?: string;

  // Maps from input 1 axis to general axis id
  protected input1Indices: string[] = [];
  // Maps from input 2 axis to general axis id
  protected input2Indices: string[] = [];

  // Maps from output axis to general axis id
  protected outputIndices: string[] = [];

  protected implicit: boolean;
}

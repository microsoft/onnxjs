// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import ndarray from 'ndarray';

import {Attribute} from '../../../attribute';
import {BinaryOp} from '../../../ops/binary-op';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

type BinaryOpLambda = (e1: number, e2: number) => number;

export class CpuBinaryOp extends BinaryOp {
  constructor(
      typeConstraint: ReadonlyArray<Tensor.DataType>, private opLambda?: (e1: number, e2: number) => number,
      opType?: string, resultType?: Tensor.DataType) {
    super(typeConstraint, opType, resultType);
  }
  initialize(attributes: Attribute): void {
    if (!this.opType && !this.opLambda) {
      throw new Error(`Both opType and opLambda cannot be missing for a binary op`);
    }
    // Expose functionality to construct opLambdas on the fly
    // This is not costly as initialize() should be invoked only once after the model is resolved to a graph object
    if (!this.opLambda) {
      switch (this.opType) {
        default:
          throw new Error(`Binary op could not be initialized. Missing op lambda.`);
      }
    }
  }

  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    const output = binaryOp(inputs[0], inputs[1], this.opLambda!, this.resultType);
    return [output];
  }
}

export function binaryOp(x: Tensor, y: Tensor, opLambda: BinaryOpLambda, resultType?: Tensor.DataType): Tensor {
  const result =
      BroadcastUtil.calc(ndarray(x.numberData, x.dims.slice(0)), ndarray(y.numberData, y.dims.slice(0)), opLambda);
  if (!result) {
    throw new Error('not broadcastable');
  }
  const output = new Tensor(result.shape, resultType ? resultType : x.type);
  output.numberData.set(result.data);
  return output;
}

// specific operator lambdas
// arithmetic ops
export const addLambda: BinaryOpLambda = (e1, e2) => (e1 + e2);
export const subLambda: BinaryOpLambda = (e1, e2) => (e1 - e2);
export const mulLambda: BinaryOpLambda = (e1, e2) => (e1 * e2);
export const divLambda: BinaryOpLambda = (e1, e2) => (e1 / e2);

// logical ops
export const xorLambda: BinaryOpLambda = (e1, e2) => (e1 ^ e2);
export const orLambda: BinaryOpLambda = (e1, e2) => (e1 || e2);
export const andLambda: BinaryOpLambda = (e1, e2) => (e1 && e2);
export const equalLambda: BinaryOpLambda = (e1, e2) => (e1 === e2) ? 1 : 0;
export const greaterLambda: BinaryOpLambda = (e1, e2) => (e1 > e2) ? 1 : 0;
export const lessLambda: BinaryOpLambda = (e1, e2) => (e1 < e2) ? 1 : 0;

// misc ops
export const pReluLambda: BinaryOpLambda = (e1, e2) => (e1 >= 0 ? e1 : e1 * e2);
export const powLambda: BinaryOpLambda = (e1, e2) => Math.pow(e1, e2);

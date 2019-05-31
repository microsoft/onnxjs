// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {BinaryOp} from '../../../ops/binary-op';
import {Tensor} from '../../../tensor';
import {BroadcastUtil} from '../../../util';
import {CpuInferenceHandler} from '../inference-handler';

export class CpuBinaryOp extends BinaryOp {
  constructor(
      typeConstraint: ReadonlyArray<Tensor.DataType>, private opLambda?: (e1: number, e2: number) => number,
      opType?: string, resultType?: Tensor.DataType) {
    super(typeConstraint, opType, resultType);
  }

  // overriding the initialize() in the base class
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
    const output = binaryOp(inputs[0], inputs[1], this.opLambda!, false, this.resultType);
    return [output];
  }
}

function binaryOp(
    x: Tensor, y: Tensor, opLambda: (e1: number, e2: number) => number, inplace: boolean,
    resultType?: Tensor.DataType): Tensor {
  const result = BroadcastUtil.calc(x, y, opLambda, inplace, resultType);
  if (!result) {
    throw new Error('not broadcastable');
  }
  return result;
}

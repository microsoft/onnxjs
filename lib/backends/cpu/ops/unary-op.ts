// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {UnaryOp} from '../../../ops/unary-op';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

type UnaryOpCoreFunction = (input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) => void;

export class CpuUnaryOp extends UnaryOp {
  constructor(
      typeConstraint: ReadonlyArray<Tensor.DataType>, private func: UnaryOpCoreFunction, resultType?: Tensor.DataType) {
    super(typeConstraint, resultType);
  }

  initialize(attributes: Attribute): void {
    this.attributes = attributes;
  }

  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    // TODO:  use webpack + ts-loader + CustomTransformer
    // tslint:disable-next-line:max-line-length
    // https://github.com/TypeStrong/ts-loader#getcustomtransformers-----before-transformerfactory-after-transformerfactory--
    const output = unaryOp(inputs[0], this.func, this.attributes, this.resultType);
    return [output];
  }

  private attributes: Attribute;
}

export function unaryOp(
    x: Tensor, func: UnaryOpCoreFunction, attributes: Attribute, resultType?: Tensor.DataType): Tensor {
  const output = new Tensor(x.dims, resultType ? resultType : x.type);
  const inputNumberData = x.data as Tensor.NumberType;
  const outputNumberData = output.data as Tensor.NumberType;
  func(inputNumberData, outputNumberData, attributes);
  return output;
}

// specific implementations pertaining to each unary-op.
// although this can be accomplished with an op lambda
// that approach was found to be detrimental to performance
// so we use this approach which involves slight code duplication

export function abs(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.abs(input[i]);
  }
}

export function neg(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = -input[i];
  }
}

export function acos(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.acos(input[i]);
  }
}

export function ceil(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.ceil(input[i]);
  }
}

export function cos(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.cos(input[i]);
  }
}

export function sin(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.sin(input[i]);
  }
}

export function tan(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.tan(input[i]);
  }
}

export function tanh(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.tanh(input[i]);
  }
}

export function exp(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.exp(input[i]);
  }
}

export function floor(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.floor(input[i]);
  }
}

export function atan(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.atan(input[i]);
  }
}

export function relu(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.max(0, input[i]);
  }
}

export function leakyRelu(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  const alpha = attributes.getFloat('alpha', 0.01);
  for (let i = 0; i < input.length; i++) {
    const value = input[i];
    output[i] = value >= 0 ? value : alpha * value;
  }
}

export function log(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.log(input[i]);
  }
}

export function sqrt(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.sqrt(input[i]);
  }
}

export function asin(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.asin(input[i]);
  }
}

export function sigmoid(input: Tensor.NumberType, output: Tensor.NumberType, attributes: Attribute) {
  for (let i = 0; i < input.length; i++) {
    output[i] = (1 / (1 + Math.exp(-input[i])));
  }
}

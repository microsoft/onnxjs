// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../../../attribute';
import {UnaryOp} from '../../../ops/unary-op';
import {Tensor} from '../../../tensor';
import {CpuInferenceHandler} from '../inference-handler';

type UnaryOpCoreFunction<T> = (input: Tensor.NumberType, output: Tensor.NumberType, attributes?: T) => void;

export class CpuUnaryOp<T = unknown> extends UnaryOp {
  private attributes?: T;

  constructor(
      typeConstraint: ReadonlyArray<Tensor.DataType>, private func: UnaryOpCoreFunction<T>,
      private attributesInitializer?: (attributes: Attribute) => T, resultType?: Tensor.DataType) {
    super(typeConstraint, resultType);
  }

  initialize(attributes: Attribute) {
    if (this.attributesInitializer) {
      this.attributes = this.attributesInitializer(attributes);
    }
  }

  run(inferenceHandler: CpuInferenceHandler, inputs: Tensor[]): Tensor[] {
    // TODO:  use webpack + ts-loader + CustomTransformer
    // tslint:disable-next-line:max-line-length
    // https://github.com/TypeStrong/ts-loader#getcustomtransformers-----before-transformerfactory-after-transformerfactory--
    const output = unaryOp(inputs[0], this.func, this.attributes, this.resultType);
    return [output];
  }
}

export function unaryOp<T>(
    x: Tensor, func: UnaryOpCoreFunction<T>, attributes: T, resultType?: Tensor.DataType): Tensor {
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

export function abs(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.abs(input[i]);
  }
}

export function acos(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.acos(input[i]);
  }
}

export function acosh(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.acosh(input[i]);
  }
}

export function asin(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.asin(input[i]);
  }
}

export function asinh(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.asinh(input[i]);
  }
}

export function atan(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.atan(input[i]);
  }
}

export function atanh(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.atanh(input[i]);
  }
}

export function ceil(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.ceil(input[i]);
  }
}

export function clipInitializer(attributes: Attribute) {
  return {
    min: attributes.getFloat('min', -3.4028234663852886e+38),
    max: attributes.getFloat('max', 3.4028234663852886e+38)
  };
}

export function clip(input: Tensor.NumberType, output: Tensor.NumberType, attributes: {min: number, max: number}) {
  const min = attributes.min;
  const max = attributes.max;
  for (let i = 0; i < input.length; i++) {
    const value = input[i];
    output[i] = (value < min) ? min : (value > max) ? max : value;
  }
}

export function cos(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.cos(input[i]);
  }
}

export function cosh(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.cosh(input[i]);
  }
}

export function eluInitializer(attributes: Attribute) {
  return attributes.getFloat('alpha', 1.0);
}

export function elu(input: Tensor.NumberType, output: Tensor.NumberType, attributes: number) {
  const alpha = attributes;
  for (let i = 0; i < input.length; i++) {
    const value = input[i];
    output[i] = value >= 0 ? value : alpha * (Math.exp(value) - 1.0);
  }
}

export function exp(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.exp(input[i]);
  }
}

export function floor(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.floor(input[i]);
  }
}

export function isNan(input: Tensor.NumberType, output: Tensor.BooleanType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Number.isNaN(input[i]) ? 1 : 0;
  }
}

export function leakyReluInitializer(attributes: Attribute) {
  return attributes.getFloat('alpha', 0.01);
}

export function leakyRelu(input: Tensor.NumberType, output: Tensor.NumberType, attributes: number) {
  const alpha = attributes;
  for (let i = 0; i < input.length; i++) {
    const value = input[i];
    output[i] = value >= 0 ? value : alpha * value;
  }
}

export function log(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.log(input[i]);
  }
}

export function neg(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = -input[i];
  }
}

export function not(input: Tensor.BooleanType, output: Tensor.BooleanType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = input[i] ? 0 : 1;
  }
}

export function reciprocal(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = 1.0 / input[i];
  }
}

export function relu(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.max(0, input[i]);
  }
}

export function sigmoid(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = (1 / (1 + Math.exp(-input[i])));
  }
}

export function sign(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = input[i] > 0 ? 1 : input[i] < 0 ? -1 : 0;
  }
}

export function sin(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.sin(input[i]);
  }
}

export function sinh(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.sinh(input[i]);
  }
}

export function sqrt(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.sqrt(input[i]);
  }
}

export function tan(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.tan(input[i]);
  }
}

export function tanh(input: Tensor.NumberType, output: Tensor.NumberType) {
  for (let i = 0; i < input.length; i++) {
    output[i] = Math.tanh(input[i]);
  }
}

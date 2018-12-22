// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {argMax as argMaxImpl} from '../backends/cpu/ops/argMax';
import {binaryOp} from '../backends/cpu/ops/binary-op';
import {concat as concatImpl} from '../backends/cpu/ops/concat';
import {gather as gatherImpl} from '../backends/cpu/ops/gather';
import {reduceMax as reduceMaxImpl} from '../backends/cpu/ops/reduce';
import {reshape as reshapeImpl} from '../backends/cpu/ops/reshape';
import {slice as sliceImpl} from '../backends/cpu/ops/slice';
import {softmax as softmaxImpl} from '../backends/cpu/ops/softmax';
import {tile as tileImpl} from '../backends/cpu/ops/tile';
import {transpose as transposeImpl} from '../backends/cpu/ops/transpose';
import * as unaryOps from '../backends/cpu/ops/unary-op';
import {Tensor as InternalTensor} from '../tensor';
import {getActualAxisFromNegativeValue, ShapeUtil} from '../util';

import {Tensor} from './tensor';
import {TensorTransformUtils, toApiTensor, toInternalTensor, validateIndices} from './tensor-impl-utils';

type NumberDataType = Uint8Array|Int32Array|Float32Array;

export function zeros(shape: ReadonlyArray<number>, type: Tensor.NumberOrBoolType): Tensor {
  if (type !== 'float32' && type !== 'int32' && type !== 'bool') {
    throw new Error('Unsupported type for creating all zero Tensor');
  }
  validateIndices(shape);
  return new Tensor(TensorTransformUtils.createTypedArray(type, ShapeUtil.size(shape)), type, shape);
}

export function linspace(start: number, stop: number, num: number): Tensor {
  if (num === 0) {
    throw new Error('Must request atleast one sample');
  }
  const increments = (stop - start) / (num - 1);
  const data = TensorTransformUtils.createTypedArray('float32', num);
  data[0] = start;
  for (let i = 1; i < data.length; i++) {
    data[i] = data[i - 1] + increments;
  }
  return new Tensor(data, 'float32', [num]);
}

export function range(start: number, stop: number, step = 1, type: Tensor.NumberType = 'float32'): Tensor {
  if (step === 0) {
    throw new Error('Step size of 0 is not acceptable');
  }
  // adjust default values
  if (stop < step && step === 1) {
    step = -1;
  }
  // the following conditions cannot generate any data
  if (start === step || (start < stop && step < 0) || (stop < start && step > 0)) {
    return new Tensor(TensorTransformUtils.createTypedArray(type, 1), type, [1]);
  }
  const size = Math.abs(Math.ceil((stop - start) / step));
  const data = TensorTransformUtils.createTypedArray(type, size);
  data[0] = start;
  for (let i = 1; i < data.length; i++) {
    data[i] = data[i - 1] + step;
  }
  return new Tensor(data, type, [size]);
}

export function as1d(x: Tensor): Tensor {
  return reshape(x, [x.data.length]);
}

export function scalar(value: number, type: Tensor.NumberType = 'float32'): Tensor {
  if (type !== 'float32' && type !== 'int32') {
    throw new Error('Unsupported type for this transformation');
  }
  const data = TensorTransformUtils.createTypedArray(type, 1);
  data[0] = value;
  return new Tensor(data, type, [1]);
}

export function exp(x: Tensor): Tensor {
  if (x.type !== 'float32' && x.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return toApiTensor(unaryOps.unaryOp(toInternalTensor(x), unaryOps.exp, new Attribute(null)));
}

export function sigmoid(x: Tensor): Tensor {
  if (x.type !== 'float32' && x.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return toApiTensor(unaryOps.unaryOp(toInternalTensor(x), unaryOps.sigmoid, new Attribute(null)));
}

export function add(a: Tensor, b: Tensor): Tensor {
  if ((a.type !== 'float32' && a.type !== 'int32') || (b.type !== 'float32' && b.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (a.type !== b.type) {
    throw new Error('Types are not homogeneous');
  }
  return toApiTensor(binaryOp(toInternalTensor(a), toInternalTensor(b), (e1, e2) => (e1 + e2), a.type));
}

export function sub(a: Tensor, b: Tensor): Tensor {
  if ((a.type !== 'float32' && a.type !== 'int32') || (b.type !== 'float32' && b.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (a.type !== b.type) {
    throw new Error('Types are not homogeneous');
  }
  return toApiTensor(binaryOp(toInternalTensor(a), toInternalTensor(b), (e1, e2) => (e1 - e2), a.type));
}

export function mul(a: Tensor, b: Tensor): Tensor {
  if ((a.type !== 'float32' && a.type !== 'int32') || (b.type !== 'float32' && b.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (a.type !== b.type) {
    throw new Error('Types are not homogeneous');
  }
  return toApiTensor(binaryOp(toInternalTensor(a), toInternalTensor(b), (e1, e2) => (e1 * e2), a.type));
}

export function div(a: Tensor, b: Tensor): Tensor {
  if ((a.type !== 'float32' && a.type !== 'int32') || (b.type !== 'float32' && b.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (a.type !== b.type) {
    throw new Error('Types are not homogeneous');
  }
  return toApiTensor(binaryOp(toInternalTensor(a), toInternalTensor(b), (e1, e2) => (e1 / e2), a.type));
}

export function softmax(x: Tensor, axis = 1): Tensor {
  if (x.type !== 'float32' && x.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return toApiTensor(softmaxImpl(toInternalTensor(x), axis));
}

export function concat(x: Tensor[], axis: number): Tensor {
  if (x.length < 2) {
    throw new Error('Must have atleast 2 tensors to concatenate');
  }
  const types: Tensor.Type[] = [];
  x.forEach(t => {
    types.push(t.type);
  });
  TensorTransformUtils.validateSameTypes(types);
  const internalTensors: InternalTensor[] = [];
  x.forEach(t => {
    internalTensors.push(toInternalTensor(t));
  });
  return toApiTensor(concatImpl(internalTensors, axis));
}

export function slice(x: Tensor, starts: number[], ends: number[], axes?: number[]): Tensor {
  if (x.type === 'string') {
    throw new Error('Unspported type for this transformation');
  }
  return toApiTensor(sliceImpl(toInternalTensor(x), starts, ends, axes || []));
}

export function stack(x: Tensor[], axis = 0): Tensor {
  if (x.length < 2) {
    throw new Error('Must have at least 2 tensors to stack');
  }

  const types: Tensor.Type[] = [];
  const shapes: Array<ReadonlyArray<number>> = [];
  x.forEach(t => {
    types.push(t.type);
    shapes.push(t.dims ? t.dims : [t.data.length]);
  });
  TensorTransformUtils.validateSameTypes(types);
  TensorTransformUtils.validateEqualDims(shapes);
  const rank = x[0].dims ? x[0].dims.length : 1;
  axis = getActualAxisFromNegativeValue(axis, rank);
  const expanded = x.map(t => expandDims(t, axis));
  const internalTensors: InternalTensor[] = [];
  expanded.forEach(t => {
    internalTensors.push(toInternalTensor(t));
  });
  return toApiTensor(concatImpl(internalTensors, axis));
}

export function gather(x: Tensor, indices: Tensor, axis = 0): Tensor {
  if (x.type === 'string') {
    throw new Error('Unspported type for this transformation');
  }
  if (indices.type !== 'int32' || (indices.dims && indices.dims.length > 1)) {
    throw new Error('Indices tensor not of specified format');
  }
  return toApiTensor(gatherImpl(toInternalTensor(x), toInternalTensor(indices), axis));
}

export function tile(x: Tensor, repeats: ReadonlyArray<number>): Tensor {
  if (x.type === 'string') {
    throw new Error('Unspported type for this transformation');
  }
  const dims = x.dims ? x.dims : [x.data.length];
  const rank = dims.length;
  if (rank !== repeats.length) {
    throw new Error('Repetitions must be of the same rank as input dims');
  }
  return toApiTensor(tileImpl(
      toInternalTensor(x),
      new InternalTensor([repeats.length], 'int32', undefined, undefined, Int32Array.from(repeats))));
}

export function transpose(x: Tensor, perm?: number[]): Tensor {
  return toApiTensor(transposeImpl(toInternalTensor(x), perm));
}

export function expandDims(x: Tensor, axis: number): Tensor {
  axis = getActualAxisFromNegativeValue(axis, x.dims ? x.dims.length : 1);
  const dims = x.dims ? x.dims : [x.data.length];
  const changedShapeLength = dims.length + 1;
  const changedShape = new Array<number>(changedShapeLength);
  let iter = 0;
  for (let i = 0; i < changedShapeLength; ++i) {
    if (i === axis) {
      changedShape[i] = 1;
    } else {
      changedShape[i] = dims[iter++];
    }
  }
  return new Tensor(x.data, x.type, changedShape);
}

export function reshape(x: Tensor, shape: number[]): Tensor {
  return toApiTensor(reshapeImpl(
      toInternalTensor(x), new InternalTensor([shape.length], 'int32', undefined, undefined, Int32Array.from(shape))));
}

export function greaterEqual(a: Tensor, b: Tensor): Tensor {
  if ((a.type !== 'float32' && a.type !== 'int32') || (b.type !== 'float32' && b.type !== 'int32')) {
    throw new Error('Unsupported type for transform');
  }
  if (a.type !== b.type) {
    throw new Error('Types are not homogeneous');
  }
  return toApiTensor(binaryOp(toInternalTensor(a), toInternalTensor(b), (e1, e2) => (e1 >= e2 ? 1 : 0), 'bool'));
}

export function where(condition: Tensor, a: Tensor, b: Tensor): Tensor {
  // validate shape and types of input tensors and condition tensor
  ShapeUtil.areEqual(a.dims ? a.dims : [a.data.length], b.dims ? b.dims : [b.data.length]);
  TensorTransformUtils.validateSameTypes([a.type, b.type]);
  if (condition.type !== 'bool') {
    throw new Error('Condition tensor must be bool type');
  }

  // create output
  const outputShape = a.dims ? a.dims : [a.data.length];
  const output =
      new Tensor(TensorTransformUtils.createTypedArray(a.type, ShapeUtil.size(outputShape)), a.type, outputShape);
  const outputData = output.data;

  // input data
  const conditionData = condition.data;
  const X = a.data;
  const Y = b.data;

  // condition is 1D rank
  if (!condition.dims || condition.dims.length === 1) {
    // the outermost dimension of the input tensors and condition tensor must be the same
    const conditionDims = condition.dims ? condition.dims : [condition.data.length];
    const aDims = a.dims ? a.dims : [a.data.length];
    if (conditionDims[0] !== aDims[0]) {
      throw new Error('Outermost dimensions of input tensors and condition tensor must match');
    }

    let offset = 1;
    // Input tensors are not 1-D. Need to compute offset.
    if (a.dims && a.dims.length > 1) {
      for (let i = 1; i < a.dims.length; ++i) {
        offset *= a.dims[i];
      }
    }

    for (let i = 0; i < conditionData.length; ++i) {
      for (let j = 0; j < offset; ++j) {
        outputData[i * offset + j] = conditionData[i] > 0 ? X[i * offset + j] : Y[i * offset + j];
      }
    }

  } else {
    // The shapes of input tensors and condition tensor must be the same
    ShapeUtil.areEqual(condition.dims, b.dims ? b.dims : [b.data.length]);

    for (let i = 0; i < conditionData.length; ++i) {
      outputData[i] = conditionData[i] > 0 ? X[i] : Y[i];
    }
  }
  return output;
}

export function cast(x: Tensor, type: Tensor.Type): Tensor {
  // TODO: If the requested type and the given type are the same, return same tensor ?
  // Need to investigate if it breaks some basic assumptions
  switch (type) {
    case 'int32':
      return new Tensor(Int32Array.from(x.data as NumberDataType), 'int32', x.dims ? x.dims : [x.data.length]);
    case 'float32':
      return new Tensor(Float32Array.from(x.data as NumberDataType), 'float32', x.dims ? x.dims : [x.data.length]);
    case 'bool':
      return new Tensor(Uint8Array.from(x.data as NumberDataType), 'bool', x.dims ? x.dims : [x.data.length]);
    default:
      throw new Error('Unsupported type for casting');
  }
}

export function argMax(x: Tensor, axis = 0, keepdims = 1): Tensor {
  if (x.type !== 'float32' && x.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  return toApiTensor(argMaxImpl(toInternalTensor(x), axis, keepdims));
}

export function reduceMax(x: Tensor, axes?: number[], keepdims?: number): Tensor {
  if (x.type !== 'float32' && x.type !== 'int32') {
    throw new Error('Unsupported type for transform');
  }
  const rank = x.dims ? x.dims.length : 1;
  if (axes) {
    axes = axes.map(axis => getActualAxisFromNegativeValue(axis, rank));
  }
  return toApiTensor(reduceMaxImpl(toInternalTensor(x), axes || [], keepdims || 1));
}

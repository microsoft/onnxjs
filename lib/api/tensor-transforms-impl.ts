// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from './tensor';
export function zeros(shape: ReadonlyArray<number>, dtype: Tensor.NumberOrBoolType): Tensor {
  throw new Error('Method not implemented.');
}

export function linspace(start: number, stop: number, num: number): Tensor {
  throw new Error('Method not implemented.');
}

export function range(start: number, stop: number, step: number, dtype: Tensor.NumberType): Tensor {
  throw new Error('Method not implemented.');
}

export function as1d(t: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function scalar(value: number, dtype: Tensor.NumberType): Tensor {
  throw new Error('Method not implemented.');
}

export function exp(t: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function sigmoid(t: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function add(t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function sub(t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function mul(t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function div(t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function softmax(t: Tensor, dim: number): Tensor {
  throw new Error('Method not implemented.');
}

export function concat(tensors: Tensor[], axis: number): Tensor {
  throw new Error('Method not implemented.');
}

export function stack(tensors: Tensor[], axis: number): Tensor {
  throw new Error('Method not implemented.');
}

export function gather(t: Tensor, indices: ReadonlyArray<number>, axis: number): Tensor {
  throw new Error('Method not implemented.');
}

export function tile(t: Tensor, reps: ReadonlyArray<number>): Tensor {
  throw new Error('Method not implemented.');
}

export function transpose(t: Tensor, perm: ReadonlyArray<number>): Tensor {
  throw new Error('Method not implemented.');
}

export function expandDims(t: Tensor, axis: number): Tensor {
  throw new Error('Method not implemented.');
}

export function greaterEqual(t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function where(condition: ReadonlyArray<boolean>, t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}

export function cast(t: Tensor, dtype: Tensor.NumberOrBoolType): Tensor {
  throw new Error('Method not implemented.');
}

export function argMax(t: Tensor, axis: number): Tensor {
  throw new Error('Method not implemented.');
}

export function max(t: Tensor, axis: number, keepDims: boolean): Tensor {
  throw new Error('Method not implemented.');
}

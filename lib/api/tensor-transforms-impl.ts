// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from './tensor';

export function zeros(shape: ReadonlyArray<number>, type: Tensor.NumberOrBoolType): Tensor {
  throw new Error('Method not implemented.');
}
export function linspace(start: number, stop: number, num: number, type: Tensor.NumberType): Tensor {
  throw new Error('Method not implemented.');
}
export function range(start: number, stop: number, step: number, type: Tensor.NumberType): Tensor {
  throw new Error('Method not implemented.');
}
export function as1d(t: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function scalar(value: number, type: Tensor.NumberType): Tensor {
  throw new Error('Method not implemented.');
}
export function exp(x: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function sigmoid(x: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function add(a: Tensor, b: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function sub(a: Tensor, b: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function mul(a: Tensor, b: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function div(a: Tensor, b: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function softmax(x: Tensor, axis: number): Tensor {
  throw new Error('Method not implemented.');
}
export function concat(x: Tensor[], axis: number): Tensor {
  throw new Error('Method not implemented.');
}
export function stack(x: Tensor[], axis: number): Tensor {
  throw new Error('Method not implemented.');
}
export function gather(x: Tensor, indices: ReadonlyArray<number>, axis: number): Tensor {
  throw new Error('Method not implemented.');
}
export function tile(x: Tensor, repeats: ReadonlyArray<number>): Tensor {
  throw new Error('Method not implemented.');
}
export function transpose(x: Tensor, perm: ReadonlyArray<number>): Tensor {
  throw new Error('Method not implemented.');
}
export function expandDims(x: Tensor, axis: number): Tensor {
  throw new Error('Method not implemented.');
}
export function greaterEqual(a: Tensor, b: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function where(condition: ReadonlyArray<boolean>, t1: Tensor, t2: Tensor): Tensor {
  throw new Error('Method not implemented.');
}
export function cast(x: Tensor, type: Tensor.NumberOrBoolType): Tensor {
  throw new Error('Method not implemented.');
}
export function argMax(x: Tensor, axis: number): Tensor {
  throw new Error('Method not implemented.');
}
export function max(x: Tensor, axis: number): Tensor {
  throw new Error('Method not implemented.');
}

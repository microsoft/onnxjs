// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from './tensor';
import {TensorTransformsInterface} from './tensor-transforms';

// Placeholder
export function setTransforms(onnx: TensorTransforms) {}
export type NumberDataType = Uint8Array|Int32Array|Float32Array;

export class TensorTransforms implements TensorTransformsInterface {
  zeros(shape: ReadonlyArray<number>, dtype: Tensor.NumberOrBoolType): Tensor {
    throw new Error('Method not implemented.');
  }

  linspace(start: number, stop: number, num: number): Tensor {
    throw new Error('Method not implemented.');
  }

  range(start: number, stop: number, step: number, dtype: Tensor.NumberType): Tensor {
    throw new Error('Method not implemented.');
  }
  as1d(t: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  scalar(value: number, dtype: Tensor.NumberType): Tensor {
    throw new Error('Method not implemented.');
  }
  exp(t: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  sigmoid(t: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  add(t1: Tensor, t2: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  sub(t1: Tensor, t2: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  mul(t1: Tensor, t2: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  div(t1: Tensor, t2: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  softmax(t: Tensor, dim: number): Tensor {
    throw new Error('Method not implemented.');
  }
  concat(tensors: Tensor[], axis: number): Tensor {
    throw new Error('Method not implemented.');
  }
  stack(tensors: Tensor[], axis: number): Tensor {
    throw new Error('Method not implemented.');
  }
  gather(t: Tensor, indices: ReadonlyArray<number>, axis: number): Tensor {
    throw new Error('Method not implemented.');
  }
  tile(t: Tensor, reps: ReadonlyArray<number>): Tensor {
    throw new Error('Method not implemented.');
  }
  transpose(t: Tensor, perm: ReadonlyArray<number>): Tensor {
    throw new Error('Method not implemented.');
  }
  expandDims(t: Tensor, axis: number): Tensor {
    throw new Error('Method not implemented.');
  }
  greaterEqual(t1: Tensor, t2: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  where(condition: ReadonlyArray<boolean>, t1: Tensor, t2: Tensor): Tensor {
    throw new Error('Method not implemented.');
  }
  cast(t: Tensor, dtype: Tensor.NumberOrBoolType): Tensor {
    throw new Error('Method not implemented.');
  }
  argMax(t: Tensor, axis: number): Tensor {
    throw new Error('Method not implemented.');
  }
  max(t: Tensor, axis: number, keepDims: boolean): Tensor {
    throw new Error('Method not implemented.');
  }
}

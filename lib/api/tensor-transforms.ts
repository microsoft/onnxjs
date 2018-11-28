// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Exposes a set of light-weight tensor(s) transformations

import {Tensor} from './tensor';

export interface UtilityTensorCreators {
  zeros(shape: ReadonlyArray<number>, dtype: Tensor.NumberOrBoolType): Tensor;
  linspace(start: number, stop: number, num: number): Tensor;
  range(start: number, stop: number, step: number, dtype: Tensor.NumberType): Tensor;
}

export interface BasicMathTensorTransforms {
  exp(t: Tensor): Tensor;
  sigmoid(t: Tensor): Tensor;
}

export interface ArithmeticTensorTransforms {
  add(t1: Tensor, t2: Tensor): Tensor;
  sub(t1: Tensor, t2: Tensor): Tensor;
  mul(t1: Tensor, t2: Tensor): Tensor;
  div(t1: Tensor, t2: Tensor): Tensor;
}

export interface NormalizationTensorTransforms {
  softmax(t: Tensor, dim: number): Tensor;
}

export interface SliceAndJoinTensorTransforms {
  concat(tensors: Tensor[], axis: number): Tensor;
  stack(tensors: Tensor[], axis: number): Tensor;
  gather(t: Tensor, indices: ReadonlyArray<number>, axis: number): Tensor;
  tile(t: Tensor, reps: ReadonlyArray<number>): Tensor;
}

export interface PermutationTensorTransforms {
  transpose(t: Tensor, perm: ReadonlyArray<number>): Tensor;
}

export interface ShapeTensorTransforms {
  expandDims(t: Tensor, axis: number): Tensor;
}

export interface LogicalTensorTransforms {
  greaterEqual(t1: Tensor, t2: Tensor): Tensor;
  where(condition: ReadonlyArray<boolean>, t1: Tensor, t2: Tensor): Tensor;
}

export interface CastTensorTransforms {
  cast(t: Tensor, dtype: Tensor.NumberOrBoolType): Tensor;
}

export interface ReductionTensorTransforms {
  argMax(t: Tensor, axis: number): Tensor;
  max(t: Tensor, axis: number, keepDims: boolean): Tensor;
}

export interface TensorTransforms extends UtilityTensorCreators, BasicMathTensorTransforms, ArithmeticTensorTransforms,
                                          NormalizationTensorTransforms, SliceAndJoinTensorTransforms,
                                          PermutationTensorTransforms, ShapeTensorTransforms, LogicalTensorTransforms,
                                          CastTensorTransforms, ReductionTensorTransforms {}

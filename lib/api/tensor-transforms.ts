// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// Exposes a set of light-weight tensor(s) transformations

import {Tensor} from './tensor';

interface UtilityTensorCreators {
  /**
   * Creates a Tensor of given shape and type with elements all zeros.
   * @param shape The desired output Tensor shape
   * @param type The data type of the output Tensor
   */
  zeros(shape: ReadonlyArray<number>, type: Tensor.NumberOrBoolType): Tensor;
  /**
   * Creates an evenly spaced sequence of numbers over the given interval.
   * @param start The start value of the sequence
   * @param stop The end value of the sequence
   * @param num The number of values to generate.
   * @param type The output Tensor data type.
   */
  linspace(start: number, stop: number, num: number, type: Tensor.NumberType): Tensor;
  /**
   * Creates a 1-D Tensor filled with an arithmetic sequence.
   * @param start The start value of the sequence.
   * @param end The end value of the sequence.
   * @param step The increment value.
   * @param type The output Tensor data type.
   */
  range(start: number, stop: number, step: number, type: Tensor.NumberType): Tensor;
  /**
   * Reshapes a Tensor to 1-D Tensor
   * @param x Input Tensor
   */
  as1d(t: Tensor): Tensor;
  /**
   * Creates a scalar Tensor (rank = 0) with given value and type
   * @param value The data value of the Tensor
   * @param type Data type of the scalar Tensor
   */
  scalar(value: number, type: Tensor.NumberType): Tensor;
}

interface BasicMathTensorTransforms {
  /**
   * Calculates the exponential of the given input tensor, element-wise.
   * @param x  The input Tensor
   */
  exp(x: Tensor): Tensor;
  /**
   * Sigmoid takes one input data (Tensor) and produces one output data (Tensor) where the sigmoid function, y = 1 / (1
   * + exp(-x)), is applied to the tensor elementwise.
   * @param x The input Tensor
   */
  sigmoid(x: Tensor): Tensor;
}

interface ArithmeticTensorTransforms {
  /**
   * Performs element-wise binary addition (with Numpy-style broadcasting support).
   * @param a The first operand
   * @param b The second operand
   */
  add(a: Tensor, b: Tensor): Tensor;
  /**
   * Performs element-wise binary subtraction (with Numpy-style broadcasting support).
   * @param a The first operand
   * @param b The second operand
   */
  sub(a: Tensor, b: Tensor): Tensor;
  /**
   * Performs element-wise binary multiplication (with Numpy-style broadcasting support).
   * @param a The first operand
   * @param b The second operand
   */
  mul(a: Tensor, b: Tensor): Tensor;
  /**
   * Performs element-wise binary division (with Numpy-style broadcasting support).
   * @param a The first operand
   * @param b The second operand
   */
  div(a: Tensor, b: Tensor): Tensor;
}

interface NormalizationTensorTransforms {
  /**
   * Computes the softmax (normalized exponential) values.
   * @param x The input Tensor
   * @param axis The axis of the inputs when coerced to 2D. Default to 1;
   */
  softmax(x: Tensor, axis: number): Tensor;
}

interface SliceAndJoinTensorTransforms {
  /**
   * Concatenate a list of tensors into a single tensor.
   * @param x List of tensors for concatenation
   * @param axis Which axis to concat on
   */
  concat(x: Tensor[], axis: number): Tensor;
  stack(x: Tensor[], axis: number): Tensor;
  /**
   * Gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, and
   * concatenates them in an output tensor of rank q + (r - 1).
   * @param x Tensor of rank r >= 1
   * @param indices Tensor of int32/int64 indices, of any rank q.
   * @param axis Which axis to gather on, defaults to 0. Negative value means counting dimensions from the back.
   *     Accepted range in [-r, r-1]
   */
  gather(x: Tensor, indices: ReadonlyArray<number>, axis: number): Tensor;
  /**
   * Constructs a tensor by tiling a given tensor.
   * @param x Input tensor of any shape.
   * @param repeats 1D int64 tensor of the same length as input's dimension number, includes numbers of repeated copies
   *     along input's dimensions.
   */
  tile(x: Tensor, repeats: ReadonlyArray<number>): Tensor;
}

interface PermutationTensorTransforms {
  /**
   * Transpose the input tensor similar to numpy.transpose. For example, when perm=(1, 0, 2), given an input tensor of
   * shape (1, 2, 3), the output shape will be (2, 1, 3).
   * @param x The input Tensor
   * @param perm A list of integers. By default, reverse the dimensions, otherwise permute the axes according to the
   *     values given.
   */
  transpose(x: Tensor, perm: ReadonlyArray<number>): Tensor;
}

interface ShapeTensorTransforms {
  /**
   * Creates a Tensor with rank expanded at the specified axis
   * @param x The input Tensor
   * @param axis The dimension index where to expand.
   */
  expandDims(x: Tensor, axis: number): Tensor;
}

interface LogicalTensorTransforms {
  /**
   * Returns the tensor resulted from performing the greater logical operation elementwise on the input tensors A and B
   * (with Numpy-style broadcasting support).
   * @param a First input operand for the logical operator.
   * @param b Second input operand for the logical operator.
   */
  greaterEqual(a: Tensor, b: Tensor): Tensor;
  where(condition: ReadonlyArray<boolean>, t1: Tensor, t2: Tensor): Tensor;
}

interface CastTensorTransforms {
  /**
   * The operator casts the elements of a given input tensor to a data type specified by the 'type' argument and returns
   * an output tensor of the same size in the converted type. NOTE: casting to string is not supported yet.
   * @param x The input Tensor
   * @param type The data type to which the elements of the input tensor are cast. Strictly must be one of the types
   *     from Tensor.NumberOrBoolType
   */
  cast(x: Tensor, type: Tensor.NumberOrBoolType): Tensor;
}

interface ReductionTensorTransforms {
  /**
   * Computes the indices of the max elements of the input tensor's element along the provided axis. The resulted tensor
   * has the same rank as the input if keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced
   * dimension pruned.
   * @param x The input Tensor.
   * @param axis The axis in which to compute the arg indices.
   */
  argMax(x: Tensor, axis: number): Tensor;
  /**
   * Element-wise max of each of the input tensors. All inputs and outputs must have the same shape and data type.
   * @param x The input Tensor.
   * @param axis The axis in which to compute the arg indices.
   */
  max(x: Tensor, axis: number): Tensor;
}

export interface TensorTransforms extends UtilityTensorCreators, BasicMathTensorTransforms, ArithmeticTensorTransforms,
                                          NormalizationTensorTransforms, SliceAndJoinTensorTransforms,
                                          PermutationTensorTransforms, ShapeTensorTransforms, LogicalTensorTransforms,
                                          CastTensorTransforms, ReductionTensorTransforms {}

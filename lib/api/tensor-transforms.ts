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
   */
  linspace(start: number, stop: number, num: number): Tensor;
  /**
   * Creates a 1-D Tensor filled with an arithmetic sequence.
   * @param start The start value of the sequence.
   * @param end The end value of the sequence.
   * @param step The increment value. Optional. Default is 1.
   * @param type The output Tensor data type. Optional. Default is "float32"
   */
  range(start: number, stop: number, step?: number, type?: Tensor.NumberType): Tensor;
  /**
   * Reshapes a Tensor to 1-D Tensor
   * @param x Input Tensor
   */
  as1d(x: Tensor): Tensor;
  /**
   * Creates a scalar Tensor (rank = 0) with given value and type
   * @param value The data value of the Tensor
   * @param type Data type of the scalar Tensor. Default is float32
   */
  scalar(value: number, type?: Tensor.NumberType): Tensor;
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
  softmax(x: Tensor, axis?: number): Tensor;
}

interface SliceAndJoinTensorTransforms {
  /**
   * Concatenate a list of tensors into a single tensor.
   * @param x List of tensors for concatenation
   * @param axis Which axis to concat on.
   */
  concat(x: Tensor[], axis: number): Tensor;
  /**
   * Gather entries of the axis dimension of data (by default outer-most one as axis=0) indexed by indices, and
   * concatenates them in an output tensor of rank q + (r - 1).
   * @param x Tensor of rank r >= 1
   * @param indices Tensor of int32/int64 indices, of any rank q.
   * @param axis Which axis to gather on, defaults to 0. Negative value means counting dimensions from the back.
   *     Accepted range in [-r, r-1]
   */
  gather(x: Tensor, indices: Tensor, axis?: number): Tensor;
  /**
   * Produces a slice of the input tensor along multiple axes.
   * @param x The input Tensor to slice from
   * @param starts Starting indices of corresponding axis in "axes"
   * @param ends Ending indices (exclusive) of corresponding axis in "axes"
   * @param axes Axes that "starts" and "ends" apply to. Optional. Default = [0, 1, ..., len("starts" - 1)]
   */
  slice(x: Tensor, starts: number[], ends: number[], axes?: number[]): Tensor;
  /**
   * Stack a list of tensors into a single tensor.
   * @param x List of tensors of the same shape and type.
   * @param axis Which axis to concat on. Default is 0 (first dim).
   */
  stack(x: Tensor[], axis?: number): Tensor;

  /**
   * Constructs a tensor by tiling a given tensor.
   * @param x Input tensor of any shape.
   * @param repeats A number array of the same length as input's dimension number, specifying numbers of repeated copies
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
  transpose(x: Tensor, perm?: number[]): Tensor;
}

interface ShapeTensorTransforms {
  /**
   * Creates a Tensor with rank expanded at the specified axis.
   * @param x The input Tensor.
   * @param axis The dimension index where to expand. Optional. Defaults to 0.
   */
  expandDims(x: Tensor, axis?: number): Tensor;
  /**
   * Reshapes the input Tensor.
   * @param x The input Tensor.
   * @param shape Specified shape for output.
   */
  reshape(x: Tensor, shape: ReadonlyArray<number>): Tensor;
}

interface LogicalTensorTransforms {
  /**
   * Returns the tensor resulted from performing the greater logical operation elementwise on the input tensors A and B
   * (with Numpy-style broadcasting support).
   * @param a First input operand for the logical operator.
   * @param b Second input operand for the logical operator.
   */
  greaterEqual(a: Tensor, b: Tensor): Tensor;
  /**
   * Returns the elements, either a or b depending on the condition.
   * If the condition is true, select from a, otherwise select from b.
   * @param condition The input condition. Must be of data type bool.
   * @param a  If condition is rank 1, a may have a higher rank but its first dimension must match the size of
   *     condition.
   * @param b A tensor with the same shape and type as a.
   */
  where(condition: Tensor, a: Tensor, b: Tensor): Tensor;
}

interface CastTensorTransforms {
  /**
   * The operator casts the elements of a given input tensor to a data type specified by the 'type' argument and returns
   * an output tensor of the same size in the converted type. NOTE: casting to string is not supported yet.
   * @param x The input Tensor
   * @param type The data type to which the elements of the input tensor are cast. Strictly must be one of the types
   *     from Tensor.NumberOrBoolType
   */
  cast(x: Tensor, type: Tensor.Type): Tensor;
}

interface ReductionTensorTransforms {
  /**
   * Computes the indices of the max elements of the input tensor's element along the provided axis. The resulted tensor
   * has the same rank as the input if keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced
   * dimension pruned.
   * @param x The input Tensor.
   * @param axis The axis in which to compute the arg indices. Default is 0.
   * @param keepdims Keep the reduced dimension or not, default 1 mean keep reduced dimension.
   */
  argMax(x: Tensor, axis?: number, keepdims?: number): Tensor;
  /**
   * Computes the max of the input tensor's element along the provided axes. The resulted tensor has the same rank as
   * the input if keepdims equal 1. If keepdims equal 0, then the resulted tensor have the reduced dimension pruned.
   * @param x An input tensor.
   * @param axis A list of integers, along which to reduce. The default is to reduce over all the dimensions of the
   *     input tensor.
   * @param keepdims Keep the reduced dimension or not, default 1 mean keep reduced dimension.
   */
  reduceMax(x: Tensor, axes?: number[], keepdims?: number): Tensor;
}

export interface TensorTransforms extends UtilityTensorCreators, BasicMathTensorTransforms, ArithmeticTensorTransforms,
                                          NormalizationTensorTransforms, SliceAndJoinTensorTransforms,
                                          PermutationTensorTransforms, ShapeTensorTransforms, LogicalTensorTransforms,
                                          CastTensorTransforms, ReductionTensorTransforms {}

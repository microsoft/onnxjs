// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/**
 * represent a tensor with specified dimensions and data type.
 */
export interface Tensor {
  /**
   * get the dimensions of the tensor
   */
  readonly dims: ReadonlyArray<number>;

  /**
   * get the data type of the tensor
   */
  readonly type: Tensor.Type;

  /**
   * get the number of elements in the tensor
   */
  readonly size: number;

  /**
   * get the underlying data of the tensor
   */
  readonly data: Tensor.DataType;

  /**
   * get value of an element
   * @param indices the indices to the element
   * @returns value of the element
   */
  get(...indices: number[]): Tensor.ElementType;

  /**
   * get value of an element
   * @param indices the indices to the element
   * @returns value of the element
   */
  get(indices: ReadonlyArray<number>): Tensor.ElementType;

  /**
   * set value of an element
   * @param value the value to set
   * @param indices the indices to the element
   */
  set(value: Tensor.ElementType, ...indices: number[]): void;

  /**
   * set value of an element
   * @param value the value to set
   * @param indices the indices to the element
   */
  set(value: Tensor.ElementType, indices: ReadonlyArray<number>): void;
}

export declare namespace Tensor {
  interface DataTypeMap {
    bool: Uint8Array;
    float32: Float32Array;
    int32: Int32Array;
    string: string[];
  }

  interface ElementTypeMap {
    bool: boolean;
    float32: number;
    int32: number;
    string: string;
  }

  type DataType = DataTypeMap[Type];
  type ElementType = ElementTypeMap[Type];

  /**
   * represent the data type of a tensor
   */
  export type Type = keyof DataTypeMap;
}

export interface TensorConstructor {
  /**
   * Create a Tensor with provided data, dimension, and type
   * @param data The value of the tensor. It could a flat array or a TypedArray.
   * @param type The data type. Should match the value of the tensor, else throw
   *     TypeError exception.
   * @param dims Optional. Should match the length of the value provided. If not
   *     specified, dims will be inferred as a 1d tensor.
   */
  new(data: Tensor.DataType|boolean[]|number[], type: Tensor.Type, dims?: ReadonlyArray<number>): Tensor;
}

export interface TensorConstructor {
  // Tensor factory functions
}

export interface Tensor {
  // Tensor utilities
}

import * as TensorImpl from './tensor-impl';
export const Tensor: TensorConstructor = TensorImpl.Tensor;

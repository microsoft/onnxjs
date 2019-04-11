// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// tslint:disable:use-named-parameter

import {Tensor as InternalTensor} from '../tensor';
import {Tensor as TensorInterface} from './tensor';

import * as Utils from './tensor-impl-utils';

type DataType = TensorInterface.DataType;
type Type = TensorInterface.Type;
type ElementType = TensorInterface.ElementType;

export class Tensor implements TensorInterface {
  internalTensor: InternalTensor;
  constructor(data: DataType|number[]|boolean[], type: Type, dims?: ReadonlyArray<number>) {
    const inferredDims = dims ? dims : [data.length];
    if (data.length === 0) {
      throw new RangeError(`Tensor data should contain at least one element.`);
    }
    // convert regular arrays to typeArrays
    if (Array.isArray(data) && type !== 'string') {
      if (type === 'float32') {
        // convert number[] to Float32Array
        this.data = Float32Array.from(data as number[]);
      } else if (type === 'bool') {
        // convert boolean[] to Uint8Array
        this.data = Uint8Array.from(data as number[]);
      } else if (type === 'int32') {
        // convert number[] to Int32Array
        this.data = Int32Array.from(data as number[]);
      }
    } else {
      this.data = data as DataType;
    }

    this.dims = inferredDims;
    this.type = type;
    this.internalTensor = new InternalTensor(this.dims, this.type, undefined, undefined, this.data);
    this.size = this.internalTensor.size;
  }

  dims: ReadonlyArray<number>;
  type: Type;
  size: number;
  data: DataType;
  get(...indices: number[]): ElementType;
  get(indices: ReadonlyArray<number>): ElementType;
  get(indices?: ReadonlyArray<number>|number, ...rest: number[]): ElementType {
    let flatIndices = 0;
    let indexArray: ReadonlyArray<number> = [];
    if (typeof indices === 'number') {
      indexArray = [indices, ...rest];
    } else if (indices) {
      indexArray = indices;
    } else {
      throw new Error(`Input index array is undefined. `);
    }
    // check dims
    Utils.validateIndices(indexArray);
    if (indexArray.length !== this.dims.length) {
      throw new RangeError(`Input index array dims don't match the tensor dims.`);
    }
    // compute the flattened index
    indexArray.forEach((dim: number, idx: number) => {
      if (dim >= this.dims[idx]) {
        throw new RangeError(`Input index array dims don't match the tensor dims.`);
      }
      flatIndices += idx < indexArray.length - 1 ? dim * this.dims.slice(idx + 1).reduce((a, b) => a * b) : dim;
    });
    if (this.type === 'bool') {
      return this.data[flatIndices] === 1 ? true : false;
    }
    return this.data[flatIndices];
  }
  set(value: ElementType, ...indices: number[]): void;
  set(value: ElementType, indices: ReadonlyArray<number>): void;
  set(value: ElementType, indices?: ReadonlyArray<number>|number, ...rest: number[]) {
    Utils.matchElementType(this.type, value);
    let flatIndices = 0;
    let indexArray: ReadonlyArray<number> = [];
    if (typeof indices === 'number') {
      indexArray = [indices, ...rest];
    } else if (indices) {
      indexArray = indices;
    } else {
      throw new Error(`Input index array is undefined.`);
    }
    // check dims
    Utils.validateIndices(indexArray);
    if (indexArray.length !== this.dims.length) {
      throw new RangeError(`Input index array dims don't match the tensor dims.`);
    }
    // compute the flattened index
    indexArray.forEach((dim: number, idx: number) => {
      if (dim >= this.dims[idx]) {
        throw new RangeError(`Input index array dims don't match the tensor dims.`);
      }
      flatIndices += idx < indexArray.length - 1 ? dim * this.dims.slice(idx + 1).reduce((a, b) => a * b) : dim;
    });

    if (typeof value === 'boolean') {
      this.data[flatIndices] = value ? 1 : 0;
    } else if (typeof value === 'string') {
      this.data[flatIndices] = value;
    } else if (ArrayBuffer.isView(this.data)) {
      this.data.set([value], flatIndices);
    } else {
      throw new TypeError(`Value type is not supported. `);
    }
  }
}

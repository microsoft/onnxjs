// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor as InternalTensor} from '../tensor';
import {Tensor as TensorInterface} from './tensor';
import {Tensor as ApiTensor} from './tensor-impl';

export function toApiTensor(internalTensor: InternalTensor): ApiTensor {
  switch (internalTensor.type) {
    case 'bool':
      return new ApiTensor(new Uint8Array(internalTensor.integerData), 'bool', internalTensor.dims);
    case 'float32':
      return new ApiTensor(internalTensor.floatData as Float32Array, 'float32', internalTensor.dims);
    case 'float64':
      return new ApiTensor(new Float32Array(internalTensor.floatData), 'float32', internalTensor.dims);
    case 'string':
      return new ApiTensor(internalTensor.stringData, 'string', internalTensor.dims);
    case 'int8' || 'uint8' || 'int16' || 'uint16' || 'uint32':
      return new ApiTensor(new Int32Array(internalTensor.integerData), 'int32', internalTensor.dims);
    case 'int32':
      return new ApiTensor(internalTensor.integerData as Int32Array, 'int32', internalTensor.dims);
    default:
      throw new TypeError('Tensor type is not supported. ');
  }
}

export function toInternalTensor(apiTensor: TensorInterface): InternalTensor {
  return new InternalTensor(apiTensor.dims, apiTensor.type, undefined, undefined, apiTensor.data);
}

export function matchElementType(type: TensorInterface.Type, element: TensorInterface.ElementType) {
  switch (typeof element) {
    case 'string':
      if (type !== 'string') {
        throw new TypeError(`The new element type doesn't match the tensor data type.`);
      }
      break;
    case 'number':
      if (type !== 'float32' && type !== 'int32') {
        throw new TypeError(`The new element type doesn't match the tensor data type.`);
      }
      if (type === 'float32' && Number.isInteger(element)) {
        throw new TypeError(`The new element type doesn't match the tensor data type.`);
      }
      if (type === 'int32' && !Number.isInteger(element)) {
        throw new TypeError(`The new element type doesn't match the tensor data type.`);
      }
      break;
    case 'boolean':
      if (type !== 'bool') {
        throw new TypeError(`The new element type doesn't match the tensor data type.`);
      }
      break;
    default:
      throw new TypeError(`The new element type is not supported.`);
  }
}

export function validateIndices(indices: ReadonlyArray<number>) {
  if (indices.length < 0 || indices.length > 6) {
    throw new RangeError(`Only rank 0 to 6 is supported for tensor shape.`);
  }
  for (const n of indices) {
    if (!Number.isInteger(n)) {
      throw new TypeError(`Invalid index: ${n} is not an integer`);
    }
    if (n < 0 || n > 2147483647) {
      throw new TypeError(`Invalid index: length ${n} is not allowed`);
    }
  }
}

export class TensorTransformUtils {
  static createTypedArray(type: string, size: number): Uint8Array|Int32Array|Float32Array {
    switch (type) {
      case 'bool':
        return new Uint8Array(size);
      case 'int32':
        return new Int32Array(size);
      case 'float32':
        return new Float32Array(size);
      default:
        throw new Error('Unsupported type');
    }
  }
  static validateSameTypes(typesArray: TensorInterface.Type[]) {
    if (typesArray.length < 2) {
      throw new Error('must contain atleast 2 types to compare equality');
    }
    const baseType = typesArray[0];
    for (let i = 0; i < typesArray.length; ++i) {
      if (typesArray[i] !== baseType) {
        throw new Error('input types are ');
      }
    }
  }

  static validateEqualDims(dimsArray: Array<ReadonlyArray<number>>) {
    if (dimsArray.length < 2) {
      throw new Error('must contain atleast 2 shapes to compare equality');
    }
    const baseDims = dimsArray[0];
    const baseRank = baseDims.length;
    for (let i = 1; i < dimsArray.length; ++i) {
      const dims = dimsArray[i];
      if (dims.length !== baseRank) {
        throw new Error('rank is not the same for given inpu shapes');
      }
      for (let j = 0; j < baseRank; ++j) {
        if (baseDims[j] !== dims[j]) {
          throw new Error('input shapes are not the same');
        }
      }
    }
  }

  /**
   * Splits a given `dims` into 2 mutually exclusive `dims`
   * @param dims ReadonlyArray<number>
   * @param pick number - picks the dim along this axis and composes a new `dims`.
   * The remnants make up another `dims`
   */
  static splitDimsIntoTwo(dims: ReadonlyArray<number>, pick: number): [number[], number[]] {
    const picked: number[] = [];
    const remnants: number[] = [];

    for (let i = 0; i < dims.length; ++i) {
      if (i === pick) {
        picked.push(dims[i]);
      } else {
        remnants.push(dims[i]);
      }
    }
    return [picked, remnants];
  }
}

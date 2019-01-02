// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor as InternalTensor} from '../tensor';
import {Tensor as TensorInterface} from './tensor';
import {Tensor as ApiTensor} from './tensor-impl';

export function fromInternalTensor(internalTensor: InternalTensor): ApiTensor {
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

export function toInternalTensor(tensor: ApiTensor): InternalTensor {
  return new InternalTensor(tensor.dims, tensor.type, undefined, undefined, tensor.data);
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

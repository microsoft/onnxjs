// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import Long from 'long';
import {onnx} from 'onnx-proto';
import {onnxruntime} from './ort_format_schema';

import {Tensor} from './tensor';
import {LongUtil} from './util';

export declare namespace Attribute {
  export interface DataTypeMap {
    float: number;
    int: number;
    string: string;
    tensor: Tensor;
    floats: number[];
    ints: number[];
    strings: string[];
    tensors: Tensor[];
  }

  export type DataType = keyof DataTypeMap;
}

type ValueTypes = Attribute.DataTypeMap[Attribute.DataType];

type Value = [ValueTypes, Attribute.DataType];

export class Attribute {
  constructor(attributes: onnx.IAttributeProto[]|null|undefined|onnxruntime.experimental.fbs.Attribute[]) {
    this._attributes = new Map();
    if (attributes !== null && attributes !== undefined) {
      for (const attr of attributes) {
        if (attr instanceof onnx.AttributeProto) {
          this._attributes.set(attr.name, [Attribute.getValue(attr), Attribute.getType(attr)]);
        } else if (attr instanceof onnxruntime.experimental.fbs.Attribute) {
          this._attributes.set(attr.name()!, [Attribute.getValue(attr), Attribute.getType(attr)]);
        }
      }
      if (this._attributes.size < attributes.length) {
        throw new Error('duplicated attribute names');
      }
    }
  }

  set(key: string, type: Attribute.DataType, value: ValueTypes): void {
    this._attributes.set(key, [value, type]);
  }
  delete(key: string): void {
    this._attributes.delete(key);
  }
  getFloat(key: string, defaultValue?: Attribute.DataTypeMap['float']) {
    return this.get(key, 'float', defaultValue);
  }

  getInt(key: string, defaultValue?: Attribute.DataTypeMap['int']) {
    return this.get(key, 'int', defaultValue);
  }

  getString(key: string, defaultValue?: Attribute.DataTypeMap['string']) {
    return this.get(key, 'string', defaultValue);
  }

  getTensor(key: string, defaultValue?: Attribute.DataTypeMap['tensor']) {
    return this.get(key, 'tensor', defaultValue);
  }

  getFloats(key: string, defaultValue?: Attribute.DataTypeMap['floats']) {
    return this.get(key, 'floats', defaultValue);
  }

  getInts(key: string, defaultValue?: Attribute.DataTypeMap['ints']) {
    return this.get(key, 'ints', defaultValue);
  }

  getStrings(key: string, defaultValue?: Attribute.DataTypeMap['strings']) {
    return this.get(key, 'strings', defaultValue);
  }

  getTensors(key: string, defaultValue?: Attribute.DataTypeMap['tensors']) {
    return this.get(key, 'tensors', defaultValue);
  }

  private get<V extends Attribute.DataTypeMap[Attribute.DataType]>(
      key: string, type: Attribute.DataType, defaultValue?: V): V {
    const valueAndType = this._attributes.get(key);
    if (valueAndType === undefined) {
      if (defaultValue !== undefined) {
        return defaultValue;
      }
      throw new Error(`required attribute not found: ${key}`);
    }
    if (valueAndType[1] !== type) {
      throw new Error(`type mismatch: expected ${type} but got ${valueAndType[1]}`);
    }
    return valueAndType[0] as V;
  }

  private static getType(attr: onnx.IAttributeProto|onnxruntime.experimental.fbs.Attribute): Attribute.DataType {
    const type =
        attr instanceof onnx.AttributeProto ? (attr).type : (attr as onnxruntime.experimental.fbs.Attribute).type();
    switch (type) {
      case onnx.AttributeProto.AttributeType.FLOAT|onnxruntime.experimental.fbs.AttributeType.FLOAT:
        return 'float';
      case onnx.AttributeProto.AttributeType.INT|onnxruntime.experimental.fbs.AttributeType.INT:
        return 'int';
      case onnx.AttributeProto.AttributeType.STRING|onnxruntime.experimental.fbs.AttributeType.STRING:
        return 'string';
      case onnx.AttributeProto.AttributeType.TENSOR|onnxruntime.experimental.fbs.AttributeType.TENSOR:
        return 'tensor';
      case onnx.AttributeProto.AttributeType.FLOATS|onnxruntime.experimental.fbs.AttributeType.FLOATS:
        return 'floats';
      case onnx.AttributeProto.AttributeType.INTS|onnxruntime.experimental.fbs.AttributeType.INTS:
        return 'ints';
      case onnx.AttributeProto.AttributeType.STRINGS|onnxruntime.experimental.fbs.AttributeType.STRINGS:
        return 'strings';
      case onnx.AttributeProto.AttributeType.TENSORS|onnxruntime.experimental.fbs.AttributeType.TENSORS:
        return 'tensors';
      default:
        throw new Error(`attribute type is not supported yet: ${onnx.AttributeProto.AttributeType[type]}`);
    }
  }

  private static getValue(attr: onnx.IAttributeProto|onnxruntime.experimental.fbs.Attribute) {
    const attrType =
        attr instanceof onnx.AttributeProto ? attr.type : (attr as onnxruntime.experimental.fbs.Attribute).type();
    if (attrType === onnx.AttributeProto.AttributeType.GRAPH || attrType === onnx.AttributeProto.AttributeType.GRAPHS) {
      throw new Error('graph attribute is not supported yet');
    }

    const value = attr instanceof onnx.AttributeProto ?
        this.getValueNoCheck(attr) :
        this.getValueNoCheckFromOrtFormat(attr as onnxruntime.experimental.fbs.Attribute);

    // cast LONG to number
    if (attrType === onnx.AttributeProto.AttributeType.INT && LongUtil.isLong(value)) {
      return LongUtil.longToNumber(value as Long | flatbuffers.Long);
    }

    // cast LONG[] to number[]
    if (attrType === onnx.AttributeProto.AttributeType.INTS) {
      const arr = (value as Array<number|Long|flatbuffers.Long>);
      const numberValue: number[] = new Array<number>(arr.length);

      for (let i = 0; i < arr.length; i++) {
        const maybeLong = arr[i];
        numberValue[i] = LongUtil.longToNumber(maybeLong);
      }

      return numberValue;
    }

    // cast onnx.TensorProto to onnxjs.Tensor
    if (attrType === onnx.AttributeProto.AttributeType.TENSOR) {
      return attr instanceof onnx.AttributeProto ? Tensor.fromProto(value as onnx.ITensorProto) :
                                                   Tensor.fromOrtTensor(value as onnxruntime.experimental.fbs.Tensor);
    }

    // cast onnx.TensorProto[] to onnxjs.Tensor[]
    if (attrType === onnx.AttributeProto.AttributeType.TENSORS) {
      if (attr instanceof onnx.AttributeProto) {
        const tensorProtos = value as onnx.ITensorProto[];
        return tensorProtos.map(value => Tensor.fromProto(value));
      } else if (attr instanceof onnxruntime.experimental.fbs.Attribute) {
        const tensorProtos = value as onnxruntime.experimental.fbs.Tensor[];
        return tensorProtos.map(value => Tensor.fromOrtTensor(value));
      }
    }

    // cast Uint8Array to string
    if (attrType === onnx.AttributeProto.AttributeType.STRING) {
      // string in onnx attribute is of uint8array type
      if (attr instanceof onnx.AttributeProto) {
        const utf8String = value as Uint8Array;
        return Buffer.from(utf8String.buffer, utf8String.byteOffset, utf8String.byteLength).toString();
      }
    }

    // cast Uint8Array[] to string[]
    if (attrType === onnx.AttributeProto.AttributeType.STRINGS) {
      // strings in onnx attribute is uint8array[]
      if (attr instanceof onnx.AttributeProto) {
        const utf8Strings = value as Uint8Array[];
        return utf8Strings.map(
            utf8String => Buffer.from(utf8String.buffer, utf8String.byteOffset, utf8String.byteLength).toString());
      }
    }

    return value as ValueTypes;
  }

  private static getValueNoCheck(attr: onnx.IAttributeProto) {
    switch (attr.type!) {
      case onnx.AttributeProto.AttributeType.FLOAT:
        return attr.f;
      case onnx.AttributeProto.AttributeType.INT:
        return attr.i;
      case onnx.AttributeProto.AttributeType.STRING:
        return attr.s;
      case onnx.AttributeProto.AttributeType.TENSOR:
        return attr.t;
      case onnx.AttributeProto.AttributeType.GRAPH:
        return attr.g;
      case onnx.AttributeProto.AttributeType.FLOATS:
        return attr.floats;
      case onnx.AttributeProto.AttributeType.INTS:
        return attr.ints;
      case onnx.AttributeProto.AttributeType.STRINGS:
        return attr.strings;
      case onnx.AttributeProto.AttributeType.TENSORS:
        return attr.tensors;
      case onnx.AttributeProto.AttributeType.GRAPHS:
        return attr.graphs;
      default:
        throw new Error(`unsupported attribute type: ${onnx.AttributeProto.AttributeType[attr.type!]}`);
    }
  }

  private static getValueNoCheckFromOrtFormat(attr: onnxruntime.experimental.fbs.Attribute) {
    switch (attr.type()) {
      case onnxruntime.experimental.fbs.AttributeType.FLOAT:
        return attr.f();
      case onnxruntime.experimental.fbs.AttributeType.INT:
        return attr.i();
      case onnxruntime.experimental.fbs.AttributeType.STRING:
        return attr.s();
      case onnxruntime.experimental.fbs.AttributeType.TENSOR:
        return attr.t();
      case onnxruntime.experimental.fbs.AttributeType.GRAPH:
        return attr.g();
      case onnxruntime.experimental.fbs.AttributeType.FLOATS:
        return attr.floatsArray();
      case onnxruntime.experimental.fbs.AttributeType.INTS:
        const ints = [];
        for (let i = 0; i < attr.intsLength(); i++) {
          ints.push(attr.ints(i)!);
        }
        return ints;
      case onnxruntime.experimental.fbs.AttributeType.STRINGS:
        const strings = [];
        for (let i = 0; i < attr.stringsLength(); i++) {
          strings.push(attr.strings(i));
        }
        return strings;
      case onnxruntime.experimental.fbs.AttributeType.TENSORS:
        const tensors = [];
        for (let i = 0; i < attr.tensorsLength(); i++) {
          tensors.push(attr.tensors(i)!);
        }
        return tensors;
      case onnxruntime.experimental.fbs.AttributeType.GRAPHS:
        // TODO: Subgraph not supported yet.
        const graphs = [];
        for (let i = 0; i < attr.graphsLength(); i++) {
          graphs.push(attr.graphs(i)!);
        }
        return graphs;
      default:
        throw new Error(`unsupported attribute type: ${onnxruntime.experimental.fbs.AttributeType[attr.type()]}`);
    }
  }

  protected _attributes: Map<string, Value>;
}

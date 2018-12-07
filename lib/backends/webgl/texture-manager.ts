// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger, Profiler} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';

import {TextureData, TextureLayout} from './texture-data';
import {Encoder} from './texture-data-encoder';
import {TextureLayoutStrategy} from './texture-layout-strategy';
import {WebGLContext} from './webgl-context';

/**
 * Texture Manager is the mainly responsible for caching Textures
 * Textures are cached in 2 levels:
 *   1. the texures which are associated with a dataId (from Tensor)
 *    Caching these is crucial to performance. These are In-use Textures
 *   2. textures which are not in use by any current ProgramInfo/Tensor
 *     These are called Free Textures
 * TextureManager is also used to help creating textures. For this it
 * uses WebGLContext and TextureLayoutStrategy
 */
export class TextureManager {
  glContext: WebGLContext;
  free: Map<string, WebGLTexture[]>;
  gl: WebGLRenderingContext;
  layoutStrategy: TextureLayoutStrategy;
  profiler: Readonly<Profiler>;

  constructor(context: WebGLContext, layoutStrategy: TextureLayoutStrategy, profiler: Readonly<Profiler>) {
    this.glContext = context;
    this.gl = context.gl;
    this.free = new Map();
    this.layoutStrategy = layoutStrategy;
    this.profiler = profiler;
  }
  createTextureFromLayout(dataType: Tensor.DataType, layout: TextureLayout, data?: Tensor.NumberType) {
    let texture: WebGLTexture;
    const textureDataType = this.toEncoderType(dataType);
    const size = `${layout.width}-${layout.height}`;
    const textureList = this.free.get(size);
    if (!textureList || textureList.length === 0) {
      Logger.verbose('TextureManager', `No cached texture; Creating new of size ${size}`);
      texture = this.glContext.allocateTexture(
          layout.width, layout.height, textureDataType, layout.channels, this.toTextureData(dataType, data));
    } else {
      Logger.verbose('TextureManager', `Found a texture in cache of size ${size}`);
      texture = textureList.shift()!;
      if (data) {
        this.glContext.updateTexture(
            texture, layout.width, layout.height, 'float' /*this.toEncoderType(dataType)*/, layout.channels,
            this.toTextureData(dataType, data)!);
      }
    }
    return {...layout, dataType, texture, arrayType: textureDataType};
  }
  createTexture(
      dataType: Tensor.DataType, shape: ReadonlyArray<number>, strides?: ReadonlyArray<number>,
      data?: Tensor.NumberType, channels?: number, width?: number, height?: number,
      unpackedShape?: ReadonlyArray<number>): TextureData {
    return this.profiler.event('backend', 'TextureManager.createTexture', () => {
      if (!width || !height) {
        [width, height] = this.layoutStrategy.computeTextureWH(shape);
      }
      if (!strides) {
        strides = ShapeUtil.computeStrides(shape);
      }
      if (!channels || channels === 1) {
        channels = 1;
        unpackedShape = shape;
      }
      if (channels > 1 && !unpackedShape) {
        throw new Error('unpacked shape is needed when the number of channels is > 1');
      }
      const layout = {width, height, channels, shape, strides, unpackedShape: unpackedShape!};
      return this.createTextureFromLayout(dataType, layout, data);
    });
  }
  readTexture(td: TextureData, dataType: Tensor.DataType, channels?: number): Tensor.NumberType {
    if (!channels) {
      channels = 1;
    }
    return this.profiler.event('backend', 'TextureManager.readTexture', () => {
      const dataSize = td.shape.reduce((a, b) => a * b) * channels!;
      const data = this.glContext.readTexture(
          td.texture, td.width, td.height, dataSize, this.toEncoderType(dataType), channels!);
      return this.toTensorData(dataType, data);
    });
  }
  saveTexture(texture: WebGLTexture, dims: ReadonlyArray<number>): void {
    return this.profiler.event('backend', 'TextureManager.saveTexture', () => {
      const size = `${dims[0]}-${dims[1]}`;
      Logger.verbose('TextureManager', `caching texture of size ${size}`);
      let textureList = this.free.get(size);
      if (!textureList) {
        textureList = [];
      }
      textureList.push(texture);
      this.free.set(size, textureList);
    });
  }
  createPaddedTexture(inputTextureData: TextureData, outputLayout: TextureLayout): TextureData {
    const inputTexture = inputTextureData.texture;
    const [inputWidth, inputHeight] = [inputTextureData.width, inputTextureData.height];
    const outputTD = this.createTextureFromLayout(inputTextureData.dataType, outputLayout);

    const gl = this.gl;
    this.glContext.attachFramebuffer(inputTexture, inputWidth, inputHeight);
    gl.bindTexture(gl.TEXTURE_2D, outputTD.texture);
    gl.copyTexSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 0, 0, inputWidth, inputHeight);

    return outputTD;
  }
  toTensorData(dataType: Tensor.DataType, data: Encoder.DataArrayType): Tensor.NumberType {
    return (data.constructor === Float32Array) ? data as Float32Array : new Float32Array(data);
    /*
    switch (dataType) {
      case 'int16':
        return new Int16Array(data);
      case 'int32':
        return new Int32Array(data);
      case 'int8':
        return new Int8Array(data);
      case 'uint16':
        return new Uint16Array(data);
      case 'uint32':
        return data as Uint32Array;
      case 'uint8':
      case 'bool':
        return data as Uint8Array;
      case 'float32':
        return data as Float32Array;
      case 'float64':
        return new Float64Array(data);
      default:
        throw new Error(`TensorData type ${dataType} is not supported`);
    }
    */
  }
  toTextureData(dataType: Tensor.DataType, data: Tensor.NumberType|undefined): Encoder.DataArrayType|undefined {
    if (!data) {
      return undefined;
    }
    return (data.constructor === Float32Array) ? data as Float32Array : new Float32Array(data);
    /*
    switch (dataType) {
      case 'int16':
      case 'int32':
      case 'uint16':
      case 'uint32':
        return (data.constructor === Uint32Array) ? data as Uint32Array : new Uint32Array(data);
      case 'int8':
      case 'uint8':
      case 'bool':
        return (data.constructor === Uint8Array) ? data as Uint8Array : new Uint8Array(data);
      case 'float32':
      case 'float64':
        return (data.constructor === Float32Array) ? data as Float32Array : new Float32Array(data);
      default:
        throw new Error(`TensorData type ${dataType} is not supported`);
    }
    */
  }
  toEncoderType(dataType: Tensor.DataType): Encoder.DataType {
    return 'float';
    // switch (dataType) {
    //   case 'int16':
    //   case 'int32':
    //   case 'uint16':
    //   case 'uint32':
    //     return 'int';
    //   case 'uint8':
    //   case 'bool':
    //     return 'byte';
    //   case 'float32':
    //   case 'float64':
    //     return 'float';
    //   default:
    //     throw new Error(`TensorData type ${dataType} is not supported`);
    // }
  }
  dispose(): void {
    this.free.forEach(value => value.forEach(t => this.glContext.deleteTexture(t)));
  }
}

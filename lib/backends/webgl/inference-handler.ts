// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Logger} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';
import {WebGLBackend} from '../backend-webgl';

import {ProgramManager} from './program-manager';
import {WebGLSessionHandler} from './session-handler';
import {TextureData, TextureLayout} from './texture-data';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {TextureManager} from './texture-manager';
import {getPackedShape} from './utils';

/**
 * GlInferencContext is reponsible for mapping from Tensors to TextureData
 * and back
 * Throughout WebGL backend operations TextureData is used as the data carrier
 */
export class WebGLInferenceHandler implements InferenceHandler {
  textureManager: TextureManager;
  programManager: ProgramManager;
  private tensorToTexture: Map<Tensor, TextureData>;
  private textureToTensor: Map<TextureData, Tensor>;
  constructor(public backend: WebGLBackend, public session: WebGLSessionHandler) {
    this.textureManager = session.textureManager;
    this.programManager = session.programManager;
    this.tensorToTexture = new Map();
    this.textureToTensor = new Map();
  }
  protected lookupTextureData(tensor: Tensor): TextureData|undefined {
    return this.session.isInitializer(tensor) ? this.session.getTextureData(tensor) : this.tensorToTexture.get(tensor);
  }
  getOrCreate(tensor: Tensor, layout?: TextureLayout): TextureData {
    let td = this.lookupTextureData(tensor);
    if (!td) {
      Logger.verbose('InferenceHandler', `Creating new TextureData for dims: [${tensor.dims}]`);
      if (!layout) {
        layout = this.createBasicTextureLayout(tensor.dims.slice());
      }
      td = this.createTextureDataFromLayout(layout, tensor.type, tensor.numberData);
      this.setTextureData(tensor, td);
    } else {
      Logger.verbose('InferenceHandler', `Retrieving TextureData from cache: [${tensor.dims}]`);
    }
    return td;
  }
  getTextureData(tensor: Tensor): TextureData|undefined {
    return this.lookupTextureData(tensor);
  }
  setTextureData(tensor: Tensor, td: TextureData): void {
    if (this.session.isInitializer(tensor)) {
      this.session.setTextureData(tensor, td);
      return;
    }
    this.tensorToTexture.set(tensor, td);
    this.textureToTensor.set(td, tensor);
  }
  getTensor(td: TextureData): Tensor {
    let tensor: Tensor|undefined;
    tensor = this.textureToTensor.get(td);
    if (!tensor) {
      Logger.verbose('InferenceHandler', `Creating new Tensor from texture data: [${td.unpackedShape}]`);
      /**
       * We're creating a Tensor without converting data from Texture onto CPU
       * Instead we're passing a closure which is only executed if Tesor.data is accessed
       * This allows for the execution of the graph without paying the penalty of
       * data movement from GPU to CPU
       */
      tensor = new Tensor(td.unpackedShape, td.dataType, (id: Tensor.Id) => {
        const values = this.textureManager.readTexture(td, td.dataType, td.channels);
        return values;
      });
      this.setTextureData(tensor, td);
    } else {
      Logger.verbose('InferenceHandler', `Retrieving Tensor from cache for:[${td.unpackedShape}]`);
    }
    return tensor;
  }
  getOrCreateTextureLayout(tensor: Tensor, channels = 1, unpackedShape?: ReadonlyArray<number>): TextureLayout {
    const td = this.getTextureData(tensor);
    if (td) {
      return td;
    }
    return this.createBasicTextureLayout(
        channels === 1 ? tensor.dims.slice() : getPackedShape(tensor.dims.slice()), channels, unpackedShape);
  }
  dispose(): void {
    this.tensorToTexture.forEach(td => this.textureManager.saveTexture(td.texture, [td.width, td.height]));
    this.tensorToTexture = new Map();
    this.textureToTensor = new Map();
  }
  createTextureData(
      dataType: Tensor.DataType, shape: ReadonlyArray<number>, strides?: ReadonlyArray<number>,
      data?: Tensor.NumberType, channels?: number, width?: number, height?: number): TextureData {
    Logger.verbose('InferenceHandler', `Creating TextureData: shape:[${shape}], channels:${channels ? channels : 1}`);
    const td = this.textureManager.createTexture(dataType, shape, strides, data, channels, width, height);
    return td;
  }
  createTextureDataFromLayout(layout: TextureLayout, dataType: Tensor.DataType, data?: Tensor.NumberType): TextureData {
    Logger.verbose('InferenceHandler', `Creating TextureData: layout:[${JSON.stringify(layout)}]`);
    const td = this.textureManager.createTextureFromLayout(dataType, layout, data);
    return td;
  }
  createBasicTextureLayout(
      shape: ReadonlyArray<number>, channels = 1, unpackedShape?: ReadonlyArray<number>,
      prefs?: WidthHeightPrefs): TextureLayout {
    const [width, height] = this.session.layoutStrategy.computeTextureWH(shape, prefs);
    if (channels === 1) {
      unpackedShape = shape;
    } else if (!unpackedShape) {
      throw new Error('Unpacked shape is needed when using channels > 1');
    }
    return {
      width,
      height,
      channels: channels ? channels : 1,
      shape,
      strides: ShapeUtil.computeStrides(shape),
      unpackedShape
    };
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Logger} from '../../instrument';
import {Tensor} from '../../tensor';
import {ShapeUtil} from '../../util';

import {WebGLUint8Encode} from './ops/uint8-encode';
import {ProgramManager} from './program-manager';
import {WebGLSessionHandler} from './session-handler';
import {Encoder} from './texture-data-encoder';
import {TextureHelper} from './texture-helper';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {TextureData, TextureLayout, WebGLOperator} from './types';
import {getPackedShape} from './utils';

export class WebGLInferenceHandler implements InferenceHandler {
  textureHelper: TextureHelper;
  programManager: ProgramManager;
  private tensorToTexture: Map<Tensor, TextureData>;
  private textureToTensor: Map<TextureData, Tensor>;
  constructor(public session: WebGLSessionHandler) {
    this.textureHelper = session.textureHelper;
    this.programManager = session.programManager;
    this.tensorToTexture = new Map();
    this.textureToTensor = new Map();
  }

  run(op: WebGLOperator, inputs: Tensor[]): Tensor[] {
    let artifact = this.programManager.getArtifact(op);
    if (!artifact) {
      const programInfo = op.createProgramInfo(this, inputs);
      artifact = this.programManager.build(programInfo);
      this.programManager.setArtifact(op, artifact);
    }
    const runData = op.createRunData(this, artifact.programInfo, inputs);
    this.programManager.run(artifact, runData);
    return [this.getTensor(runData.outputTextureData)];
  }

  /**
   * Create a TextureData object from a tensor.
   * Usage = Encoder.Usage.UploadOnly.
   * If a related texture data is found in cache, returns it;
   * Otherwise:
   *   Creates a new texture layout if not provided;
   *   Creates WebGLTexture with the layout;
   *   Upload tensor data to the texture;
   *   Creates a texture data object associated with the given tensor.
   * @param tensor the tensor with data to upload
   */
  getOrCreateTextureData(tensor: Tensor, layout?: TextureLayout): TextureData {
    let td = this.getTextureData(tensor);
    if (!td) {
      Logger.verbose('InferenceHandler', `Creating new TextureData for dims: [${tensor.dims}]`);
      if (!layout) {
        layout = this.createTextureLayoutFromShape(tensor.dims.slice());
      }
      // graph inputs or initializers
      td = this.createTextureDataFromLayout(layout, tensor.type, tensor.numberData, Encoder.Usage.UploadOnly);
      this.setTextureData(tensor, td);
    } else {
      Logger.verbose('InferenceHandler', `Retrieving TextureData from cache: [${tensor.dims}]`);
    }
    return td;
  }

  /**
   * Create a TextureData object from the given data type and texture layout.
   * Usage = Encoder.Usage.Default.
   * @param dataType the tensor data type
   */
  createTextureDataFromLayout(
      layout: TextureLayout, dataType: Tensor.DataType, data?: Tensor.NumberType, usage?: Encoder.Usage): TextureData {
    Logger.verbose('InferenceHandler', `Creating TextureData: layout:[${JSON.stringify(layout)}]`);
    return {...layout, dataType, texture: this.textureHelper.createTextureFromLayout(dataType, layout, data, usage)};
  }

  getTextureData(tensor: Tensor): TextureData|undefined {
    return this.session.isInitializer(tensor) ? this.session.getTextureData(tensor) : this.tensorToTexture.get(tensor);
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
        return this.readTexture(td);
      });
      this.setTextureData(tensor, td);
    } else {
      Logger.verbose('InferenceHandler', `Retrieving Tensor from cache for:[${td.unpackedShape}]`);
    }
    return tensor;
  }

  /**
   * Create a TextureLayout object from a tensor. If a related texture data is found, returns the cached texture layout.
   */
  getOrCreateTextureLayout(tensor: Tensor, channels = 1, unpackedShape?: ReadonlyArray<number>): TextureLayout {
    const td = this.getTextureData(tensor);
    if (td) {
      return td;
    }
    return this.createTextureLayoutFromShape(
        channels === 1 ? tensor.dims.slice() : getPackedShape(tensor.dims.slice()), channels, unpackedShape);
  }
  /**
   * Create a TextureLayout object from shape.
   */
  createTextureLayoutFromShape(
      shape: ReadonlyArray<number>, channels = 1, unpackedShape?: ReadonlyArray<number>,
      prefs?: WidthHeightPrefs): TextureLayout {
    const [width, height] = this.session.layoutStrategy.computeTextureWH(shape, prefs);
    let inferredDims = shape;
    if (shape.length === 0) {
      inferredDims = [1];
    }
    if (channels === 1) {
      // unpackedShape will take `shape` and not `inferredDims` so as to create a scalar Tensor if need be
      unpackedShape = shape;
    } else if (!unpackedShape) {
      throw new Error('Unpacked shape is needed when using channels > 1');
    }
    return {
      width,
      height,
      channels: channels ? channels : 1,
      shape: inferredDims,
      strides: ShapeUtil.computeStrides(inferredDims),
      unpackedShape
    };
  }

  dispose(): void {
    this.textureHelper.clearActiveTextures();
    this.tensorToTexture.forEach(td => this.textureHelper.releaseTexture(td));
    this.tensorToTexture = new Map();
    this.textureToTensor = new Map();
  }

  readTexture(textureData: TextureData): Tensor.NumberType {
    if (this.session.backend.forceUint8Reads) {
      const op = new WebGLUint8Encode();
      const uint8TD = op.runInternal(this, textureData);
      return this.textureHelper.readUint8TextureAsFloat(uint8TD);
    }
    const values = this.textureHelper.readTexture(textureData, textureData.dataType, textureData.channels);
    return values;
  }
}

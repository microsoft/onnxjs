// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../tensor';
import {Encoder} from './texture-data-encoder';

/**
 * Layout info is used for mapping n-dimensional array to 2D textures
 * The layout is created by the TextureLayoutStrategy based on
 * the Tensor's dimensions and strides
 */

export interface TextureLayout {
  width: number;
  height: number;
  channels: number;
  shape: number[];
  strides: number[];
  unpackedShape: number[];
}
export interface TextureData extends TextureLayout {
  dataType: Tensor.DataType;
  texture: WebGLTexture;
  arrayType: Encoder.DataType;
}

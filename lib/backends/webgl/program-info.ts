// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {GlslPositionalFunction, GlslValueFunction} from './glsl-definitions';
import {TextureLayout} from './texture-data';

export interface ProgramInfo {
  inputLayouts: TextureLayout[];
  shaderSource: string;
  outputLayout: TextureLayout;
  hasMain: boolean;
  positionalSubFunctions?: GlslPositionalFunction[];
  valueSubFunctions?: GlslValueFunction[];
  blockSize?: [number, number];
  params?: {[name: string]: number|number[]|string};
}

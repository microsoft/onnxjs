// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../tensor';

import {WebGLInferenceHandler} from './inference-handler';
import {WebGLContext} from './webgl-context';

/**
 * Represent an operator instance that can run in WebGL backend
 */
export interface WebGLOperator {
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo;
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData;
}

/**
 * Layout info is used for mapping n-dimensional array to 2D textures
 * The layout is created by the TextureLayoutStrategy based on
 * the Tensor's dimensions and strides
 */
export interface TextureLayout {
  width: number;
  height: number;
  /**
   * specify the number of value that encoded in a single pixel
   */
  channels: 1|2|3|4;
  /**
   * the normalized shape
   */
  shape: ReadonlyArray<number>;
  /**
   * the stride of each dimensions, calculated according to shape
   */
  strides: ReadonlyArray<number>;
  /**
   * the original shape(dims) of the corresponding tensor
   */
  unpackedShape: ReadonlyArray<number>;
}
export interface TextureData extends TextureLayout {
  tensor: Tensor;
  texture: WebGLTexture;
}

/**
 * A set of data that represent a shader program
 */
export interface ProgramInfo {
  /**
   * texture layouts for each input
   */
  inputLayouts: TextureLayout[];
  /**
   * texture layout for output
   */
  outputLayout: TextureLayout;
  shaderSource: string;
  hasMain: boolean;
  params?: {[name: string]: number|number[]|string};
}

/**
 * Information extracted from Shader source to help with binding later
 */
export interface VariableInfo {
  type: string;
  name: string;
  isVec: boolean;
  arraySuffix?: string;
}

/**
 * LocationInfo contains a mappig from a variable name (inside shader)
 * to its "location" in the compiled program
 */
export interface LocationInfo {
  variable: VariableInfo;
  location: WebGLUniformLocation|number;
}

/**
 * Artifact is the result of compilation
 * It does not contain input of output data
 * However anything that could be run as a "program"
 */
export interface Artifact {
  programInfo: ProgramInfo;
  program: WebGLProgram;
  uniformLocations: {[name: string]: LocationInfo};
  attribLocations: {[name: string]: LocationInfo};
}

export interface UniformData {
  [name: string]: number|number[];
}

/**
 * RunData contains all inputs that required to run a "program"
 */
export interface RunData {
  inputTextureDatas: TextureData[];
  outputTextureData: TextureData;
  uniformData: UniformData;
  preRun?: (glContext: WebGLContext, artifact: Artifact) => void;
  postRun?: (glContext: WebGLContext, artifact: Artifact) => void;
  draw?: (glContext: WebGLContext, artifact: Artifact) => void;
}

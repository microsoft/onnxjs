// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Operator} from '../../operators';
import {Tensor} from '../../tensor';

import {GlslPositionalFunction, GlslValueFunction} from './glsl-definitions';
import {WebGLInferenceHandler} from './inference-handler';
import {WebGLContext} from './webgl-context';

export interface PositionalSubOperator extends Operator {
  getPositionalFunction(handler: WebGLInferenceHandler, inputShape: ReadonlyArray<number>, name?: string):
      GlslPositionalFunction;
}
export interface WebGLRunnable extends Operator {
  addPositionalSub(positionalSubOperator: PositionalSubOperator): void;
  positionalSubs: PositionalSubOperator[];
}
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
  channels: number;
  shape: ReadonlyArray<number>;
  strides: ReadonlyArray<number>;
  unpackedShape: ReadonlyArray<number>;
}
export interface TextureData extends TextureLayout {
  dataType: Tensor.DataType;
  texture: WebGLTexture;
}

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

/**
 * Information extracted from Shader source to help with binding later
 */
export class VariableInfo {
  type: string;
  name: string;
  isVec: boolean;
  arraySuffix?: string;
}
/**
 * LocationInfo contains a mappig from a variable name (inside shader)
 * to its "location" in the compiled program
 */
export class LocationInfo {
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

export class RunData {
  inputTextureDatas: TextureData[];
  outputTextureData: TextureData;
  uniformData: UniformData;
  preRun?: (glContext: WebGLContext, artifact: Artifact) => void;
  postRun?: (glContext: WebGLContext, artifact: Artifact) => void;
  draw?: (glContext: WebGLContext, artifact: Artifact) => void;
}

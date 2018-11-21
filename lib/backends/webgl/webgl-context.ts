// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {DataEncoder, Encoder} from './texture-data-encoder';

/**
 * Abstraction and wrapper around WebGLRenderingContext and its operations
 */
export interface WebGLContext {
  canvas: HTMLCanvasElement;
  gl: WebGLRenderingContext;
  contextAttributes: WebGLContextAttributes;
  maxTextureSize: number;
  textureFloatExtension: OES_texture_float|null;
  textureHalfFloatExtension: OES_texture_half_float|null;
  createDefaultGeometry(): Float32Array;
  createVertexbuffer(): WebGLBuffer;
  createFramebuffer(): WebGLFramebuffer;
  allocateTexture(
      width: number, height: number, dataType: Encoder.DataType, channels: number,
      data?: Encoder.DataArrayType): WebGLTexture;
  updateTexture(
      texture: WebGLTexture, width: number, height: number, dataType: Encoder.DataType, channels: number,
      data: Encoder.DataArrayType): void;
  attachFramebuffer(texture: WebGLTexture, width: number, height: number): void;
  readTexture(
      texture: WebGLTexture, width: number, height: number, dataSize: number, dataType: Encoder.DataType,
      channels: number): Encoder.DataArrayType;
  isFramebufferReady(): boolean;
  getActiveTexture(): string;
  getTextureBinding(): WebGLTexture;
  getFramebufferBinding(): WebGLFramebuffer;
  setVertexAttributes(positionHandle: number, textureCoordHandle: number): void;
  compileShader(shaderSource: string, shaderType: number): WebGLShader;
  createProgram(vertexShader: WebGLShader, fragShader: WebGLShader): WebGLProgram;
  deleteShader(shader: WebGLShader): void;
  bindTextureToUniform(texture: WebGLTexture, position: number, uniformHandle: WebGLUniformLocation): void;
  draw(): void;
  checkError(): void;
  deleteTexture(texture: WebGLTexture): void;
  deleteProgram(program: WebGLProgram): void;
  getEncoder(dataType: Encoder.DataType, channels: number): DataEncoder;
  dispose(): void;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {DataEncoder, Encoder, RGBAFloat32DataEncoder, Uint8DataEncoder} from './texture-data-encoder';
import {Disposable} from './utils';
import {WebGLContext} from './webgl-context';

/**
 * Basic implementation of WebGLContext
 */
export abstract class BaseWebGLContext implements WebGLContext, Disposable {
  canvas: HTMLCanvasElement;
  gl: WebGLRenderingContext;
  contextAttributes: WebGLContextAttributes;
  maxTextureSize: number;
  textureFloatExtension: OES_texture_float|null;
  textureHalfFloatExtension: OES_texture_half_float|null;
  vertexbuffer: WebGLBuffer;
  framebuffer: WebGLFramebuffer;
  maxCombinedTextureImageUnits: number;
  maxTextureImageUnits: number;
  maxCubeMapTextureSize: number;
  shadingLanguageVersion: string;
  webglVendor: string;
  webglVersion: string;
  disposed: boolean;
  frameBufferBound = false;

  init(): void {
    this.getExtensions();
    this.vertexbuffer = this.createVertexbuffer();
    this.framebuffer = this.createFramebuffer();
    this.queryVitalParameters();
  }
  dispose(): void {
    if (this.disposed) {
      return;
    }
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.deleteFramebuffer(this.framebuffer);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.deleteBuffer(this.vertexbuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    gl.finish();
    this.disposed = true;
  }
  createDefaultGeometry(): Float32Array {
    // Sets of x,y,z(=0),s,t coordinates.
    return new Float32Array([
      -1.0, 1.0,  0.0, 0.0, 1.0,  // upper left
      -1.0, -1.0, 0.0, 0.0, 0.0,  // lower left
      1.0,  1.0,  0.0, 1.0, 1.0,  // upper right
      1.0,  -1.0, 0.0, 1.0, 0.0
    ]);  // lower right
  }
  createVertexbuffer(): WebGLBuffer {
    const gl = this.gl;
    const buffer = gl.createBuffer();
    if (!buffer) {
      throw new Error('createBuffer() returned null');
    }
    const geometry = this.createDefaultGeometry();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, geometry, gl.STATIC_DRAW);
    this.checkError();
    return buffer;
  }
  createFramebuffer(): WebGLFramebuffer {
    const fb = this.gl.createFramebuffer();
    if (!fb) {
      throw new Error('createFramebuffer returned null');
    }
    return fb;
  }
  allocateTexture(
      width: number, height: number, dataType: Encoder.DataType, channels: number,
      data?: Encoder.DataArrayType): WebGLTexture {
    const gl = this.gl;
    if (!channels) {
      channels = 1;
    }
    // create the texture
    const texture = gl.createTexture();
    // bind the texture so the following methods effect this texture.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const encoder = this.getEncoder(dataType, channels);
    const buffer = data ? encoder.encode(data, width * height) : null;
    gl.texImage2D(
        gl.TEXTURE_2D,
        0,  // Level of detail.
        encoder.internalFormat, width, height,
        0,  // Always 0 in OpenGL ES.
        encoder.format, encoder.channelType, buffer);
    this.checkError();
    return texture as WebGLTexture;
  }
  updateTexture(
      texture: WebGLTexture, width: number, height: number, dataType: Encoder.DataType, channels: number,
      data: Encoder.DataArrayType): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    const encoder = this.getEncoder(dataType, channels);
    const buffer = encoder.encode(data, width * height);
    gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,  // level
        0,  // xoffset
        0,  // yoffset
        width, height, encoder.format, encoder.channelType, buffer);
    this.checkError();
  }
  attachFramebuffer(texture: WebGLTexture, width: number, height: number): void {
    const gl = this.gl;
    // Make it the target for framebuffer operations - including rendering.
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.framebuffer);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
        0);  // 0, we aren't using MIPMAPs
    this.checkError();
    gl.viewport(0, 0, width, height);
  }
  readTexture(
      texture: WebGLTexture, width: number, height: number, dataSize: number, dataType: Encoder.DataType,
      channels: number): Encoder.DataArrayType {
    const gl = this.gl;
    if (!channels) {
      channels = 1;
    }
    if (!this.frameBufferBound) {
      this.attachFramebuffer(texture, width, height);
    }
    const encoder = this.getEncoder(dataType, channels);
    const buffer = encoder.allocate(width * height);
    // bind texture to framebuffer
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.framebufferTexture2D(
        gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture,
        0);  // 0, we aren't using MIPMAPs
    // TODO: Check if framebuffer is ready
    gl.readPixels(0, 0, width, height, gl.RGBA, encoder.channelType, buffer);
    this.checkError();
    // unbind FB
    return encoder.decode(buffer, dataSize);
  }
  isFramebufferReady(): boolean {
    // TODO: Implement logic to check if the framebuffer is ready
    return true;
  }
  getActiveTexture(): string {
    const gl = this.gl;
    const n = gl.getParameter(this.gl.ACTIVE_TEXTURE);
    return `TEXTURE${(n - gl.TEXTURE0)}`;
  }
  getTextureBinding(): WebGLTexture {
    return this.gl.getParameter(this.gl.TEXTURE_BINDING_2D);
  }
  getFramebufferBinding(): WebGLFramebuffer {
    return this.gl.getParameter(this.gl.FRAMEBUFFER_BINDING);
  }
  checkError(): void {
    // TODO: Implement WebGL error checks
  }
  setVertexAttributes(positionHandle: number, textureCoordHandle: number): void {
    const gl = this.gl;
    gl.vertexAttribPointer(positionHandle, 3, gl.FLOAT, false, 20, 0);
    gl.vertexAttribPointer(textureCoordHandle, 2, gl.FLOAT, false, 20, 12);
    gl.enableVertexAttribArray(positionHandle);
    gl.enableVertexAttribArray(textureCoordHandle);
    this.checkError();
  }
  createProgram(
      vertexShader: WebGLShader,
      fragShader: WebGLShader,
      ): WebGLProgram {
    const gl = this.gl;
    const program = gl.createProgram()!;

    // the program consists of our shaders
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragShader);
    gl.linkProgram(program);
    return program;
  }
  compileShader(shaderSource: string, shaderType: number): WebGLShader {
    const gl = this.gl;
    const shader = gl.createShader(shaderType);
    if (!shader) {
      throw new Error('createShader() returned null');
    }

    gl.shaderSource(shader, shaderSource);
    gl.compileShader(shader);
    // TODO: check if the compilation was a success
    return shader;
  }
  deleteShader(shader: WebGLShader): void {
    this.gl.deleteShader(shader);
  }
  bindTextureToUniform(texture: WebGLTexture, position: number, uniformHandle: WebGLUniformLocation): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + position);
    this.checkError();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    this.checkError();
    gl.uniform1i(uniformHandle, position);
    this.checkError();
  }
  draw(): void {
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
    this.checkError();
  }
  deleteTexture(texture: WebGLTexture): void {
    this.gl.deleteTexture(texture);
  }
  deleteProgram(program: WebGLProgram): void {
    this.gl.deleteProgram(program);
  }
  getEncoder(dataType: Encoder.DataType, channels: number): DataEncoder {
    switch (dataType) {
      case 'float':
        return new RGBAFloat32DataEncoder(channels);
      case 'int':
        throw new Error('not implemented');
      case 'byte':
        return new Uint8DataEncoder(channels);
      default:
        throw new Error(`Invalid dataType: ${dataType}`);
    }
  }
  clearActiveTextures(): void {
    const gl = this.gl;
    for (let unit = 0; unit < this.maxTextureImageUnits; ++unit) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
  }
  protected queryVitalParameters(): void {
    const gl = this.gl;
    this.maxCombinedTextureImageUnits = gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS);
    this.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
    this.maxTextureImageUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
    this.maxCubeMapTextureSize = gl.getParameter(gl.MAX_CUBE_MAP_TEXTURE_SIZE);
    this.shadingLanguageVersion = gl.getParameter(gl.SHADING_LANGUAGE_VERSION);
    this.webglVendor = gl.getParameter(gl.VENDOR);
    this.webglVersion = gl.getParameter(gl.VERSION);
  }
  protected getExtensions(): void {
    this.textureFloatExtension = this.gl.getExtension('OES_texture_float');
    this.textureHalfFloatExtension = this.gl.getExtension('OES_texture_half_float');
  }
}

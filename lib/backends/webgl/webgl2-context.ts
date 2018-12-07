// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BaseWebGLContext} from './base-webgl-context';
import {DataEncoder, Encoder, RedFloat32DataEncoder,} from './texture-data-encoder';

export class WebGL2Context extends BaseWebGLContext {
  max3DTextureSize: number;
  maxArrayTextureLayers: number;
  maxColorAttachments: number;
  maxDrawBuffers: number;
  colorBufferFloatExtension: {};

  constructor(
      public canvas: HTMLCanvasElement, public gl: WebGLRenderingContext,
      public contextAttributes: WebGLContextAttributes) {
    super();
    this.init();
  }
  getEncoder(dataType: Encoder.DataType, channels: number): DataEncoder {
    return new RedFloat32DataEncoder(channels);
  }
  protected queryVitalParameters(): void {
    super.queryVitalParameters();
    const gl = this.gl;
    this.max3DTextureSize = gl.getParameter(WebGL2RenderingContext.MAX_3D_TEXTURE_SIZE);
    this.maxArrayTextureLayers = gl.getParameter(WebGL2RenderingContext.MAX_ARRAY_TEXTURE_LAYERS);
    this.maxColorAttachments = gl.getParameter(WebGL2RenderingContext.MAX_COLOR_ATTACHMENTS);
    this.maxDrawBuffers = gl.getParameter(WebGL2RenderingContext.MAX_DRAW_BUFFERS);
  }
  protected getExtensions() {
    this.colorBufferFloatExtension = this.gl.getExtension('EXT_color_buffer_float');
  }
}

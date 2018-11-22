// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {BaseWebGLContext} from './base-webgl-context';

export class WebGLExperimentalContext extends BaseWebGLContext {
  constructor(
      public canvas: HTMLCanvasElement, public gl: WebGLRenderingContext,
      public contextAttributes: WebGLContextAttributes) {
    super();
    this.init();
  }
}

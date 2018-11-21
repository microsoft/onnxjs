// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger} from '../../instrument';

import {WebGLContext} from './webgl-context';
import {WebGLExperimentalContext} from './webgl-experimental-context';
import {WebGL1Context} from './webgl1-context';
import {WebGL2Context} from './webgl2-context';

/**
 * This factory class creates proper WebGLRenderingContext based on
 * the current browsers capabilities
 * The order is from higher/most recent versions to most basic
 */
export class WebGLContextFactory {
  static create(contextAttributes: WebGLContextAttributes|null): WebGLContext {
    const canvas = this.createCanvas();
    if (contextAttributes == null) {
      contextAttributes = {
        alpha: false,
        depth: false,
        antialias: false,
        stencil: false,
        preserveDrawingBuffer: false,
        premultipliedAlpha: false,
        failIfMajorPerformanceCaveat: false
      };
    }
    let gl: WebGLRenderingContext|null;
    const ca = contextAttributes;
    gl = canvas.getContext('webgl2', ca) as WebGLRenderingContext | null;
    if (gl) {
      try {
        return new WebGL2Context(canvas, gl, ca);
      } catch (err) {
        Logger.warning('GlContextFactory', `failed to create WebGL2Context. Error: ${err}`);
      }
    }

    gl = canvas.getContext('webgl', ca);
    if (gl) {
      try {
        return new WebGL1Context(canvas, gl, ca);
      } catch (err) {
        Logger.warning('GlContextFactory', `failed to create WebGL1Context. Error: ${err}`);
      }
    }

    gl = canvas.getContext('experimental-webgl', ca);
    if (gl) {
      try {
        return new WebGLExperimentalContext(canvas, gl, ca);
      } catch (err) {
        Logger.warning('GlContextFactory', `failed to create WebGLExperimentalContext. Error: ${err}`);
      }
    }

    throw new Error('WebGL is not supported');
  }
  static createCanvas(): HTMLCanvasElement {
    const canvas: HTMLCanvasElement = document.createElement('canvas');
    canvas.width = 1;
    canvas.height = 1;
    return canvas;
  }
  static forceCreate(webgl: string, contextAttributes: WebGLContextAttributes|null): WebGLContext {
    const canvas = this.createCanvas();
    if (contextAttributes == null) {
      contextAttributes = {
        alpha: false,
        depth: false,
        antialias: false,
        stencil: false,
        preserveDrawingBuffer: false,
        premultipliedAlpha: false,
        failIfMajorPerformanceCaveat: true
      };
    }
    const gl = canvas.getContext(webgl, contextAttributes);
    if (!gl) {
      throw new Error('WebGL is not supported');
    }
    switch (webgl) {
      case 'webgl':
        return new WebGL1Context(canvas, gl as WebGLRenderingContext, contextAttributes);
      case 'webgl2':
        return new WebGL2Context(canvas, gl as WebGL2RenderingContext, contextAttributes);
      case 'experimental-webgl':
        return new WebGLExperimentalContext(canvas, gl as WebGLRenderingContext, contextAttributes);
      default:
        throw new Error('Invalid WebGL spec: ' + webgl);
    }
  }
}

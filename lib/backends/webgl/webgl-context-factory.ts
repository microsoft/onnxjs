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
  static create(contextId?: 'webgl'|'webgl2'|'experimental-webgl', contextAttributes?: WebGLContextAttributes):
      WebGLContext {
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
    if (!contextId || contextId === 'webgl2') {
      gl = canvas.getContext('webgl2', ca);
      if (gl) {
        try {
          return new WebGL2Context(canvas, gl, ca);
        } catch (err) {
          Logger.warning('GlContextFactory', `failed to create WebGL2Context. Error: ${err}`);
        }
      }
    }
    if (!contextId || contextId === 'webgl') {
      gl = canvas.getContext('webgl', ca);
      if (gl) {
        try {
          return new WebGL1Context(canvas, gl, ca);
        } catch (err) {
          Logger.warning('GlContextFactory', `failed to create WebGL1Context. Error: ${err}`);
        }
      }
    }
    if (!contextId || contextId === 'experimental-webgl') {
      gl = canvas.getContext('experimental-webgl', ca);
      if (gl) {
        try {
          return new WebGLExperimentalContext(canvas, gl, ca);
        } catch (err) {
          Logger.warning('GlContextFactory', `failed to create WebGLExperimentalContext. Error: ${err}`);
        }
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
}

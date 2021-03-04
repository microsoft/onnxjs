// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebGLSessionHandler} from './webgl/session-handler';
import {WebGLContext} from './webgl/webgl-context';
import {createWebGLContext} from './webgl/webgl-context-factory';

type WebGLOptions = BackendInterface.WebGLOptions;

/**
 * WebGLBackend is the entry point for all WebGL opeartions
 * When it starts it created the WebGLRenderingContext
 * and other main framework components such as Program and Texture Managers
 */
export class WebGLBackend implements Backend, WebGLOptions {
  disabled?: boolean;
  glContext: WebGLContext;
  contextId?: 'webgl'|'webgl2';
  matmulMaxBatchSize?: number;
  textureCacheMode?: 'initializerOnly'|'full';
  pack?: boolean;

  initialize(): boolean {
    try {
      this.glContext = createWebGLContext(this.contextId);
      if (typeof this.matmulMaxBatchSize !== 'number') {
        this.matmulMaxBatchSize = 16;
      }
      if (typeof this.textureCacheMode !== 'string') {
        this.textureCacheMode = 'full';
      }
      if (typeof this.pack !== 'boolean') {
        this.pack = false;
      }
      Logger.verbose(
          'WebGLBackend',
          `Created WebGLContext: ${typeof this.glContext} with matmulMaxBatchSize: ${
              this.matmulMaxBatchSize}; textureCacheMode: ${this.textureCacheMode}; pack: ${this.pack}.`);
      return true;
    } catch (e) {
      Logger.warning('WebGLBackend', `Unable to initialize WebGLBackend. ${e}`);
      return false;
    }
  }
  createSessionHandler(context: Session.Context): SessionHandler {
    return new WebGLSessionHandler(this, context);
  }
  dispose(): void {
    this.glContext.dispose();
  }
}

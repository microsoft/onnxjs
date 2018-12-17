// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebGLSessionHandler} from './webgl/session-handler';
import {WebGLContext} from './webgl/webgl-context';
import {WebGLContextFactory} from './webgl/webgl-context-factory';

type WebGLOptions = BackendInterface.WebGLOptions;

/**
 * WebGLBackend is the entry point for all WebGL opeartions
 * When it starts it created the WebGLRenderingContext
 * and other main framework components such as Program and Texture Managers
 */
export class WebGLBackend implements Backend, WebGLOptions {
  glContext: WebGLContext;
  contextId?: 'webgl'|'webgl2'|'experimental-webgl';

  initialize(): boolean {
    try {
      this.glContext = WebGLContextFactory.create(this.contextId);
      Logger.verbose('WebGLBackend', `Created WebGLContext: ${typeof this.glContext}`);
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

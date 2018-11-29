// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {getOnnxObject} from '../api/onnx-impl';
import {Backend, SessionHandler} from '../backend';
import {Session} from '../session';

import {CpuSessionHandler} from './cpu/session-handler';

type CpuOptions = BackendInterface.CpuOptions;

class CpuBackend implements Backend, CpuOptions {
  initialize(): boolean {
    return true;
  }
  createSessionHandler(context: Session.Context): SessionHandler {
    return new CpuSessionHandler(this, context);
  }
  dispose(): void {}
}

// register CPU backend
getOnnxObject().backend.cpu = new CpuBackend();

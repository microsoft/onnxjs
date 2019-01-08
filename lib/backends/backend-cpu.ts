// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {Backend, SessionHandler} from '../backend';
import {Session} from '../session';

import {CpuSessionHandler} from './cpu/session-handler';

type CpuOptions = BackendInterface.CpuOptions;

export class CpuBackend implements Backend, CpuOptions {
  disabled?: boolean;

  initialize(): boolean {
    return true;
  }
  createSessionHandler(context: Session.Context): SessionHandler {
    return new CpuSessionHandler(this, context);
  }
  dispose(): void {}
}

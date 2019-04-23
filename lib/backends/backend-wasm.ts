// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';
import * as wasmBinding from '../wasm-binding';

import {WasmSessionHandler} from './wasm/session-handler';

export let bindingInitPromise: Promise<void>|undefined;

type WasmOptions = BackendInterface.WasmOptions;

export class WasmBackend implements Backend, WasmOptions {
  disabled?: boolean;
  worker: number;
  cpuFallback: boolean;
  constructor() {
    // default parameters that users can override using the onnx global object

    // by default fallback to pure JS cpu ops if not resolved in wasm backend
    this.cpuFallback = true;

    // by default use 3 workers
    // TODO: Have logic to determing optimal fallback worker numbers based on CPU cores
    this.worker = 0;
  }
  async initialize(): Promise<boolean> {
    this.checkIfNumWorkersIsValid();
    const init = await this.isWasmSupported();
    if (!init) {
      return false;
    }
    return true;
  }
  createSessionHandler(context: Session.Context): SessionHandler {
    return new WasmSessionHandler(this, context, this.cpuFallback);
  }
  dispose(): void {}
  checkIfNumWorkersIsValid() {
    if (!Number.isFinite(this.worker) || Number.isNaN(this.worker)) {
      throw new Error(`${this.worker} is not valid number of workers`);
    }
    if (!Number.isInteger(this.worker)) {
      throw new Error(`${this.worker} is not an integer and hence not valid number of workers`);
    }
  }
  async isWasmSupported(): Promise<boolean> {
    try {
      await wasmBinding.init(this.worker);
      return true;
    } catch (e) {
      Logger.warning('WebAssembly', `Unable to initialize WebAssembly backend. ${e}`);
      return false;
    }
  }
}

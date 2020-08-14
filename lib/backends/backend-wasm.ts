// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import * as platform from 'platform';

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
  initTimeout: number;
  constructor() {
    // default parameters that users can override using the onnx global object

    // by default fallback to pure JS cpu ops if not resolved in wasm backend
    this.cpuFallback = true;

    this.worker = defaultNumWorkers();

    this.initTimeout = 5000;
  }
  async initialize(): Promise<boolean> {
    checkIfNumWorkersIsValid(this.worker);
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

  async isWasmSupported(): Promise<boolean> {
    try {
      await wasmBinding.init(this.worker, this.initTimeout);
      return true;
    } catch (e) {
      Logger.warning('WebAssembly', `Unable to initialize WebAssembly backend. ${e}`);
      return false;
    }
  }
}

function defaultNumWorkers(): number {
  if (typeof navigator !== 'undefined' && navigator) {
    // by default use ([navigator.hardwareConcurrency / 2] - 1) workers
    if (typeof navigator.hardwareConcurrency === 'number') {
      return Math.max(Math.ceil(navigator.hardwareConcurrency / 2) - 1, 0);
    }

    // if object 'navigator' exists, but 'navigator.hardwareConcurrency' does not. This may mean:
    // - The environment is Safari (macOS/iOS), or
    // - it's not any mainstream browser.
    if (platform.name === 'Safari') {
      if (platform.os && (platform.os.family === 'iOS' || platform.os.family === 'OS X')) {
        return 1;
      }
    }
  }

  return 0;
}

function checkIfNumWorkersIsValid(worker: number) {
  if (!Number.isFinite(worker) || Number.isNaN(worker)) {
    throw new Error(`${worker} is not valid number of workers`);
  }
  if (!Number.isInteger(worker)) {
    throw new Error(`${worker} is not an integer and hence not valid number of workers`);
  }
}

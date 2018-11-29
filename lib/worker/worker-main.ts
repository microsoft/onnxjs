// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/// <reference lib="webworker" />
import {init, WasmBinding} from '../wasm-binding-core';

class WorkerBinding extends WasmBinding {
  static instance: WorkerBinding;
  static getInstance() {
    if (!WorkerBinding.instance) {
      WorkerBinding.instance = new WorkerBinding();
    }
    return WorkerBinding.instance;
  }
}

let instance: WorkerBinding;

init().then(
    () => {
      instance = WorkerBinding.getInstance();
      postMessage({type: 'init-success'});
    },
);

onmessage = (e) => {
  if (e && e.data && e.data.type) {
    if (e.data.type === 'ccall') {
      const func: string = e.data.func;
      const buffer: ArrayBuffer = e.data.buffer;

      const perfData = instance.ccallRaw(func, new Uint8Array(buffer));

      postMessage({type: 'ccall', buffer, perfData}, [buffer]);
    } else {
      throw new Error(`unknown message type from main thread: ${e.data.type}`);
    }
  } else {
    throw new Error(`missing message type from main thread`);
  }
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {CpuBackend} from '../backends/backend-cpu';
import {WasmBackend} from '../backends/backend-wasm';
import {WebGLBackend} from '../backends/backend-webgl';

import {Backend} from './onnx';

export * from './onnx';
export * from './tensor';
export * from './inference-session';

export const backend: Backend = {
  cpu: new CpuBackend(),
  wasm: new WasmBackend(),
  webgl: new WebGLBackend()
};

export let debug = false;

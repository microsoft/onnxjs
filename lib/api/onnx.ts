// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceSessionConstructor} from './inference-session';
import {TensorConstructor} from './tensor';

//#region Backends

export declare namespace Backend {
  /**
   * set options for the CPU backend
   */
  interface CpuOptions {}
  /**
   * set options for the WebGL backend
   */
  interface WebGLOptions {}
  /**
   * set options for the WebAssembly backend
   */
  interface WasmOptions {
    /**
     * set or get number of worker(s)
     */
    worker?: number;
    /**
     * set or get a flag specifying if the fallback cpu implementations can be used in case of missing ops
     */
    cpuFallback?: boolean;
  }
}

// tslint:disable-next-line:no-any
type BackendOptions = any;

/**
 * represent all available backends and settings of them
 */
export interface Backend {
  /**
   * set one or more string(s) as hint for onnx session to resolve the corresponding backend
   */
  hint?: string|ReadonlyArray<string>;

  cpu: Backend.CpuOptions;
  webgl: Backend.WebGLOptions;
  wasm: Backend.WasmOptions;

  /**
   * set options for the specific backend
   */
  [name: string]: BackendOptions;
}

//#endregion Backends

/**
 * represent runtime environment settings and status of ONNX.js
 */
export interface Environment {
  /**
   * a global flag to indicate whether to run ONNX.js in debug mode
   */
  debug: boolean;
}

export interface Onnx extends Environment {
  /**
   * represent a tensor with specified dimensions and data type.
   */
  readonly Tensor: TensorConstructor;
  /**
   * represent a runtime instance of an ONNX model
   */
  readonly InferenceSession: InferenceSessionConstructor;
  /**
   * represent all available backends and settings of them
   */
  readonly backend: Backend;
}

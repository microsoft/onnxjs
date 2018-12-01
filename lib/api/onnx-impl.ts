// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceSessionConstructor} from './inference-session';
import {InferenceSession} from './inference-session-impl';
import {Backend, Environment, Onnx} from './onnx';
import {TensorConstructor} from './tensor';
import {Tensor} from './tensor-impl';

interface OnnxGlobal {
  onnx?: {Tensor?: TensorConstructor; InferenceSession?: InferenceSessionConstructor; backend?: Partial<Backend>;}&
      Partial<Environment>;
}

/**
 * get the onnx object from global context. if it does not exist, create a new object, initialize and return it
 */
export function getOnnxObject(): Onnx {
  // set onnx object
  const global = getGlobal();
  global.onnx = global.onnx || {};
  const onnx = global.onnx;

  // initialize onnx object
  onnx.Tensor = onnx.Tensor || Tensor;
  onnx.InferenceSession = onnx.InferenceSession || InferenceSession;

  // set backend object
  onnx.backend = onnx.backend || {};

  // set environment properties
  setEnvironment(onnx);

  return onnx as Onnx;
}

function getGlobal(): OnnxGlobal {
  return ((typeof window !== 'undefined') ? window : global) as OnnxGlobal;
}

function setEnvironment(env: Partial<Environment>) {
  // placeholder to implement environment
}

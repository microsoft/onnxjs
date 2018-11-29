// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceSessionConstructor} from './inference-session';
import {InferenceSession} from './inference-session-impl';
import {Backend, Onnx} from './onnx';
import {TensorConstructor} from './tensor';
import {Tensor} from './tensor-impl';

interface OnnxGlobal {
  onnx?: {
    Tensor?: TensorConstructor;
    InferenceSession?: InferenceSessionConstructor;
    backend?: Partial<Backend>;
    debug?: boolean;
  };
}

function getGlobal(): OnnxGlobal {
  return ((typeof window !== 'undefined') ? window : global) as OnnxGlobal;
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
  if (typeof onnx.Tensor === 'undefined') {
    onnx.Tensor = Tensor;
  }
  if (typeof onnx.InferenceSession === 'undefined') {
    onnx.InferenceSession = InferenceSession;
  }

  // set backend object
  onnx.backend = onnx.backend || {};

  return global.onnx as Onnx;
}

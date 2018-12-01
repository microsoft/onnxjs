// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceSession as InferenceSessionInterface} from './inference-session';
import {Onnx} from './onnx';
import {getOnnxObject} from './onnx-impl';
import {Tensor as TensorInterface} from './tensor';

// get or create the onnx object in the global context
const onnx = getOnnxObject();

// set module exported object to global.onnx
export = onnx;

// declaration merging for 'Tensor' and 'InferenceSession'
declare namespace onnx {
  export type Tensor = TensorInterface;
  export type InferenceSession = InferenceSessionInterface;
}

// declaration of object global.onnx
declare global {
  /**
   * the global onnxjs exported object
   */
  const onnx: Onnx;
}

// attach pre-built backends

// tslint:disable-next-line:no-import-side-effect
import '../backends';

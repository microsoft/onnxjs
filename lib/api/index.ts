// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Onnx} from './onnx';
import * as OnnxImpl from './onnx-impl';

// get or create the onnx object in the global context
const onnx: Onnx = OnnxImpl;
const onnxGlobal = ((typeof window !== 'undefined') ? window : global) as {onnx?: Onnx};
onnxGlobal.onnx = onnx;

// set module exported object to global.onnx
export = OnnxImpl;

// declaration of object global.onnx
declare global {
  /**
   * the global onnxjs exported object
   */
  const onnx: Onnx;
}

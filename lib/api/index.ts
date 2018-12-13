// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Onnx} from './onnx';
import * as onnxImpl from './onnx-impl';

// get or create the onnx object in the global context
const onnxGlobal = ((typeof window !== 'undefined') ? window : global) as {onnx?: Onnx};
const onnx: Onnx = onnxImpl;
onnxGlobal.onnx = onnx;

// set module exported object to global.onnx
export = onnxImpl;

// declaration of object global.onnx
declare global {
  /**
   * the global onnxjs exported object
   */
  const onnx: Onnx;
}

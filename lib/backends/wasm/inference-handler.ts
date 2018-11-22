// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Profiler} from '../../instrument';

import {WasmSessionHandler} from './session-handler';

export class WasmInferenceHandler implements InferenceHandler {
  constructor(public readonly session: WasmSessionHandler, public readonly profiler?: Readonly<Profiler>) {}

  dispose(): void {}
}

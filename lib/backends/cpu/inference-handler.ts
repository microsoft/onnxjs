// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Profiler} from '../../instrument';

import {CpuSessionHandler} from './session-handler';

export class CpuInferenceHandler implements InferenceHandler {
  constructor(public readonly session: CpuSessionHandler, public readonly profiler?: Readonly<Profiler>) {}

  dispose(): void {}
}

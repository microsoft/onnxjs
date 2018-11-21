// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend, InferenceHandler, SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {Session} from '../../session';

import {CpuInferenceHandler} from './inference-handler';
import {resolve} from './ops-resolve';

export class CpuSessionHandler implements SessionHandler {
  constructor(readonly backend: Backend, readonly context: Session.Context) {}

  createInferenceHandler(): InferenceHandler {
    return new CpuInferenceHandler(this, this.context.profiler);
  }

  dispose(): void {}

  resolve(node: Graph.Node, domain: string, version: number): Operator {
    // We have kept the ops resolve logic separately to be leveraged by other components (if needed)
    // This is valid only if there is no statefulness associated with the op resolution logic (which is currently true)
    return resolve(node, domain, version);
  }
}

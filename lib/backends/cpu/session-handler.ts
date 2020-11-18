// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend, InferenceHandler, SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';

import {CpuInferenceHandler} from './inference-handler';
import {CPU_OP_RESOLVE_RULES} from './op-resolve-rules';

export class CpuSessionHandler implements SessionHandler {
  constructor(readonly backend: Backend, readonly context: Session.Context) {}

  createInferenceHandler(): InferenceHandler {
    return new CpuInferenceHandler(this, this.context.profiler);
  }

  dispose(): void {}

  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>, graph: Graph): Operator {
    const op = resolveOperator(node, opsets, CPU_OP_RESOLVE_RULES);
    op.initialize(node.attributes, node, graph);
    return op;
  }
}

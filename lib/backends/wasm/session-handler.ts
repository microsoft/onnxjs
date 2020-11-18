// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend, InferenceHandler, SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {CPU_OP_RESOLVE_RULES} from '../cpu/op-resolve-rules';

import {WasmInferenceHandler} from './inference-handler';
import {WASM_OP_RESOLVE_RULES} from './op-resolve-rules';

export class WasmSessionHandler implements SessionHandler {
  private opResolveRules: ReadonlyArray<OpSet.ResolveRule>;
  constructor(readonly backend: Backend, readonly context: Session.Context, fallbackToCpuOps: boolean) {
    this.opResolveRules = fallbackToCpuOps ? WASM_OP_RESOLVE_RULES.concat(CPU_OP_RESOLVE_RULES) : WASM_OP_RESOLVE_RULES;
  }

  createInferenceHandler(): InferenceHandler {
    return new WasmInferenceHandler(this, this.context.profiler);
  }

  dispose(): void {}

  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>, graph: Graph): Operator {
    const op = resolveOperator(node, opsets, this.opResolveRules);
    op.initialize(node.attributes, node, graph);
    return op;
  }
}

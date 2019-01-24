// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend, InferenceHandler, SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {Session} from '../../session';
import {resolve} from '../cpu/ops-resolve';
import {WasmInferenceHandler} from './inference-handler';
import {WasmBatchNormalization} from './ops/batch-normalization';
import {WasmConv} from './ops/conv';
import {WasmGemm} from './ops/gemm';
import {WasmInstanceNormalization} from './ops/instance-normalization';
import {WasmAveragePool, WasmGlobalAveragePool, WasmGlobalMaxPool, WasmMaxPool} from './ops/pool';
import {WasmSoftmax} from './ops/softmax';
import {WasmSum} from './ops/sum';

export class WasmSessionHandler implements SessionHandler {
  constructor(readonly backend: Backend, readonly context: Session.Context, private fallbackToCpuOps: boolean) {}

  createInferenceHandler(): InferenceHandler {
    return new WasmInferenceHandler(this, this.context.profiler);
  }

  dispose(): void {}

  resolve(node: Graph.Node, domain: string, version: number): Operator {
    const op = this.createOperator(node, domain, version);
    op.initialize(node.attributes);
    return op;
  }

  private createOperator(node: Graph.Node, domain: string, version: number): Operator {
    // assume domain=ai.onnx, version=v7
    switch (node.opType) {
      case 'Conv':
        return new WasmConv();
      case 'BatchNormalization':
        return new WasmBatchNormalization();
      case 'Gemm':
        return new WasmGemm();
      case 'Softmax':
        return new WasmSoftmax();
      case 'Sum':
        return new WasmSum();
      case 'AveragePool':
        return new WasmAveragePool();
      case 'MaxPool':
        return new WasmMaxPool();
      case 'GlobalMaxPool':
        return new WasmGlobalMaxPool();
      case 'GlobalAveragePool':
        return new WasmGlobalAveragePool();
      case 'InstanceNormalization':
        return new WasmInstanceNormalization();
      default:
        if (this.fallbackToCpuOps) {
          return resolve(node, domain, version);
        }
        throw new TypeError(`unrecognized operator '${node.opType}'`);
    }
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Graph} from '../../graph';
import {FLOAT_TYPES, NUMBER_TYPES, Operator} from '../../operators';

import {CpuArgMax} from './ops/argMax';
import {CpuBatchNormalization} from './ops/batch-normalization';
import {CpuBinaryOp} from './ops/binary-op';
import {CpuConcat} from './ops/concat';
import {CpuConv} from './ops/conv';
import {CpuDropout} from './ops/dropout';
import {CpuFlatten} from './ops/flatten';
import {CpuGather} from './ops/gather';
import {CpuGemm} from './ops/gemm';
import {CpuImageScaler} from './ops/image-scaler';
import {CpuInstanceNormalization} from './ops/instance-normalization';
import {CpuLrn} from './ops/lrn';
import {CpuMatMul} from './ops/matmul';
import {CpuAveragePool, CpuGlobalAveragePool, CpuGlobalMaxPool, CpuMaxPool} from './ops/pool';
import * as cpuReduce from './ops/reduce';
import {CpuReshape} from './ops/reshape';
import {CpuSlice} from './ops/slice';
import {CpuSoftmax} from './ops/softmax';
import {CpuSqueeze} from './ops/squeeze';
import {CpuSum} from './ops/sum';
import {CpuTile} from './ops/tile';
import {CpuTranspose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
import {CpuUnsqueeze} from './ops/unsqueeze';

export function resolve(node: Graph.Node, domain: string, version: number): Operator {
  const op = createOperator(node, domain, version);
  op.initialize(node.attributes);
  return op;
}

function createOperator(node: Graph.Node, domain: string, version: number): Operator {
  // assume domain=ai.onnx, version=v7
  switch (node.opType) {
    // Unary ops
    case 'Abs':
      return new unaryOps.CpuUnaryOp(NUMBER_TYPES, unaryOps.abs);
    case 'Neg':
      return new unaryOps.CpuUnaryOp(NUMBER_TYPES, unaryOps.neg);
    case 'Acos':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.acos);
    case 'Ceil':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.ceil);
    case 'Cos':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.cos);
    case 'Sin':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.sin);
    case 'Tan':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.tan);
    case 'Tanh':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.tanh);
    case 'Exp':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.exp);
    case 'Floor':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.floor);
    case 'Atan':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.atan);
    case 'Relu':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.relu);
    case 'Log':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.log);
    case 'Sqrt':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.sqrt);
    case 'Asin':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.asin);
    case 'Sigmoid':
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.sigmoid);
    // Binary arithmetic ops
    case 'Add':
      return new CpuBinaryOp(NUMBER_TYPES, (e1, e2) => (e1 + e2));
    case 'Sub':
      return new CpuBinaryOp(NUMBER_TYPES, (e1, e2) => (e1 - e2));
    case 'Mul':
      return new CpuBinaryOp(NUMBER_TYPES, (e1, e2) => (e1 * e2));
    case 'Div':
      // TODO: Handle division by zero
      return new CpuBinaryOp(NUMBER_TYPES, (e1, e2) => (e1 / e2));
    // Binary logical ops
    case 'Xor':
      return new CpuBinaryOp(['bool'], (e1, e2) => (e1 ^ e2));
    case 'Or':
      return new CpuBinaryOp(['bool'], (e1, e2) => (e1 || e2));
    case 'And':
      return new CpuBinaryOp(['bool'], (e1, e2) => (e1 && e2));
    // Non-unary and non-binary ops
    case 'ArgMax':
      return new CpuArgMax();
    case 'BatchNormalization':
      return new CpuBatchNormalization();
    case 'Concat':
      return new CpuConcat();
    case 'Conv':
      return new CpuConv();
    case 'Dropout':
      return new CpuDropout();
    case 'Flatten':
      return new CpuFlatten();
    case 'Gemm':
      return new CpuGemm();
    case 'ImageScaler':
      return new CpuImageScaler();
    case 'LRN':
      return new CpuLrn();
    case 'LeakyRelu':
      // opLambda will be resolved when the op is initialized at which time it will have context of the attribute
      // 'alpha'
      return new unaryOps.CpuUnaryOp(FLOAT_TYPES, unaryOps.leakyRelu);
    case 'MatMul':
      return new CpuMatMul();
    case 'AveragePool':
      return new CpuAveragePool();
    case 'MaxPool':
      return new CpuMaxPool();
    case 'Gather':
      return new CpuGather();
    case 'GlobalMaxPool':
      return new CpuGlobalMaxPool();
    case 'GlobalAveragePool':
      return new CpuGlobalAveragePool();
    case 'InstanceNormalization':
      return new CpuInstanceNormalization();
    case 'PRelu':
      return new CpuBinaryOp(NUMBER_TYPES, (e1, e2) => (e1 >= 0 ? e1 : e1 * e2));
    case 'Reshape':
      return new CpuReshape();
    case 'ReduceLogSum':
      return new cpuReduce.CpuReduceLogSum();
    case 'ReduceMax':
      return new cpuReduce.CpuReduceMax();
    case 'ReduceMean':
      return new cpuReduce.CpuReduceMean();
    case 'ReduceMin':
      return new cpuReduce.CpuReduceMin();
    case 'ReduceProd':
      return new cpuReduce.CpuReduceProd();
    case 'ReduceSum':
      return new cpuReduce.CpuReduceSum();
    case 'ReduceSumSquare':
      return new cpuReduce.CpuReduceSumSquare();
    case 'Slice':
      return new CpuSlice();
    case 'Softmax':
      return new CpuSoftmax();
    case 'Squeeze':
      return new CpuSqueeze();
    case 'Sum':
      return new CpuSum();
    case 'Tile':
      return new CpuTile();
    case 'Transpose':
      return new CpuTranspose();
    case 'Unsqueeze':
      return new CpuUnsqueeze();
    default:
      throw new TypeError(`unrecognized operator '${node.opType}'`);
  }
}

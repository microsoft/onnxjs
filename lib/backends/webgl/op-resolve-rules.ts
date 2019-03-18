// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {FLOAT_TYPES, NUMBER_TYPES} from '../../operators';
import {OpSet} from '../../opset';

import {WebGLBatchNormalization} from './ops/batch-normalization';
import * as binaryOps from './ops/binary-op';
import {WebGLConcat} from './ops/concat';
import {WebGLConv} from './ops/conv';
import {WebGLDropout} from './ops/dropout';
import {WebGLElu} from './ops/elu';
import {WebGLFlatten} from './ops/flatten';
import {WebGLGather} from './ops/gather';
import {WebGLGemm} from './ops/gemm';
import {WebGLImageScaler} from './ops/image-scaler';
import {WebGLLeakyRelu} from './ops/leaky-relu';
import {WebGLMatMul} from './ops/matmul';
import {WebGLPad} from './ops/pad';
import {WebGLAveragePool, WebGLGlobalAveragePool, WebGLGlobalMaxPool, WebGLMaxPool} from './ops/pool';
import * as reduceOps from './ops/reduce';
import {WebGLReshape} from './ops/reshape';
import {WebGLSlice} from './ops/slice';
import {WebGLSoftmax} from './ops/softmax';
import {WebGLSplit} from './ops/split';
import {WebGLSqueeze} from './ops/squeeze';
import {WebGLSum} from './ops/sum';
import {WebGLTile} from './ops/tile';
import {WebGLTranspose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
import {WebGLUnsqueeze} from './ops/unsqueeze';

export const WEBGL_OP_RESOLVE_RULES: ReadonlyArray<OpSet.ResolveRule> = [
  ['Abs', '', '7+', () => new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslAbs())],
  ['Acos', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAcos())],
  ['Add', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslAdd())],
  ['And', '', '7+', () => new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslAnd())],
  ['Asin', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAsin())],
  ['Atan', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAtan())],
  ['AveragePool', '', '7+', () => new WebGLAveragePool()],
  ['BatchNormalization', '', '7+', () => new WebGLBatchNormalization()],
  ['Ceil', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslCeil())],
  ['Concat', '', '4+', () => new WebGLConcat()],
  ['Conv', '', '1+', () => new WebGLConv()],
  ['Cos', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslCos())],
  ['Div', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslDiv())],
  ['Dropout', '', '7+', () => new WebGLDropout()],
  ['Equal', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslEqual(), undefined, 'bool')],
  ['Elu', '', '6+', () => new WebGLElu()],
  ['Exp', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslExp())],
  ['Flatten', '', '1+', () => new WebGLFlatten()],
  ['Floor', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslFloor())],
  ['Gather', '', '1+', () => new WebGLGather()],
  ['Gemm', '', '7+', () => new WebGLGemm()],
  ['GlobalAveragePool', '', '1+', () => new WebGLGlobalAveragePool()],
  ['GlobalMaxPool', '', '1+', () => new WebGLGlobalMaxPool()],
  ['Greater', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslGreater(), undefined, 'bool')],
  ['Identity', '', '1+', () => new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslIdentity())],
  ['ImageScaler', '', '1+', () => new WebGLImageScaler()],
  ['LeakyRelu', '', '6+', () => new WebGLLeakyRelu()],
  ['Less', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslLess(), undefined, 'bool')],
  ['Log', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslLog())],
  ['MatMul', '', '1+', () => new WebGLMatMul()],
  ['MaxPool', '', '1+', () => new WebGLMaxPool()],
  ['Mul', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslMul())],
  ['Neg', '', '6+', () => new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslNeg())],
  ['Not', '', '1+', () => new unaryOps.WebGLUnaryOp(['bool'], unaryOps.glslNot())],
  ['Or', '', '7+', () => new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslOr())],
  ['Pad', '', '2+', () => new WebGLPad()],
  ['Pow', '', '7+', () => new binaryOps.WebGLBinaryOp(FLOAT_TYPES, binaryOps.glslPow())],
  ['PRelu', '', '7+', () => new binaryOps.WebGLBinaryOp(FLOAT_TYPES, binaryOps.glslPRelu())],
  ['ReduceLogSum', '', '1+', () => new reduceOps.WebGLReduceLogSum()],
  ['ReduceMax', '', '1+', () => new reduceOps.WebGLReduceMax()],
  ['ReduceMean', '', '1+', () => new reduceOps.WebGLReduceMean()],
  ['ReduceMin', '', '1+', () => new reduceOps.WebGLReduceMin()],
  ['ReduceProd', '', '1+', () => new reduceOps.WebGLReduceProd()],
  ['ReduceSum', '', '1+', () => new reduceOps.WebGLReduceSum()],
  ['ReduceSumSquare', '', '1+', () => new reduceOps.WebGLReduceSumSquare()],
  ['Relu', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslRelu())],
  ['Reshape', '', '5+', () => new WebGLReshape()],
  ['Sigmoid', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSigmoid())],
  ['Sin', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSin())],
  ['Slice', '', '1+', () => new WebGLSlice()],
  ['Softmax', '', '1+', () => new WebGLSoftmax()],
  // 'Split' operator has an optional attribute 'split'
  // this attribute determines how the specified axis of input data
  // is split. When the attribute is missing, we need the count of number of outputs
  // so that we can determine the 'split' attribute from the runtime input to the Operator
  ['Split', '', '2+', (node) => new WebGLSplit(node.outputs.length)],
  ['Sqrt', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSqrt())],
  ['Squeeze', '', '1+', () => new WebGLSqueeze()],
  ['Sub', '', '7+', () => new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslSub())],
  ['Sum', '', '6+', () => new WebGLSum()],
  ['Tan', '', '7+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslTan())],
  ['Tanh', '', '6+', () => new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslTanh())],
  ['Tile', '', '6+', () => new WebGLTile()],
  ['Transpose', '', '1+', () => new WebGLTranspose()],
  ['Unsqueeze', '', '1+', () => new WebGLUnsqueeze()],
  ['Xor', '', '7+', () => new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslXor())],
];

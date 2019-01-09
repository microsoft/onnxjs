// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Graph} from '../../graph';
import {Logger} from '../../instrument';
import {FLOAT_TYPES, NUMBER_TYPES, Operator} from '../../operators';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {WebGLBackend} from '../backend-webgl';

import {SessionHandler} from './../../backend';
import {WebGLInferenceHandler} from './inference-handler';
import {WebGLBatchNormalization} from './ops/batch-normalization';
import * as binaryOps from './ops/binary-op';
import {WebGLConcat} from './ops/concat';
import {WebGLConv} from './ops/conv';
import {WebGLDropout} from './ops/dropout';
import {WebGLGather} from './ops/gather';
import {WebGLGemm} from './ops/gemm';
import {WebGLImageScaler} from './ops/image-scaler';
import {WebGLLeakyRelu} from './ops/leaky-relu';
import {WebGLMatMul} from './ops/matmul';
import {WebGLPad} from './ops/pad';
import {WebGLAveragePool, WebGLGlobalAveragePool, WebGLGlobalMaxPool, WebGLMaxPool} from './ops/pool';
import {WebGLReduceSum} from './ops/reduce';
import {WebGLReduceMean} from './ops/reduce';
import {WebGLReduceMax} from './ops/reduce';
import {WebGLReduceMin} from './ops/reduce';
import {WebGLReduceProd} from './ops/reduce';
import {WebGLReduceLogSum} from './ops/reduce';
import {WebGLReduceSumSquare} from './ops/reduce';
import {WebGLReshape} from './ops/reshape';
import {WebGLSlice} from './ops/slice';
import {WebGLSoftmax} from './ops/softmax';
import {WebGLSplit} from './ops/split';
import {WebGLSum} from './ops/sum';
import {WebGLTile} from './ops/tile';
import {WebGLTranspose} from './ops/transpose';
import * as unaryOps from './ops/unary-op';
import {ProgramManager} from './program-manager';
import {TextureData} from './texture-data';
import {TextureHelper} from './texture-helper';
import {AlwaysKeepOriginalSizeStrategy, TextureLayoutStrategy} from './texture-layout-strategy';

export class WebGLSessionHandler implements SessionHandler {
  programManager: ProgramManager;
  textureHelper: TextureHelper;
  layoutStrategy: TextureLayoutStrategy;
  textureDataCache: Map<Tensor, TextureData>;
  initializers: Set<Tensor>;

  constructor(public readonly backend: WebGLBackend, public readonly context: Session.Context) {
    this.programManager = new ProgramManager(this.context.profiler, backend.glContext);
    this.layoutStrategy = new AlwaysKeepOriginalSizeStrategy(backend.glContext.maxTextureSize);
    this.textureHelper = new TextureHelper(backend.glContext, this.layoutStrategy, this.context.profiler);
    this.textureDataCache = new Map();
  }

  createInferenceHandler() {
    return new WebGLInferenceHandler(this.backend, this);
  }
  onGraphInitialized(graph: Graph): void {
    const initializers = graph.getValues().filter(v => v.from === -1).map(v => v.tensor).filter(t => (t)) as Tensor[];
    this.initializers = new Set(initializers);
  }
  isInitializer(t: Tensor): boolean {
    return this.initializers ? this.initializers.has(t) : false;
  }
  getTextureData(tensor: Tensor): TextureData|undefined {
    return this.textureDataCache.get(tensor);
  }
  setTextureData(tensor: Tensor, textureData: TextureData): void {
    Logger.verbose('WebGLSessionHandler', 'Storing Texture data in cache');
    this.textureDataCache.set(tensor, textureData);
  }
  dispose(): void {
    this.programManager.dispose();
    this.textureHelper.clearActiveTextures();
    this.textureDataCache.forEach(td => this.textureHelper.releaseTexture(td.texture));
    this.textureDataCache = new Map();
  }
  resolve(node: Graph.Node, domain: string, version: number): Operator {
    const op = this.createOperator(node, domain, version);
    op.initialize(node.attributes);
    return op;
  }

  private createOperator(node: Graph.Node, domain: string, version: number): Operator {
    // assume domain=ai.onnx, version=v7
    switch (node.opType) {
      // Unary ops
      case 'Abs':
        return new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslAbs());
      case 'Acos':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAcos());
      case 'Add':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslAdd());
      case 'And':
        return new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslAnd());
      case 'Asin':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAsin());
      case 'Atan':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslAtan());
      case 'AveragePool':
        return new WebGLAveragePool();
      case 'BatchNormalization':
        return new WebGLBatchNormalization();
      case 'Ceil':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslCeil());
      case 'Cos':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslCos());
      case 'Concat':
        return new WebGLConcat();
      case 'Conv':
        return new WebGLConv();
      case 'Div':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslDiv());
      case 'Dropout':
        return new WebGLDropout();
      case 'Equal':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslEqual(), 'bool');
      case 'Exp':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslExp());
      case 'Floor':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslFloor());
      case 'Gather':
        return new WebGLGather();
      case 'Gemm':
        return new WebGLGemm();
      case 'GlobalAveragePool':
        return new WebGLGlobalAveragePool();
      case 'GlobalMaxPool':
        return new WebGLGlobalMaxPool();
      case 'Greater':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslGreater(), 'bool');
      case 'Identity':
        return new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslIdentity());
      case 'ImageScaler':
        return new WebGLImageScaler();
      case 'LeakyRelu':
        return new WebGLLeakyRelu();
      case 'Less':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslLess(), 'bool');
      case 'Log':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslLog());
      case 'MatMul':
        return new WebGLMatMul();
      case 'MaxPool':
        return new WebGLMaxPool();
      case 'Mul':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslMul());
      case 'Neg':
        return new unaryOps.WebGLUnaryOp(NUMBER_TYPES, unaryOps.glslNeg());
      case 'Not':
        return new unaryOps.WebGLUnaryOp(['bool'], unaryOps.glslNot());
      case 'Or':
        return new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslOr());
      case 'Pad':
        return new WebGLPad();
      case 'Pow':
        return new binaryOps.WebGLBinaryOp(FLOAT_TYPES, binaryOps.glslPow());
      case 'PRelu':
        return new binaryOps.WebGLBinaryOp(FLOAT_TYPES, binaryOps.glslPRelu());
      case 'Relu':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslRelu());
      case 'Reshape':
        return new WebGLReshape();
      case 'Sigmoid':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSigmoid());
      case 'Sin':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSin());
      case 'ReduceSum':
        return new WebGLReduceSum();
      case 'ReduceMean':
        return new WebGLReduceMean();
      case 'ReduceMax':
        return new WebGLReduceMax();
      case 'ReduceMin':
        return new WebGLReduceMin();
      case 'ReduceProd':
        return new WebGLReduceProd();
      case 'ReduceLogSum':
        return new WebGLReduceLogSum();
      case 'ReduceSumSquare':
        return new WebGLReduceSumSquare();
      case 'Softmax':
        return new WebGLSoftmax();
      case 'Split':
        // 'Split' operator has an optional attribute 'split'
        // this attribute determines how the specified axis of input data
        // is split. When the attribute is missing, we need the count of number of outputs
        // so that we can determine the 'split' attribute from the runtime input to the Operator
        return new WebGLSplit(node.outputs.length);
      case 'Sqrt':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslSqrt());
      case 'Sub':
        return new binaryOps.WebGLBinaryOp(NUMBER_TYPES, binaryOps.glslSub());
      case 'Sum':
        return new WebGLSum();
      case 'Slice':
        return new WebGLSlice();
      case 'Tan':
        return new unaryOps.WebGLUnaryOp(FLOAT_TYPES, unaryOps.glslTan());
      case 'Transpose':
        return new WebGLTranspose();
      case 'Tile':
        return new WebGLTile();
      case 'Xor':
        return new binaryOps.WebGLBinaryOp(['bool'], binaryOps.glslXor());
      default:
        throw new TypeError(`unrecognized operator '${node.opType}'`);
    }
  }
}

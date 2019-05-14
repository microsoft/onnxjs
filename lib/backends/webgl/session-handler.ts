// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Logger} from '../../instrument';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {WebGLBackend} from '../backend-webgl';

import {WebGLInferenceHandler} from './inference-handler';
import {WEBGL_OP_RESOLVE_RULES} from './op-resolve-rules';
import {ProgramManager} from './program-manager';
import {TextureHelper} from './texture-helper';
import {AlwaysKeepOriginalSizeStrategy, TextureLayoutStrategy} from './texture-layout-strategy';
import {TextureData} from './types';

export class WebGLSessionHandler implements SessionHandler {
  programManager: ProgramManager;
  textureHelper: TextureHelper;
  layoutStrategy: TextureLayoutStrategy;
  textureDataCache: Map<Tensor.Id, TextureData>;
  initializers: Set<Tensor.Id>;

  constructor(public readonly backend: WebGLBackend, public readonly context: Session.Context) {
    this.programManager = new ProgramManager(this.context.profiler, backend.glContext);
    this.layoutStrategy = new AlwaysKeepOriginalSizeStrategy(backend.glContext.maxTextureSize);
    this.textureHelper = new TextureHelper(backend.glContext, this.layoutStrategy, this.context.profiler);
    this.textureDataCache = new Map();
  }

  transformGraph(graphTransformer: Graph.Transformer): void {
    if (!this.backend.glContext.isFloat32DownloadSupported) {
      graphTransformer.appendNodeToOutputs('webgl_DownloadFloat');
    }
  }

  createInferenceHandler() {
    return new WebGLInferenceHandler(this);
  }
  onGraphInitialized(graph: Graph): void {
    const initializers = graph.getValues().filter(v => v.from === -1 && v.tensor).map(v => v.tensor!.dataId);
    this.initializers = new Set(initializers);
  }
  isInitializer(tensorId: Tensor.Id): boolean {
    return this.initializers ? this.initializers.has(tensorId) : false;
  }
  getTextureData(tensorId: Tensor.Id): TextureData|undefined {
    return this.textureDataCache.get(tensorId);
  }
  setTextureData(tensorId: Tensor.Id, textureData: TextureData): void {
    Logger.verbose('WebGLSessionHandler', 'Storing Texture data in cache');
    this.textureDataCache.set(tensorId, textureData);
  }
  dispose(): void {
    this.programManager.dispose();
    this.textureHelper.clearActiveTextures();
    this.textureDataCache.forEach(td => this.textureHelper.releaseTexture(td));
    this.textureDataCache = new Map();
  }
  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>): Operator {
    const op = resolveOperator(node, opsets, WEBGL_OP_RESOLVE_RULES);
    op.initialize(node.attributes);
    return op;
  }
}

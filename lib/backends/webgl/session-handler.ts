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
  textureDataCache: Map<Tensor, TextureData>;
  initializers: Set<Tensor>;

  constructor(public readonly backend: WebGLBackend, public readonly context: Session.Context) {
    this.programManager = new ProgramManager(this.context.profiler, backend.glContext);
    this.layoutStrategy = new AlwaysKeepOriginalSizeStrategy(backend.glContext.maxTextureSize);
    this.textureHelper = new TextureHelper(backend.glContext, this.layoutStrategy, this.context.profiler);
    this.textureDataCache = new Map();
  }

  createInferenceHandler() {
    return new WebGLInferenceHandler(this);
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
    this.textureDataCache.forEach(td => this.textureHelper.releaseTexture(td));
    this.textureDataCache = new Map();
  }
  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>): Operator {
    const op = resolveOperator(node, opsets, WEBGL_OP_RESOLVE_RULES);
    op.initialize(node.attributes);
    return op;
  }
}

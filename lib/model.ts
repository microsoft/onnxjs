// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {flatbuffers} from 'flatbuffers';
import {onnx} from 'onnx-proto';

import {Graph} from './graph';
import {OpSet} from './opset';
import {onnxruntime} from './ortSchema/ort_generated';
import ortFbs = onnxruntime.experimental.fbs;
import {LongUtil} from './util';

export class Model {
  // empty model
  constructor() {}

  load(buf: Buffer, graphInitializer?: Graph.Initializer, isOrtFormat?: boolean): void {
    if (!isOrtFormat) {
      this.loadFromOnnxFormat(buf, graphInitializer);
    } else {
      this.loadFromOrtFormat(buf, graphInitializer);
    }
  }

  private loadFromOnnxFormat(buf: Buffer, graphInitializer?: Graph.Initializer): void {
    const modelProto = onnx.ModelProto.decode(buf);
    const irVersion = LongUtil.longToNumber(modelProto.irVersion);
    if (irVersion < 3) {
      throw new Error('only support ONNX model with IR_VERSION>=3');
    }

    this._opsets = modelProto.opsetImport.map(i => {
      return {domain: i.domain as string, version: LongUtil.longToNumber(i.version!)};
    });

    this._graph = Graph.from(modelProto.graph!, graphInitializer);
  }

  private loadFromOrtFormat(buf: Buffer, graphInitializer?: Graph.Initializer): void {
    const fb = new flatbuffers.ByteBuffer(buf);
    const ortModel = ortFbs.InferenceSession.getRootAsInferenceSession(fb).model()!;
    const irVersion = LongUtil.longToNumber(ortModel.irVersion());
    if (irVersion < 3) {
      throw new Error('only support ONNX model with IR_VERSION>=3');
    }
    this._opsets = [];
    for (let i = 0; i < ortModel.opsetImportLength(); i++) {
      const opsetId = ortModel.opsetImport(i);
      this._opsets.push({domain: opsetId?.domain() as string, version: LongUtil.longToNumber(opsetId?.version()!)});
    }

    this._graph = Graph.from(ortModel.graph()!, graphInitializer);
  }

  private _graph: Graph;
  get graph(): Graph {
    return this._graph;
  }

  private _opsets: OpSet[];
  get opsets(): ReadonlyArray<OpSet> {
    return this._opsets;
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';
import {Backend, InferenceHandler, SessionHandler} from '../../../../lib/backend';
import {WebGLInferenceHandler} from '../../../../lib/backends/webgl/inference-handler';
import {Profiler} from '../../../../lib/instrument';
import {Tensor} from '../../../../lib/tensor';

let backend: Backend|undefined;
let sessionhandler: SessionHandler|undefined;
let inferenceHandler: InferenceHandler|undefined;

describe('#UnitTest# - textureCache', () => {
  before('Initialize Context', async () => {
    const profiler = Profiler.create();
    backend = await Backend('webgl');
    sessionhandler = backend.createSessionHandler({profiler});
    inferenceHandler = sessionhandler.createInferenceHandler();
  });

  const webglInferenceHandler = inferenceHandler as WebGLInferenceHandler;
  const tensor = new Tensor([2, 3], 'int32');
  const packedTd = webglInferenceHandler.getOrCreateTextureData(tensor, undefined, true);
  const unpackedTd = webglInferenceHandler.getOrCreateTextureData(tensor, undefined, false);
  webglInferenceHandler.setTextureData(tensor.dataId, packedTd, true);
  webglInferenceHandler.setTextureData(tensor.dataId, unpackedTd, false);
  // make sure packed texture data is not overwritten.
  expect(webglInferenceHandler.getTextureData(tensor.dataId, true)?.isPacked === true);
});

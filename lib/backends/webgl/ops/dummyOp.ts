// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {DummyOp} from '../../../ops/dummyOp';
import {Tensor} from '../../../tensor';
// import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
// import {TextureLayout} from '../types';
// import {getPackedShape} from '../utils';

export class WebGLDummyOp extends DummyOp {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inputs;
  }
}

// export function reshape(
//     inferenceHandler: WebGLInferenceHandler, input: Tensor, reshapedDims: ReadonlyArray<number>): Tensor {
//   const inputTD = inferenceHandler.getOrCreateTextureData(input);
//   let packedShape = reshapedDims;
//   if (inputTD.channels === 4) {
//     packedShape = getPackedShape(reshapedDims);
//   }
//   const newTextureLayout: TextureLayout = {
//     channels: inputTD.channels,
//     height: inputTD.height,
//     width: inputTD.width,
//     // handle reshaping into scalar Tensors
//     shape: packedShape.length !== 0 ? packedShape : [1],
//     strides: ShapeUtil.computeStrides(packedShape),
//     unpackedShape: reshapedDims,
//   };

//   const newTextureData = inferenceHandler.createSharedTextureData(newTextureLayout, input.type, inputTD.texture);
//   return newTextureData.tensor;
// }

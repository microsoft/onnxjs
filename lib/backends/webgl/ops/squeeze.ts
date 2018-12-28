// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Squeeze} from '../../../ops/squeeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {getPackedShape} from '../utils';

export class WebGLSqueeze extends Squeeze {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const inputDimensions = inputs[0].dims;

    const outputDims = new Array<number>();

    for (let i = 0; i < inputDimensions.length; i++) {
      if (this.axes.length > 0) {
        if (this.axes.indexOf(i) === -1) {  // not in squeeze list
          outputDims.push(inputDimensions[i]);
        } else {  // in squeeze list
          if (inputDimensions[i] !== 1) {
            throw new Error(`squeeze an axis of size different than 1`);
          }
        }
      } else {  // any axis with size=1 is squeezed
        if (inputDimensions[i] > 1) {
          outputDims.push(inputDimensions[i]);
        }
      }
    }

    const reshapedDims = outputDims;
    const inputTD = inferenceHandler.getOrCreate(inputs[0]);
    const isInitializer = inferenceHandler.session.isInitializer(inputs[0]);
    let packedShape: ReadonlyArray<number> = reshapedDims;
    if (inputTD.channels === 4) {
      packedShape = getPackedShape(reshapedDims);
    }
    const newTD = {
      channels: inputTD.channels,
      dataType: inputs[0].type,
      texture: inputTD.texture,
      height: inputTD.height,
      width: inputTD.width,
      shape: packedShape,
      strides: ShapeUtil.computeStrides(packedShape),
      unpackedShape: reshapedDims,
      arrayType: inputTD.arrayType
    };
    const newTensor = new Tensor(newTD.unpackedShape, newTD.dataType, (id: Tensor.Id) => {
      const values = inferenceHandler.textureHelper.readTexture(newTD, newTD.dataType, newTD.channels);
      return values;
    });
    if (isInitializer) {
      inferenceHandler.session.setTextureData(newTensor, newTD);
    } else {
      inferenceHandler.setTextureData(newTensor, newTD);
    }
    return [newTensor];
  }
}

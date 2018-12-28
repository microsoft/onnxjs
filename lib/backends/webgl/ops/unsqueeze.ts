// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Unsqueeze} from '../../../ops/unsqueeze';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {WebGLInferenceHandler} from '../inference-handler';
import {getPackedShape} from '../utils';

export class WebGLUnsqueeze extends Unsqueeze {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    const x = inputs[0];

    const inputDimensions = x.dims;

    const outputDims = new Array<number>(x.dims.length + this.axes.length);

    // initialize the array elements to 0
    outputDims.fill(0);

    // set all axes indices to 1 in outputDims and check for duplicates
    for (let i = 0; i < this.axes.length; i++) {
      const axis = this.axes[i];
      if (axis >= outputDims.length) {
        throw new Error(`'axes' has an out of range axis`);
      }
      if (outputDims[axis] !== 0) {
        throw new Error(`'axes' has a duplicate axis`);
      }

      outputDims[axis] = 1;
    }

    // fill in the zero entries of outputDims with the input tensor's shape
    let inputDimsIterator = 0;
    for (let i = 0; i < outputDims.length; i++) {
      if (outputDims[i] === 0) {
        outputDims[i] = inputDimensions[inputDimsIterator++];
      }
    }

    // sanity check assertion. 'inputDimsIterator'
    // should be equal to the length of 'inputDimensions'
    if (inputDimsIterator !== inputDimensions.length) {
      throw new Error('the unsqueezed dimension could not be established');
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

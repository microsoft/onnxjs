// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// import {assert} from 'chai';

import {Reshape} from '../../../ops/reshape';
// import {Upsample} from '../../../ops/upsample';
import {Tensor} from '../../../tensor';
import {ShapeUtil} from '../../../util';
import {getGlsl} from '../glsl-source';
import {WebGLInferenceHandler} from '../inference-handler';
import {ProgramInfo, RunData, WebGLOperator} from '../types';
import {unpackFromChannel} from './packing_utils';
// import {getCoordsDataType} from '../utils';

// import {unpackFromChannel} from './packing_utils';

export class WebGLReshapePacked extends Reshape implements WebGLOperator {
  run(inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    return inferenceHandler.run(this, inputs);
  }
  createProgramInfo(handler: WebGLInferenceHandler, inputs: Tensor[]): ProgramInfo {
    if (inputs.length !== 2) {
      throw new Error(`resize kernel should have input tensor count to 2.`);
    }
    const originInputShape = inputs[0].dims;
    const inputShape3D = processDims3D(inputs[0].dims);
    inputs[0].dims = inputShape3D;
    const inputLayout = handler.getOrCreateTextureLayout2(inputShape3D, 4, true, inputShape3D, true);

    // TODO: double check inputs[1] should not be uploaded
    const outputShape = ShapeUtil.calculateReshapedDims(originInputShape, inputs[1].integerData);
    const squeezedOutputShape = processDims3D(outputShape);

    const outputLayout = handler.createTextureLayoutFromShape(
        squeezedOutputShape, 4, squeezedOutputShape, {isPacked: true, reverseWH: true});

    let mainLoop = ``;
    // TODO: optimize the loop
    for (let i = 0; i < 4; i++) {
      let thisRC = `thisRC = rc;`;
      if (i > 1) {
        thisRC += `thisRC.z += 1;`;
      }
      if (i % 2 === 1) {
        thisRC += `thisRC.y += 1;`;
      }

      mainLoop += `
        ${thisRC}
        ${i > 0 ? `if(thisRC.y < rows && thisRC.z < cols){` : ''}
          int flatIndex = getFlatIndex(thisRC);

          ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          result[${i}] = getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);

        ${i > 0 ? '}' : ''}
      `;
    }
    const glsl = getGlsl(handler.session.backend.glContext.version);
    // const inputShape = inputs[0].dims;
    // const squeezedInputShape = processDims3D(inputShape);

    const shaderSource = `
      ${getReshapedInputCoords(inputShape3D)}
      ${getFlatIndexFrom3D(squeezedOutputShape)}
      ${unpackFromChannel()}
      // testing 6 here
      void main() {
        ivec3 rc = getOutputCoords();

        vec4 result = vec4(0.0);

        ivec3 thisRC;
        int rows = ${squeezedOutputShape[2]};
        int cols = ${squeezedOutputShape[1]};

        ${mainLoop}

        ${glsl.output} = result;
      }
    `;

    return {
      inputLayouts: [inputLayout],
      outputLayout,
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      expectPackedInputs: true,
      expectPackedOutputs: true,
    };
  }
  createRunData(handler: WebGLInferenceHandler, programInfo: ProgramInfo, inputs: Tensor[]): RunData {
    const inputTDs =
        [handler.getOrCreateTextureData(inputs[0], handler.getOrCreateTextureLayout(inputs[0], 1, false, [], false))];
    return {
      inputTextureDatas: inputTDs,
      outputTextureData: handler.createTextureDataFromLayout(programInfo.outputLayout, inputTDs[0].tensor.type),
      uniformData: {}
    };
  }
}

function processDims3D(shpae: readonly number[]|ReadonlyArray<number>|Tensor.IntegerType): [number, number, number] {
  // TODO: squeeze other shapes to 2D case
  const batchDims = shpae.length >= 3 ? shpae.slice(0, shpae.length - 2) : [1];
  let batch = 1;
  for (let i = 0; i < batchDims.length; ++i) {
    batch *= batchDims[i];
  }
  // batchDims.reduce((accumulator, currentValue) => accumulator + currentValue);

  return [batch, shpae.length > 1 ? shpae[shpae.length - 2] : 1, shpae[shpae.length - 1]];
}
function getReshapedInputCoords(shape: [number, number, number]): string {
  // const coordsFromIndexSnippet = shader_util.getLogicalCoordinatesFromFlatIndex(['r', 'c', 'd'], shape);
  const strides = ShapeUtil.computeStrides(shape);
  const coords = ['r', 'c', 'd'];
  const index = 'index';
  const coordsFromIndexSnippet = strides
                                     .map((stride, i) => {
                                       const line1 = `int ${coords[i]} = ${index} / ${stride}`;
                                       const line2 = i === strides.length - 1 ?
                                           `int ${coords[i + 1]} = ${index} - ${coords[i]} * ${stride}` :
                                           `index -= ${coords[i]} * ${stride}`;
                                       return `${line1}; ${line2};`;
                                     })
                                     .join('');

  return `
    ivec3 inputCoordsFromReshapedOutCoords(int index) {
      ${coordsFromIndexSnippet}
      return ivec3(r, c, d);
    }
  `;
}

function getFlatIndexFrom3D(shape: [number, number, number]): string {
  const strides = ShapeUtil.computeStrides(shape);

  return `
  int getFlatIndex(ivec3 coords) {
    // reverse y, z order
    return coords.x * ${strides[0]} + coords.z * ${strides[1]} + coords.y;
  }
`;
}

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

    let mainLoop = ``;
    // TODO: optimize the loop
    for (let i = 0; i < 1; i++) {
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


            if(flatIndex <4){
              vec4 t = getA(0, 0, 2);
              result = t;
            }
            else{
              vec4 t = getA(0, 1, 0);
              result[0] = t[0];
              result[1] = t[1];
              result[2] = t[2];
              result[3] = t[3];
            }

          // ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
          // vec2 inputRCInnerDims = vec2(float(inputRC.y),float(inputRC.z));

          // // reverse inputRC.z and inputRC.y's order as input's width and height is reversed
          // //result[${i}] = getChannel(getA(inputRC.x, inputRC.z, inputRC.y), inputRCInnerDims);
          // vec4 t = getA(inputRC.x, inputRC.z, inputRC.y);
          // result = t;
          // //result[${i}] = float(inputRC.y);
          // //result[${i}] = float(flatIndex);
          // //result[${i}] = t[${i}];

        ${i > 0 ? '}' : ''}
      `;
    }
    const glsl = getGlsl(handler.session.backend.glContext.version);
    const inputShape = inputs[0].dims;
    const squeezedInputShape = processDims3D(inputShape);

    // const outputShape = inputs[1].integerData;  // TODO: double check inputs[1] should not be uploaded
    // const outputShape = inputShape;  // TODO: double check inputs[1] should not be uploaded
    const outputShape = ShapeUtil.calculateReshapedDims(inputs[0].dims, inputs[1].integerData);
    const squeezedOutputShape = processDims3D(outputShape);
    let shaderSource = `
      ${getReshapedInputCoords(squeezedInputShape)}
      ${getFlatIndexFrom3D(squeezedOutputShape)}
      ${unpackFromChannel()}
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

    const outputLayout =
        handler.createTextureLayoutFromShape(outputShape, 4, outputShape, {isPacked: true, reverseWH: true});
    return {
      inputLayouts: [handler.getOrCreateTextureLayout(inputs[0], 4, true, inputs[0].dims, true)],
      outputLayout,
      samplers: ['A'],
      shaderSource,
      hasMain: true,
      expectPackedInputs: true,
      expectPackedoutputs: true,
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
  // TODO: why we need it to be 2D?
  const batchDims = shpae.length > 3 ? shpae.slice(0, shpae.length - 2) : [1];
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

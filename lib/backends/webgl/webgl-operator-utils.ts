// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Tensor} from '../../tensor';

import {GlslPositionalFunction} from './glsl-definitions';
import {WebGLInferenceHandler} from './inference-handler';
import {TextureLayout} from './texture-data';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {getPackedShape} from './utils';
import {PositionalSubOperator, WebGLOperator} from './webgl-operator';

export class WebGLOperatorHelper {
  static run(op: WebGLOperator, inferenceHandler: WebGLInferenceHandler, inputs: Tensor[]): Tensor[] {
    let artifact = inferenceHandler.programManager.getArtifact(op);
    if (!artifact) {
      const programInfo = op.createProgramInfo(inferenceHandler, inputs);
      artifact = inferenceHandler.programManager.build(programInfo);
      inferenceHandler.programManager.setArtifact(op, artifact);
    }
    const runData = op.createRunData(inferenceHandler, artifact.programInfo, inputs);
    inferenceHandler.programManager.run(artifact, runData);
    return [inferenceHandler.getTensor(runData.outputTextureData)];
  }
  static getFinalLayout(
      inferenceHandler: WebGLInferenceHandler, positionalSubFunctions: GlslPositionalFunction[], outputShape: number[],
      channels: number, prefs?: WidthHeightPrefs): TextureLayout {
    let finalShape = outputShape;
    if (positionalSubFunctions.length > 0) {
      finalShape = positionalSubFunctions[positionalSubFunctions.length - 1].outputShape;
    }
    return channels === 4 ?
        inferenceHandler.createBasicTextureLayout(getPackedShape(finalShape), channels, finalShape, prefs) :
        inferenceHandler.createBasicTextureLayout(finalShape, 1, finalShape, prefs);
  }
  static getPositionalFunctions(
      inferenceHandler: WebGLInferenceHandler, subOperators: PositionalSubOperator[],
      outputShape: number[]): GlslPositionalFunction[] {
    let shape = outputShape;
    if (subOperators && subOperators.length > 0) {
      const result = new Array<GlslPositionalFunction>(subOperators.length);
      subOperators.forEach((sub, i) => {
        result[i] = sub.getPositionalFunction(inferenceHandler, shape);
        shape = result[i].outputShape;
      });
      return result;
    }
    return [];
  }
}

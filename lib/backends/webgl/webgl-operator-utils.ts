// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {GlslPositionalFunction} from './glsl-definitions';
import {WebGLInferenceHandler} from './inference-handler';
import {WidthHeightPrefs} from './texture-layout-strategy';
import {PositionalSubOperator, TextureLayout} from './types';
import {getPackedShape} from './utils';

export class WebGLOperatorHelper {
  static getFinalLayout(
      inferenceHandler: WebGLInferenceHandler, positionalSubFunctions: GlslPositionalFunction[],
      outputShape: ReadonlyArray<number>, channels: number, prefs?: WidthHeightPrefs): TextureLayout {
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
      outputShape: ReadonlyArray<number>): GlslPositionalFunction[] {
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

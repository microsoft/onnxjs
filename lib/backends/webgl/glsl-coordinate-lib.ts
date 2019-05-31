// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {GlslContext, GlslLib, GlslLibRoutine} from './glsl-definitions';
import {getGlsl} from './glsl-source';

/**
 * GLSL Library responsible for data types and routines for manipulating
 * coordinates and mapping to/from tensor indices
 */
export class CoordsGlslLib extends GlslLib {
  returnType: string;

  constructor(context: GlslContext) {
    super(context);
  }
  getFunctions(): {[name: string]: GlslLibRoutine;} {
    return {...this.offsetToCoords(), ...this.coordsToOffset(), ...this.toVec(), ...this.valueFrom()};
  }
  getCustomTypes() {
    return {};
  }
  /**
   * Produces a function that can map from
   * 2D normalzied coordinates (s,t) to a flat offset
   */
  protected offsetToCoords(): {[name: string]: GlslLibRoutine} {
    const funcName = `offsetToCoords`;
    return {
      offsetToCoords: new GlslLibRoutine(`
      vec2 ${funcName}(int offset, int width, int height) {
        int t = offset / width;
        int s = offset - t*width;
        vec2 coords = (vec2(s,t) + vec2(0.5,0.5)) / vec2(width, height);
        return coords;
      }
      `)
    };
  }
  /**
   * Produces a function that can map from
   * 2D normalzied coordinates (s,t) to a flat offset
   */
  protected coordsToOffset(): {[name: string]: GlslLibRoutine} {
    const funcName = `coordsToOffset`;
    return {
      coordsToOffset: new GlslLibRoutine(`
      int ${funcName}(vec2 coords, int width, int height) {
        float s = coords.s * float(width);
        float t = coords.t * float(height);
        int offset = int(t) * width + int(s);
        return offset;
      }
      `)
    };
  }
  /**
   * This is the main function to map from the given texture coordiantes (s,t)
   * to logical indices for the output
   * There will only be one single variation of this
   * Also see coordsToOffset and offsetToIndices for input-specific versions
   */
  protected toVec(): {[name: string]: GlslLibRoutine;} {
    const output = this.context.programInfo.outputLayout;
    const rank = output.shape.length;
    const strides = output.strides;
    const xScale = output.width;
    const yScale = output.height;

    const stridesBlock = [];
    for (let i = 0; i < rank - 1; ++i) {
      stridesBlock.push(`
        c[${i}] = offset / ${strides[i]};`);
      stridesBlock.push(`
        offset -= c[${i}] * ${strides[i]};`);
    }
    stridesBlock.push(`
        c[${rank - 1}] = offset;`);
    const body = `
      void toVec(vec2 texCoords, out int c[${rank}]) {
        int offset = coordsToOffset(texCoords, ${xScale}, ${yScale});
        ${stridesBlock.join('')}
      }
      void toVec(int offset, out int c[${rank}]) {
        ${stridesBlock.join('')}
      }
    `;
    return {toVec: new GlslLibRoutine(body, ['coordinates.coordsToOffset'])};
  }
  /**
   * These are value getter functions generated for each input
   * Each function is hardwired to the name and dimensions of the input
   * An '_T' variation is also produced which accesses values as if the
   * input was transposed
   */
  protected valueFrom(): {[name: string]: GlslLibRoutine} {
    const programInfo = this.context.programInfo;
    const result: {[name: string]: GlslLibRoutine} = {};
    this.context.programInfo.samplers.forEach((name, i) => {
      const layout = programInfo.inputLayouts[i];
      const shape = layout.shape;
      const rank = shape.length;
      let funcName = `_${name}`;
      result[funcName] = new GlslLibRoutine(
          this.getValueFromSingle(name, rank, layout.width, layout.height, false),
          [`shapeUtils.indicesToOffset${funcName}`, `coordinates.offsetToCoords`, `fragcolor.getColorAsFloat`]);
      funcName = funcName + '_T';
      result[funcName] = new GlslLibRoutine(
          this.getValueFromSingle(name, rank, layout.width, layout.height, true),
          [`shapeUtils.indicesToOffset${funcName}`, `coordinates.offsetToCoords`, `fragcolor.getColorAsFloat`]);
    });
    return result;
  }
  /**
   * Produces one value getter function for the name and rank given
   * If a transpose is set proper offsetToCoords mapping will be used
   * @param name name of the function
   * @param rank rank of the input
   * @param transpose whether or not should generate a transpose variation
   */
  protected getValueFromSingle(varName: string, rank: number, width: number, height: number, transpose: boolean):
      string {
    let name = `_${varName}`;
    if (transpose) {
      name = name + '_T';
    }
    const glsl = getGlsl(this.context.glContext.version);
    return `
        float ${name}(int m[${rank}]) {
          int offset = indicesToOffset${name}(m);
          vec2 coords = offsetToCoords(offset, ${width}, ${height});
          float value = getColorAsFloat(${glsl.texture2D}(${varName}, coords));
          return value;
        }
        `;
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {ArrayUtil, BroadcastUtil, ShapeUtil} from '../../util';

import {GlslContext, GlslLib, GlslLibRoutine} from './glsl-definitions';
import {getGlsl} from './glsl-source';
import {TextureLayout} from './types';
import {generateShaderFuncNameFromInputSamplerName} from './utils';
import {generateShaderFuncNameFromInputSamplerNameAtOutCoords,} from './utils';
import {getCoordsDataType, getSqueezedParams, squeezeInputShape} from './utils';
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
    return {
      ...this.offsetToCoords(),
      ...this.coordsToOffset(),
      ...this.toVec(),
      ...this.valueFrom(),
      // TODO return these only when packing is enabled.
      ...this.GetCommonPackedUtilFuncs(),
      ...this.getPackedInputsSamplingSnippets(),
      ...this.getPackedOutputSamplingSnippet()
    };
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
   * Generates code for output sampler.
   */
  protected getPackedOutputSamplingSnippet(): {[name: string]: GlslLibRoutine;} {
    const outputLayout = this.context.programInfo.outputLayout;
    const outShape = outputLayout.unpackedShape;
    const outTexShape = [outputLayout.width, outputLayout.height];
    const result: {[name: string]: GlslLibRoutine} = {};
    const funcName = 'getOutputCoords';
    switch (outShape.length) {
      case 0:
        result[funcName] = this.getOutputScalarCoords();
        break;
      case 1:
        result[funcName] = this.getOutputPacked1DCoords(outShape as [number], outTexShape as [number, number]);
        break;
      case 2:
        result[funcName] = this.getOutputPacked2DCoords(outShape as [number, number], outTexShape as [number, number]);
        break;
      case 3:
        result[funcName] =
            this.getOutputPacked3DCoords(outShape as [number, number, number], outTexShape as [number, number]);
        break;
      default:
        result[funcName] = this.getOutputPackedNDCoords(outShape, outTexShape as [number, number]);
    }
    const glsl = getGlsl(this.context.glContext.version);
    // TODO we need this to properly return a packed vec4 from kernels.
    // Replace all '{glsl.output} = result' with 'setOutput(result)' in all kernels.
    const floatTextureSetRGBASource = `
      void setOutput(vec4 val) {
        ${glsl.output} = val;
      }
    `;
    const floatTextureSetRGBAFuncName = 'floatTextureSetRGBA';
    result[floatTextureSetRGBAFuncName] = new GlslLibRoutine(floatTextureSetRGBASource);
    return result;
  }

  /**
   * Scalar output coordinates.
   */
  protected getOutputScalarCoords(): GlslLibRoutine {
    return new GlslLibRoutine(`
      int getOutputCoords() {
        return 0;
      }
    `);
  }

  /**
   * 1D output coordinates.
   */
  protected getOutputPacked1DCoords(shape: [number], texShape: [number, number]): GlslLibRoutine {
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    let source = '';
    if (packedTexShape[0] === 1) {
      source = `
          int getOutputCoords() {
            return 2 * int(TexCoords.x * ${packedTexShape[1]}.0);
          }
        `;
      return new GlslLibRoutine(source);
    }

    if (packedTexShape[1] === 1) {
      source = `
          int getOutputCoords() {
            return 2 * int(TexCoords.y * ${packedTexShape[0]}.0);
          }
        `;
      return new GlslLibRoutine(source);
    }

    source = `
        int getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.yx *
                                 vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
          return 2 * (resTexRC.x * ${packedTexShape[1]} + resTexRC.y);
        }
      `;
    return new GlslLibRoutine(source);
  }

  /**
   * 2D output coordinates.
   */
  protected getOutputPacked2DCoords(shape: [number, number], texShape: [number, number]): GlslLibRoutine {
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    let source = '';
    if (ArrayUtil.arraysEqual(shape, texShape)) {
      source = `
        ivec2 getOutputCoords() {
          return 2 * ivec2(TexCoords.yx * vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
        }
      `;
      return new GlslLibRoutine(source);
    }

    // texels needed to accommodate a logical row
    const texelsInLogicalRow = Math.ceil(shape[1] / 2);

    /**
     * getOutputCoords
     *
     * resTexRC: The rows and columns of the texels. If you move over one
     * texel to the right in the packed texture, you are moving over one column
     * (not two).
     *
     * index: The texel index
     */
    source = `
        ivec2 getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.yx *
                                vec2(${packedTexShape[0]}, ${packedTexShape[1]}));

          int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;
          int r = 2 * (index / ${texelsInLogicalRow});
          int c = imod(index, ${texelsInLogicalRow}) * 2;

          return ivec2(r, c);
        }
      `;
    return new GlslLibRoutine(source);
  }

  /**
   * 3D output coordinates.
   */
  protected getOutputPacked3DCoords(shape: [number, number, number], texShape: [number, number]): GlslLibRoutine {
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const texelsInLogicalRow = Math.ceil(shape[2] / 2);
    const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[1] / 2);
    const source = `
        ivec3 getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.yx *
                                vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
          int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

          int b = index / ${texelsInBatch};
          index -= b * ${texelsInBatch};

          int r = 2 * (index / ${texelsInLogicalRow});
          int c = imod(index, ${texelsInLogicalRow}) * 2;

          return ivec3(b, r, c);
        }
      `;
    return new GlslLibRoutine(source);
  }
  /**
   * ND output coordinates.
   */
  protected getOutputPackedNDCoords(shape: ReadonlyArray<number>, texShape: [number, number]): GlslLibRoutine {
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];

    const texelsInLogicalRow = Math.ceil(shape[shape.length - 1] / 2);
    const texelsInBatch = texelsInLogicalRow * Math.ceil(shape[shape.length - 2] / 2);
    let texelsInBatchN = texelsInBatch;
    let batches = ``;
    let coords = 'b, r, c';

    for (let b = 2; b < shape.length - 1; b++) {
      texelsInBatchN *= shape[shape.length - b - 1];
      batches = `
      int b${b} = index / ${texelsInBatchN};
      index -= b${b} * ${texelsInBatchN};
    ` + batches;
      coords = `b${b}, ` + coords;
    }
    const source = `
      ivec${shape.length} getOutputCoords() {
        ivec2 resTexRC = ivec2(TexCoords.yx *
                              vec2(${packedTexShape[0]}, ${packedTexShape[1]}));
        int index = resTexRC.x * ${packedTexShape[1]} + resTexRC.y;

        ${batches}

        int b = index / ${texelsInBatch};
        index -= b * ${texelsInBatch};

        int r = 2 * (index / ${texelsInLogicalRow});
        int c = imod(index, ${texelsInLogicalRow}) * 2;

        return ivec${shape.length}(${coords});
      }
    `;
    return new GlslLibRoutine(source);
  }

  /**
   * Generates code for common packed coord computation utility functions.
   */
  protected GetCommonPackedUtilFuncs(): {[name: string]: GlslLibRoutine;} {
    const result: {[name: string]: GlslLibRoutine} = {};
    let funcName = `Sampler1DSnippet`;
    result[funcName] = new GlslLibRoutine(`
      vec2 packedUVfrom1D(int texNumR, int texNumC, int index) {
        int texelIndex = index / 2;
        int texR = texelIndex / texNumC;
        int texC = texelIndex - texR * texNumC;
        return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
      }
      `);
    funcName = `Sampler2DSnippet`;
    result[funcName] = new GlslLibRoutine(`
      vec2 packedUVfrom2D(int texelsInLogicalRow, int texNumR,
        int texNumC, int row, int col) {
        int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
        int texR = texelIndex / texNumC;
        int texC = texelIndex - texR * texNumC;
        return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
      }
      `);
    funcName = `Sampler3DSnippet`;
    result[funcName] = new GlslLibRoutine(`
      vec2 packedUVfrom3D(int texNumR, int texNumC,
          int texelsInBatch, int texelsInLogicalRow, int b,
          int row, int col) {
        int index = b * texelsInBatch + (row / 2) * texelsInLogicalRow + (col / 2);
        int texR = index / texNumC;
        int texC = index - texR * texNumC;
        return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
      }
      `);
    return result;
  }

  /**
   * Constructing snippets for packed inputs
   */
  protected getPackedInputsSamplingSnippets(): {[name: string]: GlslLibRoutine;} {
    const result: {[name: string]: GlslLibRoutine} = {};
    const outputLayout = this.context.programInfo.outputLayout;
    this.context.programInfo.inputLayouts.forEach((inputLayout, i) => {
      const name = this.context.programInfo.samplers[i];
      const funcName = generateShaderFuncNameFromInputSamplerName(name);
      result[funcName] = this.getPackedSamplerFromInput(funcName, name, inputLayout);

      const outCoordFuncName = generateShaderFuncNameFromInputSamplerNameAtOutCoords(name);
      if (inputLayout.unpackedShape.length <= outputLayout.unpackedShape.length) {
        result[outCoordFuncName] =
            this.getPackedSamplerAtOutputCoords(outCoordFuncName, inputLayout, outputLayout, name);
      }
    });

    return result;
  }

  /**
   * Constructing snippets for output coordinates of samplers
   */
  protected getPackedSamplerAtOutputCoords(
      funcName: string, inputLayout: TextureLayout, outputLayout: TextureLayout, name: string): GlslLibRoutine {
    const inShape = inputLayout.unpackedShape;
    const outShape = outputLayout.unpackedShape;
    const texName = name;
    const texFuncSnippet = generateShaderFuncNameFromInputSamplerName(texName);

    const inRank = inShape.length;
    const outRank = outShape.length;

    const broadcastDims = BroadcastUtil.getBroadcastDims(inShape, outShape);

    const type = getCoordsDataType(outRank);
    const rankDiff = outRank - inRank;
    let coordsSnippet: string;
    const fields = ['x', 'y', 'z', 'w', 'u', 'v'];

    if (inRank === 0) {
      coordsSnippet = '';
    } else if (outRank < 2 && broadcastDims.length >= 1) {
      coordsSnippet = 'coords = 0;';
    } else {
      coordsSnippet = broadcastDims.map(d => `coords.${fields[d + rankDiff]} = 0;`).join('\n');
    }
    let unpackedCoordsSnippet = '';
    if (outRank < 2 && inRank > 0) {
      unpackedCoordsSnippet = 'coords';
    } else {
      unpackedCoordsSnippet = inShape.map((s, i) => `coords.${fields[i + rankDiff]}`).join(', ');
    }

    let output = `return outputValue;`;
    const inSize = ShapeUtil.size(inShape);
    const isInputScalar = inSize === 1;
    const outSize = ShapeUtil.size(outShape);
    const isOutputScalar = outSize === 1;

    if (inRank === 1 && !isInputScalar && !isOutputScalar) {
      output = `
        return vec4(outputValue.xy, outputValue.xy);
      `;
    } else if (isInputScalar && !isOutputScalar) {
      if (outRank === 1) {
        output = `
          return vec4(outputValue.x, outputValue.x, 0., 0.);
        `;
      } else {
        output = `
          return vec4(outputValue.x);
        `;
      }
    } else if (broadcastDims.length) {
      const rows = inRank - 2;
      const cols = inRank - 1;

      if (broadcastDims.indexOf(rows) > -1 && broadcastDims.indexOf(cols) > -1) {
        output = `return vec4(outputValue.x);`;
      } else if (broadcastDims.indexOf(rows) > -1) {
        output = `return vec4(outputValue.x, outputValue.y, ` +
            `outputValue.x, outputValue.y);`;
      } else if (broadcastDims.indexOf(cols) > -1) {
        output = `return vec4(outputValue.xx, outputValue.zz);`;
      }
    }
    const source = `
      vec4 ${funcName}() {
        ${type} coords = getOutputCoords();
        ${coordsSnippet}
        vec4 outputValue = get${texFuncSnippet}(${unpackedCoordsSnippet});
        ${output}
      }
    `;
    return new GlslLibRoutine(source);
  }
  /**
   * Constructing snippets for packed operations.
   */
  protected getPackedSamplerFromInput(funcName: string, name: string, inputLayout: TextureLayout): GlslLibRoutine {
    switch (inputLayout.unpackedShape.length) {
      case 0:
        return this.getPackedSamplerScalar(funcName, name);
      case 1:
        return this.getPackedSampler1D(funcName, name, inputLayout);
      case 2:
        return this.getPackedSampler2D(funcName, name, inputLayout);
      case 3:
        return this.getPackedSampler3D(funcName, name, inputLayout);
      default:
        return this.getPackedSamplerND(funcName, name, inputLayout);
    }
  }

  /**
   * Packed scalar snippet.
   */
  protected getPackedSamplerScalar(funcName: string, name: string): GlslLibRoutine {
    const glsl = getGlsl(this.context.glContext.version);
    const source = `
          vec4 ${funcName}() {
            return ${glsl.texture2D}(${name}, halfCR);
          }
        `;
    return new GlslLibRoutine(source);
  }

  /**
   * Packed 1D snippet.
   */
  protected getPackedSampler1D(funcName: string, name: string, inputLayout: TextureLayout): GlslLibRoutine {
    const texShape = [inputLayout.width, inputLayout.height];
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const glsl = getGlsl(this.context.glContext.version);
    const source = `
          vec4 ${funcName}(int index) {
            vec2 uv = packedUVfrom1D(
              ${packedTexShape[0]}, ${packedTexShape[1]}, index);
            return ${glsl.texture2D}(${name}, uv);
          }
        `;
    return new GlslLibRoutine(source);
  }

  /**
   * Packed 2D snippet.
   */
  protected getPackedSampler2D(funcName: string, name: string, inputLayout: TextureLayout): GlslLibRoutine {
    const shape = inputLayout.unpackedShape;
    const texShape = [inputLayout.width, inputLayout.height];
    const glsl = getGlsl(this.context.glContext.version);
    const texNumR = texShape[0];
    const texNumC = texShape[1];

    if (texShape != null && ArrayUtil.arraysEqual(shape, texShape)) {
      return new GlslLibRoutine(`
        vec4 ${funcName}(int row, int col) {
          vec2 uv = (vec2(col, row) + halfCR) / vec2(${texNumC}.0, ${texNumR}.0);

          return ${glsl.texture2D}(${name}, uv);
        }
      `);
    }
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const valuesPerRow = Math.ceil(shape[1] / 2);
    const source = `
        vec4 ${funcName}(int row, int col) {
          vec2 uv = packedUVfrom2D(${valuesPerRow}, ${packedTexShape[0]}, ${packedTexShape[1]}, row, col);
          return ${glsl.texture2D}(${name}, uv);
        }
      `;
    return new GlslLibRoutine(source);
  }

  /**
   * Packed 3D snippet.
   */
  protected getPackedSampler3D(funcName: string, name: string, inputLayout: TextureLayout): GlslLibRoutine {
    const shape = inputLayout.unpackedShape;
    const texShape = [inputLayout.width, inputLayout.width];
    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const glsl = getGlsl(this.context.glContext.version);

    if (shape[0] === 1) {
      const squeezedShape = shape.slice(1);
      const keptDims = [1, 2];
      const newInputShape = squeezeInputShape(shape, squeezedShape);
      const params = ['b', 'row', 'col'];
      // Deep copy of input texture layout.
      const newInputLayout: TextureLayout = JSON.parse(JSON.stringify(inputLayout));
      newInputLayout.unpackedShape = newInputShape;
      const source = `
              ${this.getPackedSamplerFromInput(funcName, name, newInputLayout)}
              vec4 ${funcName}(int b, int row, int col) {
                return ${funcName}(${getSqueezedParams(params, keptDims)});
              }
            `;
      return new GlslLibRoutine(source);
    }
    const texNumR = packedTexShape[0];
    const texNumC = packedTexShape[1];

    const valuesPerRow = Math.ceil(shape[2] / 2);
    const texelsInBatch = valuesPerRow * Math.ceil(shape[1] / 2);

    const source = `
      vec4 ${funcName}(int b, int row, int col) {
        vec2 uv = packedUVfrom3D(
          ${texNumR}, ${texNumC}, ${texelsInBatch}, ${valuesPerRow}, b, row, col);
        return ${glsl.texture2D}(${name}, uv);
      }
    `;
    return new GlslLibRoutine(source);
  }

  /**
   * Packed ND snippet.
   */
  protected getPackedSamplerND(funcName: string, name: string, inputLayout: TextureLayout): GlslLibRoutine {
    const shape = inputLayout.unpackedShape;
    const rank = shape.length;
    const texShape = [inputLayout.width, inputLayout.height];
    const glsl = getGlsl(this.context.glContext.version);

    const packedTexShape = [Math.ceil(texShape[0] / 2), Math.ceil(texShape[1] / 2)];
    const texNumR = packedTexShape[0];
    const texNumC = packedTexShape[1];
    const valuesPerRow = Math.ceil(shape[rank - 1] / 2);
    let texelsInBatch = valuesPerRow * Math.ceil(shape[rank - 2] / 2);
    let params = `int b, int row, int col`;
    let index = `b * ${texelsInBatch} + (row / 2) * ${valuesPerRow} + (col / 2)`;
    for (let b = 2; b < rank - 1; b++) {
      params = `int b${b}, ` + params;
      texelsInBatch *= shape[rank - b - 1];
      index = `b${b} * ${texelsInBatch} + ` + index;
    }
    const source = `
        vec4 ${funcName}(${params}) {
          int index = ${index};
          int texR = index / ${texNumC};
          int texC = index - texR * ${texNumC};
          vec2 uv = (vec2(texC, texR) + halfCR) / vec2(${texNumC}, ${texNumR});
          return ${glsl.texture2D}(${name}, uv);
        }
      `;
    return new GlslLibRoutine(source);
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

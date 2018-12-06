// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {GlslContext, GlslLib, GlslLibRoutine} from './glsl-definitions';

/**
 * GLSL Library responsible for data types and routines for manipulating
 * coordinates and mapping to/from tensor indices
 */
export class ShapeUtilsGlslLib extends GlslLib {
  constructor(context: GlslContext) {
    super(context);
  }
  getFunctions(): {[name: string]: GlslLibRoutine;} {
    return {
      ...this.bcastIndex(),
      ...this.bcastMatmulIndex(),
      ...this.offsetToIndices(),
      ...this.indicesToOffset(),
      ...this.incrementIndices()
    };
  }
  getCustomTypes() {
    return {};
  }
  protected bcastIndex(): {[name: string]: GlslLibRoutine;} {
    const programInfo = this.context.programInfo;
    const outputRank = programInfo.outputLayout.shape.length;
    if (outputRank === 0) {
      throw new Error(`No Glsl method inclusion for scalar tensor`);
    }
    const result: {[name: string]: GlslLibRoutine} = {};
    this.context.uniformInfo.filter(ui => ui.type === 'sampler2D').forEach((ui, i) => {
      const name = ui.name;
      const shape = programInfo.inputLayouts[i].shape;
      if (shape.length <= outputRank) {
        const rank = shape.length;
        const dimOffset = outputRank - rank;
        const funcName = `bcastIndices_${name}`;
        let block = '';
        for (let i = 0; i < rank; ++i) {
          block += `
          realIndices[${i}] = int( mod(float(bcastedIndices[${dimOffset + i}]), ${shape[i]}.0) );
          `;
        }
        const body = `
        void ${funcName} (int bcastedIndices[${outputRank}], out int realIndices[${rank}]) {
          ${block}
        }
        `;
        result[funcName] = new GlslLibRoutine(body);
      }
    });
    return result;
  }
  protected bcastMatmulIndex(): {[name: string]: GlslLibRoutine;} {
    const programInfo = this.context.programInfo;
    const outputRank = programInfo.outputLayout.shape.length;
    //    if (outputRank === 0) {
    //      throw new Error(`No Glsl method inclusion for scalar tensor`);
    //    }
    const result: {[name: string]: GlslLibRoutine} = {};
    this.context.uniformInfo.filter(ui => ui.type === 'sampler2D').forEach((ui, i) => {
      const name = ui.name;
      const shape = programInfo.inputLayouts[i].shape;
      if (!(shape.length < 2 || shape.length > outputRank)) {
        const rank = shape.length;
        const dimOffset = outputRank - rank;
        const funcName = `bcastMatmulIndices_${name}`;
        let block = '';
        for (let i = 0; i < rank - 2; ++i) {
          block += `
          realIndices[${i}] = int( mod(float(bcastedIndices[${dimOffset + i}]), ${shape[i]}.0) );
          `;
        }
        const body = `
        void ${funcName}(int bcastedIndices[${outputRank}], out int realIndices[${rank}]) {
          ${block}
          realIndices[${rank - 1}] = bcastedIndices[${outputRank - 1}];
          realIndices[${rank - 2}] = bcastedIndices[${outputRank - 2}];
        }
        `;
        result[funcName] = new GlslLibRoutine(body);
      }
    });
    return result;
  }
  protected indicesToOffset(): {[name: string]: GlslLibRoutine;} {
    const programInfo = this.context.programInfo;
    const result: {[name: string]: GlslLibRoutine} = {};
    this.context.uniformInfo.filter(ui => ui.type === 'sampler2D').forEach((ui, i) => {
      const name = ui.name;
      const shape = programInfo.inputLayouts[i].shape;
      const strides = programInfo.inputLayouts[i].strides;
      const rank = shape.length;
      //      if (rank === 0) {
      //        throw new Error(`No Glsl method inclusion for scalar tensor`);
      //      }
      let funcName = `indicesToOffset_${name}`;
      result[funcName] = new GlslLibRoutine(ShapeUtilsGlslLib.indexToOffsetSingle(funcName, rank, strides));
      funcName = `indicesToOffset_${name}_T`;
      result[funcName] =
          new GlslLibRoutine(ShapeUtilsGlslLib.indexToOffsetSingle(funcName, rank, strides.slice().reverse()));
    });
    return result;
  }
  static indexToOffsetSingle(name: string, rank: number, strides: ReadonlyArray<number>): string {
    //    if (rank === 0) {
    //      throw new Error(`No Glsl method inclusion for scalar tensor`);
    //    }
    let block = '';
    for (let i = rank - 1; i >= 0; --i) {
      block += `
        offset += indices[${i}] * ${strides[i]};
        `;
    }
    return `
      int ${name}(int indices[${rank}]) {
        int offset = 0;
        ${block}
        return offset;
      }
      `;
  }
  protected offsetToIndices(): {[name: string]: GlslLibRoutine;} {
    const programInfo = this.context.programInfo;
    const result: {[name: string]: GlslLibRoutine} = {};
    this.context.uniformInfo.filter(ui => ui.type === 'sampler2D').forEach((ui, i) => {
      const name = ui.name;
      const shape = programInfo.inputLayouts[i].shape;
      const strides = programInfo.inputLayouts[i].strides;
      const rank = shape.length;
      //      if (rank === 0) {
      //        throw new Error(`No Glsl method inclusion for scalar tensor`);
      //      }
      let funcName = `offsetToIndices_${name}`;
      result[funcName] = new GlslLibRoutine(ShapeUtilsGlslLib.offsetToIndicesSingle(funcName, rank, strides));
      funcName = `offsetToIndices_${name}_T`;
      result[funcName] =
          new GlslLibRoutine(ShapeUtilsGlslLib.offsetToIndicesSingle(funcName, rank, strides.slice().reverse()));
    });
    return result;
  }
  static offsetToIndicesSingle(name: string, rank: number, strides: ReadonlyArray<number>): string {
    //    if (rank === 0) {
    //      throw new Error(`No Glsl method inclusion for scalar tensor`);
    //    }
    const stridesBlock = [];
    for (let i = 0; i < rank - 1; ++i) {
      stridesBlock.push(`
      indices[${i}] = offset / ${strides[i]};`);
      stridesBlock.push(`
        offset -= indices[${i}] * ${strides[i]};`);
    }
    stridesBlock.push(`
      indices[${rank - 1}] = offset;`);
    return `
      void ${name}(int offset, out int indices[${rank}]) {
        ${stridesBlock.join('')}
      }
      `;
  }
  protected incrementIndices(): {[name: string]: GlslLibRoutine;} {
    const programInfo = this.context.programInfo;
    const result: {[name: string]: GlslLibRoutine} = {};
    this.context.uniformInfo.filter(ui => ui.type === 'sampler2D').forEach((ui, i) => {
      const name = ui.name;
      const shape = programInfo.inputLayouts[i].shape;
      const rank = shape.length;
      //      if (rank === 0) {
      //        throw new Error(`No Glsl method inclusion for scalar tensor`);
      //      }
      const funcName = `incrementIndices_${name}`;
      let shapeInit = '';
      for (let i = 0; i < rank; ++i) {
        shapeInit += `
        shape[${i}] = ${shape[i]};`;
      }
      const body = `
        void ${funcName}(int axis, out int indices[${rank}]) {
          int shape[${rank}];
          ${shapeInit};
          for(int i = ${rank} -1 ; i >= 0; --i) {
            if(i > axis) continue;
            indices[i] += 1;
            if(indices[i] < shape[i]) {
              break;
            }
            indices[i] = 0;
          }
        }
        `;
      result[funcName] = new GlslLibRoutine(body);
    });
    return result;
  }
}

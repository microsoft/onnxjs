// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger} from '../../instrument';

/** Layout preferences */
export interface WidthHeightPrefs {
  breakAxis: number;
}
/**
 * TextureLayoutStrategy is an abstraction for different plans
 * for mapping n-dimensional arrays to 2D textures (and back)
 */
export interface TextureLayoutStrategy {
  computeTextureWH(shape: ReadonlyArray<number>, prefs?: WidthHeightPrefs): [number, number];
}

/**
 * This strategy try to find the minimal max(W,H) that fulfills (W * H == totalSize)
 */
export class AlwaysKeepOriginalSizeStrategy implements TextureLayoutStrategy {
  constructor(public maxTextureSize: number) {}
  computeTextureWH(shape: ReadonlyArray<number>, prefs?: WidthHeightPrefs): [number, number] {
    // scalar tensor
    if (shape.length === 0) {
      return [1, 1];
    }
    const maxTextureSize = this.maxTextureSize;
    if (prefs) {
      // check to see if dims fit
      const wsize = prefs.breakAxis >= shape.length ? 1 : shape.slice(prefs.breakAxis).reduce((a, b) => a * b);
      const hsize = prefs.breakAxis <= 0 ? 1 : shape.slice(0, prefs.breakAxis).reduce((a, b) => a * b);
      if (wsize > maxTextureSize || hsize > maxTextureSize) {
        // ignore preferences
        // continue with default layout
        Logger.verbose(
            'TextureLayout',
            `Given width/height preferences were unattainable: shape:${shape}, breakAxis:${prefs.breakAxis}`);
      } else {
        return [wsize, hsize];
      }
    }
    const totalSize = shape.reduce((a, b) => a * b);

    let width = Math.floor(Math.sqrt(totalSize));

    for (; width < maxTextureSize && width < totalSize; width++) {
      if (totalSize % width === 0) {
        break;
      }
    }

    if (width >= maxTextureSize || totalSize % width !== 0) {
      throw new Error(`The given dimensions are outside this GPU\'s boundaries: ${shape}`);
    }
    return [width, totalSize / width];
  }
}

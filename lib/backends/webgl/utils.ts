// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

export interface Disposable {
  dispose(): void;
}

export function using<T extends Disposable>(resource: T, func: (resource: T) => void) {
  try {
    func(resource);
  } finally {
    resource.dispose();
  }
}
export function expandArray(shape: ReadonlyArray<number>, newLength: number, fill: number) {
  if (shape.length === newLength) {
    return shape;
  }
  const newShape = new Array(newLength);
  newShape.fill(fill);
  newShape.splice(newLength - shape.length, shape.length, ...shape);
  return newShape;
}
/**
 * Given a non RGBA shape calculate the R version
 * It is assumed that the dimensions are multiples of given channels
 * NOTE: it is always the last dim that gets packed.
 * @param unpackedShape original shape to create a packed version from
 */
export function getPackedShape(unpackedShape: ReadonlyArray<number>): ReadonlyArray<number> {
  const len = unpackedShape.length;
  return unpackedShape.slice(0, len - 1).concat(unpackedShape[len - 1] / 4);
}

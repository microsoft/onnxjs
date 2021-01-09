// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

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

export function repeatedTry(
    checkFn: () => boolean, delayFn = (counter: number) => 0, maxCounter?: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    let tryCount = 0;

    const tryFn = () => {
      if (checkFn()) {
        resolve();
        return;
      }

      tryCount++;

      const nextBackoff = delayFn(tryCount);

      if (maxCounter != null && tryCount >= maxCounter) {
        reject();
        return;
      }
      setTimeout(tryFn, nextBackoff);
    };

    tryFn();
  });
}

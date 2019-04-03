// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

export function base64toBuffer(data: string): Uint8Array {
  return Buffer.from(data, 'base64');
}

export function bufferToBase64(buffer: Uint8Array): string {
  return Buffer.from(buffer).toString('base64');
}

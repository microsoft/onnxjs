// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

export function getVertexShaderSource(): string {
  return `
      precision highp float;
      attribute vec3 position;
      attribute vec2 textureCoord;

      varying vec2 TexCoords;

      void main()
      {
          gl_Position = vec4(position, 1.0);
          TexCoords = textureCoord;
      }`;
}

export function getFragShaderPreamble(): string {
  return `
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    varying vec2 TexCoords;

    `;
}

export function getDefaultFragShaderMain(outputShapeLength: number): string {
  return `
  void main() {
    int indices[${outputShapeLength}];
    toVec(TexCoords, indices);
    vec4 result = vec4(process(indices));
    gl_FragColor = result;
  }
  `;
}

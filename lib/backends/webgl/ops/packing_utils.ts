export function getVecChannels(name: string, rank: number): string[] {
  return ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank).map(d => `${name}.${d}`);
}

export function getChannels(name: string, rank: number): string[] {
  if (rank === 1) {
    return [name];
  }
  return getVecChannels(name, rank);
}

export function getChannelValue(width: number, height: number, texture2D: string, allChannels = false) {
  let channalName = '';
  let returnType = 'vec4 ';

  if (allChannels === false) {
    channalName = '.r ';
    returnType = 'float ';
  }

  return returnType + `getA(int row, int col) {
  vec2 uv = (vec2(col, row) + 0.5) / vec2(${width}, ${height});
    return ${texture2D}(A, uv)${channalName};
  }
  `;
}

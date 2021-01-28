export function getVecChannels(name: string, rank: number): string[] {
  return ['x', 'y', 'z', 'w', 'u', 'v'].slice(0, rank).map(d => `${name}.${d}`);
}

export function getChannels(name: string, rank: number): string[] {
  if (rank === 1) {
    return [name];
  }
  return getVecChannels(name, rank);
}

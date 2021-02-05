export function assert(expr: boolean, msg: () => string) {
  if (!expr) {
    throw new Error(typeof msg === 'string' ? msg : msg());
  }
}

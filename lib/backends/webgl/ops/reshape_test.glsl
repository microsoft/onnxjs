#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
in vec2 TexCoords;
out vec4 outputColor;
const vec2 halfCR = vec2(0.5, 0.5);

// Custom vector types to handle higher dimenalities.
struct ivec5 {
  int x;
  int y;
  int z;
  int w;
  int u;
};

struct ivec6 {
  int x;
  int y;
  int z;
  int w;
  int u;
  int v;
};

int imod(int x, int y) { return x - y * (x / y); }

uniform sampler2D A;

vec2 packedUVfrom2D(int texNumR, int texNumC, int texelsInLogicalRow, int row,
                    int col) {
  int texelIndex = (row / 2) * texelsInLogicalRow + (col / 2);
  int texR = texelIndex / texNumC;
  int texC = texelIndex - texR * texNumC;
  return (vec2(texC, texR) + halfCR) / vec2(texNumC, texNumR);
}

vec4 getA(int row, int col) {
  vec2 uv = packedUVfrom2D(1, 2, 2, row, col);
  return texture(A, uv);
}
vec4 getA(int b, int row, int col) { return getA(row, col); }

ivec3 getOutputCoords() {
  ivec2 resTexRC = ivec2(TexCoords.xy * vec2(1, 2));
  int index = resTexRC.y * 1 + resTexRC.x;

  int b = index / 2;
  index -= b * 2;

  // reverse r and c order for packed texture
  int r = imod(index, 1) * 2;
  int c = 2 * (index / 1);

  return ivec3(b, r, c);
}

ivec3 inputCoordsFromReshapedOutCoords(int index) {
  int r = index / 8;
  index -= r * 8;
  int c = index / 4;
  index -= c * 4;
  int d = index / 1;
  int undefined = index - d * 1;
  return ivec3(r, c, d);
}

int getFlatIndex(ivec3 coords) {
  // reverse y, z order
  return coords.x * 8 + coords.z * 2 + coords.y;
}

float getChannel(vec4 frag, int dim) {
  int modCoord = imod(dim, 2);
  return modCoord == 0 ? frag.r : frag.g;
}

float getChannel(vec4 frag, vec2 innerDims) {
  vec2 modCoord = mod(innerDims, 2.);
  return modCoord.x == 0. ? (modCoord.y == 0. ? frag.r : frag.g)
                          : (modCoord.y == 0. ? frag.b : frag.a);
}

void main() {
  ivec3 rc = getOutputCoords();

  vec4 result = vec4(0.0);

  ivec3 thisRC;
  int rows = 2;
  int cols = 4;

  thisRC = rc;

  int flatIndex = getFlatIndex(thisRC);

  ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
  vec2 inputRCInnerDims = vec2(float(inputRC.y), float(inputRC.z));

  // result[0] = getChannel(getA(inputRC.x, inputRC.y, inputRC.z),
  // inputRCInnerDims);
  float t = getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
  result[0] = float(flatIndex);

  thisRC = rc;
  thisRC.y += 1;
  if (thisRC.y < rows && thisRC.z < cols) {
    int flatIndex = getFlatIndex(thisRC);

    ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
    vec2 inputRCInnerDims = vec2(float(inputRC.y), float(inputRC.z));

    // result[1] = getChannel(getA(inputRC.x, inputRC.y, inputRC.z),
    // inputRCInnerDims);
    float t =
        getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
    result[1] = float(flatIndex);
  }

  thisRC = rc;
  thisRC.z += 1;
  if (thisRC.y < rows && thisRC.z < cols) {
    int flatIndex = getFlatIndex(thisRC);

    ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
    vec2 inputRCInnerDims = vec2(float(inputRC.y), float(inputRC.z));

    // result[2] = getChannel(getA(inputRC.x, inputRC.y, inputRC.z),
    // inputRCInnerDims);
    float t =
        getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
    result[2] = float(flatIndex);
  }

  thisRC = rc;
  thisRC.z += 1;
  thisRC.y += 1;
  if (thisRC.y < rows && thisRC.z < cols) {
    int flatIndex = getFlatIndex(thisRC);

    ivec3 inputRC = inputCoordsFromReshapedOutCoords(flatIndex);
    vec2 inputRCInnerDims = vec2(float(inputRC.y), float(inputRC.z));

    // result[3] = getChannel(getA(inputRC.x, inputRC.y, inputRC.z),
    // inputRCInnerDims);
    float t =
        getChannel(getA(inputRC.x, inputRC.y, inputRC.z), inputRCInnerDims);
    result[3] = float(flatIndex);
  }

  outputColor = result;
}

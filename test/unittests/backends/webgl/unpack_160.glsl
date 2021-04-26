#version 300 es
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    in vec2 TexCoords;
    out vec4 outputColor;
    const vec2 halfCR = vec2(0.5, 0.5);

    // Custom vector types to handle higher dimenalities.
    struct ivec5
    {
      int x;
      int y;
      int z;
      int w;
      int u;
    };

    struct ivec6
    {
      int x;
      int y;
      int z;
      int w;
      int u;
      int v;
    };

    int imod(int x, int y) {
      return x - y * (x / y);
    }


    uniform sampler2D A;
    vec4 getA(int b2, int b, int row, int col) {
      int index = b2 * 3840 + b * 3840 + (row / 2) * 80 + (col / 2);
      int texR = index / 80;
      int texC = index - texR * 80;
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(80, 48);
      return texture(A, uv);
    }

      ivec4 getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.xy *
                                vec2(96, 160));
          int index = resTexRC.y * 96 + resTexRC.x;
          int r = index / 15360; index -= r * 15360;int c = index / 15360; index -= c * 15360;int d = index / 160; int d2 = index - d * 160;
          return ivec4(r, c, d, d2);
        }




    float getChannel(vec4 frag, int dim) {
      int modCoord = imod(dim, 2);
      return modCoord == 0 ? frag.r : frag.g;
    }

    float getChannel(vec4 frag, vec2 innerDims) {
      vec2 modCoord = mod(innerDims, 2.);
      return modCoord.x == 0. ?
        (modCoord.y == 0. ? frag.r : frag.g) :
        (modCoord.y == 0. ? frag.b : frag.a);
    }

        void main() {
          ivec4 rc = getOutputCoords();

          // Sample the texture with the coords to get the rgba channel value.
          vec4 packedInput = getA(rc.x,rc.y,rc.z,rc.w);

          outputColor = vec4(getChannel(packedInput, vec2(rc.z,rc.w)), 0, 0, 0);
        }

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

    vec2 uvFromFlat(int texNumR, int texNumC, int index) {
      int texR = index / texNumC;
      int texC = index - texR * texNumC;
      // TODO: swap texR, texC order in following function so row is corresponding to u and column is corresponding to v.
      return (vec2(texR, texC) + halfCR) / vec2(texNumR, texNumC);
    }


        float sampleTexture(sampler2D textureSampler, vec2 uv) {
            return texture(textureSampler, uv).r;
        }

      int coordsToOffset(vec2 coords, int width, int height) {
        float s = coords.s * float(width);
        float t = coords.t * float(height);
        int offset = int(t) * width + int(s);
        return offset;
      }


        float getA(int row, int col) {
          // Explicitly use integer operations as dot() only works on floats.
          int offset_A = coordsToOffset(TexCoords, 4, 2);
          int index = row * 4 + col + offset_A;
          vec2 uv = uvFromFlat(4, 2, index);
          return sampleTexture(A, uv);
        }


        ivec2 getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.xy *
                                vec2(2, 1));

          int index = resTexRC.y * 1 + resTexRC.x;
          int r = 2 * (index / 1);
          int c = imod(index, 1) * 2;

          return ivec2(r, c);
        }



        void main() {
          // TODO(TJ): implement getOutputCoords() to map input uv to output xy.
          ivec2 rc = getOutputCoords();

          if(rc.x >= 2||rc.y >= 4) {
            outputColor = vec4(0);
          } else {

    int r = rc.x;
    int c = rc.y;
    int rp1 = rc.x + 1;
    int cp1 = rc.y + 1;
    bool rEdge = rp1 >= 2;
    bool cEdge = cp1 >= 4;


    // outputColor = vec4(getA(r, c),
          rEdge ? 0. : getA(rp1, c),
          cEdge ? 0. : getA(r, cp1),
          rEdge || cEdge ? 0. : getA(rp1, cp1));
            vec4 tmp = vec4(getA(r, c),
          rEdge ? 0. : getA(rp1, c),
          cEdge ? 0. : getA(r, cp1),
          rEdge || cEdge ? 0. : getA(rp1, cp1));
            outputColor = vec4(getA(0, 0), getA(0, 1), getA(1, 0), getA(1, 1));
          }
        }

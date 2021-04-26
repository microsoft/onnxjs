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
      int texC = index / texNumR;
      int texR = index - texC * texNumR;
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
          int offset_A = coordsToOffset(TexCoords, 48, 80);
          int index = row * 80 + col + offset_A;
          vec2 uv = uvFromFlat(48, 80, index);
          return sampleTexture(A, uv);
        }

          float getA(int row, int col, int depth, int depth2) {
            return getA(depth, depth2);
          }


      ivec4 getOutputCoords() {
        ivec2 resTexRC = ivec2(TexCoords.xy *
                              vec2(40, 24));
        int index = resTexRC.y * 40 + resTexRC.x;


      int b2 = index / 960;
      index -= b2 * 960;


        int b = index / 960;
        index -= b * 960;

        // reverse r and c order for packed texture
        int r = imod(index, 40) * 2;
        int c = 2 * (index / 40);

        return ivec4(b2, b, r, c);
      }



    // test place holder resize
        void main() {
          ivec4 rc = getOutputCoords();

          if(rc.z >= 80||rc.w >= 48) {
            outputColor = vec4(0);
          } else {

    int r = rc.z;
    int c = rc.w;
    int rp1 = rc.z + 1;
    int cp1 = rc.w + 1;
    bool rEdge = rp1 >= 80;
    bool cEdge = cp1 >= 48;


            outputColor = vec4(getA(rc.x,rc.y,r, c),
          rEdge ? 0. : getA(rc.x,rc.y,rp1, c),
          cEdge ? 0. : getA(rc.x,rc.y,r, cp1),
          rEdge || cEdge ? 0. : getA(rc.x,rc.y,rp1, cp1));
          }
        }

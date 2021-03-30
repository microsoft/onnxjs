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

        float sampleTexture(sampler2D textureSampler, vec2 uv) {
            return texture(textureSampler, uv).r;
        }

          float getA(int row, int col) {
            vec2 uv = (vec2(row, col) + halfCR) / vec2(6.0, 3.0);
            return sampleTexture(A, uv);
          }


        ivec2 getOutputCoords() {
          ivec2 resTexRC = ivec2(TexCoords.xy *
                                vec2(3, 2));

          int index = resTexRC.y * 3 + resTexRC.x;

          // reverse r and c order for packed texture
          int r = imod(index, 3) * 2;
          int c = 2 * (index / 3);

          return ivec2(r, c);
        }



        void main() {
          ivec2 rc = getOutputCoords();

          if(rc.x >= 6||rc.y >= 3) {
            outputColor = vec4(0);
          } else {

    int r = rc.x;
    int c = rc.y;
    int rp1 = rc.x + 1;
    int cp1 = rc.y + 1;
    bool rEdge = rp1 >= 6;
    bool cEdge = cp1 >= 3;


            outputColor = vec4(getA(r, c),
          rEdge ? 0. : getA(rp1, c),
          cEdge ? 0. : getA(r, cp1),
          rEdge || cEdge ? 0. : getA(rp1, cp1));
          }
        }

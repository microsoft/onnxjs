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
      int index = b2 * 960 + b * 960 + (row / 2) * 40 + (col / 2);
      int texR = index / 40;
      int texC = index - texR * 40;
      vec2 uv = (vec2(texC, texR) + halfCR) / vec2(40, 24);
      return texture(A, uv);
    }

      ivec4 getOutputCoords() {
        ivec2 resTexRC = ivec2(TexCoords.xy *
                              vec2(80, 48));
        int index = resTexRC.y * 80 + resTexRC.x;


      int b2 = index / 3840;
      index -= b2 * 3840;


        int b = index / 3840;
        index -= b * 3840;

        // reverse r and c order for packed texture
        int r = imod(index, 80) * 2;
        int c = 2 * (index / 80);

        return ivec4(b2, b, r, c);
      }



        const vec2 inputWH = vec2(48.0, 80.0);
        const vec4 scaleWHWH = vec4(2.0, 2.0, 2.0, 2.0);

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


        vec4 getSourceFracIndex(ivec4 coords){
          return vec4(coords) / scaleWHWH;
        }

        float getAValue(int x10, int r, int c, int d) {
          return getChannel(getA(x10, r, c, d), vec2(c, d));
        }
        // test place holder 0 resize
        void main() {
          ivec4 rc = getOutputCoords();

          int batch = rc[0];
          int depth = rc[1];

          // retrieve the 4 coordinates that is used in the 4 packed output values.
          ivec4 coords = ivec4(rc.wz, rc.w + 1, rc.z + 1);

          // calculate the source index in fraction
          vec4 sourceFrac = getSourceFracIndex(coords);

          // get the lower and upper bound of the 4 values that will be packed into one texel.
          ivec4 x00 = ivec4(max(sourceFrac.xy, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.xy)));
          ivec4 x01 = ivec4(max(sourceFrac.xw, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.xw)));
          ivec4 x10 = ivec4(max(sourceFrac.zy, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.zy)));
          ivec4 x11 = ivec4(max(sourceFrac.zw, vec2(0.0)), min(inputWH - 1.0, ceil(sourceFrac.zw)));

          bool hasNextRow = rc.w < 95;
          bool hasNextCol = rc.z < 159;

          // pack x00, x01, x10, x11's top-left corner into one vec4 structure
          vec4 topLeft = vec4(
            getAValue(batch, depth, x00.x, x00.y),
            hasNextCol ? getAValue(batch, depth, x01.x, x01.y)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.x, x10.y)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.x, x11.y) : 0.0);

          // pack x00, x01, x10, x11's top-right corner into one vec4 structure
          vec4 topRight = vec4(
            getAValue(batch, depth, x00.x, x00.w),
            hasNextCol ? getAValue(batch, depth, x01.x, x01.w)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.x, x10.w)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.x, x11.w) : 0.0);

          // pack x00, x01, x10, x11's bottom-left corner into one vec4 structure
          vec4 bottomLeft = vec4(
            getAValue(batch, depth, x00.z, x00.y),
            hasNextCol ? getAValue(batch, depth, x01.z, x01.y)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.z, x10.y)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.z, x11.y) : 0.0);

          // pack x00, x01, x10, x11's bottom-right corner into one vec4 structure
          vec4 bottomRight = vec4(
            getAValue(batch, depth, x00.z, x00.w),
            hasNextCol ? getAValue(batch, depth, x01.z, x01.w)
                      : 0.0,
            hasNextRow ? getAValue(batch, depth, x10.z, x10.w)
                      : 0.0,
            (hasNextRow && hasNextCol) ?
              getAValue(batch, depth, x11.z, x11.w) : 0.0);

          // calculate the interpolation fraction on u and v direction
          vec4 frac = vec4(sourceFrac) - floor(sourceFrac);
          vec4 clampFrac = clamp(frac, vec4(0.0), vec4(1.0));

          vec4 top = mix(topLeft, topRight, clampFrac.ywyw);
          vec4 bottom = mix(bottomLeft, bottomRight, clampFrac.ywyw);
          vec4 newValue = mix(top, bottom, clampFrac.xxzz);

          outputColor = vec4(newValue);
        }

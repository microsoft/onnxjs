import {Tensor} from '../../../lib/api';

describe('#UnitTest# - API - Tensor Utility Creater Tests', () => {
  it('zeros() test', () => {
    const actual = onnx.zeros([2, 2], 'int32');
    const expected = new Tensor(new Int32Array(4), 'int32', [2, 2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Zeros Test failed');
    }
  });

  it('linspace() test', () => {
    const actual = onnx.linspace(0, 9, 10);
    const expected = new Tensor(Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 'float32', [10]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Linspace Test failed');
    }
  });

  it('range() test', () => {
    const actual = onnx.range(0, 9, 2);
    const expected = new Tensor(Float32Array.from([0, 2, 4, 6, 8]), 'float32', [5]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Range Test failed');
    }
  });

  it('as1d() test',
     () => {

     });

  it('scalar() test', () => {
    const actual = onnx.scalar(3.14, 'float32');
    const data = new Float32Array(1);
    data[0] = 3.14;
    const expected = new Tensor(data, 'float32', [1]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Scalar Test failed');
    }
  });
});

describe('#UnitTest# - API - BasicMathTensorTransforms', () => {
  it('exp() test', () => {
    const actual = onnx.exp(new Tensor([1, 2, -3], 'float32'));
    const expected = new Tensor(Float32Array.from([2.7182817, 7.3890562, 0.0497871]), 'float32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Exp Test failed');
    }
  });
  it('sigmoid() test', () => {
    const actual = onnx.sigmoid(new Tensor([0, -1, 2, -3], 'float32'));
    const expected = new Tensor(Float32Array.from([0.5, 0.2689414, 0.8807971, 0.0474259]), 'float32', [4]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Sigmoid Test failed');
    }
  });
});

describe('#UnitTest# - API - ArithmeticTensorTransforms', () => {
  it('add() test', () => {
    const actual = onnx.add(new Tensor([1, 2, 3], 'float32'), new Tensor([10, 20, 30], 'float32'));
    const expected = new Tensor(Float32Array.from([11, 22, 33]), 'float32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Add Test failed');
    }
  });
  it('sub() test', () => {
    const actual = onnx.sub(new Tensor([1, 2, 3], 'float32'), new Tensor([10, 20, 30], 'float32'));
    const expected = new Tensor(Float32Array.from([-9, -18, -27]), 'float32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Sub Test failed');
    }
  });
  it('mul() test', () => {
    const actual = onnx.mul(new Tensor([1, 2, 3], 'float32'), new Tensor([10, 20, 30], 'float32'));
    const expected = new Tensor(Float32Array.from([10, 40, 90]), 'float32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Mul Test failed');
    }
  });
  it('div() test', () => {
    const actual = onnx.div(new Tensor([10, 20, 30], 'float32'), new Tensor([1, 2, 3], 'float32'));
    const expected = new Tensor(Float32Array.from([10, 10, 10]), 'float32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Div Test failed');
    }
  });
});

describe('#UnitTest# - API - NormalizationTensorTransforms', () => {
  it('softmax() test', () => {
    const actual = onnx.softmax(new Tensor([2, 4, 6, 1, 2, 3], 'float32', [2, 3]));
    const expected = new Tensor(
        Float32Array.from([0.0158762, 0.1173104, 0.8668135, 0.0900306, 0.2447284, 0.6652408]), 'float32', [2, 3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Softmax Test failed');
    }
  });
});

describe('#UnitTest# - API - SliceAndJoinTensorTransforms', () => {
  it('concat() test', () => {
    const a = new Tensor([1, 2], 'float32');
    const b = new Tensor([3, 4], 'float32');
    const actual = onnx.concat([a, b], 0);
    const expected = new Tensor(Float32Array.from([1, 2, 3, 4]), 'float32', [4]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Concat Test failed');
    }
  });
  it('gather() test', () => {
    const indices = new Tensor(Int32Array.from([1, 3, 3]), 'int32');
    const actual = onnx.gather(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32'), indices);
    const expected = new Tensor(Int32Array.from([2, 4, 4]), 'int32');
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Gather Test failed');
    }
  });
  it('slice() test', () => {
    const actual = onnx.slice(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2, 2]), [1, 0], [1, 2]);
    const expected = new Tensor(Int32Array.from([3, 4]), 'int32', [1, 2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Slice Test failed');
    }
  });
  it('stack() test', () => {
    const a = new Tensor([1, 2], 'float32');
    const b = new Tensor([3, 4], 'float32');
    const c = new Tensor([5, 6], 'float32');
    const actual = onnx.stack([a, b, c]);
    const expected = new Tensor(Float32Array.from([1, 2, 3, 4, 5, 6]), 'float32', [3, 2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Stack Test failed');
    }
  });
  it('tile() test', () => {
    const actual = onnx.tile(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2, 2]), [1, 2]);
    const expected = new Tensor(Int32Array.from([1, 2, 1, 2, 3, 4, 3, 4]), 'int32', [2, 4]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Tile Test failed');
    }
  });
});

describe('#UnitTest# - API - PermutationTensorTransforms', () => {
  it('transpose() test', () => {
    const actual = onnx.transpose(new Tensor([1, 2, 3, 4, 5, 6], 'float32', [2, 3]));
    const expected = new Tensor(Float32Array.from([1, 4, 2, 5, 3, 6]), 'float32', [3, 2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Transpose Test failed');
    }
  });
});

describe('#UnitTest# - API - ShapeTensorTransforms', () => {
  it('expandDims() test', () => {
    const actual = onnx.expandDims(new Tensor([1, 2, 3, 4], 'float32'));
    const expected = new Tensor(Float32Array.from([1, 2, 3, 4]), 'float32', [1, 4]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('ExpandDims Test failed');
    }
  });
  it('reshape() test', () => {
    const actual = onnx.reshape(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32'), [2, 2]);
    const expected = new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2, 2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Reshape Test failed');
    }
  });
});

describe('#UnitTest# - API - LogicalTensorTransforms', () => {
  it('greaterEqual() test', () => {
    const actual = onnx.greaterEqual(new Tensor([1, 2, 3], 'float32'), new Tensor([2, 2, 2], 'float32'));
    const expected = new Tensor(Uint8Array.from([0, 1, 1]), 'bool', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('GreaterEqual Test failed');
    }
  });
  it('where() 1d test', () => {
    const cond = new Tensor([false, false, true], 'bool');
    const a = new Tensor(Int32Array.from([1, 2, 3]), 'int32');
    const b = new Tensor(Int32Array.from([-1, -2, -3]), 'int32');
    const actual = onnx.where(cond, a, b);
    const expected = new Tensor(Int32Array.from([-1, -2, 3]), 'int32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Where 1D Test failed');
    }
  });
  it('where() 2d test', () => {
    const cond = new Tensor([false, false, true, true], 'bool', [2, 2]);
    const a = new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2, 2]);
    const b = new Tensor(Int32Array.from([-1, -2, -3, -4]), 'int32', [2, 2]);
    const actual = onnx.where(cond, a, b);
    const expected = new Tensor(Int32Array.from([-1, -2, 3, 4]), 'int32', [2, 2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Where 2D Test failed');
    }
  });
});

describe('#UnitTest# - API - CastTensorTransforms', () => {
  it('cast() test', () => {
    const actual = onnx.cast(new Tensor([1.5, 2.5, 3], 'float32'), 'int32');
    const expected = new Tensor(Int32Array.from([1, 2, 3]), 'int32', [3]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Cast Test failed');
    }
  });
});

describe('#UnitTest# - API - ReductionTensorTransforms', () => {
  it('argMax() 1D test', () => {
    const actual = onnx.argMax(new Tensor(Int32Array.from([1, 2, 3]), 'int32'));
    const expected = new Tensor(Int32Array.from([2]), 'int32', [1]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('ArgMax 1D Test failed');
    }
  });
  it('argMax() 2D test', () => {
    const actual = onnx.argMax(new Tensor(Int32Array.from([1, 2, 4, 3]), 'int32', [2, 2]), 1);
    const expected = new Tensor(Int32Array.from([1, 0]), 'int32', [2]);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('ArgMax 2D Test failed');
    }
  });
  it('reduceMax() 1D test', () => {
    const actual = onnx.reduceMax(new Tensor(Int32Array.from([1, 2, 3]), 'int32'), [0], 0);
    const expected = new Tensor(Int32Array.from([3]), 'int32', [1]);
    console.log('actual ', actual);
    console.log('expected ', expected);
    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Max 1D Test failed');
    }
  });
  it('reduceMax() 2D test', () => {
    const actual = onnx.reduceMax(new Tensor(Int32Array.from([1, 2, 3, 4]), 'int32', [2, 2]), [1], 0);
    const expected = new Tensor(Int32Array.from([2, 4]), 'int32', [2]);

    if (!assertTensorEquality(actual, expected)) {
      throw new Error('Max 2D Test failed');
    }
  });
});

// Helper for tests
function assertTensorEquality(t1: Tensor, t2: Tensor) {
  // type doesn't match
  if (t1.type !== t2.type) {
    return false;
  }

  // dims don't match
  if (t1.dims.length !== t2.dims.length) {
    return false;
  }

  // dims don't match
  for (let i = 0; i < t1.dims.length; ++i) {
    if (t1.dims[i] !== t2.dims[i]) {
      return false;
    }
  }

  // data doesn't match
  if (t1.data.length !== t2.data.length) {
    return false;
  }

  // data doesn't match
  for (let i = 0; i < t2.data.length; ++i) {
    if (t1.data[i] !== t2.data[i]) {
      if (t1.type === 'string') {
        return false;
      }
      // number type. so allow some precision loss.
      if (Math.abs((t1.data[i] as number) - (t2.data[i] as number)) > 0.0001) {
        return false;
      }
    }
  }

  return true;
}

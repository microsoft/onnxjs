// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';

import {Tensor} from '../../../lib/api';

let t: Tensor;

describe('#UnitTest# - API - Tensor constructors', () => {
  it('1. Create a new Tensor with specific type', () => {
    t = new Tensor([2, 3], 'int32');              // ok, a int32-typed 1D tensor with 2 elements
    t = new Tensor(new Int32Array(10), 'int32');  // ok, a int32-typed 1D tensor with 10 elements
    t = new Tensor(['a', 'b', 'c'], 'string');    // ok, a string-typed 1D tensor with 3 elements
    t = new Tensor([], 'float32');                // ok, a zero-sized 1D tensor

    expect(() => {
      t = new Tensor(new Int32Array(10), 'float32');  // throws TypeError. data and type don't match.
    }).to.throw(TypeError);
  });

  it('2. Create a new Tensor with type and dims', () => {
    t = new Tensor([2, 3], 'float32', [1, 1, 2]);         // ok
    t = new Tensor([2, 3, 4, 5, 6, 7], 'int32', [3, 2]);  // ok
    t = new Tensor([2], 'float32', [1]);                  // ok: this is a 1-d tensor with size 1

    expect(() => {
      t = new Tensor([2, 3], 'float32', [1, 0.5, 4]);  // throws TypeError. input dims is not integer.
    }).to.throw(TypeError);
    expect(() => {
      t = new Tensor([2, 3], 'float32', [1, 1, 3]);  // throws RangeError. input dims doesn't match data length.
    }).to.throw(RangeError);
  });

  it('3. Create a scalar Tensor', () => {
    t = new Tensor([1], 'float32', []);
    expect(t.data).to.have.lengthOf(1);
    expect(t.data[0]).to.equal(1);
    expect(t.data).to.be.an('Float32Array');
    expect(t.dims).to.have.lengthOf(0);
    expect(() => {
      t = new Tensor([2, 3, 4], 'float32', []);  // throws RangeError. input dims doesn't match data length.
    }).to.throw(RangeError);
  });
});

describe('#UnitTest# - API - Tensor Raw Data Manipulation', () => {
  it('Get Raw Data from Boolean Tensor', () => {
    t = new Tensor([true, false, false], 'bool');
    expect(t.data).to.have.lengthOf(3);
    expect(t.data[0]).to.equal(1);
    expect(t.data[1]).to.equal(0);
    expect(t.data[2]).to.equal(0);
    expect(t.data).to.be.an('UInt8Array');
  });

  it('Access and Change Raw Data from Int32Array Tensor', () => {
    t = new Tensor(new Int32Array([1, 3, 2]), 'int32');
    const data = t.data;
    data[1] = 2;
    data[2] = 3;
    expect(data).to.have.lengthOf(3);
    expect(data).to.deep.equal(new Int32Array([1, 2, 3]));
    expect(data).to.be.a('Int32Array');
  });

  it('Access and Change Raw Data from Array Tensor', () => {
    t = new Tensor([1, 3, 2], 'int32');
    const data = t.data;
    data[1] = 2;
    data[2] = 3;
    expect(data).to.have.lengthOf(3);
    expect(data).to.deep.equal(new Int32Array([1, 2, 3]));
    expect(data).to.be.a('Int32Array');
  });
});

describe('#UnitTest# - API - Tensor Data Getter', () => {
  it('Get Tensor Element By Index Array', () => {
    t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
    expect(t.get([0, 1, 0])).to.equal(3);
    expect(t.get([0, 1, 0])).to.be.a('number');
    expect(() => {
      t.get([2, 1]);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.get([1, 3, 2]);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.get([1, 3, 5, 6]);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.get([2.5, 1.5, 1.0]);  // throws TypeError. Input dims is not integer.
    }).to.throw(TypeError);
  });

  it('Get Tensor Element By Indices', () => {
    t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
    expect(t.get(0, 1, 0)).to.equal(3);
    expect(t.get(0, 1, 0)).to.be.a('number');
    expect(() => {
      t.get(2, 1);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.get(1, 3, 2);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.get(0, 0, 0, 0);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.get(2.5, 1.5, 1.0);  // throws TypeError. Input dims is not integer.
    }).to.throw(TypeError);
  });

  it('Get Tensor Boolean Element', () => {
    t = new Tensor([false, true, true, false], 'bool', [1, 2, 2]);
    expect(t.get(0, 1, 0)).to.equal(true);
    expect(t.get(0, 1, 0)).to.be.a('boolean');
    expect(t.get(0, 1, 1)).to.equal(false);
    expect(t.get(0, 1, 1)).to.be.a('boolean');
  });

  it('Get Tensor String Element', () => {
    t = new Tensor(['a', 'b', 'c', 'd', 'e', 'f'], 'string', [2, 3, 1]);
    expect(t.get(1, 1, 0)).to.equal('e');
    expect(t.get(1, 1, 0)).to.be.a('string');
    expect(t.get(1, 2, 0)).to.equal('f');
    expect(t.get(1, 2, 0)).to.be.a('string');
  });
});

describe('#UnitTest# - API - Tensor Data Setter', () => {
  it('Set Tensor Element By Index Array', () => {
    t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
    t.set(2, [0, 1, 0]);
    t.set(3, [0, 2, 0]);
    expect(t.get([0, 1, 0])).to.equal(2);
    expect(t.get([0, 2, 0])).to.equal(3);
    expect(t.data[0]).to.equal(1);
    expect(t.data[1]).to.equal(2);
    expect(t.data[2]).to.equal(3);
    expect(t.size).to.equal(3);
    expect(() => {
      t.set(2.5, [0, 2, 0]);  // throws TypeError. The new element type doesn't match the tensor data type
    }).to.throw(TypeError);
    expect(() => {
      t.set(2, 1);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.set(1, 3, 2);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.set(1, 3, 5, 6);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.set(2, 2.5, 1.5, 1.0);  // throws TypeError. Input dims is not integer.
    }).to.throw(TypeError);
  });
  it('Set Tensor Element By Indices', () => {
    t = new Tensor([1, 3, 2], 'int32', [1, 3, 1]);
    t.set(2, 0, 1, 0);
    t.set(3, 0, 2, 0);
    expect(t.get([0, 1, 0])).to.equal(2);
    expect(t.data[0]).to.equal(1);
    expect(t.data[1]).to.equal(2);
    expect(t.data[2]).to.equal(3);

    expect(t.size).to.equal(3);
    expect(() => {
      t.set(2.5, 0, 2, 0);  // throws TypeError. The new element type doesn't match the tensor data type
    }).to.throw(TypeError);
    expect(() => {
      t.set(2, 1);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.set(1, 3, 2);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.set(1, 3, 5, 6);  // throws RangeError. Input index array dims don't match the tensor dims.
    }).to.throw(RangeError);
    expect(() => {
      t.set(2, 2.5, 1.5, 1.0);  // throws TypeError. Input dims is not integer.
    }).to.throw(TypeError);
  });
});

//
// typescript does not support to suppress --noUnusedLocals, --noUnusedParameters per file
// by-pass unused variable check
t = unused();
unused(t);
function unused<U>(...t: unknown[]) {
  return t as unknown as U;
}

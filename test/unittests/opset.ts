// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';

import {Attribute} from '../../lib/attribute';
import {Graph} from '../../lib/graph';
import {Operator} from '../../lib/operators';
import {resolveOperator} from '../../lib/opset';

describe('#UnitTest# - resolveOperator', () => {
  const nodeAbs = createTestGraphNode('Abs_1', 'Abs');
  const opset7 = [{domain: '', version: 7}];
  it('ExpectFail - no rule available', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, []);
    }).to.throw(TypeError);
  });
  it('ExpectFail - no matching rule', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['And', '', '7', dummyOpConstructor], ['Sub', '', '7', dummyOpConstructor]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (exact match)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '6', dummyOpConstructor]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (range match)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '8+', dummyOpConstructor]]);
    }).to.throw(TypeError);
  });
  it('ExpectPass - version match (exact match)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '7', dummyOpConstructor]]);
  });
  it('ExpectPass - version match (range match)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '5+', dummyOpConstructor]]);
  });
});

function createTestGraphNode(name: string, opType: string): Graph.Node {
  return {name, opType, inputs: [], outputs: [], attributes: new Attribute(null)};
}

function dummyOpConstructor(): Operator {
  // tslint:disable-next-line:no-any
  return {} as any as Operator;
}

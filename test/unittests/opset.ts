// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {expect} from 'chai';

import {Attribute} from '../../lib/attribute';
import {CPU_OP_RESOLVE_RULES} from '../../lib/backends/cpu/op-resolve-rules';
import {WASM_OP_RESOLVE_RULES} from '../../lib/backends/wasm/op-resolve-rules';
import {WEBGL_OP_RESOLVE_RULES} from '../../lib/backends/webgl/op-resolve-rules';
import {Graph} from '../../lib/graph';
import {Operator} from '../../lib/operators';
import {OpSet, resolveOperator} from '../../lib/opset';

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
  it('ExpectFail - version not match (minimum version match)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '8+', dummyOpConstructor]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (range match 1)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '4-6', dummyOpConstructor]]);
    }).to.throw(TypeError);
  });
  it('ExpectFail - version not match (range match 2)', () => {
    expect(() => {
      resolveOperator(nodeAbs, opset7, [['Abs', '', '8-10', dummyOpConstructor]]);
    }).to.throw(TypeError);
  });
  it('ExpectPass - version match (exact match)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '7', dummyOpConstructor]]);
  });
  it('ExpectPass - version match (minimum version match)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '5+', dummyOpConstructor]]);
  });
  it('ExpectPass - version match (range match 1)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '5-7', dummyOpConstructor]]);
  });
  it('ExpectPass - version match (range match 2)', () => {
    resolveOperator(nodeAbs, opset7, [['Abs', '', '6-9', dummyOpConstructor]]);
  });
});

describe('#UnitTest# - resolve rules', () => {
  const cpuCheckOnlyRules =
      CPU_OP_RESOLVE_RULES.map(rule => [rule[0], rule[1], rule[2], dummyOpConstructor] as OpSet.ResolveRule);
  const wasmCheckOnlyRules =
      WASM_OP_RESOLVE_RULES.map(rule => [rule[0], rule[1], rule[2], dummyOpConstructor] as OpSet.ResolveRule);
  const webglCheckOnlyRules =
      WEBGL_OP_RESOLVE_RULES.map(rule => [rule[0], rule[1], rule[2], dummyOpConstructor] as OpSet.ResolveRule);
  it('Consistency check - onnx.ai - cpu', () => {
    checkConsistency(cpuCheckOnlyRules);
  });
  it('Consistency check - onnx.ai - wasm', () => {
    checkConsistency(wasmCheckOnlyRules);
  });
  it('Consistency check - onnx.ai - webgl', () => {
    checkConsistency(webglCheckOnlyRules);
  });
});

function createTestGraphNode(name: string, opType: string): Graph.Node {
  return {name, opType, inputs: [], outputs: [], attributes: new Attribute(null)};
}

function dummyOpConstructor(): Operator {
  // tslint:disable-next-line:no-any
  return {} as any as Operator;
}

function checkConsistency(rules: ReadonlyArray<OpSet.ResolveRule>) {
  const VERSION_MIN = 1, VERSION_MAX = 10;
  const typeRules = new Map<string, OpSet.ResolveRule[]>();
  rules.forEach(rule => {
    let ruleSet = typeRules.get(rule[0]);
    if (!ruleSet) {
      ruleSet = [];
      typeRules.set(rule[0], ruleSet);
    }
    ruleSet.push(rule);
  });

  typeRules.forEach((rules, type) => {
    for (let i = VERSION_MIN; i < VERSION_MAX; i++) {
      let match = false;
      for (const r of rules) {
        try {
          resolveOperator(createTestGraphNode('', type), [{domain: '', version: i}], [r]);
        } catch {
          continue;
        }
        expect(match, `multiple rules overlapped: opType='${type}', domain='', version=${i}`).to.be.false;
        match = true;
      }
    }
  });
}

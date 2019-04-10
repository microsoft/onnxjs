// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

// tslint:disable-next-line:no-import-side-effect
import '../lib/api';

import {Logger} from '../lib/instrument';
import {Test} from './test-types';

// tslint:disable-next-line:no-require-imports
const ONNX_JS_TEST_CONFIG = require('./testdata') as Test.Config;

// Set logging configuration
for (const logConfig of ONNX_JS_TEST_CONFIG.log) {
  Logger.set(logConfig.category, logConfig.config);
}

import {ModelTestContext, OpTestContext, runModelTestSet, runOpTest} from './test-runner';

// Unit test
if (ONNX_JS_TEST_CONFIG.unittest) {
  // tslint:disable-next-line:no-require-imports
  require('./unittests');
}

// Set file cache
ModelTestContext.setCache(ONNX_JS_TEST_CONFIG.fileCache);

// ModelTests
for (const group of ONNX_JS_TEST_CONFIG.model) {
  describe(`#ModelTest# - ${group.name}`, () => {
    for (const test of group.tests) {
      const describeTest = (!test.cases || test.cases.length === 0) ? describe.skip : describe;
      describeTest(`[${test.backend}] ${test.name}`, () => {
        let context: ModelTestContext;

        before('prepare session', async () => {
          context = await ModelTestContext.create(test, ONNX_JS_TEST_CONFIG.profile);
        });

        after('release session', () => {
          if (context) {
            context.release();
          }
        });

        for (const testCase of test.cases) {
          it(testCase.name, async () => {
            await runModelTestSet(context, testCase);
          });
        }
      });
    }
  });
}

// OpTests
for (const group of ONNX_JS_TEST_CONFIG.op) {
  describe(`#OpTest# - ${group.name}`, () => {
    for (const test of group.tests) {
      const describeTest = (!test.cases || test.cases.length === 0) ? describe.skip : describe;
      describeTest(`[${test.backend!}]${test.operator} - ${test.name}`, () => {
        let context: OpTestContext;

        before('Initialize Context', async () => {
          context = new OpTestContext(test);
          await context.init();
          if (ONNX_JS_TEST_CONFIG.profile) {
            OpTestContext.profiler.start();
          }
        });

        after('Dispose Context', () => {
          if (ONNX_JS_TEST_CONFIG.profile) {
            OpTestContext.profiler.stop();
          }
          context.dispose();
        });

        for (const testCase of test.cases) {
          it(testCase.name, async () => {
            await runOpTest(testCase, context);
          });
        }
      });
    }
  });
}

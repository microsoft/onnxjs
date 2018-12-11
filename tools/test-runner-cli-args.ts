// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import minimist from 'minimist';
import logger from 'npmlog';

import {Backend} from '../lib/api';
import {Logger} from '../lib/instrument';
import {Test} from '../test/test-types';

const HELP_MESSAGE = `
test-runner-cli

Run ONNX.js tests, models, benchmarks in different environments.

Usage:
 test-runner-cli <mode> ... [options]

Modes:
 suite0                      Run all unittests, all operator tests and node model tests that described in white list
 suite1                      Run all unittests, all operator tests and all model tests that described in white list
 model                       Run a single model test
 unittest                    Run all unittests
 op                          Run a single operator test

Options:

*** General Options ***

 -h, --help                  Print this message.
 -v, --verbose               Verbose the output of test runner cli
 -d, --debug                 Specify to run test runner in debug mode.
 -b=<...>, --backend=<...>   Specify one or more backend(s) to run the test upon.
                               Backends can be one or more of the following, splitted by comma:
                                 cpu
                                 webgl
                                 wasm
 -e=<...>, --env=<...>       Specify the environment to run the test. Should be one of the following:
                               chrome (default)
                               edge
                               firefox
                               electron
                               node

*** Logging Options ***

 -p, --profile               Enable profiler.
                               Profiler will generate extra logs which include the information of events time
                               consumption
 -P, --perf                  Generate performance number. Cannot be used with flag --debug.

 --log-verbose=<...>         Set log level to verbose
 --log-info=<...>            Set log level to info
 --log-warning=<...>         Set log level to warning
 --log-error=<...>           Set log level to error
                               The 4 flags above specify the logging configuration. Each flag allows to specify one or
                               more category(s), splitted by comma. If use the flags without value, the log level will
                               be applied to all category.

*** Backend Options ***

 --wasm-worker               Set the WASM worker number
 --webgl-context-id          Set the WebGL context ID

*** Browser Options ***

 --no-sandbox                This flag will be passed to Chrome.
                               Sometimes Chrome need this flag to work together with Karma.

Examples:

 Run all suite0 tests:
 > test-runner-cli suite0

 Run single model test (test_relu) on CPU backend and show verbose log
 > test-runner-cli model test_relu --verbose --backend=cpu

 Debug unittest
 > test-runner-cli unittest --debug

 Debug operator matmul, highlight verbose log from BaseGlContext and WebGLBackend
 > test-runner-cli op matmul --debug --info --verbose=BaseGlContext,WebGLBackend

 Profile the model ResNet50 on WebGL backend
 > test-runner-cli model resnet50 --profile --backend=webgl
 `;

export interface TestRunnerCliArgs {
  debug: boolean;
  mode: 'suite0'|'suite1'|'model'|'unittest'|'op';
  param?: string;
  backend: ['cpu'|'webgl'|'wasm'];
  env: 'chrome'|'edge'|'firefox'|'electron'|'node';

  /**
   * Bundle Mode
   *
   * this field affects the behavior of Karma and Webpack.
   *
   * Mode   | Folder       | Files                   | Source Map         | Webpack Config
   * ------ | ------------ | ----------------------- | ------------------ | --------------
   * prod   | /dist/       | onnx.min.js + test.js   | source-map         | production
   * dev    | /test/       | onnx.dev.js             | inline-source-map  | development
   * perf   | /test/       | onnx.perf.js            | source-map         | production
   */
  bundleMode: 'prod'|'dev'|'perf';

  logConfig: Test.Config['log'];
  profile: boolean;

  worker?: Backend.WasmOptions['worker'];
  // contextId?: Backend.WebGLOptions['contextId'];

  noSandbox?: boolean;
}

export function parseTestRunnerCliArgs(cmdlineArgs: string[]): TestRunnerCliArgs {
  const args = minimist(cmdlineArgs);

  if (args.help || args.h) {
    console.log(HELP_MESSAGE);
    process.exit();
  }

  if (typeof (args.verbose || args.v) === 'boolean' || (args.debug || args.d)) {
    logger.level = 'verbose';
  }
  logger.verbose('TestRunnerCli.Init', `Parsing commandline arguments...`);

  const mode = args._.length === 0 ? 'suite0' : args._[0];

  // Option: -d, --debug
  const debug = (args.debug || args.d) ? true : false;

  // Option: -b=<...>, --backend=<...>
  const backendArgs = args.b || args.backend;
  const backend = (typeof backendArgs !== 'string') ? ['cpu', 'webgl', 'wasm'] : backendArgs.split(',');
  for (const b of backend) {
    if (b !== 'cpu' && b !== 'webgl' && b !== 'wasm') {
      throw new Error(`not supported backend ${b}`);
    }
  }

  // Option: -e=<...>, --env=<...>
  const envArg = args.e || args.env;
  const env = (typeof envArg !== 'string') ? 'chrome' : envArg;
  if (['chrome', 'edge', 'firefox', 'electron', 'node'].indexOf(env) === -1) {
    throw new Error(`not supported env ${env}`);
  }

  // Options:
  // --log-verbose=<...>
  // --log-info=<...>
  // --log-warning=<...>
  // --log-error=<...>
  const logConfig = parseLogConfig(args);

  // Option: -p, --profile
  const profile = (args.profile || args.p) ? true : false;
  if (profile) {
    logConfig.push({category: 'Profiler.session', config: {minimalSeverity: 'verbose'}});
    logConfig.push({category: 'Profiler.node', config: {minimalSeverity: 'verbose'}});
    logConfig.push({category: 'Profiler.op', config: {minimalSeverity: 'verbose'}});
    logConfig.push({category: 'Profiler.backend', config: {minimalSeverity: 'verbose'}});
  }

  const perf = (args.perf || args.P) ? true : false;
  if (debug && perf) {
    throw new Error('Flag "perf" cannot be used together with flag "debug".');
  }

  // Option: --no-sandbox
  const noSandbox = !!args['no-sandbox'];

  logger.verbose('TestRunnerCli.Init', ` Mode:              ${mode}`);
  logger.verbose('TestRunnerCli.Init', ` Env:               ${env}`);
  logger.verbose('TestRunnerCli.Init', ` Debug:             ${debug}`);
  logger.verbose('TestRunnerCli.Init', ` Backend:           ${backend}`);
  logger.verbose('TestRunnerCli.Init', `Parsing commandline arguments... DONE`);

  return {
    debug,
    mode: mode as TestRunnerCliArgs['mode'],
    param: args._.length > 1 ? args._[1] : undefined,
    backend: backend as TestRunnerCliArgs['backend'],
    bundleMode: perf ? 'perf' : (debug ? 'debug' : 'dev'),
    env: env as TestRunnerCliArgs['env'],
    logConfig,
    profile,
    noSandbox
  };
}

function parseLogLevel<T>(arg: T) {
  let v: string[]|boolean;
  if (typeof arg === 'string') {
    v = arg.split(',');
  } else if (Array.isArray(arg)) {
    v = [];
    for (const e of arg) {
      v.push(...e.split(','));
    }
  } else {
    v = arg ? true : false;
  }
  return v;
}
function parseLogConfig(args: minimist.ParsedArgs) {
  const config: Array<{category: string, config: Logger.Config}> = [];
  const verbose = parseLogLevel(args['log-verbose']);
  const info = parseLogLevel(args['log-info']);
  const warning = parseLogLevel(args['log-warning']);
  const error = parseLogLevel(args['log-error']);

  if (typeof error === 'boolean' && error) {
    config.push({category: '*', config: {minimalSeverity: 'error'}});
  } else if (typeof warning === 'boolean' && warning) {
    config.push({category: '*', config: {minimalSeverity: 'warning'}});
  } else if (typeof info === 'boolean' && info) {
    config.push({category: '*', config: {minimalSeverity: 'info'}});
  } else if (typeof verbose === 'boolean' && verbose) {
    config.push({category: '*', config: {minimalSeverity: 'verbose'}});
  }

  if (Array.isArray(error)) {
    config.push(...error.map(i => ({category: i, config: {minimalSeverity: 'error' as Logger.Severity}})));
  }
  if (Array.isArray(warning)) {
    config.push(...warning.map(i => ({category: i, config: {minimalSeverity: 'warning' as Logger.Severity}})));
  }
  if (Array.isArray(info)) {
    config.push(...info.map(i => ({category: i, config: {minimalSeverity: 'info' as Logger.Severity}})));
  }
  if (Array.isArray(verbose)) {
    config.push(...verbose.map(i => ({category: i, config: {minimalSeverity: 'verbose' as Logger.Severity}})));
  }

  return config;
}

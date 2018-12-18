// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs';
import * as globby from 'globby';
import minimist from 'minimist';
import logger from 'npmlog';
import * as path from 'path';
import stripJsonComments from 'strip-json-comments';
import {inspect} from 'util';

import {Logger} from '../lib/instrument';
import {Test} from '../test/test-types';

const args = minimist(process.argv.slice(2));

if (args.help || args.h) {
  console.log(`
test-runner-cli

Generate file 'testdata.js' to use in unittest.

Usage:
 test-runner-cli [options] <mode> ...

Modes:
 suite0                      Run all unittests, all operator tests and node model tests that described in white list
 suite1                      Run all unittests, all operator tests and all model tests that described in white list
 model                       Run a single model test
 unittest                    Run all unittests
 op                          Run a single operator test

Options:
 -h, --help                  Print this message.
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
 -p, --profile               Enable profiler.
                               Profiler will generate extra logs which include the information of events time
                               consumption
 --log-verbose=<...>         Set log level to verbose
 --log-info=<...>            Set log level to info
 --log-warning=<...>         Set log level to warning
 --log-error=<...>           Set log level to error
                               The 4 flags above specify the logging configuration. Each flag allows to specify one or
                               more category(s), splitted by comma. If use the flags without value, the log level will
                               be applied to all category.
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
 `);
  process.exit();
}

logger.info('TestRunnerCli', 'Initializing...');

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
if (['chrome', 'edge', 'firefox', 'electron', 'safari', 'node'].indexOf(env) === -1) {
  throw new Error(`not supported env ${env}`);
}

// Options:
// --log-verbose=<...>
// --log-info=<...>
// --log-warning=<...>
// --log-error=<...>
const logConfig = parseLogConfig();

// Option: -p, --profile
const profile = (args.profile || args.p) ? true : false;
if (profile) {
  logConfig.push({category: 'Profiler.session', config: {minimalSeverity: 'verbose'}});
  logConfig.push({category: 'Profiler.node', config: {minimalSeverity: 'verbose'}});
  logConfig.push({category: 'Profiler.op', config: {minimalSeverity: 'verbose'}});
  logConfig.push({category: 'Profiler.backend', config: {minimalSeverity: 'verbose'}});
}

// Option: --no-sandbox
const noSandbox = args['no-sandbox'];

logger.verbose('TestRunnerCli.Init', ` Mode:              ${mode}`);
logger.verbose('TestRunnerCli.Init', ` Env:               ${env}`);
logger.verbose('TestRunnerCli.Init', ` Debug:             ${debug}`);
logger.verbose('TestRunnerCli.Init', ` Backend:           ${backend}`);
logger.verbose('TestRunnerCli.Init', `Parsing commandline arguments... DONE`);

const TEST_ROOT = path.join(__dirname, '..', 'test');
const TEST_DATA_ROOT = path.join(__dirname, '..', 'deps', 'data', 'data', 'test');
const TEST_DATA_NODE_ROOT = path.join(TEST_DATA_ROOT, 'node');
const TEST_DATA_ONNX_ROOT = path.join(TEST_DATA_ROOT, 'onnx', 'v7');
const TEST_DATA_OP_ROOT = path.join(TEST_DATA_ROOT, 'ops');

const TEST_DATA_BASE = env === 'node' ? TEST_ROOT : '/base/test/';

logger.verbose('TestRunnerCli.Init', `Loading whitelist...`);

// The following is a whitelist of unittests for already implemented operators.
// Modify this list to control what node tests to run.
const jsonWithComments = fs.readFileSync(path.resolve(TEST_ROOT, './unittest-whitelist.jsonc')).toString();
const json = stripJsonComments(jsonWithComments, {whitespace: true});
const whitelist = JSON.parse(json) as Test.WhiteList;
logger.verbose('TestRunnerCli.Init', `Loading whitelist... DONE`);

const shouldLoadSuiteTestData = (mode === 'suite0' || mode === 'suite1');
if (shouldLoadSuiteTestData) {
  logger.verbose('TestRunnerCli.Init', `Loading test groups for suite test...`);
}

const nodeTestsCpu = shouldLoadSuiteTestData ? nodeTests('cpu') : null;
const nodeTestsWebGL = shouldLoadSuiteTestData ? nodeTests('webgl') : null;
const nodeTestsWasm = shouldLoadSuiteTestData ? nodeTests('wasm') : null;

const onnxTestsCpu = shouldLoadSuiteTestData ? onnxTests('cpu') : null;
const onnxTestsWebGL = shouldLoadSuiteTestData ? onnxTests('webgl') : null;
const onnxTestsWasm = shouldLoadSuiteTestData ? onnxTests('wasm') : null;

const opTestsCpu = shouldLoadSuiteTestData ? opTests('cpu') : null;
const opTestsWebGL = shouldLoadSuiteTestData ? opTests('webgl') : null;
const opTestsWasm = shouldLoadSuiteTestData ? opTests('wasm') : null;

if (shouldLoadSuiteTestData) {
  logger.verbose('TestRunnerCli.Init', `Loading test groups for suite test... DONE`);

  logger.verbose('TestRunnerCli.Init', `Validate whitelist...`);
  validateWhiteList();
  logger.verbose('TestRunnerCli.Init', `Validate whitelist... DONE`);
}

const modelTestGroups: Test.ModelTestGroup[] = [];
const opTestGroups: Test.OperatorTestGroup[] = [];
let unittest = false;

logger.verbose('TestRunnerCli.Init', `Preparing test config...`);
switch (mode) {
  case 'suite1':
    if (backend.indexOf('cpu') !== -1) {
      modelTestGroups.push(onnxTestsCpu!);  // model test : ONNX model (CPU)
    }
    if (backend.indexOf('webgl') !== -1) {
      modelTestGroups.push(onnxTestsWebGL!);  // model test : ONNX model (WebGL)
    }
    if (backend.indexOf('wasm') !== -1) {
      modelTestGroups.push(onnxTestsWasm!);  // model test : ONNX model (Wasm)
    }

  case 'suite0':
    if (backend.indexOf('cpu') !== -1) {
      modelTestGroups.push(nodeTestsCpu!);  // model test : node (CPU)
      opTestGroups.push(...opTestsCpu!);    // operator test (CPU)
    }
    if (backend.indexOf('webgl') !== -1) {
      modelTestGroups.push(nodeTestsWebGL!);  // model test : node (WebGL)
      opTestGroups.push(...opTestsWebGL!);    // operator test (WebGL)
    }
    if (backend.indexOf('wasm') !== -1) {
      modelTestGroups.push(nodeTestsWasm!);  // model test : node (Wasm)
      opTestGroups.push(...opTestsWasm!);    // operator test (Wasm)
    }
    unittest = true;
    break;

  case 'model':
    if (args._.length < 2) {
      throw new Error(`the test folder should be specified in mode 'node'`);
    }
    const testFolderSearchPattern = args._[1];
    const testFolder = tryLocateModelTestFolder(testFolderSearchPattern);
    for (const b of backend) {
      modelTestGroups.push({name: testFolder, tests: [modelTestFromFolder(testFolder, b)]});
    }
    break;

  case 'unittest':
    unittest = true;
    break;

  case 'op':
    if (args._.length < 2) {
      throw new Error(`the test manifest should be specified in mode 'op'`);
    }
    const manifestFileSearchPattern = args._[1];
    const manifestFile = tryLocateOpTestManifest(manifestFileSearchPattern);
    for (const b of backend) {
      opTestGroups.push(opTestFromManifest(manifestFile, b));
    }
    break;
  default:
    throw new Error(`unsupported mode '${mode}'`);
}

logger.verbose('TestRunnerCli.Init', `Preparing test config... DONE`);

logger.info('TestRunnerCli', 'Initialization completed. Start to run tests...');
run({unittest, model: modelTestGroups, op: opTestGroups, log: logConfig, profile});
logger.info('TestRunnerCli', 'Tests completed successfully');

process.exit();

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
function parseLogConfig() {
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

function validateWhiteList() {
  if (nodeTestsCpu) {
    const nodeModelTestsCpuNames = nodeTestsCpu.tests.map(i => i.name);
    for (const testCase of whitelist.cpu.node) {
      if (nodeModelTestsCpuNames.indexOf(testCase) === -1) {
        throw new Error(`node model test case '${testCase}' in white list does not exist.`);
      }
    }
  }
  if (nodeTestsWebGL) {
    const nodeModelTestsWebGLNames = nodeTestsWebGL.tests.map(i => i.name);
    for (const testCase of whitelist.webgl.node) {
      if (nodeModelTestsWebGLNames.indexOf(testCase) === -1) {
        throw new Error(`node model test case '${testCase}' in white list does not exist.`);
      }
    }
  }
  if (nodeTestsWasm) {
    const nodeModelTestsWasmNames = nodeTestsWasm.tests.map(i => i.name);
    for (const testCase of whitelist.wasm.node) {
      if (nodeModelTestsWasmNames.indexOf(testCase) === -1) {
        throw new Error(`node model test case '${testCase}' in white list does not exist.`);
      }
    }
  }

  if (onnxTestsCpu) {
    const onnxModelTestsCpuNames = onnxTestsCpu.tests.map(i => i.name);
    for (const testCase of whitelist.cpu.onnx) {
      if (onnxModelTestsCpuNames.indexOf(testCase) === -1) {
        throw new Error(`onnx model test case '${testCase}' in white list does not exist.`);
      }
    }
  }
  if (onnxTestsWebGL) {
    const onnxModelTestsWebGLNames = onnxTestsWebGL.tests.map(i => i.name);
    for (const testCase of whitelist.webgl.onnx) {
      if (onnxModelTestsWebGLNames.indexOf(testCase) === -1) {
        throw new Error(`onnx model test case '${testCase}' in white list does not exist.`);
      }
    }
  }
  if (onnxTestsWasm) {
    const onnxModelTestsWasmNames = onnxTestsWasm.tests.map(i => i.name);
    for (const testCase of whitelist.wasm.onnx) {
      if (onnxModelTestsWasmNames.indexOf(testCase) === -1) {
        throw new Error(`onnx model test case '${testCase}' in white list does not exist.`);
      }
    }
  }

  if (opTestsCpu) {
    const opTestsCpuNames = opTestsCpu.map(i => i.name);
    for (const testCase of whitelist.cpu.ops) {
      if (opTestsCpuNames.indexOf(testCase) === -1) {
        throw new Error(`operator test case '${testCase}' in white list does not exist.`);
      }
    }
  }
  if (opTestsWebGL) {
    const opTestsWebGLNames = opTestsWebGL.map(i => i.name);
    for (const testCase of whitelist.webgl.ops) {
      if (opTestsWebGLNames.indexOf(testCase) === -1) {
        throw new Error(`operator test case '${testCase}' in white list does not exist.`);
      }
    }
  }
  if (opTestsWasm) {
    const opTestsWasmNames = opTestsWasm.map(i => i.name);
    for (const testCase of whitelist.wasm.ops) {
      if (opTestsWasmNames.indexOf(testCase) === -1) {
        throw new Error(`operator test case '${testCase}' in white list does not exist.`);
      }
    }
  }
}

function nodeTests(backend: string): Test.ModelTestGroup {
  return suiteFromFolder(`node-${backend}`, TEST_DATA_NODE_ROOT, backend, whitelist[backend].node);
}

function onnxTests(backend: string): Test.ModelTestGroup {
  return suiteFromFolder(`onnx-${backend}`, TEST_DATA_ONNX_ROOT, backend, whitelist[backend].onnx);
}

function suiteFromFolder(
    name: string, suiteRootFolder: string, backend: string, whitelist?: ReadonlyArray<string>): Test.ModelTestGroup {
  const sessions: Test.ModelTest[] = [];
  const tests = fs.readdirSync(suiteRootFolder);
  for (const test of tests) {
    const skip = whitelist && whitelist.indexOf(test) === -1;
    sessions.push(modelTestFromFolder(path.resolve(suiteRootFolder, test), backend, skip));
  }
  return {name, tests: sessions};
}

function modelTestFromFolder(testDataRootFolder: string, backend: string, skip = false): Test.ModelTest {
  if (skip) {
    logger.verbose('TestRunnerCli.Init.Model', `Skip test data from folder: ${testDataRootFolder}`);
    return {name: path.basename(testDataRootFolder), backend, modelUrl: '', cases: []};
  }

  let modelUrl: string|null = null;
  const cases: Test.ModelTestCase[] = [];

  logger.verbose('TestRunnerCli.Init.Model', `Start to prepare test data from folder: ${testDataRootFolder}`);

  try {
    for (const thisPath of fs.readdirSync(testDataRootFolder)) {
      const thisFullPath = path.join(testDataRootFolder, thisPath);
      const stat = fs.lstatSync(thisFullPath);
      if (stat.isFile()) {
        const ext = path.extname(thisPath);
        if (ext.toLowerCase() === '.onnx') {
          if (modelUrl === null) {
            modelUrl = path.join(TEST_DATA_BASE, path.relative(TEST_ROOT, thisFullPath));
          } else {
            throw new Error('there are multiple model files under the folder specified');
          }
        }
      } else if (stat.isDirectory()) {
        const dataFiles: string[] = [];
        for (const dataFile of fs.readdirSync(thisFullPath)) {
          const dataFileFullPath = path.join(thisFullPath, dataFile);
          const ext = path.extname(dataFile);

          if (ext.toLowerCase() === '.pb') {
            dataFiles.push(path.join(TEST_DATA_BASE, path.relative(TEST_ROOT, dataFileFullPath)));
          }
        }
        if (dataFiles.length > 0) {
          cases.push({dataFiles, name: thisPath});
        }
      }
    }

    if (modelUrl === null) {
      throw new Error('there are no model file under the folder specified');
    }
  } catch (e) {
    logger.error('TestRunnerCli.Init.Model', `Failed to prepare test data. Error: ${inspect(e)}`);
    throw e;
  }

  logger.verbose('TestRunnerCli.Init.Model', `Finished preparing test data.`);
  logger.verbose('TestRunnerCli.Init.Model', `===============================================================`);
  logger.verbose('TestRunnerCli.Init.Model', ` Model file: ${modelUrl}`);
  logger.verbose('TestRunnerCli.Init.Model', ` Backend: ${backend}`);
  logger.verbose('TestRunnerCli.Init.Model', ` Test set(s): ${cases.length}`);
  logger.verbose('TestRunnerCli.Init.Model', `===============================================================`);

  return {name: path.basename(testDataRootFolder), modelUrl, backend, cases};
}

function tryLocateModelTestFolder(searchPattern: string): string {
  for (const folderCandidate of globby.sync(
           [searchPattern, path.join(TEST_DATA_ROOT, '**', searchPattern)], {onlyDirectories: true, absolute: true})) {
    const modelCandidates = globby.sync('*.onnx', {onlyFiles: true, cwd: folderCandidate});
    if (modelCandidates && modelCandidates.length === 1) {
      return folderCandidate;
    }
  }

  throw new Error(`no model folder found: ${searchPattern}`);
}

function opTests(backend: string): Test.OperatorTestGroup[] {
  const groups: Test.OperatorTestGroup[] = [];
  for (const thisPath of fs.readdirSync(TEST_DATA_OP_ROOT)) {
    const thisFullPath = path.join(TEST_DATA_OP_ROOT, thisPath);
    const stat = fs.lstatSync(thisFullPath);
    const ext = path.extname(thisFullPath);
    if (stat.isFile() && (ext === '.json' || ext === '.jsonc')) {
      const skip = whitelist[backend].ops.indexOf(thisPath) === -1;
      groups.push(opTestFromManifest(thisFullPath, backend, skip));
    }
  }

  return groups;
}

function opTestFromManifest(manifestFile: string, backend: string, skip = false): Test.OperatorTestGroup {
  let tests: Test.OperatorTest[] = [];
  const filePath = path.resolve(process.cwd(), manifestFile);
  if (skip) {
    logger.verbose('TestRunnerCli.Init.Op', `Skip test data from manifest file: ${filePath}`);
  } else {
    logger.verbose('TestRunnerCli.Init.Op', `Start to prepare test data from manifest file: ${filePath}`);
    const jsonWithComments = fs.readFileSync(filePath).toString();
    const json = stripJsonComments(jsonWithComments, {whitespace: true});
    tests = JSON.parse(json) as Test.OperatorTest[];
    // field 'verbose' and 'backend' is not set
    for (const test of tests) {
      test.backend = backend;
    }
    logger.verbose('TestRunnerCli.Init.Op', `Finished preparing test data.`);
    logger.verbose('TestRunnerCli.Init.Op', `===============================================================`);
    logger.verbose('TestRunnerCli.Init.Op', ` Test Group: ${path.relative(TEST_DATA_OP_ROOT, filePath)}`);
    logger.verbose('TestRunnerCli.Init.Op', ` Backend: ${backend}`);
    logger.verbose('TestRunnerCli.Init.Op', ` Test case(s): ${tests.length}`);
    logger.verbose('TestRunnerCli.Init.Op', `===============================================================`);
  }
  return {name: path.relative(TEST_DATA_OP_ROOT, filePath), tests};
}

function tryLocateOpTestManifest(searchPattern: string): string {
  for (const manifestCandidate of globby.sync(
           [
             searchPattern, path.join(TEST_DATA_ROOT, '**', searchPattern),
             path.join(TEST_DATA_ROOT, '**', searchPattern + '.json'),
             path.join(TEST_DATA_ROOT, '**', searchPattern + '.jsonc')
           ],
           {onlyFiles: true, absolute: true})) {
    return manifestCandidate;
  }

  throw new Error(`no OP test manifest found: ${searchPattern}`);
}

function run(config: Test.Config) {
  // STEP 1. we write the config to testdata.js
  logger.info('TestRunnerCli.Run', '(1/4) Writing config to file: testdata.js ...');
  saveConfig(config);
  logger.info('TestRunnerCli.Run', '(1/4) Writing config to file: testdata.js ... DONE');

  // STEP 2. get npm bin folder
  logger.info('TestRunnerCli.Run', '(2/4) Retrieving npm bin folder...');
  const npmBin = execSync('npm bin', {encoding: 'utf8'}).trimRight();
  logger.info('TestRunnerCli.Run', `(2/4) Retrieving npm bin folder... DONE, folder: ${npmBin}`);

  if (env === 'node') {
    // STEP 3. use webpack to generate ONNX.js
    logger.info('TestRunnerCli.Run', '(3/4) Running tsc...');
    const tscCommand = path.join(npmBin, 'tsc');
    const webpack = spawnSync(tscCommand, {shell: true, stdio: 'inherit'});
    if (webpack.status !== 0) {
      console.error(webpack.error);
      process.exit(webpack.status);
    }
    logger.info('TestRunnerCli.Run', '(3/4) Running tsc... DONE');

    // STEP 4. run mocha
    logger.info('TestRunnerCli.Run', '(4/4) Running mocha...');
    const mochaCommand = path.join(npmBin, 'mocha');
    const mochaArgs = [path.join(TEST_ROOT, 'unittest'), '--timeout 60000'];
    logger.info('TestRunnerCli.Run', `CMD: ${mochaCommand} ${mochaArgs.join(' ')}`);
    const mocha = spawnSync(mochaCommand, mochaArgs, {shell: true, stdio: 'inherit'});
    if (mocha.status !== 0) {
      console.error(mocha.error);
      process.exit(mocha.status);
    }
    logger.info('TestRunnerCli.Run', '(4/4) Running mocha... DONE');

  } else {
    // STEP 3. use webpack to generate ONNX.js
    logger.info('TestRunnerCli.Run', '(3/4) Running webpack to generate ONNX.js...');
    const webpackCommand = path.join(npmBin, 'webpack');
    const webpackArgs = ['--mode', 'development'];
    logger.info('TestRunnerCli.Run', `CMD: ${webpackCommand} ${webpackArgs.join(' ')}`);
    const webpack = spawnSync(webpackCommand, webpackArgs, {shell: true, stdio: 'inherit'});
    if (webpack.status !== 0) {
      console.error(webpack.error);
      process.exit(webpack.status);
    }
    logger.info('TestRunnerCli.Run', '(3/4) Running webpack to generate ONNX.js... DONE');

    // STEP 4. use Karma to run test
    logger.info('TestRunnerCli.Run', '(4/4) Running karma to start test runner...');
    const karmaCommand = path.join(npmBin, 'karma');
    // currently only ChromeDebug, ChromeTest, Edge, Firefox and Electron browsers are supported
    const browser = (env === 'chrome') ?
        (debug ? 'ChromeDebug' : 'ChromeTest') :
        (env === 'edge') ?
        'Edge' :
        (env === 'firefox') ? 'Firefox' : (env === 'electron') ? 'Electron' : (env === 'safari') ? 'Safari' : '';
    const karmaArgs = ['start', `--browsers ${browser}`];
    if (debug) {
      karmaArgs.push('--log-level info');
    } else {
      karmaArgs.push('--single-run');
    }
    if (noSandbox) {
      karmaArgs.push('--no-sandbox');
    }
    // == Special treatment to Microsoft Edge ==
    //
    // == Edge's Auto Recovery Feature ==
    // when Edge starts, if it found itself was terminated forcely last time, it always recovers all previous pages.
    // this always happen in Karma because `karma-edge-launcher` uses `taskkill` command to kill Edge every time.
    //
    // == The Problem ==
    // every time when a test is completed, it will be added to the recovery page list.
    // if we run the test 100 times, there will be 100 previous tabs when we launch Edge again.
    // this run out of resources quickly and fails the futher test.
    // and it cannot recover by itself because every time it is terminated forcely or crashes.
    // and the auto recovery feature has no way to disable by configuration/commandline/registry
    //
    // == The Solution ==
    // for Microsoft Edge, we should clean up the previous active page before each run
    // delete the files stores in the specific folder to clean up the recovery page list.
    // see also: https://www.laptopmag.com/articles/edge-browser-stop-tab-restore
    if (browser === 'Edge') {
      const deleteEdgeActiveRecoveryCommand =
          // tslint:disable-next-line:max-line-length
          `del /F /Q %LOCALAPPDATA%\\Packages\\Microsoft.MicrosoftEdge_8wekyb3d8bbwe\\AC\\MicrosoftEdge\\User\\Default\\Recovery\\Active\\*`;
      logger.info('TestRunnerCli.Run', `CMD: ${deleteEdgeActiveRecoveryCommand}`);
      spawnSync(deleteEdgeActiveRecoveryCommand, {shell: true, stdio: 'inherit'});
    }
    logger.info('TestRunnerCli.Run', `CMD: ${karmaCommand} ${karmaArgs.join(' ')}`);
    const karma = spawnSync(karmaCommand, karmaArgs, {shell: true, stdio: 'inherit'});
    if (karma.status !== 0) {
      console.error(karma.error);
      process.exit(karma.status);
    }
    logger.info('TestRunnerCli.Run', '(4/4) Running karma to start test runner... DONE');
  }
}

function saveConfig(config: Test.Config) {
  fs.writeFileSync(path.join(TEST_ROOT, './testdata.js'), `module.exports=${JSON.stringify(config, null, 2)};`);
}

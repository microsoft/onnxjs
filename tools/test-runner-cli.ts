// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {execSync, spawnSync} from 'child_process';
import * as fs from 'fs';
import * as globby from 'globby';
import {default as minimatch} from 'minimatch';
import logger from 'npmlog';
import * as path from 'path';
import stripJsonComments from 'strip-json-comments';
import {inspect} from 'util';

import {bufferToBase64} from '../test/test-shared';
import {Test} from '../test/test-types';

import {parseTestRunnerCliArgs, TestRunnerCliArgs} from './test-runner-cli-args';

logger.info('TestRunnerCli', 'Initializing...');

const args = parseTestRunnerCliArgs(process.argv.slice(2));

logger.verbose('TestRunnerCli.Init.Config', inspect(args));

const TEST_ROOT = path.join(__dirname, '..', 'test');
const TEST_DATA_MODEL_NODE_ROOT = path.join(TEST_ROOT, 'data', 'node');
const TEST_DATA_MODEL_ONNX_ROOT = path.join(__dirname, '..', 'deps/data/data/test/onnx/v7');
const TEST_DATA_OP_ROOT = path.join(TEST_ROOT, 'data', 'ops');

const TEST_DATA_BASE = args.env === 'node' ? TEST_ROOT : '/base/test/';

logger.verbose('TestRunnerCli.Init', `Loading whitelist...`);

// The following is a whitelist of unittests for already implemented operators.
// Modify this list to control what node tests to run.
const jsonWithComments = fs.readFileSync(path.resolve(TEST_ROOT, './unittest-whitelist.jsonc')).toString();
const json = stripJsonComments(jsonWithComments, {whitespace: true});
const whitelist = JSON.parse(json) as Test.WhiteList;
logger.verbose('TestRunnerCli.Init', `Loading whitelist... DONE`);

// The default backends and opset version lists. Those will be used in suite tests.
const DEFAULT_BACKENDS: ReadonlyArray<TestRunnerCliArgs.Backend> =
    args.env === 'node' ? ['cpu', 'wasm'] : ['cpu', 'wasm', 'webgl'];
const DEFAULT_OPSET_VERSIONS: ReadonlyArray<number> = [10, 9, 8, 7];

const fileCache: Test.FileCache = {};

const nodeTests = new Map<string, Test.ModelTestGroup[]>();
const onnxTests = new Map<string, Test.ModelTestGroup>();
const opTests = new Map<string, Test.OperatorTestGroup[]>();

const shouldLoadSuiteTestData = (args.mode === 'suite0' || args.mode === 'suite1');
if (shouldLoadSuiteTestData) {
  logger.verbose('TestRunnerCli.Init', `Loading test groups for suite test...`);

  for (const backend of DEFAULT_BACKENDS) {
    for (const version of DEFAULT_OPSET_VERSIONS) {
      let nodeTest = nodeTests.get(backend);
      if (!nodeTest) {
        nodeTest = [];
        nodeTests.set(backend, nodeTest);
      }
      nodeTest.push(loadNodeTests(backend, version));
    }
    onnxTests.set(backend, loadOnnxTests(backend));
    opTests.set(backend, loadOpTests(backend));
  }
}

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
switch (args.mode) {
  case 'suite1':
    for (const backend of DEFAULT_BACKENDS) {
      if (args.backends.indexOf(backend) !== -1) {
        modelTestGroups.push(onnxTests.get(backend)!);  // model test : ONNX model
      }
    }

  case 'suite0':
    for (const backend of DEFAULT_BACKENDS) {
      if (args.backends.indexOf(backend) !== -1) {
        modelTestGroups.push(...nodeTests.get(backend)!);  // model test : node
        opTestGroups.push(...opTests.get(backend)!);       // operator test
      }
    }
    unittest = true;
    break;

  case 'model':
    if (!args.param) {
      throw new Error(`the test folder should be specified in mode 'node'`);
    }
    const testFolderSearchPattern = args.param;
    const testFolder = tryLocateModelTestFolder(testFolderSearchPattern);
    for (const b of args.backends) {
      modelTestGroups.push(
          {name: testFolder, tests: [modelTestFromFolder(testFolder, b, false, undefined, args.times)]});
    }
    break;

  case 'unittest':
    unittest = true;
    break;

  case 'op':
    if (!args.param) {
      throw new Error(`the test manifest should be specified in mode 'op'`);
    }
    const manifestFileSearchPattern = args.param;
    const manifestFile = tryLocateOpTestManifest(manifestFileSearchPattern);
    for (const b of args.backends) {
      opTestGroups.push(opTestFromManifest(manifestFile, b));
    }
    break;
  default:
    throw new Error(`unsupported mode '${args.mode}'`);
}

logger.verbose('TestRunnerCli.Init', `Preparing test config... DONE`);

logger.info('TestRunnerCli', 'Initialization completed. Start to run tests...');
run({
  unittest,
  model: modelTestGroups,
  op: opTestGroups,
  fileCache,
  log: args.logConfig,
  profile: args.profile,
  options: {debug: args.debug, cpu: args.cpuOptions, webgl: args.webglOptions, wasm: args.wasmOptions}
});
logger.info('TestRunnerCli', 'Tests completed successfully');

process.exit();

function validateWhiteList() {
  for (const backend of DEFAULT_BACKENDS) {
    const nodeTest = nodeTests.get(backend);
    if (nodeTest) {
      for (const testCase of whitelist[backend].node) {
        const testCaseName = typeof testCase === 'string' ? testCase : testCase.name;
        let found = false;
        for (const testGroup of nodeTest) {
          found =
              found || testGroup.tests.some(test => minimatch(test.modelUrl, path.join('**', testCaseName, '*.onnx')));
        }
        if (!found) {
          throw new Error(`node model test case '${testCaseName}' in white list does not exist.`);
        }
      }
    }

    const onnxTest = onnxTests.get(backend);
    if (onnxTest) {
      const onnxModelTests = onnxTest.tests.map(i => i.name);
      for (const testCase of whitelist[backend].onnx) {
        const testCaseName = typeof testCase === 'string' ? testCase : testCase.name;
        if (onnxModelTests.indexOf(testCaseName) === -1) {
          throw new Error(`onnx model test case '${testCaseName}' in white list does not exist.`);
        }
      }
    }

    const opTest = opTests.get(backend);
    if (opTest) {
      const opTests = opTest.map(i => i.name);
      for (const testCase of whitelist[backend].ops) {
        const testCaseName = typeof testCase === 'string' ? testCase : testCase.name;
        if (opTests.indexOf(testCaseName) === -1) {
          throw new Error(`operator test case '${testCaseName}' in white list does not exist.`);
        }
      }
    }
  }
}

function loadNodeTests(backend: string, version: number): Test.ModelTestGroup {
  return suiteFromFolder(
      `node-opset_v${version}-${backend}`, path.join(TEST_DATA_MODEL_NODE_ROOT, `v${version}`), backend, true,
      whitelist[backend].node);
}

function loadOnnxTests(backend: string): Test.ModelTestGroup {
  return suiteFromFolder(`onnx-${backend}`, TEST_DATA_MODEL_ONNX_ROOT, backend, false, whitelist[backend].onnx);
}

function suiteFromFolder(
    name: string, suiteRootFolder: string, backend: string, preload: boolean,
    whitelist?: ReadonlyArray<Test.WhiteList.Test>): Test.ModelTestGroup {
  const sessions: Test.ModelTest[] = [];
  const tests = fs.readdirSync(suiteRootFolder);
  for (const test of tests) {
    let condition: Test.Condition|undefined;
    let times: number|undefined;
    if (whitelist) {
      const matches = whitelist.filter(
          p => minimatch(path.join(suiteRootFolder, test), path.join('**', typeof p === 'string' ? p : p.name)));
      if (matches.length === 0) {
        times = 0;
      } else if (matches.length === 1) {
        const match = matches[0];
        if (typeof match !== 'string') {
          condition = match.condition;
        }
      } else {
        throw new Error(`multiple whitelist rules matches test: ${path.join(suiteRootFolder, test)}`);
      }
    }
    sessions.push(modelTestFromFolder(path.resolve(suiteRootFolder, test), backend, preload, condition, times));
  }
  return {name, tests: sessions};
}

function modelTestFromFolder(
    testDataRootFolder: string, backend: string, preload: boolean, condition?: Test.Condition,
    times?: number): Test.ModelTest {
  if (times === 0) {
    logger.verbose('TestRunnerCli.Init.Model', `Skip test data from folder: ${testDataRootFolder}`);
    return {name: path.basename(testDataRootFolder), backend, modelUrl: '', cases: []};
  }

  let modelUrl: string|null = null;
  let cases: Test.ModelTestCase[] = [];

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
            if (preload && !fileCache[modelUrl]) {
              fileCache[modelUrl] = bufferToBase64(fs.readFileSync(thisFullPath));
            }
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
            const dataFileUrl = path.join(TEST_DATA_BASE, path.relative(TEST_ROOT, dataFileFullPath));
            dataFiles.push(dataFileUrl);
            if (preload && !fileCache[dataFileUrl]) {
              fileCache[dataFileUrl] = bufferToBase64(fs.readFileSync(dataFileFullPath));
            }
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

  const caseCount = cases.length;
  if (times !== undefined) {
    if (times > caseCount) {
      for (let i = 0; cases.length < times; i++) {
        const origin = cases[i % caseCount];
        const duplicated = {name: `${origin.name} - copy ${Math.floor(i / caseCount)}`, dataFiles: origin.dataFiles};
        cases.push(duplicated);
      }
    } else {
      cases = cases.slice(0, times);
    }
  }

  logger.verbose('TestRunnerCli.Init.Model', `Finished preparing test data.`);
  logger.verbose('TestRunnerCli.Init.Model', `===============================================================`);
  logger.verbose('TestRunnerCli.Init.Model', ` Model file: ${modelUrl}`);
  logger.verbose('TestRunnerCli.Init.Model', ` Backend: ${backend}`);
  logger.verbose('TestRunnerCli.Init.Model', ` Test set(s): ${cases.length} (${caseCount})`);
  logger.verbose('TestRunnerCli.Init.Model', `===============================================================`);

  return {name: path.basename(testDataRootFolder), condition, modelUrl, backend, cases};
}

function tryLocateModelTestFolder(searchPattern: string): string {
  const folderCandidates: string[] = [];
  // 1 - check whether search pattern is a directory
  if (fs.existsSync(searchPattern) && fs.lstatSync(searchPattern).isDirectory()) {
    folderCandidates.push(searchPattern);
  }

  // 2 - check the globby result of searchPattern
  // 3 - check the globby result of ONNX root combined with searchPattern
  const globbyPattern = [searchPattern, path.join(TEST_DATA_MODEL_ONNX_ROOT, '**', searchPattern)];
  // 4 - check the globby result of NODE root combined with opset versions and searchPattern
  globbyPattern.push(
      ...DEFAULT_OPSET_VERSIONS.map(v => path.join(TEST_DATA_MODEL_NODE_ROOT, `v${v}`, '**', searchPattern)));

  folderCandidates.push(...globby.sync(
      [
        searchPattern, path.join(TEST_DATA_MODEL_ONNX_ROOT, '**', searchPattern),
        path.join(TEST_DATA_MODEL_NODE_ROOT, '**', searchPattern)
      ],
      {onlyDirectories: true, absolute: true}));

  // pick the first folder that matches the pattern
  for (const folderCandidate of folderCandidates) {
    const modelCandidates = globby.sync('*.onnx', {onlyFiles: true, cwd: folderCandidate});
    if (modelCandidates && modelCandidates.length === 1) {
      return folderCandidate;
    }
  }

  throw new Error(`no model folder found: ${searchPattern}`);
}

function loadOpTests(backend: string): Test.OperatorTestGroup[] {
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
             searchPattern, path.join(TEST_DATA_OP_ROOT, '**', searchPattern),
             path.join(TEST_DATA_OP_ROOT, '**', searchPattern + '.json'),
             path.join(TEST_DATA_OP_ROOT, '**', searchPattern + '.jsonc')
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

  if (args.env === 'node') {
    // STEP 3. use tsc to build ONNX.js
    logger.info('TestRunnerCli.Run', '(3/4) Running tsc...');
    const tscCommand = path.join(npmBin, 'tsc');
    const tsc = spawnSync(tscCommand, {shell: true, stdio: 'inherit'});
    if (tsc.status !== 0) {
      console.error(tsc.error);
      process.exit(tsc.status);
    }
    logger.info('TestRunnerCli.Run', '(3/4) Running tsc... DONE');

    // STEP 4. run mocha
    logger.info('TestRunnerCli.Run', '(4/4) Running mocha...');
    const mochaCommand = path.join(npmBin, 'mocha');
    const mochaArgs = [path.join(TEST_ROOT, 'test-main'), '--timeout 60000'];
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
    const webpackArgs = [
      '--mode',
      args.bundleMode === 'dev' ? 'development' : 'production',
      `--bundle-mode=${args.bundleMode}`,
    ];
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
    const browser = getBrowserNameFromEnv(args.env, args.debug);
    const karmaArgs = ['start', `--browsers ${browser}`];
    if (args.debug) {
      karmaArgs.push('--log-level info');
    } else {
      karmaArgs.push('--single-run');
    }
    if (args.noSandbox) {
      karmaArgs.push('--no-sandbox');
    }
    karmaArgs.push(`--bundle-mode=${args.bundleMode}`);
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
  let setOptions = '';
  if (config.options.debug !== undefined) {
    setOptions += `onnx.ENV.debug = ${config.options.debug};`;
  }
  if (config.options.webgl && config.options.webgl.disabled !== undefined) {
    setOptions += `onnx.backend.webgl.disabled = ${config.options.webgl.disabled};`;
  }
  if (config.options.wasm && config.options.wasm.disabled !== undefined) {
    setOptions += `onnx.backend.wasm.disabled = ${config.options.wasm.disabled};`;
  }
  if (config.options.webgl && config.options.webgl.contextId) {
    setOptions += `onnx.backend.webgl.contextId = ${JSON.stringify(config.options.webgl.contextId)};`;
  }
  if (config.options.wasm && config.options.wasm.worker !== undefined) {
    setOptions += `onnx.backend.wasm.worker = ${JSON.stringify(config.options.wasm.worker)};`;
  }
  if (config.options.wasm && config.options.wasm.cpuFallback !== undefined) {
    setOptions += `onnx.backend.wasm.cpuFallback = ${JSON.stringify(config.options.wasm.cpuFallback)};`;
  }
  if (config.options.wasm && config.options.wasm.initTimeout !== undefined) {
    setOptions += `onnx.backend.wasm.initTimeout = ${JSON.stringify(config.options.wasm.initTimeout)};`;
  }
  if (config.model.some(testGroup => testGroup.tests.some(test => test.backend === 'onnxruntime'))) {
    setOptions += `require('onnxjs-node');`;
  }

  fs.writeFileSync(path.join(TEST_ROOT, './testdata.js'), `${setOptions}

module.exports=${JSON.stringify(config, null, 2)};`);
}

function getBrowserNameFromEnv(env: TestRunnerCliArgs['env'], debug?: boolean) {
  switch (env) {
    case 'chrome':
      return debug ? 'ChromeDebug' : 'ChromeTest';
    case 'edge':
      return 'Edge';
    case 'firefox':
      return 'Firefox';
    case 'electron':
      return 'Electron';
    case 'safari':
      return 'Safari';
    case 'bs':
      return process.env.ONNXJS_TEST_BS_BROWSERS!;
    default:
      throw new Error(`env "${env}" not supported.`);
  }
}

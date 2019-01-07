module.exports = function(config) {
  const bundleMode = require('minimist')(process.argv)['bundle-mode'] || 'dev';  // 'dev'|'perf'|undefined;

  const mainFile = bundleMode === 'perf' ? 'test/onnx.perf.js' : 'test/onnx.dev.js';

  config.set({
    // global config of your BrowserStack account
    browserStack: {
      username: process.env.BROWSER_STACK_USERNAME,
      accessKey: process.env.BROWSER_STACK_ACCESS_KEY,
      forceLocal: true,
      startTunnel: true,
    },
    frameworks: ['mocha'],
    files: [
      {pattern: mainFile},
      {pattern: 'test/onnx-worker.js', included: false},
      {pattern: 'deps/data/data/test/**/*', included: false},
      {pattern: 'dist/onnx-wasm.wasm', included: false},
    ],
    proxies: {
      '/onnx-wasm.wasm': '/base/dist/onnx-wasm.wasm',
      '/onnx-worker.js': '/base/test/onnx-worker.js',
    },
    client: {captureConsole: true, mocha: {expose: ['body'], timeout: 60000}},
    preprocessors: {mainFile: ['sourcemap']},
    reporters: ['mocha'],
    browsers: [
      'ChromeTest',
      'ChromeDebug',
      'Edge',
      'Firefox',
      'Electron',
      'Safari',
      'bs_chrome_win',
      'bs_edge_win',
      'bs_ff_win',
      'bs_chrome_mac',
      'bs_ff_mac',
    ],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    customLaunchers: {
      ChromeTest: {base: 'Chrome', flags: ['--window-size=1,1']},
      ChromeDebug: {debug: true, base: 'Chrome', flags: ['--remote-debugging-port=9333']},
      bs_chrome_win: {
        base: 'BrowserStack',
        browser: 'Chrome',
        browser_version: '71.0',
        os: 'Windows',
        os_version: '10',
      },
      bs_edge_win: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Edge',
        browser_version: '18.0',
      },
      bs_ff_win: {
        base: 'BrowserStack',
        os: 'Windows',
        os_version: '10',
        browser: 'Firefox',
        browser_version: '63.0',
      },
      bs_chrome_mac: {
        base: 'BrowserStack',
        browser: 'Chrome',
        browser_version: '71.0',
        os: 'OS X',
        os_version: 'High Sierra',
      },
      bs_ff_mac: {
        base: 'BrowserStack',
        os: 'OS X',
        os_version: 'High Sierra',
        browser: 'Firefox',
        browser_version: '63.0',
      }
    }
  });
};

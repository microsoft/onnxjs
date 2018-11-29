module.exports = function(config) {
  config.set({
    frameworks: ['mocha'],
    files: [
      {pattern: 'test/onnx.dev.js'},
      {pattern: 'test/onnx-worker.js', included: false},
      {pattern: 'deps/data/data/test/**/*', included: false},
      {pattern: 'dist/onnx-wasm.wasm', included: false},
    ],
    proxies: {
      '/onnx-wasm.wasm': '/base/dist/onnx-wasm.wasm',
      '/onnx-worker.js': '/base/test/onnx-worker.js',
    },
    client: {captureConsole: true, mocha: {expose: ['body'], timeout: 60000}},
    preprocessors: {'test/onnx.dev.js': ['sourcemap']},
    reporters: ['mocha'],
    browsers: ['ChromeTest', 'ChromeDebug', 'Edge', 'Firefox', 'Electron'],
    captureTimeout: 120000,
    reportSlowerThan: 100,
    browserDisconnectTimeout: 600000,
    browserNoActivityTimeout: 300000,
    browserDisconnectTolerance: 0,
    browserSocketTimeout: 60000,
    customLaunchers: {
      ChromeTest: {base: 'Chrome', flags: ['--window-size=1,1']},
      ChromeDebug: {debug: true, base: 'Chrome', flags: ['--remote-debugging-port=9333']}
    }
  });
};

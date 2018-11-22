const path = require('path');
const webpack = require('webpack');
const HardSourceWebpackPlugin = require('hard-source-webpack-plugin');

module.exports = (env, argv) => {
  const config = {
    resolve: {extensions: ['.ts', '.js']},
    plugins: [new webpack.WatchIgnorePlugin([/\.js$/, /\.d\.ts$/])],
    module: {rules: [{test: /\.tsx?$/, loader: 'ts-loader'}]},
    node: {fs: 'empty'}
  };

  if (argv.mode === 'production') {
    config.mode = 'production';
    config.devtool = 'source-map';
    config.entry = path.resolve(__dirname, 'lib/api/index.ts');
    config.output = {path: path.resolve(__dirname, 'dist'), filename: 'onnx.min.js', libraryTarget: 'umd'};
  } else {
    config.mode = 'development';
    config.devtool = 'inline-source-map';
    config.entry = path.resolve(__dirname, 'test/unittest.ts');
    config.plugins.push(new HardSourceWebpackPlugin());
    config.output = {path: path.resolve(__dirname, 'test'), filename: 'onnx.dev.js', libraryTarget: 'umd'};
  }

  return config;
};

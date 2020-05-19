// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {execSync, spawnSync} from 'child_process';
import npmlog from 'npmlog';
import * as path from 'path';

// tslint:disable: non-literal-fs-path

// Path variables
const ROOT = path.join(__dirname, '..');

npmlog.info('pre-commit', 'Running Lint...');
const lint = spawnSync('npm run lint', {shell: true, stdio: 'inherit', cwd: ROOT});
if (lint.status !== 0) {
  if (lint.error) {
    console.error(lint.error);
  }
  process.exit(lint.status === null ? undefined : lint.status);
}

npmlog.info('pre-commit', 'Running Clang-Format on ts and cpp...');
const clangFormat = spawnSync('npm run format', {shell: true, stdio: 'inherit', cwd: ROOT});
if (clangFormat.status !== 0) {
  if (clangFormat.error) {
    console.error(clangFormat.error);
  }
  process.exit(clangFormat.status === null ? undefined : clangFormat.status);
}

npmlog.info('pre-commit', 'Running gen-doc...');
const genDoc = spawnSync('npm run build:doc', {shell: true, stdio: 'inherit', cwd: ROOT});
if (genDoc.status !== 0) {
  if (genDoc.error) {
    console.error(genDoc.error);
  }
  process.exit(genDoc.status === null ? undefined : genDoc.status);
}

npmlog.info('pre-commit', 'Running prettier on markdown...');
const prettierMd = spawnSync('npm run format:md', {shell: true, stdio: 'inherit', cwd: ROOT});
if (prettierMd.status !== 0) {
  if (prettierMd.error) {
    console.error(prettierMd.error);
  }
  process.exit(prettierMd.status === null ? undefined : prettierMd.status);
}

npmlog.info('pre-commit', 'Running prettier on jsonc...');
const prettierJsonc = spawnSync('npm run format:jsonc', {shell: true, stdio: 'inherit', cwd: ROOT});
if (prettierJsonc.status !== 0) {
  if (prettierJsonc.error) {
    console.error(prettierJsonc.error);
  }
  process.exit(prettierJsonc.status === null ? undefined : prettierJsonc.status);
}

const lsFiles = execSync('git ls-files -m', {encoding: 'utf8', cwd: ROOT});
const modifiedFiles = lsFiles.split('\\n').map(i => i.trim());
const notFormattedFound = modifiedFiles.filter(i => ['.ts', '.cpp', '.jsonc', '.md'].some(ext => i.endsWith(ext)));
if (notFormattedFound.length > 0) {
  npmlog.error('pre-commit', 'File(s) not formatted:');
  for (const file of notFormattedFound) {
    npmlog.error('pre-commit', file);
  }
  process.exit(1);
}

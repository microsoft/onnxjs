// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

/**
 * GLSL preprocessor class responsible for resolving @inline directives
 */
export class GlslFunctionInliner {
  // tslint:disable-next-line:variable-name
  static readonly InlineFuncDefRegex: RegExp =
      /@inline[\s\n\r]+(\w+)[\s\n\r]+([0-9a-zA-Z_]+)\s*\(([^)]*)\)\s*{(([^}]|[\n\r])*)}/gm;
  // tslint:disable-next-line:variable-name
  static readonly FuncCallRegex = '(\\w+)?\\s+([_0-9a-zA-Z]+)\\s+=\\s+__FUNC__\\((.*)\\)\\s*;';

  inline(script: string) {
    const inlineDefs: {[name: string]: {params: Array<{type: string, name: string}|null>, body: string}} = {};
    let match;
    while ((match = GlslFunctionInliner.InlineFuncDefRegex.exec(script)) !== null) {
      const params = match[3]
                         .split(',')
                         .map(s => {
                           const tokens = s.trim().split(' ');
                           if (tokens && tokens.length === 2) {
                             return {type: tokens[0], name: tokens[1]};
                           }
                           return null;
                         })
                         .filter(v => v !== null);
      inlineDefs[match[2]] = {params, body: match[4]};
    }
    for (const name in inlineDefs) {
      const regexString = GlslFunctionInliner.FuncCallRegex.replace('__FUNC__', name);
      const regex = new RegExp(regexString, 'gm');
      while ((match = regex.exec(script)) !== null) {
        const type = match[1];
        const variable = match[2];
        const params = match[3].split(',');
        const declLine = (type) ? `${type} ${variable};` : '';
        let newBody: string = inlineDefs[name].body;
        let paramRedecLine = '';
        inlineDefs[name].params.forEach((v, i) => {
          if (v) {
            paramRedecLine += `${v.type} ${v.name} = ${params[i]};\n`;
          }
        });
        newBody = `${paramRedecLine}\n ${newBody}`;
        newBody = newBody.replace('return', `${variable} = `);
        const replacement = `
        ${declLine}
        {
          ${newBody}
        }
        `;
        script = script.replace(match[0], replacement);
      }
    }
    script = script.replace(GlslFunctionInliner.InlineFuncDefRegex, '');
    return script;
  }
}

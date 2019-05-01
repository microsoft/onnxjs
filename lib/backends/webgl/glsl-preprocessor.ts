// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {GlslContext, GlslLib, GlslLibRoutineNode, TopologicalSortGlslRoutines} from './glsl-definitions';
import {GlslFunctionInliner} from './glsl-function-inliner';
import {glslRegistry} from './glsl-registered-libs';
import {ProgramInfo, VariableInfo} from './types';
import {WebGLContext} from './webgl-context';

/**
 * Preprocessor for the additions to the GLSL language
 * It deals with:
 *  @include directives
 *  @inline
 *  Loop unrolling (not implemented)
 *  Macro resolution (not implemented)
 */
export class GlslPreprocessor {
  context: GlslContext;
  libs: {[name: string]: GlslLib};
  inliner: GlslFunctionInliner;
  glslLibRoutineDependencyGraph: {[routineName: string]: GlslLibRoutineNode} = {};
  shaderSource: string;

  constructor(glContext: WebGLContext, programInfo: ProgramInfo) {
    this.context = new GlslContext(glContext, programInfo, [], []);
    this.inliner = new GlslFunctionInliner();
    this.libs = {};
    Object.keys(glslRegistry).forEach((name: string) => {
      const lib = new glslRegistry[name](this.context);
      this.libs[name] = lib;
    });
    this.shaderSource = this.extractUniformInfo(programInfo.shaderSource);
    this.glslLibRoutineDependencyGraph = this.constructGlslRoutineDependencyGraph();
  }

  preprocess(): string {
    let s = this.shaderSource;
    if (!this.context.programInfo.hasMain) {
      s = this.addClosing(s);
    }
    s = this.processImports(s);
    s = this.processMacros(s);
    s = this.processInlines(s);
    s = this.addUniforms(s, this.context.uniformInfo);
    s = this.addPreamble(s);
    return s;
  }
  extractAttribInfo(shaderSource: string): void {
    const attribRegex = /^\s*attribute (\w+) (\w+);/gm;
    for (const item of this.getVariableMatches(attribRegex, shaderSource)) {
      const matches = item as string[];
      this.context.attribInfo.push({type: matches[0], name: matches[1], isVec: matches[2] ? true : false});
    }
  }
  extractUniformInfo(shaderSource: string): string {
    const uniformRegex = /^\s*uniform (?:\w+ )?(\w+) (\w+)(\[\d+\])?;/gm;
    for (const item of this.getVariableMatches(uniformRegex, shaderSource)) {
      const matches = item as string[];
      this.context.uniformInfo.push(
          {type: matches[0], name: matches[1], isVec: matches[2] ? true : false, arraySuffix: matches[2]});
    }
    return shaderSource.replace(uniformRegex, '');
  }
  protected addPreamble(script: string): string {
    return `
    precision highp float;
    precision highp int;
    precision highp sampler2D;
    varying vec2 TexCoords;

    ${script}
    `;
  }
  protected addClosing(script: string): string {
    const rank = this.context.programInfo.outputLayout.shape.length;
    return `
    ${script}
    void main() {
      int indices[${rank}];
      toVec(TexCoords, indices);
      vec4 result = vec4(process(indices));
      gl_FragColor = result;
    }
    `;
  }
  protected processImports(script: string): string {
    const routinesIncluded = this.selectGlslLibRoutinesToBeIncluded(script);

    if (routinesIncluded.length === 0) {
      return `
      ${script}
      `;
    }

    let routines = ``;
    for (let i = 0; i < routinesIncluded.length; ++i) {
      if (routinesIncluded[i].routineBody) {
        routines += routinesIncluded[i].routineBody + `\n`;
      } else {
        throw new Error(`Missing body for the Glsl Library routine: ${routinesIncluded[i].name}`);
      }
    }

    return `
    ${routines}
    ${script}
    `;
  }
  protected addUniforms(script: string, uniforms: VariableInfo[]): string {
    const uniformLines: string[] = [];
    uniforms.forEach(vi => {
      const arraySuffix = vi.arraySuffix ? vi.arraySuffix : '';
      uniformLines.push(`uniform ${vi.type} ${vi.name}${arraySuffix};`);
    });
    return `
    ${uniformLines.join('\n')}
    ${script}
    `;
  }
  private selectGlslLibRoutinesToBeIncluded(script: string): GlslLibRoutineNode[] {
    const nodes: GlslLibRoutineNode[] = [];

    Object.keys(this.glslLibRoutineDependencyGraph).forEach(classAndRoutine => {
      const routine = classAndRoutine.split('.')[1];
      if (script.indexOf(routine) !== -1) {
        nodes.push(this.glslLibRoutineDependencyGraph[classAndRoutine]);
      }
    });

    return TopologicalSortGlslRoutines.returnOrderedNodes(nodes);
  }

  private constructGlslRoutineDependencyGraph(): {[routineName: string]: GlslLibRoutineNode;} {
    const map: {[routineName: string]: GlslLibRoutineNode;} = {};
    for (const libName in this.libs) {
      const lib = this.libs[libName];
      const routinesInLib = lib.getFunctions();
      for (const routine in routinesInLib) {
        const key = libName + '.' + routine;
        let currentNode: GlslLibRoutineNode;
        if (map[key]) {
          currentNode = map[key];
          currentNode.routineBody = routinesInLib[routine].routineBody;
        } else {
          currentNode = new GlslLibRoutineNode(key, routinesInLib[routine].routineBody);
          map[key] = currentNode;
        }
        const dependencies = routinesInLib[routine].dependencies;
        if (dependencies) {
          for (let i = 0; i < dependencies.length; ++i) {
            if (!map[dependencies[i]]) {
              const node = new GlslLibRoutineNode(dependencies[i]);
              map[dependencies[i]] = node;
              currentNode.addDependency(node);
            } else {
              currentNode.addDependency(map[dependencies[i]]);
            }
          }
        }
      }
    }
    return map;
  }

  protected processMacros(script: string): string {
    return script;
  }
  protected processInlines(script: string): string {
    return this.inliner.inline(script);
  }
  protected getVariableMatches(regex: RegExp, src: string): object[] {
    const result: object[] = [];
    let match;
    while ((match = regex.exec(src)) !== null) {
      if (match.length === 4) {
        result.push([match[1], match[2], match[3]]);
      } else {
        result.push([match[1], match[2], null]);
      }
    }
    return result;
  }
}

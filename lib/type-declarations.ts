// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

declare module 'ndarray-gemm' {
import ndarray from 'ndarray';
  export default function matrixProduct(c: ndarray, a: ndarray, b: ndarray, alpha?: number, beta?: number): void;
}

declare module 'ndarray-ops' {
import ndarray from 'ndarray';
  export function assign<T>(dest: ndarray<T>, src: ndarray<T>): ndarray<T>;
  export function assigns<T>(dest: ndarray<T>, val: T): ndarray<T>;

  // add[,s,eq,seq] - Addition, +
  export function add(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function adds(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function addeq(dest: ndarray, arg1: ndarray): ndarray;
  export function addseq(dest: ndarray, scalar: number): ndarray;

  // sub[,s,eq,seq] - Subtraction, -
  export function sub(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function subs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function subeq(dest: ndarray, arg1: ndarray): ndarray;
  export function subseq(dest: ndarray, scalar: number): ndarray;

  // mul[,s,eq,seq] - Multiplication, *
  export function mul(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function muls(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function muleq(dest: ndarray, arg1: ndarray): ndarray;
  export function mulseq(dest: ndarray, scalar: number): ndarray;

  // div[,s,eq,seq] - Division, /
  export function div(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function divs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function diveq(dest: ndarray, arg1: ndarray): ndarray;
  export function divseq(dest: ndarray, scalar: number): ndarray;

  // mod[,s,eq,seq] - Modulo, %
  export function mod(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function mods(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function modeq(dest: ndarray, arg1: ndarray): ndarray;
  export function modseq(dest: ndarray, scalar: number): ndarray;

  // band[,s,eq,seq] - Bitwise And, &
  export function band(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function bands(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function bandeq(dest: ndarray, arg1: ndarray): ndarray;
  export function bandseq(dest: ndarray, scalar: number): ndarray;

  // bor[,s,eq,seq] - Bitwise Or, &
  export function bor(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function bors(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function boreq(dest: ndarray, arg1: ndarray): ndarray;
  export function borseq(dest: ndarray, scalar: number): ndarray;

  // bxor[,s,eq,seq] - Bitwise Xor, ^
  export function bxor(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function bxors(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function bxoreq(dest: ndarray, arg1: ndarray): ndarray;
  export function bxorseq(dest: ndarray, scalar: number): ndarray;

  // lshift[,s,eq,seq] - Left shift, <<
  export function lshift(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function lshifts(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function lshifteq(dest: ndarray, arg1: ndarray): ndarray;
  export function lshiftseq(dest: ndarray, scalar: number): ndarray;

  // rshift[,s,eq,seq] - Signed right shift, >>
  export function rshift(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function rshifts(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function rshifteq(dest: ndarray, arg1: ndarray): ndarray;
  export function rshiftseq(dest: ndarray, scalar: number): ndarray;

  // rrshift[,s,eq,seq] - Unsigned right shift, >>>
  export function rrshift(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function rrshifts(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function rrshifteq(dest: ndarray, arg1: ndarray): ndarray;
  export function rrshiftseq(dest: ndarray, scalar: number): ndarray;

  // lt[,s,eq,seq] - Less than, <
  export function lt(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function lts(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function lteq(dest: ndarray, arg1: ndarray): ndarray;
  export function ltseq(dest: ndarray, scalar: number): ndarray;

  // gt[,s,eq,seq] - Greater than, >
  export function gt(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function gts(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function gteq(dest: ndarray, arg1: ndarray): ndarray;
  export function gtseq(dest: ndarray, scalar: number): ndarray;

  // leq[,s,eq,seq] - Less than or equal, <=
  export function leq(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function leqs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function leqeq(dest: ndarray, arg1: ndarray): ndarray;
  export function leqseq(dest: ndarray, scalar: number): ndarray;

  // geq[,s,eq,seq] - Greater than or equal >=
  export function geq(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function geqs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function geqeq(dest: ndarray, arg1: ndarray): ndarray;
  export function geqseq(dest: ndarray, scalar: number): ndarray;

  // eq[,s,eq,seq] - Equals, ===
  export function eq(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function eqs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function eqeq(dest: ndarray, arg1: ndarray): ndarray;
  export function eqseq(dest: ndarray, scalar: number): ndarray;

  // neq[,s,eq,seq] - Not equals, !==
  export function neq(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function neqs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function neqeq(dest: ndarray, arg1: ndarray): ndarray;
  export function neqseq(dest: ndarray, scalar: number): ndarray;

  // and[,s,eq,seq] - Boolean And, &&
  export function and(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function ands(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function andeq(dest: ndarray, arg1: ndarray): ndarray;
  export function andseq(dest: ndarray, scalar: number): ndarray;

  // or[,s,eq,seq] - Boolean Or, ||
  export function or(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function ors(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function oreq(dest: ndarray, arg1: ndarray): ndarray;
  export function orseq(dest: ndarray, scalar: number): ndarray;

  // max[,s,eq,seq] - Maximum, Math.max
  export function max(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function maxs(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function maxeq(dest: ndarray, arg1: ndarray): ndarray;
  export function maxseq(dest: ndarray, scalar: number): ndarray;

  // min[,s,eq,seq] - Minimum, Math.min
  export function min(dest: ndarray, arg1: ndarray, arg2: ndarray): ndarray;
  export function mins(dest: ndarray, arg1: ndarray, scalar: number): ndarray;
  export function mineq(dest: ndarray, arg1: ndarray): ndarray;
  export function minseq(dest: ndarray, scalar: number): ndarray;

  // not[,eq] - Boolean not, !
  export function not(dest: ndarray, arg: ndarray): ndarray;
  export function noteq(dest: ndarray): ndarray;

  // bnot[,eq] - Bitwise not, ~
  export function bnot(dest: ndarray, arg: ndarray): ndarray;
  export function bnoteq(dest: ndarray): ndarray;

  // neg[,eq] - Negative, -
  export function neg(dest: ndarray, arg: ndarray): ndarray;
  export function negeq(dest: ndarray): ndarray;

  // recip[,eq] - Reciprocal, 1.0/
  export function recip(dest: ndarray, arg: ndarray): ndarray;
  export function recipeq(dest: ndarray): ndarray;

  // abs[,eq] - Absolute value, Math.abs
  export function abs(dest: ndarray, arg: ndarray): ndarray;
  export function abseq(dest: ndarray): ndarray;

  // acos[,eq] - Inverse cosine, Math.acos
  export function acos(dest: ndarray, arg: ndarray): ndarray;
  export function acoseq(dest: ndarray): ndarray;

  // asin[,eq] - Inverse sine, Math.asin
  export function asin(dest: ndarray, arg: ndarray): ndarray;
  export function asineq(dest: ndarray): ndarray;

  // atan[,eq] - Inverse tangent, Math.atan
  export function atan(dest: ndarray, arg: ndarray): ndarray;
  export function ataneq(dest: ndarray): ndarray;

  // ceil[,eq] - Ceiling, Math.ceil
  export function ceil(dest: ndarray, arg: ndarray): ndarray;
  export function ceileq(dest: ndarray): ndarray;

  // cos[,eq] - Cosine, Math.cos
  export function cos(dest: ndarray, arg: ndarray): ndarray;
  export function coseq(dest: ndarray): ndarray;

  // exp[,eq] - Exponent, Math.exp
  export function exp(dest: ndarray, arg: ndarray): ndarray;
  export function expeq(dest: ndarray): ndarray;

  // floor[,eq] - Floor, Math.floor
  export function floor(dest: ndarray, arg: ndarray): ndarray;
  export function flooreq(dest: ndarray): ndarray;

  // log[,eq] - Logarithm, Math.log
  export function log(dest: ndarray, arg: ndarray): ndarray;
  export function logeq(dest: ndarray): ndarray;

  // round[,eq] - Round, Math.round
  export function round(dest: ndarray, arg: ndarray): ndarray;
  export function roundeq(dest: ndarray): ndarray;

  // sin[,eq] - Sine, Math.sin
  export function sin(dest: ndarray, arg: ndarray): ndarray;
  export function sineq(dest: ndarray): ndarray;

  // sqrt[,eq] - Square root, Math.sqrt
  export function sqrt(dest: ndarray, arg: ndarray): ndarray;
  export function sqrteq(dest: ndarray): ndarray;

  // tan[,eq] - Tangent, Math.tan
  export function tan(dest: ndarray, arg: ndarray): ndarray;
  export function taneq(dest: ndarray): ndarray;
}

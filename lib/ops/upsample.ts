// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Attribute} from '../attribute';
import {InferenceHandler} from '../backend';
import {Graph} from '../graph';
import {Operator} from '../operators';
import {Tensor} from '../tensor';

export declare namespace Upsample {
  interface GetNearestPixelFunc {
    (a: number, b: boolean): number;
  }
  interface GetOriginalCoordinateFunc {
    (xResized: number, xScale: number, lengthResized: number, lengthOriginal: number, roiStart: number,
     roiEnd: number): number;
  }
}

export abstract class Upsample implements Operator {
  constructor(protected opset: number) {}

  abstract run(inferenceHandler: InferenceHandler, inputs: Tensor[]): Tensor[]|Promise<Tensor[]>;

  initialize(attributes: Attribute, node: Graph.Node, graph: Graph): void {
    this.isResize = (this.opset >= 10);

    this.mode = attributes.getString('mode', 'nearest');
    if (this.mode !== 'nearest' && this.mode !== 'linear' && (this.opset < 11 || this.mode !== 'cubic')) {
      throw new Error(`unrecognized mode: ${this.mode}`);
    }

    if (this.opset < 9) {
      this.scales = attributes.getFloats('scales');
      scalesValidataion(this.scales, this.mode, this.isResize);
    }

    this.extrapolationValue = attributes.getFloat('extrapolation_value', 0.0);

    this.coordinateTransformMode =
        this.opset > 10 ? attributes.getString('coordinate_transformation_mode', 'half_pixel') : 'asymmetric';
    if ([
          'asymmetric', 'pytorch_half_pixel', 'tf_half_pixel_for_nn', 'align_corners', 'tf_crop_and_resize',
          'half_pixel'
        ].indexOf(this.coordinateTransformMode) === -1) {
      throw new Error(`coordinate_transform_mode '${this.coordinateTransformMode}' is not supported`);
    }
    this.useExtrapolation = this.needRoiInput = (this.coordinateTransformMode === 'tf_crop_and_resize');

    this.nearestMode =
        (this.mode === 'nearest' && this.opset >= 11) ? attributes.getString('nearest_mode', 'round_prefer_floor') : '';
    if (['round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil', ''].indexOf(this.nearestMode) === -1) {
      throw new Error(`nearest_mode '${this.nearestMode}' is not supported`);
    }

    this.cubicCoefficientA = attributes.getFloat('cubic_coeff_a', -0.75);
    this.excludeOutside = attributes.getInt('exclude_outside', 0) !== 0;
    if (this.excludeOutside && this.mode !== 'cubic') {
      throw new Error('exclude_outside can be set to 1 only when mode is CUBIC.');
    }

    this.useNearest2xOptimization = (this.opset < 11) ?
        true :
        (this.mode === 'nearest' && this.coordinateTransformMode === 'asymmetric' && this.nearestMode === 'floor');

    if (this.opset > 10) {
      this.roiInputIdx = 1;
      this.scalesInputIdx = 2;
      this.sizesInputIdx = 3;
    } else if (this.opset === 9) {
      this.scalesInputIdx = 1;
    }

    if (this.scalesInputIdx > 0) {
      const scale = graph.getValues()[node.inputs[this.scalesInputIdx]].tensor;

      if (scale && scale.dims.length > 0) {
        this.scales = Array.from(scale.floatData);
        scalesValidataion(this.scales, this.mode, this.isResize);
      }
    }

    // roi is only needed when coordinate transformation mode is tf_crop_and_resize
    // for all other modes no need to read roi input
    if (this.roiInputIdx > 0 && this.needRoiInput) {
      const roi = graph.getValues()[node.inputs[this.roiInputIdx]].tensor;
      if (roi) {
        this.roi = Array.from(roi.floatData);
      }
    }
    this.useExtrapolation = this.needRoiInput = (this.coordinateTransformMode === 'tf_crop_and_resize');

    this.nearestMode =
        (this.mode === 'nearest' && this.opset >= 11) ? attributes.getString('nearest_mode', 'round_prefer_floor') : '';
    if (['round_prefer_floor', 'round_prefer_ceil', 'floor', 'ceil', ''].indexOf(this.nearestMode) === -1) {
      throw new Error(`nearest_mode '${this.nearestMode}' is not supported`);
    }

    this.cubicCoefficientA = attributes.getFloat('cubic_coeff_a', -0.75);
    this.excludeOutside = attributes.getInt('exclude_outside', 0) !== 0;
    if (this.excludeOutside && this.mode !== 'cubic') {
      throw new Error('exclude_outside can be set to 1 only when mode is CUBIC.');
    }

    this.useNearest2xOptimization = (this.opset < 11) ?
        true :
        (this.mode === 'nearest' && this.coordinateTransformMode === 'asymmetric' && this.nearestMode === 'floor');

    this.getOriginalCoordinate = getOriginalCoordinateFromResizedCoordinate(this.coordinateTransformMode);
    this.getNearestPixel = getNearestPixelFromOriginal(this.nearestMode);
  }

  checkInputs(inputs: Tensor[]): boolean {
    if (!inputs || (this.opset < 9 && inputs.length !== 1) ||
        (this.opset >= 9 && this.opset < 11 && inputs.length !== 2) ||
        (this.opset >= 11 && inputs.length !== 3 && inputs.length !== 4)) {
      return false;
    }

    if (this.scales && inputs[0].dims.length !== this.scales.length) {
      return false;
    }

    return this.checkInputTypes(inputs);
  }

  protected checkInputTypes(inputs: Tensor[]): boolean {
    if (inputs[0].type === 'string') {
      return false;
    }

    return true;
  }

  protected prepare(inputs: Tensor[]): [number[], number[], ReadonlyArray<number>] {
    const x = inputs[0];
    const xDims = x.dims;

    // get roi data
    let roi = this.roi;
    if (!roi) {
      if (this.needRoiInput) {
        if (this.roiInputIdx <= 0) {
          throw new Error('Invalid roi input index.');
        }
        roi = parseRoiData(inputs[this.roiInputIdx]);
      } else {
        roi = new Array(xDims.length * 2).fill(0);
      }
    }

    let scales = this.scales;
    let outputSizes: number[]|undefined;
    if (!scales) {
      const scalesTensor = inputs[this.scalesInputIdx];
      if (scalesTensor && scalesTensor.size !== 0) {
        if (inputs[this.sizesInputIdx]) {
          throw new Error('Only one of scales or sizes must be provided as input.');
        }
        scales = parseScalesData(scalesTensor, this.mode, this.isResize);
      } else {
        const sizesTensor = inputs[this.sizesInputIdx];
        if (!sizesTensor || sizesTensor.size === 0) {
          throw new Error('Either scales or sizes MUST be provided as input.');
        }

        outputSizes = Array.from(sizesTensor.integerData);
        scales = parseScalesDataFromOutputSize(outputSizes, xDims, this.mode, this.isResize);
      }
    } else {
      if (inputs[this.sizesInputIdx]) {
        throw new Error('Only one of scales or sizes must be provided as input.');
      }
    }

    const yDims = outputSizes || computeOutputShape(scales, xDims);

    return [roi, scales, yDims];
  }

  protected isResize: boolean;
  protected mode: string;
  protected scales: number[];
  protected extrapolationValue: number;
  protected coordinateTransformMode: string;
  protected useExtrapolation: boolean;
  protected needRoiInput: boolean;
  protected nearestMode: string;
  protected cubicCoefficientA: number;
  protected excludeOutside: boolean;
  protected useNearest2xOptimization: boolean;
  protected roiInputIdx: number;
  protected scalesInputIdx: number;
  protected sizesInputIdx: number;
  protected roi: number[];

  protected getOriginalCoordinate: Upsample.GetOriginalCoordinateFunc;
  protected getNearestPixel: Upsample.GetNearestPixelFunc;
}

function scalesValidataion(scales: number[], mode: string, isResize: boolean) {
  if (!isResize) {
    for (const scale of scales) {
      if (scale < 1) {
        throw new Error('Scale value should be greater than or equal to 1.');
      }
    }
  } else {
    for (const scale of scales) {
      if (scale <= 0) {
        throw new Error('Scale value should be greater than 0.');
      }
    }
  }
  if (mode === 'linear' || mode === 'cubic') {
    if (scales.length !== 2 && (scales.length !== 4 || scales[0] !== 1 || scales[1] !== 1)) {
      throw new Error(`'Linear' mode and 'Cubic' mode only support 2-D inputs ('Bilinear', 'Bicubic') or 4-D inputs\
with the corresponding outermost 2 scale values being 1 in the ${isResize ? 'Resize' : 'Upsample'} opeartor.`);
    }
  }
}

export function parseRoiData(roi: Tensor): number[] {
  return roi.size > 0 ? Array.from(roi.floatData) : [];
}

export function parseScalesData(scale: Tensor, mode: string, isResize: boolean): number[] {
  const scales = Array.from(scale.floatData);
  scalesValidataion(scales, mode, isResize);
  return scales;
}

export function parseScalesDataFromOutputSize(
    yDims: ReadonlyArray<number>, xDims: ReadonlyArray<number>, mode: string, isResize: boolean): number[] {
  const length = xDims.length;
  const scales = new Array<number>(length);

  for (let i = 0, end = length; i < end; i++) {
    if (xDims[i] === 0) {
      if (yDims[i] !== 0) {
        throw new Error('Input dim is zero but required output dim is non-zero.');
      }
      scales[i] = 1;
    } else {
      scales[i] = yDims[i] / xDims[i];
    }
  }
  scalesValidataion(scales, mode, isResize);
  return scales;
}

export function computeOutputShape(scales: ReadonlyArray<number>, inputDims: ReadonlyArray<number>): number[] {
  return inputDims.map((dim, i) => Math.floor(dim * scales[i]));
}

function getOriginalCoordinateFromResizedCoordinate(mode: string): Upsample.GetOriginalCoordinateFunc {
  switch (mode) {
    case 'asymmetric':
      return (xResized: number, xScale: number) => xResized / xScale;
    case 'pytorch_half_pixel':
      return (xResized: number, xScale: number, lengthResized: number) =>
                 lengthResized > 1 ? (xResized + 0.5) / xScale - 0.5 : 0.0;
    case 'tf_half_pixel_for_nn':
      return (xResized: number, xScale: number) => (xResized + 0.5) / xScale;
    case 'align_corners':
      return (xResized: number, xScale: number, lengthResized: number, lengthOriginal: number) =>
                 lengthResized === 1 ? 0 : xResized * (lengthOriginal - 1) / (lengthResized - 1);
    case 'tf_crop_and_resize':
      return (xResized: number, xScale: number, lengthResized: number, lengthOriginal: number, roiStart: number,
              roiEnd: number) =>
                 (lengthResized > 1 ? roiStart * (lengthOriginal - 1) +
                          (xResized * (roiEnd - roiStart) * (lengthOriginal - 1)) / (lengthResized - 1) :
                                      0.5 * (roiStart + roiEnd) * (lengthOriginal - 1));
    default:  //'half_pixel'
      return (xResized: number, xScale: number) => (xResized + 0.5) / xScale - 0.5;
  }
}

function getNearestPixelFromOriginal(mode: string): Upsample.GetNearestPixelFunc {
  switch (mode) {
    case '':
      return (xOriginal: number, isDownSample: boolean) => isDownSample ? Math.ceil(xOriginal) : Math.floor(xOriginal);
    case 'round_prefer_ceil':
      return (xOriginal: number) => Math.round(xOriginal);
    case 'floor':
      return (xOriginal: number) => Math.floor(xOriginal);
    case 'ceil':
      return (xOriginal: number) => Math.ceil(xOriginal);
    default:  // round_prefer_floor
      return (xOriginal: number) =>
                 xOriginal === Math.floor(xOriginal) + 0.5 ? Math.floor(xOriginal) : Math.round(xOriginal);
  }
}

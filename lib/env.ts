import * as platform from 'platform';

import * as onnx from './api';
import {Backend, Environment, Onnx} from './api';

interface ENV extends Environment {
  readonly onnx: Onnx;
  readonly backend: Backend;
  readonly platform: Platform;
}

class EnvironmentImpl implements ENV {
  public readonly onnx = onnx;
  public readonly backend = onnx.backend;
  public readonly platform = platform;

  public debug = false;
}

export const env: ENV = new EnvironmentImpl();

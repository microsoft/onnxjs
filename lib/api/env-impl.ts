// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {env} from '../env';

import {Environment} from './env';

class ENV implements Environment {
  public set debug(value: boolean) {
    env.debug = value;
  }
  public get debug(): boolean {
    return env.debug;
  }

  public set pack(value: boolean) {
    env.pack = value;
  }
  public get pack(): boolean {
    return env.pack;
  }
}

export const envImpl = new ENV();

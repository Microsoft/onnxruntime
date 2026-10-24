// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import type { InferenceSession } from 'onnxruntime-common';

import { getInstance } from './wasm-factory';
import { allocWasmString, checkLastError, iterateExtraOptions } from './wasm-utils';

const getGraphOptimzationLevel = (graphOptimizationLevel: string | unknown): number => {
  switch (graphOptimizationLevel) {
    case 'disabled':
      return 0;
    case 'basic':
      return 1;
    case 'extended':
      return 2;
    case 'layout':
      return 3;
    case 'all':
      return 99;
    default:
      throw new Error(`unsupported graph optimization level: ${graphOptimizationLevel}`);
  }
};

const getExecutionMode = (executionMode: 'sequential' | 'parallel'): number => {
  switch (executionMode) {
    case 'sequential':
      return 0;
    case 'parallel':
      return 1;
    default:
      throw new Error(`unsupported execution mode: ${executionMode}`);
  }
};

const appendDefaultOptions = (options: InferenceSession.SessionOptions): void => {
  if (!options.extra) {
    options.extra = {};
  }
  if (!options.extra.session) {
    options.extra.session = {};
  }
  const session = options.extra.session as Record<string, string>;
  if (!session.use_ort_model_bytes_directly) {
    // eslint-disable-next-line camelcase
    session.use_ort_model_bytes_directly = '1';
  }

  // if using JSEP with WebGPU, always disable memory pattern
  if (
    options.executionProviders &&
    options.executionProviders.some((ep) => (typeof ep === 'string' ? ep : ep.name) === 'webgpu')
  ) {
    options.enableMemPattern = false;
  }
};

const appendSessionConfig = (sessionOptionsHandle: number, key: string, value: string, allocs: number[]): void => {
  const keyDataOffset = allocWasmString(key, allocs);
  const valueDataOffset = allocWasmString(value, allocs);
  if (getInstance()._OrtAddSessionConfigEntry(sessionOptionsHandle, keyDataOffset, valueDataOffset) !== 0) {
    checkLastError(`Can't set a session config entry: ${key} - ${value}.`);
  }
};

const appendEpOption = (epOptions: Array<[number, number]>, key: string, value: string, allocs: number[]): void => {
  const keyDataOffset = allocWasmString(key, allocs);
  const valueDataOffset = allocWasmString(value, allocs);
  epOptions.push([keyDataOffset, valueDataOffset]);
};

// OrtErrorCode values are part of the public C API ABI.
const ORT_NOT_IMPLEMENTED = 9;

/**
 * Get the OrtEpDevice instances registered with the environment, grouped by execution provider name.
 *
 * An execution provider may register more than one device, e.g. one per GPU, and all of them share its name.
 *
 * @returns a map of execution provider name to the OrtEpDevice pointers registered under it, or undefined if the
 * build does not support the EP device API.
 */
const getEpDevicesByEpName = (): Map<string, number[]> | undefined => {
  const wasm = getInstance();
  const stack = wasm.stackSave();
  try {
    const epDevicesPtr = wasm.stackAlloc(wasm.PTR_SIZE);
    const numEpDevicesPtr = wasm.stackAlloc(wasm.PTR_SIZE);
    const errorCode = wasm._OrtGetEpDevices(epDevicesPtr, numEpDevicesPtr);
    if (errorCode === ORT_NOT_IMPLEMENTED) {
      return undefined;
    }
    if (errorCode !== 0) {
      checkLastError("Can't get execution provider devices.");
    }
    // The array is owned by the environment, so only the pointers are read out here.
    // getValue() with '*' returns a BigInt in a wasm64 build, so convert before doing pointer arithmetic.
    const epDevices = Number(wasm.getValue(epDevicesPtr, '*'));
    const numEpDevices = Number(wasm.getValue(numEpDevicesPtr, '*'));

    const epDevicesByEpName = new Map<string, number[]>();
    for (let i = 0; i < numEpDevices; i++) {
      const epDevice = Number(wasm.getValue(epDevices + i * wasm.PTR_SIZE, '*'));
      const epName = wasm.UTF8ToString(wasm._OrtEpDevice_EpName(epDevice));
      const devices = epDevicesByEpName.get(epName);
      if (devices) {
        devices.push(epDevice);
      } else {
        epDevicesByEpName.set(epName, [epDevice]);
      }
    }
    return epDevicesByEpName;
  } finally {
    wasm.stackRestore(stack);
  }
};

const setExecutionProviders = async (
  sessionOptionsHandle: number,
  sessionOptions: InferenceSession.SessionOptions,
  allocs: number[],
): Promise<void> => {
  const executionProviders = sessionOptions.executionProviders!;
  for (const ep of executionProviders) {
    let epName = typeof ep === 'string' ? ep : ep.name;
    const epOptions: Array<[number, number]> = [];
    let selectedEpDevices: number[] | undefined;

    // check EP name
    switch (epName) {
      case 'webnn':
        epName = 'WEBNN';
        // Disable QDQ fusion so DQ/Q nodes are preserved as individual ops for WebNN EP.
        appendSessionConfig(sessionOptionsHandle, 'session.disable_quant_qdq', '1', allocs);
        // Forcibly prevent constant folding from replacing DQ nodes with constants.
        appendSessionConfig(sessionOptionsHandle, 'session.disable_qdq_constant_folding', '1', allocs);
        if (typeof ep !== 'string') {
          const webnnOptions = ep as InferenceSession.WebNNExecutionProviderOption;
          // const context = (webnnOptions as InferenceSession.WebNNOptionsWithMLContext)?.context;
          const deviceType = (webnnOptions as InferenceSession.WebNNContextOptions)?.deviceType;
          if (deviceType) {
            appendSessionConfig(sessionOptionsHandle, 'deviceType', deviceType, allocs);
          }
        }
        break;
      case 'webgpu':
        if (!BUILD_DEFS.DISABLE_WEBGPU) {
          epName = 'WebGPU';
          const epDevicesByEpName = getEpDevicesByEpName();
          if (epDevicesByEpName !== undefined) {
            // Non-minimal builds select by OrtEpDevice under the EP's canonical name. Minimal builds do not support
            // the EP device API and continue to use the built-in EP name table.
            epName = 'WebGpuExecutionProvider';
            selectedEpDevices = epDevicesByEpName.get(epName);
            if (selectedEpDevices === undefined) {
              const registered = [...epDevicesByEpName.keys()].join(', ') || '(none)';
              throw new Error(`no execution provider device is registered for: ${epName}. registered: ${registered}.`);
            }
          }
          let customDevice: GPUDevice | undefined;

          if (typeof ep !== 'string') {
            const webgpuOptions = ep as InferenceSession.WebGpuExecutionProviderOption;

            // set custom GPU device
            if (webgpuOptions.device) {
              if (typeof GPUDevice !== 'undefined' && webgpuOptions.device instanceof GPUDevice) {
                customDevice = webgpuOptions.device;
              } else {
                throw new Error('Invalid GPU device set in WebGPU EP options.');
              }
            }

            // set graph capture option from session options
            const { enableGraphCapture } = sessionOptions;
            if (typeof enableGraphCapture === 'boolean' && enableGraphCapture) {
              appendEpOption(epOptions, 'enableGraphCapture', '1', allocs);
            }

            // set layout option
            if (typeof webgpuOptions.preferredLayout === 'string') {
              appendEpOption(epOptions, 'preferredLayout', webgpuOptions.preferredLayout, allocs);
            }

            // set force CPU fallback nodes
            if (webgpuOptions.forceCpuNodeNames) {
              const names = Array.isArray(webgpuOptions.forceCpuNodeNames)
                ? webgpuOptions.forceCpuNodeNames
                : [webgpuOptions.forceCpuNodeNames];

              appendEpOption(epOptions, 'forceCpuNodeNames', names.join('\n'), allocs);
            }

            // set validation mode
            if (webgpuOptions.validationMode) {
              appendEpOption(epOptions, 'validationMode', webgpuOptions.validationMode, allocs);
            }

            if (typeof webgpuOptions.disableSubgroupMatrix === 'boolean') {
              appendEpOption(epOptions, 'disableSubgroupMatrix', webgpuOptions.disableSubgroupMatrix ? '1' : '0', allocs);
            }

            // set f32 accumulation for the MatMulNBits kernels
            if (typeof webgpuOptions.enableMatmulFp32Accumulation === 'boolean') {
              appendEpOption(
                epOptions,
                'enableMatmulFp32Accumulation',
                webgpuOptions.enableMatmulFp32Accumulation ? '1' : '0',
                allocs,
              );
            }

            // set buffer cache modes
            for (const key of [
              'storageBufferCacheMode',
              'uniformBufferCacheMode',
              'queryResolveBufferCacheMode',
              'defaultBufferCacheMode',
            ] as const) {
              const mode = webgpuOptions[key];
              if (mode) {
                if (mode !== 'disabled' && mode !== 'lazyRelease' && mode !== 'simple' && mode !== 'bucket') {
                  throw new Error(`${key} must be one of 'disabled', 'lazyRelease', 'simple' or 'bucket': ${mode}`);
                }
                appendEpOption(epOptions, key, mode, allocs);
              }
            }
          }

          const info = getInstance().webgpuRegisterDevice!(customDevice);
          if (info) {
            const [deviceId, instanceHandle, deviceHandle] = info;
            appendEpOption(epOptions, 'deviceId', deviceId.toString(), allocs);
            appendEpOption(epOptions, 'webgpuInstance', instanceHandle.toString(), allocs);
            appendEpOption(epOptions, 'webgpuDevice', deviceHandle.toString(), allocs);
          }
        } else {
          epName = 'JS';
          if (typeof ep !== 'string') {
            const webgpuOptions = ep as InferenceSession.WebGpuExecutionProviderOption;
            if (webgpuOptions?.preferredLayout) {
              if (webgpuOptions.preferredLayout !== 'NCHW' && webgpuOptions.preferredLayout !== 'NHWC') {
                throw new Error(`preferredLayout must be either 'NCHW' or 'NHWC': ${webgpuOptions.preferredLayout}`);
              }
              appendSessionConfig(sessionOptionsHandle, 'preferredLayout', webgpuOptions.preferredLayout, allocs);
            }
          }
        }
        break;
      case 'wasm':
      case 'cpu':
        continue;
      default:
        throw new Error(`not supported execution provider: ${epName}`);
    }

    const epOptionsCount = epOptions.length;
    let keysOffset = 0;
    let valuesOffset = 0;
    if (epOptionsCount > 0) {
      keysOffset = getInstance()._malloc(epOptionsCount * getInstance().PTR_SIZE);
      allocs.push(keysOffset);
      valuesOffset = getInstance()._malloc(epOptionsCount * getInstance().PTR_SIZE);
      allocs.push(valuesOffset);
      for (let i = 0; i < epOptionsCount; i++) {
        getInstance().setValue(keysOffset + i * getInstance().PTR_SIZE, epOptions[i][0], '*');
        getInstance().setValue(valuesOffset + i * getInstance().PTR_SIZE, epOptions[i][1], '*');
      }
    }

    let appendErrorCode: number;
    if (selectedEpDevices !== undefined) {
      // Select a single OrtEpDevice, even though more than one may match the EP name.
      //
      // Necessary: the WebGPU EP backs a session with one Dawn device, so its factory rejects any device
      // count other than 1 (see WebGpuEpFactory::CreateIExecutionProvider for the built-in EP and
      // Factory::CreateEpImpl in core/providers/webgpu/ep/factory.cc for the plugin EP).
      //
      // Safe to take the first: the factory reads nothing from the device except whether it is virtual, and
      // Emscripten device discovery reports a single GPU entry derived from navigator.gpu, so there is at
      // most one real candidate. Which GPU is used is decided by the adapter ORT Web hands to Dawn, not by
      // this device (see env.webgpu.adapter and the WebGPU EP `device` option). A second entry only appears
      // when virtual devices are enabled, and the factory registers those after the real ones, so the first
      // is still a real device. Should that order ever change, EP creation fails with an explicit
      // "selected on a virtual GPU device" error rather than silently running on the wrong device.
      const epDevicesOffset = getInstance()._malloc(getInstance().PTR_SIZE);
      allocs.push(epDevicesOffset);
      getInstance().setValue(epDevicesOffset, selectedEpDevices[0], '*');

      appendErrorCode = await getInstance()._OrtAppendExecutionProviderV2(
        sessionOptionsHandle,
        epDevicesOffset,
        1,
        keysOffset,
        valuesOffset,
        epOptionsCount,
      );
    } else {
      const epNameDataOffset = allocWasmString(epName, allocs);
      appendErrorCode = await getInstance()._OrtAppendExecutionProvider(
        sessionOptionsHandle,
        epNameDataOffset,
        keysOffset,
        valuesOffset,
        epOptionsCount,
      );
    }

    if (appendErrorCode !== 0) {
      checkLastError(`Can't append execution provider: ${epName}.`);
    }
  }
};

export const setSessionOptions = async (options?: InferenceSession.SessionOptions): Promise<[number, number[]]> => {
  const wasm = getInstance();
  let sessionOptionsHandle = 0;
  const allocs: number[] = [];

  const sessionOptions: InferenceSession.SessionOptions = options || {};
  appendDefaultOptions(sessionOptions);

  try {
    const graphOptimizationLevel = getGraphOptimzationLevel(sessionOptions.graphOptimizationLevel ?? 'all');
    const executionMode = getExecutionMode(sessionOptions.executionMode ?? 'sequential');
    const logIdDataOffset =
      typeof sessionOptions.logId === 'string' ? allocWasmString(sessionOptions.logId, allocs) : 0;

    const logSeverityLevel = sessionOptions.logSeverityLevel ?? 2; // Default to 2 - warning
    if (!Number.isInteger(logSeverityLevel) || logSeverityLevel < 0 || logSeverityLevel > 4) {
      throw new Error(`log severity level is not valid: ${logSeverityLevel}`);
    }

    const logVerbosityLevel = sessionOptions.logVerbosityLevel ?? 0; // Default to 0 - verbose
    if (!Number.isInteger(logVerbosityLevel) || logVerbosityLevel < 0 || logVerbosityLevel > 4) {
      throw new Error(`log verbosity level is not valid: ${logVerbosityLevel}`);
    }

    const optimizedModelFilePathOffset =
      typeof sessionOptions.optimizedModelFilePath === 'string'
        ? allocWasmString(sessionOptions.optimizedModelFilePath, allocs)
        : 0;

    sessionOptionsHandle = wasm._OrtCreateSessionOptions(
      graphOptimizationLevel,
      !!sessionOptions.enableCpuMemArena,
      !!sessionOptions.enableMemPattern,
      executionMode,
      !!sessionOptions.enableProfiling,
      0,
      logIdDataOffset,
      logSeverityLevel,
      logVerbosityLevel,
      optimizedModelFilePathOffset,
    );
    if (sessionOptionsHandle === 0) {
      checkLastError("Can't create session options.");
    }

    if (sessionOptions.executionProviders) {
      await setExecutionProviders(sessionOptionsHandle, sessionOptions, allocs);
    }

    if (sessionOptions.enableGraphCapture !== undefined) {
      if (typeof sessionOptions.enableGraphCapture !== 'boolean') {
        throw new Error(`enableGraphCapture must be a boolean value: ${sessionOptions.enableGraphCapture}`);
      }
      appendSessionConfig(
        sessionOptionsHandle,
        'enableGraphCapture',
        sessionOptions.enableGraphCapture.toString(),
        allocs,
      );
    }

    if (sessionOptions.freeDimensionOverrides) {
      for (const [name, value] of Object.entries(sessionOptions.freeDimensionOverrides)) {
        if (typeof name !== 'string') {
          throw new Error(`free dimension override name must be a string: ${name}`);
        }
        if (typeof value !== 'number' || !Number.isInteger(value) || value < 0) {
          throw new Error(`free dimension override value must be a non-negative integer: ${value}`);
        }
        const nameOffset = allocWasmString(name, allocs);
        if (wasm._OrtAddFreeDimensionOverride(sessionOptionsHandle, nameOffset, value) !== 0) {
          checkLastError(`Can't set a free dimension override: ${name} - ${value}.`);
        }
      }
    }

    if (sessionOptions.extra !== undefined) {
      iterateExtraOptions(sessionOptions.extra, '', new WeakSet<Record<string, unknown>>(), (key, value) => {
        appendSessionConfig(sessionOptionsHandle, key, value, allocs);
      });
    }

    return [sessionOptionsHandle, allocs];
  } catch (e) {
    if (sessionOptionsHandle !== 0) {
      if (wasm._OrtReleaseSessionOptions(sessionOptionsHandle) !== 0) {
        checkLastError("Can't release session options.");
      }
    }
    allocs.forEach((alloc) => wasm._free(alloc));
    throw e;
  }
};

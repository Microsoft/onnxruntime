// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { InferenceSession as InferenceSessionImpl } from './inference-session-impl.js';
import type { LoraAdapter } from './lora-adapter.js';
import { OnnxModelOptions } from './onnx-model.js';
import { OnnxValue, OnnxValueDataLocation } from './onnx-value.js';
import type { Tensor } from './tensor.js';
import { TryGetGlobalType } from './type-helper.js';

/* eslint-disable @typescript-eslint/no-redeclare */

export declare namespace InferenceSession {
  // #region input/output types

  type OnnxValueMapType = { readonly [name: string]: OnnxValue };
  type NullableOnnxValueMapType = { readonly [name: string]: OnnxValue | null };

  /**
   * A feeds (model inputs) is an object that uses input names as keys and OnnxValue as corresponding values.
   */
  type FeedsType = OnnxValueMapType;

  /**
   * A fetches (model outputs) could be one of the following:
   *
   * - Omitted. Use model's output names definition.
   * - An array of string indicating the output names.
   * - An object that use output names as keys and OnnxValue or null as corresponding values.
   *
   * @remarks
   * different from input argument, in output, OnnxValue is optional. If an OnnxValue is present it will be
   * used as a pre-allocated value by the inference engine; if omitted, inference engine will allocate buffer
   * internally.
   */
  type FetchesType = readonly string[] | NullableOnnxValueMapType;

  /**
   * A inferencing return type is an object that uses output names as keys and OnnxValue as corresponding values.
   */
  type ReturnType = OnnxValueMapType;

  // #endregion

  // #region session options

  /**
   * A set of configurations for session behavior.
   */
  export interface SessionOptions extends OnnxModelOptions {
    /**
     * Configure loading external data referenced by EPContext nodes.
     *
     * The callback is invoked synchronously with the external data name and must return a `Uint8Array`. The returned
     * bytes are copied into ONNX Runtime-owned memory during the enclosing native operation. Do not modify or reuse the
     * returned view's backing storage until that operation completes; return a fresh array when using mutable shared
     * storage. Empty arrays are supported. Callback invocations may originate from multiple ONNX Runtime threads; the
     * native JavaScript bindings serialize calls before invoking JavaScript.
     *
     * `maxDataSize` is a required, finite, positive safe integer. Session creation fails if the callback throws, returns
     * another type, or returns more bytes than this limit. ONNX Runtime does not fall back to loading the data from disk
     * after a callback failure.
     *
     * The binding retains and enforces this limit in its callback state before allocating the native output buffer.
     * It is not configured in the native API and does not constrain allocations made inside the JavaScript callback.
     *
     * This setting is available only in the Node.js and React Native bindings. ONNX Runtime Web rejects it because a
     * JavaScript callback cannot currently be registered safely through the WebAssembly ABI.
     *
     * JavaScript does not currently expose model compilation, so the corresponding EPContext data write callback is not
     * available.
     */
    epContextDataRead?: {
      callback: (name: string) => Uint8Array;
      maxDataSize: number;
    };

    /**
     * An array of execution provider options.
     *
     * An execution provider option can be a string indicating the name of the execution provider,
     * or an object of corresponding type.
     */
    executionProviders?: readonly ExecutionProviderConfig[];

    /**
     * The intra OP threads number.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native).
     */
    intraOpNumThreads?: number;

    /**
     * The inter OP threads number.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native).
     */
    interOpNumThreads?: number;

    /**
     * The free dimension override.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    freeDimensionOverrides?: { readonly [dimensionName: string]: number };

    /**
     * The optimization level.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    graphOptimizationLevel?: 'disabled' | 'basic' | 'extended' | 'layout' | 'all';

    /**
     * Whether enable CPU memory arena.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    enableCpuMemArena?: boolean;

    /**
     * Whether enable memory pattern.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    enableMemPattern?: boolean;

    /**
     * Execution mode.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    executionMode?: 'sequential' | 'parallel';

    /**
     * Optimized model file path.
     *
     * If this setting is specified, the optimized model will be dumped. In browser, a blob will be created
     * with a pop-up window.
     */
    optimizedModelFilePath?: string;

    /**
     * Whether enable profiling.
     *
     * This setting is a placeholder for a future use.
     */
    enableProfiling?: boolean;

    /**
     * File prefix for profiling.
     *
     * This setting is a placeholder for a future use.
     */
    profileFilePrefix?: string;

    /**
     * Log ID.
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    logId?: string;

    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    logSeverityLevel?: 0 | 1 | 2 | 3 | 4;

    /**
     * Log verbosity level.
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     */
    logVerbosityLevel?: number;

    /**
     * Specify string as a preferred data location for all outputs, or an object that use output names as keys and a
     * preferred data location as corresponding values.
     *
     * This setting is available only in ONNXRuntime Web for WebGL and WebGPU EP.
     */
    preferredOutputLocation?: OnnxValueDataLocation | { readonly [outputName: string]: OnnxValueDataLocation };

    /**
     * Whether enable graph capture.
     * This setting is available only in ONNXRuntime Web for WebGPU EP.
     */
    enableGraphCapture?: boolean;

    /**
     * Store configurations for a session. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/
     * onnxruntime_session_options_config_keys.h
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     *
     * @example
     * ```js
     * extra: {
     *   session: {
     *     set_denormal_as_zero: "1",
     *     disable_prepacking: "1"
     *   },
     *   optimization: {
     *     enable_gelu_approximation: "1"
     *   }
     * }
     * ```
     */
    extra?: Record<string, unknown>;
  }

  // #region execution providers

  // Currently, we have the following backends to support execution providers:
  // Backend Node.js binding: supports 'cpu', 'dml' (win32), 'coreml' (macOS) and 'cuda' (linux).
  // Backend WebAssembly: supports 'cpu', 'wasm', 'webgpu' and 'webnn'.
  // Backend ONNX.js: supports 'webgl'.
  // Backend React Native: supports 'cpu', 'xnnpack', 'coreml' (iOS), 'nnapi' (Android).
  interface ExecutionProviderOptionMap {
    coreml: CoreMLExecutionProviderOption;
    cpu: CpuExecutionProviderOption;
    cuda: CudaExecutionProviderOption;
    dml: DmlExecutionProviderOption;
    nnapi: NnapiExecutionProviderOption;
    tensorrt: TensorRtExecutionProviderOption;
    wasm: WebAssemblyExecutionProviderOption;
    webgl: WebGLExecutionProviderOption;
    webgpu: WebGpuExecutionProviderOption;
    webnn: WebNNExecutionProviderOption;
    qnn: QnnExecutionProviderOption;
    xnnpack: XnnpackExecutionProviderOption;
  }

  type ExecutionProviderName = keyof ExecutionProviderOptionMap;
  type ExecutionProviderConfig =
    | ExecutionProviderOptionMap[ExecutionProviderName]
    | ExecutionProviderOption
    | ExecutionProviderName
    | string;

  export interface ExecutionProviderOption {
    readonly name: string;
  }
  export interface CpuExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'cpu';
    useArena?: boolean;
  }
  export interface CudaExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'cuda';
    deviceId?: number;
  }
  export interface DmlExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'dml';
    deviceId?: number;
  }
  export interface TensorRtExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'tensorrt';
    deviceId?: number;
  }
  export interface WebAssemblyExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'wasm';
  }
  export interface WebGLExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'webgl';
    // TODO: add flags
  }
  export interface XnnpackExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'xnnpack';
  }
  export interface WebGpuExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'webgpu';

    /**
     * Specify the preferred layout when running layout sensitive operators.
     *
     * @default 'NCHW'
     */
    preferredLayout?: 'NCHW' | 'NHWC';

    /**
     * Specify a list of node names that should be executed on CPU even when WebGPU EP is used.
     */
    forceCpuNodeNames?: readonly string[];

    /**
     * Specify the validation mode for WebGPU execution provider.
     * - 'disabled': Disable all validation.
     * When used in Node.js, disable validation may cause process crash if WebGPU errors occur. Be cautious when using
     * this mode.
     * When used in web, this mode is equivalent to 'wgpuOnly'.
     * - 'wgpuOnly': Perform WebGPU internal validation only.
     * - 'basic': Perform basic validation including WebGPU internal validation. This is the default mode.
     * - 'full': Perform full validation. This mode may have performance impact. Use it for debugging purpose.
     *
     * @default 'basic'
     */
    validationMode?: 'disabled' | 'wgpuOnly' | 'basic' | 'full';

    /**
     * Disable subgroup matrix kernels and use their existing fallbacks, including with a custom GPU device.
     * This is a global, first-context-wins option. Later
     * conflicting values on an existing context are ignored with a warning.
     *
     * @default false
     */
    disableSubgroupMatrix?: boolean;

    /**
     * Enable robust buffer access for an ORT-created Dawn device. This is a global, first-device-wins option. Later
     * conflicting values and values supplied with an external device are ignored with a warning.
     *
     * This setting is available only in ONNX Runtime (Node.js binding).
     *
     * @default `true` in Debug builds; `false` in Release and RelWithDebInfo builds
     */
    enableRobustness?: boolean;

    /**
     * Select accelerated external-weight loading for the native Windows WebGPU execution provider.
     * - 'off': Disable accelerated weight loading.
     * - 'preferred': Use D3D12 accelerated loading when available and otherwise use the ordinary loading path.
     * - 'required': Require D3D12 accelerated loading support.
     *
     * This setting is available only in ONNX Runtime (Node.js binding) builds with D3D12 file loading support.
     *
     * @default 'off'
     */
    weightLoadAcceleration?: 'off' | 'preferred' | 'required';

    /**
     * Specify the cache mode for storage buffers.
     * - 'disabled': Disable buffer cache. Buffers are destroyed when no longer in use.
     * - 'lazyRelease': Buffers are released lazily, at the end of the current run.
     * - 'simple': Released buffers are cached and reused only for requests of the exact same size.
     * - 'bucket': Released buffers are cached and reused using predefined size buckets. This is the default mode.
     *
     * For static-shape models, 'simple' may reduce GPU memory usage, because exact-size buffers are reused across
     * runs instead of allocating new bucket-sized buffers.
     *
     * @default 'bucket'
     */
    storageBufferCacheMode?: 'disabled' | 'lazyRelease' | 'simple' | 'bucket';

    /**
     * Specify the cache mode for uniform buffers.
     *
     * See {@link storageBufferCacheMode} for a description of the available modes.
     *
     * @default 'simple'
     */
    uniformBufferCacheMode?: 'disabled' | 'lazyRelease' | 'simple' | 'bucket';

    /**
     * Specify the cache mode for query resolve buffers.
     *
     * See {@link storageBufferCacheMode} for a description of the available modes.
     *
     * @default 'disabled'
     */
    queryResolveBufferCacheMode?: 'disabled' | 'lazyRelease' | 'simple' | 'bucket';

    /**
     * Specify the cache mode for buffers not covered by the other buffer cache mode options.
     *
     * See {@link storageBufferCacheMode} for a description of the available modes.
     *
     * @default 'disabled'
     */
    defaultBufferCacheMode?: 'disabled' | 'lazyRelease' | 'simple' | 'bucket';

    /**
     * Accumulate the dot products in f32 instead of in the output element type. The input and
     * weight tensors keep their own type, so global memory traffic is identical either way.
     *
     * When this is false the accumulator follows the output element type. Partial sums along K
     * can exceed the f16 maximum (65504) on backends that round strictly at every step, which
     * saturates the accumulator to Inf; setting this avoids that at the cost of registers and
     * workgroup memory.
     *
     * Where a fused kernel computes its epilogue on the accumulators, that epilogue carries the
     * wider type too. On the fused MLP decode fast path the bias, the SiLU and the gate/up product
     * are applied to the f32 accumulators and rounded once at the final store rather than after
     * every step, so with the option on its output can differ from the same graph run unfused by
     * more than the accumulation change alone. Fused MLP shapes that do not take that fast path
     * materialize the gate and up tensors in the output element type before the activation, and
     * are unaffected in their epilogue.
     *
     * This currently applies to MatMulNBits and its fused variants. Coverage of the unquantized
     * MatMul family is planned as follow-up work under the same option.
     *
     * This option is read by the native WebGPU execution provider only. Builds of onnxruntime-web
     * that use the JSEP WebGPU backend ignore it, and their MatMulNBits shaders keep accumulating
     * in the output element type.
     *
     * @default false
     */
    enableMatmulFp32Accumulation?: boolean;

    /**
     * Specify an optional WebGPU device to be used by the WebGPU execution provider.
     */
    device?: TryGetGlobalType<'GPUDevice'>;
  }

  // #region WebNN options

  interface WebNNExecutionProviderName extends ExecutionProviderOption {
    readonly name: 'webnn';
  }

  /**
   * Represents a set of options for creating a WebNN MLContext.
   *
   * @see https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions
   */
  export interface WebNNContextOptions {
    deviceType?: 'cpu' | 'gpu' | 'npu';
    numThreads?: number;
    powerPreference?: 'default' | 'low-power' | 'high-performance';
  }

  /**
   * Represents a set of options for WebNN execution provider without MLContext.
   */
  export interface WebNNOptionsWithoutMLContext extends WebNNExecutionProviderName, WebNNContextOptions {
    context?: never;
  }

  /**
   * Represents a set of options for WebNN execution provider with MLContext.
   *
   * When MLContext is provided, the deviceType is also required so that the WebNN EP can determine the preferred
   * channel layout.
   *
   * @see https://www.w3.org/TR/webnn/#dom-ml-createcontext
   */
  export interface WebNNOptionsWithMLContext
    extends
      WebNNExecutionProviderName,
      Omit<WebNNContextOptions, 'deviceType'>,
      Required<Pick<WebNNContextOptions, 'deviceType'>> {
    context: TryGetGlobalType<'MLContext'>;
  }

  /**
   * Represents a set of options for WebNN execution provider with MLContext which is created from GPUDevice.
   *
   * @see https://www.w3.org/TR/webnn/#dom-ml-createcontext-gpudevice
   */
  export interface WebNNOptionsWebGpu extends WebNNExecutionProviderName {
    context: TryGetGlobalType<'MLContext'>;
    gpuDevice: TryGetGlobalType<'GPUDevice'>;
  }

  /**
   * Options for WebNN execution provider.
   */
  export type WebNNExecutionProviderOption =
    | WebNNOptionsWithoutMLContext
    | WebNNOptionsWithMLContext
    | WebNNOptionsWebGpu;

  // #endregion

  export interface QnnExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'qnn';
    /**
     * Specify the QNN backend type. E.g., 'cpu' or 'htp'.
     * Mutually exclusive with `backendPath`.
     *
     * @default 'htp'
     */
    backendType?: string;
    /**
     * Specify a path to the QNN backend library.
     * Mutually exclusive with `backendType`.
     */
    backendPath?: string;
    /**
     * Specify whether to enable HTP FP16 precision.
     *
     * @default true
     */
    enableFp16Precision?: boolean;
  }
  export interface CoreMLExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'coreml';
    /**
     * The bit flags for CoreML execution provider.
     *
     * ```
     * COREML_FLAG_USE_CPU_ONLY = 0x001
     * COREML_FLAG_ENABLE_ON_SUBGRAPH = 0x002
     * COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004
     * COREML_FLAG_ONLY_ALLOW_STATIC_INPUT_SHAPES = 0x008
     * COREML_FLAG_CREATE_MLPROGRAM = 0x010
     * COREML_FLAG_USE_CPU_AND_GPU = 0x020
     * ```
     *
     * See include/onnxruntime/core/providers/coreml/coreml_provider_factory.h for more details.
     *
     * This flag is available only in ONNXRuntime (Node.js binding).
     */
    coreMlFlags?: number;
    /**
     * Specify whether to use CPU only in CoreML EP.
     *
     * This setting is available only in ONNXRuntime (react-native).
     */
    useCPUOnly?: boolean;
    useCPUAndGPU?: boolean;
    /**
     * Specify whether to enable CoreML EP on subgraph.
     *
     * This setting is available only in ONNXRuntime (react-native).
     */
    enableOnSubgraph?: boolean;
    /**
     * Specify whether to only enable CoreML EP for Apple devices with ANE (Apple Neural Engine).
     *
     * This setting is available only in ONNXRuntime (react-native).
     */
    onlyEnableDeviceWithANE?: boolean;
  }
  export interface NnapiExecutionProviderOption extends ExecutionProviderOption {
    readonly name: 'nnapi';
    useFP16?: boolean;
    useNCHW?: boolean;
    cpuDisabled?: boolean;
    cpuOnly?: boolean;
  }
  // #endregion

  // #endregion

  // #region run options

  /**
   * A set of configurations for inference run behavior
   */
  export interface RunOptions {
    /**
     * Log severity level. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/logging/severity.h
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    logSeverityLevel?: 0 | 1 | 2 | 3 | 4;

    /**
     * Log verbosity level.
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     */
    logVerbosityLevel?: number;

    /**
     * Terminate all incomplete OrtRun calls as soon as possible if true
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     */
    terminate?: boolean;

    /**
     * A tag for the Run() calls using this
     *
     * This setting is available only in ONNXRuntime (Node.js binding and react-native) or WebAssembly backend
     */
    tag?: string;

    /**
     * Set a single run configuration entry. See
     * https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/session/
     * onnxruntime_run_options_config_keys.h
     *
     * This setting is available only in WebAssembly backend. Will support Node.js binding and react-native later
     *
     * @example
     *
     * ```js
     * extra: {
     *   memory: {
     *     enable_memory_arena_shrinkage: "1",
     *   }
     * }
     * ```
     */
    extra?: Record<string, unknown>;

    /**
     * A list of LoRA adapters to activate for this run. See `LoraAdapter`.
     *
     * The adapters must be created by the same backend as the session. Parameters of different adapters that are
     * active at the same time must not overlap.
     *
     * This setting is available only in WebAssembly backend. It is not supported yet in proxy mode
     * (`env.wasm.proxy`), or for a session that uses IO binding. IO binding is used when an output is preferred to be
     * on GPU (see `SessionOptions.preferredOutputLocation`), when graph capture is enabled, and in some cases with the
     * WebNN execution provider.
     */
    activeLoraAdapters?: readonly LoraAdapter[];
  }

  // #endregion

  // #region value metadata

  /**
   * The common part of the value metadata type for both tensor and non-tensor values.
   */
  export interface ValueMetadataBase {
    /**
     * The name of the specified input or output.
     */
    readonly name: string;
  }

  /**
   * Represents the metadata of a non-tensor value.
   */
  export interface NonTensorValueMetadata extends ValueMetadataBase {
    /**
     * Get a value indicating whether the value is a tensor.
     */
    readonly isTensor: false;
  }

  /**
   * Represents the metadata of a tensor value.
   */
  export interface TensorValueMetadata extends ValueMetadataBase {
    /**
     * Get a value indicating whether the value is a tensor.
     */
    readonly isTensor: true;
    /**
     * Get the data type of the tensor.
     */
    readonly type: Tensor.Type;
    /**
     * Get the shape of the tensor.
     *
     * If the shape is not defined, the value will an empty array. Otherwise, it will be an array representing the shape
     * of the tensor. Each element in the array can be a number or a string. If the element is a number, it represents
     * the corresponding dimension size. If the element is a string, it represents a symbolic dimension.
     */
    readonly shape: ReadonlyArray<number | string>;
  }

  /**
   * Represents the metadata of a value.
   */
  export type ValueMetadata = NonTensorValueMetadata | TensorValueMetadata;

  // #endregion
}

/**
 * Represent a runtime instance of an ONNX model.
 */
export interface InferenceSession {
  // #region run()

  /**
   * Execute the model asynchronously with the given feeds and options.
   *
   * @param feeds - Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param options - Optional. A set of options that controls the behavior of model inference.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  run(feeds: InferenceSession.FeedsType, options?: InferenceSession.RunOptions): Promise<InferenceSession.ReturnType>;

  /**
   * Execute the model asynchronously with the given feeds, fetches and options.
   *
   * @param feeds - Representation of the model input. See type description of `InferenceSession.InputType` for detail.
   * @param fetches - Representation of the model output. See type description of `InferenceSession.OutputType` for
   * detail.
   * @param options - Optional. A set of options that controls the behavior of model inference.
   * @returns A promise that resolves to a map, which uses output names as keys and OnnxValue as corresponding values.
   */
  run(
    feeds: InferenceSession.FeedsType,
    fetches: InferenceSession.FetchesType,
    options?: InferenceSession.RunOptions,
  ): Promise<InferenceSession.ReturnType>;

  // #endregion

  // #region release()

  /**
   * Release the inference session and the underlying resources.
   */
  release(): Promise<void>;

  // #endregion

  // #region profiling

  /**
   * Start profiling.
   */
  startProfiling(): void;

  /**
   * End profiling.
   */
  endProfiling(): void;

  // #endregion

  // #region metadata

  /**
   * Get input names of the loaded model.
   */
  readonly inputNames: readonly string[];

  /**
   * Get output names of the loaded model.
   */
  readonly outputNames: readonly string[];

  /**
   * Get input metadata of the loaded model.
   */
  readonly inputMetadata: readonly InferenceSession.ValueMetadata[];

  /**
   * Get output metadata of the loaded model.
   */
  readonly outputMetadata: readonly InferenceSession.ValueMetadata[];

  // #endregion
}

export interface InferenceSessionFactory {
  // #region create()

  /**
   * Create a new inference session and load model asynchronously from an ONNX model file.
   *
   * @param uri - The URI or file path of the model to load.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(uri: string, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from an array bufer.
   *
   * @param buffer - An ArrayBuffer representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: ArrayBufferLike, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from segment of an array bufer.
   *
   * @param buffer - An ArrayBuffer representation of an ONNX model.
   * @param byteOffset - The beginning of the specified portion of the array buffer.
   * @param byteLength - The length in bytes of the array buffer.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(
    buffer: ArrayBufferLike,
    byteOffset: number,
    byteLength?: number,
    options?: InferenceSession.SessionOptions,
  ): Promise<InferenceSession>;

  /**
   * Create a new inference session and load model asynchronously from a Uint8Array.
   *
   * @param buffer - A Uint8Array representation of an ONNX model.
   * @param options - specify configuration for creating a new inference session.
   * @returns A promise that resolves to an InferenceSession object.
   */
  create(buffer: Uint8Array, options?: InferenceSession.SessionOptions): Promise<InferenceSession>;

  // #endregion
}

// eslint-disable-next-line @typescript-eslint/naming-convention
export const InferenceSession: InferenceSessionFactory = InferenceSessionImpl;

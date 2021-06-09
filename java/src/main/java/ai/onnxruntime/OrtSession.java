/*
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.providers.CoreMLFlags;
import ai.onnxruntime.providers.NNAPIFlags;
import ai.onnxruntime.providers.OrtFlags;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Wraps an ONNX model and allows inference calls.
 *
 * <p>Allows the inspection of the model's input and output nodes. Produced by an {@link
 * OrtEnvironment}.
 *
 * <p>Most instance methods throw {@link IllegalStateException} if the session is closed and the
 * methods are called.
 */
public class OrtSession implements AutoCloseable {

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  private final long nativeHandle;

  private final OrtAllocator allocator;

  private final long numInputs;

  private final Set<String> inputNames;

  private final long numOutputs;

  private final Set<String> outputNames;

  private OnnxModelMetadata metadata;

  private boolean closed = false;

  /**
   * Create a session loading the model from disk.
   *
   * @param env The environment.
   * @param modelPath The path to the model.
   * @param allocator The allocator to use.
   * @param options Session configuration options.
   * @throws OrtException If the file could not be read, or the model was corrupted etc.
   */
  OrtSession(OrtEnvironment env, String modelPath, OrtAllocator allocator, SessionOptions options)
      throws OrtException {
    this(
        createSession(OnnxRuntime.ortApiHandle, env.nativeHandle, modelPath, options.nativeHandle),
        allocator);
  }

  /**
   * Creates a session reading the model from the supplied byte array.
   *
   * @param env The environment.
   * @param modelArray The model protobuf as a byte array.
   * @param allocator The allocator to use.
   * @param options Session configuration options.
   * @throws OrtException If the model was corrupted or some other error occurred in native code.
   */
  OrtSession(OrtEnvironment env, byte[] modelArray, OrtAllocator allocator, SessionOptions options)
      throws OrtException {
    this(
        createSession(OnnxRuntime.ortApiHandle, env.nativeHandle, modelArray, options.nativeHandle),
        allocator);
  }

  /**
   * Private constructor to build the Java object wrapped around a native session.
   *
   * @param nativeHandle The pointer to the native session.
   * @param allocator The allocator to use.
   * @throws OrtException If the model's inputs, outputs or metadata could not be read.
   */
  private OrtSession(long nativeHandle, OrtAllocator allocator) throws OrtException {
    this.nativeHandle = nativeHandle;
    this.allocator = allocator;
    numInputs = getNumInputs(OnnxRuntime.ortApiHandle, nativeHandle);
    inputNames =
        new LinkedHashSet<>(
            Arrays.asList(getInputNames(OnnxRuntime.ortApiHandle, nativeHandle, allocator.handle)));
    numOutputs = getNumOutputs(OnnxRuntime.ortApiHandle, nativeHandle);
    outputNames =
        new LinkedHashSet<>(
            Arrays.asList(
                getOutputNames(OnnxRuntime.ortApiHandle, nativeHandle, allocator.handle)));
  }

  /**
   * Returns the number of inputs this model expects.
   *
   * @return The number of inputs.
   */
  public long getNumInputs() {
    if (!closed) {
      return numInputs;
    } else {
      throw new IllegalStateException("Asking for inputs from a closed OrtSession.");
    }
  }

  /**
   * Returns the number of outputs this model expects.
   *
   * @return The number of outputs.
   */
  public long getNumOutputs() {
    if (!closed) {
      return numOutputs;
    } else {
      throw new IllegalStateException("Asking for outputs from a closed OrtSession.");
    }
  }

  /**
   * Returns the input names. The underlying collection is sorted based on the input id number.
   *
   * @return The input names.
   */
  public Set<String> getInputNames() {
    if (!closed) {
      return inputNames;
    } else {
      throw new IllegalStateException("Asking for inputs from a closed OrtSession.");
    }
  }

  /**
   * Returns the output names. The underlying collection is sorted based on the output id number.
   *
   * @return The output names.
   */
  public Set<String> getOutputNames() {
    if (!closed) {
      return outputNames;
    } else {
      throw new IllegalStateException("Asking for outputs from a closed OrtSession.");
    }
  }

  /**
   * Returns the info objects for the inputs, including their names and types. The underlying
   * collection is sorted based on the input id number.
   *
   * @return The input information.
   * @throws OrtException If there was an error in native code.
   */
  public Map<String, NodeInfo> getInputInfo() throws OrtException {
    if (!closed) {
      return wrapInMap(getInputInfo(OnnxRuntime.ortApiHandle, nativeHandle, allocator.handle));
    } else {
      throw new IllegalStateException("Asking for inputs from a closed OrtSession.");
    }
  }

  /**
   * Returns the info objects for the outputs, including their names and types. The underlying
   * collection is sorted based on the output id number.
   *
   * @return The output information.
   * @throws OrtException If there was an error in native code.
   */
  public Map<String, NodeInfo> getOutputInfo() throws OrtException {
    if (!closed) {
      return wrapInMap(getOutputInfo(OnnxRuntime.ortApiHandle, nativeHandle, allocator.handle));
    } else {
      throw new IllegalStateException("Asking for outputs from a closed OrtSession.");
    }
  }

  /**
   * Scores an input feed dict, returning the map of all inferred outputs.
   *
   * <p>The outputs are sorted based on their id number.
   *
   * @param inputs The inputs to score.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input names are invalid, or if
   *     there are zero or too many inputs.
   */
  public Result run(Map<String, OnnxTensor> inputs) throws OrtException {
    return run(inputs, outputNames);
  }

  /**
   * Scores an input feed dict, returning the map of all inferred outputs.
   *
   * <p>The outputs are sorted based on their id number.
   *
   * @param inputs The inputs to score.
   * @param runOptions The RunOptions to control this run.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input names are invalid, or if
   *     there are zero or too many inputs.
   */
  public Result run(Map<String, OnnxTensor> inputs, RunOptions runOptions) throws OrtException {
    return run(inputs, outputNames, runOptions);
  }

  /**
   * Scores an input feed dict, returning the map of requested inferred outputs.
   *
   * <p>The outputs are sorted based on the supplied set traversal order.
   *
   * @param inputs The inputs to score.
   * @param requestedOutputs The requested outputs.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input or output names are
   *     invalid, or if there are zero or too many inputs or outputs.
   */
  public Result run(Map<String, OnnxTensor> inputs, Set<String> requestedOutputs)
      throws OrtException {
    return run(inputs, requestedOutputs, null);
  }

  /**
   * Scores an input feed dict, returning the map of requested inferred outputs.
   *
   * <p>The outputs are sorted based on the supplied set traveral order.
   *
   * @param inputs The inputs to score.
   * @param requestedOutputs The requested outputs.
   * @param runOptions The RunOptions to control this run.
   * @return The inferred outputs.
   * @throws OrtException If there was an error in native code, the input or output names are
   *     invalid, or if there are zero or too many inputs or outputs.
   */
  public Result run(
      Map<String, OnnxTensor> inputs, Set<String> requestedOutputs, RunOptions runOptions)
      throws OrtException {
    if (!closed) {
      if (inputs.isEmpty() || (inputs.size() > numInputs)) {
        throw new OrtException(
            "Unexpected number of inputs, expected [1," + numInputs + ") found " + inputs.size());
      }
      if (requestedOutputs.isEmpty() || (requestedOutputs.size() > numOutputs)) {
        throw new OrtException(
            "Unexpected number of requestedOutputs, expected [1,"
                + numOutputs
                + ") found "
                + requestedOutputs.size());
      }
      String[] inputNamesArray = new String[inputs.size()];
      long[] inputHandles = new long[inputs.size()];
      int i = 0;
      for (Map.Entry<String, OnnxTensor> t : inputs.entrySet()) {
        if (inputNames.contains(t.getKey())) {
          inputNamesArray[i] = t.getKey();
          inputHandles[i] = t.getValue().getNativeHandle();
          i++;
        } else {
          throw new OrtException(
              "Unknown input name " + t.getKey() + ", expected one of " + inputNames.toString());
        }
      }
      String[] outputNamesArray = new String[requestedOutputs.size()];
      i = 0;
      for (String s : requestedOutputs) {
        if (outputNames.contains(s)) {
          outputNamesArray[i] = s;
          i++;
        } else {
          throw new OrtException(
              "Unknown output name " + s + ", expected one of " + outputNames.toString());
        }
      }
      long runOptionsHandle = runOptions == null ? 0 : runOptions.nativeHandle;

      OnnxValue[] outputValues =
          run(
              OnnxRuntime.ortApiHandle,
              nativeHandle,
              allocator.handle,
              inputNamesArray,
              inputHandles,
              inputNamesArray.length,
              outputNamesArray,
              outputNamesArray.length,
              runOptionsHandle);
      return new Result(outputNamesArray, outputValues);
    } else {
      throw new IllegalStateException("Trying to score a closed OrtSession.");
    }
  }

  /**
   * Gets the metadata for the currently loaded model.
   *
   * @return The metadata.
   * @throws OrtException If the native call failed.
   */
  public OnnxModelMetadata getMetadata() throws OrtException {
    if (metadata == null) {
      metadata = constructMetadata(OnnxRuntime.ortApiHandle, nativeHandle, allocator.handle);
    }
    return metadata;
  }

  /**
   * Returns the timestamp that profiling started in nanoseconds.
   *
   * @return the profiling start time in ns.
   * @throws OrtException If the native call failed.
   */
  public long getProfilingStartTimeInNs() throws OrtException {
    return getProfilingStartTimeInNs(OnnxRuntime.ortApiHandle, nativeHandle);
  }

  /**
   * Ends the profiling session and returns the output of the profiler.
   *
   * <p>Profiling should be enabled in the {@link SessionOptions} used to construct this {@code
   * Session}.
   *
   * @return The profiling output.
   * @throws OrtException If the native call failed.
   */
  public String endProfiling() throws OrtException {
    return endProfiling(OnnxRuntime.ortApiHandle, nativeHandle, allocator.handle);
  }

  @Override
  public String toString() {
    return "OrtSession(numInputs=" + numInputs + ",numOutputs=" + numOutputs + ")";
  }

  /**
   * Closes the session, releasing it's resources.
   *
   * @throws OrtException If it failed to close.
   */
  @Override
  public void close() throws OrtException {
    if (!closed) {
      closeSession(OnnxRuntime.ortApiHandle, nativeHandle);
      closed = true;
    } else {
      throw new IllegalStateException("Trying to close an already closed OrtSession.");
    }
  }

  /**
   * Converts a NodeInfo array into a map from node name to node info.
   *
   * @param infos The NodeInfo array to convert.
   * @return A Map from String to NodeInfo.
   */
  private static Map<String, NodeInfo> wrapInMap(NodeInfo[] infos) {
    Map<String, NodeInfo> output = new LinkedHashMap<>();

    for (NodeInfo info : infos) {
      output.put(info.getName(), info);
    }

    return output;
  }

  private static native long createSession(
      long apiHandle, long envHandle, String modelPath, long optsHandle) throws OrtException;

  private static native long createSession(
      long apiHandle, long envHandle, byte[] modelArray, long optsHandle) throws OrtException;

  private native long getNumInputs(long apiHandle, long nativeHandle) throws OrtException;

  private native String[] getInputNames(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native NodeInfo[] getInputInfo(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native long getNumOutputs(long apiHandle, long nativeHandle) throws OrtException;

  private native String[] getOutputNames(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native NodeInfo[] getOutputInfo(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  /**
   * The native run call. runOptionsHandle can be zero (i.e. the null pointer), but all other
   * handles must be valid pointers.
   *
   * @param apiHandle The pointer to the api.
   * @param nativeHandle The pointer to the session.
   * @param allocatorHandle The pointer to the allocator.
   * @param inputNamesArray The input names.
   * @param inputs The input tensors.
   * @param numInputs The number of inputs.
   * @param outputNamesArray The requested output names.
   * @param numOutputs The number of requested outputs.
   * @param runOptionsHandle The (possibly null) pointer to the run options.
   * @return The OnnxValues produced by this run.
   * @throws OrtException If the native call failed in some way.
   */
  private native OnnxValue[] run(
      long apiHandle,
      long nativeHandle,
      long allocatorHandle,
      String[] inputNamesArray,
      long[] inputs,
      long numInputs,
      String[] outputNamesArray,
      long numOutputs,
      long runOptionsHandle)
      throws OrtException;

  private native long getProfilingStartTimeInNs(long apiHandle, long nativeHandle)
      throws OrtException;

  private native String endProfiling(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native void closeSession(long apiHandle, long nativeHandle) throws OrtException;

  /**
   * Builds the {@link OnnxModelMetadata} for this session.
   *
   * @param ortApiHandle The api pointer.
   * @param nativeHandle The native session pointer.
   * @param allocatorHandle The OrtAllocator pointer.
   * @return The metadata.
   * @throws OrtException If the native runtime failed to access or allocate the metadata.
   */
  private native OnnxModelMetadata constructMetadata(
      long ortApiHandle, long nativeHandle, long allocatorHandle) throws OrtException;

  /**
   * Represents the options used to construct this session.
   *
   * <p>Used to set the number of threads, optimisation level, computation backend and other
   * options.
   *
   * <p>Modifying this after the session has been constructed will have no effect.
   *
   * <p>The SessionOptions object must not be closed until all sessions which use it are closed, as
   * otherwise it could release resources that are in use.
   */
  public static class SessionOptions implements AutoCloseable {

    /**
     * The optimisation level to use. Needs to be kept in sync with the GraphOptimizationLevel enum
     * in the C API.
     */
    public enum OptLevel {
      NO_OPT(0),
      BASIC_OPT(1),
      EXTENDED_OPT(2),
      ALL_OPT(99);

      private final int id;

      OptLevel(int id) {
        this.id = id;
      }

      /**
       * Gets the int id used in native code for this optimisation level.
       *
       * @return The int id.
       */
      public int getID() {
        return id;
      }
    }

    /**
     * The execution mode to use. Needs to be kept in sync with the ExecutionMode enum in the C API.
     */
    public enum ExecutionMode {
      SEQUENTIAL(0),
      PARALLEL(1);
      private final int id;

      ExecutionMode(int id) {
        this.id = id;
      }

      /**
       * Gets the int id used in native code for the execution mode.
       *
       * @return The int id.
       */
      public int getID() {
        return id;
      }
    }

    private final long nativeHandle;

    private final List<Long> customLibraryHandles;

    private final Map<String, String> configEntries;

    private boolean closed = false;

    /** Create an empty session options. */
    public SessionOptions() {
      nativeHandle = createOptions(OnnxRuntime.ortApiHandle);
      customLibraryHandles = new ArrayList<>();
      configEntries = new LinkedHashMap<String, String>();
    }

    /** Closes the session options, releasing any memory acquired. */
    @Override
    public void close() {
      if (!closed) {
        if (customLibraryHandles.size() > 0) {
          long[] longArray = new long[customLibraryHandles.size()];
          for (int i = 0; i < customLibraryHandles.size(); i++) {
            longArray[i] = customLibraryHandles.get(i);
          }
          closeCustomLibraries(longArray);
        }
        closeOptions(OnnxRuntime.ortApiHandle, nativeHandle);
        closed = true;
      } else {
        throw new IllegalStateException("Trying to close a closed SessionOptions.");
      }
    }

    /** Checks if the SessionOptions is closed, if so throws {@link IllegalStateException}. */
    private void checkClosed() {
      if (closed) {
        throw new IllegalStateException("Trying to use a closed SessionOptions");
      }
    }

    /**
     * Sets the execution mode of this options object, overriding the old setting.
     *
     * @param mode The execution mode to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setExecutionMode(ExecutionMode mode) throws OrtException {
      checkClosed();
      setExecutionMode(OnnxRuntime.ortApiHandle, nativeHandle, mode.getID());
    }

    /**
     * Sets the optimization level of this options object, overriding the old setting.
     *
     * @param level The optimization level to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setOptimizationLevel(OptLevel level) throws OrtException {
      checkClosed();
      setOptimizationLevel(OnnxRuntime.ortApiHandle, nativeHandle, level.getID());
    }

    /**
     * Sets the size of the CPU thread pool used for executing multiple request concurrently, if
     * executing on a CPU.
     *
     * @param numThreads The number of threads to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setInterOpNumThreads(int numThreads) throws OrtException {
      checkClosed();
      setInterOpNumThreads(OnnxRuntime.ortApiHandle, nativeHandle, numThreads);
    }

    /**
     * Sets the size of the CPU thread pool used for executing a single graph, if executing on a
     * CPU.
     *
     * @param numThreads The number of threads to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setIntraOpNumThreads(int numThreads) throws OrtException {
      checkClosed();
      setIntraOpNumThreads(OnnxRuntime.ortApiHandle, nativeHandle, numThreads);
    }

    /**
     * Sets the output path for the optimized model.
     *
     * @param outputPath The output path to write the model to.
     * @throws OrtException If there was an error in native code.
     */
    public void setOptimizedModelFilePath(String outputPath) throws OrtException {
      checkClosed();
      setOptimizationModelFilePath(OnnxRuntime.ortApiHandle, nativeHandle, outputPath);
    }

    /**
     * Sets the logger id to use.
     *
     * @param loggerId The logger id string.
     * @throws OrtException If there was an error in native code.
     */
    public void setLoggerId(String loggerId) throws OrtException {
      checkClosed();
      setLoggerId(OnnxRuntime.ortApiHandle, nativeHandle, loggerId);
    }

    /**
     * Enables profiling in sessions using this SessionOptions.
     *
     * @param filePath The file to write profile information to.
     * @throws OrtException If there was an error in native code.
     */
    public void enableProfiling(String filePath) throws OrtException {
      checkClosed();
      enableProfiling(OnnxRuntime.ortApiHandle, nativeHandle, filePath);
    }

    /**
     * Disables profiling in sessions using this SessionOptions.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void disableProfiling() throws OrtException {
      checkClosed();
      disableProfiling(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    /**
     * Turns on memory pattern optimizations, where memory is preallocated if all shapes are known.
     *
     * @param memoryPatternOptimization If true enable memory pattern optimizations.
     * @throws OrtException If there was an error in native code.
     */
    public void setMemoryPatternOptimization(boolean memoryPatternOptimization)
        throws OrtException {
      checkClosed();
      setMemoryPatternOptimization(
          OnnxRuntime.ortApiHandle, nativeHandle, memoryPatternOptimization);
    }

    /**
     * Sets the CPU to use an arena memory allocator.
     *
     * @param useArena If true use an arena memory allocator for the CPU execution provider.
     * @throws OrtException If there was an error in native code.
     */
    public void setCPUArenaAllocator(boolean useArena) throws OrtException {
      checkClosed();
      setCPUArenaAllocator(OnnxRuntime.ortApiHandle, nativeHandle, useArena);
    }

    /**
     * Sets the Session's logging level.
     *
     * @param logLevel The log level to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setSessionLogLevel(OrtLoggingLevel logLevel) throws OrtException {
      checkClosed();
      setSessionLogLevel(OnnxRuntime.ortApiHandle, nativeHandle, logLevel.getValue());
    }

    /**
     * Sets the Session's logging verbosity level.
     *
     * @param logLevel The logging verbosity to use.
     * @throws OrtException If there was an error in native code.
     */
    public void setSessionLogVerbosityLevel(int logLevel) throws OrtException {
      checkClosed();
      setSessionLogVerbosityLevel(OnnxRuntime.ortApiHandle, nativeHandle, logLevel);
    }

    /**
     * Registers a library of custom ops for use with {@link OrtSession}s using this SessionOptions.
     *
     * @param path The path to the library on disk.
     * @throws OrtException If there was an error loading the library.
     */
    public void registerCustomOpLibrary(String path) throws OrtException {
      checkClosed();
      long customHandle = registerCustomOpLibrary(OnnxRuntime.ortApiHandle, nativeHandle, path);
      customLibraryHandles.add(customHandle);
    }

    /**
     * Sets the value of a symbolic dimension. Fixed dimension computations may have more
     * optimizations applied to them.
     *
     * @param dimensionName The name of the symbolic dimension.
     * @param dimensionValue The value to set that dimension to.
     * @throws OrtException If there was an error in native code.
     */
    public void setSymbolicDimensionValue(String dimensionName, long dimensionValue)
        throws OrtException {
      checkClosed();
      addFreeDimensionOverrideByName(
          OnnxRuntime.ortApiHandle, nativeHandle, dimensionName, dimensionValue);
    }

    /**
     * Disables the per session thread pools. Must be used in conjunction with an environment
     * containing global thread pools.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void disablePerSessionThreads() throws OrtException {
      checkClosed();
      disablePerSessionThreads(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    /**
     * Adds a single session configuration entry as a pair of strings.
     *
     * @param configKey The config key string.
     * @param configValue The config value string.
     * @throws OrtException If there was an error in native code.
     */
    public void addConfigEntry(String configKey, String configValue) throws OrtException {
      checkClosed();
      addConfigEntry(OnnxRuntime.ortApiHandle, nativeHandle, configKey, configValue);
      configEntries.put(configKey, configValue);
    }

    /**
     * Returns an unmodifiable view of the map contains all session configuration entries.
     *
     * @return All session configuration entries
     */
    public Map<String, String> getConfigEntries() {
      checkClosed();
      return Collections.unmodifiableMap(configEntries);
    }

    /**
     * Add CUDA as an execution backend, using device 0.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void addCUDA() throws OrtException {
      addCUDA(0);
    }

    /**
     * Add CUDA as an execution backend, using the specified CUDA device id.
     *
     * @param deviceNum The CUDA device id.
     * @throws OrtException If there was an error in native code.
     */
    public void addCUDA(int deviceNum) throws OrtException {
      checkClosed();
      if (OnnxRuntime.extractCUDA()) {
        addCUDA(OnnxRuntime.ortApiHandle, nativeHandle, deviceNum);
      } else {
        throw new OrtException(
            OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find CUDA shared provider");
      }
    }

    /**
     * Adds the CPU as an execution backend, using the arena allocator if desired.
     *
     * <p>By default this backend is used, but if other backends are requested, it should be
     * requested last.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addCPU(boolean useArena) throws OrtException {
      checkClosed();
      addCPU(OnnxRuntime.ortApiHandle, nativeHandle, useArena ? 1 : 0);
    }

    /**
     * Adds Intel's Deep Neural Network Library as an execution backend.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addDnnl(boolean useArena) throws OrtException {
      checkClosed();
      if (OnnxRuntime.extractDNNL()) {
        addDnnl(OnnxRuntime.ortApiHandle, nativeHandle, useArena ? 1 : 0);
      } else {
        throw new OrtException(
            OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find DNNL shared provider");
      }
    }

    /**
     * Adds OpenVINO as an execution backend.
     *
     * @param deviceId The id of the OpenVINO execution device.
     * @throws OrtException If there was an error in native code.
     */
    public void addOpenVINO(String deviceId) throws OrtException {
      checkClosed();
      if (OnnxRuntime.extractOpenVINO()) {
        addOpenVINO(OnnxRuntime.ortApiHandle, nativeHandle, deviceId);
      } else {
        throw new OrtException(
            OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find OpenVINO shared provider");
      }
    }

    /**
     * Adds Nvidia's TensorRT as an execution backend.
     *
     * @param deviceNum The id of the CUDA device.
     * @throws OrtException If there was an error in native code.
     */
    public void addTensorrt(int deviceNum) throws OrtException {
      checkClosed();
      if (OnnxRuntime.extractTensorRT()) {
        addTensorrt(OnnxRuntime.ortApiHandle, nativeHandle, deviceNum);
      } else {
        throw new OrtException(
            OrtException.OrtErrorCode.ORT_EP_FAIL, "Failed to find TensorRT shared provider");
      }
    }

    /**
     * Adds Android's NNAPI as an execution backend. Uses the default empty flag.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void addNnapi() throws OrtException {
      addNnapi(EnumSet.noneOf(NNAPIFlags.class));
    }

    /**
     * Adds Android's NNAPI as an execution backend.
     *
     * @param flags The flags which control the NNAPI configuration.
     * @throws OrtException If there was an error in native code.
     */
    public void addNnapi(EnumSet<NNAPIFlags> flags) throws OrtException {
      checkClosed();
      addNnapi(OnnxRuntime.ortApiHandle, nativeHandle, OrtFlags.aggregateToInt(flags));
    }

    /**
     * Adds Nuphar as an execution backend.
     *
     * @param allowUnalignedBuffers Allow unaligned memory buffers.
     * @param settings See the documentation for valid settings strings.
     * @throws OrtException If there was an error in native code.
     */
    public void addNuphar(boolean allowUnalignedBuffers, String settings) throws OrtException {
      checkClosed();
      addNuphar(OnnxRuntime.ortApiHandle, nativeHandle, allowUnalignedBuffers ? 1 : 0, settings);
    }

    /**
     * Adds DirectML as an execution backend.
     *
     * @param deviceId The id of the DirectML device.
     * @throws OrtException If there was an error in native code.
     */
    public void addDirectML(int deviceId) throws OrtException {
      checkClosed();
      addDirectML(OnnxRuntime.ortApiHandle, nativeHandle, deviceId);
    }

    /**
     * Adds the ARM Compute Library as an execution backend.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addACL(boolean useArena) throws OrtException {
      checkClosed();
      addACL(OnnxRuntime.ortApiHandle, nativeHandle, useArena ? 1 : 0);
    }

    /**
     * Adds the ARM Neural Net library as an execution backend.
     *
     * @param useArena If true use the arena memory allocator.
     * @throws OrtException If there was an error in native code.
     */
    public void addArmNN(boolean useArena) throws OrtException {
      checkClosed();
      addArmNN(OnnxRuntime.ortApiHandle, nativeHandle, useArena ? 1 : 0);
    }

    /**
     * Adds ROCM as an execution backend.
     *
     * @param deviceID The ROCM device ID.
     * @param memLimit The maximum amount of memory available.
     * @throws OrtException If there was an error in native code.
     */
    public void addROCM(int deviceID, long memLimit) throws OrtException {
      checkClosed();
      addROCM(OnnxRuntime.ortApiHandle, nativeHandle, deviceID, memLimit);
    }

    /**
     * Adds Apple's CoreML as an execution backend. Uses the default empty flag.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void addCoreML() throws OrtException {
      addCoreML(EnumSet.noneOf(CoreMLFlags.class));
    }

    /**
     * Adds Apple's CoreML as an execution backend.
     *
     * @param flags The flags which control the CoreML configuration.
     * @throws OrtException If there was an error in native code.
     */
    public void addCoreML(EnumSet<CoreMLFlags> flags) throws OrtException {
      checkClosed();
      addCoreML(OnnxRuntime.ortApiHandle, nativeHandle, OrtFlags.aggregateToInt(flags));
    }

    private native void setExecutionMode(long apiHandle, long nativeHandle, int mode)
        throws OrtException;

    private native void setOptimizationLevel(long apiHandle, long nativeHandle, int level)
        throws OrtException;

    private native void setInterOpNumThreads(long apiHandle, long nativeHandle, int numThreads)
        throws OrtException;

    private native void setIntraOpNumThreads(long apiHandle, long nativeHandle, int numThreads)
        throws OrtException;

    private native void setOptimizationModelFilePath(
        long apiHandle, long nativeHandle, String modelPath) throws OrtException;

    private native long createOptions(long apiHandle);

    private native void setLoggerId(long apiHandle, long nativeHandle, String loggerId)
        throws OrtException;

    private native void enableProfiling(long apiHandle, long nativeHandle, String filePrefix)
        throws OrtException;

    private native void disableProfiling(long apiHandle, long nativeHandle) throws OrtException;

    private native void setMemoryPatternOptimization(
        long apiHandle, long nativeHandle, boolean memoryPatternOptimization) throws OrtException;

    private native void setCPUArenaAllocator(long apiHandle, long nativeHandle, boolean useArena)
        throws OrtException;

    private native void setSessionLogLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native void setSessionLogVerbosityLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native long registerCustomOpLibrary(long apiHandle, long nativeHandle, String path)
        throws OrtException;

    private native void closeCustomLibraries(long[] nativeHandle);

    private native void closeOptions(long apiHandle, long nativeHandle);

    private native void addFreeDimensionOverrideByName(
        long apiHandle, long nativeHandle, String dimensionName, long dimensionValue)
        throws OrtException;

    private native void disablePerSessionThreads(long apiHandle, long nativeHandle)
        throws OrtException;

    private native void addConfigEntry(
        long apiHandle, long nativeHandle, String configKey, String configValue)
        throws OrtException;

    /*
     * To use additional providers, you must build ORT with the extra providers enabled. Then call one of these
     * functions to enable them in the session:
     *   OrtSessionOptionsAppendExecutionProvider_CPU
     *   OrtSessionOptionsAppendExecutionProvider_CUDA
     *   OrtSessionOptionsAppendExecutionProvider_<remaining providers...>
     * The order they care called indicates the preference order as well. In other words call this method
     * on your most preferred execution provider first followed by the less preferred ones.
     * If none are called Ort will use its internal CPU execution provider.
     *
     * If a backend is unavailable then it throws an OrtException
     */
    private native void addCPU(long apiHandle, long nativeHandle, int useArena) throws OrtException;

    private native void addCUDA(long apiHandle, long nativeHandle, int deviceNum)
        throws OrtException;

    private native void addDnnl(long apiHandle, long nativeHandle, int useArena)
        throws OrtException;

    private native void addOpenVINO(long apiHandle, long nativeHandle, String deviceId)
        throws OrtException;

    private native void addTensorrt(long apiHandle, long nativeHandle, int deviceNum)
        throws OrtException;

    private native void addNnapi(long apiHandle, long nativeHandle, int nnapiFlags)
        throws OrtException;

    private native void addNuphar(
        long apiHandle, long nativeHandle, int allowUnalignedBuffers, String settings)
        throws OrtException;

    private native void addDirectML(long apiHandle, long nativeHandle, int deviceId)
        throws OrtException;

    private native void addACL(long apiHandle, long nativeHandle, int useArena) throws OrtException;

    private native void addArmNN(long apiHandle, long nativeHandle, int useArena)
        throws OrtException;

    private native void addROCM(long apiHandle, long nativeHandle, int deviceID, long memLimit)
        throws OrtException;

    private native void addCoreML(long apiHandle, long nativeHandle, int coreMLFlags)
        throws OrtException;
  }

  /** Used to control logging and termination of a call to {@link OrtSession#run}. */
  public static class RunOptions implements AutoCloseable {

    private final long nativeHandle;

    private boolean closed = false;

    /**
     * Creates a RunOptions.
     *
     * @throws OrtException If the construction of the native RunOptions failed.
     */
    public RunOptions() throws OrtException {
      this.nativeHandle = createRunOptions(OnnxRuntime.ortApiHandle);
    }

    /**
     * Sets the current logging level on this RunOptions.
     *
     * @param level The new logging level.
     * @throws OrtException If the native call failed.
     */
    public void setLogLevel(OrtLoggingLevel level) throws OrtException {
      checkClosed();
      setLogLevel(OnnxRuntime.ortApiHandle, nativeHandle, level.getValue());
    }

    /**
     * Gets the current logging level set on this RunOptions.
     *
     * @return The logging level.
     * @throws OrtException If the native call failed.
     */
    public OrtLoggingLevel getLogLevel() throws OrtException {
      checkClosed();
      return OrtLoggingLevel.mapFromInt(getLogLevel(OnnxRuntime.ortApiHandle, nativeHandle));
    }

    /**
     * Sets the current logging verbosity level on this RunOptions.
     *
     * @param level The new logging verbosity level.
     * @throws OrtException If the native call failed.
     */
    public void setLogVerbosityLevel(int level) throws OrtException {
      checkClosed();
      setLogVerbosityLevel(OnnxRuntime.ortApiHandle, nativeHandle, level);
    }

    /**
     * Gets the current logging verbosity level set on this RunOptions.
     *
     * @return The logging verbosity level.
     * @throws OrtException If the native call failed.
     */
    public int getLogVerbosityLevel() throws OrtException {
      checkClosed();
      return getLogVerbosityLevel(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    /**
     * Sets the run tag used in logging.
     *
     * @param runTag The run tag in logging output.
     * @throws OrtException If the native library call failed.
     */
    public void setRunTag(String runTag) throws OrtException {
      checkClosed();
      setRunTag(OnnxRuntime.ortApiHandle, nativeHandle, runTag);
    }

    /**
     * Gets the String used to log information about this run.
     *
     * @return The run tag.
     * @throws OrtException If the native library call failed.
     */
    public String getRunTag() throws OrtException {
      checkClosed();
      return getRunTag(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    /**
     * Sets a flag so that all incomplete {@link OrtSession#run} calls using this instance of {@code
     * RunOptions} will terminate as soon as possible. If the flag is false, it resets this {@code
     * RunOptions} so it can be used with other calls to {@link OrtSession#run}.
     *
     * @param terminate If true terminate all runs associated with this RunOptions.
     * @throws OrtException If the native library call failed.
     */
    public void setTerminate(boolean terminate) throws OrtException {
      checkClosed();
      setTerminate(OnnxRuntime.ortApiHandle, nativeHandle, terminate);
    }

    /** Checks if the RunOptions is closed, if so throws {@link IllegalStateException}. */
    private void checkClosed() {
      if (closed) {
        throw new IllegalStateException("Trying to use a closed RunOptions");
      }
    }

    @Override
    public void close() {
      if (!closed) {
        close(OnnxRuntime.ortApiHandle, nativeHandle);
        closed = true;
      } else {
        throw new IllegalStateException("Trying to close an already closed RunOptions");
      }
    }

    private static native long createRunOptions(long apiHandle) throws OrtException;

    private native void setLogLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native int getLogLevel(long apiHandle, long nativeHandle) throws OrtException;

    private native void setLogVerbosityLevel(long apiHandle, long nativeHandle, int logLevel)
        throws OrtException;

    private native int getLogVerbosityLevel(long apiHandle, long nativeHandle) throws OrtException;

    private native void setRunTag(long apiHandle, long nativeHandle, String runTag)
        throws OrtException;

    private native String getRunTag(long apiHandle, long nativeHandle) throws OrtException;

    private native void setTerminate(long apiHandle, long nativeHandle, boolean terminate)
        throws OrtException;

    private static native void close(long apiHandle, long nativeHandle);
  }

  /**
   * An {@link AutoCloseable} wrapper around a {@link Map} containing {@link OnnxValue}s.
   *
   * <p>When this is closed it closes all the {@link OnnxValue}s inside it. If you maintain a
   * reference to a value after this object has been closed it will throw an {@link
   * IllegalStateException} upon access.
   */
  public static class Result implements AutoCloseable, Iterable<Map.Entry<String, OnnxValue>> {

    private static final Logger logger = Logger.getLogger(Result.class.getName());

    private final Map<String, OnnxValue> map;

    private final List<OnnxValue> list;

    private boolean closed;

    /**
     * Creates a Result from the names and values produced by {@link OrtSession#run(Map)}.
     *
     * @param names The output names.
     * @param values The output values.
     */
    Result(String[] names, OnnxValue[] values) {
      map = new LinkedHashMap<>();
      list = new ArrayList<>();

      if (names.length != values.length) {
        throw new IllegalArgumentException(
            "Expected same number of names and values, found names.length = "
                + names.length
                + ", values.length = "
                + values.length);
      }

      for (int i = 0; i < names.length; i++) {
        map.put(names[i], values[i]);
        list.add(values[i]);
      }
      this.closed = false;
    }

    @Override
    public void close() {
      if (!closed) {
        closed = true;
        for (OnnxValue t : map.values()) {
          t.close();
        }
      } else {
        logger.warning("Closing an already closed Result");
      }
    }

    @Override
    public Iterator<Map.Entry<String, OnnxValue>> iterator() {
      if (!closed) {
        return map.entrySet().iterator();
      } else {
        throw new IllegalStateException("Result is closed");
      }
    }

    /**
     * Gets the value from the container at the specified index.
     *
     * <p>Throws {@link IllegalStateException} if the container has been closed, and {@link
     * IndexOutOfBoundsException} if the index is invalid.
     *
     * @param index The index to lookup.
     * @return The value at the index.
     */
    public OnnxValue get(int index) {
      if (!closed) {
        return list.get(index);
      } else {
        throw new IllegalStateException("Result is closed");
      }
    }

    /**
     * Returns the number of outputs in this Result.
     *
     * @return The number of outputs.
     */
    public int size() {
      return map.size();
    }

    /**
     * Gets the value from the container assuming it's not been closed.
     *
     * <p>Throws {@link IllegalStateException} if the container has been closed.
     *
     * @param key The key to lookup.
     * @return Optional.of the value if it exists.
     */
    public Optional<OnnxValue> get(String key) {
      if (!closed) {
        OnnxValue value = map.get(key);
        if (value != null) {
          return Optional.of(value);
        } else {
          return Optional.empty();
        }
      } else {
        throw new IllegalStateException("Result is closed");
      }
    }
  }
}

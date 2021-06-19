/*
 * Copyright (c) 2019, 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import ai.onnxruntime.OnnxMl.TensorProto;
import ai.onnxruntime.OnnxMl.TensorProto.DataType;
import ai.onnxruntime.OrtSession.Result;
import ai.onnxruntime.OrtSession.SessionOptions;
import ai.onnxruntime.OrtSession.SessionOptions.ExecutionMode;
import ai.onnxruntime.OrtSession.SessionOptions.OptLevel;
import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.function.BiFunction;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/** Tests for the onnx-runtime Java interface. */
public class InferenceTest {
  private static final Logger logger = Logger.getLogger(InferenceTest.class.getName());
  private static final Pattern LOAD_PATTERN = Pattern.compile("[,\\[\\] ]");

  private static final String propertiesFile = "Properties.txt";

  private static final Pattern inputPBPattern = Pattern.compile("input_*.pb");
  private static final Pattern outputPBPattern = Pattern.compile("output_*.pb");

  private static Path getResourcePath(String path) {
    return new File(InferenceTest.class.getResource(path).getFile()).toPath();
  }

  @Test
  public void repeatedCloseTest() throws OrtException {
    Logger.getLogger(OrtEnvironment.class.getName()).setLevel(Level.SEVERE);
    OrtEnvironment env = OrtEnvironment.getEnvironment("repeatedCloseTest");
    try (OrtEnvironment otherEnv = OrtEnvironment.getEnvironment()) {
      assertFalse(otherEnv.isClosed());
    }
    assertFalse(env.isClosed());
    env.close();
    assertTrue(env.isClosed());
  }

  @Test
  public void createSessionFromPath() throws OrtException {
    String modelPath = getResourcePath("/squeezenet.onnx").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment("createSessionFromPath");
        OrtSession.SessionOptions options = new SessionOptions()) {
      try (OrtSession session = env.createSession(modelPath, options)) {
        assertNotNull(session);
        assertEquals(1, session.getNumInputs()); // 1 input node
        Map<String, NodeInfo> inputInfoList = session.getInputInfo();
        assertNotNull(inputInfoList);
        assertEquals(1, inputInfoList.size());
        NodeInfo input = inputInfoList.get("data_0");
        assertEquals("data_0", input.getName()); // input node name
        assertTrue(input.getInfo() instanceof TensorInfo);
        TensorInfo inputInfo = (TensorInfo) input.getInfo();
        assertEquals(OnnxJavaType.FLOAT, inputInfo.type);
        int[] expectedInputDimensions = new int[] {1, 3, 224, 224};
        assertEquals(expectedInputDimensions.length, inputInfo.shape.length);
        for (int i = 0; i < expectedInputDimensions.length; i++) {
          assertEquals(expectedInputDimensions[i], inputInfo.shape[i]);
        }

        assertEquals(1, session.getNumOutputs()); // 1 output node
        Map<String, NodeInfo> outputInfoList = session.getOutputInfo();
        assertNotNull(outputInfoList);
        assertEquals(1, outputInfoList.size());
        NodeInfo output = outputInfoList.get("softmaxout_1");
        assertEquals("softmaxout_1", output.getName()); // output node name
        assertTrue(output.getInfo() instanceof TensorInfo);
        TensorInfo outputInfo = (TensorInfo) output.getInfo();
        assertEquals(OnnxJavaType.FLOAT, outputInfo.type);
        int[] expectedOutputDimensions = new int[] {1, 1000, 1, 1};
        assertEquals(expectedOutputDimensions.length, outputInfo.shape.length);
        for (int i = 0; i < expectedOutputDimensions.length; i++) {
          assertEquals(expectedOutputDimensions[i], outputInfo.shape[i]);
        }
      }
    }
  }

  @Test
  public void morePartialInputsTest() throws OrtException {
    String modelPath = getResourcePath("/partial-inputs-test-2.onnx").toString();
    try (OrtEnvironment env =
            OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "partialInputs");
        OrtSession.SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      assertNotNull(session);
      assertEquals(3, session.getNumInputs());
      assertEquals(1, session.getNumOutputs());

      // Input and output collections.
      Map<String, OnnxTensor> inputMap = new HashMap<>();
      Set<String> requestedOutputs = new HashSet<>();

      BiFunction<Result, String, Float> unwrapFunc =
          (r, s) -> {
            try {
              return ((float[]) r.get(s).get().getValue())[0];
            } catch (OrtException e) {
              return Float.NaN;
            }
          };

      // Graph has three scalar inputs, a, b, c, and a single output, ab.
      OnnxTensor a = OnnxTensor.createTensor(env, new float[] {2.0f});
      OnnxTensor b = OnnxTensor.createTensor(env, new float[] {3.0f});
      OnnxTensor c = OnnxTensor.createTensor(env, new float[] {5.0f});

      // Request all outputs, supply all inputs
      inputMap.put("a:0", a);
      inputMap.put("b:0", b);
      inputMap.put("c:0", c);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        assertEquals(1, r.size());
        float abVal = unwrapFunc.apply(r, "ab:0");
        assertEquals(6.0f, abVal, 1e-10);
      }

      // Don't specify an output, expect all of them returned.
      try (Result r = session.run(inputMap)) {
        assertEquals(1, r.size());
        float abVal = unwrapFunc.apply(r, "ab:0");
        assertEquals(6.0f, abVal, 1e-10);
      }

      inputMap.clear();
      requestedOutputs.clear();

      // Request single output ab, supply required inputs
      inputMap.put("a:0", a);
      inputMap.put("b:0", b);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        assertEquals(1, r.size());
        float abVal = unwrapFunc.apply(r, "ab:0");
        assertEquals(6.0f, abVal, 1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();

      // Request output but don't supply the inputs
      inputMap.put("c:0", c);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        fail("Expected to throw OrtException due to incorrect inputs");
      } catch (OrtException e) {
        // System.out.println(e.getMessage());
        // pass
      }
      inputMap.clear();
      requestedOutputs.clear();

      // Request output but don't supply all the inputs
      inputMap.put("b:0", b);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        fail("Expected to throw OrtException due to incorrect inputs");
      } catch (OrtException e) {
        // System.out.println(e.getMessage());
        // pass
      }
    }
  }

  @Test
  public void partialInputsTest() throws OrtException {
    String modelPath = getResourcePath("/partial-inputs-test.onnx").toString();
    try (OrtEnvironment env =
            OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "partialInputs");
        OrtSession.SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      assertNotNull(session);
      assertEquals(3, session.getNumInputs());
      assertEquals(3, session.getNumOutputs());

      // Input and output collections.
      Map<String, OnnxTensor> inputMap = new HashMap<>();
      Set<String> requestedOutputs = new HashSet<>();

      BiFunction<Result, String, Float> unwrapFunc =
          (r, s) -> {
            try {
              return ((float[]) r.get(s).get().getValue())[0];
            } catch (OrtException e) {
              return Float.NaN;
            }
          };

      // Graph has three scalar inputs, a, b, c, and three outputs, ab, bc, ab + bc.
      OnnxTensor a = OnnxTensor.createTensor(env, new float[] {2.0f});
      OnnxTensor b = OnnxTensor.createTensor(env, new float[] {3.0f});
      OnnxTensor c = OnnxTensor.createTensor(env, new float[] {5.0f});

      // Request all outputs, supply all inputs
      inputMap.put("a:0", a);
      inputMap.put("b:0", b);
      inputMap.put("c:0", c);
      requestedOutputs.add("ab:0");
      requestedOutputs.add("bc:0");
      requestedOutputs.add("abc:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        assertEquals(3, r.size());
        float abVal = unwrapFunc.apply(r, "ab:0");
        assertEquals(6.0f, abVal, 1e-10);
        float bcVal = unwrapFunc.apply(r, "bc:0");
        assertEquals(15.0f, bcVal, 1e-10);
        float abcVal = unwrapFunc.apply(r, "abc:0");
        assertEquals(21.0f, abcVal, 1e-10);
      }

      // Don't specify an output, expect all of them returned.
      try (Result r = session.run(inputMap)) {
        assertEquals(3, r.size());
        float abVal = unwrapFunc.apply(r, "ab:0");
        assertEquals(6.0f, abVal, 1e-10);
        float bcVal = unwrapFunc.apply(r, "bc:0");
        assertEquals(15.0f, bcVal, 1e-10);
        float abcVal = unwrapFunc.apply(r, "abc:0");
        assertEquals(21.0f, abcVal, 1e-10);
      }

      inputMap.clear();
      requestedOutputs.clear();

      // Request single output ab, supply all inputs
      inputMap.put("a:0", a);
      inputMap.put("b:0", b);
      inputMap.put("c:0", c);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        assertEquals(1, r.size());
        float abVal = unwrapFunc.apply(r, "ab:0");
        assertEquals(6.0f, abVal, 1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();

      // Request single output abc, supply all inputs
      inputMap.put("a:0", a);
      inputMap.put("b:0", b);
      inputMap.put("c:0", c);
      requestedOutputs.add("abc:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        assertEquals(1, r.size());
        float abcVal = unwrapFunc.apply(r, "abc:0");
        assertEquals(21.0f, abcVal, 1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();

      /* The native library does all the computations, rather than the requested subset.
       * Leaving these tests commented out until it's fixed.
      // Request single output ab, supply required inputs
      inputMap.put("a:0",a);
      inputMap.put("b:0",b);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap,requestedOutputs)) {
          assertEquals(1,r.size());
          float abVal = unwrapFunc.apply(r,"ab:0");
          assertEquals(6.0f,abVal,1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();

      // Request single output bc, supply required inputs
      inputMap.put("b:0",b);
      inputMap.put("c:0",c);
      requestedOutputs.add("bc:0");
      try (Result r = session.run(inputMap,requestedOutputs)) {
          assertEquals(1,r.size());
          float bcVal = unwrapFunc.apply(r,"bc:0");
          assertEquals(15.0f,bcVal,1e-10);
      }
      inputMap.clear();
      requestedOutputs.clear();
      */

      // Request output but don't supply the inputs
      inputMap.put("c:0", c);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        fail("Expected to throw OrtException due to incorrect inputs");
      } catch (OrtException e) {
        // System.out.println(e.getMessage());
        // pass
      }
      inputMap.clear();
      requestedOutputs.clear();

      // Request output but don't supply all the inputs
      inputMap.put("b:0", b);
      requestedOutputs.add("ab:0");
      try (Result r = session.run(inputMap, requestedOutputs)) {
        fail("Expected to throw OrtException due to incorrect inputs");
      } catch (OrtException e) {
        // System.out.println(e.getMessage());
        // pass
      }
    }
  }

  @Test
  public void createSessionFromByteArray() throws IOException, OrtException {
    Path modelPath = getResourcePath("/squeezenet.onnx");
    byte[] modelBytes = Files.readAllBytes(modelPath);
    try (OrtEnvironment env = OrtEnvironment.getEnvironment("createSessionFromByteArray");
        OrtSession.SessionOptions options = new SessionOptions()) {
      try (OrtSession session = env.createSession(modelBytes, options)) {
        assertNotNull(session);
        assertEquals(1, session.getNumInputs()); // 1 input node
        Map<String, NodeInfo> inputInfoList = session.getInputInfo();
        assertNotNull(inputInfoList);
        assertEquals(1, inputInfoList.size());
        NodeInfo input = inputInfoList.get("data_0");
        assertEquals("data_0", input.getName()); // input node name
        assertTrue(input.getInfo() instanceof TensorInfo);
        TensorInfo inputInfo = (TensorInfo) input.getInfo();
        assertEquals(OnnxJavaType.FLOAT, inputInfo.type);
        int[] expectedInputDimensions = new int[] {1, 3, 224, 224};
        assertEquals(expectedInputDimensions.length, inputInfo.shape.length);
        for (int i = 0; i < expectedInputDimensions.length; i++) {
          assertEquals(expectedInputDimensions[i], inputInfo.shape[i]);
        }

        assertEquals(1, session.getNumOutputs()); // 1 output node
        Map<String, NodeInfo> outputInfoList = session.getOutputInfo();
        assertNotNull(outputInfoList);
        assertEquals(1, outputInfoList.size());
        NodeInfo output = outputInfoList.get("softmaxout_1");
        assertEquals("softmaxout_1", output.getName()); // output node name
        assertTrue(output.getInfo() instanceof TensorInfo);
        TensorInfo outputInfo = (TensorInfo) output.getInfo();
        assertEquals(OnnxJavaType.FLOAT, outputInfo.type);
        int[] expectedOutputDimensions = new int[] {1, 1000, 1, 1};
        assertEquals(expectedOutputDimensions.length, outputInfo.shape.length);
        for (int i = 0; i < expectedOutputDimensions.length; i++) {
          assertEquals(expectedOutputDimensions[i], outputInfo.shape[i]);
        }

        // Check the metadata can be extracted
        OnnxModelMetadata metadata = session.getMetadata();
        assertEquals("onnx-caffe2", metadata.getProducerName());
        assertEquals("squeezenet_old", metadata.getGraphName());
        assertEquals("", metadata.getDomain());
        assertEquals("", metadata.getDescription());
        assertEquals(0x7FFFFFFFFFFFFFFFL, metadata.getVersion());
        assertTrue(metadata.getCustomMetadata().isEmpty());
      }
    }
  }

  @Test
  public void environmentThreadPoolTest() throws OrtException {
    Path squeezeNet = getResourcePath("/squeezenet.onnx");
    String modelPath = squeezeNet.toString();
    float[] inputData = loadTensorFromFile(getResourcePath("/bench.in"));
    float[] expectedOutput = loadTensorFromFile(getResourcePath("/bench.expected_out"));
    Map<String, OnnxTensor> container = new HashMap<>();

    OrtEnvironment.ThreadingOptions threadOpts = new OrtEnvironment.ThreadingOptions();
    threadOpts.setGlobalInterOpNumThreads(2);
    threadOpts.setGlobalIntraOpNumThreads(2);
    threadOpts.setGlobalDenormalAsZero();
    threadOpts.setGlobalSpinControl(true);
    try (OrtEnvironment env =
            OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "environmentThreadPoolTest", threadOpts);
        OrtSession.SessionOptions options = new SessionOptions();
        OrtSession.SessionOptions disableThreadOptions = new SessionOptions()) {
      disableThreadOptions.disablePerSessionThreads();

      // Check that the regular session executes
      try (OrtSession session = env.createSession(modelPath, options)) {
        NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
        long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
        Object tensorData = OrtUtil.reshape(inputData, inputShape);
        OnnxTensor tensor = OnnxTensor.createTensor(env, tensorData);
        container.put(inputMeta.getName(), tensor);
        try (OrtSession.Result result = session.run(container)) {
          OnnxValue resultTensor = result.get(0);
          float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
          assertEquals(expectedOutput.length, resultArray.length);
          assertArrayEquals(expectedOutput, resultArray, 1e-6f);
        }
        container.clear();
        tensor.close();
      }

      // Check that the session using the env thread pool executes
      try (OrtSession session = env.createSession(modelPath, disableThreadOptions)) {
        NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
        long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
        Object tensorData = OrtUtil.reshape(inputData, inputShape);
        OnnxTensor tensor = OnnxTensor.createTensor(env, tensorData);
        container.put(inputMeta.getName(), tensor);
        try (OrtSession.Result result = session.run(container)) {
          OnnxValue resultTensor = result.get(0);
          float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
          assertEquals(expectedOutput.length, resultArray.length);
          assertArrayEquals(expectedOutput, resultArray, 1e-6f);
        }
        container.clear();
        tensor.close();
      }
    }

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("test")) {
      try {
        OrtEnvironment newEnv =
            OrtEnvironment.getEnvironment(
                OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, "fail", threadOpts);
        // fail as we can't recreate environments with different threading options
        fail("Should have thrown IllegalStateException");
      } catch (IllegalStateException e) {
        // pass
      }
    }

    threadOpts.close();
  }

  @Test
  public void inferenceTest() throws OrtException {
    canRunInferenceOnAModel(OptLevel.NO_OPT, ExecutionMode.PARALLEL);
    canRunInferenceOnAModel(OptLevel.NO_OPT, ExecutionMode.SEQUENTIAL);
    canRunInferenceOnAModel(OptLevel.ALL_OPT, ExecutionMode.PARALLEL);
    canRunInferenceOnAModel(OptLevel.ALL_OPT, ExecutionMode.SEQUENTIAL);
  }

  private void canRunInferenceOnAModel(OptLevel graphOptimizationLevel, ExecutionMode exectionMode)
      throws OrtException {
    String modelPath = getResourcePath("/squeezenet.onnx").toString();

    // Set the graph optimization level for this session.
    try (OrtEnvironment env = OrtEnvironment.getEnvironment("canRunInferenceOnAModel");
        SessionOptions options = new SessionOptions()) {
      options.setOptimizationLevel(graphOptimizationLevel);
      options.setExecutionMode(exectionMode);

      try (OrtSession session = env.createSession(modelPath, options)) {
        Map<String, NodeInfo> inputMetaMap = session.getInputInfo();
        Map<String, OnnxTensor> container = new HashMap<>();
        NodeInfo inputMeta = inputMetaMap.values().iterator().next();

        float[] inputData = loadTensorFromFile(getResourcePath("/bench.in"));
        // this is the data for only one input tensor for this model
        Object tensorData =
            OrtUtil.reshape(inputData, ((TensorInfo) inputMeta.getInfo()).getShape());
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, tensorData);
        container.put(inputMeta.getName(), inputTensor);

        // Run the inference
        try (OrtSession.Result results = session.run(container)) {
          assertEquals(1, results.size());

          float[] expectedOutput = loadTensorFromFile(getResourcePath("/bench.expected_out"));
          // validate the results
          // Only iterates once
          for (Map.Entry<String, OnnxValue> r : results) {
            OnnxValue resultValue = r.getValue();
            assertTrue(resultValue instanceof OnnxTensor);
            OnnxTensor resultTensor = (OnnxTensor) resultValue;
            int[] expectedDimensions =
                new int[] {1, 1000, 1, 1}; // hardcoded for now for the test data
            long[] resultDimensions = resultTensor.getInfo().getShape();
            assertEquals(expectedDimensions.length, resultDimensions.length);

            for (int i = 0; i < expectedDimensions.length; i++) {
              assertEquals(expectedDimensions[i], resultDimensions[i]);
            }

            float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
            assertEquals(expectedOutput.length, resultArray.length);
            assertArrayEquals(expectedOutput, resultArray, 1e-6f);
          }
        } finally {
          inputTensor.close();
        }
      }
    }
  }

  @Test
  public void throwWrongInputName() throws OrtException {
    SqueezeNetTuple tuple = openSessionSqueezeNet();
    try (OrtEnvironment env = tuple.env;
        OrtSession session = tuple.session) {
      float[] inputData = tuple.inputData;
      NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
      int[] inputDataInt = new int[inputData.length];
      for (int i = 0; i < inputData.length; i++) {
        inputDataInt[i] = (int) inputData[i];
      }
      Object tensor = OrtUtil.reshape(inputDataInt, inputShape);
      container.put("wrong_name", OnnxTensor.createTensor(env, tensor));
      try {
        session.run(container);
        OnnxValue.close(container.values());
        fail("Should throw exception for incorrect name.");
      } catch (OrtException e) {
        OnnxValue.close(container.values());
        String msg = e.getMessage();
        assertTrue(msg.contains("Unknown input name"));
      }
    }
  }

  @Test
  public void throwWrongInputType() throws OrtException {
    SqueezeNetTuple tuple = openSessionSqueezeNet();
    try (OrtEnvironment env = tuple.env;
        OrtSession session = tuple.session) {

      float[] inputData = tuple.inputData;
      NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
      int[] inputDataInt = new int[inputData.length];
      for (int i = 0; i < inputData.length; i++) {
        inputDataInt[i] = (int) inputData[i];
      }
      Object tensor = OrtUtil.reshape(inputDataInt, inputShape);
      container.put(inputMeta.getName(), OnnxTensor.createTensor(env, tensor));
      try {
        session.run(container);
        OnnxValue.close(container.values());
        fail("Should throw exception for incorrect type.");
      } catch (OrtException e) {
        OnnxValue.close(container.values());
        String msg = e.getMessage();
        assertTrue(msg.contains("Unexpected input data type"));
      }
    }
  }

  @Test
  public void throwExtraInputs() throws OrtException {
    SqueezeNetTuple tuple = openSessionSqueezeNet();
    try (OrtEnvironment env = tuple.env;
        OrtSession session = tuple.session) {

      float[] inputData = tuple.inputData;
      NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
      int[] inputDataInt = new int[inputData.length];
      for (int i = 0; i < inputData.length; i++) {
        inputDataInt[i] = (int) inputData[i];
      }
      Object tensor = OrtUtil.reshape(inputDataInt, inputShape);
      container.put(inputMeta.getName(), OnnxTensor.createTensor(env, tensor));
      container.put("extra", OnnxTensor.createTensor(env, tensor));
      try {
        session.run(container);
        OnnxValue.close(container.values());
        fail("Should throw exception for too many inputs.");
      } catch (OrtException e) {
        OnnxValue.close(container.values());
        String msg = e.getMessage();
        assertTrue(msg.contains("Unexpected number of inputs"));
      }
    }
  }

  @Test
  public void testMultiThreads() throws OrtException, InterruptedException {
    int numThreads = 10;
    int loop = 10;
    SqueezeNetTuple tuple = openSessionSqueezeNet();
    try (OrtEnvironment env = tuple.env;
        OrtSession session = tuple.session) {

      float[] inputData = tuple.inputData;
      float[] expectedOutput = tuple.outputData;
      NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
      Object tensor = OrtUtil.reshape(inputData, inputShape);
      container.put(inputMeta.getName(), OnnxTensor.createTensor(env, tensor));
      ExecutorService executor = Executors.newFixedThreadPool(numThreads);
      for (int i = 0; i < numThreads; i++) {
        executor.submit(
            () -> {
              for (int j = 0; j < loop; j++) {
                try (OrtSession.Result result = session.run(container)) {
                  OnnxValue resultTensor = result.get(0);
                  float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
                  assertEquals(expectedOutput.length, resultArray.length);
                  assertArrayEquals(expectedOutput, resultArray, 1e-6f);
                } catch (OrtException e) {
                  throw new IllegalStateException("Failed to execute a scoring operation", e);
                }
              }
            });
      }
      executor.shutdown();
      executor.awaitTermination(1, TimeUnit.MINUTES);
      OnnxValue.close(container.values());
      assertTrue(executor.isTerminated());
    }
  }

  @Test
  public void testProviders() {
    EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
    int providersSize = providers.size();
    assertTrue(providersSize > 0);
    assertTrue(providers.contains(OrtProvider.CPU));

    // Check that the providers are a copy of the original, note this does not enable the DNNL
    // provider
    providers.add(OrtProvider.DNNL);
    assertEquals(providersSize, OrtEnvironment.getAvailableProviders().size());
  }

  @Test
  public void testSymbolicDimensionAssignment() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/capi_symbolic_dims.onnx").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testSymbolicDimensionAssignment")) {
      // Check the dimension is symbolic
      try (SessionOptions options = new SessionOptions()) {
        try (OrtSession session = env.createSession(modelPath, options)) {
          Map<String, NodeInfo> infoMap = session.getInputInfo();
          TensorInfo aInfo = (TensorInfo) infoMap.get("A").getInfo();
          assertArrayEquals(new long[] {-1, 2}, aInfo.shape);
        }
      }
      // Check that when the options are assigned it overrides the symbolic dimension
      try (SessionOptions options = new SessionOptions()) {
        options.setSymbolicDimensionValue("n", 5);
        try (OrtSession session = env.createSession(modelPath, options)) {
          Map<String, NodeInfo> infoMap = session.getInputInfo();
          TensorInfo aInfo = (TensorInfo) infoMap.get("A").getInfo();
          assertArrayEquals(new long[] {5, 2}, aInfo.shape);
        }
      }
    }
  }

  @Test
  @EnabledIfSystemProperty(named = "USE_CUDA", matches = "1")
  public void testCUDA() throws OrtException {
    runProvider(OrtProvider.CUDA);
  }

  @Test
  @EnabledIfSystemProperty(named = "USE_TENSORRT", matches = "1")
  public void testTensorRT() throws OrtException {
    runProvider(OrtProvider.TENSOR_RT);
  }

  @Test
  @EnabledIfSystemProperty(named = "USE_OPENVINO", matches = "1")
  public void testOpenVINO() throws OrtException {
    runProvider(OrtProvider.OPEN_VINO);
  }

  @Test
  @EnabledIfSystemProperty(named = "USE_DNNL", matches = "1")
  public void testDNNL() throws OrtException {
    runProvider(OrtProvider.DNNL);
  }

  private void runProvider(OrtProvider provider) throws OrtException {
    EnumSet<OrtProvider> providers = OrtEnvironment.getAvailableProviders();
    assertTrue(providers.size() > 1);
    assertTrue(providers.contains(OrtProvider.CPU));
    assertTrue(providers.contains(provider));
    SqueezeNetTuple tuple = openSessionSqueezeNet(EnumSet.of(provider));
    try (OrtEnvironment env = tuple.env;
        OrtSession session = tuple.session) {
      float[] inputData = tuple.inputData;
      float[] expectedOutput = tuple.outputData;
      NodeInfo inputMeta = session.getInputInfo().values().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] inputShape = ((TensorInfo) inputMeta.getInfo()).shape;
      Object tensor = OrtUtil.reshape(inputData, inputShape);
      container.put(inputMeta.getName(), OnnxTensor.createTensor(env, tensor));
      try (OrtSession.Result result = session.run(container)) {
        OnnxValue resultTensor = result.get(0);
        float[] resultArray = TestHelpers.flattenFloat(resultTensor.getValue());
        assertEquals(expectedOutput.length, resultArray.length);
        assertArrayEquals(expectedOutput, resultArray, 1e-6f);
      } catch (OrtException e) {
        throw new IllegalStateException("Failed to execute a scoring operation", e);
      }
      OnnxValue.close(container.values());
    }
  }

  private static File getTestModelsDir() throws IOException {
    // get build directory, append downloaded models location
    String cwd = System.getProperty("user.dir");
    List<String> props = Files.readAllLines(Paths.get(cwd, propertiesFile));
    String modelsRelDir = props.get(0).split("=")[1].trim();
    File modelsDir = Paths.get(cwd, "../../..", modelsRelDir, "models").toFile();
    return modelsDir;
  }

  private static Map<String, String> getSkippedModels() {
    Map<String, String> skipModels = new HashMap<>();
    skipModels.put("mxnet_arcface", "Model is an invalid ONNX model");
    skipModels.put("tf_inception_v2", "TODO: Debug failing model, skipping for now");
    skipModels.put("fp16_inception_v1", "16-bit float not supported type in C#.");
    skipModels.put("fp16_shufflenet", "16-bit float not supported type in C#.");
    skipModels.put("fp16_tiny_yolov2", "16-bit float not supported type in C#.");
    skipModels.put(
        "BERT_Squad",
        "Could not find an implementation for the node bert / embeddings / one_hot:OneHot(9)");
    skipModels.put("mlperf_ssd_mobilenet_300", "Could not find file output_0.pb");
    skipModels.put("tf_resnet_v1_50", "result mismatch when Conv BN Fusion is applied");
    skipModels.put("tf_resnet_v1_101", "result mismatch when Conv BN Fusion is applied");
    skipModels.put("tf_resnet_v1_152", "result mismatch when Conv BN Fusion is applied");

    // The following models fails on nocontribops win CI
    skipModels.put("test_tiny_yolov2", "Fails when ContribOps is disabled");
    skipModels.put("mask_rcnn_keras", "Pad is not a registered function/op");

    return skipModels;
  }

  public static List<String[]> getModelsForTest() throws IOException {
    File modelsDir = getTestModelsDir();
    Map<String, String> skipModels = getSkippedModels();
    ArrayList<String[]> output = new ArrayList<>();

    for (File opsetDirectory : modelsDir.listFiles(File::isDirectory)) {
      for (File modelDir : opsetDirectory.listFiles(File::isDirectory)) {
        if (!skipModels.containsKey(modelDir.getName())) {
          output.add(new String[] {modelDir.getParentFile().getName(), modelDir.getName()});
        }
      } // model
    } // opset

    return output;
  }

  public static Set<String[]> getSkippedModelForTest() throws IOException {
    File modelsDir = getTestModelsDir();
    Map<String, String> skipModels = getSkippedModels();
    Set<String[]> output = new HashSet<>();

    for (File opsetDirectory : modelsDir.listFiles(File::isDirectory)) {
      for (File modelDir : opsetDirectory.listFiles(File::isDirectory)) {
        if (skipModels.containsKey(modelDir.getName())) {
          output.add(new String[] {modelDir.getParentFile().getName(), modelDir.getName()});
        }
      } // model
    } // opset

    return output;
  }

  @Disabled
  @Test
  public void testAllPretrainedModels() throws IOException {
    List<String[]> testModels = getModelsForTest();
    Set<String[]> skipModels = getSkippedModelForTest();
    for (String[] model : testModels) {
      if (!skipModels.contains(model)) {
        testPreTrainedModel(model[0], model[1]);
      }
    }
  }

  public void testPreTrainedModel(String opset, String modelName) throws IOException {
    File modelsDir = getTestModelsDir();
    String onnxModelFileName = null;

    Path modelDir = Paths.get(modelsDir.getAbsolutePath(), opset, modelName);

    try {
      File[] onnxModelFiles =
          modelDir.toFile().listFiles((dir, filename) -> filename.contains(".onnx"));
      boolean validModelFound = false;
      if (onnxModelFiles.length > 0) {
        // TODO remove file "._resnet34v2.onnx" from test set
        for (int i = 0; i < onnxModelFiles.length; i++) {
          if (!onnxModelFiles[i].getName().equals("._resnet34v2.onnx")) {
            onnxModelFiles[0] = onnxModelFiles[i];
            validModelFound = true;
          }
        }
      }

      if (validModelFound) {
        onnxModelFileName = onnxModelFiles[0].getAbsolutePath();
      } else {
        String modelNamesList =
            Stream.of(onnxModelFiles).map(File::getName).collect(Collectors.joining(","));
        throw new RuntimeException(
            "Opset "
                + opset
                + " Model "
                + modelName
                + ". Can't determine model file name. Found these: "
                + modelNamesList);
      }

      try (OrtEnvironment env = OrtEnvironment.getEnvironment();
          OrtSession session = env.createSession(onnxModelFileName)) {
        String testDataDirNamePattern;
        if (opset.equals("opset9") && modelName.equals("LSTM_Seq_lens_unpacked")) {
          testDataDirNamePattern = "seq_lens"; // discrepency in data directory
        } else {
          testDataDirNamePattern = "test_data";
        }
        Map<String, NodeInfo> inMeta = session.getInputInfo();
        Map<String, NodeInfo> outMeta = session.getOutputInfo();
        File testDataDir =
            modelDir.toFile().listFiles(f -> f.getName().startsWith(testDataDirNamePattern))[0];
        Map<String, OnnxTensor> inputContainer = new HashMap<>();
        Map<String, OnnxTensor> outputContainer = new HashMap<>();
        for (File f :
            testDataDir.listFiles((dir, name) -> inputPBPattern.matcher(name).matches())) {
          StringTensorPair o = loadTensorFromFilePb(env, f, inMeta);
          inputContainer.put(o.string, o.tensor);
        }
        for (File f :
            testDataDir.listFiles((dir, name) -> outputPBPattern.matcher(name).matches())) {
          StringTensorPair o = loadTensorFromFilePb(env, f, outMeta);
          outputContainer.put(o.string, o.tensor);
        }

        try (Result resultCollection = session.run(inputContainer)) {
          for (Map.Entry<String, OnnxValue> result : resultCollection) {
            Assertions.assertTrue(outMeta.containsKey(result.getKey()));
            OnnxTensor outputValue = outputContainer.get(result.getKey());
            if (outputValue == null) {
              outputValue =
                  outputContainer
                      .values()
                      .iterator()
                      .next(); // in case the output data file does not contain the name
            }
            if (result.getValue() instanceof OnnxTensor) {
              OnnxTensor resultTensor = (OnnxTensor) result.getValue();
              Assertions.assertEquals(outputValue.getByteBuffer(), resultTensor.getByteBuffer());
            } else {
              fail("testPretrainedModel cannot handle non-tensor outputs yet");
            }
          }
        }
      }
    } catch (OrtException ex) {
      String msg =
          "Opset "
              + opset
              + ", Model "
              + modelName
              + ": ModelFile = "
              + onnxModelFileName
              + " error = "
              + ex.getMessage();
      throw new RuntimeException(msg, ex);
    }
  }

  @Test
  public void testModelInputFLOAT() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_FLOAT.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputFLOAT");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      long[] shape = new long[] {1, 5};
      Map<String, OnnxTensor> container = new HashMap<>();
      float[] flatInput = new float[] {1.0f, 2.0f, -3.0f, Float.MIN_VALUE, Float.MAX_VALUE};
      Object tensorIn = OrtUtil.reshape(flatInput, shape);
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      float[] resultArray;
      try (OrtSession.Result res = session.run(container)) {
        resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray, 1e-6f);
        float[] resultBufferArray = new float[flatInput.length];
        ((OnnxTensor) res.get(0)).getFloatBuffer().get(resultBufferArray);
        assertArrayEquals(flatInput, resultBufferArray, 1e-6f);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelInputBuffer() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_FLOAT.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputFLOAT");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      long[] shape = new long[] {1, 5};
      Map<String, OnnxTensor> container = new HashMap<>();
      float[] inputArr =
          new float[] {
            1.0f, -2.0f, 3.0f, -4.0f, 5.0f, -6.0f, 7.0f, -8.0f, 9.0f, -10.0f, 11.0f, -12.0f, 13.0f,
            -14.0f, 15
          };
      FloatBuffer buffer = FloatBuffer.wrap(inputArr);
      FloatBuffer directBuffer =
          ByteBuffer.allocateDirect(inputArr.length * 4)
              .order(ByteOrder.nativeOrder())
              .asFloatBuffer()
              .put(buffer);
      buffer.rewind();
      directBuffer.rewind();
      float[] resultArray;

      // Test loading from buffer
      for (int i = 0; i < 3; i++) {
        // Set limits
        buffer.position(i * 5);
        buffer.limit((i + 1) * 5);
        directBuffer.position(i * 5);
        directBuffer.limit((i + 1) * 5);

        // Check regular buffer (copies to direct)
        OnnxTensor newTensor = OnnxTensor.createTensor(env, buffer, shape);
        container.put(inputName, newTensor);
        try (OrtSession.Result res = session.run(container)) {
          resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
          assertArrayEquals(Arrays.copyOfRange(inputArr, i * 5, (i + 1) * 5), resultArray, 1e-6f);
          OnnxValue.close(container);
        }
        container.clear();
        // buffer should be unchanged
        assertEquals(i * 5, buffer.position());

        // Check direct buffer (no-copy)
        newTensor = OnnxTensor.createTensor(env, directBuffer, shape);
        container.put(inputName, newTensor);
        try (OrtSession.Result res = session.run(container)) {
          resultArray = TestHelpers.flattenFloat(res.get(0).getValue());
          assertArrayEquals(Arrays.copyOfRange(inputArr, i * 5, (i + 1) * 5), resultArray, 1e-6f);
          OnnxValue.close(container);
        }
        container.clear();
        // direct buffer should be unchanged
        assertEquals(i * 5, directBuffer.position());
      }
    }
  }

  @Test
  public void testRunOptions() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_BOOL.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testRunOptions");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options);
        OrtSession.RunOptions runOptions = new OrtSession.RunOptions()) {
      runOptions.setRunTag("monkeys");
      assertEquals("monkeys", runOptions.getRunTag());
      runOptions.setLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL);
      assertEquals(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL, runOptions.getLogLevel());
      runOptions.setLogVerbosityLevel(9000);
      assertEquals(9000, runOptions.getLogVerbosityLevel());
      runOptions.setTerminate(true);
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      boolean[] flatInput = new boolean[] {true, false, true, false, true};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container, runOptions)) {
        fail("Should have terminated.");
      } catch (OrtException e) {
        assertTrue(e.getMessage().contains("Exiting due to terminate flag being set to true."));
        assertEquals(OrtException.OrtErrorCode.ORT_FAIL, e.getCode());
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testExtraSessionOptions() throws OrtException, IOException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_BOOL.pb").toString();
    File tmpPath = File.createTempFile("onnx-runtime-profiling", "file");
    tmpPath.deleteOnExit();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testExtraSessionOptions")) {
      try (SessionOptions options = new SessionOptions()) {
        options.setCPUArenaAllocator(true);
        options.setMemoryPatternOptimization(true);
        options.enableProfiling(tmpPath.getAbsolutePath());
        options.setLoggerId("monkeys");
        options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_FATAL);
        options.setSessionLogVerbosityLevel(5);
        Map<String, String> configEntries = options.getConfigEntries();
        assertTrue(configEntries.isEmpty());
        options.addConfigEntry("key", "value");
        assertEquals("value", configEntries.get("key"));
        try {
          options.addConfigEntry("", "invalid key");
          fail("Add config entry with empty key should have failed");
        } catch (OrtException e) {
          assertTrue(e.getMessage().contains("Config key is empty"));
          assertEquals(OrtException.OrtErrorCode.ORT_INVALID_ARGUMENT, e.getCode());
        }
        try (OrtSession session = env.createSession(modelPath, options)) {
          String inputName = session.getInputNames().iterator().next();
          Map<String, OnnxTensor> container = new HashMap<>();
          boolean[] flatInput = new boolean[] {true, false, true, false, true};
          Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
          OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
          container.put(inputName, ov);
          try (OrtSession.Result res = session.run(container)) {
            boolean[] resultArray = TestHelpers.flattenBoolean(res.get(0).getValue());
            assertArrayEquals(flatInput, resultArray);
          }
          // Check that the profiling start time doesn't throw
          long profilingStartTime = session.getProfilingStartTimeInNs();

          // Check the profiling output doesn't throw
          String profilingOutput = session.endProfiling();
          File profilingOutputFile = new File(profilingOutput);
          profilingOutputFile.deleteOnExit();
          try (OrtSession.Result res = session.run(container)) {
            boolean[] resultArray = TestHelpers.flattenBoolean(res.get(0).getValue());
            assertArrayEquals(flatInput, resultArray);
          }
          OnnxValue.close(container);
        }
      }
      try (SessionOptions options = new SessionOptions()) {
        options.setCPUArenaAllocator(false);
        options.setMemoryPatternOptimization(false);
        options.enableProfiling(tmpPath.getAbsolutePath());
        options.disableProfiling();
        options.setSessionLogVerbosityLevel(0);
        try (OrtSession session = env.createSession(modelPath, options)) {
          String inputName = session.getInputNames().iterator().next();
          Map<String, OnnxTensor> container = new HashMap<>();
          boolean[] flatInput = new boolean[] {true, false, true, false, true};
          Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
          OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
          container.put(inputName, ov);
          try (OrtSession.Result res = session.run(container)) {
            boolean[] resultArray = TestHelpers.flattenBoolean(res.get(0).getValue());
            assertArrayEquals(flatInput, resultArray);
          }
          OnnxValue.close(container);
        }
      }
    }
  }

  @Test
  public void testLoadCustomLibrary() throws OrtException {
    // This test is disabled on Android.
    if (!OnnxRuntime.isAndroid()) {
      String customLibraryName = "";
      String osName = System.getProperty("os.name").toLowerCase();
      if (osName.contains("windows")) {
        // In windows we start in the wrong working directory relative to the custom_op_library.dll
        // So we look it up as a classpath resource and resolve it to a real path
        customLibraryName = getResourcePath("/custom_op_library.dll").toString();
      } else if (osName.contains("mac")) {
        customLibraryName = getResourcePath("/libcustom_op_library.dylib").toString();
      } else if (osName.contains("linux")) {
        customLibraryName = getResourcePath("/libcustom_op_library.so").toString();
      } else {
        fail("Unknown os/platform '" + osName + "'");
      }
      String customOpLibraryTestModel =
          getResourcePath("/custom_op_library/custom_op_test.onnx").toString();

      try (OrtEnvironment env = OrtEnvironment.getEnvironment("testLoadCustomLibrary");
          SessionOptions options = new SessionOptions()) {
        options.registerCustomOpLibrary(customLibraryName);
        try (OrtSession session = env.createSession(customOpLibraryTestModel, options)) {
          Map<String, OnnxTensor> container = new HashMap<>();

          // prepare expected inputs and outputs
          float[] flatInputOne =
              new float[] {
                1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f, 10.0f, 11.1f, 12.2f, 13.3f,
                14.4f, 15.5f
              };
          Object tensorIn = OrtUtil.reshape(flatInputOne, new long[] {3, 5});
          OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
          container.put("input_1", ov);

          float[] flatInputTwo =
              new float[] {
                15.5f, 14.4f, 13.3f, 12.2f, 11.1f, 10.0f, 9.9f, 8.8f, 7.7f, 6.6f, 5.5f, 4.4f, 3.3f,
                2.2f, 1.1f
              };
          tensorIn = OrtUtil.reshape(flatInputTwo, new long[] {3, 5});
          ov = OnnxTensor.createTensor(env, tensorIn);
          container.put("input_2", ov);

          int[] flatOutput = new int[] {17, 17, 17, 17, 17, 17, 18, 18, 18, 17, 17, 17, 17, 17, 17};

          try (OrtSession.Result res = session.run(container)) {
            OnnxTensor outputTensor = (OnnxTensor) res.get(0);
            assertArrayEquals(new long[] {3, 5}, outputTensor.getInfo().shape);
            int[] resultArray = TestHelpers.flattenInteger(res.get(0).getValue());
            assertArrayEquals(flatOutput, resultArray);
          }
          OnnxValue.close(container);
        }
      }
    }
  }

  @Test
  public void testModelInputBOOL() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_BOOL.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputBOOL");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      boolean[] flatInput = new boolean[] {true, false, true, false, true};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        boolean[] resultArray = TestHelpers.flattenBoolean(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelInputINT32() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_INT32.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT32");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      int[] flatInput = new int[] {1, -2, -3, Integer.MIN_VALUE, Integer.MAX_VALUE};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        int[] resultArray = TestHelpers.flattenInteger(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelInputDOUBLE() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_DOUBLE.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputDOUBLE");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      double[] flatInput = new double[] {1.0, 2.0, -3.0, 5, 5};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        double[] resultArray = TestHelpers.flattenDouble(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray, 1e-6f);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelInputINT8() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_INT8.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT8");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      byte[] flatInput = new byte[] {1, 2, -3, Byte.MIN_VALUE, Byte.MAX_VALUE};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        byte[] resultArray = TestHelpers.flattenByte(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelInputINT16() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_INT16.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT16");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      short[] flatInput = new short[] {1, 2, 3, Short.MIN_VALUE, Short.MAX_VALUE};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        short[] resultArray = TestHelpers.flattenShort(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelInputINT64() throws OrtException {
    // model takes 1x5 input of fixed type, echoes back
    String modelPath = getResourcePath("/test_types_INT64.pb").toString();

    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelInputINT64");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {
      String inputName = session.getInputNames().iterator().next();
      Map<String, OnnxTensor> container = new HashMap<>();
      long[] flatInput = new long[] {1, 2, -3, Long.MIN_VALUE, Long.MAX_VALUE};
      Object tensorIn = OrtUtil.reshape(flatInput, new long[] {1, 5});
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);
      try (OrtSession.Result res = session.run(container)) {
        long[] resultArray = TestHelpers.flattenLong(res.get(0).getValue());
        assertArrayEquals(flatInput, resultArray);
      }
      OnnxValue.close(container);
    }
  }

  @Test
  public void testModelSequenceOfMapIntFloat() throws OrtException {
    // test model trained using lightgbm classifier
    // produces 2 named outputs
    //   "label" is a tensor,
    //   "probabilities" is a sequence<map<int64, float>>
    // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py

    String modelPath = getResourcePath("/test_sequence_map_int_float.pb").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelSequenceOfMapIntFloat");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {

      Map<String, NodeInfo> outputInfos = session.getOutputInfo();
      Iterator<NodeInfo> valuesItr = outputInfos.values().iterator();
      NodeInfo firstOutputInfo = valuesItr.next();
      NodeInfo secondOutputInfo = valuesItr.next();
      assertTrue(firstOutputInfo.getInfo() instanceof TensorInfo);
      assertTrue(secondOutputInfo.getInfo() instanceof SequenceInfo);
      assertEquals(OnnxJavaType.INT64, ((TensorInfo) firstOutputInfo.getInfo()).type);
      assertTrue(((SequenceInfo) secondOutputInfo.getInfo()).sequenceOfMaps);
      assertEquals(OnnxJavaType.UNKNOWN, ((SequenceInfo) secondOutputInfo.getInfo()).sequenceType);
      MapInfo mapInfo = ((SequenceInfo) secondOutputInfo.getInfo()).mapInfo;
      assertNotNull(mapInfo);
      assertEquals(OnnxJavaType.INT64, mapInfo.keyType);
      assertEquals(OnnxJavaType.FLOAT, mapInfo.valueType);

      Map<String, OnnxTensor> container = new HashMap<>();
      long[] shape = new long[] {1, 2};
      float[] flatInput = new float[] {5.8f, 2.8f};
      Object tensorIn = OrtUtil.reshape(flatInput, shape);
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(session.getInputNames().iterator().next(), ov);

      try (OrtSession.Result outputs = session.run(container)) {
        assertEquals(2, outputs.size());

        // first output is a tensor containing label
        OnnxValue firstOutput = outputs.get(firstOutputInfo.getName()).get();
        assertTrue(firstOutput instanceof OnnxTensor);

        // try-cast as a tensor
        long[] labelOutput = (long[]) firstOutput.getValue();

        // Label 1 should have highest probability
        assertEquals(1, labelOutput[0]);
        assertEquals(1, labelOutput.length);

        // second output is a sequence<map<int64, float>>
        // try-cast to an sequence of NOV
        OnnxValue secondOutput = outputs.get(secondOutputInfo.getName()).get();
        assertTrue(secondOutput instanceof OnnxSequence);
        SequenceInfo sequenceInfo = ((OnnxSequence) secondOutput).getInfo();
        assertTrue(sequenceInfo.sequenceOfMaps);
        assertEquals(OnnxJavaType.INT64, sequenceInfo.mapInfo.keyType);
        assertEquals(OnnxJavaType.FLOAT, sequenceInfo.mapInfo.valueType);

        // try-cast first element in sequence to map/dictionary type
        @SuppressWarnings("unchecked")
        Map<Long, Float> map = (Map<Long, Float>) ((List<Object>) secondOutput.getValue()).get(0);
        assertEquals(0.25938290, map.get(0L), 1e-6);
        assertEquals(0.40904793, map.get(1L), 1e-6);
        assertEquals(0.33156919, map.get(2L), 1e-6);
      }
      ov.close();
    }
  }

  @Test
  public void testModelSequenceOfMapStringFloat() throws OrtException {
    // test model trained using lightgbm classifier
    // produces 2 named outputs
    //   "label" is a tensor,
    //   "probabilities" is a sequence<map<int64, float>>
    // https://github.com/onnx/sklearn-onnx/blob/master/docs/examples/plot_pipeline_lightgbm.py
    String modelPath = getResourcePath("/test_sequence_map_string_float.pb").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testModelSequenceOfMapStringFloat");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {

      Map<String, NodeInfo> outputInfos = session.getOutputInfo();
      Iterator<NodeInfo> valuesItr = outputInfos.values().iterator();
      NodeInfo firstOutputInfo = valuesItr.next();
      NodeInfo secondOutputInfo = valuesItr.next();
      assertTrue(firstOutputInfo.getInfo() instanceof TensorInfo);
      assertTrue(secondOutputInfo.getInfo() instanceof SequenceInfo);
      assertEquals(OnnxJavaType.STRING, ((TensorInfo) firstOutputInfo.getInfo()).type);
      assertTrue(((SequenceInfo) secondOutputInfo.getInfo()).sequenceOfMaps);
      assertEquals(OnnxJavaType.UNKNOWN, ((SequenceInfo) secondOutputInfo.getInfo()).sequenceType);
      MapInfo mapInfo = ((SequenceInfo) secondOutputInfo.getInfo()).mapInfo;
      assertNotNull(mapInfo);
      assertEquals(OnnxJavaType.STRING, mapInfo.keyType);
      assertEquals(OnnxJavaType.FLOAT, mapInfo.valueType);

      Map<String, OnnxTensor> container = new HashMap<>();
      long[] shape = new long[] {1, 2};
      float[] flatInput = new float[] {5.8f, 2.8f};
      Object tensorIn = OrtUtil.reshape(flatInput, shape);
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(session.getInputNames().iterator().next(), ov);

      try (OrtSession.Result outputs = session.run(container)) {
        assertEquals(2, outputs.size());

        // first output is a tensor containing label
        OnnxValue firstOutput = outputs.get(firstOutputInfo.getName()).get();
        assertTrue(firstOutput instanceof OnnxTensor);

        // try-cast as a tensor
        String[] labelOutput = (String[]) firstOutput.getValue();

        // Label 1 should have highest probability
        assertEquals("1", labelOutput[0]);
        assertEquals(1, labelOutput.length);

        // second output is a sequence<map<int64, float>>
        // try-cast to an sequence of NOV
        OnnxValue secondOutput = outputs.get(secondOutputInfo.getName()).get();
        assertTrue(secondOutput instanceof OnnxSequence);
        SequenceInfo sequenceInfo = ((OnnxSequence) secondOutput).getInfo();
        assertTrue(sequenceInfo.sequenceOfMaps);
        assertEquals(OnnxJavaType.STRING, sequenceInfo.mapInfo.keyType);
        assertEquals(OnnxJavaType.FLOAT, sequenceInfo.mapInfo.valueType);

        // try-cast first element in sequence to map/dictionary type
        @SuppressWarnings("unchecked")
        Map<String, Float> map =
            (Map<String, Float>) ((List<Object>) secondOutput.getValue()).get(0);
        assertEquals(0.25938290, map.get("0"), 1e-6);
        assertEquals(0.40904793, map.get("1"), 1e-6);
        assertEquals(0.33156919, map.get("2"), 1e-6);
      }
      ov.close();
    }
  }

  @Test
  public void testModelSerialization() throws OrtException, IOException {
    String cwd = System.getProperty("user.dir");
    Path squeezeNet = getResourcePath("/squeezenet.onnx");
    String modelPath = squeezeNet.toString();
    File tmpFile = File.createTempFile("optimized-squeezenet", ".onnx");
    String modelOutputPath = tmpFile.getAbsolutePath();
    Assertions.assertEquals(0, tmpFile.length());
    try (OrtEnvironment env = OrtEnvironment.getEnvironment()) {
      // Set the optimized model file path to assert that no exception are thrown.
      SessionOptions options = new SessionOptions();
      options.setOptimizedModelFilePath(modelOutputPath);
      options.setOptimizationLevel(OptLevel.BASIC_OPT);
      try (OrtSession session = env.createSession(modelPath, options)) {
        Assertions.assertNotNull(session);
      } finally {
        Assertions.assertTrue(tmpFile.length() > 0);
      }
    } finally {
      tmpFile.delete();
    }
  }

  @Test
  public void testStringIdentity() throws OrtException {
    String modelPath = getResourcePath("/identity_string.onnx").toString();
    try (OrtEnvironment env = OrtEnvironment.getEnvironment("testStringIdentity");
        SessionOptions options = new SessionOptions();
        OrtSession session = env.createSession(modelPath, options)) {

      Map<String, NodeInfo> outputInfos = session.getOutputInfo();
      ValueInfo firstOutputInfo = outputInfos.values().iterator().next().getInfo();
      assertTrue(firstOutputInfo instanceof TensorInfo);
      assertEquals(OnnxJavaType.STRING, ((TensorInfo) firstOutputInfo).type);

      String inputName = session.getInputNames().iterator().next();

      Map<String, OnnxTensor> container = new HashMap<>();
      String[][] tensorIn =
          new String[][] {new String[] {"this", "is"}, new String[] {"identity", "test \u263A"}};
      OnnxTensor ov = OnnxTensor.createTensor(env, tensorIn);
      container.put(inputName, ov);

      try (OrtSession.Result outputs = session.run(container)) {
        assertEquals(1, outputs.size());

        OnnxValue firstOutput = outputs.get(0);
        assertTrue(firstOutput instanceof OnnxTensor);

        String[] labelOutput = (String[]) firstOutput.getValue();

        assertEquals("this", labelOutput[0]);
        assertEquals("is", labelOutput[1]);
        assertEquals("identity", labelOutput[2]);
        assertEquals("test \u263A", labelOutput[3]);
        assertEquals(4, labelOutput.length);

        OnnxValue.close(container);
        container.clear();
      }

      String[] tensorInFlatArr = new String[] {"this", "is", "identity", "test \u263A"};
      ov = OnnxTensor.createTensor(env, tensorInFlatArr, new long[] {2, 2});
      container.put(inputName, ov);

      try (OrtSession.Result outputs = session.run(container)) {
        assertEquals(1, outputs.size());

        OnnxValue firstOutput = outputs.get(0);
        assertTrue(firstOutput instanceof OnnxTensor);

        String[] labelOutput = (String[]) firstOutput.getValue();

        assertEquals("this", labelOutput[0]);
        assertEquals("is", labelOutput[1]);
        assertEquals("identity", labelOutput[2]);
        assertEquals("test \u263A", labelOutput[3]);
        assertEquals(4, labelOutput.length);
      }
    }
  }

  /** Carrier tuple for the squeeze net model. */
  private static class SqueezeNetTuple {
    public final OrtEnvironment env;
    public final OrtSession session;
    public final float[] inputData;
    public final float[] outputData;

    public SqueezeNetTuple(
        OrtEnvironment env, OrtSession session, float[] inputData, float[] outputData) {
      this.env = env;
      this.session = session;
      this.inputData = inputData;
      this.outputData = outputData;
    }
  }

  private static SqueezeNetTuple openSessionSqueezeNet() throws OrtException {
    return openSessionSqueezeNet(EnumSet.noneOf(OrtProvider.class));
  }

  /**
   * Loads the squeezenet model into a session using the supplied providers.
   *
   * @param providers The providers to activate.
   * @return The squeezenet session, input and output.
   * @throws OrtException If the native code failed.
   */
  private static SqueezeNetTuple openSessionSqueezeNet(EnumSet<OrtProvider> providers)
      throws OrtException {
    Path squeezeNet = getResourcePath("/squeezenet.onnx");
    String modelPath = squeezeNet.toString();
    OrtEnvironment env = OrtEnvironment.getEnvironment();
    SessionOptions options = new SessionOptions();
    for (OrtProvider p : providers) {
      switch (p) {
        case CUDA:
          options.addCUDA();
          break;
        case DNNL:
          options.addDnnl(false);
          break;
        case OPEN_VINO:
          options.addOpenVINO("");
          break;
        case TENSOR_RT:
          options.addTensorrt(0);
          break;
        case NNAPI:
          options.addNnapi();
          break;
        case DIRECT_ML:
          options.addDirectML(0);
          break;
        case ACL:
          options.addACL(false);
          break;
        case ARM_NN:
          options.addArmNN(false);
          break;
        case ROCM:
          options.addROCM(0, 4 * 1024 * 1024);
          break;
        case CORE_ML:
          options.addCoreML();
          break;
        case NUPHAR:
          options.addNuphar(true, "");
          break;
        case VITIS_AI:
        case RK_NPU:
        case MI_GRAPH_X:
        default:
          logger.warning("Unsupported provider in Java test " + p);
      }
    }
    OrtSession session = env.createSession(modelPath, options);
    float[] inputData = loadTensorFromFile(getResourcePath("/bench.in"));
    float[] expectedOutput = loadTensorFromFile(getResourcePath("/bench.expected_out"));
    return new SqueezeNetTuple(env, session, inputData, expectedOutput);
  }

  private static float[] loadTensorFromFile(Path filename) {
    return loadTensorFromFile(filename, true);
  }

  private static float[] loadTensorFromFile(Path filename, boolean skipHeader) {
    // read data from file
    try (BufferedReader reader = new BufferedReader(new FileReader(filename.toFile()))) {
      if (skipHeader) {
        reader.readLine(); // skip the input name
      }
      String[] dataStr = LOAD_PATTERN.split(reader.readLine());
      List<Float> tensorData = new ArrayList<>();
      for (int i = 0; i < dataStr.length; i++) {
        if (!dataStr[i].isEmpty()) {
          tensorData.add(Float.parseFloat(dataStr[i]));
        }
      }
      return TestHelpers.toPrimitiveFloat(tensorData);
    } catch (IOException e) {
      throw new UncheckedIOException(e);
    }
  }

  private static class TypeWidth {
    public final OnnxJavaType type;
    public final int width;

    public TypeWidth(OnnxJavaType type, int width) {
      this.type = type;
      this.width = width;
    }
  }

  private static TypeWidth getTypeAndWidth(TensorProto.DataType elemType) {
    OnnxJavaType type;
    int width;
    switch (elemType) {
      case FLOAT:
        type = OnnxJavaType.FLOAT;
        width = 4;
        break;
      case UINT8:
      case INT8:
        type = OnnxJavaType.INT8;
        width = 1;
        break;
      case UINT16:
      case INT16:
        type = OnnxJavaType.INT16;
        width = 2;
        break;
      case INT32:
      case UINT32:
        type = OnnxJavaType.INT32;
        width = 4;
        break;
      case INT64:
      case UINT64:
        type = OnnxJavaType.INT64;
        width = 8;
        break;
      case STRING:
        type = OnnxJavaType.STRING;
        width = 1;
        break;
      case BOOL:
        type = OnnxJavaType.BOOL;
        width = 1;
        break;
      case FLOAT16:
        type = OnnxJavaType.FLOAT;
        width = 2;
        break;
      case DOUBLE:
        type = OnnxJavaType.DOUBLE;
        width = 8;
        break;
      default:
        type = null;
        width = 0;
        break;
    }
    return new TypeWidth(type, width);
  }

  private static StringTensorPair loadTensorFromFilePb(
      OrtEnvironment env, File filename, Map<String, NodeInfo> nodeMetaDict)
      throws IOException, OrtException {
    InputStream is = new BufferedInputStream(new FileInputStream(filename), 1024 * 1024 * 4);
    OnnxMl.TensorProto tensor = OnnxMl.TensorProto.parseFrom(is);
    is.close();

    TypeWidth tw = getTypeAndWidth(DataType.forNumber(tensor.getDataType()));
    int width = tw.width;
    OnnxJavaType tensorElemType = tw.type;
    long[] intDims = new long[tensor.getDimsCount()];
    for (int i = 0; i < tensor.getDimsCount(); i++) {
      intDims[i] = tensor.getDims(i);
    }

    TensorInfo nodeMeta = null;
    String nodeName = "";
    if (nodeMetaDict.size() == 1) {
      for (Map.Entry<String, NodeInfo> e : nodeMetaDict.entrySet()) {
        nodeMeta = (TensorInfo) e.getValue().getInfo();
        nodeName = e.getKey(); // valid for single node input
      }
    } else if (nodeMetaDict.size() > 1) {
      if (!tensor.getName().isEmpty()) {
        nodeMeta = (TensorInfo) nodeMetaDict.get(tensor.getName()).getInfo();
        nodeName = tensor.getName();
      } else {
        boolean matchfound = false;
        // try to find from matching type and shape
        for (Map.Entry<String, NodeInfo> e : nodeMetaDict.entrySet()) {
          if (e.getValue().getInfo() instanceof TensorInfo) {
            TensorInfo meta = (TensorInfo) e.getValue().getInfo();
            if (tensorElemType == meta.type && tensor.getDimsCount() == meta.shape.length) {
              int i = 0;
              for (; i < meta.shape.length; i++) {
                if (meta.shape[i] != -1 && meta.shape[i] != intDims[i]) {
                  break;
                }
              }
              if (i >= meta.shape.length) {
                matchfound = true;
                nodeMeta = meta;
                nodeName = e.getKey();
                break;
              }
            }
          }
        }
        if (!matchfound) {
          // throw error
          throw new IllegalStateException(
              "No matching Tensor found in InputOutputMetadata corresponding to the serialized tensor loaded from "
                  + filename);
        }
      }
    } else {
      // throw error
      throw new IllegalStateException(
          "While reading the serialized tensor loaded from "
              + filename
              + ", metaDataDict has 0 elements");
    }

    Assertions.assertEquals(tensorElemType, nodeMeta.type);
    Assertions.assertEquals(nodeMeta.shape.length, tensor.getDimsCount());
    for (int i = 0; i < nodeMeta.shape.length; i++) {
      Assertions.assertTrue((nodeMeta.shape[i] == -1) || (nodeMeta.shape[i] == intDims[i]));
    }

    ByteBuffer buffer = ByteBuffer.wrap(tensor.getRawData().toByteArray());

    OnnxTensor onnxTensor = OnnxTensor.createTensor(env, buffer, intDims, tensorElemType);

    return new StringTensorPair(nodeName, onnxTensor);
  }

  private static class StringTensorPair {
    public final String string;
    public final OnnxTensor tensor;

    public StringTensorPair(String string, OnnxTensor tensor) {
      this.string = string;
      this.tensor = tensor;
    }
  }
}

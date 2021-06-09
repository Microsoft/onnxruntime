/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * A Java interface to the ONNX Runtime.
 *
 * <p>Provides access to the same execution backends as the C library. Non-representable types in
 * Java (such as fp16) are converted into the nearest Java primitive type when accessed through this
 * API.
 *
 * <p>There are two shared libraries required: <code>onnxruntime</code> and <code>onnxruntime4j_jni
 * </code>. The loader is in {@link ai.onnxruntime.OnnxRuntime} and the logic is in this order:
 *
 * <ol>
 *   <li>The user may signal to skip loading of a shared library using a property in the form <code>
 *       onnxruntime.native.LIB_NAME.skip</code> with a value of <code>true</code>. This means the
 *       user has decided to load the library by some other means.
 *   <li>The user may specify an explicit location of all native library files using a property in
 *       the form <code>onnxruntime.native.path</code>. This uses {@link java.lang.System#load}.
 *   <li>The user may specify an explicit location of the shared library file using a property in
 *       the form <code>onnxruntime.native.LIB_NAME.path</code>. This uses {@link
 *       java.lang.System#load}.
 *   <li>The shared library is autodiscovered:
 *       <ol>
 *         <li>If the shared library is present in the classpath resources, load using {@link
 *             java.lang.System#load} via a temporary file. Ideally, this should be the default use
 *             case when adding JAR's/dependencies containing the shared libraries to your
 *             classpath.
 *         <li>If the shared library is not present in the classpath resources, then load using
 *             {@link java.lang.System#loadLibrary}, which usually looks elsewhere on the filesystem
 *             for the library. The semantics and behavior of that method are system/JVM dependent.
 *             Typically, the <code>java.library.path</code> property is used to specify the
 *             location of native libraries.
 *       </ol>
 * </ol>
 *
 * For troubleshooting, all shared library loading events are reported to Java logging at the level
 * FINE.
 *
 * <p>Note that CUDA, DNNL, OpenVINO and TensorRT are all "shared library execution providers", and
 * must be stored either in the directory containing the ONNX Runtime core native library, or as a
 * classpath resource. This is because these providers are loaded by the ONNX Runtime native library
 * itself and the Java API cannot control the loading location.
 */
package ai.onnxruntime;

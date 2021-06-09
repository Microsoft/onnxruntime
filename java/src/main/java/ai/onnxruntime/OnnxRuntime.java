/*
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.Locale;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Static loader for the JNI binding. No public API, but called from various classes in this package
 * to ensure shared libraries are properly loaded.
 */
final class OnnxRuntime {
  private static final Logger logger = Logger.getLogger(OnnxRuntime.class.getName());

  // The initial release of the ORT API.
  private static final int ORT_API_VERSION_1 = 1;
  // Post 1.0 builds of the ORT API.
  private static final int ORT_API_VERSION_2 = 2;
  // Post 1.3 builds of the ORT API
  private static final int ORT_API_VERSION_3 = 3;
  // Post 1.6 builds of the ORT API
  private static final int ORT_API_VERSION_7 = 7;

  /** The short name of the ONNX runtime shared library */
  static final String ONNXRUNTIME_LIBRARY_NAME = "onnxruntime";
  /** The short name of the ONNX runtime JNI shared library */
  static final String ONNXRUNTIME_JNI_LIBRARY_NAME = "onnxruntime4j_jni";

  /** The short name of the ONNX runtime shared provider library */
  static final String ONNXRUNTIME_LIBRARY_SHARED_NAME = "onnxruntime_providers_shared";
  /** The short name of the ONNX runtime cuda provider library */
  static final String ONNXRUNTIME_LIBRARY_CUDA_NAME = "onnxruntime_providers_cuda";

  private static final String OS_ARCH_STR = initOsArch();

  private static boolean loaded = false;

  /** The API handle. */
  static long ortApiHandle;

  /** The available runtime providers */
  static EnumSet<OrtProvider> providers;

  private OnnxRuntime() {}

  /* Computes and initializes OS_ARCH_STR (such as linux-x64) */
  private static String initOsArch() {
    String detectedOS = null;
    String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
    if (os.contains("mac") || os.contains("darwin")) {
      detectedOS = "osx";
    } else if (os.contains("win")) {
      detectedOS = "win";
    } else if (os.contains("nux")) {
      detectedOS = "linux";
    } else if (isAndroid()) {
      detectedOS = "android";
    } else {
      throw new IllegalStateException("Unsupported os:" + os);
    }
    String detectedArch = null;
    String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
      detectedArch = "x64";
    } else if (arch.startsWith("x86")) {
      // 32-bit x86 is not supported by the Java API
      detectedArch = "x86";
    } else if (arch.startsWith("aarch64")) {
      detectedArch = "aarch64";
    } else if (arch.startsWith("ppc64")) {
      detectedArch = "ppc64";
    } else if (isAndroid()) {
      detectedArch = arch;
    } else {
      throw new IllegalStateException("Unsupported arch:" + arch);
    }
    return detectedOS + '-' + detectedArch;
  }

  /**
   * Loads the native C library.
   *
   * @throws IOException If it can't write to disk to copy out the library from the jar file.
   */
  static synchronized void init() throws IOException {
    if (loaded) {
      return;
    }
    Path tempDirectory = isAndroid() ? null : Files.createTempDirectory("onnxruntime-java");
    try {
      // Extract and prepare the shared provider libraries but don't try to load them, Onnxruntime
      // itself will load them
      load(tempDirectory, ONNXRUNTIME_LIBRARY_SHARED_NAME, false);
      load(tempDirectory, ONNXRUNTIME_LIBRARY_CUDA_NAME, false);

      load(tempDirectory, ONNXRUNTIME_LIBRARY_NAME, true);
      load(tempDirectory, ONNXRUNTIME_JNI_LIBRARY_NAME, true);
      ortApiHandle = initialiseAPIBase(ORT_API_VERSION_7);
      providers = initialiseProviders(ortApiHandle);
      loaded = true;
    } finally {
      if (tempDirectory != null) {
        cleanUp(tempDirectory.toFile(), false);
      }
    }
  }

  /**
   * Attempt to remove a file and then mark for delete on exit if it cannot be deleted at this point
   * in time.
   *
   * @param file The file to remove.
   * @param onExitOnly Delete the file on exit only, vs trying to do it immediately
   */
  private static void cleanUp(File file, boolean onExitOnly) {
    if (!file.exists()) {
      return;
    }
    logger.log(Level.FINE, "Deleting " + file);
    if (onExitOnly || !file.delete()) {
      logger.log(Level.FINE, "Deleting " + file + " on exit");
      file.deleteOnExit();
    }
  }

  /**
   * Check if we're running on Android.
   *
   * @return True if the property java.vendor equals The Android Project, false otherwise.
   */
  static boolean isAndroid() {
    return System.getProperty("java.vendor", "generic").equals("The Android Project");
  }

  /**
   * Load a shared library by name.
   *
   * @param tempDirectory The temp directory to write the library resource to.
   * @param library The bare name of the library.
   * @param systemLoad If system.Load(..) should be called on the library vs just preparing it
   * @throws IOException If the file failed to read or write.
   */
  private static void load(Path tempDirectory, String library, boolean systemLoad)
      throws IOException {
    // On Android, we simply use System.loadLibrary
    if (isAndroid()) {
      System.loadLibrary("onnxruntime4j_jni");
      return;
    }

    // 1) The user may skip loading of this library:
    String skip = System.getProperty("onnxruntime.native." + library + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skip)) {
      logger.log(Level.FINE, "Skipping load of native library '" + library + "'");
      return;
    }

    // Resolve the platform dependent library name.
    String libraryFileName = System.mapLibraryName(library).replace("jnilib", "dylib");

    // 2) The user may explicitly specify the path to a directory containing all shared libraries:
    String libraryDirPathProperty = System.getProperty("onnxruntime.native.path");
    if (libraryDirPathProperty != null) {
      logger.log(
              Level.FINE,
              "Attempting to load native library '"
                      + library
                      + "' from specified path: "
                      + libraryDirPathProperty);
      File libraryFile = Path.of(libraryDirPathProperty,libraryFileName).toFile();
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }
      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 3) The user may explicitly specify the path to their shared library:
    String libraryPathProperty = System.getProperty("onnxruntime.native." + library + ".path");
    if (libraryPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryPathProperty);
      File libraryFile = new File(libraryPathProperty);
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }
      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 4) try loading from resources or library path:
    // generate a platform specific library name
    // replace Mac's jnilib extension to dylib
    String resourcePath = "/ai/onnxruntime/native/" + OS_ARCH_STR + '/' + libraryFileName;
    File tempFile = tempDirectory.resolve(libraryFileName).toFile();
    try (InputStream is = OnnxRuntime.class.getResourceAsStream(resourcePath)) {
      if (is == null) {
        // 3a) Not found in resources, load from library path
        if (!systemLoad) {
          return; // Failure is expected for optional components we don't need to load
        }
        logger.log(
            Level.FINE, "Attempting to load native library '" + library + "' from library path");
        System.loadLibrary(library);
        logger.log(Level.FINE, "Loaded native library '" + library + "' from library path");
      } else {
        // 3b) Found in resources, load via temporary file
        logger.log(
            Level.FINE,
            "Attempting to load native library '"
                + library
                + "' from resource path "
                + resourcePath
                + " copying to "
                + tempFile);
        byte[] buffer = new byte[1024];
        int readBytes;
        try (FileOutputStream os = new FileOutputStream(tempFile)) {
          while ((readBytes = is.read(buffer)) != -1) {
            os.write(buffer, 0, readBytes);
          }
        }
        if (systemLoad) {
          System.load(tempFile.getAbsolutePath());
          logger.log(Level.FINE, "Loaded native library '" + library + "' from resource path");
        } else {
          logger.log(Level.FINE, "Extracted native library '" + library + "' from resource path");
        }
      }
    } finally {
      cleanUp(tempFile, !systemLoad);
    }
  }

  /**
   * Extracts the providers array from the C API, converts it into an EnumSet.
   *
   * <p>Throws IllegalArgumentException if a provider isn't recognised (note this exception should
   * only happen during development of ONNX Runtime, if it happens at any other point, file an issue
   * on Github).
   *
   * @param ortApiHandle The API Handle.
   * @return The enum set.
   */
  private static EnumSet<OrtProvider> initialiseProviders(long ortApiHandle) {
    String[] providersArray = getAvailableProviders(ortApiHandle);

    EnumSet<OrtProvider> providers = EnumSet.noneOf(OrtProvider.class);

    for (String provider : providersArray) {
      providers.add(OrtProvider.mapFromName(provider));
    }

    return providers;
  }

  /**
   * Get a reference to the API struct.
   *
   * @param apiVersionNumber The API version to use.
   * @return A pointer to the API struct.
   */
  private static native long initialiseAPIBase(int apiVersionNumber);

  /**
   * Gets the array of available providers.
   *
   * @param ortApiHandle The API handle
   * @return The array of providers
   */
  private static native String[] getAvailableProviders(long ortApiHandle);
}

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
import java.nio.file.Paths;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
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
  // Post 1.7 builds of the ORT API
  private static final int ORT_API_VERSION_8 = 8;

  /**
   * The name of the system property which when set gives the path on disk where the ONNX Runtime
   * native libraries are stored.
   */
  static final String ONNXRUNTIME_NATIVE_PATH = "onnxruntime.native.path";

  /** The short name of the ONNX runtime shared library */
  static final String ONNXRUNTIME_LIBRARY_NAME = "onnxruntime";
  /** The short name of the ONNX runtime JNI shared library */
  static final String ONNXRUNTIME_JNI_LIBRARY_NAME = "onnxruntime4j_jni";

  /** The short name of the ONNX runtime shared provider library */
  static final String ONNXRUNTIME_LIBRARY_SHARED_NAME = "onnxruntime_providers_shared";
  /** The short name of the ONNX runtime CUDA provider library */
  static final String ONNXRUNTIME_LIBRARY_CUDA_NAME = "onnxruntime_providers_cuda";
  /** The short name of the ONNX runtime DNNL provider library */
  static final String ONNXRUNTIME_LIBRARY_DNNL_NAME = "onnxruntime_providers_dnnl";
  /** The short name of the ONNX runtime OpenVINO provider library */
  static final String ONNXRUNTIME_LIBRARY_OPENVINO_NAME = "onnxruntime_providers_openvino";
  /** The short name of the ONNX runtime TensorRT provider library */
  static final String ONNXRUNTIME_LIBRARY_TENSORRT_NAME = "onnxruntime_providers_tensorrt";

  /** The OS & CPU architecture string */
  private static final String OS_ARCH_STR = initOsArch();

  /** Have the core ONNX Runtime native libraries been loaded */
  private static boolean loaded = false;

  /** The temp directory where native libraries are extracted */
  private static Path tempDirectory;

  /** The value of the {@link #ONNXRUNTIME_NATIVE_PATH} system property */
  private static String libraryDirPathProperty;

  /** Tracks if the shared providers have been extracted */
  private static final Set<String> extractedSharedProviders = new HashSet<>();

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
    tempDirectory = isAndroid() ? null : Files.createTempDirectory("onnxruntime-java");
    try {
      libraryDirPathProperty = System.getProperty(ONNXRUNTIME_NATIVE_PATH);
      // Extract and prepare the shared provider library but don't try to load it,
      // the ONNX Runtime native library will load it
      extractProviderLibrary(ONNXRUNTIME_LIBRARY_SHARED_NAME);

      load(ONNXRUNTIME_LIBRARY_NAME);
      load(ONNXRUNTIME_JNI_LIBRARY_NAME);
      ortApiHandle = initialiseAPIBase(ORT_API_VERSION_8);
      providers = initialiseProviders(ortApiHandle);
      loaded = true;
    } finally {
      if (tempDirectory != null) {
        cleanUp(tempDirectory.toFile());
      }
    }
  }

  /**
   * Marks the file for delete on exit.
   *
   * @param file The file to remove.
   */
  private static void cleanUp(File file) {
    if (!file.exists()) {
      return;
    }
    logger.log(Level.FINE, "Deleting " + file + " on exit");
    file.deleteOnExit();
  }

  /**
   * Extracts the CUDA provider library from the classpath resources if present, or checks to see if
   * the CUDA provider library is in the directory specified by {@link #ONNXRUNTIME_NATIVE_PATH}.
   *
   * @return True if the CUDA provider library is ready for loading, false otherwise.
   */
  static boolean extractCUDA() {
    return extractProviderLibrary(ONNXRUNTIME_LIBRARY_CUDA_NAME);
  }

  /**
   * Extracts the DNNL provider library from the classpath resources if present, or checks to see if
   * the DNNL provider library is in the directory specified by {@link #ONNXRUNTIME_NATIVE_PATH}.
   *
   * @return True if the DNNL provider library is ready for loading, false otherwise.
   */
  static boolean extractDNNL() {
    return extractProviderLibrary(ONNXRUNTIME_LIBRARY_DNNL_NAME);
  }

  /**
   * Extracts the OpenVINO provider library from the classpath resources if present, or checks to
   * see if the OpenVINO provider library is in the directory specified by {@link
   * #ONNXRUNTIME_NATIVE_PATH}.
   *
   * @return True if the OpenVINO provider library is ready for loading, false otherwise.
   */
  static boolean extractOpenVINO() {
    return extractProviderLibrary(ONNXRUNTIME_LIBRARY_OPENVINO_NAME);
  }

  /**
   * Extracts the TensorRT provider library from the classpath resources if present, or checks to
   * see if the TensorRT provider library is in the directory specified by {@link
   * #ONNXRUNTIME_NATIVE_PATH}.
   *
   * @return True if the TensorRT provider library is ready for loading, false otherwise.
   */
  static boolean extractTensorRT() {
    return extractProviderLibrary(ONNXRUNTIME_LIBRARY_TENSORRT_NAME);
  }

  /**
   * Extracts a shared provider library from the classpath resources if present, or checks to see if
   * that library is in the directory specified by {@link #ONNXRUNTIME_NATIVE_PATH}.
   *
   * @param libraryName The shared provider library to load.
   * @return True if the library is ready for loading by ORT's native code, false otherwise.
   */
  static synchronized boolean extractProviderLibrary(String libraryName) {
    // Check if we've already extracted or check this provider, and it's ready
    if (extractedSharedProviders.contains(libraryName)) {
      return true;
    }
    // Otherwise extract the file from the classpath resources
    Optional<File> file = extractFromResources(libraryName);
    if (file.isPresent()) {
      extractedSharedProviders.add(libraryName);
      return true;
    } else {
      // If we failed to extract it, check if there is a valid cache directory
      // that contains it
      if (libraryDirPathProperty != null) {
        String libraryFileName = mapLibraryName(libraryName);
        File libraryFile = Paths.get(libraryDirPathProperty, libraryFileName).toFile();
        if (libraryFile.exists()) {
          extractedSharedProviders.add(libraryName);
          return true;
        } else {
          return false;
        }
      } else {
        return false;
      }
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
   * <p>If the library path is not specified via a system property then it attempts to extract the
   * library from the classpath before loading it.
   *
   * @param library The bare name of the library.
   * @throws IOException If the file failed to read or write.
   */
  private static void load(String library) throws IOException {
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
    String libraryFileName = mapLibraryName(library);

    // 2) The user may explicitly specify the path to a directory containing all shared libraries:
    if (libraryDirPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryDirPathProperty);
      // TODO: Switch this to Path.of when the minimum Java version is 11.
      File libraryFile = Paths.get(libraryDirPathProperty, libraryFileName).toFile();
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
    Optional<File> extractedPath = extractFromResources(library);
    if (extractedPath.isPresent()) {
      // extracted library from resources
      System.load(extractedPath.get().getAbsolutePath());
      logger.log(Level.FINE, "Loaded native library '" + library + "' from resource path");
    } else {
      // failed to load library from resources, try to load it from the library path
      logger.log(
          Level.FINE, "Attempting to load native library '" + library + "' from library path");
      System.loadLibrary(library);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from library path");
    }
  }

  /**
   * Extracts the library from the classpath resources. returns optional.empty if it failed to
   * extract or couldn't be found.
   *
   * @param library The library name
   * @return An optional containing the file if it is successfully extracted, or an empty optional
   *     if it failed to extract or couldn't be found.
   */
  private static Optional<File> extractFromResources(String library) {
    String libraryFileName = mapLibraryName(library);
    String resourcePath = "/ai/onnxruntime/native/" + OS_ARCH_STR + '/' + libraryFileName;
    File tempFile = tempDirectory.resolve(libraryFileName).toFile();
    try (InputStream is = OnnxRuntime.class.getResourceAsStream(resourcePath)) {
      if (is == null) {
        // Not found in classpath resources
        return Optional.empty();
      } else {
        // Found in classpath resources, load via temporary file
        logger.log(
            Level.FINE,
            "Attempting to load native library '"
                + library
                + "' from resource path "
                + resourcePath
                + " copying to "
                + tempFile);
        byte[] buffer = new byte[4096];
        int readBytes;
        try (FileOutputStream os = new FileOutputStream(tempFile)) {
          while ((readBytes = is.read(buffer)) != -1) {
            os.write(buffer, 0, readBytes);
          }
        }
        logger.log(Level.FINE, "Extracted native library '" + library + "' from resource path");
        return Optional.of(tempFile);
      }
    } catch (IOException e) {
      logger.log(
          Level.WARNING, "Failed to extract library '" + library + "' from the resources", e);
      return Optional.empty();
    } finally {
      cleanUp(tempFile);
    }
  }

  /**
   * Maps the library name into a platform dependent library filename. Converts macOS's "jnilib" to
   * "dylib" but otherwise is the same as {@link System#mapLibraryName(String)}.
   *
   * @param library The library name
   * @return The library filename.
   */
  private static String mapLibraryName(String library) {
    return System.mapLibraryName(library).replace("jnilib", "dylib");
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

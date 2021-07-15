/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;

/** Util code for interacting with Java arrays. */
public final class OrtUtil {

  /** Private constructor for static util class. */
  private OrtUtil() {}

  /**
   * Converts an long shape into a int shape.
   *
   * <p>Validates that the shape has more than 1 elements, less than 9 elements, each element is
   * less than {@link Integer#MAX_VALUE} and that each entry is non-negative.
   *
   * @param shape The long shape.
   * @return The int shape.
   */
  public static int[] transformShape(long[] shape) {
    if (shape.length == 0 || shape.length > TensorInfo.MAX_DIMENSIONS) {
      throw new IllegalArgumentException(
          "Arrays with less than 1 and greater than "
              + TensorInfo.MAX_DIMENSIONS
              + " dimensions are not supported.");
    }
    int[] newShape = new int[shape.length];
    for (int i = 0; i < shape.length; i++) {
      long curDim = shape[i];
      if (curDim < 1 || curDim > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "Invalid shape for a Java array, expected positive entries smaller than Integer.MAX_VALUE. Found "
                + Arrays.toString(shape));
      } else {
        newShape[i] = (int) curDim;
      }
    }
    return newShape;
  }

  /**
   * Converts an int shape into a long shape.
   *
   * <p>Validates that the shape has more than 1 element, less than 9 elements and that each entry
   * is non-negative.
   *
   * @param shape The int shape.
   * @return The long shape.
   */
  public static long[] transformShape(int[] shape) {
    if (shape.length == 0 || shape.length > 8) {
      throw new IllegalArgumentException(
          "Arrays with less than 1 and greater than "
              + TensorInfo.MAX_DIMENSIONS
              + " dimensions are not supported.");
    }
    long[] newShape = new long[shape.length];
    for (int i = 0; i < shape.length; i++) {
      long curDim = shape[i];
      if (curDim < 1) {
        throw new IllegalArgumentException(
            "Invalid shape for a Java array, expected positive entries smaller than Integer.MAX_VALUE. Found "
                + Arrays.toString(shape));
      } else {
        newShape[i] = curDim;
      }
    }
    return newShape;
  }

  /**
   * Creates a new primitive boolean array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A boolean array.
   */
  public static Object newBooleanArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(boolean.class, intShape);
  }

  /**
   * Creates a new primitive byte array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A byte array.
   */
  public static Object newByteArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(byte.class, intShape);
  }

  /**
   * Creates a new primitive short array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A short array.
   */
  public static Object newShortArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(short.class, intShape);
  }

  /**
   * Creates a new primitive int array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A int array.
   */
  public static Object newIntArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(int.class, intShape);
  }

  /**
   * Creates a new primitive long array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A long array.
   */
  public static Object newLongArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(long.class, intShape);
  }

  /**
   * Creates a new primitive float array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A float array.
   */
  public static Object newFloatArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(float.class, intShape);
  }

  /**
   * Creates a new primitive double array of up to 8 dimensions, using the supplied shape.
   *
   * <p>
   *
   * @param shape The shape of array to create.
   * @return A double array.
   */
  public static Object newDoubleArray(long[] shape) {
    int[] intShape = transformShape(shape);
    return Array.newInstance(double.class, intShape);
  }

  /**
   * Reshapes a boolean array into the desired n-dimensional array assuming the boolean array is
   * stored in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number
   * of elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The boolean array.
   * @param shape The desired shape.
   * @return An n-dimensional boolean array.
   */
  public static Object reshape(boolean[] input, long[] shape) {
    Object output = OrtUtil.newBooleanArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a byte array into the desired n-dimensional array assuming the byte array is stored in
   * n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The byte array.
   * @param shape The desired shape.
   * @return An n-dimensional byte array.
   */
  public static Object reshape(byte[] input, long[] shape) {
    Object output = OrtUtil.newByteArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a short array into the desired n-dimensional array assuming the short array is stored
   * in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The short array.
   * @param shape The desired shape.
   * @return An n-dimensional short array.
   */
  public static Object reshape(short[] input, long[] shape) {
    Object output = OrtUtil.newShortArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes an int array into the desired n-dimensional array, assuming the int array is stored in
   * n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The int array.
   * @param shape The desired shape.
   * @return An n-dimensional int array.
   */
  public static Object reshape(int[] input, long[] shape) {
    Object output = OrtUtil.newIntArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a long array into the desired n-dimensional array, assuming the long array is stored
   * in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The long array.
   * @param shape The desired shape.
   * @return An n-dimensional long array.
   */
  public static Object reshape(long[] input, long[] shape) {
    Object output = OrtUtil.newLongArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a float array into the desired n-dimensional array assuming the float array is stored
   * in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number of
   * elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The float array.
   * @param shape The desired shape.
   * @return An n-dimensional float array.
   */
  public static Object reshape(float[] input, long[] shape) {
    Object output = OrtUtil.newFloatArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Reshapes a double array into the desired n-dimensional array assuming the double array is
   * stored in n-dimensional row-major order. Throws {@link IllegalArgumentException} if the number
   * of elements doesn't match between the shape and the input or the shape is invalid.
   *
   * @param input The double array.
   * @param shape The desired shape.
   * @return An n-dimensional double array.
   */
  public static Object reshape(double[] input, long[] shape) {
    Object output = OrtUtil.newDoubleArray(shape);
    reshape(input, output, 0);
    return output;
  }

  /**
   * Copies elements from the flat input array to the appropriate primitive array of the output.
   * Recursively calls itself as it traverses the output array.
   *
   * @param input The input array.
   * @param output The output multidimensional array.
   * @param position The current position in the input array.
   * @return The new position in the input array.
   */
  private static int reshape(Object input, Object output, int position) {
    if (output.getClass().isArray()) {
      Object[] outputArray = (Object[]) output;
      for (Object outputElement : outputArray) {
        Class<?> outputElementClass = outputElement.getClass();
        if (outputElementClass.isArray()) {
          if (outputElementClass.getComponentType().isPrimitive()) {
            int length = Array.getLength(outputElement);
            System.arraycopy(input, position, outputElement, 0, length);
            position += length;
          } else {
            position = reshape(input, outputElement, position);
          }
        } else {
          throw new IllegalStateException(
              "Found element type when expecting an array. Class " + outputElementClass);
        }
      }
    } else {
      throw new IllegalStateException(
          "Found element type when expecting an array. Class " + output.getClass());
    }

    return position;
  }

  /**
   * Counts the number of elements stored in a Tensor of this shape.
   *
   * <p>Multiplies all the elements together if they are positive, throws an {@link
   * IllegalArgumentException} otherwise.
   *
   * @param shape The shape to use.
   * @return The number of elements.
   */
  public static long elementCount(long[] shape) {
    long count = 1;
    for (int i = 0; i < shape.length; i++) {
      if (shape[i] > 0) {
        count *= shape[i];
      } else {
        throw new IllegalArgumentException(
            "Received non-positive value in shape " + Arrays.toString(shape) + " .");
      }
    }
    return count;
  }

  /**
   * Checks that the shape is a valid shape for a Java array (i.e. that the values are all positive
   * and representable by an int).
   *
   * @param shape The shape to check.
   * @return True if the shape is valid.
   */
  public static boolean validateShape(long[] shape) {
    boolean valid = true;
    for (int i = 0; i < shape.length; i++) {
      valid &= shape[i] > 0;
      valid &= ((int) shape[i]) == shape[i];
    }
    return valid && shape.length <= TensorInfo.MAX_DIMENSIONS;
  }

  /**
   * Flatten a multidimensional String array into a single dimensional String array, reading it in a
   * multidimensional row-major order.
   *
   * @param o A multidimensional String array.
   * @return A single dimensional String array.
   */
  public static String[] flattenString(Object o) {
    ArrayList<String> output = new ArrayList<>();

    flattenString((Object[]) o, output);

    return output.toArray(new String[0]);
  }

  /**
   * Flattens a multidimensional String array into the ArrayList.
   *
   * @param input The multidimensional String array.
   * @param output The output ArrayList.
   */
  private static void flattenString(Object[] input, ArrayList<String> output) {
    for (Object i : input) {
      Class<?> iClazz = i.getClass();
      if (iClazz.isArray()) {
        if (iClazz.getComponentType().isArray()) {
          flattenString((Object[]) i, output);
        } else if (iClazz.getComponentType().equals(String.class)) {
          output.addAll(Arrays.asList((String[]) i));
        } else {
          throw new IllegalStateException("Found a non-String, non-array element type, " + iClazz);
        }
      } else {
        throw new IllegalStateException(
            "Found an element type where there should have been an array. Class = " + iClazz);
      }
    }
  }

  /**
   * Stores a boxed primitive in a single element array of the unboxed type.
   *
   * <p>If it's not a boxed primitive then it returns null.
   *
   * @param javaType The type of the boxed primitive.
   * @param data The boxed primitive.
   * @return The primitive in an array.
   */
  static Object convertBoxedPrimitiveToArray(OnnxJavaType javaType, Object data) {
    switch (javaType) {
      case FLOAT:
        float[] floatArr = new float[1];
        floatArr[0] = (Float) data;
        return floatArr;
      case DOUBLE:
        double[] doubleArr = new double[1];
        doubleArr[0] = (Double) data;
        return doubleArr;
      case UINT8:
      case INT8:
        byte[] byteArr = new byte[1];
        byteArr[0] = (Byte) data;
        return byteArr;
      case INT16:
        short[] shortArr = new short[1];
        shortArr[0] = (Short) data;
        return shortArr;
      case INT32:
        int[] intArr = new int[1];
        intArr[0] = (Integer) data;
        return intArr;
      case INT64:
        long[] longArr = new long[1];
        longArr[0] = (Long) data;
        return longArr;
      case BOOL:
        boolean[] booleanArr = new boolean[1];
        booleanArr[0] = (Boolean) data;
        return booleanArr;
      case STRING:
      case UNKNOWN:
      default:
        return null;
    }
  }
}

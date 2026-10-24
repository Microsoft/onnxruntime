/*
 * A minimal, complete ONNX Runtime C API program: it loads a model, feeds it one float32
 * input tensor, runs inference and prints the output tensor.
 *
 * The model must have exactly one float32 input and one float32 output. See
 * docs/c_cxx/C_API_Quickstart.md for how to build this program and for a model to try it on.
 */

#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

#include "onnxruntime_c_api.h"

/* Every C API call that can fail returns an OrtStatus*. A NULL status means success, anything
   else is an error that owns its message and has to be released with ReleaseStatus. */
#define ORT_CHECK(call)                                                                    \
  do {                                                                                     \
    status = (call);                                                                       \
    if (status != NULL) {                                                                  \
      fprintf(stderr, "%s failed: %s (error code %d)\n", #call, api->GetErrorMessage(status), \
              (int)api->GetErrorCode(status));                                             \
      api->ReleaseStatus(status);                                                          \
      goto cleanup;                                                                        \
    }                                                                                      \
  } while (0)

/* Prints "label 'name' has shape [1, 2]". */
static void print_shape(const char* label, const char* name, const int64_t* dims, size_t count) {
  printf("%s '%s' has shape [", label, name);
  for (size_t i = 0; i < count; ++i) {
    printf("%s%lld", i == 0 ? "" : ", ", (long long)dims[i]);
  }
  printf("]\n");
}

/* Returns the number of elements of a tensor with the given shape. */
static size_t element_count(const int64_t* dims, size_t count) {
  size_t elements = 1;
  for (size_t i = 0; i < count; ++i) {
    elements *= (size_t)dims[i];
  }
  return elements;
}

#ifdef _WIN32
/* The C API takes a model path as ORTCHAR_T, which is wchar_t on Windows. A narrow argv
   cannot be handed to CreateSession there, so on Windows the program starts from wmain and
   receives the path in its native wide form. */
int wmain(int argc, wchar_t** argv) {
#else
int main(int argc, char** argv) {
#endif
  if (argc != 2) {
#ifdef _WIN32
    fwprintf(stderr, L"usage: %ls <model.onnx>\n", argv[0]);
#else
    fprintf(stderr, "usage: %s <model.onnx>\n", argv[0]);
#endif
    return EXIT_FAILURE;
  }
  const ORTCHAR_T* model_path = argv[1];

  /* The C API is a table of function pointers. GetApi returns the table that matches the
     version of the header this file was compiled against, and NULL when the loaded library is
     older than that header. */
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (api == NULL) {
    fprintf(stderr, "the loaded ONNX Runtime does not support API version %d\n", ORT_API_VERSION);
    return EXIT_FAILURE;
  }

  OrtEnv* env = NULL;
  OrtSession* session = NULL;
  OrtAllocator* allocator = NULL;
  OrtTypeInfo* type_info = NULL;
  const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
  OrtMemoryInfo* memory_info = NULL;
  OrtValue* input = NULL;
  OrtValue* output = NULL;
  OrtTensorTypeAndShapeInfo* output_tensor_info = NULL;
  char* input_name = NULL;
  char* output_name = NULL;
  int64_t* input_dims = NULL;
  int64_t* output_dims = NULL;
  float* input_data = NULL;
  float* output_data = NULL;
  const char* input_names[1];
  const OrtValue* input_values[1];
  const char* output_names[1];
  size_t input_count = 0;
  size_t output_count = 0;
  size_t dim_count = 0;
  size_t output_dim_count = 0;
  size_t elements = 0;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  ONNXTensorElementDataType output_element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  OrtStatus* status = NULL;
  int result = EXIT_FAILURE;

  /* CreateEnv: the environment owns the logging and the thread pools. The log id only tags the
     log messages. */
  ORT_CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "quickstart", &env));

  /* CreateSession: loads the model from disk. NULL session options select the defaults, that
     is the CPU execution provider and the default graph optimization level. */
  ORT_CHECK(api->CreateSession(env, model_path, NULL, &session));

  /* GetAllocatorWithDefaultOptions: the default allocator is owned by ONNX Runtime and must not
     be released. The strings returned by the API are allocated with it, so they are freed with
     the same allocator. */
  ORT_CHECK(api->GetAllocatorWithDefaultOptions(&allocator));

  /* SessionGetInputCount and SessionGetOutputCount: how many inputs and outputs does the model
     have? SessionGetInputName and SessionGetOutputName: the names it gives them. The names have
     to be passed to Run unchanged. */
  ORT_CHECK(api->SessionGetInputCount(session, &input_count));
  ORT_CHECK(api->SessionGetOutputCount(session, &output_count));
  if (input_count != 1 || output_count != 1) {
    fprintf(stderr, "this example needs a model with exactly one input and one output\n");
    goto cleanup;
  }
  ORT_CHECK(api->SessionGetInputName(session, 0, allocator, &input_name));
  ORT_CHECK(api->SessionGetOutputName(session, 0, allocator, &output_name));

  /* Ask the session what the input looks like: its element type and its shape. The tensor info
     belongs to the type info, so it is not released separately. */
  ORT_CHECK(api->SessionGetInputTypeInfo(session, 0, &type_info));
  ORT_CHECK(api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
  ORT_CHECK(api->GetTensorElementType(tensor_info, &element_type));
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    fprintf(stderr, "this example only supports float32 inputs, the model uses element type %d\n",
            (int)element_type);
    goto cleanup;
  }

  /* GetDimensionsCount and then GetDimensions: the shape of the input tensor. The caller
     allocates the array that GetDimensions writes into. */
  ORT_CHECK(api->GetDimensionsCount(tensor_info, &dim_count));
  input_dims = (int64_t*)calloc(dim_count + 1, sizeof(int64_t));
  if (input_dims == NULL) {
    fprintf(stderr, "out of memory\n");
    goto cleanup;
  }
  ORT_CHECK(api->GetDimensions(tensor_info, input_dims, dim_count));

  /* GetDimensions reports a symbolic dimension as -1. A concrete size has to be chosen for it
     before the buffer is sized, because a negative dimension turns into a huge element count
     and the allocation then fails for the wrong reason. This example resolves every symbolic
     dimension to 1, which is a valid batch size for the models it supports. */
  for (size_t i = 0; i < dim_count; ++i) {
    if (input_dims[i] < 0) {
      input_dims[i] = 1;
    }
  }
  print_shape("model input", input_name, input_dims, dim_count);

  /* Fill the input buffer. CreateTensorWithDataAsOrtValue does not copy the data, so this
     buffer has to stay alive until Run has returned. */
  elements = element_count(input_dims, dim_count);
  input_data = (float*)calloc(elements, sizeof(float));
  if (input_data == NULL) {
    fprintf(stderr, "out of memory\n");
    goto cleanup;
  }
  for (size_t i = 0; i < elements; ++i) {
    input_data[i] = (float)(i + 1);
  }

  /* CreateCpuMemoryInfo: describes where the input tensor lives. */
  ORT_CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

  /* CreateTensorWithDataAsOrtValue: wraps the buffer in an OrtValue. */
  ORT_CHECK(api->CreateTensorWithDataAsOrtValue(memory_info, input_data, elements * sizeof(float),
                                                 input_dims, dim_count,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input));

  /* Run: the arrays of names and of values must have the same length, and the names must match
     the names used by the model. NULL run options mean "use the defaults". */
  input_names[0] = input_name;
  input_values[0] = input;
  output_names[0] = output_name;
  ORT_CHECK(api->Run(session, NULL, input_names, input_values, 1, output_names, 1, &output));

  /* The output tensor is owned by the OrtValue that Run filled in. */
  ORT_CHECK(api->GetTensorTypeAndShape(output, &output_tensor_info));
  ORT_CHECK(api->GetTensorElementType(output_tensor_info, &output_element_type));
  if (output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    fprintf(stderr, "this example only supports float32 outputs, the model produces element type %d\n",
            (int)output_element_type);
    goto cleanup;
  }
  ORT_CHECK(api->GetDimensionsCount(output_tensor_info, &output_dim_count));
  output_dims = (int64_t*)calloc(output_dim_count + 1, sizeof(int64_t));
  if (output_dims == NULL) {
    fprintf(stderr, "out of memory\n");
    goto cleanup;
  }
  ORT_CHECK(api->GetDimensions(output_tensor_info, output_dims, output_dim_count));

  /* The output of a finished run has concrete dimensions. Guard against a symbolic one anyway,
     so that the element count below cannot be computed from a negative number. */
  for (size_t i = 0; i < output_dim_count; ++i) {
    if (output_dims[i] < 0) {
      fprintf(stderr, "the output tensor reports a symbolic dimension, which this example does not handle\n");
      goto cleanup;
    }
  }
  print_shape("output", output_name, output_dims, output_dim_count);

  ORT_CHECK(api->GetTensorMutableData(output, (void**)&output_data));
  printf("output values:");
  for (size_t i = 0; i < element_count(output_dims, output_dim_count); ++i) {
    printf(" %g", (double)output_data[i]);
  }
  printf("\n");

  result = EXIT_SUCCESS;

cleanup:
  /* Release everything in the reverse order of creation. All the release functions accept NULL,
     so this is safe even after an early failure. */
  api->ReleaseTensorTypeAndShapeInfo(output_tensor_info);
  api->ReleaseValue(output);
  api->ReleaseValue(input);
  api->ReleaseMemoryInfo(memory_info);
  api->ReleaseTypeInfo(type_info);
  api->ReleaseSession(session);
  api->ReleaseEnv(env);
  if (allocator != NULL) {
    allocator->Free(allocator, output_name);
    allocator->Free(allocator, input_name);
  }
  free(output_dims);
  free(input_dims);
  free(input_data);
  return result;
}

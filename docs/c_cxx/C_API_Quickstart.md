# ONNX Runtime C API quickstart

A starting point for calling ONNX Runtime from C. This page walks through one complete program
that creates an environment, loads a model, asks the model what it expects, feeds it an input
tensor, runs inference and prints the result, and it explains what every argument of every call
means.

The C API is the native interface of ONNX Runtime: the C#, Java, JavaScript and Python bindings
are all built on top of it, so a C program can use every model, every execution provider and
every feature of the runtime.

For everything that is not covered here, see the generated C API reference at
<https://onnxruntime.ai/docs/api/c/>.

## Getting the library and the header

A C program needs two things:

- the shared library: `libonnxruntime.so`, `libonnxruntime.dylib` or `onnxruntime.dll`;
- the public header `onnxruntime_c_api.h`.

Both come with the prebuilt packages on the
[release page](https://github.com/microsoft/onnxruntime/releases). Unpack an archive and the
library is in `lib/` and the headers are in `include/`, so the header is
`include/onnxruntime_c_api.h`. In this repository the same header is
[`include/onnxruntime/core/session/onnxruntime_c_api.h`](../../include/onnxruntime/core/session/onnxruntime_c_api.h).

Compile and link against them with:

```sh
cc -std=c11 -I <onnxruntime include dir> c_api_quickstart.c -L <onnxruntime lib dir> \
   -lonnxruntime -o quickstart
```

On Windows with MSVC:

```bat
cl /I <onnxruntime include dir> c_api_quickstart.c /link <onnxruntime lib dir>\onnxruntime.lib
```

The C API takes a model path as `ORTCHAR_T`, which is `wchar_t` on Windows, so the program
declares `wmain` there and receives the path in its native wide form. MSVC links `wmain` as the
entry point automatically; a MinGW build has to pass `-municode` so that the wide entry point is
used.

Use the header that comes with the library. The header declares an API version number, and
`GetApi` returns NULL when the loaded library is older than that header, so a header from a
newer release than the library fails with "the requested API version is not available".

## A complete program

The program below loads a model, feeds it one float32 input tensor and prints the output. It is
also checked in as [`c_api_quickstart.c`](c_api_quickstart.c); the two copies are identical.

The model has to have exactly one float32 input and one float32 output. One to try the program
with can be created with the `onnx` package:

```python
import onnx
from onnx import helper, TensorProto

x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
node = helper.make_node("Add", ["input", "input"], ["output"])
graph = helper.make_graph([node], "add", [x], [y])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
onnx.save(model, "add.onnx")
```

`ir_version` is pinned on purpose: the default of the `onnx` package is often newer than the
IR version the installed runtime accepts, and loading the model then fails with "Unsupported
model IR version".

```c
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
```

Build and run it:

```sh
./quickstart add.onnx
```

The shared library has to be found when the program starts, not only when it is linked. The `-L`
above tells the linker where to look, but the loader still has to be pointed at the same
directory, otherwise the program fails before `main` is reached:

| Platform | What the loader needs |
| --- | --- |
| Linux | `LD_LIBRARY_PATH=<onnxruntime lib dir> ./quickstart add.onnx` |
| macOS | `DYLD_LIBRARY_PATH=<onnxruntime lib dir> ./quickstart add.onnx` |
| Windows | the directory that holds `onnxruntime.dll` on `PATH`, or the DLL next to `quickstart.exe` |

On Linux and macOS an rpath (`-Wl,-rpath,<onnxruntime lib dir>`) is an alternative to setting
the environment variable for every run.

which prints:

```
model input 'input' has shape [1, 2]
output 'output' has shape [1, 2]
output values: 2 4
```

## What the calls do

| Call | Arguments |
| --- | --- |
| `OrtGetApiBase()->GetApi(ORT_API_VERSION)` | `ORT_API_VERSION` is the API version the header declares. Returns the table of function pointers, or NULL when the loaded library does not support that version. |
| `CreateEnv(log_severity_level, logid, &env)` | The minimum severity of the messages that are logged, a free-form string that tags the log messages, and the output parameter for the new environment. |
| `CreateSession(env, model_path, options, &session)` | The environment, the path of the `.onnx` file as an `ORTCHAR_T` string, the session options (NULL selects the defaults) and the output parameter for the new session. |
| `GetAllocatorWithDefaultOptions(&allocator)` | The default CPU allocator. It belongs to ONNX Runtime and must not be released. |
| `SessionGetInputCount(session, &count)` | The number of inputs of the model. |
| `SessionGetInputName(session, index, allocator, &name)` | The name of the input at `index`. The string is allocated with `allocator` and has to be freed with it. |
| `SessionGetOutputCount(session, &count)` | The number of outputs of the model. |
| `SessionGetOutputName(session, index, allocator, &name)` | The name of the output at `index`, allocated with `allocator`. |
| `SessionGetInputTypeInfo(session, index, &type_info)` | The type of the input at `index`. Has to be released with `ReleaseTypeInfo`. |
| `CastTypeInfoToTensorInfo(type_info, &tensor_info)` | The tensor view of a type. It belongs to `type_info` and is not released separately. |
| `GetTensorElementType(tensor_info, &type)` | The element type, for example `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT`. |
| `GetDimensionsCount(tensor_info, &count)` | The number of dimensions of the tensor. |
| `GetDimensions(tensor_info, dims, count)` | The `count` dimensions of the tensor, written into the `dims` array that the caller allocates. A symbolic dimension is reported as `-1`; the program replaces it with `1` before it sizes the buffer. |
| `CreateCpuMemoryInfo(type, mem_type, &info)` | Describes a memory region. `OrtArenaAllocator` with `OrtMemTypeDefault` is ordinary CPU memory. |
| `CreateTensorWithDataAsOrtValue(info, data, len, shape, shape_len, type, &value)` | Wraps the `len` bytes at `data` in a tensor of shape `shape` and element type `type`. The data is not copied. |
| `CreateTensorWithDataAndDeleterAsOrtValue(allocator, data, len, shape, shape_len, type, deleter, &value)` | The same, but the tensor takes ownership of the buffer: `deleter` is called with `allocator` when the `OrtValue` is released. |
| `Run(session, run_options, input_names, inputs, input_len, output_names, output_names_len, outputs)` | Runs the model. The names are the ones the model uses, the arrays of names and of values have the same length, `run_options` may be NULL and the results are written into `outputs`. |
| `GetTensorTypeAndShape(value, &info)` | The shape and the element type of the tensor in `value`. |
| `GetTensorMutableData(value, &data)` | The address of the tensor data. The memory belongs to the `OrtValue`. |
| `ReleaseStatus(status)` | Frees the `OrtStatus` that a failed call returned. |

## Error handling

Every call that can fail returns an `OrtStatus*`. A NULL status means success, anything else is
an error that has to be released with `ReleaseStatus`. `GetErrorMessage(status)` returns the
message and `GetErrorCode(status)` the numeric code, for example `ORT_INVALID_ARGUMENT`. The
message is owned by the status, so read it before releasing the status.

The `ORT_CHECK` macro in the program does exactly that: it prints the message and the code and
then jumps to the cleanup code.

## Memory ownership

- Every `Create*` call returns an object that the caller owns and has to release with the
  matching `Release*` call: `ReleaseEnv`, `ReleaseSession`, `ReleaseValue`, `ReleaseMemoryInfo`,
  `ReleaseTypeInfo` or `ReleaseTensorTypeAndShapeInfo`.
- The strings returned by `SessionGetInputName` and `SessionGetOutputName` belong to the
  allocator that was passed in, so they are freed with `allocator->Free(allocator, name)`.
- `CreateTensorWithDataAsOrtValue` does not copy the data, so the buffer has to stay alive until
  `Run` has returned. The caller keeps ownership of it and frees it when it is done.
- `CreateTensorWithDataAndDeleterAsOrtValue` also does not copy the data, but it *transfers*
  ownership: it takes an `OrtAllocator*` and a deleter of the form
  `void (*)(OrtAllocator* allocator, void* data)`, and it calls that deleter with the allocator
  when the `OrtValue` is released. Use it when the buffer should be freed with the runtime's
  allocator rather than with `free`. It cannot make a buffer that expires before the `OrtValue`
  safe: the deleter runs when the `OrtValue` is released, so the storage has to live at least
  that long.
- The output tensor that `Run` produces is owned by the `OrtValue`; its memory is released with
  `ReleaseValue`.
- Release the objects in the reverse order of their creation: the values before the session and
  the session before the environment.

## Where to go next

- The generated C API reference: <https://onnxruntime.ai/docs/api/c/>.
- [`docs/C_API_Guidelines.md`](../C_API_Guidelines.md) explains how the C API is designed and
  what a new API is expected to look like.
- [`docs/c_cxx/README.md`](README.md) describes how the reference documentation is generated.

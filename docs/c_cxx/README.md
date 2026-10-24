# ONNX Runtime C/C++ docs source files

This directory contains doxygen configuration to generate the C/C++ API docs for ONNX Runtime.

The actual generation is performed by a GitHub actions workflow: [publish-c-apidocs.yml](../../.github/workflows/publish-c-apidocs.yml).

## User documentation

- [C API quickstart](C_API_Quickstart.md): a short starting point guide for calling ONNX Runtime
  from C, with a complete, copy-pasteable example program and an explanation of the arguments of
  every call it makes.

# C/C++ API Documentation Conventions

## Handling API changes across versions
When a new API is added or an existing API is changed, indicate this in the API's documentation with the Doxygen `\since` command.
This lets readers know the availability and behavior in different versions.

For a new API in version X, add `\since Version X.`. If an API's behavior changes in version Y, add `\since Version Y: <change description>`.
Put each API change entry for a version on its own line.

For example:
```c++
/**
 * \brief My API function.
 * \since Version X.
 * \since Version Y: Does things differently.
 */
void MyApi();
```

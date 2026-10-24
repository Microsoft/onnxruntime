# Compiled Model Encryption

## Overview

This document describes how an application can generate an *encrypted* compiled model and later load
it for inference, with **all encryption/decryption performed by the application**. ONNX Runtime (ORT)
and the execution providers (EPs) never write plaintext to disk on the application's behalf and never
hold the encryption keys.

The piece that makes this possible for **EPContext binary data** — the compiled kernels/engines an EP
emits when embed mode is disabled (e.g. a serialized TensorRT engine or a QNN context binary) — is a
pair of application-supplied **named-buffer I/O callbacks** plus a small EP-facing configuration
handle. That mechanism, together with a reference helper for EP authors, was added in
[microsoft/onnxruntime#28624](https://github.com/microsoft/onnxruntime/pull/28624) and refined in
[microsoft/onnxruntime#29294](https://github.com/microsoft/onnxruntime/pull/29294).

The model-level and initializer-level hooks that this feature builds on (the compiled-model write
callback, the per-initializer location callback, in-memory external initializers, and EPContext embed
mode) are **pre-existing** ORT compile/session APIs and are described here only for context.

> **API status:** The callback transport is part of the stable C and C++ API starting in ONNX Runtime 1.31.
> Applications should use the `OrtApi`, `OrtCompileApi`, `OrtEpApi`, and `Ort::EpContextConfig` members described below.

## Background

A model is composed of the following "model assets":

- ONNX model file
- External initializer files referenced from the ONNX model
- EPContext node binary data (embedded in the model, or stored in external files, for compiled models)

ORT already supports model compilation via the compilation API (`OrtCompileApi`). These pre-existing
APIs let an application:

- Provide an input model from a buffer (`ModelCompilationOptions_SetInputModelFromBuffer`), allowing
  the application to decrypt the model before handing it to ORT.
- Stream out compiled ONNX model bytes via a write callback
  (`ModelCompilationOptions_SetOutputModelWriteFunc`), enabling the application to encrypt the output.
- Control per-initializer storage via a callback
  (`ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc`), enabling the application to
  encrypt and store initializers externally.

For loading and inference, ORT lets an application inject external initializer files from an in-memory
buffer (`AddExternalInitializersFromFilesInMemory`), so the application can decrypt initializer files
and hand the bytes to the session.

### Current Gaps (addressed by #28624 / #29294)

- **Compilation:** there was no hook to intercept EPContext binary data writes when embed mode is
  disabled — the EP wrote the context binary directly to a file, defeating application-side encryption.
- **Load:** there was no way to supply decrypted EPContext binary data to the EP without writing
  plaintext to disk and without double-buffering. Before this feature the EP loaded its EPContext
  binary by opening the external file itself (e.g. `std::ifstream` on the path stored in the
  `ep_cache_context` attribute). For an encrypted model that forced two undesirable outcomes:
  - **Plaintext on disk.** Because the EP reads from a file path, the application had to first decrypt
    the binary to a temporary plaintext file for the EP to open — directly violating the
    "no unencrypted data on disk" property this feature exists to preserve. Even a short-lived temp
    file exposes the decrypted engine/context binary to other processes and to crash/interruption
    scenarios, and adds extra I/O.
  - **Double-buffering (2× peak memory).** Even if the application decrypted into memory instead, there
    was no API to hand that buffer to the EP. The EP would allocate its own buffer and read the bytes
    in, so the decrypted payload existed twice at once — once in the application's buffer and once in
    the EP's copy. For EPContext binaries that can be hundreds of MB to several GB (TensorRT engines,
    QNN / Ryzen AI context binaries), that doubled peak is significant.

  The load-side read callback closes both gaps: the application allocates the final buffer once via the
  ORT-provided allocator and returns the decrypted bytes directly to the EP — no temporary plaintext
  file and no second copy (1× peak). See [Named-Buffer I/O Callbacks](#named-buffer-io-callbacks) and
  [Memory Usage Analysis](#memory-usage-analysis).

## Requirements

- **No unencrypted data on disk.** All encryption/decryption happens in memory, in the application;
  ORT and the EPs never persist plaintext on the application's behalf.
- **Minimize peak memory.** Avoid double-buffering (app buffer + ORT copy). Prefer a read callback
  that lets the application allocate the output directly via an ORT-provided allocator (1× peak).
- **Reuse existing API patterns.** A write callback for the compile side and a read callback (with an
  ORT-provided allocator) for the load side. Keep the callbacks generic ("named buffer") so the
  contract can be reused for other named payloads.
- **Minimize EP / plugin API changes.** Keep `Compile()` unchanged and add one optional, append-only `OrtEp` support
  query. Encryption is application-controlled, not ORT- or EP-controlled.
- **Backward compatible.** Applications that register no callbacks behave exactly as before. A registered callback is
  a strict requirement for a participating external-EPContext operation; an EP that does not advertise the required
  READ or WRITE support is rejected before `Compile()` instead of silently writing or reading a fallback file.

## Assumptions

- Uncompiled model assets are originally encrypted.
- The uncompiled model may have a mix of embedded initializers and initializers stored in external
  (encrypted) files.
- The application knows the location of all model assets (both uncompiled and compiled).
- Just-in-time (JIT) compilation is used, with a requirement that no unencrypted assets are written to
  disk during compilation.
- The original uncompiled model is encrypted and must be decrypted during JIT compilation.
- Compiled models may have EPContext nodes with embedded binary data or binary data stored in
  encrypted external files.

## Security and Support Boundary

- ORT does not encrypt, decrypt, authenticate, or manage keys. The application controls those operations inside its
  callbacks.
- The callbacks cover only bytes routed through them. They do not prevent an EP compiler, device driver, or third-party
  library from creating its own temporary files. An EP that cannot meet an application's no-plaintext-on-disk policy
  must reject that configuration with a clear error.
- The sample filesystem fallback performs containment and file-type checks, but those checks are advisory under
  concurrent filesystem mutation (TOCTOU). Applications that require stronger guarantees should use callbacks backed
  by a sandboxed store and platform-native secure-open semantics.
- The stable API transports callbacks and state, not payload-size policy. The application is responsible for
  imposing allocation limits in its callback; the EP is responsible for validating the returned data and imposing
  deserialization limits before consumption. ORT does not invoke the callback or inspect its result. The sample
  helper accepts an EP-selected `max_data_size`, but that parameter is not part of the stable ABI. Its tests
  demonstrate helper enforcement, not an ORT-provided size guarantee.

## Approaches Considered

### Where encryption lives

**A. Application-controlled I/O callbacks (chosen).** The application supplies read/write callbacks;
ORT and the EPs route EPContext binary data through them and never handle keys or plaintext on disk.

- Pros: keys and crypto never enter ORT; the application can encrypt, compress, or redirect to cloud
  storage; no crypto dependency is added to ORT.
- Cons: the application must implement and register the callbacks; the contract (allocator ownership,
  state lifetime and thread-safety) must be documented carefully.

**B. ORT- or EP-managed encryption.** ORT (or each EP) encrypts/decrypts using a key handed to it.

- Pros: less application code for the common case.
- Cons: ORT/EPs would own key material and a crypto implementation (large surface, policy and
  compliance burden) and would be inflexible for cloud/compression use cases. Rejected.

**C. Filesystem / OS-level encryption.** Rely on an encrypted volume.

- Pros: no ORT changes.
- Cons: does not satisfy "no unencrypted data on disk" for JIT flows (plaintext transits the
  filesystem) and is outside the application's per-asset control. Rejected.

### How the EP obtains the callbacks

**A. Getter-based `OrtEpContextConfig` + reference helper (chosen).** The application registers
callbacks on the session/compile options; the EP extracts an opaque `OrtEpContextConfig` in
`CreateEp()` and pulls the callback/state via `OrtEpApi` accessors. The "callback if present, else
file I/O" logic and untrusted-name hardening live in the sample `ep_context_data_utils` helper.

- Pros: zero `OrtEp` struct changes; the opaque handle can gain fields later without breaking EPs; the
  fallback/validation policy can evolve in sample code without an ABI commitment.
- Cons: EPs must copy/adapt the helper (it is not ABI); slightly more EP code than a single ORT call.

**B. ORT-implemented `OrtEpApi::Read/WriteEpContextData` utility (proposed, not shipped).** ORT owns a
function that performs the callback-or-disk fallback internally.

- Pros: a single call for the EP; consistent fallback behavior across EPs.
- Cons: bakes path-resolution and fallback policy into the ABI (hard to iterate on) and grows the
  stable surface. Deferred in favor of a sample helper — see [Rejected Alternatives](#rejected-alternatives).

**C. Change the `OrtEp` struct / `Compile()` signature.** Pass the callbacks/config directly to the EP.

- Pros: explicit.
- Cons: breaks the EP ABI and every existing EP. Rejected.

### Read-side memory model

**A. Read callback allocates via the ORT allocator; `EpContextData` adopts it (chosen).** Peak memory
is 1× — only the final buffer exists.

**B. Callback returns its own buffer and ORT copies into a `std::vector`.** Simpler ownership, but 2×
peak for potentially multi-GB engine/context binaries. Rejected: `EpContextData` already provides
convenient zero-copy access, so no separate copy-into-`std::vector` reader is kept.

### EPContext storage

**Embed (`embed = true`)** keeps EPContext bytes inside the model, encrypted with the model bytes — no
EPContext-specific callback is needed; this is simplest when the model stays under the 2 GB protobuf
limit. **External (`embed = false`)** stores the bytes in separate files and is the case that requires
the write/read callbacks. Both are supported; embed is recommended first when size permits.

## Named-Buffer I/O Callbacks

The callbacks are intentionally generic ("named buffer") so the same contract can be reused for other
named payloads in the future; today they are used for EPContext binary data. Each invocation is
synchronous, and ORT does not serialize invocations made by different EP instances or worker threads.

### `OrtWriteNamedBufferFunc`

Called during compilation to write named binary data. `name` identifies which EPContext binary is
being written so the application can choose a storage location. The application's implementation can
process the data in any way (e.g. encrypt and store, upload to cloud storage, or compress) before
persisting it.

Each invocation represents one complete write for `name`; the signature carries no offset or
final-chunk marker, so any chunked ordering/completion contract must be defined by the invoking EP.
Current EPContext use should prefer a single invocation per EPContext binary unless the EP documents
chunking semantics.

```c
typedef OrtStatus*(ORT_API_CALL* OrtWriteNamedBufferFunc)(_In_ void* state,
                                                          _In_ const char* name,
                                                          _In_ const void* buffer,
                                                          _In_ size_t buffer_num_bytes);
```

### `OrtReadNamedBufferFunc`

Called during session load to obtain named binary data. ORT provides an allocator so the application
can allocate the output buffer directly, avoiding double-buffering. The application reads, processes
(e.g. decrypts, decompresses, downloads), allocates via the provided allocator, fills the buffer, and
returns. Peak memory for the payload is 1× (only the final buffer exists).

```c
typedef OrtStatus*(ORT_API_CALL* OrtReadNamedBufferFunc)(_In_ void* state,
                                                         _In_ const char* name,
                                                         _In_ OrtAllocator* allocator,
                                                         _Outptr_ void** buffer,
                                                         _Out_ size_t* data_size);
```

Both callbacks return `OrtStatus*` (nullptr on success; use `CreateStatus` on failure). The `state`
pointer is opaque and owned by the application, which must keep it valid — and synchronize it if it
can be used concurrently — for as long as an EP might invoke the callback.

## Registering the Callbacks (Application Side)

### Read callback — `OrtApi::SessionOptionsSetEpContextDataReadFunc`

Registers the read callback on the session options. Reading happens at session load, so this is
configured on `OrtSessionOptions`. Passing `NULL` for `read_func` clears any previously set callback
(and its state).

```c
OrtStatus* status = api->SessionOptionsSetEpContextDataReadFunc(options, read_func, state);
```

### Write callback — `OrtCompileApi::ModelCompilationOptions_SetEpContextDataWriteFunc`

Registers the write callback used during compilation when embed mode is disabled. Writing happens only
at compile time, so this is configured on `OrtModelCompilationOptions`. It may be used together with
`ModelCompilationOptions_SetEpContextBinaryInformation`, whose binary information still describes the
compiled-model/output location EPs use to generate stable logical names or as a file-fallback
location. Passing `NULL` for `write_func` clears any previously set callback (and its state).

```c
OrtStatus* ModelCompilationOptions_SetEpContextDataWriteFunc(
  OrtModelCompilationOptions* model_compile_options,
  OrtWriteNamedBufferFunc write_func,
  void* state);
```

## EP-Facing: `OrtEpContextConfig` and Accessors

### `OrtEpContextConfig` (opaque handle)

An opaque immutable snapshot that holds ORT's copy of the EPContext callback function pointers and their opaque
state pointers extracted from an `OrtSessionOptions` instance. Later changes to the session options affect
only subsequently created configs and sessions. The handle **does not** own the application-provided state — the
application remains responsible for keeping that state valid and synchronized. The EP creates the handle during
`CreateEp()` (while the session options are still valid) and releases it in its destructor.

> The originally-proposed *typed EP-context generation-option accessors* (embed mode, file path, node
> name prefix, weightless flag, etc.) are **not** part of this implementation. `OrtEpContextConfig`
> currently carries only the I/O callbacks and their state. See
> [Open Questions](#open-questions).

### Extract / release — `OrtEpApi::SessionOptionsGetEpContextConfig` / `OrtEpApi::ReleaseEpContextConfig`

```c
// Extract the EPContext configuration (callbacks + state) from session options. On success *config is
// a non-NULL handle that must be released with OrtEpApi::ReleaseEpContextConfig; on failure *config is
// left unmodified. Call during CreateEp() while session_options is valid; store the handle for Compile().
OrtStatus* SessionOptionsGetEpContextConfig(const OrtSessionOptions* session_options,
                                            OrtEpContextConfig** config);

// Release the handle. May be NULL.
void ReleaseEpContextConfig(OrtEpContextConfig* config);
```

### Retrieve callbacks — `OrtEpApi::EpContextConfigGetEpContextData{Read,Write}Func`

The EP pulls the registered callback (and its state) out of the config. If none was registered,
`*func` and `*state` are set to `NULL`, and the EP should use its own normal disk read/write path.

```c
OrtStatus* EpContextConfigGetEpContextDataReadFunc(const OrtEpContextConfig* config,
                                                   OrtReadNamedBufferFunc* read_func,
                                                   void** state);

OrtStatus* EpContextConfigGetEpContextDataWriteFunc(const OrtEpContextConfig* config,
                                                    OrtWriteNamedBufferFunc* write_func,
                                                    void** state);
```

### C++ convenience wrapper — `Ort::EpContextConfig`

A move-only RAII wrapper (in `onnxruntime_cxx_api.h`) that owns the handle and exposes the
callback accessors. Typical EP usage: construct from the session options during `CreateEp()`, keep the
wrapper for the EP's lifetime, and query the callbacks via `GetReadFunc()` / `GetWriteFunc()`.

```cpp
namespace Ort {
class EpContextConfig : public detail::Base<OrtEpContextConfig> {
 public:
  explicit EpContextConfig(std::nullptr_t) noexcept;
  explicit EpContextConfig(const SessionOptions& session_options);
  explicit EpContextConfig(ConstSessionOptions session_options);   // extracts via GetEpContextConfig

  void GetReadFunc(OrtReadNamedBufferFunc& read_func, void*& state) const;
  void GetWriteFunc(OrtWriteNamedBufferFunc& write_func, void*& state) const;
};
}  // namespace Ort
```

The C++ session setter registers the callback and its state:

```cpp
session_options.SetEpContextDataReadFunc(read_func, state);
```

## Reference Helper: `ep_context_data_utils` (sample, not ABI)

Location: `onnxruntime/test/autoep/library/ep_context_data_utils.h`. This is **sample/reference code**
for EP authors, shared by the example plugin EP and its tests. It is intentionally **outside the ORT C
and EP ABI**; EPs are expected to copy and adapt it. It wraps the "use the callback if present,
otherwise fall back to file I/O" pattern and hardens untrusted, model-derived names. Production EPs
should additionally apply their own sandboxing, size limits, and path policies.

### `EpContextData` — zero-copy owning buffer (#29294)

A move-only RAII owner for the bytes returned by a read. On the callback path it **adopts** the
allocator-provided buffer directly (no copy) and frees it via the same allocator on destruction; on the
file-fallback path it reads straight from disk into a buffer allocated from the same (optionally
caller-supplied) allocator. Either way the bytes are accessed through `data()` / `size()` without an
extra copy, and the allocator (which is not owned) must outlive the object. Ownership is transferred
into the object **only on success**, so it stays empty on any error path.

```cpp
class EpContextData {
 public:
  const char* data() const noexcept;   // valid until destroyed/reassigned; may be null only when empty
  size_t size() const noexcept;
  bool empty() const noexcept;
  // move-only; frees the adopted allocator buffer (if any) on destruction
};
```

### Read with file fallback

```cpp
// Zero-copy: if the config carries a read callback it is invoked with the (optionally caller-supplied)
// allocator and the returned buffer is adopted by `out`; otherwise the file is read into an allocator-
// backed buffer (resolved against the model dir). `allocator` is optional (nullptr = ORT default) and is
// not owned by `out`: it must outlive `out`.
OrtStatus* ReadEpContextData(const OrtApi& api, const OrtEpContextConfig* config,
                             const char* file_name, const OrtGraph* graph, EpContextData& out,
                             OrtAllocator* allocator = nullptr);
```

### Write with file fallback

```cpp
// If the config carries a write callback it is invoked with `file_name`; otherwise the bytes are
// written to a file. An overload takes a separate `fallback_file_name` for the on-disk path.
OrtStatus* WriteEpContextDataWithFileFallback(const OrtApi& api, const OrtEpContextConfig* config,
                                              const char* file_name, const OrtGraph* graph,
                                              const void* buffer, size_t buffer_size);
```

### Untrusted name handling (#28624 + #29294)

The helper distinguishes trust based on the `OrtGraph*`:

- **Logical (callback-namespace) names** are opaque keys owned by the application's I/O callbacks: they
  are passed through unmodified and are **not** validated as filesystem paths. Path rules are enforced
  only where a name is actually used as a path (below), not at authoring time.
- **Model-derived file names** (`graph != nullptr`, i.e. read from an EPContext node's
  `ep_cache_context` attribute) are joined with the model directory, canonicalized with
  `std::filesystem::weakly_canonical` (resolving `.` / `..` and symlinks), and accepted **only if the
  resolved path stays within the model directory** (`IsResolvedPathWithinBase`). A benign `a/b/../file`
  is accepted; anything escaping the model directory — including via a symlink — is rejected.
- **Trusted callers** (`graph == nullptr`) own the path and may pass an absolute path or one containing
  `..`; there is no model directory to contain against, so no traversal check is applied there.
- **File type** is checked where the path is actually used, not during resolution: a read target must
  exist and be a regular file (`EnsureRegularFileForRead`), and a write target must be a regular file
  *if* it already exists (`EnsureRegularFileForWrite`). Both use `std::filesystem::status`, which
  follows symlinks and so is consistent with the canonicalization above. This rejects directories and
  special files — notably a FIFO left inside the model directory, which passes containment and would
  block an `ifstream` open (a cheap DoS). The check is advisory, not a security boundary: the path can
  change between the check and the open (TOCTOU); a race-free variant would inspect the opened handle.
- **A symlink write target is rejected** (`symlink_status`, which inspects the link itself rather than
  following it). This covers what containment cannot: a *dangling* symlink has no target for
  `weakly_canonical` to resolve, so the link path survives resolution, stays lexically inside the model
  directory and passes the containment check — yet an `ofstream` would follow it and create the file
  wherever it points. A `status`-based test cannot catch it either, because the missing target reports
  as `not_found`, the same result as the ordinary "new file" case that writes must allow. Note the
  division of labor: a model-relative symlink that *does* resolve is replaced by its target during
  resolution, so containment is what constrains those writes; this check covers the dangling case and
  trusted (`graph == nullptr`) paths, which skip resolution altogether.

## Application Flows

### Flow 1: JIT Compilation (Encrypt Compiled Model)

```text
┌─────────────────────────────────────────────────────────────────┐
│ Application                                                       │
│                                                                   │
│  1. Create OrtSessionOptions                                      │
│  2. Decrypt external initializer files into buffers               │
│  3. Inject via AddExternalInitializersFromFilesInMemory           │
│  4. Create OrtModelCompilationOptions from session options        │
│  5. Decrypt ONNX model into buffer                                │
│  6. ModelCompilationOptions_SetInputModelFromBuffer(buffer)       │
│  7. ModelCompilationOptions_SetOutputModelWriteFunc(encrypt_cb)   │
│  8. ModelCompilationOptions_SetOutputModelGetInitializerLocation… │
│  9. ModelCompilationOptions_SetEpContextEmbedMode(true|false)     │
│ 10. [If embed=false]                                              │
│     ModelCompilationOptions_SetEpContextDataWriteFunc(write_cb)   │
│ 11. CompileModel(...)                                             │
│                                                                   │
│  During CompileModel:                                             │
│  ├─ ORT calls the model write callback with compiled ONNX chunks  │
│  ├─ ORT calls the initializer-location callback per initializer   │
│  └─ [If embed=false]                                              │
│     ├─ EP extracts OrtEpContextConfig in CreateEp()               │
│     ├─ ORT calls ep->Compile(...)                                 │
│     │  Inside EP's Compile():                                     │
│     │  ├─ EP generates compiled binary (e.g., a TRT engine)       │
│     │  ├─ EP calls ep_context_data_utils::WriteEpContextData…     │
│     │  │     (api, config, name, graph, data, size)               │
│     │  └─ helper pulls write_cb from config and invokes it        │
│     │     → app encrypts and stores the data                      │
│                                                                   │
│  After CompileModel: all assets are encrypted on disk/storage     │
└───────────────────────────────────────────────────────────────────┘
```

#### Step-by-step

1. Create `OrtSessionOptions` and configure the desired execution providers.
2. Decrypt external initializer files into memory buffers.
3. Inject initializers via `OrtApi::AddExternalInitializersFromFilesInMemory` (copies buffers). Release
   app-side buffers after this call.
4. Create `OrtModelCompilationOptions` from the session options via
   `OrtCompileApi::CreateModelCompilationOptionsFromSessionOptions`.
5. Decrypt the uncompiled ONNX model into a memory buffer.
6. Set the input model buffer via `OrtCompileApi::ModelCompilationOptions_SetInputModelFromBuffer`.
7. Set the model write callback via `OrtCompileApi::ModelCompilationOptions_SetOutputModelWriteFunc`.
   ORT calls it with chunks of compiled ONNX bytes; the app encrypts and writes them.
8. Set the initializer location callback via
   `OrtCompileApi::ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc`. Per initializer,
   the app can:
   - Embed it in the ONNX model (return `new_external_info = NULL`); it is encrypted with the model
     bytes via the model write callback.
   - Store it externally: encrypt and write the initializer, then return an `OrtExternalInitializerInfo`
     pointing to the file.

   > Note: ONNX files cannot exceed 2 GB due to protobuf limitations. Large initializers should be
   > stored externally.
9. Set the EPContext embed mode via `OrtCompileApi::ModelCompilationOptions_SetEpContextEmbedMode`.
   - `embed = true`: EPContext binary data is stored inside the `ep_cache_context` attribute of
     EPContext nodes and is encrypted with the model bytes. No additional API is needed.
   - `embed = false`: EPContext binary data is stored in external files. Proceed to step 10.
10. _(If `embed = false`)_ Register the write callback **(new in #28624)** via
    `OrtCompileApi::ModelCompilationOptions_SetEpContextDataWriteFunc`. ORT stores it in the session
    options; when the EP calls `OrtEpApi::SessionOptionsGetEpContextConfig` during `CreateEp()`, the
    write callback is carried by the returned `OrtEpContextConfig`. During `Compile()`, when the EP
    produces compiled binary data (e.g. a serialized TensorRT engine), it writes it through the config
    (directly via `OrtEpApi::EpContextConfigGetEpContextDataWriteFunc`, or via the
    `ep_context_data_utils::WriteEpContextDataWithFileFallback` helper) instead of writing to disk. The
    `name` parameter lets the application distinguish multiple EPContext binaries (one per compiled
    subgraph).
11. Compile the model via `OrtCompileApi::CompileModel`. After this call, all compiled model assets are
    generated and encrypted.

### Flow 2: Load, Decrypt, and Run Inference

```text
┌─────────────────────────────────────────────────────────────────┐
│ Application                                                       │
│                                                                   │
│  1. Create OrtSessionOptions                                      │
│  2. Decrypt external initializer files into buffers               │
│  3. Inject via AddExternalInitializersFromFilesInMemory           │
│  4. [If embed=false]                                              │
│     SessionOptions_SetEpContextDataReadFunc(decrypt_read_cb)      │
│  5. Decrypt compiled ONNX model into buffer                       │
│  6. CreateSessionFromArray(buffer, ...)                           │
│                                                                   │
│  During session initialization:                                   │
│  ├─ EP extracts OrtEpContextConfig in CreateEp()                  │
│  │   → EP stores the handle (e.g. Ort::EpContextConfig)           │
│  ├─ ORT calls ep->Compile(...)                                    │
│  │  Inside EP's Compile():                                        │
│  │  ├─ EP hits an EPContext node with an external file reference  │
│  │  ├─ EP calls ep_context_data_utils::ReadEpContextData(         │
│  │  │     api, config, name, graph, ep_context_data)              │
│  │  ├─ helper pulls read_cb from config and invokes it:           │
│  │  │   read_cb(state, name, allocator, &buf, &size)              │
│  │  │   → app reads, decrypts, allocates via ORT allocator        │
│  │  └─ EP deserializes from ep_context_data.data()/size()         │
│  │                                                                │
│  7. Run inference                                                 │
└───────────────────────────────────────────────────────────────────┘
```

#### Step-by-step

1. Create `OrtSessionOptions` and configure the desired execution providers.
2. Decrypt external initializer files into memory buffers.
3. Inject initializers via `OrtApi::AddExternalInitializersFromFilesInMemory` (copies buffers). Release
   app-side buffers after this call.
4. _(If `embed = false`)_ Register the read callback **(new in #28624)** via
   `OrtApi::SessionOptions_SetEpContextDataReadFunc`. When invoked, the callback receives an
   ORT-provided allocator; the application:
   1. Reads the source data (e.g. from encrypted storage, cloud, a compressed file).
   2. Processes it as needed (e.g. decrypts, decompresses).
   3. Allocates a buffer using the ORT allocator.
   4. Writes the output into the buffer and returns it.

   > This eliminates double-buffering. Peak memory for EPContext data is 1× (only the final buffer
   > exists).
5. Decrypt the compiled ONNX model into a memory buffer.
6. Create the session via `OrtApi::CreateSessionFromArray` with the decrypted model buffer. During
   session initialization, for each EP:
  - The EP calls `OrtEpApi::SessionOptionsGetEpContextConfig` during `CreateEp()` (while session
     options are still valid) and stores the returned `OrtEpContextConfig` handle (typically wrapped in
    `Ort::EpContextConfig`).
   - ORT calls `ep->Compile(...)`.
   - When the EP encounters an EPContext node with an external file reference, it reads the data through
     the config — via `ep_context_data_utils::ReadEpContextData(api, config, name, graph, out)` (or by
    pulling the callback with `OrtEpApi::EpContextConfigGetEpContextDataReadFunc`). If no read callback
     is present, the helper falls back to reading the file from the model directory.
   - The EP releases the config (RAII wrapper does this in its destructor).
7. Run inference as usual.

## Memory Usage Analysis

| Asset | Compilation Peak | Load Peak |
| --- | --- | --- |
| ONNX model | 1× (app decrypts into buffer, streams out via write callback) | 1× (app decrypts into buffer for `CreateSessionFromArray`) |
| External initializers | 2× during inject (`AddExternalInitializersFromFilesInMemory` copies) | 2× during inject (same API, same copy behavior) |
| EPContext binary (embed=true) | Included in ONNX model bytes | Included in ONNX model bytes |
| EPContext binary (embed=false) | Streaming (near 0×) via write callback | **1×** via read callback (app allocates with the ORT allocator; `EpContextData` adopts it) |

### Notes on Memory

- External initializers still use the existing `AddExternalInitializersFromFilesInMemory` API, which
  copies buffers (2× peak). A future optimization could add a read-callback variant for initializers,
  matching the EPContext pattern.
- EPContext binary data can be large (hundreds of MB to GBs for TensorRT engines, QNN context
  binaries). The read-callback + `EpContextData` approach is specifically designed to avoid
  double-buffering here.
- Embed mode (`embed = true`) is the simplest path: all binary data flows through the existing model
  write/read callbacks with no EPContext-specific callback needed. Consider it first if the total model
  size (including embedded EPContext data) stays within the 2 GB protobuf limit.

## EP Impact

Plugin EPs implement the `OrtEp` struct and access ORT functionality through `OrtEpApi`.
`OrtEp::Compile()` remains unchanged. The stable feature adds the optional, append-only
`OrtEp::GetEpContextDataCallbackSupport` query and uses:

- `OrtEpApi::SessionOptionsGetEpContextConfig` / `OrtEpApi::ReleaseEpContextConfig` — obtain and release
  the config handle.
- `OrtEpApi::EpContextConfigGetEpContextDataReadFunc` / `...GetEpContextDataWriteFunc` — retrieve the
  application's callback and state, or `NULL` if none was registered.
- `OrtEp::GetEpContextDataCallbackSupport` — advertise READ and/or WRITE only after the corresponding external-data path
  honors the callback and propagates callback errors without filesystem fallback.
- The `ep_context_data_utils` reference helper (sample code) — encapsulates the callback-or-file
  fallback and the untrusted-name hardening.

### Adoption by EPs

- **OpenVINO:** snapshots session callbacks and supports native external EPContext import and export. Embedded
  contexts bypass the callbacks. Callback errors propagate without filesystem fallback, and import buffers are
  released after deserialization. Native binary metadata and all blob ranges are checked against the supplied stream
  length before allocating their storage. Total native-blob storage is also bounded by that length, preventing
  overlapping entries from amplifying allocations. Each distinct external data name is read once per provider
  initialization. Models containing multiple native EPContext nodes import each cached partition separately.
  Callbacks are incompatible with shared EP contexts; callback-backed binaries with
  external shared-weight metadata are rejected before inference. External OVIR (XML plus weights) remains a
  filesystem-only format and is explicitly rejected when a read callback is registered.
- **New EPs:** extract the `OrtEpContextConfig` in `CreateEp()` and store it (e.g. as a
  `Ort::EpContextConfig` member). During `Compile()`, read/write EPContext binaries
  through the config (directly via the getters, or via the reference helper, which handles the
  callback-vs-disk fallback transparently). Enforce EP-specific limits before deserialization, advertise the implemented
  flags through `GetEpContextDataCallbackSupport`, and release the config in the destructor.
- **Old EPs / no callback registered:** with no callback, EPs continue reading/writing files directly. If a callback
  is required for external EPContext data, an old or non-advertising EP is rejected before accessing that data,
  including access during capability discovery rather than only during `Compile()`.

### Example: TensorRT-style EP (Compilation — Writing)

```cpp
// In the EP's Compile(): write the serialized engine, honoring an app write callback if present.
nvinfer1::IHostMemory* serialized = trt_engine->serialize();

// ep_context_config_ is an Ort::EpContextConfig obtained in CreateEp().
RETURN_IF_ERROR(ep_context_data_utils::WriteEpContextDataWithFileFallback(
    ort_api, ep_context_config_,
    engine_cache_name,   // logical name / file-fallback name
    graph,               // model graph, for file-fallback path resolution
    serialized->data(), serialized->size()));
// No std::ofstream needed — the helper invokes the write callback or writes to disk.
```

### Example: TensorRT-style EP (Load — Reading)

```cpp
// In the EP's Compile(): read a cached engine, honoring an app read callback if present.
ep_context_data_utils::EpContextData ctx;
RETURN_IF_ERROR(ep_context_data_utils::ReadEpContextData(
    ort_api, ep_context_config_, ep_cache_context_name, graph, ctx));

engine = runtime->deserializeCudaEngine(ctx.data(), ctx.size());
// `ctx` frees the adopted allocator buffer on destruction — no manual free.
```

Both examples replace the EP's existing `std::ofstream` / `std::ifstream` code with a single helper
call that handles both the custom (callback) and standard (disk) cases.

### Backward Compatibility

This is a backward-compatible, append-only C API addition. `OrtEp::GetEpContextDataCallbackSupport` is appended to the
EP-implemented struct and is called only for an EP compiled against API version 31 or later; `OrtEp::Compile()` is
unchanged. Applications that register no callbacks behave exactly as before.

### Provider Adoption

The stable transport and strict support negotiation are exercised end-to-end by the example plugin EP. Built-in EPs
adopt the transport separately because each provider owns its external-context format, sharing behavior, and
filesystem optimizations. A provider must route only non-embedded EPContext data through the callbacks, reject
incompatible modes before filesystem I/O, and add hardware-backed tests before advertising support.

The in-tree CUDA and WebGPU plugin EPs do not currently emit EPContext models. TensorRT has an EPContext
implementation but still requires provider-owned callback integration, support advertisement, and hardware-backed
tests before it can claim support.

An EP should claim support only after its tests prove callback write/read, callback-error propagation without disk
fallback, no extra payload copy on read, legacy disk behavior with no callback, and embed-mode bypass.

Built-in EPs report provider-specific EPContext data callback requirements through the side-effect-free
`IExecutionProvider::GetEpContextDataCallbackRequirements()` query. ORT checks the registered callbacks against these
requirements before each `GetCapability()` call, including AOT inlining, NHWC passes, and ORT-format loading — this
same check covers EPs, such as VitisAI, that can produce external EPContext data as a side effect of
`GetCapability()` itself, before `Compile()` is ever called. VitisAI has a direct-assignment path through
`GetComputeCapabilityOps()` and compiles its backend model in `GetCapability()`; it reports WRITE from
`GetEpContextDataCallbackRequirements()` whenever its effective configuration would produce external context data.
Merely moving context-node creation into its `Compile()` would skip direct-assignment-only sessions; changing that
lifecycle requires coordinated VAIP changes. TensorRT reports WRITE when its effective provider options select
external context output, even if ORT's compilation options request embedded output. QNN reports READ before
capability-time loading of external contexts for VTCM backup buffer sharing or file-mapped weights. Embedded data
and sessions without the corresponding callback do not require that callback's support.

## API Summary

Stable callback typedefs (in `onnxruntime_c_api.h`):

| API | Location | Purpose |
| --- | --- | --- |
| `OrtWriteNamedBufferFunc` | Callback typedef | Write named (EPContext) binary data during compilation |
| `OrtReadNamedBufferFunc` | Callback typedef | Read/process named data, allocate via the ORT allocator, return the buffer, during load |
| `OrtEpContextDataCallbackSupportFlags` | Support flags | Advertise callback-capable external EPContext read/write paths |

Stable functions (since ONNX Runtime 1.31):

| API | Group | Purpose |
| --- | --- | --- |
| `SessionOptionsSetEpContextDataReadFunc` | `OrtApi` | Register the read callback on session options (load) |
| `ModelCompilationOptions_SetEpContextDataWriteFunc` | `OrtCompileApi` | Register compile-time write callback |
| `SessionOptionsGetEpContextConfig` | `OrtEpApi` | Extract config during `CreateEp()` |
| `ReleaseEpContextConfig` | `OrtEpApi` | Release the `OrtEpContextConfig` handle |
| `EpContextConfigGetEpContextDataReadFunc` | `OrtEpApi` | EP retrieves the read callback + state (or `NULL`) |
| `EpContextConfigGetEpContextDataWriteFunc` | `OrtEpApi` | EP retrieves the write callback + state (or `NULL`) |
| `OrtEpContextConfig` | Opaque type | Snapshots callbacks and state pointers from session options |
| `GetEpContextDataCallbackSupport` | `OrtEp` | Advertise strict READ/WRITE callback support |

C++ wrapper and reference helper:

| Symbol | Location | Purpose |
| --- | --- | --- |
| `Ort::EpContextConfig` | `onnxruntime_cxx_api.h` | RAII config wrapper with callback getters |
| `ep_context_data_utils::EpContextData` | `test/autoep/library/ep_context_data_utils.h` (sample) | Zero-copy owning buffer for a read result |
| `ep_context_data_utils::ReadEpContextData` | sample | Read via callback or file fallback into an owning `EpContextData` |
| `ep_context_data_utils::WriteEpContextDataWithFileFallback` | sample | Write via callback or file fallback |

Pre-existing APIs used (no changes needed):

| API | Location | Purpose |
| --- | --- | --- |
| `AddExternalInitializersFromFilesInMemory` | `OrtApi` | Inject decrypted external initializers |
| `ModelCompilationOptions_SetInputModelFromBuffer` | `OrtCompileApi` | Compile from a decrypted model buffer |
| `ModelCompilationOptions_SetOutputModelWriteFunc` | `OrtCompileApi` | Stream-encrypt compiled ONNX model bytes |
| `ModelCompilationOptions_SetOutputModelGetInitializerLocationFunc` | `OrtCompileApi` | Per-initializer embed/external decision + encryption |
| `ModelCompilationOptions_SetEpContextEmbedMode` | `OrtCompileApi` | Control EPContext binary embedding |
| `ModelCompilationOptions_SetEpContextBinaryInformation` | `OrtCompileApi` | Describe the compiled-model/output location for stable names / file fallback |
| `CreateSessionFromArray` | `OrtApi` | Create a session from a decrypted model buffer |

## Internal Wiring

This section is illustrative implementation detail — applications and EPs see only the public APIs
above.

- **Application → session options.** `SetEpContextDataReadFunc` (on `OrtSessionOptions`) and
  `SetEpContextDataWriteFunc` (on `OrtModelCompilationOptions`, which forwards into the underlying
  session options) store the callback function pointer and opaque state.
- **Session options → `OrtEpContextConfig`.** `OrtEpApi::SessionOptionsGetEpContextConfig` allocates a
  config handle that snapshots those callback pointers and state values (it does not own the
  application state).
- **`OrtEpContextConfig` → EP.** The EP obtains the handle during `CreateEp()` and keeps it. During
  `Compile()`, the getters return the stored callback/state; when none was registered, they return
  `NULL` and the EP (or the reference helper) falls back to disk I/O. The EP releases the handle in its
  destructor.

## Rejected Alternatives

- **Stable payload-size policy.** Registering a limit without ORT invoking the callback would add permanent API
  surface without runtime enforcement and require duplicated application configuration. Centralized invocation
  could check the returned size, but not prevent an oversized allocation within the callback without additional
  allocation constraints. Passing a limit to the callback avoids duplicate configuration but still relies on
  callback/EP enforcement. Instead, allocation and deserialization limits remain application/EP responsibilities;
  the sample helper may enforce an EP-selected limit without making it part of the ABI.
- **ORT-implemented `OrtEpApi::Read/WriteEpContextData` utility.** The original proposal had ORT expose
  ABI functions that performed the callback-or-disk fallback and path resolution internally. This was
  not shipped: it would bake fallback and path-resolution policy into the (experimental/stable) ABI,
  making the untrusted-name hardening hard to iterate on. Instead, EPs retrieve the raw callback via
  the config getters and use the sample `ep_context_data_utils` helper (copyable, non-ABI) for the
  fallback and validation logic.
- **Typed EP-context generation-option accessors on `OrtEpContextConfig`.** The proposal also floated
  typed accessors for EP-context generation options (`ep.context_enable`, `ep.context_embed_mode`,
  `ep.context_file_path`, etc.) that EPs read today as string key/value pairs via
  `GetSessionConfigEntry`. These PRs implement only the I/O callbacks + state on the config; typed
  accessors remain a possible future extension (see [Open Questions](#open-questions)).
- **ORT- or EP-managed encryption** and **filesystem-level encryption** — see
  [Approaches Considered](#approaches-considered).

## Open Questions

1. **Streaming decryption for external initializers.** `AddExternalInitializersFromFilesInMemory`
   copies buffers (2× peak). Should a read-callback variant be added for initializers to match the
   EPContext pattern?
2. **Streaming decryption for the ONNX model itself.** A `CreateSessionFromReadFunc`-style API could let
   the application stream-decrypt the ONNX model during session creation, avoiding holding the entire
   decrypted model in memory. Protobuf supports parsing from a stream, so this is feasible.
3. **Multiple EPContext binaries per compilation.** The write callback includes a `name` parameter to
   distinguish multiple EPContext binaries. Should ORT guarantee the ordering of write invocations
   (e.g. all data for name A before name B)? The current contract prefers a single invocation per name
   unless the EP documents chunking.
4. **Thread safety of the read callback.** Each EP gets its own `OrtEpContextConfig`, so the design is
   thread-safe across EPs by construction. If a single EP calls the read path from multiple threads for
   different files, the application's callback must itself be thread-safe.
5. **Typed EP-context generation-option accessors (not implemented).** `OrtEpContextConfig` could also
   expose the EP-context generation options that EPs read today as string key/value pairs via
   `GetSessionConfigEntry` (e.g. `ep.context_enable`, `ep.context_embed_mode`, `ep.context_file_path`).
   These PRs implement only the I/O callbacks and state; typed accessors remain a possible future extension.

## Appendix: EP-Impact Cases (draft)

Rough case analysis of how compilation settings affect EP changes:

- **Embed mode = 1, internal weights:** no EP changes.
- **Embed mode = 1, external initializer file:**
  - No weightless: no EP changes.
  - Weightless: app sets the encryption callback; ORT writes weights to disk by invoking the callback
    after the EP's `Compile()` call to EP context; no EP changes.

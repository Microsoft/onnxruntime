# Making cuDNN Optional for the CUDA Execution Provider

Status: Draft / Proposal
Owner: (CUDA EP)
Scope: `onnxruntime/core/providers/cuda` (main static CUDA EP), CUDA Plugin EP
(`BUILD_CUDA_EP_AS_PLUGIN`), and the CUDA unit tests. **Out of scope:** TensorRT EP (it creates
and owns its own cuDNN handles and inherently depends on cuDNN/TensorRT). The existing
build-time `USE_CUDA_MINIMAL` path is also out of scope; it is used by TensorRT integration
and should remain available.

---

## 1. Motivation

Today the CUDA EP has a **hard link‑time and load‑time dependency** on cuDNN
(`libcudnn*.so` / `cudnn64_*.dll`) and on the header‑only `cudnn_frontend` library. If the
cuDNN shared libraries are not present on the machine, the ORT shared library
(`libonnxruntime.so` / `onnxruntime.dll`, or the CUDA provider DLL) **fails to load at all** —
even for models that use no cuDNN‑backed operators.

cuDNN is large (hundreds of MB across its sub‑libraries) and is only needed by a subset of
operators (Conv, Pooling, BatchNorm, LRN, RNN/LSTM/GRU, the cuDNN reduction path, the cuDNN
softmax path, etc.). Many transformer / LLM models do not require any of these.

**Goal:** Ship a *single* CUDA EP binary that

1. loads and runs **without** cuDNN present, and
2. **lazily** attempts to load cuDNN at first use; if cuDNN is available, cuDNN‑backed
   operators work exactly as today; if it is absent, those operators fail with a clear,
   actionable error (Phase 1) and, incrementally, fall back to native CUDA kernels
   (Phases 2 and 3).

This is delivered in **three phases**:

- **Phase 1 — Make cuDNN optional.** No new compute kernels. When cuDNN is missing, any op
  that needs it throws a clear `NOT_IMPLEMENTED` ("cuDNN is required for operator X but was
  not found") error at `Run` time. Everything that does not need cuDNN runs normally.
- **Phase 2 — Remove LLM‑relevant cuDNN dependencies.** Replace the cuDNN Softmax /
  LogSoftmax and reduction paths with existing native CUDA / CUB paths. Phase 1 + Phase 2 is
  the first milestone because LLM models may use these ops.
- **Phase 3 — Replace the remaining cuDNN‑backed NN ops.** Add native CUDA / CUTLASS /
  Triton‑cubin fallbacks for Pooling, normalization, LRN, Conv, ConvTranspose, and FusedConv.

---

## 2. Current state (as of this writing)

### 2.1 How cuDNN is linked

- `cmake/onnxruntime_providers_cuda.cmake`:
  `target_link_libraries(... CUDNN::cudnn_all cudnn_frontend ...)` — a normal link dependency,
  resolved at process load time.
- `cmake/onnxruntime_providers_cuda_plugin.cmake`: same (`CUDNN::cudnn_all`, `cudnn_frontend`).
- `cmake/onnxruntime_unittests.cmake`: links `cudnn_frontend` and includes `CUDNN_INCLUDE_DIR`.
- `cmake/deps.txt`: pins `cudnn_frontend` v1.x (header‑only C++ wrapper over the cuDNN v9
  *backend* API).
- `cmake/external/cudnn_frontend.cmake` fetches `cudnn_frontend`, sets `CUDNN_PATH` from
  `onnxruntime_CUDNN_HOME`, disables its samples/tests/python bindings, and marks its headers
  as system includes. In ORT this is a **compile-time header dependency**, not a runtime DLL.
- There is **no delay‑load** configured for cuDNN today (delay‑load is only used for DML /
  WebGPU / a few Win32 API sets).

### 2.2 How the handle is created and reached

- The cuDNN handle is created **eagerly**:
  - `CudaStream` constructor — `cudnnCreate(&cudnn_handle_)` / `cudnnSetStream(...)`
    (`onnxruntime/core/providers/cuda/cuda_stream_handle.cc`).
  - `CUDAExecutionProvider::PerThreadContext` — `cudnnCreate(&cudnn_handle_)`
    (`onnxruntime/core/providers/cuda/cuda_execution_provider.cc`).
  - CUDA Plugin EP — `cuda_stream_plugin.cc` and `cuda_kernel_adapter.h`.
- Kernels obtain the handle through:
  - `CudaKernel::GetCudnnHandle(context)` → `stream->cudnn_handle_`
    (`onnxruntime/core/providers/cuda/cuda_kernel.h`).
  - `CUDAExecutionProvider::PerThreadDefaultCudnnHandle()`.
  - The public `CudaContext` resource API (`cuda_context.h`,
    `CudaResource::cudnn_handle_t`) — used by custom ops.

### 2.3 Call sites and macros

- `CUDNN_CALL` / `CUDNN_CALL_THROW` (in `shared_inc/cuda_call.h`) and
  `CUDNN_RETURN_IF_ERROR` / `CUDNN2_RETURN_IF_ERROR` (in `cuda_common.h`).
- `CUDNN_FE_CALL` / `CUDNN_FE_RETURN_IF_ERROR` for the frontend (`cudnn_fe_call.*`).
- `CudaErrString<cudnnStatus_t>` calls `cudnnGetErrorString` (`cuda_call.cc`).
- All of the above are already gated by `#ifndef USE_CUDA_MINIMAL` in shared CUDA code. That
  build‑time path is used by TensorRT-related builds and is a useful
  inventory of cuDNN‑touching code, but it is *not* the CUDA EP runtime behavior we want.

### 2.4 Operators / components that depend on cuDNN

| Area | Files | cuDNN usage |
|---|---|---|
| Conv / ConvTranspose (v9 graph) | `nn/conv.cc`, `nn/conv_transpose.cc` | `cudnn_frontend` graph API + `cudnnAddTensor` |
| Conv / ConvTranspose (legacy) | `nn/conv_8.h`, `nn/conv_transpose_8.h` | `cudnnConvolutionForward`, `cudnnConvolutionBackwardData`, algo search |
| FusedConv (contrib) | `contrib_ops/cuda/fused_conv.cc` | `cudnnConvolutionBiasActivationForward`, activation desc |
| Pooling | `nn/pool.cc` | `cudnnPoolingForward`, pooling desc |
| BatchNormalization | `nn/batch_norm.cc` | `cudnnBatchNormalizationForwardInference/Training` |
| InstanceNormalization | `nn/instance_norm.cc` | BatchNorm training helper |
| LRN | `nn/lrn.cc` | `cudnnLRNCrossChannelForward` |
| RNN / LSTM / GRU | `rnn/cudnn_rnn_base.*`, `rnn/{rnn,lstm,gru}.h` | `cudnnRNNForward`, RNN/dropout descriptors |
| Reductions | `reduction/reduction_ops.*` | `cudnnReduceTensor` (Reduce\*, ArgMax/ArgMin) |
| Softmax (cuDNN path) | `math/softmax_common.cc` | `cudnnSoftmaxForward/Backward` |
| Einsum | `math/einsum_utils/*` | passes cuDNN handle into helpers |
| Dropout descriptor | `cudnn_common.h` (`CudnnDropout`) | used by RNN |
| Tensor/Filter descriptors | `cudnn_common.*` | `cudnnCreate*Descriptor`, `cudnnSet*Descriptor` |
| Contrib attention (optional) | `contrib_ops/cuda/bert/group_query_attention.cc`, `quantization/attention_quantization.cc`, `math/bias_softmax.cc` | optional cuDNN flash attention / handle passthrough |

> Note: Several of these ops already have, or can trivially get, a **non‑cuDNN** path
> (e.g. Softmax has a native warp/block kernel; reductions have CUB‑based paths; pooling and
> simple elementwise norms are straightforward). Softmax and reductions are Phase‑2
> candidates; the remaining NN ops are Phase‑3 candidates.

### 2.5 `cudnn_frontend` usage

`cudnn_frontend` is currently used as a header-only C++ graph API wrapper around cuDNN's
backend API:

- `onnxruntime/core/providers/cuda/nn/conv.h` includes `<cudnn_frontend.h>` and stores
  `cudnn_frontend::graph::Graph`, `Tensor_attributes`, `Pointwise_attributes`, and variant
  packs in `CudnnConvState`.
- `onnxruntime/core/providers/cuda/nn/conv.cc` builds cuDNN frontend graphs for v9 Conv,
  optional bias fusion, optional activation fusion, heuristic selection, support checks, plan
  building, workspace sizing, and graph execution.
- `onnxruntime/core/providers/cuda/nn/conv_transpose.cc` does the same for ConvTranspose
  using `Conv_dgrad_attributes`.
- `onnxruntime/core/providers/cuda/cudnn_common.{h,cc}` defines `CudnnFeTensor`, a small ORT
  helper that maps ORT tensor shapes/types into `cudnn_frontend::graph::Tensor_attributes`.
- `onnxruntime/core/providers/cuda/shared_inc/cudnn_fe_call.h` and
  `onnxruntime/core/providers/cuda/cudnn_fe_call.cc` adapt `cudnn_frontend::error_t` into
  ORT's `CudaCall` error-handling path.
- `onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc` has a cuDNN SDPA feature path
  selected through kernel options, but it does not directly include the high-level frontend
  graph headers in the same way Conv/ConvTranspose do.

Important frontend detail: `cudnn_frontend` already has a dynamic-loading mode gated by
`NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING`. In that mode, `cudnn_frontend_shim.h` does **not**
link directly against `cudnnBackend*` symbols. Instead, it expects the embedding library to
define `cudnn_frontend::cudnn_dlhandle`, then resolves symbols from that handle with
`dlsym` / `GetProcAddress`.

The frontend backend-symbol surface used by ORT's current graph path includes at least:

- `cudnnGetVersion`, `cudnnGetErrorString`, and for cuDNN 9, `cudnnGetLastErrorString`.
- `cudnnBackendCreateDescriptor`, `cudnnBackendDestroyDescriptor`,
  `cudnnBackendSetAttribute`, `cudnnBackendGetAttribute`, and `cudnnBackendFinalize`.
- `cudnnBackendExecute`.
- Version-gated helpers such as `cudnnBackendPopulateCudaGraph`,
  `cudnnBackendUpdateCudaGraph`, and `cudnnGetExecutionPlanWorkspaceSize` if ORT starts using
  frontend features that require them.

Therefore, `cudnn_frontend` should stay in the build while ORT still has cuDNN frontend
Conv/ConvTranspose paths, but it should be compiled in dynamic-loading mode and wired to the
same ORT-owned cuDNN loader used by direct cuDNN calls.

---

## 3. Design overview

The core idea: **break the hard dependency by routing every cuDNN symbol through a thin,
lazily‑resolved trampoline layer**, plus an availability flag the EP and kernels consult.

```mermaid
flowchart TD
    K[CUDA kernel<br/>e.g. Conv, Pool] -->|cudnnXxx(...)| S[cuDNN shim<br/>trampolines]
  FE[cudnn_frontend<br/>header-only] -->|dynamic mode dlsym(cudnn_dlhandle)| L
    S -->|first call: dlopen/LoadLibrary| L[cuDNN loader]
    L -->|present| R[(real libcudnn*)]
    L -->|absent| U[mark unavailable<br/>return sentinel status]
    K -.->|enable_cudnn && IsCudnnAvailable()?| L
```

### 3.1 The shim (no hard link)

We **stop linking** `CUDNN::cudnn_all`. In its place we compile a generated translation unit,
`cudnn_stub.cc`, that **defines every direct cuDNN entry point ORT references**. Each
definition is a trampoline:

```cpp
// Pseudocode for one entry
cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t h, /* ... */) {
  auto fn = CudnnLibrary::Get().convolution_forward;  // resolved lazily
  if (fn == nullptr) return CUDNN_STATUS_NOT_INITIALIZED;  // cuDNN unavailable
  return fn(h, /* ... */);
}
```

Because the trampolines have the **exact** cuDNN symbol names and signatures, ORT's direct
calls link against *our* definitions — so the final binary has **no `NEEDED`/import entry for
libcudnn** from those calls.

`cudnn_frontend` is handled separately: compile it with
`NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` and define `cudnn_frontend::cudnn_dlhandle` in ORT.
When the ORT loader successfully opens cuDNN, it sets that handle to the loaded cuDNN library
handle. `cudnn_frontend` then resolves `cudnnBackend*` symbols from the same handle. When
`enable_cudnn=0` or cuDNN is unavailable, ORT must guard all frontend graph-build/execute
entry points before calling frontend APIs so the frontend shim never tries to resolve symbols
from a null handle.

This means `cudnn_frontend` stays a compile‑time‑only dependency. We still need its headers
and the cuDNN headers to build, but not cuDNN import libraries at link time.

A single loader object owns the `dlopen`/`LoadLibrary` handles and the resolved function
pointers:

```cpp
class CudnnLibrary {            // onnxruntime/core/providers/cuda/cudnn_loader.{h,cc}
 public:
  static CudnnLibrary& Get();   // thread-safe singleton (std::call_once)
  bool Available() const;       // true iff all required libs + symbols resolved
  // ... function pointer members, one per cuDNN entry ORT/frontend uses ...
};
```

Loader responsibilities:

- On first use, attempt to load the cuDNN runtime. For cuDNN **9**, this is a small set of
  sub‑libraries (`libcudnn.so.9` umbrella plus, depending on packaging,
  `libcudnn_graph`, `libcudnn_ops`, `libcudnn_cnn`, `libcudnn_engines_*`). On Windows the
  corresponding `cudnn*64_9.dll` set. We load the umbrella `libcudnn` first; cuDNN itself
  dlopens its sub‑libraries.
- Resolve each required symbol with `dlsym` / `GetProcAddress`.
- If the umbrella library or any **required** symbol is missing, set `available_ = false`.
- Do not accept a provider option that names a cuDNN runtime path. A provider-controlled
  native library path is equivalent to a native code loading hook if provider options are
  influenced by untrusted input.

On Windows, cuDNN 9 is split into multiple DLLs. The loader should not rely on the process
working directory. Application or package code that needs an explicit directory should use a
trusted process-level preload mechanism, such as Python `preload_dlls(cudnn=True,
directory=...)`, before creating the session. On Linux, the C++ loader uses the default
dynamic loader search behavior for the umbrella `libcudnn.so.9`; deployment should provide
trusted library paths via the system loader, container image, package manager, or explicit
application preload.

The loader must not run when the CUDA provider option `enable_cudnn=0` is set (see §3.3).
This keeps "force no cuDNN" tests deterministic even on machines where cuDNN is installed.

**Symbol manifest.** The set of symbols is finite and enumerable (see §2.4 plus
`cudnn_common.*` and `cudnn_rnn_base.*`). We maintain direct ORT cuDNN calls as a single
header list (`cudnn_symbols.inc`, an X‑macro list) consumed by both the loader and the stub
generator so the two never drift. Frontend backend symbols are resolved by
`cudnn_frontend`'s own dynamic-loading shim from the same ORT-owned cuDNN handle; maintain a
separate frontend-symbol audit list for testing and version checks.

> **Alternative considered (delay‑load only):** Windows `/DELAYLOAD:cudnn*.dll` gets us lazy
> load on Windows, but Linux has no equivalent, and `cudnn_frontend` requires explicit dynamic
> loading support to avoid backend API imports. Rejected because the request requires
> identical behavior on Linux and Windows from a single binary. The direct-call trampoline
> plus frontend dynamic-loading approach is uniform across both.

### 3.2 Availability flag and handle lifecycle

- `cudnnCreate` is only invoked through the shim. The eager `cudnnCreate` calls in
  `CudaStream` / `PerThreadContext` / plugin stream become **conditional and non‑fatal**:
  - Attempt `CudnnLibrary::Get().Available()`; if false, leave `cudnn_handle_ == nullptr` and
    **do not throw**.
  - If true, create the handle as today.
- `GetCudnnHandle(context)` returns `nullptr` when cuDNN is unavailable (it already returns a
  raw handle; today it's never null).
- New helpers in `cuda_common.h`:

```cpp
bool CudnnAvailable(const OpKernelContext* context);  // provider option && runtime availability

// For kernels: fail fast with a clear message if cuDNN is required but missing.
#define ORT_RETURN_IF_CUDNN_UNAVAILABLE(context, op_name)                         \
  ORT_RETURN_IF_ERROR(::onnxruntime::cuda::CheckCudnnAvailable(context, op_name))
```

- `CudaErrString<cudnnStatus_t>` must not call `cudnnGetErrorString` when cuDNN is unavailable
  (route through the shim, which returns a static string in that case).

### 3.3 CUDA provider option: `enable_cudnn`

cuDNN can be disabled explicitly with a CUDA provider option:

```text
enable_cudnn = 1  # default: try to load and use cuDNN when it is present
enable_cudnn = 0  # do not load cuDNN; force native CUDA paths / Phase-1 NOT_IMPLEMENTED
```

`enable_cudnn` is the policy switch. When it is `0`, ORT must not attempt to load cuDNN. ORT
intentionally does not provide a `cudnn_path` provider option because provider options can be
supplied by higher-level configuration systems, and allowing them to choose a native DLL/SO
path would create a library-loading security risk.

Implementation details:

- Add `constexpr const char* kEnableCudnn = "enable_cudnn"` in
  `cuda::provider_option_names`.
- Add `bool enable_cudnn{true};` to `CUDAExecutionProviderInfo`.
- Parse it with `ProviderOptionsParser::AddAssignmentToReference(...)` in
  `CUDAExecutionProviderInfo::FromProviderOptions(...)`.
- Emit it from `CUDAExecutionProviderInfo::ToProviderOptions(...)`.
- Include it in `std::hash<CUDAExecutionProviderInfo>` because it changes the EP behavior.
- Do **not** add a field to `OrtCUDAProviderOptionsV2` for Phase 1. That struct is public C
  ABI surface; string-key provider options are sufficient and can be set through existing
  provider-options APIs.
- Add an EP helper such as `CUDAExecutionProvider::IsCudnnEnabled()` or
  `CudaKernel::IsCudnnEnabled()` so kernels can distinguish:
  - cuDNN disabled by user (`enable_cudnn=0`), and
  - cuDNN enabled but unavailable at runtime.

The effective condition for cuDNN use is:

```text
effective_cudnn_available = info.enable_cudnn && CudnnLibrary::Get().Available()
```

If `enable_cudnn=0`, ORT must not call `dlopen` / `LoadLibrary` for cuDNN and must not create
a cuDNN handle. If `enable_cudnn=1`, ORT uses trusted process-level library discovery: the
system loader search path, package/container deployment, or an explicit preload performed by
application code before session creation.

### 3.4 Phase 1 fallback behavior (chosen: throw at Run time)

Per the agreed design, in Phase 1 cuDNN‑dependent kernels **remain registered** but **fail
fast** when executed without cuDNN. The check is added at the top of each cuDNN op's
`ComputeInternal` (or centralized in shared base helpers such as `CudnnConvState` setup,
`CudnnRnnBase`, the reduction helper, etc.):

```cpp
Status Conv<T>::ComputeInternal(OpKernelContext* context) const {
  ORT_RETURN_IF_CUDNN_UNAVAILABLE(context, "Conv");
  // ... existing cuDNN path ...
}
```

The guard should include the reason in the message:

- `enable_cudnn=0`: "Operator 'Conv' on the CUDA EP requires cuDNN, but cuDNN was disabled
  by the CUDA provider option 'enable_cudnn'."
- cuDNN missing: "Operator 'Conv' on the CUDA EP requires cuDNN, but cuDNN was not found at
  runtime. Install cuDNN, or disable CUDA execution for this op/model."

Rationale for "throw" over "don't register / fall back to CPU":

- Keeps kernel registration tables identical regardless of runtime cuDNN presence (no
  divergence between build/load configurations; simpler, lower‑risk).
- Produces a clear, attributable error instead of silent CPU fallback that can mask perf
  cliffs.
- CPU fallback for individual nodes is still achievable by the user via EP assignment; we are
  not removing that option, only not making it implicit.

### 3.5 Builds in scope

- **Main static CUDA EP** — primary target; shim + loader compiled in.
- **CUDA Plugin EP** (`BUILD_CUDA_EP_AS_PLUGIN`) — same shim/loader; the plugin's
  `cuda_stream_plugin.cc` / `cuda_kernel_adapter.h` handle creation becomes conditional.
- **Unit tests** — link the shim instead of cuDNN; add tests that exercise both
  cuDNN‑present and cuDNN‑absent behavior (the latter by forcing the loader into the
  unavailable state, see §7).

Python wheel packaging is unchanged by this design: cuDNN DLLs are not packed in the wheel
today, so Phase 1 is not introducing a new "CUDA-minimal" wheel flavor. The runtime loader
simply makes the existing package tolerant of environments where cuDNN is absent.

The TensorRT EP is untouched and continues to link cuDNN as before. (If both a TRT EP and the
shimmed CUDA EP are in the same process, symbol collision must be avoided — see §8 Risks.)

### 3.6 Applying the same pattern to cuFFT

The identical shim + lazy‑loader technique is used to drop the hard dependency on **cuFFT**
(`libcufft.so.*` / `cufft64_*.dll`). cuFFT is only needed by the FFT contrib operators
(`Rfft` / `Irfft` in `contrib_ops/cuda/math`), which use a small, enumerable set of entry
points: `cufftCreate`, `cufftDestroy`, `cufftSetStream`, `cufftXtMakePlanMany`, and
`cufftXtExec`.

Implementation:

- **Loader** — `onnxruntime/core/providers/cuda/cufft_loader.{h,cc}` defines a
  `CufftLibrary` singleton that mirrors `CudnnLibrary`: it lazily `dlopen`/`LoadLibrary`s the
  cuFFT runtime, resolves symbols on demand, caches them, and exposes `Available()` / `Error()`.
  It uses the same security‑conscious search behavior as the cuDNN loader (Windows
  `LOAD_LIBRARY_SEARCH_DEFAULT_DIRS` plus a PATH fallback that never loads from the current
  working directory). The candidate library name is selected at **compile time** from the
  `CUDA_VERSION` macro (cuFFT 12 for CUDA 13.x, cuFFT 11 for CUDA 12.x), because cuFFT's
  SONAME/DLL version tracks the CUDA major version. Unsupported CUDA major versions fail at
  compile time until their cuFFT library mapping is added explicitly.
- **Shim** — `onnxruntime/core/providers/cuda/cufft_stub.cc` defines the five entry points
  above as trampolines that forward to the resolved symbol (returning `CUFFT_INTERNAL_ERROR`
  when the library is unavailable), so ORT's direct calls link against *our* definitions and
  the final binary has **no import entry for libcufft**.
- **Availability guard** — `CudaCall<cufftResult>` (in `cuda_call.cc`) and the plugin's
  `CUFFT_RETURN_IF_ERROR` macro check `CufftLibrary::Available()` and return a clear
  `NOT_IMPLEMENTED` status ("cuFFT is unavailable …") when cuFFT is missing, exactly like the
  cuDNN path.

Unlike cuDNN, cuFFT needs **no provider option** (there is no `enable_cufft`): it is always
loaded lazily on first FFT‑op use, and there is no `cufft_frontend` equivalent to configure.
cuFFT headers (part of the CUDA toolkit) are still required at build time, but `CUDA::cufft`
is no longer linked in `onnxruntime_providers_cuda.cmake`,
`onnxruntime_providers_cuda_plugin.cmake`, or the CUDA unit‑test target.

---

## 4. Phase 1 — Make cuDNN optional (no new kernels)

**Outcome:** ORT CUDA EP loads and runs without cuDNN. cuDNN‑backed ops throw a clear
`NOT_IMPLEMENTED` error when cuDNN is absent; everything else runs normally. When cuDNN *is*
present, behavior is byte‑for‑byte identical to today.

### 4.1 Task breakdown

1. **Symbol inventory & manifest.**
   - Enumerate every direct `cudnn*` symbol referenced by ORT code (`cudnnCreate`,
     descriptor APIs, legacy conv APIs, BN/LRN/pooling/reduction/softmax APIs, RNN APIs,
     etc.).
   - Capture direct calls as `cudnn_symbols.inc` (X‑macro: name, return type, signature).
   - Separately audit the `cudnn_frontend` backend-symbol surface resolved by its dynamic
     shim: `cudnnGetVersion`, `cudnnGetErrorString`, `cudnnBackend*` descriptor APIs,
     `cudnnBackendExecute`, and version-gated graph helpers such as
     `cudnnBackendPopulateCudaGraph`, `cudnnBackendUpdateCudaGraph`, and
     `cudnnGetExecutionPlanWorkspaceSize`.
   - *Verification:* link a probe binary that references only the direct-call manifest and
     diff against `nm -D`/`dumpbin` of the real cuDNN to ensure completeness; add a frontend
     graph-build/execute smoke test to prove `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` resolves
     backend symbols through ORT's loaded cuDNN handle.

2. **Loader (`cudnn_loader.{h,cc}`).**
   - `dlopen`/`LoadLibrary` of the cuDNN umbrella lib with versioned name candidates
     (`libcudnn.so.9`, `libcudnn.so`, `cudnn64_9.dll`, …).
   - Do not accept a provider-supplied cuDNN path. Rely on trusted deployment/library-search
     mechanisms or application-controlled preloading.
   - On Windows, avoid relying on the process working directory. Python package users who need
     an explicit directory should call `preload_dlls(cudnn=True, directory=...)` from trusted
     application code before creating a CUDA EP session.
   - Resolve all manifest symbols; populate function‑pointer table.
   - `Available()` + thread‑safe one‑time init; report loader diagnostics without exposing a
     provider-controlled library path option.
   - Expose the raw library handle to `cudnn_frontend` dynamic-loading mode.
   - Define and maintain `cudnn_frontend::cudnn_dlhandle` in one ORT translation unit when
     `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` is enabled. Set it to the loader's cuDNN handle
     after a successful load; keep it null when cuDNN is disabled or unavailable.
   - Add an explicit "disabled" path: when the EP has `enable_cudnn=0`, skip loader
     initialization entirely and report "disabled by provider option" to the error helper.

3. **Stub/trampoline TU (`cudnn_stub.cc`).**
   - Generate one trampoline per manifest entry forwarding to the loader's pointer; return a
     sentinel `cudnnStatus_t` when unavailable.
   - Handle the few non‑`cudnnStatus_t` entries (`cudnnGetErrorString`, `cudnnGetVersion`).

  The stubs must be compiled into the same target that currently links cuDNN. For Linux,
  prefer hidden visibility for the stub definitions where possible to avoid exporting cuDNN
  names from ORT provider binaries. The loader should use `RTLD_LOCAL` when opening cuDNN.

4. **CMake changes.**
   - Remove `CUDNN::cudnn_all` from `target_link_libraries` for the CUDA EP, plugin EP, and
     tests; **keep** `CUDNN_INCLUDE_DIR` (headers) and `cudnn_frontend` (headers).
   - Compile `cudnn_stub.cc` + `cudnn_loader.cc` into the EP.
   - Compile CUDA EP targets that include `cudnn_frontend` with
     `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING`. This is required so frontend graph code uses
     `dlsym` / `GetProcAddress` on `cudnn_frontend::cudnn_dlhandle` instead of creating
     link-time imports for `cudnnBackend*` symbols.
   - Keep `USE_CUDA_MINIMAL` working. It is used by TensorRT-related builds and is not
     replaced by the optional-cuDNN runtime shim.

   Current CMake anchor points:

   - `cmake/onnxruntime_providers_cuda.cmake`: replace `CUDNN::cudnn_all` with the shim
     sources/library while retaining `include(cudnn_frontend)` and the cuDNN include dirs;
     add `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` for the provider target.
   - `cmake/onnxruntime_providers_cuda_plugin.cmake`: same for
     `onnxruntime_providers_cuda_plugin`.
   - `cmake/onnxruntime_unittests.cmake`: unit tests should link against the shim path, not
     against cuDNN import libraries, and should use the same frontend dynamic-loading define.
   - `cmake/onnxruntime_python.cmake`: on Windows, the generated `version_info.py` currently
     searches for `cudnn64_*.dll` and fails if it is missing. That fatal check must be
     relaxed because cuDNN is optional; `cudnn_version` should be omitted, set to `None`, or
     set to `"optional"` when no DLL is found.

5. **Conditional handle creation.**
   - `cuda_stream_handle.cc`, `cuda_execution_provider.cc`, `cuda_stream_plugin.cc`,
     `cuda_kernel_adapter.h`: create the handle only if
     `info.enable_cudnn && CudnnLibrary::Get().Available()`, otherwise leave it null and do
     not throw.
   - When `info.enable_cudnn` is false, skip the loader call entirely.
   - External/custom-op resource behavior: `CudaResource::cudnn_handle_t` may be `nullptr`.
     Any internal custom-op adapter that assumes a non-null handle must return the same clear
     cuDNN-required error.

6. **Guard all cuDNN op entry points.**
   - Add `ORT_RETURN_IF_CUDNN_UNAVAILABLE(context, "<Op>")` or an equivalent helper to the
     `ComputeInternal` of every op in the §2.4 table (centralize in shared bases where
     possible: `CudnnRnnBase`, conv state setup, reduction helper, pooling,
     batch/instance norm, LRN, cuDNN softmax path).
   - For the cuDNN **softmax** and **reduction** paths that *already* have native
     alternatives, prefer routing to the native path when cuDNN is absent instead of
     throwing (Phase‑2 work; see §5). Otherwise throw.
   - Guard frontend graph creation as well as frontend graph execution. `cudnn_frontend`'s
     dynamic shim throws if it cannot resolve backend symbols, so ORT should fail with the
     clearer provider-option / cuDNN-missing message before calling `validate()`,
     `build_operation_graph()`, `create_execution_plans()`, `check_support()`,
     `build_plans()`, or `execute()`.

7. **Provider-option plumbing.**
  - Add and parse `enable_cudnn` in `CUDAExecutionProviderInfo`.
  - Return it from `GetProviderOptions()` / `ToProviderOptions()`.
  - Include it in the EP hash.
  - Add tests for parsing: `enable_cudnn` default true, `"0"` false, `"1"` true, invalid
    values rejected.

8. **Error‑string safety.**
   - Make `CudaErrString<cudnnStatus_t>` shim‑safe.
   - Make `CudaErrString<cudnn_frontend::error_t>` report frontend dynamic-loading failures
     without assuming cuDNN is available.

9. **Docs & messaging.**
   - Document the new behavior and `enable_cudnn`.
   - Update Python package guidance for `onnxruntime.preload_dlls(cuda=True, cudnn=True,
     directory=...)`: users can still preload a known cuDNN directory, but preloading is now
     optional for CUDA EP load because the provider itself lazy-loads cuDNN.
   - Update `onnxruntime/__init__.py` behavior as needed so missing cuDNN does not produce a
     scary install warning by default in optional-cuDNN packages. If the user explicitly calls
     `preload_dlls(cudnn=True)`, keep diagnostics useful and include the missing DLL name.

10. **CI workflow for no-cuDNN builds.**
    - Add a focused CI workflow/job that configures and builds the CUDA EP without cuDNN
      import libraries available at link time. It should still provide cuDNN headers, because
      the optional-cuDNN design remains source-compatible with cuDNN APIs and
      `cudnn_frontend` headers.
    - The job should verify the produced CUDA provider binary has no direct cuDNN runtime
      dependency (`readelf -d` / `ldd` on Linux, `dumpbin /dependents` on Windows).
    - Run at least a smoke test that imports ORT, initializes the CUDA EP, and executes a
      non-cuDNN CUDA model with cuDNN runtime libraries absent from the runtime library path.
    - Run a negative smoke test for one cuDNN-backed op, such as Conv, and assert the clear
      `NOT_IMPLEMENTED` error rather than a dynamic-loader failure.
    - Start with Linux CUDA CI, then add the equivalent Windows CUDA CI leg once the Windows
      sub-DLL search behavior is implemented and stable.

### 4.2 Acceptance criteria (Phase 1)

- With cuDNN **removed** from the system:
  - `libonnxruntime`/CUDA provider loads; a model with no cuDNN ops runs correctly on CUDA.
  - A model with a cuDNN op (e.g. Conv) fails with the clear `NOT_IMPLEMENTED` message, not a
    crash or loader error.
- With cuDNN **present** and `enable_cudnn=0`:
  - ORT does not load cuDNN or create a cuDNN handle.
  - Phase‑1 cuDNN ops fail with the "disabled by provider option" `NOT_IMPLEMENTED` message.
  - Phase‑2 / Phase‑3 native fallback ops run through the native path once implemented.
- With cuDNN **present**: full existing test suite passes unchanged (no perf/accuracy
  regression).
- Both main CUDA EP and plugin EP build and pass.
- A dedicated no-cuDNN CI job builds the CUDA EP without cuDNN import libraries, confirms the
  provider binary has no direct cuDNN runtime dependency, and runs the no-cuDNN smoke tests.

---

## 5. Phase 2 — Replace LLM‑relevant cuDNN paths

**Outcome:** LLM‑focused CUDA workloads can run without cuDNN for common Softmax /
LogSoftmax and reduction patterns. This phase is part of the first milestone with Phase 1:
Phase 1 makes cuDNN optional, and Phase 2 removes the cuDNN dependency from ops that LLM
models may still use.

### 5.1 Scope

1. **Softmax / LogSoftmax** — native warp/block kernels already exist; make them the default
   and drop the cuDNN path (or keep cuDNN only as an opt‑in fast path).
2. **Reductions / ArgMax / ArgMin** — CUB‑based implementations; remove the
   `cudnnReduceTensor` dependency.

### 5.2 Mechanism

For these ops, prefer the native implementation regardless of cuDNN availability, unless a
specific cuDNN fast path is intentionally kept behind an opt‑in provider option:

```text
use native CUDA / CUB implementation
optional: if provider option requests cuDNN fast path and cuDNN is available, use cuDNN
```

The important Phase‑2 property is that these ops must not require a non-null cuDNN handle.
If `enable_cudnn=0`, or if cuDNN is absent, they should still run through the native path.

### 5.3 Acceptance criteria (Phase 2, per op)

- Native path matches cuDNN within tolerance on the op's existing unit tests.
- With cuDNN absent, the op runs (no `NOT_IMPLEMENTED`).
- With cuDNN present, no regression in correctness; any retained cuDNN fast path is explicit
  and test-covered.
- With `enable_cudnn=0`, no dynamic cuDNN load is attempted.

---

## 6. Phase 3 — Replace remaining cuDNN‑backed NN ops

**Outcome:** Broader CNN / vision-style CUDA workloads can run without cuDNN where practical.
These ops are outside the first Phase 1 + Phase 2 milestone because they are less central to
LLM workloads and, for convolution, much more expensive to replace well.

### 6.1 Scope

1. **Pooling** (Max/Average, global variants) — straightforward native kernels.
2. **BatchNormalization / InstanceNormalization (inference)** — elementwise affine over
  precomputed stats; native kernel is simple.
3. **LRN** — native kernel.
4. **Conv / ConvTranspose / FusedConv** — the hard part. Options:
  - implicit‑GEMM CUDA kernels for common cases,
  - CUTLASS conv,
  - precompiled Triton conv cubins,
  - or im2col + existing GEMM as a correctness fallback.
  Keep cuDNN as the preferred fast path when available; native kernel as fallback.

RNN / LSTM / GRU are intentionally not part of the Phase‑3 scope for now. They are the
heaviest to replace and may remain cuDNN‑only with a clear `NOT_IMPLEMENTED` when cuDNN is
absent unless there is product demand.

### 6.2 Mechanism

For each Phase‑3 op, introduce a dispatch at `ComputeInternal`:

```text
if info.enable_cudnn and CudnnLibrary::Get().Available() and (cuDNN path preferred): use cuDNN
else: use native CUDA / CUTLASS / Triton-cubin fallback
```

This preserves cuDNN performance where present while removing the hard requirement.
Triton cubins (see `docs/ORT_Use_Triton_Kernel.md`) are an option for fused/normalization
kernels; CUTLASS (already vendored) for conv/GEMM‑shaped work.

### 6.3 Acceptance criteria (Phase 3, per op)

- Native path matches cuDNN within tolerance on the op's existing unit tests.
- With cuDNN absent, the op runs (no `NOT_IMPLEMENTED`).
- With cuDNN present, no regression (cuDNN path still selected unless configured otherwise).
- With `enable_cudnn=0`, the op uses the native fallback and does not load cuDNN.

---

## 7. Testing strategy

- **Loader unit tests:** simulate cuDNN present vs absent.
  - "Absent" is forced via a test hook on `CudnnLibrary` (e.g. an internal
    `SetForceUnavailableForTest(true)` compiled only in test/internal builds), avoiding the
    need to physically remove cuDNN in CI.
  - "Disabled" is tested with CUDA provider option `enable_cudnn=0`; this should not touch
    the dynamic loader at all.
- **Op‑level tests:** for each cuDNN op, assert the clear `NOT_IMPLEMENTED` error in the
  forced‑absent mode (Phase 1), and correctness in present mode.
- **cuDNN frontend tests:** add a Conv / ConvTranspose test that exercises frontend graph
  creation and execution with cuDNN present, and verifies that `enable_cudnn=0` fails before
  `cudnn_frontend` attempts to resolve backend symbols.
- **Binary dependency tests:** inspect the CUDA provider binary (`ldd` / `readelf -d` on
  Linux, `dumpbin /dependents` on Windows) and confirm there is no direct dependency on
  `libcudnn*` / `cudnn64_*.dll` even though `cudnn_frontend` headers are compiled in.
- **Phase‑2 native path tests:** run Softmax / LogSoftmax and reduction tests with
  `enable_cudnn=0` and with the forced‑absent loader hook. These should pass, not throw.
- **Phase‑3 native path tests:** as each Phase‑3 op is implemented, add the same
  `enable_cudnn=0` / forced‑absent coverage for that op.
- **Load test:** a process that initializes the CUDA EP with the cuDNN libs unavailable and
  runs a non‑cuDNN model end‑to‑end.
- **No-cuDNN CI workflow:** add a workflow/job, initially in Linux CUDA CI, that removes cuDNN
  import libraries from the link/runtime environment while keeping cuDNN headers available.
  It should build the CUDA EP, inspect dynamic dependencies, run the non-cuDNN smoke model,
  and verify a cuDNN-backed op fails with ORT's `NOT_IMPLEMENTED` message. Add the Windows
  equivalent after the Windows cuDNN sub-DLL loading path is covered.
- **Python preload tests:** verify `onnxruntime.preload_dlls(cudnn=True, directory=...)`
  still preloads a user-provided cuDNN directory, while normal `import onnxruntime` and CUDA
  EP initialization do not fail or print install guidance solely because cuDNN is missing.
- **Regression:** full existing CUDA suite with cuDNN present must stay green.
- CI legs that genuinely lack cuDNN can additionally validate the real (not forced) path.

---

## 8. Risks and mitigations

- **Symbol‑manifest completeness for direct cuDNN calls.** Missing a direct-call symbol in
  `cudnn_symbols.inc` would cause an unresolved-symbol link error or a runtime trampoline
  failure. *Mitigation:* the probe-binary diff in §4.1-1, and CI that builds with cuDNN
  import libs unavailable.
- **`cudnn_frontend` dynamic-loading integration.** If
  `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` is not applied consistently, frontend graph code may
  still create imports for `cudnnBackend*` symbols. If `cudnn_frontend::cudnn_dlhandle` is not
  defined/set by ORT, frontend calls will throw while resolving backend symbols. *Mitigation:*
  apply the compile definition to every target that includes frontend headers, define the
  global handle once in ORT, and add binary dependency plus Conv/ConvTranspose frontend smoke
  tests.
- **`cudnn_frontend` may add new backend symbols across dependency bumps.** A future
  `cudnn_frontend` version may resolve additional backend, CUDA, CUDART, NVRTC, or experimental
  symbols. *Mitigation:* after every `cudnn_frontend` update, grep/audit
  `cudnn_frontend_shim.h` and experimental shims, then update the frontend-symbol audit tests.
- **cuDNN sub‑library packaging differences (v9 split libs; distro/conda/pip layouts).**
  *Mitigation:* rely on trusted deployment mechanisms for library discovery. Python users who
  need a specific directory can call `onnxruntime.preload_dlls(cudnn=True, directory=...)`
  from application code before creating the session; C++ deployments should use container,
  package-manager, or system loader configuration.
- **Native library path provider options can become code-loading hooks.** If an attacker can
  influence a provider option that names a DLL/SO directory, they can potentially cause ORT
  to load attacker-controlled native code. *Mitigation:* do not expose a `cudnn_path` provider
  option. Keep custom library discovery at trusted process/deployment layers instead.
- **Python preload behavior can conflict with optional cuDNN.** Today
  `onnxruntime.preload_dlls(cudnn=True)` tries to load cuDNN and prints installation guidance
  on failure. That is useful when the user requested preloading, but too alarming if cuDNN is
  optional. *Mitigation:* keep explicit preloading available, but avoid invoking or requiring
  cuDNN preload as part of normal optional-cuDNN package import / CUDA EP load. Update docs so
  users who need a specific cuDNN directory can call `preload_dlls(cudnn=True, directory=...)`
  before creating the session.
- **Python version metadata currently assumes cuDNN on Windows CUDA builds.**
  `cmake/onnxruntime_python.cmake` fails if no `cudnn64_*.dll` is found when generating
  `version_info.py`. *Mitigation:* make `cudnn_version` optional in that generated file.
- **Symbol collision when a cuDNN‑linking EP (TensorRT) and the shimmed CUDA EP coexist in
  one process.** Our trampolines define real cuDNN symbol names; if TRT's cuDNN is also
  loaded, the dynamic linker could bind either. *Mitigation:* keep the shim symbols with
  internal/hidden visibility where possible and resolve the real library explicitly via the
  loader (`RTLD_LOCAL`); document the constraint; consider a build option that keeps the
  classic hard‑link behavior for TRT‑combined packages.
- **ABI/version skew** (built against cuDNN headers vN, loads runtime vM). *Mitigation:*
  check `cudnnGetVersion` in the loader and refuse (mark unavailable) on incompatible major
  versions.
- **Performance:** one extra indirect call per cuDNN entry — negligible relative to kernel
  cost.

---

## 9. Backward compatibility

- Default behavior with cuDNN installed is unchanged (same kernels, same perf).
- The build still requires cuDNN **headers** (and `cudnn_frontend` headers) to compile; it no
  longer requires cuDNN **import libraries** to link the in‑scope targets.
- `cudnn_frontend` remains a compile-time dependency while ORT uses frontend graph APIs for
  Conv / ConvTranspose. It should not introduce a runtime cuDNN dependency when compiled with
  `NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING` and wired to ORT's loader handle.
- The wheel package is unchanged: cuDNN DLLs are not packed into the wheel today, and this
  design does not introduce a separate CUDA-minimal wheel.
- `onnxruntime.preload_dlls()` remains supported for users who want Python to preload CUDA /
  cuDNN libraries from PyTorch, NVIDIA site packages, or an explicit directory. It becomes an
  optional convenience path for cuDNN, not a requirement for importing ORT or initializing the
  CUDA EP without cuDNN.
- No public API change is required for Phase 1. The custom‑op `CudaContext::cudnn_handle`
  may now be `nullptr`; this is documented, and `FetchResource` returns null gracefully.

---

## 10. Resolved decisions

- Force-disabling cuDNN is a **CUDA provider option**, not a session option. Use
  `enable_cudnn=0`.
- Providing a custom cuDNN runtime directory is **not** a CUDA provider option. Use trusted
  deployment/library-search configuration, or Python `preload_dlls(cudnn=True, directory=...)`
  from application code before creating the session.
- No new "CUDA-minimal" wheel is required for Phase 1. cuDNN DLLs are not packed in the wheel
  today.
- Keep the existing `USE_CUDA_MINIMAL` build-time path. It is used by TensorRT-related EP
  builds and is not replaced by the CUDA EP runtime shim.

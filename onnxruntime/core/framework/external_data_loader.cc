// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/external_data_loader.h"
#if defined(__wasm__)
#include <emscripten.h>
#endif

namespace onnxruntime {

#if defined(__wasm__)

// Error codes returned by the JS loader bodies below. The async and sync
// loaders share this contract - keep them in sync:
//   0 - success
//   1 - Module.MountedFiles is not available
//   2 - file not found in preloaded files
//   3 - out-of-bounds
//   4 - unknown error during memory copy / GPU upload
//   5 - Blob stream returned an unexpected number of bytes

#if defined(ORT_WASM_JSPI)

// Asynchronous external data loader, available in JSPI builds only.
//
// The offset/length parameters are doubles so the loader itself can address beyond
// 4GB, but the current wasm caller still rejects offset + length >= 4GB before reaching here.

// clang-format off
EM_ASYNC_JS(int, OrtLoadWebAssemblyExternalDataAsync,
            (const char* data_file_path, double data_offset, double data_length,
             void* tensor_data, int32_t load_type), {

  if (typeof Module == 'undefined' || !Module.MountedFiles) {
    return 1;  // "Module.MountedFiles" is not available.
  }
  let fileName = UTF8ToString(Number(data_file_path >>> 0));
  if (fileName.startsWith('./')) {
    fileName = fileName.substring(2);
  }
  const fileData = Module.MountedFiles.get(fileName);
  if (!fileData) {
    return 2;  // file not found in preloaded files.
  }
  const offset = data_offset;
  const length = data_length;
  const dataIdOrBuffer = Number(tensor_data >>> 0);

  if (offset < 0 || length < 0) {
    return 3;
  }

  let ownScratchLock = false;
  try {
    let data;
    if (typeof Blob !== 'undefined' && fileData instanceof Blob) {
      if (offset + length > fileData.size) {
        return 3;  // Out of bounds.
      }
      if (length === 0) {
        data = new Uint8Array(0);
      } else {
        // Stream into a reused scratch buffer rather than slice().arrayBuffer(), which
        // creates a large short-lived buffer per initializer and inflates peak memory.
        let scratch;
        if (!Module.ortExtDataScratchBusy) {
          Module.ortExtDataScratchBusy = true;
          ownScratchLock = true;
          scratch = Module.ortExtDataScratch;
          if (!scratch || scratch.length < length) {
            scratch = Module.ortExtDataScratch = new Uint8Array(length);
          }
        } else {
          scratch = new Uint8Array(length);
        }
        const stream = fileData.slice(offset, offset + length).stream();
        // BYOB reader avoids per-chunk allocations; fall back if unavailable.
        let reader;
        let byob = false;
        try {
          reader = stream.getReader({ 'mode': 'byob' });
          byob = true;
        } catch (e) {
          reader = stream.getReader();
        }
        let pos = 0;
        if (byob) {
          let chunk = Module.ortExtDataChunk || new ArrayBuffer(1048576);
          Module.ortExtDataChunk = null;  // BYOB read detaches; one owner per load.
          for (;;) {
            const r = await reader.read(new Uint8Array(chunk, 0, chunk.byteLength));
            if (r.value) {
              if (r.value.byteLength > 0) {
                if (pos + r.value.byteLength > length) {
                  Module.ortExtDataChunk = r.value.buffer;
                  return 5;  // Stream yielded more bytes than requested.
                }
                scratch.set(r.value, pos);
                pos += r.value.byteLength;
              }
              chunk = r.value.buffer;  // reclaim the (detached) chunk buffer
            }
            if (r.done) break;
          }
          Module.ortExtDataChunk = chunk;
        } else {
          for (;;) {
            const r = await reader.read();
            if (r.done) break;
            if (pos + r.value.byteLength > length) {
              return 5;  // Stream yielded more bytes than requested.
            }
            scratch.set(r.value, pos);
            pos += r.value.byteLength;
          }
        }
        if (pos !== length) {
          return 5;  // Reading from the Blob returned an unexpected number of bytes.
        }
        data = scratch.subarray(0, length);
      }
    } else {
      if (offset + length > fileData.byteLength) {
        return 3;  // Out of bounds.
      }
      data = fileData.subarray(offset, offset + length);
    }

    switch (load_type) {
      case 0:
        // Load external data to CPU memory.
        // Copy the file data (fileData,offset,length) into WebAssembly memory
        // (HEAPU8,buffer,length).
        HEAPU8.set(data, dataIdOrBuffer);
        break;
      case 1:
        // Load external data to GPU.
        // TODO: use a unified interface for upload external buffer.
        if (Module.webgpuUploadExternalBuffer) {
          Module.webgpuUploadExternalBuffer(dataIdOrBuffer, data);
        } else {
          Module.jsepUploadExternalBuffer(dataIdOrBuffer, data);
        }
        break;
      default:
        return 4;  // Unknown error occurred in memory copy.
    }
    return 0;
  } catch (e) {
    console.error('Failed to load external data "' + fileName + '":', e);
    return 4;
  } finally {
    // `data` may alias the shared scratch; release it only after the copy/upload
    // above has consumed it.
    if (ownScratchLock) {
      Module.ortExtDataScratchBusy = false;
    }
  }
});
// clang-format on

#endif

common::Status ValidateWebAssemblyExternalDataRange(const std::filesystem::path& data_file_path,
                                                    FileOffsetType data_offset,
                                                    SafeInt<size_t> data_length) {
  // clang-format off
  auto err_code = EM_ASM_INT(({
                               if (typeof Module == 'undefined' || !Module.MountedFiles) {
                                 return 1;
                               }
                               let fileName = UTF8ToString(Number($0 >>> 0));
                               if (fileName.startsWith('./')) {
                                 fileName = fileName.substring(2);
                               }
                               const fileData = Module.MountedFiles.get(fileName);
                               if (!fileData) {
                                 return 2;
                               }
                               const fileSize = typeof Blob !== 'undefined' && fileData instanceof Blob
                                                    ? fileData.size
                                                    : fileData.byteLength;
                               if (typeof fileSize !== 'number') {
                                 return 4;
                               }
                               return Number($1 >>> 0) + Number($2 >>> 0) > fileSize ? 3 : 0;
                             }),
                             data_file_path.c_str(),
                             static_cast<int32_t>(data_offset),
                             static_cast<int32_t>(data_length));
  // clang-format on
  switch (err_code) {
    case 0:
      return Status::OK();
    case 1:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Module.MountedFiles is not available.");
    case 2:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "External data file not found: ", data_file_path);
    case 3:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "External data file is out of bounds: ", data_file_path);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid mounted external data file: ", data_file_path);
  }
}

common::Status LoadWebAssemblyExternalData(const Env& env,
                                           const std::filesystem::path& data_file_path,
                                           FileOffsetType data_offset,
                                           SafeInt<size_t> data_length,
                                           ExternalDataLoadType load_type,
                                           void* tensor_data) {
#if defined(ORT_WASM_JSPI)
  auto err_code = OrtLoadWebAssemblyExternalDataAsync(data_file_path.c_str(),
                                                      static_cast<double>(data_offset),
                                                      static_cast<double>(static_cast<size_t>(data_length)),
                                                      tensor_data,
                                                      static_cast<int32_t>(load_type));
#else
  auto err_code = EM_ASM_INT(({
                               // If available, "Module.MountedFiles" is a Map for all preloaded files.
                               if (typeof Module == 'undefined' || !Module.MountedFiles) {
                                 return 1;  // "Module.MountedFiles" is not available.
                               }
                               let fileName = UTF8ToString(Number($0 >>> 0));
                               if (fileName.startsWith('./')) {
                                 fileName = fileName.substring(2);
                               }
                               const fileData = Module.MountedFiles.get(fileName);
                               if (!fileData) {
                                 return 2;  // File not found in preloaded files.
                               }
                               const offset = Number($1 >>> 0);
                               const length = Number($2 >>> 0);
                               const dataIdOrBuffer = Number($3 >>> 0);
                               const loadType = $4;

                               if (offset + length > fileData.byteLength) {
                                 return 3;  // Out of bounds.
                               }

                               try {
                                 const data = fileData.subarray(offset, offset + length);
                                 switch (loadType) {
                                   case 0:
                                     // Load external data to CPU memory.
                                     // Copy the file data (fileData,offset,length) into WebAssembly memory
                                     // (HEAPU8,buffer,length).
                                     HEAPU8.set(data, dataIdOrBuffer);
                                     break;
                                   case 1:
                                     // Load external data to GPU.
                                     // TODO: use a unified interface for upload external buffer.
                                     if (Module.webgpuUploadExternalBuffer) {
                                       Module.webgpuUploadExternalBuffer(dataIdOrBuffer, data);
                                     } else {
                                       Module.jsepUploadExternalBuffer(dataIdOrBuffer, data);
                                     }
                                     break;
                                   default:
                                     return 4;  // Unknown error occurred in memory copy.
                                 }
                                 return 0;
                               } catch {
                                 return 4;
                               }
                             }),
                             data_file_path.c_str(),
                             static_cast<int32_t>(data_offset),
                             static_cast<int32_t>(data_length),
                             tensor_data,
                             static_cast<int32_t>(load_type));
#endif
  const char* err_msg;
  switch (err_code) {
    case 0:
      return Status::OK();
    case 1:
      err_msg = "Module.MountedFiles is not available.";
      break;
    case 2:
      err_msg = "File not found in preloaded files.";
      break;
    case 3:
      err_msg = "Out of bounds.";
      break;
#if defined(ORT_WASM_JSPI)
    case 5:
      err_msg = "Reading from the Blob returned an unexpected number of bytes.";
      break;
#endif
    default:
      err_msg = "Unknown error occurred in memory copy.";
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to load external data file \"", data_file_path,
                         "\", error: ", err_msg);
}

#endif

}  // namespace onnxruntime

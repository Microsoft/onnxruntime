// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const NODEJS_TEST_CASES = [
  './node-test-main-no-threads.js',
  './node-test-main.js',
  './node-test-main.mjs',
  './node-test-wasm-path-override-filename.js',
  './node-test-wasm-path-override-prefix.js',
];

// [test_for_same_origin, test_for_cross_origin, main_js, ort_main_js, [test_args]]
const BROWSER_TEST_CASES = [
  // IIFE
  [true, true, './browser-test-webgl.js', 'ort.all.min.js'], // webgl
  [true, true, './browser-test-webgl.js', 'ort.webgl.min.js'], // webgl
  [true, true, './browser-test-wasm.js', 'ort.wasm.min.js'], // wasm, ort.wasm
  [true, true, './browser-test-wasm-fp6-external-data.js', 'ort.wasm.min.js'], // wasm, external FP6
  [true, true, './browser-test-wasm-multi-session-create.js', 'ort.min.js'], // wasm, multi-session create
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=1']], // wasm, 1 thread
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=2']], // wasm, 2 threads
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=2', 'proxy=1']], // wasm, 2 threads, proxy
  [true, true, './browser-test-wasm.js', 'ort.min.js', ['num_threads=1', 'proxy=1']], // wasm, 1 thread, proxy

  // ort.min.mjs
  [true, true, './browser-test-webgl.js', 'ort.webgl.min.mjs'], // webgl
  [true, true, './browser-test-wasm.js', 'ort.min.mjs', ['num_threads=1']], // wasm, 1 thread
  [true, true, './browser-test-wasm.js', 'ort.min.mjs', ['num_threads=2']], // wasm, 2 threads
  [true, true, './browser-test-wasm.js', 'ort.min.mjs', ['num_threads=2', 'proxy=1']], // wasm, 2 threads, proxy
  [true, true, './browser-test-wasm.js', 'ort.min.mjs', ['num_threads=1', 'proxy=1']], // wasm, 1 thread, proxy

  // ort.bundle.min.mjs
  [true, true, './browser-test-wasm.js', 'ort.bundle.min.mjs', ['num_threads=1']], // 1 thread
  [true, true, './browser-test-wasm.js', 'ort.bundle.min.mjs', ['num_threads=2']], // 2 threads
  [true, true, './browser-test-wasm.js', 'ort.bundle.min.mjs', ['num_threads=2', 'proxy=1']], // 2 threads, proxy
  [true, true, './browser-test-wasm.js', 'ort.bundle.min.mjs', ['num_threads=1', 'proxy=1']], // 1 thread, proxy

  // wasm binary override:
  [true, false, './browser-test-wasm-binary-override.js', 'ort.min.js'],

  // path override:
  // wasm, path override filenames for both mjs and wasm, same origin
  [true, false, './browser-test-wasm-path-override-filename-jsep.js', 'ort.min.js', ['port=9876', 'files=mjs,wasm']],
  [true, false, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['port=9876', 'files=mjs,wasm']],
  // wasm, path override filenames for both mjs and wasm, cross origin
  [false, true, './browser-test-wasm-path-override-filename-jsep.js', 'ort.min.js', ['port=8081', 'files=mjs,wasm']],
  [false, true, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['port=8081', 'files=mjs,wasm']],
  // wasm, path override filename for wasm, same origin
  [true, false, './browser-test-wasm-path-override-filename-jsep.js', 'ort.min.js', ['port=9876', 'files=wasm']],
  [true, false, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['port=9876', 'files=wasm']],
  // wasm, path override filename for wasm, cross origin
  [false, true, './browser-test-wasm-path-override-filename-jsep.js', 'ort.min.js', ['port=8081', 'files=wasm']],
  [false, true, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['port=8081', 'files=wasm']],
  // wasm, path override filename for mjs, same origin
  [true, false, './browser-test-wasm-path-override-filename-jsep.js', 'ort.min.js', ['port=9876', 'files=mjs']],
  [true, false, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['port=9876', 'files=mjs']],
  // wasm, path override filename for mjs, cross origin
  [false, true, './browser-test-wasm-path-override-filename-jsep.js', 'ort.min.js', ['port=8081', 'files=mjs']],
  [false, true, './browser-test-wasm-path-override-filename.js', 'ort.wasm.min.js', ['port=8081', 'files=mjs']],
  // wasm, path override prefix, same origin
  [true, false, './browser-test-wasm-path-override-prefix.js', 'ort.min.js', ['port=9876']],
  [true, false, './browser-test-wasm-path-override-prefix.js', 'ort.wasm.min.js', ['port=9876']],
  // wasm, path override prefix, cross origin
  [false, true, './browser-test-wasm-path-override-prefix.js', 'ort.min.js', ['port=8081']],
  [false, true, './browser-test-wasm-path-override-prefix.js', 'ort.wasm.min.js', ['port=8081']],

  [true, true, './browser-test-wasm-image-tensor-image.js', 'ort.min.js', ['port=8081']], // pre-post-process
  [true, true, './browser-test-webgpu-custom-device.js', 'ort.webgpu.min.js'], // user-provided GPUDevice
  [true, true, './browser-test-webgpu-external-data.js', 'ort.webgpu.min.js'], // external data
  [true, true, './browser-test-webgpu-external-data-blob.js', 'ort.webgpu.min.js'], // external data as Blob (fallback)
  [true, true, './browser-test-webgpu-external-data-blob.js', 'ort.jspi.min.js'], // external data as Blob (on-demand)
];

// [bundle_path, format]
const BUNDLER_TEST_CASES = [
  ['./dist/webpack_esm_js/ort-test-e2e.bundle.mjs', 'esm'],
  ['./dist/webpack_umd_js/ort-test-e2e.bundle.js', 'iife'],
  ['./dist/rollup_esm_js/ort-test-e2e.bundle.mjs', 'esm'],
  ['./dist/rollup_umd_js/ort-test-e2e.bundle.js', 'iife'],
  ['./dist/parcel_esm_js/main.js', 'esm'],
  ['./dist/parcel_umd_js/main.js', 'iife'],
  ['./dist/vite_esm_js/ort-test-e2e.bundle.mjs', 'esm'],
  ['./dist/vite_umd_js/ort-test-e2e.bundle.js', 'iife'],
];

module.exports = {
  NODEJS_TEST_CASES,
  BROWSER_TEST_CASES,
  BUNDLER_TEST_CASES,
};

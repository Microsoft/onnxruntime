// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

it('Browser E2E testing - WASM backend with external packed FP6 data', async function () {
  const session = await ort.InferenceSession.create('./fp6-external-data.onnx', {
    executionProviders: ['wasm'],
    externalData: [{ data: './fp6-external-data.bin', path: 'fp6-external-data.bin' }],
    // FP6 Cast first appears in the under-development ONNX opset 28.
    extra: { session: { allow_released_opsets_only: '0' } },
  });

  const outputs = await session.run({});
  for (const [name, expected] of [
    ['E2M3', [1, -2, 4, -6, 0.5]],
    ['E3M2', [1, -0.5, 2, -16, 0.25]],
  ]) {
    const output = outputs[name];
    assert(output instanceof ort.Tensor);
    assert(output.type === 'float32');
    assert(output.dims.length === 1 && output.dims[0] === expected.length);
    assert(output.data.length === expected.length);
    for (let i = 0; i < expected.length; i++) {
      assert(output.data[i] === expected[i]);
    }
  }

  await session.release();
});

it('Browser E2E testing - reject oversized external FP6 data before WASM allocation', async function () {
  for (const type of ['e2m3', 'e3m2']) {
    let session;
    try {
      session = await ort.InferenceSession.create(`./fp6-external-data-short-${type}.onnx`, {
        executionProviders: ['wasm'],
        externalData: [{ data: './fp6-external-data.bin', path: 'fp6-external-data.bin' }],
        extra: { session: { allow_released_opsets_only: '0' } },
      });
      assert(false, `expected an out-of-bounds error for ${type}`);
    } catch (e) {
      assert(String(e).toLowerCase().includes('out of bounds'), `${type}: ${e}`);
    } finally {
      if (session) {
        await session.release();
      }
    }
  }
});

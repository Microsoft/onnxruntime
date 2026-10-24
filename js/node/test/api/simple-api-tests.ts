// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import assert from 'assert';
import { InferenceSession, Tensor } from 'onnxruntime-common';
import * as path from 'path';

import { assertDataEqual, TEST_DATA_ROOT } from '../test-utils';

const MODEL_TEST_TYPES_CASES: Array<{
  model: string;
  type: Tensor.Type;
  input0: Tensor.DataType;
  expectedOutput0: Tensor.DataType;
}> = [
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_bool.onnx'),
    type: 'bool',
    input0: Uint8Array.from([1, 0, 0, 1, 0]),
    expectedOutput0: Uint8Array.from([1, 0, 0, 1, 0]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_double.onnx'),
    type: 'float64',
    input0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
    expectedOutput0: Float64Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_float.onnx'),
    type: 'float32',
    input0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
    expectedOutput0: Float32Array.from([1.0, 2.0, 3.0, 4.0, 5.0]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int8.onnx'),
    type: 'int8',
    input0: Int8Array.from([1, -2, 3, 4, -5]),
    expectedOutput0: Int8Array.from([1, -2, 3, 4, -5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int16.onnx'),
    type: 'int16',
    input0: Int16Array.from([1, -2, 3, 4, -5]),
    expectedOutput0: Int16Array.from([1, -2, 3, 4, -5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int32.onnx'),
    type: 'int32',
    input0: Int32Array.from([1, -2, 3, 4, -5]),
    expectedOutput0: Int32Array.from([1, -2, 3, 4, -5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_int64.onnx'),
    type: 'int64',
    input0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)]),
    expectedOutput0: BigInt64Array.from([BigInt(1), BigInt(-2), BigInt(3), BigInt(4), BigInt(-5)]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_string.onnx'),
    type: 'string',
    input0: ['a', 'b', 'c', 'd', 'e'],
    expectedOutput0: ['a', 'b', 'c', 'd', 'e'],
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint8.onnx'),
    type: 'uint8',
    input0: Uint8Array.from([1, 2, 3, 4, 5]),
    expectedOutput0: Uint8Array.from([1, 2, 3, 4, 5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint16.onnx'),
    type: 'uint16',
    input0: Uint16Array.from([1, 2, 3, 4, 5]),
    expectedOutput0: Uint16Array.from([1, 2, 3, 4, 5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint32.onnx'),
    type: 'uint32',
    input0: Uint32Array.from([1, 2, 3, 4, 5]),
    expectedOutput0: Uint32Array.from([1, 2, 3, 4, 5]),
  },
  {
    model: path.join(TEST_DATA_ROOT, 'test_types_uint64.onnx'),
    type: 'uint64',
    input0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)]),
    expectedOutput0: BigUint64Array.from([BigInt(1), BigInt(2), BigInt(3), BigInt(4), BigInt(5)]),
  },
];

describe('API Tests - simple API tests', () => {
  it('recognizes every WebGPU weightLoadAcceleration value', async () => {
    const values: ReadonlyArray<NonNullable<InferenceSession.WebGpuExecutionProviderOption['weightLoadAcceleration']>> =
      ['off', 'preferred', 'required'];
    const model = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');
    let unavailableProviderError: string | undefined;

    for (const weightLoadAcceleration of values) {
      try {
        const session = await InferenceSession.create(model, {
          executionProviders: [{ name: 'webgpu', weightLoadAcceleration }],
        });
        await session.release();
      } catch (error) {
        const message = String(error);
        assert.doesNotMatch(message, /WebGPU EP has an unrecognized option: 'weightLoadAcceleration'/);
        if (weightLoadAcceleration === 'off') {
          // A build without WebGPU, or a machine without a usable adapter, can
          // fail before an acceleration mode matters. Later modes must not hide
          // a different, unexpected failure.
          unavailableProviderError = message;
        } else if (weightLoadAcceleration === 'required') {
          if (message !== unavailableProviderError) {
            assert.match(message, /weightLoadAcceleration|disk-to-GPU|DXGI|D3D12/);
          }
        } else {
          assert.doesNotMatch(message, /weightLoadAcceleration|disk-to-GPU|DXGI|D3D12/);
        }
      }
    }
  });

  it('recognizes the WebGPU disableSubgroupMatrix option', async () => {
    const model = path.join(TEST_DATA_ROOT, 'test_types_float.onnx');

    for (const disableSubgroupMatrix of [true, false]) {
      try {
        const session = await InferenceSession.create(model, {
          executionProviders: [{ name: 'webgpu', disableSubgroupMatrix }],
        });
        await session.release();
      } catch (error) {
        // A build without WebGPU, or a machine without a usable adapter, can fail for unrelated reasons.
        assert.doesNotMatch(String(error), /WebGPU EP has an unrecognized option: 'disableSubgroupMatrix'/);
      }
    }
  });

  MODEL_TEST_TYPES_CASES.forEach((testCase) => {
    it(`${testCase.model}`, async () => {
      const session = await InferenceSession.create(testCase.model);
      const output = await session.run({ input: new Tensor(testCase.type, testCase.input0, [1, 5]) });
      assert(Object.prototype.hasOwnProperty.call(output, 'output'), "'output' should be in the result object.");
      assert(output.output instanceof Tensor, 'result[output] should be a Tensor object.');
      assert.strictEqual(output.output.size, 5, `output size expected 5, got ${output.output.size}.`);
      assert.strictEqual(
        output.output.type,
        testCase.type,
        `tensor type expected ${testCase.type}, got ${output.output.type}.`,
      );
      assert.strictEqual(
        Object.getPrototypeOf(output.output.data),
        Object.getPrototypeOf(testCase.expectedOutput0),
        `tensor data expected ${Object.getPrototypeOf(testCase.expectedOutput0).constructor.name}, got ${
          Object.getPrototypeOf(output.output.data).constructor.name
        }`,
      );
      assertDataEqual(testCase.type, output.output.data, testCase.expectedOutput0);
    });
  });
});

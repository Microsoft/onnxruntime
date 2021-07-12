# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# -*- coding: UTF-8 -*-
import unittest
import os
import numpy as np
import gc

import onnxruntime as onnxrt
import threading
import sys
from helper import get_name
from onnxruntime.capi.onnxruntime_pybind11_state import Fail

class TestInferenceSession(unittest.TestCase):

    def run_model(self, session_object, run_options):
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = session_object.get_inputs()[0].name
        res = session_object.run([], {input_name: x}, run_options=run_options)
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testModelSerialization(self):
        try:
            so = onnxrt.SessionOptions()
            so.log_verbosity_level = 1
            so.logid = "TestModelSerialization"
            so.optimized_model_filepath = "./PythonApiTestOptimizedModel.onnx"
            onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
            self.assertTrue(os.path.isfile(so.optimized_model_filepath))
        except Fail as onnxruntime_error:
            if str(onnxruntime_error) == "[ONNXRuntimeError] : 1 : FAIL : Unable to serialize model as it contains" \
                " compiled nodes. Please disable any execution providers which generate compiled nodes.":
                pass
            else:
                raise onnxruntime_error

    def testGetProviders(self):
        self.assertTrue('CPUExecutionProvider' in onnxrt.get_available_providers())
        # get_all_providers() returns the default EP order from highest to lowest.
        # CPUExecutionProvider should always be last.
        self.assertTrue('CPUExecutionProvider' == onnxrt.get_all_providers()[-1])
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
        self.assertTrue('CPUExecutionProvider' in sess.get_providers())

    def testEnablingAndDisablingTelemetry(self):
        onnxrt.disable_telemetry_events()

        # no-op on non-Windows builds
        # may be no-op on certain Windows builds based on build configuration
        onnxrt.enable_telemetry_events()

    def testSetProviders(self):
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
            # confirm that CUDA Provider is in list of registered providers.
            self.assertTrue('CUDAExecutionProvider' in sess.get_providers())
            # reset the session and register only CPU Provider.
            sess.set_providers(['CPUExecutionProvider'])
            # confirm only CPU Provider is registered now.
            self.assertEqual(['CPUExecutionProvider'], sess.get_providers())

    def testSetProvidersWithOptions(self):
        if 'TensorrtExecutionProvider' in onnxrt.get_available_providers():
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
            self.assertIn('TensorrtExecutionProvider', sess.get_providers())

            options = sess.get_provider_options()
            option = options['TensorrtExecutionProvider']
            self.assertIn('device_id', option)
            self.assertIn('trt_max_partition_iterations', option)
            self.assertIn('trt_min_subgraph_size', option)
            self.assertIn('trt_max_workspace_size', option)
            self.assertIn('trt_dump_subgraphs', option)
            self.assertIn('trt_engine_cache_enable', option)
            self.assertIn('trt_engine_cache_path', option)
            self.assertIn('trt_force_sequential_engine_build', option)

            max_partition_iterations = option['trt_max_partition_iterations']
            new_max_partition_iterations = int(max_partition_iterations) + 1
            min_subgraph_size = option['trt_min_subgraph_size']
            new_min_subgraph_size = int(min_subgraph_size) + 1
            ori_max_workspace_size = option['trt_max_workspace_size']
            new_max_workspace_size = int(ori_max_workspace_size) // 2

            option = {}
            option['trt_max_partition_iterations'] = new_max_partition_iterations
            option['trt_min_subgraph_size'] = new_min_subgraph_size
            option['trt_max_workspace_size'] = new_max_workspace_size
            dump_subgraphs = "true"
            option['trt_dump_subgraphs'] = dump_subgraphs
            engine_cache_enable = "true"
            option['trt_engine_cache_enable'] = engine_cache_enable
            engine_cache_path = './engine_cache'
            option['trt_engine_cache_path'] = engine_cache_path
            force_sequential_engine_build = "true"
            option['trt_force_sequential_engine_build'] = force_sequential_engine_build
            sess.set_providers(['TensorrtExecutionProvider'], [option])

            options = sess.get_provider_options()
            option = options['TensorrtExecutionProvider']
            self.assertEqual(option['trt_max_partition_iterations'], str(new_max_partition_iterations))
            self.assertEqual(option['trt_min_subgraph_size'], str(new_min_subgraph_size))
            self.assertEqual(option['trt_max_workspace_size'], str(new_max_workspace_size))
            self.assertEqual(option['trt_dump_subgraphs'], '1')
            self.assertEqual(option['trt_engine_cache_enable'], '1')
            self.assertEqual(option['trt_engine_cache_path'], str(engine_cache_path))
            self.assertEqual(option['trt_force_sequential_engine_build'], '1')

            # We currently disable following test code since that not all test machines/GPUs have nvidia int8 capability

            '''
            int8_use_native_calibration_table = "false"
            option['trt_int8_use_native_calibration_table'] = int8_use_native_calibration_table 
            int8_enable = "true"
            option['trt_int8_enable'] = int8_enable
            calib_table_name = '/home/onnxruntime/table.flatbuffers' # this file is not existed
            option['trt_int8_calibration_table_name'] = calib_table_name
            with self.assertRaises(RuntimeError):
                sess.set_providers(['TensorrtExecutionProvider'], [option])
            '''

        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            import sys
            import ctypes
            CUDA_SUCCESS = 0
            def runBaseTest1():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
                self.assertTrue('CUDAExecutionProvider' in sess.get_providers())

                option1 = {'device_id': 0}
                sess.set_providers(['CUDAExecutionProvider'], [option1])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())
                option2 = {'device_id': -1}
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option2])
                sess.set_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'], [option1, {}])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())

            def runBaseTest2():
                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
                self.assertIn('CUDAExecutionProvider', sess.get_providers())

                # test get/set of "gpu_mem_limit" configuration.
                options = sess.get_provider_options()
                self.assertIn('CUDAExecutionProvider', options)
                option = options['CUDAExecutionProvider']
                self.assertIn('gpu_mem_limit', option)
                ori_mem_limit = option['gpu_mem_limit']
                new_mem_limit = int(ori_mem_limit) // 2
                option['gpu_mem_limit'] = new_mem_limit
                sess.set_providers(['CUDAExecutionProvider'], [option])
                options = sess.get_provider_options()
                self.assertEqual(options['CUDAExecutionProvider']['gpu_mem_limit'], str(new_mem_limit))

                option['gpu_mem_limit'] = ori_mem_limit
                sess.set_providers(['CUDAExecutionProvider'], [option])
                options = sess.get_provider_options()
                self.assertEqual(options['CUDAExecutionProvider']['gpu_mem_limit'], ori_mem_limit)

                def test_get_and_set_option_with_values(option_name, option_values):
                    provider_options = sess.get_provider_options()
                    self.assertIn('CUDAExecutionProvider', provider_options)
                    cuda_options = options['CUDAExecutionProvider']
                    self.assertIn(option_name, cuda_options)
                    for option_value in option_values:
                        cuda_options[option_name] = option_value
                        sess.set_providers(['CUDAExecutionProvider'], [cuda_options])
                        new_provider_options = sess.get_provider_options()
                        self.assertEqual(
                            new_provider_options.get('CUDAExecutionProvider', {}).get(option_name),
                            str(option_value))

                test_get_and_set_option_with_values(
                    'arena_extend_strategy', ['kNextPowerOfTwo', 'kSameAsRequested'])

                test_get_and_set_option_with_values(
                    'cudnn_conv_algo_search', ["DEFAULT", "EXHAUSTIVE", "HEURISTIC"])

                test_get_and_set_option_with_values(
                    'do_copy_in_default_stream', [0, 1])

                option['gpu_external_alloc'] = '0'
                option['gpu_external_free'] = '0'
                sess.set_providers(['CUDAExecutionProvider'], [option])
                options = sess.get_provider_options()
                self.assertEqual(options['CUDAExecutionProvider']['gpu_external_alloc'], '0')
                self.assertEqual(options['CUDAExecutionProvider']['gpu_external_free'], '0')
                #
                # Note: Tests that throw an exception leave an empty session due to how set_providers currently works,
                #       so run them last. Each set_providers call will attempt to re-create a session, so it's
                #       fine for a test that fails to run immediately after another one that fails.
                #       Alternatively a valid call to set_providers could be used to recreate the underlying session
                #       after a failed call.
                #
                option['arena_extend_strategy'] = 'wrong_value'
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                option['gpu_mem_limit'] = -1024
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                option['gpu_mem_limit'] = 1024.1024
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                option['gpu_mem_limit'] = 'wrong_value'
                with self.assertRaises(RuntimeError):
                    sess.set_providers(['CUDAExecutionProvider'], [option])

            def getCudaDeviceCount():
                import ctypes

                num_device = ctypes.c_int()
                result = ctypes.c_int()
                error_str = ctypes.c_char_p()

                result = cuda.cuInit(0)
                result = cuda.cuDeviceGetCount(ctypes.byref(num_device))
                if result != CUDA_SUCCESS:
                    cuda.cuGetErrorString(result, ctypes.byref(error_str))
                    print("cuDeviceGetCount failed with error code %d: %s" % (result, error_str.value.decode()))
                    return -1

                return num_device.value

            def setDeviceIdTest(i):
                import ctypes
                import onnxruntime as onnxrt

                device = ctypes.c_int()
                result = ctypes.c_int()
                error_str = ctypes.c_char_p()

                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
                option = {'device_id': i}
                sess.set_providers(['CUDAExecutionProvider'], [option])
                self.assertEqual(['CUDAExecutionProvider', 'CPUExecutionProvider'], sess.get_providers())
                result = cuda.cuCtxGetDevice(ctypes.byref(device))
                if result != CUDA_SUCCESS:
                    cuda.cuGetErrorString(result, ctypes.byref(error_str))
                    print("cuCtxGetDevice failed with error code %d: %s" % (result, error_str.value.decode()))

                self.assertEqual(result, CUDA_SUCCESS)
                self.assertEqual(i, device.value)

            def runAdvancedTest():
                num_device = getCudaDeviceCount()
                if num_device < 0:
                    return

                # Configure session to be ready to run on all available cuda devices
                for i in range(num_device):
                    setDeviceIdTest(i)

                sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))

                # configure session with invalid option values and that should fail
                with self.assertRaises(RuntimeError):
                    option = {'device_id': num_device}
                    sess.set_providers(['CUDAExecutionProvider'], [option])
                    option = {'device_id': 'invalid_value'}
                    sess.set_providers(['CUDAExecutionProvider'], [option])

                # configure session with invalid option should fail
                with self.assertRaises(RuntimeError):
                    option = {'invalid_option': 123}
                    sess.set_providers(['CUDAExecutionProvider'], [option])

            libnames = ('libcuda.so', 'libcuda.dylib', 'cuda.dll')
            for libname in libnames:
                try:
                    cuda = ctypes.CDLL(libname)
                    runBaseTest1()
                    runBaseTest2()
                    runAdvancedTest()

                except OSError:
                    continue
                else:
                    break
            else:
                runBaseTest1()
                runBaseTest2()
                # raise OSError("could not load any of: " + ' '.join(libnames))

    def testInvalidSetProviders(self):
        with self.assertRaises(RuntimeError) as context:
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
            sess.set_providers(['InvalidProvider'])
        self.assertTrue('Unknown Provider Type: InvalidProvider' in str(context.exception))

    def testSessionProviders(self):
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            # create session from scratch, but constrain it to only use the CPU.
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), providers=['CPUExecutionProvider'])
            self.assertEqual(['CPUExecutionProvider'], sess.get_providers())

    def testRunModel(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 2])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModelFromBytes(self):
        with open(get_name("mul_1.onnx"), "rb") as f:
            content = f.read()
        sess = onnxrt.InferenceSession(content)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 2])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testRunModel2Contiguous(self):
        sess = onnxrt.InferenceSession(get_name("matmul_1.onnx"))
        x = np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float32)[:, [1, 0]]
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        self.assertEqual(input_shape, [3, 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [3, 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)
        xcontiguous = np.ascontiguousarray(x)
        rescontiguous = sess.run([output_name], {input_name: xcontiguous})
        np.testing.assert_allclose(output_expected, rescontiguous[0], rtol=1e-05, atol=1e-08)

    def testRunModelMultipleThreads(self):
        available_providers = onnxrt.get_available_providers()

        # Skip this test for a "pure" DML onnxruntime python wheel. We keep this test enabled for instances where both DML and CUDA
        # EPs are available (Windows GPU CI pipeline has this config) - this test will pass because CUDA has higher precendence than DML
        # and the nodes are assigned to only the CUDA EP (which supports this test)
        if ('DmlExecutionProvider' in available_providers and not 'CUDAExecutionProvider' in available_providers):
            print("Skipping testRunModelMultipleThreads as the DML EP does not support calling Run() on different threads using the same session object ")
        else:
            so = onnxrt.SessionOptions()
            so.log_verbosity_level = 1
            so.logid = "MultiThreadsTest"
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
            ro1 = onnxrt.RunOptions()
            ro1.logid = "thread1"
            t1 = threading.Thread(target=self.run_model, args=(sess, ro1))
            ro2 = onnxrt.RunOptions()
            ro2.logid = "thread2"
            t2 = threading.Thread(target=self.run_model, args=(sess, ro2))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

    def testListAsInput(self):
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        res = sess.run([], {input_name: x.tolist()})
        output_expected = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testStringListAsInput(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'], dtype=str).reshape((2, 2))
        x_name = sess.get_inputs()[0].name
        res = sess.run([], {x_name: x.tolist()})
        np.testing.assert_equal(x, res[0])

    def testRunDevice(self):
        device = onnxrt.get_device()
        self.assertTrue('CPU' in device or 'GPU' in device)

    def testRunModelSymbolicInput(self):
        sess = onnxrt.InferenceSession(get_name("matmul_2.onnx"))
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "X")
        input_shape = sess.get_inputs()[0].shape
        # Input X has an unknown dimension.
        self.assertEqual(input_shape, ['None', 2])
        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_shape = sess.get_outputs()[0].shape
        # Output X has an unknown dimension.
        self.assertEqual(output_shape, ['None', 1])
        res = sess.run([output_name], {input_name: x})
        output_expected = np.array([[5.0], [11.0], [17.0]], dtype=np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

    def testBooleanInputs(self):
        sess = onnxrt.InferenceSession(get_name("logicaland.onnx"))
        a = np.array([[True, True], [False, False]], dtype=bool)
        b = np.array([[True, False], [True, False]], dtype=bool)

        # input1:0 is first in the protobuf, and input:0 is second
        # and we maintain the original order.
        a_name = sess.get_inputs()[0].name
        self.assertEqual(a_name, "input1:0")
        a_shape = sess.get_inputs()[0].shape
        self.assertEqual(a_shape, [2, 2])
        a_type = sess.get_inputs()[0].type
        self.assertEqual(a_type, 'tensor(bool)')

        b_name = sess.get_inputs()[1].name
        self.assertEqual(b_name, "input:0")
        b_shape = sess.get_inputs()[1].shape
        self.assertEqual(b_shape, [2, 2])
        b_type = sess.get_inputs()[0].type
        self.assertEqual(b_type, 'tensor(bool)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(bool)')

        output_expected = np.array([[True, False], [False, False]], dtype=bool)
        res = sess.run([output_name], {a_name: a, b_name: b})
        np.testing.assert_equal(output_expected, res[0])

    def testStringInput1(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'], dtype=str).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0])

    def testStringInput2(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['Olá', '你好', '여보세요', 'hello'], dtype=str).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0])

    def testInputBytes(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array([b'this', b'is', b'identity', b'test']).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0].astype('|S8'))

    def testInputObject(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        x = np.array(['this', 'is', 'identity', 'test'], object).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})
        np.testing.assert_equal(x, res[0])

    def testInputVoid(self):
        sess = onnxrt.InferenceSession(get_name("identity_string.onnx"))
        # numpy 1.20+ doesn't automatically pad the bytes based entries in the array when dtype is np.void,
        # so we use inputs where that is the case
        x = np.array([b'must', b'have', b'same', b'size'], dtype=np.void).reshape((2, 2))

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "input:0")
        x_shape = sess.get_inputs()[0].shape
        self.assertEqual(x_shape, [2, 2])
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'tensor(string)')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output:0")
        output_shape = sess.get_outputs()[0].shape
        self.assertEqual(output_shape, [2, 2])
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(string)')

        res = sess.run([output_name], {x_name: x})

        expr = np.array([['must', 'have'], ['same', 'size']], dtype=object)
        np.testing.assert_equal(expr, res[0])

    def testRaiseWrongNumInputs(self):
        with self.assertRaises(ValueError) as context:
            sess = onnxrt.InferenceSession(get_name("logicaland.onnx"))
            a = np.array([[True, True], [False, False]], dtype=bool)
            res = sess.run([], {'input:0': a})

        self.assertTrue('Model requires 2 inputs' in str(context.exception))

    def testModelMeta(self):
        model_path = "../models/opset8/test_squeezenet/model.onnx"
        if not os.path.exists(model_path):
            return
        sess = onnxrt.InferenceSession(model_path)
        modelmeta = sess.get_modelmeta()
        self.assertEqual('onnx-caffe2', modelmeta.producer_name)
        self.assertEqual('squeezenet_old', modelmeta.graph_name)
        self.assertEqual('', modelmeta.domain)
        self.assertEqual('', modelmeta.description)
        self.assertEqual('', modelmeta.graph_description)

    def testProfilerWithSessionOptions(self):
        so = onnxrt.SessionOptions()
        so.enable_profiling = True
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
        x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        sess.run([], {'X': x})
        profile_file = sess.end_profiling()

        tags = ['pid', 'dur', 'ts', 'ph', 'X', 'name', 'args']
        with open(profile_file) as f:
            lines = f.readlines()
            self.assertTrue('[' in lines[0])
            for i in range(1, 8):
                for tag in tags:
                    self.assertTrue(tag in lines[i])
            self.assertTrue(']' in lines[8])

    def testProfilerGetStartTimeNs(self):
        def getSingleSessionProfilingStartTime():
            so = onnxrt.SessionOptions()
            so.enable_profiling = True
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so)
            return sess.get_profiling_start_time_ns()

        # Get 1st profiling's start time
        start_time_1 = getSingleSessionProfilingStartTime()
        # Get 2nd profiling's start time
        start_time_2 = getSingleSessionProfilingStartTime()
        # Get 3rd profiling's start time
        start_time_3 = getSingleSessionProfilingStartTime()

        # Chronological profiling's start time
        self.assertTrue(start_time_1 <= start_time_2 <= start_time_3)

    def testGraphOptimizationLevel(self):
        opt = onnxrt.SessionOptions()
        # default should be all optimizations optimization
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL)
        opt.graph_optimization_level = onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        self.assertEqual(opt.graph_optimization_level, onnxrt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED)
        sess = onnxrt.InferenceSession(get_name("logicaland.onnx"), sess_options=opt)
        a = np.array([[True, True], [False, False]], dtype=bool)
        b = np.array([[True, False], [True, False]], dtype=bool)

        res = sess.run([], {'input1:0': a, 'input:0': b})

    def testSequenceLength(self):
        sess = onnxrt.InferenceSession(get_name("sequence_length.onnx"))
        x = [
            np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3)),
            np.array([1.0, 0.0, 3.0, 44.0, 23.0, 11.0], dtype=np.float32).reshape((2, 3))
        ]

        x_name = sess.get_inputs()[0].name
        self.assertEqual(x_name, "X")
        x_type = sess.get_inputs()[0].type
        self.assertEqual(x_type, 'seq(tensor(float))')

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "Y")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'tensor(int64)')

        output_expected = np.array(2, dtype=np.int64)
        res = sess.run([output_name], {x_name: x})
        self.assertEqual(output_expected, res[0])

    def testSequenceConstruct(self):
        sess = onnxrt.InferenceSession(get_name("sequence_construct.onnx"))

        self.assertEqual(sess.get_inputs()[0].type, 'tensor(int64)')
        self.assertEqual(sess.get_inputs()[1].type, 'tensor(int64)')

        self.assertEqual(sess.get_inputs()[0].name, "tensor1")
        self.assertEqual(sess.get_inputs()[1].name, "tensor2")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(tensor(int64))')

        output_expected = [
            np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
            np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3))
        ]

        res = sess.run(
            [output_name], {
                "tensor1": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
                "tensor2": np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((2, 3))
            })

        np.testing.assert_array_equal(output_expected, res[0])

    def testSequenceInsert(self):
        opt = onnxrt.SessionOptions()
        opt.execution_mode = onnxrt.ExecutionMode.ORT_SEQUENTIAL
        sess = onnxrt.InferenceSession(get_name("sequence_insert.onnx"), sess_options=opt)

        self.assertEqual(sess.get_inputs()[0].type, 'seq(tensor(int64))')
        self.assertEqual(sess.get_inputs()[1].type, 'tensor(int64)')

        self.assertEqual(sess.get_inputs()[0].name, "input_seq")
        self.assertEqual(sess.get_inputs()[1].name, "tensor")

        output_name = sess.get_outputs()[0].name
        self.assertEqual(output_name, "output_sequence")
        output_type = sess.get_outputs()[0].type
        self.assertEqual(output_type, 'seq(tensor(int64))')

        output_expected = [np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3))]
        res = sess.run([output_name], {
            "tensor": np.array([1, 0, 3, 44, 23, 11], dtype=np.int64).reshape((2, 3)),
            "input_seq": []
        })
        np.testing.assert_array_equal(output_expected, res[0])

    def testOrtExecutionMode(self):
        opt = onnxrt.SessionOptions()
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_SEQUENTIAL)
        opt.execution_mode = onnxrt.ExecutionMode.ORT_PARALLEL
        self.assertEqual(opt.execution_mode, onnxrt.ExecutionMode.ORT_PARALLEL)

    def testLoadingSessionOptionsFromModel(self):
        try:
            os.environ['ORT_LOAD_CONFIG_FROM_MODEL'] = str(1)
            sess = onnxrt.InferenceSession(get_name("model_with_valid_ort_config_json.onnx"))
            session_options = sess.get_session_options()

            self.assertEqual(session_options.inter_op_num_threads, 5)  # from the ORT config

            self.assertEqual(session_options.intra_op_num_threads, 2)  # from the ORT config

            self.assertEqual(session_options.execution_mode,
                             onnxrt.ExecutionMode.ORT_SEQUENTIAL)  # default option (not from the ORT config)

            self.assertEqual(session_options.graph_optimization_level,
                             onnxrt.GraphOptimizationLevel.ORT_ENABLE_ALL)  # from the ORT config

            self.assertEqual(session_options.enable_profiling, True)  # from the ORT config

        except Exception:
            raise

        finally:
            # Make sure the usage of the feature is disabled after this test
            os.environ['ORT_LOAD_CONFIG_FROM_MODEL'] = str(0)

    def testSessionOptionsAddFreeDimensionOverrideByDenotation(self):
        so = onnxrt.SessionOptions()
        so.add_free_dimension_override_by_denotation("DATA_BATCH", 3)
        so.add_free_dimension_override_by_denotation("DATA_CHANNEL", 5)
        sess = onnxrt.InferenceSession(get_name("abs_free_dimensions.onnx"), so)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "x")
        input_shape = sess.get_inputs()[0].shape
        # Free dims with denotations - "DATA_BATCH" and "DATA_CHANNEL" have values assigned to them.
        self.assertEqual(input_shape, [3, 5, 5])

    def testSessionOptionsAddFreeDimensionOverrideByName(self):
        so = onnxrt.SessionOptions()
        so.add_free_dimension_override_by_name("Dim1", 4)
        so.add_free_dimension_override_by_name("Dim2", 6)
        sess = onnxrt.InferenceSession(get_name("abs_free_dimensions.onnx"), so)
        input_name = sess.get_inputs()[0].name
        self.assertEqual(input_name, "x")
        input_shape = sess.get_inputs()[0].shape
        # "Dim1" and "Dim2" have values assigned to them.
        self.assertEqual(input_shape, [4, 6, 5])

    def testSessionOptionsAddConfigEntry(self):
        so = onnxrt.SessionOptions()
        key = "CONFIG_KEY"
        val = "CONFIG_VAL"
        so.add_session_config_entry(key, val)
        self.assertEqual(so.get_session_config_entry(key), val)

    def testInvalidSessionOptionsConfigEntry(self):
        so = onnxrt.SessionOptions()
        invalide_key = "INVALID_KEY"
        with self.assertRaises(RuntimeError) as context:
            so.get_session_config_entry(invalide_key)
        self.assertTrue(
            'SessionOptions does not have configuration with key: ' + invalide_key in str(context.exception))

    def testSessionOptionsAddInitializer(self):
        # Create an initializer and add it to a SessionOptions instance
        so = onnxrt.SessionOptions()
        # This initializer is different from the actual initializer in the model for "W"
        ortvalue_initializer = onnxrt.OrtValue.ortvalue_from_numpy(np.array([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=np.float32))
        # The user should manage the life cycle of this OrtValue and should keep it in scope
        # as long as any session that is going to be reliant on it is in scope
        so.add_initializer("W", ortvalue_initializer)

        # Create an InferenceSession that only uses the CPU EP and validate that it uses the
        # initializer provided via the SessionOptions instance (overriding the model initializer)
        # We only use the CPU EP because the initializer we created is on CPU and we want the model to use that
        sess = onnxrt.InferenceSession(get_name("mul_1.onnx"), so, ['CPUExecutionProvider'])
        res = sess.run(["Y"], {"X": np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)})
        self.assertTrue(np.array_equal(res[0], np.array([[2.0, 2.0], [12.0, 12.0], [30.0, 30.0]], dtype=np.float32)))

    def testRegisterCustomOpsLibrary(self):
        if sys.platform.startswith("win"):
            shared_library = 'custom_op_library.dll'
            if not os.path.exists(shared_library):
                raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

        elif sys.platform.startswith("darwin"):
            shared_library = 'libcustom_op_library.dylib'
            if not os.path.exists(shared_library):
                raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

        else:
            shared_library = './libcustom_op_library.so'
            if not os.path.exists(shared_library):
                raise FileNotFoundError("Unable to find '{0}'".format(shared_library))

        this = os.path.dirname(__file__)
        custom_op_model = os.path.join(this, "testdata", "custom_op_library", "custom_op_test.onnx")
        if not os.path.exists(custom_op_model):
            raise FileNotFoundError("Unable to find '{0}'".format(custom_op_model))

        so1 = onnxrt.SessionOptions()
        so1.register_custom_ops_library(shared_library)

        # Model loading successfully indicates that the custom op node could be resolved successfully
        sess1 = onnxrt.InferenceSession(custom_op_model, so1)
        #Run with input data
        input_name_0 = sess1.get_inputs()[0].name
        input_name_1 = sess1.get_inputs()[1].name
        output_name = sess1.get_outputs()[0].name
        input_0 = np.ones((3,5)).astype(np.float32)
        input_1 = np.zeros((3,5)).astype(np.float32)
        res = sess1.run([output_name], {input_name_0: input_0, input_name_1: input_1})
        output_expected = np.ones((3,5)).astype(np.float32)
        np.testing.assert_allclose(output_expected, res[0], rtol=1e-05, atol=1e-08)

        # Create an alias of SessionOptions instance
        # We will use this alias to construct another InferenceSession
        so2 = so1

        # Model loading successfully indicates that the custom op node could be resolved successfully
        sess2 = onnxrt.InferenceSession(custom_op_model, so2)

        # Create another SessionOptions instance with the same shared library referenced
        so3 = onnxrt.SessionOptions()
        so3.register_custom_ops_library(shared_library)
        sess3 = onnxrt.InferenceSession(custom_op_model, so3)

    def testOrtValue(self):

        numpy_arr_input = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        numpy_arr_output = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]], dtype=np.float32)

        def test_session_with_ortvalue_input(ortvalue):
            sess = onnxrt.InferenceSession(get_name("mul_1.onnx"))
            res = sess.run(["Y"], {"X": ortvalue})
            self.assertTrue(np.array_equal(res[0], numpy_arr_output))

        ortvalue1 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input)
        self.assertEqual(ortvalue1.device_name(), "cpu")
        self.assertEqual(ortvalue1.shape(), [3, 2])
        self.assertEqual(ortvalue1.data_type(), "tensor(float)")
        self.assertEqual(ortvalue1.is_tensor(), True)
        self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

        # Pass in the constructed OrtValue to a session via Run() and check results
        test_session_with_ortvalue_input(ortvalue1)

        # The constructed OrtValue should still be valid after being used in a session
        self.assertTrue(np.array_equal(ortvalue1.numpy(), numpy_arr_input))

        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            ortvalue2 = onnxrt.OrtValue.ortvalue_from_numpy(numpy_arr_input, 'cuda', 0)
            self.assertEqual(ortvalue2.device_name(), "cuda")
            self.assertEqual(ortvalue2.shape(), [3, 2])
            self.assertEqual(ortvalue2.data_type(), "tensor(float)")
            self.assertEqual(ortvalue2.is_tensor(), True)
            self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))

            # Pass in the constructed OrtValue to a session via Run() and check results
            test_session_with_ortvalue_input(ortvalue2)

            # The constructed OrtValue should still be valid after being used in a session
            self.assertTrue(np.array_equal(ortvalue2.numpy(), numpy_arr_input))
    def testNumpyToCudaCpu(self):
        shape = [2,2]
        cuda_device = onnxrt.OrtDevice.make('cuda', 0)
        data_to_copy = np.array([1, 2, 3, 4], np.int32).reshape(shape)
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            cuda_result = onnxrt.numpy_array_to_cuda(cuda_device, data_to_copy)
            self.assertEqual(list(cuda_result.shape), shape)
            self.assertEqual(cuda_result.dtype, np.dtype(np.int32))

            cpu_result = onnxrt.numpy_array_to_cpu(cuda_result)
            self.assertEqual(list(cpu_result.shape), shape)
            self.assertEqual(cpu_result.dtype, np.dtype(np.int32))
            self.assertTrue(np.array_equal(cpu_result, data_to_copy))

            # Test unsupported dtype
            unsupported_str = np.array(['xyz', 'zyx', 'xzy', 'zxy'], np.dtype(str)).reshape(shape)
            unsupported_obj = np.array(['xyz', 'zyx', 'xzy', 'zxy'], np.dtype(object)).reshape(shape)
            with self.assertRaises(RuntimeError):
                onnxrt.numpy_array_to_cuda(cuda_device, unsupported_str)
            with self.assertRaises(RuntimeError):
                onnxrt.numpy_array_to_cuda(cuda_device, unsupported_obj)
        else:
            # No CUDA
            with self.assertRaises(RuntimeError):
                onnxrt.numpy_array_to_cuda(cuda_device, data_to_copy)

    def testSparseTensorCooFormat(self):
        cpu_device = onnxrt.OrtDevice.make('cpu', 0)
        shape = [9,9]
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        # Linear indices
        indices = np.array([3, 5, 15], dtype=np.int64)
        sparse_tensor = onnxrt.SparseTensor.sparse_coo_from_numpy(shape, values, indices, cpu_device)
        self.assertEqual(sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_COO)
        self.assertEqual(sparse_tensor.shape(), shape)
        self.assertEqual(sparse_tensor.data_type(), "sparse_tensor(float)")
        self.assertEqual(sparse_tensor.device_name(), 'cpu')

        # Get Data View on a numeric type.
        values_ret = sparse_tensor.values()
        self.assertFalse(values_ret.flags.writeable)
        indices_ret = sparse_tensor.as_coo_view().indices()
        self.assertFalse(indices_ret.flags.writeable)
        # Run GC to test that values_ret still exhibits expected data
        gc.collect()
        self.assertTrue(np.array_equal(values, values_ret))
        self.assertTrue(np.array_equal(indices, indices_ret))

        # Test new Ortvalue interfaces
        ort_value = onnxrt.OrtValue.ort_value_from_sparse_tensor(sparse_tensor)
        sparse_tensor = ort_value.as_sparse_tensor()
        values_ret = sparse_tensor.values()
        self.assertFalse(values_ret.flags.writeable)
        indices_ret = sparse_tensor.as_coo_view().indices()
        self.assertFalse(indices_ret.flags.writeable)
        gc.collect()

        # Test string data on cpu only, need to subst values only
        str_values = np.array(['xyz', 'yxz', 'zyx'], dtype=str)
        str_sparse_tensor = onnxrt.SparseTensor.sparse_coo_from_numpy(shape, str_values, indices, cpu_device)
        self.assertEqual(str_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_COO)
        self.assertEqual(str_sparse_tensor.shape(), shape)
        self.assertEqual(str_sparse_tensor.data_type(), "sparse_tensor(string)")
        self.assertEqual(str_sparse_tensor.device_name(), 'cpu')

        # Get string values back
        str_values_ret = str_sparse_tensor.values()
        self.assertTrue(np.array_equal(str_values, str_values_ret))
        # Check indices
        str_indices_ret = str_sparse_tensor.as_coo_view().indices()
        gc.collect()
        self.assertFalse(str_indices_ret.flags.writeable)
        self.assertTrue(np.array_equal(indices, str_indices_ret))

        cuda_device = onnxrt.OrtDevice.make('cuda', 0)
        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            values_on_gpu = onnxrt.numpy_array_to_cuda(cuda_device, values)
            indices_on_gpu = onnxrt.numpy_array_to_cuda(cuda_device, indices)
            cuda_sparse_tensor = onnxrt.SparseTensor.sparse_coo_from_numpy(shape, values_on_gpu, indices_on_gpu, cuda_device)
            self.assertEqual(cuda_sparse_tensor.device_name(), 'cuda')
            # Test to_cuda
            copy_on_cuda = sparse_tensor.to_cuda(cuda_device)
            self.assertEqual(copy_on_cuda.shape(), shape)
            self.assertEqual(copy_on_cuda.data_type(), "sparse_tensor(float)")
            self.assertEqual(copy_on_cuda.device_name(), 'cuda')

            # Test that gpu copy would fail to copy to cuda
            with self.assertRaises(RuntimeError):
                copy_on_cuda.to_cuda(cuda_device)
            # Test that string tensor copy would fail
            with self.assertRaises(RuntimeError):
                str_sparse_tensor.to_cuda(cuda_device)
        else:
            # No cuda available
            with self.assertRaises(RuntimeError):
                sparse_tensor.to_cuda(cuda_device)

    def testSparseTensorCsrFormat(self):
        cpu_device = onnxrt.OrtDevice.make('cpu', 0)
        shape = [9,9]
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        inner_indices = np.array([1, 1, 1], dtype=np.int64)
        outer_indices = np.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 3], dtype=np.int64)
        sparse_tensor = onnxrt.SparseTensor.sparse_csr_from_numpy(shape, values, inner_indices, outer_indices, cpu_device)
        self.assertEqual(sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
        self.assertEqual(sparse_tensor.shape(), shape)
        self.assertEqual(sparse_tensor.data_type(), "sparse_tensor(float)")
        self.assertEqual(sparse_tensor.device_name(), 'cpu')

        # Test CSR(C) indices
        inner_indices_ret = sparse_tensor.as_csrc_view().inner()
        outer_indices_ret = sparse_tensor.as_csrc_view().outer()
        self.assertFalse(inner_indices_ret.flags.writeable)
        self.assertFalse(outer_indices_ret.flags.writeable)
        gc.collect()
        self.assertTrue(np.array_equal(inner_indices, inner_indices_ret))
        self.assertTrue(np.array_equal(outer_indices, outer_indices_ret))

        # Test with strings
        str_values = np.array(['xyz', 'yxz', 'zyx'], dtype=str)
        str_sparse_tensor = onnxrt.SparseTensor.sparse_csr_from_numpy(shape, str_values, inner_indices, outer_indices, cpu_device)
        self.assertEqual(str_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
        self.assertEqual(str_sparse_tensor.shape(), shape)
        self.assertEqual(str_sparse_tensor.data_type(), "sparse_tensor(string)")
        self.assertEqual(str_sparse_tensor.device_name(), 'cpu')

        if 'CUDAExecutionProvider' in onnxrt.get_available_providers():
            cuda_device = onnxrt.OrtDevice.make('cuda', 0)
            values_on_gpu = onnxrt.numpy_array_to_cuda(cuda_device, values)
            inner_indices_on_gpu = onnxrt.numpy_array_to_cuda(cuda_device, inner_indices)
            outer_indices_on_gpu = onnxrt.numpy_array_to_cuda(cuda_device, outer_indices)
            cuda_sparse_tensor = onnxrt.SparseTensor.sparse_csr_from_numpy(shape, values_on_gpu, inner_indices_on_gpu, outer_indices_on_gpu, cuda_device)
            self.assertEqual(cuda_sparse_tensor.device_name(), 'cuda')
            self.assertEqual(cuda_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
            self.assertEqual(cuda_sparse_tensor.shape(), shape)
            self.assertEqual(cuda_sparse_tensor.data_type(), "sparse_tensor(float)")

            cuda_sparse_tensor = sparse_tensor.to_cuda(cuda_device)
            self.assertEqual(cuda_sparse_tensor.device_name(), 'cuda')
            self.assertEqual(cuda_sparse_tensor.format(), onnxrt.OrtSparseFormat.ORT_SPARSE_CSRC)
            self.assertEqual(cuda_sparse_tensor.shape(), shape)
            self.assertEqual(cuda_sparse_tensor.data_type(), "sparse_tensor(float)")


    def testRunModelWithCudaCopyStream(self):
        available_providers = onnxrt.get_available_providers()

        if (not 'CUDAExecutionProvider' in available_providers):
            print("Skipping testRunModelWithCudaCopyStream when CUDA is not available")
        else:
            # adapted from issue #4829 for a race condition when copy is not on default stream
            # note:
            # 1. if there are intermittent failure in this test, something is wrong
            # 2. it's easier to repro on slower GPU (like M60, Geforce 1070)

            # to repro #4829, set the CUDA EP do_copy_in_default_stream option to False
            providers = [("CUDAExecutionProvider", {"do_copy_in_default_stream": True}), "CPUExecutionProvider"]

            session = onnxrt.InferenceSession(get_name("issue4829.onnx"), providers=providers)
            shape = np.array([2,2], dtype=np.int64)
            for iteration in range(100000):
                result = session.run(output_names=['output'], input_feed={'shape': shape})

    def testSharedAllocatorUsingCreateAndRegisterAllocator(self):
        # Create and register an arena based allocator

        # ort_arena_cfg = onnxrt.OrtArenaCfg(0, -1, -1, -1) (create an OrtArenaCfg like this template if you want to use non-default parameters)
        ort_memory_info = onnxrt.OrtMemoryInfo("Cpu", onnxrt.OrtAllocatorType.ORT_ARENA_ALLOCATOR, 0, onnxrt.OrtMemType.DEFAULT)
        # Use this option if using non-default OrtArenaCfg : onnxrt.create_and_register_allocator(ort_memory_info, ort_arena_cfg)
        onnxrt.create_and_register_allocator(ort_memory_info, None)

        # Create a session that will use the registered arena based allocator
        so1 = onnxrt.SessionOptions()
        so1.log_severity_level = 1
        so1.add_session_config_entry("session.use_env_allocators", "1");
        onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so1)

        # Create a session that will NOT use the registered arena based allocator
        so2 = onnxrt.SessionOptions()
        so2.log_severity_level = 1
        onnxrt.InferenceSession(get_name("mul_1.onnx"), sess_options=so2)

    def testCheckAndNormalizeProviderArgs(self):
        from onnxruntime.capi.onnxruntime_inference_collection import check_and_normalize_provider_args

        valid_providers = ["a", "b", "c"]

        def check_success(providers, provider_options, expected_providers, expected_provider_options):
            actual_providers, actual_provider_options = check_and_normalize_provider_args(
                providers, provider_options, valid_providers)
            self.assertEqual(actual_providers, expected_providers)
            self.assertEqual(actual_provider_options, expected_provider_options)

        check_success(None, None, [], [])

        check_success(["a"], None, ["a"], [{}])

        check_success(["a", "b"], None, ["a", "b"], [{}, {}])

        check_success([("a", {1: 2}), "b"], None, ["a", "b"], [{"1": "2"}, {}])

        check_success(["a", "b"], [{1: 2}, {}], ["a", "b"], [{"1": "2"}, {}])

        with self.assertWarns(UserWarning):
            check_success(["a", "b", "a"], [{"x": 1}, {}, {"y": 2}], ["a", "b"], [{"x": "1"}, {}])

        def check_failure(providers, provider_options):
            with self.assertRaises(ValueError):
                check_and_normalize_provider_args(providers, provider_options, valid_providers)

        # disable this test
        # provider not valid
        #check_failure(["d"], None)

        # providers not sequence
        check_failure(3, None)

        # providers value invalid
        check_failure([3], None)

        # provider_options not sequence
        check_failure(["a"], 3)

        # provider_options value invalid
        check_failure(["a"], ["not dict"])

        # providers and provider_options length mismatch
        check_failure(["a", "b"], [{1: 2}])

        # provider options unsupported mixed specification
        check_failure([("a", {1: 2})], [{3: 4}])

    def testRegisterCustomEPsLibrary(self):
        # exclude for macos and linux
        if not sys.platform.startswith("win"):
            return

        shared_library = 'test_execution_provider.dll'
        if not os.path.exists(shared_library):
            raise FileNotFoundError("Unable to find '{0}'".format(shared_library))
        
        this = os.path.dirname(__file__)
        custom_op_model = os.path.join(this, "testdata", "custom_execution_provider_library", "test_model.onnx")
        if not os.path.exists(custom_op_model):
            raise FileNotFoundError("Unable to find '{0}'".format(custom_op_model))

        from onnxruntime.capi import _pybind_state as C
        session_options = C.get_default_session_options()
        sess = C.InferenceSession(session_options, custom_op_model, True, True)
        sess.initialize_session(['my_ep'], 
                        [{'shared_lib_path': shared_library,
                          'device_id':'1', 'some_config':'val'}], 
                        set())
        print("Create session with customize execution provider successfully!")

if __name__ == '__main__':
    unittest.main()

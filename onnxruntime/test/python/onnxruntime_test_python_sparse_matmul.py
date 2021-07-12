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

class TestSparseToDenseMatmul(unittest.TestCase):
    def testRunSparseOutputOnly(self):
        '''
        Try running models using the new run_with_ort_values
        sparse_initializer_as_output.onnx - requires no inputs, but only one output
        that comes from the initializer
        '''
        # The below values are a part of the model
        dense_shape = [3,3]
        values = np.array([1.764052391052246, 0.40015721321105957, 0.978738009929657], np.float)
        indices = np.array([2, 3, 5], np.int64)
        sess = onnxrt.InferenceSession(get_name("sparse_initializer_as_output.onnx"))
        res = sess.run_with_ort_values(["values"], {})
        self.assertEqual(len(res), 1)
        ort_value = res[0]
        self.assertTrue(isinstance(ort_value, onnxrt.OrtValue))
        sparse_output = ort_value.as_sparse_tensor()
        self.assertTrue(isinstance(sparse_output, onnxrt.SparseTensor))
        self.assertEqual(dense_shape, sparse_output.shape())
        self.assertTrue(np.array_equal(values, sparse_output.values()))
        self.assertTrue(np.array_equal(indices, sparse_output.as_coo_rep().indices()))

    def testRunContribSparseMatMul(self):
        '''
        Mutliple sparse COO tensor to dense
        '''
        common_shape = [9,9] # inputs and oputputs same shape
        A_values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                    10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
                    18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
                    26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0,
                    34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0,
                    42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0,
                    50.0, 51.0, 52.0,  53.0], np.float32)
        # 2-D index
        A_indices = np.array([0, 1, 0, 2, 0, 6, 0, 7, 0, 8, 1, 0, 1,
                                1, 1, 2, 1, 6, 1, 7, 1, 8, 2, 0, 2, 1,
                                2, 2, 2, 6, 2, 7, 2, 8, 3, 3, 3, 4, 3,
                                5, 3, 6, 3, 7, 3, 8, 4, 3, 4, 4, 4, 5,
                                4, 6, 4, 7, 4, 8, 5, 3, 5, 4, 5, 5, 5,
                                6, 5, 7, 5, 8, 6, 0, 6, 1, 6, 2, 6, 3,
                                6, 4, 6, 5, 7, 0, 7, 1, 7, 2, 7, 3, 7,
                                4, 7, 5, 8, 0, 8, 1, 8, 2, 8, 3, 8, 4,
                                8, 5], np.int64).reshape((len(A_values), 2))

        cpu_device = onnxrt.OrtDevice.make('cpu', 0)
        sparse_tensor = onnxrt.SparseTensor.sparse_coo_from_numpy(common_shape, A_values, A_indices, cpu_device)
        A_ort_value = onnxrt.OrtValue.ort_value_from_sparse_tensor(sparse_tensor)

        B_data = np.array([0, 1, 2, 0, 0, 0, 3, 4, 5,
                            6, 7, 8, 0, 0, 0, 9, 10, 11,
                            12, 13, 14, 0, 0, 0, 15, 16, 17,
                            0, 0, 0, 18, 19, 20, 21, 22, 23,
                            0, 0, 0, 24, 25, 26, 27, 28, 29,
                            0, 0, 0, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 0, 0, 0,
                            42, 43, 44, 45, 46, 47, 0, 0, 0,
                            48, 49, 50, 51, 52, 53, 0, 0, 0], np.float32).reshape(common_shape)
        B_ort_value = onnxrt.OrtValue.ortvalue_from_numpy(B_data)

        Y_result = np.array([546, 561, 576, 552, 564, 576, 39, 42, 45,
                            1410, 1461, 1512, 1362, 1392, 1422, 201, 222, 243,
                            2274, 2361, 2448, 2172, 2220, 2268, 363, 402, 441,
                            2784, 2850, 2916, 4362, 4485, 4608, 1551, 1608, 1665,
                            3540, 3624, 3708, 5604, 5763, 5922, 2037, 2112, 2187,
                            4296, 4398, 4500, 6846, 7041, 7236, 2523, 2616, 2709,
                            678, 789, 900, 2892, 3012, 3132, 4263, 4494, 4725,
                            786, 915, 1044, 3324, 3462, 3600, 4911, 5178, 5445,
                            894, 1041, 1188, 3756, 3912, 4068, 5559, 5862, 6165], np.float).reshape(common_shape)

        sess = onnxrt.InferenceSession(get_name("sparse_to_dense_matmul.onnx"))
        res = sess.run_with_ort_values(["dense_Y"], { "sparse_A" : A_ort_value, "dense_B" : B_ort_value })
        self.assertEqual(len(res), 1)
        ort_value = res[0]
        self.assertTrue(isinstance(ort_value, onnxrt.OrtValue))
        self.assertTrue(ort_value.is_tensor())
        self.assertEqual(ort_value.data_type(), "tensor(float)")
        self.assertEqual(ort_value.shape(), common_shape)
        result = ort_value.numpy()
        self.assertEqual(list(result.shape), common_shape)
        self.assertTrue(np.array_equal(Y_result, result))

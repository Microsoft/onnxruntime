# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import subprocess
from logger import get_logger

log = get_logger("amd_hipify")

contrib_ops_path = 'onnxruntime/contrib_ops'
providers_path = 'onnxruntime/core/providers'
training_ops_path = 'orttraining/orttraining/training_ops'

contrib_ops_excluded_files = [
                    'bert/attention.cc',
                    'bert/attention.h',
                    'bert/attention_impl.cu',
                    'bert/attention_impl.h',
                    'bert/attention_transpose.cu',
                    'bert/attention_past.cu',
                    'bert/embed_layer_norm.cc',
                    'bert/embed_layer_norm.h',
                    'bert/embed_layer_norm_impl.cu',
                    'bert/embed_layer_norm_impl.h',
                    'bert/fast_gelu_impl.cu',
                    'bert/layer_norm.cuh',
                    'bert/longformer_attention.cc',
                    'bert/longformer_attention.h',
                    'bert/longformer_attention_softmax.cu',
                    'bert/longformer_attention_softmax.h',
                    'bert/longformer_attention_impl.cu',
                    'bert/longformer_attention_impl.h',
                    'bert/longformer_global_impl.cu',
                    'bert/longformer_global_impl.h',
                    'math/bias_softmax.cc',
                    'math/bias_softmax.h',
                    'math/bias_softmax_impl.cu',
                    'math/complex_mul.cc',
                    'math/complex_mul.h',
                    'math/complex_mul_impl.cu',
                    'math/complex_mul_impl.h',
                    'math/cufft_plan_cache.h',
                    'math/fft_ops.cc',
                    'math/fft_ops.h',
                    'math/fft_ops_impl.cu',
                    'math/fft_ops_impl.h',
                    'quantization/attention_quantization.cc',
                    'quantization/attention_quantization.h',
                    'quantization/attention_quantization_impl.cu',
                    'quantization/attention_quantization_impl.cuh',
                    'quantization/quantize_dequantize_linear.cc',
                    'tensor/crop.cc',
                    'tensor/crop.h',
                    'tensor/crop_impl.cu',
                    'tensor/crop_impl.h',
                    'tensor/dynamicslice.cc',
                    'tensor/image_scaler.cc',
                    'tensor/image_scaler.h',
                    'tensor/image_scaler_impl.cu',
                    'tensor/image_scaler_impl.h',
                    'conv_transpose_with_dynamic_pads.cc',
                    'conv_transpose_with_dynamic_pads.h',
                    'cuda_contrib_kernels.cc',
                    'cuda_contrib_kernels.h',
                    'inverse.cc',
                    'fused_conv.cc'
]

provider_excluded_files = [
                'atomic/common.cuh',
                'controlflow/if.cc',
                'controlflow/if.h',
                'controlflow/loop.cc',
                'controlflow/loop.h',
                'controlflow/scan.cc',
                'controlflow/scan.h',
                'cu_inc/common.cuh',
                'math/einsum_utils/einsum_auxiliary_ops.cc',
                'math/einsum_utils/einsum_auxiliary_ops.h',
                'math/einsum_utils/einsum_auxiliary_ops_diagonal.cu',
                'math/einsum_utils/einsum_auxiliary_ops_diagonal.h',
                'math/einsum.cc',
                'math/einsum.h',
                'math/gemm.cc',
                'math/matmul.cc',
                'math/matmul_integer.cc',
                'math/matmul_integer.cu',
                'math/matmul_integer.cuh',
                'math/matmul_integer.h',
                'math/softmax_impl.cu',
                'math/softmax.cc',
                'math/topk.cc',
                'math/topk.h',
                'math/topk_impl.cu',
                'math/topk_impl.h',
                'nn/batch_norm.cc',
                'nn/batch_norm.h',
                'nn/conv.cc',
                'nn/conv.h',
                'nn/conv_transpose.cc',
                'nn/conv_transpose.h',
                'nn/instance_norm.cc',
                'nn/instance_norm.h',
                'nn/instance_norm_impl.cu',
                'nn/instance_norm_impl.h',
                'nn/lrn.cc',
                'nn/lrn.h',
                'nn/max_pool_with_index.cu',
                'nn/max_pool_with_index.h',
                'nn/pool.cc',
                'nn/pool.h',
                'object_detection/non_max_suppression.cc',
                'object_detection/non_max_suppression.h',
                'object_detection/non_max_suppression_impl.cu',
                'object_detection/non_max_suppression_impl.h',
                'object_detection/roialign.cc',
                'object_detection/roialign.h',
                'object_detection/roialign_impl.cu',
                'object_detection/roialign_impl.h',
                'reduction/reduction_ops.cc',
                'reduction/reduction_ops.h',
                'rnn/cudnn_rnn_base.cc',
                'rnn/cudnn_rnn_base.h',
                'rnn/gru.cc',
                'rnn/gru.h',
                'rnn/lstm.cc',
                'rnn/lstm.h',
                'rnn/rnn.cc',
                'rnn/rnn.h',
                'rnn/rnn_impl.cu',
                'rnn/rnn_impl.h',
                'shared_inc/cuda_call.h',
                'shared_inc/fast_divmod.h',
                'shared_inc/fpgeneric.h',
                'shared_inc/integer_gemm.h',
                'tensor/gather_nd_impl.cu',
                'tensor/quantize_linear.cc',
                'tensor/quantize_linear.cu',
                'tensor/quantize_linear.cuh',
                'tensor/quantize_linear.h',
                'tensor/resize.cc',
                'tensor/resize.h',
                'tensor/resize_impl.cu',
                'tensor/resize_impl.h',
                'tensor/transpose.cc',
                'tensor/transpose.h',
                'tensor/upsample.cc',
                'tensor/upsample.h',
                'tensor/upsample_impl.cu',
                'tensor/upsample_impl.h',
                'cuda_allocator.cc',
                'cuda_allocator.h',
                'cuda_call.cc',
                'cuda_common.cc',
                'cuda_common.h',
                'cuda_execution_provider_info.cc',
                'cuda_execution_provider_info.h',
                'cuda_execution_provider.cc',
                'cuda_execution_provider.h',
                'cuda_memory_check.cc',
                'cuda_memory_check.h',
                'cuda_fence.cc',
                'cuda_fence.h',
                'cuda_fwd.h',
                'cuda_kernel.h',
                'cuda_pch.cc',
                'cuda_pch.h',
                'cuda_provider_factory.cc',
                'cuda_utils.cu',
                'cudnn_common.cc',
                'cudnn_common.h',
                'fpgeneric.cu',
                'gpu_data_transfer.cc',
                'gpu_data_transfer.h',
                'integer_gemm.cc',
                'symbols.txt',
]

training_ops_excluded_files = [
                    'activation/gelu_grad_impl_common.cuh',
                    'collective/adasum_kernels.cc',
                    'collective/adasum_kernels.h',
                    'collective/nccl_common.cc',
                    'collective/ready_event.cc',
                    'collective/ready_event.h',
                    'communication/common.h',
                    'communication/nccl_service.cc',
                    'communication/nccl_service.h',
                    'communication/recv.cc',
                    'communication/recv.h',
                    'communication/send.cc',
                    'communication/send.h',
                    'controlflow/record.cc',
                    'controlflow/record.h',
                    'controlflow/wait.cc',
                    'controlflow/wait.h',
                    'math/div_grad.cc',
                    'math/softmax_grad_impl.cu',
                    'math/softmax_grad.cc',
                    'nn/batch_norm_grad.cc',
                    'nn/batch_norm_grad.h',
                    'nn/batch_norm_internal.cc',
                    'nn/batch_norm_internal.h',
                    'nn/conv_grad.cc',
                    'nn/conv_grad.h',
                    'reduction/reduction_all.cc',
                    'reduction/reduction_ops.cc',
                    'tensor/gather_nd_grad_impl.cu',
                    'cuda_training_kernels.cc',
                    'cuda_training_kernels.h',
]

HIPIFY_PERL = '/opt/rocm/bin/hipify-perl'


def hipify(src_file_path, dst_file_path):
    log.debug('Hipifying: "{}" -> "{}"'.format(src_file_path, dst_file_path))

    dst_file_path = dst_file_path.replace('cuda', 'rocm')
    dir_name = os.path.dirname(dst_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    with open(dst_file_path, 'w') as f:
        subprocess.run([HIPIFY_PERL, src_file_path], stdout=f)
    with open(dst_file_path) as f:
        s = f.read().replace('kCudaExecutionProvider', 'kRocmExecutionProvider')
        s = s.replace('CublasHandle', 'RocblasHandle')
        s = s.replace('cublas_handle', 'rocblas_handle_var')
        s = s.replace('hipblasHandle_t', 'rocblas_handle')
        s = s.replace('CudaAsyncBuffer', 'RocmAsyncBuffer')
        s = s.replace('CudaKernel', 'RocmKernel')
        s = s.replace('ToCudaType', 'ToHipType')
        s = s.replace('CudaT', 'HipT')
        s = s.replace('CUDA_LONG', 'HIP_LONG')
        s = s.replace('CUDA_RETURN_IF_ERROR', 'HIP_RETURN_IF_ERROR')
        s = s.replace('CUDA_KERNEL_ASSERT', 'HIP_KERNEL_ASSERT')
        s = s.replace('CUDA_CALL', 'HIP_CALL')
        s = s.replace('SliceCuda', 'SliceRocm')
        s = s.replace('thrust::cuda', 'thrust::hip')
        s = s.replace('cuda', 'rocm')
        # s = s.replace('Cuda', 'Rocm')
        s = s.replace('CUDA', 'ROCM')

        s = s.replace('GPU_WARP_SIZE = 32', 'GPU_WARP_SIZE = 64')
        s = s.replace('std::exp', 'expf')
        s = s.replace('std::log', 'logf')
        s = s.replace('#include <cub/device/device_radix_sort.cuh>',
                      '#include <hipcub/hipcub.hpp>\n#include <hipcub/backend/rocprim/device/device_radix_sort.hpp>')
        s = s.replace('#include <cub/device/device_reduce.cuh>',
                      '#include <hipcub/backend/rocprim/device/device_reduce.hpp>')
        s = s.replace('#include <cub/device/device_run_length_encode.cuh>',
                      '#include <hipcub/backend/rocprim/device/device_run_length_encode.hpp>')
        s = s.replace('#include <cub/device/device_scan.cuh>',
                      '#include <hipcub/backend/rocprim/device/device_scan.hpp>')
        s = s.replace('#include <cub/iterator/counting_input_iterator.cuh>',
                      '#include <hipcub/backend/rocprim/iterator/counting_input_iterator.hpp>')
        s = s.replace('#include <cub/iterator/discard_output_iterator.cuh>',
                      '#include <hipcub/backend/rocprim/iterator/discard_output_iterator.hpp>')
        s = s.replace('typedef half MappedType', 'typedef __half MappedType')
        # CUBLAS -> ROCBLAS
        # s = s.replace('CUBLAS', 'HIPBLAS')
        # s = s.replace('Cublas', 'Hipblas')
        # s = s.replace('cublas', 'hipblas')

        # CURAND -> HIPRAND
        s = s.replace('CURAND', 'HIPRAND')
        s = s.replace('Curand', 'Hiprand')
        s = s.replace('curand', 'hiprand')

        # NCCL -> RCCL
        # s = s.replace('NCCL_CALL', 'RCCL_CALL')
        s = s.replace('#include <nccl.h>', '#include <rccl.h>')

        # CUDNN -> MIOpen
        s = s.replace('CUDNN', 'MIOPEN')
        s = s.replace('Cudnn', 'Miopen')
        s = s.replace('cudnn', 'miopen')
        # hipify seems to have a bug for MIOpen, cudnn.h -> hipDNN.h, cudnn -> hipdnn
        s = s.replace('#include <hipDNN.h>', '#include <miopen/miopen.h>')
        s = s.replace('hipdnn', 'miopen')
        s = s.replace('HIPDNN_STATUS_SUCCESS', 'miopenStatusSuccess')
        s = s.replace('HIPDNN', 'MIOPEN')

        # CUSPARSE -> HIPSPARSE
        s = s.replace('CUSPARSE', 'HIPSPARSE')

        # CUFFT -> HIPFFT
        s = s.replace('CUFFT', 'HIPFFT')
    with open(dst_file_path, 'w') as f:
        f.write(s)


def list_files(prefix, path):
    all_files = []
    curr_path = os.path.join(prefix, path)
    for root, dirs, files in os.walk(curr_path):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(os.path.relpath(full_path, curr_path))
    return all_files


def amd_hipify(config_build_dir):
    cuda_contrib_path = os.path.join(contrib_ops_path, 'cuda')
    rocm_contrib_path = os.path.join(config_build_dir, 'amdgpu', contrib_ops_path, 'rocm')
    contrib_files = list_files(cuda_contrib_path, '')
    for file in contrib_files:
        if file not in contrib_ops_excluded_files:
            src_file_path = os.path.join(cuda_contrib_path, file)
            dst_file_path = os.path.join(rocm_contrib_path, file)
            hipify(src_file_path, dst_file_path)

    cuda_provider_path = os.path.join(providers_path, 'cuda')
    rocm_provider_path = os.path.join(config_build_dir, 'amdgpu', providers_path, 'rocm')
    provider_files = list_files(cuda_provider_path, '')
    for file in provider_files:
        if file not in provider_excluded_files:
            src_file_path = os.path.join(cuda_provider_path, file)
            dst_file_path = os.path.join(rocm_provider_path, file)
            hipify(src_file_path, dst_file_path)

    cuda_training_path = os.path.join(training_ops_path, 'cuda')
    rocm_training_path = os.path.join(config_build_dir, 'amdgpu', training_ops_path, 'rocm')
    training_files = list_files(cuda_training_path, '')
    for file in training_files:
        if file not in training_ops_excluded_files:
            src_file_path = os.path.join(cuda_training_path, file)
            dst_file_path = os.path.join(rocm_training_path, file)
            hipify(src_file_path, dst_file_path)

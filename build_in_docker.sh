pip uninstall onnxruntime_gpu onnxruntime_gpu_noopenmp -y

rm -rf build/Linux/RelWithDebInfo/dist/
./build.sh --enable_training --use_cuda --config=RelWithDebInfo --cuda_version=10.2 --cuda_home=/usr/local/cuda-10.2 --cudnn_home=/usr/local/cuda-10.2 --update --build --build_wheel --parallel --mpi_home /usr/local/mpi/
pip install  build/Linux/RelWithDebInfo/dist/onnxruntime_gpu_noopenmp-1.5.2-cp36-cp36m-linux_x86_64.whl

# rm -rf build/Linux/Debug/dist/
# ./build.sh --enable_training --use_cuda --config=Debug --cuda_version=10.2 --cuda_home=/usr/local/cuda-10.2 --cudnn_home=/usr/local/cuda-10.2 --update --build --build_wheel --parallel --mpi_home /usr/local/mpi --cmake_extra_defines onnxruntime_DEBUG_NODE_INPUTS_OUTPUTS=1
# pip install  build/Linux/Debug/dist/onnxruntime_gpu_noopenmp-1.5.2-cp36-cp36m-linux_x86_64.whl

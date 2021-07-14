#!/bin/bash
set -e -o -x

id

SCRIPT_DIR="$( dirname "${BASH_SOURCE[0]}" )"
YOCTO_VERSION="4.19"

while getopts d:x:o:y: parameter_Option
do case "${parameter_Option}"
in
d) BUILD_DEVICE=${OPTARG};;
x) BUILD_EXTR_PAR=${OPTARG};;
o) BUILD_OS=${OPTARG};;
# YOCTO 4.19 + ACL 19.05, YOCTO 4.14 + ACL 19.02
y) YOCTO_VERSION=${OPTARG};;
esac
done

export PATH=$PATH:/usr/local/gradle/bin

if [ $BUILD_OS = "android" ]; then
    pushd /onnxruntime_src
    mkdir build-android && cd build-android
    if [ $BUILD_DEVICE = "nnapi" ]; then
        cmake -DCMAKE_TOOLCHAIN_FILE=/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_USE_NNAPI_BUILTIN=ON ../cmake
    else
        cmake -DCMAKE_TOOLCHAIN_FILE=/android-ndk/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc ../cmake
    fi
    make -j$(nproc)
elif [ $BUILD_OS = "yocto" ]; then
    YOCTO_FOLDER="4.19-warrior"
    if [ $YOCTO_VERSION = "4.14" ]; then
        YOCTO_FOLDER="4.14-sumo"
    fi
    pushd /onnxruntime_src
    if [ ! -d build ]; then
        mkdir build
    fi
    cd build
    . /opt/fsl-imx-xwayland/$YOCTO_FOLDER/environment-setup-aarch64-poky-linux
    alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
    cmake ../cmake -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -Donnxruntime_DEV_MODE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_NSYNC=OFF -Donnxruntime_CUDNN_HOME= -Donnxruntime_USE_JEMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_USE_EIGEN_FOR_BLAS=ON -Donnxruntime_USE_OPENBLAS=OFF -Donnxruntime_USE_ACL=ON -Donnxruntime_USE_MKLDNN=OFF -Donnxruntime_USE_MKLML=OFF -Donnxruntime_USE_OPENMP=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_NUPHAR=OFF -Donnxruntime_USE_EIGEN_THREADPOOL=OFF -Donnxruntime_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES:PATH=/opt/fsl-imx-xwayland/$YOCTO_FOLDER/sysroots/aarch64-poky-linux/usr/include -DCMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES:PATH=/opt/fsl-imx-xwayland/$YOCTO_FOLDER/sysroots/aarch64-poky-linux/usr/include -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc

    make -j$(nproc)
else
    COMMON_BUILD_ARGS="--skip_submodule_sync --enable_onnx_tests --parallel --build_shared_lib --cmake_path /usr/bin/cmake --ctest_path /usr/bin/ctest"

    if [ $BUILD_DEVICE = "gpu" ]; then
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Release $COMMON_BUILD_ARGS \
            --use_cuda \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cudnn-$_CUDNN_VERSION/cuda $BUILD_EXTR_PAR
    elif [[ $BUILD_DEVICE = "tensorrt"* ]]; then
        if [ $BUILD_DEVICE = "tensorrt-v7.1" ]; then
            CUR_PWD=$(pwd)
            cd $SCRIPT_DIR/../../../../cmake/external/onnx-tensorrt/
            git remote update
            git checkout 7.1
            cd $CUR_PWD
            COMMON_BUILD_ARGS=${COMMON_BUILD_ARGS/"--skip_submodule_sync"}
        fi
        _CUDNN_VERSION=$(echo $CUDNN_VERSION | cut -d. -f1-2)
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Release $COMMON_BUILD_ARGS \
            --use_tensorrt --tensorrt_home /workspace/tensorrt \
            --cuda_home /usr/local/cuda \
            --cudnn_home /usr/local/cuda $BUILD_EXTR_PAR
    else #cpu and openvino
        python3 $SCRIPT_DIR/../../build.py --build_dir /build \
            --config Release $COMMON_BUILD_ARGS $BUILD_EXTR_PAR
    fi
fi


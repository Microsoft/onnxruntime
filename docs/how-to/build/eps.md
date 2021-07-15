---
title: Build with different EPs
parent: Build ORT
grand_parent: How to
nav_order: 3
---

# Build ONNX Runtime with Execution Providers
{: .no_toc }

## Contents
{: .no_toc }

* TOC placeholder
{:toc}

## Execution Provider Shared Libraries

The oneDNN, TensorRT, and OpenVINO providers are built as shared libraries vs being statically linked into the main onnxruntime. This enables them to be loaded only when needed, and if the dependent libraries of the provider are not installed onnxruntime will still run fine, it just will not be able to use that provider. For non shared library providers, all dependencies of the provider must exist to load onnxruntime.

### Built files
{: .no_toc }

On Windows, shared provider libraries will be named 'onnxruntime_providers_\*.dll' (for example onnxruntime_providers_openvino.dll).
On Unix, they will be named 'libonnxruntime_providers_\*.so'
On Mac, they will be named 'libonnxruntime_providers_\*.dylib'.

There is also a shared library that shared providers depend on called onnxruntime_providers_shared (with the same naming convension applied as above).

Note: It is not recommended to put these libraries in a system location or added to a library search path (like LD_LIBRARY_PATH on Unix). If multiple versions of onnxruntime are installed on the system this can make them find the wrong libraries and lead to undefined behavior.

### Loading the shared providers
{: .no_toc }

Shared provider libraries are loaded by the onnxruntime code (do not load or depend on them in your client code). The API for registering shared or non shared providers is identical, the difference is that shared ones will be loaded at runtime when the provider is added to the session options (through a call like OrtSessionOptionsAppendExecutionProvider_OpenVINO or SessionOptionsAppendExecutionProvider_OpenVINO in the C API).
If a shared provider library cannot be loaded (if the file doesn't exist, or its dependencies don't exist or not in the path) then an error will be returned.

The onnxruntime code will look for the provider shared libraries in the same location as the onnxruntime shared library is (or the executable statically linked to the static library version).

---

## CUDA

### Prerequisites
{: .no_toc }

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
  * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home` parameter
  * The path to the cuDNN installation (include the `cuda` folder in the path) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home` parameter. The cuDNN path should contain `bin`, `include` and `lib` directories.
  * The path to the cuDNN bin directory must be added to the PATH environment variable so that cudnn64_8.dll is found.


### Build Instructions
{: .no_toc }

#### Windows

```
.\build.bat --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

#### Linux

```
./build.sh --use_cuda --cudnn_home <cudnn home path> --cuda_home <cuda home path>
```

A Dockerfile is available [here](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles#cuda).

### Notes
{: .no_toc }

* Depending on compatibility between the CUDA, cuDNN, and Visual Studio 2017 versions you are using, you may need to explicitly install an earlier version of the MSVC toolset.
 * CUDA 10.0 is [known to work](https://devblogs.microsoft.com/cppblog/cuda-10-is-now-available-with-support-for-the-latest-visual-studio-2017-versions/) with toolsets from 14.11 up to 14.16 (Visual Studio 2017 15.9), and should continue to work with future Visual Studio versions
 * CUDA 9.2 is known to work with the 14.11 MSVC toolset (Visual Studio 15.3 and 15.4)
    * To install the 14.11 MSVC toolset, see [this page](https://blogs.msdn.microsoft.com/vcblog/2017/11/15/side-by-side-minor-version-msvc-toolsets-in-visual-studio-2017).
    * To use the 14.11 toolset with a later version of Visual Studio 2017 you have two options:
     1. Setup the Visual Studio environment variables to point to the 14.11 toolset by running vcvarsall.bat, prior to running the build script. e.g. if you have VS2017 Enterprise, an x64 build would use the following command `"C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" amd64 -vcvars_ver=14.11` For convenience, .\build.amd64.1411.bat will do this and can be used in the same way as .\build.bat. e.g. ` .\build.amd64.1411.bat --use_cuda`

     2. Alternatively, if you have CMake 3.13 or later you can specify the toolset version via the `--msvc_toolset` build script parameter. e.g. `.\build.bat --msvc_toolset 14.11`

* If you have multiple versions of CUDA installed on a Windows machine and are building with Visual Studio, CMake will use the build files for the highest version of CUDA it finds in the BuildCustomization folder.
e.g. C:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\Common7\IDE\VC\VCTargets\BuildCustomizations\.
If you want to build with an earlier version, you must temporarily remove the 'CUDA x.y.*' files for later versions from this directory.

---

## TensorRT

See more information on the TensorRT Execution Provider [here](../../reference/execution-providers/TensorRT-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [CUDA](https://developer.nvidia.com/cuda-toolkit) and [cuDNN](https://developer.nvidia.com/cudnn)
   * The TensorRT execution provider for ONNX Runtime is built and tested with CUDA 10.2/11.0/11.1 and cuDNN 8.0.
   * The path to the CUDA installation must be provided via the CUDA_PATH environment variable, or the `--cuda_home` parameter. The CUDA path should contain `bin`, `include` and `lib` directories.
   * The path to the CUDA `bin` directory must be added to the PATH environment variable so that `nvcc` is found.
   * The path to the cuDNN installation (path to folder that contains libcudnn.so) must be provided via the cuDNN_PATH environment variable, or `--cudnn_home` parameter.
 * Install [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download)
   * The TensorRT execution provider for ONNX Runtime is built on TensorRT 7.2.2 and is tested with TensorRT 7.2.2.3.
   * The path to TensorRT installation must be provided via the `--tensorrt_home` parameter.

### Build Instructions
{: .no_toc }

#### Windows
```
.\build.bat --cudnn_home <path to cuDNN home> --cuda_home <path to CUDA home> --use_tensorrt --tensorrt_home <path to TensorRT home>
```

#### Linux

```
./build.sh --cudnn_home <path to cuDNN e.g. /usr/lib/x86_64-linux-gnu/> --cuda_home <path to folder for CUDA e.g. /usr/local/cuda> --use_tensorrt --tensorrt_home <path to TensorRT home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#tensorrt)

---

## NVIDIA Jetson TX1/TX2/Nano/Xavier

### Build Instructions
{: .no_toc }

These instructions are for JetPack SDK 4.4.

1. Clone the ONNX Runtime repo on the Jetson host

    ```bash
    git clone --recursive https://github.com/microsoft/onnxruntime
    ```

2. Specify the CUDA compiler, or add its location to the PATH.

   Cmake can't automatically find the correct nvcc if it's not in the PATH.

    ```bash
    export CUDACXX="/usr/local/cuda/bin/nvcc"

    ```

    or:

    ```bash
    export PATH="/usr/local/cuda/bin:${PATH}"
    ```

3. Install the ONNX Runtime build dependencies on the Jetpack 4.4 host:

    ```bash
    sudo apt install -y --no-install-recommends \
      build-essential software-properties-common libopenblas-dev \
      libpython3.6-dev python3-pip python3-dev python3-setuptools python3-wheel
    ```

4. Cmake is needed to build ONNX Runtime. Because the minimum required version is 3.13,
   it is necessary to build CMake from source. Download Unix/Linux sources from https://cmake.org/download/
   and follow https://cmake.org/install/ to build from source. Version 3.17.5 and 3.18.4 have been tested on Jetson.

5. Build the ONNX Runtime Python wheel:

    ```bash
    ./build.sh --config Release --update --build --parallel --build_wheel \
    --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu
    ```

    Note: You may optionally build with experimental TensorRT support.

    ```bash
    ./build.sh --config Release --update --build --parallel --build_wheel \
    --use_tensorrt --cuda_home /usr/local/cuda --cudnn_home /usr/lib/aarch64-linux-gnu \
    --tensorrt_home /usr/lib/aarch64-linux-gnu
    ```

---

## oneDNN

See more information on ondDNN (formerly DNNL) [here](../reference/execution-providers/DNNL-ExecutionProvider.md).

### Build Instructions
{: .no_toc }


The DNNL execution provider can be built for Intel CPU or GPU. To build for Intel GPU, install [Intel SDK for OpenCL Applications](https://software.intel.com/content/www/us/en/develop/tools/opencl-sdk.html). Install the latest GPU driver - [Windows graphics driver](https://downloadcenter.intel.com/product/80939/Graphics), [Linux graphics compute runtime and OpenCL driver](https://github.com/intel/compute-runtime/releases).

#### Windows
`.\build.bat --use_dnnl`

#### Linux
`./build.sh --use_dnnl`

To build for Intel GPU, replace dnnl_opencl_root with the path of the Intel SDK for OpenCL Applications.

#### Windows 

`.\build.bat --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "c:\program files (x86)\intelswtools\sw_dev_tools\opencl\sdk"`
#### Linux

`./build.sh --use_dnnl --dnnl_gpu_runtime ocl --dnnl_opencl_root "/opt/intel/sw_dev_tools/opencl-sdk"`s

---

## OpenVINO

See more information on the OpenVINO Execution Provider [here](../../reference/execution-providers/OpenVINO-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

1. Install the Intel<sup>®</sup> Distribution of OpenVINO<sup>TM</sup> Toolkit **Release 2021.4** for the appropriate OS and target hardware:
   * [Linux - CPU, GPU, VPU, VAD-M](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux)
   * [Linux - FPGA](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-linux-fpga)
   * [Windows - CPU, GPU, VPU, VAD-M](https://software.intel.com/en-us/openvino-toolkit/choose-download/free-download-windows).

   Follow [documentation](https://docs.openvinotoolkit.org/2021.4/index.html) for detailed instructions.

  *2021.4 is the recommended OpenVINO version. [OpenVINO 2020.3](https://docs.openvinotoolkit.org/2020.3/index.html) is minimal OpenVINO version requirement.*
  *The minimum ubuntu version to support 2021.4 is 18.04.*

2. Configure the target hardware with specific follow on instructions:
   * To configure Intel<sup>®</sup> Processor Graphics(GPU) please follow these instructions: [Windows](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_windows.html#Install-GPU), [Linux](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#additional-GPU-steps)
   * To configure Intel<sup>®</sup> Movidius<sup>TM</sup> USB, please follow this getting started guide: [Linux](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#additional-NCS-steps)
   * To configure Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs, please follow this configuration guide: [Windows](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_windows.html#hddl-myriad), [Linux](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_linux.html#install-VPU). Follow steps 3 and 4 to complete the configuration.
   * To configure Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA, please follow this configuration guide: [Linux](https://docs.openvinotoolkit.org/2021.4/openvino_docs_install_guides_installing_openvino_linux_fpga.html)

3. Initialize the OpenVINO environment by running the setupvars script as shown below:
   * For Linux run:
   ```
      $ source <openvino_install_directory>/bin/setupvars.sh
   ```
   * For Windows run:
   ```
      C:\ <openvino_install_directory>\bin\setupvars.bat
   ```

4. Extra configuration step for Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs:
   * After setting the environment using setupvars script, follow these steps to change the default scheduler of VAD-M to Bypass:
      * Edit the hddl_service.config file from $HDDL_INSTALL_DIR/config/hddl_service.config and change the field "bypass_device_number" to 8.
      * Restart the hddl daemon for the changes to take effect.
      * Note that if OpenVINO was installed with root permissions, this file has to be changed with the same permissions.


### Build Instructions
{: .no_toc }

#### Windows

```
.\build.bat --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib
```

*Note: The default Windows CMake Generator is Visual Studio 2017, but you can also use the newer Visual Studio 2019 by passing `--cmake_generator "Visual Studio 16 2019"` to `.\build.bat`*

#### Linux

```bash
./build.sh --config RelWithDebInfo --use_openvino <hardware_option> --build_shared_lib
```

* `--use_openvino` builds the OpenVINO Execution Provider in ONNX Runtime.
* `<hardware_option>`: Specifies the default hardware target for building OpenVINO Execution Provider. This can be overriden dynamically at runtime with another option (refer to [OpenVINO-ExecutionProvider](../../reference/execution-providers/OpenVINO-ExecutionProvider.md#summary-of-options) for more details on dynamic device selection). Below are the options for different Intel target devices.

| Hardware Option | Target Device |
| --------------- | ------------------------|
| <code>CPU_FP32</code> | Intel<sup>®</sup> CPUs |
| <code>GPU_FP32</code> | Intel<sup>®</sup> Integrated Graphics |
| <code>GPU_FP16</code> | Intel<sup>®</sup> Integrated Graphics with FP16 quantization of models |
| <code>MYRIAD_FP16</code> | Intel<sup>®</sup> Movidius<sup>TM</sup> USB sticks | 
| <code>VAD-M_FP16</code> | Intel<sup>®</sup> Vision Accelerator Design based on 8 Movidius<sup>TM</sup> MyriadX VPUs |
| <code>VAD-F_FP32</code> | Intel<sup>®</sup> Vision Accelerator Design with an Intel<sup>®</sup> Arria<sup>®</sup> 10 FPGA |
| <code>HETERO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |
| <code>MULTI:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...</code> | All Intel<sup>®</sup> silicons mentioned above |

Specifying Hardware Target for HETERO or Multi-Device Build:

HETERO:DEVICE_TYPE_1,DEVICE_TYPE_2,DEVICE_TYPE_3...
The DEVICE_TYPE can be any of these devices from this list ['CPU','GPU','MYRIAD','FPGA','HDDL']

A minimum of two device's should be specified for a valid HETERO or Multi-Device Build.

```
Example's: HETERO:MYRIAD,CPU or HETERO:HDDL,GPU,CPU or MULTI:MYRIAD,GPU,CPU
```

For more information on OpenVINO Execution Provider&#39;s ONNX Layer support, Topology support, and Intel hardware enabled, please refer to the document [OpenVINO-ExecutionProvider](../../reference/execution-providers/OpenVINO-ExecutionProvider.md)

---

## NUPHAR
See more information on the Nuphar Execution Provider [here](../reference/execution-providers/Nuphar-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* The Nuphar execution provider for ONNX Runtime is built and tested with LLVM 9.0.0. Because of TVM's requirement when building with LLVM, you need to build LLVM from source. To build the debug flavor of ONNX Runtime, you need the debug build of LLVM.
   * Windows (Visual Studio 2017):
   ```
   REM download llvm source code 9.0.0 and unzip to \llvm\source\path, then install to \llvm\install\path
   cd \llvm\source\path
   mkdir build
   cd build
   cmake .. -G "Visual Studio 15 2017 Win64" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_DIA_SDK=OFF
   msbuild llvm.sln /maxcpucount /p:Configuration=Release /p:Platform=x64
   cmake -DCMAKE_INSTALL_PREFIX=\llvm\install\path -DBUILD_TYPE=Release -P cmake_install.cmake
   ```

*Note that following LLVM cmake patch is necessary to make the build work on Windows, Linux does not need to apply the patch.*
The patch is to fix the linking warning LNK4199 caused by this [LLVM commit](https://github.com/llvm-mirror/llvm/commit/148f823e4845c9a13faea62e3105abb80b39e4bc)

```
diff --git "a/lib\\Support\\CMakeLists.txt" "b/lib\\Support\\CMakeLists.txt"
index 7dfa97c..6d99e71 100644
--- "a/lib\\Support\\CMakeLists.txt"
+++ "b/lib\\Support\\CMakeLists.txt"
@@ -38,12 +38,6 @@ elseif( CMAKE_HOST_UNIX )
   endif()
 endif( MSVC OR MINGW )

-# Delay load shell32.dll if possible to speed up process startup.
-set (delayload_flags)
-if (MSVC)
-  set (delayload_flags delayimp -delayload:shell32.dll -delayload:ole32.dll)
-endif()
-
 # Link Z3 if the user wants to build it.
 if(LLVM_WITH_Z3)
   set(Z3_LINK_FILES ${Z3_LIBRARIES})
@@ -187,7 +181,7 @@ add_llvm_library(LLVMSupport
   ${LLVM_MAIN_INCLUDE_DIR}/llvm/ADT
   ${LLVM_MAIN_INCLUDE_DIR}/llvm/Support
   ${Backtrace_INCLUDE_DIRS}
-  LINK_LIBS ${system_libs} ${delayload_flags} ${Z3_LINK_FILES}
+  LINK_LIBS ${system_libs} ${Z3_LINK_FILES}
   )

 set_property(TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS "${system_libs}")
```
   * Linux
   Download llvm source code 9.0.0 and unzip to /llvm/source/path, then install to /llvm/install/path
   ```
   cd /llvm/source/path
   mkdir build
   cd build
   cmake .. -DLLVM_TARGETS_TO_BUILD=X86 -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   cmake -DCMAKE_INSTALL_PREFIX=/llvm/install/path -DBUILD_TYPE=Release -P cmake_install.cmake
   ```

### Build Instructions
{: .no_toc }

#### Windows
```
.\build.bat --llvm_path=\llvm\install\path\lib\cmake\llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

* These instructions build the release flavor. The Debug build of LLVM would be needed to build with the Debug flavor of ONNX Runtime.

#### Linux:
```
./build.sh --llvm_path=/llvm/install/path/lib/cmake/llvm --use_mklml --use_nuphar --build_shared_lib --build_csharp --enable_pybind --config=Release
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/tree/master/dockerfiles#nuphar).


---

## DirectML
See more information on the DirectML execution provider [here](../reference/execution-providers/DirectML-ExecutionProvider.md).
### Windows
{: .no_toc }

```
.\build.bat --use_dml
```
### Notes
{: .no_toc }

The DirectML execution provider supports building for both x64 and x86 architectures. DirectML is only supported on Windows.

---

## ARM Compute Library
See more information on the ACL Execution Provider [here](../reference/execution-providers/ACL-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment
```
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```
* See [Build ARM](#ARM) below for information on building for ARM devices

### Build Instructions
{: .no_toc }

1. Configure ONNX Runtime with ACL support:
```
cmake ../onnxruntime-arm-upstream/cmake -DONNX_CUSTOM_PROTOC_EXECUTABLE=/usr/bin/protoc -Donnxruntime_RUN_ONNX_TESTS=OFF -Donnxruntime_GENERATE_TEST_REPORTS=ON -Donnxruntime_DEV_MODE=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -Donnxruntime_USE_CUDA=OFF -Donnxruntime_USE_NSYNC=OFF -Donnxruntime_CUDNN_HOME= -Donnxruntime_USE_JEMALLOC=OFF -Donnxruntime_ENABLE_PYTHON=OFF -Donnxruntime_BUILD_CSHARP=OFF -Donnxruntime_BUILD_SHARED_LIB=ON -Donnxruntime_USE_EIGEN_FOR_BLAS=ON -Donnxruntime_USE_OPENBLAS=OFF -Donnxruntime_USE_ACL=ON -Donnxruntime_USE_DNNL=OFF -Donnxruntime_USE_MKLML=OFF -Donnxruntime_USE_OPENMP=ON -Donnxruntime_USE_TVM=OFF -Donnxruntime_USE_LLVM=OFF -Donnxruntime_ENABLE_MICROSOFT_INTERNAL=OFF -Donnxruntime_USE_BRAINSLICE=OFF -Donnxruntime_USE_NUPHAR=OFF -Donnxruntime_USE_EIGEN_THREADPOOL=OFF -Donnxruntime_BUILD_UNIT_TESTS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
The ```-Donnxruntime_USE_ACL=ON``` option will use, by default, the 19.05 version of the Arm Compute Library. To set the right version you can use:
```-Donnxruntime_USE_ACL_1902=ON```, ```-Donnxruntime_USE_ACL_1905=ON```, ```-Donnxruntime_USE_ACL_1908=ON``` or ```-Donnxruntime_USE_ACL_2002=ON```;

To use a library outside the normal environment you can set a custom path by using ```-Donnxruntime_ACL_HOME``` and ```-Donnxruntime_ACL_LIBS``` tags that defines the path to the ComputeLibrary directory and the build directory respectively.

```-Donnxruntime_ACL_HOME=/path/to/ComputeLibrary```, ```-Donnxruntime_ACL_LIBS=/path/to/build```


2. Build ONNX Runtime library, test and performance application:
```
make -j 6
```

3. Deploy ONNX runtime on the i.MX 8QM board
```
libonnxruntime.so.0.5.0
onnxruntime_perf_test
onnxruntime_test_all
```

### Native Build Instructions 
{: .no_toc }

*Validated on Jetson Nano and Jetson Xavier*

1. Build ACL Library (skip if already built)

    ```bash
    cd ~
    git clone -b v20.02 https://github.com/Arm-software/ComputeLibrary.git
    cd ComputeLibrary
    sudo apt-get install -y scons g++-arm-linux-gnueabihf
    scons -j8 arch=arm64-v8a  Werror=1 debug=0 asserts=0 neon=1 opencl=1 examples=1 build=native
    ```

1. Cmake is needed to build ONNX Runtime. Because the minimum required version is 3.13,
   it is necessary to build CMake from source. Download Unix/Linux sources from https://cmake.org/download/
   and follow https://cmake.org/install/ to build from source. Version 3.17.5 and 3.18.4 have been tested on Jetson.

1. Build onnxruntime with --use_acl flag with one of the supported ACL version flags. (ACL_1902 | ACL_1905 | ACL_1908 | ACL_2002)

---

## ArmNN

See more information on the ArmNN Execution Provider [here](../reference/execution-providers/ArmNN-ExecutionProvider.md).

### Prerequisites
{: .no_toc }


* Supported backend: i.MX8QM Armv8 CPUs
* Supported BSP: i.MX8QM BSP
  * Install i.MX8QM BSP: `source fsl-imx-xwayland-glibc-x86_64-fsl-image-qt5-aarch64-toolchain-4*.sh`
* Set up the build environment

```bash
source /opt/fsl-imx-xwayland/4.*/environment-setup-aarch64-poky-linux
alias cmake="/usr/bin/cmake -DCMAKE_TOOLCHAIN_FILE=$OECORE_NATIVE_SYSROOT/usr/share/cmake/OEToolchainConfig.cmake"
```

* See [Build ARM](#ARM) below for information on building for ARM devices

### Build Instructions
{: .no_toc }


```bash
./build.sh --use_armnn
```

The Relu operator is set by default to use the CPU execution provider for better performance. To use the ArmNN implementation build with --armnn_relu flag

```bash
./build.sh --use_armnn --armnn_relu
```

The Batch Normalization operator is set by default to use the CPU execution provider. To use the ArmNN implementation build with --armnn_bn flag

```bash
./build.sh --use_armnn --armnn_bn
```

To use a library outside the normal environment you can set a custom path by providing the --armnn_home and --armnn_libs parameters to define the path to the ArmNN home directory and build directory respectively. 
The ARM Compute Library home directory and build directory must also be available, and can be specified if needed using --acl_home and --acl_libs respectively.

```bash
./build.sh --use_armnn --armnn_home /path/to/armnn --armnn_libs /path/to/armnn/build  --acl_home /path/to/ComputeLibrary --acl_libs /path/to/acl/build
```

---

## RKNPU
See more information on the RKNPU Execution Provider [here](../reference/execution-providers/RKNPU-ExecutionProvider.md).

### Prerequisites
{: .no_toc }


* Supported platform: RK1808 Linux
* See [Build ARM](#ARM) below for information on building for ARM devices
* Use gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu instead of gcc-linaro-6.3.1-2017.05-x86_64_arm-linux-gnueabihf, and modify CMAKE_CXX_COMPILER & CMAKE_C_COMPILER in tool.cmake:
  
```
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
```

### Build Instructions
{: .no_toc }

#### Linux

1. Download [rknpu_ddk](#https://github.com/airockchip/rknpu_ddk.git) to any directory.

2. Build ONNX Runtime library and test:

    ```bash
    ./build.sh --arm --use_rknpu --parallel --build_shared_lib --build_dir build_arm --config MinSizeRel --cmake_extra_defines RKNPU_DDK_PATH=<Path To rknpu_ddk> CMAKE_TOOLCHAIN_FILE=<Path To tool.cmake> ONNX_CUSTOM_PROTOC_EXECUTABLE=<Path To protoc>
    ```

3. Deploy ONNX runtime and librknpu_ddk.so on the RK1808 board:

    ```bash
    libonnxruntime.so.1.2.0
    onnxruntime_test_all
    rknpu_ddk/lib64/librknpu_ddk.so
    ```

---

## Vitis-AI
See more information on the Xilinx Vitis-AI execution provider [here](../reference/execution-providers/Vitis-AI-ExecutionProvider.md).

For instructions to setup the hardware environment: [Hardware setup](../reference/execution-providers/Vitis-AI-ExecutionProvider.md#Hardware-setup)

### Linux
{: .no_toc }


```bash
./build.sh --use_vitisai
```

### Notes
{: .no_toc }

The Vitis-AI execution provider is only supported on Linux.

---

## AMD MIGraphX

See more information on the MIGraphX Execution Provider [here](../reference/execution-providers/MIGraphX-ExecutionProvider.md).

### Prerequisites
{: .no_toc }

* Install [ROCM](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html)
  * The MIGraphX execution provider for ONNX Runtime is built and tested with ROCM3.3
* Install [MIGraphX](https://github.com/ROCmSoftwarePlatform/AMDMIGraphX)
  * The path to MIGraphX installation must be provided via the `--migraphx_home parameter`.

### Build Instructions
{: .no_toc }

#### Linux

```bash
./build.sh --config <Release|Debug|RelWithDebInfo> --use_migraphx --migraphx_home <path to MIGraphX home>
```

Dockerfile instructions are available [here](https://github.com/microsoft/onnxruntime/blob/master/dockerfiles#migraphx).

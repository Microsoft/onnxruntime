# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.import argparse
import argparse
import os
import platform
import shlex
import sys
import warnings

from util import (
    is_linux,
    is_macOS,
    is_windows,
)


def _str_to_bool(s: str) -> bool:
    """Convert string to bool (in argparse context) using match/case."""
    match s.lower():
        case "true":
            return True
        case "false":
            return False
        case _:
            raise ValueError(f"Invalid boolean value: {s!r}. Use 'true' or 'false'.")
    return False


# --- Argument Verification Helpers ---
def _qnn_verify_library_kind(library_kind: str) -> str:
    """Verifies the library kind for the QNN Execution Provider."""
    choices = ["shared_lib", "static_lib"]
    if library_kind not in choices:
        print("\nYou have specified an invalid library kind for QNN EP.")
        print(f"The invalid library kind was: {library_kind}")
        print("Provide a library kind from the following options: ", choices)
        print(f"Example: --use_qnn {choices[0]}")
        sys.exit("Incorrect build configuration")
    return library_kind


def _openvino_verify_device_type(device_read: str) -> str:
    """Verifies the device type string for the OpenVINO Execution Provider."""
    choices = ["CPU", "GPU", "NPU"]
    choices1 = [
        "CPU_NO_PARTITION",
        "GPU_NO_PARTITION",
        "NPU_NO_PARTITION",
        "NPU_NO_CPU_FALLBACK",
    ]
    status_hetero = True
    res = False
    if device_read in choices:
        res = True
    elif device_read in choices1:
        res = True
    elif device_read.startswith(("HETERO:", "MULTI:", "AUTO:")):
        res = True
        parts = device_read.split(":")
        if len(parts) < 2 or not parts[1]:
            print("Hetero/Multi/Auto mode requires devices to be specified after the colon.")
            status_hetero = False
        else:
            comma_separated_devices = parts[1].split(",")
            if len(comma_separated_devices) < 2:
                print("At least two devices required in Hetero/Multi/Auto Mode")
                status_hetero = False
            dev_options = ["CPU", "GPU", "NPU"]
            for dev in comma_separated_devices:
                if dev not in dev_options:
                    status_hetero = False
                    print(f"Invalid device '{dev}' found in Hetero/Multi/Auto specification.")
                    break

    def invalid_hetero_build() -> None:
        print("\nIf trying to build Hetero/Multi/Auto, specify the supported devices along with it.\n")
        print("Specify the keyword HETERO or MULTI or AUTO followed by a colon and comma-separated devices ")
        print("in the order of priority you want to build (e.g., HETERO:GPU,CPU).\n")
        print("The different hardware devices that can be added are ['CPU','GPU','NPU'] \n")
        print("An example of how to specify the hetero build type: --use_openvino HETERO:GPU,CPU \n")
        print("An example of how to specify the MULTI build type: --use_openvino MULTI:GPU,CPU \n")
        print("An example of how to specify the AUTO build type: --use_openvino AUTO:GPU,CPU \n")
        sys.exit("Wrong Build Type selected")

    if res is False:
        print("\nYou have selected wrong configuration for the build.")
        print("Pick the build type for specific Hardware Device from following options: ", choices)
        print("(or) from the following options with graph partitioning disabled: ", choices1)
        print("\n")
        if not (device_read.startswith(("HETERO", "MULTI", "AUTO"))):
            invalid_hetero_build()  # Will exit
        sys.exit("Wrong Build Type selected")  # Should not be reached if invalid_hetero_build exits

    if status_hetero is False:
        invalid_hetero_build()  # Will exit

    return device_read


def _webgpu_verify_library_kind(library_kind: str) -> str:
    """Verifies the library kind for the WebGPU Execution Provider."""
    choices = ["shared_lib", "static_lib", "static_plugin"]
    if library_kind not in choices:
        print("\nYou have specified an invalid library kind for WebGPU EP.")
        print(f"The invalid library kind was: {library_kind}")
        print("Provide a library kind from the following options: ", choices)
        print(f"Example: --use_webgpu {choices[0]}")
        sys.exit("Incorrect build configuration")
    return library_kind


# --- Argument Grouping Functions ---


def add_core_build_args(parser: argparse.ArgumentParser) -> None:
    """Adds core build process arguments."""
    parser.add_argument("--build_dir", required=True, help="Path to the build directory.")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["Debug"],
        choices=["Debug", "MinSizeRel", "Release", "RelWithDebInfo"],
        help="Configuration(s) to build.",
    )
    parser.add_argument("--update", action="store_true", help="Update makefiles.")
    parser.add_argument("--build", action="store_true", help="Build.")
    parser.add_argument(
        "--clean", action="store_true", help="Run 'cmake --build --target clean' for the selected config/s."
    )
    parser.add_argument(
        "--parallel",
        nargs="?",
        const="0",
        default="1",
        type=int,
        help="Use parallel build. Optional value specifies max jobs (0=num CPUs).",
    )
    parser.add_argument(
        "--target",
        nargs=1,
        action="extend",
        metavar="TARGET",
        dest="targets",
        help="Build a specific CMake target (e.g., winml_dll).",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        action="extend",
        default=[],
        help="Build one or more specific CMake targets.",
    )
    parser.add_argument(
        "--compile_no_warning_as_error",
        action="store_true",
        help="Prevent warnings from being treated as errors during compile. Only works for cmake targets that honor the COMPILE_WARNING_AS_ERROR property",
    )
    parser.add_argument("--build_shared_lib", action="store_true", help="Build a shared library for ONNXRuntime.")
    parser.add_argument(
        "--build_apple_framework", action="store_true", help="Build a macOS/iOS framework for ONNXRuntime."
    )
    parser.add_argument("--enable_lto", action="store_true", help="Enable Link Time Optimization (LTO).")
    parser.add_argument("--use_cache", action="store_true", help="Use ccache in CI")
    parser.add_argument(
        "--use_binskim_compliant_compile_flags",
        action="store_true",
        help="[MS Internal] Use preset compile flags for BinSkim compliance.",
    )


def add_cmake_build_config_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to CMake and general build system configuration."""
    parser.add_argument(
        "--cmake_extra_defines",
        nargs="+",
        action="extend",
        default=[],
        help="Extra CMake definitions (-D<key>=<value>). Provide as <key>=<value>.",
    )
    parser.add_argument("--cmake_path", default="cmake", help="Path to the CMake executable.")
    parser.add_argument(
        "--cmake_generator",
        choices=[
            "MinGW Makefiles",
            "Ninja",
            "NMake Makefiles",
            "NMake Makefiles JOM",
            "Unix Makefiles",
            "Visual Studio 17 2022",
            "Visual Studio 18 2026",
            "Xcode",
        ],
        default=None,  # Will be set later based on OS and WASM
        help="Specify the generator for CMake.",
    )
    parser.add_argument(
        "--use_vcpkg", action="store_true", help="Use vcpkg for dependencies (requires CMAKE_TOOLCHAIN_FILE)."
    )
    parser.add_argument(
        "--use_vcpkg_ms_internal_asset_cache", action="store_true", help="[MS Internal] Use internal vcpkg asset cache."
    )
    parser.add_argument("--terrapin_retrieval_tool_path", help="Path to TerrapinRetrievalTool binary.")
    parser.add_argument("--skip_submodule_sync", action="store_true", help="Skip 'git submodule update'.")
    parser.add_argument("--skip_pip_install", action="store_true", help="Skip 'pip install'.")


def add_testing_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to running tests."""
    parser.add_argument("--test", action="store_true", help="Run unit tests.")
    parser.add_argument("--skip_tests", action="store_true", help="Skip all tests.")
    parser.add_argument(
        "--ctest_path",
        default="ctest",
        help="Path to CTest. Empty string uses script to drive tests.",
    )
    parser.add_argument(
        "--enable_onnx_tests",
        action="store_true",
        help="Run onnx_test_runner against test data. Only used in ONNX Runtime's CI pipelines",
    )
    parser.add_argument("--path_to_protoc_exe", help="Path to protoc executable.")
    parser.add_argument("--cmake_deps_mirror_dir", help="Path to the local mirror of cmake dependencies.")
    parser.add_argument("--fuzz_testing", action="store_true", help="Enable Fuzz testing.")
    parser.add_argument(
        "--enable_symbolic_shape_infer_tests",
        action="store_true",
        help="Run symbolic shape inference tests.",
    )
    parser.add_argument("--skip_onnx_tests", action="store_true", help="Explicitly disable ONNX related tests.")
    parser.add_argument("--skip_winml_tests", action="store_true", help="Explicitly disable WinML related tests.")
    parser.add_argument("--skip_nodejs_tests", action="store_true", help="Explicitly disable Node.js binding tests.")
    parser.add_argument("--ctest_timeout", default="10800", help="Timeout provided to CTest --timeout (seconds).")
    parser.add_argument(
        "--test_parallel",
        default=None,
        type=int,
        help="Max CTest parallel jobs. Defaults to --parallel. Optional value 0 uses num CPUs.",
    )
    parser.add_argument("--enable_transformers_tool_test", action="store_true", help="Enable transformers tool test.")
    parser.add_argument("--build_micro_benchmarks", action="store_true", help="Build ONNXRuntime micro-benchmarks.")
    parser.add_argument("--code_coverage", action="store_true", help="Generate code coverage report (Android only).")


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to ONNX Runtime Training."""
    parser.add_argument(
        "--enable_training",
        action="store_true",
        help="Enable full ORT Training (ORTModule, Training APIs).",
    )
    parser.add_argument("--enable_training_apis", action="store_true", help="Enable ORT Training APIs.")
    parser.add_argument("--enable_training_ops", action="store_true", help="Enable training ops in inference graph.")
    parser.add_argument("--enable_nccl", action="store_true", help="Enable NCCL for distributed training.")
    parser.add_argument("--nccl_home", help="Path to NCCL installation directory.")


def add_general_profiling_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to general (non-EP specific) profiling."""
    parser.add_argument("--enable_memory_profile", action="store_true", help="Enable memory profiling.")


def add_debugging_sanitizer_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to debugging, sanitizers, and compliance."""
    parser.add_argument(
        "--enable_address_sanitizer", action="store_true", help="Enable Address Sanitizer (ASan) (Linux/macOS/Windows)."
    )


def add_documentation_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to documentation generation."""
    parser.add_argument(
        "--gen_doc",
        nargs="?",
        const="yes",
        type=str,
        help="Generate operator/type docs. Use '--gen_doc validate' to check against /docs.",
    )


def add_cross_compile_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for cross-compiling to non-Windows target CPU architectures."""
    parser.add_argument(
        "--rv64",
        action="store_true",
        help="[cross-compiling] Target RISC-V 64-bit.",
    )
    parser.add_argument(
        "--riscv_toolchain_root",
        type=str,
        default="",
        help="Path to RISC-V toolchain root.",
    )
    parser.add_argument(
        "--riscv_qemu_path",
        type=str,
        default="",
        help="Path to RISC-V qemu executable.",
    )


def add_android_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Android platform builds."""
    parser.add_argument("--android", action="store_true", help="Build for Android.")
    parser.add_argument(
        "--android_abi",
        default="arm64-v8a",
        choices=["armeabi-v7a", "arm64-v8a", "x86", "x86_64"],
        help="Target Android ABI.",
    )
    parser.add_argument("--android_api", type=int, default=27, help="Android API Level (e.g., 21).")
    parser.add_argument(
        "--android_sdk_path", type=str, default=os.environ.get("ANDROID_HOME", ""), help="Path to Android SDK."
    )
    parser.add_argument(
        "--android_ndk_path", type=str, default=os.environ.get("ANDROID_NDK_HOME", ""), help="Path to Android NDK."
    )
    parser.add_argument(
        "--android_cpp_shared",
        action="store_true",
        help="Link shared libc++ instead of static (default).",
    )
    parser.add_argument(
        "--android_run_emulator", action="store_true", help="Start an Android emulator if needed for tests."
    )


def add_apple_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Apple platform builds (iOS, macOS, visionOS, tvOS)."""
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument("--ios", action="store_true", help="Build for iOS.")
    platform_group.add_argument("--visionos", action="store_true", help="Build for visionOS.")
    platform_group.add_argument("--tvos", action="store_true", help="Build for tvOS.")
    platform_group.add_argument(
        "--macos",
        choices=["MacOSX", "Catalyst"],
        help="Target platform for macOS build (requires --build_apple_framework).",
    )

    parser.add_argument("--apple_sysroot", default="", help="Specify the macOS platform SDK location name.")
    parser.add_argument(
        "--ios_toolchain_file",
        default="",
        help="Path to iOS CMake toolchain file (defaults to cmake/onnxruntime_ios.toolchain.cmake).",
    )
    parser.add_argument(
        "--visionos_toolchain_file",
        default="",
        help="Path to visionOS CMake toolchain file (defaults to cmake/onnxruntime_visionos.toolchain.cmake).",
    )
    parser.add_argument(
        "--tvos_toolchain_file",
        default="",
        help="Path to tvOS CMake toolchain file (defaults to cmake/onnxruntime_tvos.toolchain.cmake).",
    )
    parser.add_argument("--xcode_code_signing_team_id", default="", help="Development team ID for Xcode code signing.")
    parser.add_argument(
        "--xcode_code_signing_identity", default="", help="Development identity for Xcode code signing."
    )
    parser.add_argument(
        "--use_xcode",
        action="store_const",
        const="Xcode",
        dest="cmake_generator",  # Overwrites the general cmake_generator if specified
        help="Use Xcode CMake generator (macOS only, non-Catalyst).",
    )
    parser.add_argument(
        "--osx_arch",
        default="arm64" if platform.machine() == "arm64" else "x86_64",
        choices=["arm64", "arm64e", "x86_64"],
        help="Target architecture for macOS/iOS (macOS host only).",
    )
    parser.add_argument(
        "--apple_deploy_target",
        type=str,
        help="Minimum deployment target version (e.g., 11.0 for macOS).",
    )


def add_webassembly_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for WebAssembly (WASM) platform builds."""
    parser.add_argument("--build_wasm", action="store_true", help="Build for WebAssembly.")
    parser.add_argument("--build_wasm_static_lib", action="store_true", help="Build WebAssembly static library.")
    parser.add_argument("--emsdk_version", default="4.0.23", help="Specify version of emsdk.")
    parser.add_argument(
        "--enable_wasm_jspi", action="store_true", help="Enable WebAssembly JavaScript Promise Integration."
    )
    parser.add_argument("--enable_wasm_simd", action="store_true", help="Enable WebAssembly SIMD.")
    parser.add_argument("--enable_wasm_relaxed_simd", action="store_true", help="Enable WebAssembly Relaxed SIMD.")
    parser.add_argument("--enable_wasm_threads", action="store_true", help="Enable WebAssembly multi-threading.")
    parser.add_argument("--disable_wasm_exception_catching", action="store_true", help="Disable exception catching.")
    parser.add_argument(
        "--enable_wasm_api_exception_catching",
        action="store_true",
        help="Catch exceptions only at top-level API calls.",
    )
    parser.add_argument(
        "--enable_wasm_exception_throwing_override",
        action="store_true",
        help="Override default behavior to allow throwing exceptions even when catching is generally disabled.",
    )
    parser.add_argument("--wasm_run_tests_in_browser", action="store_true", help="Run WASM tests in a browser.")
    parser.add_argument(
        "--enable_wasm_profiling", action="store_true", help="Enable WASM profiling and preserve function names."
    )
    parser.add_argument("--enable_wasm_debug_info", action="store_true", help="Build WASM with DWARF debug info.")
    parser.add_argument("--wasm_malloc", help="Specify memory allocator for WebAssembly (e.g., dlmalloc).")
    parser.add_argument(
        "--emscripten_settings",
        nargs="+",
        action="extend",
        default=[],
        help="Extra emscripten settings (-s <key>=<value>). Provide as <key>=<value>.",
    )


def add_gdk_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for GDK (Xbox) platform builds."""
    parser.add_argument("--use_gdk", action="store_true", help="Build with the GDK toolchain.")
    default_gdk_edition = ""
    gdk_latest_env = os.environ.get("GameDKLatest", "")  # noqa: SIM112
    if gdk_latest_env:
        try:
            default_gdk_edition = os.path.basename(os.path.normpath(gdk_latest_env))
        except Exception as e:
            warnings.warn(f"Failed to determine GDK edition from GameDKLatest env var: {e}")
    parser.add_argument(
        "--gdk_edition",
        default=default_gdk_edition,
        help="Specific GDK edition to build with (defaults to latest installed via GameDKLatest env var).",
    )
    parser.add_argument("--gdk_platform", default="Scarlett", help="GDK target platform (e.g., Scarlett, XboxOne).")


def add_windows_specific_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments specific to Windows builds or Windows cross-compilation."""
    # Build tools / config
    parser.add_argument("--msvc_toolset", help="MSVC toolset version (e.g., 14.11). Must be >=14.40")
    parser.add_argument("--windows_sdk_version", help="Windows SDK version (e.g., 10.0.19041.0).")
    parser.add_argument("--enable_msvc_static_runtime", action="store_true", help="Statically link MSVC runtimes.")
    parser.add_argument("--caller_framework", type=str, help="Name of the framework calling ONNX Runtime.")

    # Cross-compilation targets hosted on Windows
    parser.add_argument(
        "--x86",
        action="store_true",
        help="[Windows cross-compiling] Target Windows x86.",
    )
    parser.add_argument(
        "--arm",
        action="store_true",
        help="[Windows cross-compiling] Target Windows ARM.",
    )
    parser.add_argument(
        "--arm64",
        action="store_true",
        help="[Windows cross-compiling] Target Windows ARM64.",
    )
    parser.add_argument(
        "--arm64ec",
        action="store_true",
        help="[Windows cross-compiling] Target Windows ARM64EC.",
    )
    parser.add_argument(
        "--buildasx",
        action="store_true",
        help="[Windows cross-compiling] Create ARM64X Binary.",
    )

    parser.add_argument(
        "--disable_memleak_checker",
        action="store_true",
        help="Disable memory leak checker (enabled by default in Debug builds).",
    )
    parser.add_argument(
        "--enable_pix_capture", action="store_true", help="Enable Pix support for GPU debugging (requires D3D12)."
    )

    parser.add_argument(
        "--enable_wcos",
        action="store_true",
        help="Build for Windows Core OS. Link to Windows umbrella libraries instead of kernel32.lib.",
    )

    add_gdk_args(parser)

    # --- WinML ---
    winml_group = parser.add_argument_group("WinML API (Windows)")
    winml_group.add_argument(
        "--use_winml", action="store_true", help="Enable WinML API (Windows). Requires --enable_wcos."
    )
    winml_group.add_argument(
        "--winml_root_namespace_override", type=str, help="Override the namespace WinML builds into."
    )


def add_linux_specific_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments specific to Linux builds."""
    parser.add_argument(
        "--allow_running_as_root",
        action="store_true",
        help="Allow build script to run as root (disallowed by default).",
    )
    parser.add_argument(
        "--enable_external_custom_op_schemas",
        action="store_true",
        help="Enable loading custom op schemas from external shared libraries (Ubuntu only).",
    )


def add_dependency_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to external dependencies."""
    parser.add_argument("--use_full_protobuf", action="store_true", help="Use the full (non-lite) protobuf library.")
    parser.add_argument("--use_mimalloc", action="store_true", help="Use mimalloc memory allocator.")
    parser.add_argument(
        "--external_graph_transformer_path", type=str, help="Path to external graph transformer directory."
    )


def add_extension_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments related to ONNX Runtime Extensions."""
    parser.add_argument(
        "--use_extensions",
        action="store_true",
        help="Enable ONNX Runtime Extensions (uses submodule by default).",
    )
    parser.add_argument(
        "--extensions_overridden_path",
        type=str,
        help="Path to an external ONNX Runtime Extensions source directory.",
    )


def add_size_reduction_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for reducing the binary size."""
    parser.add_argument(
        "--minimal_build",
        default=None,
        nargs="*",
        type=str.lower,
        help="Create a minimal build supporting only ORT format models. "
        "Options: 'extended' (runtime kernel compilation), 'custom_ops'. "
        "e.g., '--minimal_build extended custom_ops'. RTTI disabled automatically.",
    )
    parser.add_argument(
        "--include_ops_by_config",
        type=str,
        help="Include only ops specified in the config file (see docs/Reduced_Operator_Kernel_build.md).",
    )
    parser.add_argument(
        "--enable_reduced_operator_type_support",
        action="store_true",
        help="Further reduce size by limiting operator data types based on --include_ops_by_config file.",
    )
    parser.add_argument("--disable_contrib_ops", action="store_true", help="Disable contrib operators.")
    parser.add_argument("--disable_ml_ops", action="store_true", help="Disable traditional ML operators.")
    parser.add_argument(
        "--disable_generation_ops",
        action="store_true",
        help="Disable generation contrib operators (BeamSearch, WhisperBeamSearch, GreedySearch, Sampling).",
    )
    parser.add_argument("--disable_rtti", action="store_true", help="Disable Run-Time Type Information (RTTI).")
    parser.add_argument(
        "--disable_types",
        nargs="+",
        default=[],
        choices=["float4", "float8", "optional", "sparsetensor", "string"],
        help="Disable selected data types.",
    )
    parser.add_argument(
        "--disable_exceptions",
        action="store_true",
        help="Disable exceptions.",
    )


def add_client_package_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for client package build package."""
    parser.add_argument(
        "--client_package_build",
        action="store_true",
        help="Create ORT package with default settings more appropriate for client/on-device workloads.",
    )


def add_threadpool_callback_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for per-session thread pool work callbacks."""
    parser.add_argument(
        "--enable_session_threadpool_callbacks",
        action="store_true",
        help="Enable per-session thread pool work callbacks.",
    )


def add_python_binding_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Python bindings."""
    parser.add_argument("--enable_pybind", action="store_true", help="Enable Python bindings.")
    parser.add_argument("--build_wheel", action="store_true", help="Build Python wheel package.")
    parser.add_argument(
        "--wheel_name_suffix",
        help="Suffix for wheel name (used for nightly builds).",
    )
    parser.add_argument("--skip-keras-test", action="store_true", help="Skip Keras-related tests.")


def add_csharp_binding_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for C# bindings."""
    parser.add_argument(
        "--build_csharp",
        action="store_true",
        help="Build C# DLL and NuGet package (CI usage). Use --build_nuget for local build.",
    )
    parser.add_argument(
        "--build_nuget",
        action="store_true",
        help="Build C# DLL and NuGet package locally (Windows/Linux only).",
    )
    parser.add_argument(
        "--msbuild_extra_options",
        nargs="+",
        action="extend",
        default=[],
        help="Extra MSBuild properties (/p:key=value). Provide as key=value.",
    )


def add_java_binding_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Java bindings."""
    parser.add_argument("--build_java", action="store_true", help="Build Java bindings.")


def add_nodejs_binding_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Node.js bindings."""
    parser.add_argument("--build_nodejs", action="store_true", help="Build Node.js binding and NPM package.")
    # Note: --skip_nodejs_tests is handled in add_testing_args


def add_objc_binding_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for Objective-C bindings."""
    parser.add_argument("--build_objc", action="store_true", help="Build Objective-C binding.")


def add_execution_provider_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for enabling various Execution Providers (EPs)."""
    # --- CUDA ---
    cuda_group = parser.add_argument_group("CUDA Execution Provider")
    cuda_group.add_argument("--use_cuda", action="store_true", help="Enable CUDA EP.")
    cuda_group.add_argument("--cuda_version", help="CUDA toolkit version (e.g., 11.8). Auto-detect if omitted.")
    cuda_group.add_argument("--cuda_home", help="Path to CUDA toolkit (uses CUDA_HOME env var if unset).")
    cuda_group.add_argument("--cudnn_home", help="Path to cuDNN (uses CUDNN_HOME env var if unset).")
    cuda_group.add_argument("--enable_cuda_line_info", action="store_true", help="Enable CUDA line info for debugging.")
    cuda_group.add_argument("--enable_cuda_nhwc_ops", action="store_true", help="[Deprecated] Default enabled.")
    cuda_group.add_argument("--disable_cuda_nhwc_ops", action="store_true", help="Disable CUDA NHWC layout ops.")
    cuda_group.add_argument("--enable_cuda_minimal_build", action="store_true", help="Enable CUDA minimal build.")
    cuda_group.add_argument(
        "--nvcc_threads",
        default=4,
        type=int,
        help="Max NVCC threads per parallel job (default is 4).",
    )
    cuda_group.add_argument(
        "--flash_nvcc_threads",
        default=-1,
        type=int,
        help="Max NVCC threads per parallel job for flash attention (default is same value of --nvcc_threads).",
    )
    # CUDA-specific profiling
    cuda_group.add_argument(
        "--enable_nvtx_profile", action="store_true", help="Enable NVTX profile markers for CUDA EP."
    )
    cuda_group.add_argument(
        "--enable_cuda_profiling",
        action="store_true",
        help="Enable CUDA kernel profiling (requires CUPTI in PATH).",
    )

    # --- CPU ---
    cpu_group = parser.add_argument_group("CPU Execution Provider")
    cpu_group.add_argument("--no_sve", action="store_true", help="Disable building with SVE support.")
    # The following enables building ORT with NCHWc Neon ARM kernels.
    # At the time of writing, it is turned OFF by default because its performance relative to "regular" NCHW kernels
    # is not good at smaller thread counts. But its speed-up is non-negligible with higher thread counts on supporting
    # ARM platforms.
    # Once the gap is closed for smaller thread counts, it can be turned on by default.
    # See https://github.com/microsoft/onnxruntime/pull/25580#issuecomment-3335056846 for benchmarking details.
    cpu_group.add_argument(
        "--enable_arm_neon_nchwc", action="store_true", help="Enables building with NCHWc ARM kernels."
    )
    cpu_group.add_argument(
        "--enable_rvv",
        action="store_true",
        help="Enable riscv64 MLAS kernels that use the RISC-V Vector extension.",
    )

    # --- DNNL (formerly MKL-DNN / oneDNN) ---
    dnnl_group = parser.add_argument_group("DNNL Execution Provider")
    dnnl_group.add_argument("--use_dnnl", action="store_true", help="Enable DNNL EP.")
    dnnl_group.add_argument(
        "--dnnl_gpu_runtime",
        action="store",
        default="",
        type=str.lower,
        choices=["ocl", ""],
        help="DNNL GPU backend (e.g., ocl for OpenCL).",
    )
    dnnl_group.add_argument("--dnnl_opencl_root", action="store", default="", help="Path to OpenCL SDK (for DNNL GPU).")
    dnnl_group.add_argument(
        "--dnnl_aarch64_runtime",
        action="store",
        default="",
        type=str.lower,
        choices=["acl", ""],
        help="DNNL AArch64 backend (e.g., acl for Arm Compute Library).",
    )
    dnnl_group.add_argument(
        "--dnnl_acl_root", action="store", default="", help="Path to Arm Compute Library (ACL) root."
    )

    # --- OpenVINO ---
    openvino_group = parser.add_argument_group("OpenVINO Execution Provider")
    openvino_group.add_argument(
        "--use_openvino",
        nargs="?",
        const="CPU",  # Default device if only flag is present
        type=_openvino_verify_device_type,
        help="Enable OpenVINO EP for specific hardware (e.g., CPU, GPU, NPU, HETERO:GPU,CPU).",
    )

    # --- TensorRT ---
    trt_group = parser.add_argument_group("TensorRT Execution Provider")
    trt_group.add_argument("--use_tensorrt", action="store_true", help="Enable TensorRT EP.")
    trt_group.add_argument(
        "--use_tensorrt_builtin_parser",
        action="store_true",
        default=True,
        help="Use TensorRT internal ONNX parser (default).",
    )
    trt_group.add_argument("--use_tensorrt_oss_parser", action="store_true", help="Use TensorRT OSS ONNX parser.")
    trt_group.add_argument("--tensorrt_home", help="Path to TensorRT installation directory.")
    # --- DirectML ---
    dml_group = parser.add_argument_group("DirectML Execution Provider (Windows)")
    dml_group.add_argument("--use_dml", action="store_true", help="Enable DirectML EP (Windows).")
    dml_group.add_argument("--dml_path", type=str, default="", help="Path to custom DirectML SDK.")
    dml_group.add_argument("--dml_external_project", action="store_true", help="Build DirectML as an external project.")

    # --- NNAPI ---
    nnapi_group = parser.add_argument_group("NNAPI Execution Provider (Android)")
    nnapi_group.add_argument("--use_nnapi", action="store_true", help="Enable NNAPI EP (Android).")
    nnapi_group.add_argument("--nnapi_min_api", type=int, help="Minimum Android API level for NNAPI (>= 27).")

    # --- CoreML ---
    coreml_group = parser.add_argument_group("CoreML Execution Provider (Apple)")
    coreml_group.add_argument("--use_coreml", action="store_true", help="Enable CoreML EP (Apple platforms).")

    # --- QNN ---
    qnn_group = parser.add_argument_group("QNN Execution Provider (Qualcomm)")
    qnn_group.add_argument(
        "--use_qnn",
        nargs="?",
        const="shared_lib",  # Default linkage if only flag is present
        type=_qnn_verify_library_kind,
        help="Enable QNN EP. Optionally specify 'shared_lib' (default) or 'static_lib'.",
    )
    qnn_group.add_argument("--qnn_home", help="Path to QNN SDK directory.")

    # --- SNPE ---
    snpe_group = parser.add_argument_group("SNPE Execution Provider (Qualcomm)")
    snpe_group.add_argument("--use_snpe", action="store_true", help="Enable SNPE EP.")
    snpe_group.add_argument("--snpe_root", help="Path to SNPE SDK root directory.")

    # --- Vitis-AI ---
    vitis_group = parser.add_argument_group("Vitis-AI Execution Provider (Xilinx)")
    vitis_group.add_argument("--use_vitisai", action="store_true", help="Enable Vitis-AI EP.")

    # --- ACL (Arm Compute Library) ---
    acl_group = parser.add_argument_group("[DEPRECATED] ACL Execution Provider")
    acl_group.add_argument(
        "--use_acl",
        action="store_true",
        help="Enable ACL EP (ARM architectures). The ACL EP is deprecated and will be removed in a future release.",
    )
    acl_group.add_argument("--acl_home", help="Path to ACL home directory.")
    acl_group.add_argument("--acl_libs", help="Path to ACL libraries directory.")
    acl_group.add_argument("--no_kleidiai", action="store_true", help="Disable KleidiAI integration (used with ACL).")

    # --- Qualcomm QMX Library ---
    qmx_group = parser.add_argument_group("QMX kernel library")
    qmx_group.add_argument(
        "--use_qmx", action="store_true", help="Enable Qualcomm QMX kernel to coexist with Arm KleidiAI."
    )

    # --- RKNPU ---
    rknpu_group = parser.add_argument_group("RKNPU Execution Provider")
    rknpu_group.add_argument("--use_rknpu", action="store_true", help="Enable RKNPU EP.")

    # --- CANN (Huawei Ascend) ---
    cann_group = parser.add_argument_group("CANN Execution Provider")
    cann_group.add_argument("--use_cann", action="store_true", help="Enable CANN EP.")
    cann_group.add_argument("--cann_home", help="Path to CANN installation directory.")

    # --- MIGraphX (AMD) ---
    migx_group = parser.add_argument_group("MIGraphX Execution Provider")
    migx_group.add_argument("--use_migraphx", action="store_true", help="Enable MIGraphX EP.")
    migx_group.add_argument("--migraphx_home", help="Path to MIGraphX installation directory.")
    # --rocm_version and --rocm_home are deprecated. See https://github.com/microsoft/onnxruntime/issues/26801.
    migx_group.add_argument("--rocm_version", help="ROCm stack version. This option is deprecated and has no effect.")
    migx_group.add_argument(
        "--rocm_home", help="Path to ROCm installation directory. This option is deprecated and has no effect."
    )

    # --- WebNN ---
    webnn_group = parser.add_argument_group("WebNN Execution Provider")
    webnn_group.add_argument("--use_webnn", action="store_true", help="Enable WebNN EP.")

    # --- JSEP (JavaScript EP for WASM) ---
    jsep_group = parser.add_argument_group("JSEP Execution Provider (WebAssembly)")
    jsep_group.add_argument("--use_jsep", action="store_true", help="Enable JavaScript EP (used with WebAssembly).")

    # --- WebGPU ---
    webgpu_group = parser.add_argument_group("WebGPU Execution Provider")
    webgpu_group.add_argument(
        "--use_webgpu",
        nargs="?",
        const="static_lib",
        type=_webgpu_verify_library_kind,
        help="Enable WebGPU EP. Optionally specify 'static_lib' (default), 'shared_lib', or 'static_plugin'. "
        "'static_lib' builds the EP as an internal ORT EP. 'shared_lib' builds it as a separate plugin EP "
        "library that is loaded at runtime. 'static_plugin' builds it as a plugin EP that is linked into the "
        "ORT binary and registered at environment creation.",
    )
    webgpu_group.add_argument(
        "--use_external_dawn", action="store_true", help="Use external Dawn dependency for WebGPU."
    )
    webgpu_group.add_argument(
        "--use_dawn_agility_sdk",
        action="store_true",
        help=(
            "Build Dawn's D3D12 backend with the Agility SDK "
            "(Windows desktop x86, x64, or ARM64 only; Python wheels are supported; "
            "C#, NuGet, Java, and Node.js packaging through build.py is unsupported. "
            "Package the WebGPU plugin separately for Python or NuGet)."
        ),
    )
    webgpu_group.add_argument(
        "--wgsl_template",
        choices=["static"],
        default="static",
        help="Deprecated no-op. WGSL template generation is always static; kept for backward compatibility.",
    )

    # --- XNNPACK ---
    xnn_group = parser.add_argument_group("XNNPACK Execution Provider")
    xnn_group.add_argument("--use_xnnpack", action="store_true", help="Enable XNNPACK EP.")

    # --- VSINPU (VeriSilicon NPU) ---
    vsi_group = parser.add_argument_group("VSINPU Execution Provider")
    vsi_group.add_argument("--use_vsinpu", action="store_true", help="Enable VSINPU EP.")

    # --- Azure ---
    azure_group = parser.add_argument_group("Azure Execution Provider")
    azure_group.add_argument("--use_azure", action="store_true", help="Enable Azure EP.")

    # --- DX Interop Feature ---
    dx_interop_group = parser.add_argument_group("DirectX Interop Feature")
    dx_interop_group.add_argument(
        "--enable_dx_interop",
        action="store_true",
        help="Enable DirectX Interop feature for graphics API synchronization.",
    )


def add_other_feature_args(parser: argparse.ArgumentParser) -> None:
    """Adds arguments for other miscellaneous features."""
    parser.add_argument("--enable_lazy_tensor", action="store_true", help="Enable ORT backend for PyTorch LazyTensor.")
    parser.add_argument("--ms_experimental", action="store_true", help="Build Microsoft experimental operators.")
    parser.add_argument(
        "--enable_msinternal", action="store_true", help="[MS Internal] Enable Microsoft internal build features."
    )
    parser.add_argument("--use_lock_free_queue", action="store_true", help="Use lock-free task queue for threadpool.")
    parser.add_argument(
        "--enable_generic_interface",
        action="store_true",
        help="Build ORT shared lib with compatible bridge for primary EPs (TRT, OV, QNN, VitisAI), excludes tests.",
    )
    # Telemetry arguments (cross-platform)
    telemetry_group = parser.add_mutually_exclusive_group()
    telemetry_group.add_argument(
        "--no_telemetry",
        dest="use_telemetry",
        action="store_false",
        help="Disable telemetry. Telemetry is enabled by default for supported native builds.",
    )
    telemetry_group.add_argument(
        "--use_telemetry",
        dest="use_telemetry_legacy",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    telemetry_group.add_argument(
        "--use_windows_telemetry",
        action="store_true",
        help="Use the legacy Windows TraceLogging telemetry backend instead of 1DS.",
    )


def is_cross_compiling(args: argparse.Namespace) -> bool:
    return any(
        [
            # Check existence before accessing for conditionally added args
            getattr(args, "x86", False),
            getattr(args, "arm", False),
            getattr(args, "arm64", False),
            getattr(args, "arm64ec", False),
            args.rv64,  # General cross-compile arg
            args.android,
            # Check existence for macOS/Apple specific args
            getattr(args, "ios", False),
            getattr(args, "visionos", False),
            getattr(args, "tvos", False),
            args.build_wasm,
            getattr(args, "use_gdk", False),  # GDK args added conditionally
        ]
    )


def target_supports_telemetry(args: argparse.Namespace) -> bool:
    """Returns whether the selected build target has a telemetry provider."""
    if (
        args.build_wasm
        or getattr(args, "disable_exceptions", False)
        or getattr(args, "rv64", False)
        or getattr(args, "visionos", False)
        or getattr(args, "tvos", False)
        or getattr(args, "macos", None) == "Catalyst"
    ):
        return False

    if getattr(args, "android", False) or is_windows() or is_macOS():
        return True

    if not is_linux():
        return False

    return platform.machine().lower() in {
        "aarch64",
        "amd64",
        "arm",
        "arm64",
        "armv6l",
        "armv7l",
        "armv8l",
        "i386",
        "i686",
        "x86",
        "x86_64",
    }


# --- Main Argument Parsing Function ---
def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for the ONNX Runtime build."""

    class Parser(argparse.ArgumentParser):
        # override argument file line parsing behavior
        # - allow multiple arguments per line and handle quotes
        # - allow comment lines starting with '#'
        def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
            if arg_line.lstrip().startswith("#"):  # ignore comment lines
                return []
            return shlex.split(arg_line)

    parser = Parser(
        description="ONNXRuntime CI build driver.",
        usage=f"""
        Default behavior is --update --build --test for native architecture builds.
        Default behavior is --update --build for cross-compiled builds.

        The Update phase will update git submodules, and run cmake to generate makefiles.
        The Build phase will build all projects.
        The Test phase will run all unit tests, and optionally the ONNX tests.

        Use the individual flags (--update, --build, --test) to only run specific stages.

        Arguments can also be passed in an argument file prefixed with '@'.
        E.g., `{sys.argv[0]} @arguments.txt`.
        Argument files may contain comment lines starting with '#'. They will be ignored.
        """,
        fromfile_prefix_chars="@",  # Allow args from file (@filename)
    )

    # Add arguments by category
    add_core_build_args(parser)
    add_cmake_build_config_args(parser)
    add_testing_args(parser)
    add_training_args(parser)
    add_general_profiling_args(parser)
    add_debugging_sanitizer_args(parser)
    add_documentation_args(parser)
    add_cross_compile_args(parser)  # Non-Windows cross-compile args
    add_android_args(parser)
    add_webassembly_args(parser)
    add_dependency_args(parser)
    add_extension_args(parser)
    add_size_reduction_args(parser)
    add_client_package_args(parser)
    add_threadpool_callback_args(parser)

    # Language Bindings
    add_python_binding_args(parser)
    add_csharp_binding_args(parser)
    add_java_binding_args(parser)
    add_nodejs_binding_args(parser)
    add_objc_binding_args(parser)

    # Execution Providers (now includes EP-specific profiling args)
    add_execution_provider_args(parser)

    # Other Features
    add_other_feature_args(parser)

    # Platform specific args (now includes Windows cross-compile targets & specific config/debug args)
    if is_windows():
        add_windows_specific_args(parser)
    elif is_macOS():
        add_apple_args(parser)
    else:  # Assuming Linux or other non-Windows, non-macOS Unix-like
        add_linux_specific_args(parser)

    # --- Parse Arguments ---
    args: argparse.Namespace = parser.parse_args()

    # --- Post-processing and Defaults ---

    # Normalize paths
    if args.android_sdk_path:
        args.android_sdk_path = os.path.normpath(args.android_sdk_path)
    if args.android_ndk_path:
        args.android_ndk_path = os.path.normpath(args.android_ndk_path)

    # Treat --build_wasm_static_lib as implying --build_wasm
    if args.build_wasm_static_lib:
        args.build_wasm = True

    if args.use_telemetry_legacy:
        warnings.warn(
            "--use_telemetry is deprecated because telemetry is enabled by default. "
            "For native Windows targets it retains its historical TraceLogging behavior; use "
            "--use_windows_telemetry to request that backend explicitly.",
            FutureWarning,
            stacklevel=2,
        )
        if not target_supports_telemetry(args):
            parser.error("--use_telemetry requires a telemetry-capable target")
        args.use_telemetry = True
        if is_windows() and not args.android:
            args.use_windows_telemetry = True

    if not target_supports_telemetry(args):
        args.use_telemetry = False
    if args.use_windows_telemetry and (not is_windows() or args.android or not target_supports_telemetry(args)):
        parser.error("--use_windows_telemetry requires a telemetry-capable native Windows target")

    # Handle WASM exception logic
    if args.enable_wasm_api_exception_catching:
        args.disable_wasm_exception_catching = True  # Catching at API level implies disabling broader catching
    if not args.disable_wasm_exception_catching or args.enable_wasm_api_exception_catching:
        # Doesn't make sense to catch if nothing throws
        args.enable_wasm_exception_throwing_override = True

    # Set default CMake generator if not specified
    # Check if cmake_generator attribute exists (it might if --use_xcode was used)
    # before checking if it's None.
    if not hasattr(args, "cmake_generator") or args.cmake_generator is None:
        if is_windows():
            # Default to Ninja for WASM on Windows for potential speedup, VS otherwise
            args.cmake_generator = "Ninja" if args.build_wasm else "Visual Studio 17 2022"
        # else: Linux/macOS default (usually Makefiles or Ninja) is handled by CMake itself

    # Handle deprecated args
    if hasattr(args, "enable_cuda_nhwc_ops") and args.enable_cuda_nhwc_ops:
        warnings.warn("The argument '--enable_cuda_nhwc_ops' is deprecated and enabled by default.", FutureWarning)

    if args.use_acl:
        warnings.warn("The ACL EP is deprecated and will be removed in a future release.", FutureWarning)

    # Default behavior (update/build/test) if no action flags are specified
    # Determine if it's a cross-compiled build (approximated by checking common cross-compile flags)
    if not (args.update or args.build or args.test or args.clean or args.gen_doc):
        args.update = True
        args.build = True
        # Only default to running tests for native builds if tests aren't explicitly skipped
        if not is_cross_compiling(args) and not args.skip_tests:
            args.test = True
        elif is_cross_compiling(args):
            print(
                "Cross-compiling build detected: Defaulting to --update --build. Specify --test explicitly to run tests."
            )

    # The Emscripten vcpkg triplets disable the dependencies' exceptions only for minimal builds, so a
    # non-minimal wasm build with --disable_exceptions would mix exception settings.
    if args.build_wasm and args.disable_exceptions and args.minimal_build is None:
        parser.error("--build_wasm --disable_exceptions requires --minimal_build to be specified.")
    if is_windows():
        if getattr(args, "use_winml", False) and not getattr(args, "enable_wcos", False):
            parser.error("--use_winml requires --enable_wcos to be specified.")
        if hasattr(args, "msvc_toolset") and args.msvc_toolset:
            try:
                # Extract major.minor version parts (e.g., "14.36")
                version_parts = args.msvc_toolset.split(".")
                if len(version_parts) >= 2:
                    major = int(version_parts[0])
                    minor = int(version_parts[1])
                    # Check known problematic range based on previous script comments/help text
                    # Refined check: >= 14.36 and <= 14.39
                    # Help text now says >= 14.40 is required, so check < 14.40
                    if major == 14 and minor < 40:
                        # You could make this an error or just a warning
                        # parser.error(f"MSVC toolset version {args.msvc_toolset} is not supported. Use 14.40 or higher.")
                        warnings.warn(
                            f"Specified MSVC toolset version {args.msvc_toolset} might have compatibility issues. Version 14.40 or higher is recommended."
                        )

            except (ValueError, IndexError):
                warnings.warn(
                    f"Could not parse MSVC toolset version: {args.msvc_toolset}. Skipping compatibility check."
                )

    elif is_macOS():
        if getattr(args, "build_apple_framework", False) and not any(
            [
                getattr(args, "ios", False),
                getattr(args, "macos", None),
                getattr(args, "visionos", False),
                getattr(args, "tvos", False),
            ]
        ):
            parser.error("--build_apple_framework requires --ios, --macos, --visionos, or --tvos to be specified.")

        if getattr(args, "macos", None) and not getattr(args, "build_apple_framework", False):
            parser.error("--macos target requires --build_apple_framework.")
    return args

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# Licensed under the MIT License.
from __future__ import annotations

import contextlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def version_to_tuple(version: str) -> tuple:
    v = []
    for s in version.split("."):
        with contextlib.suppress(ValueError):
            v.append(int(s))
    return tuple(v)


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.insert(0, os.path.join(REPO_DIR, "tools", "python"))
import util.android as android  # noqa: E402
from build_args import parse_arguments  # noqa: E402
from util import (  # noqa: E402
    generate_android_triplets,
    generate_linux_triplets,
    generate_macos_triplets,
    generate_vcpkg_triplets_for_emscripten,
    generate_windows_triplets,
    get_logger,
    is_linux,
    is_macOS,
    is_windows,
    parse_qnn_version_from_sdk_yaml,
    run,
)
from vcpkg_tool_info import get_vcpkg_release_tag  # noqa: E402

log = get_logger("build")


class BaseError(Exception):
    """Base class for errors originating from build.py."""


class BuildError(BaseError):
    """Error from running build steps."""

    def __init__(self, *messages):
        super().__init__("\n".join(messages))


class UsageError(BaseError):
    """Usage related error."""

    def __init__(self, message):
        super().__init__(message)


def _check_python_version():
    required_minor_version = 8
    if (sys.version_info.major, sys.version_info.minor) < (3, required_minor_version):
        raise UsageError(
            f"Invalid Python version. At least Python 3.{required_minor_version} is required. "
            f"Actual Python version: {sys.version}"
        )


_check_python_version()


def is_reduced_ops_build(args):
    return args.include_ops_by_config is not None


def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    if command_or_path and command_or_path.strip():
        executable_path = shutil.which(command_or_path)
        if executable_path is None:
            raise BuildError(f"Failed to resolve executable path for '{command_or_path}'.")
        return os.path.abspath(executable_path)
    else:
        return None


def get_linux_distro():
    try:
        with open("/etc/os-release") as f:
            dist_info = dict(line.strip().split("=", 1) for line in f)
        return dist_info.get("NAME", "").strip('"'), dist_info.get("VERSION", "").strip('"')
    except (OSError, ValueError):
        return "", ""


def get_config_build_dir(build_dir, config):
    # build directory per configuration
    return os.path.join(build_dir, config)


def run_subprocess(
    args,
    cwd=None,
    capture_stdout=False,
    dll_path=None,
    shell=False,
    env=None,
    python_path=None,
    cuda_home=None,
):
    if env is None:
        env = {}
    if isinstance(args, str):
        raise ValueError("args should be a sequence of strings, not a string")

    my_env = os.environ.copy()
    if dll_path:
        if is_windows():
            if "PATH" in my_env:
                my_env["PATH"] = dll_path + os.pathsep + my_env["PATH"]
            else:
                my_env["PATH"] = dll_path
        else:
            if "LD_LIBRARY_PATH" in my_env:
                my_env["LD_LIBRARY_PATH"] += os.pathsep + dll_path
            else:
                my_env["LD_LIBRARY_PATH"] = dll_path
    # Add nvcc's folder to PATH env so that our cmake file can find nvcc
    if cuda_home:
        my_env["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + my_env["PATH"]

    if python_path:
        if "PYTHONPATH" in my_env:
            my_env["PYTHONPATH"] += os.pathsep + python_path
        else:
            my_env["PYTHONPATH"] = python_path

    my_env.update(env)
    log.info(" ".join(args))
    return run(*args, cwd=cwd, capture_stdout=capture_stdout, shell=shell, env=my_env)


def get_onnx_backend_test_environment(_use_cuda):
    return {
        "ALLOW_RELEASED_ONNX_OPSET_ONLY": "1",
        "ORT_BACKEND_TEST_ALLOW_UNRELEASED_OPSETS": "0",
    }


def update_submodules(source_dir):
    run_subprocess(["git", "submodule", "sync", "--recursive"], cwd=source_dir)
    run_subprocess(["git", "submodule", "update", "--init", "--recursive"], cwd=source_dir)


def setup_test_data(source_onnx_model_dir, dest_model_dir_name, build_dir, configs):
    # create the symlink/shortcut of onnx models dir under build_dir
    # currently, there're 2 sources of onnx models, one is build in OS image, another is
    # from {source_dir}/js/test, which is downloaded from onnx web.
    if is_windows():
        src_model_dir = os.path.join(build_dir, dest_model_dir_name)
        if os.path.exists(source_onnx_model_dir) and not os.path.exists(src_model_dir):
            log.debug(f"creating shortcut {source_onnx_model_dir} -> {src_model_dir}")
            run_subprocess(["mklink", "/D", "/J", src_model_dir, source_onnx_model_dir], shell=True)
        for config in configs:
            config_build_dir = get_config_build_dir(build_dir, config)
            os.makedirs(config_build_dir, exist_ok=True)
            dest_model_dir = os.path.join(config_build_dir, dest_model_dir_name)
            if os.path.exists(source_onnx_model_dir) and not os.path.exists(dest_model_dir):
                log.debug(f"creating shortcut {source_onnx_model_dir} -> {dest_model_dir}")
                run_subprocess(["mklink", "/D", "/J", dest_model_dir, source_onnx_model_dir], shell=True)
            elif os.path.exists(src_model_dir) and not os.path.exists(dest_model_dir):
                log.debug(f"creating shortcut {src_model_dir} -> {dest_model_dir}")
                run_subprocess(["mklink", "/D", "/J", dest_model_dir, src_model_dir], shell=True)
    else:
        src_model_dir = os.path.join(build_dir, dest_model_dir_name)
        if os.path.exists(source_onnx_model_dir) and not os.path.exists(src_model_dir):
            log.debug(f"create symlink {source_onnx_model_dir} -> {src_model_dir}")
            os.symlink(source_onnx_model_dir, src_model_dir, target_is_directory=True)


def use_dev_mode(args):
    if args.compile_no_warning_as_error:
        return False
    if args.use_acl:
        return False
    if is_macOS() and (args.ios or args.visionos or args.tvos):
        return False
    if args.use_qnn:
        return True
    SYSTEM_COLLECTIONURI = os.getenv("SYSTEM_COLLECTIONURI")  # noqa: N806
    if SYSTEM_COLLECTIONURI:
        return False
    return True


def add_default_definition(definition_list, key, default_value):
    for x in definition_list:
        if x.startswith(key + "="):
            return definition_list
    definition_list.append(key + "=" + default_value)


def number_of_parallel_jobs(args):
    return os.cpu_count() if args.parallel == 0 else args.parallel


def number_of_test_parallel_jobs(args):
    if args.test_parallel is None:
        return number_of_parallel_jobs(args)
    return os.cpu_count() if args.test_parallel == 0 else args.test_parallel


def number_of_nvcc_threads(args):
    if args.nvcc_threads >= 0:
        return args.nvcc_threads

    return 4


def number_of_flash_nvcc_threads(args):
    if args.flash_nvcc_threads >= 0:
        return args.flash_nvcc_threads

    return number_of_nvcc_threads(args)


# See https://learn.microsoft.com/en-us/vcpkg/commands/install
def generate_vcpkg_install_options(build_dir, args):
    # NOTE: each option string should not contain any whitespace.
    vcpkg_install_options = ["--x-feature=tests"]
    if args.use_acl:
        vcpkg_install_options.append("--x-feature=acl-ep")
    if args.use_azure:
        vcpkg_install_options.append("--x-feature=azure-ep")
    if args.use_cann:
        vcpkg_install_options.append("--x-feature=cann-ep")
    if args.use_coreml:
        vcpkg_install_options.append("--x-feature=coreml-ep")
    if args.use_cuda:
        vcpkg_install_options.append("--x-feature=cuda-ep")
    if args.use_dml:
        vcpkg_install_options.append("--x-feature=dml-ep")
    if args.use_dnnl:
        vcpkg_install_options.append("--x-feature=dnnl-ep")
    if args.use_jsep:
        vcpkg_install_options.append("--x-feature=js-ep")
    if args.use_migraphx:
        vcpkg_install_options.append("--x-feature=migraphx-ep")
    if args.use_nnapi:
        vcpkg_install_options.append("--x-feature=nnapi-ep")
    if args.use_openvino:
        vcpkg_install_options.append("--x-feature=openvino-ep")
    if args.use_qnn:
        vcpkg_install_options.append("--x-feature=qnn-ep")
    if args.use_rknpu:
        vcpkg_install_options.append("--x-feature=rknpu-ep")
    if args.use_tensorrt:
        vcpkg_install_options.append("--x-feature=tensorrt-ep")
    if args.use_vitisai:
        vcpkg_install_options.append("--x-feature=vitisai-ep")
    if args.use_vsinpu:
        vcpkg_install_options.append("--x-feature=vsinpu-ep")
    if args.use_webgpu:
        vcpkg_install_options.append("--x-feature=webgpu-ep")
    if args.use_webnn:
        vcpkg_install_options.append("--x-feature=webnn-ep")
    if args.use_xnnpack:
        vcpkg_install_options.append("--x-feature=xnnpack-ep")
    if args.use_telemetry and not args.use_windows_telemetry and not args.android and not args.build_wasm:
        vcpkg_install_options.append("--x-feature=telemetry")
    overlay_triplets_dir = None

    folder_name_parts = []
    if args.enable_address_sanitizer:
        folder_name_parts.append("asan")
    if args.use_binskim_compliant_compile_flags and not args.android and not args.build_wasm:
        folder_name_parts.append("binskim")
    if args.disable_rtti:
        folder_name_parts.append("nortti")
    if args.build_wasm and not args.disable_wasm_exception_catching and not args.disable_exceptions:
        folder_name_parts.append("exception_catching")
    if args.disable_exceptions:
        folder_name_parts.append("noexception")
    if args.minimal_build is not None:
        folder_name_parts.append("minimal")
    if args.build_wasm or len(folder_name_parts) == 0:
        # It's hard to tell whether we must use a custom triplet or not. The official triplets work fine for most common situations. However, if a Windows build has set msvc toolset version via args.msvc_toolset then we need to, because we need to ensure all the source code are compiled by the same MSVC toolset version otherwise we will hit link errors like "error LNK2019: unresolved external symbol __std_mismatch_4 referenced in function ..."
        # So, to be safe we always use a custom triplet.
        folder_name = "default"
    else:
        folder_name = "_".join(folder_name_parts)
    overlay_triplets_dir = (Path(build_dir) / folder_name).absolute()

    vcpkg_install_options.append(f"--overlay-triplets={overlay_triplets_dir}")
    if "AGENT_TEMPDIRECTORY" in os.environ:
        temp_dir = os.environ["AGENT_TEMPDIRECTORY"]
        vcpkg_install_options.append(f"--x-buildtrees-root={temp_dir}")
    elif "RUNNER_TEMP" in os.environ:
        temp_dir = os.environ["RUNNER_TEMP"]
        vcpkg_install_options.append(f"--x-buildtrees-root={temp_dir}")

    # Config asset cache
    if args.use_vcpkg_ms_internal_asset_cache:
        terrapin_path_candidates = [
            args.terrapin_retrieval_tool_path,
            shutil.which("TerrapinRetrievalTool"),
        ]
        if is_windows():
            terrapin_path_candidates.append("C:\\local\\Terrapin\\TerrapinRetrievalTool.exe")

        terrapin_cmd_path = next(
            (path for path in terrapin_path_candidates if path is not None and os.path.exists(path)),
            None,
        )

        if terrapin_cmd_path is not None:
            quoted_terrapin_cmd_path = f'"{terrapin_cmd_path}"' if is_windows() else shlex.quote(terrapin_cmd_path)
            vcpkg_install_options.append(
                "--x-asset-sources=x-script,"
                + quoted_terrapin_cmd_path
                + " -b https://vcpkg.storage.devpackages.microsoft.io/artifacts/ -a true -u Environment -p {url} -s {sha512} -d {dst}\\;x-block-origin"
            )
        else:
            vcpkg_install_options.append(
                "--x-asset-sources=x-azurl,https://vcpkg.storage.devpackages.microsoft.io/artifacts/\\;x-block-origin"
            )

    return vcpkg_install_options


def _get_vctools_install_dir(args):
    vctools_dir = os.environ.get("VCToolsInstallDir")  # noqa: SIM112
    if vctools_dir:
        return Path(vctools_dir)

    vswhere_candidates = []
    program_files_x86 = os.environ.get("ProgramFiles(x86)")  # noqa: SIM112
    if program_files_x86:
        vswhere_candidates.append(Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe")
    if vswhere_path := shutil.which("vswhere.exe"):
        vswhere_candidates.append(Path(vswhere_path))

    installation_paths = []
    for vswhere_path in vswhere_candidates:
        if not vswhere_path.is_file():
            continue
        try:
            result = subprocess.run(
                [
                    str(vswhere_path),
                    "-products",
                    "*",
                    "-property",
                    "installationPath",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            continue
        if result.returncode == 0 and result.stdout.strip():
            installation_paths = [Path(path) for path in result.stdout.splitlines() if path.strip()]
            break

    for installation_path in installation_paths:
        msvc_root = installation_path / "VC" / "Tools" / "MSVC"
        if args.msvc_toolset:
            matching_toolsets = sorted(
                (path for path in msvc_root.glob(f"{args.msvc_toolset}*") if path.is_dir()),
                key=lambda path: version_to_tuple(path.name),
                reverse=True,
            )
            if matching_toolsets:
                return matching_toolsets[0]
            continue

        default_version_file = installation_path / "VC" / "Auxiliary" / "Build" / "Microsoft.VCToolsVersion.default.txt"
        try:
            default_version = default_version_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if default_version:
            vctools_dir = msvc_root / default_version
            if vctools_dir.is_dir():
                return vctools_dir

    return None


def get_msvc_spectre_lib_dir(args):
    """Return the directory that holds the MSVC Spectre-mitigated CRT/STL static libraries for the
    target architecture, or None if it cannot be located.

    The /Qspectre compile flag only mitigates ONNX Runtime's own object files. The prebuilt MSVC
    CRT/STL static libraries (libcmt.lib, libcpmt.lib, libvcruntime.lib) that get linked into the
    binaries also need to be the Spectre-mitigated variants, otherwise BinSkim BA2024
    (EnableSpectreMitigations) still fails. Those variants ship in the "C++ Spectre-mitigated libs"
    Visual Studio component under %VCToolsInstallDir%\\lib\\spectre\\<arch>. When the build is not
    running in a Visual Studio Developer Command Prompt, locate the selected toolset with vswhere.
    """
    vctools_dir = _get_vctools_install_dir(args)
    if vctools_dir is None:
        return None
    if args.arm:
        arch = "arm"
    elif args.arm64:
        arch = "arm64"
    elif args.arm64ec:
        arch = "arm64ec"
    elif args.x86:
        arch = "x86"
    else:
        # Default to the target architecture selected by vcvarsall.bat (x86, x64, arm, arm64),
        # falling back to x64 which is what the official Windows release packages use.
        arch = os.environ.get("VSCMD_ARG_TGT_ARCH", "x64")
    spectre_dir = vctools_dir / "lib" / "spectre" / arch
    if spectre_dir.is_dir():
        return str(spectre_dir)
    # Some toolsets do not ship a dedicated arm64ec folder; those reuse the arm64 Spectre libraries.
    if args.arm64ec:
        fallback = vctools_dir / "lib" / "spectre" / "arm64"
        if fallback.is_dir():
            return str(fallback)
    return None


def stale_python_cache_args(config_build_dir: str) -> list[str]:
    """CMake arguments that drop the FindPython cache when the build tree was configured for another Python.

    FindPython re-resolves most results for a new Python_EXECUTABLE but keeps the cached header directory, so
    Development.Module and NumPy are then reported missing and the Python::NumPy target is not created.
    """
    cache = Path(config_build_dir) / "CMakeCache.txt"
    if not cache.is_file():
        return []
    prefix = "_Python_EXECUTABLE:INTERNAL="
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            cached = os.path.normcase(os.path.realpath(line[len(prefix) :]))
            if cached != os.path.normcase(os.path.realpath(sys.executable)):
                return ["-U", "_Python_*"]
            break
    return []


def generate_build_tree(
    cmake_path,
    source_dir,
    build_dir,
    cuda_home,
    cudnn_home,
    nccl_home,
    tensorrt_home,
    migraphx_home,
    acl_home,
    acl_libs,
    qnn_home,
    snpe_root,
    cann_home,
    path_to_protoc_exe,
    configs,
    cmake_extra_defines,
    args,
    cmake_extra_args,
):
    log.info("Generating CMake build tree")
    cmake_dir = os.path.join(source_dir, "cmake")
    cmake_args = [cmake_path, cmake_dir]
    if not use_dev_mode(args):
        cmake_args += ["--compile-no-warning-as-error"]

    types_to_disable = args.disable_types
    # enable/disable float 8 types
    disable_float8_types = args.android or ("float8" in types_to_disable)
    # enable/disable float 4 type
    disable_float4_types = args.android or ("float4" in types_to_disable)
    disable_optional_type = "optional" in types_to_disable
    disable_sparse_tensors = "sparsetensor" in types_to_disable
    disable_string_type = "string" in types_to_disable
    # VitisAI and OpenVINO providers currently only support the full protobuf option. Resolve this once: the
    # vcpkg triplets (which decide how the ONNX port is built) and the CMake configure must agree, otherwise
    # ONNX and ONNX Runtime end up with different protobuf runtimes in the same binary.
    use_full_protobuf = bool(
        args.use_full_protobuf or args.use_openvino or args.use_vitisai or args.gen_doc or args.enable_generic_interface
    )
    # Select 1DS by default, including on Windows; TraceLogging requires explicit opt-in.
    telemetry_backend = "WINDOWS" if args.use_windows_telemetry else "1DS"

    cmake_args.append("-Donnxruntime_USE_TELEMETRY=" + ("ON" if args.use_telemetry else "OFF"))
    cmake_args.append("-Donnxruntime_TELEMETRY_BACKEND=" + telemetry_backend)
    cmake_args.append("-Donnxruntime_USE_WINDOWS_TELEMETRY=" + ("ON" if args.use_windows_telemetry else "OFF"))
    if is_windows():
        cmake_args += [
            "-Donnxruntime_USE_DML=" + ("ON" if args.use_dml else "OFF"),
            "-Donnxruntime_USE_WINML=" + ("ON" if args.use_winml else "OFF"),
            "-Donnxruntime_ENABLE_PIX_FOR_WEBGPU_EP=" + ("ON" if args.enable_pix_capture else "OFF"),
        ]

        if args.caller_framework:
            cmake_args.append("-Donnxruntime_CALLER_FRAMEWORK=" + args.caller_framework)
        if args.winml_root_namespace_override:
            cmake_args.append("-Donnxruntime_WINML_NAMESPACE_OVERRIDE=" + args.winml_root_namespace_override)
        if args.disable_memleak_checker or args.enable_address_sanitizer:
            cmake_args.append("-Donnxruntime_ENABLE_MEMLEAK_CHECKER=OFF")
        else:
            cmake_args.append("-Donnxruntime_ENABLE_MEMLEAK_CHECKER=ON")

        if args.use_winml:
            cmake_args.append("-Donnxruntime_BUILD_WINML_TESTS=" + ("OFF" if args.skip_winml_tests else "ON"))
        if args.dml_path:
            cmake_args += [
                "-Donnxruntime_USE_CUSTOM_DIRECTML=ON",
                "-Ddml_INCLUDE_DIR=" + os.path.join(args.dml_path, "include"),
                "-Ddml_LIB_DIR=" + os.path.join(args.dml_path, "lib"),
            ]

        if args.dml_external_project:
            cmake_args += [
                "-Donnxruntime_USE_CUSTOM_DIRECTML=ON",
                "-Ddml_EXTERNAL_PROJECT=ON",
            ]

        if args.use_gdk:
            cmake_args += [
                "-DCMAKE_TOOLCHAIN_FILE=" + os.path.join(source_dir, "cmake", "gdk_toolchain.cmake"),
                "-DGDK_EDITION=" + args.gdk_edition,
                "-DGDK_PLATFORM=" + args.gdk_platform,
                "-Donnxruntime_BUILD_UNIT_TESTS=OFF",  # gtest doesn't build for GDK
            ]
            if args.use_dml and not (args.dml_path or args.dml_external_project):
                raise BuildError("You must set dml_path or dml_external_project when building with the GDK.")
    elif not is_macOS():
        cmake_args.append(
            "-Donnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS="
            + ("ON" if args.enable_external_custom_op_schemas else "OFF")
        )
    cmake_args += [
        "-Donnxruntime_RUN_ONNX_TESTS=" + ("ON" if args.enable_onnx_tests else "OFF"),
        "-Donnxruntime_GENERATE_TEST_REPORTS=ON",
        "-DPython_EXECUTABLE=" + sys.executable,
        "-Donnxruntime_USE_VCPKG=" + ("ON" if args.use_vcpkg else "OFF"),
        "-Donnxruntime_USE_MIMALLOC=" + ("ON" if args.use_mimalloc else "OFF"),
        "-Donnxruntime_ENABLE_PYTHON=" + ("ON" if args.enable_pybind else "OFF"),
        "-Donnxruntime_BUILD_CSHARP=" + ("ON" if args.build_csharp else "OFF"),
        "-Donnxruntime_BUILD_JAVA=" + ("ON" if args.build_java else "OFF"),
        "-Donnxruntime_BUILD_NODEJS=" + ("ON" if args.build_nodejs else "OFF"),
        "-Donnxruntime_BUILD_OBJC=" + ("ON" if args.build_objc else "OFF"),
        "-Donnxruntime_BUILD_SHARED_LIB=" + ("ON" if args.build_shared_lib else "OFF"),
        "-Donnxruntime_BUILD_APPLE_FRAMEWORK=" + ("ON" if args.build_apple_framework else "OFF"),
        "-Donnxruntime_USE_DNNL=" + ("ON" if args.use_dnnl else "OFF"),
        "-Donnxruntime_USE_NNAPI_BUILTIN=" + ("ON" if args.use_nnapi else "OFF"),
        "-Donnxruntime_USE_VSINPU=" + ("ON" if args.use_vsinpu else "OFF"),
        "-Donnxruntime_USE_RKNPU=" + ("ON" if args.use_rknpu else "OFF"),
        "-Donnxruntime_ENABLE_MICROSOFT_INTERNAL=" + ("ON" if args.enable_msinternal else "OFF"),
        "-Donnxruntime_USE_VITISAI=" + ("ON" if args.use_vitisai else "OFF"),
        "-Donnxruntime_USE_TENSORRT=" + ("ON" if args.use_tensorrt else "OFF"),
        "-Donnxruntime_USE_TENSORRT_BUILTIN_PARSER="
        + ("ON" if args.use_tensorrt_builtin_parser and not args.use_tensorrt_oss_parser else "OFF"),
        # interface variables are used only for building onnxruntime/onnxruntime_shared.dll but not EPs
        "-Donnxruntime_USE_TENSORRT_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        "-Donnxruntime_USE_CUDA_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        "-Donnxruntime_USE_OPENVINO_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        "-Donnxruntime_USE_VITISAI_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        "-Donnxruntime_USE_QNN_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        "-Donnxruntime_USE_MIGRAPHX_INTERFACE=" + ("ON" if args.enable_generic_interface else "OFF"),
        # set vars for migraphx
        "-Donnxruntime_USE_MIGRAPHX=" + ("ON" if args.use_migraphx else "OFF"),
        "-Donnxruntime_DISABLE_CONTRIB_OPS=" + ("ON" if args.disable_contrib_ops else "OFF"),
        "-Donnxruntime_DISABLE_ML_OPS=" + ("ON" if args.disable_ml_ops else "OFF"),
        "-Donnxruntime_DISABLE_GENERATION_OPS=" + ("ON" if args.disable_generation_ops else "OFF"),
        "-Donnxruntime_DISABLE_RTTI="
        + ("ON" if args.disable_rtti or (args.minimal_build is not None and not args.enable_pybind) else "OFF"),
        "-Donnxruntime_DISABLE_EXCEPTIONS=" + ("ON" if args.disable_exceptions else "OFF"),
        # Need to use 'is not None' with minimal_build check as it could be an empty list.
        "-Donnxruntime_MINIMAL_BUILD=" + ("ON" if args.minimal_build is not None else "OFF"),
        "-Donnxruntime_EXTENDED_MINIMAL_BUILD="
        + ("ON" if args.minimal_build and "extended" in args.minimal_build else "OFF"),
        "-Donnxruntime_MINIMAL_BUILD_CUSTOM_OPS="
        + (
            "ON"
            if (args.minimal_build is not None and ("custom_ops" in args.minimal_build or args.use_extensions))
            else "OFF"
        ),
        "-Donnxruntime_REDUCED_OPS_BUILD=" + ("ON" if is_reduced_ops_build(args) else "OFF"),
        "-Donnxruntime_CLIENT_PACKAGE_BUILD=" + ("ON" if args.client_package_build else "OFF"),
        "-Donnxruntime_ENABLE_SESSION_THREADPOOL_CALLBACKS="
        + ("ON" if args.enable_session_threadpool_callbacks else "OFF"),
        "-Donnxruntime_BUILD_MS_EXPERIMENTAL_OPS=" + ("ON" if args.ms_experimental else "OFF"),
        "-Donnxruntime_ENABLE_LTO=" + ("ON" if args.enable_lto else "OFF"),
        "-Donnxruntime_USE_ACL=" + ("ON" if args.use_acl else "OFF"),
        "-Donnxruntime_USE_JSEP=" + ("ON" if args.use_jsep else "OFF"),
        "-Donnxruntime_USE_WEBGPU=" + ("ON" if args.use_webgpu else "OFF"),
        # The WebGPU EP library kind selects these. Emit them unconditionally, including the OFF case, so that
        # reusing a build directory cannot leak a previous build's setting in via the CMake cache.
        "-Donnxruntime_USE_EP_API_ADAPTERS=" + ("ON" if args.use_webgpu in ("shared_lib", "static_plugin") else "OFF"),
        "-Donnxruntime_WEBGPU_STATIC_PLUGIN=" + ("ON" if args.use_webgpu == "static_plugin" else "OFF"),
        "-Donnxruntime_USE_EXTERNAL_DAWN=" + ("ON" if args.use_external_dawn else "OFF"),
        "-DDAWN_USE_AGILITY_SDK=" + ("ON" if args.use_dawn_agility_sdk else "OFF"),
        # Training related flags
        "-Donnxruntime_ENABLE_NVTX_PROFILE=" + ("ON" if args.enable_nvtx_profile else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING=" + ("ON" if args.enable_training else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING_OPS=" + ("ON" if args.enable_training_ops else "OFF"),
        "-Donnxruntime_ENABLE_TRAINING_APIS=" + ("ON" if args.enable_training_apis else "OFF"),
        # Enable advanced computations such as AVX for some traininig related ops.
        "-Donnxruntime_ENABLE_CPU_FP16_OPS=" + ("ON" if args.enable_training else "OFF"),
        "-Donnxruntime_USE_NCCL=" + ("ON" if args.enable_nccl else "OFF"),
        "-Donnxruntime_BUILD_BENCHMARKS=" + ("ON" if args.build_micro_benchmarks else "OFF"),
        "-Donnxruntime_GCOV_COVERAGE=" + ("ON" if args.code_coverage else "OFF"),
        "-Donnxruntime_ENABLE_MEMORY_PROFILE=" + ("ON" if args.enable_memory_profile else "OFF"),
        "-Donnxruntime_ENABLE_CUDA_LINE_NUMBER_INFO=" + ("ON" if args.enable_cuda_line_info else "OFF"),
        "-Donnxruntime_USE_CUDA_NHWC_OPS=" + ("ON" if args.use_cuda and not args.disable_cuda_nhwc_ops else "OFF"),
        "-Donnxruntime_BUILD_WEBASSEMBLY_STATIC_LIB=" + ("ON" if args.build_wasm_static_lib else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_CATCHING="
        + ("OFF" if args.disable_wasm_exception_catching else "ON"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_API_EXCEPTION_CATCHING="
        + ("ON" if args.enable_wasm_api_exception_catching else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_EXCEPTION_THROWING="
        + ("ON" if args.enable_wasm_exception_throwing_override else "OFF"),
        "-Donnxruntime_WEBASSEMBLY_RUN_TESTS_IN_BROWSER=" + ("ON" if args.wasm_run_tests_in_browser else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_JSPI=" + ("ON" if args.enable_wasm_jspi else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_THREADS=" + ("ON" if args.enable_wasm_threads else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_DEBUG_INFO=" + ("ON" if args.enable_wasm_debug_info else "OFF"),
        "-Donnxruntime_ENABLE_WEBASSEMBLY_PROFILING=" + ("ON" if args.enable_wasm_profiling else "OFF"),
        "-Donnxruntime_ENABLE_LAZY_TENSOR=" + ("ON" if args.enable_lazy_tensor else "OFF"),
        "-Donnxruntime_ENABLE_CUDA_PROFILING=" + ("ON" if args.enable_cuda_profiling else "OFF"),
        "-Donnxruntime_USE_XNNPACK=" + ("ON" if args.use_xnnpack else "OFF"),
        "-Donnxruntime_USE_WEBNN=" + ("ON" if args.use_webnn else "OFF"),
        "-Donnxruntime_USE_CANN=" + ("ON" if args.use_cann else "OFF"),
        "-Donnxruntime_DISABLE_FLOAT8_TYPES=" + ("ON" if disable_float8_types else "OFF"),
        "-Donnxruntime_DISABLE_FLOAT4_TYPES=" + ("ON" if disable_float4_types else "OFF"),
        "-Donnxruntime_DISABLE_SPARSE_TENSORS=" + ("ON" if disable_sparse_tensors else "OFF"),
        "-Donnxruntime_DISABLE_OPTIONAL_TYPE=" + ("ON" if disable_optional_type else "OFF"),
        "-Donnxruntime_DISABLE_STRING_TYPE=" + ("ON" if disable_string_type else "OFF"),
        "-Donnxruntime_CUDA_MINIMAL=" + ("ON" if args.enable_cuda_minimal_build else "OFF"),
    ]
    if args.minimal_build is not None:
        add_default_definition(cmake_extra_defines, "ONNX_MINIMAL_BUILD", "ON")
    if args.rv64:
        add_default_definition(cmake_extra_defines, "onnxruntime_CROSS_COMPILING", "ON")
        if not args.riscv_toolchain_root:
            raise BuildError("The --riscv_toolchain_root option is required to build for riscv64.")
        if not args.skip_tests and not args.riscv_qemu_path:
            raise BuildError("The --riscv_qemu_path option is required for testing riscv64.")

        cmake_args += [
            "-DRISCV_TOOLCHAIN_ROOT:PATH=" + args.riscv_toolchain_root,
            "-DRISCV_QEMU_PATH:PATH=" + args.riscv_qemu_path,
            "-DCMAKE_TOOLCHAIN_FILE=" + os.path.join(source_dir, "cmake", "riscv64.toolchain.cmake"),
        ]
    emscripten_cmake_toolchain_file = None
    emsdk_dir = None
    if args.build_wasm:
        emsdk_dir = os.path.join(cmake_dir, "external", "emsdk")
        emscripten_cmake_toolchain_file = os.path.join(
            emsdk_dir, "upstream", "emscripten", "cmake", "Modules", "Platform", "Emscripten.cmake"
        )

    if args.use_vcpkg:
        # Setup CMake flags for vcpkg

        # Find VCPKG's toolchain cmake file
        vcpkg_cmd_path = shutil.which("vcpkg")
        vcpkg_toolchain_path = None
        if vcpkg_cmd_path is not None:
            vcpkg_toolchain_path = Path(vcpkg_cmd_path).parent / "scripts" / "buildsystems" / "vcpkg.cmake"
            if not vcpkg_toolchain_path.exists():
                if is_windows():
                    raise BuildError(
                        "Cannot find VCPKG's toolchain cmake file. Please check if your vcpkg command was provided by Visual Studio"
                    )
                # Fallback to the next
                vcpkg_toolchain_path = None
        # Fallback to use the "VCPKG_INSTALLATION_ROOT" env var
        vcpkg_installation_root = os.environ.get("VCPKG_INSTALLATION_ROOT")
        if vcpkg_installation_root is None:
            # Fallback to checkout vcpkg from github
            vcpkg_installation_root = os.path.join(os.path.abspath(build_dir), "vcpkg")
            if not os.path.exists(vcpkg_installation_root):
                run_subprocess(
                    [
                        "git",
                        "clone",
                        "-b",
                        get_vcpkg_release_tag(),
                        "https://github.com/microsoft/vcpkg.git",
                        "--recursive",
                    ],
                    cwd=build_dir,
                )
        vcpkg_toolchain_path = Path(vcpkg_installation_root) / "scripts" / "buildsystems" / "vcpkg.cmake"

        if args.build_wasm:
            # While vcpkg has a toolchain cmake file for most build platforms(win/linux/mac/android/...), it doesn't have one to wasm. Therefore we need to do some tricks here.
            # Here we will generate two toolchain cmake files. One for building the ports, one for building onnxruntime itself.
            # The following one is for building onnxruntime itself
            new_toolchain_file = Path(build_dir) / "toolchain.cmake"
            old_toolchain_lines = []
            # First, we read the official toolchain cmake file provided by emscripten into memory
            emscripten_root_path = os.path.join(emsdk_dir, "upstream", "emscripten")
            with open(emscripten_cmake_toolchain_file, encoding="utf-8") as f:
                old_toolchain_lines = f.readlines()
            emscripten_root_path_cmake_path = emscripten_root_path.replace("\\", "/")
            # This file won't be used by vcpkg-tools when invoking 0.vcpkg_dep_info.cmake or vcpkg/scripts/ports.cmake
            with open(new_toolchain_file, "w", encoding="utf-8") as f:
                f.write(f'set(EMSCRIPTEN_ROOT_PATH "{emscripten_root_path_cmake_path}")\n')

                # Copy emscripten's toolchain cmake file to ours.
                f.writelines(old_toolchain_lines)
                vcpkg_toolchain_path_cmake_path = str(vcpkg_toolchain_path).replace("\\", "/")
                # Add an extra line at the bottom of the file to include vcpkg's toolchain file.
                f.write(f"include({vcpkg_toolchain_path_cmake_path})")
                # Then tell cmake to use this toolchain file.
                vcpkg_toolchain_path = new_toolchain_file.absolute()

            # This file is for building the vcpkg ports.
            empty_toolchain_file = Path(build_dir) / "emsdk_vcpkg_toolchain.cmake"
            with open(empty_toolchain_file, "w", encoding="utf-8") as f:
                f.write(f'set(EMSCRIPTEN_ROOT_PATH "{emscripten_root_path_cmake_path}")\n')
                # The variable VCPKG_MANIFEST_INSTALL is OFF while building the ports
                f.write("if(NOT VCPKG_MANIFEST_INSTALL)\n")
                flags_to_pass = [
                    "CXX_FLAGS",
                    "CXX_FLAGS_DEBUG",
                    "CXX_FLAGS_RELEASE",
                    "C_FLAGS",
                    "C_FLAGS_DEBUG",
                    "C_FLAGS_RELEASE",
                    "LINKER_FLAGS",
                    "LINKER_FLAGS_DEBUG",
                    "LINKER_FLAGS_RELEASE",
                ]
                # Overriding the cmake flags
                f.writelines("SET(CMAKE_" + flag + ' "${VCPKG_' + flag + '}")\n' for flag in flags_to_pass)
                # Copy emscripten's toolchain cmake file to ours.
                f.writelines(old_toolchain_lines)
                f.write("endif()")
            # We must define the VCPKG_CHAINLOAD_TOOLCHAIN_FILE cmake variable, otherwise vcpkg won't let us go.
            add_default_definition(
                cmake_extra_defines, "VCPKG_CHAINLOAD_TOOLCHAIN_FILE", str(empty_toolchain_file.absolute())
            )
            generate_vcpkg_triplets_for_emscripten(
                build_dir,
                configs,
                emscripten_root_path,
                args.enable_wasm_jspi,
                not args.disable_rtti,
                not args.disable_wasm_exception_catching,
                args.minimal_build is not None,
                args.enable_address_sanitizer,
                use_full_protobuf,
            )
        elif args.android:
            generate_android_triplets(
                build_dir,
                configs,
                args.android_cpp_shared,
                args.android_api,
                use_full_protobuf,
            )
        elif is_windows():
            generate_windows_triplets(build_dir, configs, args.msvc_toolset, use_full_protobuf)
        elif is_macOS():
            osx_target = args.apple_deploy_target
            if args.apple_deploy_target is None:
                osx_target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
            if osx_target is not None:
                log.info(f"Setting VCPKG_OSX_DEPLOYMENT_TARGET to {osx_target}")
            generate_macos_triplets(build_dir, configs, osx_target, use_full_protobuf, args.use_telemetry)
        else:
            # Linux, *BSD, AIX or other platforms
            generate_linux_triplets(build_dir, configs, use_full_protobuf, args.use_telemetry)
        add_default_definition(cmake_extra_defines, "CMAKE_TOOLCHAIN_FILE", str(vcpkg_toolchain_path))

        # Choose the cmake triplet
        triplet = None
        if args.build_wasm:
            triplet = "wasm32-emscripten"
        elif args.android:
            if args.android_abi == "armeabi-v7a":
                triplet = "arm-neon-android"
            elif args.android_abi == "arm64-v8a":
                triplet = "arm64-android"
            elif args.android_abi == "x86_64":
                triplet = "x64-android"
            elif args.android_abi == "x86":
                triplet = "x86-android"
            else:
                raise BuildError("Unknown android_abi")
        elif is_windows():
            target_arch = platform.machine()
            if args.arm64:
                target_arch = "ARM64"
            elif args.arm64ec:
                target_arch = "ARM64EC"
            cpu_arch = platform.architecture()[0]
            if target_arch == "AMD64":
                if cpu_arch == "32bit" or args.x86:
                    triplet = "x86-windows-static" if args.enable_msvc_static_runtime else "x86-windows-static-md"
                else:
                    triplet = "x64-windows-static" if args.enable_msvc_static_runtime else "x64-windows-static-md"
            elif target_arch == "ARM64":
                triplet = "arm64-windows-static" if args.enable_msvc_static_runtime else "arm64-windows-static-md"
            elif target_arch == "ARM64EC":
                triplet = "arm64ec-windows-static" if args.enable_msvc_static_runtime else "arm64ec-windows-static-md"
            else:
                raise BuildError("unknown python arch")
        elif is_macOS():
            for kvp in cmake_extra_defines:
                parts = kvp.split("=")
                if len(parts) != 2:
                    continue
                key = parts[0]
                value = parts[1]
                if key == "CMAKE_OSX_ARCHITECTURES" and len(value.split(";")) == 2:
                    triplet = "universal2-osx"
        if triplet:
            log.info(f"setting target triplet to {triplet}")
            add_default_definition(cmake_extra_defines, "VCPKG_TARGET_TRIPLET", triplet)

    # By default on Windows we currently support only cross compiling for ARM/ARM64
    if is_windows() and (args.arm64 or args.arm64ec or args.arm) and platform.architecture()[0] != "AMD64":
        # The onnxruntime_CROSS_COMPILING flag is deprecated. Prefer to use CMAKE_CROSSCOMPILING.
        add_default_definition(cmake_extra_defines, "onnxruntime_CROSS_COMPILING", "ON")
        if args.use_extensions:
            add_default_definition(cmake_extra_defines, "OPENCV_SKIP_SYSTEM_PROCESSOR_DETECTION", "ON")
    if args.use_cache:
        cmake_args.append("-Donnxruntime_BUILD_CACHE=ON")
        if not (is_windows() and args.cmake_generator != "Ninja"):
            cmake_args.append("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
            cmake_args.append("-DCMAKE_C_COMPILER_LAUNCHER=ccache")
            if args.use_cuda:
                cmake_args.append("-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache")
    if args.external_graph_transformer_path:
        cmake_args.append("-Donnxruntime_EXTERNAL_TRANSFORMER_SRC_PATH=" + args.external_graph_transformer_path)

    if args.use_dnnl:
        cmake_args.append("-Donnxruntime_DNNL_GPU_RUNTIME=" + args.dnnl_gpu_runtime)
        cmake_args.append("-Donnxruntime_DNNL_OPENCL_ROOT=" + args.dnnl_opencl_root)
        cmake_args.append("-Donnxruntime_DNNL_AARCH64_RUNTIME=" + args.dnnl_aarch64_runtime)
        cmake_args.append("-Donnxruntime_DNNL_ACL_ROOT=" + args.dnnl_acl_root)
    if args.build_wasm:
        cmake_args.append("-Donnxruntime_ENABLE_WEBASSEMBLY_SIMD=" + ("ON" if args.enable_wasm_simd else "OFF"))
        if args.enable_wasm_relaxed_simd:
            if not args.enable_wasm_simd:
                raise BuildError(
                    "Wasm Relaxed SIMD (--enable_wasm_relaxed_simd) is only available with Wasm SIMD (--enable_wasm_simd)."
                )
            cmake_args += ["-Donnxruntime_ENABLE_WEBASSEMBLY_RELAXED_SIMD=ON"]
    if args.use_migraphx:
        cmake_args.append("-Donnxruntime_MIGRAPHX_HOME=" + migraphx_home)

    if args.use_tensorrt:
        cmake_args.append("-Donnxruntime_TENSORRT_HOME=" + tensorrt_home)

    if args.use_cuda:
        nvcc_threads = number_of_nvcc_threads(args)
        cmake_args.append("-Donnxruntime_NVCC_THREADS=" + str(nvcc_threads))

        flash_nvcc_threads = number_of_flash_nvcc_threads(args)
        cmake_args.append("-Donnxruntime_FLASH_NVCC_THREADS=" + str(flash_nvcc_threads))

        cmake_args.append(f"-DCMAKE_CUDA_COMPILER={cuda_home}/bin/nvcc")
        add_default_definition(cmake_extra_defines, "onnxruntime_USE_CUDA", "ON")
        if args.cuda_version:
            add_default_definition(cmake_extra_defines, "onnxruntime_CUDA_VERSION", args.cuda_version)
        # TODO: this variable is not really needed
        add_default_definition(cmake_extra_defines, "onnxruntime_CUDA_HOME", cuda_home)
        if cudnn_home:
            add_default_definition(cmake_extra_defines, "onnxruntime_CUDNN_HOME", cudnn_home)

    if is_windows():
        if args.enable_msvc_static_runtime:
            add_default_definition(
                cmake_extra_defines, "CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            )
        # Set flags for 3rd-party libs
        if not args.use_vcpkg:
            if args.enable_msvc_static_runtime:
                add_default_definition(cmake_extra_defines, "ONNX_USE_MSVC_STATIC_RUNTIME", "ON")
                add_default_definition(cmake_extra_defines, "protobuf_MSVC_STATIC_RUNTIME", "ON")
                # The following build option was added in ABSL 20240722.0 and it must be explicitly set
                add_default_definition(cmake_extra_defines, "ABSL_MSVC_STATIC_RUNTIME", "ON")
                add_default_definition(cmake_extra_defines, "gtest_force_shared_crt", "OFF")
            else:
                # CMAKE_MSVC_RUNTIME_LIBRARY is default to MultiThreaded$<$<CONFIG:Debug>:Debug>DLL
                add_default_definition(cmake_extra_defines, "ONNX_USE_MSVC_STATIC_RUNTIME", "OFF")
                add_default_definition(cmake_extra_defines, "protobuf_MSVC_STATIC_RUNTIME", "OFF")
                add_default_definition(cmake_extra_defines, "ABSL_MSVC_STATIC_RUNTIME", "OFF")
                add_default_definition(cmake_extra_defines, "gtest_force_shared_crt", "ON")

    if acl_home and os.path.exists(acl_home):
        cmake_args += ["-Donnxruntime_ACL_HOME=" + acl_home]

    if acl_libs and os.path.exists(acl_libs):
        cmake_args += ["-Donnxruntime_ACL_LIBS=" + acl_libs]

    if nccl_home and os.path.exists(nccl_home):
        cmake_args += ["-Donnxruntime_NCCL_HOME=" + nccl_home]

    if qnn_home and os.path.exists(qnn_home):
        cmake_args += ["-Donnxruntime_QNN_HOME=" + qnn_home]

    if snpe_root and os.path.exists(snpe_root):
        cmake_args += ["-DSNPE_ROOT=" + snpe_root]

    if cann_home and os.path.exists(cann_home):
        cmake_args += ["-Donnxruntime_CANN_HOME=" + cann_home]

    if args.use_openvino:
        cmake_args += [
            "-Donnxruntime_USE_OPENVINO=ON",
            "-Donnxruntime_NPU_NO_FALLBACK=" + ("ON" if args.use_openvino == "NPU_NO_CPU_FALLBACK" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_GPU=" + ("ON" if args.use_openvino == "GPU" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_CPU=" + ("ON" if args.use_openvino == "CPU" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_NPU=" + ("ON" if args.use_openvino == "NPU" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_GPU_NP=" + ("ON" if args.use_openvino == "GPU_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_CPU_NP=" + ("ON" if args.use_openvino == "CPU_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_NPU_NP=" + ("ON" if args.use_openvino == "NPU_NO_PARTITION" else "OFF"),
            "-Donnxruntime_USE_OPENVINO_HETERO=" + ("ON" if args.use_openvino.startswith("HETERO") else "OFF"),
            "-Donnxruntime_USE_OPENVINO_DEVICE=" + (args.use_openvino),
            "-Donnxruntime_USE_OPENVINO_MULTI=" + ("ON" if args.use_openvino.startswith("MULTI") else "OFF"),
            "-Donnxruntime_USE_OPENVINO_AUTO=" + ("ON" if args.use_openvino.startswith("AUTO") else "OFF"),
        ]

    if use_full_protobuf:
        cmake_args += ["-Donnxruntime_USE_FULL_PROTOBUF=ON", "-DProtobuf_USE_STATIC_LIBS=ON"]

    if args.use_cuda and not is_windows():
        nvml_stub_path = cuda_home + "/lib64/stubs"
        cmake_args += ["-DCUDA_CUDA_LIBRARY=" + nvml_stub_path]

    if args.nnapi_min_api:
        cmake_args += ["-Donnxruntime_NNAPI_MIN_API=" + str(args.nnapi_min_api)]

    if args.android:
        if not args.android_ndk_path:
            raise BuildError("android_ndk_path required to build for Android")
        if not args.android_sdk_path:
            raise BuildError("android_sdk_path required to build for Android")
        android_toolchain_cmake_path = os.path.join(args.android_ndk_path, "build", "cmake", "android.toolchain.cmake")
        cmake_args += [
            "-DANDROID_PLATFORM=android-" + str(args.android_api),
            "-DANDROID_ABI=" + str(args.android_abi),
            "-DANDROID_MIN_SDK=" + str(args.android_api),
            "-DANDROID_USE_LEGACY_TOOLCHAIN_FILE=false",
        ]
        if args.disable_rtti:
            add_default_definition(cmake_extra_defines, "CMAKE_ANDROID_RTTI", "OFF")
        if args.disable_exceptions:
            add_default_definition(cmake_extra_defines, "CMAKE_ANDROID_EXCEPTIONS", "OFF")
        if not args.use_vcpkg:
            cmake_args.append("-DCMAKE_TOOLCHAIN_FILE=" + android_toolchain_cmake_path)
        else:
            cmake_args.append("-DVCPKG_CHAINLOAD_TOOLCHAIN_FILE=" + android_toolchain_cmake_path)

        if args.android_cpp_shared:
            cmake_args += ["-DANDROID_STL=c++_shared"]

    if is_macOS() and not args.android:
        add_default_definition(cmake_extra_defines, "CMAKE_OSX_ARCHITECTURES", args.osx_arch)
        # Code sign the binaries, if the code signing development identity and/or team id are provided
        if args.xcode_code_signing_identity:
            cmake_args += ["-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGN_IDENTITY=" + args.xcode_code_signing_identity]
        if args.xcode_code_signing_team_id:
            cmake_args += ["-DCMAKE_XCODE_ATTRIBUTE_DEVELOPMENT_TEAM=" + args.xcode_code_signing_team_id]

    if args.use_qnn:
        if args.qnn_home is None or os.path.exists(args.qnn_home) is False:
            raise BuildError("qnn_home=" + qnn_home + " not valid." + " qnn_home paths must be specified and valid.")
        cmake_args += ["-Donnxruntime_USE_QNN=ON"]

        if args.use_qnn == "static_lib":
            cmake_args += ["-Donnxruntime_BUILD_QNN_EP_STATIC_LIB=ON"]
        if args.android and args.use_qnn != "static_lib":
            raise BuildError("Only support Android + QNN builds with QNN EP built as a static library.")
        if args.use_qnn == "static_lib" and args.enable_generic_interface:
            raise BuildError("Generic ORT interface only supported with QNN EP built as a shared library.")

    if args.use_coreml:
        cmake_args += ["-Donnxruntime_USE_COREML=ON"]

    if args.use_webnn:
        if not args.build_wasm:
            raise BuildError("WebNN is only available for WebAssembly build.")
        cmake_args += ["-Donnxruntime_USE_WEBNN=ON"]

    # TODO: currently we allows building with both --use_jsep and --use_webgpu in this working branch.
    #       This situation is temporary. Eventually, those two flags will be mutually exclusive.
    #
    # if args.use_jsep and args.use_webgpu:
    #     raise BuildError("JSEP (--use_jsep) and WebGPU (--use_webgpu) cannot be enabled at the same time.")

    if args.enable_wasm_jspi:
        if args.use_jsep:
            raise BuildError("JSEP (--use_jsep) and WASM JSPI (--enable_wasm_jspi) cannot be enabled at the same time.")
        if args.disable_wasm_exception_catching:
            raise BuildError("Cannot set WebAssembly exception catching in JSPI build.")

    if not args.use_webgpu:
        if args.use_external_dawn:
            raise BuildError("External Dawn (--use_external_dawn) must be enabled with WebGPU (--use_webgpu).")
        if args.use_dawn_agility_sdk:
            raise BuildError("Dawn Agility SDK (--use_dawn_agility_sdk) must be enabled with WebGPU (--use_webgpu).")
        if is_windows():
            if args.enable_pix_capture:
                raise BuildError(
                    "Enable PIX Capture (--enable_pix_capture) must be enabled with WebGPU (--use_webgpu) on Windows"
                )
    elif args.use_webgpu == "shared_lib":
        # Shared library build (plugin EP)
        if args.build_wasm:
            raise BuildError("Only static library build of WebGPU EP is supported for WebAssembly build.")

    if args.use_dawn_agility_sdk:
        if not is_windows():
            raise BuildError("Dawn Agility SDK (--use_dawn_agility_sdk) is only supported on Windows.")

        if args.arm or args.arm64ec:
            raise BuildError(
                "Dawn Agility SDK (--use_dawn_agility_sdk) does not support Windows ARM32 or ARM64EC. "
                "Use an x86, x64, or ARM64 target."
            )

        if args.build_csharp or args.build_nuget or args.build_java or args.build_nodejs:
            raise BuildError(
                "--use_dawn_agility_sdk is not supported with --build_csharp, --build_nuget, "
                "--build_java, or --build_nodejs because those build paths do not deploy "
                "the required D3D12 runtime DLLs. "
                "For the standalone WebGPU EP NuGet package, use plugin-ep-webgpu/csharp/pack_nuget.py."
            )

    if args.use_snpe:
        cmake_args += ["-Donnxruntime_USE_SNPE=ON"]

    if not args.no_kleidiai:
        cmake_args += ["-Donnxruntime_USE_KLEIDIAI=ON"]
        if args.use_qmx:
            cmake_args += ["-Donnxruntime_USE_QMX_KLEIDIAI_COEXIST=ON"]

    if args.enable_arm_neon_nchwc:
        cmake_args += ["-Donnxruntime_USE_ARM_NEON_NCHWC=ON"]

    if args.enable_rvv:
        cmake_args += ["-Donnxruntime_USE_RVV=ON"]

    if not args.no_sve:
        cmake_args += ["-Donnxruntime_USE_SVE=ON"]

    if is_macOS() and (args.macos or args.ios or args.visionos or args.tvos):
        # Note: Xcode CMake generator doesn't have a good support for Mac Catalyst yet.
        if args.macos == "Catalyst" and args.cmake_generator == "Xcode":
            raise BuildError("Xcode CMake generator ('--cmake_generator Xcode') doesn't support Mac Catalyst build.")

        if (args.ios or args.visionos or args.tvos or args.macos == "MacOSX") and not args.cmake_generator == "Xcode":
            raise BuildError(
                "iOS/MacOS framework build requires use of the Xcode CMake generator ('--cmake_generator Xcode')."
            )

        needed_args = [
            args.apple_sysroot,
            args.apple_deploy_target,
        ]
        arg_names = [
            "--apple_sysroot          " + "<the location or name of the macOS platform SDK>",
            "--apple_deploy_target  " + "<the minimum version of the target platform>",
        ]
        if not all(needed_args):
            raise BuildError(
                "iOS/MacOS framework build on MacOS canceled due to missing arguments: "
                + ", ".join(val for val, cond in zip(arg_names, needed_args, strict=False) if not cond)
            )
        # note: this value is mainly used in framework_info.json file to specify the build osx type
        platform_name = "macabi" if args.macos == "Catalyst" else args.apple_sysroot
        cmake_args += [
            "-Donnxruntime_BUILD_SHARED_LIB=ON",
            "-DCMAKE_OSX_SYSROOT=" + args.apple_sysroot,
            # we do not need protoc binary for ios cross build
            "-Dprotobuf_BUILD_PROTOC_BINARIES=OFF",
            "-DPLATFORM_NAME=" + platform_name,
        ]
        if args.ios:
            cmake_args += [
                "-DCMAKE_SYSTEM_NAME=iOS",
                "-DCMAKE_TOOLCHAIN_FILE="
                + (args.ios_toolchain_file if args.ios_toolchain_file else "../cmake/onnxruntime_ios.toolchain.cmake"),
            ]
        # for catalyst build, we need to manually specify cflags for target e.g. x86_64-apple-ios14.0-macabi, etc.
        # https://forums.developer.apple.com/forums/thread/122571
        if args.macos == "Catalyst":
            macabi_target = f"{args.osx_arch}-apple-ios{args.apple_deploy_target}-macabi"
            cmake_args += [
                f"-DCMAKE_CXX_FLAGS=--target={macabi_target}",
                f"-DCMAKE_C_FLAGS=--target={macabi_target}",
                f"-DCMAKE_ASM_FLAGS=--target={macabi_target}",
            ]
        else:
            cmake_args += [
                "-DCMAKE_OSX_DEPLOYMENT_TARGET=" + args.apple_deploy_target,
            ]

        if args.visionos:
            cmake_args += [
                "-DCMAKE_SYSTEM_NAME=visionOS",
                "-DCMAKE_TOOLCHAIN_FILE="
                + (
                    args.visionos_toolchain_file
                    if args.visionos_toolchain_file
                    else "../cmake/onnxruntime_visionos.toolchain.cmake"
                ),
                "-Donnxruntime_ENABLE_CPUINFO=OFF",
            ]
        if args.tvos:
            cmake_args += [
                "-DCMAKE_SYSTEM_NAME=tvOS",
                "-DCMAKE_TOOLCHAIN_FILE="
                + (
                    args.tvos_toolchain_file
                    if args.tvos_toolchain_file
                    else "../cmake/onnxruntime_tvos.toolchain.cmake"
                ),
            ]

    if args.build_wasm:
        if not args.use_vcpkg:
            cmake_args.append("-DCMAKE_TOOLCHAIN_FILE=" + emscripten_cmake_toolchain_file)
        if args.disable_wasm_exception_catching:
            # WebAssembly unittest requires exception catching to work. If this feature is disabled, we do not build
            # unit test.
            cmake_args += [
                "-Donnxruntime_BUILD_UNIT_TESTS=OFF",
            ]

        # add default emscripten settings
        emscripten_settings = list(args.emscripten_settings)

        # set -s MALLOC
        if args.wasm_malloc is not None:
            add_default_definition(emscripten_settings, "MALLOC", args.wasm_malloc)
        add_default_definition(emscripten_settings, "MALLOC", "dlmalloc")

        # set -s STACK_SIZE=5242880
        add_default_definition(emscripten_settings, "STACK_SIZE", "5242880")

        if emscripten_settings:
            cmake_args += [f"-Donnxruntime_EMSCRIPTEN_SETTINGS={';'.join(emscripten_settings)}"]

    # Append onnxruntime-extensions cmake options
    if args.use_extensions:
        cmake_args += ["-Donnxruntime_USE_EXTENSIONS=ON"]

        # default path of onnxruntime-extensions, using git submodule
        for config in configs:
            onnxruntime_extensions_path = os.path.join(build_dir, config, "_deps", "extensions-src")
            onnxruntime_extensions_path = os.path.abspath(onnxruntime_extensions_path)

            if args.extensions_overridden_path and os.path.exists(args.extensions_overridden_path):
                # use absolute path here because onnxruntime-extensions is outside onnxruntime
                onnxruntime_extensions_path = os.path.abspath(args.extensions_overridden_path)
                cmake_args += ["-Donnxruntime_EXTENSIONS_OVERRIDDEN=ON"]
                print("[onnxruntime-extensions] Loading onnxruntime-extensions from: ", onnxruntime_extensions_path)
            else:
                print("[onnxruntime-extensions] Loading onnxruntime-extensions from: FetchContent")

            cmake_args += ["-Donnxruntime_EXTENSIONS_PATH=" + onnxruntime_extensions_path]

            if is_reduced_ops_build(args):
                operators_config_file = os.path.abspath(args.include_ops_by_config)
                cmake_tool_dir = os.path.join(onnxruntime_extensions_path, "tools")

                # generate _selectedoplist.cmake by operators config file
                run_subprocess([sys.executable, "gen_selectedops.py", operators_config_file], cwd=cmake_tool_dir)

    if path_to_protoc_exe:
        cmake_args += [f"-DONNX_CUSTOM_PROTOC_EXECUTABLE={path_to_protoc_exe}"]

    if args.cmake_deps_mirror_dir:
        cmake_args += [f"-Donnxruntime_CMAKE_DEPS_MIRROR_DIR={args.cmake_deps_mirror_dir}"]

    if args.fuzz_testing:
        if not (
            args.build_shared_lib
            and is_windows()
            and args.cmake_generator in ("Visual Studio 17 2022", "Visual Studio 18 2026")
            and args.use_full_protobuf
        ):
            raise BuildError("Fuzz test has only be tested with build shared libs option using MSVC on windows")
        cmake_args += [
            "-Donnxruntime_BUILD_UNIT_TESTS=ON",
            "-Donnxruntime_FUZZ_TEST=ON",
            "-Donnxruntime_USE_FULL_PROTOBUF=ON",
        ]

    if args.enable_lazy_tensor:
        import torch  # noqa: PLC0415

        cmake_args += [f"-Donnxruntime_PREBUILT_PYTORCH_PATH={os.path.dirname(torch.__file__)}"]
        cmake_args += ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]

    if args.enable_dx_interop:
        cmake_args += ["-Donnxruntime_USE_DX_INTEROP=ON"]

    if args.use_azure:
        add_default_definition(cmake_extra_defines, "onnxruntime_USE_AZURE", "ON")

    if args.use_lock_free_queue:
        add_default_definition(cmake_extra_defines, "onnxruntime_USE_LOCK_FREE_QUEUE", "ON")

    if is_windows():
        if not args.android and not args.build_wasm:
            if args.use_cache:
                add_default_definition(
                    cmake_extra_defines,
                    "CMAKE_MSVC_DEBUG_INFORMATION_FORMAT",
                    "$<$<CONFIG:Debug,RelWithDebInfo>:Embedded>",
                )
            else:
                # Always enable debug info even in release build. The debug information is in separated *.pdb files that
                # can be easily discarded when debug symbols are not needed. We enable it by default because many auditting
                # tools need to use the symbols.
                add_default_definition(cmake_extra_defines, "CMAKE_MSVC_DEBUG_INFORMATION_FORMAT", "ProgramDatabase")

        if number_of_parallel_jobs(args) > 0:
            # https://devblogs.microsoft.com/cppblog/improved-parallelism-in-msbuild/
            # NOTE: this disables /MP if set (according to comments on blog post).
            # By default, MultiProcMaxCount and CL_MPCount value are equal to the number of CPU logical processors.
            # See logic around setting CL_MPCount below
            cmake_args += ["-DCMAKE_VS_GLOBALS=UseMultiToolTask=true;EnforceProcessCountAcrossBuilds=true"]

    cmake_args += [f"-D{define}" for define in cmake_extra_defines]

    cmake_args += cmake_extra_args

    # ADO pipelines will store the pipeline build number
    # (e.g. 20260615.4) and source version in environment
    # variables. If present, use these values to define the
    # WinML/ORT DLL versions.
    build_number = os.getenv("Build_BuildNumber")  # noqa: SIM112
    source_version = os.getenv("Build_SourceVersion")  # noqa: SIM112
    if build_number and source_version:
        build_matches = re.fullmatch(r"(\d\d)(\d\d)(\d\d)(\d\d)\.(\d+)", build_number)
        if build_matches:
            MM = build_matches.group(3)  # noqa: N806
            DD = build_matches.group(4)  # noqa: N806

            # Get ORT major, minor, and patch number
            with open(os.path.join(source_dir, "VERSION_NUMBER")) as f:
                first_line = f.readline()
                ort_version_matches = re.match(r"(\d+)\.(\d+)\.(\d+)", first_line)
                if not ort_version_matches:
                    raise BuildError("Couldn't read version from VERSION_NUMBER")
                ort_major = ort_version_matches.group(1)
                ort_minor = ort_version_matches.group(2)
                ort_patch = ort_version_matches.group(3)
                # Example (VERSION_NUMBER: 1.27.0, BuildNumber: 20260615.4,
                # SourceVersion: 8f0278c77bf44b0cc83c098c6c722b92a36ac4b5)
                # MajorPart = 1
                # MinorPart = 27
                # BuildPart = 0
                # PrivatePart = 615
                # String = 1.27.0.20260615.4.8f0278c
                cmake_args += [
                    f"-DVERSION_MAJOR_PART={ort_major}",
                    f"-DVERSION_MINOR_PART={ort_minor}",
                    f"-DVERSION_BUILD_PART={ort_patch}",
                    f"-DVERSION_PRIVATE_PART={MM}{DD}",
                    f"-DVERSION_STRING={ort_major}.{ort_minor}.{ort_patch}.{build_number}.{source_version[0:7]}",
                ]

    for config in configs:
        cflags = []
        cxxflags = None
        ldflags = None
        nvcc_flags = []
        if is_windows() and not args.android and not args.build_wasm:
            njobs = number_of_parallel_jobs(args)
            if args.use_cuda:
                nvcc_flags.append("-allow-unsupported-compiler")
            if njobs > 1:
                if args.parallel == 0:
                    cflags += ["/MP"]
                else:
                    cflags += [f"/MP{njobs}"]
        # Setup default values for cflags/cxxflags/ldflags.
        # The values set here are purely for security and compliance purposes. ONNX Runtime should work fine without these flags.
        if (args.use_binskim_compliant_compile_flags or args.enable_address_sanitizer) and not args.android:
            if is_windows() and not args.build_wasm:
                cflags += ["/guard:cf", "/DWIN32", "/D_WINDOWS"]
                if not args.use_gdk:
                    # Target Windows 10
                    cflags += [
                        "/DWINAPI_FAMILY=100",
                        "/DWINVER=0x0A00",
                        "/D_WIN32_WINNT=0x0A00",
                        "/DNTDDI_VERSION=0x0A000000",
                    ]
                # The "/profile" flag implies "/DEBUG:FULL /DEBUGTYPE:cv,fixup /OPT:REF /OPT:NOICF /INCREMENTAL:NO /FIXED:NO". We set it for satisfying a Microsoft internal compliance requirement. External users
                # do not need to have it.
                ldflags = ["/profile", "/DYNAMICBASE"]
                # Address Sanitizer libs do not have a Qspectre version. So they two cannot be both enabled.
                if not args.enable_address_sanitizer:
                    cflags += ["/Qspectre"]
                    # /Qspectre only mitigates ONNX Runtime's own object files. The prebuilt MSVC
                    # CRT/STL static libraries (libcmt.lib, libcpmt.lib, libvcruntime.lib) that are
                    # linked into the binaries also have to be the Spectre-mitigated variants,
                    # otherwise BinSkim BA2024 (EnableSpectreMitigations) still fails. Prepend the
                    # Spectre lib directory to the linker search path so those libraries are
                    # resolved ahead of the default (non-mitigated) CRT libraries.
                    spectre_lib_dir = get_msvc_spectre_lib_dir(args)
                    if spectre_lib_dir is not None:
                        ldflags = [f'/LIBPATH:"{spectre_lib_dir}"', *ldflags]
                    else:
                        log.warning(
                            "Could not locate the MSVC Spectre-mitigated CRT/STL libraries. The "
                            "resulting binaries may fail BinSkim BA2024 (EnableSpectreMitigations). "
                            "Install the 'C++ Spectre-mitigated libs' component from the Visual "
                            "Studio installer and build from a Developer Command Prompt."
                        )
                if config == "Release":
                    cflags += ["/O2", "/Ob2", "/DNDEBUG"]
                elif config == "RelWithDebInfo":
                    cflags += ["/O2", "/Ob1", "/DNDEBUG"]
                elif config == "Debug":
                    cflags += ["/Ob0", "/Od", "/RTC1"]
                elif config == "MinSizeRel":
                    cflags += ["/O1", "/Ob1", "/DNDEBUG"]
                if args.enable_address_sanitizer:
                    cflags += ["/fsanitize=address"]
                cxxflags = cflags.copy()
                if not args.disable_exceptions:
                    cxxflags.append("/EHsc")
                if args.use_cuda:
                    # nvcc flags is like --name=value or -name
                    # MSVC flags are like /name=value or /name
                    msvc_flags = ""
                    for compile_flag in cflags:
                        if compile_flag.startswith("-"):
                            nvcc_flags.append(compile_flag)
                        else:
                            if not compile_flag.startswith("/"):
                                log.warning(
                                    "Flag (%s) is not started with - or /. It will be passed to host (MSVC) compiler.",
                                    compile_flag,
                                )
                            msvc_flags = msvc_flags + " " + compile_flag
                    if len(msvc_flags) != 0:
                        nvcc_flags.append(f'-Xcompiler="{msvc_flags}"')
            elif is_linux() or is_macOS() or args.build_wasm:
                if is_linux() and not args.build_wasm:
                    ldflags = ["-Wl,-Bsymbolic-functions", "-Wl,-z,relro", "-Wl,-z,now", "-Wl,-z,noexecstack"]
                else:
                    ldflags = []
                if config == "Release":
                    cflags = [
                        "-DNDEBUG",
                        "-U_FORTIFY_SOURCE",
                        "-D_FORTIFY_SOURCE=2",
                        "-Wp,-D_GLIBCXX_ASSERTIONS",
                        "-fstack-protector-strong",
                        "-O3",
                        "-pipe",
                    ]
                    if is_linux() and not args.build_wasm:
                        ldflags += ["-Wl,--strip-all"]
                elif config == "RelWithDebInfo":
                    cflags = [
                        "-DNDEBUG",
                        "-U_FORTIFY_SOURCE",
                        "-D_FORTIFY_SOURCE=2",
                        "-Wp,-D_GLIBCXX_ASSERTIONS",
                        "-fstack-protector-strong",
                        "-O3",
                        "-pipe",
                        "-g",
                    ]
                elif config == "Debug":
                    cflags = ["-g", "-O0"]
                    if args.enable_address_sanitizer:
                        cflags += ["-fsanitize=address"]
                        ldflags += ["-fsanitize=address"]
                elif config == "MinSizeRel":
                    cflags = [
                        "-DNDEBUG",
                        "-U_FORTIFY_SOURCE",
                        "-D_FORTIFY_SOURCE=2",
                        "-Wp,-D_GLIBCXX_ASSERTIONS",
                        "-fstack-protector-strong",
                        "-Os",
                        "-pipe",
                        "-g",
                    ]
                if is_linux() and platform.machine() == "x86_64" and not args.build_wasm:
                    # The following flags needs GCC 8 and newer
                    cflags += ["-fstack-clash-protection"]
                    if not args.rv64:
                        cflags += ["-fcf-protection"]
                cxxflags = cflags.copy()
                if args.use_cuda:
                    nvcc_flags = cflags.copy()
        if cxxflags is None and cflags is not None and len(cflags) != 0:
            cxxflags = cflags.copy()
        config_build_dir = get_config_build_dir(build_dir, config)
        os.makedirs(config_build_dir, exist_ok=True)
        temp_cmake_args = cmake_args.copy() + stale_python_cache_args(config_build_dir)
        if cflags is not None and cxxflags is not None and len(cflags) != 0 and len(cxxflags) != 0:
            temp_cmake_args += [
                "-DCMAKE_C_FLAGS={}".format(" ".join(cflags)),
                "-DCMAKE_CXX_FLAGS={}".format(" ".join(cxxflags)),
            ]
        if nvcc_flags is not None and len(nvcc_flags) != 0:
            temp_cmake_args += ["-DCMAKE_CUDA_FLAGS_INIT={}".format(" ".join(nvcc_flags))]
        if ldflags is not None and len(ldflags) != 0:
            temp_cmake_args += [
                "-DCMAKE_EXE_LINKER_FLAGS_INIT={}".format(" ".join(ldflags)),
                "-DCMAKE_MODULE_LINKER_FLAGS_INIT={}".format(" ".join(ldflags)),
                "-DCMAKE_SHARED_LINKER_FLAGS_INIT={}".format(" ".join(ldflags)),
            ]
        env = {}
        if args.use_vcpkg:
            # append VCPKG_INSTALL_OPTIONS
            #
            # VCPKG_INSTALL_OPTIONS is a CMake list. It must be joined by semicolons
            # Therefore, if any of the option string contains a semicolon, it must be escaped
            temp_cmake_args += [
                "-DVCPKG_INSTALL_OPTIONS={}".format(
                    ";".join(generate_vcpkg_install_options(Path(build_dir) / config, args))
                )
            ]

            vcpkg_keep_env_vars = ["TRT_UPLOAD_AUTH_TOKEN"]

            if args.build_wasm:
                emsdk_vars = ["EMSDK", "EMSDK_NODE", "EMSDK_PYTHON"]

                # If environment variables 'EMSDK' is not set, run emsdk_env to set them
                if "EMSDK" not in os.environ:
                    if is_windows():
                        # Run `cmd /s /c call .\\emsdk_env && set` to run emsdk_env and dump the environment variables
                        emsdk_env = run_subprocess(
                            ["cmd", "/s", "/c", "call .\\emsdk_env && set"],
                            cwd=emsdk_dir,
                            capture_stdout=True,
                        )
                    else:
                        # Run `sh -c ". ./emsdk_env.sh && printenv"` to run emsdk_env and dump the environment variables
                        emsdk_env = run_subprocess(
                            ["sh", "-c", ". ./emsdk_env.sh && printenv"],
                            cwd=emsdk_dir,
                            capture_stdout=True,
                        )

                    # check for EMSDK environment variables and set them in the environment
                    for line in emsdk_env.stdout.decode().splitlines():
                        if "=" in line:
                            key, value = line.rstrip().split("=", 1)
                            if key in emsdk_vars:
                                os.environ[key] = value

                for var in emsdk_vars:
                    if var in os.environ:
                        env[var] = os.environ[var]
                    elif var == "EMSDK":
                        # EMSDK must be set, but EMSDK_NODE and EMSDK_PYTHON are optional
                        raise BuildError(
                            "EMSDK environment variable is not set correctly. Please run `emsdk_env` to set them."
                        )

                vcpkg_keep_env_vars += emsdk_vars

            #
            # Workaround for vcpkg failed to find the correct path of Python
            #
            # Since vcpkg does not inherit the environment variables `PATH` from the parent process, CMake will fail to
            # find the Python executable if the Python executable is not in the default location. This usually happens
            # to the Python installed by Anaconda.
            #
            # To minimize the impact of this problem, we set the `Python3_ROOT_DIR` environment variable to the
            # directory of current Python executable.
            #
            # see https://cmake.org/cmake/help/latest/module/FindPython3.html
            #
            env["Python3_ROOT_DIR"] = str(Path(os.path.dirname(sys.executable)).resolve())
            vcpkg_keep_env_vars += ["Python3_ROOT_DIR"]

            env["VCPKG_KEEP_ENV_VARS"] = ";".join(vcpkg_keep_env_vars)

        run_subprocess(
            [*temp_cmake_args, f"-DCMAKE_BUILD_TYPE={config}"],
            cwd=config_build_dir,
            cuda_home=cuda_home,
            env=env,
        )


def clean_targets(cmake_path, build_dir, configs):
    for config in configs:
        log.info("Cleaning targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path, "--build", build_dir2, "--config", config, "--target", "clean"]

        run_subprocess(cmd_args)


def build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs, targets: list[str] | None):
    for config in configs:
        log.info("Building targets for %s configuration", config)
        build_dir2 = get_config_build_dir(build_dir, config)
        cmd_args = [cmake_path, "--build", build_dir2, "--config", config]
        if targets:
            log.info(f"Building specified targets: {targets}")
            cmd_args.extend(["--target", *targets])

        build_tool_args = []
        if num_parallel_jobs != 0:
            if is_windows() and args.cmake_generator != "Ninja" and not args.build_wasm:
                build_tool_args += [
                    f"/maxcpucount:{num_parallel_jobs}",
                    f"/p:CL_MPCount={num_parallel_jobs}",
                    # if nodeReuse is true, msbuild processes will stay around for a bit after the build completes
                    "/nodeReuse:False",
                ]
            elif args.cmake_generator == "Xcode":
                build_tool_args += [
                    "-parallelizeTargets",
                    "-jobs",
                    str(num_parallel_jobs),
                ]
            else:
                build_tool_args += [f"-j{num_parallel_jobs}"]

        if build_tool_args:
            cmd_args += ["--"]
            cmd_args += build_tool_args

        env = {}
        if args.android:
            env["ANDROID_SDK_ROOT"] = args.android_sdk_path
            env["ANDROID_NDK_HOME"] = args.android_ndk_path

        run_subprocess(cmd_args, env=env)


def add_dir_if_exists(directory, dir_list):
    if os.path.isdir(directory):
        dir_list.append(directory)


def setup_cuda_vars(args):
    cuda_home = ""
    cudnn_home = ""

    if args.use_cuda:
        cuda_home = args.cuda_home if args.cuda_home else os.getenv("CUDA_HOME")
        cudnn_home = args.cudnn_home if args.cudnn_home else os.getenv("CUDNN_HOME")

        cuda_home_valid = cuda_home is not None and os.path.exists(cuda_home)
        cudnn_home_valid = cudnn_home is not None and os.path.exists(cudnn_home)

        if not cuda_home_valid or (not is_windows() and not cudnn_home_valid):
            raise BuildError(
                "cuda_home and cudnn_home paths must be specified and valid.",
                f"cuda_home='{cuda_home}' valid={cuda_home_valid}. cudnn_home='{cudnn_home}' valid={cudnn_home_valid}",
            )

    return cuda_home, cudnn_home


def setup_cann_vars(args):
    cann_home = ""

    if args.use_cann:
        cann_home = args.cann_home if args.cann_home else os.getenv("ASCEND_HOME_PATH")

        cann_home_valid = cann_home is not None and os.path.exists(cann_home)

        if not cann_home_valid:
            raise BuildError(
                "cann_home paths must be specified and valid.",
                f"cann_home='{cann_home}' valid={cann_home_valid}.",
            )

    return cann_home


def setup_tensorrt_vars(args):
    tensorrt_home = ""
    if args.use_tensorrt:
        tensorrt_home = args.tensorrt_home if args.tensorrt_home else os.getenv("TENSORRT_HOME")
        tensorrt_home_valid = tensorrt_home is not None and os.path.exists(tensorrt_home)
        if not tensorrt_home_valid:
            raise BuildError(
                "tensorrt_home paths must be specified and valid.",
                f"tensorrt_home='{tensorrt_home}' valid={tensorrt_home_valid}.",
            )

        # Set maximum workspace size in byte for
        # TensorRT (1GB = 1073741824 bytes).
        os.environ["ORT_TENSORRT_MAX_WORKSPACE_SIZE"] = "1073741824"

        # Set maximum number of iterations to detect unsupported nodes
        # and partition the models for TensorRT.
        os.environ["ORT_TENSORRT_MAX_PARTITION_ITERATIONS"] = "1000"

        # Set minimum subgraph node size in graph partitioning
        # for TensorRT.
        os.environ["ORT_TENSORRT_MIN_SUBGRAPH_SIZE"] = "1"

        # Set FP16 flag
        os.environ["ORT_TENSORRT_FP16_ENABLE"] = "0"

    return tensorrt_home


def setup_migraphx_vars(args):
    migraphx_home = None

    if args.use_migraphx:
        print(f"migraphx_home = {args.migraphx_home}")
        migraphx_home = args.migraphx_home or os.getenv("MIGRAPHX_HOME") or None

        migraphx_home_not_valid = migraphx_home and not os.path.exists(migraphx_home)

        if migraphx_home_not_valid:
            raise BuildError(
                "migraphx_home paths must be specified and valid.",
                f"migraphx_home='{migraphx_home}' valid={migraphx_home_not_valid}.",
            )
    return migraphx_home or ""


def setup_dml_build(args, cmake_path, build_dir, configs):
    if not args.use_dml:
        return

    if args.dml_path:
        for expected_file in ["bin/DirectML.dll", "lib/DirectML.lib", "include/DirectML.h"]:
            file_path = os.path.join(args.dml_path, expected_file)
            if not os.path.exists(file_path):
                raise BuildError("dml_path is invalid.", f"dml_path='{args.dml_path}' expected_file='{file_path}'.")
    elif not args.dml_external_project:
        for config in configs:
            # Run the RESTORE_PACKAGES target to perform the initial
            # NuGet setup.
            cmd_args = [
                cmake_path,
                "--build",
                get_config_build_dir(build_dir, config),
                "--config",
                config,
                "--target",
                "RESTORE_PACKAGES",
            ]
            run_subprocess(cmd_args)

    if args.minimal_build is not None:
        raise BuildError("use_dml and minimal_build may not both be set")


def run_android_tests(args, source_dir, build_dir, config, cwd):
    if args.android_abi != "x86_64":
        log.info(f"--android_abi ({args.android_abi}) is not x86_64, skipping running of Android tests on emulator.")
        return

    sdk_tool_paths = android.get_sdk_tool_paths(args.android_sdk_path)
    device_dir = "/data/local/tmp"

    def adb_push(src, dest, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, "push", src, dest], **kwargs)

    def adb_shell(*args, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, "shell", *args], **kwargs)

    def adb_logcat(*args, **kwargs):
        return run_subprocess([sdk_tool_paths.adb, "logcat", *args], **kwargs)

    def run_adb_shell(cmd):
        # GCOV_PREFIX_STRIP specifies the depth of the directory hierarchy to strip and
        # GCOV_PREFIX specifies the root directory
        # for creating the runtime code coverage files.
        if args.code_coverage:
            adb_shell(f"cd {device_dir} && GCOV_PREFIX={device_dir} GCOV_PREFIX_STRIP={cwd.count(os.sep) + 1} {cmd}")
        else:
            adb_shell(f"cd {device_dir} && {cmd}")

    with contextlib.ExitStack() as context_stack:
        if args.android_run_emulator:
            avd_name = "ort_android"
            system_image = f"system-images;android-{args.android_api};default;{args.android_abi}"

            android.create_virtual_device(sdk_tool_paths, system_image, avd_name)
            emulator_proc = context_stack.enter_context(
                android.start_emulator(
                    sdk_tool_paths=sdk_tool_paths,
                    avd_name=avd_name,
                    extra_args=["-partition-size", "2047", "-wipe-data"],
                )
            )
            context_stack.callback(android.stop_emulator, emulator_proc)

        all_android_tests_passed = False

        def dump_logs_on_failure():
            if not all_android_tests_passed:
                log.warning("Android test failed. Dumping logs.")
                adb_logcat("-d")  # dump logs

        context_stack.callback(dump_logs_on_failure)

        adb_logcat("-c")  # clear logs

        adb_push("testdata", device_dir, cwd=cwd)
        if is_linux() and os.path.exists("/data/onnx"):
            adb_push("/data/onnx", device_dir + "/test", cwd=cwd)
        else:
            test_data_dir = os.path.join(source_dir, "cmake", "external", "onnx", "onnx", "backend", "test")
            if os.path.exists(test_data_dir):
                adb_push(test_data_dir, device_dir + "/test", cwd=cwd)

        for test_program_name in ["onnxruntime_test_all", "onnxruntime_provider_test"]:
            adb_push(test_program_name, device_dir, cwd=cwd)
            adb_shell(f"chmod +x {device_dir}/{test_program_name}")
            run_adb_shell(f"{device_dir}/{test_program_name}")

            # remove test program when we are done testing to free up space
            adb_shell(f"rm {device_dir}/{test_program_name}")

        if args.build_java:
            # use the gradle wrapper under <repo root>/java
            gradle_executable = os.path.join(source_dir, "java", "gradlew.bat" if is_windows() else "gradlew")
            android_test_path = os.path.join(cwd, "java", "androidtest", "android")
            run_subprocess(
                [
                    gradle_executable,
                    "--no-daemon",
                    f"-DminSdkVer={args.android_api}",
                    "clean",
                    "connectedDebugAndroidTest",
                ],
                cwd=android_test_path,
            )

        adb_push("onnx_test_runner", device_dir, cwd=cwd)
        adb_shell(f"chmod +x {device_dir}/onnx_test_runner")

        if args.use_nnapi:
            run_adb_shell(f"{device_dir}/onnx_test_runner -e nnapi {device_dir}/test")
        else:
            run_adb_shell(f"{device_dir}/onnx_test_runner {device_dir}/test")

        # run shared library tests if applicable
        if args.build_shared_lib:
            shared_library_test_program_names = [
                "onnxruntime_autoep_test",
                "onnxruntime_customopregistration_test",
                "onnxruntime_shared_lib_test",
            ]

            shared_libraries = [
                "libcustom_op_get_const_input_test_library.so",
                "libcustom_op_library.so",
                "libexample_plugin_ep_virt_gpu.so",
                "libexample_plugin_ep.so",
                "libexample_plugin_ep_kernel_registry.so",
                "libonnxruntime_runtime_path_test_shared_library.so",
                "libonnxruntime.so",
            ]

            for file_to_copy in shared_library_test_program_names + shared_libraries:
                adb_push(file_to_copy, device_dir, cwd=cwd)

            # run test programs
            for test_program_name in shared_library_test_program_names:
                adb_shell(f"chmod +x {device_dir}/{test_program_name}")
                run_adb_shell(f"LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{device_dir} {device_dir}/{test_program_name}")

        all_android_tests_passed = True


def run_ios_tests(args, source_dir, config, cwd):
    is_targeting_iphone_simulator = "iphonesimulator" in args.apple_sysroot.lower()
    if not is_targeting_iphone_simulator:
        log.info(
            f"Could not detect iphonesimulator target from --apple_sysroot ({args.apple_sysroot}), "
            "skipping running of iOS tests on simulator."
        )
        return

    host_arch = platform.machine()
    if host_arch != args.osx_arch:
        log.info(
            f"Host arch ({host_arch}) and --osx_arch ({args.osx_arch}) mismatch, "
            "skipping running of iOS tests on simulator."
        )
        return

    simulator_device_info = subprocess.check_output(
        [
            sys.executable,
            os.path.join(source_dir, "tools", "ci_build", "github", "apple", "get_simulator_device_info.py"),
        ],
        text=True,
    ).strip()
    log.debug(f"Simulator device info:\n{simulator_device_info}")

    simulator_device_info = json.loads(simulator_device_info)

    xc_test_schemes = [
        "onnxruntime_test_all_xc",
        "onnxruntime_provider_test_xc",
    ]

    if args.build_shared_lib:
        xc_test_schemes += [
            "onnxruntime_shared_lib_test_xc",
            "onnxruntime_customopregistration_test_xc",
        ]

    for xc_test_scheme in xc_test_schemes:
        run_subprocess(
            [
                "xcodebuild",
                "test-without-building",
                "-project",
                "./onnxruntime.xcodeproj",
                "-configuration",
                config,
                "-scheme",
                xc_test_scheme,
                "-destination",
                f"platform=iOS Simulator,id={simulator_device_info['device_udid']}",
            ],
            cwd=cwd,
        )

    if args.build_apple_framework:
        package_test_py = os.path.join(source_dir, "tools", "ci_build", "github", "apple", "test_apple_packages.py")
        framework_info_file = os.path.join(cwd, "framework_info.json")
        dynamic_framework_dir = os.path.join(cwd, config + "-" + args.apple_sysroot)
        static_framework_dir = os.path.join(cwd, config + "-" + args.apple_sysroot, "static_framework")
        # test dynamic framework
        run_subprocess(
            [
                sys.executable,
                package_test_py,
                "--c_framework_dir",
                dynamic_framework_dir,
                "--framework_info_file",
                framework_info_file,
                "--skip_macos_test",
            ],
            cwd=cwd,
        )
        # test static framework
        run_subprocess(
            [
                sys.executable,
                package_test_py,
                "--c_framework_dir",
                static_framework_dir,
                "--framework_info_file",
                framework_info_file,
                "--skip_macos_test",
            ],
            cwd=cwd,
        )


def run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs):
    for config in configs:
        log.info("Running tests for %s configuration", config)
        cwd = get_config_build_dir(build_dir, config)
        cwd = os.path.abspath(cwd)

        if args.android:
            run_android_tests(args, source_dir, build_dir, config, cwd)
            continue
        elif is_macOS() and args.ios:
            run_ios_tests(args, source_dir, config, cwd)
            continue
        dll_path_list = []
        if args.use_tensorrt:
            dll_path_list.append(os.path.join(args.tensorrt_home, "lib"))

        dll_path = None
        if len(dll_path_list) > 0:
            dll_path = os.pathsep.join(dll_path_list)

        if not ctest_path and not is_windows():
            executables = ["onnxruntime_test_all", "onnxruntime_mlas_test", "onnxruntime_provider_test"]
            if args.build_shared_lib:
                executables.append("onnxruntime_shared_lib_test")
                executables.append("onnxruntime_global_thread_pools_test")
                executables.append("onnxruntime_customopregistration_test")
            for exe in executables:
                test_output = f"--gtest_output=xml:{cwd}/{exe}.{config}.results.xml"
                run_subprocess([os.path.join(cwd, exe), test_output], cwd=cwd, dll_path=dll_path)
        else:
            num_parallel_jobs = number_of_test_parallel_jobs(args)
            ctest_cmd = [
                ctest_path,
                "--build-config",
                config,
                "--verbose",
                "--timeout",
                args.ctest_timeout,
                "--parallel",
                str(num_parallel_jobs),
                "--output-on-failure",
            ]
            run_subprocess(ctest_cmd, cwd=cwd, dll_path=dll_path)

        if args.enable_pybind:
            python_path = None

            # Disable python tests in a reduced build as we don't know which ops have been included and which
            # models can run.
            if is_reduced_ops_build(args) or args.minimal_build is not None:
                return

            if is_windows():
                cwd = os.path.join(cwd, config)

            if not args.skip_pip_install and args.enable_transformers_tool_test and not args.disable_contrib_ops:
                # PyTorch is required for transformers tests, and optional for some python tests.
                # Install cpu only version of torch when cuda is not enabled in Linux.
                extra = [] if args.use_cuda and is_linux() else ["--index-url", "https://download.pytorch.org/whl/cpu"]
                run_subprocess(
                    [sys.executable, "-m", "pip", "install", "torch==2.10.0", "torchvision==0.25.0", *extra],
                    cwd=cwd,
                    dll_path=dll_path,
                    python_path=python_path,
                )
            elif not args.skip_pip_install and args.use_cuda and is_linux() and not args.disable_contrib_ops:
                # CUDA weight-packer parity tests also run without the transformers suite.
                run_subprocess(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch==2.10.0",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                    ],
                    cwd=cwd,
                    dll_path=dll_path,
                    python_path=python_path,
                )

            run_subprocess(
                [sys.executable, "onnxruntime_test_python.py"], cwd=cwd, dll_path=dll_path, python_path=python_path
            )
            run_subprocess(
                [sys.executable, "onnxruntime_test_python_fpa_intb_tune.py"],
                cwd=cwd,
                dll_path=dll_path,
                python_path=python_path,
            )

            if not args.disable_contrib_ops:
                log.info("Testing QMoE expert distribution analysis")
                run_subprocess(
                    [
                        sys.executable,
                        os.path.join(
                            source_dir,
                            "onnxruntime",
                            "test",
                            "python",
                            "test_qmoe_expert_distribution.py",
                        ),
                    ],
                    cwd=cwd,
                    dll_path=dll_path,
                    python_path=python_path,
                )

            log.info("Testing Global Thread Pool feature")
            run_subprocess([sys.executable, "onnxruntime_test_python_global_threadpool.py"], cwd=cwd, dll_path=dll_path)

            log.info("Testing AutoEP feature")
            run_subprocess([sys.executable, "onnxruntime_test_python_autoep.py"], cwd=cwd, dll_path=dll_path)

            if not args.disable_contrib_ops:
                run_subprocess([sys.executable, "onnxruntime_test_python_sparse_matmul.py"], cwd=cwd, dll_path=dll_path)

            if args.enable_symbolic_shape_infer_tests:
                run_subprocess(
                    [sys.executable, "onnxruntime_test_python_symbolic_shape_infer.py"], cwd=cwd, dll_path=dll_path
                )

            # For CUDA or DML enabled builds test IOBinding feature
            if args.use_cuda or args.use_dml:
                log.info("Testing IOBinding feature")
                run_subprocess([sys.executable, "onnxruntime_test_python_iobinding.py"], cwd=cwd, dll_path=dll_path)

            if args.use_cuda:
                log.info("Testing CUDA kernel execution")
                run_subprocess([sys.executable, "onnxruntime_test_python_cuda_kernel.py"], cwd=cwd, dll_path=dll_path)
                log.info("Testing CUDA Graph feature")
                run_subprocess([sys.executable, "onnxruntime_test_python_cudagraph.py"], cwd=cwd, dll_path=dll_path)
                log.info("Testing running inference concurrently")
                run_subprocess([sys.executable, "onnxruntime_test_python_ort_parallel.py"], cwd=cwd, dll_path=dll_path)

            if args.use_dml:
                log.info("Testing DML Graph feature")
                run_subprocess([sys.executable, "onnxruntime_test_python_dmlgraph.py"], cwd=cwd, dll_path=dll_path)

            if not args.disable_ml_ops and not args.use_tensorrt:
                run_subprocess([sys.executable, "onnxruntime_test_python_mlops.py"], cwd=cwd, dll_path=dll_path)

            if args.use_tensorrt:
                run_subprocess(
                    [sys.executable, "onnxruntime_test_python_nested_control_flow_op.py"], cwd=cwd, dll_path=dll_path
                )

            try:
                import onnx  # noqa: F401, PLC0415

                onnx_test = True
            except ImportError as error:
                log.exception(error)
                log.warning("onnx is not installed. The ONNX tests will be skipped.")
                onnx_test = False

            if onnx_test:
                log.info("Testing Symlink ONNX Model and External Data")
                run_subprocess([sys.executable, "onnxruntime_test_python_symlink_data.py"], cwd=cwd, dll_path=dll_path)

                # Disable python onnx tests for TensorRT and CANN EP, because many tests are
                # not supported yet.
                if args.use_tensorrt or args.use_cann:
                    return

                run_subprocess(
                    [sys.executable, "onnxruntime_test_python_backend.py"],
                    cwd=cwd,
                    dll_path=dll_path,
                    python_path=python_path,
                )

                if not args.disable_contrib_ops:
                    run_subprocess(
                        [sys.executable, "-m", "unittest", "discover", "-s", "quantization", "-v"],
                        cwd=cwd,
                        dll_path=dll_path,
                    )
                    run_subprocess(
                        [
                            sys.executable,
                            "-m",
                            "pytest",
                            "-q",
                            "-p",
                            "no:cacheprovider",
                            "--rootdir=.",
                            "--confcutdir=.",
                            "test_reference.py",
                        ],
                        cwd=os.path.join(cwd, "quantization", "qmoe_packed_reference"),
                        dll_path=dll_path,
                    )

                    if args.enable_transformers_tool_test and (sys.version_info.major, sys.version_info.minor) < (
                        3,
                        15,
                    ):
                        import google.protobuf  # noqa: PLC0415
                        import numpy  # noqa: PLC0415

                        numpy_init_version = numpy.__version__
                        pb_init_version = google.protobuf.__version__
                        if not args.skip_pip_install:
                            run_subprocess(
                                [
                                    sys.executable,
                                    "-m",
                                    "pip",
                                    "install",
                                    "-r",
                                    "requirements/transformers-test/requirements.txt",
                                ],
                                cwd=SCRIPT_DIR,
                            )
                        run_subprocess([sys.executable, "-m", "pytest", "--durations=0", "transformers"], cwd=cwd)
                        if not args.skip_pip_install:
                            # Restore initial numpy/protobuf version in case other tests use it
                            run_subprocess([sys.executable, "-m", "pip", "install", "numpy==" + numpy_init_version])
                            run_subprocess([sys.executable, "-m", "pip", "install", "protobuf==" + pb_init_version])

                if not args.disable_ml_ops:
                    run_subprocess(
                        [sys.executable, "onnxruntime_test_python_backend_mlops.py"], cwd=cwd, dll_path=dll_path
                    )

                run_subprocess(
                    [
                        sys.executable,
                        os.path.join(source_dir, "onnxruntime", "test", "onnx", "gen_test_models.py"),
                        "--output_dir",
                        "test_models",
                    ],
                    cwd=cwd,
                )

                if not args.disable_contrib_ops:
                    log.info("Testing Python Compile API")
                    run_subprocess(
                        [sys.executable, "onnxruntime_test_python_compile_api.py"], cwd=cwd, dll_path=dll_path
                    )

                if not args.skip_onnx_tests:
                    run_subprocess([os.path.join(cwd, "onnx_test_runner"), "test_models"], cwd=cwd)
                    if config != "Debug":
                        # Set the opset policy explicitly so the child process does not inherit the CI default.
                        run_subprocess(
                            [sys.executable, "onnx_backend_test_series.py"],
                            cwd=cwd,
                            dll_path=dll_path,
                            env=get_onnx_backend_test_environment(args.use_cuda),
                        )

            if not args.skip_keras_test:
                try:
                    import keras  # noqa: F401, PLC0415
                    import onnxmltools  # noqa: F401, PLC0415

                    onnxml_test = True
                except ImportError:
                    log.warning("onnxmltools and keras are not installed. The keras tests will be skipped.")
                    onnxml_test = False
                if onnxml_test:
                    run_subprocess([sys.executable, "onnxruntime_test_python_keras.py"], cwd=cwd, dll_path=dll_path)


def run_nodejs_tests(nodejs_binding_dir):
    args = ["npm", "test", "--", "--timeout=90000"]
    if is_windows():
        args = ["cmd", "/c", *args]
    run_subprocess(args, cwd=nodejs_binding_dir)


def parse_cuda_version_from_json(cuda_home):
    version_file_path = os.path.join(cuda_home, "version.json")
    if not os.path.exists(version_file_path):
        print(f"version.json not found in {cuda_home}.")
    else:
        try:
            with open(version_file_path) as version_file:
                version_data = json.load(version_file)
                cudart_info = version_data.get("cuda")
                if cudart_info and "version" in cudart_info:
                    parts = cudart_info["version"].split(".")
                    return ".".join(parts[:2])
        except FileNotFoundError:
            print(f"version.json not found in {cuda_home}.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON from version.json in {cuda_home}.")

    return ""


def build_python_wheel(
    source_dir,
    build_dir,
    configs,
    use_cuda,
    cuda_home,
    cuda_version,
    use_migraphx,
    use_dnnl,
    use_tensorrt,
    use_openvino,
    use_vitisai,
    use_acl,
    use_dml,
    use_webgpu,
    use_cann,
    use_azure,
    use_qnn,
    qnn_home,
    wheel_name_suffix,
    enable_training,
    nightly_build=False,
    default_training_package_device=False,
    use_ninja=False,
    enable_training_apis=False,
):
    for config in configs:
        cwd = get_config_build_dir(build_dir, config)
        if is_windows() and not use_ninja:
            cwd = os.path.join(cwd, config)

        args = [sys.executable, os.path.join(source_dir, "setup.py"), "bdist_wheel"]

        # Any combination of the following arguments can be applied
        if nightly_build:
            args.append("--nightly_build")
        if default_training_package_device:
            args.append("--default_training_package_device")
        if wheel_name_suffix:
            args.append(f"--wheel_name_suffix={wheel_name_suffix}")
        if enable_training:
            args.append("--enable_training")
        if enable_training_apis:
            args.append("--enable_training_apis")

        # The following arguments are mutually exclusive
        if use_cuda:
            # The following line assumes no other EP is enabled
            args.append("--wheel_name_suffix=gpu")
            cuda_version = cuda_version or parse_cuda_version_from_json(cuda_home)
            if cuda_version:
                args.append(f"--cuda_version={cuda_version}")
        elif use_migraphx:
            args.append("--use_migraphx")
        elif use_openvino:
            args.append("--use_openvino")
        elif use_dnnl:
            args.append("--use_dnnl")
        elif use_vitisai:
            args.append("--use_vitisai")
        elif use_acl:
            args.append("--use_acl")
        elif use_dml:
            args.append("--wheel_name_suffix=directml")
        elif use_webgpu:
            args.append("--wheel_name_suffix=webgpu")
        elif use_cann:
            args.append("--use_cann")
        elif use_qnn:
            args.append("--use_qnn")
            qnn_version = parse_qnn_version_from_sdk_yaml(qnn_home)
            if qnn_version:
                args.append(f"--qnn_version={qnn_version}")
        elif use_azure:
            args.append("--use_azure")

        run_subprocess(args, cwd=cwd)


def build_nuget_package(
    cmake_path,
    source_dir,
    build_dir,
    configs,
    use_cuda,
    use_openvino,
    use_tensorrt,
    use_dnnl,
    use_winml,
    use_qnn,
    use_dml,
    use_migraphx,
    enable_training_apis,
    msbuild_extra_options,
):
    if not (is_windows() or is_linux()):
        raise BuildError(
            "Currently csharp builds and nuget package creation is only supported on Windows and Linux platforms."
        )

    csharp_build_dir = os.path.join(source_dir, "csharp")

    # in most cases we don't want/need to include the MAUI mobile targets, as doing so means the mobile workloads
    # must be installed on the machine.
    # they are only included in the Microsoft.ML.OnnxRuntime nuget package
    sln = "OnnxRuntime.DesktopOnly.CSharp.sln"
    have_exclude_mobile_targets_option = "IncludeMobileTargets=false" in msbuild_extra_options

    # derive package name and execution provider based on the build args
    target_name = "/t:CreatePackage"
    execution_provider = "/p:ExecutionProvider=None"
    package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime"
    enable_training_tests = "/p:TrainingEnabledNativeBuild=false"

    if enable_training_apis:
        enable_training_tests = "/p:TrainingEnabledNativeBuild=true"
        if use_cuda:
            package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.Training.Gpu"
        else:
            package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.Training"
    elif use_winml:
        package_name = "/p:OrtPackageId=Microsoft.AI.MachineLearning"
        target_name = "/t:CreateWindowsAIPackage"
    elif use_openvino:
        execution_provider = "/p:ExecutionProvider=openvino"
        package_name = "/p:OrtPackageId=Intel.ML.OnnxRuntime.OpenVino"
    elif use_tensorrt:
        execution_provider = "/p:ExecutionProvider=tensorrt"
        package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.TensorRT"
    elif use_migraphx:
        execution_provider = "/p:ExecutionProvider=migraphx"
        package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.MIGraphX"
    elif use_dnnl:
        execution_provider = "/p:ExecutionProvider=dnnl"
        package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.DNNL"
    elif use_cuda:
        package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.Gpu"
    elif use_dml:
        package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.DirectML"
    elif use_qnn:
        if use_qnn != "shared_lib":
            raise BuildError("Currently NuGet packages with QNN require QNN EP to be built as a shared library.")
        execution_provider = "/p:ExecutionProvider=qnn"
        package_name = "/p:OrtPackageId=Microsoft.ML.OnnxRuntime.QNN"
    elif any("OrtPackageId=" in x for x in msbuild_extra_options):
        pass
    else:
        # we currently only allow building with mobile targets on Windows.
        # it should be possible to allow building with android targets on Linux but that requires updating the
        # csproj to separate the inclusion of ios and android targets.
        if is_windows() and have_exclude_mobile_targets_option is False:
            # use the sln that include the mobile targets
            sln = "OnnxRuntime.CSharp.sln"

    # expand extra_options to add prefix
    extra_options = ["/p:" + option for option in msbuild_extra_options]

    # explicitly exclude mobile targets in this case
    if sln != "OnnxRuntime.CSharp.sln" and have_exclude_mobile_targets_option is False:
        extra_options.append("/p:IncludeMobileTargets=false")

    # we have to use msbuild directly if including Xamarin targets as dotnet only supports MAUI (.net6)
    use_dotnet = sln != "OnnxRuntime.CSharp.sln"

    # build csharp bindings and create nuget package for each config
    for config in configs:
        configuration = "/p:Configuration=" + config
        extra_options += [configuration, "/p:Platform=Any CPU"]
        if use_dotnet:
            cmd_args = ["dotnet", "restore", sln, "--configfile", "NuGet.CSharp.config", *extra_options]
        else:
            cmd_args = ["msbuild", sln, "/t:restore", "/p:RestoreConfigFile=NuGet.CSharp.config", *extra_options]

        # set build directory based on build_dir arg
        native_dir = os.path.normpath(os.path.join(source_dir, build_dir))
        ort_build_dir = "/p:OnnxRuntimeBuildDirectory=" + native_dir

        run_subprocess(cmd_args, cwd=csharp_build_dir)

        if not use_winml:
            cmd_args = ["dotnet"] if use_dotnet else []
            cmd_args += [
                "msbuild",
                sln,
                package_name,
                ort_build_dir,
                enable_training_tests,
                *extra_options,
            ]

            run_subprocess(cmd_args, cwd=csharp_build_dir)
        else:
            winml_interop_dir = os.path.join(source_dir, "csharp", "src", "Microsoft.AI.MachineLearning.Interop")
            winml_interop_project = os.path.join(winml_interop_dir, "Microsoft.AI.MachineLearning.Interop.csproj")
            winml_interop_project = os.path.normpath(winml_interop_project)
            cmd_args = [
                "dotnet",
                "msbuild",
                winml_interop_project,
                configuration,
                "/p:Platform=Any CPU",
                ort_build_dir,
                "-restore",
            ]
            run_subprocess(cmd_args, cwd=csharp_build_dir)

        if is_windows():
            if not use_winml:
                # user needs to make sure nuget is installed and added to the path variable
                nuget_exe = "nuget.exe"
            else:
                # this path is setup by cmake/nuget_helpers.cmake for MSVC on Windows
                nuget_exe = os.path.normpath(os.path.join(native_dir, config, "nuget_exe", "src", "nuget.exe"))
        else:
            # `dotnet pack` is used on Linux
            nuget_exe = "NugetExe_not_set"

        nuget_exe_arg = '/p:NugetExe="' + nuget_exe + '"'

        cmd_args = ["dotnet"] if use_dotnet else []
        cmd_args += [
            "msbuild",
            "OnnxRuntime.CSharp.proj",
            target_name,
            package_name,
            execution_provider,
            ort_build_dir,
            nuget_exe_arg,
            *extra_options,
        ]

        run_subprocess(cmd_args, cwd=csharp_build_dir)

        log.info(f"nuget package was created in the {config} build output directory.")


def run_csharp_tests(
    source_dir,
    build_dir,
    use_cuda,
    use_openvino,
    use_tensorrt,
    use_dnnl,
    enable_training_apis,
    configs,
    msbuild_extra_options,
):
    # Currently only running tests on windows.
    if not is_windows():
        return
    csharp_source_dir = os.path.join(source_dir, "csharp")

    # define macros based on build args
    macros = []
    if use_openvino:
        macros.append("USE_OPENVINO")
    if use_tensorrt:
        macros.append("USE_TENSORRT")
    if use_dnnl:
        macros.append("USE_DNNL")
    if use_cuda:
        macros.append("USE_CUDA")
    if enable_training_apis:
        macros += ["__TRAINING_ENABLED_NATIVE_BUILD__", "__ENABLE_TRAINING_APIS__"]

    define_constants = ""
    if macros:
        define_constants = '/p:DefineConstants="' + ";".join(macros) + '"'

    # set build directory based on build_dir arg
    native_build_dir = os.path.normpath(os.path.join(source_dir, build_dir))
    ort_build_dir = '/p:OnnxRuntimeBuildDirectory="' + native_build_dir + '"'
    # expand extra_options to add prefix
    extra_options = ["/p:" + option for option in msbuild_extra_options]
    for config in configs:
        extra_options.append("/p:Configuration=" + config)
        # Skip pretrained models test. Only run unit tests as part of the build
        # add "--verbosity", "detailed" to this command if required
        cmd_args = [
            "dotnet",
            "test",
            "test\\Microsoft.ML.OnnxRuntime.Tests.NetCoreApp\\Microsoft.ML.OnnxRuntime.Tests.NetCoreApp.csproj",
            "--filter",
            "FullyQualifiedName!=Microsoft.ML.OnnxRuntime.Tests.InferenceTest.TestPreTrainedModels",
            define_constants,
            ort_build_dir,
        ]
        cmd_args += extra_options
        run_subprocess(cmd_args, cwd=csharp_source_dir)


def generate_documentation(source_dir, build_dir, configs, validate):
    # Randomly choose one build config
    config = next(iter(configs))
    cwd = get_config_build_dir(build_dir, config)
    if is_windows():
        cwd = os.path.join(cwd, config)

    contrib_op_doc_path = os.path.join(source_dir, "docs", "ContribOperators.md")
    opkernel_doc_path = os.path.join(source_dir, "docs", "OperatorKernels.md")
    shutil.copy(os.path.join(source_dir, "tools", "python", "gen_contrib_doc.py"), cwd)
    shutil.copy(os.path.join(source_dir, "tools", "python", "gen_opkernel_doc.py"), cwd)
    # limit to just com.microsoft (excludes purely internal stuff like com.microsoft.nchwc).
    run_subprocess(
        [sys.executable, "gen_contrib_doc.py", "--output_path", contrib_op_doc_path, "--domains", "com.microsoft"],
        cwd=cwd,
    )
    # we currently limit the documentation created by a build to a subset of EP's.
    # Run get_opkernel_doc.py directly if you need/want documentation from other EPs that are enabled in the build.
    run_subprocess(
        [
            sys.executable,
            "gen_opkernel_doc.py",
            "--output_path",
            opkernel_doc_path,
            "--providers",
            "CPU",
            "CUDA",
            "DML",
        ],
        cwd=cwd,
    )

    if validate:
        try:
            have_diff = False

            def diff_file(path, regenerate_qualifiers=""):
                diff = subprocess.check_output(["git", "diff", "--ignore-blank-lines", path], cwd=source_dir).decode(
                    "utf-8"
                )
                if diff:
                    nonlocal have_diff
                    have_diff = True
                    log.warning(
                        f"The updated document {path} is different from the checked in version. "
                        f"Please regenerate the file{regenerate_qualifiers}, or copy the updated version from the "
                        "CI build's published artifacts if applicable."
                    )
                    log.debug("diff:\n" + diff)  # noqa: G003

            diff_file(opkernel_doc_path, " with CPU, CUDA and DML execution providers enabled")
            diff_file(contrib_op_doc_path)

            if have_diff:
                # Output for the CI to publish the updated md files as an artifact
                print("##vso[task.setvariable variable=DocUpdateNeeded]true")
                raise BuildError("Generated documents have diffs. Check build output for details.")

        except subprocess.CalledProcessError:
            raise BuildError("git diff returned non-zero error code")  # noqa: B904


def main():
    log.debug("Command line arguments:\n  {}".format(" ".join(shlex.quote(arg) for arg in sys.argv[1:])))  # noqa: G001

    args = parse_arguments()

    print(args)

    if os.getenv("ORT_BUILD_WITH_CACHE") == "1":
        args.use_cache = True

    # VCPKG's scripts/toolchains/android.cmake has logic for autodetecting NDK home when the ANDROID_NDK_HOME env is not set, but it is only implemented for Windows
    if args.android and args.use_vcpkg and args.android_ndk_path is not None and os.path.exists(args.android_ndk_path):
        os.environ["ANDROID_NDK_HOME"] = args.android_ndk_path

    if not is_windows() and not is_macOS():
        if not args.allow_running_as_root:
            is_root_user = os.geteuid() == 0
            if is_root_user:
                raise BuildError(
                    "Running as root is not allowed. If you really want to do that, use '--allow_running_as_root'."
                )

    cmake_extra_defines = list(args.cmake_extra_defines)

    if args.use_tensorrt:
        args.use_cuda = True

    if args.build_wheel or args.gen_doc or args.enable_training:
        args.enable_pybind = True

    if (
        args.build_csharp
        or args.build_nuget
        or args.build_java
        or args.build_nodejs
        or (args.enable_pybind and not args.enable_training)
    ):
        # If pyhon bindings are enabled, we embed the shared lib in the python package.
        # If training is enabled, we don't embed the shared lib in the python package since training requires
        # torch interop.
        args.build_shared_lib = True

    if args.enable_pybind:
        if args.disable_rtti:
            raise BuildError("Python bindings use typeid so you can't disable RTTI")

        if args.disable_exceptions:
            raise BuildError("Python bindings require exceptions to be enabled.")

        if args.minimal_build is not None:
            raise BuildError("Python bindings are not supported in a minimal build.")

    if args.nnapi_min_api:
        if not args.use_nnapi:
            raise BuildError("Using --nnapi_min_api requires --use_nnapi")
        if args.nnapi_min_api < 27:
            raise BuildError("--nnapi_min_api should be 27+")

    if args.build_wasm:
        if not args.disable_wasm_exception_catching and args.disable_exceptions:
            # When '--disable_exceptions' is set, we set '--disable_wasm_exception_catching' as well
            args.disable_wasm_exception_catching = True
        if args.test and args.disable_wasm_exception_catching and not args.minimal_build:
            raise BuildError("WebAssembly tests need exception catching enabled to run if it's not minimal build")
        if args.test and args.enable_wasm_debug_info:
            # With flag --enable_wasm_debug_info, the test program .wasm file will be very huge (>1GB). This will fail
            # Node.js when trying to load the .wasm file.
            # To debug ONNX Runtime WebAssembly, use ONNX Runtime Web to debug ort-wasm.wasm in browsers.
            raise BuildError("WebAssembly tests cannot be enabled with flag --enable_wasm_debug_info")

        if args.wasm_malloc is not None:
            # mark --wasm_malloc as deprecated
            log.warning(
                "Flag '--wasm_malloc=<Value>' is deprecated. Please use '--emscripten_settings MALLOC=<Value>'."
            )

    if args.code_coverage and not args.android:
        raise BuildError("Using --code_coverage requires --android")

    # Disabling unit tests for GPU on nuget creation
    if args.use_openvino and args.use_openvino != "CPU" and args.build_nuget:
        args.test = False

    # GDK builds don't support testing
    if is_windows() and args.use_gdk:
        args.test = False

    # enable_training is a higher level flag that enables all training functionality.
    if args.enable_training:
        args.enable_training_apis = True
        args.enable_training_ops = True

    configs = set(args.config)

    # setup paths and directories
    # cmake_path and ctest_path can be None. For example, if a person only wants to run the tests, he/she doesn't need
    # to have cmake/ctest.
    cmake_path = resolve_executable_path(args.cmake_path)
    ctest_path = resolve_executable_path(args.ctest_path)
    build_dir = args.build_dir
    script_dir = os.path.realpath(os.path.dirname(__file__))
    source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))

    # if using cuda, setup cuda paths and env vars
    cuda_home = ""
    cudnn_home = ""
    if args.use_cuda:
        cuda_home, cudnn_home = setup_cuda_vars(args)

    nccl_home = args.nccl_home

    snpe_root = args.snpe_root

    acl_home = args.acl_home
    acl_libs = args.acl_libs

    qnn_home = ""
    if args.use_qnn:
        qnn_home = args.qnn_home

    # if using tensorrt, setup tensorrt paths
    tensorrt_home = ""
    if args.use_tensorrt:
        tensorrt_home = setup_tensorrt_vars(args)

    # if using migraphx, setup migraphx paths
    migraphx_home = setup_migraphx_vars(args)

    # if using cann, setup cann paths
    cann_home = setup_cann_vars(args)

    if args.update or args.build:
        for config in configs:
            os.makedirs(get_config_build_dir(build_dir, config), exist_ok=True)

    log.info("Build started")

    if args.update:
        if is_reduced_ops_build(args):
            from reduce_op_kernels import reduce_ops  # noqa: PLC0415

            is_extended_minimal_build_or_higher = args.minimal_build is None or "extended" in args.minimal_build
            for config in configs:
                reduce_ops(
                    config_path=args.include_ops_by_config,
                    build_dir=get_config_build_dir(build_dir, config),
                    enable_type_reduction=args.enable_reduced_operator_type_support,
                    use_cuda=args.use_cuda,
                    is_extended_minimal_build_or_higher=is_extended_minimal_build_or_higher,
                )

        cmake_extra_args = []
        path_to_protoc_exe = None
        if args.path_to_protoc_exe:
            path_to_protoc_exe = Path(args.path_to_protoc_exe)
            if not path_to_protoc_exe.exists():
                raise BuildError("The value to --path_to_protoc_exe is invalid.")
        if not args.skip_submodule_sync:
            update_submodules(source_dir)
        if is_windows() and not args.build_wasm:
            cpu_arch = platform.architecture()[0]
            if args.cmake_generator == "Ninja":
                if cpu_arch == "32bit" or args.arm or args.arm64 or args.arm64ec:
                    raise BuildError(
                        "To cross-compile with Ninja, load the toolset "
                        "environment for the target processor (e.g. Cross "
                        "Tools Command Prompt for VS)"
                    )
                cmake_extra_args = ["-G", args.cmake_generator]
            elif args.arm or args.arm64 or args.arm64ec:
                if args.arm:
                    cmake_extra_args = ["-A", "ARM"]
                elif args.arm64:
                    cmake_extra_args = ["-A", "ARM64"]
                    if args.buildasx:
                        cmake_extra_args += ["-D", "BUILD_AS_ARM64X=ARM64"]
                elif args.arm64ec:
                    cmake_extra_args = ["-A", "ARM64EC"]
                    if args.buildasx:
                        cmake_extra_args += ["-D", "BUILD_AS_ARM64X=ARM64EC"]
                cmake_extra_args += ["-G", args.cmake_generator]
                # Cannot test on host build machine for cross-compiled
                # builds (Override any user-defined behavior for test if any)
                if args.test:
                    log.warning(
                        "Cannot test on host build machine for cross-compiled "
                        "ARM(64) builds. Will skip test running after build."
                    )
                    args.test = False
            else:
                target_arch = platform.machine()
                if target_arch == "AMD64":
                    if cpu_arch == "32bit" or args.x86:
                        target_arch = "Win32"
                    else:
                        target_arch = "x64"
                    host_arch = "x64"
                elif target_arch == "ARM64":
                    host_arch = "ARM64"
                else:
                    raise BuildError("unknown python arch")
                if args.msvc_toolset:
                    toolset = "host=" + host_arch + ",version=" + args.msvc_toolset
                else:
                    toolset = "host=" + host_arch
                if args.use_cuda and args.cuda_version:
                    toolset += ",cuda=" + args.cuda_version
                elif args.use_cuda and args.cuda_home:
                    toolset += ",cuda=" + args.cuda_home
                if args.windows_sdk_version:
                    target_arch += ",version=" + args.windows_sdk_version
                cmake_extra_args = ["-A", target_arch, "-T", toolset, "-G", args.cmake_generator]
            if args.enable_wcos:
                cmake_extra_defines.append("CMAKE_USER_MAKE_RULES_OVERRIDE=wcos_rules_override.cmake")

        elif args.cmake_generator is not None:
            cmake_extra_args += ["-G", args.cmake_generator]

        if is_macOS():
            if (
                not (args.ios or args.visionos or args.tvos)
                and args.macos != "Catalyst"
                and not args.android
                and args.osx_arch == "arm64"
                and platform.machine() == "x86_64"
            ):
                if args.test:
                    log.warning("Cannot test ARM64 build on X86_64. Will skip test running after build.")
                    args.test = False

        if args.build_wasm:
            if is_windows() and platform.architecture()[0] == "32bit":
                raise BuildError("Please use a 64-bit python to run this script")
            if args.build_wheel or args.enable_pybind:
                raise BuildError("WASM does not support pybind")
            emsdk_version = args.emsdk_version
            emsdk_dir = os.path.join(source_dir, "cmake", "external", "emsdk")
            emsdk_file = os.path.join(emsdk_dir, "emsdk.bat") if is_windows() else os.path.join(emsdk_dir, "emsdk")

            log.info("Installing emsdk...")
            run_subprocess([emsdk_file, "install", emsdk_version], cwd=emsdk_dir)
            log.info("Activating emsdk...")
            run_subprocess([emsdk_file, "activate", emsdk_version], cwd=emsdk_dir)

        if not args.skip_pip_install and args.enable_pybind and is_windows():
            run_subprocess(
                [sys.executable, "-m", "pip", "install", "-r", "requirements/pybind/requirements.txt"],
                cwd=SCRIPT_DIR,
            )

        generate_build_tree(
            cmake_path,
            source_dir,
            build_dir,
            cuda_home,
            cudnn_home,
            nccl_home,
            tensorrt_home,
            migraphx_home,
            acl_home,
            acl_libs,
            qnn_home,
            snpe_root,
            cann_home,
            path_to_protoc_exe,
            configs,
            cmake_extra_defines,
            args,
            cmake_extra_args,
        )

    if args.clean:
        clean_targets(cmake_path, build_dir, configs)

    if is_windows():
        # if using DML, perform initial nuget package restore
        setup_dml_build(args, cmake_path, build_dir, configs)

    if args.build:
        if args.parallel < 0:
            raise BuildError(f"Invalid parallel job count: {args.parallel}")
        num_parallel_jobs = number_of_parallel_jobs(args)
        build_targets(args, cmake_path, build_dir, configs, num_parallel_jobs, args.targets)

    if args.test:
        if args.test_parallel is not None and args.test_parallel < 0:
            raise BuildError(f"Invalid test parallel job count: {args.test_parallel}")

        if args.enable_onnx_tests:
            source_onnx_model_dir = "C:\\local\\models" if is_windows() else "/data/models"
            setup_test_data(source_onnx_model_dir, "models", build_dir, configs)

        run_onnxruntime_tests(args, source_dir, ctest_path, build_dir, configs)

        # run node.js binding tests
        if args.build_nodejs and not args.skip_nodejs_tests:
            nodejs_binding_dir = os.path.normpath(os.path.join(source_dir, "js", "node"))
            run_nodejs_tests(nodejs_binding_dir)

    # Build packages after running the tests.
    # NOTE: if you have a test that rely on a file which only get copied/generated during packaging step, it could
    # fail unexpectedly. Similar, if your packaging step forgot to copy a file into the package, we don't know it
    # either.
    if args.build:
        # TODO: find asan DLL and copy it to onnxruntime/capi folder when args.enable_address_sanitizer is True and
        #  the target OS is Windows
        if args.build_wheel:
            nightly_build = bool(os.getenv("NIGHTLY_BUILD") == "1")
            default_training_package_device = bool(os.getenv("DEFAULT_TRAINING_PACKAGE_DEVICE") == "1")
            build_python_wheel(
                source_dir,
                build_dir,
                configs,
                args.use_cuda,
                cuda_home,
                args.cuda_version,
                args.use_migraphx,
                args.use_dnnl,
                args.use_tensorrt,
                args.use_openvino,
                args.use_vitisai,
                args.use_acl,
                args.use_dml,
                args.use_webgpu,
                args.use_cann,
                args.use_azure,
                args.use_qnn,
                args.qnn_home,
                args.wheel_name_suffix,
                args.enable_training,
                nightly_build=nightly_build,
                default_training_package_device=default_training_package_device,
                use_ninja=(args.cmake_generator == "Ninja"),
                enable_training_apis=args.enable_training_apis,
            )

        if args.build_nuget:
            build_nuget_package(
                cmake_path,
                source_dir,
                build_dir,
                configs,
                args.use_cuda,
                args.use_openvino,
                args.use_tensorrt,
                args.use_dnnl,
                getattr(args, "use_winml", False),
                args.use_qnn,
                getattr(args, "use_dml", False),
                args.use_migraphx,
                args.enable_training_apis,
                args.msbuild_extra_options,
            )

    if args.test and args.build_nuget:
        run_csharp_tests(
            source_dir,
            build_dir,
            args.use_cuda,
            args.use_openvino,
            args.use_tensorrt,
            args.use_dnnl,
            args.enable_training_apis,
            configs,
            args.msbuild_extra_options,
        )

    if args.gen_doc:
        # special case CI where we create the build config separately to building
        if args.update and not args.build:
            pass
        else:
            # assumes build has occurred for easier use in CI where we don't always build via build.py and need to run
            # documentation generation as a separate task post-build
            generate_documentation(source_dir, build_dir, configs, args.gen_doc == "validate")

    log.info("Build complete")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BaseError as e:
        log.error(str(e))
        sys.exit(1)

#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import subprocess
import tempfile
import unittest
from pathlib import Path

_RUNTIME_LIBRARIES = (
    "libcmt.lib",
    "libcmtd.lib",
    "libcpmt.lib",
    "libcpmtd.lib",
    "libvcruntime.lib",
    "libvcruntimed.lib",
)


class OneCoreSpectreTest(unittest.TestCase):
    def _configure(
        self,
        *,
        platform: str = "x64",
        c_flags: str = "",
        cxx_flags: str = "",
        standard_libraries: str = "",
        gdk: bool = False,
        cross_compiling: bool = False,
        missing_library: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path, str]:
        repo_root = Path(__file__).resolve().parents[2]
        temporary_directory = tempfile.TemporaryDirectory(prefix="ort-onecore-spectre-")
        self.addCleanup(temporary_directory.cleanup)
        root = Path(temporary_directory.name)
        source_dir = root / "source"
        build_dir = root / "build"
        toolset_dir = root / "toolset"
        compiler = toolset_dir / "bin" / "Hostx64" / "x64" / "cl.exe"
        generic_spectre_dir = toolset_dir / "lib" / "spectre" / platform.lower()
        selected_platform = "ARM64" if platform == "ARM64EC" else platform
        spectre_onecore_dir = toolset_dir / "lib" / "spectre" / "onecore" / selected_platform
        onecore_dir = toolset_dir / "lib" / "onecore" / platform

        source_dir.mkdir()
        compiler.parent.mkdir(parents=True)
        compiler.touch()
        generic_spectre_dir.mkdir(parents=True)
        spectre_onecore_dir.mkdir(parents=True)
        onecore_dir.mkdir(parents=True)
        for library in _RUNTIME_LIBRARIES:
            if library != missing_library:
                (spectre_onecore_dir / library).touch()

        cmake_module = repo_root / "cmake" / "onnxruntime_msvc_runtime.cmake"
        (source_dir / "CMakeLists.txt").write_text(
            f"""
cmake_minimum_required(VERSION 3.28)
project(OneCoreSpectreSelection LANGUAGES NONE)
set(WIN32 TRUE)
set(GDK_PLATFORM "${{TEST_GDK}}")
set(CMAKE_CROSSCOMPILING "${{TEST_CROSS_COMPILING}}")
set(CMAKE_C_COMPILER "${{TEST_COMPILER}}")
set(CMAKE_C_FLAGS "${{TEST_C_FLAGS}}")
set(CMAKE_CXX_FLAGS "${{TEST_CXX_FLAGS}}")
set(CMAKE_CXX_STANDARD_LIBRARIES "${{TEST_STANDARD_LIBRARIES}}")
set(onnxruntime_target_platform "${{TEST_PLATFORM}}")
link_directories("${{TEST_GENERIC_SPECTRE_DIR}}")
include("{cmake_module.as_posix()}")
onnxruntime_configure_msvc_onecore_runtime()
get_property(link_directories DIRECTORY PROPERTY LINK_DIRECTORIES)
file(WRITE "${{TEST_RESULT_FILE}}" "${{link_directories}}")
""",
            encoding="utf-8",
        )

        result_file = root / "link_directories.txt"
        command = [
            "cmake",
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            f"-DTEST_PLATFORM={platform}",
            f"-DTEST_C_FLAGS={c_flags}",
            f"-DTEST_CXX_FLAGS={cxx_flags}",
            f"-DTEST_STANDARD_LIBRARIES={standard_libraries}",
            f"-DTEST_GDK={'ON' if gdk else 'OFF'}",
            f"-DTEST_CROSS_COMPILING={'ON' if cross_compiling else 'OFF'}",
            f"-DTEST_COMPILER={compiler}",
            f"-DTEST_GENERIC_SPECTRE_DIR={generic_spectre_dir}",
            f"-DTEST_RESULT_FILE={result_file}",
        ]
        result = subprocess.run(command, capture_output=True, check=False, text=True, timeout=30)
        return result, result_file, toolset_dir, result.stdout + result.stderr

    def _assert_selected(self, expected_relative_path: str, **kwargs) -> None:
        result, result_file, toolset_dir, output = self._configure(**kwargs)
        self.assertEqual(result.returncode, 0, output)
        link_directories = result_file.read_text(encoding="utf-8").split(";")
        expected = toolset_dir / expected_relative_path
        generic_spectre_dir = toolset_dir / "lib" / "spectre" / kwargs.get("platform", "x64").lower()
        self.assertEqual(len(link_directories), 2)
        self.assertTrue(Path(link_directories[0]).samefile(expected))
        self.assertTrue(Path(link_directories[1]).samefile(generic_spectre_dir))
        selected_lines = [
            line.partition(": ")[2]
            for line in output.splitlines()
            if line.startswith("-- MSVC OneCore runtime library directory: ")
        ]
        self.assertEqual(len(selected_lines), 1)
        self.assertTrue(Path(selected_lines[0]).samefile(expected))

    def _assert_override_skipped(self, **kwargs) -> None:
        result, result_file, _, output = self._configure(**kwargs)
        self.assertEqual(result.returncode, 0, output)
        link_directories = result_file.read_text(encoding="utf-8").split(";")
        self.assertEqual(len(link_directories), 1)
        self.assertNotIn("MSVC OneCore runtime library directory:", output)

    def test_x64_spectre_in_cxx_flags(self):
        self._assert_selected("lib/spectre/onecore/x64", cxx_flags="/Qspectre")

    def test_x64_spectre_in_c_flags(self):
        self._assert_selected("lib/spectre/onecore/x64", c_flags="/Qspectre")

    def test_arm64_spectre(self):
        self._assert_selected("lib/spectre/onecore/ARM64", platform="ARM64", cxx_flags="/Qspectre")

    def test_arm64ec_uses_arm64_hybrid_libraries(self):
        self._assert_selected("lib/spectre/onecore/ARM64", platform="ARM64EC", cxx_flags="/Qspectre")

    def test_no_spectre_flag_preserves_existing_selection(self):
        self._assert_selected("lib/onecore/x64")

    def test_disabled_spectre_flag_preserves_existing_selection(self):
        self._assert_selected("lib/onecore/x64", cxx_flags="/Qspectre-")

    def test_final_disabled_spectre_flag_preserves_existing_selection(self):
        self._assert_selected("lib/onecore/x64", cxx_flags="/Qspectre /O2 /Qspectre-")

    def test_final_enabled_spectre_flag_selects_spectre_libraries(self):
        self._assert_selected("lib/spectre/onecore/x64", cxx_flags="/Qspectre- /O2 /Qspectre")

    def test_desktop_standard_libraries_skip_override(self):
        self._assert_override_skipped(standard_libraries="kernel32.lib")

    def test_gdk_skips_override(self):
        self._assert_override_skipped(gdk=True)

    def test_cross_compiling_skips_override(self):
        self._assert_override_skipped(cross_compiling=True)

    def _assert_missing_library_fails(self, library: str) -> None:
        result, _, toolset_dir, output = self._configure(cxx_flags="/Qspectre", missing_library=library)
        expected = (toolset_dir / "lib" / "spectre" / "onecore" / "x64" / library).as_posix()
        self.assertNotEqual(result.returncode, 0, output)
        self.assertIn("Required Spectre-mitigated OneCore runtime library is missing:", output)
        emitted_paths = [Path(line.strip()) for line in output.splitlines() if line.strip().endswith(f"/{library}")]
        self.assertEqual(len(emitted_paths), 1)
        self.assertEqual(emitted_paths[0].name, library)
        self.assertTrue(emitted_paths[0].parent.samefile(Path(expected).parent))


def _add_missing_library_test(library: str) -> None:
    def test(self):
        self._assert_missing_library_fails(library)

    test.__name__ = f"test_missing_{library.replace('.', '_')}_fails"
    setattr(OneCoreSpectreTest, test.__name__, test)


for _library in _RUNTIME_LIBRARIES:
    _add_missing_library_test(_library)


if __name__ == "__main__":
    unittest.main()

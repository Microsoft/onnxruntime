#!/usr/bin/python3
import argparse
import os
import sys


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", required=True, help="input symbol file")
    parser.add_argument("--output", required=True, help="output file")
    parser.add_argument("--output_source", required=True, help="output file")
    parser.add_argument("--version_file", required=True, help="VERSION_NUMBER file")
    parser.add_argument("--style", required=True, choices=["gcc", "vc", "xcode", "aix"])
    parser.add_argument("--config", required=True, nargs="+")
    parser.add_argument(
        "--extra_symbol_file",
        action="append",
        default=[],
        help="Additional symbols to export without adding them to GetFunctionEntryByName",
    )
    return parser.parse_args()


args = parse_arguments()
print(f"Generating symbol file for {args.config!s}")
with open(args.version_file) as f:
    VERSION_STRING = f.read().strip()

print(f"VERSION:{VERSION_STRING}")

provider_symbols = set()
for c in args.config:
    file_name = os.path.join(args.src_root, "core", "providers", c, "symbols.txt")
    with open(file_name) as file:
        for line in file:
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue
            if line in provider_symbols:
                sys.exit(f"dup symbol in {file_name}: {line}")
            provider_symbols.add(line)

symbols = set(provider_symbols)
for file_name in args.extra_symbol_file:
    with open(file_name) as file:
        for line in file:
            line = line.strip()  # noqa: PLW2901
            if not line:
                continue
            if line in symbols:
                sys.exit(f"dup symbol in {file_name}: {line}")
            symbols.add(line)

symbols = sorted(symbols)

symbol_index = 1
with open(args.output, "w") as file:
    if args.style == "vc":
        file.write("LIBRARY\n")
        file.write("EXPORTS\n")
    elif args.style in ["xcode", "aix"]:
        pass  # Both xcode and AIX export files don't need a specific header.
    else:
        file.write(f"VERS_{VERSION_STRING} {{\n")
        file.write(" global:\n")

    for symbol in symbols:
        if args.style == "vc":
            file.write(f" {symbol} @{symbol_index}\n")
        elif args.style == "xcode":
            file.write(f"_{symbol}\n")
        elif args.style == "aix":
            file.write(f"{symbol}\n")  # AIX just needs the name
        else:
            file.write(f"  {symbol};\n")
        symbol_index += 1

    if args.style == "gcc":
        file.write(" local:\n")
        file.write("    *;\n")
        file.write("};   \n")

with open(args.output_source, "w") as file:
    file.write("#include <onnxruntime_c_api.h>\n")
    for c in args.config:
        # WinML adapter should not be exported in platforms other than Windows.
        # Exporting OrtGetWinMLAdapter is exported without issues using .def file when compiling for Windows
        # so it isn't necessary to include it in generated_source.c

        # external symbols are removed, xnnpack ep will be created via the standard ORT API.
        # https://github.com/microsoft/onnxruntime/pull/11798
        if c not in (
            "vitisai",
            "winml",
            "cuda",
            "migraphx",
            "qnn",
            "snpe",
            "xnnpack",
            "cann",
            "dnnl",
            "tensorrt",
            "azure",
            "webgpu",
        ):
            file.write(f"#include <core/providers/{c}/{c}_provider_factory.h>\n")
    file.write("void* GetFunctionEntryByName(const char* name){\n")
    for symbol in sorted(provider_symbols):
        if symbol != "OrtGetWinMLAdapter":
            file.write(f'if(strcmp(name,"{symbol}") ==0) return (void*)&{symbol};\n')
    file.write("return NULL;\n")
    file.write("}\n")

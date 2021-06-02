#!/usr/bin/env python
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import io
import os
import pathlib
from collections import defaultdict

import onnxruntime.capi.onnxruntime_pybind11_state as rtpy


def format_version_range(v):
    if (v[1] >= 2147483647):
        return str(v[0])+'+'
    else:
        if (v[0] == v[1]):
            return str(v[0])
        else:
            return '['+str(v[0])+', '+str(v[1])+']'


def format_type_constraints(tc):
    counter = 0
    tcstr = ''
    firsttcitem = True
    for tcitem in tc:
        counter += 1
        if firsttcitem:
            firsttcitem = False
        else:
            tcstr += ', '
        tcstr += tcitem
    return tcstr


def format_param_strings(params):
    firstparam = True
    s = ''
    if params:
        for param in sorted(params):
            if firstparam:
                firstparam = False
            else:
                s += '<br><br>or<br><br>'
            s += param
    return s


def expand_providers(provider_filter: [str]):
    providers = set()
    if provider_filter:
        for provider in provider_filter:
            p = provider.lower()
            if not p.endswith('executionprovider'):
                p += 'executionprovider'
            providers.add(p)

    return providers


def main(output_path: pathlib.Path, provider_filter: [str]):

    providers = expand_providers(provider_filter)

    with io.open(output_path, 'w', newline='', encoding="utf-8") as fout:
        fout.write('## Supported Operators and Data Types\n')
        fout.write(
            "*This file is automatically generated from the registered kernels by "
            "[this script](https://github.com/microsoft/onnxruntime/blob/master/tools/python/gen_opkernel_doc.py).\n"
            "Do not modify directly.*\n\n")
        opdef = rtpy.get_all_operator_schema()
        paramdict = {}
        for schema in opdef:
            inputs = schema.inputs
            domain = schema.domain
            if (domain == ''):
                domain = 'ai.onnx'
            fullname = domain+'.'+schema.name
            paramstr = ''
            firstinput = True
            if inputs:
                for inp in inputs:
                    if firstinput:
                        firstinput = False
                    else:
                        paramstr += '<br> '
                    paramstr += '*in* {}:**{}**'.format(inp.name, inp.typeStr)

            outputs = schema.outputs
            if outputs:
                for outp in outputs:
                    if firstinput:
                        firstinput = False
                    else:
                        paramstr += '<br> '
                    paramstr += '*out* {}:**{}**'.format(outp.name, outp.typeStr)

            paramstr += ''
            paramset = paramdict.get(fullname, None)
            if paramset is None:
                paramdict[fullname] = set()

            paramdict[fullname].add(paramstr)

        index = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for op in rtpy.get_all_opkernel_def():
            domain = op.domain
            if (domain == ''):
                domain = 'ai.onnx'
            index[op.provider][domain][op.op_name].append(op)

        # TOC
        fout.write('## Execution Providers\n\n')
        for provider in sorted(index.keys()):
            if providers and provider.lower() not in providers:
                continue
            fout.write('- [{}](#{})\n'.format(provider, provider.lower()))
        fout.write('\n---------------')

        for provider, domainmap in sorted(index.items()):
            if providers and provider.lower() not in providers:
                continue

            fout.write('\n\n<a name="{}"/>\n'.format(provider.lower()))
            fout.write('## Operators implemented by {}\n\n'.format(provider))
            fout.write('| Op Name | Parameters | OpSet Version | Types Supported |\n')
            fout.write('|---------|------------|---------------|-----------------|\n')
            for domain, namemap in sorted(domainmap.items()):
                fout.write('|**Operator Domain:** *'+domain+'*||||\n')
                for name, ops in sorted(namemap.items()):
                    version_type_index = defaultdict(lambda: defaultdict(set))
                    for op in ops:
                        for tname, tclist in op.type_constraints.items():
                            for c in tclist:
                                version_type_index[op.version_range][tname].add(c)

                    namefirsttime = True
                    for version_range, typemap in sorted(version_type_index.items(), key=lambda x: x[0], reverse=True):
                        if (namefirsttime):
                            params = paramdict.get(domain+'.'+name, None)
                            fout.write('|' + name + '|' + format_param_strings(params) + '|')
                            namefirsttime = False
                        else:
                            fout.write('|||')
                        fout.write(format_version_range(version_range) + '|')
                        tnameindex = 0
                        for tname, tcset in sorted(typemap.items()):
                            tnameindex += 1
                            tclist = []
                            for tc in sorted(tcset):
                                tclist.append(tc)
                            fout.write('**'+tname+'** = '+format_type_constraints(tclist))
                            if (tnameindex < len(typemap)):
                                fout.write('<br/> ')
                        fout.write('|\n')

                fout.write('| |\n| |\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ONNX Runtime Operator Kernel Documentation Generator')
    parser.add_argument('--providers', nargs='+',
                        help="Filter to specified execution providers. Case-insensitive. "
                             "Matches provider names from <ORT>/include/onnxruntime/core/graph/constants.h'. "
                             "'ExecutionProvider' is automatically appended as needed. "
                             "e.g. `--providers cpu cuda` will match CPUExecutionProvider and CUDAExecutionProvider.")
    parser.add_argument('--output_path', help='output markdown file path', type=pathlib.Path, required=True,
                        default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'OperatorKernels.md'))
    args = parser.parse_args()

    main(args.output_path, args.providers)

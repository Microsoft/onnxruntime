# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from setuptools import setup, Extension
from distutils import log as logger
from distutils.command.build_ext import build_ext as _build_ext
from glob import glob
from os import path, getcwd, environ, remove, listdir
from shutil import copyfile, copytree, rmtree
import platform
import subprocess
import sys
import datetime

nightly_build = False
featurizers_build = False
package_name = 'onnxruntime'
wheel_name_suffix = None


def parse_arg_remove_boolean(argv, arg_name):
    arg_value = False
    if arg_name in sys.argv:
        arg_value = True
        argv.remove(arg_name)

    return arg_value


def parse_arg_remove_string(argv, arg_name_equal):
    arg_value = None
    for arg in sys.argv[1:]:
        if arg.startswith(arg_name_equal):
            arg_value = arg[len(arg_name_equal):]
            sys.argv.remove(arg)
            break

    return arg_value


# Any combination of the following arguments can be applied
featurizers_build = parse_arg_remove_boolean(sys.argv, '--use_featurizers')

if parse_arg_remove_boolean(sys.argv, '--nightly_build'):
    package_name = 'ort-nightly'
    nightly_build = True

wheel_name_suffix = parse_arg_remove_string(sys.argv, '--wheel_name_suffix=')

cuda_version = None
rocm_version = None
# The following arguments are mutually exclusive
if parse_arg_remove_boolean(sys.argv, '--use_tensorrt'):
    package_name = 'onnxruntime-gpu-tensorrt' if not nightly_build else 'ort-trt-nightly'
elif wheel_name_suffix == 'gpu':
    #TODO: how to support multiple CUDA versions?
    cuda_version = parse_arg_remove_string(sys.argv, '--cuda_version=')
elif parse_arg_remove_boolean(sys.argv, '--use_rocm'):
    package_name = 'onnxruntime-rocm' if not nightly_build else 'ort-rocm-nightly'
    rocm_version = parse_arg_remove_string(sys.argv, '--rocm_version=')
elif parse_arg_remove_boolean(sys.argv, '--use_openvino'):
    package_name = 'onnxruntime-openvino'
elif parse_arg_remove_boolean(sys.argv, '--use_dnnl'):
    package_name = 'onnxruntime-dnnl'
elif parse_arg_remove_boolean(sys.argv, '--use_nuphar'):
    package_name = 'onnxruntime-nuphar'
elif parse_arg_remove_boolean(sys.argv, '--use_vitisai'):
    package_name = 'onnxruntime-vitisai'
elif parse_arg_remove_boolean(sys.argv, '--use_acl'):
    package_name = 'onnxruntime-acl'
elif parse_arg_remove_boolean(sys.argv, '--use_armnn'):
    package_name = 'onnxruntime-armnn'


# PEP 513 defined manylinux1_x86_64 and manylinux1_i686
# PEP 571 defined manylinux2010_x86_64 and manylinux2010_i686
# PEP 599 defines the following platform tags:
# manylinux2014_x86_64
# manylinux2014_i686
# manylinux2014_aarch64
# manylinux2014_armv7l
# manylinux2014_ppc64
# manylinux2014_ppc64le
# manylinux2014_s390x
manylinux_tags = [
    'manylinux1_x86_64',
    'manylinux1_i686',
    'manylinux2010_x86_64',
    'manylinux2010_i686',
    'manylinux2014_x86_64',
    'manylinux2014_i686',
    'manylinux2014_aarch64',
    'manylinux2014_armv7l',
    'manylinux2014_ppc64',
    'manylinux2014_ppc64le',
    'manylinux2014_s390x',
]
is_manylinux = environ.get('AUDITWHEEL_PLAT', None) in manylinux_tags

class build_ext(_build_ext):
    def build_extension(self, ext):
        dest_file = self.get_ext_fullpath(ext.name)
        logger.info('copying %s -> %s', ext.sources[0], dest_file)
        copyfile(ext.sources[0], dest_file)


try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

    class bdist_wheel(_bdist_wheel):
        def finalize_options(self):
            _bdist_wheel.finalize_options(self)
            if not is_manylinux:
                self.root_is_pure = False

        def _rewrite_ld_preload(self, to_preload):
            with open('onnxruntime/capi/_ld_preload.py', 'rt') as f:
                ld_preload = f.read().splitlines()
            with open('onnxruntime/capi/_ld_preload.py', 'wt') as f:
                for line in ld_preload:
                    f.write(line)
                    f.write('\n')
                    if 'LD_PRELOAD_BEGIN_MARK' in line:
                        break
                if len(to_preload) > 0:
                    f.write('from ctypes import CDLL, RTLD_GLOBAL\n')
                    for library in to_preload:
                        f.write('_{} = CDLL("{}", mode=RTLD_GLOBAL)\n'.format(library.split('.')[0], library))

        def run(self):
            if is_manylinux:
                source = 'onnxruntime/capi/onnxruntime_pybind11_state.so'
                dest = 'onnxruntime/capi/onnxruntime_pybind11_state_manylinux1.so'
                logger.info('copying %s -> %s', source, dest)
                copyfile(source, dest)
                result = subprocess.run(['patchelf', '--print-needed', dest], check=True, stdout=subprocess.PIPE, universal_newlines=True)
                dependencies = ['librccl.so', 'libamdhip64.so', 'librocblas.so', 'libMIOpen.so', 'libhsa-runtime64.so', 'libhsakmt.so']
                to_preload = []
                args = ['patchelf', '--debug']
                for line in result.stdout.split('\n'):
                    for dependency in dependencies:
                        if dependency in line:
                            to_preload.append(line)
                            args.extend(['--remove-needed', line])
                args.append(dest)
                if len(args) > 3:
                    subprocess.run(args, check=True, stdout=subprocess.PIPE)

                dest = 'onnxruntime/capi/libonnxruntime_providers_cuda.so'
                if path.isfile(dest):
                    result = subprocess.run(['patchelf', '--print-needed', dest], check=True, stdout=subprocess.PIPE, universal_newlines=True)
                    cuda_dependencies = ['libcublas.so', 'libcublasLt.so', 'libcudnn.so', 'libcudart.so', 'libcurand.so', 'libcufft.so', 'libnvToolsExt.so']
                    args = ['patchelf', '--debug']
                    for line in result.stdout.split('\n'):
                        for dependency in cuda_dependencies:
                            if dependency in line:
                                if not dependency in to_preload:
                                    to_preload.append(line)
                                args.extend(['--remove-needed', line])
                    args.append(dest)
                    if len(args) > 3:
                        subprocess.run(args, check=True, stdout=subprocess.PIPE)

                self._rewrite_ld_preload(to_preload)
            _bdist_wheel.run(self)
            if is_manylinux:
                file = glob(path.join(self.dist_dir, '*linux*.whl'))[0]
                logger.info('repairing %s for manylinux1', file)
                try:
                    subprocess.run(['auditwheel', 'repair', '-w', self.dist_dir, file], check=True, stdout=subprocess.PIPE)
                finally:
                    logger.info('removing %s', file)
                    remove(file)

except ImportError as error:
    print("Error importing dependencies:")
    print(error)
    bdist_wheel = None

# Additional binaries
if platform.system() == 'Linux':
  libs = ['onnxruntime_pybind11_state.so', 'libdnnl.so.2', 'libmklml_intel.so', 'libmklml_gnu.so', 'libiomp5.so', 'mimalloc.so']
  dl_libs = ['libonnxruntime_providers_shared.so', 'libonnxruntime_providers_cuda.so']
  # DNNL, TensorRT & OpenVINO EPs are built as shared libs
  libs.extend(['libonnxruntime_providers_shared.so'])
  libs.extend(['libonnxruntime_providers_dnnl.so'])
  libs.extend(['libonnxruntime_providers_tensorrt.so'])
  libs.extend(['libonnxruntime_providers_openvino.so'])
  libs.extend(['libonnxruntime_providers_cuda.so'])
  # Nuphar Libs
  libs.extend(['libtvm.so.0.5.1'])
  if nightly_build:
    libs.extend(['libonnxruntime_pywrapper.so'])
elif platform.system() == "Darwin":
  libs = ['onnxruntime_pybind11_state.so', 'libdnnl.2.dylib', 'mimalloc.so'] # TODO add libmklml and libiomp5 later.
  # DNNL & TensorRT EPs are built as shared libs
  libs.extend(['libonnxruntime_providers_shared.dylib'])
  libs.extend(['libonnxruntime_providers_dnnl.dylib'])
  libs.extend(['libonnxruntime_providers_tensorrt.dylib'])
  libs.extend(['libonnxruntime_providers_cuda.dylib'])
  if nightly_build:
    libs.extend(['libonnxruntime_pywrapper.dylib'])
else:
  libs = ['onnxruntime_pybind11_state.pyd', 'dnnl.dll', 'mklml.dll', 'libiomp5md.dll']
  # DNNL, TensorRT & OpenVINO EPs are built as shared libs
  libs.extend(['onnxruntime_providers_shared.dll'])
  libs.extend(['onnxruntime_providers_dnnl.dll'])
  libs.extend(['onnxruntime_providers_tensorrt.dll'])
  libs.extend(['onnxruntime_providers_openvino.dll'])
  libs.extend(['onnxruntime_providers_cuda.dll'])
  # DirectML Libs
  libs.extend(['DirectML.dll'])
  # Nuphar Libs
  libs.extend(['tvm.dll'])
  if nightly_build:
    libs.extend(['onnxruntime_pywrapper.dll'])

if is_manylinux:
    data = ['capi/libonnxruntime_pywrapper.so'] if nightly_build else []
    data += [path.join('capi', x) for x in dl_libs if path.isfile(path.join('onnxruntime', 'capi', x))]
    ext_modules = [
        Extension(
            'onnxruntime.capi.onnxruntime_pybind11_state',
            ['onnxruntime/capi/onnxruntime_pybind11_state_manylinux1.so'],
        ),
    ]
else:
    data = [path.join('capi', x) for x in libs if path.isfile(path.join('onnxruntime', 'capi', x))]
    ext_modules = []

# Additional examples
examples_names = ["mul_1.onnx", "logreg_iris.onnx", "sigmoid.onnx"]
examples = [path.join('datasets', x) for x in examples_names]

# Extra files such as EULA and ThirdPartyNotices
extra = ["LICENSE", "ThirdPartyNotices.txt", "Privacy.md"]

# Description
README = path.join(getcwd(), "docs/python/README.rst")
if not path.exists(README):
    this = path.dirname(__file__)
    README = path.join(this, "docs/python/README.rst")
if not path.exists(README):
    raise FileNotFoundError("Unable to find 'README.rst'")
with open(README) as f:
    long_description = f.read()

packages = [
    'onnxruntime',
    'onnxruntime.backend',
    'onnxruntime.capi',
    'onnxruntime.capi.training',
    'onnxruntime.datasets',
    'onnxruntime.tools',
    'onnxruntime.tools.ort_format_model',
    'onnxruntime.tools.ort_format_model.ort_flatbuffers_py',
    'onnxruntime.tools.ort_format_model.ort_flatbuffers_py.experimental',
    'onnxruntime.tools.ort_format_model.ort_flatbuffers_py.experimental.fbs',
    'onnxruntime.quantization',
    'onnxruntime.quantization.operators',
    'onnxruntime.quantization.CalTableFlatBuffers',
    'onnxruntime.transformers',
    'onnxruntime.transformers.longformer',
]

requirements_file = "requirements.txt"

local_version = None
enable_training = parse_arg_remove_boolean(sys.argv, '--enable_training')
default_training_package_device = parse_arg_remove_boolean(sys.argv, '--default_training_package_device')

package_data = {}
data_files = []

if enable_training:
    packages.extend(['onnxruntime.training',
                     'onnxruntime.training.amp',
                     'onnxruntime.training.optim',
                     'onnxruntime.training.ortmodule',
                     'onnxruntime.training.ortmodule.torch_cpp_extensions',
                     'onnxruntime.training.ortmodule.torch_cpp_extensions.aten_op_executor',
                     'onnxruntime.training.ortmodule.torch_cpp_extensions.torch_gpu_allocator'])
    package_data['onnxruntime.training.ortmodule.torch_cpp_extensions.aten_op_executor'] = ['*.cc']
    package_data['onnxruntime.training.ortmodule.torch_cpp_extensions.torch_gpu_allocator'] = ['*.cc']
    requirements_file = "requirements-training.txt"
    # with training, we want to follow this naming convention:
    # stable:
    # onnxruntime-training-1.7.0+cu111-cp36-cp36m-linux_x86_64.whl
    # nightly:
    # onnxruntime-training-1.7.0.dev20210408+cu111-cp36-cp36m-linux_x86_64.whl
    # this is needed immediately by pytorch/ort so that the user is able to
    # install an onnxruntime training package with matching torch cuda version.
    package_name = 'onnxruntime-training'

    # we want put default training packages to pypi. pypi does not accept package with a local version.
    if (not default_training_package_device) or nightly_build:
        def get_torch_version():
            try:
                import torch
                torch_version = torch.__version__
                torch_version_plus_pos = torch_version.find('+')
                if torch_version_plus_pos != -1:
                    torch_version = torch_version[:torch_version_plus_pos]
                torch_version = torch_version.replace('.', '')
                return torch_version
            except ImportError as error:
                print("Error importing torch to get torch version:")
                print(error)
                return None

        torch_version = get_torch_version()
        if cuda_version:
            # removing '.' to make local Cuda version number in the same form as Pytorch.
            if torch_version:
                local_version = '+torch' + torch_version + '.'\
                    + 'cu' + cuda_version.replace('.', '')
            else:
                local_version = '+cu' + cuda_version.replace('.', '')
        if rocm_version:
            # removing '.' to make Cuda version number in the same form as Pytorch.
            rocm_version = rocm_version.replace('.', '')
            if torch_version:
                local_version = '+torch' + torch_version + '.'\
                    + 'rocm' + rocm_version
            else:
                local_version = '+rocm' + rocm_version

if package_name == 'onnxruntime-nuphar':
    packages += ["onnxruntime.nuphar"]
    extra += [path.join('nuphar', 'NUPHAR_CACHE_VERSION')]

if featurizers_build:
    # Copy the featurizer data from its current directory into the onnx runtime directory so that the
    # content can be included as module data.

    # Apparently, the root_dir is different based on how the script is invoked
    source_root_dir = None
    dest_root_dir = None

    for potential_source_prefix, potential_dest_prefix in [
        (getcwd(), getcwd()),
        (path.dirname(__file__), path.dirname(__file__)),
        (path.join(getcwd(), ".."), getcwd()),
    ]:
        potential_dir = path.join(potential_source_prefix, "external", "FeaturizersLibrary", "Data")
        if path.isdir(potential_dir):
            source_root_dir = potential_source_prefix
            dest_root_dir = potential_dest_prefix

            break

    if source_root_dir is None:
        raise Exception("Unable to find the build root dir")

    assert dest_root_dir is not None

    featurizer_source_dir = path.join(source_root_dir, "external", "FeaturizersLibrary", "Data")
    assert path.isdir(featurizer_source_dir), featurizer_source_dir

    featurizer_dest_dir = path.join(dest_root_dir, "onnxruntime", "FeaturizersLibrary", "Data")
    if path.isdir(featurizer_dest_dir):
        rmtree(featurizer_dest_dir)

    for item in listdir(featurizer_source_dir):
        this_featurizer_source_fullpath = path.join(featurizer_source_dir)
        assert path.isdir(this_featurizer_source_fullpath), this_featurizer_source_fullpath

        copytree(this_featurizer_source_fullpath, featurizer_dest_dir)

        packages.append("onnxruntime.FeaturizersLibrary.Data.{}".format(item))
        package_data[packages[-1]] = listdir(path.join(featurizer_dest_dir, item))

package_data["onnxruntime"] = data + examples + extra

version_number = ''
with open('VERSION_NUMBER') as f:
    version_number = f.readline().strip()
if nightly_build:
    # https://docs.microsoft.com/en-us/azure/devops/pipelines/build/variables
    build_suffix = environ.get('BUILD_BUILDNUMBER')
    if build_suffix is None:
        # The following line is only for local testing
        build_suffix = str(datetime.datetime.now().date().strftime("%Y%m%d"))
        print("build_suffix1: ", build_suffix)
    else:
        print("build_suffix2: ", build_suffix)
        build_suffix = build_suffix.replace('.', '')

    if len(build_suffix) > 8 and len(build_suffix) < 12:
        # we want to format the build_suffix to avoid (the 12th run on 20210630 vs the first run on 20210701):
        # 2021063012 > 20210701
        # in above 2021063012 is treated as the latest which is incorrect.
        # we want to convert the format to:
        # 20210630012 < 20210701001
        # where the first 8 digits are date. the last 3 digits are run count.
        # as long as there are less than 1000 runs per day, we will not have the problem.
        def check_date_format(date_str):
            try:
                datetime.datetime.strptime(date_str, '%Y%m%d')
                return True
            except: # noqa
                return False

        def reformat_run_count(count_str):
            try:
                count = int(count_str)
                if count >= 0 and count < 1000:
                    return "{:03}".format(count)
                return ""
            except: # noqa
                return ""

        build_suffix_is_date_format = check_date_format(build_suffix[:8])
        build_suffix_run_count = reformat_run_count(build_suffix[8:])
        if build_suffix_is_date_format and build_suffix_run_count:
            build_suffix = build_suffix[:8] + build_suffix_run_count

    if enable_training:
        from packaging import version
        from packaging.version import Version
        # with training package, we need to bump up version minor number so that
        # nightly releases take precedence over the latest release when --pre is used during pip install.
        # eventually this shall be the behavior of all onnxruntime releases.
        # alternatively we may bump up version number right after every release.
        ort_version = version.parse(version_number)
        if isinstance(ort_version, Version):
            # TODO: this is the last time we have to do this!!!
            # We shall bump up release number right after release cut.
            if ort_version.major == 1 and ort_version.minor == 8 and ort_version.micro == 0:
                version_number = '{major}.{minor}.{macro}'.format(
                    major=ort_version.major,
                    minor=ort_version.minor + 1,
                    macro=ort_version.micro)

    version_number = version_number + ".dev" + build_suffix

if local_version:
    version_number = version_number + local_version

if wheel_name_suffix:
    if not (enable_training and wheel_name_suffix == 'gpu'):
        # for training packages, local version is used to indicate device types
        package_name = "{}-{}".format(package_name, wheel_name_suffix)

cmd_classes = {}
if bdist_wheel is not None:
    cmd_classes['bdist_wheel'] = bdist_wheel
cmd_classes['build_ext'] = build_ext

requirements_path = path.join(getcwd(), requirements_file)
if not path.exists(requirements_path):
    this = path.dirname(__file__)
    requirements_path = path.join(this, requirements_file)
if not path.exists(requirements_path):
    raise FileNotFoundError("Unable to find " + requirements_file)
with open(requirements_path) as f:
    install_requires = f.read().splitlines()


if enable_training:
    def save_build_and_package_info(package_name, version_number, cuda_version, rocm_version):
        sys.path.append(path.join(path.dirname(__file__), 'onnxruntime', 'python'))
        from onnxruntime_collect_build_info import find_cudart_versions

        version_path = path.join('onnxruntime', 'capi', 'build_and_package_info.py')
        with open(version_path, 'w') as f:
            f.write("package_name = '{}'\n".format(package_name))
            f.write("__version__ = '{}'\n".format(version_number))

            if cuda_version:
                f.write("cuda_version = '{}'\n".format(cuda_version))

                # cudart_versions are integers
                cudart_versions = find_cudart_versions(build_env=True)
                if cudart_versions and len(cudart_versions) == 1:
                    f.write("cudart_version = {}\n".format(cudart_versions[0]))
                else:
                    print(
                        "Error getting cudart version. ",
                        "did not find any cudart library"
                        if not cudart_versions or len(cudart_versions) == 0
                        else "found multiple cudart libraries")
            elif rocm_version:
                f.write("rocm_version = '{}'\n".format(rocm_version))

    save_build_and_package_info(package_name, version_number, cuda_version, rocm_version)

# Setup
setup(
    name=package_name,
    version=version_number,
    description='ONNX Runtime is a runtime accelerator for Machine Learning models',
    long_description=long_description,
    author='Microsoft Corporation',
    author_email='onnxruntime@microsoft.com',
    cmdclass=cmd_classes,
    license="MIT License",
    packages=packages,
    ext_modules=ext_modules,
    package_data=package_data,
    url="https://onnxruntime.ai",
    download_url='https://github.com/microsoft/onnxruntime/tags',
    data_files=data_files,
    install_requires=install_requires,
    keywords='onnx machine learning',
    entry_points= {
        'console_scripts': [
            'onnxruntime_test = onnxruntime.tools.onnxruntime_test:main',
        ]
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'],
    )

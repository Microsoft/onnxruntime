# ------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------

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

# Any combination of the following arguments can be applied
if '--use_featurizers' in sys.argv:
    featurizers_build = True
    sys.argv.remove('--use_featurizers')

if '--nightly_build' in sys.argv:
    package_name = 'ort-nightly'
    nightly_build = True
    sys.argv.remove('--nightly_build')

for arg in sys.argv[1:]:
    if arg.startswith("--wheel_name_suffix="):
        wheel_name_suffix = arg[len("--wheel_name_suffix="):]

        sys.argv.remove(arg)

        break

# The following arguments are mutually exclusive
if '--use_tensorrt' in sys.argv:
    package_name = 'onnxruntime-gpu-tensorrt' if not nightly_build else 'ort-trt-nightly'
    sys.argv.remove('--use_tensorrt')
elif '--use_cuda' in sys.argv:
    package_name = 'onnxruntime-gpu' if not nightly_build else 'ort-gpu-nightly'
    sys.argv.remove('--use_cuda')
elif '--use_openvino' in sys.argv:
    package_name = 'onnxruntime-openvino'
    sys.argv.remove('--use_openvino')
elif '--use_dnnl' in sys.argv:
    package_name = 'onnxruntime-dnnl'
    sys.argv.remove('--use_dnnl')
elif '--use_nuphar' in sys.argv:
    package_name = 'onnxruntime-nuphar'
    sys.argv.remove('--use_nuphar')
elif '--use_vitisai' in sys.argv:
    package_name = 'onnxruntime-vitisai'
    sys.argv.remove('--use_vitisai')
elif '--use_acl' in sys.argv:
    package_name = 'onnxruntime-acl'
    sys.argv.remove('--use_acl')
elif '--use_armnn' in sys.argv:
    package_name = 'onnxruntime-armnn'
    sys.argv.remove('--use_armnn')
elif '--use_dml' in sys.argv:
    package_name = 'onnxruntime-dml'
    sys.argv.remove('--use_dml')

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
                result = subprocess.run(
                    ['patchelf', '--print-needed', dest], check=True, stdout=subprocess.PIPE, universal_newlines=True)
                cuda_dependencies = [
                    'libcublas.so', 'libcudnn.so', 'libcudart.so', 'libcurand.so', 'libcufft.so', 'libnvToolsExt.so']
                to_preload = []
                args = ['patchelf', '--debug']
                for line in result.stdout.split('\n'):
                    for dependency in cuda_dependencies:
                        if dependency in line:
                            to_preload.append(line)
                            args.extend(['--remove-needed', line])
                args.append(dest)
                if len(to_preload) > 0:
                    subprocess.run(args, check=True, stdout=subprocess.PIPE)
                self._rewrite_ld_preload(to_preload)
            _bdist_wheel.run(self)
            if is_manylinux:
                file = glob(path.join(self.dist_dir, '*linux*.whl'))[0]
                logger.info('repairing %s for manylinux1', file)
                try:
                    subprocess.run(
                        ['auditwheel', 'repair', '-w', self.dist_dir, file], check=True, stdout=subprocess.PIPE)
                finally:
                    logger.info('removing %s', file)
                    remove(file)

except ImportError as error:
    print("Error importing dependencies:")
    print(error)
    bdist_wheel = None

# Additional binaries
if platform.system() == 'Linux':
    libs = ['onnxruntime_pybind11_state.so',
            'libdnnl.so.1',
            'libmklml_intel.so',
            'libmklml_gnu.so',
            'libiomp5.so',
            'mimalloc.so']
    # DNNL, TensorRT & OpenVINO EPs are built as shared libs
    libs.extend(['libonnxruntime_providers_shared.so'])
    libs.extend(['libonnxruntime_providers_dnnl.so'])
    libs.extend(['libonnxruntime_providers_tensorrt.so'])
    libs.extend(['libonnxruntime_providers_openvino.so'])
    # Nuphar Libs
    libs.extend(['libtvm.so.0.5.1'])
    if nightly_build:
        libs.extend(['libonnxruntime_pywrapper.so'])
elif platform.system() == "Darwin":
    libs = ['onnxruntime_pybind11_state.so', 'libdnnl.1.dylib', 'mimalloc.so']  # TODO add libmklml and libiomp5 later.
    # DNNL & TensorRT EPs are built as shared libs
    libs.extend(['libonnxruntime_providers_shared.dylib'])
    libs.extend(['libonnxruntime_providers_dnnl.dylib'])
    libs.extend(['libonnxruntime_providers_tensorrt.dylib'])
    if nightly_build:
        libs.extend(['libonnxruntime_pywrapper.dylib'])
else:
    libs = ['onnxruntime_pybind11_state.pyd', 'dnnl.dll', 'mklml.dll', 'libiomp5md.dll']
    # DNNL, TensorRT & OpenVINO EPs are built as shared libs
    libs.extend(['onnxruntime_providers_shared.dll'])
    libs.extend(['onnxruntime_providers_dnnl.dll'])
    libs.extend(['onnxruntime_providers_tensorrt.dll'])
    libs.extend(['onnxruntime_providers_openvino.dll'])
    # DirectML Libs
    libs.extend(['directml.dll'])
    # Nuphar Libs
    libs.extend(['tvm.dll'])
    if nightly_build:
        libs.extend(['onnxruntime_pywrapper.dll'])

if is_manylinux:
    data = ['capi/libonnxruntime_pywrapper.so'] if nightly_build else []
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
    'onnxruntime.quantization',
    'onnxruntime.quantization.operators',
    'onnxruntime.quantization.CalTableFlatBuffers',
    'onnxruntime.transformers',
    'onnxruntime.transformers.longformer',
]

requirements_file = "requirements.txt"

if '--enable_training' in sys.argv:
    packages.extend(['onnxruntime.training',
                     'onnxruntime.training.amp',
                     'onnxruntime.training.optim'])
    sys.argv.remove('--enable_training')
    requirements_file = "requirements-training.txt"

package_data = {}
data_files = []

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
    else:
        build_suffix = build_suffix.replace('.', '')

    version_number = version_number + ".dev" + build_suffix

if wheel_name_suffix:
    package_name = "{}_{}".format(package_name, wheel_name_suffix)

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
    entry_points={
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

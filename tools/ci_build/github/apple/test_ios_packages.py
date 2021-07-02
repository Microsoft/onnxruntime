#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import pathlib
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))


def _test_ios_packages(args):
    # check if CocoaPods is installed
    if shutil.which('pod') is None:
        if args.fail_if_cocoapods_missing:
            raise ValueError('CocoaPods is required for this test')
        else:
            print('CocoaPods is not installed, ignore this test')
            return

    # Now we need to create a zip file contains the framework and the podspec file, both of these 2 files
    # should be under the c_framework_dir
    c_framework_dir = args.c_framework_dir.resolve()
    if not c_framework_dir.is_dir():
        raise FileNotFoundError('c_framework_dir {} is not a folder.'.format(c_framework_dir))

    has_framework = pathlib.Path(os.path.join(c_framework_dir, 'onnxruntime.framework')).exists()
    has_xcframework = pathlib.Path(os.path.join(c_framework_dir, 'onnxruntime.xcframework')).exists()

    if not has_framework and not has_xcframework:
        raise FileNotFoundError('{} does not have onnxruntime.framework/xcframework'.format(c_framework_dir))

    if has_framework and has_xcframework:
        raise ValueError('Cannot proceed when both onnxruntime.framework '
                         'and onnxruntime.xcframework exist')

    framework_name = 'onnxruntime.framework' if has_framework else 'onnxruntime.xcframework'

    # create a temp folder
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # create a zip file contains the framework
        # TODO, move this into a util function
        local_pods_dir = os.path.join(temp_dir, 'local_pods')
        os.makedirs(local_pods_dir, exist_ok=True)
        # shutil.make_archive require target file as full path without extension
        zip_base_filename = os.path.join(local_pods_dir, 'onnxruntime-mobile')
        zip_file_path = zip_base_filename + '.zip'
        shutil.make_archive(zip_base_filename, 'zip', root_dir=c_framework_dir, base_dir=framework_name)

        # copy the test project to the temp_dir
        test_proj_path = os.path.join(REPO_DIR, 'onnxruntime', 'test', 'platform', 'ios', 'ios_package_test')
        target_proj_path = os.path.join(temp_dir, 'ios_package_test')
        shutil.copytree(test_proj_path, target_proj_path)

        # update the podspec to point to the local framework zip file
        local_podspec_path = os.path.join(target_proj_path, 'onnxruntime-mobile.podspec')
        local_podspec_template = os.path.join(target_proj_path, 'onnxruntime-mobile.podspec.template')
        with open(local_podspec_template, 'r') as file:
            file_data = file.read()

        # replace the target strings
        file_data = file_data.replace('${ORT_BASE_FRAMEWORK_ARCHIVE}', 'file:' + zip_file_path)
        file_data = file_data.replace('${ORT_FRAMEWORK_NAME}', framework_name)
        with open(os.path.join(REPO_DIR, 'VERSION_NUMBER')) as version_file:
            file_data = file_data.replace('${ORT_VERSION}', version_file.readline().strip())

        # write the updated podspec
        with open(local_podspec_path, 'w') as file:
            file.write(file_data)

        # install pods first
        subprocess.run(['pod', 'install'], shell=False, check=True, cwd=target_proj_path)

        # run the tests
        subprocess.run(['xcrun', 'xcodebuild', 'test',
                        '-workspace', './ios_package_test.xcworkspace',
                        '-scheme', 'ios_package_test',
                        '-destination', 'platform=iOS Simulator,OS=latest,name=iPhone SE (2nd generation)'],
                       shell=False, check=True, cwd=target_proj_path)


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description='Test iOS framework using CocoaPods package.'
    )

    parser.add_argument('--fail_if_cocoapods_missing', action='store_true',
                        help='This script will fail if CocoaPods is not installed, '
                        'will not throw error unless fail_if_cocoapod_missing is set.')

    parser.add_argument('--c_framework_dir', type=pathlib.Path, required=True,
                        help='Provide the parent directory for C/C++ framework')

    return parser.parse_args()


def main():
    args = parse_args()
    _test_ios_packages(args)


if __name__ == '__main__':
    main()

#!/bin/bash

# This script will run a baseline minimal ort build for android arm64-v8a ABI
# and report the binary size to the ort mysql DB

set -e
set -x
export PATH=/opt/python/cp37-cp37m/bin:$PATH
# Create an empty file to be used with build --include_ops_by_config, which will include no operators at all
echo -n > /home/onnxruntimedev/.test_data/include_no_operators.config

# Run a baseline minimal build of ORT Android arm64-v8a
# Generate binary size as /build/MinSizeRel/binary_size_data.txt
python3 /onnxruntime_src/tools/ci_build/build.py \
    --build_dir /build --cmake_generator Ninja \
    --config MinSizeRel \
    --skip_submodule_sync \
    --parallel \
    --android \
    --android_sdk_path /android_home \
    --android_ndk_path /ndk_home \
    --android_abi=arm64-v8a \
    --android_api=29 \
    --minimal_build \
    --build_shared_lib \
    --build_java \
    --disable_ml_ops \
    --disable_exceptions \
    --include_ops_by_config /home/onnxruntimedev/.test_data/include_no_operators.config

# set current size limit to 1165KB.
python3 /onnxruntime_src/tools/ci_build/github/linux/ort_minimal/check_build_binary_size.py \
    --threshold=1185000 \
    /build/MinSizeRel/libonnxruntime.so

# Post the binary size info to ort mysql DB
# The report script's DB connection failure will not fail the pipeline
# To reduce noise, we only report binary size for Continuous integration (a merge to master)
if [[ $BUILD_REASON == "IndividualCI" || $BUILD_REASON == "BatchedCI" ]] && [[ $BUILD_BRANCH == "refs/heads/master" ]]; then
    # Install the mysql connector
    python3 -m pip install --user mysql-connector-python

    python3 /onnxruntime_src/tools/ci_build/github/windows/post_binary_sizes_to_dashboard.py \
        --ignore_db_error \
        --commit_hash=$BUILD_SOURCEVERSION \
        --size_data_file=/build/MinSizeRel/binary_size_data.txt \
        --build_project=onnxruntime \
        --build_id=$BUILD_ID
else
    echo "No binary size report for build reason: [$BUILD_REASON] and build branch: [$BUILD_BRANCH]"
    echo "The content of binary_size_data.txt"
    cat /build/MinSizeRel/binary_size_data.txt
fi

# Clear the build
rm -rf /build/MinSizeRel

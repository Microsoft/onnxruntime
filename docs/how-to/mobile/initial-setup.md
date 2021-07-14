---
title: Initial setup
parent: Deploy ONNX Runtime Mobile
grand_parent: How to
has_children: false
nav_order: 2
---

{::options toc_levels="2..3" /}

## Contents
{: .no_toc}

* TOC
{:toc}

## Initial setup if using a pre-built package

### Android

##### Java/Kotlin

In your Android Studio Project, make the following changes to:

1. build.gradle (Project):
    ```
    repositories {
        mavenCentral()
    }
    ```

2. build.gradle (Module):
    ```
    dependencies {
        implementation 'com.microsoft.onnxruntime:onnxruntime-mobile:<onnxruntime mobile version>'
    }
    ```

##### C/C++

Download the onnxruntime-mobile AAR hosted at [MavenCentral](https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile), change the file extension from `.aar` to `.zip`, and unzip it. Include the header files from the `headers` folder, and the relevant `libonnxruntime.so` dynamic library from the `jni` folder in your NDK project.


### iOS

In your CocoaPods `Podfile`, add the `onnxruntime-mobile-c` or `onnxruntime-mobile-objc` pod depending on which API you wish to use.

##### C/C++

  ```
  use_frameworks!

  pod 'onnxruntime-mobile-c'
  ```

##### Objective-C

  ```
  use_frameworks!

  pod 'onnxruntime-mobile-objc'
  ```

Run `pod install`.

### Install ONNX Runtime python package

Install the onnxruntime python package from [https://pypi.org/project/onnxruntime/](https://pypi.org/project/onnxruntime/) in order to convert models from ONNX format to the internal ORT format.
Version v1.8 or higher is required.

- `pip install onnxruntime` will install the latest release

## Initial setup if performing a custom build

### Clone ONNX Runtime repository

Use git to clone the ONNX Runtime repository
  - `git clone --recursive https://github.com/Microsoft/onnxruntime`
    - this will create an 'onnxruntime' directory with the repository contents
  - See the [Build for inferencing](../build/inferencing) documentation for further details on supported environments. Ignore the build instructions on that page as they are for a full build and we will cover the mobile build instructions here.

Select the branch you wish to use. The latest release is recommended.
  - `git checkout <branch>`
    - e.g. `git checkout rel-1.8.0`

It is suggested you do not use the unreleased 'master' branch unless there is a specific new feature you require.

| Release | Date | Branch |
|---------|--------|
| 1.8 | 2021-06-02 | rel-1.8.0 |
| 1.7 | 2021-03-03 | rel-1.7.2 |
| 1.6 | 2020-12-11 | rel-1.6.0 |
| 1.5 | 2020-10-30 | rel-1.5.3 |
| Unreleased | | master |

The directory the ONNX Runtime repository was cloned into is referred to as `<ONNX Runtime repository root>` in this documentation.

### Install ONNX Runtime python package

Install the onnxruntime python package from [https://pypi.org/project/onnxruntime/](https://pypi.org/project/onnxruntime/) in order to convert models from ONNX format to the internal ORT format. Version 1.5.3 or higher is required.

- `pip install onnxruntime` will install the latest release

You must match the python package version to the branch of the ONNX Runtime repository you checked out
  - e.g. if you wanted to use the 1.7 release
    - `git checkout rel-1.7.2` in your local git repository
    - `pip install onnxruntime==1.7.2`

If you are using the `master` branch in the git repository you should use the nightly ONNX Runtime python package
  - `pip install -U -i https://test.pypi.org/simple/ ort-nightly`


-------

Next: [Converting ONNX models to ORT format](model-conversion)
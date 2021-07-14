# ONNX Runtime coding conventions and standards


## C++ Code Style

Google style from https://google.github.io/styleguide/cppguide.html with a few minor alterations:

* Max line length 120
  *	Aim for 80, but up to 120 is fine.
* Exceptions
  *	Allowed to throw fatal errors that are expected to result in a top level handler catching them, logging them and terminating the program.
* Non-const references
  *	Allowed
  * Use a non-const reference for arguments that are modifiable but cannot be nullptr so the API clearly advertises the intent
  *	Const correctness and usage of smart pointers (shared_ptr and unique_ptr) is expected. A non-const reference equates to “this is a non-null object that you can change but are not being given ownership of”.
* 'using namespace' permitted with limited scope
  * Not allowing 'using namespace' at all is overly restrictive. Follow the C++ Core Guidelines:
    * [SF.6: Use using namespace directives for transition, for foundation libraries (such as std), or within a local scope (only)](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using)
    * [SF.7: Don't write using namespace at global scope in a header file](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive)

Other
* Qualify usages of 'auto' with 'const', '*', '&' and '&&' where applicable to more clearly express the intent
* When adding a new class, disable copy/assignment/move until you have a proven need for these capabilities. If a need arises, enable copy/assignment/move selectively, and when doing so validate that the implementation of the class supports what is being enabled.
  * Use ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE initially
  * See the other ORT_DISALLOW_* macros in https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/common.h
* Don't use else after return. see: [https://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return](https://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return)
* Don't overuse std::shared\_ptr. Use std::shared\_ptr only if it's not clear when and where the object will be deallocated. See also: [https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rf-shared_ptr](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rf-shared_ptr)
* Avoid using the 'long' type, which could be either 32 bits or 64 bits.
* If there is a legitimate need to allocate objects on the heap, prefer using std::make_unique(). References for the reasoning:
  * https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rh-make_unique
  * https://herbsutter.com/2013/05/29/gotw-89-solution-smart-pointers/
  * https://abseil.io/tips/126
* Use [SafeInt](https://github.com/dcleblanc/SafeInt) when calculating the size of memory to allocate to protect against overflow errors
  * `#include "core/common/safeint.h"`
  * search for `SafeInt<size_t>` in the code for examples
* The following C++ warnings should never be disabled in onnxruntime VC++ projects(Required by [Binskim](https://github.com/microsoft/binskim/blob/d9afb65c89a621411efded74c27999281d87867e/src/BinSkim.Rules/PERules/BA2007.EnableCriticalCompilerWarnings.cs)).
  1. [4018](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4018) 'token' : signed/unsigned mismatch
  2. [4146](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4146?view=msvc-160) unary minus operator applied to unsigned type, result still unsigned
  3. [4244](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4244?view=msvc-160) 'argument' : conversion from 'type1' to 'type2', possible loss of data. For example, casting a int64_t to size_t. 
  4. [4267](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4267?view=msvc-160) 'var' : conversion from 'size_t' to 'type', possible loss of data.
  5. [4302](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4302?view=msvc-160) 'conversion' : truncation from 'type 1' to 'type 2'
  6. [4308](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4308?view=msvc-160) negative integral constant converted to unsigned type 
  7. [4532](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4532?view=msvc-160) 'continue' : jump out of \_\_finally/finally block has undefined behavior during termination handling
  8. [4533](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4533?view=msvc-160) initialization of 'variable' is skipped by 'instruction'
  9. [4700](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-and-level-4-c4700?view=msvc-160) uninitialized local variable 'name' used
  10. [4789](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4789?view=msvc-160) buffer 'identifier' of size N bytes will be overrun; M bytes will be written starting at offset L
  11. [4995](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4995?view=msvc-160) 'function': name was marked as #pragma deprecated 
  12. [4996](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4996?view=msvc-160) Your code uses a function, class member, variable, or typedef that's marked deprecated

#### Clang-format

Clang-format will handle automatically formatting code to these rules. There’s a Visual Studio plugin that can format on save at https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.ClangFormat, or alternatively the latest versions of Visual Studio 2017 include [clang-format support](https://blogs.msdn.microsoft.com/vcblog/2018/03/13/clangformat-support-in-visual-studio-2017-15-7-preview-1/).  

There is a .clang-format file in the root directory that has the max line length override and defaults to the google rules. This should be automatically discovered by the clang-format tools. 

## Code analysis

Visual Studio Code Analysis with [C++ Core guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md) rules enabled is configured to run on build for the onnxruntime_common, onnxruntime_graph and onnxruntime_util libraries. Updating the onnxruntime_framework and onnxruntime_provider libraries to enable Code Analysis and build warning free is pending. 

Code changes should build with no Code Analysis warnings, however this is somewhat difficult to achieve consistently as the Code Analysis implementation is in fairly constant flux. Different minor releases may have less false positives (a build with the latest version may be warning free, and a build with an earlier version may not), or detect additional problems (an earlier version builds warning free and a later version doesn't). 

We use [BinSkim Binary Analyzer](https://github.com/microsoft/binskim) scanning our binraries.

## Unit Testing and Code Coverage

There should be unit tests that cover the core functionality of the product, expected edge cases, and expected errors. 
Code coverage from these tests should aim at maintaining over 80% coverage. 

All changes should be covered by new or existing unit tests. 

In order to check that all the code you expect to be covered by testing is covered, run code coverage in Visual Studio using 'Analyze Code Coverage' under the Test menu. 

There is a configuration file in onnxruntime\VSCodeCoverage.runsettings that can be used to configure code coverage so that it reports numbers for just the onnxruntime code. Select that file in Visual Studio via the Test menu: 'Test' -> 'Test Settings' -> 'Select Test Settings File'. 

Using 'Show Code Coverage Coloring' will allow you to visually inspect which lines were hit by the tests. See <https://docs.microsoft.com/en-us/visualstudio/test/using-code-coverage-to-determine-how-much-code-is-being-tested?view=vs-2017>.

## Python Code Style

Please adhere to the [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/). 
A maximum line length of 120 characters is allowed for consistency with the C++ code.

Code can be validated with [flake8](https://pypi.org/project/flake8/) using the configuration file in the root directory called [.flake8](https://github.com/microsoft/onnxruntime/tree/master/.flake8).

The [autopep8](https://pypi.org/project/autopep8/) tool can be used to automatically fix a range of PEP8 issues, as can [yapf](https://github.com/google/yapf). There's a yapf configuration file [here](https://github.com/microsoft/onnxruntime/tree/master/onnxruntime/.style.yapf).

Editors such as PyCharm [(see here)](https://www.jetbrains.com/help/pycharm/code-inspection.html) and Visual Studio Code [(see here)](https://code.visualstudio.com/docs/python/linting#_flake8) can be configured to check for PEP8 issues. 

## Objective-C/C++ Code Style

Please follow the [Google Objective-C/C++ Style Guide](https://google.github.io/styleguide/objcguide.html).

Clang-format can be used to format Objective-C/C++ code. The .clang-format file is in the repository root directory.

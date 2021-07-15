REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

@ECHO ON
SETLOCAL EnableDelayedExpansion

SET TargetFramework=netcoreapp2.1
SET TargetArch=x64
SET dn="C:\Program Files\dotnet\dotnet"
SET CurrentOnnxRuntimeVersion=""
SET OrtEp=""
SET DefineConstants=""

SET LocalNuGetRepo=%1
IF NOT "%2"=="" (SET TargetFramework=%2)
IF NOT "%3"=="" (SET TargetArch=%3)
IF NOT "%4"=="" (
    SET CurrentOnnxRuntimeVersion=%4
) ELSE (
    echo "Usage: runtest.bat LocalNuGetRepoPath TargetFramework TargetArch NuGetPackageVersion"
)

IF "%TargetArch%"=="x64" (
  SET RuntimeIdentifier=win-x64
  SET PlatformTarget=x64
)

IF "%TargetArch%"=="x86" (
  SET dn="C:\Program Files (x86)\dotnet\dotnet"
  SET RuntimeIdentifier=win-x86
  SET PlatformTarget=x86
)

ECHO Target Framework is %TargetFramework%

REM Update if CUDA lib paths if set
SET PATH=%CUDA_PATH%\bin;%CUDNN_PATH%\bin;%PATH%

IF EXIST test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages RMDIR /S /Q test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages
IF EXIST test\Microsoft.ML.OnnxRuntime.EndToEndTests\bin RMDIR /S /Q test\Microsoft.ML.OnnxRuntime.EndToEndTests\bin
IF EXIST test\Microsoft.ML.OnnxRuntime.EndToEndTests\obj RMDIR /S /Q test\Microsoft.ML.OnnxRuntime.EndToEndTests\obj

@echo %PackageName%
@echo %CurrentOnnxRuntimeVersion%
%dn% clean test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj
%dn% add test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj package Microsoft.ML.OnnxRuntime.Managed --no-restore -v %CurrentOnnxRuntimeVersion%
%dn% restore test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --configfile .\Nuget.CSharp.config --no-cache --packages test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages --source https://api.nuget.org/v3/index.json --source  %LocalNuGetRepo%

IF NOT errorlevel 0 (
    @echo "Failed to restore nuget packages for the test project"
    EXIT 1
)

%dn% list test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj package
dir test\Microsoft.ML.OnnxRuntime.EndToEndTests\packages\

%dn% test test\Microsoft.ML.OnnxRuntime.EndToEndTests\Microsoft.ML.OnnxRuntime.EndToEndTests.csproj --no-restore /p:DefineConstants="USE_TENSORRT"
IF NOT errorlevel 0 (
    @echo "Failed to build or execute the end-to-end test"
    EXIT 1
)

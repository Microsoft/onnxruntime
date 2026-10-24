// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/js/js_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"

#include "conv.h"

namespace onnxruntime {
namespace js {

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kMSInternalNHWCDomain,
    22,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Conv<true>);

ONNX_OPERATOR_KERNEL_EX(
    Conv,
    kOnnxDomain,
    22,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Conv<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kMSInternalNHWCDomain,
    11, 21,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Conv<true>);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    11, 21,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Conv<false>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kMSInternalNHWCDomain,
    1, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Conv<true>);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Conv,
    kOnnxDomain,
    1, 10,
    kJsExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()),
    Conv<false>);

}  // namespace js
}  // namespace onnxruntime

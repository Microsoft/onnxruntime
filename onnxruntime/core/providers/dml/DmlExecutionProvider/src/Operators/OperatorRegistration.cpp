// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlDFT.h"
#include "DmlSTFT.h"
#include "DmlGridSample.h"
#include "OperatorRegistration.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorVersions.h"
#include "core/providers/dml/DmlExecutionProvider/inc/IWinmlExecutionProvider.h"
#include "core/framework/customregistry.h"
#include "onnx/defs/operator_sets.h"
#include <wrl/client.h>
#include <wrl/implements.h>
#include <mutex>

using namespace Microsoft::WRL;

#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/sequence/sequence_ops.h"
#include "core/providers/cpu/tensor/concatbase.h"
#include "core/providers/cpu/optional/optional_ops.h"
#include "core/providers/cpu/controlflow/if.h"

namespace onnxruntime {

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceAt);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceConstruct);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceEmpty);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceLength);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, ConcatFromSequence);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceErase);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceInsert);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 15, OptionalHasElement);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 15, OptionalGetElement);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 18, OptionalHasElement);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 18, OptionalGetElement);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 7, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 13, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 16, If);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 19, If);

}

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    SequenceAt,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceAt);

ONNX_OPERATOR_KERNEL_EX(
    SequenceConstruct,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SequenceConstruct);

ONNX_OPERATOR_KERNEL_EX(
    SequenceEmpty,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes()),
    SequenceEmpty);

ONNX_OPERATOR_KERNEL_EX(
    SequenceLength,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .OutputMemoryType(OrtMemTypeCPUInput, 0)
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    SequenceLength);

ONNX_OPERATOR_KERNEL_EX(
    SequenceErase,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceErase);

ONNX_OPERATOR_KERNEL_EX(
    SequenceInsert,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .TypeConstraint("S", DataTypeImpl::AllFixedSizeSequenceTensorTypes())
        .TypeConstraint("I", std::vector<MLDataType>{
                                 DataTypeImpl::GetTensorType<int32_t>(),
                                 DataTypeImpl::GetTensorType<int64_t>()}),
    SequenceInsert);

ONNX_OPERATOR_KERNEL_EX(
    OptionalHasElement,
    kOnnxDomain,
    15,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
    OptionalHasElement);

ONNX_OPERATOR_KERNEL_EX(
    OptionalGetElement,
    kOnnxDomain,
    15,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("O", DataTypeImpl::AllOptionalTypes())
        .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        // We may be able to re-use the input for the output as is unless the output
        // is a graph output. We provide this hint to the allocation planner
        // to make the re-use call.
        .Alias(0, 0),
    OptionalGetElement);

ONNX_OPERATOR_KERNEL_EX(
    OptionalHasElement,
    kOnnxDomain,
    18,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypes())
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>()),
    OptionalHasElement);

ONNX_OPERATOR_KERNEL_EX(
    OptionalGetElement,
    kOnnxDomain,
    18,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("O", DataTypeImpl::AllTensorAndSequenceTensorAndOptionalTypes())
        .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes())
        // We may be able to re-use the input for the output as is unless the output
        // is a graph output. We provide this hint to the allocation planner
        // to make the re-use call.
        .Alias(0, 0),
    OptionalGetElement);

ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    7,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
    If);

// output shape rules requiring the output shapes of the 'THEN' and 'ELSE'
// branches to be the same were relaxed in opset-11
ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    11,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("V", DataTypeImpl::AllFixedSizeTensorTypes()),
    If);

ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    13,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
    If);

ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    16,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypes()),
    If);

ONNX_OPERATOR_KERNEL_EX(
    If,
    kOnnxDomain,
    19,
    kDmlExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 0)  // 'cond' needs to be on CPU
        .TypeConstraint("B", DataTypeImpl::GetTensorType<bool>())
        .TypeConstraint("V", DataTypeImpl::AllTensorAndSequenceTensorTypesIRv9()),
    If);
}

namespace Dml
{

enum class SupportedTensorDataTypes : uint64_t
{
    Undefined = 1LLU<<0,
    Float32 = 1LLU<<1,
    UInt4 = 1LLU<<2,
    Int4 = 1LLU<<3,
    UInt8 = 1LLU<<4,
    Int8 = 1LLU<<5,
    UInt16 = 1LLU<<6,
    Int16 = 1LLU<<7,
    Int32 = 1LLU<<8,
    Int64 = 1LLU<<9,
    String = 1LLU<<10,
    Bool = 1LLU<<11,
    Float16 = 1LLU<<12,
    Float64 = 1LLU<<13,
    UInt32 = 1LLU<<14,
    UInt64 = 1LLU<<15,
    Complex64 = 1LLU<<16,
    Complex128 = 1LLU<<17,
    SequenceFloat32 = 1LLU<<18,
    SequenceUInt4 = 1LLU<<19,
    SequenceInt4 = 1LLU<<20,
    SequenceUInt8 = 1LLU<<21,
    SequenceInt8 = 1LLU<<22,
    SequenceUInt16 = 1LLU<<23,
    SequenceInt16 = 1LLU<<24,
    SequenceInt32 = 1LLU<<25,
    SequenceInt64 = 1LLU<<26,
    SequenceString = 1LLU<<27,
    SequenceBool = 1LLU<<28,
    SequenceFloat16 = 1LLU<<29,
    SequenceFloat64 = 1LLU<<30,
    SequenceUInt32 = 1LLU<<31,
    SequenceUInt64 = 1LLU<<32,
    SequenceComplex64 = 1LLU<<33,
    SequenceComplex128 = 1LLU<<34,
    Ints8to32 = UInt8|Int8|UInt16|Int16|UInt32|Int32,
    Ints32to64 = UInt32|Int32|UInt64|Int64,
    Ints8to64 = UInt8|Int8|UInt16|Int16|UInt32|Int32|UInt64|Int64,
    UInt8to64 = UInt8|UInt16|UInt32|UInt64,
    Float16to32 = Float16|Float32,
    Float16to64 = Float16|Float32|Float64,
    NumericDefault = Ints8to32|Float16to32, // Only simple numbers, not bool, complex, or string.
    Scalars8to32 = UInt8|Int8|UInt16|Int16|UInt32|Int32|Float16to32|Bool,
    AllScalars = UInt8|Int8|UInt16|Int16|UInt32|Int32|UInt64|Int64|Float16|Float32|Float64|Bool,
    AllSequences = SequenceUInt8|SequenceInt8|SequenceUInt16|SequenceInt16|SequenceUInt32|SequenceInt32|
                   SequenceUInt64|SequenceInt64|SequenceFloat16|SequenceFloat32|SequenceFloat64|SequenceBool,
    Ints8Bit = UInt8|Int8,
    Ints16Bit = UInt16|Int16,
    Ints32Bit = UInt32|Int32,
    Ints64Bit = UInt64|Int64,
    All = static_cast<uint64_t>(-1),
};
DEFINE_ENUM_FLAG_OPERATORS(Dml::SupportedTensorDataTypes);

enum class DmlGraphSupport : uint32_t
{
    Supported    = 0,
    NotSupported = 1,
};
DEFINE_ENUM_FLAG_OPERATORS(DmlGraphSupport);

struct OperatorRegistrationInformation
{
    const char* operatorName;
    int sinceVersion;
    const char* domain;
    MLOperatorKernelCreateFn creationFunction;
    MLOperatorShapeInferenceFunction shapeInferenceFunction;
    std::pair<std::array<const std::pair<uint32_t, uint32_t>, 4>, int> aliases;

    gsl::span<char const* const> tensorTypeNames;
    gsl::span<const SupportedTensorDataTypes> supportedTensorDataTypes;
    DmlGraphSupport dmlGraphSupport;

    std::pair<std::array<const uint32_t, 4>, int> requiredConstantCpuInputs = {{}, 0};

    // For use by operators such as Sum, which may require multiple calls to DML, in which case they
    // can't be represented as nodes in an optimized graph yet.
    std::optional<uint32_t> requiredInputCountForDmlGraphSupport;

    MLOperatorSupportQueryFunction supportQueryFunction;
    bool allowDynamicInputShapes = false;
};

DML_OP_EXTERN_CREATION_FUNCTION(Copy);
DML_OP_EXTERN_CREATION_FUNCTION(FC);
DML_OP_EXTERN_CREATION_FUNCTION(Conv);
DML_OP_EXTERN_CREATION_FUNCTION(NhwcConv);
DML_OP_EXTERN_CREATION_FUNCTION(ConvTranspose);
DML_OP_EXTERN_CREATION_FUNCTION(ConvTransposeWithDynamicPads);
DML_OP_EXTERN_CREATION_FUNCTION(AveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalAveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(MaxPool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalMaxPool);
DML_OP_EXTERN_CREATION_FUNCTION(LpPool);
DML_OP_EXTERN_CREATION_FUNCTION(GlobalLpPool);
DML_OP_EXTERN_CREATION_FUNCTION(MaxRoiPool);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearAveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearGlobalAveragePool);
DML_OP_EXTERN_CREATION_FUNCTION(RoiAlign10);
DML_OP_EXTERN_CREATION_FUNCTION(RoiAlign16);
DML_OP_EXTERN_CREATION_FUNCTION(InstanceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(BatchNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(BatchNormalization15);
DML_OP_EXTERN_CREATION_FUNCTION(LayerNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(LayerNormalization17);
DML_OP_EXTERN_CREATION_FUNCTION(SkipLayerNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(EmbedLayerNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(BiasSplitGelu);
DML_OP_EXTERN_CREATION_FUNCTION(BiasAdd);
DML_OP_EXTERN_CREATION_FUNCTION(LRN);
DML_OP_EXTERN_CREATION_FUNCTION(MeanVarianceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(GroupNorm);
DML_OP_EXTERN_CREATION_FUNCTION(SimplifiedLayerNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(SkipSimplifiedLayerNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(LpNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(RNN);
DML_OP_EXTERN_CREATION_FUNCTION(GRU);
DML_OP_EXTERN_CREATION_FUNCTION(LSTM);
DML_OP_EXTERN_CREATION_FUNCTION(Gather);
DML_OP_EXTERN_CREATION_FUNCTION(Flatten);
DML_OP_EXTERN_CREATION_FUNCTION(Split7);
DML_OP_EXTERN_CREATION_FUNCTION(Split11);
DML_OP_EXTERN_CREATION_FUNCTION(Split13);
DML_OP_EXTERN_CREATION_FUNCTION(Split18);
DML_OP_EXTERN_CREATION_FUNCTION(Transpose);
DML_OP_EXTERN_CREATION_FUNCTION(Tile);
DML_OP_EXTERN_CREATION_FUNCTION(Concat);
DML_OP_EXTERN_CREATION_FUNCTION(ConcatFromSequence);
DML_OP_EXTERN_CREATION_FUNCTION(Slice7);
DML_OP_EXTERN_CREATION_FUNCTION(Slice10);
DML_OP_EXTERN_CREATION_FUNCTION(Slice11);
DML_OP_EXTERN_CREATION_FUNCTION(Slice13);
DML_OP_EXTERN_CREATION_FUNCTION(Pad7);
DML_OP_EXTERN_CREATION_FUNCTION(Pad11);
DML_OP_EXTERN_CREATION_FUNCTION(Pad13);
DML_OP_EXTERN_CREATION_FUNCTION(Pad18);
DML_OP_EXTERN_CREATION_FUNCTION(Pad19);
DML_OP_EXTERN_CREATION_FUNCTION(Pad21);
DML_OP_EXTERN_CREATION_FUNCTION(SpaceToDepth);
DML_OP_EXTERN_CREATION_FUNCTION(DepthToSpace);
DML_OP_EXTERN_CREATION_FUNCTION(Sqrt);
DML_OP_EXTERN_CREATION_FUNCTION(Reciprocal);
DML_OP_EXTERN_CREATION_FUNCTION(Pow);
DML_OP_EXTERN_CREATION_FUNCTION(Exp);
DML_OP_EXTERN_CREATION_FUNCTION(Log);
DML_OP_EXTERN_CREATION_FUNCTION(Abs);
DML_OP_EXTERN_CREATION_FUNCTION(Ceil);
DML_OP_EXTERN_CREATION_FUNCTION(Floor);
DML_OP_EXTERN_CREATION_FUNCTION(Clip7);
DML_OP_EXTERN_CREATION_FUNCTION(Clip11);
DML_OP_EXTERN_CREATION_FUNCTION(Clip12);
DML_OP_EXTERN_CREATION_FUNCTION(Clip13);
DML_OP_EXTERN_CREATION_FUNCTION(Greater);
DML_OP_EXTERN_CREATION_FUNCTION(Less);
DML_OP_EXTERN_CREATION_FUNCTION(GreaterOrEqual);
DML_OP_EXTERN_CREATION_FUNCTION(LessOrEqual);
DML_OP_EXTERN_CREATION_FUNCTION(Equal);
DML_OP_EXTERN_CREATION_FUNCTION(Not);
DML_OP_EXTERN_CREATION_FUNCTION(And);
DML_OP_EXTERN_CREATION_FUNCTION(Or);
DML_OP_EXTERN_CREATION_FUNCTION(Xor);
DML_OP_EXTERN_CREATION_FUNCTION(Add);
DML_OP_EXTERN_CREATION_FUNCTION(Sub);
DML_OP_EXTERN_CREATION_FUNCTION(Mul);
DML_OP_EXTERN_CREATION_FUNCTION(Div);
DML_OP_EXTERN_CREATION_FUNCTION(Sum);
DML_OP_EXTERN_CREATION_FUNCTION(Mean);
DML_OP_EXTERN_CREATION_FUNCTION(Max);
DML_OP_EXTERN_CREATION_FUNCTION(Min);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceSum);
DML_OP_EXTERN_CREATION_FUNCTION(Einsum12);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMean);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceProd);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceLogSum);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceLogSumExp);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceSumSquare);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceL1);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceL2);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMax);
DML_OP_EXTERN_CREATION_FUNCTION(ReduceMin);
DML_OP_EXTERN_CREATION_FUNCTION(ArgMax);
DML_OP_EXTERN_CREATION_FUNCTION(ArgMin);
DML_OP_EXTERN_CREATION_FUNCTION(Gemm);
DML_OP_EXTERN_CREATION_FUNCTION(Neg);
DML_OP_EXTERN_CREATION_FUNCTION(Crop);
DML_OP_EXTERN_CREATION_FUNCTION(ImageScaler);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample7);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample9);
DML_OP_EXTERN_CREATION_FUNCTION(Upsample10);
DML_OP_EXTERN_CREATION_FUNCTION(Sigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(HardSigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(ScaledTanh);
DML_OP_EXTERN_CREATION_FUNCTION(Relu);
DML_OP_EXTERN_CREATION_FUNCTION(LeakyRelu);
DML_OP_EXTERN_CREATION_FUNCTION(PRelu);
DML_OP_EXTERN_CREATION_FUNCTION(ThresholdedRelu);
DML_OP_EXTERN_CREATION_FUNCTION(Elu);
DML_OP_EXTERN_CREATION_FUNCTION(Celu);
DML_OP_EXTERN_CREATION_FUNCTION(Selu);
DML_OP_EXTERN_CREATION_FUNCTION(Softmax);
DML_OP_EXTERN_CREATION_FUNCTION(Softmax13);
DML_OP_EXTERN_CREATION_FUNCTION(LogSoftmax);
DML_OP_EXTERN_CREATION_FUNCTION(LogSoftmax13);
DML_OP_EXTERN_CREATION_FUNCTION(Hardmax);
DML_OP_EXTERN_CREATION_FUNCTION(Hardmax13);
DML_OP_EXTERN_CREATION_FUNCTION(Softsign);
DML_OP_EXTERN_CREATION_FUNCTION(Softplus);
DML_OP_EXTERN_CREATION_FUNCTION(ParametricSoftplus);
DML_OP_EXTERN_CREATION_FUNCTION(Affine);
DML_OP_EXTERN_CREATION_FUNCTION(Dropout);
DML_OP_EXTERN_CREATION_FUNCTION(MatMul);
DML_OP_EXTERN_CREATION_FUNCTION(FusedMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(FusedMatMulActivation);
DML_OP_EXTERN_CREATION_FUNCTION(DynamicQuantizeMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(Cast);
DML_OP_EXTERN_CREATION_FUNCTION(CastLike15);
DML_OP_EXTERN_CREATION_FUNCTION(CastLike19);
DML_OP_EXTERN_CREATION_FUNCTION(CastLike21);
DML_OP_EXTERN_CREATION_FUNCTION(MemcpyFromHost);
DML_OP_EXTERN_CREATION_FUNCTION(MemcpyToHost);
DML_OP_EXTERN_CREATION_FUNCTION(TopK7);
DML_OP_EXTERN_CREATION_FUNCTION(TopK10);
DML_OP_EXTERN_CREATION_FUNCTION(TopK11);
DML_OP_EXTERN_CREATION_FUNCTION(Expand);
DML_OP_EXTERN_CREATION_FUNCTION(Cos);
DML_OP_EXTERN_CREATION_FUNCTION(Sin);
DML_OP_EXTERN_CREATION_FUNCTION(Tan);
DML_OP_EXTERN_CREATION_FUNCTION(Acos);
DML_OP_EXTERN_CREATION_FUNCTION(Asin);
DML_OP_EXTERN_CREATION_FUNCTION(Atan);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedConv);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedConvTranspose);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedInstanceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedBatchNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedMeanVarianceNormalization);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedGemm);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedAdd);
DML_OP_EXTERN_CREATION_FUNCTION(DmlFusedSum);
DML_OP_EXTERN_CREATION_FUNCTION(QuantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(QuantizeLinear21);
DML_OP_EXTERN_CREATION_FUNCTION(DequantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(DequantizeLinear21);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearSigmoid);
DML_OP_EXTERN_CREATION_FUNCTION(Sign);
DML_OP_EXTERN_CREATION_FUNCTION(IsNaN);
DML_OP_EXTERN_CREATION_FUNCTION(Sinh);
DML_OP_EXTERN_CREATION_FUNCTION(Cosh);
DML_OP_EXTERN_CREATION_FUNCTION(Tanh);
DML_OP_EXTERN_CREATION_FUNCTION(Asinh);
DML_OP_EXTERN_CREATION_FUNCTION(Acosh);
DML_OP_EXTERN_CREATION_FUNCTION(Atanh);
DML_OP_EXTERN_CREATION_FUNCTION(Erf);
DML_OP_EXTERN_CREATION_FUNCTION(Where);
DML_OP_EXTERN_CREATION_FUNCTION(Shrink);
DML_OP_EXTERN_CREATION_FUNCTION(Gelu);
DML_OP_EXTERN_CREATION_FUNCTION(BiasGelu);
DML_OP_EXTERN_CREATION_FUNCTION(FastGelu);
DML_OP_EXTERN_CREATION_FUNCTION(OneHot);
DML_OP_EXTERN_CREATION_FUNCTION(EyeLike);
DML_OP_EXTERN_CREATION_FUNCTION(MaxUnpool);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter9);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter11);
DML_OP_EXTERN_CREATION_FUNCTION(Scatter13);
DML_OP_EXTERN_CREATION_FUNCTION(Resize10);
DML_OP_EXTERN_CREATION_FUNCTION(Resize11);
DML_OP_EXTERN_CREATION_FUNCTION(Resize13);
DML_OP_EXTERN_CREATION_FUNCTION(ConstantOfShape);
DML_OP_EXTERN_CREATION_FUNCTION(IsInf);
DML_OP_EXTERN_CREATION_FUNCTION(Mod);
DML_OP_EXTERN_CREATION_FUNCTION(BitShift);
DML_OP_EXTERN_CREATION_FUNCTION(CumSum11);
DML_OP_EXTERN_CREATION_FUNCTION(CumSum14);
DML_OP_EXTERN_CREATION_FUNCTION(GatherElements);
DML_OP_EXTERN_CREATION_FUNCTION(GatherND);
DML_OP_EXTERN_CREATION_FUNCTION(Range);
DML_OP_EXTERN_CREATION_FUNCTION(ReverseSequence);
DML_OP_EXTERN_CREATION_FUNCTION(Round);
DML_OP_EXTERN_CREATION_FUNCTION(ScatterElements);
DML_OP_EXTERN_CREATION_FUNCTION(ScatterND);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearAdd);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearConv);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearMatMul);
DML_OP_EXTERN_CREATION_FUNCTION(QLinearConcat);
DML_OP_EXTERN_CREATION_FUNCTION(DynamicQuantizeLinear);
DML_OP_EXTERN_CREATION_FUNCTION(MatMulInteger);
DML_OP_EXTERN_CREATION_FUNCTION(MatMulIntegerToFloat);
DML_OP_EXTERN_CREATION_FUNCTION(ConvInteger);
DML_OP_EXTERN_CREATION_FUNCTION(Trilu);

#if DML_TARGET_VERSION >= 0x6400
DML_OP_EXTERN_CREATION_FUNCTION(Col2Im);
DML_OP_EXTERN_CREATION_FUNCTION(Resize18);
DML_OP_EXTERN_CREATION_FUNCTION(Resize19);
#endif

DML_OP_EXTERN_CREATION_FUNCTION(Shape);
DML_OP_EXTERN_CREATION_FUNCTION(Size);
DML_OP_EXTERN_CREATION_FUNCTION(QAttention);
DML_OP_EXTERN_CREATION_FUNCTION(Attention);
DML_OP_EXTERN_CREATION_FUNCTION(MultiHeadAttention);
DML_OP_EXTERN_CREATION_FUNCTION(GroupQueryAttention);
DML_OP_EXTERN_CREATION_FUNCTION(NonZero);
DML_OP_EXTERN_CREATION_FUNCTION(QuickGelu);
DML_OP_EXTERN_CREATION_FUNCTION(BitwiseAnd);
DML_OP_EXTERN_CREATION_FUNCTION(BitwiseOr);
DML_OP_EXTERN_CREATION_FUNCTION(BitwiseXor);
DML_OP_EXTERN_CREATION_FUNCTION(BitwiseNot);
DML_OP_EXTERN_CREATION_FUNCTION(RotaryEmbedding);
DML_OP_EXTERN_CREATION_FUNCTION(MatMulNBits);

DML_OP_EXTERN_QUERY_FUNCTION(MaxPool);
DML_OP_EXTERN_QUERY_FUNCTION(Slice);
DML_OP_EXTERN_QUERY_FUNCTION(Resize);
DML_OP_EXTERN_QUERY_FUNCTION(SpaceToDepth);
DML_OP_EXTERN_QUERY_FUNCTION(Scatter);
DML_OP_EXTERN_QUERY_FUNCTION(EinSum);
DML_OP_EXTERN_QUERY_FUNCTION(RecurrentNeuralNetwork);
DML_OP_EXTERN_QUERY_FUNCTION(BatchNormalization);
DML_OP_EXTERN_QUERY_FUNCTION(Pad);
DML_OP_EXTERN_QUERY_FUNCTION(LayerNormalization);
DML_OP_EXTERN_QUERY_FUNCTION(SkipLayerNormalization);
DML_OP_EXTERN_QUERY_FUNCTION(QLinearSigmoid);
DML_OP_EXTERN_QUERY_FUNCTION(QAttention);
DML_OP_EXTERN_QUERY_FUNCTION(Attention);
DML_OP_EXTERN_QUERY_FUNCTION(MatMulNBits);

constexpr static std::array<const char*, 1> typeNameListDefault = {"T"};
constexpr static std::array<const char*, 1> typeNameListDefaultV = {"V"};
constexpr static std::array<const char*, 2> typeNameListAttention = {"T", "M"};
constexpr static std::array<const char*, 2> typeNameListRotaryEmbedding = {"T", "M"};
constexpr static std::array<const char*, 2> typeNameListTwo = { "T1", "T2" };
constexpr static std::array<const char*, 2> typeNameListLayerNorm = { "T", "U" };
constexpr static std::array<const char*, 3> typeNameListLayerNormContrib = { "T", "U", "V" };
constexpr static std::array<const char*, 3> typeNameListThree = { "T1", "T2", "T3" };
constexpr static std::array<const char*, 4> typeNameListFour = { "T1", "T2", "T3", "T4" };
constexpr static std::array<const char*, 2> typeNameListTopK = { "T", "I" };
constexpr static std::array<const char*, 2> typeNameListMaxPool = { "T", "I" };
constexpr static std::array<const char*, 2> typeNameListLogicalComparison = { "T", "T1" };
constexpr static std::array<const char*, 2> typeNameListPow12 = {"T", "T1"};
constexpr static std::array<const char*, 2> typeNameListConstantOfShape = { "T1", "T2" };
constexpr static std::array<const char*, 2> typeNameListScatterGather = { "T", "Tind" };
constexpr static std::array<const char*, 1> typeNameListScatterGatherND = { "T" }; // Tind is curiously missing, only allowing 64-bit.
constexpr static std::array<const char*, 2> typeNameListSlice10 = { "T", "Tind" };
constexpr static std::array<const char*, 2> typeNameListWhere = { "B", "T" };
constexpr static std::array<const char*, 2> typeNameListEyeLike = { "T1", "T2" };
constexpr static std::array<const char*, 2> typeNameShape = { "T", "T1" };
constexpr static std::array<const char*, 2> typeNameSize = { "T", "T1" };
constexpr static std::array<const char*, 2> typeNameListGroupNorm = {"T", "M"};
constexpr static std::array<const char*, 3> typeNameListQLinearConcat= {"TF", "T8", "TV"};

constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListAll = {SupportedTensorDataTypes::All};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat32 = {SupportedTensorDataTypes::Float32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32 = {SupportedTensorDataTypes::Float16to32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Ints8 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Ints32 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::UInt32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Ints8to32 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit | SupportedTensorDataTypes::Ints16Bit | SupportedTensorDataTypes::Ints32Bit};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Ints8to64 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit | SupportedTensorDataTypes::Ints16Bit | SupportedTensorDataTypes::Ints32Bit | SupportedTensorDataTypes::Ints64Bit};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32SignedInts8to32 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int8 | SupportedTensorDataTypes::Int16 | SupportedTensorDataTypes::Int32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListFloat16to32Ints32to64 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints32Bit | SupportedTensorDataTypes::Ints64Bit};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListUInt8to64 = {SupportedTensorDataTypes::UInt8to64};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListAllIntegers = {SupportedTensorDataTypes::Ints8to64};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListNumericDefault = { SupportedTensorDataTypes::NumericDefault };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListAllScalars = {SupportedTensorDataTypes::AllScalars};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListAllScalarsAndSequences = {SupportedTensorDataTypes::AllScalars | SupportedTensorDataTypes::AllSequences};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListEyeLike = { SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::AllScalars};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListBool = {SupportedTensorDataTypes::Bool};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListPow12 = {SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::NumericDefault};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListTopK = {SupportedTensorDataTypes::NumericDefault | SupportedTensorDataTypes::Ints64Bit, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListMaxPool = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListMaxUnpool = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListIndices = { SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListCast = { SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::AllScalars };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListScalars8to32 = { SupportedTensorDataTypes::Scalars8to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListScatterGather = { SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListScatterGatherND = { SupportedTensorDataTypes::AllScalars };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListSlice10 = { SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListQuantizeLinear = { SupportedTensorDataTypes::Float32 | SupportedTensorDataTypes::Int32, SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListQuantizeLinear19 = { SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int32, SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListQuantizeLinear21 = { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::UInt4 | SupportedTensorDataTypes::Int4 | SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListDequantizeLinear = { SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 | SupportedTensorDataTypes::Int32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListDequantizeLinear19 = { SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8 | SupportedTensorDataTypes::Int32, SupportedTensorDataTypes::Float16to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListDequantizeLinear21 = { SupportedTensorDataTypes::UInt4 | SupportedTensorDataTypes::Int4 | SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8, SupportedTensorDataTypes::Float16to32 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListIsNan = { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListIsInf = { SupportedTensorDataTypes::Float32, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListConstantOfShape = { SupportedTensorDataTypes::Int64, SupportedTensorDataTypes::AllScalars };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListWhere = { SupportedTensorDataTypes::Bool, SupportedTensorDataTypes::AllScalars };
constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListOneHot = /* indices, depth, values */ { SupportedTensorDataTypes::Ints32to64, SupportedTensorDataTypes::AllScalars, SupportedTensorDataTypes::AllScalars };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListLogicalComparison7 = /* A&B,C */ { SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListLogicalComparison9 = /* A&B,C */ { SupportedTensorDataTypes::Float16to32|SupportedTensorDataTypes::Ints8to64, SupportedTensorDataTypes::Bool };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListSigned = { SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int64 | SupportedTensorDataTypes::Int32 | SupportedTensorDataTypes::Int16 | SupportedTensorDataTypes::Int8 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListRange = {SupportedTensorDataTypes::Int16|SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64|SupportedTensorDataTypes::Float32};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListResize11 = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Int8 | SupportedTensorDataTypes::UInt8, SupportedTensorDataTypes::Float16to32 /* ROI read by CPU */};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListResize13 = supportedTypeListResize11;
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListResize18 = supportedTypeListResize11;
constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListInteger = {SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8, SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8, SupportedTensorDataTypes::Int32 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListInteger8 = {SupportedTensorDataTypes::Int8|SupportedTensorDataTypes::UInt8 };
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListRoiAlign = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int32|SupportedTensorDataTypes::Int64 };
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListArgMinMax = {SupportedTensorDataTypes::Float16to32|SupportedTensorDataTypes::Ints8to64};
constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListLayerNormalizationContrib = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Float16to32};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListLayerNormalization = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Float32};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListShape = {SupportedTensorDataTypes::All, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListSize = {SupportedTensorDataTypes::All, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListQLinearSigmoid = {SupportedTensorDataTypes::UInt8 | SupportedTensorDataTypes::Int8};

constexpr static std::array<SupportedTensorDataTypes, 4> supportedTypeListQAttention = {
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Float16to32,
    SupportedTensorDataTypes::Int32
};

constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListAttention = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int32};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListRotaryEmbedding = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Int64};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListGroupNorm = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::Float16to32};
constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListNonZero = {SupportedTensorDataTypes::Float16to32 | SupportedTensorDataTypes::Ints8Bit | SupportedTensorDataTypes::Ints16Bit | SupportedTensorDataTypes::Ints32Bit | SupportedTensorDataTypes::Bool};
constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListMatMulNBits = {SupportedTensorDataTypes::Float16to32, SupportedTensorDataTypes::UInt8};

constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListQLinearMatMul = {
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit
};

constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListMatMulIntegerToFloat = {
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Float16to32
};

constexpr static std::array<SupportedTensorDataTypes, 4> supportedTypeListQLinearConv = {
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Int32
};


constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListDynamicQuantizeLinear = {
    SupportedTensorDataTypes::Float32,
    SupportedTensorDataTypes::Ints8Bit
};

constexpr static std::array<SupportedTensorDataTypes, 2> supportedTypeListDynamicQuantizeMatMul= {
    SupportedTensorDataTypes::Float32,
    SupportedTensorDataTypes::Ints8Bit,
};

constexpr static std::array<SupportedTensorDataTypes, 3> supportedTypeListQLinearConcat= {
    SupportedTensorDataTypes::Float32,
    SupportedTensorDataTypes::Ints8Bit,
    SupportedTensorDataTypes::Ints8Bit|SupportedTensorDataTypes::Float32,
};

constexpr static std::array<SupportedTensorDataTypes, 1> supportedTypeListQLinearAveragePool = {
    SupportedTensorDataTypes::Ints8Bit
};

template<typename... Args>
constexpr auto requiredConstantCpuInputs(Args... args)
{
    std::array<const uint32_t, 4> inputs = {static_cast<uint32_t>(args)...};
    return std::make_pair(inputs, static_cast<int>(sizeof...(args)));
}

template<typename... Args>
constexpr auto Aliases(Args... args)
{
    if constexpr (sizeof...(args) == 0)
    {
        std::array<const std::pair<uint32_t, uint32_t>, 4> aliases = {std::make_pair<uint32_t, uint32_t>(0, 0)};
        return std::make_pair(aliases, 0);
    }
    else
    {
        std::array<const std::pair<uint32_t, uint32_t>, 4> aliases = {static_cast<std::pair<uint32_t, uint32_t>>(args)...};
        return std::make_pair(aliases, static_cast<int>(sizeof...(args)));
    }
}

// Define a single row of OperatorRegistrationInformation.
#define REG_INFO(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, Aliases(), ##__VA_ARGS__,

#define REG_INFO_DYNAMIC_OUTPUTS(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName, nullptr, Aliases(), ##__VA_ARGS__,

// Versioned operator
#define REG_INFO_VER(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, Create##operatorName##version, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName##version>, Aliases(), ##__VA_ARGS__,

// Identity operators use Copy, alias their first input, and use elementwise identity operators
// when needed for striding support, but issue actual copies outside the graph.
#define REG_INFO_COPY(version, operatorName, ...) \
    #operatorName, OnnxOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kOnnxDomain, CreateCopy, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName##version>, Aliases(std::make_pair(0, 0)), ##__VA_ARGS__,

// MS-domain operators
#define REG_INFO_MS(version, operatorName, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, Aliases(), ##__VA_ARGS__,

// MS-domain operators
#define REG_INFO_MS_ALIAS(version, operatorName, aliases, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, aliases, ##__VA_ARGS__,

// MS-domain operators
#define REG_INFO_MSDML(version, operatorName, ...) \
    #operatorName, MsftOperatorSet##version::sc_sinceVer_##operatorName, onnxruntime::kMSDmlDomain, Create##operatorName, ShapeInferenceFunction<ShapeInferenceHelper_##operatorName>, Aliases(), ##__VA_ARGS__,

constexpr static OperatorRegistrationInformation operatorRegistrationInformationTable[] =
{
///  Domain/Type, Ver,  Name,                               TypeNames,                       Types,                                 Graph Support,                  Required const CPU inputs,
///                                                                                                                                                                 Input count required for graph support,
///                                                                                                                                                                 Support query function

    // Deep Learning Standard Layers
    {REG_INFO(      7,  Conv,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Conv,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  NhwcConv,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ConvTranspose,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ConvTranspose,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     19,  AveragePool,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  GlobalAveragePool,                  typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MaxPool,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      8,  MaxPool,                            typeNameListMaxPool,            supportedTypeListMaxPool,               DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},
    {REG_INFO(      10, MaxPool,                            typeNameListMaxPool,            supportedTypeListMaxPool,               DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},
    {REG_INFO(      11, MaxPool,                            typeNameListMaxPool,            supportedTypeListMaxPool,               DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},
    {REG_INFO(      12, MaxPool,                            typeNameListMaxPool,            supportedTypeListMaxPool,               DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryMaxPool)},

    {REG_INFO(      7,  GlobalMaxPool,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LpPool,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  LpPool,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     18,  LpPool,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  GlobalLpPool,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MaxRoiPool,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 10,  RoiAlign,                           typeNameListTwo,                supportedTypeListRoiAlign,              DmlGraphSupport::Supported)},
    {REG_INFO_VER( 16,  RoiAlign,                           typeNameListTwo,                supportedTypeListRoiAlign,              DmlGraphSupport::Supported)},
    {REG_INFO(      7,  InstanceNormalization,              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  BatchNormalization,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  BatchNormalization,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},  // v9 just removes 'spatial' attribute.
    {REG_INFO(     14,  BatchNormalization,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryBatchNormalization)},  // v14 adds training_mode attribute
    {REG_INFO(     15,  BatchNormalization,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryBatchNormalization)},  // v15 adds differing types for scale and bias vs input.
    {REG_INFO_VER( 17,  LayerNormalization,                 typeNameListLayerNorm,          supportedTypeListLayerNormalization,    DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryLayerNormalization)},
    {REG_INFO(      7,  LRN,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  LRN,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MeanVarianceNormalization,          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  MeanVarianceNormalization,          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  MeanVarianceNormalization,          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LpNormalization,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  RNN,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported)},
    {REG_INFO(     14,  RNN,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported,   requiredConstantCpuInputs(), std::nullopt, QueryRecurrentNeuralNetwork)},
    {REG_INFO(      7,  GRU,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported)},
    {REG_INFO(     14,  GRU,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported,   requiredConstantCpuInputs(), std::nullopt, QueryRecurrentNeuralNetwork)},
    {REG_INFO(      7,  LSTM,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported)},
    {REG_INFO(     14,  LSTM,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::NotSupported,   requiredConstantCpuInputs(), std::nullopt, QueryRecurrentNeuralNetwork)},
    {REG_INFO_MS(   1,  ConvTransposeWithDynamicPads,       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(2))},

    // Data Reorganization Layers
    {REG_INFO_VER(  7,  Split,                              typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_VER( 11,  Split,                              typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},  // Adds negative axis.
    {REG_INFO_VER( 13,  Split,                              typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},  // Moves splits from constant parameter to dynamic input.
    {REG_INFO_VER( 18,  Split,                              typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  Transpose,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Transpose,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(     21,  Transpose,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Concat,                             typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Concat,                             typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},  // Adds negative axis.
    {REG_INFO(     13,  Concat,                             typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},  // Adds negative axis.
    {REG_INFO_DYNAMIC_OUTPUTS(11, ConcatFromSequence,       typeNameListDefault,            supportedTypeListAllScalarsAndSequences,DmlGraphSupport::NotSupported)}, // Adds negative axis.
    {REG_INFO_VER(  7,  Slice,                              typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_VER( 10,  Slice,                              typeNameListSlice10,            supportedTypeListSlice10,               DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3, 4), std::nullopt, QuerySlice)},  // Adds negative axes.
    {REG_INFO_VER( 11,  Slice,                              typeNameListSlice10,            supportedTypeListSlice10,               DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3, 4), std::nullopt, QuerySlice)},
    {REG_INFO_VER( 13,  Slice,                              typeNameListSlice10,            supportedTypeListSlice10,               DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3, 4), std::nullopt, QuerySlice)},
    {REG_INFO_VER(  7,  Pad,                                typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(), std::nullopt, QueryPad)},
    {REG_INFO_VER( 11,  Pad,                                typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2) /*pads, value*/)}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
    {REG_INFO_VER( 13,  Pad,                                typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2) /*pads, value*/)}, // https://microsoft.visualstudio.com/OS/_workitems/edit/26007728
    {REG_INFO_VER( 18,  Pad,                                typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*pads, value, axes*/)},
    {REG_INFO_VER( 21,  Pad,                                typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*pads, value, axes*/)},

#if DML_TARGET_VERSION >= 0x6400
    {REG_INFO_VER( 19,  Pad,                                typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*pads, value, axes*/)},
#endif

    {REG_INFO(      7,  SpaceToDepth,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QuerySpaceToDepth)},
    {REG_INFO(     13,  SpaceToDepth,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QuerySpaceToDepth)},
    {REG_INFO(     28,  SpaceToDepth,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QuerySpaceToDepth)},
    {REG_INFO(      7,  DepthToSpace,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(     11,  DepthToSpace,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(     13,  DepthToSpace,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(     28,  DepthToSpace,                       typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Tile,                               typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},
    {REG_INFO(     13,  Tile,                               typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},
    {REG_INFO(      8,  Expand,                             typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},
    {REG_INFO(     13,  Expand,                             typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,      requiredConstantCpuInputs(1))},
    {REG_INFO(      9,  ConstantOfShape,                    typeNameListConstantOfShape,    supportedTypeListConstantOfShape,       DmlGraphSupport::Supported,      requiredConstantCpuInputs(0))},
    {REG_INFO(     21,  ConstantOfShape,                    typeNameListConstantOfShape,    supportedTypeListConstantOfShape,       DmlGraphSupport::Supported,      requiredConstantCpuInputs(0))},
    {REG_INFO(      7,  Gather,                             typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Gather,                             typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Gather,                             typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     11,  GatherElements,                     typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     13,  GatherElements,                     typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     11,  GatherND,                           typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported)},
    {REG_INFO(     12,  GatherND,                           typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported)},
    {REG_INFO(     13,  GatherND,                           typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported)},
    {REG_INFO_VER(  9,  Scatter,                            typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO_VER( 11,  Scatter,                            typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO_VER( 13,  Scatter,                            typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ScatterElements,                    typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ScatterElements,                    typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported)},
    {REG_INFO(     16,  ScatterElements,                    typeNameListScatterGather,      supportedTypeListScatterGather,         DmlGraphSupport::Supported,     requiredConstantCpuInputs(), std::nullopt, QueryScatter)},
    {REG_INFO(     11,  ScatterND,                          typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ScatterND,                          typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported)},
    {REG_INFO(     16,  ScatterND,                          typeNameListScatterGatherND,    supportedTypeListScatterGatherND,       DmlGraphSupport::Supported,     requiredConstantCpuInputs(), std::nullopt, QueryScatter)},
    {REG_INFO(      9,  EyeLike,                            typeNameListEyeLike,            supportedTypeListEyeLike,               DmlGraphSupport::Supported)},
    {REG_INFO(     14,  Trilu,                              typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},

#if DML_TARGET_VERSION >= 0x6400
    {REG_INFO(     18,  Col2Im,                             typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1, 2))},
#endif

    // Data reorganization that merely changes the dimensions while keeping the data identical.
    {REG_INFO_COPY( 7,  Identity,                           typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(13,  Identity,                           typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(14,  Identity,                           typeNameListDefaultV,           supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(16,  Identity,                           typeNameListDefaultV,           supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(19,  Identity,                           typeNameListDefaultV,           supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(21,  Identity,                           typeNameListDefaultV,           supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY( 7,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY( 9,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(11,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(13,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(21,  Flatten,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY( 7,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(11,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(13,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY(21,  Squeeze,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY( 7,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(11,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_COPY(13,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY(21,  Unsqueeze,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY( 7,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY(13,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY(14,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY(19,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_COPY(21,  Reshape,                            typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},

    // Elementwise
    {REG_INFO(      7,  Sqrt,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Sqrt,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Reciprocal,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Reciprocal,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Pow,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     12,  Pow,                                typeNameListPow12,              supportedTypeListPow12,                 DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Pow,                                typeNameListPow12,              supportedTypeListPow12,                 DmlGraphSupport::Supported)},  // 13 added bfloat16 to T.
    {REG_INFO(     15,  Pow,                                typeNameListPow12,              supportedTypeListPow12,                 DmlGraphSupport::Supported)},  // 15 added bfloat16 to T1.
    {REG_INFO(      7,  Exp,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Exp,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Log,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Log,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Abs,                                typeNameListDefault,            supportedTypeListSigned,                DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Abs,                                typeNameListDefault,            supportedTypeListSigned,                DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Ceil,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Ceil,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Floor,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Floor,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER(  7,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 11,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(1,2))},
    {REG_INFO_VER( 12,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported,     requiredConstantCpuInputs(1,2))},
    {REG_INFO_VER( 13,  Clip,                               typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported,     requiredConstantCpuInputs(1,2))},
    {REG_INFO(      7,  Add,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Add,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     14,  Add,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Sub,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Sub,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     14,  Sub,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Mul,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Mul,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     14,  Mul,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Div,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Div,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     14,  Div,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(     13,  Sum,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(     13,  Mean,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Max,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Max,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      12, Max,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      13, Max,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Min,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      8,  Min,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      12, Min,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      13, Min,                                typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},
    {REG_INFO(      7,  Cos,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Sin,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Tan,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Acos,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Asin,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Atan,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Affine,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      10, QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantizeLinear,        DmlGraphSupport::Supported)},
    {REG_INFO(      13, QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantizeLinear,        DmlGraphSupport::Supported)},
    {REG_INFO(      19, QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantizeLinear19,      DmlGraphSupport::Supported)},
    {REG_INFO_VER(  21, QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantizeLinear21,      DmlGraphSupport::Supported)},
    {REG_INFO(      10, DequantizeLinear,                   typeNameListDefault,            supportedTypeListDequantizeLinear,      DmlGraphSupport::Supported)},
    {REG_INFO(      13, DequantizeLinear,                   typeNameListDefault,            supportedTypeListDequantizeLinear,      DmlGraphSupport::Supported)},
    {REG_INFO(      19, DequantizeLinear,                   typeNameListTwo,                supportedTypeListDequantizeLinear19,    DmlGraphSupport::Supported)},
    {REG_INFO_VER(  21, DequantizeLinear,                   typeNameListTwo,                supportedTypeListDequantizeLinear21,    DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  QuantizeLinear,                     typeNameListTwo,                supportedTypeListQuantizeLinear19,      DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  DequantizeLinear,                   typeNameListTwo,                supportedTypeListDequantizeLinear19,    DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Sign,                               typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Sign,                               typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(      9,  IsNaN,                              typeNameListTwo,                supportedTypeListIsNan,                 DmlGraphSupport::Supported)},
    {REG_INFO(     13,  IsNaN,                              typeNameListTwo,                supportedTypeListIsNan,                 DmlGraphSupport::Supported)},
    {REG_INFO(     20,  IsNaN,                              typeNameListTwo,                supportedTypeListIsNan,                 DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Sinh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Cosh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Asinh,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Acosh,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Atanh,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Erf,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Erf,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Where,                              typeNameListWhere,              supportedTypeListWhere,                 DmlGraphSupport::Supported)},
    {REG_INFO(     16,  Where,                              typeNameListWhere,              supportedTypeListWhere,                 DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ReduceSum,                          typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceSum,                          typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceSum,                          typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_VER( 12,  Einsum,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), std::nullopt, QueryEinSum )},
    {REG_INFO(      7,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceMean,                         typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceProd,                         typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(     28,  ReduceLogSum,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(     28,  ReduceLogSumExp,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceSumSquare,                    typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceL1,                           typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceL2,                           typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     12,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceMax,                          typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(     20,  ReduceMax,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     12,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32Ints8to64,  DmlGraphSupport::Supported)},
    {REG_INFO(     18,  ReduceMin,                          typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(     20,  ReduceMin,                          typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  ArgMax,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ArgMax,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(     12,  ArgMax,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ArgMax,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ArgMin,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(     11,  ArgMin,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(     12,  ArgMin,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(     13,  ArgMin,                             typeNameListDefault,            supportedTypeListArgMinMax,             DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Gemm,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Neg,                                typeNameListDefault,            supportedTypeListSigned,                DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Neg,                                typeNameListDefault,            supportedTypeListSigned,                DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison7,    DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Greater,                            typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison7,    DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Less,                               typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     12,  GreaterOrEqual,                     typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     16,  GreaterOrEqual,                     typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     12,  LessOrEqual,                        typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     16,  LessOrEqual,                        typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison7,    DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(     19,  Equal,                              typeNameListLogicalComparison,  supportedTypeListLogicalComparison9,    DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Not,                                typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  And,                                typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Or,                                 typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Xor,                                typeNameListDefault,            supportedTypeListBool,                  DmlGraphSupport::Supported)},
    {REG_INFO(      18,  BitwiseAnd,                        typeNameListDefault,            supportedTypeListAllIntegers,           DmlGraphSupport::Supported)},
    {REG_INFO(      18,  BitwiseOr,                         typeNameListDefault,            supportedTypeListAllIntegers,           DmlGraphSupport::Supported)},
    {REG_INFO(      18,  BitwiseXor,                        typeNameListDefault,            supportedTypeListAllIntegers,           DmlGraphSupport::Supported)},
    {REG_INFO(      18,  BitwiseNot,                        typeNameListDefault,            supportedTypeListAllIntegers,           DmlGraphSupport::Supported)},

    // Imaging Operators
    {REG_INFO(      7,  Crop,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ImageScaler,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER(  7,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER(  9,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1) /*scales*/)},
    {REG_INFO_VER( 10,  Upsample,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1) /*scales*/)},
    {REG_INFO_VER( 10,  Resize,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,      requiredConstantCpuInputs(1) /*scales*/)},
    {REG_INFO_VER( 11,  Resize,                             typeNameListTwo,                supportedTypeListResize11,              DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*roi, scales, sizes*/, std::nullopt, QueryResize)},
    {REG_INFO_VER( 13,  Resize,                             typeNameListTwo,                supportedTypeListResize13,              DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*roi, scales, sizes*/, std::nullopt, QueryResize)},
#if DML_TARGET_VERSION >= 0x6400
    {REG_INFO_VER( 18,  Resize,                             typeNameListTwo,                supportedTypeListResize18,              DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*roi, scales, sizes*/, std::nullopt, QueryResize)},
    {REG_INFO_VER( 19,  Resize,                             typeNameListTwo,                supportedTypeListResize18,              DmlGraphSupport::Supported,      requiredConstantCpuInputs(1, 2, 3) /*roi, scales, sizes*/, std::nullopt, QueryResize)},
#endif
    // Activation Functions
    {REG_INFO(      7,  Sigmoid,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Sigmoid,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  HardSigmoid,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Tanh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Tanh,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ScaledTanh,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Relu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Relu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     14,  Relu,                               typeNameListDefault,            supportedTypeListFloat16to32SignedInts8to32, DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LeakyRelu,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     16,  LeakyRelu,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)}, // bfloat added to T in 16
    {REG_INFO(      7,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32SignedInts8to32, DmlGraphSupport::Supported)},
    {REG_INFO(     16,  PRelu,                              typeNameListDefault,            supportedTypeListFloat16to32SignedInts8to32, DmlGraphSupport::Supported)}, // bfloat added to T in 16
    {REG_INFO(      7,  ThresholdedRelu,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  ThresholdedRelu,                    typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Elu,                                typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      12, Celu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Selu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Softmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Softmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 13,  Softmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LogSoftmax,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  LogSoftmax,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 13,  LogSoftmax,                         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Hardmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Hardmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_VER( 13,  Hardmax,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Softsign,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Softplus,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  ParametricSoftplus,                 typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Dropout,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Shrink,                             typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},

    // Uncategorized
    {REG_INFO(      7,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      9,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     13,  MatMul,                             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(      7,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(      9,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(     19,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(     21,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(     28,  Cast,                               typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO_VER( 15,  CastLike,                           typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO_VER( 19,  CastLike,                           typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO_VER( 21,  CastLike,                           typeNameListTwo,                supportedTypeListCast,                  DmlGraphSupport::Supported)},
    {REG_INFO(      7,  MemcpyFromHost,                     typeNameListDefault,            supportedTypeListAll)},
    {REG_INFO(      7,  MemcpyToHost,                       typeNameListDefault,            supportedTypeListAll)},
    {REG_INFO_VER(  7,  TopK,                               typeNameListTopK,               supportedTypeListTopK,                  DmlGraphSupport::Supported)},
    {REG_INFO_VER( 10,  TopK,                               typeNameListTopK,               supportedTypeListTopK,                  DmlGraphSupport::Supported, requiredConstantCpuInputs(1))},
    {REG_INFO_VER( 11,  TopK,                               typeNameListTopK,               supportedTypeListTopK,                  DmlGraphSupport::Supported, requiredConstantCpuInputs(1))},
    {REG_INFO(      9,  OneHot,                             typeNameListThree,              supportedTypeListOneHot,                DmlGraphSupport::Supported, requiredConstantCpuInputs(1))},
    {REG_INFO(     11,  OneHot,                             typeNameListThree,              supportedTypeListOneHot,                DmlGraphSupport::Supported, requiredConstantCpuInputs(1))},
    {REG_INFO(      7,  Shape,                              typeNameShape,                  supportedTypeListShape,                 DmlGraphSupport::NotSupported)},
    {REG_INFO(     13,  Shape,                              typeNameShape,                  supportedTypeListShape,                 DmlGraphSupport::NotSupported)},
    {REG_INFO(     15,  Shape,                              typeNameShape,                  supportedTypeListShape,                 DmlGraphSupport::NotSupported)},
    {REG_INFO(     19,  Shape,                              typeNameShape,                  supportedTypeListShape,                 DmlGraphSupport::NotSupported)},
    {REG_INFO(     21,  Shape,                              typeNameShape,                  supportedTypeListShape,                 DmlGraphSupport::NotSupported)},
    {REG_INFO(      7,  Size,                               typeNameSize,                   supportedTypeListSize,                  DmlGraphSupport::NotSupported)},
    {REG_INFO(     13,  Size,                               typeNameSize,                   supportedTypeListSize,                  DmlGraphSupport::NotSupported)},
    {REG_INFO(     19,  Size,                               typeNameSize,                   supportedTypeListSize,                  DmlGraphSupport::NotSupported)},
    {REG_INFO(     21,  Size,                               typeNameSize,                   supportedTypeListSize,                  DmlGraphSupport::NotSupported)},
    {REG_INFO_DYNAMIC_OUTPUTS( 9,  NonZero,                 typeNameListDefault,            supportedTypeListNonZero,               DmlGraphSupport::NotSupported)},
    {REG_INFO_DYNAMIC_OUTPUTS(13,  NonZero,                 typeNameListDefault,            supportedTypeListNonZero,               DmlGraphSupport::NotSupported)},

    // DmlFused operators
    {REG_INFO_MSDML(1,  DmlFusedConv,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedConvTranspose,              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedInstanceNormalization,      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedBatchNormalization,         typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedMeanVarianceNormalization,  typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedGemm,                       typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedMatMul,                     typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedAdd,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MSDML(1,  DmlFusedSum,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported,     requiredConstantCpuInputs(), 2)},

    // Contrib operators
    {REG_INFO_MS(   1,  Gelu,                               typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  BiasGelu,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  FastGelu,                           typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  FusedMatMul,                        typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  DynamicQuantizeMatMul,              typeNameListTwo,                supportedTypeListDynamicQuantizeLinear, DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  FusedMatMulActivation,              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  QLinearSigmoid,                     typeNameListDefault,            supportedTypeListQLinearSigmoid,        DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryQLinearSigmoid)},
    {REG_INFO_MS(   1,  QAttention,                         typeNameListFour,               supportedTypeListQAttention,            DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryQAttention)},
    {REG_INFO_MS(   1,  Attention,                          typeNameListAttention,          supportedTypeListAttention,             DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryAttention)},
    {REG_INFO_MS(   1,  MultiHeadAttention,                 typeNameListAttention,          supportedTypeListAttention,             DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  RotaryEmbedding,                    typeNameListRotaryEmbedding,    supportedTypeListRotaryEmbedding,       DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  QLinearConcat,                      typeNameListQLinearConcat,      supportedTypeListQLinearConcat,         DmlGraphSupport::Supported)},

    {REG_INFO(     10,  IsInf,                              typeNameListTwo,                supportedTypeListIsInf,                 DmlGraphSupport::Supported)},
    {REG_INFO(     20,  IsInf,                              typeNameListTwo,                supportedTypeListIsInf,                 DmlGraphSupport::Supported)},
    {REG_INFO(     10,  Mod,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},
    {REG_INFO(     13,  Mod,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},
    {REG_INFO(     28,  Mod,                                typeNameListDefault,            supportedTypeListNumericDefault,        DmlGraphSupport::Supported)},
    {REG_INFO(     11,  BitShift,                           typeNameListDefault,            supportedTypeListUInt8to64,             DmlGraphSupport::Supported)},
    {REG_INFO(     11,  Round,                              typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  ReverseSequence,                    typeNameListDefault,            supportedTypeListAllScalars,            DmlGraphSupport::Supported)},
    {REG_INFO_VER( 11,  CumSum,                             typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO_VER( 14,  CumSum,                             typeNameListDefault,            supportedTypeListFloat16to32Ints32to64, DmlGraphSupport::Supported,     requiredConstantCpuInputs(1))},
    {REG_INFO(     11,  Range,                              typeNameListDefault,            supportedTypeListRange,                 DmlGraphSupport::Supported,     requiredConstantCpuInputs(0,1,2))},

    {REG_INFO(      9,  MaxUnpool,                          typeNameListTwo,                supportedTypeListMaxUnpool,             DmlGraphSupport::Supported,      requiredConstantCpuInputs(2))},
    {REG_INFO(     11,  MaxUnpool,                          typeNameListTwo,                supportedTypeListMaxUnpool,             DmlGraphSupport::Supported,      requiredConstantCpuInputs(2))},  // 11 is identical to 9.

    {REG_INFO_MS(  1,   QLinearAveragePool,                 typeNameListDefault,            supportedTypeListQLinearAveragePool,    DmlGraphSupport::Supported)},
    {REG_INFO_MS(  1,   QLinearGlobalAveragePool,           typeNameListDefault,            supportedTypeListQLinearAveragePool,    DmlGraphSupport::Supported)},
    {REG_INFO_MS(  1,   QLinearAdd,                         typeNameListDefault,            supportedTypeListInteger8,              DmlGraphSupport::Supported)},
    {REG_INFO(     10,  QLinearConv,                        typeNameListFour,               supportedTypeListQLinearConv,           DmlGraphSupport::Supported)},
    {REG_INFO(     10,  QLinearMatMul,                      typeNameListThree,              supportedTypeListQLinearMatMul,         DmlGraphSupport::Supported)},
    {REG_INFO(     21,  QLinearMatMul,                      typeNameListThree,              supportedTypeListQLinearMatMul,         DmlGraphSupport::Supported)},
    {REG_INFO(     10,  MatMulInteger,                      typeNameListThree,              supportedTypeListInteger,               DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  MatMulIntegerToFloat,               typeNameListThree,              supportedTypeListMatMulIntegerToFloat,  DmlGraphSupport::Supported)},
    {REG_INFO(     10,  ConvInteger,                        typeNameListThree,              supportedTypeListInteger,               DmlGraphSupport::Supported)},
    {REG_INFO(     11,  DynamicQuantizeLinear,              typeNameListTwo,                supportedTypeListDynamicQuantizeLinear, DmlGraphSupport::Supported)},
    {REG_INFO(      7,  LayerNormalization,                 typeNameListLayerNormContrib,   supportedTypeListLayerNormalizationContrib, DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryLayerNormalization)},
    {REG_INFO(      7,  SimplifiedLayerNormalization,       typeNameListLayerNormContrib,   supportedTypeListLayerNormalizationContrib, DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryLayerNormalization)},
    {REG_INFO_MS(   1,  SkipLayerNormalization,             typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QuerySkipLayerNormalization)},
    {REG_INFO_MS(   1,  SkipSimplifiedLayerNormalization,   typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QuerySkipLayerNormalization)},
    {REG_INFO_MS(   1,  EmbedLayerNormalization,            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  BiasSplitGelu,                      typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  BiasAdd,                            typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  QuickGelu,                          typeNameListDefault,            supportedTypeListFloat16to32,           DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  GroupNorm,                          typeNameListGroupNorm,          supportedTypeListGroupNorm,             DmlGraphSupport::Supported)},
    {REG_INFO(      21, GroupNorm,                          typeNameListGroupNorm,          supportedTypeListGroupNorm,             DmlGraphSupport::Supported)},
    {REG_INFO_MS(   1,  MatMulNBits,                        typeNameListTwo,                supportedTypeListMatMulNBits,           DmlGraphSupport::Supported, requiredConstantCpuInputs(), std::nullopt, QueryMatMulNBits)},

    // Operators that need to alias an input with an output
    {REG_INFO_MS_ALIAS(1, GroupQueryAttention, Aliases(std::make_pair(3, 1), std::make_pair(4, 2)), typeNameListAttention, supportedTypeListAttention, DmlGraphSupport::Supported, requiredConstantCpuInputs(6))},
};

template<typename T>
MLOperatorEdgeDescription TensorEdgeDesc()
{
    return {MLOperatorEdgeType::Tensor, static_cast<uint64_t>(MLTypeTraits<T>::TensorType)};
}

template<typename T>
MLOperatorEdgeDescription SequenceEdgeDesc()
{
    return {MLOperatorEdgeType::SequenceTensor, static_cast<uint64_t>(MLTypeTraits<T>::TensorType)};
}

void RegisterCpuOperatorsAsDml(onnxruntime::KernelRegistry* registry)
{
    using namespace onnxruntime;

    static const BuildKernelCreateInfoFn function_table[] = {
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceAt)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceConstruct)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceEmpty)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceLength)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceErase)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, SequenceInsert)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 15, OptionalHasElement)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 15, OptionalGetElement)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 18, OptionalHasElement)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 18, OptionalGetElement)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 7, If)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 11, If)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 13, If)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 16, If)>,
        BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kDmlExecutionProvider, kOnnxDomain, 19, If)>,
    };

    for (auto& function_table_entry : function_table) {
        KernelCreateInfo info = function_table_entry();
        if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
            ORT_THROW_IF_ERROR(registry->Register(std::move(info)));
        }
    }
}

void RegisterDmlOperators(IMLOperatorRegistry* registry)
{
    ComPtr<IMLOperatorRegistryPrivate> registryPrivate;
    ORT_THROW_IF_FAILED(registry->QueryInterface(registryPrivate.GetAddressOf()));

    std::vector<MLOperatorEdgeTypeConstrant> typeConstraints;
    std::vector<MLOperatorEdgeDescription> edgeDescs;

    for (const OperatorRegistrationInformation& information : operatorRegistrationInformationTable)
    {
        assert(information.tensorTypeNames.size() == information.supportedTensorDataTypes.size());

        MLOperatorKernelDescription desc = {};
        desc.domain = information.domain;
        desc.name = information.operatorName;
        desc.executionType = MLOperatorExecutionType::D3D12;

        // The graph must be configured with operators from only the legacy DML API, or only the new DML API
        bool kernelSupportsGraph = !bool(information.dmlGraphSupport & DmlGraphSupport::NotSupported);

        desc.options = information.allowDynamicInputShapes ?
            MLOperatorKernelOptions::AllowDynamicInputShapes : MLOperatorKernelOptions::None;

        desc.minimumOperatorSetVersion = information.sinceVersion;

        typeConstraints.resize(information.tensorTypeNames.size());
        desc.typeConstraints = typeConstraints.data();
        desc.typeConstraintCount = static_cast<uint32_t>(typeConstraints.size());

#if _DEBUG
        // If some version of the operator is supported for fusion, check that each registered version is also supported.
        // This ensures that table of operators and versions supporting fusion does not become stale as operator sets are added.
        FusionHelpers::AssertFusableOperatorSupportsVersionIfExists(desc.name, desc.domain, desc.minimumOperatorSetVersion);
#endif

        // edgeDescs will accumulate the edge descriptions across all type constraints.
        // The values of allowedTypeCount will indicate how many elements of edgeDescs
        // belong to each type constraint.
        edgeDescs.clear();
        size_t lastEdgeDescSize = 0;

        // Append all supported tensor data types for each tensor type.
        for (gsl::index i = 0, ci = information.tensorTypeNames.size(); i < ci; ++i)
        {
            typeConstraints[i].typeLabel = information.tensorTypeNames[i];

            std::vector<MLOperatorTensorDataType> supportedTypeList;
            SupportedTensorDataTypes supportedTypes = information.supportedTensorDataTypes[i];

            // Scalars
            if (bool(supportedTypes & SupportedTensorDataTypes::Float32)) edgeDescs.push_back(TensorEdgeDesc<float>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt4  )) edgeDescs.push_back(TensorEdgeDesc<::MLUInt4x2>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int4   )) edgeDescs.push_back(TensorEdgeDesc<::MLInt4x2>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt8  )) edgeDescs.push_back(TensorEdgeDesc<uint8_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int8   )) edgeDescs.push_back(TensorEdgeDesc<int8_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt16 )) edgeDescs.push_back(TensorEdgeDesc<uint16_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int16  )) edgeDescs.push_back(TensorEdgeDesc<int16_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int32  )) edgeDescs.push_back(TensorEdgeDesc<int32_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Int64  )) edgeDescs.push_back(TensorEdgeDesc<int64_t>());
            //if (bool(supportedTypes & SupportedTensorDataTypes::String )) edgeDescs.push_back(TensorEdgeDesc<std::string>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Bool   )) edgeDescs.push_back(TensorEdgeDesc<bool>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Float16)) edgeDescs.push_back(TensorEdgeDesc<::MLFloat16>());
            if (bool(supportedTypes & SupportedTensorDataTypes::Float64)) edgeDescs.push_back(TensorEdgeDesc<double>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt32 )) edgeDescs.push_back(TensorEdgeDesc<uint32_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::UInt64 )) edgeDescs.push_back(TensorEdgeDesc<uint64_t>());

            // Sequences
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceFloat32)) edgeDescs.push_back(SequenceEdgeDesc<float>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceUInt8  )) edgeDescs.push_back(SequenceEdgeDesc<uint8_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceInt8   )) edgeDescs.push_back(SequenceEdgeDesc<int8_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceUInt16 )) edgeDescs.push_back(SequenceEdgeDesc<uint16_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceInt16  )) edgeDescs.push_back(SequenceEdgeDesc<int16_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceInt32  )) edgeDescs.push_back(SequenceEdgeDesc<int32_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceInt64  )) edgeDescs.push_back(SequenceEdgeDesc<int64_t>());
            //if (bool(supportedTypes & SupportedTensorDataTypes::SequenceString )) edgeDescs.push_back(SequenceEdgeDesc<std::string>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceBool   )) edgeDescs.push_back(SequenceEdgeDesc<bool>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceFloat16)) edgeDescs.push_back(SequenceEdgeDesc<::MLFloat16>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceFloat64)) edgeDescs.push_back(SequenceEdgeDesc<double>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceUInt32 )) edgeDescs.push_back(SequenceEdgeDesc<uint32_t>());
            if (bool(supportedTypes & SupportedTensorDataTypes::SequenceUInt64 )) edgeDescs.push_back(SequenceEdgeDesc<uint64_t>());
            typeConstraints[i].allowedTypeCount = static_cast<uint32_t>(edgeDescs.size() - lastEdgeDescSize);
            lastEdgeDescSize = edgeDescs.size();
        }

        // Now that the edge descriptions list won't be re-allocated, assign pointers to its memory
        // into the type constraints entries
        size_t totalTypeCount = 0;
        for (gsl::index i = 0, ci = information.tensorTypeNames.size(); i < ci; ++i)
        {
            typeConstraints[i].allowedTypes = &edgeDescs.data()[totalTypeCount];
            totalTypeCount += typeConstraints[i].allowedTypeCount;
        }

        ComPtr<MLOperatorKernelFactory> factory =  Dml::SafeMakeOrThrow<MLOperatorKernelFactory>(information.creationFunction);
        ComPtr<MLOperatorShapeInferrer> shapeInferrer;

        if (information.shapeInferenceFunction)
        {
            shapeInferrer = Dml::SafeMakeOrThrow<MLOperatorShapeInferrer>(information.shapeInferenceFunction);
        }

        ComPtr<IMLOperatorSupportQueryPrivate> supportQuery;
        if (information.supportQueryFunction)
        {
            supportQuery = Dml::SafeMakeOrThrow<MLOperatorSupportQuery>(information.supportQueryFunction);
        }

        ORT_THROW_IF_FAILED(registryPrivate->RegisterOperatorKernel(
            &desc,
            factory.Get(),
            shapeInferrer.Get(),
            supportQuery.Get(),
            true, // isInternalOperator
            kernelSupportsGraph, // supportsGraph
            information.requiredInputCountForDmlGraphSupport ? &(*information.requiredInputCountForDmlGraphSupport) : nullptr,
            information.requiredConstantCpuInputs.first.data(),
            static_cast<uint32_t>(information.requiredConstantCpuInputs.second),
            information.aliases.first.data(),
            static_cast<uint32_t>(information.aliases.second)
        ));
    }

    GpuDFTOperatorFactory::RegisterDFTKernel(registry, 17);
    GpuDFTOperatorFactory::RegisterDFTKernel(registry, 20);
    DmlSTFTOperatorFactory::RegisterSTFTKernel(registry);
    DmlGridSampleOperatorFactory::RegisterGridSampleKernel(registry);
}

} // namespace Dml

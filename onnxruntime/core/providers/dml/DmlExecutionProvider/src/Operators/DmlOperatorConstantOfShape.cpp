
//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------
#include "precomp.h"

namespace Dml
{

class DmlOperatorConstantOfShape : public DmlOperator, public ConstantOfShapeHelper
{
public:
    using Self = DmlOperatorConstantOfShape;

    DmlOperatorConstantOfShape(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        ConstantOfShapeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1); // ignored shape tensor
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1); // output tensor

        std::vector<std::optional<uint32_t>> inputIndices = {}; // The shape tensor is not GPU bound.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        Initialize(kernelCreationContext, inputIndices, outputIndices);

        // Read the tensor attribute for the output fill pattern.
        if (kernelCreationContext.HasAttribute(AttrName::Value, MLOperatorAttributeTypeTensor))
        {
            ComPtr<IMLOperatorKernelCreationContext> kernelCreationContextInterface = kernelCreationContext.GetInterface();
            ComPtr<IMLOperatorAttributes1> attributes;
            ComPtr<IMLOperatorTensor> valueTensor;

            // Get the extended attributes to be able to access the constant tensor.
            THROW_IF_FAILED(kernelCreationContextInterface.As(&attributes));
            THROW_IF_FAILED(attributes->GetTensorAttribute(AttrName::Value, &valueTensor));
            MLOperatorTensor wrappedValueTensor(valueTensor.Get());

            // Read the raw bytes from the tensor, agnostic to data type, which becomes the GPU fill pattern.
            ML_CHECK_VALID_ARGUMENT(wrappedValueTensor.IsCpuData());
            const uint32_t elementCount = wrappedValueTensor.GetTotalElementCount();
            ML_CHECK_VALID_ARGUMENT(elementCount == 1); // Expect exactly one element.
            const size_t rawDataByteSize = GetByteSizeFromMlDataType(wrappedValueTensor.GetTensorDataType());
            const std::byte* rawData = static_cast<const std::byte*>(valueTensor->GetData());
            valueBytes.assign(rawData, rawData + rawDataByteSize);
        }
        // Else valueBytes is empty, and the default fill pattern is 0.
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override
    {
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);
        THROW_IF_FAILED(m_executionProvider->FillTensorWithPattern(outputTensors.front(), valueBytes));
    }

private:
    std::vector<std::byte> valueBytes;
};

DML_OP_DEFINE_CREATION_FUNCTION(ConstantOfShape, DmlOperatorConstantOfShape);

} // namespace Dml

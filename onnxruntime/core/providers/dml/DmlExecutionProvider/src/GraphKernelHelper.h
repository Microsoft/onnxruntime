// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "GraphDescBuilder.h"

namespace Dml
{
namespace GraphKernelHelper 
{
    using namespace Windows::AI::MachineLearning::Adapter;

    template <typename T>
    static T AlignToPow2(T offset, T alignment)
    {
        static_assert(std::is_unsigned_v<T>);
        assert(alignment != 0);
        assert((alignment & (alignment - 1)) == 0);
        return (offset + alignment - 1) & ~(alignment - 1);
    }
    
    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateResource(
        Dml::IExecutionProvider* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize);

    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateCpuResource(
        Dml::IExecutionProvider* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize);

    void UnwrapTensor(
        IWinmlExecutionProvider* winmlProvider,
        const onnxruntime::Tensor* tensor,
        ID3D12Resource** resource,
        uint64_t* allocId);

    bool GetGraphInputConstness(
        uint32_t index,
        const onnxruntime::OpKernelInfo& kernelInfo,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeInputDefs,
        const std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap);

    void ProcessInputData(
        Dml::IExecutionProvider* provider,
        IWinmlExecutionProvider* winmlProvider,
        const std::vector<uint8_t>& inputsConstant,
        const onnxruntime::OpKernelInfo& kernelInfo,
        const Dml::GraphDescBuilder::GraphDesc& graphDesc,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeInputDefs,
        _Out_ std::vector<bool>& inputsUsed,
        _Inout_ std::vector<DML_BUFFER_BINDING>& initInputBindings,
        _Inout_ std::vector<ComPtr<ID3D12Resource>>& initInputResources,
        _Inout_ std::vector<ComPtr<ID3D12Resource>>& nonOwnedGraphInputsFromInitializers,
        _Inout_ std::vector<ComPtr<ID3D12Resource>>& initializeResourceRefs,
        _Inout_opt_ std::vector<std::vector<std::byte>>* inputRawData,
        _Inout_ std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap);

    void ConvertGraphDesc(
        const Dml::GraphDescBuilder::GraphDesc& graphDesc,
        _Out_ DML_GRAPH_DESC& dmlGraphDesc,
        const onnxruntime::OpKernelInfo& kernelInfo,
        _Inout_ std::vector<DML_OPERATOR_GRAPH_NODE_DESC>& dmlOperatorGraphNodes,
        _Inout_ std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges);

    std::string GetFusedNodeArgNameMatchingGraph(const std::string& fusedNodeArgeName);
    
}  // namespace GraphKernelHelper
}  // namespace Dml
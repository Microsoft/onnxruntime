#include "precomp.h"

#include "GraphKernelHelper.h"

namespace Dml
{
namespace GraphKernelHelper 
{
    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateResource(
        Dml::IExecutionProvider* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;

        D3D12_HEAP_PROPERTIES heapProperties = {
            D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};

        D3D12_RESOURCE_DESC resourceDesc = {D3D12_RESOURCE_DIMENSION_BUFFER,
                                            0,
                                            static_cast<uint64_t>((tensorByteSize + 3) & ~3),
                                            1,
                                            1,
                                            1,
                                            DXGI_FORMAT_UNKNOWN,
                                            {1, 0},
                                            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

        Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
        THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

        THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(buffer.GetAddressOf())));

        THROW_IF_FAILED(provider->UploadToResource(buffer.Get(), tensorPtr, tensorByteSize));

        return buffer;
    }

    Microsoft::WRL::ComPtr<ID3D12Resource>
    CreateCpuResource(
        Dml::IExecutionProvider* provider,
        const std::byte* tensorPtr,
        size_t tensorByteSize)
    {
        Microsoft::WRL::ComPtr<ID3D12Resource> buffer;

        D3D12_HEAP_PROPERTIES heapProperties = {
            D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0, 0, 0};

        D3D12_RESOURCE_DESC resourceDesc = {D3D12_RESOURCE_DIMENSION_BUFFER,
                                            0,
                                            static_cast<uint64_t>((tensorByteSize + 3) & ~3),
                                            1,
                                            1,
                                            1,
                                            DXGI_FORMAT_UNKNOWN,
                                            {1, 0},
                                            D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                                            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS};

        Microsoft::WRL::ComPtr<ID3D12Device> d3dDevice;
        THROW_IF_FAILED(provider->GetD3DDevice(d3dDevice.GetAddressOf()));

        THROW_IF_FAILED(d3dDevice->CreateCommittedResource(
            &heapProperties,
            D3D12_HEAP_FLAG_NONE,
            &resourceDesc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(buffer.GetAddressOf())));

        // Map the buffer and copy the data
        void* bufferData = nullptr;
        D3D12_RANGE range = {0, tensorByteSize};
        THROW_IF_FAILED(buffer->Map(0, &range, &bufferData));
        memcpy(bufferData, tensorPtr, tensorByteSize);
        buffer->Unmap(0, &range);

        return buffer;
    }

    void UnwrapTensor(
        IWinmlExecutionProvider* winmlProvider,
        const onnxruntime::Tensor* tensor,
        ID3D12Resource** resource,
        uint64_t* allocId) 
    {
        IUnknown* allocationUnk = static_cast<IUnknown*>(const_cast<void*>(tensor->DataRaw()));
        Microsoft::WRL::ComPtr<IUnknown> resourceUnk;
        winmlProvider->GetABIDataInterface(false, allocationUnk, &resourceUnk);

        *allocId = winmlProvider->TryGetPooledAllocationId(allocationUnk, 0);

        THROW_IF_FAILED(resourceUnk->QueryInterface(resource));
    }

    bool GetGraphInputConstness(
        uint32_t index,
        const onnxruntime::OpKernelInfo& kernelInfo,
        const onnxruntime::ConstPointerContainer<std::vector<onnxruntime::NodeArg*>>& fusedNodeInputDefs,
        const std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap) 
    {
        // Transferred initializers are uploaded to GPU memory
        auto iter = transferredInitializerMap.find(GetFusedNodeArgNameMatchingGraph(fusedNodeInputDefs[index]->Name()));
        if (iter != transferredInitializerMap.end())
        {
            return true;
        }

        // If an initializer wasn't transferred, the constant input may be available from ORT
        const onnxruntime::Tensor* inputTensor = nullptr;
        if (!kernelInfo.TryGetConstantInput(index, &inputTensor) || inputTensor == nullptr)
        {
            return false;
        }

        // Check that the constant ORT input is in GPU memory
        if (!strcmp(inputTensor->Location().name, onnxruntime::CPU) ||
            inputTensor->Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput ||
            inputTensor->Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput)
        {
            return false;
        }

        return true;
    };

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
        _Inout_ std::unordered_map<std::string, onnx::TensorProto>& transferredInitializerMap)
    {
        const uint32_t graphInputCount = kernelInfo.GetInputCount();
        // Determine the last input which uses an initializer, so initializers can be freed incrementally
        // while processing each input in order.
        std::map<const onnx::TensorProto*, uint32_t> initializerToLastInputIndexMap;
        for (uint32_t i = 0; i < graphInputCount; i++) 
        {
            auto iter = transferredInitializerMap.find(GetFusedNodeArgNameMatchingGraph(fusedNodeInputDefs[i]->Name()));
            if (iter != transferredInitializerMap.end()) {
                initializerToLastInputIndexMap[&iter->second] = i;
            }
        }

        // Walk through each graph edge and mark used inputs
        inputsUsed.assign(graphInputCount, false);
        for (const DML_INPUT_GRAPH_EDGE_DESC& edge : graphDesc.inputEdges) {
            inputsUsed[edge.GraphInputIndex] = true;
        }

        for (uint32_t i = 0; i < initInputBindings.size(); i++)
        {
            // If the input isn't actually used by the graph, nothing ever needs to be bound (either for
            // initialization or execution). So just throw away the transferred initializer and skip this input.
            if (!inputsUsed[i])
            {
                transferredInitializerMap.erase(GetFusedNodeArgNameMatchingGraph(fusedNodeInputDefs[i]->Name()));

                if (inputRawData)
                {
                    inputRawData->push_back(std::vector<std::byte>());
                }

                continue;
            }

            // Look for the initializer among those transferred from the graph during partitioning
            auto iter = transferredInitializerMap.find(GetFusedNodeArgNameMatchingGraph(fusedNodeInputDefs[i]->Name()));
            if (iter != transferredInitializerMap.end())
            {
                std::byte* tensorPtr = nullptr;
                size_t tensorByteSize = 0;
                std::unique_ptr<std::byte[]> unpackedTensor;

                auto& initializer = iter->second;

                // The tensor may be stored as raw data or in typed fields.
                if (initializer.has_raw_data())
                {
                    tensorPtr = (std::byte*)(initializer.raw_data().c_str());
                    tensorByteSize = initializer.raw_data().size();
                }
                else
                {
                    std::tie(unpackedTensor, tensorByteSize) = UnpackTensor(initializer);
                    tensorPtr = unpackedTensor.get(); 
                }

                // Tensor sizes in DML must be a multiple of 4 bytes large.
                tensorByteSize = AlignToPow2<size_t>(tensorByteSize, 4);

                if (inputRawData)
                {
                    inputRawData->push_back(std::vector<std::byte>(tensorPtr, tensorPtr + tensorByteSize));
                }

                if (!inputsConstant[i])
                {
                    // Store the resource to use during execution
                    ComPtr<ID3D12Resource> defaultBuffer = CreateResource(provider, tensorPtr, tensorByteSize);
                    nonOwnedGraphInputsFromInitializers[i] = defaultBuffer;
                    initializeResourceRefs.push_back(std::move(defaultBuffer));
                }
                else
                {
                    ComPtr<ID3D12Resource> initializeInputBuffer;

                    // D3D_FEATURE_LEVEL_1_0_CORE doesn't support Custom heaps
                    if (provider->IsMcdmDevice())
                    {
                        initializeInputBuffer = CreateResource(provider, tensorPtr, tensorByteSize);
                    }
                    else
                    {
                        initializeInputBuffer = CreateCpuResource(provider, tensorPtr, tensorByteSize);
                    }

                    // Set the binding for operator initialization to the buffer
                    initInputBindings[i].Buffer = initializeInputBuffer.Get();
                    initInputBindings[i].SizeInBytes = tensorByteSize;
                    initializeResourceRefs.push_back(std::move(initializeInputBuffer));
                }

                // Free the initializer if this is the last usage of it.
                if (initializerToLastInputIndexMap[&initializer] == i)
                {
                    transferredInitializerMap.erase(iter);
                }
            }
            else if (inputsConstant[i])
            {                
                const onnxruntime::Tensor* inputTensor = nullptr;
                THROW_HR_IF(E_UNEXPECTED, !kernelInfo.TryGetConstantInput(i, &inputTensor));

                const std::byte* tensorData = reinterpret_cast<const std::byte*>(inputTensor->DataRaw());

                if (inputRawData)
                {
                    inputRawData->push_back(
                        std::vector<std::byte>(tensorData, tensorData + inputTensor->SizeInBytes()));
                }

                uint64_t allocId;
                UnwrapTensor(winmlProvider, inputTensor, &initInputBindings[i].Buffer, &allocId);
                initInputBindings[i].SizeInBytes = initInputBindings[i].Buffer->GetDesc().Width;

                initInputBindings[i].Buffer->Release(); // Avoid holding an additional reference
                initInputResources.push_back(initInputBindings[i].Buffer);
            } 
            else if (inputRawData)
            {
                inputRawData->push_back(std::vector<std::byte>());
            }
        }

        // All initializers should have been consumed and freed above
        assert(transferredInitializerMap.empty());
    }

    void ConvertGraphDesc(
        const Dml::GraphDescBuilder::GraphDesc& graphDesc,
        _Out_ DML_GRAPH_DESC& dmlGraphDesc,
        const onnxruntime::OpKernelInfo& kernelInfo,
        _Inout_ std::vector<DML_OPERATOR_GRAPH_NODE_DESC>& dmlOperatorGraphNodes,
        _Inout_ std::vector<DML_GRAPH_NODE_DESC>& dmlGraphNodes,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlInputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlOutputEdges,
        _Inout_ std::vector<DML_GRAPH_EDGE_DESC>& dmlIntermediateEdges)
    {
        const uint32_t graphInputCount = kernelInfo.GetInputCount();

        for (size_t i = 0; i < graphDesc.nodes.size(); ++i)
        {
            dmlOperatorGraphNodes[i] = DML_OPERATOR_GRAPH_NODE_DESC{graphDesc.nodes[i].op.Get()};
            dmlGraphNodes[i] = DML_GRAPH_NODE_DESC{DML_GRAPH_NODE_TYPE_OPERATOR, &dmlOperatorGraphNodes[i]};
        }

        for (size_t i = 0; i < graphDesc.inputEdges.size(); ++i)
        {
            dmlInputEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INPUT, &graphDesc.inputEdges[i]};
        }

        for (size_t i = 0; i < graphDesc.outputEdges.size(); ++i)
        {
            dmlOutputEdges[i] = DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_OUTPUT, &graphDesc.outputEdges[i]};
        }

        for (size_t i = 0; i < graphDesc.intermediateEdges.size(); ++i)
        {
            dmlIntermediateEdges[i] =
                DML_GRAPH_EDGE_DESC{DML_GRAPH_EDGE_TYPE_INTERMEDIATE, &graphDesc.intermediateEdges[i]};
        }

        dmlGraphDesc.InputCount = graphInputCount;
        dmlGraphDesc.OutputCount = kernelInfo.GetOutputCount();
        dmlGraphDesc.NodeCount = gsl::narrow_cast<uint32_t>(dmlGraphNodes.size());
        dmlGraphDesc.Nodes = dmlGraphNodes.data();
        dmlGraphDesc.InputEdgeCount = gsl::narrow_cast<uint32_t>(dmlInputEdges.size());
        dmlGraphDesc.InputEdges = dmlInputEdges.data();
        dmlGraphDesc.OutputEdgeCount = gsl::narrow_cast<uint32_t>(dmlOutputEdges.size());
        dmlGraphDesc.OutputEdges = dmlOutputEdges.data();
        dmlGraphDesc.IntermediateEdgeCount = gsl::narrow_cast<uint32_t>(dmlIntermediateEdges.size());
        dmlGraphDesc.IntermediateEdges = dmlIntermediateEdges.data();
    }

    // TODO: This is a hack which strips the suffix added within Lotus transforms that insert mem copies.
    // This shouldn't be necessary if Lotus exposes the inputs/ouputs in the same order between the kernel
    // for a function, and the graph for that function exposed as a kernel property.  When the ordering 
    // mismatch is fixed (WindowsAI: 21114358, Lotus: 1953), this workaround should be removed.
    std::string GetFusedNodeArgNameMatchingGraph(const std::string& fusedNodeArgeName)
    {
        const char* suffix = nullptr;
        
        // The suffix used when inserting mem copies is equal to the below, probably followed by an incrementing number.
        if (!suffix) 
        {
            suffix = strstr(fusedNodeArgeName.c_str(), "_DmlExecutionProvider_");
        }

        // The suffix used when inserting mem copies is equal to the below, not followed by an incrementing number.
        if (!suffix) 
        {
            suffix = strstr(fusedNodeArgeName.c_str(), "_DmlExecutionProvider");
        }
        
        if (!suffix) 
        {
            suffix = strstr(fusedNodeArgeName.c_str(), "_token_");
        }

        if (suffix)
        {
            return std::string(
                fusedNodeArgeName.begin(),
                fusedNodeArgeName.begin() + (suffix - fusedNodeArgeName.c_str())
            );
        } 
        else 
        {
            return fusedNodeArgeName;
        }
    }
}  // namespace GraphKernelHelper
}  // namespace Dml
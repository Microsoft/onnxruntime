// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "IExecutionProvider.h"
#include "ExecutionProvider.h"
#include "PooledUploadHeap.h"
#include "ReadbackHeap.h"
#include "ExecutionContext.h"
#include "BucketizedBufferAllocator.h"
#include "MLOperatorAuthorImpl.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorHelper.h"
#include "core/providers/dml/OperatorAuthorHelper/OperatorHelper.h"
#include "AbiCustomRegistry.h"
#include "GraphPartitioner.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/framework/compute_capability.h"

#ifdef ERROR
#undef ERROR
#endif
#include "core/session/inference_session.h"
#define ERROR 0

#include "core/session/onnxruntime_c_api.h"
#include <wil/wrl.h>
#include <dxgi1_6.h>

#define ENABLE_GRAPH_COMPILATION

using namespace Windows::AI::MachineLearning::Adapter;

namespace Dml
{
    using namespace onnxruntime::common;

    ExecutionProvider::~ExecutionProvider()
    { 
        if (m_impl)
        {
            m_impl->Close();
        }
    }

    static void CreateDmlKernelRegistry(
        _Out_ std::shared_ptr<onnxruntime::KernelRegistry>* registry,
        _Out_ std::shared_ptr<const InternalRegistrationInfoMap>* internalRegInfoMap)
    {
        ComPtr<AbiCustomRegistry> abiRegistry = wil::MakeOrThrow<AbiCustomRegistry>();
        Dml::RegisterDmlOperators(abiRegistry.Get());

        assert(abiRegistry->GetRegistries().size() == 1);
        
        auto customRegistry = *abiRegistry->GetRegistries().begin();
        *registry = customRegistry->GetKernelRegistry();
        *internalRegInfoMap = abiRegistry->GetInternalRegInfoMap();
    }

    ExecutionProvider::ExecutionProvider(
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* commandQueue,
        bool enableMetacommands) :
            IExecutionProvider(onnxruntime::kDmlExecutionProvider)
    {
        D3D12_COMMAND_LIST_TYPE queueType = commandQueue->GetDesc().Type;
        if (queueType != D3D12_COMMAND_LIST_TYPE_DIRECT && queueType != D3D12_COMMAND_LIST_TYPE_COMPUTE)
        {
            // DML requires either DIRECT or COMPUTE command queues.
            THROW_HR(E_INVALIDARG);
        }

        ComPtr<ID3D12Device> device;
        THROW_IF_FAILED(commandQueue->GetDevice(IID_PPV_ARGS(&device)));

        m_impl = wil::MakeOrThrow<ExecutionProviderImpl>(dmlDevice, device.Get(), commandQueue, enableMetacommands);

        // Register the allocators with ORT, through concrete ORT methods on the IExecutionProvider base class
        InsertAllocator(m_impl->GetGpuAllocator());
        InsertAllocator(m_impl->GetCpuInputAllocator());
        InsertAllocator(m_impl->GetCpuOutputAllocator());
    }

    std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
    ExecutionProvider::GetCapability(
        const onnxruntime::GraphViewer& graph,
        const std::vector<const onnxruntime::KernelRegistry*>& kernel_registries) const
    {
#ifdef ENABLE_GRAPH_COMPILATION
        return m_impl->GetCapability(graph, kernel_registries);
#endif
        return onnxruntime::IExecutionProvider::GetCapability(graph, kernel_registries);
    }

    void ExecutionProviderImpl::Close()
    {
        m_context->Close();
    }
    
    HRESULT __stdcall ExecutionProviderImpl::AllocatePooledResource(
        size_t size, 
        AllocatorRoundingMode roundingMode,
        ID3D12Resource **d3dResource, 
        IUnknown** pooledResource
    ) const noexcept try
    {
        ComPtr<IUnknown> allocation;
        allocation.Attach(static_cast<IUnknown* >(m_allocator->Alloc(size, roundingMode)));

        const auto* allocInfo = m_allocator->DecodeDataHandle(allocation.Get());

        ComPtr<ID3D12Resource> resource = allocInfo->GetResource();
        resource.CopyTo(d3dResource);
        *pooledResource = allocation.Detach();
        return S_OK;
    }
    CATCH_RETURN();

    ID3D12Resource* __stdcall ExecutionProviderImpl::DecodeResource(void* allocation) const noexcept
    {
        try
        {
            const AllocationInfo* allocInfo = m_allocator->DecodeDataHandle(allocation);
            return allocInfo->GetResource();
        }
        catch(...)
        {
            return nullptr;
        }
    }

// ORT release pipelines agent pools do not have 19H1 SDK installed which defines D3D_FEATURE_LEVEL_1_0_CORE.
// Once ORT/WinML github project can be built with VS2019, we can update these pools to use install the 19H1 SDK 
// using the command line installer tool with VS2019
// Task 24384515: Update ORT AIInfra release agent pool to install 19H1 SDK on VM bootstrap
#define D3D_FEATURE_LEVEL_1_0_CORE_PRIVATE ((D3D_FEATURE_LEVEL)0x1000)

    ExecutionProviderImpl::ExecutionProviderImpl(IDMLDevice* dmlDevice, ID3D12Device* d3d12Device, ID3D12CommandQueue* queue, bool enableMetacommands)
        : m_d3d12Device(d3d12Device),
          m_dmlDevice(dmlDevice),
          m_areMetacommandsEnabled(enableMetacommands)
    {

        D3D12_FEATURE_DATA_FEATURE_LEVELS featureLevels = {};

        D3D_FEATURE_LEVEL featureLevelsList[] = {
            D3D_FEATURE_LEVEL_1_0_CORE_PRIVATE,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_12_0,
            D3D_FEATURE_LEVEL_12_1
        };

        featureLevels.NumFeatureLevels = ARRAYSIZE(featureLevelsList);
        featureLevels.pFeatureLevelsRequested = featureLevelsList;
        THROW_IF_FAILED(d3d12Device->CheckFeatureSupport(
            D3D12_FEATURE_FEATURE_LEVELS,
            &featureLevels,
            sizeof(featureLevels)
            ));

        m_isMcdmDevice = (featureLevels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_1_0_CORE_PRIVATE);

        m_context = std::make_shared<ExecutionContext>(m_d3d12Device.Get(), m_dmlDevice.Get(), queue);

        // Create an allocator for D3D12 buffers used to hold tensor data. The returned buffers from the allocator
        // should be DEFAULT heap buffers which can be used as UAVs, and which start in UAV state.
        m_allocator = std::make_shared<BucketizedBufferAllocator>(
            m_d3d12Device.Get(),
            m_context,
            CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
            D3D12_HEAP_FLAG_NONE,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        m_context->SetAllocator(m_allocator);

        m_uploadHeap = std::make_unique<PooledUploadHeap>(m_d3d12Device.Get(), m_context);
        m_readbackHeap = std::make_unique<ReadbackHeap>(m_d3d12Device.Get(), m_context);
        
        // CPU Allocator used to create buffers for the MemcpyFromHost operator.
        m_cpuInputAllocator = std::make_shared<CPUAllocator>(OrtMemType::OrtMemTypeCPUInput);
        m_cpuOutputAllocator = std::make_shared<CPUAllocator>(OrtMemType::OrtMemTypeCPUOutput);

        CreateDmlKernelRegistry(&m_kernelRegistry, &m_internalRegInfoMap);
    }

    HRESULT __stdcall ExecutionProviderImpl::GetD3DDevice(_COM_Outptr_ ID3D12Device** d3dDevice) const noexcept
    {
        m_d3d12Device.CopyTo(d3dDevice);
        _Analysis_assume_(*d3dDevice != NULL);
        return S_OK;
    }

    HRESULT __stdcall ExecutionProviderImpl::GetDmlDevice(_COM_Outptr_ IDMLDevice** dmlDevice) const noexcept
    {
        m_dmlDevice.CopyTo(dmlDevice);
        _Analysis_assume_(*dmlDevice != NULL);
        return S_OK;
    }

    HRESULT __stdcall ExecutionProviderImpl::ExecuteCommandList(
        ID3D12GraphicsCommandList* commandList,
        _Outptr_ ID3D12Fence** fence,
        _Out_ uint64_t* completionValue
        ) const noexcept try
    {
        assert(!m_closed);
        m_context->ExecuteCommandList(commandList, fence, completionValue);

        return S_OK;
    }
    CATCH_RETURN();

    HRESULT __stdcall ExecutionProviderImpl::AddUAVBarrier() const noexcept try
    {
        assert(!m_closed);

        m_context->AddUAVBarrier();

        return S_OK;
    }
    CATCH_RETURN();

    HRESULT __stdcall ExecutionProviderImpl::InitializeOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<const DML_BUFFER_BINDING> inputBindings
        ) const noexcept try
    {
        assert(!m_closed);

        bool hasInputsToBind = false;
        std::vector<DML_BUFFER_BINDING> inputBufferBindings(inputBindings.size());

        for (gsl::index i = 0; i < inputBindings.size(); i++)
        {
            if (inputBindings[i].Buffer)
            {
                hasInputsToBind = true;
                inputBufferBindings[i] = { inputBindings[i].Buffer, inputBindings[i].Offset, inputBindings[i].SizeInBytes };
            }
        }

        DML_BINDING_DESC persistentResourceBindingDesc =
            persistentResourceBinding
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, persistentResourceBinding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        DML_BUFFER_ARRAY_BINDING inputBufferArrayDesc;
        inputBufferArrayDesc.BindingCount = gsl::narrow_cast<uint32_t>(inputBufferBindings.size());
        inputBufferArrayDesc.Bindings = inputBufferBindings.data();

        DML_BINDING_DESC inputArrayBindingDesc = hasInputsToBind ? 
            DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER_ARRAY, &inputBufferArrayDesc } : 
            DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        m_context->InitializeOperator(
            op,
            persistentResourceBindingDesc,
            inputArrayBindingDesc);

        return S_OK;
    }
    CATCH_RETURN();

    HRESULT __stdcall ExecutionProviderImpl::ExecuteOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<IMLOperatorTensor*> inputTensors,
        gsl::span<IMLOperatorTensor*> outputTensors
        ) const noexcept try
    {
        assert(!m_closed);

        std::vector<uint32_t> shape;

        for (IMLOperatorTensor* tensor : inputTensors)
        {
            if (tensor)
            {
                shape.resize(tensor->GetDimensionCount());
                THROW_IF_FAILED(tensor->GetShape(tensor->GetDimensionCount(), shape.data()));

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    return S_OK;
                }
            }
        }

        for (IMLOperatorTensor* tensor : outputTensors)
        {
            if (tensor)
            {
                shape.resize(tensor->GetDimensionCount());
                THROW_IF_FAILED(tensor->GetShape(tensor->GetDimensionCount(), shape.data()));

                if (OperatorHelper::ContainsEmptyDimensions(shape))
                {
                    return S_OK;
                }
            }
        }

        auto FillBindings = [this](auto& bufferBindings, auto& bindingDescs, auto& tensors)
        {
            for (IMLOperatorTensor* tensor : tensors)
            {
                if (tensor)
                {
                    assert(tensor->IsDataInterface());
                    const AllocationInfo* allocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(tensor).GetDataInterface().Get());
                    ID3D12Resource* resource = allocInfo->GetResource();
                    D3D12_RESOURCE_DESC resourceDesc = resource->GetDesc();
                    bufferBindings.push_back({ resource, 0, resourceDesc.Width });
                    bindingDescs.push_back({ DML_BINDING_TYPE_BUFFER, &bufferBindings.back() });
                }
                else
                {
                    bufferBindings.push_back({ nullptr, 0, 0 });
                    bindingDescs.push_back({ DML_BINDING_TYPE_NONE, nullptr });
                }
            }
        };

        std::vector<DML_BUFFER_BINDING> inputBufferBindings;
        inputBufferBindings.reserve(inputTensors.size());
        std::vector<DML_BINDING_DESC> inputBindings;
        inputBindings.reserve(inputTensors.size());
        FillBindings(inputBufferBindings, inputBindings, inputTensors);

        std::vector<DML_BUFFER_BINDING> outputBufferBindings;
        outputBufferBindings.reserve(outputTensors.size());
        std::vector<DML_BINDING_DESC> outputBindings;
        outputBindings.reserve(outputTensors.size());
        FillBindings(outputBufferBindings, outputBindings, outputTensors);

        THROW_IF_FAILED(ExecuteOperator(op, persistentResourceBinding, inputBindings, outputBindings));
        
        return S_OK;
    }
    CATCH_RETURN();

    HRESULT __stdcall ExecutionProviderImpl::ExecuteOperator(
        IDMLCompiledOperator* op,
        _In_opt_ const DML_BUFFER_BINDING* persistentResourceBinding,
        gsl::span<DML_BINDING_DESC> inputTensors,
        gsl::span<DML_BINDING_DESC> outputTensors
        ) const noexcept try
    {
        assert(!m_closed);
        
        DML_BINDING_DESC persistentResourceBindingDesc =
            persistentResourceBinding
            ? DML_BINDING_DESC{ DML_BINDING_TYPE_BUFFER, persistentResourceBinding }
            : DML_BINDING_DESC{ DML_BINDING_TYPE_NONE, nullptr };

        m_context->ExecuteOperator(
            op, 
            persistentResourceBindingDesc,
            inputTensors,
            outputTensors);

        return S_OK;
    }
    CATCH_RETURN();

    static gsl::span<const std::byte> AsByteSpan(const void* data, size_t sizeInBytes)
    {
        return gsl::make_span(static_cast<const std::byte*>(data), sizeInBytes);
    }

    static gsl::span<std::byte> AsByteSpan(void* data, size_t sizeInBytes)
    {
        return gsl::make_span(static_cast<std::byte*>(data), sizeInBytes);
    }

    HRESULT __stdcall ExecutionProviderImpl::CopyTensor(IMLOperatorTensor* dst, IMLOperatorTensor* src) const noexcept try
    {
        assert(!m_closed);

        const size_t sourceSizeInBytes = ComputeByteSizeFromTensor(*src);
        const size_t dataSizeInBytes = ComputeByteSizeFromTensor(*dst);
        THROW_HR_IF(E_INVALIDARG, dataSizeInBytes != sourceSizeInBytes); // Tensors must be the same size

        if (dataSizeInBytes == 0)
        {
            return S_OK;
        }

        if (src->IsCpuData() && !dst->IsCpuData())
        {
            // 
            // CPU -> GPU copy (upload)
            // 
            const AllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());
            
            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            const void* srcData = src->GetData();

            const uint64_t dstOffset = 0;
            const auto dstState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            m_uploadHeap->BeginUploadToGpu(dstData, dstOffset, dstState, AsByteSpan(srcData, dataSizeInBytes));
        }
        else if (!src->IsCpuData() && dst->IsCpuData())
        {
            // 
            // GPU -> CPU copy (readback)
            // 

            void* dstData = dst->GetData();
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());

            ID3D12Resource* srcData = srcAllocInfo->GetResource();

            const uint64_t srcOffset = 0;
            const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

            // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
            m_readbackHeap->ReadbackFromGpu(AsByteSpan(dstData, dataSizeInBytes), srcData, srcOffset, srcState);
        }
        else if (!src->IsCpuData() && !dst->IsCpuData())
        {
            // 
            // GPU -> GPU copy
            // 
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(src).GetDataInterface().Get());
            const AllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());

            ID3D12Resource* srcData = srcAllocInfo->GetResource();
            ID3D12Resource* dstData = dstAllocInfo->GetResource();
            m_context->CopyBufferRegion(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, srcData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, dataSizeInBytes);
        }
        else
        {
            // CPU -> CPU copies not supported
            THROW_HR(E_INVALIDARG);
        }

        return S_OK;
    }
    CATCH_RETURN();

    HRESULT STDMETHODCALLTYPE ExecutionProviderImpl::FillTensorWithPattern(
        IMLOperatorTensor* dst,
        gsl::span<const std::byte> value // Data type agnostic value, treated as raw bits
        ) const noexcept try
    {
        const AllocationInfo* dstAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(dst).GetDataInterface().Get());
        ID3D12Resource* dstData = dstAllocInfo->GetResource();
        m_context->FillBufferWithPattern(dstData, value);

        return S_OK;
    }
    CATCH_RETURN();

    HRESULT __stdcall ExecutionProviderImpl::UploadToResource(ID3D12Resource* dstData, const void* srcData, uint64_t srcDataSize) const noexcept try
    {
        assert(!m_closed);

        m_uploadHeap->BeginUploadToGpu(dstData, 0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, AsByteSpan(srcData, static_cast<size_t>(srcDataSize)));

        return S_OK;
    }
    CATCH_RETURN();

    uint32_t ExecutionProviderImpl::GetSupportedDeviceDataTypeMask() const
    {
        // The DML provider registers all supported kernels up-front regardless of actual device capability,
        // but this is problematic later when executing the graph because DirectML will fail to create
        // the operator, and by that late phase, it's long past too late to recover. So, this function queries
        // the actual type capabilities so the partitioner may assigns nodes to the CPU if the GPU cannot
        // handle them, similar to the fallback in CUDAExecutionProvider::GetCapability for certain RNN/GRU/Conv
        // attributes.

        return Dml::GetSupportedDeviceDataTypeMask(m_dmlDevice.Get());
    }

    std::vector<std::unique_ptr<onnxruntime::ComputeCapability>>
    ExecutionProviderImpl::GetCapability(
        const onnxruntime::GraphViewer& graph,
        const std::vector<const onnxruntime::KernelRegistry*>& registries) const
    {
        std::string partitionKernelPrefix = std::to_string(m_partitionKernelPrefixVal++) + "_";
        uint32_t deviceDataTypeMask = GetSupportedDeviceDataTypeMask();

        return PartitionGraph(
            graph,
            *m_internalRegInfoMap,
            registries,
            deviceDataTypeMask,
            m_kernelRegistry.get(),
            partitionKernelPrefix
        );
    }
    
    bool IsGpuTensor(const onnxruntime::Tensor& tensor) 
    {
        return strcmp(tensor.Location().name, onnxruntime::CPU) && 
            !(tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput || tensor.Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput);
    }

    Status ExecutionProviderImpl::CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst) const
    {
        assert(!m_closed);

        auto provider = const_cast<ExecutionProviderImpl*>(this);

        TensorWrapper destInternal(
            &dst, 
            IsGpuTensor(dst), 
            provider,
            true);

        TensorWrapper srcInternal(
            const_cast<onnxruntime::Tensor*>(&src), 
            IsGpuTensor(src), 
            provider,
            true);

        THROW_IF_FAILED(CopyTensor(&destInternal, &srcInternal));

        return onnxruntime::common::Status::OK();
    }

    Status ExecutionProviderImpl::CopyTensors(const std::vector<onnxruntime::IDataTransfer::SrcDstPair>& src_dst_pairs) const
    {
        // Source and destination for batched GPU -> CPU copies
        std::vector<ID3D12Resource*> srcDatas;
        std::vector<void*> dstDatas;
        std::vector<uint32_t> dataSizesInBytes;

        assert(!m_closed);
        auto provider = const_cast<ExecutionProviderImpl*>(this);

        for (uint32_t i = 0; i < src_dst_pairs.size(); ++i)
        {
            // This batching implementation only handles GPU -> CPU copies.  Other copies do not require synchronization
            // and are batched across multiple calls to CopyTensor.
            if (!IsGpuTensor(src_dst_pairs[i].src) || IsGpuTensor(src_dst_pairs[i].dst)) 
            {
                ORT_RETURN_IF_ERROR(CopyTensor(src_dst_pairs[i].src, src_dst_pairs[i].dst));
                continue;
            }
            
            TensorWrapper srcWrapper = TensorWrapper(
                const_cast<onnxruntime::Tensor*>(&src_dst_pairs[i].src.get()), 
                true,
                provider,
                true);

            TensorWrapper dstWrapper = TensorWrapper(
                &src_dst_pairs[i].dst.get(), 
                false, 
                provider,
                true);

            const size_t dataSizeInBytes = ComputeByteSizeFromTensor(dstWrapper);
            THROW_HR_IF(E_INVALIDARG, dataSizeInBytes != ComputeByteSizeFromTensor(srcWrapper)); // Tensors must be the same size

            if (dataSizeInBytes == 0)
            {
                return onnxruntime::common::Status::OK();
            }

            dataSizesInBytes.push_back(static_cast<uint32_t>(ComputeByteSizeFromTensor(dstWrapper)));
            THROW_HR_IF(E_INVALIDARG, dataSizesInBytes[i] != ComputeByteSizeFromTensor(srcWrapper)); // Tensors must be the same size

            dstDatas.push_back(dstWrapper.GetData());
            const AllocationInfo* srcAllocInfo = m_allocator->DecodeDataHandle(MLOperatorTensor(&srcWrapper).GetDataInterface().Get());

            srcDatas.push_back(srcAllocInfo->GetResource());
        }

        const uint64_t srcOffset = 0;
        const auto srcState = D3D12_RESOURCE_STATE_UNORDERED_ACCESS; // GPU resources are always kept in UAV state

        // Performs a blocking call to synchronize and read back data from the GPU into the destination buffer
        m_readbackHeap->ReadbackFromGpu(dstDatas, dataSizesInBytes, srcDatas, srcState);
        
        return onnxruntime::common::Status::OK();
    }

    void __stdcall ExecutionProviderImpl::Flush() const
    {
        assert(!m_closed);
        m_context->Flush();
    }

    void ExecutionProviderImpl::SetDefaultRoundingMode(AllocatorRoundingMode roundingMode)
    {
        m_allocator->SetDefaultRoundingMode(roundingMode);
    }

    void ExecutionProviderImpl::ReleaseCompletedReferences()
    {
         m_context->ReleaseCompletedReferences();
    }

    void ExecutionProviderImpl::QueueReference(IUnknown* object) 
    {
        assert(!m_closed);
        m_context->QueueReference(object);
    }

    void ExecutionProviderImpl::GetShadowCopyIfRequired(
        bool isInternalOperator,
        IUnknown* data,
        IUnknown** dataCopy) const
    {
        assert(!m_closed);

        *dataCopy = data;
        data->AddRef();
    }

    void ExecutionProviderImpl::GetABIDataInterface(
        bool isInternalOperator,
        IUnknown* data,
        IUnknown** abiData) const
    {
        assert(!m_closed);

        if (isInternalOperator)
        {
            *abiData = data;
            data->AddRef();
        }
        else
        {         
            ComPtr<ID3D12Resource> resource = m_allocator->DecodeDataHandle(data)->GetResource();
            *abiData = resource.Detach();
        } 
    }

    uint64_t ExecutionProviderImpl::TryGetPooledAllocationId(
        IUnknown* data,
        bool isInternalOperator)
    {
        assert(!isInternalOperator);
        return m_allocator->DecodeDataHandle(data)->GetPooledResourceId();
    }

    void ExecutionProviderImpl::GetABIExecutionInterface(
        bool isInternalOperator,
        IUnknown** abiExecutionObject) const 
    {
        assert(!m_closed);

        if (isInternalOperator)
        {
            ComPtr<IUnknown> thisPtr = const_cast<IExecutionProvider*>(static_cast<const IExecutionProvider*>(this));
            *abiExecutionObject = thisPtr.Detach();
        }
        else
        {
            ComPtr<ID3D12GraphicsCommandList> commandList;
            m_context->GetCommandListForRecording(commandList.GetAddressOf());
            *abiExecutionObject = commandList.Detach();
        }  
    }
    
    bool ExecutionProviderImpl::TransitionsRequiredForOperator(
        bool isInternalOperator
    )
    {
        // External operators receive resources in Common state, while internal operators receive
        // them in UAV state. Resources are otherwise kept in UAV state (or are promotable to UAV).
        return !isInternalOperator;
    }

    void ExecutionProviderImpl::TransitionResourcesForOperator(
        bool isBeforeOp,
        uint32_t resourceCount,
        IUnknown** resources
    ) 
    {
        std::vector<D3D12_RESOURCE_BARRIER> barriers;
        barriers.reserve(resourceCount);

        for (uint32_t i = 0; i < resourceCount; ++i)
        {
            ComPtr<ID3D12Resource> resource;
            THROW_IF_FAILED(resources[i]->QueryInterface(resource.GetAddressOf()));

            // Custom operators receive resources in Common state and must return them to Common
            // state when finished.  Resources are otherwise kept in UAV state (or are promotable to UAV).
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(
                resource.Get(), 
                isBeforeOp ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS : D3D12_RESOURCE_STATE_COMMON, 
                isBeforeOp ? D3D12_RESOURCE_STATE_COMMON : D3D12_RESOURCE_STATE_UNORDERED_ACCESS
            ));
        }

        if (!barriers.empty())
        {
            m_context->ResourceBarrier(barriers);
        }
    }

    D3D12_COMMAND_LIST_TYPE __stdcall ExecutionProviderImpl::GetCommandListTypeForQueue() const
    {
        return m_context->GetCommandListTypeForQueue();
    }
    
    bool __stdcall ExecutionProviderImpl::IsMcdmDevice() const noexcept
    {
        return m_isMcdmDevice;
    }

    bool __stdcall ExecutionProviderImpl::MetacommandsEnabled() const noexcept
    {
        return m_areMetacommandsEnabled;
    }

    std::shared_ptr<const Windows::AI::MachineLearning::Adapter::InternalRegistrationInfoMap> 
    ExecutionProviderImpl::GetInternalRegistrationInfoMap() const
    {
        return m_internalRegInfoMap;
    }

    std::shared_ptr<onnxruntime::IAllocator> ExecutionProviderImpl::GetGpuAllocator()
    {
        return m_allocator;
    }

    std::shared_ptr<onnxruntime::IAllocator> ExecutionProviderImpl::GetCpuInputAllocator()
    {
        return m_cpuInputAllocator;
    }

    std::shared_ptr<onnxruntime::IAllocator> ExecutionProviderImpl::GetCpuOutputAllocator()
    {
        return m_cpuOutputAllocator;
    }

    
    onnxruntime::common::Status ExecutionProviderImpl::OnSessionInitializationEnd() 
    {
        // Flush and trim resources, including staging memory used to upload weights.
        // This reduces memory usage immediately after session creation, and avoids
        // performance impact of deallocation during first evaluation.
        Flush();
        m_context->GetCurrentCompletionEvent().WaitForSignal();
        m_context->ReleaseCompletedReferences();
        m_uploadHeap->Trim();

        return onnxruntime::common::Status::OK();
    }

    std::unique_ptr<onnxruntime::IExecutionProvider> CreateExecutionProvider(
        IDMLDevice* dmlDevice,
        ID3D12CommandQueue* commandQueue,
        bool enableMetacommands)
    {
        return std::make_unique<Dml::ExecutionProvider>(dmlDevice, commandQueue, enableMetacommands);
    }

    ID3D12Resource* GetD3D12ResourceFromAllocation(onnxruntime::IAllocator* allocator, void* ptr)
    {
        Dml::BucketizedBufferAllocator* pAllocationInfo = static_cast<Dml::BucketizedBufferAllocator*>(allocator);
        return pAllocationInfo->DecodeDataHandle(ptr)->GetResource();
    }

    void FlushContext(onnxruntime::IExecutionProvider* provider)
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        dmlexecutionprovider->Flush();
    }

    void SetDefaultRoundingMode(onnxruntime::IExecutionProvider* provider, AllocatorRoundingMode roundingMode)
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        dmlexecutionprovider->SetDefaultRoundingMode(roundingMode);
    }

    void ReleaseCompletedReferences(onnxruntime::IExecutionProvider * provider)
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        dmlexecutionprovider->ReleaseCompletedReferences();
    }

    onnxruntime::common::Status CopyTensor(
        onnxruntime::IExecutionProvider* provider, 
        const onnxruntime::Tensor& src, 
        onnxruntime::Tensor& dst
    )
    {
        ExecutionProvider* dmlexecutionprovider = static_cast<Dml::ExecutionProvider*>(provider);
        return dmlexecutionprovider->GetImpl()->CopyTensor(src, dst);
    }

    void* CreateGPUAllocationFromD3DResource(ID3D12Resource* pResource)
    {
        uint64_t pooledResourceId = 0; // Not a pooled resource
        ComPtr<AllocationInfo> allocInfo = wil::MakeOrThrow<AllocationInfo>(nullptr, 0, pooledResourceId, pResource, (size_t)pResource->GetDesc().Width);
        return allocInfo.Detach();
    }
    void FreeGPUAllocation(void* ptr)
    {
        ComPtr<AllocationInfo> allocInfo;
        allocInfo.Attach(static_cast<AllocationInfo*>(ptr));
    }

} // namespace Dml

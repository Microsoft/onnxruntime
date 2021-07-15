// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "DmlCommandRecorder.h"
#include "CommandQueue.h"
#include "BucketizedBufferAllocator.h"

using namespace Dml;

DmlCommandRecorder::DmlCommandRecorder(
    ID3D12Device* d3dDevice,
    IDMLDevice* dmlDevice, 
    std::shared_ptr<CommandQueue> commandQueue)
    : m_queue(std::move(commandQueue)),
      m_d3dDevice(d3dDevice),
      m_dmlDevice(dmlDevice),
      m_descriptorPool(d3dDevice, 2048),
      m_commandAllocatorRing(d3dDevice, m_queue->GetType(), m_queue->GetCurrentCompletionEvent())
{
    THROW_IF_FAILED(dmlDevice->CreateOperatorInitializer(0, nullptr, IID_PPV_ARGS(&m_initializer)));
    THROW_IF_FAILED(dmlDevice->CreateCommandRecorder(IID_PPV_ARGS(&m_recorder)));
}

void DmlCommandRecorder::SetAllocator(std::weak_ptr<BucketizedBufferAllocator> allocator)
{
    m_bufferAllocator = allocator;
}

void DmlCommandRecorder::InitializeOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    const DML_BINDING_DESC& inputArrayBinding)
{
    // Reset the initializer to reference the input operator.
    IDMLCompiledOperator* ops[] = { op };
    THROW_IF_FAILED(m_initializer->Reset(ARRAYSIZE(ops), ops));

    DML_BINDING_PROPERTIES initBindingProps = m_initializer->GetBindingProperties();
    DML_BINDING_PROPERTIES execBindingProps = op->GetBindingProperties();

    const uint32_t numDescriptors = initBindingProps.RequiredDescriptorCount;
    DescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
        numDescriptors,
        m_queue->GetNextCompletionEvent());

    // Create a binding table for initialization.
    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = m_initializer.Get();
    bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
    bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
    bindingTableDesc.SizeInDescriptors = numDescriptors;

    ComPtr<IDMLBindingTable> bindingTable;
    THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

    // Create a temporary resource for initializing the op, if it's required.
    UINT64 temporaryResourceSize = initBindingProps.TemporaryResourceSize;
    if (temporaryResourceSize > 0)
    {
        auto allocator = m_bufferAllocator.lock();

        // Allocate and immediately free a temporary buffer. The buffer resource will still be
        // alive (managed by the pool); freeing allows the resource to be shared with other operators.
        void* tempResourceHandle = allocator->Alloc(static_cast<size_t>(temporaryResourceSize), AllocatorRoundingMode::Enabled);
        if (!tempResourceHandle)
        {
            THROW_HR(E_OUTOFMEMORY);
        }

        ID3D12Resource* buffer = allocator->DecodeDataHandle(tempResourceHandle)->GetResource();
        allocator->Free(tempResourceHandle);

        // Bind the temporary resource.
        DML_BUFFER_BINDING bufferBinding = { buffer, 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindTemporaryResource(&bindingDesc);
    }

    // Bind inputs, if provided.
    if (inputArrayBinding.Type != DML_BINDING_TYPE_NONE)
    {
        // An operator with inputs to bind MUST use a BUFFER_ARRAY.
        assert(inputArrayBinding.Type == DML_BINDING_TYPE_BUFFER_ARRAY);
        bindingTable->BindInputs(1, &inputArrayBinding);
    }

    // Bind the persistent resource, which is an output of initialization.
    if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE)
    {
        // Persistent resources MUST be bound as buffers.
        assert(persistentResourceBinding.Type == DML_BINDING_TYPE_BUFFER);
        bindingTable->BindOutputs(1, &persistentResourceBinding);
    }

    // Record the initialization work.
    SetDescriptorHeap(descriptorRange.heap);
    m_recorder->RecordDispatch(m_currentCommandList.Get(), m_initializer.Get(), bindingTable.Get());
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier if there's an output (i.e. persistent resource), or if any temps are used.
    if ((persistentResourceBinding.Type != DML_BINDING_TYPE_NONE) ||
        (temporaryResourceSize > 0))
    {
        m_currentCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
    }
}

void DmlCommandRecorder::ExecuteOperator(
    IDMLCompiledOperator* op,
    const DML_BINDING_DESC& persistentResourceBinding,
    gsl::span<const DML_BINDING_DESC> inputBindings,
    gsl::span<const DML_BINDING_DESC> outputBindings)
{
    DML_BINDING_PROPERTIES execBindingProps = op->GetBindingProperties();

    const uint32_t numDescriptors = execBindingProps.RequiredDescriptorCount;
    DescriptorRange descriptorRange = m_descriptorPool.AllocDescriptors(
        numDescriptors,
        m_queue->GetNextCompletionEvent());

    // Create a binding table for execution.
    DML_BINDING_TABLE_DESC bindingTableDesc = {};
    bindingTableDesc.Dispatchable = op;
    bindingTableDesc.CPUDescriptorHandle = descriptorRange.cpuHandle;
    bindingTableDesc.GPUDescriptorHandle = descriptorRange.gpuHandle;
    bindingTableDesc.SizeInDescriptors = numDescriptors;

    ComPtr<IDMLBindingTable> bindingTable;
    THROW_IF_FAILED(m_dmlDevice->CreateBindingTable(&bindingTableDesc, IID_PPV_ARGS(&bindingTable)));

    // Create a temporary resource for executing the op, if it's required.
    UINT64 temporaryResourceSize = execBindingProps.TemporaryResourceSize;
    if (temporaryResourceSize > 0)
    {
        auto allocator = m_bufferAllocator.lock();

        // Allocate and immediately free a temporary buffer. The buffer resource will still be
        // alive (managed by the pool); freeing allows the resource to be shared with other operators.
        void* tempResourceHandle = allocator->Alloc(static_cast<size_t>(temporaryResourceSize), AllocatorRoundingMode::Enabled);
        if (!tempResourceHandle)
        {
            THROW_HR(E_OUTOFMEMORY);
        }

        ID3D12Resource* buffer = allocator->DecodeDataHandle(tempResourceHandle)->GetResource();
        allocator->Free(tempResourceHandle);

        // Bind the temporary resource.
        DML_BUFFER_BINDING bufferBinding = { buffer, 0, temporaryResourceSize };
        DML_BINDING_DESC bindingDesc = { DML_BINDING_TYPE_BUFFER, &bufferBinding };
        bindingTable->BindTemporaryResource(&bindingDesc);
    }

    if (persistentResourceBinding.Type != DML_BINDING_TYPE_NONE)
    {
        bindingTable->BindPersistentResource(&persistentResourceBinding);
    }

    bindingTable->BindInputs(gsl::narrow<uint32_t>(inputBindings.size()), inputBindings.data());
    bindingTable->BindOutputs(gsl::narrow<uint32_t>(outputBindings.size()), outputBindings.data());

    // Record the execution work.
    SetDescriptorHeap(descriptorRange.heap);
    m_recorder->RecordDispatch(m_currentCommandList.Get(), op, bindingTable.Get());
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier all outputs.
    #pragma warning(suppress: 6387)
        m_currentCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
}

void DmlCommandRecorder::CopyBufferRegion(
    ID3D12Resource* dstBuffer,
    uint64_t dstOffset,
    ID3D12Resource* srcBuffer,
    uint64_t srcOffset,
    uint64_t byteCount)
{
    m_currentCommandList->CopyBufferRegion(dstBuffer, dstOffset, srcBuffer, srcOffset, byteCount);
    m_operationsRecordedInCurrentCommandList = true;
}
    
void DmlCommandRecorder::FillBufferWithPattern(
    ID3D12Resource* dstBuffer,
    gsl::span<const std::byte> value /* Data type agnostic value, treated as raw bits */)
{
    // The fill pattern for ClearUnorderedAccessViewUint is 16 bytes.
    union
    {
        uint32_t integers[4];
        std::byte bytes[16];
    } fillPattern = {};

    assert(ARRAYSIZE(fillPattern.bytes) == 16);
    assert(value.size() <= ARRAYSIZE(fillPattern.bytes)); // No element is expected larger than 128 bits (e.g. complex128).

    if (!value.empty())
    {
        assert(ARRAYSIZE(fillPattern.bytes) % value.size() == 0); // Should fit evenly into 16 bytes (e.g. uint8, float16, uint32, float64...).

        // Repeat the value multiple times into the pattern buffer.
        size_t valueIndex = 0;
        for (std::byte& p : fillPattern.bytes)
        {
            p = value[valueIndex++];
            valueIndex = (valueIndex == value.size()) ? 0 : valueIndex;
        }
    }
    // Else just leave fill pattern as zeroes.

    // Create a RAW buffer UAV over the resource.
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc = {};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    uavDesc.Buffer.NumElements = gsl::narrow<uint32_t>(dstBuffer->GetDesc().Width / sizeof(uint32_t));
    uavDesc.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;

    const uint32_t neededDescriptorCount = 1;
    DescriptorRange descriptorRangeCpu = m_descriptorPool.AllocDescriptors(neededDescriptorCount, m_queue->GetNextCompletionEvent(), D3D12_DESCRIPTOR_HEAP_FLAG_NONE);
    DescriptorRange descriptorRangeGpu = m_descriptorPool.AllocDescriptors(neededDescriptorCount, m_queue->GetNextCompletionEvent(), D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE);
    m_d3dDevice->CreateUnorderedAccessView(dstBuffer, nullptr, &uavDesc, descriptorRangeCpu.cpuHandle);
    m_d3dDevice->CreateUnorderedAccessView(dstBuffer, nullptr, &uavDesc, descriptorRangeGpu.cpuHandle);

    SetDescriptorHeap(descriptorRangeGpu.heap);

    // Record a ClearUAV onto the command list.
    m_currentCommandList->ClearUnorderedAccessViewUint(
        descriptorRangeGpu.gpuHandle,
        descriptorRangeCpu.cpuHandle,
        dstBuffer,
        fillPattern.integers,
        0,
        nullptr);
    m_operationsRecordedInCurrentCommandList = true;

    // Barrier all outputs.
    #pragma warning(suppress: 6387)
        m_currentCommandList->ResourceBarrier(1, &CD3DX12_RESOURCE_BARRIER::UAV(nullptr));
}

void DmlCommandRecorder::ExecuteCommandList(
    ID3D12GraphicsCommandList* commandList,
    _Outptr_ ID3D12Fence** fence,
    _Out_ uint64_t* completionValue
    )
{    
    THROW_IF_FAILED(m_currentCommandList->Close());

    if (m_operationsRecordedInCurrentCommandList)
    {            
        m_pendingCommandLists.push_back(m_currentCommandList.Get());
        m_pendingCommandListsCacheable.push_back(true);
    }
    else
    {
        m_cachedCommandLists.push_back(m_currentCommandList.Get());
    }

    m_currentCommandList = nullptr;
    m_operationsRecordedInCurrentCommandList = false;

    m_pendingCommandLists.push_back(commandList);
    m_pendingCommandListsCacheable.push_back(false);

    // Remember the descriptor heap and apply it to the next command list
    auto heap = m_currentDescriptorHeap;
    m_currentDescriptorHeap = nullptr;
    Open();

    // The caller can re-use relevant resources after the next set of work to be
    // flushed has completed.  Its command list hasn't been executed yet, just batched.
    GpuEvent gpuEvent = m_queue->GetNextCompletionEvent();
    gpuEvent.fence.CopyTo(fence);
    *completionValue = gpuEvent.fenceValue;

    // Trigger a flush of the command list, with the assumption that it contains enough GPU work that this
    // will help parallelize GPU work with subsequent CPU work.  This policy is related to the choice of
    // minNodeCountToReuseCommandList within FusedGraphKernel, so both should be tuned together.
    CloseAndExecute();
    Open();

    SetDescriptorHeap(heap);
}

ComPtr<ID3D12GraphicsCommandList> DmlCommandRecorder::GetCommandList()
{ 
    // Assume operations are added by the caller after this returns
    m_operationsRecordedInCurrentCommandList = true; 
    return m_currentCommandList; 
}

void DmlCommandRecorder::ResourceBarrier(gsl::span<const D3D12_RESOURCE_BARRIER> barriers)
{
    m_currentCommandList->ResourceBarrier(gsl::narrow_cast<uint32_t>(barriers.size()), barriers.data());
    m_operationsRecordedInCurrentCommandList = true; 
}

void DmlCommandRecorder::AddUAVBarrier()
{
    #pragma warning(suppress: 6387)
        auto barrier = CD3DX12_RESOURCE_BARRIER::UAV(nullptr);
    m_currentCommandList->ResourceBarrier(1, &barrier);
    m_operationsRecordedInCurrentCommandList = true; 
}

void DmlCommandRecorder::Open()
{
    assert(m_currentDescriptorHeap == nullptr);

    ID3D12CommandAllocator* allocator = m_commandAllocatorRing.GetCurrentAllocator();

    if (m_cachedCommandLists.empty())
    {
        THROW_IF_FAILED(m_d3dDevice->CreateCommandList(
            0,
            m_queue->GetType(),
            m_commandAllocatorRing.GetCurrentAllocator(),
            nullptr,
            IID_PPV_ARGS(&m_currentCommandList)));   
    }
    else
    {
        m_currentCommandList = m_cachedCommandLists.front();
        m_cachedCommandLists.pop_front();
        THROW_IF_FAILED(m_currentCommandList->Reset(allocator, nullptr));
    }

    // The current command allocator will become eligible for reset once this command list completes execution
    m_commandAllocatorRing.AdvanceAllocator(m_queue->GetNextCompletionEvent());
}

void DmlCommandRecorder::CloseAndExecute()
{
    THROW_IF_FAILED(m_currentCommandList->Close());

    if (m_operationsRecordedInCurrentCommandList)
    {            
        m_pendingCommandLists.push_back(m_currentCommandList.Get());
        m_pendingCommandListsCacheable.push_back(true);
    }
    else
    {
        m_cachedCommandLists.push_back(m_currentCommandList.Get());
    }

    m_currentCommandList = nullptr;
    m_operationsRecordedInCurrentCommandList = false;

    if (!m_pendingCommandLists.empty())
    {
        // Close and execute the command list
        m_queue->ExecuteCommandLists(
            gsl::span<ID3D12CommandList*>(reinterpret_cast<ID3D12CommandList**>(m_pendingCommandLists.data()), m_pendingCommandLists.size()));

        assert(m_pendingCommandLists.size() == m_pendingCommandListsCacheable.size());
        for (size_t i = 0; i < m_pendingCommandLists.size(); ++i)
        {
            if (m_pendingCommandListsCacheable[i])
            {
                m_cachedCommandLists.push_back(m_pendingCommandLists[i]);
            }
        }

        m_pendingCommandLists.clear();
        m_pendingCommandListsCacheable.clear();
    }

    // The descriptor heap must be set on the command list the next time it's opened.
    m_currentDescriptorHeap = nullptr;

    // Fail early if something horrifying happens
    THROW_IF_FAILED(m_dmlDevice->GetDeviceRemovedReason());
    THROW_IF_FAILED(m_d3dDevice->GetDeviceRemovedReason());
}

void DmlCommandRecorder::SetDescriptorHeap(ID3D12DescriptorHeap* descriptorHeap)
{
    if (descriptorHeap != nullptr && descriptorHeap != m_currentDescriptorHeap)
    {
        m_currentDescriptorHeap = descriptorHeap;

        ID3D12DescriptorHeap* descriptorHeaps[] = { descriptorHeap };
        m_currentCommandList->SetDescriptorHeaps(ARRAYSIZE(descriptorHeaps), descriptorHeaps);
    }
}
/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmHelper.h"

namespace ommhelper
{
    inline ID3D12Device5* OpacityMicroMapsHelper::GetD3D12Device5()
    {
        ID3D12Device* d3d12Device = (ID3D12Device*)NRI.GetDeviceNativeObject(*m_Device);
        if (!d3d12Device)
        {
            printf("[FAILED] ID3D12Device* d3d12Device = NRI.GetDeviceNativeObject(*m_Device)");
            std::abort();
        }
        ID3D12Device5* d3d12Device5 = nullptr;
        if (d3d12Device->QueryInterface(IID_PPV_ARGS(&d3d12Device5)) != S_OK)
        {
            printf("[FAILED] d3d12Device->QueryInterface(IID_PPV_ARGS(&d3d12Device5))");
            std::abort();
        }
        return d3d12Device5;
    }

    inline ID3D12GraphicsCommandList4* OpacityMicroMapsHelper::GetD3D12GraphicsCommandList4(nri::CommandBuffer* commandBuffer)
    {
        ID3D12GraphicsCommandList4* commandList = nullptr;
        {
            ID3D12GraphicsCommandList* graphicsCommandList = (ID3D12GraphicsCommandList*)NRI.GetCommandBufferNativeObject(*commandBuffer);
            if (graphicsCommandList->QueryInterface(IID_PPV_ARGS(&commandList)) != S_OK)
            {
                printf("[FAIL]: ID3D12GraphicsCommandList::QueryInterface(ID3D12GraphicsCommandList4)\n");
                std::abort();
            }
        }
        return commandList;
    }

    void OpacityMicroMapsHelper::InitializeD3D12()
    {
        _NvAPI_Status nvResult = NvAPI_Initialize();
        if (nvResult != NVAPI_OK)
        {
            printf("[FAIL]: NvAPI_Initialize\n");
            std::abort();
        }

        _NVAPI_D3D12_SET_CREATE_PIPELINE_STATE_OPTIONS_PARAMS_V1 createPsoParams = {};
        createPsoParams.version = NVAPI_D3D12_SET_CREATE_PIPELINE_STATE_OPTIONS_PARAMS_VER;
        createPsoParams.flags = NVAPI_D3D12_PIPELINE_CREATION_STATE_FLAGS_ENABLE_OMM_SUPPORT;
        nvResult = NvAPI_D3D12_SetCreatePipelineStateOptions(GetD3D12Device5(), &createPsoParams);
        if (nvResult != NVAPI_OK)
        {
            printf("[FAIL]: NvAPI_D3D12_SetCreatePipelineStateOptions\n");
            std::abort();
        }
    }

    inline D3D12_RESOURCE_DESC InitBufferResourceDesc(size_t size, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE)
    {
        D3D12_RESOURCE_DESC result = {};
        result.Width = size;
        result.Flags = flags;
        result.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        result.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        result.Height = 1;
        result.MipLevels = 1;
        result.DepthOrArraySize = 1;
        result.Format = DXGI_FORMAT_UNKNOWN;
        result.SampleDesc.Count = 1;
        result.SampleDesc.Quality = 0;
        result.Alignment = 0;

        return result;
    }

    inline D3D12_RESOURCE_BARRIER InitUavBarrier(ID3D12Resource* resource)
    {
        D3D12_RESOURCE_BARRIER result = {};
        result.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        result.UAV.pResource = resource;
        return result;
    }

    inline NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS FillOmmArrayInputsDesc(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* ommArrayData, ID3D12Resource* ommDescArray)
    {
        NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS vmInput = {};
        vmInput.flags = NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_BUILD_FLAG_PREFER_FAST_TRACE;
        vmInput.numOMMUsageCounts = inputs.descArrayHistogramNum;
        vmInput.pOMMUsageCounts = (NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT*)inputs.descArrayHistogram;

        uint64_t ommArrayDataOffset = inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].offset;
        uint64_t ommDescArrayOffset = inputs.buffers[(uint32_t)OmmDataLayout::DescArray].offset;
        vmInput.inputBuffer = ommArrayData ? ommArrayData->GetGPUVirtualAddress() + ommArrayDataOffset : NULL;
        vmInput.perOMMDescs.StartAddress = ommDescArray ? ommDescArray->GetGPUVirtualAddress() + ommDescArrayOffset : NULL;
        vmInput.perOMMDescs.StrideInBytes = sizeof(_NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_DESC);

        return vmInput;
    }

    inline NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX FillGeometryDescEx(MaskedGeometryBuildDesc::Inputs& inputs, ID3D12Resource* indexData, ID3D12Resource* vertexData, ID3D12Resource* ommArray, ID3D12Resource* ommIndexBuffer)
    {
        NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX geometryDescEx = {};
        geometryDescEx.flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
        geometryDescEx.type = NVAPI_D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES_EX;
        geometryDescEx.ommTriangles = {};

        NVAPI_D3D12_RAYTRACING_GEOMETRY_OMM_TRIANGLES_DESC& vmTriangles = geometryDescEx.ommTriangles;

        D3D12_RAYTRACING_GEOMETRY_TRIANGLES_DESC& trianglesDesc = vmTriangles.triangles;
        trianglesDesc.IndexBuffer = indexData ? indexData->GetGPUVirtualAddress() + inputs.indices.offset : NULL;
        trianglesDesc.IndexFormat = (DXGI_FORMAT)nri::ConvertNRIFormatToDXGI(inputs.indices.format);
        trianglesDesc.IndexCount = (UINT)inputs.indices.numElements;

        trianglesDesc.VertexCount = (UINT)inputs.vertices.numElements;
        trianglesDesc.VertexFormat = (DXGI_FORMAT)nri::ConvertNRIFormatToDXGI(inputs.vertices.format);
        trianglesDesc.VertexBuffer.StrideInBytes = inputs.vertices.stride;
        trianglesDesc.VertexBuffer.StartAddress = vertexData ? vertexData->GetGPUVirtualAddress() + inputs.vertices.offset : NULL;

        vmTriangles.ommAttachment.opacityMicromapArray = ommArray ? ommArray->GetGPUVirtualAddress() : NULL;
        vmTriangles.ommAttachment.opacityMicromapBaseLocation = 0;
        vmTriangles.ommAttachment.opacityMicromapIndexBuffer = {};

        size_t ommIndexOffset = inputs.buffers[(uint32_t)OmmDataLayout::Indices].offset;
        vmTriangles.ommAttachment.opacityMicromapIndexBuffer.StartAddress = ommIndexBuffer ? ommIndexBuffer->GetGPUVirtualAddress() + ommIndexOffset : NULL;
        vmTriangles.ommAttachment.opacityMicromapIndexBuffer.StrideInBytes = inputs.ommIndexStride;
        vmTriangles.ommAttachment.opacityMicromapIndexFormat = (DXGI_FORMAT)nri::ConvertNRIFormatToDXGI(inputs.ommIndexFormat);

        vmTriangles.ommAttachment.numOMMUsageCounts = inputs.indexHistogramNum;
        vmTriangles.ommAttachment.pOMMUsageCounts = (NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT*)inputs.indexHistogram;
        return geometryDescEx;
    }

    inline NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX FillDefaultBlasInputsDesc()
    {
        NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX inputDescEx = {};
        inputDescEx.type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        inputDescEx.flags = NVAPI_D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE_EX;
        inputDescEx.numDescs = 1;
        inputDescEx.descsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        inputDescEx.geometryDescStrideInBytes = sizeof(NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX);
        return inputDescEx;
    }

    inline static size_t Align(size_t size)
    {
        constexpr size_t a = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        return ((size + a - 1) / a) * a;
    }

    void OpacityMicroMapsHelper::ReleaseMemoryD3D12()
    {
        if (m_D3D12ScratchBuffer)
            m_D3D12ScratchBuffer->Release();
        m_D3D12ScratchBuffer = nullptr;

        for (auto& heap : m_D3D12GeometryHeaps)
            heap->Release();
        m_D3D12GeometryHeaps.clear();

        m_CurrentHeapOffset = 0;
    }

    void OpacityMicroMapsHelper::AllocateMemoryD3D12(uint64_t size)
    {
        m_D3D12GeometryHeaps.reserve(16);
        ID3D12Device5* device = GetD3D12Device5();
        ID3D12Heap*& newHeap = m_D3D12GeometryHeaps.emplace_back();
        D3D12_HEAP_DESC desc = {};
        desc.Properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        desc.SizeInBytes = size > m_DefaultHeapSize ? size : m_DefaultHeapSize;
        desc.SizeInBytes = m_D3D12ScratchBuffer ? desc.SizeInBytes : desc.SizeInBytes + m_SctrachSize;
        device->CreateHeap(&desc, IID_PPV_ARGS(&newHeap));
        m_CurrentHeapOffset = 0;

        if (!m_D3D12ScratchBuffer)
        {
            D3D12_RESOURCE_DESC resourceDesc = InitBufferResourceDesc(m_SctrachSize, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
            GetD3D12Device5()->CreatePlacedResource(m_D3D12GeometryHeaps.back(), 0, &resourceDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&m_D3D12ScratchBuffer));
            m_CurrentHeapOffset += Align(m_SctrachSize);
        }
    }

    void OpacityMicroMapsHelper::BindResourceToMemoryD3D12(ID3D12Resource*& resource, size_t size)
    {
        if (m_D3D12GeometryHeaps.empty() || (m_CurrentHeapOffset + size) > m_DefaultHeapSize)
            AllocateMemoryD3D12(size);

        ID3D12Heap* heap = m_D3D12GeometryHeaps.back();
        D3D12_RESOURCE_DESC resourceDesc = InitBufferResourceDesc(size, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        GetD3D12Device5()->CreatePlacedResource(heap, m_CurrentHeapOffset, &resourceDesc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&resource));
        m_CurrentHeapOffset += Align(size);
    }

    void OpacityMicroMapsHelper::GetPreBuildInfoD3D12(MaskedGeometryBuildDesc** queue, const size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            MaskedGeometryBuildDesc& desc = *queue[i];
            {// get omm prebuild info
                NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS vmInput = FillOmmArrayInputsDesc(desc.inputs, NULL, NULL);
                NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO ommPrebuildInfo = {};
                NVAPI_GET_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO_PARAMS ommGetPrebuildInfoParams = {};
                ommGetPrebuildInfoParams.pDesc = &vmInput;
                ommGetPrebuildInfoParams.pInfo = &ommPrebuildInfo;
                ommGetPrebuildInfoParams.version = NVAPI_GET_RAYTRACING_OPACITY_MICROMAP_ARRAY_PREBUILD_INFO_PARAMS_VER;

                _NvAPI_Status nvResult = NvAPI_D3D12_GetRaytracingOpacityMicromapArrayPrebuildInfo(GetD3D12Device5(), &ommGetPrebuildInfoParams);
                if (nvResult != NVAPI_OK)
                {
                    printf("[FAIL]: NvAPI_D3D12_GetRaytracingOpacityMicromapArrayPrebuildInfo\n");
                    std::abort();
                }
                desc.prebuildInfo.ommArraySize = ommPrebuildInfo.resultDataMaxSizeInBytes;
                desc.prebuildInfo.maxScratchDataSize = ommPrebuildInfo.scratchDataSizeInBytes;
            }

            {//get blas prebuild info
                nri::Buffer* nriOmmIndexData = desc.inputs.buffers[(uint32_t)OmmDataLayout::Indices].buffer;
                ID3D12Resource* ommIndexData = nriOmmIndexData ? (ID3D12Resource*)NRI.GetBufferNativeObject(*nriOmmIndexData, 0) : nullptr;
                NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX geometryDescEx = FillGeometryDescEx(desc.inputs, NULL, NULL, NULL, ommIndexData);

                NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX inputDescEx = FillDefaultBlasInputsDesc();
                inputDescEx.pGeometryDescs = &geometryDescEx;

                D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPrebuildInfo = {};
                NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX asInputs = {};
                NVAPI_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_EX_PARAMS blaskGetPrebuildInfoParams = {};
                blaskGetPrebuildInfoParams.pInfo = &blasPrebuildInfo;
                blaskGetPrebuildInfoParams.pDesc = &inputDescEx;
                blaskGetPrebuildInfoParams.version = NVAPI_GET_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO_EX_PARAMS_VER;

                NvAPI_Status nvapiStatus = NVAPI_OK;
                nvapiStatus = NvAPI_D3D12_GetRaytracingAccelerationStructurePrebuildInfoEx(GetD3D12Device5(), &blaskGetPrebuildInfoParams);
                if (nvapiStatus != NVAPI_OK)
                {
                    printf("[FAIL]: NvAPI_D3D12_GetRaytracingAccelerationStructurePrebuildInfoEx\n");
                    std::abort();
                }
                desc.prebuildInfo.blasSize = blasPrebuildInfo.ResultDataMaxSizeInBytes;
                desc.prebuildInfo.maxScratchDataSize = std::max(blasPrebuildInfo.ScratchDataSizeInBytes, desc.prebuildInfo.maxScratchDataSize);
            }
        }
    }

    void OpacityMicroMapsHelper::BuildOmmArrayD3D12(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer)
    {
        if (!desc.inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].buffer)
            return;

        ID3D12Resource* ommArrayData = (ID3D12Resource*)NRI.GetBufferNativeObject(*desc.inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].buffer, 0);
        ID3D12Resource* ommDescArray = (ID3D12Resource*)NRI.GetBufferNativeObject(*desc.inputs.buffers[(uint32_t)OmmDataLayout::DescArray].buffer, 0);

        NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_INPUTS vmInput = FillOmmArrayInputsDesc(desc.inputs, ommArrayData, ommDescArray);
        {
            ID3D12Resource* ommArrayBuffer = nullptr;
            BindResourceToMemoryD3D12(ommArrayBuffer, desc.prebuildInfo.ommArraySize);

            NVAPI_D3D12_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC vmArrayDesc = {};
            vmArrayDesc.destOpacityMicromapArrayData = ommArrayBuffer->GetGPUVirtualAddress();
            vmArrayDesc.inputs = vmInput;
            vmArrayDesc.scratchOpacityMicromapArrayData = m_D3D12ScratchBuffer->GetGPUVirtualAddress();

            NVAPI_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_PARAMS buildVmParams = {};
            buildVmParams.numPostbuildInfoDescs = 0;
            buildVmParams.pPostbuildInfoDescs = nullptr;
            buildVmParams.pDesc = &vmArrayDesc;
            buildVmParams.version = NVAPI_BUILD_RAYTRACING_OPACITY_MICROMAP_ARRAY_PARAMS_VER;

            NvAPI_Status nvapiStatus = NvAPI_D3D12_BuildRaytracingOpacityMicromapArray(GetD3D12GraphicsCommandList4(commandBuffer), &buildVmParams);
            if (nvapiStatus != NVAPI_OK)
            {
                printf("[FAIL]: NvAPI_D3D12_BuildRaytracingOpacityMicromapArray\n");
                std::abort();
            }
            D3D12_RESOURCE_BARRIER barriers[] = { InitUavBarrier(m_D3D12ScratchBuffer) };
            GetD3D12GraphicsCommandList4(commandBuffer)->ResourceBarrier(_countof(barriers), barriers);

            nri::BufferD3D12Desc wrappedBufferDesc = { ommArrayBuffer , 0 };
            NRI.CreateBufferD3D12(*m_Device, wrappedBufferDesc, desc.outputs.ommArray);
            ommArrayBuffer->Release();//dereference the resource to ensure it's destruction via NRI
        }
    }

    void OpacityMicroMapsHelper::BuildBlasD3D12(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer)
    {
        if (!desc.outputs.ommArray)
            return;

        ID3D12Resource* indexData = (ID3D12Resource*)NRI.GetBufferNativeObject(*desc.inputs.indices.nriBufferOrPtr.buffer, 0);
        ID3D12Resource* vertexData = (ID3D12Resource*)NRI.GetBufferNativeObject(*desc.inputs.vertices.nriBufferOrPtr.buffer, 0);
        ID3D12Resource* ommArray = (ID3D12Resource*)NRI.GetBufferNativeObject(*desc.outputs.ommArray, 0);
        ID3D12Resource* ommIndexData = (ID3D12Resource*)NRI.GetBufferNativeObject(*desc.inputs.buffers[(uint32_t)OmmDataLayout::Indices].buffer, 0);

        NVAPI_D3D12_RAYTRACING_GEOMETRY_DESC_EX geometryDescEx = FillGeometryDescEx(desc.inputs, indexData, vertexData, ommArray, ommIndexData);
        NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS_EX inputDescEx = FillDefaultBlasInputsDesc();
        inputDescEx.pGeometryDescs = &geometryDescEx;

        ID3D12Resource* blas = nullptr;
        BindResourceToMemoryD3D12(blas, desc.prebuildInfo.blasSize);
        {
            NVAPI_D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC_EX asDesc = {};
            asDesc.destAccelerationStructureData = blas->GetGPUVirtualAddress();
            asDesc.inputs = inputDescEx;
            asDesc.scratchAccelerationStructureData = m_D3D12ScratchBuffer->GetGPUVirtualAddress();

            NVAPI_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_EX_PARAMS asExParams = {};
            asExParams.numPostbuildInfoDescs = 0;
            asExParams.pPostbuildInfoDescs = nullptr;
            asExParams.pDesc = &asDesc;
            asExParams.version = NVAPI_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_EX_PARAMS_VER;

            NvAPI_Status nvapiStatus = NvAPI_D3D12_BuildRaytracingAccelerationStructureEx(GetD3D12GraphicsCommandList4(commandBuffer), &asExParams);
            if (nvapiStatus != NVAPI_OK)
            {
                printf("[FAIL]: NvAPI_D3D12_BuildRaytracingAccelerationStructureEx\n");
                std::abort();
            }
        }
        D3D12_RESOURCE_BARRIER barriers[] = { InitUavBarrier(m_D3D12ScratchBuffer) };
        GetD3D12GraphicsCommandList4(commandBuffer)->ResourceBarrier(_countof(barriers), barriers);

        nri::AccelerationStructureD3D12Desc asDesc = {};
        asDesc.d3d12Resource = blas;
        asDesc.scratchDataSizeInBytes = desc.prebuildInfo.maxScratchDataSize;
        asDesc.updateScratchDataSizeInBytes = desc.prebuildInfo.maxScratchDataSize;
        NRI.CreateAccelerationStructureD3D12(*m_Device, asDesc, desc.outputs.blas);
        blas->Release();//dereference the resource to ensure it's destruction via NRI
    }

    void OpacityMicroMapsHelper::BuildMaskedGeometryD3D12(MaskedGeometryBuildDesc** queue, const size_t count, nri::CommandBuffer* commandBuffer)
    {
        GetPreBuildInfoD3D12(queue, count);

        for (size_t i = 0; i < count; ++i)
        {//build omm then blas to increase memory locality
            BuildOmmArrayD3D12(*queue[i], commandBuffer);
            BuildBlasD3D12(*queue[i], commandBuffer);
        }
    }
}
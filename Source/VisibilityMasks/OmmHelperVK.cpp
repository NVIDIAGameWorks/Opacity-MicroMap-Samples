/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmHelper.h"
#include <cassert>

#define VK_CALL(result)\
    if(result != VK_SUCCESS)\
        { printf("[FAIL] {%s}", #result); std::abort(); };

#define GET_VK_FUNC(getter, source, name) (PFN_vk ## name)getter(source, "vk" #name)

#define INIT_VK_FUNC(getter, source, name)\
    VK.name = GET_VK_FUNC(getter, source, name);\
    if(VK.name == nullptr)\
        { printf("[FAIL] Unable to get VK device function: {%s}", "vk" #name ); std::abort(); };

#define DECLARE_VK_FUNC(name) PFN_vk ## name name = nullptr;

constexpr uint64_t VK_PLACEMENT_ALIGNMENT = 256;

namespace ommhelper
{
    struct VkInterface
    {
        DECLARE_VK_FUNC(GetMicromapBuildSizesEXT);
        DECLARE_VK_FUNC(GetAccelerationStructureBuildSizesKHR);
        DECLARE_VK_FUNC(CmdBuildMicromapsEXT);
        DECLARE_VK_FUNC(CmdBuildAccelerationStructuresKHR);
        DECLARE_VK_FUNC(CreateBuffer);
        DECLARE_VK_FUNC(GetBufferMemoryRequirements);
        DECLARE_VK_FUNC(GetPhysicalDeviceMemoryProperties);
        DECLARE_VK_FUNC(AllocateMemory);
        DECLARE_VK_FUNC(DestroyBuffer);
        DECLARE_VK_FUNC(FreeMemory);
        DECLARE_VK_FUNC(BindBufferMemory);
        DECLARE_VK_FUNC(CreateMicromapEXT);
        DECLARE_VK_FUNC(GetBufferDeviceAddress);
        DECLARE_VK_FUNC(CreateAccelerationStructureKHR);
        DECLARE_VK_FUNC(GetAccelerationStructureDeviceAddressKHR);
        DECLARE_VK_FUNC(CmdPipelineBarrier);
        DECLARE_VK_FUNC(DestroyMicromapEXT);
    } VK = {};

    inline VkDevice OpacityMicroMapsHelper::GetVkDevice()
    {
        return (VkDevice)NRI.GetDeviceNativeObject(*m_Device);
    }

    inline VkIndexType GetVkIndexType(nri::Format format)
    {
        switch (format)
        {
        case nri::Format::R32_UINT: return VkIndexType::VK_INDEX_TYPE_UINT32;
        case nri::Format::R16_UINT: return VkIndexType::VK_INDEX_TYPE_UINT16;
        default: return VkIndexType::VK_INDEX_TYPE_NONE_KHR;
        }
    }

    inline VkBufferDeviceAddressInfo GetBufferAddressInfo(VkBuffer& buffer)
    {
        VkBufferDeviceAddressInfo bufferAddressInfo = { VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO };
        bufferAddressInfo.pNext = nullptr;
        bufferAddressInfo.buffer = buffer;
        return bufferAddressInfo;
    }

    void OpacityMicroMapsHelper::InitializeVK()
    {
        VkInstance vkInstance = (VkInstance)NRI.GetVkInstance(*m_Device);
        VkDevice vkDevice = (VkDevice)NRI.GetDeviceNativeObject(*m_Device);

        { // Get required vk function pointers
            PFN_vkGetDeviceProcAddr getDeviceProcAddr = (PFN_vkGetDeviceProcAddr)NRI.GetVkGetDeviceProcAddr(*m_Device);
            PFN_vkGetInstanceProcAddr getInstanceProcAddr = (PFN_vkGetInstanceProcAddr)NRI.GetVkGetInstanceProcAddr(*m_Device);

            INIT_VK_FUNC(getInstanceProcAddr, vkInstance, GetPhysicalDeviceMemoryProperties);

            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, GetMicromapBuildSizesEXT);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, CreateMicromapEXT);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, CmdBuildMicromapsEXT);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, DestroyMicromapEXT);

            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, GetAccelerationStructureBuildSizesKHR);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, CreateAccelerationStructureKHR);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, GetAccelerationStructureDeviceAddressKHR);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, CmdBuildAccelerationStructuresKHR);

            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, AllocateMemory);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, FreeMemory);

            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, CreateBuffer);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, GetBufferMemoryRequirements);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, BindBufferMemory);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, GetBufferDeviceAddress);
            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, DestroyBuffer);

            INIT_VK_FUNC(getDeviceProcAddr, vkDevice, CmdPipelineBarrier);
        }

        {//create a buffer with properties required to store ommArrays, blases and scratch buffer to query memory type for future allocations
            VkBufferCreateInfo bufferDesc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            bufferDesc.pNext = NULL;
            bufferDesc.size = m_DefaultHeapSize;
            bufferDesc.flags = 0;
            bufferDesc.usage = VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

            VkBuffer buffer = NULL;
            VK_CALL(VK.CreateBuffer(GetVkDevice(), &bufferDesc, nullptr, &buffer));

            VkMemoryRequirements memoryRequirments = {};
            VK.GetBufferMemoryRequirements(GetVkDevice(), buffer, &memoryRequirments);

            VkPhysicalDeviceMemoryProperties memoryProperties = {};
            VK.GetPhysicalDeviceMemoryProperties((VkPhysicalDevice)NRI.GetVkPhysicalDevice(*m_Device), &memoryProperties);

            const uint32_t memProperty = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
            for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
            {
                if ((memoryRequirments.memoryTypeBits & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & memProperty) == memProperty)
                {
                    m_VkMemoryTypeId = i;
                    break;
                }
            }
            VK.DestroyBuffer(GetVkDevice(), buffer, nullptr);

            if (m_VkMemoryTypeId == uint32_t(-1))
            {
                printf("[FAIL] Current device doesn't support requested memory type\n");
                std::abort();
            }
        }
    }

    void OpacityMicroMapsHelper::DestroyOmmArrayVK(nri::Buffer* ommArray)
    {
        VkMicromapEXT micromap = reinterpret_cast<VkMicromapEXT>(ommArray);
        VK.DestroyMicromapEXT(GetVkDevice(), micromap, nullptr);
    }

    void OpacityMicroMapsHelper::ReleaseMemoryVK()
    {
        if (m_VkScrathBuffer)
            VK.DestroyBuffer(GetVkDevice(), m_VkScrathBuffer, nullptr);
        m_VkScrathBuffer = NULL;

        for (auto& buffer : m_VkBuffers)
            VK.DestroyBuffer(GetVkDevice(), buffer, nullptr);
        m_VkBuffers.clear();

        for(auto& memory : m_VkMemories)
            VK.FreeMemory(GetVkDevice(), memory, nullptr);

        m_VkMemories.clear();
        m_CurrentHeapOffset = 0;
    }

    void OpacityMicroMapsHelper::AllocateMemoryVK(uint64_t size)
    {
        m_CurrentHeapOffset = 0;
        VkDeviceMemory& newMemory = m_VkMemories.emplace_back();

        uint64_t allocationSize = (size > m_DefaultHeapSize) ? size : m_DefaultHeapSize; //make the initial heap bigger to store the scratch resource
        allocationSize = (m_VkScrathBuffer == NULL) ? allocationSize + m_SctrachSize : allocationSize; //make the initial heap bigger to store the scratch resource

        VkMemoryAllocateFlagsInfo flagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
        flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT | VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.allocationSize = allocationSize;
        allocInfo.memoryTypeIndex = m_VkMemoryTypeId;
        allocInfo.pNext = &flagsInfo;

        for (uint32_t i = 0; i < NRI.GetDeviceDesc(*m_Device).physicalDeviceNum; ++i)
        {
            flagsInfo.deviceMask = 1 << i;
            VK_CALL(VK.AllocateMemory(GetVkDevice(), &allocInfo, nullptr, &newMemory));
        }

        if (m_VkScrathBuffer == NULL)
        {//create scratch buffer and put it in the end of the heap
            VkBufferCreateInfo scratchDesc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            scratchDesc.pNext = NULL;
            scratchDesc.size = m_SctrachSize;
            scratchDesc.flags = 0;
            scratchDesc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            VK_CALL(VK.CreateBuffer(GetVkDevice(), &scratchDesc, nullptr, &m_VkScrathBuffer));
            VK_CALL(VK.BindBufferMemory(GetVkDevice(), m_VkScrathBuffer, newMemory, m_DefaultHeapSize));
        }

        VkBufferCreateInfo bufferDesc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferDesc.pNext = NULL;
        bufferDesc.size = m_DefaultHeapSize;
        bufferDesc.flags = 0;
        bufferDesc.usage = VK_BUFFER_USAGE_MICROMAP_STORAGE_BIT_EXT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

        VkBuffer& buffer = m_VkBuffers.emplace_back();
        VK_CALL(VK.CreateBuffer(GetVkDevice(), &bufferDesc, nullptr, &buffer));
        VK_CALL(VK.BindBufferMemory(GetVkDevice(), buffer, newMemory, 0));
    }

    inline static uint64_t Align(uint64_t s, uint64_t a)
    {
        return ((s + a - 1) / a) * a;
    }

    inline VkMicromapBuildInfoEXT FillMicromapBuildInfo(const MaskedGeometryBuildDesc::Inputs& inputs, VkMicromapEXT micromap,  VkDeviceAddress arrayDataAddress, VkDeviceAddress descArrayAddress, VkDeviceAddress scratch)
    {
        VkMicromapBuildInfoEXT buildDesc = { VK_STRUCTURE_TYPE_MICROMAP_BUILD_INFO_EXT };
        buildDesc.pNext = nullptr;
        buildDesc.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
        buildDesc.mode = VK_BUILD_MICROMAP_MODE_BUILD_EXT;
        buildDesc.dstMicromap = micromap;
        buildDesc.usageCountsCount = inputs.descArrayHistogramNum;
        buildDesc.pUsageCounts = (VkMicromapUsageEXT*)inputs.descArrayHistogram;
        buildDesc.data.deviceAddress = arrayDataAddress;
        buildDesc.scratchData.deviceAddress = scratch;
        buildDesc.triangleArray.deviceAddress = descArrayAddress;
        buildDesc.triangleArrayStride = sizeof(VkMicromapTriangleEXT);
        return buildDesc;
    }

    inline VkAccelerationStructureTrianglesOpacityMicromapEXT FillOmmTrianglesDesc(const MaskedGeometryBuildDesc& desc, VkDeviceAddress ommIndicesAddress)
    {
        VkAccelerationStructureTrianglesOpacityMicromapEXT ommBlasDesc = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_TRIANGLES_OPACITY_MICROMAP_EXT };
        ommBlasDesc.pNext = nullptr;
        ommBlasDesc.indexType = GetVkIndexType(desc.inputs.ommIndexFormat);
        ommBlasDesc.indexBuffer.deviceAddress = ommIndicesAddress;
        ommBlasDesc.indexStride = desc.inputs.ommIndexStride;
        ommBlasDesc.baseTriangle = 0;
        ommBlasDesc.usageCountsCount = desc.inputs.indexHistogramNum;
        ommBlasDesc.pUsageCounts = (VkMicromapUsageEXT*)desc.inputs.indexHistogram;
        ommBlasDesc.micromap = reinterpret_cast<VkMicromapEXT>(desc.outputs.ommArray);
        return ommBlasDesc;
    }

    inline VkAccelerationStructureGeometryKHR FillGeometryDesc(const MaskedGeometryBuildDesc& desc, VkAccelerationStructureTrianglesOpacityMicromapEXT* ommTriangles, VkDeviceAddress indices, VkDeviceAddress vertices)
    {
        VkAccelerationStructureGeometryKHR geometryDesc = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR };
        geometryDesc.pNext = nullptr;
        geometryDesc.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometryDesc.flags = 0;

        VkAccelerationStructureGeometryTrianglesDataKHR& triangles = geometryDesc.geometry.triangles;
        triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        triangles.pNext = ommTriangles;
        triangles.indexData.deviceAddress = indices;
        triangles.indexType = GetVkIndexType(desc.inputs.indices.format);
        triangles.maxVertex = (uint32_t)desc.inputs.vertices.numElements;
        triangles.vertexData.deviceAddress = vertices;
        triangles.vertexFormat = (VkFormat)nri::ConvertNRIFormatToVK(desc.inputs.vertices.format);
        triangles.vertexStride = desc.inputs.vertices.stride;
        triangles.transformData.hostAddress = nullptr;
        return geometryDesc;
    }

    inline VkAccelerationStructureBuildGeometryInfoKHR FillBlasBuildInfo(VkAccelerationStructureKHR blas, VkAccelerationStructureGeometryKHR& geometryDesc, VkDeviceAddress scratch)
    {
        VkAccelerationStructureBuildGeometryInfoKHR blasDesc = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR };
        blasDesc.pNext = nullptr;
        blasDesc.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        blasDesc.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        blasDesc.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        blasDesc.srcAccelerationStructure = NULL;
        blasDesc.dstAccelerationStructure = blas;
        blasDesc.geometryCount = 1;
        blasDesc.pGeometries = &geometryDesc;
        blasDesc.scratchData.deviceAddress = scratch;
        return blasDesc;
    }

    void OpacityMicroMapsHelper::GetPreBuildInfoVK(MaskedGeometryBuildDesc** queue, const size_t count)
    {
        if (count == 0)
            return;

        uint64_t maxMicromapSize = 0;
        uint64_t maxScratchSize = 0;
        for (size_t i = 0; i < count; ++i)
        {
            MaskedGeometryBuildDesc& desc = *queue[i];
            const MaskedGeometryBuildDesc::Inputs& inputs = desc.inputs;
            { // Get OMM array prebuid info
                VkMicromapBuildInfoEXT buildDesc = FillMicromapBuildInfo(inputs, NULL, NULL, NULL, NULL);
                VkMicromapBuildSizesInfoEXT preBuildInfo = { VK_STRUCTURE_TYPE_MICROMAP_BUILD_SIZES_INFO_EXT };
                VK.GetMicromapBuildSizesEXT(GetVkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildDesc, &preBuildInfo);

                desc.prebuildInfo.ommArraySize = preBuildInfo.micromapSize;
                desc.prebuildInfo.maxScratchDataSize = preBuildInfo.buildScratchSize;
                maxMicromapSize = std::max(preBuildInfo.micromapSize, maxMicromapSize);
                maxScratchSize = std::max(preBuildInfo.buildScratchSize, maxScratchSize);
            }
        }

        VkDeviceMemory tmpMemory = {};
        VkBuffer scratch = {};
        VkBuffer tmpOmmBuffer = {};
        { // Perform temporal allocation to store empty micromaps. This is required to get correct blas size
            uint64_t allocationSize = Align(maxMicromapSize, VK_PLACEMENT_ALIGNMENT) + Align(maxScratchSize, VK_PLACEMENT_ALIGNMENT);
            VkMemoryAllocateFlagsInfo flagsInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
            flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT | VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;

            VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            allocInfo.allocationSize = allocationSize;
            allocInfo.memoryTypeIndex = m_VkMemoryTypeId;
            allocInfo.pNext = &flagsInfo;
            for (uint32_t i = 0; i < NRI.GetDeviceDesc(*m_Device).physicalDeviceNum; ++i)
            {
                flagsInfo.deviceMask = 1 << i;
                VK_CALL(VK.AllocateMemory(GetVkDevice(), &allocInfo, nullptr, &tmpMemory));
            }

            VkBufferCreateInfo desc = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            desc.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
            desc.size = maxMicromapSize;
            VK_CALL(VK.CreateBuffer(GetVkDevice(), &desc, nullptr, &tmpOmmBuffer));
            VK_CALL(VK.BindBufferMemory(GetVkDevice(), tmpOmmBuffer, tmpMemory, 0));

            desc.size = maxScratchSize;
            VK_CALL(VK.CreateBuffer(GetVkDevice(), &desc, nullptr, &scratch));
            VK_CALL(VK.BindBufferMemory(GetVkDevice(), scratch, tmpMemory, Align(maxMicromapSize, VK_PLACEMENT_ALIGNMENT)));
        }

        for (size_t i = 0; i < count; ++i)
        { // Get BLAS prebuild info
            MaskedGeometryBuildDesc& desc = *queue[i];
            const MaskedGeometryBuildDesc::Inputs& inputs = desc.inputs;
            const GpuBakerBuffer* buffers = inputs.buffers;
            {
                nri::Buffer* nriOmmIndices = buffers[(uint32_t)OmmDataLayout::Indices].buffer;

                VkDeviceAddress ommIndicesAddress = NULL;
                if (nriOmmIndices)
                {
                    VkBuffer ommIndices = nriOmmIndices ? (VkBuffer)NRI.GetBufferNativeObject(*nriOmmIndices, nri::WHOLE_DEVICE_GROUP) : NULL;
                    VkBufferDeviceAddressInfo ommIndicesbufferAddressInfo = GetBufferAddressInfo(ommIndices);
                    ommIndicesAddress = ommIndices ? VK.GetBufferDeviceAddress(GetVkDevice(), &ommIndicesbufferAddressInfo) : NULL;
                    ommIndicesAddress += buffers[(uint32_t)OmmDataLayout::Indices].offset;
                }

                VkBuffer indices = (VkBuffer)NRI.GetBufferNativeObject(*inputs.indices.nriBufferOrPtr.buffer, nri::WHOLE_DEVICE_GROUP);
                VkBuffer vertices = (VkBuffer)NRI.GetBufferNativeObject(*inputs.vertices.nriBufferOrPtr.buffer, nri::WHOLE_DEVICE_GROUP);

                VkBufferDeviceAddressInfo indicesBufferAddressInfo = GetBufferAddressInfo(indices);
                VkBufferDeviceAddressInfo verticesBufferAddressInfo = GetBufferAddressInfo(vertices);

                VkDeviceAddress indicesAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &indicesBufferAddressInfo) + inputs.indices.offset;
                VkDeviceAddress verticesAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &verticesBufferAddressInfo) + inputs.vertices.offset;

                VkAccelerationStructureTrianglesOpacityMicromapEXT ommBlasDesc = FillOmmTrianglesDesc(desc, ommIndicesAddress);
                VkMicromapEXT tmpMicromap = {};
                { // No need to build the micromap. just allocate...
                    VkMicromapCreateInfoEXT ommArrayDesc = { VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT };
                    ommArrayDesc.buffer = tmpOmmBuffer;
                    ommArrayDesc.size = uint64_t(desc.prebuildInfo.ommArraySize);
                    ommArrayDesc.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
                    VK_CALL(VK.CreateMicromapEXT(GetVkDevice(), &ommArrayDesc, nullptr, &tmpMicromap));
                }
                ommBlasDesc.micromap = tmpMicromap;

                VkAccelerationStructureGeometryKHR geometryDesc = FillGeometryDesc(desc, &ommBlasDesc, indicesAddress, verticesAddress);
                VkAccelerationStructureBuildGeometryInfoKHR blasDesc = FillBlasBuildInfo(NULL, geometryDesc, NULL);

                uint32_t maxPrimitiveCount = uint32_t(inputs.indices.numElements / 3);
                VkAccelerationStructureBuildSizesInfoKHR preBuildInfo = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR };
                VK.GetAccelerationStructureBuildSizesKHR(GetVkDevice(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasDesc, &maxPrimitiveCount, &preBuildInfo);

                desc.prebuildInfo.blasSize = preBuildInfo.accelerationStructureSize;
                desc.prebuildInfo.maxScratchDataSize = std::max(preBuildInfo.buildScratchSize, desc.prebuildInfo.maxScratchDataSize);
                VK.DestroyMicromapEXT(GetVkDevice(), tmpMicromap, nullptr);
            }
        }
        VK.DestroyBuffer(GetVkDevice(), scratch, nullptr);
        VK.DestroyBuffer(GetVkDevice(), tmpOmmBuffer, nullptr);
        VK.FreeMemory(GetVkDevice(), tmpMemory, nullptr);
    }

    inline void InsertUavBarrier(VkCommandBuffer commandBuffer, VkBuffer buffer, uint64_t size, uint64_t offset)
    {
        VkBufferMemoryBarrier barrier = { VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER };
        barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = (~0U);
        barrier.dstQueueFamilyIndex = (~0U);
        barrier.buffer = buffer;
        barrier.offset = offset;
        barrier.size = size;

        uint32_t stageBit = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        VK.CmdPipelineBarrier(commandBuffer, stageBit, stageBit, 0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    void OpacityMicroMapsHelper::BindOmmToMemoryVK(VkMicromapEXT& ommArray, size_t size)
    {
        if (m_VkMemories.empty() || m_CurrentHeapOffset + size > m_DefaultHeapSize)
            AllocateMemoryVK(size);

        VkBuffer& currentBuffer = m_VkBuffers.back();

        VkMicromapCreateInfoEXT ommArrayDesc = { VK_STRUCTURE_TYPE_MICROMAP_CREATE_INFO_EXT };
        ommArrayDesc.pNext = nullptr;
        ommArrayDesc.createFlags = 0;
        ommArrayDesc.buffer = currentBuffer;
        ommArrayDesc.offset = m_CurrentHeapOffset;
        assert(ommArrayDesc.offset % VK_PLACEMENT_ALIGNMENT == 0);
        ommArrayDesc.size = uint64_t(size);
        ommArrayDesc.type = VK_MICROMAP_TYPE_OPACITY_MICROMAP_EXT;
        ommArrayDesc.deviceAddress = 0;
        VK_CALL(VK.CreateMicromapEXT(GetVkDevice(), &ommArrayDesc, nullptr, &ommArray));
        uint64_t alignedOffset = Align(size, VK_PLACEMENT_ALIGNMENT);
        m_CurrentHeapOffset += alignedOffset;
    }

    void OpacityMicroMapsHelper::BuildOmmArrayVK(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer)
    {
        if (!desc.inputs.buffers[(uint32_t)OmmDataLayout::ArrayData].buffer)
            return;

        VkMicromapEXT ommArray = {};
        BindOmmToMemoryVK(ommArray, desc.prebuildInfo.ommArraySize);

        const MaskedGeometryBuildDesc::Inputs& inputs = desc.inputs;
        const GpuBakerBuffer* buffers = inputs.buffers;

        VkBuffer ommArrayData = (VkBuffer)NRI.GetBufferNativeObject(*buffers[(uint32_t)OmmDataLayout::ArrayData].buffer, nri::WHOLE_DEVICE_GROUP);
        VkBuffer ommDescArray = (VkBuffer)NRI.GetBufferNativeObject(*buffers[(uint32_t)OmmDataLayout::DescArray].buffer, nri::WHOLE_DEVICE_GROUP);

        uint64_t ommArrayDataOffset = buffers[(uint32_t)OmmDataLayout::ArrayData].offset;
        uint64_t ommDescArrayOffset = buffers[(uint32_t)OmmDataLayout::DescArray].offset;

        VkBufferDeviceAddressInfo ommArrayDataAddressInfo = GetBufferAddressInfo(ommArrayData);
        VkBufferDeviceAddressInfo ommDescArrayAddressInfo = GetBufferAddressInfo(ommDescArray);
        VkBufferDeviceAddressInfo scratchAddressInfo = GetBufferAddressInfo(m_VkScrathBuffer);

        VkDeviceAddress ommArrayDataAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &ommArrayDataAddressInfo) + ommArrayDataOffset;
        VkDeviceAddress ommDescArrayAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &ommDescArrayAddressInfo) + ommDescArrayOffset;
        VkDeviceAddress scratchAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &scratchAddressInfo);

        VkMicromapBuildInfoEXT buildDesc = FillMicromapBuildInfo(inputs, ommArray, ommArrayDataAddress, ommDescArrayAddress, scratchAddress);

        VkCommandBuffer vkCommandBuffer = (VkCommandBuffer)NRI.GetCommandBufferNativeObject(*commandBuffer);
        VK.CmdBuildMicromapsEXT(vkCommandBuffer, 1, &buildDesc);
        InsertUavBarrier(vkCommandBuffer, m_VkScrathBuffer, m_SctrachSize, 0);

        desc.outputs.ommArray = reinterpret_cast<nri::Buffer*>(ommArray);
    }

    void OpacityMicroMapsHelper::BindBlasToMemoryVK(VkAccelerationStructureKHR& blas, size_t size)
    {
        if (m_VkMemories.empty() || m_CurrentHeapOffset + size > m_DefaultHeapSize)
            AllocateMemoryVK(size);

        VkBuffer& currentBuffer = m_VkBuffers.back();
        VkAccelerationStructureCreateInfoKHR blasDesc = { VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR };
        blasDesc.pNext = nullptr;
        blasDesc.buffer = currentBuffer;
        blasDesc.offset = m_CurrentHeapOffset;
        assert(blasDesc.offset % VK_PLACEMENT_ALIGNMENT == 0);
        blasDesc.size = size;
        blasDesc.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VK.CreateAccelerationStructureKHR(GetVkDevice(), &blasDesc, nullptr, &blas);
        uint64_t alignedOffset = Align(size, VK_PLACEMENT_ALIGNMENT);
        m_CurrentHeapOffset += alignedOffset;
    }

    void OpacityMicroMapsHelper::BuildBlasVK(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer)
    {
        if (!desc.outputs.ommArray)
            return;

        const MaskedGeometryBuildDesc::Inputs& inputs = desc.inputs;
        const GpuBakerBuffer* buffers = inputs.buffers;

        nri::Buffer* nriOmmIndices = buffers[(uint32_t)OmmDataLayout::Indices].buffer;
        VkBuffer ommIndices = (VkBuffer)NRI.GetBufferNativeObject(*nriOmmIndices, nri::WHOLE_DEVICE_GROUP);
        VkBufferDeviceAddressInfo ommIndicesAddressInfo = GetBufferAddressInfo(ommIndices);
        VkDeviceAddress ommIndicesAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &ommIndicesAddressInfo);
        ommIndicesAddress += buffers[(uint32_t)OmmDataLayout::Indices].offset;

        VkBuffer indices = (VkBuffer)NRI.GetBufferNativeObject(*inputs.indices.nriBufferOrPtr.buffer, nri::WHOLE_DEVICE_GROUP);
        VkBuffer vertices = (VkBuffer)NRI.GetBufferNativeObject(*inputs.vertices.nriBufferOrPtr.buffer, nri::WHOLE_DEVICE_GROUP);

        VkBufferDeviceAddressInfo indicesAddressInfo = GetBufferAddressInfo(indices);
        VkBufferDeviceAddressInfo verticesAddressInfo = GetBufferAddressInfo(vertices);
        VkBufferDeviceAddressInfo scratchAddressInfo = GetBufferAddressInfo(m_VkScrathBuffer);

        VkDeviceAddress indicesAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &indicesAddressInfo) + inputs.indices.offset;
        VkDeviceAddress verticesAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &verticesAddressInfo) + inputs.vertices.offset;
        VkDeviceAddress scratchAddress = VK.GetBufferDeviceAddress(GetVkDevice(), &scratchAddressInfo);

        VkAccelerationStructureKHR blas = {};
        BindBlasToMemoryVK(blas, desc.prebuildInfo.blasSize);

        VkAccelerationStructureTrianglesOpacityMicromapEXT ommBlasDesc = FillOmmTrianglesDesc(desc, ommIndicesAddress);
        VkAccelerationStructureGeometryKHR geometryDesc = FillGeometryDesc(desc, &ommBlasDesc, indicesAddress, verticesAddress);
        VkAccelerationStructureBuildGeometryInfoKHR blasDesc = FillBlasBuildInfo(blas, geometryDesc, scratchAddress);

        VkAccelerationStructureBuildRangeInfoKHR range = {};
        range.primitiveCount = uint32_t(inputs.indices.numElements / 3);
        const VkAccelerationStructureBuildRangeInfoKHR* rangeArrays[1] = { &range };

        VkCommandBuffer vkCommandBuffer = (VkCommandBuffer)NRI.GetCommandBufferNativeObject(*commandBuffer);
        VK.CmdBuildAccelerationStructuresKHR(vkCommandBuffer, 1, &blasDesc, rangeArrays); // Known issue: Vulkan Debug Layer crashes here
        InsertUavBarrier(vkCommandBuffer, m_VkScrathBuffer, m_SctrachSize, 0);

        nri::AccelerationStructureVulkanDesc wrapperDesc = {};
        wrapperDesc.buildScratchSize = desc.prebuildInfo.maxScratchDataSize;
        wrapperDesc.physicalDeviceMask = nri::WHOLE_DEVICE_GROUP;
        wrapperDesc.updateScratchSize = 0;
        wrapperDesc.vkAccelerationStructure = (nri::NRIVkAccelerationStructureKHR)blas;
        NRI.CreateAccelerationStructureVK(*m_Device, wrapperDesc, desc.outputs.blas);
    }

    void OpacityMicroMapsHelper::BuildMaskedGeometryVK(MaskedGeometryBuildDesc** queue, const size_t count, nri::CommandBuffer* commandBuffer)
    {
        GetPreBuildInfoVK(queue, count);

        for (size_t i = 0; i < count; ++i)
        { // Build omm then blas to increase memory locality
            BuildOmmArrayVK(*queue[i], commandBuffer);
            BuildBlasVK(*queue[i], commandBuffer);
        }
    }
}


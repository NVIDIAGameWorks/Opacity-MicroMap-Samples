/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmHelper.h"
#include <filesystem>
#include "ImGui/imgui.h"

namespace ommhelper
{
    void OpacityMicroMapsHelper::Initialize(nri::Device* device)
    {
        m_Device = device;
        if (m_Device)
        {
            uint32_t nriResult = (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
            nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI);
            nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI);

            if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
            {
                nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::WrapperD3D12Interface), (nri::WrapperD3D12Interface*)&NRI);
                InitializeD3D12();
            }
            else if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VULKAN)
            {
                nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::WrapperVKInterface), (nri::WrapperVKInterface*)&NRI);
                InitializeVK();
            }
            else
            {
                printf("[FAIL]: Unsupported Graphics API\n");
                std::abort();
            }

            m_GpuBakerIntegration.Initialize(*m_Device);

            omm::BakerCreationDesc desc = {};
            desc.enableValidation = false;
            desc.type = omm::BakerType::CPU;
            if (omm::CreateOpacityMicromapBaker(desc, &m_OmmCpuBaker) != omm::Result::SUCCESS)
            {
                printf("[FAIL]: omm::CreateOpacityMicromapBaker\n");
                std::abort();
            }
        }
    }

    void OpacityMicroMapsHelper::Destroy()
    {
        m_GpuBakerIntegration.Destroy();
        omm::DestroyOpacityMicromapBaker(m_OmmCpuBaker);
        ReleaseGeometryMemory();
        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
            NvAPI_Unload();
    }

#pragma region [ Utils ]

    inline omm::Cpu::TextureFormat GetOmmBakerTextureFormat(nri::Format format)
    {
        switch (format)
        {
        case nri::Format::R32_SFLOAT: return omm::Cpu::TextureFormat::FP32;
        default: printf("[FAIL] Unknown texture format passed to Cpu Baker!\n"); std::abort();
        }
    }

    inline omm::IndexFormat GetOmmBakerIndexFormat(nri::Format format)
    {
        switch (format)
        {
        case nri::Format::R16_UINT: return omm::IndexFormat::I16_UINT;
        case nri::Format::R32_UINT: return omm::IndexFormat::I32_UINT;
        default: printf("[FAIL] Unknown index format passed to Cpu Baker!\n"); std::abort();
        }
    }

    inline omm::TexCoordFormat GetOmmBakerUvFormat(nri::Format format)
    {
        switch (format)
        {
        case nri::Format::RG16_SFLOAT: return omm::TexCoordFormat::UV16_FLOAT;
        case nri::Format::RG32_SFLOAT: return omm::TexCoordFormat::UV32_FLOAT;
        case nri::Format::RG16_UNORM: return omm::TexCoordFormat::UV16_UNORM;
        default: printf("[FAIL] Unknown UV format passed to Cpu Baker!\n"); std::abort();
        }
    }

    inline omm::OMMFormat GetOmmFormat(OmmFormats format)
    {
        switch (format)
        {
        case OmmFormats::OC1_2_STATE: return omm::OMMFormat::OC1_2_State;
        case OmmFormats::OC1_4_STATE: return omm::OMMFormat::OC1_4_State;
        default: printf("[FAIL] Unknown OMM format passed to Cpu Baker!\n"); std::abort();
        }
    }

    inline nri::Format GetNriIndexFormat(omm::IndexFormat format)
    {
        switch (format)
        {
        case omm::IndexFormat::I16_UINT: return nri::Format::R16_UINT;
        case omm::IndexFormat::I32_UINT: return nri::Format::R32_UINT;
        default: printf("[FAIL] Unknown Index format returned from Cpu Baker!\n"); std::abort();
        }
    }

    inline omm::TextureAddressMode GetOmmAddressingMode(nri::AddressMode mode)
    {
        switch (mode)
        {
        case nri::AddressMode::REPEAT: return omm::TextureAddressMode::Wrap;
        case nri::AddressMode::MIRRORED_REPEAT: return omm::TextureAddressMode::Mirror;
        case nri::AddressMode::CLAMP_TO_EDGE: return omm::TextureAddressMode::Clamp;
        case nri::AddressMode::CLAMP_TO_BORDER: return omm::TextureAddressMode::Border;
        default: printf("[FAIL] Ivalid AddressMode passed to Cpu Baker!\n"); std::abort();
        }
    }

    void OpacityMicroMapsHelper::ConvertUsageCountsToApiFormat(uint8_t* outFormattedBuffer, size_t& outSize, const uint8_t* bakerOutputBuffer, size_t bakerOutputBufferSize)
    {
        size_t stride = 0;
        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
        {
            stride = sizeof(_NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT);

            size_t countsNum = bakerOutputBufferSize / sizeof(omm::Cpu::OpacityMicromapUsageCount);
            outSize = countsNum * stride;

            if (!outFormattedBuffer)
                return;

            std::vector<_NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT> sanitizedUsageCounts;
            omm::Cpu::OpacityMicromapUsageCount* ommData = (omm::Cpu::OpacityMicromapUsageCount*)bakerOutputBuffer;
            for (size_t i = 0; i < countsNum; ++i)
            {
                _NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_USAGE_COUNT usageDesc = { ommData[i].count, ommData[i].subdivisionLevel, (NVAPI_D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT)ommData[i].format };
                sanitizedUsageCounts.push_back(usageDesc);
            }
            memcpy(outFormattedBuffer, sanitizedUsageCounts.data(), outSize);
        }
        else
        {
            stride = sizeof(VkMicromapUsageEXT);

            size_t countsNum = bakerOutputBufferSize / sizeof(omm::Cpu::OpacityMicromapUsageCount);
            outSize = countsNum * stride;

            if (!outFormattedBuffer)
                return;

            std::vector<VkMicromapUsageEXT> sanitizedUsageCounts;
            omm::Cpu::OpacityMicromapUsageCount* ommData = (omm::Cpu::OpacityMicromapUsageCount*)bakerOutputBuffer;
            for (size_t i = 0; i < countsNum; ++i)
            {
                VkMicromapUsageEXT usageDesc = { ommData[i].count, ommData[i].subdivisionLevel, (uint32_t)ommData[i].format };
                sanitizedUsageCounts.push_back(usageDesc);
            }
            memcpy(outFormattedBuffer, sanitizedUsageCounts.data(), outSize);
        }
    }

    void OpacityMicroMapsHelper::DestroyMaskedGeometry(nri::AccelerationStructure* blas, nri::Buffer* ommArray)
    {
        if (blas)
            NRI.DestroyAccelerationStructure(*blas);

        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
        {
            if (ommArray)
                NRI.DestroyBuffer(*ommArray);
        }
        else
            DestroyOmmArrayVK(ommArray);
    }

    void OpacityMicroMapsHelper::ReleaseGeometryMemory()
    {
        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
            ReleaseMemoryD3D12();
        else
            ReleaseMemoryVK();
    }

#pragma endregion

#pragma region [ CPU baking ]
    static omm::Cpu::BakeFlags GetCpuBakeFlags(CpuBakerFlags cpuBakerFlags)
    {
        uint32_t result = 0;
        result |= cpuBakerFlags.enableInternalThreads ? uint32_t(omm::Cpu::BakeFlags::EnableInternalThreads) : 0;
        result |= !cpuBakerFlags.enableSpecialIndices ? uint32_t(omm::Cpu::BakeFlags::DisableSpecialIndices) : 0;
        result |= !cpuBakerFlags.enableDuplicateDetection ? uint32_t(omm::Cpu::BakeFlags::DisableDuplicateDetection) : 0;
        result |= cpuBakerFlags.enableNearDuplicateDetection ? uint32_t(omm::Cpu::BakeFlags::EnableNearDuplicateDetection) : 0;
        result |= !cpuBakerFlags.force32bitIndices ? uint32_t(omm::Cpu::BakeFlags::Force32BitIndices) : 0;
        return  omm::Cpu::BakeFlags(result);
    }

    void OpacityMicroMapsHelper::BakeOpacityMicroMapsCpu(OmmBakeGeometryDesc* queue, const size_t count, const OmmBakeDesc& desc)
    {
        for (size_t i = 0; i < count; ++i)
        {
            OmmBakeGeometryDesc& instance = queue[i];

            InputTexture& inTexture = instance.texture;

            omm::Cpu::TextureMipDesc texuteMipDesc = {};
            texuteMipDesc.width = inTexture.width;
            texuteMipDesc.height = inTexture.height;
            texuteMipDesc.textureData = (const uint8_t*)inTexture.nriTextureOrPtr.ptr;

            omm::Cpu::TextureDesc textureDesc;
            textureDesc.mipCount = 1; // TODO: Add mip array support
            textureDesc.mips = &texuteMipDesc;
            textureDesc.format = GetOmmBakerTextureFormat(inTexture.format);

            omm::Cpu::Texture vmTex = 0;
            if (omm::Cpu::CreateTexture(m_OmmCpuBaker, textureDesc, &vmTex) != omm::Result::SUCCESS)
            {
                printf("[FAIL]: omm::Cpu::CreateTexture\n");
                std::abort();
            }

            omm::Cpu::BakeInputDesc bakeDesc = {};
            bakeDesc.texture = vmTex;
            bakeDesc.alphaMode = omm::AlphaMode(instance.alphaMode);
            bakeDesc.runtimeSamplerDesc.addressingMode = GetOmmAddressingMode(inTexture.addressingMode);
            bakeDesc.runtimeSamplerDesc.filter = omm::TextureFilterMode(desc.filter);
            bakeDesc.maxSubdivisionLevel = (uint8_t)desc.subdivisionLevel;
            bakeDesc.alphaCutoff = instance.alphaCutoff;
            bakeDesc.dynamicSubdivisionScale = desc.dynamicSubdivisionScale;

            InputBuffer& inIndices = instance.indices;
            bakeDesc.indexFormat = GetOmmBakerIndexFormat(inIndices.format);
            bakeDesc.indexBuffer = (uint8_t*)inIndices.nriBufferOrPtr.ptr;
            bakeDesc.indexCount = (uint32_t)inIndices.numElements;

            InputBuffer& inUvs = instance.uvs;
            bakeDesc.texCoords = (uint8_t*)inUvs.nriBufferOrPtr.ptr;
            bakeDesc.texCoordFormat = GetOmmBakerUvFormat(inUvs.format);

            bakeDesc.bakeFlags = GetCpuBakeFlags(desc.cpuFlags);
            bakeDesc.ommFormat = GetOmmFormat(desc.format);

            omm::Cpu::BakeResult bakeResult;
            omm::Result res = omm::Cpu::BakeOpacityMicromap(m_OmmCpuBaker, bakeDesc, &bakeResult);

            if (res == omm::Result::WORKLOAD_TOO_BIG)
            {
                printf("[WARNING]: omm::Cpu::BakeOpacityMicromap - Workload size is too big.\n");
                return;
            }

            if (res != omm::Result::SUCCESS)
            {
                printf("[FAIL]: omm::Cpu::BakeVisibilityMap\n");
                std::abort();
            }

            const omm::Cpu::BakeResultDesc* resDesc = nullptr;
            res = omm::Cpu::GetBakeResultDesc(bakeResult, resDesc);

            if (res != omm::Result::SUCCESS)
            {
                printf("[FAIL]: omm::Cpu::GetBakeResultDesc\n");
                std::abort();
            }

            if (resDesc->ommArrayData)
            {
                instance.outData[(uint32_t)OmmDataLayout::ArrayData].resize(resDesc->ommArrayDataSize);
                memcpy(instance.outData[(uint32_t)OmmDataLayout::ArrayData].data(), resDesc->ommArrayData, resDesc->ommArrayDataSize);

                size_t ommDescArraySize = resDesc->ommDescArrayCount * sizeof(omm::Cpu::OpacityMicromapDesc);
                instance.outData[(uint32_t)OmmDataLayout::DescArray].resize(ommDescArraySize);
                memcpy(instance.outData[(uint32_t)OmmDataLayout::DescArray].data(), resDesc->ommDescArray, ommDescArraySize);

                size_t ommDescArrayHistogramSize = resDesc->ommDescArrayHistogramCount * sizeof(omm::Cpu::OpacityMicromapDesc);
                instance.outData[(uint32_t)OmmDataLayout::DescArrayHistogram].resize(ommDescArrayHistogramSize);
                memcpy(instance.outData[(uint32_t)OmmDataLayout::DescArrayHistogram].data(), resDesc->ommDescArrayHistogram, ommDescArrayHistogramSize);
                instance.outDescArrayHistogramCount = resDesc->ommDescArrayHistogramCount;

                size_t ommIndexHistogramSize = resDesc->ommIndexHistogramCount * sizeof(omm::Cpu::OpacityMicromapDesc);
                instance.outData[(uint32_t)OmmDataLayout::IndexHistogram].resize(ommIndexHistogramSize);
                memcpy(instance.outData[(uint32_t)OmmDataLayout::IndexHistogram].data(), resDesc->ommIndexHistogram, ommIndexHistogramSize);
                instance.outIndexHistogramCount = resDesc->ommIndexHistogramCount;

                size_t stride = resDesc->ommIndexFormat == omm::IndexFormat::I16_UINT ? sizeof(uint16_t) : sizeof(uint32_t);
                size_t indexDataSize = resDesc->ommIndexCount * stride;
                instance.outOmmIndexFormat = GetNriIndexFormat(resDesc->ommIndexFormat);
                instance.outOmmIndexStride = (uint32_t)stride;
                instance.outData[(uint32_t)OmmDataLayout::Indices].resize(indexDataSize);
                memcpy(instance.outData[(uint32_t)OmmDataLayout::Indices].data(), resDesc->ommIndexBuffer, indexDataSize);

            }
            omm::Cpu::DestroyTexture(m_OmmCpuBaker, vmTex);
            omm::Cpu::DestroyBakeResult(bakeResult);
        }
    }
#pragma endregion

#pragma region [ GPU Baking ]
    inline BakerBakeFlags GetGpuBakeFlags(const OmmBakeDesc& bakeDesc)
    {
        uint32_t result = 0;
        const GpuBakerFlags& flags = bakeDesc.gpuFlags;
        result |= !flags.enableSpecialIndices ? uint32_t(BakerBakeFlags::DisableSpecialIndices) : 0;
        result |= !flags.enableTexCoordDeduplication ? uint32_t(BakerBakeFlags::DisableTexCoordDeduplication) : 0;
        result |= flags.enablePostBuildInfo ? uint32_t(BakerBakeFlags::EnablePostBuildInfo) : 0;
        result |= bakeDesc.enableDebugMode ? uint32_t(BakerBakeFlags::EnableNsightDebugMode) : 0;
        result |= flags.force32bitIndices ? uint32_t(BakerBakeFlags::Force32BitIndices) : 0;
        result |= flags.computeOnlyWorkload ? uint32_t(BakerBakeFlags::ComputeOnly) : 0;
        return BakerBakeFlags(result);
    }

    inline void FillGpuBakerInputBufferDesc(BufferResource& bakerDesc, const InputBuffer& inDesc)
    {
        bakerDesc.buffer = (nri::Buffer*)inDesc.nriBufferOrPtr.buffer;
        bakerDesc.format = inDesc.format;
        bakerDesc.state = nri::AccessBits::SHADER_RESOURCE;
        bakerDesc.size = inDesc.bufferSize;
        bakerDesc.offset = inDesc.offset;
        bakerDesc.numElements = inDesc.numElements;
        bakerDesc.stride = inDesc.stride;
        bakerDesc.offsetInStruct = inDesc.offsetInStruct;
    }

    inline void FillGpuBakerResourceBufferDesc(BufferResource& bakerDesc, const GpuBakerBuffer& inDesc)
    {
        bakerDesc.buffer = inDesc.buffer;
        bakerDesc.offset = inDesc.offset;
        bakerDesc.size = inDesc.bufferSize;
        bakerDesc.state = nri::AccessBits::UNKNOWN;
    }

    inline void FillInputGeometryDesc(const OmmBakeGeometryDesc& desc, InputGeometryDesc& geometryDesc, const OmmBakeDesc& bakeDesc)
    {
        BakerInputs& inputs = geometryDesc.inputs;
        FillGpuBakerInputBufferDesc(inputs.inIndexBuffer, desc.indices);
        FillGpuBakerInputBufferDesc(inputs.inUvBuffer, desc.uvs);

        const InputTexture& texture = desc.texture;
        inputs.inTexture.texture = texture.nriTextureOrPtr.texture;
        inputs.inTexture.state = nri::AccessBits::SHADER_RESOURCE;
        inputs.inTexture.layout = nri::TextureLayout::SHADER_RESOURCE;
        inputs.inTexture.format = texture.format;
        inputs.inTexture.width = texture.width;
        inputs.inTexture.height = texture.height;
        inputs.inTexture.mipOffset = texture.mipOffset;
        inputs.inTexture.alphaChannelId = texture.alphaChannelId;

        for (size_t i = 0; i < omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS; ++i)
            FillGpuBakerResourceBufferDesc(inputs.inTransientPool[i], desc.transientBuffers[i]);

        BakerOutputs& outputs = geometryDesc.outputs;
        FillGpuBakerResourceBufferDesc(outputs.outArrayData, desc.gpuBuffers[(uint32_t)OmmDataLayout::ArrayData]);
        FillGpuBakerResourceBufferDesc(outputs.outDescArray, desc.gpuBuffers[(uint32_t)OmmDataLayout::DescArray]);
        FillGpuBakerResourceBufferDesc(outputs.outIndexBuffer, desc.gpuBuffers[(uint32_t)OmmDataLayout::Indices]);
        FillGpuBakerResourceBufferDesc(outputs.outArrayHistogram, desc.gpuBuffers[(uint32_t)OmmDataLayout::DescArrayHistogram]);
        FillGpuBakerResourceBufferDesc(outputs.outIndexHistogram, desc.gpuBuffers[(uint32_t)OmmDataLayout::IndexHistogram]);
        FillGpuBakerResourceBufferDesc(outputs.outPostBuildInfo, desc.gpuBuffers[(uint32_t)OmmDataLayout::GpuPostBuildInfo]);

        BakerSettings& settings = geometryDesc.settings;
        settings.alphaCutoff = desc.alphaCutoff;
        settings.borderAlpha = desc.borderAlpha;
        settings.alphaMode = BakerAlphaMode(desc.alphaMode);

        settings.numSupportedOmmFormats = 1;
        settings.globalOMMFormat = bakeDesc.format == OmmFormats::OC1_2_STATE ? BakerOmmFormat::OC1_2_State : BakerOmmFormat::OC1_4_State;
        settings.supportedOmmFormats[0] = settings.globalOMMFormat;

        settings.globalSubdivisionLevel = bakeDesc.subdivisionLevel;
        settings.maxSubdivisionLevel = bakeDesc.subdivisionLevel;

        settings.samplerAddressingMode = desc.texture.addressingMode;
        settings.samplerFilterMode = bakeDesc.filter == OmmBakeFilter::Linear ? nri::Filter::LINEAR : nri::Filter::NEAREST;
        settings.maxScratchMemorySize = BakerScratchMemoryBudget::MB_512;

        settings.dynamicSubdivisionScale = bakeDesc.dynamicSubdivisionScale;
        settings.bakeFlags = GetGpuBakeFlags(bakeDesc);
    }

    void OpacityMicroMapsHelper::GetGpuBakerPrebuildInfo(OmmBakeGeometryDesc* queue, const size_t count, const OmmBakeDesc& bakeDesc)
    {
        for (size_t i = 0; i < count; ++i)
        {
            InputGeometryDesc gpuBakerDesc = {};
            FillInputGeometryDesc(queue[i], gpuBakerDesc, bakeDesc);
            m_GpuBakerIntegration.GetPrebuildInfo(&gpuBakerDesc, 1);

            OmmBakeGeometryDesc::GpuBakerPrebuildInfo& prebuildInfo = queue[i].gpuBakerPreBuildInfo;
            PrebuildInfo& ommBakerPrebuildInfo = gpuBakerDesc.outputs.prebuildInfo;

            prebuildInfo.dataSizes[(uint32_t)OmmDataLayout::ArrayData] = ommBakerPrebuildInfo.arrayDataSize;
            prebuildInfo.dataSizes[(uint32_t)OmmDataLayout::DescArray] = ommBakerPrebuildInfo.descArraySize;
            prebuildInfo.dataSizes[(uint32_t)OmmDataLayout::Indices] = ommBakerPrebuildInfo.indexBufferSize;
            prebuildInfo.dataSizes[(uint32_t)OmmDataLayout::DescArrayHistogram] = ommBakerPrebuildInfo.ommDescArrayHistogramSize;
            prebuildInfo.dataSizes[(uint32_t)OmmDataLayout::IndexHistogram] = ommBakerPrebuildInfo.ommIndexHistogramSize;
            prebuildInfo.dataSizes[(uint32_t)OmmDataLayout::GpuPostBuildInfo] = ommBakerPrebuildInfo.postBuildInfoSize;

            memcpy(prebuildInfo.transientBufferSizes, ommBakerPrebuildInfo.transientBufferSizes, sizeof(uint64_t) * omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS);

            queue[i].outOmmIndexFormat = ommBakerPrebuildInfo.indexFormat;
            queue[i].outOmmIndexStride = (uint32_t)ommBakerPrebuildInfo.indexBufferSize / ommBakerPrebuildInfo.indexCount;
            queue[i].outDescArrayHistogramCount = uint32_t(ommBakerPrebuildInfo.ommDescArrayHistogramSize / (uint64_t)sizeof(omm::Cpu::OpacityMicromapUsageCount));
            queue[i].outIndexHistogramCount = uint32_t(ommBakerPrebuildInfo.ommIndexHistogramSize / (uint64_t)sizeof(omm::Cpu::OpacityMicromapUsageCount));
        }
    }

    void OpacityMicroMapsHelper::BakeOpacityMicroMapsGpu(nri::CommandBuffer* commandBuffer, OmmBakeGeometryDesc* queue, const size_t count, const OmmBakeDesc& bakeDesc)
    {
        std::vector<InputGeometryDesc> gpuBakerDescs(count);
        for (size_t i = 0; i < count; ++i)
            FillInputGeometryDesc(queue[i], gpuBakerDescs[i], bakeDesc);

        m_GpuBakerIntegration.Bake(*commandBuffer, gpuBakerDescs.data(), (uint32_t)gpuBakerDescs.size());
    }

    void OpacityMicroMapsHelper::GpuPostBakeCleanUp()
    {
        m_GpuBakerIntegration.ReleaseTemporalResources();
    }
#pragma endregion

#pragma region [ Geometry Builder ]
    void OpacityMicroMapsHelper::GetBlasPrebuildInfo(MaskedGeometryBuildDesc* queue, const size_t count)
    {
        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
            GetPreBuildInfoD3D12(queue, count);
        else
            GetPreBuildInfoVK(queue, count);
    }

    void OpacityMicroMapsHelper::BuildMaskedGeometry(MaskedGeometryBuildDesc* queue, const size_t count, nri::CommandBuffer* commandBuffer)
    {
        if (NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::D3D12)
            BuildMaskedGeometryD3D12(queue, count, commandBuffer);
        else
            BuildMaskedGeometryVK(queue, count, commandBuffer);
    }
#pragma endregion

#pragma region [ OMM Caching ]

    std::map<uint64_t, uint64_t> OmmCaching::m_IdentifierToDataOffset;

    uint64_t OmmCaching::PackStateMask(const OmmBakeDesc& buildDesc)
    {
        uint64_t mask = 0;
        mask |= uint64_t(buildDesc.subdivisionLevel) << 48;
        mask |= uint64_t(buildDesc.mipBias) << 32;
        mask |= uint64_t(buildDesc.filter) << 16;
        mask |= uint64_t(buildDesc.format) << 8;
        mask |= uint64_t(buildDesc.type);
        return mask;
    }

    inline uint64_t CalculateIdentifier(uint64_t a, uint64_t b)
    {
        uint64_t identifier = ((a + b) * (a + b + 1)) / 2 + b;
        return identifier;
    }

    void OmmCaching::PrewarmCache(const char* filename, FILE* file, size_t fileSize)
    {
        bool reachedEnd = false;
        while (reachedEnd != true)
        {
            MaskHeader currentHeader = {};
            size_t currentPos = ftell(file);
            if (ReadChunkFromFile(filename, file, fileSize, (void*)&currentHeader, sizeof(MaskHeader)) == false)
                return;

            uint64_t identifier = CalculateIdentifier(currentHeader.stateMask, currentHeader.hash);
            m_IdentifierToDataOffset.insert(std::make_pair(identifier, uint64_t(currentPos)));

            size_t blobSize = currentHeader.blobSize;
            currentPos = ftell(file);
            if (ValidateChunkRead(filename, file, fileSize, currentPos, blobSize) == false)
            {
                m_IdentifierToDataOffset.clear();
                return;
            }

            fseek(file, long(currentPos + blobSize), SEEK_SET);
            reachedEnd = ftell(file) == fileSize;
        }
        fseek(file, 0, SEEK_SET);
    }

    bool OmmCaching::LookForCache(const char* filename, uint64_t stateMask, uint64_t hash, size_t* dataOffset)
    {
        if (m_IdentifierToDataOffset.empty())
        {
            FILE* file = fopen(filename, "rb");
            if (file == nullptr)
                return false;//file not found

            fseek(file, 0, SEEK_END);
            size_t fileSize = ftell(file);
            fseek(file, 0, SEEK_SET);

            PrewarmCache(filename, file, fileSize);
            fclose(file);
        }

        uint64_t identifier = CalculateIdentifier(stateMask, hash);
        const auto& it = m_IdentifierToDataOffset.find(identifier);
        if (it == m_IdentifierToDataOffset.end())
            return false;
        else
        {
            if (dataOffset)
                *dataOffset = it->second;
            return true;
        }
    }

    bool OmmCaching::ReadMaskFromCache(const char* filename, OmmData& data, uint64_t stateMask, uint64_t hash, uint16_t* ommIndexFormat)
    {
        size_t dataOffset = 0;
        if (LookForCache(filename, stateMask, hash, &dataOffset) == false)
            return false;

        FILE* file = fopen(filename, "rb");
        if (file == nullptr)
        {
            printf("[FAIL] Unable to open file for reading: {%s}\n", filename);
            m_IdentifierToDataOffset.clear();
            return false;
        }

        fseek(file, 0, SEEK_END);
        size_t fileSize = ftell(file);
        fseek(file, long(dataOffset), SEEK_SET);

        MaskHeader header = {};
        if (ReadChunkFromFile(filename, file, fileSize, &header, sizeof(header)) == false)
            return false;

        std::vector<uint8_t> blob(header.blobSize);
        if (ReadChunkFromFile(filename, file, fileSize, blob.data(), header.blobSize) == false)
            return false;

        for (uint32_t i = 0; i < (uint32_t)OmmDataLayout::CpuMaxNum; ++i)
        {
            void* out = data.data[i];
            data.sizes[i] = header.sizes[i];

            if (!out)
                continue;

            memcpy(out, blob.data(), header.sizes[i]);
            blob.erase(blob.begin(), blob.begin() + header.sizes[i]);
        }

        if(ommIndexFormat)
            *ommIndexFormat = header.ommIndexFormat;

        fclose(file);
        return true;
    }

    void OmmCaching::SaveMasksToDisc(const char* filename, const OmmData& data, uint64_t stateMask, uint64_t hash, uint32_t ommIndexFormat)
    {
        if (LookForCache(filename, stateMask, hash, nullptr))
            return;//mask for this state is already cached

        FILE* outputFile = fopen(filename, "ab");
        if (outputFile == nullptr)
        {
            printf("[FAIL] Unable to open file for writing: {%s}\n", filename);
            m_IdentifierToDataOffset.clear();
            return;
        }

        fseek(outputFile, 0, SEEK_END);
        size_t fileSize = ftell(outputFile);
        fseek(outputFile, 0, SEEK_SET);

        size_t blobSize = 0;
        for (uint32_t i = 0; i < (uint32_t)OmmDataLayout::CpuMaxNum; ++i)
            blobSize += data.sizes[i];

        if (blobSize != 0)
        {
            MaskHeader header = {};
            std::vector<uint8_t> dataBlob;
            dataBlob.reserve(blobSize);

            for (uint32_t i = 0; i < (uint32_t)OmmDataLayout::CpuMaxNum; ++i)
            {
                uint64_t size = data.sizes[i];
                header.sizes[i] = size;
                size_t blobOffset = dataBlob.size();
                dataBlob.resize(dataBlob.size() + size);
                memcpy(dataBlob.data() + blobOffset, data.data[i], size);
            }

            header.hash = hash;
            header.stateMask = stateMask;
            header.ommIndexFormat = (uint16_t)ommIndexFormat;
            header.blobSize = blobSize;

            if (!WriteChunkToFile(filename, outputFile, (void*)&header, sizeof(header)))
                return;
            if (!WriteChunkToFile(filename, outputFile, (void*)dataBlob.data(), header.blobSize))
                return;

            uint64_t identifier = CalculateIdentifier(stateMask, hash);
            m_IdentifierToDataOffset.insert(std::make_pair(identifier, fileSize));
        }

        fclose(outputFile);
    }

    void OmmCaching::CreateFolder(const char* path)
    { 
        bool success = true;
        if(std::filesystem::exists(path) == false)
            success = std::filesystem::create_directory(path);
        if (!success)
            printf("[FAIL] Unable to create folder: {%s}\n", path);
    };

    inline bool OmmCaching::WriteChunkToFile(const char* fileName, FILE* file, void* data, size_t size)
    {
        if (fwrite(data, 1, size, file) != size)
        {
            printf("[FAIL] Unable to write to file: {%s}\n", fileName);
            fclose(file);
            std::filesystem::remove(fileName);
            m_IdentifierToDataOffset.clear();
            return false;
        }
        return true;
    }
    
    inline bool OmmCaching::ValidateChunkRead(const char* fileName, FILE* file, size_t fileSize, size_t currentPos, size_t dataSize)
    {
        if (currentPos + dataSize > fileSize)
        {
            printf("[FAIL] File end unexpected. Invalidating: {%s}\n", fileName);
            fclose(file);
            std::filesystem::remove(fileName);
            m_IdentifierToDataOffset.clear();
            return false;
        }
        return true;
    }
    
    inline bool OmmCaching::ReadChunkFromFile(const char* fileName, FILE* file, size_t fileSize, void* data, size_t dataSize)
    {
        size_t currentPos = ftell(file);
        if (ValidateChunkRead(fileName, file, fileSize, currentPos, dataSize) == false)
            return false;

        if (fread(data, 1, dataSize, file) != dataSize)
        {
            printf("[FAIL] Unable to read file: {%s}\n", fileName);
            fclose(file);
            m_IdentifierToDataOffset.clear();
            return false;
        }
        return true;
    }

#pragma endregion
}
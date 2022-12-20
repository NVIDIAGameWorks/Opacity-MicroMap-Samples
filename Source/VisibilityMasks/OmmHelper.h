/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include <dxgi1_6.h>
#include <d3d12.h>
#include <vulkan/vulkan.h>
#include <vector>
#include <array>
#include <map>

#include "../../External/NRIFramework/External/NRI/Include/NRI.h"
#include "../../External/NRIFramework/External/NRI/Include/Extensions/NRIDeviceCreation.h"
#include "../../External/NRIFramework/External/NRI/Include/Extensions/NRIRayTracing.h"
#include "../../External/NRIFramework/External/NRI/Include/Extensions/NRIHelper.h"
#include "../../External/NRIFramework/External/NRI/Include/Extensions/NRIWrapperD3D12.h"
#include "../../External/NRIFramework/External/NRI/Include/Extensions/NRIWrapperVK.h"

#define OMM_SUPPORTS_CPP17 (1)
#include "omm.h"

#include "../../External/NVAPI/nvapi.h"
#include "OmmBakerIntegration.h"

namespace ommhelper
{
    enum class OmmFormats
    {
        OC1_2_STATE,
        OC1_4_STATE,
        Count
    };

    enum class OmmBakeFilter
    {
        Nearest = (uint32_t)omm::TextureFilterMode::Nearest,
        Linear = (uint32_t)omm::TextureFilterMode::Linear,
        Count,
    };
        
    enum class OmmBakerType
    {
        GPU,
        CPU,
        Count
    };

    struct CpuBakerFlags
    {
        bool enableInternalThreads = true;
        bool enableSpecialIndices = true;
        bool enableDuplicateDetection = true;
        bool enableNearDuplicateDetection = false;
        bool force32bitIndices = false;
    };

    struct GpuBakerFlags
    {
        bool enablePostBuildInfo = true;
        bool enableSpecialIndices = true;
        bool enableTexCoordDeduplication = true;
        bool force32bitIndices = false;
        bool computeOnlyWorkload = false;
    };

    struct OmmBakeDesc
    {
        uint32_t subdivisionLevel = 9;// 4^N
        uint32_t mipBias = 0;
        uint32_t buildFrameId = 0;
        float dynamicSubdivisionScale = 2.0f;
        OmmBakeFilter filter = OmmBakeFilter::Linear;
        OmmFormats format = OmmFormats::OC1_4_STATE;
        OmmBakerType type = OmmBakerType::GPU;
        CpuBakerFlags cpuFlags;
        GpuBakerFlags gpuFlags;
        bool enableDebugMode = false;
        bool disableBlasBuild = false; //in case of system crash on NvApi calls
        bool enableCache = false;
    };

    enum class OmmAlphaMode
    {
        Test = (uint32_t)omm::AlphaMode::Test,
        Blend = (uint32_t)omm::AlphaMode::Blend,
        MaxNum = (uint32_t)omm::AlphaMode::MAX_NUM,
    };

    enum class OmmDataLayout
    {
        ArrayData,
        DescArray,
        Indices,
        DescArrayHistogram,
        IndexHistogram,
        GpuPostBuildInfo,
        MaxNum,
        BlasBuildGpuBuffersNum = DescArrayHistogram,
        CpuMaxNum = GpuPostBuildInfo,
        GpuOutputNum = MaxNum,
    };

    struct GpuBakerBuffer
    {
        nri::Buffer* buffer;
        uint64_t bufferSize; //total buffer size
        uint64_t dataSize;
        uint64_t offset;
    };

    struct InputBuffer
    {
        union NriBufferOrPtr
        {
            nri::Buffer* buffer;
            void* ptr;
        } nriBufferOrPtr;

        uint64_t bufferSize; //total buffer size;
        uint64_t offset;
        uint64_t numElements;
        uint64_t stride;
        uint64_t offsetInStruct;
        nri::Format format;
    };

    struct InputTexture
    {
        union NriTextureOrPtr
        {
            nri::Texture* texture;
            void* ptr;
        } nriTextureOrPtr;

        uint32_t width;
        uint32_t height;
        uint32_t mipOffset;
        uint32_t alphaChannelId;
        nri::Format format;
        nri::AddressMode addressingMode;
    };

    struct OmmBakeGeometryDesc
    {
        InputBuffer indices;
        InputBuffer uvs;
        InputTexture texture;

        GpuBakerBuffer gpuBuffers[uint32_t(OmmDataLayout::GpuOutputNum)];
        GpuBakerBuffer transientBuffers[omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS];
        GpuBakerBuffer readBackBuffers[uint32_t(OmmDataLayout::GpuOutputNum)];

        std::vector<uint8_t> outData[uint32_t(OmmDataLayout::MaxNum)]; //cpu baker outputs/gpu baker readback for caching

        struct GpuBakerPrebuildInfo
        {
            uint64_t dataSizes[(uint32_t)OmmDataLayout::GpuOutputNum];
            uint64_t transientBufferSizes[omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS];
        } gpuBakerPreBuildInfo;

        float alphaCutoff;
        float borderAlpha;

        uint32_t outIndexHistogramCount;
        uint32_t outDescArrayHistogramCount;
        uint32_t outOmmIndexStride;
        nri::Format outOmmIndexFormat;
        OmmAlphaMode alphaMode;
    };

    struct MaskedGeometryBuildDesc
    {
        struct Inputs
        {
            InputBuffer indices;
            InputBuffer vertices;
            void* descArrayHistogram;
            void* indexHistogram;

            GpuBakerBuffer buffers[uint32_t(OmmDataLayout::BlasBuildGpuBuffersNum)];

            uint64_t ommIndexStride;
            uint32_t descArrayHistogramNum;
            uint32_t indexHistogramNum;
            nri::Format ommIndexFormat;
        } inputs;

        struct PrebuildInfo
        {
            uint64_t ommArraySize;
            uint64_t blasSize;
            uint64_t maxScratchDataSize;
        } prebuildInfo;

        struct Outputs
        {
            nri::AccelerationStructure* blas;
            nri::Buffer* ommArray;
        } outputs;
    };

    struct OmmCaching
    {
        struct MaskHeader
        {
            uint64_t hash;
            uint64_t stateMask;
            uint64_t sizes[(uint32_t)OmmDataLayout::CpuMaxNum];
            uint64_t blobSize;
            uint16_t ommIndexFormat;
        };
        struct OmmData
        {
            void* data[(uint32_t)OmmDataLayout::CpuMaxNum];
            uint64_t sizes[(uint32_t)OmmDataLayout::CpuMaxNum];
        };
        static uint64_t PackStateMask(const OmmBakeDesc& buildDesc);
        static bool ReadMaskFromCache(const char* filename, OmmData& data, uint64_t stateMask, uint64_t hash, uint16_t* ommIndexFormat);
        static void SaveMasksToDisc(const char* filename, const OmmData& data, uint64_t stateMask, uint64_t hash, uint32_t ommIndexFormat);
        static void CreateFolder(const char* path);
    private:
        static void PrewarmCache(const char* filename, FILE* file, size_t fileSize);
        static bool LookForCache(const char* filename, uint64_t stateMask, uint64_t hash, size_t* dataOffset = nullptr);
        static bool WriteChunkToFile(const char* fileName, FILE* file, void* data, size_t size);
        static bool ValidateChunkRead(const char* fileName, FILE* file, size_t fileSize, size_t currentPos, size_t dataSize);
        static bool ReadChunkFromFile(const char* fileName, FILE* file, size_t fileSize, void* data, size_t dataSize);
        static std::map<uint64_t, uint64_t> m_IdentifierToDataOffset;
    };

    class OpacityMicroMapsHelper
    {
    public:
        void Initialize(nri::Device* device);

        void GetGpuBakerPrebuildInfo(OmmBakeGeometryDesc* queue, const size_t count, const OmmBakeDesc& desc);
        void BakeOpacityMicroMapsGpu(nri::CommandBuffer* commandBuffer, OmmBakeGeometryDesc* queue, const size_t count, const OmmBakeDesc& bakeDesc);
        void GpuPostBakeCleanUp();

        void BakeOpacityMicroMapsCpu(OmmBakeGeometryDesc* queue, const size_t count, const OmmBakeDesc& desc);
        void ConvertUsageCountsToApiFormat(uint8_t* outFormattedBuffer, size_t& outSize, const uint8_t* bakerOutputBuffer, size_t bakerOutputBufferSize);

        void GetBlasPrebuildInfo(MaskedGeometryBuildDesc* queue, const size_t count);
        void BuildMaskedGeometry(MaskedGeometryBuildDesc* queue, const size_t count, nri::CommandBuffer* commandBuffer);
        void DestroyMaskedGeometry(nri::AccelerationStructure* blas, nri::Buffer* ommArray);
        void ReleaseGeometryMemory();

        void Destroy();

    private:
        //D3D12:
        void InitializeD3D12();
        void GetPreBuildInfoD3D12(MaskedGeometryBuildDesc* queue, const size_t count);
        void BindResourceToMemoryD3D12(ID3D12Resource*& resource, size_t size);
        void AllocateMemoryD3D12();
        void ReleaseMemoryD3D12();
        void BuildMaskedGeometryD3D12(MaskedGeometryBuildDesc* queue, const size_t count, nri::CommandBuffer* commandBuffer);
        void BuildOmmArrayD3D12(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer);
        void BuildBlasD3D12(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer);
        ID3D12Device5* GetD3D12Device5();
        ID3D12GraphicsCommandList4* GetD3D12GraphicsCommandList4(nri::CommandBuffer* commandBuffer);

        //VK:
        void InitializeVK();
        void AllocateMemoryVK();
        void ReleaseMemoryVK();
        void GetPreBuildInfoVK(MaskedGeometryBuildDesc* queue, const size_t count);
        void BindOmmToMemoryVK(VkMicromapEXT& ommArray, size_t size);
        void BindBlasToMemoryVK(VkAccelerationStructureKHR& blas, size_t size);
        void BuildMaskedGeometryVK(MaskedGeometryBuildDesc* queue, const size_t count, nri::CommandBuffer* commandBuffer);
        void BuildOmmArrayVK(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer);
        void BuildBlasVK(MaskedGeometryBuildDesc& desc, nri::CommandBuffer* commandBuffer);
        void DestroyOmmArrayVK(nri::Buffer* ommArray);
        VkDevice GetVkDevice();

    private:
        //internal memory for masked geometry
        //TODO: when micromaps are supported in NRI, move memory managment to the sample main part
        const uint64_t m_HeapSize = 100 * 1024 * 1024;
        const uint64_t m_SctrachSize = 10 * 1024 * 1024;
        uint64_t m_CurrentHeapOffset;

        //D3D12:
        std::vector<ID3D12Heap*> m_D3D12GeometryHeaps;
        ID3D12Resource* m_D3D12ScratchBuffer = nullptr;

        //VK:
        std::vector<VkDeviceMemory> m_VkMemories;
        std::vector<VkBuffer> m_VkBuffers;
        uint32_t m_VkMemoryTypeId = uint32_t(~0);
        VkBuffer m_VkScrathBuffer;

        //common
        struct NriInterface
            : public nri::CoreInterface
            , public nri::RayTracingInterface
            , public nri::HelperInterface
            , public nri::WrapperD3D12Interface
            , public nri::WrapperVKInterface
        {};
        NriInterface NRI = {};

        OmmBakerGpuIntegration m_GpuBakerIntegration;
        omm::Baker m_OmmCpuBaker = 0;
        nri::Device* m_Device;
    };
}

/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include <vector>
#include <map>

#include "../../External/NRIFramework/External/NRI/Include/NRI.h"
#include "../../External/NRIFramework/External/NRI/Include/Extensions/NRIHelper.h"

#define OMM_SUPPORTS_CPP17 (1)
#include "omm.h"

struct TextureResource
{
    nri::Texture* texture;
    nri::Format format;
    nri::AccessBits state;
    nri::TextureLayout layout;
    uint32_t width;
    uint32_t height;
    uint32_t mipOffset;
    uint32_t alphaChannelId;
};

struct BufferResource
{
    nri::Buffer* buffer;
    nri::Format format = nri::Format::R32_UINT;
    uint64_t size;
    uint64_t offset;
    uint64_t stride;
    uint64_t offsetInStruct;
    uint64_t numElements;
    nri::AccessBits state;
};

struct PrebuildInfo
{
    uint64_t arrayDataSize;
    uint64_t descArraySize;
    uint64_t indexBufferSize;
    uint64_t ommDescArrayHistogramSize;
    uint64_t ommIndexHistogramSize;
    uint64_t postBuildInfoSize;
    uint64_t transientBufferSizes[OMM_MAX_TRANSIENT_POOL_BUFFERS];

    uint32_t indexCount;
    nri::Format indexFormat;
};

enum class BakerAlphaMode : uint32_t
{
    Test = (uint32_t)ommAlphaMode_Test,
    Blend = (uint32_t)ommAlphaMode_Blend,
    Count
};

enum class BakerOmmFormat : uint16_t
{
    OC1_2_State = 1,
    OC1_4_State = 2,
};

enum class BakerScratchMemoryBudget : uint64_t
{
    Undefined = (uint64_t)ommGpuScratchMemoryBudget_Undefined,

    MB_4 = (uint64_t)ommGpuScratchMemoryBudget_MB_4,
    MB_32 = (uint64_t)ommGpuScratchMemoryBudget_MB_32,
    MB_64 = (uint64_t)ommGpuScratchMemoryBudget_MB_64,
    MB_128 = (uint64_t)ommGpuScratchMemoryBudget_MB_128,
    MB_256 = (uint64_t)ommGpuScratchMemoryBudget_MB_256,
    MB_512 = (uint64_t)ommGpuScratchMemoryBudget_MB_512,
    MB_1024 = (uint64_t)ommGpuScratchMemoryBudget_MB_1024,

    Default = (uint64_t)ommGpuScratchMemoryBudget_Default,
};

enum class BakerBakeFlags : uint32_t
{
    Invalid = (uint32_t)ommGpuBakeFlags_Invalid,
    PerformBake = (uint32_t)ommGpuBakeFlags_PerformBake,
    PerformSetup = (uint32_t)ommGpuBakeFlags_PerformSetup,
    EnablePostBuildInfo = (uint32_t)ommGpuBakeFlags_EnablePostDispatchInfoStats,
    DisableSpecialIndices = (uint32_t)ommGpuBakeFlags_DisableSpecialIndices,
    DisableTexCoordDeduplication = (uint32_t)ommGpuBakeFlags_DisableTexCoordDeduplication,
    Force32BitIndices = (uint32_t)ommGpuBakeFlags_Force32BitIndices,
    ComputeOnly = (uint32_t)ommGpuBakeFlags_ComputeOnly,
    EnableNsightDebugMode = (uint32_t)ommGpuBakeFlags_EnableNsightDebugMode,
};

struct BakerSettings
{
    uint32_t maxSubdivisionLevel;

    float dynamicSubdivisionScale;
    float alphaCutoff;
    float borderAlpha;

    BakerAlphaMode alphaMode;

    nri::Filter samplerFilterMode;
    nri::AddressMode samplerAddressingMode;

    BakerOmmFormat globalOMMFormat;
    BakerScratchMemoryBudget maxScratchMemorySize;
    BakerBakeFlags bakeFlags;
};

struct BakerInputs
{
    TextureResource inTexture;
    BufferResource inUvBuffer;
    BufferResource inIndexBuffer;
    BufferResource inSubdivisionLevelBuffer; //currently unused
    BufferResource inTransientPool[OMM_MAX_TRANSIENT_POOL_BUFFERS];
};

struct BakerOutputs
{
    BufferResource outArrayData = { nullptr, nri::Format::R32_UINT };
    BufferResource outDescArray = { nullptr, nri::Format::R32_UINT };
    BufferResource outIndexBuffer = { nullptr, nri::Format::R32_UINT };
    BufferResource outArrayHistogram = { nullptr, nri::Format::R32_UINT };
    BufferResource outIndexHistogram = { nullptr, nri::Format::R32_UINT };
    BufferResource outPostBuildInfo = { nullptr, nri::Format::R32_UINT };

    PrebuildInfo prebuildInfo;
};

struct InputGeometryDesc
{
    BakerInputs inputs;
    BakerOutputs outputs;
    BakerSettings settings;
};

class OmmBakerGpuIntegration
{
public:
    void Initialize(nri::Device& device);                                                               //0.
    void GetPrebuildInfo(InputGeometryDesc* geometryDesc, uint32_t geometryNum);                        //1. Get info on output resources sizes
    void Bake(nri::CommandBuffer& commandBuffer, InputGeometryDesc* geometryDesc, uint32_t geometryNum);//2. After the queue is ready kick off the baker
    void ReleaseTemporalResources();                                                                    //3. Clean up internal data after work is finished
    void Destroy();                                                                                     //4.

private:
    struct NRIInterface
        : public nri::CoreInterface
        , public nri::HelperInterface
    {};

    enum class GpuStaticResources
    {
        IndexBuffer,
        VertexBuffer,
        Count,
    };

    struct FrameBuffer
    {
        nri::FrameBuffer* frameBuffer;
        nri::Texture* texture;
        nri::Memory* memory;
        nri::Descriptor* descriptor;
        nri::AccessBits state = nri::AccessBits::UNKNOWN;
    };

    struct GeometryQueueInstance
    {
        InputGeometryDesc* desc;
        ommGpuDispatchConfigDesc dispatchConfigDesc;
    };

private:
    //On Init
    void CreateFrameBuffers(uint32_t pipelineNum);
    void CreateSamplers(const ommGpuPipelineInfoDesc* pipelinesInfo);
    void CreatePipelines(const ommGpuPipelineInfoDesc* pipelinesInfo);
    void CreateComputePipeline(uint32_t id, const ommGpuPipelineInfoDesc* pipelineInfo);
    void CreateGraphicsPipeline(uint32_t id, const ommGpuPipelineInfoDesc* pipelineInfo);
    void CreateStaticResources(nri::CommandQueue* commandQueue);

    //On Submit
    void AddGeometryToQueue(InputGeometryDesc* geometryDesc, uint32_t geometryNum);

    //On Build
    nri::DescriptorSet* PrepareDispatch(nri::CommandBuffer& commandBuffer, const ommGpuResource* resources, uint32_t resourceNum, uint32_t pipelineIndex, uint32_t geometryId);
    void InsertUavBarriers(nri::CommandBuffer& commandBuffer, const ommGpuResource* resources, uint32_t resourceNum, uint32_t geometryId);
    void PerformResourceTransition(const ommGpuResource& resource, uint32_t geometryId, std::vector<nri::BufferTransitionBarrierDesc>& bufferBarriers);
    BufferResource& GetBuffer(const ommGpuResource& resource, uint32_t geometryId);

    void UpdateDescriptorPool(uint32_t geometryId, const ommGpuDispatchChain* dispatchChain);
    void UpdateGlobalConstantBuffer();

    nri::Descriptor* GetDescriptor(const ommGpuResource& resource, uint32_t geometryId);
    void DispatchCompute(nri::CommandBuffer& commandBuffer, const ommGpuComputeDesc& desc, uint32_t geometryId);
    void DispatchComputeIndirect(nri::CommandBuffer& commandBuffer, const ommGpuComputeIndirectDesc& desc, uint32_t geometryId);
    void DispatchDrawIndexedIndirect(nri::CommandBuffer& commandBuffer, const ommGpuDrawIndexedIndirectDesc& desc, uint32_t geometryId);

    void GenerateVisibilityMaskGPU(nri::CommandBuffer& commandBuffer, uint32_t geometryId);

private:
    std::vector<GeometryQueueInstance> m_GeometryQueue;

    //resources
    BufferResource m_StaticBuffers[(uint32_t)GpuStaticResources::Count];
    std::map<uint64_t, nri::Descriptor*> m_NriDescriptors;
    std::map<uint64_t, nri::DescriptorSet*> m_NriDescriptorSets;
    std::vector<nri::Memory*> m_NriStaticMemories;
    std::vector<nri::DescriptorPool*> m_NriDescriptorPools;

    //samplers
    std::vector<nri::Descriptor*> m_Samplers;

    //pipelines
    std::vector<nri::Pipeline*> m_NriPipelines;
    std::vector<nri::PipelineLayout*> m_NriPipelineLayouts;

    //vars
    NRIInterface NRI = {};
    nri::Device* m_Device;

    //CB
    nri::Descriptor* m_ConstantBufferView;
    nri::Buffer* m_ConstantBuffer;
    nri::Memory* m_ConstantBufferHeap;
    uint32_t m_ConstantBufferViewSize;
    uint32_t m_ConstantBufferSize;
    uint32_t m_ConstantBufferOffset;

    //framebuffers
    FrameBuffer m_FrameBuffers[2];
    std::vector<FrameBuffer*> m_FrameBufferPerPipeline;
    const uint32_t m_EmptyFrameBufferId = 0;
    const uint32_t m_DebugFrameBufferId = 1;
    const nri::Format m_DebugTexFormat = nri::Format::RGBA8_SNORM;

    //ommbaker
    const ommGpuPipelineInfoDesc* m_PipelineInfo;
    ommBaker m_GpuBaker;
    ommGpuPipeline m_Pipeline;
};


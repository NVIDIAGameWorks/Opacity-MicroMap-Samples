/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "OmmBakerIntegration.h"

#define NRI_ABORT_ON_FAILURE(result) \
    if ((result) != nri::Result::SUCCESS) \
        exit(1);

void OmmBakerGpuIntegration::Initialize(nri::Device& device)
{
    m_Device = &device;

    uint32_t nriResult = (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI);
    nriResult |= (uint32_t)nri::GetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI);
    if (nriResult != (uint32_t)nri::Result::SUCCESS)
    {
        printf("[FAIL]: nri::GetInterface\n");
        std::abort();
    }

    ommBakerCreationDesc bakerCreationDesc = {};
    bakerCreationDesc.enableValidation = true;
    bakerCreationDesc.type = ommBakerType_GPU;
    ommResult ommResult = ommCreateBaker(&bakerCreationDesc, &m_GpuBaker);
    if (ommResult != ommResult_SUCCESS)
    {
        printf("[FAIL]: ommCreateBaker\n");
        std::abort();
    }

    ommGpuRenderAPI renderApi = NRI.GetDeviceDesc(*m_Device).graphicsAPI == nri::GraphicsAPI::VULKAN ? ommGpuRenderAPI_Vulkan : ommGpuRenderAPI_DX12;
    ommGpuPipelineConfigDesc bakePipelineDesc = { renderApi };
    ommResult = ommGpuCreatePipeline(m_GpuBaker, &bakePipelineDesc, &m_Pipeline);
    if (ommResult != ommResult_SUCCESS)
    {
        printf("[FAIL]: ommGpuCreatePipeline\n");
        std::abort();
    }

    ommResult = ommGpuGetPipelineDesc(m_Pipeline, &m_PipelineInfo);
    if (ommResult != ommResult_SUCCESS)
    {
        printf("[FAIL]: ommGpuGetPipelineDesc\n");
        std::abort();
    }

    nri::CommandQueue* commandQueue = nullptr;
    NRI_ABORT_ON_FAILURE(NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, commandQueue));
    {
        CreateStaticResources(commandQueue);
        CreateSamplers(m_PipelineInfo);
        CreateFrameBuffers(m_PipelineInfo->pipelineNum);
        CreatePipelines(m_PipelineInfo);
    }
}

ommTexCoordFormat GetOmmTexcoordFormat(nri::Format format)
{
    switch (format)
    {
    case nri::Format::RG16_UNORM: return ommTexCoordFormat_UV16_UNORM;
    case nri::Format::RG16_SFLOAT: return ommTexCoordFormat_UV16_FLOAT;
    case nri::Format::RG32_SFLOAT: return ommTexCoordFormat_UV32_FLOAT;
    default: printf("[FAIL] Unsupported texCoord format\n"); std::abort();
    }
}

ommIndexFormat GetOmmIndexFormat(nri::Format inFormat)
{
    switch (inFormat)
    {
    case nri::Format::R16_UINT: return ommIndexFormat_I16_UINT;
    case nri::Format::R32_UINT: return ommIndexFormat_I32_UINT;
    default: printf("[FAIL] Unsupported index format\n"); std::abort();
    }
}

nri::Format GetNriIndexFormat(ommIndexFormat  inFormat)
{
    switch (inFormat)
    {
    case ommIndexFormat_I16_UINT : return nri::Format::R16_UINT;
    case ommIndexFormat_I32_UINT : return nri::Format::R32_UINT;
    default: printf("[FAIL] Unsupported index format\n"); std::abort();
    }
}

ommTextureFilterMode GetOmmFilterMode(nri::Filter mode)
{
    switch (mode)
    {
    case nri::Filter::LINEAR: return ommTextureFilterMode_Linear;
    case nri::Filter::NEAREST: return ommTextureFilterMode_Nearest;
    default: printf("[FAIL] Invalid ommTextureFilterMode\n"); std::abort();
    }
}

ommTextureAddressMode GetOmmAddressingMode(nri::AddressMode mode)
{
    switch (mode)
    {
    case nri::AddressMode::REPEAT: return ommTextureAddressMode_Wrap;
    case nri::AddressMode::MIRRORED_REPEAT: return ommTextureAddressMode_Mirror;
    case nri::AddressMode::CLAMP_TO_EDGE: return ommTextureAddressMode_Clamp;
    case nri::AddressMode::CLAMP_TO_BORDER: return ommTextureAddressMode_Border;
    default: printf("[FAIL] Invalid ommTextureAddressMode\n"); std::abort();
    }
}

nri::DescriptorType GetNriDescriptorType(ommGpuDescriptorType ommType)
{
    switch (ommType)
    {
    case ommGpuDescriptorType_TextureRead: return nri::DescriptorType::TEXTURE;
    case ommGpuDescriptorType_BufferRead: return nri::DescriptorType::BUFFER;
    case ommGpuDescriptorType_RawBufferRead: return nri::DescriptorType::STRUCTURED_BUFFER;
    case ommGpuDescriptorType_RawBufferWrite: return nri::DescriptorType::STORAGE_STRUCTURED_BUFFER;
    default: printf("[FAIL] Invalid ommGpuDescriptorType"); std::abort();
    }
}

nri::AddressMode GetNriAddressMode(ommTextureAddressMode mode)
{
    switch (mode)
    {
    case ommTextureAddressMode_Wrap: return nri::AddressMode::REPEAT;
    case ommTextureAddressMode_Mirror: return nri::AddressMode::MIRRORED_REPEAT;
    case ommTextureAddressMode_Clamp: return nri::AddressMode::CLAMP_TO_EDGE;
    case ommTextureAddressMode_Border: return nri::AddressMode::CLAMP_TO_BORDER;
    case ommTextureAddressMode_MirrorOnce: return nri::AddressMode::MIRRORED_REPEAT;
    default: printf("[FAIL] Invalid ommTextureAddressMode\n"); std::abort();
    }
}

nri::Filter GetNriFilterMode(ommTextureFilterMode mode)
{
    switch (mode)
    {
    case ommTextureFilterMode_Linear: return nri::Filter::LINEAR;
    case ommTextureFilterMode_Nearest: return nri::Filter::NEAREST;
    default: printf("[FAIL] Invalid ommTextureFilterMode\n"); std::abort();
    }
}

nri::AccessBits GetNriResourceState(ommGpuDescriptorType descriptorType)
{
    switch (descriptorType)
    {
    case ommGpuDescriptorType_BufferRead: return nri::AccessBits::SHADER_RESOURCE;
    case ommGpuDescriptorType_RawBufferRead: return nri::AccessBits::SHADER_RESOURCE;
    case ommGpuDescriptorType_RawBufferWrite: return nri::AccessBits::SHADER_RESOURCE_STORAGE;
    case ommGpuDescriptorType_TextureRead: return nri::AccessBits::SHADER_RESOURCE;
    default: printf("[FAIL] Invalid ommGpuDescriptorType\n"); std::abort();
    }
}

BufferResource& OmmBakerGpuIntegration::GetBuffer(const ommGpuResource& resource, uint32_t geometryId)
{
    BakerInputs& inputs = m_GeometryQueue[geometryId].desc->inputs;
    BakerOutputs& outputs = m_GeometryQueue[geometryId].desc->outputs;
    switch (resource.type)
    {
    case ommGpuResourceType_IN_TEXCOORD_BUFFER: return inputs.inUvBuffer;
    case ommGpuResourceType_IN_INDEX_BUFFER: return inputs.inIndexBuffer;
    case ommGpuResourceType_IN_SUBDIVISION_LEVEL_BUFFER: return inputs.inSubdivisionLevelBuffer;
    case ommGpuResourceType_OUT_OMM_ARRAY_DATA: return outputs.outArrayData;
    case ommGpuResourceType_OUT_OMM_DESC_ARRAY: return outputs.outDescArray;
    case ommGpuResourceType_OUT_OMM_INDEX_BUFFER: return outputs.outIndexBuffer;
    case ommGpuResourceType_OUT_OMM_DESC_ARRAY_HISTOGRAM: return outputs.outArrayHistogram;
    case ommGpuResourceType_OUT_OMM_INDEX_HISTOGRAM: return outputs.outIndexHistogram;
    case ommGpuResourceType_OUT_POST_DISPATCH_INFO: return outputs.outPostBuildInfo;
    case ommGpuResourceType_TRANSIENT_POOL_BUFFER: return inputs.inTransientPool[resource.indexInPool];
    case ommGpuResourceType_STATIC_VERTEX_BUFFER: return m_StaticBuffers[(uint32_t)GpuStaticResources::VertexBuffer];
    case ommGpuResourceType_STATIC_INDEX_BUFFER: return m_StaticBuffers[(uint32_t)GpuStaticResources::IndexBuffer];
    default: std::abort();
    }
}

nri::BufferViewType GetNriBufferViewType(ommGpuDescriptorType type)
{
    switch (type)
    {
    case ommGpuDescriptorType_BufferRead: return nri::BufferViewType::SHADER_RESOURCE;
    case ommGpuDescriptorType_RawBufferRead: return nri::BufferViewType::SHADER_RESOURCE;
    case ommGpuDescriptorType_RawBufferWrite: return nri::BufferViewType::SHADER_RESOURCE_STORAGE;
    case ommGpuDescriptorType_TextureRead:
    default: printf("[FAIL] Invalid BufferDescriptorType\n"); std::abort();
    }
}

ommGpuBakeFlags GetBakeFlags(BakerBakeFlags flags)
{
    static_assert((uint32_t)BakerBakeFlags::Invalid == (uint32_t)ommGpuBakeFlags_Invalid);
    static_assert((uint32_t)BakerBakeFlags::EnablePostBuildInfo == (uint32_t)ommGpuBakeFlags_EnablePostDispatchInfoStats);
    static_assert((uint32_t)BakerBakeFlags::DisableSpecialIndices == (uint32_t)ommGpuBakeFlags_DisableSpecialIndices);
    static_assert((uint32_t)BakerBakeFlags::DisableTexCoordDeduplication == (uint32_t)ommGpuBakeFlags_DisableTexCoordDeduplication);
    static_assert((uint32_t)BakerBakeFlags::EnableNsightDebugMode == (uint32_t)ommGpuBakeFlags_EnableNsightDebugMode);
    return (ommGpuBakeFlags)flags;
}

ommGpuScratchMemoryBudget GetScratchMemoryBudget(BakerScratchMemoryBudget budget)
{
    static_assert((uint64_t)ommGpuScratchMemoryBudget_Undefined == (uint64_t)BakerScratchMemoryBudget::Undefined);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_4 == (uint64_t)BakerScratchMemoryBudget::MB_4);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_32 == (uint64_t)BakerScratchMemoryBudget::MB_32);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_64 == (uint64_t)BakerScratchMemoryBudget::MB_64);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_128 == (uint64_t)BakerScratchMemoryBudget::MB_128);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_256 == (uint64_t)BakerScratchMemoryBudget::MB_256);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_512 == (uint64_t)BakerScratchMemoryBudget::MB_512);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_MB_1024 == (uint64_t)BakerScratchMemoryBudget::MB_1024);
    static_assert((uint64_t)ommGpuScratchMemoryBudget_Default == (uint64_t)BakerScratchMemoryBudget::Default);
    return (ommGpuScratchMemoryBudget)budget;
}

void FillDescriptorRangeDescs(uint32_t count, const ommGpuDescriptorRangeDesc* ommDesc, nri::DescriptorRangeDesc* nriDesc)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        nriDesc[i].baseRegisterIndex = ommDesc[i].baseRegisterIndex;
        nriDesc[i].descriptorNum = ommDesc[i].descriptorNum;
        nriDesc[i].descriptorType = GetNriDescriptorType(ommDesc[i].descriptorType);
        nriDesc[i].visibility = nri::ShaderStage::ALL;
    }
}

void OmmBakerGpuIntegration::CreateGraphicsPipeline(uint32_t pipelineId, const ommGpuPipelineInfoDesc* pipelineInfo)
{
    const ommGpuGraphicsPipelineDesc& pipelineDesc = pipelineInfo->pipelines[pipelineId].graphics;
    static_assert(OMM_GRAPHICS_PIPELINE_DESC_VERSION == 3, "ommGpuGraphicsPipelineDesc has changed\n");

    std::vector<nri::DescriptorRangeDesc> descriptorRangeDescs(pipelineDesc.descriptorRangeNum + 1);
    FillDescriptorRangeDescs(pipelineDesc.descriptorRangeNum, pipelineDesc.descriptorRanges, descriptorRangeDescs.data());

    nri::DescriptorRangeDesc& staticSamplersRange = descriptorRangeDescs.back();
    staticSamplersRange.baseRegisterIndex = 0;
    staticSamplersRange.descriptorNum = (uint32_t)m_Samplers.size();
    staticSamplersRange.descriptorType = nri::DescriptorType::SAMPLER;
    staticSamplersRange.visibility = nri::ShaderStage::ALL;

    nri::DescriptorSetDesc descriptorSetDescs = {};
    descriptorSetDescs.rangeNum = (uint32_t)descriptorRangeDescs.size();
    descriptorSetDescs.ranges = descriptorRangeDescs.data();
    descriptorSetDescs.dynamicConstantBufferNum = 0;

    nri::DynamicConstantBufferDesc dynamicConstantBufferDesc;
    {
        dynamicConstantBufferDesc.registerIndex = pipelineInfo->globalConstantBufferDesc.registerIndex;
        dynamicConstantBufferDesc.visibility = nri::ShaderStage::ALL;
        descriptorSetDescs.dynamicConstantBuffers = &dynamicConstantBufferDesc;
        descriptorSetDescs.dynamicConstantBufferNum = 1;
    }

    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.descriptorSets = &descriptorSetDescs;
    layoutDesc.descriptorSetNum = 1;
    layoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::ALL_GRAPHICS;
    nri::PushConstantDesc pushConstantDesc = {};
    {
        pushConstantDesc.registerIndex = pipelineInfo->localConstantBufferDesc.registerIndex;
        pushConstantDesc.size = pipelineInfo->localConstantBufferDesc.maxDataSize;
        pushConstantDesc.visibility = nri::ShaderStage::ALL;
        layoutDesc.pushConstants = &pushConstantDesc;
        layoutDesc.pushConstantNum = 1;
    }
    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, layoutDesc, m_NriPipelineLayouts.emplace_back()));

    nri::GraphicsPipelineDesc nriPipelineDesc = {};
    nriPipelineDesc.pipelineLayout = m_NriPipelineLayouts.back();

    ommGpuGraphicsPipelineInputElementDesc inputElementDesc = ommGpuGraphicsPipelineInputElementDescDefault();
    nri::VertexAttributeDesc vertextAttributes = {};
    vertextAttributes.format = nri::Format::R32_UINT;
    vertextAttributes.d3d.semanticIndex = inputElementDesc.semanticIndex;
    vertextAttributes.d3d.semanticName = inputElementDesc.semanticName;
    vertextAttributes.vk.location = 0;
    vertextAttributes.streamIndex = 0;

    nri::VertexStreamDesc vertexStreamDesc = {};
    vertexStreamDesc.bindingSlot = 0;
    vertexStreamDesc.stride = sizeof(uint32_t);

    nri::InputAssemblyDesc inputAssemblyDesc = {};
    nriPipelineDesc.inputAssembly = &inputAssemblyDesc;

    inputAssemblyDesc.attributes = &vertextAttributes;
    inputAssemblyDesc.attributeNum = 1;
    inputAssemblyDesc.streams = &vertexStreamDesc;
    inputAssemblyDesc.streamNum = 1;
    inputAssemblyDesc.topology = nri::Topology::TRIANGLE_LIST;

    nri::RasterizationDesc rasterizationDesc = {};
    nriPipelineDesc.rasterization = &rasterizationDesc;
    rasterizationDesc.viewportNum = 1;
    rasterizationDesc.cullMode = nri::CullMode::NONE;
    rasterizationDesc.sampleNum = 1;
    rasterizationDesc.sampleMask = 0xFFFF;
    rasterizationDesc.conservativeRasterization = pipelineDesc.conservativeRasterization;

    nri::OutputMergerDesc outputMergerDesc = {};
    nriPipelineDesc.outputMerger = &outputMergerDesc;
    outputMergerDesc.colorNum = pipelineDesc.numRenderTargets;
    std::vector<nri::ColorAttachmentDesc> colorAttachments;
    for (uint32_t i = 0; i < outputMergerDesc.colorNum; ++i)
    {
        nri::ColorAttachmentDesc colorAttachment = {};
        colorAttachment.blendEnabled = false;
        colorAttachment.format = m_DebugTexFormat;
        colorAttachment.colorWriteMask = nri::ColorWriteBits::RGBA;
        colorAttachments.push_back(colorAttachment);
    }
    outputMergerDesc.color = colorAttachments.data();
    outputMergerDesc.depth.write = false;

    m_FrameBufferPerPipeline[pipelineId] = outputMergerDesc.colorNum ? &m_FrameBuffers[m_DebugFrameBufferId] : &m_FrameBuffers[m_EmptyFrameBufferId];

    std::vector<nri::ShaderDesc> shaderStages;
    if (pipelineDesc.vertexShader.data)
    {
        nri::ShaderDesc& desc = shaderStages.emplace_back();
        desc.bytecode = pipelineDesc.vertexShader.data;
        desc.size = pipelineDesc.vertexShader.size;
        desc.entryPointName = pipelineDesc.vertexShaderEntryPointName;
        desc.stage = nri::ShaderStage::VERTEX;
    }
    if (pipelineDesc.geometryShader.data)
    {
        nri::ShaderDesc& desc = shaderStages.emplace_back();
        desc.bytecode = pipelineDesc.geometryShader.data;
        desc.size = pipelineDesc.geometryShader.size;
        desc.entryPointName = pipelineDesc.geometryShaderEntryPointName;
        desc.stage = nri::ShaderStage::GEOMETRY;
    }
    if (pipelineDesc.pixelShader.data)
    {
        nri::ShaderDesc& desc = shaderStages.emplace_back();
        desc.bytecode = pipelineDesc.pixelShader.data;
        desc.size = pipelineDesc.pixelShader.size;
        desc.entryPointName = pipelineDesc.pixelShaderEntryPointName;
        desc.stage = nri::ShaderStage::FRAGMENT;
    }

    nriPipelineDesc.shaderStages = shaderStages.data();
    nriPipelineDesc.shaderStageNum = (uint32_t)shaderStages.size();
    NRI_ABORT_ON_FAILURE(NRI.CreateGraphicsPipeline(*m_Device, nriPipelineDesc, m_NriPipelines.emplace_back()));
}

void OmmBakerGpuIntegration::CreateComputePipeline(uint32_t id, const ommGpuPipelineInfoDesc* pipelineInfo)
{
    const ommGpuComputePipelineDesc& pipelineDesc = pipelineInfo->pipelines[id].compute;

    std::vector<nri::DescriptorRangeDesc> descriptorRangeDescs(pipelineDesc.descriptorRangeNum + 1);
    FillDescriptorRangeDescs(pipelineDesc.descriptorRangeNum, pipelineDesc.descriptorRanges, descriptorRangeDescs.data());

    nri::DescriptorRangeDesc& staticSamplersRange = descriptorRangeDescs.back();
    staticSamplersRange.baseRegisterIndex = 0;
    staticSamplersRange.descriptorNum = (uint32_t)m_Samplers.size();
    staticSamplersRange.descriptorType = nri::DescriptorType::SAMPLER;
    staticSamplersRange.visibility = nri::ShaderStage::ALL;

    nri::DescriptorSetDesc descriptorSetDescs = {};
    descriptorSetDescs.rangeNum = (uint32_t)descriptorRangeDescs.size();
    descriptorSetDescs.ranges = descriptorRangeDescs.data();

    nri::DynamicConstantBufferDesc dynamicConstantBufferDesc;
    {
        dynamicConstantBufferDesc.registerIndex = pipelineInfo->globalConstantBufferDesc.registerIndex;
        dynamicConstantBufferDesc.visibility = nri::ShaderStage::COMPUTE;
        descriptorSetDescs.dynamicConstantBufferNum = 1;
        descriptorSetDescs.dynamicConstantBuffers = &dynamicConstantBufferDesc;
    }

    nri::PipelineLayoutDesc layoutDesc = {};
    layoutDesc.descriptorSets = &descriptorSetDescs;
    layoutDesc.descriptorSetNum = 1;
    layoutDesc.stageMask = nri::PipelineLayoutShaderStageBits::COMPUTE;
    nri::PushConstantDesc pushConstantDesc = {};
    {
        pushConstantDesc.registerIndex = pipelineInfo->localConstantBufferDesc.registerIndex;
        pushConstantDesc.size = pipelineInfo->localConstantBufferDesc.maxDataSize;
        pushConstantDesc.visibility = nri::ShaderStage::COMPUTE;
        layoutDesc.pushConstants = &pushConstantDesc;
        layoutDesc.pushConstantNum = 1;
    }
    NRI_ABORT_ON_FAILURE(NRI.CreatePipelineLayout(*m_Device, layoutDesc, m_NriPipelineLayouts.emplace_back()));

    nri::ComputePipelineDesc nriPipelineDesc = {};
    nriPipelineDesc.pipelineLayout = m_NriPipelineLayouts.back();
    nriPipelineDesc.computeShader.bytecode = pipelineDesc.computeShader.data;
    nriPipelineDesc.computeShader.size = pipelineDesc.computeShader.size;
    nriPipelineDesc.computeShader.entryPointName = pipelineDesc.shaderEntryPointName;
    nriPipelineDesc.computeShader.stage = nri::ShaderStage::COMPUTE;
    NRI_ABORT_ON_FAILURE(NRI.CreateComputePipeline(*m_Device, nriPipelineDesc, m_NriPipelines.emplace_back()));
}

inline void FillSamplerDesc(nri::SamplerDesc& nriDesc, const ommGpuStaticSamplerDesc& ommDesc)
{
    nriDesc.addressModes.u = GetNriAddressMode(ommDesc.desc.addressingMode);
    nriDesc.addressModes.v = GetNriAddressMode(ommDesc.desc.addressingMode);
    nriDesc.magnification = GetNriFilterMode(ommDesc.desc.filter);
    nriDesc.minification = GetNriFilterMode(ommDesc.desc.filter);
    nriDesc.mipMax = 16.0f;
    nriDesc.compareFunc = nri::CompareFunc::NONE;
}

void OmmBakerGpuIntegration::CreateSamplers(const ommGpuPipelineInfoDesc* pipelinesInfo)
{
    for (uint32_t i = 0; i < pipelinesInfo->staticSamplersNum; ++i)
    {
        nri::SamplerDesc samplerDesc = {};
        const ommGpuStaticSamplerDesc& ommDesc = pipelinesInfo->staticSamplers[i];
        FillSamplerDesc(samplerDesc, ommDesc);
        nri::Descriptor* descriptor = nullptr;
        NRI_ABORT_ON_FAILURE(NRI.CreateSampler(*m_Device, samplerDesc, descriptor));
        m_Samplers.push_back(descriptor);
    }
}

void OmmBakerGpuIntegration::CreateFrameBuffers(uint32_t pipelineNum)
{
    m_FrameBufferPerPipeline.resize(pipelineNum, nullptr);

    {//create empty framebuffer
        nri::FrameBufferDesc frameBufferDesc = {};
        frameBufferDesc.colorAttachmentNum = 0;
        frameBufferDesc.colorAttachments = nullptr;
        frameBufferDesc.size[0] = (uint16_t)NRI.GetDeviceDesc(*m_Device).frameBufferMaxDim;
        frameBufferDesc.size[1] = (uint16_t)NRI.GetDeviceDesc(*m_Device).frameBufferMaxDim;
        frameBufferDesc.layerNum = 1;
        NRI.CreateFrameBuffer(*m_Device, frameBufferDesc, m_FrameBuffers[m_EmptyFrameBufferId].frameBuffer);
    }

    {//create debug frame buffer
        nri::TextureDesc textureDesc = {};
        textureDesc.arraySize = 1;
        textureDesc.format = m_DebugTexFormat;
        textureDesc.type = nri::TextureType::TEXTURE_2D;
        constexpr uint16_t maxTexSize = 8042;
        textureDesc.size[0] = maxTexSize;
        textureDesc.size[1] = maxTexSize;
        textureDesc.size[2] = 1;
        textureDesc.usageMask = nri::TextureUsageBits::COLOR_ATTACHMENT;
        textureDesc.sampleNum = 1;
        textureDesc.mipNum = 1;
        NRI.CreateTexture(*m_Device, textureDesc, m_FrameBuffers[m_DebugFrameBufferId].texture);

        nri::ResourceGroupDesc resourceGrpoupDesc = {};
        resourceGrpoupDesc.textureNum = 1;
        resourceGrpoupDesc.textures = &m_FrameBuffers[m_DebugFrameBufferId].texture;
        resourceGrpoupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGrpoupDesc, &m_FrameBuffers[m_DebugFrameBufferId].memory));

        nri::Texture2DViewDesc textureViewDesc = {};
        textureViewDesc.viewType = nri::Texture2DViewType::COLOR_ATTACHMENT;
        textureViewDesc.mipNum = 1;
        textureViewDesc.mipOffset = 0;
        textureViewDesc.format = m_DebugTexFormat;
        textureViewDesc.texture = m_FrameBuffers[m_DebugFrameBufferId].texture;
        NRI.CreateTexture2DView(textureViewDesc, m_FrameBuffers[m_DebugFrameBufferId].descriptor);

        nri::FrameBufferDesc frameBufferDesc = {};
        frameBufferDesc.colorAttachmentNum = 1;
        frameBufferDesc.colorAttachments = &m_FrameBuffers[m_DebugFrameBufferId].descriptor;
        NRI.CreateFrameBuffer(*m_Device, frameBufferDesc, m_FrameBuffers[m_DebugFrameBufferId].frameBuffer);
    }
}

void OmmBakerGpuIntegration::CreatePipelines(const ommGpuPipelineInfoDesc* pipelinesInfo)
{
    for (uint32_t i = 0; i < pipelinesInfo->pipelineNum; ++i)
    {
        const ommGpuPipelineDesc& ommPipelineDesc = pipelinesInfo->pipelines[i];
        switch (ommPipelineDesc.type)
        {
        case ommGpuPipelineType_Compute: CreateComputePipeline(i, pipelinesInfo); break;
        case ommGpuPipelineType_Graphics: CreateGraphicsPipeline(i, pipelinesInfo); break;
        default: printf("[FAIL] Invalid ommGpuPipelineType\n"); std::abort();
        }
    }
}

void OmmBakerGpuIntegration::CreateStaticResources(nri::CommandQueue* commandQueue)
{
    ommGpuResourceType staticResources[] = { ommGpuResourceType_STATIC_INDEX_BUFFER, ommGpuResourceType_STATIC_VERTEX_BUFFER };
    nri::BufferUsageBits usageBits[] = { nri::BufferUsageBits::INDEX_BUFFER, nri::BufferUsageBits::VERTEX_BUFFER };
    nri::AccessBits nexAccessBits[] = { nri::AccessBits::INDEX_BUFFER, nri::AccessBits::VERTEX_BUFFER };
    nri::BufferUploadDesc bufferUploadDescs[(uint32_t)GpuStaticResources::Count];
    std::vector<uint8_t> uploadData[(uint32_t)GpuStaticResources::Count];

    for (uint32_t i = 0; i < (uint32_t)GpuStaticResources::Count; ++i)
    {
        size_t outSize = 0;
        ommGpuGetStaticResourceData(staticResources[i], nullptr, &outSize);
        uploadData[i].resize(outSize);
        ommGpuGetStaticResourceData(staticResources[i], uploadData[i].data(), &outSize);

        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = outSize;
        bufferDesc.usageMask = usageBits[i];
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_StaticBuffers[i].buffer));

        nri::BufferUploadDesc& uploadDesc = bufferUploadDescs[i];
        uploadDesc.buffer = m_StaticBuffers[i].buffer;
        uploadDesc.bufferOffset = 0;
        uploadDesc.data = &uploadData[i][0];
        uploadDesc.dataSize = outSize;
        uploadDesc.prevAccess = nri::AccessBits::UNKNOWN;
        uploadDesc.nextAccess = nexAccessBits[i];
    }

    nri::Buffer* buffers[] = { m_StaticBuffers[0].buffer, m_StaticBuffers[1].buffer };
    nri::ResourceGroupDesc resourceGrpoupDesc = {};
    resourceGrpoupDesc.bufferNum = (uint32_t)GpuStaticResources::Count;
    resourceGrpoupDesc.buffers = buffers;
    resourceGrpoupDesc.memoryLocation = nri::MemoryLocation::DEVICE;

    size_t currentMemoryAllocSize = m_NriStaticMemories.size();
    uint32_t allocRequestNum = NRI.CalculateAllocationNumber(*m_Device, resourceGrpoupDesc);
    m_NriStaticMemories.resize(currentMemoryAllocSize + allocRequestNum, nullptr);
    NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGrpoupDesc, m_NriStaticMemories.data() + currentMemoryAllocSize));
    NRI_ABORT_ON_FAILURE(NRI.UploadData(*commandQueue, nullptr, 0, bufferUploadDescs, (uint32_t)GpuStaticResources::Count));
}

void FillDispatchConfigDesc(ommGpuDispatchConfigDesc& dispatchConfigDesc, const InputGeometryDesc& desc)
{
    dispatchConfigDesc = ommGpuDispatchConfigDescDefault();

    const BakerInputs& inputs = desc.inputs;
    const BakerSettings& settings = desc.settings;

    dispatchConfigDesc.alphaTextureWidth = inputs.inTexture.width;
    dispatchConfigDesc.alphaTextureHeight = inputs.inTexture.height;
    dispatchConfigDesc.alphaTextureChannel = inputs.inTexture.alphaChannelId;
    
    dispatchConfigDesc.alphaMode = (ommAlphaMode)settings.alphaMode;
    dispatchConfigDesc.alphaCutoff = settings.alphaCutoff;

    dispatchConfigDesc.indexFormat = GetOmmIndexFormat(inputs.inIndexBuffer.format);
    dispatchConfigDesc.indexCount = (uint32_t)inputs.inIndexBuffer.numElements;
    dispatchConfigDesc.indexStrideInBytes = (uint32_t)inputs.inIndexBuffer.stride;

    dispatchConfigDesc.texCoordFormat = GetOmmTexcoordFormat(inputs.inUvBuffer.format);
    dispatchConfigDesc.texCoordStrideInBytes = (uint32_t)inputs.inUvBuffer.stride;
    dispatchConfigDesc.texCoordOffsetInBytes = (uint32_t)inputs.inUvBuffer.offsetInStruct;

    dispatchConfigDesc.runtimeSamplerDesc.addressingMode = GetOmmAddressingMode(settings.samplerAddressingMode);
    dispatchConfigDesc.runtimeSamplerDesc.filter = GetOmmFilterMode(settings.samplerFilterMode);
    dispatchConfigDesc.runtimeSamplerDesc.borderAlpha = settings.borderAlpha;

    dispatchConfigDesc.globalFormat = ommFormat(settings.globalOMMFormat);

    dispatchConfigDesc.maxSubdivisionLevel = (uint8_t)settings.maxSubdivisionLevel;
    dispatchConfigDesc.enableSubdivisionLevelBuffer = false; // TODO: make a var
    dispatchConfigDesc.maxScratchMemorySize = GetScratchMemoryBudget(settings.maxScratchMemorySize);
    dispatchConfigDesc.dynamicSubdivisionScale = settings.dynamicSubdivisionScale;
    dispatchConfigDesc.bakeFlags = GetBakeFlags(settings.bakeFlags);
    dispatchConfigDesc.maxOutOmmArraySize = uint32_t(~0);
}

void OmmBakerGpuIntegration::GetPrebuildInfo(InputGeometryDesc* geometryDesc, uint32_t geometryNum)
{
    for (uint32_t i = 0; i < geometryNum; ++i)
    {
        InputGeometryDesc& desc = geometryDesc[i];
        ommGpuDispatchConfigDesc dispatchConfigDesc;

        FillDispatchConfigDesc(dispatchConfigDesc, desc);

        ommGpuPreDispatchInfo info = {};
        ommResult ommResult = ommGpuGetPreDispatchInfo(m_Pipeline, &dispatchConfigDesc, &info);
        if (ommResult != ommResult_SUCCESS)
        {
            printf("[FAIL] ommGpuGetPreBakeInfo()\n");
            std::abort();
        }

        PrebuildInfo& prebuildInfo = desc.outputs.prebuildInfo;
        prebuildInfo.arrayDataSize = info.outOmmArraySizeInBytes;
        prebuildInfo.descArraySize = info.outOmmDescSizeInBytes;
        prebuildInfo.indexBufferSize = info.outOmmIndexBufferSizeInBytes;
        prebuildInfo.ommDescArrayHistogramSize = info.outOmmArrayHistogramSizeInBytes;
        prebuildInfo.ommIndexHistogramSize = info.outOmmIndexHistogramSizeInBytes;
        prebuildInfo.postBuildInfoSize = info.outOmmPostDispatchInfoSizeInBytes;
        for (size_t j = 0; j < info.numTransientPoolBuffers; ++j)
            prebuildInfo.transientBufferSizes[j] = info.transientPoolBufferSizeInBytes[j];

        prebuildInfo.indexCount = info.outOmmIndexCount;
        prebuildInfo.indexFormat = GetNriIndexFormat(info.outOmmIndexBufferFormat);
    }
}

void OmmBakerGpuIntegration::AddGeometryToQueue(InputGeometryDesc* geometryDesc, uint32_t geometryNum)
{
    m_GeometryQueue.resize(geometryNum);

    for (uint32_t i = 0; i < geometryNum; ++i)
    {
        GeometryQueueInstance& instance = m_GeometryQueue[i];
        instance.desc = &geometryDesc[i];

        FillDispatchConfigDesc(instance.dispatchConfigDesc, *instance.desc);

        ommGpuPreDispatchInfo info = {};
        ommResult ommResult = ommGpuGetPreDispatchInfo(m_Pipeline, &instance.dispatchConfigDesc, &info);
        if (ommResult != ommResult_SUCCESS)
        {
            printf("[FAIL][OMM][GPU] ommGpuGetPreDispatchInfo failed.\n");
            std::abort();
        }
    }
}

inline uint32_t GetAlignedSize(uint32_t size, uint32_t alignment)
{
    return (((size + alignment - 1) / alignment) * alignment);
}

void OmmBakerGpuIntegration::UpdateGlobalConstantBuffer()
{
    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    uint32_t newConstantBufferViewSize = GetAlignedSize(m_PipelineInfo->globalConstantBufferDesc.maxDataSize, deviceDesc.constantBufferOffsetAlignment);
    uint32_t newConstantBufferSize = newConstantBufferViewSize * (uint32_t)m_GeometryQueue.size();

    if (m_ConstantBufferSize < newConstantBufferSize)
    {
        m_ConstantBufferSize = newConstantBufferSize;
        m_ConstantBufferViewSize = 0;
        if (m_ConstantBuffer)
            NRI.DestroyBuffer(*m_ConstantBuffer);
        nri::BufferDesc bufferDesc = {};
        bufferDesc.size = m_ConstantBufferSize;
        bufferDesc.usageMask = nri::BufferUsageBits::CONSTANT_BUFFER;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_ConstantBuffer));

        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc = {};
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        resourceGroupDesc.bufferNum = 1;
        resourceGroupDesc.buffers = &m_ConstantBuffer;
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, &m_ConstantBufferHeap));
    }

    if (m_ConstantBufferViewSize < newConstantBufferViewSize)
    {
        m_ConstantBufferViewSize = newConstantBufferViewSize;
        if (m_ConstantBufferView)
            NRI.DestroyDescriptor(*m_ConstantBufferView);
        nri::BufferViewDesc constantBufferViewDesc = {};
        constantBufferViewDesc.viewType = nri::BufferViewType::CONSTANT;
        constantBufferViewDesc.buffer = m_ConstantBuffer;
        constantBufferViewDesc.size = m_ConstantBufferViewSize;
        NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(constantBufferViewDesc, m_ConstantBufferView));
    }
}

inline uint64_t ComputeHash(const void* key, uint32_t len, uint32_t geometryId)
{
    const uint8_t* p = (uint8_t*)key;
    uint64_t result = 14695981039346656037ull - geometryId;
    while (len--)
        result = (result ^ (*p++)) * 1099511628211ull;

    return result;
}

void OmmBakerGpuIntegration::UpdateDescriptorPool(uint32_t geometryId, const ommGpuDispatchChain* dispatchChain)
{
    nri::DescriptorPool*& desctriptorPool = m_NriDescriptorPools[geometryId];
    if (desctriptorPool)
        NRI.DestroyDescriptorPool(*desctriptorPool);

    nri::DescriptorPoolDesc desc = {};
    uint32_t dispatchNum = 0;
    uint32_t uniqueDescriptorSetNum = 0;
    for (uint32_t i = 0; i < dispatchChain->numDispatches; ++i)
    {//filter out labling events
        switch (dispatchChain->dispatches[i].type)
        {
        case ommGpuDispatchType_BeginLabel:
        case ommGpuDispatchType_EndLabel: break;
        default: 
        {
            uint64_t hash = ComputeHash(dispatchChain->dispatches[i].compute.resources, dispatchChain->dispatches[i].compute.resourceNum * sizeof(ommGpuResource), geometryId);
            const auto& it = m_NriDescriptorSets.find(hash);
            if (it == m_NriDescriptorSets.end())
            {
                m_NriDescriptorSets.insert(std::make_pair(hash, nullptr));
                ++uniqueDescriptorSetNum;

            for (uint32_t j = 0; j < dispatchChain->dispatches[i].compute.resourceNum; ++j)
            {
                const ommGpuResource& resource = dispatchChain->dispatches[i].compute.resources[j];
                switch (resource.stateNeeded)
                {
                    case ommGpuDescriptorType_TextureRead: ++desc.textureMaxNum; break;
                    case ommGpuDescriptorType_BufferRead: ++desc.bufferMaxNum; break;
                    case ommGpuDescriptorType_RawBufferRead: ++desc.structuredBufferMaxNum; break;
                    case ommGpuDescriptorType_RawBufferWrite: ++desc.storageStructuredBufferMaxNum; break;
                    default: break;
                }
            }
            }
            ++dispatchNum;
        }
        }
    }

    desc.descriptorSetMaxNum = uniqueDescriptorSetNum;
    desc.dynamicConstantBufferMaxNum = dispatchNum;
    desc.samplerMaxNum = uniqueDescriptorSetNum * (uint32_t)m_Samplers.size();
    NRI_ABORT_ON_FAILURE(NRI.CreateDescriptorPool(*m_Device, desc, desctriptorPool));
}

uint64_t CalculateDescriptorKey(uint32_t geometryId, const ommGpuResource& resource)
{
    bool isTransientPool = resource.type == ommGpuResourceType_TRANSIENT_POOL_BUFFER;
    uint64_t key = isTransientPool ? 0 : geometryId + 1;
    key |= uint64_t(resource.type) << 32ull;
    key |= uint64_t(resource.stateNeeded) << 40ull;
    key |= uint64_t(resource.indexInPool) << 48ull;
    return key;
}

nri::Descriptor* OmmBakerGpuIntegration::GetDescriptor(const ommGpuResource& resource, uint32_t geometryId)
{
    uint64_t key = CalculateDescriptorKey(geometryId, resource);
    nri::Descriptor* descriptor = nullptr;
    const auto& it = m_NriDescriptors.find(key);
    if (it == m_NriDescriptors.end())
    {
        BakerInputs& inputs = m_GeometryQueue[geometryId].desc->inputs;
        bool isTexture = resource.stateNeeded == ommGpuDescriptorType_TextureRead;
        bool isRaw = (resource.stateNeeded == ommGpuDescriptorType_RawBufferRead) || (resource.stateNeeded == ommGpuDescriptorType_RawBufferWrite);
        if (isTexture)
        {
            nri::Texture2DViewDesc texDesc = {};
            texDesc.mipNum = 1;
            texDesc.mipOffset = (uint16_t)inputs.inTexture.mipOffset;
            texDesc.viewType = nri::Texture2DViewType::SHADER_RESOURCE_2D;
            texDesc.format = inputs.inTexture.format;
            texDesc.texture = inputs.inTexture.texture;
            NRI_ABORT_ON_FAILURE(NRI.CreateTexture2DView(texDesc, descriptor));
        }
        else
        {
            const BufferResource& buffer = GetBuffer(resource, geometryId);
            nri::BufferViewDesc bufferDesc = {};
            bufferDesc.buffer = buffer.buffer;
            bufferDesc.offset = buffer.offset;
            bufferDesc.format = isRaw ? nri::Format::UNKNOWN : buffer.format;
            bufferDesc.size = buffer.size - buffer.offset;
            bufferDesc.viewType = GetNriBufferViewType(resource.stateNeeded);
            NRI_ABORT_ON_FAILURE(NRI.CreateBufferView(bufferDesc, descriptor));
        }
        m_NriDescriptors.insert(std::make_pair(key, descriptor));
    }
    else
        descriptor = it->second;

    return descriptor;
}

void OmmBakerGpuIntegration::PerformResourceTransition(const ommGpuResource& resource, uint32_t geometryId, std::vector<nri::BufferTransitionBarrierDesc>& bufferBarriers)
{
    if (resource.type == ommGpuResourceType_IN_ALPHA_TEXTURE)
        return;

    BufferResource& bufferResource = GetBuffer(resource, geometryId);
    nri::AccessBits currentState = bufferResource.state;
    nri::AccessBits requestedState = GetNriResourceState(resource.stateNeeded);

    if (currentState != requestedState)
    {
        nri::BufferTransitionBarrierDesc& barrier = bufferBarriers.emplace_back();
        barrier.nextAccess = requestedState;
        barrier.prevAccess = currentState;
        barrier.buffer = bufferResource.buffer;

        bufferResource.state = requestedState;
    }
}

nri::DescriptorSet* OmmBakerGpuIntegration::PrepareDispatch(nri::CommandBuffer& commandBuffer, const ommGpuResource* resources, uint32_t resourceNum, uint32_t pipelineIndex, uint32_t geometryId)
{
    std::vector<nri::Descriptor*> descriptors;
    descriptors.resize(resourceNum);

    //process requested resources. prepare range updates. perform transitions
    std::vector<nri::DescriptorRangeUpdateDesc> rangeUpdateDescs;
    std::vector<nri::BufferTransitionBarrierDesc> bufferTransitions;
    nri::DescriptorType prevRangeType = nri::DescriptorType::MAX_NUM;
    for (uint32_t i = 0; i < resourceNum; ++i)
    {
        const ommGpuResource& resource = resources[i];
        nri::DescriptorType rangeType = GetNriDescriptorType(resource.stateNeeded);
        if (rangeType != prevRangeType)
        {
            nri::DescriptorRangeUpdateDesc nextRange = {};
            nextRange.descriptors = descriptors.data() + i;
            rangeUpdateDescs.push_back(nextRange);
            prevRangeType = rangeType;
        }

        nri::DescriptorRangeUpdateDesc& currentRange = rangeUpdateDescs.back();
        descriptors[i] = GetDescriptor(resources[i], geometryId);
        currentRange.descriptorNum += 1;
        PerformResourceTransition(resource, geometryId, bufferTransitions);
    }

    nri::DescriptorRangeUpdateDesc& staticSamlersRange = rangeUpdateDescs.emplace_back();
    staticSamlersRange.descriptors = m_Samplers.data();
    staticSamlersRange.descriptorNum = (uint32_t)m_Samplers.size();
    staticSamlersRange.offsetInRange = 0;

    nri::TransitionBarrierDesc transitionBarriers = {};
    transitionBarriers.bufferNum = (uint32_t)bufferTransitions.size();
    transitionBarriers.buffers = bufferTransitions.data();
    if (transitionBarriers.bufferNum)
        NRI.CmdPipelineBarrier(commandBuffer, &transitionBarriers, nullptr, nri::BarrierDependency::ALL_STAGES);

    nri::PipelineLayout*& pipelineLayout = m_NriPipelineLayouts[pipelineIndex];
    NRI.CmdSetPipelineLayout(commandBuffer, *pipelineLayout);

    // Descriptor set
    uint64_t hash = ComputeHash(resources, resourceNum * sizeof(ommGpuResource), geometryId);
    const auto& it = m_NriDescriptorSets.find(hash);
    nri::DescriptorSet* descriptorSet = nullptr;
    if (it->second == nullptr)
    {
        NRI_ABORT_ON_FAILURE(NRI.AllocateDescriptorSets(*m_NriDescriptorPools[geometryId], *pipelineLayout, 0, &descriptorSet, 1, nri::WHOLE_DEVICE_GROUP, 0));
        NRI.UpdateDescriptorRanges(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, (uint32_t)rangeUpdateDescs.size(), rangeUpdateDescs.data());
        it->second = descriptorSet;
        NRI.UpdateDynamicConstantBuffers(*descriptorSet, nri::WHOLE_DEVICE_GROUP, 0, 1, &m_ConstantBufferView);
    }
    else
        descriptorSet = it->second;

    NRI.CmdSetPipeline(commandBuffer, *m_NriPipelines[pipelineIndex]);

    return descriptorSet;
}

void OmmBakerGpuIntegration::InsertUavBarriers(nri::CommandBuffer& commandBuffer, const ommGpuResource* resources, uint32_t resourceNum, uint32_t geometryId)
{
    std::vector<nri::BufferTransitionBarrierDesc> uavBarriers;
    for (uint32_t i = 0; i < resourceNum; ++i)
    {
        if (resources[i].stateNeeded == ommGpuDescriptorType_RawBufferWrite)
        {
            nri::BufferTransitionBarrierDesc barrier = 
            { 
                GetBuffer(resources[i], geometryId).buffer,
                nri::AccessBits::SHADER_RESOURCE_STORAGE,
                nri::AccessBits::SHADER_RESOURCE_STORAGE
            };
            uavBarriers.push_back(barrier);
        }
    }
    nri::TransitionBarrierDesc transition = {};
    transition.bufferNum = (uint32_t)uavBarriers.size();
    transition.buffers = uavBarriers.data();
    NRI.CmdPipelineBarrier(commandBuffer, &transition, nullptr, nri::BarrierDependency::ALL_STAGES);
}

void OmmBakerGpuIntegration::DispatchCompute(nri::CommandBuffer& commandBuffer, const ommGpuComputeDesc& desc, uint32_t geometryId)
{
    nri::DescriptorSet* descriptorSet = PrepareDispatch(commandBuffer, desc.resources, desc.resourceNum, desc.pipelineIndex, geometryId);

    if (desc.localConstantBufferDataSize)
        NRI.CmdSetConstants(commandBuffer, 0, desc.localConstantBufferData, desc.localConstantBufferDataSize);

    uint32_t constantBufferOffset = m_ConstantBufferOffset;
    NRI.CmdSetDescriptorSet(commandBuffer, 0, *descriptorSet, &constantBufferOffset);

    NRI.CmdDispatch(commandBuffer, desc.gridWidth, desc.gridHeight, 1);
    InsertUavBarriers(commandBuffer, desc.resources, desc.resourceNum, geometryId);
}

void OmmBakerGpuIntegration::DispatchComputeIndirect(nri::CommandBuffer& commandBuffer, const ommGpuComputeIndirectDesc& desc, uint32_t geometryId)
{
    nri::DescriptorSet* descriptorSet = PrepareDispatch(commandBuffer, desc.resources, desc.resourceNum, desc.pipelineIndex, geometryId);
    
    if (desc.localConstantBufferDataSize)
        NRI.CmdSetConstants(commandBuffer, 0, desc.localConstantBufferData, desc.localConstantBufferDataSize);

    uint32_t constantBufferOffset = m_ConstantBufferOffset;
    NRI.CmdSetDescriptorSet(commandBuffer, 0, *descriptorSet, &constantBufferOffset);

    BufferResource& argBuffer = GetBuffer(desc.indirectArg, geometryId);
    if (argBuffer.state != nri::AccessBits::ARGUMENT_BUFFER)
    {
        nri::BufferTransitionBarrierDesc bufferBarrier = { argBuffer.buffer, argBuffer.state, nri::AccessBits::ARGUMENT_BUFFER };
        nri::TransitionBarrierDesc transitionDesc = { &bufferBarrier, nullptr, 1, 0 };
        NRI.CmdPipelineBarrier(commandBuffer, &transitionDesc, nullptr, nri::BarrierDependency::ALL_STAGES);
        argBuffer.state = nri::AccessBits::ARGUMENT_BUFFER;
    }
    NRI.CmdDispatchIndirect(commandBuffer, *argBuffer.buffer, desc.indirectArgByteOffset);
    InsertUavBarriers(commandBuffer, desc.resources, desc.resourceNum, geometryId);
}

void OmmBakerGpuIntegration::DispatchDrawIndexedIndirect(nri::CommandBuffer& commandBuffer, const ommGpuDrawIndexedIndirectDesc& desc, uint32_t geometryId)
{
    nri::DescriptorSet* descriptorSet = PrepareDispatch(commandBuffer, desc.resources, desc.resourceNum, desc.pipelineIndex, geometryId);

    if (desc.localConstantBufferDataSize)
        NRI.CmdSetConstants(commandBuffer, 0, desc.localConstantBufferData, desc.localConstantBufferDataSize);

    uint32_t constantBufferOffset = m_ConstantBufferOffset;
    NRI.CmdSetDescriptorSet(commandBuffer, 0, *descriptorSet, &constantBufferOffset);

    BufferResource& argBuffer = GetBuffer(desc.indirectArg, geometryId);
    if (argBuffer.state != nri::AccessBits::ARGUMENT_BUFFER)
    {
        nri::BufferTransitionBarrierDesc bufferBarrier = { argBuffer.buffer, argBuffer.state, nri::AccessBits::ARGUMENT_BUFFER };
        nri::TransitionBarrierDesc transitionDesc = { &bufferBarrier, nullptr, 1, 0 };
        NRI.CmdPipelineBarrier(commandBuffer, &transitionDesc, nullptr, nri::BarrierDependency::ALL_STAGES);
        argBuffer.state = nri::AccessBits::ARGUMENT_BUFFER;
    }

    FrameBuffer* frameBuffer = m_FrameBufferPerPipeline[desc.pipelineIndex];
    if (frameBuffer->texture && frameBuffer->state != nri::AccessBits::COLOR_ATTACHMENT)
    { // perform debug frame buffer transition
        nri::TextureTransitionBarrierDesc textureBarrierDesc = {};
        textureBarrierDesc.texture = frameBuffer->texture;
        textureBarrierDesc.mipNum = 1;
        textureBarrierDesc.prevAccess = frameBuffer->state;
        textureBarrierDesc.nextAccess = nri::AccessBits::COLOR_ATTACHMENT;
        textureBarrierDesc.prevLayout = nri::TextureLayout::GENERAL;
        textureBarrierDesc.nextLayout = nri::TextureLayout::COLOR_ATTACHMENT;
        nri::TransitionBarrierDesc transitionDesc = { nullptr, &textureBarrierDesc, 0, 1 };
        NRI.CmdPipelineBarrier(commandBuffer, &transitionDesc, nullptr, nri::BarrierDependency::ALL_STAGES);
        frameBuffer->state = nri::AccessBits::COLOR_ATTACHMENT;
    }

    NRI.CmdBeginRenderPass(commandBuffer, *frameBuffer->frameBuffer, nri::RenderPassBeginFlag::SKIP_FRAME_BUFFER_CLEAR);
    {
        BufferResource& indexBuffer = GetBuffer(desc.indexBuffer, geometryId);
        NRI.CmdSetIndexBuffer(commandBuffer, *indexBuffer.buffer, desc.indexBufferOffset, nri::IndexType::UINT32);

        BufferResource& vertexBuffer = GetBuffer(desc.vertexBuffer, geometryId);
        uint64_t offset[] = { desc.vertexBufferOffset };
        NRI.CmdSetVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offset);

        const nri::Viewport viewport = { desc.viewport.minWidth, desc.viewport.minHeight, desc.viewport.maxWidth, desc.viewport.maxHeight, 0.0f, 1.0f };
        NRI.CmdSetViewports(commandBuffer, &viewport, 1);
        const nri::Rect scissorRect = { (int32_t)desc.viewport.minWidth, (int32_t)desc.viewport.minHeight, (uint32_t)desc.viewport.maxWidth, (uint32_t)desc.viewport.maxHeight };
        NRI.CmdSetScissors(commandBuffer, &scissorRect, 1);

        NRI.CmdDrawIndexedIndirect(commandBuffer, *argBuffer.buffer, desc.indirectArgByteOffset, 1, 20);//TODO: replace last constant with a GAPI related var
    }
    NRI.CmdEndRenderPass(commandBuffer);

    InsertUavBarriers(commandBuffer, desc.resources, desc.resourceNum, geometryId);
}

void PostBakeBufferTransition(std::vector<nri::BufferTransitionBarrierDesc>& transition, BufferResource& buffer)
{
    if (buffer.buffer)
        if (buffer.state != nri::AccessBits::UNKNOWN)
            transition.push_back({ buffer.buffer, buffer.state, nri::AccessBits::UNKNOWN });
}

void OmmBakerGpuIntegration::GenerateVisibilityMaskGPU(nri::CommandBuffer& commandBuffer, uint32_t geometryId)
{
    GeometryQueueInstance& instance = m_GeometryQueue[geometryId];
    ommGpuDispatchConfigDesc& dispatchConfigDesc = instance.dispatchConfigDesc;

    const ommGpuDispatchChain* dispatchChain = nullptr;
    ommGpuDispatch(m_Pipeline, &dispatchConfigDesc, &dispatchChain);

    //Update and set descriptor pool
    UpdateDescriptorPool(geometryId, dispatchChain);
    NRI.CmdSetDescriptorPool(commandBuffer, *m_NriDescriptorPools[geometryId]);

    // Upload constants
    if (dispatchChain->globalCBufferDataSize)
    {
        if (m_ConstantBufferOffset + m_ConstantBufferViewSize > m_ConstantBufferSize)
            m_ConstantBufferOffset = 0;

        void* data = NRI.MapBuffer(*m_ConstantBuffer, m_ConstantBufferOffset, dispatchChain->globalCBufferDataSize);
        memcpy(data, dispatchChain->globalCBufferData, dispatchChain->globalCBufferDataSize);
        NRI.UnmapBuffer(*m_ConstantBuffer);
    }

    for (uint32_t i = 0; i < dispatchChain->numDispatches; ++i)
    {
        const ommGpuDispatchDesc& dispacthDesc = dispatchChain->dispatches[i];
        switch (dispacthDesc.type)
        {
        case ommGpuDispatchType_BeginLabel: NRI.CmdBeginAnnotation(commandBuffer, dispacthDesc.beginLabel.debugName); break;
        case ommGpuDispatchType_Compute:
        {
            const ommGpuComputeDesc& desc = dispacthDesc.compute;
            DispatchCompute(commandBuffer, desc, geometryId);
            break;
        }
        case ommGpuDispatchType_ComputeIndirect:
        {
            const ommGpuComputeIndirectDesc& desc = dispacthDesc.computeIndirect;
            DispatchComputeIndirect(commandBuffer, desc, geometryId);
            break;
        }
        case ommGpuDispatchType_DrawIndexedIndirect:
        {
            const ommGpuDrawIndexedIndirectDesc& desc = dispacthDesc.drawIndexedIndirect;
            DispatchDrawIndexedIndirect(commandBuffer, desc, geometryId);
            break;
        }
        case ommGpuDispatchType_EndLabel: NRI.CmdEndAnnotation(commandBuffer); break;
        default: break;
        }
    }
    m_ConstantBufferOffset += m_ConstantBufferViewSize;

    BakerOutputs& outputs = instance.desc->outputs;
    BakerInputs& inputs = instance.desc->inputs;
    std::vector<nri::BufferTransitionBarrierDesc> outputBuffersTransition = {};
    PostBakeBufferTransition(outputBuffersTransition, outputs.outArrayData);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outDescArray);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outIndexBuffer);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outArrayHistogram);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outIndexHistogram);
    PostBakeBufferTransition(outputBuffersTransition, outputs.outPostBuildInfo);

    for (size_t i = 0; i < OMM_MAX_TRANSIENT_POOL_BUFFERS; ++i)
        PostBakeBufferTransition(outputBuffersTransition, inputs.inTransientPool[i]);

    nri::TransitionBarrierDesc transitionDesc = { outputBuffersTransition.data(), nullptr, (uint32_t)outputBuffersTransition.size(), 0 };
    NRI.CmdPipelineBarrier(commandBuffer, &transitionDesc, nullptr, nri::BarrierDependency::ALL_STAGES);
}

void OmmBakerGpuIntegration::Bake(nri::CommandBuffer& commandBuffer, InputGeometryDesc* geometryDesc, uint32_t geometryNum)
{
    if (!geometryNum)
        return;

    AddGeometryToQueue(geometryDesc, geometryNum);
    UpdateGlobalConstantBuffer();
    m_NriDescriptorPools.resize(geometryNum);

    for (uint32_t i = 0; i < geometryNum; ++i)
        GenerateVisibilityMaskGPU(commandBuffer, i);

    m_GeometryQueue.clear();
}

void OmmBakerGpuIntegration::ReleaseTemporalResources()
{
    m_GeometryQueue.resize(0);
    m_GeometryQueue.shrink_to_fit();
    m_NriDescriptorSets.clear();

    for (auto it = m_NriDescriptors.begin(); it != m_NriDescriptors.end(); )
    {
        if (it->second)
            NRI.DestroyDescriptor(*it->second);
        it = m_NriDescriptors.erase(it);
    }

    for (auto& pool : m_NriDescriptorPools)
    {
        if (pool)
        {
            NRI.DestroyDescriptorPool(*pool);
            pool = nullptr;
        }
    }
    m_NriDescriptorPools.resize(0);
    m_NriDescriptorPools.shrink_to_fit();

    if (m_ConstantBuffer) NRI.DestroyBuffer(*m_ConstantBuffer);
    if (m_ConstantBufferView) NRI.DestroyDescriptor(*m_ConstantBufferView);
    if (m_ConstantBufferHeap) NRI.FreeMemory(*m_ConstantBufferHeap);
    m_ConstantBufferViewSize = m_ConstantBufferSize = m_ConstantBufferOffset = 0;
    m_ConstantBuffer = nullptr;
    m_ConstantBufferView = nullptr;
    m_ConstantBufferHeap = nullptr;
}


void OmmBakerGpuIntegration::Destroy()
{
    for (auto& frameBuffer : m_FrameBuffers)
    {
        if (frameBuffer.descriptor)
        {
            NRI.DestroyDescriptor(*frameBuffer.descriptor);
            frameBuffer.descriptor = nullptr;
        }
        if (frameBuffer.texture)
        {
            NRI.DestroyTexture(*frameBuffer.texture);
            frameBuffer.texture = nullptr;
        }
        if (frameBuffer.frameBuffer)
        {
            NRI.DestroyFrameBuffer(*frameBuffer.frameBuffer);
            frameBuffer.frameBuffer = nullptr;
        }
        if (frameBuffer.memory)
        {
            NRI.FreeMemory(*frameBuffer.memory);
            frameBuffer.memory = nullptr;
        }
    }
    m_FrameBufferPerPipeline.resize(0);
    m_FrameBufferPerPipeline.shrink_to_fit();

    for (auto& sampler : m_Samplers)
        if (sampler) NRI.DestroyDescriptor(*sampler);

    for (auto& pipeline : m_NriPipelines)
        if (pipeline) NRI.DestroyPipeline(*pipeline);

    for (auto& layout : m_NriPipelineLayouts)
        if (layout) NRI.DestroyPipelineLayout(*layout);

    for (uint32_t i = 0; i < (uint32_t)GpuStaticResources::Count; ++i)
    {
        if (m_StaticBuffers[i].buffer)
            NRI.DestroyBuffer(*m_StaticBuffers[i].buffer);
    }

    for (auto& memory : m_NriStaticMemories)
        if (memory) NRI.FreeMemory(*memory);

    ommGpuDestroyPipeline(m_GpuBaker, m_Pipeline);
    ommDestroyBaker(m_GpuBaker);
}
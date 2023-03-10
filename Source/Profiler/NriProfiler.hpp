/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once
#include <string>
#include <vector>
#include <array>
#include "NRI.h"
#include "NRI/Include/Extensions/NRIHelper.h"

#define PROFILER_BUFFERED_FRAME_NUM 3 //TODO: this should be part of the profiler initialization.

struct ProfilerEvent
{ // TODO: simplify
    void Update(double elapasedTime);
    double GetTotalAccumulated() const { return total / double(queryNum); };
    double GetImmediateDelta() const { return immediate; };
    double GetSmoothDelta() const { return smooth; };
    double GetSmootherDelta() const { return smoother; };
    std::string name;
private:
    double total = 0.0;
    double immediate = 0.0;
    double smooth = 0.0;
    double smoother = 0.0;
    uint32_t queryNum = 0;
};

void ProfilerEvent::Update(double elapasedTime)
{
    immediate = elapasedTime;
    total += elapasedTime;

    double relativeDelta = fabs(immediate - smooth) / (std::min(immediate, smooth) + 1e-7);
    double f = relativeDelta / (1.0 + relativeDelta);
    smooth = smooth + (immediate - smooth) * std::max(f, 1.0 / 32.0);
    smoother = smoother + (smooth - smoother) * std::max(f, 1.0 / 64.0);
};

struct ProfilerTimestamp
{
    uint32_t eventID;
    uint32_t timestampID;

    ProfilerTimestamp(uint32_t name, uint32_t timestamp) :
        eventID(name),
        timestampID(timestamp) {};

};

struct ProfilerContext
{
    std::vector<ProfilerTimestamp> timestamps;
    nri::CommandBuffer* commandBuffer = nullptr;
};

class Profiler
{
public:
    void Init(nri::Device* device);

    void BeginFrame();
    void EndFrame(nri::CommandBuffer* lastCommandBufferInFrame);
    void ResolveBufferedFrame();

    ProfilerContext* BeginContext(nri::CommandBuffer* commandBuffer);
    uint32_t AllocateEvent(const char* eventName);
    uint32_t BeginTimestamp(ProfilerContext* ctx, uint32_t eventID);
    void EndTimestamp(ProfilerContext* ctx, uint32_t timestampID);
    void ProcessContexts(const nri::WorkSubmissionDesc& desc);

    const ProfilerEvent* GetPerformanceEvents(size_t& count) const { count = m_Events.size(); return m_Events.data(); };

    void Destroy();

private:
    struct NRIInterface
        : public nri::CoreInterface
        , public nri::HelperInterface
    {};

    std::array<std::vector<ProfilerContext>, PROFILER_BUFFERED_FRAME_NUM> m_Contexts{};
    std::array<nri::QueryPool*, PROFILER_BUFFERED_FRAME_NUM> m_QueryPools{};
    std::array<nri::Buffer*, PROFILER_BUFFERED_FRAME_NUM> m_QueryBuffers{};
    std::vector<nri::Memory*> m_Memories;
    std::vector<ProfilerEvent> m_Events;

    NRIInterface m_NRI;

    uint64_t m_TimestampFrequencyHz;

    const uint32_t m_QueriesNum = 16;
    const uint32_t m_QueryBufferSize = m_QueriesNum * sizeof(uint64_t);
    uint32_t m_CurrentTimestampID = uint32_t(-1);
    uint32_t m_CurrentFrameID = uint32_t(-1);
    uint32_t m_BufferedFrameID;
    uint32_t m_OldestBufferedFrameID;
};

void Profiler::ResolveBufferedFrame()
{
    if (m_CurrentFrameID < (PROFILER_BUFFERED_FRAME_NUM - 1))
        return;

    std::vector<uint64_t> dstBuffer;
    dstBuffer.resize(m_QueriesNum);

    void* srcBuffer = m_NRI.MapBuffer(*m_QueryBuffers[m_OldestBufferedFrameID], 0, m_QueryBufferSize);
    memcpy_s(dstBuffer.data(), dstBuffer.size() * sizeof(uint64_t), srcBuffer, m_QueryBufferSize);
    m_NRI.UnmapBuffer(*m_QueryBuffers[m_OldestBufferedFrameID]);

    std::vector<ProfilerContext>& targetFrameContexts = m_Contexts[m_OldestBufferedFrameID];
    for (uint32_t i = 0; i < helper::GetCountOf(targetFrameContexts); ++i)
    {
        ProfilerContext& ctx = targetFrameContexts[i];
        for (uint32_t j = 0; j < helper::GetCountOf(ctx.timestamps); j++)
        {
            ProfilerTimestamp& timestamp = ctx.timestamps[j];
            ProfilerEvent& perfEvent = m_Events[timestamp.eventID];

            uint32_t beginId = timestamp.timestampID * 2;
            uint32_t endId = beginId + 1;

            uint64_t begin = dstBuffer[beginId];
            uint64_t end = dstBuffer[endId];

            if (end > begin)
            {
                uint64_t delta = end - begin;
                double elapsedTime = ((double)delta / (double)m_TimestampFrequencyHz) * 1000.0;
                perfEvent.Update(elapsedTime);
            }
        }
    }
    targetFrameContexts.resize(0);
}

uint32_t Profiler::AllocateEvent(const char* eventName)
{
    uint32_t id = helper::GetCountOf(m_Events);
    ProfilerEvent perfevent;
    perfevent.name = eventName;
    m_Events.push_back(perfevent);
    return id;
}

ProfilerContext* Profiler::BeginContext(nri::CommandBuffer* commandBuffer)
{
    ProfilerContext ctx;
    ctx.commandBuffer = commandBuffer;
    m_Contexts[m_BufferedFrameID].push_back(ctx);
    return &m_Contexts[m_BufferedFrameID].back();
}

void Profiler::Init(nri::Device* device)
{
    NRI_ABORT_ON_FAILURE(nri::GetInterface(*device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&m_NRI));
    NRI_ABORT_ON_FAILURE(nri::GetInterface(*device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&m_NRI));
    nri::CommandQueue* commandQueue = nullptr;
    m_NRI.GetCommandQueue(*device, nri::CommandQueueType::GRAPHICS, commandQueue);

    //Query Buffers
    nri::BufferDesc bufferDesc = {};
    bufferDesc.size = m_QueryBufferSize;
    bufferDesc.usageMask = nri::BufferUsageBits::NONE;
    bufferDesc.structureStride = sizeof(uint64_t);
    for (uint32_t i = 0; i < PROFILER_BUFFERED_FRAME_NUM; ++i)
        NRI_ABORT_ON_FAILURE(m_NRI.CreateBuffer(*device, bufferDesc, m_QueryBuffers[i]));

    nri::ResourceGroupDesc rgDesc = {};
    rgDesc.bufferNum = PROFILER_BUFFERED_FRAME_NUM;
    rgDesc.buffers = m_QueryBuffers.data();
    rgDesc.memoryLocation = nri::MemoryLocation::HOST_READBACK;
    rgDesc.textures = nullptr;
    rgDesc.textureNum = 0;
    size_t currentMemoryAllocSize = m_Memories.size();
    uint32_t allocRequestNum = m_NRI.CalculateAllocationNumber(*device, rgDesc);
    m_Memories.resize(currentMemoryAllocSize + allocRequestNum, nullptr);
    NRI_ABORT_ON_FAILURE(m_NRI.AllocateAndBindMemory(*device, rgDesc, m_Memories.data() + currentMemoryAllocSize));

    //Query Pool
    nri::QueryPoolDesc queryPoolDesc = {};
    queryPoolDesc.queryType = nri::QueryType::TIMESTAMP;
    queryPoolDesc.capacity = m_QueriesNum;
    queryPoolDesc.pipelineStatsMask = {};
    queryPoolDesc.physicalDeviceMask = nri::WHOLE_DEVICE_GROUP;
    for (uint32_t i = 0; i < PROFILER_BUFFERED_FRAME_NUM; ++i)
        NRI_ABORT_ON_FAILURE(m_NRI.CreateQueryPool(*device, queryPoolDesc, m_QueryPools[i]));

    m_TimestampFrequencyHz = m_NRI.GetDeviceDesc(*device).timestampFrequencyHz;
    nri::CommandAllocator* commandAllocator = nullptr;
    nri::CommandBuffer* commandBuffer = nullptr;
    NRI_ABORT_ON_FAILURE(m_NRI.CreateCommandAllocator(*commandQueue, nri::WHOLE_DEVICE_GROUP, commandAllocator));
    NRI_ABORT_ON_FAILURE(m_NRI.CreateCommandBuffer(*commandAllocator, commandBuffer));

    m_NRI.ResetCommandAllocator(*commandAllocator);
    m_NRI.BeginCommandBuffer(*commandBuffer, nullptr, 0);
    {
        for (uint32_t i = 0; i < PROFILER_BUFFERED_FRAME_NUM; ++i)
        {
            m_NRI.CmdResetQueries(*commandBuffer, *m_QueryPools[i], 0, m_QueriesNum);
        }
    }
    m_NRI.EndCommandBuffer(*commandBuffer);

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBufferNum = 1;
    workSubmissionDesc.commandBuffers = &commandBuffer;
    workSubmissionDesc.wait = nullptr;
    workSubmissionDesc.waitNum = 0;
    workSubmissionDesc.signal = nullptr;
    workSubmissionDesc.signalNum = 0;

    m_NRI.SubmitQueueWork(*commandQueue, workSubmissionDesc, nullptr);
    m_NRI.WaitForIdle(*commandQueue);
    m_NRI.DestroyCommandBuffer(*commandBuffer);
    m_NRI.DestroyCommandAllocator(*commandAllocator);
}

void Profiler::ProcessContexts(const nri::WorkSubmissionDesc& desc)
{
    std::vector<ProfilerContext>& contexts = m_Contexts[m_BufferedFrameID];
    std::vector<ProfilerContext> sortedContexts = {};

    for (uint32_t i = 0; i < desc.commandBufferNum; ++i)
    {
        const nri::CommandBuffer* cmdBuffer = desc.commandBuffers[i];
        for (auto ctx : contexts)
        {
            if (ctx.commandBuffer == cmdBuffer)
            {
                sortedContexts.push_back(ctx);
                break;
            }
        }
    }
    contexts = sortedContexts;
}

void Profiler::BeginFrame()
{
    ++m_CurrentFrameID;
    m_CurrentTimestampID = uint32_t(-1);
    m_BufferedFrameID = m_CurrentFrameID % PROFILER_BUFFERED_FRAME_NUM;
    m_OldestBufferedFrameID = (m_CurrentFrameID + 1) % PROFILER_BUFFERED_FRAME_NUM;
    ResolveBufferedFrame();
}

void Profiler::EndFrame(nri::CommandBuffer* lastCommandBufferToExecute)
{
    uint32_t numQueries = (m_CurrentTimestampID + 1) * 2;
    m_NRI.CmdCopyQueries(*lastCommandBufferToExecute, *m_QueryPools[m_BufferedFrameID], 0, numQueries, *m_QueryBuffers[m_BufferedFrameID], 0);
    m_NRI.CmdResetQueries(*lastCommandBufferToExecute, *m_QueryPools[m_BufferedFrameID], 0, m_QueriesNum);
}

uint32_t Profiler::BeginTimestamp(ProfilerContext* ctx, uint32_t eventID)
{
    ctx->timestamps.push_back(ProfilerTimestamp(eventID, ++m_CurrentTimestampID));
    m_NRI.CmdEndQuery(*ctx->commandBuffer, *m_QueryPools[m_BufferedFrameID], m_CurrentTimestampID * 2);

    return m_CurrentTimestampID;
}

void Profiler::EndTimestamp(ProfilerContext* ctx, uint32_t timestampID)
{
    m_NRI.CmdEndQuery(*ctx->commandBuffer, *m_QueryPools[m_BufferedFrameID], timestampID * 2 + 1);
}

void Profiler::Destroy()
{
    for (auto& buffer : m_QueryBuffers)
        m_NRI.DestroyBuffer(*buffer);
    for (auto& pool : m_QueryPools)
        m_NRI.DestroyQueryPool(*pool);
    for (auto& memory : m_Memories)
        m_NRI.FreeMemory(*memory);
    m_Memories.resize(0);
    m_Memories.shrink_to_fit();
    m_Events.resize(0);
    m_Events.shrink_to_fit();
};
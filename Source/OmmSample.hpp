/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <set>
#include <map>

#include "VisibilityMasks/OmmHelper.h"
#include "NRIFramework.h"

#include "Extensions/NRIRayTracing.h"
#include "Extensions/NRIWrapperD3D11.h"
#include "Extensions/NRIWrapperD3D12.h"
#include "Extensions/NRIWrapperVK.h"

#include "NRD.h"
#include "NRDIntegration.hpp"

#include "DLSS/DLSSIntegration.hpp"
#include "NGX/NVIDIAImageScaling/NIS/NIS_Config.h"

#include "Detex/detex.h"
#include "Profiler/NriProfiler.hpp"


//=================================================================================
// Settings
//=================================================================================

// Fused or separate denoising selection
//      0 - DIFFUSE and SPECULAR
//      1 - DIFFUSE_SPECULAR
#define NRD_COMBINED                                1

// IMPORTANT: adjust same macro in "Shared.hlsli"
//      NORMAL                - common mode
//      OCCLUSION             - REBLUR OCCLUSION-only denoisers
//      SH                    - REBLUR SH (spherical harmonics) denoisers
//      DIRECTIONAL_OCCLUSION - REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION denoiser
#define NRD_MODE                                    NORMAL

constexpr uint32_t MAX_ANIMATED_INSTANCE_NUM        = 512;
constexpr auto BLAS_BUILD_BITS                      = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;
constexpr auto TLAS_BUILD_BITS                      = nri::AccelerationStructureBuildBits::PREFER_FAST_TRACE;
constexpr float ACCUMULATION_TIME                   = 0.5f; // seconds
constexpr float NEAR_Z                              = 0.001f; // m
constexpr bool CAMERA_RELATIVE                      = true;
constexpr bool CAMERA_LEFT_HANDED                   = true;

//=================================================================================
// Important tests, sensitive to regressions or just testing base functionality
//=================================================================================

const std::vector<uint32_t> interior_checkMeTests =
{{
    1, 3, 6, 8, 9, 10, 12, 13, 14, 23, 27, 28, 29, 31, 32, 35, 43, 44, 47, 53,
    59, 60, 62, 67, 75, 76, 79, 81, 95, 96, 107, 109, 111, 110, 114, 120, 124,
    126, 127, 132, 133, 134, 139, 140, 142, 145, 148, 150, 155, 156, 157, 160,
    161, 162, 168, 169
}};

//=================================================================================
// Tests, where IQ improvement would be "nice to have"
//=================================================================================

const std::vector<uint32_t> REBLUR_interior_improveMeTests =
{{
    153, 158
}};

const std::vector<uint32_t> RELAX_interior_improveMeTests =
{{
    114, 144, 148, 156, 159
}};

//=================================================================================

constexpr int32_t MAX_HISTORY_FRAME_NUM             = (int32_t)std::min(60u, std::min(nrd::REBLUR_MAX_HISTORY_FRAME_NUM, nrd::RELAX_MAX_HISTORY_FRAME_NUM));
constexpr uint32_t TEXTURES_PER_MATERIAL            = 4;
constexpr uint32_t MAX_TEXTURE_TRANSITION_NUM       = 32;

// UI
#define UI_YELLOW                                   ImVec4(1.0f, 0.9f, 0.0f, 1.0f)
#define UI_GREEN                                    ImVec4(0.5f, 0.9f, 0.0f, 1.0f)
#define UI_RED                                      ImVec4(1.0f, 0.1f, 0.0f, 1.0f)
#define UI_HEADER                                   ImVec4(0.7f, 1.0f, 0.7f, 1.0f)
#define UI_HEADER_BACKGROUND                        ImVec4(0.7f * 0.3f, 1.0f * 0.3f, 0.7f * 0.3f, 1.0f)
#define UI_DEFAULT                                  ImGui::GetStyleColorVec4(ImGuiCol_Text)

// NRD variant
#define NORMAL                                      0
#define OCCLUSION                                   1
#define SH                                          2
#define DIRECTIONAL_OCCLUSION                       3

// See HLSL
#define FLAG_FIRST_BIT                              20
#define INSTANCE_ID_MASK                            ( ( 1 << FLAG_FIRST_BIT ) - 1 )
#define FLAG_OPAQUE_OR_ALPHA_OPAQUE                 0x01
#define FLAG_TRANSPARENT                            0x02
#define FLAG_EMISSION                               0x04
#define FLAG_FORCED_EMISSION                        0x08

enum Denoiser : int32_t
{
    REBLUR,
    RELAX,
    REFERENCE,

    DENOISER_MAX_NUM
};

enum Resolution : int32_t
{
    RESOLUTION_FULL,
    RESOLUTION_FULL_PROBABILISTIC,
    RESOLUTION_HALF
};

enum MvType : int32_t
{
    MV_2D,
    MV_25D,
    MV_3D,
};

enum class Buffer : uint32_t
{
    GlobalConstants,
    InstanceDataStaging,
    WorldTlasDataStaging,
    LightTlasDataStaging,

    PrimitiveData,
    InstanceData,
    WorldScratch,
    LightScratch,

    UploadHeapBufferNum = 4
};

enum class Texture : uint32_t
{
    Ambient,
    ViewZ,
    Mv,
    Normal_Roughness,
    PrimaryMipAndCurvature,
    BaseColor_Metalness,
    DirectLighting,
    DirectEmission,
    TransparentLighting,
    Shadow,
    Diff,
    Spec,
    Unfiltered_ShadowData,
    Unfiltered_Diff,
    Unfiltered_Spec,
    Unfiltered_Shadow_Translucency,
    Validation,
    Composed_ViewZ,
    DlssOutput,
    Final,

    // History
    ComposedDiff_ViewZ,
    ComposedSpec_ViewZ,
    TaaHistory,
    TaaHistoryPrev,

    // SH
#if( NRD_MODE == SH )
    Unfiltered_DiffSh,
    Unfiltered_SpecSh,
    DiffSh,
    SpecSh,
#endif

    // Read-only
    NisData1,
    NisData2,
    MaterialTextures,

    MAX_NUM,

    // Aliases
    DlssInput = Unfiltered_Diff
};

enum class Pipeline : uint32_t
{
    AmbientRays,
    PrimaryRays,
    DirectLighting,
    IndirectRays,
    Composition,
    Temporal,
    Upsample,
    UpsampleNis,
    PreDlss,
    AfterDlss,

    MAX_NUM,
};

enum class Descriptor : uint32_t
{
    World_AccelerationStructure,
    Light_AccelerationStructure,

    LinearMipmapLinear_Sampler,
    LinearMipmapNearest_Sampler,
    Linear_Sampler,
    Nearest_Sampler,

    PrimitiveData_Buffer,
    InstanceData_Buffer,

    Ambient_Texture,
    Ambient_StorageTexture,
    ViewZ_Texture,
    ViewZ_StorageTexture,
    Mv_Texture,
    Mv_StorageTexture,
    Normal_Roughness_Texture,
    Normal_Roughness_StorageTexture,
    PrimaryMipAndCurvature_Texture,
    PrimaryMipAndCurvature_StorageTexture,
    BaseColor_Metalness_Texture,
    BaseColor_Metalness_StorageTexture,
    DirectLighting_Texture,
    DirectLighting_StorageTexture,
    DirectEmission_Texture,
    DirectEmission_StorageTexture,
    TransparentLighting_Texture,
    TransparentLighting_StorageTexture,
    Shadow_Texture,
    Shadow_StorageTexture,
    Diff_Texture,
    Diff_StorageTexture,
    Spec_Texture,
    Spec_StorageTexture,
    Unfiltered_ShadowData_Texture,
    Unfiltered_ShadowData_StorageTexture,
    Unfiltered_Diff_Texture,
    Unfiltered_Diff_StorageTexture,
    Unfiltered_Spec_Texture,
    Unfiltered_Spec_StorageTexture,
    Unfiltered_Shadow_Translucency_Texture,
    Unfiltered_Shadow_Translucency_StorageTexture,
    Validation_Texture,
    Validation_StorageTexture,
    Composed_ViewZ_Texture,
    Composed_ViewZ_StorageTexture,
    DlssOutput_Texture,
    DlssOutput_StorageTexture,
    Final_Texture,
    Final_StorageTexture,

    // History
    ComposedDiff_ViewZ_Texture,
    ComposedDiff_ViewZ_StorageTexture,
    ComposedSpec_ViewZ_Texture,
    ComposedSpec_ViewZ_StorageTexture,
    TaaHistory_Texture,
    TaaHistory_StorageTexture,
    TaaHistoryPrev_Texture,
    TaaHistoryPrev_StorageTexture,

    // SH
#if( NRD_MODE == SH )
    Unfiltered_DiffSh_Texture,
    Unfiltered_DiffSh_StorageTexture,
    Unfiltered_SpecSh_Texture,
    Unfiltered_SpecSh_StorageTexture,
    DiffSh_Texture,
    DiffSh_StorageTexture,
    SpecSh_Texture,
    SpecSh_StorageTexture,
#endif

    // Read-only
    NisData1,
    NisData2,
    MaterialTextures,

    MAX_NUM,

    // Aliases
    DlssInput_Texture = Unfiltered_Diff_Texture,
    DlssInput_StorageTexture = Unfiltered_Diff_StorageTexture
};

enum class DescriptorSet : uint32_t
{
    AmbientRays1,
    PrimaryRays1,
    DirectLighting1,
    IndirectRays1,
    Composition1,
    Temporal1a,
    Temporal1b,
    Upsample1a,
    Upsample1b,
    UpsampleNis1a,
    UpsampleNis1b,
    PreDlss1,
    AfterDlss1,
    RayTracing2,

    MAX_NUM
};

struct NRIInterface
    : public nri::CoreInterface
    , public nri::SwapChainInterface
    , public nri::RayTracingInterface
    , public nri::HelperInterface
{};

struct Frame
{
    nri::DeviceSemaphore* deviceSemaphore;
    nri::CommandAllocator* commandAllocator;
    nri::CommandBuffer* commandBuffer;
    nri::Descriptor* globalConstantBufferDescriptor;
    nri::DescriptorSet* globalConstantBufferDescriptorSet;
    uint64_t globalConstantBufferOffset;
};

struct GlobalConstantBufferData
{
    float4x4 gViewToWorld;
    float4x4 gViewToClip;
    float4x4 gWorldToView;
    float4x4 gWorldToViewPrev;
    float4x4 gWorldToClip;
    float4x4 gWorldToClipPrev;
    float4 gHitDistParams;
    float4 gCameraFrustum;
    float4 gSunDirection_gExposure;
    float4 gCameraOrigin_gMipBias;
    float4 gTrimmingParams_gEmissionIntensity;
    float4 gViewDirection_gIsOrtho;
    float2 gWindowSize;
    float2 gInvWindowSize;
    float2 gOutputSize;
    float2 gInvOutputSize;
    float2 gRenderSize;
    float2 gInvRenderSize;
    float2 gRectSize;
    float2 gInvRectSize;
    float2 gRectSizePrev;
    float2 gJitter;
    float gNearZ;
    float gAmbientAccumSpeed;
    float gAmbient;
    float gSeparator;
    float gRoughnessOverride;
    float gMetalnessOverride;
    float gUnitToMetersMultiplier;
    float gIndirectDiffuse;
    float gIndirectSpecular;
    float gTanSunAngularRadius;
    float gTanPixelAngularRadius;
    float gDebug;
    float gTransparent;
    float gReference;
    float gUsePrevFrame;
    float gMinProbability;
    uint32_t gDenoiserType;
    uint32_t gDisableShadowsAndEnableImportanceSampling;
    uint32_t gOnScreen;
    uint32_t gFrameIndex;
    uint32_t gForcedMaterial;
    uint32_t gUseNormalMap;
    uint32_t gIsWorldSpaceMotionEnabled;
    uint32_t gTracingMode;
    uint32_t gSampleNum;
    uint32_t gBounceNum;
    uint32_t gTAA;
    uint32_t gSH;
    uint32_t gPSR;
    uint32_t gValidation;
    uint32_t gHighlightAhs;
    uint32_t gAhsDynamicMipSelection;


    // NIS
    float gNisDetectRatio;
    float gNisDetectThres;
    float gNisMinContrastRatio;
    float gNisRatioNorm;
    float gNisContrastBoost;
    float gNisEps;
    float gNisSharpStartY;
    float gNisSharpScaleY;
    float gNisSharpStrengthMin;
    float gNisSharpStrengthScale;
    float gNisSharpLimitMin;
    float gNisSharpLimitScale;
    float gNisScaleX;
    float gNisScaleY;
    float gNisDstNormX;
    float gNisDstNormY;
    float gNisSrcNormX;
    float gNisSrcNormY;
    uint32_t gNisInputViewportOriginX;
    uint32_t gNisInputViewportOriginY;
    uint32_t gNisInputViewportWidth;
    uint32_t gNisInputViewportHeight;
    uint32_t gNisOutputViewportOriginX;
    uint32_t gNisOutputViewportOriginY;
    uint32_t gNisOutputViewportWidth;
    uint32_t gNisOutputViewportHeight;
};

struct Settings
{
    double      motionStartTime                    = 0.0;

    float       maxFps                             = 60.0f;
    float       camFov                             = 90.0f;
    float       sunAzimuth                         = -147.0f;
    float       sunElevation                       = 45.0f;
    float       sunAngularDiameter                 = 0.533f;
    float       exposure                           = 80.0f;
    float       roughnessOverride                  = 0.0f;
    float       metalnessOverride                  = 0.0f;
    float       emissionIntensity                  = 1.0f;
    float       debug                              = 0.0f;
    float       meterToUnitsMultiplier             = 1.0f;
    float       emulateMotionSpeed                 = 1.0f;
    float       animatedObjectScale                = 1.0f;
    float       separator                          = 0.0f;
    float       animationProgress                  = 0.0f;
    float       animationSpeed                     = 0.0f;
    float       hitDistScale                       = 3.0f;
    float       disocclusionThreshold              = 1.0f;
    float       resolutionScale                    = 1.0f;
    float       sharpness                          = 0.15f;

    int32_t     maxAccumulatedFrameNum             = 31;
    int32_t     maxFastAccumulatedFrameNum         = 7;
    int32_t     onScreen                           = 0;
    int32_t     forcedMaterial                     = 0;
    int32_t     animatedObjectNum                  = 5;
    int32_t     activeAnimation                    = 0;
    int32_t     motionMode                         = 0;
    int32_t     denoiser                           = REBLUR;
    int32_t     rpp                                = 1;
    int32_t     bounceNum                          = 1;
    int32_t     tracingMode                        = RESOLUTION_HALF;
    int32_t     mvType                             = MV_25D;

    bool        cameraJitter                       = true;
    bool        limitFps                           = false;
    bool        ambient                            = true;
    bool        PSR                                = false;
    bool        indirectDiffuse                    = true;
    bool        indirectSpecular                   = true;
    bool        normalMap                          = true;
    bool        TAA                                = true;
    bool        animatedObjects                    = false;
    bool        animateCamera                      = false;
    bool        animateSun                         = false;
    bool        nineBrothers                       = false;
    bool        blink                              = false;
    bool        pauseAnimation                     = true;
    bool        emission                           = false;
    bool        linearMotion                       = true;
    bool        emissiveObjects                    = false;
    bool        importanceSampling                 = true;
    bool        specularLobeTrimming               = true;
    bool        ortho                              = false;
    bool        adaptiveAccumulation               = true;
    bool        usePrevFrame                       = true;
    bool        DLSS                               = false;
    bool        NIS                                = true;
    bool        adaptRadiusToResolution            = true;
    bool        windowAlignment                    = true;
    bool        highLightAhs                       = true;
    bool        ahsDynamicMipSelection             = true;
};

struct DescriptorDesc
{
    const char* debugName;
    void* resource;
    nri::Format format;
    nri::TextureUsageBits textureUsage;
    nri::BufferUsageBits bufferUsage;
    bool isArray;
};

struct TextureState
{
    Texture texture;
    nri::AccessBits nextAccess;
    nri::TextureLayout nextLayout;
};

struct AnimatedInstance
{
    float3 basePosition;
    float3 rotationAxis;
    float3 elipseAxis;
    float durationSec = 5.0f;
    float progressedSec = 0.0f;
    float inverseRotation = 1.0f;
    float inverseDirection = 1.0f;
    uint32_t instanceID = 0;

    float4x4 Animate(float elapsedSeconds, float scale, float3& position)
    {
        float angle = progressedSec / durationSec;
        angle = Pi(angle * 2.0f - 1.0f);

        float3 localPosition;
        localPosition.x = Cos(angle * inverseDirection);
        localPosition.y = Sin(angle * inverseDirection);
        localPosition.z = localPosition.y;

        position = basePosition + localPosition * elipseAxis * scale;

        float4x4 transform;
        transform.SetupByRotation(angle * inverseRotation, rotationAxis);
        transform.AddScale(scale);

        progressedSec += elapsedSeconds;
        progressedSec = (progressedSec >= durationSec) ? 0.0f : progressedSec;

        return transform;
    }
};

struct PrimitiveData
{
    uint32_t uv0;
    uint32_t uv1;
    uint32_t uv2;
    uint32_t n0oct;

    uint32_t n1oct;
    uint32_t n2oct;
    uint32_t t0oct;
    uint32_t t1oct;

    uint32_t t2oct;
    uint32_t b0s_b1s;
    uint32_t b2s_worldToUvUnits;
    float curvature;
};

struct InstanceData
{
    uint32_t basePrimitiveIndex;
    uint32_t baseTextureIndex;
    uint32_t averageBaseColor;
    uint32_t unused;

    float4 mWorldToWorldPrev0;
    float4 mWorldToWorldPrev1;
    float4 mWorldToWorldPrev2;
};

struct AlphaTestedGeometry
{
    nri::Buffer* positions;
    nri::Buffer* uvs;
    nri::Buffer* indices;

    nri::Texture* alphaTexture; //on gpu
    utils::Texture* utilsTexture; //on cpu

    std::vector<uint8_t> indexData;
    std::vector<uint8_t> uvData;

    uint64_t positionBufferSize;
    uint64_t positionOffset;
    uint64_t uvBufferSize;
    uint64_t uvOffset;
    uint64_t indexBufferSize;
    uint64_t indexOffset;

    uint32_t meshIndex;
    uint32_t materialIndex;

    const nri::Format vertexFormat = nri::Format::RGB32_SFLOAT;
    const nri::Format uvFormat = nri::Format::RG32_SFLOAT;
    const nri::Format indexFormat = nri::Format::R16_UINT;
};

struct OmmGpuBakerPrebuildMemoryStats
{
    size_t maximum;
    size_t total;

    size_t outputMaxSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum];
    size_t maxTransientBufferSizes[omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS];
};

struct OmmBatch
{
    size_t offset;
    size_t count;
};

class Sample : public SampleBase
{
public:
    Sample() :
        m_Reblur(BUFFERED_FRAME_MAX_NUM, "REBLUR"),
        m_Relax(BUFFERED_FRAME_MAX_NUM, "RELAX"),
        m_Sigma(BUFFERED_FRAME_MAX_NUM, "SIGMA"),
        m_Reference(BUFFERED_FRAME_MAX_NUM, "REFERENCE")
    {
        m_SceneFile = "Bistro/BistroExterior.fbx";
        m_OutputResolution = { 1920, 1080 };
    }

    ~Sample();

    void InitCmdLine(cmdline::parser& cmdLine) override
    {
        cmdLine.add<int32_t>("dlssQuality", 'd', "DLSS quality: [-1: 3]", false, -1, cmdline::range(-1, 3));
        cmdLine.add("ommDebugMode", 0, "enable omm-bake Nsight debug mode");
        cmdLine.add("disableOmmBlasBuild", 0, "disable masked geometry building");
        cmdLine.add("enableOmmCache", 0, "enable omm init from cache");
        cmdLine.add<uint32_t>("ommBuildPostponeFrameId", 0, "build OMM on desired frameId", false, 0);
    }

    void ReadCmdLine(cmdline::parser& cmdLine) override
    {
        m_DlssQuality = cmdLine.get<int32_t>("dlssQuality");
        m_OmmBakeDesc.enableDebugMode = cmdLine.exist("ommDebugMode");
        m_OmmBakeDesc.buildFrameId = cmdLine.get<uint32_t>("ommBuildPostponeFrameId");
        m_OmmBakeDesc.disableBlasBuild = cmdLine.exist("disableOmmBlasBuild");
        m_OmmBakeDesc.enableCache = cmdLine.exist("enableOmmCache");
    }

    bool Initialize(nri::GraphicsAPI graphicsAPI) override;
    void PrepareFrame(uint32_t frameIndex) override;
    void RenderFrame(uint32_t frameIndex) override;

    inline nri::Texture*& Get(Texture index)
    { return m_Textures[(uint32_t)index]; }

    inline nri::TextureTransitionBarrierDesc& GetState(Texture index)
    { return m_TextureStates[(uint32_t)index]; }

    inline nri::Format GetFormat(Texture index)
    { return m_TextureFormats[(uint32_t)index]; }

    inline nri::Buffer*& Get(Buffer index)
    { return m_Buffers[(uint32_t)index]; }

    inline nri::Pipeline*& Get(Pipeline index)
    { return m_Pipelines[(uint32_t)index]; }

    inline nri::Descriptor*& Get(Descriptor index)
    { return m_Descriptors[(uint32_t)index]; }

    inline nri::DescriptorSet*& Get(DescriptorSet index)
    { return m_DescriptorSets[(uint32_t)index]; }

    void LoadScene();
    void SetupAnimatedObjects();

    nri::Format CreateSwapChain();
    void CreateCommandBuffers();
    void CreatePipelineLayoutAndDescriptorPool();
    void CreatePipelines();
    void CreateBottomLevelAccelerationStructures();
    void CreateTopLevelAccelerationStructures();
    void CreateSamplers();
    void CreateResources(nri::Format swapChainFormat);
    void CreateDescriptorSets();

    void CreateUploadBuffer(uint64_t size, nri::Buffer*& buffer, nri::Memory*& memory);
    void CreateScratchBuffer(nri::AccelerationStructure& accelerationStructure, nri::Buffer*& buffer, nri::Memory*& memory);
    void CreateTexture(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, nri::Format format, uint16_t width, uint16_t height, uint16_t mipNum, uint16_t arraySize, nri::TextureUsageBits usage, nri::AccessBits state);
    void CreateBuffer(std::vector<DescriptorDesc>& descriptorDescs, const char* debugName, uint64_t elements, uint32_t stride, nri::BufferUsageBits usage, nri::Format format = nri::Format::UNKNOWN);

    void UploadStaticData();
    void UpdateConstantBuffer(uint32_t frameIndex, float globalResetFactor);
    void RestoreBindings(nri::CommandBuffer& commandBuffer, const Frame& frame);
    void BuildBottomLevelAccelerationStructure(nri::AccelerationStructure& accelerationStructure, const nri::GeometryObject* objects, const uint32_t objectNum);
    void BuildTopLevelAccelerationStructure(nri::CommandBuffer& commandBuffer, uint32_t bufferedFrameIndex);
    uint32_t BuildOptimizedTransitions(const TextureState* states, uint32_t stateNum, std::array<nri::TextureTransitionBarrierDesc, MAX_TEXTURE_TRANSITION_NUM>& transitions);

    void GenerateGeometry(utils::Scene& scene);
    void GeneratePlane(utils::Scene& scene, float3 origin, float3 axisX, float3 axisY, float2 size, uint32_t subdivision, uint32_t vertexOffset, float uvScaling);
    void PushVertex(utils::Scene& scene, float positionX, float positionY, float positionZ, float texCoordU, float texCoordV);
    void ComputePrimitiveNormal(utils::Scene& scene, uint32_t vertexOffset, uint32_t indexOffset);

    inline float3 GetSunDirection() const
    {
        float3 sunDirection;
        sunDirection.x = Cos( DegToRad(m_Settings.sunAzimuth) ) * Cos( DegToRad(m_Settings.sunElevation) );
        sunDirection.y = Sin( DegToRad(m_Settings.sunAzimuth) ) * Cos( DegToRad(m_Settings.sunElevation) );
        sunDirection.z = Sin( DegToRad(m_Settings.sunElevation) );

        return sunDirection;
    }

    inline float3 GetSpecularLobeTrimming() const
    { return m_Settings.specularLobeTrimming ? float3(0.95f, 0.04f, 0.11f) : float3(1.0f, 0.0f, 0.0001f); }

    inline float GetDenoisingRange() const
    { return 4.0f * m_Scene.aabb.GetRadius(); }

private:
    NrdIntegration m_Reblur;
    NrdIntegration m_Relax;
    NrdIntegration m_Sigma;
    NrdIntegration m_Reference;
    DlssIntegration m_DLSS;
    NRIInterface NRI = {};
    Timer m_Timer;
    utils::Scene m_Scene;
    nri::Device* m_Device = nullptr;
    nri::SwapChain* m_SwapChain = nullptr;
    nri::CommandQueue* m_CommandQueue = nullptr;
    nri::QueueSemaphore* m_BackBufferAcquireSemaphore = nullptr;
    nri::QueueSemaphore* m_BackBufferReleaseSemaphore = nullptr;
    nri::AccelerationStructure* m_WorldTlas = nullptr;
    nri::AccelerationStructure* m_LightTlas = nullptr;
    nri::DescriptorPool* m_DescriptorPool = nullptr;
    nri::PipelineLayout* m_PipelineLayout = nullptr;
    std::array<Frame, BUFFERED_FRAME_MAX_NUM> m_Frames = {};
    std::vector<nri::Texture*> m_Textures;
    std::vector<nri::TextureTransitionBarrierDesc> m_TextureStates;
    std::vector<nri::Format> m_TextureFormats;
    std::vector<nri::Buffer*> m_Buffers;
    std::vector<nri::Memory*> m_MemoryAllocations;
    std::vector<nri::Descriptor*> m_Descriptors;
    std::vector<nri::DescriptorSet*> m_DescriptorSets;
    std::vector<nri::Pipeline*> m_Pipelines;
    std::vector<nri::AccelerationStructure*> m_BLASs;
    std::vector<BackBuffer> m_SwapChainBuffers;
    std::vector<AnimatedInstance> m_AnimatedInstances;
    std::array<float, 256> m_FrameTimes = {};
    nrd::RelaxDiffuseSpecularSettings m_RelaxSettings = {};
    nrd::ReblurSettings m_ReblurSettings = {};
    nrd::ReferenceSettings m_ReferenceSettings = {};
    Settings m_Settings = {};
    Settings m_PrevSettings = {};
    Settings m_DefaultSettings = {};
    const std::vector<uint32_t>* m_checkMeTests = nullptr;
    const std::vector<uint32_t>* m_improveMeTests = nullptr;
    float3 m_PrevLocalPos = {};
    uint2 m_RenderResolution = {};
    uint64_t m_ConstantBufferSize = 0;
    uint32_t m_DefaultInstancesOffset = 0;
    uint32_t m_LastSelectedTest = uint32_t(-1);
    uint32_t m_TestNum = uint32_t(-1);
    int32_t m_DlssQuality = int32_t(-1);
    float m_UiWidth = 0.0f;
    float m_AmbientAccumFrameNum = 0.0f;
    float m_MinResolutionScale = 0.5f;
    bool m_HasTransparentObjects = false;
    bool m_ShowUi = true;
    bool m_ForceHistoryReset = false;
    bool m_SH = true;
    bool m_DebugNRD = false;
    bool m_ShowValidationOverlay = true;

private: //OMM:
    void InitAlphaTestedGeometry();
    void RebuildOmmGeometry();

    void FillOmmBakerInputs(std::vector<ommhelper::OmmBakeGeometryDesc>& ommBakeQueue);
    void FillOmmBlasBuildInputs(size_t start, size_t count);

    void BakeOmmCpu(size_t* queue, size_t count);
    void BakeOmmGpu(size_t* queue, size_t count, const OmmGpuBakerPrebuildMemoryStats& memoryStats);
    OmmGpuBakerPrebuildMemoryStats GetGpuBakerPrebuildMemoryStats();
    void CreateGpuBakerBuffers(const OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void BindGpuBakerBuffers(const OmmGpuBakerPrebuildMemoryStats& memoryStats, const size_t* ids, const size_t count);

    inline uint64_t GetInstanceHash(uint32_t meshId, uint32_t materialId) { return uint64_t(meshId) << 32 | uint64_t(materialId); };
    inline std::string GetOmmCacheFilename() {return m_OmmCacheFolderName + std::string("/") + m_SceneName; };
    void InitializeOmmGeometryFromCache(std::vector<size_t>& bakeQueue, size_t start, size_t count);
    void SaveMaskCache(uint32_t id);

    inline uint64_t GetInstanceMask(uint32_t meshIndex, uint32_t materialIndex) { return uint64_t(meshIndex) << 32 | uint64_t(materialIndex); }
    nri::AccelerationStructure* GetMaskedBlas(uint64_t insatanceMask);

    void ReleaseMaskedGeometry();
    void ReleaseBakingResources();

    void AppendOmmImguiSettings();

private:
    ommhelper::OpacityMicroMapsHelper m_OmmHelper = {};

    //preprocessed alpha geometry from the scene:
    std::vector<AlphaTestedGeometry> m_OmmAlphaGeometry;
    std::vector<nri::Memory*> m_OmmAlphaGeometryMemories;
    std::vector<nri::Buffer*> m_OmmAlphaGeometryBuffers;

    //baker and builder queues
    std::vector<ommhelper::OmmBakeGeometryDesc> m_OmmBakeInstances;
    std::vector<ommhelper::MaskedGeometryBuildDesc> m_OmmGeometryBuildQueue;

    //temporal resources for baking
    std::vector<float> m_OmmRawAlphaChannelForCpuBaker;

    std::vector<nri::Buffer*> m_OmmGpuOutputBuffers;
    std::vector<nri::Buffer*> m_OmmGpuTransientBuffers;
    std::vector<nri::Buffer*> m_OmmGpuReadbackBuffers;
    std::vector<nri::Buffer*> m_OmmCpuUploadBuffers;

    std::vector<nri::Memory*> m_OmmBakerAllocations;
    std::vector<nri::Memory*> m_OmmTmpAllocations;

    //misc
    struct OmmNriContext
    {
        nri::CommandAllocator* commandAllocator;
        nri::CommandBuffer* commandBuffer;
        nri::DeviceSemaphore* deviceSemaphore;
    } m_OmmContext;

    struct OmmBlas
    {
        nri::AccelerationStructure* blas;
        //[!] VK Warning! VkMicromapExt wrapping is not supported yet. Use OmmHelper::DestroyMaskedGeometry instead of nri on release.
        nri::Buffer* ommArray; 
    };
    std::map<uint64_t, OmmBlas> m_InstanceMaskToMaskedBlasData;
    std::vector<OmmBlas> m_MaskedBlasses;
    ommhelper::OmmBakeDesc m_OmmBakeDesc = {};
    std::string m_SceneName = "Scene";
    std::string m_OmmCacheFolderName = "_OmmCache";
    size_t m_OmmWorkloadBatchSize = 32;
    bool m_EnableOmm = true;
    bool m_ShowFullSettings = false;
    bool m_IsOmmBakingActive = false;
    bool m_ShowOnlyAlphaTestedGeometry = false;

private:
    Profiler m_Profiler;
};

Sample::~Sample()
{
    if (!m_Device)
        return;

    NRI.WaitForIdle(*m_CommandQueue);

    m_DLSS.Shutdown();

    m_Reblur.Destroy();
    m_Relax.Destroy();
    m_Sigma.Destroy();
    m_Reference.Destroy();

    m_Profiler.Destroy();
    ReleaseMaskedGeometry();
    ReleaseBakingResources();
    m_OmmHelper.Destroy();

    for (auto& buffer : m_OmmAlphaGeometryBuffers)
        NRI.DestroyBuffer(*buffer);
    for (auto& memory : m_OmmAlphaGeometryMemories)
        NRI.FreeMemory(*memory);

    for (Frame& frame : m_Frames)
    {
        NRI.DestroyCommandBuffer(*frame.commandBuffer);
        NRI.DestroyDeviceSemaphore(*frame.deviceSemaphore);
        NRI.DestroyCommandAllocator(*frame.commandAllocator);
        NRI.DestroyDescriptor(*frame.globalConstantBufferDescriptor);
    }

    for (BackBuffer& backBuffer : m_SwapChainBuffers)
    {
        NRI.DestroyDescriptor(*backBuffer.colorAttachment);
        NRI.DestroyFrameBuffer(*backBuffer.frameBufferUI);
    }

    for (uint32_t i = 0; i < m_Textures.size(); i++)
        NRI.DestroyTexture(*m_Textures[i]);

    for (uint32_t i = 0; i < m_Buffers.size(); i++)
        NRI.DestroyBuffer(*m_Buffers[i]);

    for (uint32_t i = 0; i < m_Descriptors.size(); i++)
        NRI.DestroyDescriptor(*m_Descriptors[i]);

    for (uint32_t i = 0; i < m_Pipelines.size(); i++)
        NRI.DestroyPipeline(*m_Pipelines[i]);

    for (uint32_t i = 0; i < m_BLASs.size(); i++)
        NRI.DestroyAccelerationStructure(*m_BLASs[i]);

    NRI.DestroyPipelineLayout(*m_PipelineLayout);
    NRI.DestroyDescriptorPool(*m_DescriptorPool);
    NRI.DestroyAccelerationStructure(*m_WorldTlas);
    NRI.DestroyAccelerationStructure(*m_LightTlas);
    NRI.DestroyQueueSemaphore(*m_BackBufferAcquireSemaphore);
    NRI.DestroyQueueSemaphore(*m_BackBufferReleaseSemaphore);
    NRI.DestroySwapChain(*m_SwapChain);

    for (size_t i = 0; i < m_MemoryAllocations.size(); i++)
        NRI.FreeMemory(*m_MemoryAllocations[i]);

    DestroyUserInterface();

    nri::DestroyDevice(*m_Device);
}

bool Sample::Initialize(nri::GraphicsAPI graphicsAPI)
{
    Rand::Seed(106937, &m_FastRandState);

    nri::PhysicalDeviceGroup physicalDeviceGroup = {};
    if (!helper::FindPhysicalDeviceGroup(physicalDeviceGroup))
        return false;

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.spirvBindingOffsets = SPIRV_BINDING_OFFSETS;
    deviceCreationDesc.physicalDeviceGroup = &physicalDeviceGroup;
    DlssIntegration::SetupDeviceExtensions(deviceCreationDesc);
    NRI_ABORT_ON_FAILURE( nri::CreateDevice(deviceCreationDesc, m_Device) );

    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::CoreInterface), (nri::CoreInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::SwapChainInterface), (nri::SwapChainInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::RayTracingInterface), (nri::RayTracingInterface*)&NRI) );
    NRI_ABORT_ON_FAILURE( nri::GetInterface(*m_Device, NRI_INTERFACE(nri::HelperInterface), (nri::HelperInterface*)&NRI) );

    NRI_ABORT_ON_FAILURE( NRI.GetCommandQueue(*m_Device, nri::CommandQueueType::GRAPHICS, m_CommandQueue));
    NRI_ABORT_ON_FAILURE( NRI.CreateQueueSemaphore(*m_Device, m_BackBufferAcquireSemaphore));
    NRI_ABORT_ON_FAILURE( NRI.CreateQueueSemaphore(*m_Device, m_BackBufferReleaseSemaphore));

    const nri::DeviceDesc& deviceDesc = NRI.GetDeviceDesc(*m_Device);
    m_ConstantBufferSize = helper::Align(sizeof(GlobalConstantBufferData), deviceDesc.constantBufferOffsetAlignment);
    m_RenderResolution = GetOutputResolution();

    if (m_DlssQuality != -1 && m_DLSS.InitializeLibrary(*m_Device, ""))
    {
        DlssSettings dlssSettings = {};
        DlssInitDesc dlssInitDesc = {};
        dlssInitDesc.outputResolution = { GetOutputResolution().x, GetOutputResolution().y };

        if (m_DLSS.GetOptimalSettings(dlssInitDesc.outputResolution, (DlssQuality)m_DlssQuality, dlssSettings))
        {
            dlssInitDesc.quality = (DlssQuality)m_DlssQuality;
            dlssInitDesc.isContentHDR = true;

            m_DLSS.Initialize(m_CommandQueue, dlssInitDesc);

            float sx = float(dlssSettings.minRenderResolution.Width) / float(dlssSettings.renderResolution.Width);
            float sy = float(dlssSettings.minRenderResolution.Height) / float(dlssSettings.renderResolution.Height);
            float minResolutionScale = sy > sx ? sy : sx;

            m_RenderResolution = {dlssSettings.renderResolution.Width, dlssSettings.renderResolution.Height};
            m_MinResolutionScale = minResolutionScale;

            printf("Render resolution (%u, %u)\n", m_RenderResolution.x, m_RenderResolution.y);

            m_Settings.sharpness = dlssSettings.sharpness;
            m_Settings.DLSS = true;
        }
        else
        {
            m_DLSS.Shutdown();

            printf("Unsupported DLSS mode!\n");
        }
    }

    #if 0
        // README "Memory requirements" table generator
        printf("| %10s | %36s | %16s | %16s | %16s |\n", "Resolution", "Denoiser", "Working set (Mb)", "Persistent (Mb)", "Aliasable (Mb)");
        printf("|------------|--------------------------------------|------------------|------------------|------------------|\n");

        for (uint32_t j = 0; j < 3; j++)
        {
            const char* resolution = "1080p";
            uint16_t w = 1920;
            uint16_t h = 1080;

            if (j == 1)
            {
                resolution = "1440p";
                w = 2560;
                h = 1440;
            }
            else if (j == 2)
            {
                resolution = "2160p";
                w = 3840;
                h = 2160;
            }

            for (uint32_t i = 0; i <= (uint32_t)nrd::Method::REFERENCE; i++)
            {
                nrd::Method method = (nrd::Method)i;
                const char* methodName = nrd::GetMethodString(method);

                const nrd::MethodDesc methodDesc = {method, w, h};

                nrd::DenoiserCreationDesc denoiserCreationDesc = {};
                denoiserCreationDesc.requestedMethods = &methodDesc;
                denoiserCreationDesc.requestedMethodsNum = 1;

                NrdIntegration denoiser(2);
                NRI_ABORT_ON_FALSE( denoiser.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
                printf("| %10s | %36s | %16.2f | %16.2f | %16.2f |\n", i == 0 ? resolution : "", methodName, denoiser.GetTotalMemoryUsageInMb(), denoiser.GetPersistentMemoryUsageInMb(), denoiser.GetAliasableMemoryUsageInMb());
                denoiser.Destroy();
            }

            if (j != 2)
                printf("| %10s | %36s | %16s | %16s | %16s |\n", "", "", "", "", "");
        }

        __debugbreak();
    #endif

    LoadScene();
    SetupAnimatedObjects();

    nri::Format swapChainFormat = CreateSwapChain();
    CreateCommandBuffers();
    CreatePipelineLayoutAndDescriptorPool();
    CreatePipelines();
    CreateBottomLevelAccelerationStructures();
    CreateTopLevelAccelerationStructures();
    CreateSamplers();
    CreateResources(swapChainFormat);
    CreateDescriptorSets();

    UploadStaticData();

    InitAlphaTestedGeometry();
    m_OmmHelper.Initialize(m_Device);
    m_Profiler.Init(m_Device);

    m_Camera.Initialize(m_Scene.aabb.GetCenter(), m_Scene.aabb.vMin, CAMERA_RELATIVE);

    m_Scene.UnloadGeometryData();

    { // REBLUR
        const nrd::MethodDesc methodDescs[] =
        {
    #if( NRD_MODE == OCCLUSION )
        #if( NRD_COMBINED == 1 )
            { nrd::Method::REBLUR_DIFFUSE_SPECULAR_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #else
            { nrd::Method::REBLUR_DIFFUSE_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            { nrd::Method::REBLUR_SPECULAR_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #endif
    #elif( NRD_MODE == SH )
        #if( NRD_COMBINED == 1 )
            { nrd::Method::REBLUR_DIFFUSE_SPECULAR_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #else
            { nrd::Method::REBLUR_DIFFUSE_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            { nrd::Method::REBLUR_SPECULAR_SH, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #endif
    #elif( NRD_MODE == DIRECTIONAL_OCCLUSION )
            { nrd::Method::REBLUR_DIFFUSE_DIRECTIONAL_OCCLUSION, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
    #else
        #if( NRD_COMBINED == 1 )
            { nrd::Method::REBLUR_DIFFUSE_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #else
            { nrd::Method::REBLUR_DIFFUSE, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            { nrd::Method::REBLUR_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        #endif
    #endif
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Reblur.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    { // RELAX
        const nrd::MethodDesc methodDescs[] =
        {
            #if( NRD_COMBINED == 1 )
                { nrd::Method::RELAX_DIFFUSE_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            #else
                { nrd::Method::RELAX_DIFFUSE, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
                { nrd::Method::RELAX_SPECULAR, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
            #endif
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Relax.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    { // SIGMA
        const nrd::MethodDesc methodDescs[] =
        {
            { nrd::Method::SIGMA_SHADOW_TRANSLUCENCY, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Sigma.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    { // REFERENCE
        const nrd::MethodDesc methodDescs[] =
        {
            { nrd::Method::REFERENCE, (uint16_t)m_RenderResolution.x, (uint16_t)m_RenderResolution.y },
        };

        nrd::DenoiserCreationDesc denoiserCreationDesc = {};
        denoiserCreationDesc.requestedMethods = methodDescs;
        denoiserCreationDesc.requestedMethodsNum = helper::GetCountOf(methodDescs);

        NRI_ABORT_ON_FALSE( m_Reference.Initialize(denoiserCreationDesc, *m_Device, NRI, NRI) );
    }

    size_t sceneBeginNameOffset = m_SceneFile.find_last_of("/");
    sceneBeginNameOffset = sceneBeginNameOffset == std::string::npos ? 0 : ++sceneBeginNameOffset;
    size_t sceneEndNameOffset = m_SceneFile.find_last_of(".");
    sceneEndNameOffset = sceneEndNameOffset == std::string::npos ? m_SceneFile.length() : sceneEndNameOffset;
    m_SceneName = m_SceneFile.substr(sceneBeginNameOffset, sceneEndNameOffset - sceneBeginNameOffset);

    float3 cameraInitialPos = m_Scene.aabb.GetCenter();
    float3 lookAtPos = m_Scene.aabb.vMin;
    if (m_SceneFile.find("BistroExterior") != std::string::npos)
    {
        cameraInitialPos = float3(49.545f, -38.352f, 6.916f);
        float3 realLookAtPos = float3(41.304f, -26.487f, 4.805f);
        float3 hackedDir = realLookAtPos - cameraInitialPos;
        hackedDir = float3(hackedDir.y, -hackedDir.x, hackedDir.z);
        lookAtPos = cameraInitialPos + hackedDir;
    }
    m_Camera.Initialize(cameraInitialPos, lookAtPos, CAMERA_RELATIVE);
    m_Scene.UnloadGeometryData();
     
    m_DefaultSettings = m_Settings;

    return CreateUserInterface(*m_Device, NRI, NRI, swapChainFormat);
}

nri::AccelerationStructure* Sample::GetMaskedBlas(uint64_t insatanceMask)
{
    const auto& it = m_InstanceMaskToMaskedBlasData.find(insatanceMask);
    if (it != m_InstanceMaskToMaskedBlasData.end())
        return it->second.blas;
    return nullptr;
}

std::vector<uint32_t> FilterOutAlphaTestedGeometry(const utils::Scene& scene)
{ // Filter out alphaOpaque geometry by mesh and material IDs
    std::vector<uint32_t> result;
    std::set<uint64_t> processedCombinations;
    for (uint32_t instaceId = 0; instaceId < (uint32_t)scene.instances.size(); ++instaceId)
    {
        const utils::Instance& instance = scene.instances[instaceId];
        const utils::Material& material = scene.materials[instance.materialIndex];
        if (material.IsAlphaOpaque())
        {
            uint64_t mask = uint64_t(instance.meshIndex) << 32 | uint64_t(instance.materialIndex);
            size_t currentCount = processedCombinations.size();
            processedCombinations.insert(mask);
            bool isDuplicate = processedCombinations.size() == currentCount;
            if (isDuplicate == false)
                result.push_back(instaceId);
        }
    }
    return result;
}

void Sample::InitAlphaTestedGeometry()
{
    printf("[OMM] Initializing Alpha Tested Geometry\n");
    std::vector<uint32_t> alphaInstances = FilterOutAlphaTestedGeometry(m_Scene);
    m_OmmAlphaGeometry.resize(alphaInstances.size());

    size_t positionBufferSize = 0;
    size_t indexBufferSize = 0;
    size_t uvBufferSize = 0;

    for (size_t i = 0; i < alphaInstances.size(); ++i)
    { // Calculate buffer sizes
        const utils::Instance& instance = m_Scene.instances[alphaInstances[i]];
        const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];

        positionBufferSize += helper::Align(mesh.vertexNum * sizeof(float3), 256);
        indexBufferSize += helper::Align(mesh.indexNum * sizeof(utils::Index), 256);
        uvBufferSize += helper::Align(mesh.vertexNum * sizeof(float2), 256);
    }

    m_OmmAlphaGeometryBuffers.reserve(3);
    nri::Buffer*& positionBuffer = m_OmmAlphaGeometryBuffers.emplace_back();
    nri::Buffer*& indexBuffer = m_OmmAlphaGeometryBuffers.emplace_back();
    nri::Buffer*& uvBuffer = m_OmmAlphaGeometryBuffers.emplace_back();

    { // Cteate buffers
        nri::BufferDesc bufferDesc = {};
        bufferDesc.physicalDeviceMask = nri::WHOLE_DEVICE_GROUP;
        bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ACCELERATION_STRUCTURE_BUILD_READ;

        bufferDesc.size = positionBufferSize;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, positionBuffer));

        bufferDesc.size = indexBufferSize;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, indexBuffer));

        //uv buffer is used in OMM baking as a raw read buffer. For compatibility with Vulkan this buffer is required to be structured
        bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE;
        bufferDesc.size = uvBufferSize;
        bufferDesc.structureStride = sizeof(uint32_t);
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, uvBuffer));
    }

    //raw data for uploading to gpu
    std::vector<uint8_t> positions;
    std::vector<uint8_t> uvs;
    std::vector<uint8_t> indices;

    uint32_t storageAlignment = NRI.GetDeviceDesc(*m_Device).storageBufferOffsetAlignment;
    uint32_t bufferAlignment = NRI.GetDeviceDesc(*m_Device).typedBufferOffsetAlignment;

    nri::Texture** materialTextures = m_Textures.data() + (size_t)Texture::MaterialTextures;
    for (size_t i = 0; i < alphaInstances.size(); ++i)
    {
        const utils::Instance& instance = m_Scene.instances[alphaInstances[i]];
        const utils::Mesh& mesh = m_Scene.meshes[instance.meshIndex];
        const utils::Material& material = m_Scene.materials[instance.materialIndex];
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        geometry.meshIndex = instance.meshIndex;
        geometry.materialIndex = instance.materialIndex;

        geometry.alphaTexture = materialTextures[material.diffuseMapIndex];
        geometry.utilsTexture = m_Scene.textures[material.diffuseMapIndex];

        size_t uvDataSize = mesh.vertexNum * sizeof(float2);
        geometry.uvData.resize(uvDataSize);

        size_t positionDataSize = mesh.vertexNum * sizeof(float3);
        geometry.positions = positionBuffer;
        geometry.positionOffset = positions.size();
        geometry.positionBufferSize = positionBufferSize;
        positions.resize(geometry.positionOffset + helper::Align(positionDataSize, bufferAlignment));

        for (uint32_t y = 0; y < mesh.vertexNum; ++y)
        {
            uint32_t offset = mesh.vertexOffset + y;
            memcpy(geometry.uvData.data() + y * sizeof(float2), m_Scene.unpackedVertices[offset].uv, sizeof(float2));

            float3 position =
            {
                m_Scene.unpackedVertices[offset].position[0],
                m_Scene.unpackedVertices[offset].position[1],
                m_Scene.unpackedVertices[offset].position[2],
            };
            const size_t positionStride = sizeof(float3);
            void* dst = positions.data() + geometry.positionOffset + (y * positionStride);
            memcpy(dst, &position, positionStride);
        }

        size_t indexDataSize = mesh.indexNum * sizeof(utils::Index);
        geometry.indexData.resize(indexDataSize);
        memcpy(geometry.indexData.data(), m_Scene.indices.data() + mesh.indexOffset, indexDataSize);

        geometry.indices = indexBuffer;
        geometry.indexOffset = indices.size();
        geometry.indexBufferSize = indexBufferSize;
        indices.resize(geometry.indexOffset + helper::Align(indexDataSize, bufferAlignment));
        memcpy(indices.data() + geometry.indexOffset, m_Scene.indices.data() + mesh.indexOffset, indexDataSize);

        geometry.uvs = uvBuffer;
        geometry.uvOffset = uvs.size();
        geometry.uvBufferSize = uvBufferSize;
        uvs.resize(geometry.uvOffset + helper::Align(uvDataSize, storageAlignment));
        memcpy(uvs.data() + geometry.uvOffset, geometry.uvData.data(), uvDataSize);
    }

    { // Bind memories
        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.buffers = m_OmmAlphaGeometryBuffers.data();
        resourceGroupDesc.bufferNum = (uint32_t)m_OmmAlphaGeometryBuffers.size();
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        size_t allocationOffset = m_OmmAlphaGeometryMemories.size();
        m_OmmAlphaGeometryMemories.resize(allocationOffset + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_OmmAlphaGeometryMemories.data() + allocationOffset));
    }

    std::vector<nri::BufferUploadDesc> uploadDescs;
    {
        nri::BufferUploadDesc desc = {};
        desc.prevAccess = nri::AccessBits::UNKNOWN;
        desc.nextAccess = nri::AccessBits::SHADER_RESOURCE;

        desc.buffer = positionBuffer;
        desc.bufferOffset = 0;
        desc.data = positions.data();
        desc.dataSize = positionBufferSize;
        uploadDescs.push_back(desc);

        desc.buffer = uvBuffer;
        desc.bufferOffset = 0;
        desc.data = uvs.data();
        desc.dataSize = uvBufferSize;
        uploadDescs.push_back(desc);

        desc.buffer = indexBuffer;
        desc.bufferOffset = 0;
        desc.data = indices.data();
        desc.dataSize = indexBufferSize;
        uploadDescs.push_back(desc);
    }
    NRI.UploadData(*m_CommandQueue, nullptr, 0, uploadDescs.data(), (uint32_t)uploadDescs.size());
}

void PreprocessAlphaTexture(detexTexture* texture, std::vector<float>& outAlphaChannel)
{
    uint8_t* pixels = texture->data;
    std::vector<uint8_t> decompressedImage;
    uint32_t format = texture->format;
    { // Hack detex to decompress texture as BC1A to get alpha data
        uint32_t originalFormat = texture->format;
        if (originalFormat == DETEX_TEXTURE_FORMAT_BC1)
            texture->format = DETEX_TEXTURE_FORMAT_BC1A;

        if (detexFormatIsCompressed(texture->format))
        {
            uint32_t size = uint32_t(texture->width) * uint32_t(texture->height) * detexGetPixelSize(DETEX_PIXEL_FORMAT_RGBA8);
            decompressedImage.resize(size);
            detexDecompressTextureLinear(texture, &decompressedImage[0], DETEX_PIXEL_FORMAT_RGBA8);
            pixels = &decompressedImage[0];
            format = DETEX_PIXEL_FORMAT_RGBA8;
        }
        texture->format = originalFormat;
    }

    uint32_t pixelSize = detexGetPixelSize(format);
    uint32_t pixelCount = texture->width * texture->height;
    outAlphaChannel.reserve(pixelCount);

    float accumAlpha = 0.0f;
    for (uint32_t i = 0; i < pixelCount; ++i)
    {
        uint32_t offset = i * pixelSize;
        uint32_t alphaValue;
        if (pixelSize == 4)
        {
            uint32_t pixel = *(uint32_t*)(pixels + offset);
            alphaValue = detexPixel32GetA8(pixel);
        }
        else
        {
            uint64_t pixel = *(uint64_t*)(pixels + offset);
            alphaValue = (uint32_t)detexPixel64GetA16(pixel);
        }
        float alpha = (float)alphaValue / 255.0f;
        accumAlpha += alpha;
        outAlphaChannel.push_back(alpha);
    }

    accumAlpha = accumAlpha / (float)pixelCount;
    if (accumAlpha == 1.0f || accumAlpha == 0.0f)
        outAlphaChannel.resize(0);//texture is fully opaque or transparent
}

inline bool AreBakerOutputsOnGPU(const ommhelper::OmmBakeGeometryDesc& instance)
{
    bool result = true;
    for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++i)
        result &= bool(instance.gpuBuffers[i].dataSize);
    return result;
}

void Sample::FillOmmBakerInputs(std::vector<ommhelper::OmmBakeGeometryDesc>& ommBakeQueue)
{
    ommBakeQueue.resize(m_OmmAlphaGeometry.size());
    std::map<uint32_t, size_t> materialdToTextureDataOffset;
    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::CPU)
    { // Decompress textures and store alpha channel in a separate buffer for cpu baker
        std::set<uint32_t> uniqueMaterialIds;
        std::vector<float> workVector;
        for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i)
        { // Sort out unique textures to avoid resource duplication
            workVector.clear();

            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
            uint32_t materialId = geometry.materialIndex;

            size_t uniqueMaterialsNum = uniqueMaterialIds.size();
            uniqueMaterialIds.insert(materialId);
            if (uniqueMaterialsNum == uniqueMaterialIds.size())
                continue;//duplication

            size_t rawBufferOffset = m_OmmRawAlphaChannelForCpuBaker.size();
            const utils::Material& material = m_Scene.materials[materialId];
            utils::Texture* utilsTexture = m_Scene.textures[material.diffuseMapIndex];
            uint32_t minMip = utilsTexture->GetMipNum() - 1;
            uint32_t textureMipOffset = m_OmmBakeDesc.mipBias > minMip ? minMip : m_OmmBakeDesc.mipBias;
            detexTexture* texture = (detexTexture*)utilsTexture->mips[textureMipOffset];
            PreprocessAlphaTexture(texture, workVector);
            if (workVector.empty())
            {
                printf("[FAIL] Fully opaque/transparent texture has been submited for OMM baking!\n");
                std::abort();
            }
            m_OmmRawAlphaChannelForCpuBaker.insert(m_OmmRawAlphaChannelForCpuBaker.end(), workVector.begin(), workVector.end());
            materialdToTextureDataOffset.insert(std::make_pair(materialId, rawBufferOffset));
        }
    }

    for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i)
    { // Fill baking queue desc
        bool isGpuBaker = m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU;

        ommhelper::OmmBakeGeometryDesc& ommDesc = ommBakeQueue[i];
        const AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        const utils::Mesh& mesh = m_Scene.meshes[geometry.meshIndex];
        const utils::Material& material = m_Scene.materials[geometry.materialIndex];
        nri::Texture* texture = geometry.alphaTexture;

        utils::Texture* utilsTexture = m_Scene.textures[material.diffuseMapIndex];
        uint32_t minMip = utilsTexture->GetMipNum() - 1;
        uint32_t textureMipOffset = m_OmmBakeDesc.mipBias > minMip ? minMip : m_OmmBakeDesc.mipBias;

        if (isGpuBaker)
        {
            ommDesc.indices.nriBufferOrPtr.buffer = geometry.indices;
            ommDesc.uvs.nriBufferOrPtr.buffer = geometry.uvs;
            ommDesc.texture.nriTextureOrPtr.texture = texture;
        }
        else
        {
            ommDesc.indices.nriBufferOrPtr.ptr = (void*)geometry.indexData.data();
            ommDesc.uvs.nriBufferOrPtr.ptr = (void*)geometry.uvData.data();
            size_t texDataOffset = materialdToTextureDataOffset.find(geometry.materialIndex)->second;
            ommDesc.texture.nriTextureOrPtr.ptr = (void*)(m_OmmRawAlphaChannelForCpuBaker.data() + texDataOffset);
        }

        ommDesc.indices.numElements = mesh.indexNum;
        ommDesc.indices.stride = sizeof(utils::Index);
        ommDesc.indices.format = nri::Format::R16_UINT;
        ommDesc.indices.offset = geometry.indexOffset;
        ommDesc.indices.bufferSize = geometry.indexBufferSize;
        ommDesc.indices.offsetInStruct = 0;

        ommDesc.uvs.numElements = mesh.vertexNum;
        ommDesc.uvs.stride = sizeof(float2);
        ommDesc.uvs.format = nri::Format::RG32_SFLOAT;
        ommDesc.uvs.offset = geometry.uvOffset;
        ommDesc.uvs.bufferSize = geometry.uvBufferSize;
        ommDesc.uvs.offsetInStruct = 0;

        ommDesc.texture.width = reinterpret_cast<detexTexture*>(utilsTexture->mips[textureMipOffset])->width;
        ommDesc.texture.height = reinterpret_cast<detexTexture*>(utilsTexture->mips[textureMipOffset])->height;
        ommDesc.texture.mipOffset = textureMipOffset;
        ommDesc.texture.format = isGpuBaker ? utilsTexture->format : nri::Format::R32_SFLOAT;
        ommDesc.texture.addressingMode = nri::AddressMode::REPEAT;
        ommDesc.texture.alphaChannelId = 3;
        ommDesc.alphaCutoff = 0.5f;
        ommDesc.borderAlpha = 0.0f;
        ommDesc.alphaMode = ommhelper::OmmAlphaMode::Test;
    }
}

void PrepareOmmUsageCountsBuffers(ommhelper::OpacityMicroMapsHelper& ommHelper, ommhelper::OmmBakeGeometryDesc& desc)
{ // Sanitize baker outputed usageCounts buffers to fit GAPI format
    uint32_t usageCountBuffers[] = { (uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram, (uint32_t)ommhelper::OmmDataLayout::IndexHistogram };

    for (size_t i = 0; i < helper::GetCountOf(usageCountBuffers); ++i)
    {
        std::vector<uint8_t> buffer = desc.outData[usageCountBuffers[i]];
        size_t convertedCountsSize = 0;
        ommHelper.ConvertUsageCountsToApiFormat(nullptr, convertedCountsSize, buffer.data(), buffer.size());
        desc.outData[usageCountBuffers[i]].resize(convertedCountsSize);
        ommHelper.ConvertUsageCountsToApiFormat(desc.outData[usageCountBuffers[i]].data(), convertedCountsSize, buffer.data(), buffer.size());
    }
}

void PrepareCpuBuilderInputs(NRIInterface& NRI, ommhelper::OmmBakeGeometryDesc* ommBakeQueue, ommhelper::MaskedGeometryBuildDesc* ommGeometryBuildQueue, const size_t count)
{ // Copy raw mask data to the upload heaps to use during micromap and blas build
    for (size_t i = 0; i < count; ++i)
    {
        ommhelper::OmmBakeGeometryDesc& bakeResult = ommBakeQueue[i];
        if (bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram].empty())
            continue;

        ommhelper::MaskedGeometryBuildDesc& buildDesc = ommGeometryBuildQueue[i];
        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++y)
        {
            void* map = NRI.MapBuffer(*buildDesc.inputs.buffers[y].buffer, 0, bakeResult.outData[y].size());
            memcpy(map, bakeResult.outData[y].data(), bakeResult.outData[y].size());
            NRI.UnmapBuffer(*buildDesc.inputs.buffers[y].buffer);
        }
    }
}

void Sample::FillOmmBlasBuildInputs(size_t start, size_t count)
{
    nri::BufferDesc bufferDesc = {};
    bufferDesc.physicalDeviceMask = 0;
    bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE;

    size_t uploadBufferOffset = m_OmmCpuUploadBuffers.size();
    for (size_t i = start; i < start + count; ++i)
    {
        ommhelper::OmmBakeGeometryDesc& bakeResult = m_OmmBakeInstances[i];

        ommhelper::MaskedGeometryBuildDesc& buildDesc = m_OmmGeometryBuildQueue[i];
        const AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        const utils::Mesh& mesh = m_Scene.meshes[geometry.meshIndex];

        ommhelper::InputBuffer& vertices = buildDesc.inputs.vertices;
        vertices.nriBufferOrPtr.buffer = geometry.positions;
        vertices.format = geometry.vertexFormat;
        vertices.stride = sizeof(float3);
        vertices.numElements = mesh.vertexNum;
        vertices.offset = geometry.positionOffset;
        vertices.bufferSize = geometry.positionBufferSize;
        vertices.offsetInStruct = 0;

        ommhelper::InputBuffer& indices = buildDesc.inputs.indices;
        indices = bakeResult.indices;
        indices.nriBufferOrPtr.buffer = geometry.indices;

        if (bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram].empty())
            continue;

        buildDesc.inputs.ommIndexFormat = bakeResult.outOmmIndexFormat;
        buildDesc.inputs.ommIndexStride = bakeResult.outOmmIndexStride;

        PrepareOmmUsageCountsBuffers(m_OmmHelper, bakeResult);

        if(AreBakerOutputsOnGPU(bakeResult))
        {
            for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++j)
                buildDesc.inputs.buffers[j] = bakeResult.gpuBuffers[j];
        }
        else
        { // Create upload buffers to store baker output during ommArray/blas creation
            for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++j)
            {
                bufferDesc.size = bakeResult.outData[j].size();
                buildDesc.inputs.buffers[j].dataSize = bufferDesc.size;
                buildDesc.inputs.buffers[j].bufferSize = bufferDesc.size;
                NRI.CreateBuffer(*m_Device, bufferDesc, buildDesc.inputs.buffers[j].buffer);
                m_OmmCpuUploadBuffers.push_back(buildDesc.inputs.buffers[j].buffer);
            }
        }

        buildDesc.inputs.descArrayHistogram = bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram].data();
        buildDesc.inputs.descArrayHistogramNum = bakeResult.outDescArrayHistogramCount;

        buildDesc.inputs.indexHistogram = bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram].data();
        buildDesc.inputs.indexHistogramNum = bakeResult.outIndexHistogramCount;
    }

    if (m_OmmCpuUploadBuffers.empty() == false)
    { // Bind cpu baker output memories
        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.buffers = m_OmmCpuUploadBuffers.data() + uploadBufferOffset;
        resourceGroupDesc.bufferNum = (uint32_t)(m_OmmCpuUploadBuffers.size() - uploadBufferOffset);
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_UPLOAD;
        size_t allocationOffset = m_OmmTmpAllocations.size();
        m_OmmTmpAllocations.resize(allocationOffset + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_OmmTmpAllocations.data() + allocationOffset));
        PrepareCpuBuilderInputs(NRI, m_OmmBakeInstances.data() + start, m_OmmGeometryBuildQueue.data() + start, count);
    }

    for (size_t i = start; i < start + count; ++i)
    { // Release raw cpu side data. In case of cpu baker it's in the upload heaps, in case of gpu it's already saved as cache
        for (uint32_t k = 0; k < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++k)
        {
            m_OmmBakeInstances[i].outData[k].resize(0);
            m_OmmBakeInstances[i].outData[k].shrink_to_fit();
        }
    }
}

void CopyBatchToReadBackBuffer(NRIInterface& NRI, nri::CommandBuffer* commandBuffer, ommhelper::OmmBakeGeometryDesc& lastInBatch, uint32_t bufferId)
{
    ommhelper::GpuBakerBuffer& resource = lastInBatch.gpuBuffers[bufferId];
    ommhelper::GpuBakerBuffer& readback = lastInBatch.readBackBuffers[bufferId];

    nri::Buffer* src = resource.buffer;
    nri::Buffer* dst = readback.buffer;
    size_t srcOffset = 0;
    size_t dstOffset = 0;

    size_t size = resource.offset + resource.dataSize;//total size of baker output for the batch
    NRI.CmdCopyBuffer(*commandBuffer, *dst, 0, dstOffset, *src, 0, srcOffset, size);
}

void CopyFromReadBackBuffer(NRIInterface& NRI, ommhelper::OmmBakeGeometryDesc& desc, size_t id)
{
    ommhelper::GpuBakerBuffer& resource = desc.readBackBuffers[id];
    nri::Buffer* readback = resource.buffer;

    size_t offset = resource.offset;
    size_t size = resource.dataSize;
    std::vector<uint8_t>& data = desc.outData[id];
    data.resize(size);

    void* map = NRI.MapBuffer(*readback, offset, size);
    memcpy(data.data(), map, size);

    ZeroMemory(map, size);
    NRI.UnmapBuffer(*readback);
}

OmmGpuBakerPrebuildMemoryStats Sample::GetGpuBakerPrebuildMemoryStats()
{
    OmmGpuBakerPrebuildMemoryStats result = {};
    uint32_t sizeAlignment = NRI.GetDeviceDesc(*m_Device).storageBufferOffsetAlignment;
    size_t requestedMaxSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t i = 0; i < m_OmmBakeInstances.size(); ++i)
    {
        ommhelper::OmmBakeGeometryDesc& instance = m_OmmBakeInstances[i];
        ommhelper::OmmBakeGeometryDesc::GpuBakerPrebuildInfo& gpuBakerPreBuildInfo = instance.gpuBakerPreBuildInfo;

        size_t accumulation = 0;
        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
        {
            gpuBakerPreBuildInfo.dataSizes[y] = helper::Align(gpuBakerPreBuildInfo.dataSizes[y], sizeAlignment);
            accumulation += gpuBakerPreBuildInfo.dataSizes[y];
            requestedMaxSizes[y] = std::max(requestedMaxSizes[y], gpuBakerPreBuildInfo.dataSizes[y]);
        }

        result.maximum = std::max(result.maximum, accumulation);
        result.total += accumulation;

        for (size_t y = 0; y < omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS; ++y)
        {
            gpuBakerPreBuildInfo.transientBufferSizes[y] = helper::Align(gpuBakerPreBuildInfo.transientBufferSizes[y], sizeAlignment);
            result.maxTransientBufferSizes[y] = std::max(result.maxTransientBufferSizes[y], gpuBakerPreBuildInfo.transientBufferSizes[y]);
        }

    }

    auto toBytes = [](size_t sizeInMb) -> size_t { return sizeInMb * 1024 * 1024; };
    const size_t defaultSizes[] = { toBytes(64), toBytes(5), toBytes(5), toBytes(5), toBytes(5), 1024 };

    for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
        result.outputMaxSizes[y] = std::max<size_t>(requestedMaxSizes[y], defaultSizes[y]);

    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU)
    {
        uint64_t totalPrimitiveNum = 0;
        uint64_t maxPrimitiveNum = 0;
        for (auto& desc : m_OmmBakeInstances)
        {
            uint64_t numPrimitives = desc.indices.numElements / 3;
            totalPrimitiveNum += numPrimitives;
            maxPrimitiveNum = std::max<uint64_t>(maxPrimitiveNum, numPrimitives);
        }

        auto toMb = [](size_t sizeInBytes) -> double { return double(sizeInBytes) / 1024.0 / 1024.0; };
        printf("\n[OMM][GPU] PreBake Stats:\n");
        printf("Mask Format: [%s]\n", m_OmmBakeDesc.format == ommhelper::OmmFormats::OC1_2_STATE ? "OC1_2_STATE" : "OC1_4_STATE");
        printf("Subdivision Level: [%lu]\n", m_OmmBakeDesc.subdivisionLevel);
        printf("Mip Bias: [%lu]\n", m_OmmBakeDesc.mipBias);
        printf("Num Geoemetries: [%llu]\n", m_OmmBakeInstances.size());
        printf("Num Primitives: Max:[%llu],  Total:[%llu]\n", maxPrimitiveNum, totalPrimitiveNum);
        printf("Baker output memeory requested(mb): (max)%.3f | (total)%.3f\n", toMb(result.maximum), toMb(result.total));
        printf("Max ArrayDataSize(mb): %.3f\n", toMb(requestedMaxSizes[(uint32_t)ommhelper::OmmDataLayout::ArrayData]));
        printf("Max DescArraySize(mb): %.3f\n", toMb(requestedMaxSizes[(uint32_t)ommhelper::OmmDataLayout::DescArray]));
        printf("Max IndicesSize(mb): %.3f\n", toMb(requestedMaxSizes[(uint32_t)ommhelper::OmmDataLayout::Indices]));
    }
    return result;
}

std::vector<OmmBatch> GetGpuBakerBatches(const std::vector<ommhelper::OmmBakeGeometryDesc>& ommBakeQueue, const OmmGpuBakerPrebuildMemoryStats& memoryStats, const size_t batchSize)
{
    const size_t batchMaxSize = batchSize > ommBakeQueue.size() ? ommBakeQueue.size() : batchSize;
    std::vector<OmmBatch> batches(1);
    size_t accumulation[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t i = 0; i < ommBakeQueue.size(); ++i)
    {
        const ommhelper::OmmBakeGeometryDesc::GpuBakerPrebuildInfo& info = ommBakeQueue[i].gpuBakerPreBuildInfo;

        bool isAnyOverLimit = false;
        size_t nextSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum];
        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
        {
            nextSizes[y] = accumulation[y] + info.dataSizes[y];
            isAnyOverLimit |= nextSizes[y] > memoryStats.outputMaxSizes[y];
        }

        if (isAnyOverLimit)
        {
            batches.push_back({ i, 1 });
            for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
                accumulation[y] = info.dataSizes[y];
            continue;
        }

        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
            accumulation[y] = nextSizes[y];

        ++batches.back().count;
        if (batches.back().count >= batchMaxSize)
        {
            if (i + 1 < ommBakeQueue.size())
            {
                batches.push_back({ i + 1, 0 });
                for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
                    accumulation[y] = 0;
                continue;
            }
        }
    }
    return batches;
}

void Sample::BindGpuBakerBuffers(const OmmGpuBakerPrebuildMemoryStats& memoryStats, const size_t* ids, const size_t count)
{ // Bind gpu outputs and calculate offsets
    size_t offsets[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t i = 0; i < count; ++i)
    {
        size_t id = ids[i];
        ommhelper::OmmBakeGeometryDesc& desc = m_OmmBakeInstances[id];

        for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++j)
        {
            desc.gpuBuffers[j].dataSize = desc.gpuBakerPreBuildInfo.dataSizes[j];

            ommhelper::GpuBakerBuffer& resource = desc.gpuBuffers[j];
            ommhelper::GpuBakerBuffer& readback = desc.readBackBuffers[j];
            resource.buffer = m_OmmGpuOutputBuffers[j];
            readback.buffer = m_OmmGpuReadbackBuffers[j];

            resource.bufferSize = memoryStats.outputMaxSizes[j];
            readback.bufferSize = memoryStats.outputMaxSizes[j];

            resource.offset = readback.offset = offsets[j];
            readback.dataSize = resource.dataSize;

            offsets[j] += desc.gpuBakerPreBuildInfo.dataSizes[j];
        }

        for (size_t j = 0; j < m_OmmGpuTransientBuffers.size(); ++j)
        {
            desc.transientBuffers[j].buffer = m_OmmGpuTransientBuffers[j];
            desc.transientBuffers[j].bufferSize = desc.transientBuffers[j].dataSize = memoryStats.maxTransientBufferSizes[j];
            desc.transientBuffers[j].offset = 0;
        }
    }
}

void Sample::CreateGpuBakerBuffers(const OmmGpuBakerPrebuildMemoryStats& memoryStats)
{
    m_OmmGpuOutputBuffers.resize((uint32_t)ommhelper::OmmDataLayout::GpuOutputNum);
    m_OmmGpuReadbackBuffers.resize((uint32_t)ommhelper::OmmDataLayout::GpuOutputNum);

    nri::BufferDesc bufferDesc = {};
    bufferDesc.physicalDeviceMask = 0;
    bufferDesc.structureStride = sizeof(uint32_t);
    for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++i)
    {
        bufferDesc.size = memoryStats.outputMaxSizes[i];

        bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuOutputBuffers[i]));

        bufferDesc.usageMask = nri::BufferUsageBits::NONE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuReadbackBuffers[i]));
    }

    for (size_t i = 0; i < omm::Gpu::PreBakeInfo::MAX_TRANSIENT_POOL_BUFFERS; ++i)
    {
        bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ARGUMENT_BUFFER;
        bufferDesc.size = memoryStats.maxTransientBufferSizes[i];
        if (bufferDesc.size)
        {
            nri::Buffer* buffer = nullptr;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, buffer));
            m_OmmGpuTransientBuffers.push_back(buffer);
        }
    }

    { // Bind memories
        std::vector<nri::Buffer*> deviceMemoryBuffers;
        deviceMemoryBuffers.insert(deviceMemoryBuffers.end(), m_OmmGpuOutputBuffers.begin(), m_OmmGpuOutputBuffers.end());
        deviceMemoryBuffers.insert(deviceMemoryBuffers.end(), m_OmmGpuTransientBuffers.begin(), m_OmmGpuTransientBuffers.end());
        nri::ResourceGroupDesc resourceGroupDesc = {};
        resourceGroupDesc.buffers = deviceMemoryBuffers.data();
        resourceGroupDesc.bufferNum = (uint32_t)deviceMemoryBuffers.size();
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::DEVICE;
        size_t allocationOffset = m_OmmBakerAllocations.size();
        m_OmmBakerAllocations.resize(allocationOffset + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_OmmBakerAllocations.data() + allocationOffset));

        resourceGroupDesc.buffers = m_OmmGpuReadbackBuffers.data();
        resourceGroupDesc.bufferNum = (uint32_t)m_OmmGpuReadbackBuffers.size();
        resourceGroupDesc.memoryLocation = nri::MemoryLocation::HOST_READBACK;
        allocationOffset = m_OmmBakerAllocations.size();
        m_OmmBakerAllocations.resize(allocationOffset + NRI.CalculateAllocationNumber(*m_Device, resourceGroupDesc), nullptr);
        NRI_ABORT_ON_FAILURE(NRI.AllocateAndBindMemory(*m_Device, resourceGroupDesc, m_OmmBakerAllocations.data() + allocationOffset));
    }
}

void Sample::SaveMaskCache(uint32_t id)
{
    std::string cacheFileName = GetOmmCacheFilename();
    ommhelper::OmmCaching::CreateFolder(m_OmmCacheFolderName.c_str());

    AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
    uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);
    uint64_t stateMask = ommhelper::OmmCaching::PackStateMask(m_OmmBakeDesc);

    ommhelper::OmmBakeGeometryDesc& bakeResults = m_OmmBakeInstances[id];
    ommhelper::OmmCaching::OmmData data;
    for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++i)
    {
        data.data[i] = bakeResults.outData[i].data();
        data.sizes[i] = bakeResults.outData[i].size();
    }
    ommhelper::OmmCaching::SaveMasksToDisc(cacheFileName.c_str(), data, stateMask, hash, (uint16_t)bakeResults.outOmmIndexFormat);
}

void Sample::InitializeOmmGeometryFromCache(std::vector<size_t>& bakeQueue, size_t start, size_t count)
{ // Init geometry from cache. If cache not found add it to baking queue
    if (m_OmmBakeDesc.enableCache == false)
    {
        for (size_t i = start; i < start + count; ++i)
            bakeQueue.push_back(i);
        return;
    }

    printf("Read cache. ");
    for (size_t i = start; i < start + count; ++i)
    {
        ommhelper::OmmBakeGeometryDesc& instance = m_OmmBakeInstances[i];
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];

        uint64_t stateMask = ommhelper::OmmCaching::PackStateMask(m_OmmBakeDesc);
        uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);
        ommhelper::OmmCaching::OmmData data = {};
        if (ommhelper::OmmCaching::ReadMaskFromCache(GetOmmCacheFilename().c_str(), data, stateMask, hash, nullptr))
        {
            for (uint32_t j = 0; j < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++j)
            {
                instance.outData[j].resize(data.sizes[j]);
                data.data[j] = instance.outData[j].data();
            }
            ommhelper::OmmCaching::ReadMaskFromCache(GetOmmCacheFilename().c_str(), data, stateMask, hash, (uint16_t*)&instance.outOmmIndexFormat);
            instance.outOmmIndexStride = instance.outOmmIndexFormat == nri::Format::R16_UINT ? sizeof(uint16_t) : sizeof(uint32_t);
            instance.outDescArrayHistogramCount = uint32_t(data.sizes[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram] / (uint64_t)sizeof(omm::Cpu::OpacityMicromapUsageCount));
            instance.outIndexHistogramCount = uint32_t(data.sizes[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram] / (uint64_t)sizeof(omm::Cpu::OpacityMicromapUsageCount));
        }
        else
            bakeQueue.push_back(i);
    }
}

void Sample::BakeOmmGpu(size_t* queue, size_t count, const OmmGpuBakerPrebuildMemoryStats& memoryStats)
{
    if (m_OmmGpuOutputBuffers.empty())
        CreateGpuBakerBuffers(memoryStats);

    BindGpuBakerBuffers(memoryStats, queue, count);

    std::vector<ommhelper::OmmBakeGeometryDesc*> batch;
    for (size_t i = 0; i < count; ++i)
        batch.push_back(&m_OmmBakeInstances[queue[i]]);

    nri::CommandBuffer* commandBuffer = m_OmmContext.commandBuffer;
    NRI.ResetCommandAllocator(*m_OmmContext.commandAllocator);
    NRI.BeginCommandBuffer(*commandBuffer, nullptr, nri::WHOLE_DEVICE_GROUP);
    {
        m_OmmHelper.BakeOpacityMicroMapsGpu(commandBuffer, batch[0], batch.size(), m_OmmBakeDesc);

        CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch.back(), (uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram);
        CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch.back(), (uint32_t)ommhelper::OmmDataLayout::IndexHistogram);
        CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch.back(), (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBuffers = &commandBuffer;
    workSubmissionDesc.commandBufferNum = 1;
    NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, m_OmmContext.deviceSemaphore);
    NRI.WaitForSemaphore(*m_CommandQueue, *m_OmmContext.deviceSemaphore);

    m_OmmHelper.GpuPostBakeCleanUp();

    if (m_OmmBakeDesc.enableCache)
    {
        printf("Readback. ");
        NRI.ResetCommandAllocator(*m_OmmContext.commandAllocator);
        NRI.BeginCommandBuffer(*commandBuffer, nullptr, nri::WHOLE_DEVICE_GROUP);
        {
            for (size_t i = 0; i < count; ++i)
            { // Get actual data sizes from postbuild info
                size_t id = queue[i];
                ommhelper::OmmBakeGeometryDesc& desc = m_OmmBakeInstances[id];
                CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
                omm::Gpu::PostBakeInfo postbildInfo = *(omm::Gpu::PostBakeInfo*)desc.outData[(uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo].data();

                desc.gpuBuffers[(uint32_t)ommhelper::OmmDataLayout::ArrayData].dataSize = postbildInfo.outOmmArraySizeInBytes;
                desc.readBackBuffers[(uint32_t)ommhelper::OmmDataLayout::ArrayData].dataSize = postbildInfo.outOmmArraySizeInBytes;
                desc.gpuBuffers[(uint32_t)ommhelper::OmmDataLayout::DescArray].dataSize = postbildInfo.outOmmDescSizeInBytes;
                desc.readBackBuffers[(uint32_t)ommhelper::OmmDataLayout::DescArray].dataSize = postbildInfo.outOmmDescSizeInBytes;
            }

            {
                CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch.back(), (uint32_t)ommhelper::OmmDataLayout::ArrayData);
                CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch.back(), (uint32_t)ommhelper::OmmDataLayout::DescArray);
                CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch.back(), (uint32_t)ommhelper::OmmDataLayout::Indices);
            }
        }
        NRI.EndCommandBuffer(*commandBuffer);
        NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, m_OmmContext.deviceSemaphore);
        NRI.WaitForSemaphore(*m_CommandQueue, *m_OmmContext.deviceSemaphore);
    }

    for (size_t i = 0; i < count; ++i)
    {
        size_t id = queue[i];
        ommhelper::OmmBakeGeometryDesc& desc = m_OmmBakeInstances[id];
        CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram);
        CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::IndexHistogram);

        if (m_OmmBakeDesc.enableCache)
        {
            CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::ArrayData);
            CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::DescArray);
            CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::Indices);
        }
    }
}

void Sample::BakeOmmCpu(size_t* queue, size_t count)
{
    for (size_t i = 0; i < count; ++i)
    {
        size_t id = queue[i];
        ommhelper::OmmBakeGeometryDesc* instance = &m_OmmBakeInstances[id];
        m_OmmHelper.BakeOpacityMicroMapsCpu(instance, 1, m_OmmBakeDesc);
    }
}

void Sample::RebuildOmmGeometry()
{
    NRI.WaitForIdle(*m_CommandQueue);

    ReleaseMaskedGeometry();

    FillOmmBakerInputs(m_OmmBakeInstances);
    m_OmmGeometryBuildQueue.resize(m_OmmBakeInstances.size());

    OmmGpuBakerPrebuildMemoryStats memoryStats = {};
    bool isGpuBaker = m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU;
    if (isGpuBaker)
    { // Divide baking queue by memory requirments and batch size
        m_OmmHelper.GetGpuBakerPrebuildInfo(m_OmmBakeInstances.data(), m_OmmBakeInstances.size(), m_OmmBakeDesc);
        memoryStats = GetGpuBakerPrebuildMemoryStats();
    }
    std::vector<OmmBatch> batches = GetGpuBakerBatches(m_OmmBakeInstances, memoryStats, isGpuBaker ? m_OmmWorkloadBatchSize : 1);

    for (size_t i = 0; i < batches.size(); ++i)
    {
        printf("\r%s\r[OMM] Batch [%llu / %llu]: ", std::string(100, ' ').c_str(), i + 1, batches.size());
        std::vector<size_t> bakeQueue;
        InitializeOmmGeometryFromCache(bakeQueue, batches[i].offset, batches[i].count);

        if (!bakeQueue.empty())
        {
            printf("Bake. ");
            if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU)
                BakeOmmGpu(bakeQueue.data(), bakeQueue.size(), memoryStats);
            else
                BakeOmmCpu(bakeQueue.data(), bakeQueue.size()); 

            if (m_OmmBakeDesc.enableCache)
            {
                printf("Save cache. ");
                for (size_t k = 0; k < bakeQueue.size(); ++k)
                    SaveMaskCache((uint32_t)bakeQueue[k]);
            }
        }

        if (m_OmmBakeDesc.disableBlasBuild == false)
        {
            printf("Build. ");
            NRI.ResetCommandAllocator(*m_OmmContext.commandAllocator);
            NRI.BeginCommandBuffer(*m_OmmContext.commandBuffer, nullptr, nri::WHOLE_DEVICE_GROUP);
            {
                FillOmmBlasBuildInputs(batches[i].offset, batches[i].count);
                m_OmmHelper.BuildMaskedGeometry(m_OmmGeometryBuildQueue.data() + batches[i].offset, batches[i].count, m_OmmContext.commandBuffer);
            }
            NRI.EndCommandBuffer(*m_OmmContext.commandBuffer);

            nri::WorkSubmissionDesc workSubmissionDesc = {};
            workSubmissionDesc.commandBuffers = &m_OmmContext.commandBuffer;
            workSubmissionDesc.commandBufferNum = 1;
            NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, m_OmmContext.deviceSemaphore);
            NRI.WaitForSemaphore(*m_CommandQueue, *m_OmmContext.deviceSemaphore);
        }

        // Free cpu side memories with batch lifecycle
        for (auto& buffer : m_OmmCpuUploadBuffers)
            NRI.DestroyBuffer(*buffer);
        m_OmmCpuUploadBuffers.resize(0); m_OmmCpuUploadBuffers.shrink_to_fit();

        for (auto& memory : m_OmmTmpAllocations)
            NRI.FreeMemory(*memory);
        m_OmmTmpAllocations.resize(0); m_OmmTmpAllocations.shrink_to_fit();
    }
    printf("\n");

    for (size_t i = 0; i < m_OmmGeometryBuildQueue.size(); ++i)
    {
        if (!m_OmmGeometryBuildQueue[i].outputs.blas)
            continue;
        uint64_t mask = GetInstanceMask(m_OmmAlphaGeometry[i].meshIndex, m_OmmAlphaGeometry[i].materialIndex);
        OmmBlas ommBlas = { m_OmmGeometryBuildQueue[i].outputs.blas, m_OmmGeometryBuildQueue[i].outputs.ommArray };
        m_InstanceMaskToMaskedBlasData.insert(std::make_pair(mask, ommBlas));
        m_MaskedBlasses.push_back({ m_OmmGeometryBuildQueue[i].outputs.blas, m_OmmGeometryBuildQueue[i].outputs.ommArray });
    }
    ReleaseBakingResources();
}

void Sample::ReleaseMaskedGeometry()
{
    for (auto& resource : m_MaskedBlasses)
        m_OmmHelper.DestroyMaskedGeometry(resource.blas, resource.ommArray);

    m_InstanceMaskToMaskedBlasData.clear();
    m_MaskedBlasses.clear();
    m_OmmHelper.ReleaseGeometryMemory();
}

void Sample::ReleaseBakingResources()
{
    m_OmmBakeInstances.resize(0); m_OmmBakeInstances.shrink_to_fit();
    m_OmmGeometryBuildQueue.resize(0); m_OmmGeometryBuildQueue.shrink_to_fit();

    m_OmmRawAlphaChannelForCpuBaker.resize(0);
    m_OmmRawAlphaChannelForCpuBaker.shrink_to_fit();

    // Destroy buffers
    for (auto& buffer : m_OmmGpuOutputBuffers)
        NRI.DestroyBuffer(*buffer);
    m_OmmGpuOutputBuffers.resize(0); m_OmmGpuOutputBuffers.shrink_to_fit();

    for (auto& buffer : m_OmmGpuReadbackBuffers)
        NRI.DestroyBuffer(*buffer);
    m_OmmGpuReadbackBuffers.resize(0); m_OmmGpuReadbackBuffers.shrink_to_fit();

    for (auto& buffer : m_OmmCpuUploadBuffers)
        NRI.DestroyBuffer(*buffer);
    m_OmmCpuUploadBuffers.resize(0); m_OmmCpuUploadBuffers.shrink_to_fit();

    for (auto& buffer : m_OmmGpuTransientBuffers)
        NRI.DestroyBuffer(*buffer);
    m_OmmGpuTransientBuffers.resize(0); m_OmmGpuTransientBuffers.shrink_to_fit();

    // Release memories
    for (auto& memory : m_OmmTmpAllocations)
        NRI.FreeMemory(*memory);
    m_OmmTmpAllocations.resize(0); m_OmmTmpAllocations.shrink_to_fit();

    for (auto& memory : m_OmmBakerAllocations)
        NRI.FreeMemory(*memory);
    m_OmmBakerAllocations.resize(0); m_OmmBakerAllocations.shrink_to_fit();

    m_OmmHelper.GpuPostBakeCleanUp();
}

void Sample::AppendOmmImguiSettings()
{
    ImGui::PushStyleColor(ImGuiCol_Text, UI_HEADER);
    ImGui::PushStyleColor(ImGuiCol_Header, UI_HEADER_BACKGROUND);
    bool isUnfolded = ImGui::CollapsingHeader("VISIBILITY MASKS", ImGuiTreeNodeFlags_CollapsingHeader | ImGuiTreeNodeFlags_DefaultOpen);
    ImGui::PopStyleColor();
    ImGui::PopStyleColor();
    ImGui::PushID("VISIBILITY MASKS");
    {
        if (isUnfolded)
        {
            ImGui::Checkbox("Enable OMMs", &m_EnableOmm);
            ImGui::SameLine();
            ImGui::Text("[Masked Geometry Num: %llu]", m_MaskedBlasses.size());
            ImVec4 color = m_Settings.highLightAhs ? ImVec4(1.0f, 0.0f, 1.0f, 1.0f) : ImVec4(1.0f, 1.0f, 1.0f, 1.0f);
            ImGui::PushStyleColor(ImGuiCol_Text, color);
            ImGui::Checkbox("Highlight AHS", &m_Settings.highLightAhs);
            ImGui::PopStyleColor();
            ImGui::SameLine();
            ImGui::Checkbox("AHS Dynamic Mip", &m_Settings.ahsDynamicMipSelection);

            ImGui::Checkbox("Only Alpha Tested", &m_ShowOnlyAlphaTestedGeometry);

            ImGui::Separator();
            ImGui::Text("OMM Baking Settings:");

            #if _DEBUG
            {
                ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
                int batchSize = (int)m_OmmWorkloadBatchSize;
                ImGui::InputInt("OmmWorkloadBatchMaxSize", &batchSize);
                m_OmmWorkloadBatchSize = (size_t)(batchSize < 1 ? 1 : batchSize);
                ImGui::PopItemWidth();
            }
            #endif//_DEBUG

            static const char* ommBakerTypes[] = { "GPU", "CPU", };
            static int ommBakerTypeSelection = (int)m_OmmBakeDesc.type;
            ImGui::Combo("BakerType", &ommBakerTypeSelection, ommBakerTypes, helper::GetCountOf(ommBakerTypes));

            int32_t maxSubdivisionLevel = 12;
            if (ommBakerTypeSelection == 1)//if CPU
            {
                ommhelper::CpuBakerFlags& cpuFlags = m_OmmBakeDesc.cpuFlags;
                ImGui::Checkbox("SpecialIndices", &cpuFlags.enableSpecialIndices);
                ImGui::SameLine();
                ImGui::Checkbox("InternalThreads", &cpuFlags.enableInternalThreads);

                ImGui::Checkbox("DuplicateDetection", &cpuFlags.enableDuplicateDetection);
                ImGui::SameLine();
                ImGui::Checkbox("NearDuplicateDetection", &cpuFlags.enableNearDuplicateDetection);
            }
            else //if GPU
            {
                maxSubdivisionLevel = 9;//gpu baker is currently limited to level 9
                ommhelper::GpuBakerFlags& gpuFlags = m_OmmBakeDesc.gpuFlags;
                ImGui::Checkbox("SpecialIndices", &gpuFlags.enableSpecialIndices);
                ImGui::SameLine();
                ImGui::Checkbox("Compute Only", &gpuFlags.computeOnlyWorkload);
            }

            static int ommFormatSelection = (int)m_OmmBakeDesc.format;
            static const char* ommFormatNames[] = { "OC1_2_STATE", "OC1_4_STATE", };
            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::Combo("OMM Format", &ommFormatSelection, ommFormatNames, helper::GetCountOf(ommFormatNames));
            ImGui::PopItemWidth();

            static int ommFilterSelection = (int)m_OmmBakeDesc.filter;
            static const char* vmFilterNames[] = { "Nearest", "Linear", };
            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::Combo("Alpha Test Filter", &ommFilterSelection, vmFilterNames, helper::GetCountOf(ommFormatNames));
            ImGui::PopItemWidth();

            static int mipBias = m_OmmBakeDesc.mipBias;
            static int subdivisionLevel = m_OmmBakeDesc.subdivisionLevel;

            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::SliderFloat("Subdivision Scale", &m_OmmBakeDesc.dynamicSubdivisionScale, 0.0f, 100.0f);
            ImGui::PopItemWidth();

            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
            ImGui::InputInt("Max Subdivision Level", &subdivisionLevel);
            ImGui::PopItemWidth();
            subdivisionLevel = subdivisionLevel < 1 ? 1 : subdivisionLevel;
            subdivisionLevel = subdivisionLevel > maxSubdivisionLevel ? maxSubdivisionLevel : subdivisionLevel;


            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
            ImGui::InputInt("Mip Bias (if applicable)", &mipBias);
            ImGui::PopItemWidth();
            mipBias = mipBias < 0 ? 0 : mipBias;
            mipBias = mipBias > 15 ? 15 : mipBias;
            static bool enableCaching = m_OmmBakeDesc.enableCache;

            m_OmmBakeDesc.format = ommhelper::OmmFormats(ommFormatSelection);
            m_OmmBakeDesc.filter = ommhelper::OmmBakeFilter(ommFilterSelection);
            m_OmmBakeDesc.subdivisionLevel = subdivisionLevel;
            m_OmmBakeDesc.mipBias = mipBias;
            m_OmmBakeDesc.type = ommhelper::OmmBakerType(ommBakerTypeSelection);
            m_OmmBakeDesc.enableCache = enableCaching;

            static uint32_t frameId = 0;
            bool forceRebuild = frameId == m_OmmBakeDesc.buildFrameId;
            {
                if (ImGui::Button("Bake OMMs") || forceRebuild)
                    RebuildOmmGeometry();
                ImGui::SameLine();
                ImGui::Checkbox("Use OMM Cache", &enableCaching);
            }
            ++frameId;
        }
    }
    ImGui::PopID();
}

SAMPLE_MAIN(Sample, 0);

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
    int32_t     tracingMode                        = RESOLUTION_FULL;
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
    ommhelper::OmmBakeGeometryDesc bakeDesc;
    ommhelper::MaskedGeometryBuildDesc buildDesc;

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
    size_t total;
    size_t outputMaxSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum];
    size_t outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum];
    size_t maxTransientBufferSizes[OMM_SDK_TRANSIENT_BUFFER_MAX_NUM];
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

    void FillOmmBakerInputs();
    void FillOmmBlasBuildQueue(const OmmBatch& batch, std::vector<ommhelper::MaskedGeometryBuildDesc*>& outBuildQueue);

    void RunOmmSetupPass(ommhelper::OmmBakeGeometryDesc** queue, size_t count, OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void BakeOmmGpu(std::vector<ommhelper::OmmBakeGeometryDesc*>& batch);
    OmmGpuBakerPrebuildMemoryStats GetGpuBakerPrebuildMemoryStats(bool printStats);

    void CreateAndBindGpuBakerSatitcBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void CreateAndBindGpuBakerArrayDataBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats);
    void CreateAndBindGpuBakerReadbackBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats);

    inline uint64_t GetInstanceHash(uint32_t meshId, uint32_t materialId) { return uint64_t(meshId) << 32 | uint64_t(materialId); };
    inline std::string GetOmmCacheFilename() {return m_OmmCacheFolderName + std::string("/") + m_SceneName; };
    void InitializeOmmGeometryFromCache(const OmmBatch& batch, std::vector<ommhelper::OmmBakeGeometryDesc*>& outBakeQueue);
    void SaveMaskCache(const OmmBatch& batch);

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
    //std::vector<ommhelper::OmmBakeGeometryDesc> m_OmmBakeInstances;

    //temporal resources for baking
    std::vector<uint8_t> m_OmmRawAlphaChannelForCpuBaker;

    nri::Buffer* m_OmmGpuOutputBuffers[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    nri::Buffer* m_OmmGpuReadbackBuffers[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    nri::Buffer* m_OmmGpuTransientBuffers[OMM_SDK_TRANSIENT_BUFFER_MAX_NUM] = {};

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

    nri::PhysicalDeviceGroup mostPerformantPhysicalDeviceGroup = {};
    uint32_t deviceGroupNum = 1;
    NRI_ABORT_ON_FAILURE(nri::GetPhysicalDevices(&mostPerformantPhysicalDeviceGroup, deviceGroupNum));

    nri::DeviceCreationDesc deviceCreationDesc = {};
    deviceCreationDesc.graphicsAPI = graphicsAPI;
    deviceCreationDesc.enableAPIValidation = m_DebugAPI;
    deviceCreationDesc.enableNRIValidation = m_DebugNRI;
    deviceCreationDesc.spirvBindingOffsets = SPIRV_BINDING_OFFSETS;
    deviceCreationDesc.physicalDeviceGroup = &mostPerformantPhysicalDeviceGroup;
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

void BindBuffersToMemory(NRIInterface& nri, nri::Device* device, nri::Buffer** buffers, size_t count, std::vector<nri::Memory*>& memories, nri::MemoryLocation location)
{
    nri::ResourceGroupDesc resourceGroupDesc = {};
    resourceGroupDesc.buffers = buffers;
    resourceGroupDesc.bufferNum = (uint32_t)count;
    resourceGroupDesc.memoryLocation = location;
    size_t allocationOffset = memories.size();
    memories.resize(allocationOffset + nri.CalculateAllocationNumber(*device, resourceGroupDesc), nullptr);
    NRI_ABORT_ON_FAILURE(nri.AllocateAndBindMemory(*device, resourceGroupDesc, memories.data() + allocationOffset));
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
        BindBuffersToMemory(NRI, m_Device, m_OmmAlphaGeometryBuffers.data(), m_OmmAlphaGeometryBuffers.size(), m_OmmAlphaGeometryMemories, nri::MemoryLocation::DEVICE);
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

void PreprocessAlphaTexture(detexTexture* texture, std::vector<uint8_t>& outAlphaChannel)
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
        outAlphaChannel.push_back(uint8_t(alphaValue));
    }
}

inline bool AreBakerOutputsOnGPU(const ommhelper::OmmBakeGeometryDesc& instance)
{
    bool result = true;
    for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++i)
        result &= bool(instance.gpuBuffers[i].dataSize);
    return result;
}

void Sample::FillOmmBakerInputs()
{
    std::map<uint64_t, size_t> materialMaskToTextureDataOffset;
    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::CPU)
    { // Decompress textures and store alpha channel in a separate buffer for cpu baker
        std::set<uint32_t> uniqueMaterialIds;
        std::vector<uint8_t> workVector;
        for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i)
        { // Sort out unique textures to avoid resource duplication
            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
            ommhelper::InputTexture& bakerTexure = geometry.bakeDesc.texture;
            uint32_t materialId = geometry.materialIndex;

            const utils::Material& material = m_Scene.materials[materialId];
            utils::Texture* utilsTexture = m_Scene.textures[material.diffuseMapIndex];

            uint32_t minMip = utilsTexture->GetMipNum() - 1;
            uint32_t textureMipOffset = m_OmmBakeDesc.mipBias > minMip ? minMip : m_OmmBakeDesc.mipBias;
            uint32_t remainingMips = minMip - textureMipOffset + 1;
            uint32_t mipRange = m_OmmBakeDesc.mipCount > remainingMips ? remainingMips : m_OmmBakeDesc.mipCount;

            bakerTexure.mipOffset = textureMipOffset;
            bakerTexure.mipNum = mipRange;

            size_t uniqueMaterialsNum = uniqueMaterialIds.size();
            uniqueMaterialIds.insert(materialId);
            if (uniqueMaterialsNum == uniqueMaterialIds.size())
                continue;//duplication

            for (uint32_t mip = 0; mip < mipRange; ++mip)
            {
                uint32_t mipId = textureMipOffset + mip;
                detexTexture* texture = (detexTexture*)utilsTexture->mips[mipId];

                PreprocessAlphaTexture(texture, workVector);

            size_t rawBufferOffset = m_OmmRawAlphaChannelForCpuBaker.size();
                m_OmmRawAlphaChannelForCpuBaker.insert(m_OmmRawAlphaChannelForCpuBaker.end(), workVector.begin(), workVector.end());
                materialMaskToTextureDataOffset.insert(std::make_pair(uint64_t(materialId) << 32 | uint64_t(mipId), rawBufferOffset));
                workVector.clear();
            }
        }
    }

    for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i)
    { // Fill baking queue desc
        bool isGpuBaker = m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU;

        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        ommhelper::OmmBakeGeometryDesc& ommDesc = geometry.bakeDesc;
        const utils::Mesh& mesh = m_Scene.meshes[geometry.meshIndex];
        const utils::Material& material = m_Scene.materials[geometry.materialIndex];
        nri::Texture* texture = geometry.alphaTexture;

        ommhelper::InputTexture bakerTexture = ommDesc.texture;
        utils::Texture* utilsTexture = m_Scene.textures[material.diffuseMapIndex];
        if (isGpuBaker)
        {
            ommDesc.indices.nriBufferOrPtr.buffer = geometry.indices;
            ommDesc.uvs.nriBufferOrPtr.buffer = geometry.uvs;
            uint32_t minMip = utilsTexture->GetMipNum() - 1;
            uint32_t textureMipOffset = m_OmmBakeDesc.mipBias > minMip ? minMip : m_OmmBakeDesc.mipBias;
            ommDesc.texture.mipOffset = textureMipOffset;
            ommDesc.texture.mipNum = 1; // gpu baker currently doesnt support multiple mips support
            ommhelper::MipDesc& mipDesc = ommDesc.texture.mips[0];
            mipDesc.nriTextureOrPtr.texture = texture;
            mipDesc.width = reinterpret_cast<detexTexture*>(utilsTexture->mips[bakerTexture.mipOffset])->width;;
            mipDesc.height = reinterpret_cast<detexTexture*>(utilsTexture->mips[bakerTexture.mipOffset])->height;;
        }
        else
        {
            ommDesc.indices.nriBufferOrPtr.ptr = (void*)geometry.indexData.data();
            ommDesc.uvs.nriBufferOrPtr.ptr = (void*)geometry.uvData.data();

            for (uint32_t mip = 0; mip < bakerTexture.mipNum; ++mip)
            {
                uint32_t mipId = bakerTexture.mipOffset + mip;
                uint64_t materialMask = uint64_t(geometry.materialIndex) << 32 | uint64_t(mipId);
                size_t texDataOffset = materialMaskToTextureDataOffset.find(materialMask)->second;

                ommhelper::MipDesc& mipDesc = ommDesc.texture.mips[mip];
                mipDesc.nriTextureOrPtr.ptr = (void*)(m_OmmRawAlphaChannelForCpuBaker.data() + texDataOffset);
                mipDesc.width = reinterpret_cast<detexTexture*>(utilsTexture->mips[mipId])->width;
                mipDesc.height = reinterpret_cast<detexTexture*>(utilsTexture->mips[mipId])->height;
            }
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

        ommDesc.texture.format = isGpuBaker ? utilsTexture->format : nri::Format::R8_UNORM;
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

void PrepareCpuBuilderInputs(NRIInterface& NRI, const OmmBatch& batch, std::vector<AlphaTestedGeometry>& geometries)
{ // Copy raw mask data to the upload heaps to use during micromap and blas build
    for (size_t i = batch.offset; i < batch.offset + batch.count; ++i)
    {
        AlphaTestedGeometry& geometry = geometries[i];
        const ommhelper::OmmBakeGeometryDesc& bakeResult = geometry.bakeDesc;
        if (bakeResult.outData[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram].empty())
            continue;

        ommhelper::MaskedGeometryBuildDesc& buildDesc = geometry.buildDesc;
        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++y)
        {
            nri::Buffer* buffer = buildDesc.inputs.buffers[y].buffer;
            uint64_t mapSize = (uint64_t)bakeResult.outData[y].size();
            void* map = NRI.MapBuffer(*buffer, 0, mapSize);
            memcpy(map, bakeResult.outData[y].data(), bakeResult.outData[y].size());
            NRI.UnmapBuffer(*buildDesc.inputs.buffers[y].buffer);
        }
    }
}

void Sample::FillOmmBlasBuildQueue(const OmmBatch& batch, std::vector<ommhelper::MaskedGeometryBuildDesc*>& outBuildQueue)
{
    outBuildQueue.reserve(batch.count);

    size_t uploadBufferOffset = m_OmmCpuUploadBuffers.size();
    for (size_t id = batch.offset; id < batch.offset + batch.count; ++id)
    {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& bakeResult = geometry.bakeDesc;
        ommhelper::MaskedGeometryBuildDesc& buildDesc = geometry.buildDesc;
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
            nri::BufferDesc bufferDesc = {};
            bufferDesc.physicalDeviceMask = 0;
            bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE;

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
        outBuildQueue.push_back(&buildDesc);
    }

    if (m_OmmCpuUploadBuffers.empty() == false)
    { // Bind cpu baker output memories
        size_t uploadBufferCount = m_OmmCpuUploadBuffers.size() - uploadBufferOffset;
        BindBuffersToMemory(NRI, m_Device, m_OmmCpuUploadBuffers.data() + uploadBufferOffset, uploadBufferCount, m_OmmTmpAllocations, nri::MemoryLocation::HOST_UPLOAD);
        PrepareCpuBuilderInputs(NRI, batch, m_OmmAlphaGeometry);
    }

    for (size_t id = batch.offset; id < batch.offset + batch.count; ++id)
    { // Release raw cpu side data. In case of cpu baker it's in the upload heaps, in case of gpu it's already saved as cache
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& bakeResult = geometry.bakeDesc;
        for (uint32_t k = 0; k < (uint32_t)ommhelper::OmmDataLayout::BlasBuildGpuBuffersNum; ++k)
        {
             bakeResult.outData[k].resize(0);
             bakeResult.outData[k].shrink_to_fit();
        }
    }
}

void CopyBatchToReadBackBuffer(NRIInterface& NRI, nri::CommandBuffer* commandBuffer, ommhelper::OmmBakeGeometryDesc& firstInBatch, ommhelper::OmmBakeGeometryDesc& lastInBatch, uint32_t bufferId)
{
    ommhelper::GpuBakerBuffer& firstResource = firstInBatch.gpuBuffers[bufferId];
    ommhelper::GpuBakerBuffer& lastResource = lastInBatch.gpuBuffers[bufferId];
    ommhelper::GpuBakerBuffer& firstReadback = firstInBatch.readBackBuffers[bufferId];

    nri::Buffer* src = firstResource.buffer;
    nri::Buffer* dst = firstReadback.buffer;
    size_t srcOffset = firstResource.offset;
    size_t dstOffset = firstReadback.offset;

    size_t size = (lastResource.offset + lastResource.dataSize) - firstResource.offset;//total size of baker output for the batch
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

OmmGpuBakerPrebuildMemoryStats Sample::GetGpuBakerPrebuildMemoryStats(bool printStats)
{
    OmmGpuBakerPrebuildMemoryStats result = {};
    uint32_t sizeAlignment = NRI.GetDeviceDesc(*m_Device).storageBufferOffsetAlignment;
    for (size_t i = 0; i < m_OmmAlphaGeometry.size(); ++i)
    {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        ommhelper::OmmBakeGeometryDesc& instance = geometry.bakeDesc;
        ommhelper::OmmBakeGeometryDesc::GpuBakerPrebuildInfo& gpuBakerPreBuildInfo = instance.gpuBakerPreBuildInfo;

        for (uint32_t y = 0; y < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++y)
        {
            gpuBakerPreBuildInfo.dataSizes[y] = helper::Align(gpuBakerPreBuildInfo.dataSizes[y], sizeAlignment);
            result.outputTotalSizes[y] += gpuBakerPreBuildInfo.dataSizes[y];
            result.outputMaxSizes[y] = std::max<size_t>(gpuBakerPreBuildInfo.dataSizes[y], result.outputMaxSizes[y]);
            result.total += gpuBakerPreBuildInfo.dataSizes[y];
        }

        for (size_t y = 0; y < OMM_SDK_TRANSIENT_BUFFER_MAX_NUM; ++y)
        {
            gpuBakerPreBuildInfo.transientBufferSizes[y] = helper::Align(gpuBakerPreBuildInfo.transientBufferSizes[y], sizeAlignment);
            result.maxTransientBufferSizes[y] = std::max(result.maxTransientBufferSizes[y], gpuBakerPreBuildInfo.transientBufferSizes[y]);
        }
    }

    auto toBytes = [](size_t sizeInMb) -> size_t { return sizeInMb * 1024 * 1024; };
    const size_t defaultSizes[] = { toBytes(64), toBytes(5), toBytes(5), toBytes(5), toBytes(5), 1024 };

    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU && printStats)
    {
        uint64_t totalPrimitiveNum = 0;
        uint64_t maxPrimitiveNum = 0;
        for (auto& geomtry : m_OmmAlphaGeometry)
        {
            uint64_t numPrimitives = geomtry.bakeDesc.indices.numElements / 3;
            totalPrimitiveNum += numPrimitives;
            maxPrimitiveNum = std::max<uint64_t>(maxPrimitiveNum, numPrimitives);
        }

        auto toMb = [](size_t sizeInBytes) -> double { return double(sizeInBytes) / 1024.0 / 1024.0; };
        printf("\n[OMM][GPU] PreBake Stats:\n");
        printf("Mask Format: [%s]\n", m_OmmBakeDesc.format == ommhelper::OmmFormats::OC1_2_STATE ? "OC1_2_STATE" : "OC1_4_STATE");
        printf("Subdivision Level: [%lu]\n", m_OmmBakeDesc.subdivisionLevel);
        printf("Mip Bias: [%lu]\n", m_OmmBakeDesc.mipBias);
        printf("Num Geometries: [%llu]\n", m_OmmAlphaGeometry.size());
        printf("Num Primitives: Max:[%llu],  Total:[%llu]\n", maxPrimitiveNum, totalPrimitiveNum);
        printf("Baker output memeory requested(mb): (total)%.3f\n", toMb(result.total));
        printf("Total ArrayDataSize(mb): %.3f\n", toMb(result.outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::ArrayData]));
        printf("Total DescArraySize(mb): %.3f\n", toMb(result.outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::DescArray]));
        printf("Total IndicesSize(mb): %.3f\n", toMb(result.outputTotalSizes[(uint32_t)ommhelper::OmmDataLayout::Indices]));
    }
    return result;
}

std::vector<OmmBatch> GetGpuBakerBatches(const std::vector<AlphaTestedGeometry>& geometries, const OmmGpuBakerPrebuildMemoryStats& memoryStats, const size_t batchSize)
{
    const size_t batchMaxSize = batchSize > geometries.size() ? geometries.size() : batchSize;
    std::vector<OmmBatch> batches(1);
    size_t accumulation[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t i = 0; i < geometries.size(); ++i)
    {
        const AlphaTestedGeometry& geometry = geometries[i];
        const ommhelper::OmmBakeGeometryDesc& bakeDesc = geometry.bakeDesc;
        const ommhelper::OmmBakeGeometryDesc::GpuBakerPrebuildInfo& info = bakeDesc.gpuBakerPreBuildInfo;

        bool isAnyOverLimit = false;
        size_t nextSizes[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
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
            if (i + 1 < geometries.size())
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

void Sample::CreateAndBindGpuBakerReadbackBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats)
{ // for caching gpu produced omm_sdk output

    size_t dataTypeBegin = (size_t)ommhelper::OmmDataLayout::ArrayData;
    size_t dataTypeEnd = (size_t)ommhelper::OmmDataLayout::DescArrayHistogram;
    { // create and bind buffers to memory
        for (size_t i = dataTypeBegin; i < dataTypeEnd; ++i)
    {
            nri::BufferDesc bufferDesc = {};
            bufferDesc.physicalDeviceMask = 0;
            bufferDesc.structureStride = sizeof(uint32_t);
            bufferDesc.size = memoryStats.outputTotalSizes[i];
            bufferDesc.usageMask = nri::BufferUsageBits::NONE;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuReadbackBuffers[i]));
        }
        BindBuffersToMemory(NRI, m_Device, &m_OmmGpuReadbackBuffers[dataTypeBegin], dataTypeEnd - dataTypeBegin, m_OmmBakerAllocations, nri::MemoryLocation::HOST_READBACK);
    }

    { // bind baker insatnces to the buffer
        size_t perDataTypeOffsets[(size_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
        for (size_t id = 0; id < m_OmmAlphaGeometry.size(); ++id)
        {
            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
            ommhelper::OmmBakeGeometryDesc& desc = geometry.bakeDesc;
            for (size_t i = dataTypeBegin; i < dataTypeEnd; ++i)
        {
                ommhelper::GpuBakerBuffer& resource = desc.readBackBuffers[i];
                size_t& offset = perDataTypeOffsets[i];

                resource.dataSize = desc.gpuBakerPreBuildInfo.dataSizes[i];
                resource.buffer = m_OmmGpuReadbackBuffers[i];
                resource.bufferSize = memoryStats.outputTotalSizes[i];
                resource.offset = offset;
                offset += resource.dataSize;
            }
        }
    }
}

void Sample::CreateAndBindGpuBakerArrayDataBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats)
{ // in case of using setup pass of OMM-SDK, array data buffer allocation must be done separately
    const uint32_t arrayDataId = (uint32_t)ommhelper::OmmDataLayout::ArrayData;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.physicalDeviceMask = 0;
    bufferDesc.structureStride = sizeof(uint32_t);
    bufferDesc.size = memoryStats.outputTotalSizes[arrayDataId];
    bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE;
    NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuOutputBuffers[arrayDataId]));
    BindBuffersToMemory(NRI, m_Device, &m_OmmGpuOutputBuffers[arrayDataId], 1, m_OmmBakerAllocations, nri::MemoryLocation::DEVICE);

    size_t offset = 0;
    for (size_t id = 0; id < m_OmmAlphaGeometry.size(); ++id)
    {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& desc = geometry.bakeDesc;
        ommhelper::GpuBakerBuffer& resource = desc.gpuBuffers[arrayDataId];

        resource.dataSize = desc.gpuBakerPreBuildInfo.dataSizes[arrayDataId];
        resource.buffer = m_OmmGpuOutputBuffers[arrayDataId];
        resource.bufferSize = memoryStats.outputTotalSizes[arrayDataId];
        resource.offset = offset;
        offset += desc.gpuBakerPreBuildInfo.dataSizes[arrayDataId];
    }
}

void Sample::CreateAndBindGpuBakerSatitcBuffer(const OmmGpuBakerPrebuildMemoryStats& memoryStats)
{
    const size_t postBakeReadbackDataBegin = (size_t)ommhelper::OmmDataLayout::DescArrayHistogram;
    const size_t staticDataBegin = (size_t)ommhelper::OmmDataLayout::DescArray;
    const size_t buffersEnd = (size_t)ommhelper::OmmDataLayout::GpuOutputNum;

    nri::BufferDesc bufferDesc = {};
    bufferDesc.physicalDeviceMask = 0;
    bufferDesc.structureStride = sizeof(uint32_t);

    std::vector<nri::Buffer*> gpuBuffers;
    std::vector<nri::Buffer*> readbackBuffers;
    for (size_t i = staticDataBegin; i < buffersEnd; ++i)
    {
        bufferDesc.size = memoryStats.outputTotalSizes[i];
        bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuOutputBuffers[i]));
        gpuBuffers.push_back(m_OmmGpuOutputBuffers[i]);
    }

    for (size_t i = 0; i < OMM_SDK_TRANSIENT_BUFFER_MAX_NUM; ++i)
    {
        bufferDesc.size = memoryStats.maxTransientBufferSizes[i];
        if (bufferDesc.size)
        {
            bufferDesc.usageMask = nri::BufferUsageBits::SHADER_RESOURCE_STORAGE | nri::BufferUsageBits::SHADER_RESOURCE | nri::BufferUsageBits::ARGUMENT_BUFFER;
            NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuTransientBuffers[i]));
            gpuBuffers.push_back(m_OmmGpuTransientBuffers[i]);
        }
    }

    for (size_t i = postBakeReadbackDataBegin; i < buffersEnd; ++i)
    {
        bufferDesc.size = memoryStats.outputTotalSizes[i];
        bufferDesc.usageMask = nri::BufferUsageBits::NONE;
        NRI_ABORT_ON_FAILURE(NRI.CreateBuffer(*m_Device, bufferDesc, m_OmmGpuReadbackBuffers[i]));
        readbackBuffers.push_back(m_OmmGpuReadbackBuffers[i]);
    }

    { // Bind memories
        BindBuffersToMemory(NRI, m_Device, gpuBuffers.data(), gpuBuffers.size(), m_OmmBakerAllocations, nri::MemoryLocation::DEVICE);
        BindBuffersToMemory(NRI, m_Device, readbackBuffers.data(), readbackBuffers.size(), m_OmmBakerAllocations, nri::MemoryLocation::HOST_READBACK);
    }

    size_t gpuOffsetsPerType[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    size_t readBackOffsetsPerType[(uint32_t)ommhelper::OmmDataLayout::GpuOutputNum] = {};
    for (size_t id = 0; id < m_OmmAlphaGeometry.size(); ++id)
    {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& desc = geometry.bakeDesc;
        for (uint32_t j = staticDataBegin; j < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++j)
        {
            ommhelper::GpuBakerBuffer& resource = desc.gpuBuffers[j];
            size_t& offset = gpuOffsetsPerType[j];

            desc.gpuBuffers[j].dataSize = desc.gpuBakerPreBuildInfo.dataSizes[j];
            resource.buffer = m_OmmGpuOutputBuffers[j];
            resource.bufferSize = memoryStats.outputTotalSizes[j];
            resource.offset = offset;
            offset += desc.gpuBakerPreBuildInfo.dataSizes[j];
        }

        for (uint32_t j = postBakeReadbackDataBegin; j < (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum; ++j)
        {
            ommhelper::GpuBakerBuffer& resource = desc.readBackBuffers[j];
            size_t& offset = readBackOffsetsPerType[j];

            resource.dataSize = desc.gpuBakerPreBuildInfo.dataSizes[j];
            resource.buffer = m_OmmGpuReadbackBuffers[j];
            resource.bufferSize = memoryStats.outputTotalSizes[j];
            resource.offset = offset;
            offset += resource.dataSize;
        }

        for (size_t j = 0; j < OMM_SDK_TRANSIENT_BUFFER_MAX_NUM; ++j)
        {
            desc.transientBuffers[j].buffer = m_OmmGpuTransientBuffers[j];
            desc.transientBuffers[j].bufferSize = memoryStats.maxTransientBufferSizes[j];
            desc.transientBuffers[j].dataSize = memoryStats.maxTransientBufferSizes[j];
            desc.transientBuffers[j].offset = 0;
        }
    }
}

void Sample::SaveMaskCache(const OmmBatch& batch)
{
    std::string cacheFileName = GetOmmCacheFilename();
    ommhelper::OmmCaching::CreateFolder(m_OmmCacheFolderName.c_str());
    uint64_t stateMask = ommhelper::OmmCaching::CalculateSateHash(m_OmmBakeDesc);

    for (size_t id = batch.offset; id < batch.offset + batch.count; ++id)
    {
    AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
        ommhelper::OmmBakeGeometryDesc& bakeResults = geometry.bakeDesc;
    uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);

        bool isDataValid = true;
    ommhelper::OmmCaching::OmmData data;
    for (uint32_t i = 0; i < (uint32_t)ommhelper::OmmDataLayout::CpuMaxNum; ++i)
    {
        data.data[i] = bakeResults.outData[i].data();
        data.sizes[i] = bakeResults.outData[i].size();
            isDataValid &= data.sizes[i] > 0;
    }
        if(isDataValid)
    ommhelper::OmmCaching::SaveMasksToDisc(cacheFileName.c_str(), data, stateMask, hash, (uint16_t)bakeResults.outOmmIndexFormat);
}
}

void Sample::InitializeOmmGeometryFromCache(const OmmBatch& batch, std::vector<ommhelper::OmmBakeGeometryDesc*>& outBakeQueue)
{ // Init geometry from cache. If cache not found add it to baking queue
    if (m_OmmBakeDesc.enableCache == false)
    {
        for (size_t i = batch.offset; i < batch.offset + batch.count; ++i)
            outBakeQueue.push_back(&m_OmmAlphaGeometry[i].bakeDesc);
        return;
    }

    printf("Read cache. ");
    uint64_t stateMask = ommhelper::OmmCaching::CalculateSateHash(m_OmmBakeDesc);
    for (size_t i = batch.offset; i < batch.offset + batch.count; ++i)
    {
        AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[i];
        ommhelper::OmmBakeGeometryDesc& instance = geometry.bakeDesc;

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
            instance.outDescArrayHistogramCount = uint32_t(data.sizes[(uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram] / (uint64_t)sizeof(ommCpuOpacityMicromapUsageCount));
            instance.outIndexHistogramCount = uint32_t(data.sizes[(uint32_t)ommhelper::OmmDataLayout::IndexHistogram] / (uint64_t)sizeof(ommCpuOpacityMicromapUsageCount));
        }
        else
            outBakeQueue.push_back(&instance);
    }
}

void Sample::RunOmmSetupPass(ommhelper::OmmBakeGeometryDesc** queue, size_t count, OmmGpuBakerPrebuildMemoryStats& memoryStats)
{ // Run prepass to get correct size of omm array data buffer
    nri::CommandBuffer* commandBuffer = m_OmmContext.commandBuffer;
    NRI.ResetCommandAllocator(*m_OmmContext.commandAllocator);
    NRI.BeginCommandBuffer(*commandBuffer, nullptr, nri::WHOLE_DEVICE_GROUP);
    {
        m_OmmHelper.BakeOpacityMicroMapsGpu(commandBuffer, queue, count, m_OmmBakeDesc, ommhelper::OmmGpuBakerPass::Setup);
        CopyBatchToReadBackBuffer(NRI, commandBuffer, *queue[0], *queue[count - 1], (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
    }
    NRI.EndCommandBuffer(*commandBuffer);

    nri::WorkSubmissionDesc workSubmissionDesc = {};
    workSubmissionDesc.commandBuffers = &commandBuffer;
    workSubmissionDesc.commandBufferNum = 1;
    NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, m_OmmContext.deviceSemaphore);
    NRI.WaitForSemaphore(*m_CommandQueue, *m_OmmContext.deviceSemaphore);
    m_OmmHelper.GpuPostBakeCleanUp();

    for (size_t i = 0; i < count; ++i)
    { // Get actual data sizes from postbuild info
        ommhelper::OmmBakeGeometryDesc& desc = *queue[i];
        CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
        ommGpuPostBakeInfo postbildInfo = *(ommGpuPostBakeInfo*)desc.outData[(uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo].data();
        desc.gpuBakerPreBuildInfo.dataSizes[(uint32_t)ommhelper::OmmDataLayout::ArrayData] = postbildInfo.outOmmArraySizeInBytes;
    }
    memoryStats = GetGpuBakerPrebuildMemoryStats(true);
}

void Sample::BakeOmmGpu(std::vector<ommhelper::OmmBakeGeometryDesc*>& batch)
{
    nri::CommandBuffer* commandBuffer = m_OmmContext.commandBuffer;
    NRI.ResetCommandAllocator(*m_OmmContext.commandAllocator);
    NRI.BeginCommandBuffer(*commandBuffer, nullptr, nri::WHOLE_DEVICE_GROUP);
    {
        m_OmmHelper.BakeOpacityMicroMapsGpu(commandBuffer, batch.data(), batch.size(), m_OmmBakeDesc, ommhelper::OmmGpuBakerPass::Bake);
        CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::DescArrayHistogram);
        CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::IndexHistogram);
        CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
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
            for (size_t i = 0; i < batch.size(); ++i)
            { // Get actual data sizes from postbuild info
                ommhelper::OmmBakeGeometryDesc& desc = *batch[i];
                CopyFromReadBackBuffer(NRI, desc, (uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo);
                ommGpuPostBakeInfo postbildInfo = *(ommGpuPostBakeInfo*)desc.outData[(uint32_t)ommhelper::OmmDataLayout::GpuPostBuildInfo].data();

                desc.gpuBuffers[(uint32_t)ommhelper::OmmDataLayout::ArrayData].dataSize = postbildInfo.outOmmArraySizeInBytes;
                desc.readBackBuffers[(uint32_t)ommhelper::OmmDataLayout::ArrayData].dataSize = postbildInfo.outOmmArraySizeInBytes;
                desc.gpuBuffers[(uint32_t)ommhelper::OmmDataLayout::DescArray].dataSize = postbildInfo.outOmmDescSizeInBytes;
                desc.readBackBuffers[(uint32_t)ommhelper::OmmDataLayout::DescArray].dataSize = postbildInfo.outOmmDescSizeInBytes;
            }

            {
                CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::ArrayData);
                CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::DescArray);
                CopyBatchToReadBackBuffer(NRI, commandBuffer, *batch[0], *batch.back(), (uint32_t)ommhelper::OmmDataLayout::Indices);
            }
        }
        NRI.EndCommandBuffer(*commandBuffer);
        NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, m_OmmContext.deviceSemaphore);
        NRI.WaitForSemaphore(*m_CommandQueue, *m_OmmContext.deviceSemaphore);
    }

    for (size_t i = 0; i < batch.size(); ++i)
    {
        ommhelper::OmmBakeGeometryDesc& desc = *batch[i];
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

void Sample::RebuildOmmGeometry()
{
    NRI.WaitForIdle(*m_CommandQueue);

    ReleaseMaskedGeometry();

    FillOmmBakerInputs();
    OmmGpuBakerPrebuildMemoryStats memoryStats = {};
    std::vector<OmmBatch> batches = GetGpuBakerBatches(m_OmmAlphaGeometry, memoryStats, 1);

    if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU)
    {
        std::vector<ommhelper::OmmBakeGeometryDesc*> queue;
        uint64_t stateMask = ommhelper::OmmCaching::CalculateSateHash(m_OmmBakeDesc);

        for (size_t instanceId = 0; instanceId < m_OmmAlphaGeometry.size(); ++instanceId)
        { // skip prepass for instances with cache
            AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[instanceId];
            uint64_t hash = GetInstanceHash(geometry.meshIndex, geometry.materialIndex);
            if (ommhelper::OmmCaching::LookForCache(GetOmmCacheFilename().c_str(), stateMask, hash) && m_OmmBakeDesc.enableCache)
                continue;
            queue.push_back(&geometry.bakeDesc);
        }

        if (queue.empty() == false)
        { // perform setup pass
            m_OmmHelper.GetGpuBakerPrebuildInfo(queue.data(), queue.size(), m_OmmBakeDesc);
            memoryStats = GetGpuBakerPrebuildMemoryStats(false); // arrayData size calculation is conservative here

            CreateAndBindGpuBakerSatitcBuffer(memoryStats); // create buffers which sizes are correctly calculated in GetGpuBakerPrebuildInfo()
            { // get actual arrayData buffer sizes. GetGpuBakerPrebuildInfo() returns conservative arrayData size estimation
                RunOmmSetupPass(queue.data(), queue.size(), memoryStats);
            }
            CreateAndBindGpuBakerArrayDataBuffer(memoryStats);
            
            if (m_OmmBakeDesc.enableCache)
                CreateAndBindGpuBakerReadbackBuffer(memoryStats);

            batches.clear(); batches.push_back({ 0, m_OmmAlphaGeometry.size() });
        }
    }

    for (size_t batchId = 0; batchId < batches.size(); ++batchId)
    {
        const OmmBatch& batch = batches[batchId];
        printf("\r%s\r[OMM] Batch [%llu / %llu]: ", std::string(100, ' ').c_str(), batchId + 1, batches.size());
        std::vector<ommhelper::OmmBakeGeometryDesc*> bakeQueue;
        InitializeOmmGeometryFromCache(batch, bakeQueue);

        if (!bakeQueue.empty())
        {
            printf("Bake. ");
            if (m_OmmBakeDesc.type == ommhelper::OmmBakerType::GPU)
                BakeOmmGpu(bakeQueue);
            else
                m_OmmHelper.BakeOpacityMicroMapsCpu(bakeQueue.data(), bakeQueue.size(), m_OmmBakeDesc);

            if (m_OmmBakeDesc.enableCache)
            {
                printf("Save cache. ");
                SaveMaskCache(batch);
            }
        }

        if (m_OmmBakeDesc.disableBlasBuild == false)
        {
            printf("Build. ");

            std::vector<ommhelper::MaskedGeometryBuildDesc*> buildQueue = {};
            FillOmmBlasBuildQueue(batch, buildQueue);

            NRI.ResetCommandAllocator(*m_OmmContext.commandAllocator);
            NRI.BeginCommandBuffer(*m_OmmContext.commandBuffer, nullptr, nri::WHOLE_DEVICE_GROUP);
            {
                m_OmmHelper.BuildMaskedGeometry(buildQueue.data(), buildQueue.size(), m_OmmContext.commandBuffer);
            }
            NRI.EndCommandBuffer(*m_OmmContext.commandBuffer);

            nri::WorkSubmissionDesc workSubmissionDesc = {};
            workSubmissionDesc.commandBuffers = &m_OmmContext.commandBuffer;
            workSubmissionDesc.commandBufferNum = 1;
            NRI.SubmitQueueWork(*m_CommandQueue, workSubmissionDesc, m_OmmContext.deviceSemaphore);
            NRI.WaitForSemaphore(*m_CommandQueue, *m_OmmContext.deviceSemaphore);

            for (size_t id = batch.offset; id < batch.offset + batch.count; ++id)
            {
                AlphaTestedGeometry& geometry = m_OmmAlphaGeometry[id];
                ommhelper::MaskedGeometryBuildDesc& buildDesc = geometry.buildDesc;
                if (!buildDesc.outputs.blas)
                    continue;

                uint64_t mask = GetInstanceHash(m_OmmAlphaGeometry[id].meshIndex, m_OmmAlphaGeometry[id].materialIndex);
                OmmBlas ommBlas = { buildDesc.outputs.blas, buildDesc.outputs.ommArray };
                m_InstanceMaskToMaskedBlasData.insert(std::make_pair(mask, ommBlas));
                m_MaskedBlasses.push_back({ buildDesc.outputs.blas, buildDesc.outputs.ommArray });
            }
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
    for (AlphaTestedGeometry& geometry : m_OmmAlphaGeometry)
    {
        geometry.bakeDesc = {};
        geometry.buildDesc = {};
    }

    m_OmmRawAlphaChannelForCpuBaker.resize(0);
    m_OmmRawAlphaChannelForCpuBaker.shrink_to_fit();

    // Destroy buffers
    auto DestroyBuffers = [](NRIInterface& nri, nri::Buffer** buffers, uint32_t count)
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            if (buffers[i])
            {
                nri.DestroyBuffer(*buffers[i]);
                buffers[i] = nullptr;
            }
        }
    };
    DestroyBuffers(NRI, m_OmmGpuOutputBuffers, (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum);
    DestroyBuffers(NRI, m_OmmGpuReadbackBuffers, (uint32_t)ommhelper::OmmDataLayout::GpuOutputNum);
    DestroyBuffers(NRI, m_OmmGpuTransientBuffers, OMM_SDK_TRANSIENT_BUFFER_MAX_NUM);

    for (auto& buffer : m_OmmCpuUploadBuffers)
        NRI.DestroyBuffer(*buffer);
    m_OmmCpuUploadBuffers.resize(0); m_OmmCpuUploadBuffers.shrink_to_fit();

    // Release memories
    for (auto& memory : m_OmmTmpAllocations)
        NRI.FreeMemory(*memory);
    m_OmmTmpAllocations.resize(0); m_OmmTmpAllocations.shrink_to_fit();

    for (auto& memory : m_OmmBakerAllocations)
        NRI.FreeMemory(*memory);
    m_OmmBakerAllocations.resize(0); m_OmmBakerAllocations.shrink_to_fit();

    m_OmmHelper.GpuPostBakeCleanUp();
}

bool IsRebuildAvailable(ommhelper::OmmBakeDesc& updated, ommhelper::OmmBakeDesc& current)
{
    bool result = false;
    result |= updated.subdivisionLevel != current.subdivisionLevel;
    result |= updated.mipBias != current.mipBias;
    result |= updated.dynamicSubdivisionScale != current.dynamicSubdivisionScale;
    result |= updated.filter != current.filter;
    result |= updated.format != current.format;
    
    result |= updated.type != current.type;
    if (current.type == ommhelper::OmmBakerType::GPU)
    {
        result |= updated.gpuFlags.computeOnlyWorkload != current.gpuFlags.computeOnlyWorkload;
        result |= updated.gpuFlags.enablePostBuildInfo != current.gpuFlags.enablePostBuildInfo;
        result |= updated.gpuFlags.enableTexCoordDeduplication != current.gpuFlags.enableTexCoordDeduplication;
        result |= updated.gpuFlags.force32bitIndices != current.gpuFlags.force32bitIndices;
        result |= updated.gpuFlags.enableSpecialIndices != current.gpuFlags.enableSpecialIndices;
    }
    else
    {
        result |= updated.mipCount != current.mipCount;
        result |= updated.cpuFlags.enableInternalThreads != current.cpuFlags.enableInternalThreads;
        result |= updated.cpuFlags.enableSpecialIndices != current.cpuFlags.enableSpecialIndices;
        result |= updated.cpuFlags.enableDuplicateDetection != current.cpuFlags.enableDuplicateDetection;
        result |= updated.cpuFlags.enableNearDuplicateDetection != current.cpuFlags.enableNearDuplicateDetection;
        result |= updated.cpuFlags.force32bitIndices != current.cpuFlags.force32bitIndices;
    }

    result |= ((current.enableCache == false) && updated.enableCache);

    return result;
}

void Sample::AppendOmmImguiSettings()
{
    static ommhelper::OmmBakeDesc bakeDesc = m_OmmBakeDesc;

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

            static const char* ommBakerTypes[] = { "GPU", "CPU", };
            static int ommBakerTypeSelection = (int)bakeDesc.type;
            ImGui::Combo("BakerType", &ommBakerTypeSelection, ommBakerTypes, helper::GetCountOf(ommBakerTypes));

            int32_t maxSubdivisionLevel = 12;
            float maxSubdivisionScale = 12.0f;
            bool isCpuBaker = ommBakerTypeSelection == 1;
            if (isCpuBaker)//if CPU
            {
                ommhelper::CpuBakerFlags& cpuFlags = bakeDesc.cpuFlags;
                ImGui::Checkbox("SpecialIndices", &cpuFlags.enableSpecialIndices);
                ImGui::SameLine();
                ImGui::Checkbox("InternalThreads", &cpuFlags.enableInternalThreads);

                ImGui::Checkbox("DuplicateDetection", &cpuFlags.enableDuplicateDetection);
                ImGui::SameLine();
                ImGui::Checkbox("NearDuplicateDetection", &cpuFlags.enableNearDuplicateDetection);
            }
            else //if GPU
            {
                ommhelper::GpuBakerFlags& gpuFlags = bakeDesc.gpuFlags;
                maxSubdivisionLevel = gpuFlags.computeOnlyWorkload ? 10 : 9;//gpu baker is currently limited to level 9. 10 in compute only regime
                ImGui::Checkbox("SpecialIndices", &gpuFlags.enableSpecialIndices);
                ImGui::SameLine();
                ImGui::Checkbox("Compute Only", &gpuFlags.computeOnlyWorkload);
                maxSubdivisionScale = gpuFlags.computeOnlyWorkload ? maxSubdivisionScale : 9.0f;
            }

            static int ommFormatSelection = (int)bakeDesc.format;
            static const char* ommFormatNames[] = { "OC1_2_STATE", "OC1_4_STATE", };
            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::Combo("OMM Format", &ommFormatSelection, ommFormatNames, helper::GetCountOf(ommFormatNames));
            ImGui::PopItemWidth();

            static int ommFilterSelection = (int)bakeDesc.filter;
            static const char* vmFilterNames[] = { "Nearest", "Linear", };
            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
            ImGui::Combo("Alpha Test Filter", &ommFilterSelection, vmFilterNames, helper::GetCountOf(ommFormatNames));
            ImGui::PopItemWidth();

            static int mipBias = bakeDesc.mipBias;
            static int mipCount = bakeDesc.mipCount;
            static int subdivisionLevel = bakeDesc.subdivisionLevel;

            static float subdivisionScale = bakeDesc.dynamicSubdivisionScale;
            static bool enableDynamicSubdivisionScale = true;
            if (enableDynamicSubdivisionScale)
            {
                ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.66f);
                ImGui::SliderFloat("Subdivision Scale", &subdivisionScale, 0.1f, maxSubdivisionScale, "%.1f");
                ImGui::PopItemWidth();
                ImGui::SameLine();
            }

            ImGui::Checkbox(enableDynamicSubdivisionScale ? " " : "Enable Subdivision Scale", &enableDynamicSubdivisionScale);
            bakeDesc.dynamicSubdivisionScale = enableDynamicSubdivisionScale ? subdivisionScale : 0.0f;

            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
            static char buffer[128];
            sprintf(buffer, "Max Subdivision Level [1 : %d] ", maxSubdivisionLevel);
            ImGui::InputInt(buffer, &subdivisionLevel);
            ImGui::PopItemWidth();
            subdivisionLevel = subdivisionLevel < 1 ? 1 : subdivisionLevel;
            subdivisionLevel = subdivisionLevel > maxSubdivisionLevel ? maxSubdivisionLevel : subdivisionLevel;

            ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
            ImGui::InputInt("Mip Bias (if applicable)", &mipBias);
            ImGui::PopItemWidth();
            mipBias = mipBias < 0 ? 0 : mipBias;
            mipBias = mipBias > 15 ? 15 : mipBias;
            static bool enableCaching = bakeDesc.enableCache;

            if (isCpuBaker)
            {
                ImGui::PushItemWidth(ImGui::CalcItemWidth() * 0.33f);
                ImGui::InputInt("Mip Count (if applicable)", &mipCount);
                ImGui::PopItemWidth();
                int maxMipRange = OMM_MAX_MIP_NUM - mipBias;
                mipCount = mipCount < 1 ? 1 : mipCount;
                mipCount = mipCount > maxMipRange ? maxMipRange : mipCount;
            }

            bakeDesc.format = ommhelper::OmmFormats(ommFormatSelection);
            bakeDesc.filter = ommhelper::OmmBakeFilter(ommFilterSelection);
            bakeDesc.subdivisionLevel = subdivisionLevel;
            bakeDesc.mipBias = mipBias;
            bakeDesc.mipCount = mipCount;
            bakeDesc.type = ommhelper::OmmBakerType(ommBakerTypeSelection);
            bakeDesc.enableCache = enableCaching;

            bool isRebuildAvailable = IsRebuildAvailable(bakeDesc, m_OmmBakeDesc);

            static uint32_t frameId = 0;
            static ImU32 greyColor = ImGui::GetColorU32(ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
            static ImU32 greenColor = ImGui::GetColorU32(ImVec4(0.0f, 0.6f, 0.0f, 1.0f));
            bool forceRebuild = frameId == m_OmmBakeDesc.buildFrameId;
            {
                ImU32 buttonColor = isRebuildAvailable ? greenColor : greyColor;
                ImGui::PushStyleColor(ImGuiCol_::ImGuiCol_Button, buttonColor);
                if (ImGui::Button("Bake OMMs") || forceRebuild)
                {
                    m_OmmBakeDesc = bakeDesc;
                    RebuildOmmGeometry();
                }
                ImGui::PopStyleColor();

                ImGui::SameLine();
                ImGui::Checkbox("Use OMM Cache", &enableCaching);
            }
            ++frameId;
        }
    }
    ImGui::PopID();
}

SAMPLE_MAIN(Sample, 0);

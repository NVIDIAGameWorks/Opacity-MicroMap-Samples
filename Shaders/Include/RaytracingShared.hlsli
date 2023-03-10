struct PrimitiveData
{
    half2 uv0;
    half2 uv1;
    half2 uv2;
    half2 n0;

    half2 n1;
    half2 n2;
    half2 t0;
    half2 t1;

    half2 t2;
    half2 b0s_b1s;
    half2 b2s_worldToUvUnits;
    float curvature;
};

struct InstanceData
{
    uint basePrimitiveIndex;
    uint baseTextureIndex;
    uint unused1;
    uint unused2;

    // TODO: move to a separate buffer
    row_major float3x4 mWorldToWorldPrev;
};

NRI_RESOURCE( RaytracingAccelerationStructure, gWorldTlas, t, 0, 2 );
NRI_RESOURCE( RaytracingAccelerationStructure, gLightTlas, t, 1, 2 );
NRI_RESOURCE( StructuredBuffer<InstanceData>, gIn_InstanceData, t, 2, 2 );
NRI_RESOURCE( StructuredBuffer<PrimitiveData>, gIn_PrimitiveData, t, 3, 2 );
NRI_RESOURCE( Texture2D<float4>, gIn_Textures[], t, 4, 2 );

#define TEX_SAMPLER                         gLinearMipmapLinearSampler

#define FLAG_FIRST_BIT                      20 // this + number of CPU flags must be <= 24
#define INSTANCE_ID_MASK                    ( ( 1 << FLAG_FIRST_BIT ) - 1 )

// CPU flags
#define FLAG_OPAQUE_OR_ALPHA_OPAQUE         0x01
#define FLAG_TRANSPARENT                    0x02
#define FLAG_EMISSION                       0x04
#define FLAG_FORCED_EMISSION                0x08

// Local flags // TODO: all unused
#define FLAG_UNUSED1                        0x10
#define FLAG_UNUSED2                        0x20
#define FLAG_UNUSED3                        0x40
#define FLAG_ANYHIT_INVOKED                 0x80

#define GEOMETRY_ALL                        ( FLAG_OPAQUE_OR_ALPHA_OPAQUE | FLAG_EMISSION | FLAG_FORCED_EMISSION | FLAG_TRANSPARENT )
#define GEOMETRY_ONLY_EMISSIVE              ( FLAG_EMISSION | FLAG_FORCED_EMISSION )
#define GEOMETRY_ONLY_TRANSPARENT           ( FLAG_TRANSPARENT )
#define GEOMETRY_IGNORE_TRANSPARENT         ( FLAG_OPAQUE_OR_ALPHA_OPAQUE | FLAG_EMISSION | FLAG_FORCED_EMISSION )

//====================================================================================================================================
// GEOMETRY & MATERIAL PROPERTIES
//====================================================================================================================================

float3 _GetXoffset( float3 X, float3 offsetDirection )
{
    #if 1
        // Moves the ray origin further from surface to prevent self-intersections. Minimizes the distance for best results
        // ( taken from RT Gems "A Fast and Robust Method for Avoiding Self-Intersection" )
        int3 o = int3( offsetDirection * 256.0 );
        float3 a = asfloat( asint( X ) + ( X < 0.0 ? -o : o ) );
        float3 b = X + offsetDirection * ( 1.0 / 65536.0 );

        X = abs( X ) < ( 1.0 / 32.0 ) ? b : a;
    #endif

    // TODO: very accurate normal is needed to minimize offset
    return X;// + offsetDirection * ( 0.003 + length( X - gCameraOrigin ) * 0.0005 );
}

struct GeometryProps
{
    float3 X;
    float3 V;
    float4 T;
    float3 N;
    float2 uv;
    float mip;
    float tmin;
    float curvature;
    uint textureOffsetAndFlags;
    uint instanceIndex;

    float3 GetXoffset( )
    { return _GetXoffset( X, N ); }

    bool IsTransparent( )
    { return ( textureOffsetAndFlags & ( ( FLAG_TRANSPARENT ) << FLAG_FIRST_BIT ) ) != 0; }

    bool IsEmissive( )
    { return ( textureOffsetAndFlags & ( ( FLAG_EMISSION | FLAG_FORCED_EMISSION ) << FLAG_FIRST_BIT ) ) != 0; }

    bool IsForcedEmission( )
    { return ( textureOffsetAndFlags & ( ( FLAG_FORCED_EMISSION ) << FLAG_FIRST_BIT ) ) != 0; }

    uint GetBaseTexture( )
    { return textureOffsetAndFlags & INSTANCE_ID_MASK; }

    float3 GetForcedEmissionColor( )
    { return ( ( textureOffsetAndFlags >> 2 ) & 0x1 ) ? float3( 1.0, 0.0, 0.0 ) : float3( 0.0, 1.0, 0.0 ); }

    uint GetPackedMaterial( )
    { return asuint( uv.x ); }

    bool IsSky( )
    { return tmin == INF; }

    bool IsAnyHitInvoked()
    { return (textureOffsetAndFlags & ( ( FLAG_ANYHIT_INVOKED ) << FLAG_FIRST_BIT ) ) != 0; }
};

struct MaterialProps
{
    float3 Ldirect; // unshadowed
    float3 Lemi;
    float3 N;
    float3 baseColor;
    float roughness;
    float metalness;
    float curvature;
};

float2 GetConeAngleFromAngularRadius( float mip, float tanConeAngle )
{
    // In any case, we are limited by the output resolution
    tanConeAngle = max( tanConeAngle, gTanPixelAngularRadius );

    return float2( mip, tanConeAngle );
}

float2 GetConeAngleFromRoughness( float mip, float roughness )
{
    float coneAngle = tan( STL::ImportanceSampling::GetSpecularLobeHalfAngle( roughness ) ); // TODO:  * 0.33333?

    return GetConeAngleFromAngularRadius( mip, coneAngle );
}

/*
Returns:
    .x - for visibility (emission, shadow)
        We must avoid using lower mips because it can lead to significant increase in AHS invocations. Mips lower than 128x128 are skipped!
    .y - for sampling (normals...)
        Negative MIP bias is applied
    .z - for sharp sampling
        Negative MIP bias is applied (can be more negative...)
*/
float3 GetRealMip( uint textureIndex, float mip )
{
    float w, h;
    gIn_Textures[ textureIndex ].GetDimensions( w, h ); // TODO: if I only had it as a constant...

    // Taking into account real dimensions of the current texture
    float mipNum = log2( w );
    float realMip = mip + mipNum - MAX_MIP_LEVEL;

    float3 mips;
    mips.x = min( realMip, mipNum - 7.0 );
    mips.y = realMip + gMipBias * 0.5;
    mips.z = realMip + gMipBias;

    return max( mips, 0.0 );
}

MaterialProps GetMaterialProps( GeometryProps geometryProps )
{
    MaterialProps props = ( MaterialProps )0;

    float3 Csky = GetSkyIntensity( -geometryProps.V, gSunDirection, gTanSunAngularRadius );

    [branch]
    if( geometryProps.IsSky( ) )
    {
        props.Lemi = Csky;

        return props;
    }

    uint baseTexture = geometryProps.GetBaseTexture( );
    float3 mips = GetRealMip( baseTexture, geometryProps.mip );

    // Base color
    float4 color = gIn_Textures[ baseTexture ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.z );
    color.xyz *= geometryProps.IsTransparent( ) ? 1.0 : STL::Math::PositiveRcp( color.w ); // Correct handling of BC1 with pre-multiplied alpha
    float3 baseColor = saturate( color.xyz );

    // Roughness and metalness
    float3 materialProps = gIn_Textures[ baseTexture + 1 ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.z ).xyz;
    float roughness = materialProps.y;
    float metalness = materialProps.z;

    // Normal
    float2 packedNormal = gIn_Textures[ baseTexture + 2 ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.y ).xy;
    float3 N = gUseNormalMap ? STL::Geometry::TransformLocalNormal( packedNormal, geometryProps.T, geometryProps.N ) : geometryProps.N;

    // Estimate curvature
    float curvature = length( STL::Geometry::UnpackLocalNormal( packedNormal ).xy ) * float( gUseNormalMap );

    // Emission
    float3 Lemi = gIn_Textures[ baseTexture + 3 ].SampleLevel( TEX_SAMPLER, geometryProps.uv, mips.x ).xyz;
    Lemi *= ( baseColor + 0.01 ) / ( max( baseColor, max( baseColor, baseColor ) ) + 0.01 );
    Lemi *= gEmissionIntensity * float( geometryProps.IsEmissive( ) );

    // Override material
    [flatten]
    if( gForcedMaterial == MAT_GYPSUM )
    {
        roughness = 1.0;
        baseColor = 0.5;
        metalness = 0.0;
    }
    else if( gForcedMaterial == MAT_COBALT )
    {
        roughness = pow( saturate( baseColor.x * baseColor.y * baseColor.z ), 0.33333 );
        baseColor = float3( 0.672411, 0.637331, 0.585456 );
        metalness = 1.0;
    }

    metalness = gMetalnessOverride == 0.0 ? metalness : gMetalnessOverride;
    roughness = gRoughnessOverride == 0.0 ? roughness : gRoughnessOverride;

    // Force emission
    if( geometryProps.IsForcedEmission( ) )
    {
        Lemi = geometryProps.GetForcedEmissionColor( );
        baseColor = 0.0;
        roughness = 1.0;
        metalness = 0.0;
    }

    // Direct lighting ( no shadow )
    float3 Ldirect = 0;
    float NoL = saturate( dot( geometryProps.N, gSunDirection ) );
    float shadow = STL::Math::SmoothStep( 0.03, 0.1, NoL );

    [branch]
    if( shadow != 0.0 )
    {
        float3 Csun = GetSunIntensity( gSunDirection, gSunDirection, gTanSunAngularRadius );

        // Pseudo sky importance sampling
        float3 Cimp = lerp( Csky, Csun, STL::Math::SmoothStep( 0.0, 0.2, roughness ) );
        Cimp *= STL::Math::SmoothStep( -0.01, 0.05, gSunDirection.z );

        float3 albedo, Rf0;
        STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( baseColor.xyz, metalness, albedo, Rf0 );

        #if( USE_SIMPLEX_LIGHTING_MODEL == 1 )
            // Very simple "diffuse-like" model
            float m = roughness * roughness;
            float3 C = albedo * Csun + Rf0 * m * Cimp;
            float NoL = dot( geometryProps.N, gSunDirection );
            float Kdiff = NoL / STL::Math::Pi( 1.0 );

            Ldirect = Kdiff * C;
        #else
            float3 Cdiff, Cspec;
            STL::BRDF::DirectLighting( N, gSunDirection, geometryProps.V, Rf0, roughness, Cdiff, Cspec );

            Ldirect = Cdiff * albedo * Csun + Cspec * Cimp;
        #endif

        Ldirect *= shadow;
    }

    // Output
    props.Ldirect = Ldirect;
    props.Lemi = Lemi;
    props.N = N;
    props.baseColor = baseColor;
    props.roughness = roughness;
    props.metalness = metalness;
    props.curvature = geometryProps.curvature + curvature;

    return props;
}

//====================================================================================================================================
// MISC
//====================================================================================================================================

float3 GetAmbientBRDF( GeometryProps geometryProps, MaterialProps materialProps, bool approximate = false )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float3 Fenv = Rf0;
    if( !approximate )
    {
        float NoV = abs( dot( materialProps.N, geometryProps.V ) );
        Fenv = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );
    }

    float3 ambBRDF = albedo * ( 1.0 - Fenv ) + Fenv;
    ambBRDF *= float( !geometryProps.IsSky() );

    return ambBRDF;
}

float EstimateDiffuseProbability( GeometryProps geometryProps, MaterialProps materialProps, bool useMagicBoost = false )
{
    float3 albedo, Rf0;
    STL::BRDF::ConvertBaseColorMetalnessToAlbedoRf0( materialProps.baseColor, materialProps.metalness, albedo, Rf0 );

    float NoV = abs( dot( materialProps.N, geometryProps.V ) );
    float3 Fenv = STL::BRDF::EnvironmentTerm_Ross( Rf0, NoV, materialProps.roughness );

    float lumSpec = STL::Color::Luminance( Fenv );
    float lumDiff = STL::Color::Luminance( albedo * ( 1.0 - Fenv ) );

    // Boost diffuse if roughness is high
    if( useMagicBoost )
    {
        float smc = GetSpecMagicCurve( materialProps.roughness );
        lumDiff = lerp( lumDiff, 1.0, smc );
        lumSpec = lerp( lumSpec, 0.0, smc );
    }

    float diffProb = lumDiff / ( lumDiff + lumSpec + 1e-6 );

    return diffProb < 0.005 ? 0.0 : diffProb;
}

//====================================================================================================================================
// TRACER
//====================================================================================================================================

#define CheckNonOpaqueTriangle( rayQuery, mipAndCone ) \
    { \
        /* Instance */ \
        uint instanceIndex = rayQuery.CandidateInstanceID( ) & INSTANCE_ID_MASK; \
        InstanceData instanceData = gIn_InstanceData[ instanceIndex ]; \
        \
        float3x3 mObjectToWorld = (float3x3)rayQuery.CandidateObjectToWorld3x4( ); \
        \
        /* Primitive */ \
        uint primitiveIndex = instanceData.basePrimitiveIndex + rayQuery.CandidatePrimitiveIndex( ); \
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ]; \
        \
        /* Barycentrics */ \
        float3 barycentrics; \
        barycentrics.yz = rayQuery.CandidateTriangleBarycentrics( ); \
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z; \
        \
        /* Uv */ \
        float2 uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2; \
        \
        [branch]\
        if(gAhsDynamicMip) \
        { \
        /* Normal */ \
        float3 n0 = STL::Packing::DecodeUnitVector( primitiveData.n0, true ); \
        float3 n1 = STL::Packing::DecodeUnitVector( primitiveData.n1, true ); \
        float3 n2 = STL::Packing::DecodeUnitVector( primitiveData.n2, true ); \
        \
        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2; \
        N = STL::Geometry::RotateVector( mObjectToWorld, N ); \
        N = normalize( N ); \
        N = rayQuery.CandidateTriangleFrontFace( ) ? -N : N; \
        \
        /* Handling object scale embedded into the transformation matrix (assuming uniform scale) */ \
        float invObjectScale = STL::Math::Rsqrt( STL::Math::LengthSquared( mObjectToWorld[ 0 ] ) ); \
        \
        /* Mip level (TODO: doesn't take into account integrated AO / SO - i.e. diffuse = lowest mip, but what if we see the sky through a tiny hole?) */ \
        float NoR = abs( dot( rayQuery.WorldRayDirection( ), N ) ); \
        float a = rayQuery.CandidateTriangleRayT( ); \
        a *= mipAndCone.y; \
        a *= STL::Math::PositiveRcp( NoR ); \
        a *= primitiveData.b2s_worldToUvUnits.y * invObjectScale; \
        \
        float mip = log2( a ); \
        mip += MAX_MIP_LEVEL; \
        mip = max( mip, 0.0 ); \
        mip += mipAndCone.x; \
        \
        /* Alpha test */ \
        uint baseTexture = instanceData.baseTextureIndex + 0; \
        float3 mips = GetRealMip( baseTexture, mip ); \
        float alpha = gIn_Textures[ baseTexture ].SampleLevel( TEX_SAMPLER, uv, mips.x ).w; \
        \
        if( alpha > 0.5 ) \
            rayQuery.CommitNonOpaqueTriangleHit( ); \
        } \
        else \
        { \
            uint baseTexture = instanceData.baseTextureIndex + 0; \
            float alpha = gIn_Textures[ baseTexture ].SampleLevel( TEX_SAMPLER, uv, 0 ).w; \
            if( alpha > 0.5 ) \
                rayQuery.CommitNonOpaqueTriangleHit( ); \
        } \
    }

bool CastVisibilityRay_AnyHit( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    return rayQuery.CommittedStatus( ) == COMMITTED_NOTHING;
}

float CastVisibilityRay_ClosestHit( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    while( rayQuery.Proceed( ) )
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );

    return rayQuery.CommittedStatus( ) == COMMITTED_NOTHING ? INF : rayQuery.CommittedRayT( );
}

GeometryProps CastRay( float3 origin, float3 direction, float Tmin, float Tmax, float2 mipAndCone, RaytracingAccelerationStructure accelerationStructure, uint instanceInclusionMask, uint rayFlags )
{
    RayDesc rayDesc;
    rayDesc.Origin = origin;
    rayDesc.Direction = direction;
    rayDesc.TMin = Tmin;
    rayDesc.TMax = Tmax;

    RayQuery< RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES > rayQuery;
    rayQuery.TraceRayInline( accelerationStructure, rayFlags, instanceInclusionMask, rayDesc );

    bool anyHitInvoked = false;
    while( rayQuery.Proceed( ) )
    {
        anyHitInvoked = true;
        CheckNonOpaqueTriangle( rayQuery, mipAndCone );
    }

    // TODO: reuse data if committed == candidate (use T to check)
    GeometryProps props = ( GeometryProps )0;
    props.mip = mipAndCone.x;

    if( rayQuery.CommittedStatus( ) == COMMITTED_NOTHING )
        props.tmin = INF;
    else
    {
        props.tmin = rayQuery.CommittedRayT( );

        // Instance
        uint instanceIndex = rayQuery.CommittedInstanceID( ) & INSTANCE_ID_MASK;
        props.instanceIndex = instanceIndex;

        InstanceData instanceData = gIn_InstanceData[ instanceIndex ];

        float3x3 mObjectToWorld = (float3x3)rayQuery.CommittedObjectToWorld3x4( ); // TODO: 4x3?

        // Primitive
        uint primitiveIndex = instanceData.basePrimitiveIndex + rayQuery.CommittedPrimitiveIndex( );
        PrimitiveData primitiveData = gIn_PrimitiveData[ primitiveIndex ];

        // Texture offset and flags
        uint flags = rayQuery.CommittedInstanceID( ) & ~INSTANCE_ID_MASK;
        props.textureOffsetAndFlags = instanceData.baseTextureIndex | flags;
        if (anyHitInvoked)
        {
            props.textureOffsetAndFlags |= (FLAG_ANYHIT_INVOKED << FLAG_FIRST_BIT);
        }

        // Barycentrics
        float3 barycentrics;
        barycentrics.yz = rayQuery.CommittedTriangleBarycentrics( );
        barycentrics.x = 1.0 - barycentrics.y - barycentrics.z;

        // Normal
        float3 n0 = STL::Packing::DecodeUnitVector( primitiveData.n0, true );
        float3 n1 = STL::Packing::DecodeUnitVector( primitiveData.n1, true );
        float3 n2 = STL::Packing::DecodeUnitVector( primitiveData.n2, true );

        float3 N = barycentrics.x * n0 + barycentrics.y * n1 + barycentrics.z * n2;
        N = STL::Geometry::RotateVector( mObjectToWorld, N );
        N = normalize( N );
        props.N = rayQuery.CommittedTriangleFrontFace( ) ? -N : N;

        // Uv
        props.uv = barycentrics.x * primitiveData.uv0 + barycentrics.y * primitiveData.uv1 + barycentrics.z * primitiveData.uv2;

        // Tangent
        float4 t0 = float4( STL::Packing::DecodeUnitVector( primitiveData.t0, true ), primitiveData.b0s_b1s.x );
        float4 t1 = float4( STL::Packing::DecodeUnitVector( primitiveData.t1, true ), primitiveData.b0s_b1s.y );
        float4 t2 = float4( STL::Packing::DecodeUnitVector( primitiveData.t2, true ), primitiveData.b2s_worldToUvUnits.x );

        float4 T = barycentrics.x * t0 + barycentrics.y * t1 + barycentrics.z * t2;
        T.xyz = STL::Geometry::RotateVector( mObjectToWorld, T.xyz );
        T.xyz = normalize( T.xyz );
        props.T = T;

        // Curvature
        props.curvature = primitiveData.curvature;

        // Handling object scale embedded into the transformation matrix (assuming uniform scale)
        float invObjectScale = STL::Math::Rsqrt( STL::Math::LengthSquared( mObjectToWorld[ 0 ] ) );

        // Mip level (TODO: doesn't take into account integrated AO / SO - i.e. diffuse = lowest mip, but what if we see the sky through a tiny hole?)
        float NoR = abs( dot( direction, props.N ) );
        float a = props.tmin * mipAndCone.y;
        a *= STL::Math::PositiveRcp( NoR );
        a *= primitiveData.b2s_worldToUvUnits.y * invObjectScale;

        float mip = log2( a );
        mip += MAX_MIP_LEVEL;
        mip = max( mip, 0.0 );
        props.mip += mip;
    }

    props.X = origin + direction * props.tmin;
    props.V = -direction;

    return props;
}

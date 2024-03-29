/*
Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "Include/Shared.hlsli"

NRI_RESOURCE( Texture2D<float3>, gIn_Image, t, 0, 1 );
NRI_RESOURCE( Texture2D<float4>, gIn_Validation, t, 1, 1 );

NRI_RESOURCE( RWTexture2D<float3>, gOut_Image, u, 0, 1 );

[numthreads( 16, 16, 1 )]
void main( uint2 pixelPos : SV_DispatchThreadId )
{
    float2 pixelUv = float2( pixelPos + 0.5 ) * gInvWindowSize;

    // Upsampling
    float3 upsampled = BicubicFilterNoCorners( gIn_Image, gLinearSampler, pixelUv * gOutputSize, gInvOutputSize, 0.66 ).xyz;
    #if( NRD_MODE == OCCLUSION || NRD_MODE == DIRECTIONAL_OCCLUSION )
        upsampled = upsampled.xxx;
    #endif

    // Tonemap
    if( gOnScreen == SHOW_FINAL )
        upsampled = STL::Color::HdrToLinear_Uncharted( upsampled );

    // Conversion
    if( gOnScreen == SHOW_FINAL || gOnScreen == SHOW_BASE_COLOR )
        upsampled = STL::Color::LinearToSrgb( upsampled );

    // Validation layer
    if( gValidation )
    {
        float4 validation = gIn_Validation.SampleLevel( gLinearSampler, pixelUv, 0 );
        upsampled = lerp( upsampled, validation.xyz, validation.w );
    }

    gOut_Image[ pixelPos ] = upsampled;
}

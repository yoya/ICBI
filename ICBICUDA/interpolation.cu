

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "interpolation.h"


////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
float Abs(float x){
    return (x > 0) ? x : (-x);
}

float Max(float x, float y){
    return (x > y) ? x : y;
}

float Min(float x, float y){
    return (x < y) ? x : y;
}

int iDivUp(int a, int b){
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__device__ float lerpf(float a, float b, float c){
    return a + (b - a) * c;
}

__device__ float vecLen(float4 a, float4 b){
    return (
        (b.x - a.x) * (b.x - a.x) +
        (b.y - a.y) * (b.y - a.y) +
        (b.z - a.z) * (b.z - a.z)
    );
}

__device__ TColor make_color(float r, float g, float b, float a){
    return
        ((int)(a * 255.0f) << 24) |
        ((int)(b * 255.0f) << 16) |
        ((int)(g * 255.0f) <<  8) |
        ((int)(r * 255.0f) <<  0);
}

__device__ float4 TColorToFloat4(TColor c)
{
    float4 rgba;
    rgba.x = (c & 0xff) / 255.0f;
    rgba.y = ((c>>8) & 0xff) / 255.0f;
    rgba.z = ((c>>16) & 0xff) / 255.0f;
    rgba.w = ((c>>24) & 0xff) / 255.0f;
    return rgba;
}

__device__ float TColorToGray(TColor c)
{
    float4 rgba;
    float val;
    rgba.x = (c & 0xff) / 255.0f;
    rgba.y = ((c>>8) & 0xff) / 255.0f;
    rgba.z = ((c>>16) & 0xff) / 255.0f;
    rgba.w = ((c>>24) & 0xff) / 255.0f;
    val = (rgba.x + rgba.y + rgba.z)/3.0;
    return val;
}


////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////
//Texture reference and channel descriptor for image texture
texture<uchar4, 2, cudaReadModeNormalizedFloat> texImage;
cudaChannelFormatDesc uchar4tex = cudaCreateChannelDesc<uchar4>();

//CUDA array descriptor
cudaArray *a_Src;

////////////////////////////////////////////////////////////////////////////////
// kernels
////////////////////////////////////////////////////////////////////////////////
#include "interpolation_ddt_kernel.cu"
#include "interpolation_fcbi_kernel.cu"
#include "interpolation_icbi_kernel.cu"
#include "interpolation_copy_kernel.cu"

extern "C"
cudaError_t CUDA_Bind2TextureArray()
{
    return cudaBindTextureToArray(texImage, a_Src);
}

extern "C"
cudaError_t CUDA_UnbindTexture()
{
    return cudaUnbindTexture(texImage);
}

extern "C" 
cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH)
{
    cudaError_t error;

    error = cudaMallocArray(&a_Src, &uchar4tex, imageW, imageH);

    error = cudaMemcpyToArray(a_Src, 0, 0,
                              *h_Src, imageW * imageH * sizeof(uchar4),
                              cudaMemcpyHostToDevice
                              );
    return error;
}


extern "C"
cudaError_t CUDA_FreeArray()
{
    return cudaFreeArray(a_Src);    
}

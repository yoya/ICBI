/* ICBI Cuda implementation (c) 2010 andrea giachetti     
   License     : GNU GENERAL PUBLIC LICENSE v.2
   http://www.gnu.org/copyleft/gpl.html
   version 0.1   
   please report bugs/problems/tests/applications to andrea.giachetti@univr.it
   and cite related papers if used in published scientific work
*/

#define FACT 2

typedef unsigned int TColor;



///////////////////////////////////////////////////////////////////////////////
// Filter configuration
////////////////////////////////////////////////////////////////////////////////


#define BLOCKDIM_X 16
#define BLOCKDIM_Y 16

#define MAX(a,b) ((a < b) ? b : a)
#define MIN(a,b) ((a < b) ? a : b)

// functions to load images
extern "C" void LoadBMPFile(uchar4 **dst, int *width, int *height, const char *name);


// CUDA wrapper functions for allocation/freeing texture arrays
extern "C" cudaError_t CUDA_Bind2TextureArray();
extern "C" cudaError_t CUDA_UnbindTexture();
extern "C" cudaError_t CUDA_MallocArray(uchar4 **h_Src, int imageW, int imageH);
extern "C" cudaError_t CUDA_FreeArray();

// CUDA kernel functions
extern "C" void cuda_COPY( TColor *src, TColor *d_dst, int imageW, int imageH);

extern "C" void cuda_DDT( TColor *src, TColor *d_dst, int imageW, int imageH);

extern "C" void cuda_FCBI( TColor *src, TColor *d_dst, int imageW, int imageH);

extern "C" void cuda_ICBI( TColor *src, TColor *d_dst, float* d1, float* d2, float* d3, float* c1, float* c2, float* gray, int imageW, int imageH);



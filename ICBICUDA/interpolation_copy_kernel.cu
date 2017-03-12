/* ICBI Cuda implementation (c) 2010 andrea giachetti     
   License     : GNU GENERAL PUBLIC LICENSE v.2
   http://www.gnu.org/copyleft/gpl.html
   version 0.1   
   please report bugs/problems/tests/applications to andrea.giachetti@univr.it
   and cite related papers if used in published scientific work
*/

__global__ void FillCopy(
    TColor *src,
    TColor *dst,
    int origW,
    int origH
){
   const int ix = blockDim.x * blockIdx.x + threadIdx.x;
   const int iy = blockDim.y * blockIdx.y + threadIdx.y;
   const int imageW = 2*origW-1;
   const int imageH = 2*origH-1;
  
  if( ix < origW && iy < origH ){
    dst[imageW * 2*iy + 2*ix] = src[origW * iy + ix];
    dst[imageW * (2*iy+1) + 2*ix] = src[origW * iy + ix];
    dst[imageW * (2*iy+1) + 2*ix+1] = src[origW * iy + ix];
    dst[imageW * 2*iy + 2*ix+1] = src[origW * iy + ix];
  }    

}


extern "C" void
cuda_COPY(TColor *src, TColor *d_dst, int origW, int origH)
{
 
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(origW, BLOCKDIM_X), iDivUp(origH, BLOCKDIM_Y));
    FillCopy<<<grid, threads>>>(src, d_dst, origW, origH);	
	
}

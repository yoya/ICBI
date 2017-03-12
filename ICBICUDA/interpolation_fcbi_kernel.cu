/* ICBI Cuda implementation (c) 2010 andrea giachetti     
   License     : GNU GENERAL PUBLIC LICENSE v.2
   http://www.gnu.org/copyleft/gpl.html
   version 0.1   
   please report bugs/problems/tests/applications to andrea.giachetti@univr.it
   and cite related papers if used in published scientific work
*/

__global__ void FCBIStep1(
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

  }    

}

__global__ void FCBIStep2(
    TColor *dst,
    int origW,
    int origH
){
    const int tix = blockDim.x * blockIdx.x + threadIdx.x;
    const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    const int imageW = 2*origW-1;
    const int imageH = 2*origH-1;
    float d1, d2;
    float4 p1, p2;
	int ix, iy;

    if( tix < origW-1 && tiy < origH-3 && tix >1 && tiy>1){	
 	ix=2*tix+1;
	iy=2*tiy+1;

	float4 fmm = TColorToFloat4(dst[imageW * (iy-1) + ix-1]); 
	float4 fpp = TColorToFloat4(dst[imageW * (iy+1) + ix+1]);  
	float4 fmp = TColorToFloat4(dst[imageW * (iy+1) + ix-1]); 
	float4 fpm = TColorToFloat4(dst[imageW * (iy-1) + ix+1]);
	float4 fmm13 = TColorToFloat4(dst[imageW * (iy-1) + ix-3]); 
	float4 fpp13 = TColorToFloat4(dst[imageW * (iy+1) + ix+3]);  
	float4 fmm31 = TColorToFloat4(dst[imageW * (iy-3) + ix-1]); 
	float4 fpp31 = TColorToFloat4(dst[imageW * (iy+3) + ix+1]);
	float4 fmp13 = TColorToFloat4(dst[imageW * (iy-1) + ix+3]); 
	float4 fpm13 = TColorToFloat4(dst[imageW * (iy+1) + ix-3]);  
	float4 fmp31 = TColorToFloat4(dst[imageW * (iy-3) + ix+1]); 
	float4 fpm31 = TColorToFloat4(dst[imageW * (iy+3) + ix-1]);
	
	p1.x = 0.5*(fmm.x+fpp.x);
 	p2.x = 0.5*(fpm.x+fmp.x);
	p1.y = 0.5*(fmm.y+fpp.y);
 	p2.y = 0.5*(fpm.y+fmp.y);
	p1.z = 0.5*(fmm.z+fpp.z);
 	p2.z = 0.5*(fpm.z+fmp.z);
	
	d1 = fmm13.x+fmm13.y+fmm13.z+
	     fpp13.x+fpp13.y+fpp13.z+
	     fmm31.x+fmm31.y+fmm31.z+
	     fpp31.x+fpp31.y+fpp31.z+
	     - 6 * (p1.x+p1.y+p1.z) + 2*(p2.x+p2.y+p2.z);
	     
	 d2 = fmp13.x+fmp13.y+fmp13.z+
	     fpm13.x+fpm13.y+fpm13.z+
	     fmp31.x+fmp31.y+fmp31.z+
	     fpm31.x+fpm31.y+fpm31.z+
	     2* (p1.x+p1.y+p1.z) - 6*(p2.x+p2.y+p2.z);



	if(d1*d1 > d2*d2)
	  dst[imageW * iy + ix] =  make_color(p1.x,p1.y, p1.z, 0);
	 else 
	   dst[imageW * iy + ix] = make_color(p2.x,p2.y, p2.z, 0);
		
    }  
}


__global__ void FCBIStep3(
    TColor *dst,
    int origW,
    int origH
){
    const int tix = blockDim.x * blockIdx.x + threadIdx.x;
    const int tiy = blockDim.y * blockIdx.y + threadIdx.y;

    const int imageW = 2*origW-1;
    const int imageH = 2*origH-1;
    int ix, iy;

    float d1, d2;
    float4 p1, p2;

    if( tix < origW-1 && tiy < origH-1 && tix >1 && tiy>1){	
 	ix=2*tix;
	iy=2*tiy+1;
   	for(int k=0;k<2;k++){

	float4 fmm = TColorToFloat4(dst[imageW * (iy) + ix-1]); 
	float4 fpp = TColorToFloat4(dst[imageW * (iy) + ix+1]);  
	float4 fmp = TColorToFloat4(dst[imageW * (iy+1) + ix]); 
	float4 fpm = TColorToFloat4(dst[imageW * (iy-1) + ix]);
	float4 fmm13 = TColorToFloat4(dst[imageW * (iy-1) + ix-2]); 
	float4 fpp13 = TColorToFloat4(dst[imageW * (iy-1) + ix+2]);  
	float4 fmm31 = TColorToFloat4(dst[imageW * (iy+1) + ix-2]); 
	float4 fpp31 = TColorToFloat4(dst[imageW * (iy+1) + ix+2]);
	float4 fmp13 = TColorToFloat4(dst[imageW * (iy+2) + ix-1]); 
	float4 fpm13 = TColorToFloat4(dst[imageW * (iy-2) + ix-1]);  
	float4 fmp31 = TColorToFloat4(dst[imageW * (iy-2) + ix+1]); 
	float4 fpm31 = TColorToFloat4(dst[imageW * (iy+2) + ix+1]);
	
	p1.x = 0.5*(fmm.x+fpp.x);
 	p2.x = 0.5*(fpm.x+fmp.x);
	p1.y = 0.5*(fmm.y+fpp.y);
 	p2.y = 0.5*(fpm.y+fmp.y);
	p1.z = 0.5*(fmm.z+fpp.z);
 	p2.z = 0.5*(fpm.z+fmp.z);
	
	d1 = fmm13.x+fmm13.y+fmm13.z+
	     fpp13.x+fpp13.y+fpp13.z+
	     fmm31.x+fmm31.y+fmm31.z+
	     fpp31.x+fpp31.y+fpp31.z+
	     -6* (p1.x+p1.y+p1.z) + 2*(p2.x+p2.y+p2.z);
	     
	 d2 = fmp13.x+fmp13.y+fmp13.z+
	     fpm13.x+fpm13.y+fpm13.z+
	     fmp31.x+fmp31.y+fmp31.z+
	     fpm31.x+fpm31.y+fpm31.z+
	     2 * (p1.x+p1.y+p1.z) - 6 *(p2.x+p2.y+p2.z);



	if(d1*d1 > d2*d2)
	  dst[imageW * iy + ix] =  make_color(p1.x,p1.y, p1.z, 0);
	 else 
	   dst[imageW * iy + ix] = make_color(p2.x,p2.y, p2.z, 0);
			
	ix=2*tix+1;
	iy=2*tiy;
    }   
    }

  }



extern "C" void
cuda_FCBI(TColor *src, TColor *d_dst, int origW, int origH)
{
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(origW, BLOCKDIM_X), iDivUp(origH, BLOCKDIM_Y));
     
    FCBIStep1<<<grid, threads>>>(src, d_dst, origW, origH);
    FCBIStep2<<<grid, threads>>>(d_dst, origW, origH);
    FCBIStep3<<<grid, threads>>>(d_dst, origW, origH);	
	
}

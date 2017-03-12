__global__ void Step1(
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

__global__ void Step2(
    TColor *dst,
    int origW,
    int origH
){
    const int tix = blockDim.x * blockIdx.x + threadIdx.x;
    const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    const int imageW = 2*origW-1;
    const int imageH = 2*origH-1;
    float d1, d2;
    int ix, iy;

    if( tix < origW && tiy < origH){
	ix=2*tix+1;
	iy=2*tiy+1;

	float4 f1 = TColorToFloat4(dst[imageW * (iy+1) + ix+1]);
	float4 f2 = TColorToFloat4(dst[imageW * (iy-1) + ix+1]);	
	float4 f3 = TColorToFloat4(dst[imageW * (iy-1) + ix-1]);	
	float4 f4 = TColorToFloat4(dst[imageW * (iy+1) + ix-1]);	
	d1 = f1.x+f1.y+f1.z-f3.x-f3.y-f3.z;
	d2 = f2.x+f2.y+f2.z-f4.x-f4.y-f4.z;		
	if(abs(d1)>abs(d2))
	 dst[imageW * iy + ix] =  make_color(.5*(f2.x+f4.x),.5*(f2.y+f4.y), .5*(f2.z+f4.z), 0);
	 else 
	   dst[imageW * iy + ix] =  make_color(.5*(f1.x+f3.x),.5*(f1.y+f3.y), .5*(f1.z+f3.z), 0);		

    }   

}




__global__ void Step3(
    TColor *dst,
    int origW,
    int origH
){
    const int tix = blockDim.x * blockIdx.x + threadIdx.x;
    const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
    const int imageW = 2*origW-1;
    const int imageH = 2*origH-1;
    float d1, d2;
    int ix,iy;
       if( tix > 1 && tiy > 1 && tix < imageW-1 && tiy < imageH-1 ){
	ix=2*tix;
	iy=2*tiy+1;
   	for(int k=0;k<2;k++){
	float4 f1 = TColorToFloat4(dst[imageW * iy + ix+1]);	
	float4 f2 = TColorToFloat4(dst[imageW * iy + ix-1]);
	float4 f3 = TColorToFloat4(dst[imageW * (iy+1) + ix]);
	float4 f4 = TColorToFloat4(dst[imageW * (iy-1) + ix]);

	d1 = f1.x+f1.y+f1.z-f2.x-f2.y-f2.z;
	d2 = f3.x+f3.y+f3.z-f4.x-f4.y-f4.z;		
	if(abs(d1)>abs(d2))
	 dst[imageW * iy + ix] = make_color(.5*(f3.x+f4.x),.5*(f3.y+f4.y), .5*(f3.z+f4.z), 0);
	 else 
	  dst[imageW * iy + ix] = make_color(.5*(f1.x+f2.x),.5*(f1.y+f2.y), .5*(f1.z+f2.z), 0);
	ix=2*tix+1;
	iy=2*tiy;		
	}
    }
}



extern "C" void
cuda_DDT(TColor *src, TColor *d_dst, int origW, int origH)
{
 
    dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
    dim3 grid(iDivUp(origW, BLOCKDIM_X), iDivUp(origH, BLOCKDIM_Y));
    const int imageW = 2*origW-1;
    const int imageH = 2*origH-1; 
    Step1<<<grid, threads>>>(src, d_dst, origW, origH);
    Step2<<<grid, threads>>>(d_dst, origW, origH);
    Step3<<<grid, threads>>>(d_dst, origW, origH);	
	
}

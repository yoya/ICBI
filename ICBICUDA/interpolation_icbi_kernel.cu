/* ICBI Cuda implementation (c) 2010 andrea giachetti     
   License     : GNU GENERAL PUBLIC LICENSE v.2
   http://www.gnu.org/copyleft/gpl.html
   version 0.1   
   please report bugs/problems/tests/applications to andrea.giachetti@univr.it
   and cite related papers if used in published scientific work
*/

__global__ void icbiInit(
			 TColor *src,
			 TColor *dst,
			 int origW,
			 int origH
			 ){
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  
  const int imageW = 2*origW-1;
  const int imageH = 2*origH-1;
   
  if( ix < (origW) && iy < origH)
     {
    dst[imageW * 2*iy + 2*ix] = src[origW * iy + ix];

	 }
}


__global__ void icbiFill1(
			  TColor *dst,
			  int origW,
			  int origH
			  ){
  const int tix = blockDim.x * blockIdx.x + threadIdx.x;
  const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const int imageW=origW*2-1;
  const int imageH=origH*2-1;
  int ix, iy;
  float d01, d02;
  float4 p1, p2;

  if( tix < origW-2 && tiy < origH-2 && tix >1 && tiy>1){	
      ix = tix*2+1;
      iy = tiy*2+1;
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

      d01 = fmm13.x+fmm13.y+fmm13.z+
	fpp13.x+fpp13.y+fpp13.z+
	fmm31.x+fmm31.y+fmm31.z+
	fpp31.x+fpp31.y+fpp31.z+
	- 6 * (p1.x+p1.y+p1.z) + 2*(p2.x+p2.y+p2.z);
	     
      d02 = fmp13.x+fmp13.y+fmp13.z+
	fpm13.x+fpm13.y+fpm13.z+
	fmp31.x+fmp31.y+fmp31.z+
	fpm31.x+fpm31.y+fpm31.z+
	2* (p1.x+p1.y+p1.z) - 6*(p2.x+p2.y+p2.z);



      if(abs(d01) > abs(d02))
	dst[imageW * iy + ix] =  make_color(p1.x,p1.y, p1.z, 0);
      else 
	dst[imageW * iy + ix] =  make_color(p2.x,p2.y, p2.z, 0);	

  }

}

__global__ void icbiDer1(
			 TColor *dst,
			 int origW,
			 int origH,
			 float* d1,
			 float* d2,
			 float* d3,
			 float* c1,
			 float* c2,
			 float* gray
			 ){
  const int tix = blockDim.x * blockIdx.x + threadIdx.x;
  const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const int imageW=origW*2-1;
  const int imageH=origH*2-1;
  int ix, iy;

  if( tix < origW-1 && tiy < origH-2 && tix > 1 && tiy > 1){
     ix=2*tix;
     iy=2*tiy;
    
   for(int k=0;k<2;k++){
     gray[imageW *iy+ix] = TColorToGray(dst[imageW * (iy) + ix]);
		
       
      d1[imageW *iy+ix] =  TColorToGray(dst[imageW * (iy-1) + ix-1]) +
	TColorToGray(dst[imageW * (iy+1) + ix+1]) 
	- 2* TColorToGray(dst[imageW * (iy) + ix]);
      d2[imageW *iy+ix] =  TColorToGray(dst[imageW * (iy+1) + ix-1]) +
	TColorToGray(dst[imageW * (iy-1) + ix+1]) 
	- 2* TColorToGray(dst[imageW * (iy) + ix]);

      d3[imageW *iy+ix] =  0.5*(TColorToGray(dst[imageW * (iy-2) + ix]) 
				- TColorToGray(dst[imageW * (iy) + ix+2]) 
				- TColorToGray(dst[imageW * (iy) + ix-2])
				+ TColorToGray(dst[imageW * (iy+2) + ix]));

      c1[imageW *iy+ix] =  0.5* ( TColorToGray(dst[imageW * (iy-1) + ix-1]) -
				  TColorToGray(dst[imageW * (iy+1) + ix+1]));

      c2[imageW *iy+ix] =   0.5* (  TColorToGray(dst[imageW * (iy-1) + ix+1]) -
				    TColorToGray(dst[imageW * (iy+1) + ix-1]));

     ix=2*tix+1;
     iy=2*tiy+1;
  }
}

}


__global__ void icbiIter1(
			  TColor *dst,
			  int origW,
			  int origH,
			  float step,
			  float* d1,
			  float* d2,
			  float* d3,
			  float* c1,
			  float* c2,
			  float* gray
			  ){
  const int tix = blockDim.x * blockIdx.x + threadIdx.x;
  const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const int imageW=origW*2-1;
  const int imageH=origH*2-1;
  int ix, iy;

 
  if( tix < origW-1 && tiy < origH-1 && tix > 1 && tiy > 1 ){
  ix = 2*tix+1;
  iy = 2*tiy+1;
  float4 p1;
  float en;
  float ea;
  float es;
  float eiso;
    
  float w1=1, w2=1, w3=1, w4=1;
  const float alpha=1.0, beta=-1.0, gamma=3.0;


 
      if(  abs(gray[imageW * (iy) + ix] -
	       gray[imageW * (iy+1) + ix+1]) > 0.2)
	w1=0;

      if(  abs(gray[imageW * (iy) + ix] -
	       gray[imageW * (iy-1) + ix-1]) > 0.2)
	w2=0;

      if(  abs(gray[imageW * (iy) + ix] -
	       gray[imageW * (iy-1) + ix+1]) > 0.2)
	w3=0;
      if(  abs(gray[imageW * (iy) + ix] -
	       gray[imageW * (iy+1) + ix-1]) > 0.2)
	w4=0;


      en = alpha*( w1*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix+1]) + w2*abs(d1[iy*imageW+ix]- d1[(iy-1)*imageW+ix-1])) +
     alpha*( w3*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix+1]) + w4*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix-1])) +
     alpha*( w1*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix+1]) + w2*abs(d2[iy*imageW+ix]- d2[(iy-1)*imageW+ix-1])) +
     alpha*( w3*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix+1]) + w4*abs(d2[iy*imageW+ix]- d2[(iy+1)*imageW+ix-1])) +
     beta*abs(gray[imageW * (iy-2) + ix-2] +
		     gray[imageW * (iy+2) + ix+2] -
		     2.0*gray[imageW * (iy) + ix]) +
     beta*abs(gray[imageW * (iy+2) + ix-2] +
		     gray[imageW * (iy-2) + ix+2] -
		     2.0*gray[imageW * (iy) + ix]);
 
	  	  
      ea = alpha*(w1*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix+1]- 3*step) + w2*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix-1]-3*step))+
   	 alpha*(w3*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix+1]- 3*step) + w4*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix-1]-3*step)) +
	alpha*(w1*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix+1]- 3*step) + w2*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix-1]-3*step)) +
 	 alpha*(w3*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix+1]- 3*step) + w4*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix-1]-3*step)) + 
 	beta*abs(gray[imageW * (iy-2) + ix-2] +
		     gray[imageW * (iy+2) + ix+2] -
		     2.0*(gray[imageW * (iy) + ix] + step)) +
	 beta*abs(gray[imageW * (iy+2) + ix-2] +
		     gray[imageW * (iy-2) + ix+2] -
		     2.0*(gray[imageW * iy + ix] + step));
 

      es = alpha*(w1*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix+1]+3*step) + w2*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix-1]+3*step)) +
	alpha*(w3*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix+1]+3*step) + w4*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix-1]+3*step))+
	alpha*(w1*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix+1]+3*step) + w2*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix-1]+3*step))+
	alpha*(w3*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix+1]+3*step) + w4*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix-1]+3*step))+ 
	 beta*abs(gray[imageW * (iy-2) + ix-2] +
		     gray[imageW * (iy+2) + ix+2] -
		     2.0*(gray[imageW * (iy) + ix] - step)) +
	 beta*abs(gray[imageW * (iy+2) + ix-2] +
		     gray[imageW * (iy-2) + ix+2] -
		     2.0*(gray[imageW * iy + ix] - step));

      eiso = (c1[iy*imageW+ix]*c1[iy*imageW+ix]*d2[iy*imageW+ix] -2*c1[iy*imageW+ix]*c2[iy*imageW+ix]*d3[iy*imageW+ix] + c2[iy*imageW+ix]*c2[iy*imageW+ix]*d1[iy*imageW+ix])/(1+c1[iy*imageW+ix]*c1[iy*imageW+ix]+c2[iy*imageW+ix]*c2[iy*imageW+ix]);

      if(abs(eiso)>0.01)
	eiso=gamma*(float)eiso/abs(eiso);
      else eiso=0;
	  
   
      ea = ea - eiso;
      es = es + eiso;
      p1 = TColorToFloat4(dst[imageW * (iy) + ix]);

      if(en>ea && es>ea){	  	
	dst[imageW * iy + ix] = make_color(min(p1.x+step,1.0),min(p1.y+ step,1.0), min(p1.z+ step,1.0), 0);	   
     	}	
else if (en>es && ea>es){
	dst[imageW * iy + ix] = make_color(max(p1.x- step,0.0),max(p1.y-step,0.0), max(p1.z- step,0.0), 0);	   
      }
      		
  }	
}



__global__ void icbiFill2(
			  TColor *dst,
			  int origW,
			  int origH
			  ){
  const int tix = blockDim.x * blockIdx.x + threadIdx.x;
  const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const int imageW=origW*2-1;
  const int imageH=origH*2-1;
  int ix, iy;
  float d01, d02;
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
	
      d01 = fmm13.x+fmm13.y+fmm13.z+
	fpp13.x+fpp13.y+fpp13.z+
	fmm31.x+fmm31.y+fmm31.z+
	fpp31.x+fpp31.y+fpp31.z+
	-6* (p1.x+p1.y+p1.z) + 2*(p2.x+p2.y+p2.z);
	     
      d02 = fmp13.x+fmp13.y+fmp13.z+
	fpm13.x+fpm13.y+fpm13.z+
	fmp31.x+fmp31.y+fmp31.z+
	fpm31.x+fpm31.y+fpm31.z+
	2 * (p1.x+p1.y+p1.z) - 6 *(p2.x+p2.y+p2.z);



      if(abs(d01) > abs(d02))
	dst[imageW * iy + ix] = make_color(p1.x,p1.y, p1.z, 0);
      else 
	dst[imageW * iy + ix] = make_color(p2.x,p2.y, p2.z, 0);
    	
	ix=2*tix+1;
	iy=2*tiy;
	}	
  }
  
  
}
 

__global__ void icbiDer2(
			 TColor *dst,
			 int imageW,
			 int imageH,
			 float* d1,
			 float* d2,
			 float* d3,
			 float* c1,
			 float* c2,
			 float* gray
			 ){
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
    
    
  if( ix < imageW-3 && iy < imageH-3 && ix >3 && iy>3){	

    gray[imageW *iy+ix] = TColorToGray(dst[imageW * (iy) + ix]);

    d1[imageW *iy+ix] =  TColorToGray(dst[imageW * (iy-1) + ix]) +
      TColorToGray(dst[imageW * (iy+1) + ix]) 
      - 2* TColorToGray(dst[imageW * (iy) + ix]);

    d2[imageW *iy+ix] =  TColorToGray(dst[imageW * (iy) + ix-1]) +
      TColorToGray(dst[imageW * (iy) + ix+1]) 
      - 2* TColorToGray(dst[imageW * (iy) + ix]);

    d3[imageW *iy+ix] =  0.5*(TColorToGray(dst[imageW * (iy-1) + ix-1]) 
			      - TColorToGray(dst[imageW * (iy-1) + ix+1]) 
			      - TColorToGray(dst[imageW * (iy+1) + ix-1])
			      + TColorToGray(dst[imageW * (iy+1) + ix+1]));

    c1[imageW *iy+ix] =  0.5* ( TColorToGray(dst[imageW * (iy-1) + ix]) -
				TColorToGray(dst[imageW * (iy+1) + ix]));

    c2[imageW *iy+ix] =   0.5* (  TColorToGray(dst[imageW * (iy) + ix+1]) -
				  TColorToGray(dst[imageW * (iy) + ix-1]));
  }
  
}



__global__ void icbiIter2(
			  TColor *dst,
			  int origW,
			  int origH,
			  float step,
			  float* d1,
			  float* d2,
			  float* d3,
			  float* c1,
			  float* c2,
			  float* gray
			  ){

  const int tix = blockDim.x * blockIdx.x + threadIdx.x;
  const int tiy = blockDim.y * blockIdx.y + threadIdx.y;
  const int imageW=origW*2-1;
  const int imageH=origH*2-1;
  int ix, iy;

  float en;
  float ea;
  float es;
  float eiso; 
  float4 p1;
   
    
  const float w1=1, w2=1, w3=1, w4=1;
  const float alpha=1.0, beta=-1.0, gamma=3.0;


if( tix < origW-1 && tiy < origH-1 && tix >1 && tiy>1){
    ix=2*tix+1; iy=2*tiy;
    for(int k=0;k<2;k++){	

      en = alpha*( w1*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix]) + w2*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix])) + 
	alpha*( w3*abs(d1[iy*imageW+ix]-d1[(iy)*imageW+ix+1]) + w4*abs(d1[iy*imageW+ix]-d1[(iy)*imageW+ix-1]))+
	 alpha*( w1*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix]) + w2*abs(d2[iy*imageW+ix]- d2[(iy-1)*imageW+ix]))+
	 alpha*( w3*abs(d2[iy*imageW+ix]-d2[(iy)*imageW+ix+1]) + w4*abs(d2[iy*imageW+ix]-d2[(iy)*imageW+ix-1]))+
	 beta*abs(gray[imageW * (iy-2) + ix] +
		     gray[imageW * (iy+2) + ix] -
		     2.0*(gray[imageW * (iy) + ix])) +
	 beta*abs(gray[imageW * (iy) + ix-2] +
		    gray[imageW * (iy) + ix+2] -
		     2.0*(gray[imageW * (iy) + ix]));
 
	  	  
      ea = alpha*(w1*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix]- 3*step) + w2*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix]-3*step))+
	 alpha*(w3*abs(d1[iy*imageW+ix]-d1[(iy)*imageW+ix+1]- 3*step) + w4*abs(d1[iy*imageW+ix]-d1[(iy)*imageW+ix-1]-3*step))+
	 alpha*(w1*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix]- 3*step) + w2*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix]-3*step))+
	alpha*(w3*abs(d2[iy*imageW+ix]-d2[(iy)*imageW+ix+1]- 3*step) + w4*abs(d2[iy*imageW+ix]-d2[(iy)*imageW+ix-1]-3*step)) +
	 beta*abs(gray[imageW * (iy-2) + ix] +
		     gray[imageW * (iy+2) + ix] -
		     2.0*(gray[imageW * (iy) + ix] + step)) + 
	beta*abs(gray[imageW * (iy) + ix-2] +
		     gray[imageW * (iy) + ix+2] - 
		     2.0*(gray[imageW * (iy) + ix] + step));
 

      es = alpha*(w1*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix]+3*step) + w2*abs(d1[iy*imageW+ix]-d1[(iy-1)*imageW+ix]+3*step))+
	 alpha*(w3*abs(d1[iy*imageW+ix]-d1[(iy)*imageW+ix+1]+3*step) + w4*abs(d1[iy*imageW+ix]-d1[(iy+1)*imageW+ix]+3*step))+
	 alpha*(w1*abs(d2[iy*imageW+ix]-d2[(iy+1)*imageW+ix]+3*step) + w2*abs(d2[iy*imageW+ix]-d2[(iy-1)*imageW+ix]+3*step))+
	 alpha*(w3*abs(d2[iy*imageW+ix]-d2[(iy)*imageW+ix+1]+3*step) + w4*abs(d2[iy*imageW+ix]-d2[(iy)*imageW+ix-1]+3*step)) +
	 beta*abs(gray[imageW * (iy-2) + ix] +
		     gray[imageW * (iy+2) + ix] -
		     2.0*(gray[imageW * (iy) + ix] - step))+
	beta*abs(gray[imageW * (iy) + ix-2] +
		     gray[imageW * (iy) + ix+2] -
		     2.0*(gray[imageW * (iy) + ix] - step));

      eiso = (c1[iy*imageW+ix]*c1[iy*imageW+ix]*d2[iy*imageW+ix] -2*c1[iy*imageW+ix]*c2[iy*imageW+ix]*d3[iy*imageW+ix] + c2[iy*imageW+ix]*c2[iy*imageW+ix]*d1[iy*imageW+ix])/(1+c1[iy*imageW+ix]*c1[iy*imageW+ix]+c2[iy*imageW+ix]*c2[iy*imageW+ix]);

      if(abs(eiso)>0.01)
	eiso=gamma*(float)eiso/abs(eiso);
      else eiso=0;
	  

      ea = ea - eiso;
      es = es + eiso;
      p1 = TColorToFloat4(dst[imageW * (iy) + ix]);


      if(en>ea && es>ea){	  	
	dst[imageW * iy + ix] = make_color(min(1.0,p1.x+step),min(1.0,p1.y+ step),min(1.0,p1.z + step), 0);	   
      }
       else if (en>es && ea>es){
	dst[imageW * iy + ix] = make_color(max(p1.x- step,0.0),max(p1.y-step,0.0), max(p1.z- step,0.0), 0.0);	  
      } 

     ix=2*tix; iy=2*tiy+1;
   }
   

  }
}


extern "C" void
cuda_ICBI(TColor *src, TColor *d_dst, float *d1, float  *d2, float  *d3, float  *c1, float  *c2, float  *gray, int origW, int origH)
{

  dim3 threads(BLOCKDIM_X, BLOCKDIM_Y);
  dim3 grid(iDivUp(origW, BLOCKDIM_X), iDivUp(origH, BLOCKDIM_Y));
  
  const int imageW = 2*origW-1;
  const int imageH = 2*origH-1; 
  dim3 grid2(iDivUp(imageW, BLOCKDIM_X), iDivUp(imageH, BLOCKDIM_Y));         
  const int iter=8;

  icbiInit<<<grid , threads>>>(src, d_dst, origW, origH);	
  
  icbiFill1<<<grid, threads>>>(d_dst, origW, origH);
 
  for (int i=0;i<iter;i++){
    icbiDer1<<<grid, threads>>>(d_dst,origW, origH, d1,d2,d3,c1,c2,gray);
    icbiIter1<<<grid, threads>>>(d_dst, origW, origH, 0.02,d1,d2,d3,c1,c2,gray);
  }
  for (int i=0;i<iter;i++){
    icbiDer1<<<grid, threads>>>(d_dst, origW, origH, d1,d2,d3,c1,c2,gray);	
    icbiIter1<<<grid, threads>>>(d_dst, origW, origH, 0.003,d1,d2,d3,c1,c2,gray);
  }
  icbiFill2<<<grid, threads>>>(d_dst, origW, origH);

  for (int i=0;i<iter;i++){
    icbiDer2<<<grid2, threads>>>(d_dst, imageW, imageH, d1,d2,d3,c1,c2,gray);	
    icbiIter2<<<grid, threads>>>(d_dst, origW, origH, 0.02,d1,d2,d3,c1,c2,gray);
  }
  for (int i=0;i<iter;i++){
    icbiDer2<<<grid2, threads>>>(d_dst, imageW, imageH, d1,d2,d3,c1,c2,gray);	
    icbiIter2<<<grid, threads>>>(d_dst, origW, origH, 0.003,d1,d2,d3,c1,c2,gray);
  }

}

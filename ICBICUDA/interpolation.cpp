/* ICBI Cuda implementation (c) 2010 andrea giachetti     
   License     : GNU GENERAL PUBLIC LICENSE v.2
   http://www.gnu.org/copyleft/gpl.html
   version 0.1   
   please report bugs/problems/tests/applications to andrea.giachetti@univr.it
   and cite related papers if used in published scientific work
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cutil.h>

#include "interpolation.h"

#include <QtGui/QImage>
#include <QtGui/QColor>

char image_path[100];

////////////////////////////////////////////////////////////////////////////////
// Global data handlers and parameters
////////////////////////////////////////////////////////////////////////////////

//Source image on the host side
uchar4 *ho_Src, *hi_Src, *h_Src;
unsigned int *hd_Src, *hdi_Src,*d_dst;
float* d1, *d2, *d3, *c1, *c2, *gray;
 
int imageW, imageH;
int intiW, intiH;
int origW, origH;

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////

unsigned int hTimer;


#define BUFFER_DATA(i) ((char *)0 + i)


void CreateEnlargedGrid(uchar4** h_Src, uchar4* ho_Src, int origW, int origH, int* imageW, int* imageH){

 
  *imageW = 2*origW-1;
  *imageH = 2*origH-1;
  *h_Src = (uchar4 *)malloc(*imageW * *imageH * 4);
   
  for(int i=0; i<origW; i++)
    for(int j=0; j<origH; j++)
      {
	(*h_Src)[2*i+2*j*(*imageW)] = ho_Src[i+j*origW];
      }
}


int main(int argc, char **argv){

  unsigned int timer;
  cutCreateTimer(&timer);

   
  printf("Allocating host and CUDA memory and loading image file...\n");
   
  strcpy(image_path,argv[1]);
  
  LoadBMPFile(&ho_Src, &origW, &origH, image_path);



#if FACT == 4
  CreateEnlargedGrid(&hi_Src, ho_Src, origW, origH, &intiW, &intiH);
  CreateEnlargedGrid(&h_Src, hi_Src, intiW, intiH, &imageW, &imageH);


#else
  CreateEnlargedGrid(&h_Src, ho_Src,origW,origH,&imageW,&imageH);
#endif

#if FACT == 4
  CUDA_SAFE_CALL(cudaMalloc((void**)&hdi_Src, 4*intiW*intiH*sizeof(unsigned int)) );
#endif

  CUDA_SAFE_CALL(cudaMalloc((void**)&hd_Src, 4*origW*origH*sizeof(unsigned int)) );
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_dst, 4*imageW*imageH*sizeof(unsigned int)) );

  CUDA_SAFE_CALL(cudaMemcpy(hd_Src, ho_Src, 4*origW*origH, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL( CUDA_MallocArray(&h_Src, imageW, imageH) );
	
  CUDA_SAFE_CALL(cudaMalloc((void**)&d1, imageW*imageH*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d2, imageW*imageH*sizeof(float)));	
  CUDA_SAFE_CALL(cudaMalloc((void**)&c1, imageW*imageH*sizeof(float)));	
  CUDA_SAFE_CALL(cudaMalloc((void**)&c2, imageW*imageH*sizeof(float)));	
  CUDA_SAFE_CALL(cudaMalloc((void**)&d3, imageW*imageH*sizeof(float)));
  CUDA_SAFE_CALL(cudaMalloc((void**)&gray, imageW*imageH*sizeof(float)));	

  printf("Data init done!\n");
    
  /*
    #if FACT == 4
    cuda_FCBI(hd_Src,hdi_Src,origW, origH);
    cuda_FCBI(hdi_Src,d_dst,intiW, intiH);
    #else
    cuda_FCBI(hd_Src, d_dst, origW, origH);
    #endif
  */

  cutStartTimer(timer);

#if FACT == 4

/*
        cuda_DDT(hd_Src,hdi_Src,origW, origH);
  	cuda_DDT(hdi_Src,d_dst,intiW, intiH);

        cuda_FCBI(hd_Src,hdi_Src,origW, origH);
        cuda_FCBI(hdi_Src,d_dst,intiW, intiH);

	ddt and fcbi for comparison
*/

  cuda_ICBI(hd_Src,hdi_Src, d1,d2,d3,c1,c2,gray,origW, origH);
  cuda_ICBI(hdi_Src,d_dst, d1,d2,d3,c1,c2,gray,intiW, intiH);

#else
  cuda_ICBI(hd_Src,d_dst, d1,d2,d3,c1,c2,gray,origW, origH);
#endif

  cudaThreadSynchronize(); 
  cutStopTimer(timer);

  float iTime = cutGetTimerValue(timer);
  printf("time:     %0.3f ms\n", iTime);

  // allocate mem for the result on host side
  unsigned int* h_odata = (unsigned int*) malloc( imageW*imageH*sizeof(int));
  CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_dst, imageW*imageH*sizeof(int), cudaMemcpyDeviceToHost) );

  cudaThreadSynchronize(); 


// added now to save images in png format. must change also input using Qimage!

  QImage img( imageW, imageH, QImage::Format_RGB32 );
  for (int i = 0; i < imageW; i ++)
    for (int j = 0; j < imageH; j++){
      TColor c = h_odata[j*imageW+i];
      float r = (c & 0xff) ;
      float g = ((c>>8) & 0xff);
      float b = ((c>>16) & 0xff);
      float a = ((c>>24) & 0xff);
      QColor nc( r, g, b, a );
      img.setPixel( i, imageH-1-j, nc.rgb() );
    }
  img.save( "output.png", 0, -1 );

  //CUT_SAFE_CALL( cutSavePGMi( "output.pgm", h_odata, imageW, imageH));
  cudaFree(d1);cudaFree(d2); cudaFree(d3);
  cudaFree(c1);cudaFree(c2); cudaFree(gray);
  cudaFree(d_dst);
  free(h_Src);
  // CUT_EXIT(argc, argv);
}

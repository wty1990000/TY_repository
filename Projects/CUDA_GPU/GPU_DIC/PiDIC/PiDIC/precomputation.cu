
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>

#include "precomputation.cuh"

__global__ void RGradient_kernel(const double *d_InputIMGR, const double *d_InputIMGT,const double* __restrict__ d_InputBiubicMatrix,
								 double *d_OutputIMGR, double *d_OutputIMGT, 
								 double *d_OutputIMGRx, double *d_OutputIMGRy,
								 double *d_OutputIMGTx, double *d_OutputIMGTy, double *d_OutputIMGTxy, double *d_OutputdtBicubic,
								 int width, int height)
{
	/* Here the width and height have been minused by 2. That is the original image width and height should be width+2
	   and height+2	*/
	//Share memory for Bicubic computation
	__shared__ double Input_R[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ double Input_T[BLOCK_SIZE][BLOCK_SIZE];
	//temperary parameters
	double d_TaoT[16];
	double d_AlphaT[16];

	//Map the threads to the pixel positions
	int tx  = threadIdx.x;
	int ty  = threadIdx.y;
	//The size of input images
	int rowi = blockIdx.y * blockDim.y + threadIdx.y;
	int coli = blockIdx.x * blockDim.x + threadIdx.x;
	//The rows and cols of output matrix.
	int rowo = rowi-2;
	int colo = coli-2;

	if(rowi < (height+2) && coli < (width+2)){
		Input_R[ty][tx] = d_InputIMGR[rowi*(width+2)+coli];
		Input_T[ty][tx] = d_InputIMGT[rowi*(width+2)+coli];
	}
	else{
		Input_R[ty][tx] = 0.0;
		Input_T[ty][tx] = 0.0;
	}
	//When all the threads finish the loading task, continue to computation.
	__syncthreads();

	if((rowo>=0) && (rowo < height) && (colo>=0) && (colo < width)&&(tx>=2)&&(ty>=2)){
		d_OutputIMGR[rowo*width+colo]  = Input_R[ty-1][tx-1];
		d_OutputIMGRx[rowo*width+colo] = 0.5 * (Input_R[ty-1][tx] - Input_R[ty-1][tx-2]);
		d_OutputIMGRy[rowo*width+colo] = 0.5 * (Input_R[ty][tx-1] - Input_R[ty-2][tx-1]);

		d_OutputIMGT[rowo*width+colo]  = Input_T[ty-1][tx-1];
		d_OutputIMGTx[rowo*width+colo] = 0.5 * (Input_T[ty-1][tx] - Input_T[ty-1][tx-2]);
		d_OutputIMGTy[rowo*width+colo] = 0.5 * (Input_T[ty][tx-1] - Input_T[ty-2][tx-1]);
		d_OutputIMGTxy[rowo*width+colo]= 0.25 * (Input_T[ty][tx]  - Input_T[ty-2][tx] - Input_T[ty][tx-2] + Input_T[ty-2][tx-2]);
	}
	__syncthreads();
	if(rowo>=0 && colo>=0){
		if((rowo < height-1) && (colo < width-1)){
		d_TaoT[0] = d_OutputIMGT[rowo*(width)+colo];
		d_TaoT[1] = d_OutputIMGT[rowo*(width)+colo+1];
		d_TaoT[2] = d_OutputIMGT[(rowo+1)*(width)+colo];
		d_TaoT[3] = d_OutputIMGT[(rowo+1)*(width)+colo+1];
		d_TaoT[4] = d_OutputIMGTx[rowo*(width)+colo];
		d_TaoT[5] = d_OutputIMGTx[rowo*(width)+colo+1];
		d_TaoT[6] = d_OutputIMGTx[(rowo+1)*(width)+colo];
		d_TaoT[7] = d_OutputIMGTx[(rowo+1)*(width)+colo+1];
		d_TaoT[8] = d_OutputIMGTy[rowo*(width)+colo];
		d_TaoT[9] = d_OutputIMGTy[rowo*(width)+colo+1];
		d_TaoT[10] = d_OutputIMGTy[(rowo+1)*(width)+colo];
		d_TaoT[11] = d_OutputIMGTy[(rowo+1)*(width)+colo+1];
		d_TaoT[12] = d_OutputIMGTxy[rowo*(width)+colo];
		d_TaoT[13] = d_OutputIMGTxy[rowo*(width)+colo+1];
		d_TaoT[14] = d_OutputIMGTxy[(rowo+1)*(width)+colo];
		d_TaoT[15] = d_OutputIMGTxy[(rowo+1)*(width)+colo+1];
		for(int k=0; k<16; k++){
			d_AlphaT[k] = 0.0;
			for(int l=0; l<16; l++){
				d_AlphaT[k] += (d_InputBiubicMatrix[k*16+l] * d_TaoT[l]);
			}
		}
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+0] = d_AlphaT[0];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+1] = d_AlphaT[1];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+2] = d_AlphaT[2];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+3] = d_AlphaT[3];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+0] = d_AlphaT[4];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+1] = d_AlphaT[5];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+2] = d_AlphaT[6];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+3] = d_AlphaT[7];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+0] = d_AlphaT[8];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+1] = d_AlphaT[9];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+2] = d_AlphaT[10];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+3] = d_AlphaT[11];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+0] = d_AlphaT[12];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+1] = d_AlphaT[13];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+2] = d_AlphaT[14];
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+3] = d_AlphaT[15];
	}
	else{
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+0] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+1] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+2] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+0)*4+3] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+0] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+1] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+2] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+1)*4+3] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+0] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+1] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+2] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+2)*4+3] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+0] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+1] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+2] = 0;
		d_OutputdtBicubic[((rowo*(width)+colo)*4+3)*4+3] = 0;
	}
	}
	
	

}

void precompute_kernel(const double *h_InputIMGR, const double *h_InputIMGT,
								 double *h_OutputIMGR, double *h_OutputIMGT, 
								 double *h_OutputIMGRx, double *h_OutputIMGRy,
								 double *h_OutputIMGTx, double *h_OutputIMGTy, double *h_OutputIMGTxy, double *h_OutputdTBicubic,
								 int width, int height)
{
	double *d_InputIMGR, *d_InputIMGT, *d_InputBiubicMatrix;
	double *d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputIMGTx, *d_OutputIMGTy, *d_OutputIMGTxy;
	double *d_OutputdTBicubic;

	/*h_OutputIMGR = (double*)malloc(width*height*sizeof(double));
	h_OutputIMGT = (double*)malloc(width*height*sizeof(double));
	h_OutputIMGRx = (double*)malloc(width*height*sizeof(double));
	h_OutputIMGRy = (double*)malloc(width*height*sizeof(double));
	h_OutputIMGTx = (double*)malloc(width*height*sizeof(double));
	h_OutputIMGTy = (double*)malloc(width*height*sizeof(double));
	h_OutputIMGTxy = (double*)malloc(width*height*sizeof(double));
	h_OutputdTBicubic = (double*)malloc(width*height*4*4*sizeof(double));*/

	const static double h_InputBicubicMatrix[16*16] = {  
													1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
													0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
													-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
													2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
													0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
													0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ,
													0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
													0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 , 
													-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 
													0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,  
													9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 , 
													-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1, 
													2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 , 
													0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 , 
													-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
													4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 
												   };

	cudaMalloc((void**)&d_InputIMGR, (width+2)*(height+2)*sizeof(double));
	cudaMalloc((void**)&d_InputIMGT, (width+2)*(height+2)*sizeof(double));
	cudaMalloc((void**)&d_InputBiubicMatrix, 16*16*sizeof(double));

	cudaMemcpy(d_InputIMGR,h_InputIMGR,(width+2)*(height+2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_InputIMGT,h_InputIMGT,(width+2)*(height+2)*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(d_InputBiubicMatrix,h_InputBicubicMatrix,16*16*sizeof(double),cudaMemcpyHostToDevice);
	
	cudaMalloc((void**)&d_OutputIMGR, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGT, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRx, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRy, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGTx, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGTy, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGTxy, width*height*sizeof(double));
	cudaMalloc((void**)&d_OutputdTBicubic, width*height*4*4*sizeof(double));

	dim3 dimB(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 dimG((width+1)/BLOCK_SIZE+1,(height+1)/BLOCK_SIZE+1,1);

	RGradient_kernel<<<dimG, dimB>>>(d_InputIMGR,d_InputIMGT,d_InputBiubicMatrix,
								d_OutputIMGR, d_OutputIMGT, 
								 d_OutputIMGRx, d_OutputIMGRy,
								 d_OutputIMGTx, d_OutputIMGTy, d_OutputIMGTxy,d_OutputdTBicubic,
								 width, height);

	cudaDeviceSynchronize();

	cudaMemcpy(h_OutputIMGR,d_OutputIMGR,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputIMGT,d_OutputIMGT,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputIMGRx,d_OutputIMGRx,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputIMGRy,d_OutputIMGRy,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputIMGTx,d_OutputIMGTx,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputIMGTy,d_OutputIMGTy,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputIMGTxy,d_OutputIMGTxy,width*height*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_OutputdTBicubic,d_OutputdTBicubic,width*height*4*4*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_OutputIMGR);
	cudaFree(d_OutputIMGT);
	cudaFree(d_OutputIMGRx);
	cudaFree(d_OutputIMGRy);
	cudaFree(d_OutputIMGTx);
	cudaFree(d_OutputIMGTy);
	cudaFree(d_OutputIMGTxy);
	cudaFree(d_OutputdTBicubic);
}


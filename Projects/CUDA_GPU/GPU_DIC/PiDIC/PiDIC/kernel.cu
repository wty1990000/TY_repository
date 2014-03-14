#include "kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>
#include <stdlib.h>



//Precomputing kernel
__global__ void RGradient_kernel(const double *d_InputIMGR, const double *d_InputIMGT, const double* __restrict__ M,
								 double *d_OutputIMGR, double *d_OutputIMGT, 
								 double *d_OutputIMGRx, double *d_OutputIMGRy,
								 double *d_OutputIMGTx, double *d_OutputIMGTy, double *d_OutputIMGTxy,
								 int width, int height)
{
	//Map the threads to the pixel positions
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int i  = row * width;		unsigned int j  = col;
	unsigned int i1 = (row+1) * width;	unsigned int j1 = col +1;
	unsigned int i2 = (row+2) * width;  unsigned int j2 = col +2;

	if(row < height && col < width){
		d_OutputIMGR[i+j]  = d_InputIMGR[i1+j1];
		d_OutputIMGRx[i+j] = 0.5 * (d_InputIMGR[i1+j2] - d_InputIMGR[i1+j]);
		d_OutputIMGRy[i+j] = 0.5 * (d_InputIMGR[i2+j1] - d_InputIMGR[i+j1]);

		d_OutputIMGT[i+j]  = d_InputIMGT[i1+j1];
		d_OutputIMGTx[i+j] = 0.5 * (d_InputIMGT[i1+j2] - d_InputIMGT[i1+j]);
		d_OutputIMGTy[i+j] = 0.5 * (d_InputIMGT[i2+j1] - d_InputIMGT[i+j1]);
		d_OutputIMGTxy[i+j]= 0.25 * (d_InputIMGT[i2+j2] - d_InputIMGT[i+j2] - d_InputIMGT[i2+j] + d_InputIMGT[i+j]);
	}
	__syncthreads();
	if(row < height-1 && col < width-1){

	}
}
//CUFFT

//Inverse Gauss-Newton


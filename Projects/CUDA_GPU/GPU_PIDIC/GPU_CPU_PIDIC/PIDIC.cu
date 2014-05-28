#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "EasyBMP.h"
#include "PIDIC.cuh"
#include "FFTCC.h"

#include <stdio.h>
#include <iostream>

const int BLOCK_SIZE = 16;

//CUDA RUNTIME Initialization
void InitCuda()
{
	cudaFree(0);
}

__global__ void precomputation_kernel(float *d_InputIMGR, float *d_InputIMGT, const float* __restrict__ d_InputBiubicMatrix,
								 float *d_OutputIMGR, float *d_OutputIMGT, float *d_OutputIMGRx, float *d_OutputIMGRy,
								 float *d_OutputIMGTx, float *d_OutputIMGTy, float *d_OutputIMGTxy, float *d_OutputdtBicubic,
								 int width, int height)
{
	//The size of input images
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	//Temp arrays
	float d_TaoT[16];
	float d_AlphaT[16];

	//The rows and cols of output matrix.

	if((row < height) && (col < width)){
		d_OutputIMGR[row*width+col]  = d_InputIMGR[(row+1)*(width+2)+col+1];
		d_OutputIMGRx[row*width+col] = 0.5 * (d_InputIMGR[(row+1)*(width+2)+col+2] - d_InputIMGR[(row+1)*(width+2)+col]);
		d_OutputIMGRy[row*width+col] = 0.5 * (d_InputIMGR[(row+2)*(width+2)+col+1] - d_InputIMGR[(row)*(width+2)+col+1]);

		d_OutputIMGT[row*width+col]  = d_InputIMGT[(row+1)*(width+2)+col+1];
		d_OutputIMGTx[row*width+col] = 0.5 * (d_InputIMGT[(row+1)*(width+2)+col+2] -d_InputIMGT[(row+1)*(width+2)+col]);
		d_OutputIMGTy[row*width+col] = 0.5 * (d_InputIMGT[(row+2)*(width+2)+col+1] - d_InputIMGT[(row)*(width+2)+col+1]);
		d_OutputIMGTxy[row*width+col]= 0.25 * (d_InputIMGT[(row+2)*(width+2)+col+2]  - d_InputIMGT[(row)*(width+2)+col+2] -d_InputIMGT[(row+2)*(width+2)+col] + d_InputIMGT[(row)*(width+2)+col]);
	}
	__syncthreads();
	if((row < height-1) && (col < width-1)){
		d_TaoT[0] = d_OutputIMGT[row*(width)+col];
		d_TaoT[1] = d_OutputIMGT[row*(width)+col+1];
		d_TaoT[2] = d_OutputIMGT[(row+1)*(width)+col];
		d_TaoT[3] = d_OutputIMGT[(row+1)*(width)+col+1];
		d_TaoT[4] = d_OutputIMGTx[row*(width)+col];
		d_TaoT[5] = d_OutputIMGTx[row*(width)+col+1];
		d_TaoT[6] = d_OutputIMGTx[(row+1)*(width)+col];
		d_TaoT[7] = d_OutputIMGTx[(row+1)*(width)+col+1];
		d_TaoT[8] = d_OutputIMGTy[row*(width)+col];
		d_TaoT[9] = d_OutputIMGTy[row*(width)+col+1];
		d_TaoT[10] = d_OutputIMGTy[(row+1)*(width)+col];
		d_TaoT[11] = d_OutputIMGTy[(row+1)*(width)+col+1];
		d_TaoT[12] = d_OutputIMGTxy[row*(width)+col];
		d_TaoT[13] = d_OutputIMGTxy[row*(width)+col+1];
		d_TaoT[14] = d_OutputIMGTxy[(row+1)*(width)+col];
		d_TaoT[15] = d_OutputIMGTxy[(row+1)*(width)+col+1];
		for(int k=0; k<16; k++){
			d_AlphaT[k] = 0.0;
			for(int l=0; l<16; l++){
				d_AlphaT[k] += (d_InputBiubicMatrix[k*16+l] * d_TaoT[l]);
			}
		}
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+0] = d_AlphaT[0];
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+1] = d_AlphaT[1];
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+2] = d_AlphaT[2];
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+3] = d_AlphaT[3];
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+0] = d_AlphaT[4];
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+1] = d_AlphaT[5];
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+2] = d_AlphaT[6];
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+3] = d_AlphaT[7];
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+0] = d_AlphaT[8];
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+1] = d_AlphaT[9];
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+2] = d_AlphaT[10];
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+3] = d_AlphaT[11];
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+0] = d_AlphaT[12];
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+1] = d_AlphaT[13];
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+2] = d_AlphaT[14];
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+3] = d_AlphaT[15];
	}
	else if(((row >=height-1)&&(row < height)) && ((col >= width-1)&&(col<width))){
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+0] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+1] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+2] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+0)*4+3] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+0] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+1] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+2] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+1)*4+3] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+0] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+1] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+2] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+2)*4+3] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+0] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+1] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+2] = 0.0;
		d_OutputdtBicubic[((row*(width)+col)*4+3)*4+3] = 0.0;
	}

}

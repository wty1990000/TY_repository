#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "thrust\host_vector.h"
#include "thrust\device_vector.h"
#include "thrust\reduce.h"
#include "EasyBMP.h"
#include "PIDIC.cuh"
#include "FFTCC.h"

#include <stdio.h>
#include <iostream>

//Parameters
const int iMarginX = 10,	iMarginY = 10;
const int iGridX = 10,		iGridY = 10;
const int iSubsetX = 8,	iSubsetY =8;
const float fDeltaP = 0.001f;
const int iIterationNum = 5;
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
__global__ void ICGN_kernel(float* input_R, float* input_Rx, float* input_Ry, float fDeltaP,
							float *input_T,   float* input_Bicubic, int* input_iU, int* input_iV, 
							int iNumberY, int iNumberX, int iSubsetH, int iSubsetW, int width, int height, int iSubsetY, int iSubsetX, 
							int iGridSpaceX, int iGridSpaceY, int iMarginX, int iMarginY, int iIterationNum,
							float* output_dP)
/*BLOCK_SIZE: 2*(iSubsetW+1)+1, 2*(iSubsetH+1)+1
  Grid_SIZE:  iNumberX * iNumberY 
*/
{
	int x = threadIdx.x, y = threadIdx.y;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int offset = row * iNumberX + col;

	//Shared variables of each ROI
	__shared__ float fPXY[2];
	__shared__ float fT[19*19];
	__shared__ float fBicubic[19*19*4*4];
	__shared__ float fR[17*17];
	__shared__ float fRx[17*17];
	__shared__ float fRy[17*17];
	__shared__ float fdP[6], fWarp[3][3], fHessian[6][6],fInvHessian[6][6],fNumerator[6];

	//Private variables for each subset point
	float fJacobian[2][6];
	float fdU, fdV, fdUx, fdUy, fdVx, fdVy;
	float fdDU, fdDUx, fdDUy, fdDV, fdDVx, fdDVy;
	float fSubAveR, fSubNormR, fSubAveT, fSubNormT;
	
	//Load the PXY, Rx, Ry and R, T, Bicubic into shared memory of each block
	if(x==0 && y ==0){
		fPXY[0] = float(iMarginX + iSubsetY + blockIdx.y * iGridSpaceY);
		fPXY[1] = float(iMarginY + iSubsetX + blockIdx.x * iGridSpaceX);
	}
	__syncthreads();

}


void computation_interface(const float* ImgR, const float* ImgT, int iWidth, int iHeight)
{
	//Timers
	StopWatchWin WatchPrecompute, WatchICGN, WatchTotal;
	float fTimePrecopmute=0.0f, fTimeFFTCC=0.0f, fTimeICGN=0.0f, fTimeTotal=0.0f;

	
	
	//Parameters used in the computations.
	int width =  iWidth - 2;
	int height = iHeight -2;
	int iNumberX = int(floor((width - iSubsetX*2 - iMarginX*2)/float(iGridX))) + 1;
	int iNumberY = int(floor((height - iSubsetY*2 - iMarginY*2)/float(iGridY))) + 1;
	int iSubsetW = iSubsetX*2+1;
	int iSubsetH = iSubsetY*2+1;
	int iFFTSubW = iSubsetX*2;
	int iFFTSubH = iSubsetY*2;

	/*--------------------------------------Parameters for CUDA kernel use---------------------------------------------*/
	//Precomputation Parameters
	const static float h_InputBicubicCoeff[16*16] = {  
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
	float *d_InputIMGR, *d_InputIMGT,*d_InputBiubicCoeff;
	float *d_OutputIMGTx, *d_OutputIMGTy, *d_OutputIMGTxy,*d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputBicubic;
	//FFT-ZNCC Parameters
	float *hInput_dR, *hInput_dT, *fZNCC;
	int *iU, *iV;
	
	/*------------------------------Real computation starts here--------------------------------
	  Totally, there are three steps:
	  1. Precomputation of images' gradients matrix and bicubic interpolation matrix
	  2. Using FFT to transform the two images into frequency domain, and after per-
	  forming ZNCC, transforming the results back.
	  3. A Gaussian Newton's optimization method is used to estimate the warped images.
	*/
	WatchTotal.start();

	//Precomputation Starts;
	WatchPrecompute.start();
	checkCudaErrors(cudaMalloc((void**)&d_InputIMGR, (width+2)*(height+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_InputIMGT, (width+2)*(height+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_InputBiubicCoeff, 16*16*sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_InputIMGR,ImgR,(width+2)*(height+2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_InputIMGT,ImgT,(width+2)*(height+2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_InputBiubicCoeff,h_InputBicubicCoeff,16*16*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGR, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGRx, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGRy, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGT, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTx, width*height*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTy, width*height*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTxy, width*height*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputBicubic, (width*height*4*4)*sizeof(float)));
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 dimGirds((width-1)/BLOCK_SIZE+1,(height-1)/BLOCK_SIZE+1,1);
	precomputation_kernel<<<dimGirds,dimBlock>>>(d_InputIMGR,d_InputIMGT,d_InputBiubicCoeff,
						  d_OutputIMGR,d_OutputIMGT,d_OutputIMGRx,d_OutputIMGRy,
						  d_OutputIMGTx,d_OutputIMGTy,d_OutputIMGTxy,d_OutputBicubic,
						  width,height);
	cudaFree(d_OutputIMGTx);
	cudaFree(d_OutputIMGTy);
	cudaFree(d_OutputIMGTxy);
	cudaFree(d_InputIMGR);
	cudaFree(d_InputIMGT);
	cudaFree(d_InputBiubicCoeff);
	WatchPrecompute.stop();
	fTimePrecopmute = WatchPrecompute.getTime();

	//FFT-ZNCC Begins
	hInput_dR = (float*)malloc(width*height*sizeof(float));
	hInput_dT = (float*)malloc(width*height*sizeof(float));
	fZNCC = (float*)malloc(iNumberX*iNumberY*sizeof(float));
	iU = (int*)malloc(iNumberX*iNumberY*sizeof(int));
	iV = (int*)malloc(iNumberX*iNumberY*sizeof(int));
	float *fdPXY = (float*)malloc(iNumberX*iNumberY*2*sizeof(float));
	for(int i=0; i<iNumberY; i++){
		for(int j=0; j<iNumberX; j++){
			fdPXY[(i*iNumberX+j)*2+0] = float(iMarginX + iSubsetY + i*iGridY);
			fdPXY[(i*iNumberX+j)*2+1] = float(iMarginY + iSubsetX + j*iGridX);
		}
	}
	checkCudaErrors(cudaMemcpy(hInput_dR, d_OutputIMGR, width*height*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hInput_dT, d_OutputIMGT, width*height*sizeof(float), cudaMemcpyDeviceToHost));
	FFT_CC_interface(hInput_dR, hInput_dT, fdPXY, iNumberY, iNumberX, iFFTSubH, iFFTSubW, 
		width, height, iSubsetY, iSubsetX, fZNCC, iU, iV, fTimeFFTCC);

	WatchTotal.stop();
	fTimeTotal = WatchTotal.getTime();


	/*Befor ICGN starts, pass the average value in first, to simplify GPU's work.*/



	//ICGN-Begins
	WatchICGN.start();
	int *dInput_iU, *dInput_iV;
	float *dInput_fPXY, *dOutput_fDP;
	checkCudaErrors(cudaMalloc((void**)&dInput_iU, (iNumberX*iNumberY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dInput_iV, (iNumberX*iNumberY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dInput_fPXY, (iNumberX*iNumberY)*2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dOutput_fDP, (iNumberX*iNumberY)*6*sizeof(float)));
	checkCudaErrors(cudaMemcpy(dInput_iU, iU,(iNumberX*iNumberY)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dInput_iV, iV,(iNumberX*iNumberY)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dInput_fPXY, fdPXY,(iNumberX*iNumberY)*2*sizeof(float), cudaMemcpyHostToDevice));

	dim3 dimB(1,1,1);
	dim3 dimG(iNumberX,iNumberY,1);

	ICGN_kernel<<<dimG,dimB>>>(dInput_fPXY,d_OutputIMGR,d_OutputIMGRx,d_OutputIMGRy,fDeltaP,d_OutputIMGT,d_OutputBicubic,dInput_iU,dInput_iV,
		iNumberY,iNumberX,iSubsetH,iSubsetW,width,height,iSubsetY,iSubsetX,iIterationNum,dOutput_fDP);
	float *fdP = (float*)malloc(iNumberX*iNumberY*6*sizeof(float));
	checkCudaErrors(cudaMemcpy(fdP, dOutput_fDP, iNumberY*iNumberX*6*sizeof(float), cudaMemcpyDeviceToHost));
	WatchICGN.stop();
	fTimeICGN = WatchICGN.getTime();
	

	checkCudaErrors(cudaFree(d_OutputIMGR));
	checkCudaErrors(cudaFree(d_OutputIMGRx));
	checkCudaErrors(cudaFree(d_OutputIMGRy));
	checkCudaErrors(cudaFree(d_OutputIMGT));
	checkCudaErrors(cudaFree(d_OutputBicubic));
	checkCudaErrors(cudaFree(dInput_iU));
	checkCudaErrors(cudaFree(dInput_iV));
	checkCudaErrors(cudaFree(dInput_fPXY));
	checkCudaErrors(cudaFree(dOutput_fDP));
	free(hInput_dR);
	free(hInput_dT);
	free(fZNCC);
	free(fdP);
	free(fdPXY);
	free(iU);
	free(iV);
}
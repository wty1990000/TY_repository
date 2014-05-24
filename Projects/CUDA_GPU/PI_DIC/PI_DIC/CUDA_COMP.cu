#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "CUDA_COMP.cuh"
#include <iostream>
#include <vector>
#include <stdio.h>

#include "FFT-CC.h"

const int BLOCK_SIZE = 16;

//Initialize the CUDA runtime library
void cudaInit()
{
	cudaFree(0);
}


/*
--------------CUDA Kernels used for GPU computing--------------
*/
//Precopmutation kernel
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
//ICGN Kernel
__global__ void ICGN_kernel(float* fInput_dPXY, float* fInput_dR, float* fInput_dRx, float* fInput_dRy, float fDeltaP, float* fInput_dT, float* fInput_Bicubic, int* iInput_iU, int* iInput_iV,
							int iNumberY, int iNumberX, int iSubsetH, int iSubsetW, int width, int height, int iSubsetY, int iSubsetX, int iIterationNum, float* fOutput_dP)
{
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	int offset = row*iNumberX+col;

	//Used variables
	int k,l,m,n, iTemp, iTempX, iTempY;
	float fTemp, fTempX, fTempY;
	float fdU, fdV, fdUx, fdUy, fdVx, fdVy;
	float fdDU, fdDUx, fdDUy, fdDV, fdDVx, fdDVy;
	float fSubAveR = 0.0f, fSubNormR = 0.0f;
	float fSubAveT, fSubNormT;
	float fWarpX, fWarpY;
	float fdP[6], fdWarp[3][3], fJacobian[2][6], fHessian[6][6], fHessianXY[6][6], fInvHessian[6][6], fdPXY[2], fNumerator[6];
	float *fSubsetR = (float*)malloc(iSubsetH*iSubsetW*sizeof(float)), *fSubsetT = (float*)malloc(iSubsetH*iSubsetW*sizeof(float));
	float *fSubsetAveR = (float*)malloc(iSubsetH*iSubsetW*sizeof(float)), *fSubsetAveT = (float*)malloc(iSubsetH*iSubsetW*sizeof(float));
	float *fRDescent = (float*)malloc(iSubsetH*iSubsetW*6*sizeof(float));
	float fError;

	if((row<iNumberY) && (col<iNumberX)){
		fdU = float(iInput_iU[row*iNumberX+col]); fdV = float(iInput_iV[row*iNumberX+col]);	fdUx = 0.0f; fdUy = 0.0f; fdVx = 0.0f; fdVy = 0.0f;
		fdP[0] = fdU, fdP[1] = fdUx, fdP[2] = fdUy, fdP[3] = fdV, fdP[4] = fdVx, fdP[5] = fdVy;
		fdPXY[0] = fInput_dPXY[offset*2+0], fdPXY[1] = fInput_dPXY[offset*2+1];
		fdWarp[0][0] = 1+fdUx, fdWarp[0][1] = fdUy, fdWarp[0][2] = fdU, fdWarp[1][0] = fdVx, fdWarp[1][1] = 1+fdVy, fdWarp[1][2] = fdV, fdWarp[2][0] = 0.0f, fdWarp[2][1] = 0.0f, fdWarp[2][2] = 1.0f;

		//Initialize the Hessian matrix in subsetR
		for(k=0; k<6; k++){
			for(n=0; n<6; n++){
				fHessian[k][n] = 0.0f;
			}
		}
		//Fill the gray intensity value to subset R
		for(l=0; l<iSubsetH; l++){
			for(m=0; m<iSubsetW; m++){
				fSubsetR[l*iSubsetW+m] = fInput_dR[int(fdPXY[0] - iSubsetY+l)*width+int(fdPXY[1]-iSubsetX+m)];
				fSubAveR += (fSubsetR[l*iSubsetW+m]/(iSubsetH * iSubsetW));
				//Evaluate the Jacobian dW/dp at(x,0)
				fJacobian[0][0] = 1.0f, fJacobian[0][1] = float(m-iSubsetX), fJacobian[0][2] = float(l-iSubsetY), fJacobian[0][3] = 0.0f, fJacobian[0][4] = 0.0f, fJacobian[0][5] = 0.0f;
				fJacobian[1][0] = 0.0f, fJacobian[1][1] = 0.0f, fJacobian[1][2] = 0.0f, fJacobian[1][3] = 1.0f, fJacobian[1][4] = float(m-iSubsetX), fJacobian[1][5] = float(l-iSubsetY);
				for(k=0; k<6; k++){
					fRDescent[(l*iSubsetW+m)*6+k] = fInput_dRx[int(fdPXY[0] - iSubsetY+l)*width+int(fdPXY[1] - iSubsetX +m)]*fJacobian[0][k]
												   +fInput_dRy[int(fdPXY[0] - iSubsetY+l)*width+int(fdPXY[1] - iSubsetX +m)]*fJacobian[1][k];
				}
				for(k=0; k<6; k++){
					for(n=0; n<6; n++){
						fHessianXY[k][n] = fRDescent[(l*iSubsetW+m)*6+k] * fRDescent[(l*iSubsetW+m)*6+n];	//Hessian matrix at each point
						fHessian[k][n] += fHessianXY[k][n];
					}
				}
			}
		}
		for(l=0; l<iSubsetH; l++){
			for(m=0; m<iSubsetW; m++){
				fSubsetAveR[l*iSubsetW+m] = fSubsetR[l*iSubsetW+m] - fSubAveR;
				fSubNormR += pow(fSubsetAveR[l*iSubsetW+m],2);
			}
		}
		fSubNormR = sqrt(fSubNormR);
		//Inverse the Hessian matrix
		for(l=0; l<6; l++){
			for(m=0; m<6; m++){
				if( l==m ){
					fInvHessian[l][m] = 1.0f;
				}
				else{
					fInvHessian[l][m] = 0.0f;
				}
			}
		}
		for(l=0; l<6; l++){
			iTemp = 1;
			for(m=l+1; m<6; m++){
				if(fHessian[m][l] > fHessian[iTemp][l]){
					iTemp = m;
				}
			}
			if(iTemp != l){
				for(k=0; k<6; k++){
					fTemp = fHessian[l][k];
					fHessian[l][k] = fHessian[iTemp][k];
					fHessian[iTemp][k] = fTemp;
					
					fTemp = fInvHessian[l][k];
					fInvHessian[l][k] = fInvHessian[iTemp][k];
					   fInvHessian[iTemp][k] = fTemp;
				}
			}
			for(m=0; m<6; m++){
				fTemp = fHessian[m][l]; 
				if(m != l){
					for(n=0; n<6; n++){
						fInvHessian[m][n] -= fInvHessian[l][n] * fTemp / fHessian[l][l];
						fHessian[m][n]    -= fHessian[l][n] * fTemp / fHessian[l][l];
					}
				}
				else{
					for(n=0; n<6; n++){
						fInvHessian[m][n] /= fTemp;
						fHessian[m][n]    /= fTemp;
					}
				}
			}
		}
		//Initialize DeltaP
		fdDU = 0.0f, fdDUx = 0.0f, fdDUy = 0.0f, fdDV = 0.0f, fdDVx = 0.0f, fdDVy = 0.0f;
		//Perform the Newton's iterations
		for(int it = 0; it < iIterationNum; it++){
			fSubAveT = 0.0f, fSubNormT = 0.0f;
			for(l=0; l<iSubsetH; l++){
				for(m=0; m<iSubsetW; m++){
					fWarpX = fdPXY[1] + fdWarp[0][0] * float(m-iSubsetX) + fdWarp[0][1] * float(l-iSubsetY) + fdWarp[0][2];
					fWarpY = fdPXY[0] + fdWarp[1][0] * float(m-iSubsetX) + fdWarp[1][1] * float(l-iSubsetY) + fdWarp[1][2];
					iTempX = int(fWarpX);
					iTempY = int(fWarpY);
					if((iTempX >=0) && (iTempY >=0) && (iTempX < width) && (iTempY < height)){
						fTempX = fWarpX - float(iTempX);
						fTempY = fWarpY - float(iTempY);
						if((fTempX ==0.0f) && (fTempY ==0.0f)){
							fSubsetT[l*iSubsetW+m] = fInput_dT[iTempY*width+iTempX];
						}
						else{
							fSubsetT[l*iSubsetW+m] =0.0f;
							for(k=0; k<4; k++){
								for(n=0; n<4; n++){
									fSubsetT[l*iSubsetW+m] += fInput_Bicubic[((iTempY*width+iTempX)*4+k)*4+n]*pow(fTempY,k)*pow(fTempX,n);
								}
							}
						}
						fSubAveT += (fSubsetT[l*iSubsetW+m]/(iSubsetH*iSubsetW));
					}
				}
			}
			for(l=0; l<iSubsetH; l++){
				for(m=0; m<iSubsetW; m++){
					fSubsetAveT[l*iSubsetW+m] = fSubsetT[l*iSubsetW+m] - fSubAveT;
					fSubNormT += pow(fSubsetAveT[l*iSubsetW+m],2);
				}
			}
			fSubNormT = sqrt(fSubNormT);
			//Compute the error image
			for(k=0; k<6; k++){
				fNumerator[k] = 0.0f;
			}
			for(l=0; l<iSubsetH; l++){
				for(m=0; m<iSubsetW; m++){
					fError = (fSubNormR / fSubNormT) * fSubsetAveT[l*iSubsetW+m] * fSubsetAveR[l*iSubsetW+m];
					for(k=0; k<6; k++){
						fNumerator[k] += (fRDescent[(l*iSubsetW+m)*6+k] * fError);
					}
				}
			}
			//Compute DeltaP
			for(k=0; k<6; k++){
				fdP[k] = 0.0f;
				for(n=0; n<6; n++){
					fdP[k] += (fInvHessian[k][n] * fNumerator[n]);
				}
			}
			fdDU = fdP[0];
			fdDUx = fdP[1];
			fdDUy = fdP[2];
			fdDV = fdP[3];
			fdDVx = fdP[4];
			fdDVy = fdP[5];
			//Update the warp
			fTemp = (1+fdDUx) * (1+fdDVy) - fdDUy * fdDVx;
			//W(P) <- W(P) o W(DP)^-1
			fdWarp[0][0] = ((1 + fdUx) * (1 + fdDVy) - fdUy * fdDVx) / fTemp;
			fdWarp[0][1] = (fdUy * (1 + fdDUx) - (1 + fdUx) * fdDUy) / fTemp;
			fdWarp[0][2] = fdU + (fdUy * (fdDU * fdDVx - fdDV - fdDV * fdDUx) - (1 + fdUx) * (fdDU * fdDVy + fdDU - fdDUy * fdDV)) / fTemp;
			fdWarp[1][0] = (fdVx * (1 + fdDVy) - (1 + fdVy) * fdDVx) / fTemp;
			fdWarp[1][1] = ((1 + fdVy) * (1 + fdDUx) - fdVx * fdDUy) / fTemp;
			fdWarp[1][2] = fdV + ((1 + fdVy) * (fdDU * fdDVx - fdDV - fdDV * fdDUx) - fdVx * (fdDU * fdDVy + fdDU - fdDUy * fdDV)) / fTemp;
			fdWarp[2][0] = 0.0f;
			fdWarp[2][1] = 0.0f;
			fdWarp[2][2] = 1.0f;

			// Update DeltaP
			fdP[0] = fdWarp[0][2];
			fdP[1] = fdWarp[0][0] - 1;
			fdP[2] = fdWarp[0][1];
			fdP[3] = fdWarp[1][2];
			fdP[4] = fdWarp[1][0];
			fdP[5] = fdWarp[1][1] - 1;

			fdU = fdP[0];
			fdUx = fdP[1];
			fdUy = fdP[2];
			fdV = fdP[3];
			fdVx = fdP[4];
			fdVy = fdP[5];
		}
	}
	__syncthreads();
	if((row<iNumberY) && (col<iNumberX)){
		fOutput_dP[(row*iNumberX+col)*6+0] = fdP[0];
		fOutput_dP[(row*iNumberX+col)*6+1] = fdP[1];
		fOutput_dP[(row*iNumberX+col)*6+2] = fdP[2];
		fOutput_dP[(row*iNumberX+col)*6+3] = fdP[3];
		fOutput_dP[(row*iNumberX+col)*6+4] = fdP[4];
		fOutput_dP[(row*iNumberX+col)*6+5] = fdP[5];
	}
	free(fSubsetR);
	free(fSubsetT);
	free(fSubsetAveR);
	free(fSubsetAveT);
	free(fRDescent);
}


/*
--------------Interface Functions Declarition for code integration in .cu file--------------
*/
//Precomputation Interface
void precompoutation_interface(const std::vector<float>& h_InputIMGR, const std::vector<float>& h_InputIMGT, int width, int height,
							  float *d_OutputIMGR, float *d_OutputIMGT, float *d_OutputIMGRx,
							  float *d_OutputIMGRy, float *d_OutputdTBicubic, float& time);
void ICGN_interface(float* dInput_dPXY, float* dInput_dR, float* dInput_dRx, float* dInput_dRy, float* dInput_dT, float* dInput_Bicubic, float fDeltaP,
					int* dInput_iU, int* dInput_iV, int iNumberY, int iNumberX, int iSubsetH, int iSubsetW, int width, int height, int iSubsetY, int iSubsetX, int iIterationNum,
					float* dOutput_dP, float& time);

/*
--------------Interface Function CombinedComputation for host use--------------
*/
void combined_function(const std::vector<float>& hInput_IMGR, const std::vector<float>& h_InputIMGT, float* hInput_dPXY, int width, int height,
					   int iSubsetH, int iSubsetW, int iSubsetX, int iSubsetY, int iNumberX, int iNumberY, int iFFTSubH, int iFFTSubW, int iIterationNum, float fDeltaP, 
					   int* iU, int* iV, float* fZNCC, float* fdP, float& fTimePrecomputation, float& fTimeFFT, float& fTimeICGN, float& fTimeTotal)
{
	StopWatchWin TotalWatch;

	//Variables for GPU precomputation
	float* d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy,*d_OutputBicubic;
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGR, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGRx, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGRy, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGT, (width*height)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputBicubic, (width*height*4*4)*sizeof(float)));
	//Variables for CPU FFT-CC
	float *hInput_dR = (float*)malloc(width*height*sizeof(float));
	float *hInput_dT = (float*)malloc(width*height*sizeof(float));
	//Variables for GPU IC-GN
	int *dInput_iU, *dInput_iV;
	float *dInput_fPXY, *dOutput_fDP;
	checkCudaErrors(cudaMalloc((void**)&dInput_iU, (iNumberX*iNumberY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dInput_iV, (iNumberX*iNumberY)*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&dInput_fPXY, (iNumberX*iNumberY)*2*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&dOutput_fDP, (iNumberX*iNumberY)*6*sizeof(float)));

	/*--------------------------Start the whole computation--------------------------*/
	TotalWatch.start();

	precompoutation_interface(hInput_IMGR, h_InputIMGT, width, height, d_OutputIMGR, d_OutputIMGT, d_OutputIMGRx, d_OutputIMGRy, d_OutputBicubic, fTimePrecomputation);
	checkCudaErrors(cudaMemcpy(hInput_dR, d_OutputIMGR, width*height*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hInput_dT, d_OutputIMGT, width*height*sizeof(float), cudaMemcpyDeviceToHost));

	FFT_CC_interface(hInput_dR, hInput_dT, hInput_dPXY, iNumberY, iNumberX, iFFTSubH, iFFTSubW, 
		width, height, iSubsetY, iSubsetX, fZNCC, iU, iV, fTimeFFT);
	checkCudaErrors(cudaMemcpy(dInput_iU, iU,(iNumberX*iNumberY)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dInput_iV, iV,(iNumberX*iNumberY)*sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dInput_fPXY, hInput_dPXY,(iNumberX*iNumberY)*2*sizeof(float), cudaMemcpyHostToDevice));

	ICGN_interface(dInput_fPXY,d_OutputIMGR,d_OutputIMGRx,d_OutputIMGRy,d_OutputIMGT, d_OutputBicubic, fDeltaP, dInput_iU, dInput_iV, iNumberY, iNumberX, iSubsetH, iSubsetW,
		width,height,iSubsetY,iSubsetX, iIterationNum, dOutput_fDP, fTimeICGN);
	checkCudaErrors(cudaMemcpy(fdP, dOutput_fDP, iNumberY*iNumberX*6*sizeof(float), cudaMemcpyDeviceToHost));

	TotalWatch.stop();
	fTimeTotal = TotalWatch.getTime();

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

}




/*
--------------Interface Functions Definition for code integration in .cu file--------------
*/
//Precomputation Interface function
void precompoutation_interface(const std::vector<float>& h_InputIMGR, const std::vector<float>& h_InputIMGT, int width, int height,
							  float *d_OutputIMGR, float *d_OutputIMGT, float *d_OutputIMGRx,
							  float *d_OutputIMGRy, float *d_OutputdTBicubic, float& time)
/*Input: vector of image intensity values, image width and height (with 1 pixel border).
 Output: Image gradients: Rx, Ry, Tx, Ty, Txy, BicubicMatrix
Purpose: Precomputation Interface function for CPU use
*/
{
	StopWatchWin PrecomputeWatch;
	float *d_InputIMGR, *d_InputIMGT,*d_InputBiubicMatrix;
	float *d_OutputIMGTx, *d_OutputIMGTy, *d_OutputIMGTxy;
	
	const static float h_InputBicubicMatrix[16*16] = {  
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
	
	PrecomputeWatch.start();
	checkCudaErrors(cudaMalloc((void**)&d_InputIMGR, (width+2)*(height+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_InputIMGT, (width+2)*(height+2)*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_InputBiubicMatrix, 16*16*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTx, width*height*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTy, width*height*sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTxy, width*height*sizeof(float)));

	
	checkCudaErrors(cudaMemcpy(d_InputIMGR,&h_InputIMGR[0],(width+2)*(height+2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_InputIMGT,&h_InputIMGT[0],(width+2)*(height+2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_InputBiubicMatrix,h_InputBicubicMatrix,16*16*sizeof(float),cudaMemcpyHostToDevice));
	
	

	dim3 dimB(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 dimG((width-1)/BLOCK_SIZE+1,(height-1)/BLOCK_SIZE+1,1);

	precomputation_kernel<<<dimG, dimB>>>(d_InputIMGR,d_InputIMGT,d_InputBiubicMatrix,
								 d_OutputIMGR, d_OutputIMGT,  d_OutputIMGRx, d_OutputIMGRy,
								 d_OutputIMGTx, d_OutputIMGTy, d_OutputIMGTxy,d_OutputdTBicubic,
								 width, height);

	
	PrecomputeWatch.stop();
	time = PrecomputeWatch.getTime();

	cudaFree(d_OutputIMGTx);
	cudaFree(d_OutputIMGTy);
	cudaFree(d_OutputIMGTxy);
	cudaFree(d_InputIMGR);
	cudaFree(d_InputIMGT);
	cudaFree(d_InputBiubicMatrix);
}
//ICGN Interface function
void ICGN_interface(float* dInput_dPXY, float* dInput_dR, float* dInput_dRx, float* dInput_dRy, float* dInput_dT, float* dInput_Bicubic, float fDeltaP,
					int* dInput_iU, int* dInput_iV, int iNumberY, int iNumberX, int iSubsetH, int iSubsetW, int width, int height, int iSubsetY, int iSubsetX, int iIterationNum,
					float* dOutput_dP, float& time)
{
	StopWatchWin ICGNWatch;
	
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE,1);
	dim3 dimGrid((iNumberX-1)/BLOCK_SIZE+1, (iNumberY-1)/BLOCK_SIZE+1,1);

	ICGNWatch.start();
	ICGN_kernel<<<dimGrid, dimBlock>>>(dInput_dPXY, dInput_dR, dInput_dRx, dInput_dRy, fDeltaP, dInput_dT, dInput_Bicubic, dInput_iU, dInput_iV, 
		iNumberY, iNumberX, iSubsetH, iSubsetW, width, height, iSubsetY, iSubsetX, iIterationNum, dOutput_dP);
	ICGNWatch.stop();
	time = ICGNWatch.getTime();
}



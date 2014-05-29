#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "EasyBMP.h"
#include "PIDIC.cuh"
#include "FFTCC.h"

#include <stdio.h>
#include <iostream>

using namespace std;

//Parameters
const int iMarginX = 10,	iMarginY = 10;
const int iGridX = 10,		iGridY = 10;
const int iSubsetX = 16,	iSubsetY =16;
const float fDeltaP = 0.001f;
const int iIterationNum = 20;

const int BLOCK_SIZE = 16;

//CUDA kernels for parallel computation
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
__global__ void ICGN_kernel(float* fInput_dPXY, float* fInput_dR, float* fInput_dRx, float* fInput_dRy, float fDeltaP, float* fInput_dT, float* fInput_Bicubic, int* iInput_iU, int* iInput_iV,
							int iNumberY, int iNumberX, int iSubsetH, int iSubsetW, int width, int height, int iSubsetY, int iSubsetX, int iIterationNum, float* fOutput_dP)
{
	int row = blockIdx.y*blockDim.y+threadIdx.y;
	int col = blockIdx.x*blockDim.x+threadIdx.x;
	int offset = row*iNumberX+col;

	//Used variables
	int iTemp, iTempX, iTempY;
	float fTemp, fTempX, fTempY;
	float fdU, fdV, fdUx, fdUy, fdVx, fdVy;
	float fdDU, fdDUx, fdDUy, fdDV, fdDVx, fdDVy;
	float fSubAveR = 0.0, fSubNormR = 0.0;
	float fSubAveT, fSubNormT;
	float fWarpX, fWarpY;
	float fdP[6], fdWarp[3][3], fJacobian[2][6], fHessian[6][6], fHessianXY[6][6], fInvHessian[6][6], fdPXY[2], fNumerator[6];
	
	float fSubsetR[33*33], fSubsetT[33*33];
	float fSubsetAveR[33*33], fSubsetAveT[33*33];
	float fRDescent[33*33*6];
	float fError;

	//if((row<iNumberY) && (col<iNumberX)){
		fdU = float(iInput_iU[offset]); fdV = float(iInput_iV[offset]);	fdUx = 0.0; fdUy = 0.0; fdVx = 0.0; fdVy = 0.0;
		fdP[0] = fdU, fdP[1] = fdUx, fdP[2] = fdUy, fdP[3] = fdV, fdP[4] = fdVx, fdP[5] = fdVy;
		fdPXY[0] = fInput_dPXY[(offset)*2+0], fdPXY[1] = fInput_dPXY[(offset)*2+1];
		fdWarp[0][0] = 1+fdUx, fdWarp[0][1] = fdUy, fdWarp[0][2] = fdU, fdWarp[1][0] = fdVx, fdWarp[1][1] = 1+fdVy, fdWarp[1][2] = fdV, fdWarp[2][0] = 0.0, fdWarp[2][1] = 0.0, fdWarp[2][2] = 1.0;

		//Initialize the Hessian matrix in subsetR
		for(int k=0; k<6; k++){
			for(int n=0; n<6; n++){
				fHessian[k][n] = 0.0;
			}
		}
		//Fill the gray intensity value to subset R
		for(int l=0; l<iSubsetH; l++){
			for(int m=0; m<iSubsetW; m++){
				fSubsetR[l*iSubsetW+m] = fInput_dR[int(fdPXY[0] - iSubsetY+l)*width+int(fdPXY[1] - iSubsetX+m)];
				fSubAveR += (fSubsetR[l*iSubsetW+m]/float(iSubsetH * iSubsetW));
				//Evaluate the Jacobian dW/dp at(x,0)
				fJacobian[0][0] = 1.0, fJacobian[0][1] = float(m-iSubsetX), fJacobian[0][2] = float(l-iSubsetY), fJacobian[0][3] = 0.0, fJacobian[0][4] = 0.0, fJacobian[0][5] = 0.0;
				fJacobian[1][0] = 0.0, fJacobian[1][1] = 0.0, fJacobian[1][2] = 0.0, fJacobian[1][3] = 1.0, fJacobian[1][4] = float(m-iSubsetX), fJacobian[1][5] = float(l-iSubsetY);
				for(int k=0; k<6; k++){
					fRDescent[(l*iSubsetW+m)*6+k] = fInput_dRx[int(fdPXY[0] - iSubsetY+l)*width+int(fdPXY[1] - iSubsetX+m)]*fJacobian[0][k]
												   +fInput_dRy[int(fdPXY[0] - iSubsetY+l)*width+int(fdPXY[1] - iSubsetX+m)]*fJacobian[1][k];
				}
				for(int k=0; k<6; k++){
					for(int n=0; n<6; n++){
						fHessianXY[k][n] = fRDescent[(l*iSubsetW+m)*6+k] * fRDescent[(l*iSubsetW+m)*6+n];	//Hessian matrix at each point
						fHessian[k][n] += fHessianXY[k][n];
					}
				}
			}
		}
		for(int l=0; l<iSubsetH; l++){
			for(int m=0; m<iSubsetW; m++){
				fSubsetAveR[l*iSubsetW+m] = fSubsetR[l*iSubsetW+m] - fSubAveR;
				fSubNormR += pow(fSubsetAveR[l*iSubsetW+m],2);
			}
		}
		fSubNormR = sqrt(fSubNormR);
		//Inverse the Hessian matrix
		fInvHessian[0][0] = 1.0f, fInvHessian[1][1] = 1.0f, fInvHessian[2][2] = 1.0f, fInvHessian[3][3] = 1.0f, fInvHessian[4][4] = 1.0f, fInvHessian[5][5] = 1.0f;
		fInvHessian[0][1] = 0.0f, fInvHessian[0][2] = 0.0f, fInvHessian[0][3] = 0.0f, fInvHessian[0][4] = 0.0f, fInvHessian[0][5] = 0.0f;
		fInvHessian[1][0] = 0.0f, fInvHessian[1][2] = 0.0f, fInvHessian[1][3] = 0.0f, fInvHessian[1][4] = 0.0f, fInvHessian[1][5] = 0.0f;
		fInvHessian[2][0] = 0.0f, fInvHessian[2][1] = 0.0f, fInvHessian[2][3] = 0.0f, fInvHessian[2][4] = 0.0f, fInvHessian[2][5] = 0.0f;
		fInvHessian[3][0] = 0.0f, fInvHessian[3][2] = 0.0f, fInvHessian[3][1] = 0.0f, fInvHessian[3][4] = 0.0f, fInvHessian[3][5] = 0.0f;
		fInvHessian[4][0] = 0.0f, fInvHessian[4][2] = 0.0f, fInvHessian[4][1] = 0.0f, fInvHessian[4][3] = 0.0f, fInvHessian[4][5] = 0.0f;
		fInvHessian[5][0] = 0.0f, fInvHessian[5][1] = 0.0f, fInvHessian[5][2] = 0.0f, fInvHessian[5][3] = 0.0f, fInvHessian[5][4] = 0.0f;
		/*for(int l=0; l<6; l++){
			for(int m=0; m<6; m++){
				if( l==m ){
					fInvHessian[l][m] = 1.0f;
				}
				else{
					fInvHessian[l][m] = 0.0f;
				}
			}
		}*/
		for(int l=0; l<6; l++){
			iTemp = 1;
			for(int m=l+1; m<6; m++){
				if(fHessian[m][l] > fHessian[iTemp][l]){
					iTemp = m;
				}
			}
			if(iTemp != l){
				for(int k=0; k<6; k++){
					fTemp = fHessian[l][k];
					fHessian[l][k] = fHessian[iTemp][k];
					fHessian[iTemp][k] = fTemp;
					
					fTemp = fInvHessian[l][k];
					fInvHessian[l][k] = fInvHessian[iTemp][k];
					   fInvHessian[iTemp][k] = fTemp;
				}
			}
			for(int m=0; m<6; m++){
				fTemp = fHessian[m][l]; 
				if(m != l){
					for(int n=0; n<6; n++){
						fInvHessian[m][n] -= fInvHessian[l][n] * fTemp / fHessian[l][l];
						fHessian[m][n]    -= fHessian[l][n] * fTemp / fHessian[l][l];
					}
				}
				else{
					for(int n=0; n<6; n++){
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
			for(int l=0; l<iSubsetH; l++){
				for(int m=0; m<iSubsetW; m++){
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
							for(int k=0; k<4; k++){
								for(int n=0; n<4; n++){
									fSubsetT[l*iSubsetW+m] += fInput_Bicubic[((iTempY*width+iTempX)*4+k)*4+n]*pow(fTempY,k)*pow(fTempX,n);
								}
							}
						}
						fSubAveT += (fSubsetT[l*iSubsetW+m]/(iSubsetH*iSubsetW));
					}
				}
			}
			for(int l=0; l<iSubsetH; l++){
				for(int m=0; m<iSubsetW; m++){
					fSubsetAveT[l*iSubsetW+m] = fSubsetT[l*iSubsetW+m] - fSubAveT;
					fSubNormT += pow(fSubsetAveT[l*iSubsetW+m],2);
				}
			}
			fSubNormT = sqrt(fSubNormT);
			//Compute the error image
			for(int k=0; k<6; k++){
				fNumerator[k] = 0.0f;
			}
			for(int l=0; l<iSubsetH; l++){
				for(int m=0; m<iSubsetW; m++){
					fError = (fSubNormR / fSubNormT) * fSubsetAveT[l*iSubsetW+m] * fSubsetAveR[l*iSubsetW+m];
					for(int k=0; k<6; k++){
						fNumerator[k] += (fRDescent[(l*iSubsetW+m)*6+k] * fError);
					}
				}
			}
			//Compute DeltaP
			for(int k=0; k<6; k++){
				fdP[k] = 0.0f;
				for(int n=0; n<6; n++){
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
		fOutput_dP[(offset)*6+0] = fdP[0];
		fOutput_dP[(offset)*6+1] = fdP[1];
		fOutput_dP[(offset)*6+2] = fdP[2];
		fOutput_dP[(offset)*6+3] = fdP[3];
		fOutput_dP[(offset)*6+4] = fdP[4];
		fOutput_dP[(offset)*6+5] = fdP[5];

	//}
}

//CUDA RUNTIME Initialization
void InitCuda()
{
	cudaFree(0);
}

//Computation Function interface
void computation(const float* ImgR, const float* ImgT, int iWidth, int iHeight)
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


	
	dim3 dimB(iNumberY,iNumberX,1);
	dim3 dimG(1,1,1);

	ICGN_kernel<<<dimG,dimB>>>(dInput_fPXY,d_OutputIMGR,d_OutputIMGRx,d_OutputIMGRy,fDeltaP,d_OutputIMGT,d_OutputBicubic,dInput_iU,dInput_iV,
		iNumberY,iNumberX,iSubsetH,iSubsetW,width,height,iSubsetY,iSubsetX,iIterationNum,dOutput_fDP);
	float *fdP	 = (float*)malloc(iNumberX*iNumberY*6*sizeof(float));
	checkCudaErrors(cudaMemcpy(fdP, dOutput_fDP, iNumberY*iNumberX*6*sizeof(float), cudaMemcpyDeviceToHost));
	WatchICGN.stop();
	fTimeICGN = WatchICGN.getTime();



	ofstream OutputFile;
	OutputFile.open("Results.txt");
	for(int i =0; i<iNumberY; i++){
		for(int j=0; j<iNumberX; j++){
			OutputFile<<int(fdPXY[(i*iNumberX+j)*2+1])<<", "<<int(fdPXY[(i*iNumberX+j)*2+0])<<", "<<iU[i*iNumberX+j]<<", "
				<<fdP[(i*iNumberX+j)*6+0]<<", "<<fdP[(i*iNumberX+j)*6+1]<<", "<<fdP[(i*iNumberX+j)*6+2]<<", "<<fdP[(i*iNumberX+j)*6+3]<<", "<<iV[i*iNumberX+j]<<", "<<fdP[(i*iNumberX+j)*6+4]<<", "<<fdP[(i*iNumberX+j)*6+5]<<", "
				<<fZNCC[i*iNumberX+j]<<endl;
		}
	}
	OutputFile.close();	

	OutputFile.open("Time.txt");
	OutputFile << "Interval (X-axis): " << iGridX << " [pixel]" << endl;
	OutputFile << "Interval (Y-axis): " << iGridY << " [pixel]" << endl;
	OutputFile << "Number of POI: " << iNumberY*iNumberX << " = " << iNumberX << " X " << iNumberY << endl;
	OutputFile << "Subset dimension: " << iSubsetW << "x" << iSubsetH << " pixels" << endl;
	OutputFile << "Time comsumed: " << fTimeTotal << " [millisec]" << endl;
	OutputFile << "Time for Pre-computation: " << fTimePrecopmute << " [millisec]" << endl;
	OutputFile << "Time for integral-pixel registration: " << fTimeFFTCC / (iNumberY*iNumberX) << " [millisec]" << endl;
	OutputFile << "Time for sub-pixel registration: " << fTimeICGN / (iNumberY*iNumberX) << " [millisec]" << " for average iteration steps of " << float(iIterationNum) / (iNumberY*iNumberX) << endl;
	OutputFile << width << ", " << height << ", " << iGridX << ", " << iGridY << ", " << endl;

	OutputFile <<"Time for computing every FFT:"<<fTimeFFTCC<<"[miliseconds]"<<endl;
	OutputFile <<"Time for ICGN:"<<fTimeICGN<<endl;

	OutputFile.close();


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


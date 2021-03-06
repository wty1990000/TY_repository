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
#include <fstream>

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
__global__ void ICGN_kernel(float* input_R, float* input_Rx, float* input_Ry, float* input_AveR, float* input_NormR, float fDeltaP,
							float *input_T,   float* input_Bicubic, int* input_iU, int* input_iV, 
							int iNumberY, int iNumberX, int iSubsetH, int iSubsetW, int width, int height, int iSubsetY, int iSubsetX, 
							int iGridSpaceX, int iGridSpaceY, int iMarginX, int iMarginY, int iIterationNum,
							float* output_dP)
/*BLOCK_SIZE: 2*(iSubsetW+1)+1, 2*(iSubsetH+1)+1
  Grid_SIZE:  iNumberX * iNumberY 
*/
{
	int x = threadIdx.x, y = threadIdx.y;
	int offset = blockIdx.y * gridDim.x + blockIdx.x;

	//Shared variables of each ROI
	__shared__ float fPXY[2];
	__shared__ float fT[19*19];
	__shared__ float fBicubic[19*19*4*4];
	__shared__ float fR[17*17];
	__shared__ float fRx[17*17];
	__shared__ float fRy[17*17];
	__shared__ float fdP[6], fdDP[6], fdWarp[3][3], fHessian[6][6],fInvHessian[6][6],fNumerator[6];
	__shared__ float fdU, fdV, fdUx, fdUy, fdVx, fdVy;
	__shared__ float fdDU, fdDUx, fdDUy, fdDV, fdDVx, fdDVy;
	__shared__ float fSubAveR, fSubNormR, fSubAveT, fSubNormT;
	__shared__ float fTemp;

	//Private variables for each subset point
	float fJacobian[2][6], fRDescent[6], fHessianXY[6][6];
	float fSubsetR, fSubsetAveR, fSubsetT, fSubsetAveT;
	float fdError;
	float fWarpX, fWarpY;
	int iTemp, iTempX, iTempY;
	float fTempX, fTempY;
	
	//Load the auxiliary variables into shared memory of each block
	if(x==0 && y ==0){
		fPXY[0] = float(iMarginX + iSubsetY + blockIdx.y * iGridSpaceY);
		fPXY[1] = float(iMarginY + iSubsetX + blockIdx.x * iGridSpaceX);
		
		fdU = float(input_iU[offset]);		fdDU = 0.0f;
		fdV = float(input_iV[offset]);		fdDV = 0.0f;
		fdUx = 0.0f;						fdDUx = 0.0f;
		fdUy = 0.0f;						fdDUy = 0.0f;
		fdVx = 0.0f;						fdDVx = 0.0f;
		fdVy = 0.0f;						fdDVy = 0.0f;

		fdP[0] = fdU;		fdP[3] = fdV;
		fdP[1] = fdUx;		fdP[4] = fdVx;
		fdP[2] = fdUy;		fdP[5] = fdVy;
	
		fdP[0] = 0.0f;		fdP[3] = 0.0f;
		fdP[1] = 0.0f;		fdP[4] = 0.0f;
		fdP[2] = 0.0f;		fdP[5] = 0.0f;

		fdWarp[0][0] = 1 + fdUx;	fdWarp[0][1] = fdUy;		fdWarp[0][2] = fdU;		
		fdWarp[1][0] = fdVx;		fdWarp[1][1] = 1 + fdVy;	fdWarp[1][2] = fdV;
		fdWarp[2][0] = 0.0f;		fdWarp[2][1] = 0.0f;		fdWarp[2][2] = 1.0f;

		fNumerator[0] = 0.0f;	fNumerator[1] = 0.0f;	fNumerator[2] = 0.0f;	fNumerator[3] = 0.0f;	fNumerator[4] = 0.0f;	fNumerator[5] = 0.0f;
		fdDP[0] = 0.0f;	fdDP[1] = 0.0f;	fdDP[2] = 0.0f;	fdDP[3] = 0.0f;	fdDP[4] = 0.0f;	fdDP[5] = 0.0f;

		fSubAveR = input_AveR[offset];	fSubNormR = input_NormR[offset];
		fSubsetAveT = 0.0f;				fSubNormT = 0.0f;
	}
	__syncthreads();
	if( x<6 && y<6){
		if( x == y){
			fInvHessian[y][x] = 1.0f;
			fHessian[y][x]  = 0.0f;
		}
		else{
			fInvHessian[y][x] = 0.0f;
			fHessian[y][x] = 0.0f;
		}
	}
	__syncthreads();
	
	//Load PXY, Rx, Ry and R, T, Bicubic  into shared_memory
	if( x<iSubsetW && y<iSubsetH ){
		 fR[y*iSubsetW+x] = input_R[int(fPXY[0] - iSubsetY + y)*width+int(fPXY[1] - iSubsetX + x)];	
		fRx[y*iSubsetW+x] = input_Rx[int(fPXY[0] - iSubsetY + y)*width+int(fPXY[1] - iSubsetX + x)];
		fRy[y*iSubsetW+x] = input_Ry[int(fPXY[0] - iSubsetY + y)*width+int(fPXY[1] - iSubsetX + x)];
	}
	__syncthreads();

	//Load T, Bicubic with additional 1 pixel wider on each side
	fT[y*iSubsetW+x] = input_T[int(fPXY[0] - (iSubsetY+1) + y)*width+int(fPXY[1] - (iSubsetX+1) + x)];
	for(int k=0; k<4; k++){
		for(int n=0; n<4; n++){
			fBicubic[((y*iSubsetW+x)*4+k)*4+n] = input_Bicubic[((int(fPXY[0] - (iSubsetY+1) + y)*width+int(fPXY[1] - (iSubsetX+1) + x))*4+k)*4+n];
		}
	}
	__syncthreads();

	//Start computing

	if( x<iSubsetW && y<iSubsetH){
		// Evaluate the Jacbian dW/dp at (x, 0);
		fJacobian[0][0] = 1;
		fJacobian[0][1] = x - iSubsetX;
		fJacobian[0][2] = y - iSubsetY;
		fJacobian[0][3] = 0;
		fJacobian[0][4] = 0;
		fJacobian[0][5] = 0;
		fJacobian[1][0] = 0;
		fJacobian[1][1] = 0;
		fJacobian[1][2] = 0;
		fJacobian[1][3] = 1;
		fJacobian[1][4] = x - iSubsetX;
		fJacobian[1][5] = y - iSubsetY;

		for(unsigned int i=0; i<6; i++){
			fRDescent[i] = fRx[y*iSubsetW+x] * fJacobian[0][i] + fRy[y*iSubsetW+x] * fJacobian[1][i];
		}
		for(unsigned int i=0; i<6; i++){
			for(unsigned int j=0; j<6; j++){
				fHessianXY[i][j] = fRDescent[i] * fRDescent[j];
				/*fHessian[i][j] += fHessianXY[i][j];*/		//This is bad code, cannot be all added to one index.
				atomicAdd(&fHessian[i][j], fHessianXY[i][j]);	//Must use this instead.
			}
		}
		fSubsetAveR = fSubsetR - fSubAveR;
	}
	__syncthreads();
	//Invert the Hessian matrix using the first 36 threads
	if(x ==0 && y ==0){
		for(int l=0; l<6; l++){
			iTemp = l;
			for(int m=l+1; m<6; m++){
				if(fHessian[m][l] > fHessian[iTemp][l]){
					iTemp = m;
				}
			}
			//Swap the row which has maximum lth column element
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
			//Row oerpation to form required identity matrix
			for(int m=0; m<6; m++){
				fTemp = fHessian[m][l];
				if(m != l){
					for(int n=0; n<6; n++){
						fInvHessian[m][n] -= fInvHessian[l][n] * fTemp / fHessian[l][l];
						fHessian[m][n] -= fHessian[l][n] * fTemp / fHessian[l][l];
					}
				}
				else{
					for(int n=0; n<6; n++){
						fInvHessian[m][n] /= fTemp;
						fHessian[m][n] /= fTemp;
					}
				}
			}
		}
	}
	for(int it=0; it < iIterationNum; it++){
		if( x==0 && y==0){
			fSubsetAveT = 0.0f;	fSubNormT = 0.0f;
			fNumerator[0] = 0.0f; fNumerator[1] = 0.0f; fNumerator[2] = 0.0f; fNumerator[3] = 0.0f; fNumerator[4] = 0.0f; fNumerator[5] = 0.0f; fNumerator[6] = 0.0f;
		}
		__syncthreads();

		if( (x<iSubsetW) && (y<iSubsetH) ){

			fWarpX = (iSubsetX+1) + fdWarp[0][0]*(x - iSubsetX) + fdWarp[0][1]*(y - iSubsetY) + fdWarp[0][2];
			fWarpY = (iSubsetY+1) + fdWarp[1][0]*(x - iSubsetX) + fdWarp[1][1]*(y - iSubsetY) + fdWarp[1][2];

			iTempX = int(fWarpX);
			iTempY = int(fWarpY);

			if( (iTempX>=0) && (iTempY>=0) && (iTempX<(iSubsetW+1)) && (iTempY<(iSubsetH+1)) ){
				fTempX = fWarpX - float(iTempX);
				fTempY = fWarpY - float(iTempY);
				//if it is integer-pixel location ,feed the intensity of T into subset T
				if( (fTempX <= 0.000001) && (fTempY <= 0.000001) ){
					fSubsetT = fT[iTempY*(iSubsetW+2)+iTempX];
				}
				else{
					fSubsetT = 0.0f;
					for(int k=0; k<4; k++){
						for(int n=0; n<4; n++){
							fSubsetT += fBicubic[((iTempY*(iSubsetW+2)+iTempX)*4+k)*4+n]*pow(fTempY,k)*pow(fTempX,n);
						}
					}
				}
				atomicAdd(&fSubAveT, fSubsetT/float(iSubsetH*iSubsetW));
				fSubsetAveT = fSubsetT - fSubAveT;
				__syncthreads();
				atomicAdd(&fSubNormT, pow(fSubAveT,2));
			}
		}
		if( (x==0) && (y==0) ){
			fSubNormT = sqrt(fSubNormT);
		}
		__syncthreads();
		if( (x<iSubsetW) && (y<iSubsetH) ){
			//Compute Error image
			fdError = (fSubNormR / fSubNormT) * fSubAveT - fSubsetAveR;
		}
		__syncthreads();
		if( x==0 && y==0){
			for(int i=0; i<6; i++){
				atomicAdd(&(fNumerator[i]), (fRDescent[i]*fdError));
			}
			for(int k=0; k<6; k++){
				fdDP[k] = 0.0;
				for(int n=0; n<6; n++){
					fdDP[k] += (fInvHessian[k][n] * fNumerator[n]);
				}
			}
			fdDU  = fdDP[0];
			fdDUx = fdDP[1];
			fdDUy = fdDP[2];
			fdDV  = fdDP[3];
			fdDVx = fdDP[4];
			fdDVy = fdDP[5];

			fTemp = (1+fdDUx) * (1+fdDVy) - fdDUy*fdDVx;

			fdWarp[0][0] = ((1 + fdUx) * (1 + fdDVy) - fdUy * fdDVx) / fTemp;
			fdWarp[0][1] = (fdUy * (1 + fdDUx) - (1 + fdUx) * fdDUy) / fTemp;
			fdWarp[0][2] = fdU + (fdUy * (fdDU * fdDVx - fdDV - fdDV * fdDUx) - (1 + fdUx) * (fdDU * fdDVy + fdDU - fdDUy * fdDV)) / fTemp;
			fdWarp[1][0] = (fdVx * (1 + fdDVy) - (1 + fdVy) * fdDVx) / fTemp;
			fdWarp[1][1] = ((1 + fdVy) * (1 + fdDUx) - fdVx * fdDUy) / fTemp;
			fdWarp[1][2] = fdV + ((1 + fdVy) * (fdDU * fdDVx - fdDV - fdDV * fdDUx) - fdVx * (fdDU * fdDVy + fdDU - fdDUy * fdDV)) / fTemp;
			fdWarp[2][0] = 0;
			fdWarp[2][1] = 0;
			fdWarp[2][2] = 1;

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
		__syncthreads();
	}

	//Pass back the values
	if( x==0 && y==0 ){
		output_dP[offset*6+0] = fdP[0];
		output_dP[offset*6+1] = fdP[1];
		output_dP[offset*6+2] = fdP[2];
		output_dP[offset*6+3] = fdP[3];
		output_dP[offset*6+4] = fdP[4];
		output_dP[offset*6+5] = fdP[5];
	}
}


void computation_interface(const std::vector<float>& ImgR, const std::vector<float>& ImgT, int iWidth, int iHeight)
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
	checkCudaErrors(cudaMemcpy(d_InputIMGR,&ImgR[0],(width+2)*(height+2)*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_InputIMGT,&ImgT[0],(width+2)*(height+2)*sizeof(float),cudaMemcpyHostToDevice));
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


	/*Befor ICGN starts, pass the average and norm value of image R in first, to simplify GPU's work.*/
	thrust::host_vector<float> hAveR;
	thrust::host_vector<float> hNormR;

	float temp = 0.0f, temp1=0.0f, temp2 = 0.0f;
	for(int i=0; i<iNumberY; i++){
		for(int j=0; j<iNumberX; j++){
			temp = 0.0;
			for(int l=0; l<iSubsetH; l++){
				for(int m=0; m<iSubsetW; m++){
					temp += hInput_dR[int(fdPXY[(i*iNumberX+j)*2+0] - iSubsetY+l)*width
									+ int(fdPXY[(i*iNumberX+j)*2+1] - iSubsetX+m)] / float(iSubsetW*iSubsetH);
				}
			}
			hAveR.push_back(temp);
			temp1 = 0.0f, temp2 = 0.0f;
			for(int l=0; l<iSubsetH; l++){
				for(int m=0; m<iSubsetW; m++){
					temp1 = hInput_dR[int(fdPXY[(i*iNumberX+j)*2+0] - iSubsetY+l)*width
									+ int(fdPXY[(i*iNumberX+j)*2+1] - iSubsetX+m)] / float(iSubsetW*iSubsetH) - hAveR[i*iNumberX+j];
					temp2 += pow(temp1,2);
				}
			}
			hNormR.push_back(sqrt(temp2));
		}
	}
	thrust::device_vector<float> dAveR = hAveR;
	thrust::device_vector<float> dNormR = hNormR;
	float *dAveRaw = thrust::raw_pointer_cast(&dAveR[0]);
	float *dNormRaw = thrust::raw_pointer_cast(&dNormR[0]);

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

	dim3 dimB((iSubsetW+2),(iSubsetH+2),1);
	dim3 dimG(iNumberX,iNumberY,1);

	ICGN_kernel<<<dimG, dimB>>>(d_OutputIMGR,d_OutputIMGRx,d_OutputIMGRy,dAveRaw, dNormRaw,fDeltaP,d_OutputIMGT,d_OutputBicubic,dInput_iU,dInput_iV,
		iNumberY, iNumberX, iSubsetH, iSubsetW, width, height, iSubsetY, iSubsetX, iGridX, iGridY, iMarginX, iMarginY, iIterationNum, dOutput_fDP);

	float *fdP = (float*)malloc(iNumberX*iNumberY*6*sizeof(float));
	checkCudaErrors(cudaMemcpy(fdP, dOutput_fDP, iNumberY*iNumberX*6*sizeof(float), cudaMemcpyDeviceToHost));
	WatchICGN.stop();
	fTimeICGN = WatchICGN.getTime();
	
	WatchTotal.stop();
	fTimeTotal = WatchTotal.getTime();

	checkCudaErrors(cudaFree(d_OutputIMGR));
	checkCudaErrors(cudaFree(d_OutputIMGRx));
	checkCudaErrors(cudaFree(d_OutputIMGRy));
	checkCudaErrors(cudaFree(d_OutputIMGT));
	checkCudaErrors(cudaFree(d_OutputBicubic));
	checkCudaErrors(cudaFree(dInput_iU));
	checkCudaErrors(cudaFree(dInput_iV));
	checkCudaErrors(cudaFree(dInput_fPXY));
	checkCudaErrors(cudaFree(dOutput_fDP));

	std::ofstream OutputFile;
	OutputFile.open("Results.txt");
	for(int i =0; i<iNumberY; i++){
		for(int j=0; j<iNumberX; j++){
			OutputFile<<int(fdPXY[(i*iNumberX+j)*2+1])<<", "<<int(fdPXY[(i*iNumberX+j)*2+0])<<", "<<iU[i*iNumberX+j]<<", "
				<<fdP[(i*iNumberX+j)*6+0]<<", "<<fdP[(i*iNumberX+j)*6+1]<<", "<<fdP[(i*iNumberX+j)*6+2]<<", "<<fdP[(i*iNumberX+j)*6+3]<<", "<<iV[i*iNumberX+j]<<", "<<fdP[(i*iNumberX+j)*6+4]<<", "<<fdP[(i*iNumberX+j)*6+5]<<", "
				<<fZNCC[i*iNumberX+j]<<std::endl;
		}
	}
	OutputFile.close();	

	OutputFile.open("Time.txt");
	OutputFile << "Interval (X-axis): " << iGridX << " [pixel]" << std::endl;
	OutputFile << "Interval (Y-axis): " << iGridY << " [pixel]" << std::endl;
	OutputFile << "Number of POI: " << iNumberY*iNumberX << " = " << iNumberX << " X " << iNumberY << std::endl;
	OutputFile << "Subset dimension: " << iSubsetW << "x" << iSubsetH << " pixels" << std::endl;
	OutputFile << "Time comsumed: " << fTimeTotal << " [millisec]" <<std::endl;
	OutputFile << "Time for Pre-computation: " << fTimePrecopmute << " [millisec]" << std::endl;
	OutputFile << "Time for integral-pixel registration: " << fTimeFFTCC / (iNumberY*iNumberX) << " [millisec]" << std::endl;
	OutputFile << "Time for sub-pixel registration: " << fTimeICGN / (iNumberY*iNumberX) << " [millisec]" << " for average iteration steps of " << float(iIterationNum) / (iNumberY*iNumberX) << std::endl;
	OutputFile << width << ", " << height << ", " << iGridX << ", " << iGridY << ", " << std::endl;

	OutputFile <<"Time for computing every FFT:"<<fTimeFFTCC<<"[miliseconds]"<<std::endl;
	OutputFile <<"Time for ICGN:"<<fTimeICGN<<std::endl;

	OutputFile.close();

	free(hInput_dR);
	free(hInput_dT);
	free(fZNCC);
	free(fdP);
	free(fdPXY);
	free(iU);
	free(iV);
}
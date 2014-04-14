#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "IC_GN.cuh"

#include <stdio.h>

__global__ void computeICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, float m_dNormDeltaP,
							const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV,
							int m_iNumberY, int m_iNumberX, int m_iSubsetH, int m_iSubsetW,	int m_iWidth, int m_iHeight, int m_iSubsetY, int m_iSubsetX, int m_iMaxiteration,
							float* output_dP, int* m_iIterationNum)
/*Input: all the const variables
 Output: deformation P matrix
Strategy: Each block compute one of the 21*21 POIs, and within each block 32*32 threads compute other needed computations
*/
{
	unsigned int blockID = blockIdx.y * gridDim.x + blockIdx.x;  //Row * dim + Col
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;

	// Shared memory for the shared variables of each block
	__shared__ float m_dU, m_dV, m_dUx, m_dUy, m_dVx, m_dVy;	//Initial guess
	__shared__ float m_dDU, m_dDV, m_dDUx, m_dDUy, m_dDVx, m_dDVy;
	__shared__ float m_dSubAveR, m_dSubNorR, m_dSubAveT, m_dSubNorT;					//Used for atomicAdd
	__shared__ float m_dP[6], m_dDP[6], m_dPXY[2], m_dNumerator[6];
	__shared__ float m_dWarp[3][3]; 
	__shared__ float m_dHessian[6][6], m_dInvHessian[6][6];
	__shared__ float m_dTemp;
	//__shared__ int m_iIteration;
	// Local memory for each thread use
	float m_dSubsetR, m_dSubsetAveR, m_dSubsetT, m_dSubsetAveT;
	float m_dJacobian[2][6], m_dRDescent[6], m_dHessianXY[6][6];
	int m_iTemp, m_iTempX, m_iTempY;
	float m_dTempX, m_dTempY;
	float m_dWarpX, m_dWarpY;
	float m_dError;
	

	//Load the shared variables to shared memory
	if(tx ==0 && ty==0){
		m_dU = float(input_iU[blockID]);		m_dDU = 0.0;
		m_dV = float(input_iV[blockID]);		m_dDV = 0.0;
		m_dUx = 0.0;							m_dUx = 0.0;
		m_dUy = 0.0;							m_dUy = 0.0;
		m_dVx = 0.0;							m_dVx = 0.0;
		m_dVy = 0.0;							m_dVy = 0.0;

		m_dP[0] = m_dU;		m_dP[3] = m_dV;
		m_dP[1] = m_dUx;	m_dP[4] = m_dVx;
		m_dP[2] = m_dUy;	m_dP[5] = m_dVy;

		m_dWarp[0][0] = 1 + m_dUx;	m_dWarp[0][1] = m_dUy;		m_dWarp[0][2] = m_dU;		
		m_dWarp[1][0] = m_dVx;		m_dWarp[1][1] = 1 + m_dVy;	m_dWarp[1][2] = m_dV;
		m_dWarp[2][0] = 0;			m_dWarp[2][1] = 0;			m_dWarp[2][2] = 1;
		
		m_dPXY[0] = input_dPXY[blockID*2+0];	m_dPXY[1] = input_dPXY[blockID*2+1];
		m_dSubAveR=0.0;		m_dSubNorR=0.0;
		m_dSubAveT=0.0;		m_dSubNorT=0.0;		
	}
	if(tx<6 && ty<6){
		if(tx == ty){
			m_dInvHessian[ty][tx] = 1.0;

			m_dHessian[ty][tx] = 0.0;
			m_dNumerator[tx] = 0.0;
			m_dDP[tx] = 0.0;
		}
		else{
			m_dInvHessian[ty][tx] = 0.0;

			m_dHessian[ty][tx] = 0.0;
			m_dNumerator[tx] = 0.0;
			m_dDP[tx] = 0.0;
		}	
	}
	__syncthreads();
		
	// Evaluate the Jacbian dW/dp at (x, 0);
	m_dJacobian[0][0] = 1;
	m_dJacobian[0][1] = tx - m_iSubsetX;
	m_dJacobian[0][2] = ty - m_iSubsetY;
	m_dJacobian[0][3] = 0;
	m_dJacobian[0][4] = 0;
	m_dJacobian[0][5] = 0;
	m_dJacobian[1][0] = 0;
	m_dJacobian[1][1] = 0;
	m_dJacobian[1][2] = 0;
	m_dJacobian[1][3] = 1;
	m_dJacobian[1][4] = tx - m_iSubsetX;
	m_dJacobian[1][5] = ty - m_iSubsetY;

	//Compute the steepest descent image and Hessian matrix
	for(unsigned int i=0; i<6; i++){
		m_dRDescent[i] = input_mdRx[int(m_dPXY[0] - m_iSubsetY+ty)*m_iWidth+int(m_dPXY[1] - m_iSubsetX+tx)] * m_dJacobian[0][i]
							+ input_mdRy[int(m_dPXY[0] - m_iSubsetY+ty)*m_iWidth+int(m_dPXY[1] - m_iSubsetX+tx)] * m_dJacobian[1][i];
	}
	for(unsigned int i=0; i<6; i++){
			for(unsigned int j=0; j<6; j++){
			m_dHessianXY[i][j] = m_dRDescent[i] * m_dRDescent[j];
			m_dHessian[i][j] += m_dHessianXY[i][j];
		}
	}

	//Fill the intensity values in the subset R
	m_dSubsetR = input_mdR[int(m_dPXY[0] - m_iSubsetY + ty)*m_iWidth+int(m_dPXY[1] - m_iSubsetX + tx)];
	__syncthreads();
	atomicAdd(&m_dSubAveR, (m_dSubsetR/float(m_iSubsetH*m_iSubsetW))); 
	__syncthreads();
	m_dSubsetAveR = m_dSubsetR - m_dSubAveR;
	__syncthreads();
	atomicAdd(&m_dSubNorR, pow(m_dSubsetAveR,2));
	__syncthreads();

	//Invert the Hessian matrix using the first 36 threads
	if(tx ==0 && ty ==0){
		m_dSubNorR = sqrt(m_dSubNorR);
		__syncthreads();
		for(int l=0; l<6; l++){
			m_iTemp = l;
			for(int m=l+1; m<6; m++){
				if(m_dHessian[m][l] > m_dHessian[m_iTemp][l]){
					m_iTemp = m;
				}
			}
			//Swap the row which has maximum lth column element
			if(m_iTemp != l){
				for(int k=0; k<6; k++){
					m_dTemp = m_dHessian[l][k];
					m_dHessian[l][k] = m_dHessian[m_iTemp][k];
					m_dHessian[m_iTemp][k] = m_dTemp;

					m_dTemp = m_dInvHessian[l][k]; 
					m_dInvHessian[l][k] = m_dInvHessian[m_iTemp][k];
					m_dInvHessian[m_iTemp][k] = m_dTemp;
				}
			}
			//Row oerpation to form required identity matrix
			for(int m=0; m<6; m++){
				m_dTemp = m_dHessian[m][l];
				if(m != l){
					for(int n=0; n<6; n++){
						m_dInvHessian[m][n] -= m_dInvHessian[l][n] * m_dTemp / m_dHessian[l][l];
						m_dHessian[m][n] -= m_dHessian[l][n] * m_dTemp / m_dHessian[l][l];
					}
				}
				else{
					for(int n=0; n<6; n++){
						m_dInvHessian[m][n] /= m_dTemp;
						m_dHessian[m][n] /= m_dTemp;
					}
				}
			}
		}
	}
	__syncthreads();

	//Perform iterative optimization, within the maximum iteration number
	for(int m_iIteration =0; m_iIteration < m_iMaxiteration; m_iIteration++){
		if(tx ==0 && ty==0){
			m_dSubAveT=0.0;		m_dSubNorT=0.0;	
			m_dNumerator[0] = 0.0;	m_dNumerator[1] = 0.0;	m_dNumerator[2] = 0.0;	m_dNumerator[3] = 0.0;	m_dNumerator[4] = 0.0;	m_dNumerator[5] = 0.0;
		}
		__syncthreads();
		m_dWarpX = m_dPXY[1] + m_dWarp[0][0] * (tx - m_iSubsetX) + m_dWarp[0][1] * (ty - m_iSubsetY) + m_dWarp[0][2];
		m_dWarpY = m_dPXY[0] + m_dWarp[1][0] * (tx - m_iSubsetX) + m_dWarp[1][1] * (ty - m_iSubsetY) + m_dWarp[1][2];

		m_iTempX = int(m_dWarpX);
		m_iTempY = int(m_dWarpY);
		m_dTempX = m_dWarpX - float(m_iTempX);
		m_dTempY = m_dWarpY - float(m_iTempY);
		//if it is integer-pixel location ,feed the intensity of T into subset T
		if((m_dTempX ==0.0) && (m_dTempY ==0.0)){
			m_dSubsetT = input_mdT[m_iTempY * m_iWidth + m_iTempX];
		}
		else{
			m_dSubsetT =0.0;
			for(int k=0; k<4; k++){
				for(int n=0; n<4; n++){
					m_dSubsetT  += input_mBicubic[((m_iTempY*m_iWidth+m_iTempX)*4+k)*4+n] * pow(m_dTempY, k) * pow(m_dTempX,n);
				}
			}
		}
		
			__syncthreads();
			atomicAdd(&m_dSubAveT, m_dSubsetT/float(m_iSubsetH*m_iSubsetW));
			__syncthreads();
			m_dSubsetAveT = m_dSubsetT - m_dSubAveT;
			__syncthreads();
			atomicAdd(&m_dSubNorT, pow(m_dSubsetAveT,2));
			__syncthreads();

		if(tx==0 && ty==0){
			m_dSubNorT = sqrt(m_dSubNorT);
		}
		//Compute the error image
		m_dError = (m_dSubNorR / m_dSubNorT) * m_dSubsetAveT - m_dSubsetAveR;
		__syncthreads();

		if(tx<6 && ty==0){
			atomicAdd(&(m_dNumerator[tx]),(m_dRDescent[tx] * m_dError));
		}
		__syncthreads();
		//Compute DeltaP
		if(tx==0 && ty==0){
			for(int k=0; k<6; k++){
				m_dDP[k] = 0.0;
				for(int n=0; n<6; n++){
					m_dDP[k] += (m_dInvHessian[k][n] * m_dNumerator[n]);
				}
			}
			__syncthreads();
			m_dDU  = m_dDP[0];
			m_dDUx = m_dDP[1]; 
			m_dDUy = m_dDP[2];
			m_dDV  = m_dDP[3];
			m_dDVx = m_dDP[4];
			m_dDVy = m_dDP[5];

			m_dTemp = (1+m_dDUx) * (1+m_dDVy) - m_dDUy * m_dDVx;

			//W(P) <- W(P) O W(DP)^-1
			m_dWarp[0][0] = ((1 + m_dUx) * (1 + m_dDVy) - m_dUy * m_dDVx) / m_dTemp;
			m_dWarp[0][1] = (m_dUy * (1 + m_dDUx) - (1 + m_dUx) * m_dDUy) / m_dTemp;
			m_dWarp[0][2] = m_dU + (m_dUy * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - (1 + m_dUx) * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
			m_dWarp[1][0] = (m_dVx * (1 + m_dDVy) - (1 + m_dVy) * m_dDVx) / m_dTemp;
			m_dWarp[1][1] = ((1 + m_dVy) * (1 + m_dDUx) - m_dVx * m_dDUy) / m_dTemp;
			m_dWarp[1][2] = m_dV + ((1 + m_dVy) * (m_dDU * m_dDVx - m_dDV - m_dDV * m_dDUx) - m_dVx * (m_dDU * m_dDVy + m_dDU - m_dDUy * m_dDV)) / m_dTemp;
			m_dWarp[2][0] = 0;
			m_dWarp[2][1] = 0;
			m_dWarp[2][2] = 1;

			// Update DeltaP
			m_dP[0] = m_dWarp[0][2];
			m_dP[1] = m_dWarp[0][0] - 1;
			m_dP[2] = m_dWarp[0][1];
			m_dP[3] = m_dWarp[1][2];
			m_dP[4] = m_dWarp[1][0];
			m_dP[5] = m_dWarp[1][1] - 1;

			m_dU = m_dP[0];
			m_dUx = m_dP[1];
			m_dUy = m_dP[2];
			m_dV = m_dP[3];
			m_dVx = m_dP[4];
			m_dVy = m_dP[5];
			__syncthreads();

			/*if(sqrt(pow(m_dDP[blockID*6+0],2)+pow(m_dDP[blockID*6+1]*m_iSubsetX,2)+pow(m_dDP[blockID*6+2] * m_iSubsetY,2) 
				+pow(m_dDP[blockID*6+3],2)+pow(m_dDP[blockID*6+4]*m_iSubsetX,2)+pow(m_dDP[blockID*6+5]*m_iSubsetY,2))<m_dNormDeltaP){
					break;
			}*/
		}
	}
	__syncthreads();

	if(tx==0 && ty==0){
		m_iIterationNum[blockID] = 20;
		output_dP[blockID*6+0] = m_dP[0];
		output_dP[blockID*6+1] = m_dP[1];
		output_dP[blockID*6+2] = m_dP[2];
		output_dP[blockID*6+3] = m_dP[3];
		output_dP[blockID*6+4] = m_dP[4];
		output_dP[blockID*6+5] = m_dP[5];
	}
}

void launch_ICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, const float& m_dNormDeltaP,
				 const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV,
				 const int& m_iNumberY, const int& m_iNumberX, const int& m_iSubsetH, const int& m_iSubsetW, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetY, const int& m_iSubsetX, const int& m_iMaxiteration,
				 float* output_dP, int* m_iIterationNum, float& time)
{
	StopWatchWin icgn;

	dim3 dimGrid(m_iNumberY, m_iNumberX,1);
	dim3 dimBlock(m_iSubsetH, m_iSubsetW,1);

	icgn.start();
	computeICGN<<<dimGrid,dimBlock>>>(input_dPXY, input_mdR, input_mdRx, input_mdRy, m_dNormDeltaP,
									  input_mdT, input_mBicubic, input_iU, input_iV, 
									  m_iNumberY, m_iNumberX, m_iSubsetH, m_iSubsetW, m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX, m_iMaxiteration,
									  output_dP,m_iIterationNum);

	icgn.stop();
	time = icgn.getTime();

}
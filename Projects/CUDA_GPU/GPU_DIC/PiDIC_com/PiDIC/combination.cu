#include "combination.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "FFT-CC.h"

#define BLOCK_SIZE 16

__global__ void computeICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, float m_dNormDeltaP,
							const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV, 
							int m_iNumberY, int m_iNumberX, int m_iSubsetH, int m_iSubsetW,	int m_iWidth, int m_iHeight, int m_iSubsetY, int m_iSubsetX, int m_iMaxiteration,
							float* output_dP, int* m_iIterationNum,
							float* m_dSubsetR, float* m_dSubsetT, float* m_dJacobian, float* m_dRDescent, float* m_dHessianXY,float* m_dSubsetAveR, float* m_dSubsetAveT, float* m_dError, float* m_dDP)
/*Input: all the const variables
 Output: deformation P matrix
Strategy: Each thread computes one of the 21*21 POIs
*/
{
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int offset = row*m_iNumberX + col;

	float m_dU, m_dV, m_dUx, m_dUy, m_dVx, m_dVy;
	float m_dDU, m_dDV, m_dDUx, m_dDUy, m_dDVx, m_dDVy;
	float m_dSubAveR,m_dSubNorR, m_dSubAveT,m_dSubNorT;
	float m_dWarp[3][3], m_dHessian[6][6], m_dInvHessian[6][6];
	float m_dNumerator[6];
	float m_dTemp;
	int m_iTemp;
	int m_iIteration;
	float m_dWarpX,	m_dWarpY;
	int m_iTempX,m_iTempY; 
	float m_dTempX, m_dTempY;
	

	if(row<m_iNumberY && col<m_iNumberX){
		m_dU = input_iU[offset];
		m_dV = input_iV[offset];
		m_dUx = 0;
		m_dUy = 0;
		m_dVx = 0;
		m_dVy = 0;

		output_dP[offset*6+0] = m_dU;
		output_dP[offset*6+1] = m_dUx;
		output_dP[offset*6+2] = m_dUy;
		output_dP[offset*6+3] = m_dV;
		output_dP[offset*6+4] = m_dVx;
		output_dP[offset*6+5] = m_dVy;

		// Initialize the warp matrix
		m_dWarp[0][0] = 1 + m_dUx;
		m_dWarp[0][1] = m_dUy;
		m_dWarp[0][2] = m_dU;
		m_dWarp[1][0] = m_dVx;
		m_dWarp[1][1] = 1 + m_dVy;
		m_dWarp[1][2] = m_dV;
		m_dWarp[2][0] = 0;
		m_dWarp[2][1] = 0;
		m_dWarp[2][2] = 1;

		// Initialize the Hessian matrix in subset R
		for (int k = 0; k < 6; k++)
		{
			for (int n = 0; n < 6; n++)
			{
				m_dHessian[k][n] = 0;
			}
		}

		// Initialize Subset R
		m_dSubAveR = 0; // R_m
		m_dSubNorR = 0; // T_m
		// Feed the gray intensity to subset R
		for (int l = 0; l < m_iSubsetH; l++)
		{
			for (int m = 0; m < m_iSubsetW; m++)
			{
				m_dSubsetR[l*m_iSubsetW+m] = input_mdR[int(input_dPXY[offset*2+0] - m_iSubsetY + l)*m_iWidth+int(input_dPXY[offset*2+1] - m_iSubsetX + m)];
				m_dSubAveR += (m_dSubsetR[l*m_iSubsetW+m] / (m_iSubsetH * m_iSubsetW));

				// Evaluate the Jacbian dW/dp at (x, 0);
				m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+0] = 1;
				m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+1] = m - m_iSubsetX;
				m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+2] = l - m_iSubsetY;
				m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+3] = 0;
				m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+4] = 0;
				m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+5] = 0;
				m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+0] = 0;
				m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+1] = 0;
				m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+2] = 0;
				m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+3] = 1;
				m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+4] = m - m_iSubsetX;
				m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+5] = l - m_iSubsetY;

				// Compute the steepest descent image DealtR*dW/dp
				for (int k = 0; k < 6; k++)
				{
					m_dRDescent[(l*m_iSubsetW+m)*6+k] = input_mdR[int(input_dPXY[offset*2+0] - m_iSubsetY + l)*m_iWidth+int(input_dPXY[offset*2+1] - m_iSubsetX + m)] * m_dJacobian[((l*m_iSubsetW+m)*2+0)*6+k] + input_mdRy[int(input_dPXY[offset*2+0] - m_iSubsetY + l)*m_iWidth+int(input_dPXY[offset*2+1] - m_iSubsetX + m)] * m_dJacobian[((l*m_iSubsetW+m)*2+1)*6+k];
				}

				// Compute the Hessian matrix
				for (int k = 0; k < 6; k++)
				{
					for (int n = 0; n < 6; n++)
					{
						m_dHessianXY[((l*m_iSubsetW+m)*6+k)*6+n] = m_dRDescent[(l*m_iSubsetW+m)*6+k] * m_dRDescent[(l*m_iSubsetW+m)*6+n]; // Hessian matrix at each point
						m_dHessian[k][n] += m_dHessianXY[((l*m_iSubsetW+m)*6+k)*6+n]; // sum of Hessian matrix at all the points in subset R
					}
				}
			}		
		}
		__syncthreads();
		for (int l = 0; l < m_iSubsetH; l++)
		{
			for (int m = 0; m < m_iSubsetW; m++)
			{
				m_dSubsetAveR[l*m_iSubsetW+m] = m_dSubsetR[l*m_iSubsetW+m] - m_dSubAveR; // R_i - R_m
				m_dSubNorR += pow(m_dSubsetAveR[l*m_iSubsetW+m], 2);
			}
		}
		__syncthreads();
		m_dSubNorR = sqrt(m_dSubNorR); // sqrt (Sigma(R_i - R_m)^2)

		// Invert the Hessian matrix (Gauss-Jordan algorithm)
		for (int l = 0; l < 6; l++)
		{
			for (int m = 0; m < 6; m++)
			{
				if (l == m)
				{
					m_dInvHessian[l][m] = 1;
				}
				else
				{
					m_dInvHessian[l][m] = 0;
				}
			}
		}
		__syncthreads();
		for (int l = 0; l < 6; l++)
		{
			//Find pivot (maximum lth column element) in the rest (6-l) rows
			m_iTemp = l;
			for (int m = l + 1; m < 6; m++)
			{
				if (m_dHessian[m][l] > m_dHessian[m_iTemp][l])
				{
					m_iTemp = m;
				}
			}
			// Swap the row which has maximum lth column element
			if (m_iTemp != l)
			{
				for (int k = 0; k < 6; k++)
				{
					m_dTemp = m_dHessian[l][k];
					m_dHessian[l][k] = m_dHessian[m_iTemp][k];
					m_dHessian[m_iTemp][k] = m_dTemp;

					m_dTemp = m_dInvHessian[l][k];
					m_dInvHessian[l][k] = m_dInvHessian[m_iTemp][k];
					m_dInvHessian[m_iTemp][k] = m_dTemp;
				}
			}
			__syncthreads();
			// Perform row operation to form required identity matrix out of the Hessian matrix
			for (int m = 0; m < 6; m++)
			{
				m_dTemp = m_dHessian[m][l];
				if (m != l)
				{
					for (int n = 0; n < 6; n++)
					{
						m_dInvHessian[m][n] -= m_dInvHessian[l][n] * m_dTemp / m_dHessian[l][l];
						m_dHessian[m][n] -= m_dHessian[l][n] * m_dTemp / m_dHessian[l][l];
					}
				}
				else
				{
					for (int n = 0; n < 6; n++)
					{
						m_dInvHessian[m][n] /= m_dTemp;
						m_dHessian[m][n] /= m_dTemp;
					}
				}
			}
			__syncthreads();
		}

		// Initialize DeltaP,
		m_dDU = 0;
		m_dDUx = 0;
		m_dDUy = 0;
		m_dDV = 0;
		m_dDVx = 0;
		m_dDVy = 0;

		// Perform interative optimization, with pre-set maximum iteration step
		for (m_iIteration = 0; m_iIteration < m_iMaxiteration; m_iIteration++)
		{
			// Fill warpped image into Subset T
			m_dSubAveT = 0;
			m_dSubNorT = 0;
			for (int l = 0; l < m_iSubsetH; l++)
			{			
				for (int m = 0; m < m_iSubsetW; m++)
				{
					// Calculate the location of warped subset T
					m_dWarpX = input_dPXY[offset*2+1] + m_dWarp[0][0] * (m - m_iSubsetX) + m_dWarp[0][1] * (l - m_iSubsetY) + m_dWarp[0][2];
					m_dWarpY = input_dPXY[offset*2+0] + m_dWarp[1][0] * (m - m_iSubsetX) + m_dWarp[1][1] * (l - m_iSubsetY) + m_dWarp[1][2];
					m_iTempX = int(m_dWarpX);
					m_iTempY = int(m_dWarpY);


					if((m_iTempX >=0) && ( m_iTempY >=0) && (m_iTempX<m_iWidth) && (m_iTempY<m_iHeight)){
						m_dTempX = m_dWarpX - m_iTempX;
						m_dTempY = m_dWarpY - m_iTempY;
						// if it is integer-pixel location, feed the gray intensity of T into the subset T
						if ((m_dTempX == 0) && (m_dTempY == 0))
						{
							m_dSubsetT[l*m_iSubsetW+m] = input_mdT[m_iTempY*m_iSubsetW+m_iTempX];
						}
						else
						{
							// If it is sub-pixel location, estimate the gary intensity using interpolation
							m_dSubsetT[l*m_iSubsetW+m] = 0;
							for (int k = 0; k < 4; k++)
							{
								for (int n = 0; n < 4; n++)
								{
									m_dSubsetT[l*m_iSubsetW+m] += input_mBicubic[((m_iTempY*m_iWidth+m_iTempX)*4+k)*4+n] * pow(m_dTempY, k) * pow(m_dTempX, n); 
								}
							}
						}
						m_dSubAveT += (m_dSubsetT[l*m_iSubsetW+m] / (m_iSubsetH * m_iSubsetW));
					}
					else{
						break;
					}
				}
			}
			for (int l = 0; l < m_iSubsetH; l++)
			{
				for (int m = 0; m < m_iSubsetW; m++)
				{
					m_dSubsetAveT[l*m_iSubsetW+m] = m_dSubsetT[l*m_iSubsetW+m] - m_dSubAveT; // T_i - T_m
					m_dSubNorT += pow(m_dSubsetAveT[l*m_iSubsetW+m], 2);
				}
			}
			m_dSubNorT = sqrt(m_dSubNorT); // sqrt (Sigma(T_i - T_m)^2)
		
			// Compute the error image
			for (int k = 0; k < 6; k++)
			{
				m_dNumerator[k] = 0;
			}
			for (int l = 0; l < m_iSubsetH; l++)
			{
				for (int m = 0; m < m_iSubsetW; m++)
				{
					m_dError[l*m_iSubsetW+m] = (m_dSubNorR / m_dSubNorT) * m_dSubsetAveT[l*m_iSubsetW+m] - m_dSubsetAveR[l*m_iSubsetW+m];
				
					// Compute the numerator
					for (int k = 0; k < 6; k++)
					{
						m_dNumerator[k] += (m_dRDescent[(l*m_iSubsetW+m)*6+k] * m_dError[l*m_iSubsetW+m]);
					}
				}
			}

			// Compute DeltaP
			for (int k = 0; k < 6; k++)
			{
				m_dDP[offset*6+k] = 0;
				for (int n = 0; n < 6; n++)
				{
					m_dDP[offset*6+k] += (m_dInvHessian[k][n] * m_dNumerator[n]);
				}
			}
			m_dDU = m_dDP[offset*6+0];
			m_dDUx = m_dDP[offset*6+1];
			m_dDUy = m_dDP[offset*6+2];
			m_dDV = m_dDP[offset*6+3];
			m_dDVx = m_dDP[offset*6+4];
			m_dDVy = m_dDP[offset*6+5];
		
			// Update the warp
			m_dTemp = (1 + m_dDUx) * (1 + m_dDVy) - m_dDUy * m_dDVx;
			//W(P) <- W(P) o W(DP)^-1
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
			output_dP[offset*6+0] = m_dWarp[0][2];
			output_dP[offset*6+1] = m_dWarp[0][0] - 1;
			output_dP[offset*6+2] = m_dWarp[0][1];
			output_dP[offset*6+3] = m_dWarp[1][2];
			output_dP[offset*6+4] = m_dWarp[1][0];
			output_dP[offset*6+5] = m_dWarp[1][1] - 1;

			m_dU = output_dP[offset*6+0];
			m_dUx = output_dP[offset*6+1];
			m_dUy = output_dP[offset*6+2];
			m_dV = output_dP[offset*6+3];
			m_dVx = output_dP[offset*6+4];
			m_dVy =output_dP[offset*6+5];

					//Check if the norm of DeltaP is small enough
			if (sqrt(pow(m_dDP[(row*m_iNumberX+col)*6+0], 2) + pow(m_dDP[(row*m_iNumberX+col)*6+1] * m_iSubsetX, 2) + pow(m_dDP[offset*6+2] * m_iSubsetY, 2) + pow(m_dDP[offset*6+3], 2) + pow(m_dDP[offset*6+4] * m_iSubsetX, 2) + pow(m_dDP[offset*6+5] * m_iSubsetY, 2)) < m_dNormDeltaP)
			{
				break;
			}
		}
		m_iIterationNum[row*m_iNumberX+col] = m_iIteration; // save iteration steps taken at this POI		
	}
}
__global__ void RGradient_kernel(const float *d_InputIMGR, const float *d_InputIMGT, const float* __restrict__ d_InputBiubicMatrix,
								 float *d_OutputIMGR, float *d_OutputIMGT, 
								 float *d_OutputIMGRx, float *d_OutputIMGRy,
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

void initialize()
/*Purpose: Initialize GPU with CPU
*/
{
	cudaFree(0);
}

void combined_functions(const float* h_InputIMGR, const float* h_InputIMGT, const float* m_dPXY, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetH, const int& m_iSubsetW, const float& m_dNormDeltaP,
						const int& m_iSubsetX, const int& m_iSubsetY, const int& m_iNumberX, const int& m_iNumberY, const int& m_iFFTSubH, const int& m_iFFTSubW, const int& m_maxIteration,
						int* m_iU, int *m_iV, float* m_dZNCC, float* m_dP, int* m_iIterationNum,
						float& precompute_tme, float& fft_time, float& icgn_time, float& total_time)
{
	//Total timer
	StopWatchWin totalT;
	StopWatchWin precompute;
	StopWatchWin icgn;

	//Variables for precomputation
	float* d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy,*d_OutputBicubic;
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
	totalT.start();
	precompute.start();
	(cudaMalloc((void**)&d_InputIMGR, (m_iWidth+2)*(m_iHeight+2)*sizeof(float)));
	(cudaMalloc((void**)&d_InputIMGT, (m_iWidth+2)*(m_iHeight+2)*sizeof(float)));
	(cudaMalloc((void**)&d_InputBiubicMatrix, 16*16*sizeof(float)));
	cudaMalloc((void**)&d_OutputIMGR, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGRx, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGRy, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGT, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputBicubic, (m_iWidth*m_iHeight*4*4)*sizeof(float));

	(cudaMemcpy(d_InputIMGR,h_InputIMGR,(m_iWidth+2)*(m_iHeight+2)*sizeof(float),cudaMemcpyHostToDevice));
	(cudaMemcpy(d_InputIMGT,h_InputIMGT,(m_iWidth+2)*(m_iHeight+2)*sizeof(float),cudaMemcpyHostToDevice));
	(cudaMemcpy(d_InputBiubicMatrix,h_InputBicubicMatrix,16*16*sizeof(float),cudaMemcpyHostToDevice));
	
	(cudaMalloc((void**)&d_OutputIMGTx, m_iWidth*m_iHeight*sizeof(float)));
	(cudaMalloc((void**)&d_OutputIMGTy, m_iWidth*m_iHeight*sizeof(float)));
	(cudaMalloc((void**)&d_OutputIMGTxy, m_iWidth*m_iHeight*sizeof(float)));

	dim3 dimB(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 dimG((m_iWidth-1)/BLOCK_SIZE+1,(m_iHeight-1)/BLOCK_SIZE+1,1);

	RGradient_kernel<<<dimG, dimB>>>(d_InputIMGR,d_InputIMGT,d_InputBiubicMatrix,
								d_OutputIMGR, d_OutputIMGT, 
								 d_OutputIMGRx, d_OutputIMGRy,
								 d_OutputIMGTx, d_OutputIMGTy, d_OutputIMGTxy,d_OutputBicubic,
								 m_iWidth, m_iHeight);

	cudaFree(d_OutputIMGTx);
	cudaFree(d_OutputIMGTy);
	cudaFree(d_OutputIMGTxy);
	cudaFree(d_InputIMGR);
	cudaFree(d_InputIMGT);
	cudaFree(d_InputBiubicMatrix);

	precompute.stop();
	precompute_tme = precompute.getTime();



	//Variables of FFT-CC
	float *hInputm_dR; 
	float *hInputm_dT; 
	hInputm_dR = (float*)malloc(m_iWidth*m_iHeight*sizeof(float));
	hInputm_dT = (float*)malloc(m_iWidth*m_iHeight*sizeof(float));
	cudaMemcpy(hInputm_dR, d_OutputIMGR,(m_iWidth*m_iHeight)*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(hInputm_dT, d_OutputIMGT,(m_iWidth*m_iHeight)*sizeof(float),cudaMemcpyDeviceToHost);
	//-----Start FFT-CC algorithm------
	FFT_CC(hInputm_dR, hInputm_dT, m_dPXY, m_iNumberY,  m_iNumberX, 
			m_iFFTSubH,  m_iFFTSubW,  m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX,
			m_dZNCC, m_iU, m_iV,fft_time);




	//Variables of IC-GN
	int* dInput_miU;
	int* dInput_miV;
	float* dInput_mdPXY;
	float* dOutput_mdP;
	int* dOutput_miIterationNum;
	float *dm_dSubsetR, *dm_dSubsetT, *dm_dJacobian, *dm_dRdescent, *dm_dHessianXY, *dm_dSubsetAveR, *dm_dSubsetAveT, *dm_dError, *dm_dDP;

	//Variables for ICGN
	cudaMalloc((void**)&dInput_miU, (m_iNumberX*m_iNumberY)*sizeof(int));
	cudaMalloc((void**)&dInput_miV, (m_iNumberY*m_iNumberX)*sizeof(int));
	cudaMalloc((void**)&dOutput_miIterationNum, (m_iNumberY*m_iNumberX)*sizeof(int));
	cudaMalloc((void**)&dInput_mdPXY,(m_iNumberY*m_iNumberX)*2*sizeof(float));
	cudaMalloc((void**)&dOutput_mdP, (m_iNumberY*m_iNumberX)*6*sizeof(float));

	cudaMemcpy(dInput_miU, m_iU, (m_iNumberX*m_iNumberY*sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(dInput_miV, m_iV, (m_iNumberX*m_iNumberY*sizeof(int)), cudaMemcpyHostToDevice);
	cudaMemcpy(dInput_mdPXY, m_dPXY, (m_iNumberY*m_iNumberX)*2*sizeof(float), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dm_dSubsetR, m_iSubsetH*m_iSubsetW*sizeof(float));
	cudaMalloc((void**)&dm_dSubsetT, m_iSubsetH*m_iSubsetW*sizeof(float));
	cudaMalloc((void**)&dm_dJacobian, m_iSubsetH*m_iSubsetW*2*6*sizeof(float));
	cudaMalloc((void**)&dm_dRdescent, m_iSubsetH*m_iSubsetW*6*sizeof(float));
	cudaMalloc((void**)&dm_dHessianXY, m_iSubsetH*m_iSubsetW*6*6*sizeof(float));
	cudaMalloc((void**)&dm_dSubsetAveR, m_iSubsetH*m_iSubsetW*sizeof(float));
	cudaMalloc((void**)&dm_dSubsetAveT, m_iSubsetH*m_iSubsetW*sizeof(float));
	cudaMalloc((void**)&dm_dError, m_iSubsetH*m_iSubsetW*sizeof(float));
	cudaMalloc((void**)&dm_dDP, m_iNumberX*m_iNumberY*6*sizeof(float));


	dim3 dimGrid((m_iNumberY-1)/16+1, (m_iNumberX-1)/16+1,1);
	dim3 dimBlock(16, 16,1);

	icgn.start();
	computeICGN<<<dimGrid,dimBlock>>>(dInput_mdPXY, d_OutputIMGR, d_OutputIMGRx, d_OutputIMGRy, m_dNormDeltaP,
									  d_OutputIMGT, d_OutputBicubic, dInput_miU, dInput_miV, 
									  m_iNumberY, m_iNumberX, m_iSubsetH, m_iSubsetW, m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX, m_maxIteration,
									  dOutput_mdP,dOutput_miIterationNum,
									  dm_dSubsetR, dm_dSubsetT, dm_dJacobian,dm_dRdescent,dm_dHessianXY,dm_dSubsetAveR,dm_dSubsetAveT,dm_dError,dm_dDP);

	cudaMemcpy(m_dP, dOutput_mdP, (m_iNumberY*m_iNumberX)*6*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_iIterationNum, dOutput_miIterationNum, (m_iNumberY*m_iNumberX)*sizeof(int), cudaMemcpyDeviceToHost);
	icgn.stop();
	icgn_time = icgn.getTime();

	totalT.stop();
	total_time = totalT.getTime();

	cudaFree(dm_dSubsetR);
	cudaFree(dm_dSubsetT);
	cudaFree(dm_dJacobian);
	cudaFree(dm_dRdescent);
	cudaFree(dm_dHessianXY);
	cudaFree(dm_dSubsetAveR);
	cudaFree(dm_dSubsetAveT);
	cudaFree(dm_dError);
	cudaFree(dm_dDP);

	cudaFree(d_OutputIMGR);
	cudaFree(d_OutputIMGRx);
	cudaFree(d_OutputIMGRy);
	cudaFree(d_OutputIMGT);
	cudaFree(d_OutputBicubic);
	cudaFree(dInput_miU);
	cudaFree(dInput_miV);
	cudaFree(dInput_mdPXY);
	cudaFree(dOutput_mdP);
	cudaFree(dOutput_miIterationNum);

	free(hInputm_dR);
	free(hInputm_dT);
}
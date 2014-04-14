#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"


#include <stdio.h>

#include "ICGN.cuh"

__global__ void computeICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, float m_dNormDeltaP,
							const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV,
							int m_iNumberY, int m_iNumberX, int m_iSubsetH, int m_iSubsetW,	int m_iWidth, int m_iHeight, int m_iSubsetY, int m_iSubsetX, int m_iMaxiteration,
							float* output_dP, int* dm_iIterationNum)
/*Input: all the const variables
 Output: deformation P matrix
Strategy: Each block compute one of the 21*21 POIs, and within each block 32*32 threads compute other needed computations
*/
{
	unsigned int blockID = blockIdx.y * gridDim.x + blockIdx.x;  //Row * dim + Col

	if(threadIdx.x == 0 && threadIdx.y ==0){
		output_dP[blockID*6+0] = 0.0;
		output_dP[blockID*6+1] = 0.0;
		output_dP[blockID*6+2] = 0.0;
		output_dP[blockID*6+3] = 0.0;
		output_dP[blockID*6+4] = 0.0;
		output_dP[blockID*6+5] = 0.0;
		dm_iIterationNum[blockID] = 5;
	}
}
void launch_ICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, float m_dNormDeltaP,
				 const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV,
				 int m_iNumberY, int m_iNumberX, int m_iSubsetH, int m_iSubsetW, int m_iWidth, int m_iHeight, int m_iSubsetY, int m_iSubsetX, int m_iMaxiteration,
				 float* output_dP, int* dm_iIterationNum, float& time)
{
	StopWatchWin icgn;

	dim3 dimGrid(m_iNumberY, m_iNumberX,1);
	dim3 dimBlock(m_iSubsetH, m_iSubsetW,1);

	icgn.start();
	computeICGN<<<dimGrid,dimBlock>>>(input_dPXY, input_mdR, input_mdRx, input_mdRy, m_dNormDeltaP,
									  input_mdT, input_mBicubic, input_iU, input_iV, 
									  m_iNumberY, m_iNumberX, m_iSubsetH, m_iSubsetW, m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX, m_iMaxiteration,
									  output_dP,dm_iIterationNum);

	icgn.stop();
	time = icgn.getTime();

}
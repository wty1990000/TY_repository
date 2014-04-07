#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"
#include "math_functions.h"
#include "cufft.h"
#include "cufftw.h"

#include <stdio.h>
#include "FFTcc.cuh"


__global__ void FFTCC_pre_kernel(double* dInput_mdR, double* dInput_mdT, const int& m_iWidth, const int& m_iHeight,
								 double* dm_dPXY, int* dm_iFlag1, double* dm_Subset1, double* dm_Subset2,
								 const int& m_iMarginX, const int& m_iMarginY, const int& m_iGridSapceY, const int& m_iGridSapceX,
								 const int& m_iSubsetX, const int& m_iSubsetY, const int& m_iNumberX, const int& m_iNumberY, const int& m_iFFTSubW, const int& m_iFFTSubH)
{
	//Temperary variables
	double m_dAvef = 0.0, m_dModf = 0.0;
	double m_dAveg = 0.0, m_dModg = 0.0;

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if(row<m_iNumberY && col<m_iNumberX){
		dm_dPXY[(row*m_iNumberX+col)*2+0] = double(m_iMarginX + m_iSubsetY + col*m_iGridSapceY);
		dm_dPXY[(row*m_iNumberX+col)*2+1] = double(m_iMarginY + m_iSubsetX + col*m_iGridSapceX);
	
		__syncthreads();

		//Feed the gray intesity values into subsets
		for(unsigned int l=0; l<m_iFFTSubH; l++){
			for(unsigned int m=0; m<m_iFFTSubW; m++){
				dm_Subset1[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m]=dInput_mdR[int(dm_dPXY[(row*m_iNumberX+col)*2+0] - m_iSubsetY+l)*m_iWidth+int(dm_dPXY[(row*m_iNumberX+col)*2+1] - m_iSubsetX+m)];
				m_dAvef += (dm_Subset1[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m] / (m_iFFTSubH+m_iFFTSubW));
				dm_Subset2[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m]=dInput_mdT[int(dm_dPXY[(row*m_iNumberX+col)*2+0] - m_iSubsetY+l)*m_iWidth+int(dm_dPXY[(row*m_iNumberX+col)*2+1] - m_iSubsetX+m)];
				m_dAveg += (dm_Subset2[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m] / (m_iFFTSubH+m_iFFTSubW));
			}
		}
		__syncthreads();
		for(unsigned int l=0; l<m_iFFTSubH; l++){
			for(unsigned int m=0; m<m_iFFTSubW; m++){
				dm_Subset1[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m] -= m_dAvef;
				dm_Subset2[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m] -= m_dAveg;
				m_dModf += pow(dm_Subset1[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m],2);
				m_dModg += pow(dm_Subset2[((row*m_iNumberX+col)*m_iFFTSubH+l)*m_iFFTSubW+m],2);
			}
		}
		if(m_dModf <= 0.0000001 || m_dModg <= 0.0000001){
			//if one of the two subsets is full of zero intensities, set the flag to 1.
			dm_iFlag1[row*m_iWidth+col] = 1;
		}				
	}
	//Done the precomputation for the FFT-CC
}
void FFTCC_kernel(double* dInput_mdR, double* dInput_mdT, const int& m_iWidth, const int& m_iHeight,
				  double* dm_dPXY, double* dm_dP, 
				  const int& m_iMarginX, const int& m_iMarginY, const int& m_iGridSapceY, const int& m_iGridSapceX,
				  const int& m_iSubsetX, const int& m_iSubsetY, const int& m_iNumberX, const int& m_iNumberY,const int& m_iFFTSubW, const int& m_iFFTSubH,
				  double* dOutput_dZNCC, int* dOutput_iFlag1)
{
	cufftDoubleReal *dm_Subset1, *dm_Subset2, *dm_SubsetC;
	cufftDoubleComplex *dm_FreqDom1, *dm_FreqDom2, *dm_FreqDomfg;
	cufftHandle plan1, plan2, rplan;
	int n[2] = {m_iFFTSubH, m_iFFTSubW};

	cudaMalloc((void**)&dm_Subset1, m_iNumberY*m_iNumberX*m_iFFTSubH*m_iFFTSubW*sizeof(cufftDoubleReal));
	cudaMalloc((void**)&dm_Subset2, m_iNumberY*m_iNumberX*m_iFFTSubH*m_iFFTSubW*sizeof(cufftDoubleReal));
	cudaMalloc((void**)&dm_SubsetC, m_iNumberY*m_iNumberX*m_iFFTSubH*m_iFFTSubW*sizeof(cufftDoubleReal));
	
	cudaMalloc((void**)&dm_FreqDom1, m_iNumberY*m_iNumberX*m_iFFTSubW*(m_iFFTSubH/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dm_FreqDom2, m_iNumberY*m_iNumberX*m_iFFTSubW*(m_iFFTSubH/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dm_FreqDomfg, m_iNumberY*m_iNumberX*m_iFFTSubW*(m_iFFTSubH/2+1)*sizeof(cufftDoubleComplex));

	//cufftPlanMany(plan1, 2, n, 
}
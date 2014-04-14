#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"
#include "math_functions.h"
#include "cufft.h"
#include "cufftw.h"

#include <stdio.h>
#include "FFTcc.cuh"

static __device__ __host__ inline cufftDoubleComplex ComplexMul(cufftDoubleComplex a, cufftDoubleComplex b)
{
	cufftDoubleComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x * b.y + a.y * b.x;
	return c;
}
__global__ void FFTCC_pre_kernel(const double* dInput_mdR, const double* dInput_mdT, int m_iWidth, int m_iHeight,
								 double* dm_dPXY, int* dm_iFlag1, double* dm_Subset1, double* dm_Subset2,
								 int m_iMarginX, int m_iMarginY, int m_iGridSpaceY, int m_iGridSapceX,
								 int m_iSubsetX, int m_iSubsetY, int m_iNumberX, int m_iNumberY, int m_iFFTSubW, int m_iFFTSubH)
/*Input: dInput_mdR, dInput_mdT, m_* single variables
 Output: dm_dPXY, dm_iFlag1, dm_Subset1, dm_Subset2
Purpose: fill in the subsets for FFTCC use.
*/
{
	//Temperary variables
	double m_dAvef = 0.0, m_dModf = 0.0;
	double m_dAveg = 0.0, m_dModg = 0.0;

	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if((row<m_iNumberY) && (col<m_iNumberX)){
		dm_dPXY[(row*m_iNumberX+col)*2+0] = double(m_iMarginX + m_iSubsetY + col*m_iGridSpaceY);
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
		if(m_dModf <= 0.0000000001 || m_dModg <= 0.0000000001){
			//if one of the two subsets is full of zero intensities, set the flag to 1.
			dm_iFlag1[row*m_iWidth+col] = 1;
		}				
	}
	//Done the precomputation for the FFT-CC
}
__global__ void pointwiseMul(const cufftDoubleComplex* Inputa, const cufftDoubleComplex* Inputb, cufftDoubleComplex* Output, 
							 int size, int width, int height)
/*Input: Two cufftDoubleComplex arrays
 Output: Results after complex multiplication
Purpose: Complex multiplication for "size" points, done in parallel
*/
{
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	if((row<height) && (col<height)){
		for(unsigned int n=0; n<size; n++){
			Output[(row*width+col)*size+n] = ComplexMul(Inputa[(row*width+col)*size+n],Inputb[(row*width+col)*size+n]);
		}
	}
}

void FFTCC_kernel(const double* dInput_mdR, const double* dInput_mdT, int m_iWidth, int m_iHeight,
				  double* dOutputm_dPXY,  double* dOutput_dZNCC, int* dOutput_iFlag1,
				  int m_iMarginX, int m_iMarginY, int m_iGridSpaceY, int m_iGridSapceX,
				  int m_iSubsetX, int m_iSubsetY, int m_iNumberX, int m_iNumberY,int m_iFFTSubW, int m_iFFTSubH)
/*Input: mdR, mdT, other m_* single variables
 Output: dOutput_dZNCC, dOutput_iFlag1, dOutputm_dPXY, dOutputm_dP(No need to allocate memory here, allocate them in combination.cu.)
Purpose: Do the FFTCC, get ZNCC for the last output, get m_dPXY, m_iFlag1 to be used in ICGN.
*/
{
	//Parameters for cuFFT
	cufftDoubleReal *dm_Subset1, *dm_Subset2, *dm_SubsetC;
	cufftDoubleComplex *dm_FreqDom1, *dm_FreqDom2, *dm_FreqDomfg;
	cufftHandle plan, rplan;
	int n[] = {m_iFFTSubW,m_iFFTSubH};
	int nr[] = {m_iFFTSubW, (m_iFFTSubH/2+1)};

	int inembed[] = {m_iFFTSubW, m_iFFTSubH};
	int onembed[] = {m_iFFTSubW, (m_iFFTSubH/2+1)};

	//Parameters for Launching the kernels
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 dimGrid((m_iNumberY-1)/BLOCK_SIZE+1, (m_iNumberX-1)/BLOCK_SIZE+1,1);

	//Allocate the memory for FFTs
	cudaMalloc((void**)&dm_Subset1, m_iNumberY*m_iNumberX*m_iFFTSubH*m_iFFTSubW*sizeof(cufftDoubleReal));
	cudaMalloc((void**)&dm_Subset2, m_iNumberY*m_iNumberX*m_iFFTSubH*m_iFFTSubW*sizeof(cufftDoubleReal));
	cudaMalloc((void**)&dm_SubsetC, m_iNumberY*m_iNumberX*m_iFFTSubH*m_iFFTSubW*sizeof(cufftDoubleReal));
	//Fill in the values to dm_Subset1, dm_Subset2
	FFTCC_kernel<<<dimGrid, dimBlock>>>(dInput_mdR, dInput_mdT, m_iWidth, m_iHeight, 
										dOutputm_dPXY, dOutput_iFlag1, dm_Subset1, dm_Subset2, 
										m_iMarginX, m_iMarginY, m_iGridSpaceY, m_iGridSpaceX,
										m_iSubsetX, m_iSubsetY, m_iNumberX, m_iNumberY, m_iFFTSubW, m_iFFTSubH);
	
	//Allocate the memory for thr frequency domain
	cudaMalloc((void**)&dm_FreqDom1, m_iNumberY*m_iNumberX*m_iFFTSubW*(m_iFFTSubH/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dm_FreqDom2, m_iNumberY*m_iNumberX*m_iFFTSubW*(m_iFFTSubH/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dm_FreqDomfg, m_iNumberY*m_iNumberX*m_iFFTSubW*(m_iFFTSubH/2+1)*sizeof(cufftDoubleComplex));
	
	//CUFFT plans for forward FFT and reverse FFT
	cufftPlanMany(&plan,2,n,inembed,1,m_iFFTSubH*m_iFFTSubW,onembed,1,m_iFFTSubW*(m_iFFTSubH/2+1),CUFFT_D2Z,m_iNumberY*m_iNumberX);
	cufftPlanMany(&rplan,2,nr,onembed,1,m_iFFTSubW*(m_iFFTSubH/2+1),inembed,1,m_iFFTSubH*m_iFFTSubW,CUFFT_Z2D,m_iNumberY*m_iNumberX);
	//Execute the plan
	cufftExecD2Z(plan, dm_Subset1, dm_FreqDom1);
	cufftExecD2Z(plan, dm_Subset2, dm_FreqDom2);

	//Multiplication in the frequncy domain
	pointwiseMul<<<dimGrid,dimBlock>>>(dm_FreqDom1, dm_FreqDom2, dm_FreqDomfg, m_iFFTSubW*(m_iFFTSubH/2+1),m_iNumberX,m_iNumberY);
	//Execute the reverse FFT
	cufftExecZ2D(rplan,dm_FreqDomfg,dm_SubsetC);
}
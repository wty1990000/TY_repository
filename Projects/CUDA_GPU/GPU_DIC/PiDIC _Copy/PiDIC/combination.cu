#include "combination.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "precomputation.cuh"
#include "FFT-CC.h"

void initialize()
/*Purpose: Initialize GPU with CPU
*/
{
	cudaFree(0);
}

void combined_functions(const float* h_InputIMGR, const float* h_InputIMGT, const float* m_dPXY, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetH, const int& m_iSubsetW,
						const int& m_iSubsetX, const int& m_iSubsetY, const int& m_iNumberX, const int& m_iNumberY, const int& m_iFFTSubH, const int& m_iFFTSubW,
						int* m_iU, int *m_iV, float* m_dZNCC, float* m_dP, int* m_iFlag1,
						float& precompute_tme, float& fft_time, float& icgn_time, float& total_time)
{
	//Total timer
	StopWatchWin totalT;
	//Variables for precomputation
	float* d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputIMGT, *d_OutputBicubic;

	//Variables of FFT-CC
	float *m_dR; 
	float *m_dT; 

	//Variables for ICGN
	cudaMalloc((void**)&d_OutputIMGR, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGRx, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGRy, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGT, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputBicubic, (m_iWidth*m_iHeight*4*4)*sizeof(float));


	/*-----------------------------Start Computation----------------------------------*/
	//-----Start pre-computation------
	totalT.start();
	precompute_kernel(h_InputIMGR, h_InputIMGT, d_OutputIMGR, d_OutputIMGT, 
		d_OutputIMGRx, d_OutputIMGRy, d_OutputBicubic,m_iWidth,m_iHeight, precompute_tme);
	m_dR = (float*)malloc(m_iWidth*m_iHeight*sizeof(float));
	m_dT = (float*)malloc(m_iWidth*m_iHeight*sizeof(float));
	cudaMemcpy(m_dR, d_OutputIMGR,(m_iWidth*m_iHeight)*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(m_dT, d_OutputIMGT,(m_iWidth*m_iHeight)*sizeof(float),cudaMemcpyDeviceToHost);

	//-----Start FFT-CC algorithm------
	FFT_CC(m_dR, m_dT, m_dPXY, m_iNumberY,  m_iNumberX, 
			m_iFFTSubH,  m_iFFTSubW,  m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX,
			m_dZNCC, m_iU, m_iV,fft_time);


}
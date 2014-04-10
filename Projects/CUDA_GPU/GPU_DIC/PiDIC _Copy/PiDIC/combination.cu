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

void combined_functions(const double* h_InputIMGR, const double* h_InputIMGT, const double* m_dPXY, const int& m_iWidth, const int& m_iHeight, 
						const int& m_iSubsetX, const int& m_iSubsetY, const int& m_iNumberX, const int& m_iNumberY, const int& m_iFFTSubH, const int& m_iFFTSubW,
						double* m_iU, double *m_iV, double* m_dZNCC, double* m_dP, int* m_iFlag1,
						float& precompute_tme, float& fft_time, float& icgn_time, float& total_time)
{
	//Total timer
	StopWatchWin totalT;
	//Variables for precomputation
	double* d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputIMGT, *d_OutputBicubic;

	//Variables of FFT-CC
	double *m_dR; 
	double *m_dT; 
	int *m_iCorrPeakX, *m_iCorrPeakY;

	//Variables for ICGN



	/*-----------------------------Start Computation----------------------------------*/
	totalT.start();
	//-----Start pre-computation------
	cudaMalloc((void**)&d_OutputIMGR, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRx, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRy, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGT, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputBicubic, (m_iWidth*m_iHeight*4*4)*sizeof(double));
	precompute_kernel(h_InputIMGR, h_InputIMGT, d_OutputIMGR, d_OutputIMGT, 
		d_OutputIMGRx, d_OutputIMGRy, d_OutputBicubic,m_iWidth,m_iHeight, precompute_tme);

	m_dR = (double*)malloc(m_iWidth*m_iHeight*sizeof(double));
	m_dT = (double*)malloc(m_iWidth*m_iHeight*sizeof(double));
	cudaMemcpy(m_dR, d_OutputIMGR,(m_iWidth*m_iHeight)*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(m_dT, d_OutputIMGT,(m_iWidth*m_iHeight)*sizeof(double),cudaMemcpyDeviceToHost);

	//-----Start FFT-CC algorithm------
	m_iCorrPeakX = new int[m_iNumberX*m_iNumberY];
	m_iCorrPeakY = new int[m_iNumberY*m_iNumberX];
	FFT_CC(m_dR, m_dT, m_dPXY, m_iNumberY,  m_iNumberX, 
			m_iFFTSubH,  m_iFFTSubW,  m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX,
			m_iFlag1, m_dZNCC, m_dP, m_iCorrPeakX, m_iCorrPeakY,fft_time);


}
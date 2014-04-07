#include "combination.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "precomputation.cuh"

void combined_functions(const double* h_InputIMGR, const double* h_InputIMGT, const int& m_iWidth, const int& m_iHeight, 
						const int& m_iSubsetX, const int& m_iSubsetY, double* m_dZNCC, double* m_dP,
						double* m_iU, double *m_iV,
						float& precompute_tme, float& fft_time, float& icgn_time, float& total_time)
{
	//Total timer
	StopWatchWin totalT;
	//Variables for precomputation
	double* d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputIMGT, *d_OutputBicubic;

	//Variables of FFT-CC
	
	//-----Start pre-computation------
	cudaMalloc((void**)&d_OutputIMGR, (m_iWidth*m_iHeight)*sizeof(double));
	totalT.start();
	cudaMalloc((void**)&d_OutputIMGRx, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRy, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGT, (m_iWidth*m_iHeight)*sizeof(double));
	cudaMalloc((void**)&d_OutputBicubic, (m_iWidth*m_iHeight*4*4)*sizeof(double));
	precompute_kernel(h_InputIMGR, h_InputIMGT, d_OutputIMGR, d_OutputIMGT, 
		d_OutputIMGRx, d_OutputIMGRy, d_OutputBicubic,m_iWidth,m_iHeight, precompute_tme);
	
	//----Start FFT-CC ---------------
	
}
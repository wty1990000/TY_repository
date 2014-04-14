#include "combination.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "precomputation.cuh"
#include "FFT-CC.h"
#include "IC_GN.cuh"

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

	//Variables for precomputation
	float* d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy,*d_OutputBicubic;
	//Variables of FFT-CC
	float *hInputm_dR; 
	float *hInputm_dT; 
	//Variables of IC-GN
	int* dInput_miU;
	int* dInput_miV;
	float* dInput_mdPXY;
	float* dOutput_mdP;
	int* dOutput_miIterationNum;

	cudaMalloc((void**)&d_OutputIMGR, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGRx, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGRy, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputIMGT, (m_iWidth*m_iHeight)*sizeof(float));
	cudaMalloc((void**)&d_OutputBicubic, (m_iWidth*m_iHeight*4*4)*sizeof(float));


	cudaMalloc((void**)&dInput_miU, (m_iNumberX*m_iNumberY)*sizeof(int));
	cudaMalloc((void**)&dInput_miV, (m_iNumberY*m_iNumberX)*sizeof(int));
	cudaMalloc((void**)&dOutput_miIterationNum, (m_iNumberY*m_iNumberX)*sizeof(int));
	cudaMalloc((void**)&dInput_mdPXY,(m_iNumberY*m_iNumberX)*2*sizeof(float));
	cudaMalloc((void**)&dOutput_mdP, (m_iNumberY*m_iNumberX)*6*sizeof(float));


	/*-----------------------------Start Computation----------------------------------*/
	//-----Start pre-computation------
	totalT.start();
	precompute_kernel(h_InputIMGR, h_InputIMGT, d_OutputIMGR, d_OutputIMGT, 
		d_OutputIMGRx, d_OutputIMGRy, d_OutputBicubic,m_iWidth,m_iHeight, precompute_tme);

	hInputm_dR = (float*)malloc(m_iWidth*m_iHeight*sizeof(float));
	hInputm_dT = (float*)malloc(m_iWidth*m_iHeight*sizeof(float));
	cudaMemcpy(hInputm_dR, d_OutputIMGR,(m_iWidth*m_iHeight)*sizeof(float),cudaMemcpyDeviceToHost);
	cudaMemcpy(hInputm_dT, d_OutputIMGT,(m_iWidth*m_iHeight)*sizeof(float),cudaMemcpyDeviceToHost);



	//-----Start FFT-CC algorithm------
	FFT_CC(hInputm_dR, hInputm_dT, m_dPXY, m_iNumberY,  m_iNumberX, 
			m_iFFTSubH,  m_iFFTSubW,  m_iWidth, m_iHeight, m_iSubsetY, m_iSubsetX,
			m_dZNCC, m_iU, m_iV,fft_time);
	
	cudaMemcpy(dInput_miU, m_iU, (m_iNumberX*m_iNumberY)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dInput_miV, m_iV, (m_iNumberX*m_iNumberY)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dInput_mdPXY,m_dPXY, (m_iNumberY*m_iNumberX)*2*sizeof(float), cudaMemcpyHostToDevice);



	//-----Start IC-GN algorithm------
	launch_ICGN(dInput_mdPXY, d_OutputIMGR,d_OutputIMGRx, d_OutputIMGRy, m_dNormDeltaP,d_OutputIMGT, d_OutputBicubic,
		dInput_miU, dInput_miV, m_iNumberY, m_iNumberX, m_iSubsetH,m_iSubsetW,m_iWidth,m_iHeight,m_iSubsetY,m_iSubsetX,m_maxIteration,dOutput_mdP,dOutput_miIterationNum,icgn_time);
	
	cudaMemcpy(m_dP, dOutput_mdP, (m_iNumberY*m_iNumberX)*6*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(m_iIterationNum, dOutput_miIterationNum, (m_iNumberY*m_iNumberX)*sizeof(int), cudaMemcpyDeviceToHost);

	totalT.stop();
	total_time = totalT.getTime();




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
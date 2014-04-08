#include "combination.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"

#include "FFTcc.cuh"
#include "precomputation.cuh"

void initialize_CUDA()
/*Purpose: Initialize GPU with CPU
*/
{
	cudaFree(0);
}

void combined_functions(const double* h_InputIMGR, const double* h_InputIMGT, double* h_OutputFFTcc)
{
	//Total timer
	float FFTtime=0.0, precomputetime=0.0;

	//Variables for precomputation
	double *d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputBicubic;

	cudaMalloc((void**)&d_OutputIMGR, (254*254)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRx, (254*254)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGRy, (254*254)*sizeof(double));
	cudaMalloc((void**)&d_OutputIMGT, (254*254)*sizeof(double));
	cudaMalloc((void**)&d_OutputBicubic, (254*254*4*4)*sizeof(double));

	//Variables of FFT-CC
	double *dOutputm_dPXY, *dm_SubsetC;
	int *dm_iFlag1;
	cudaMalloc((void**)&dOutputm_dPXY, (21*21*2)*sizeof(double));
	cudaMalloc((void**)&dm_iFlag1, (21*21)*sizeof(int));
	cudaMalloc((void**)&dm_SubsetC, 21*21*32*32*sizeof(double));


	//-----Start pre-computation------
	precompute_kernel(h_InputIMGR, h_InputIMGT, d_OutputIMGR, d_OutputIMGT, 
		d_OutputIMGRx, d_OutputIMGRy, d_OutputBicubic,254,254, precomputetime);


	//----Start FFT-CC ---------------
	FFTCC_kernel(d_OutputIMGR,d_OutputIMGT,254,254,dOutputm_dPXY,dm_iFlag1,dm_SubsetC,10,10,10,10,16,16,21,21,32,32, FFTtime);
	cudaMemcpy(h_OutputFFTcc,dm_SubsetC,21*21*32*32*sizeof(double),cudaMemcpyDeviceToHost);

	//Free the memory
	cudaFree(d_OutputIMGR);
	cudaFree(d_OutputIMGRx);
	cudaFree(d_OutputIMGRy);
	cudaFree(d_OutputIMGT);
	cudaFree(d_OutputBicubic);

	cudaFree(dOutputm_dPXY);
	cudaFree(dm_iFlag1);
	cudaFree(dm_SubsetC);

	printf("Precomputation time: %f\n",precomputetime);
	printf("FFT time: %f",FFTtime);
	
}
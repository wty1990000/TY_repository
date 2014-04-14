#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "comb.cuh"
#include "kernel.cuh"


void combine(float* output, const int& sizeo, const int& sizei)
{
	float *dOutput;
	checkCudaErrors(cudaMalloc((void**)&dOutput, sizeo*sizeo*sizeof(float)));

	launch_kernel(dOutput, sizeo, sizei);

	checkCudaErrors(cudaMemcpy(output, dOutput, sizeo*sizeo*sizeof(float),cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(dOutput));
}
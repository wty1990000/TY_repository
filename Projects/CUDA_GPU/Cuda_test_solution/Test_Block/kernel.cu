#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"

#include <stdio.h>

__global__ void assignment(float* output)
{
	unsigned int blockID = blockIdx.y * gridDim.x + blockIdx.x;

	if(threadIdx.x ==0 && threadIdx.y == 0){
		output[blockID] = 20.0;
	}
}

void launch_kernel(float *dOutput, const int& sizeo, const int& sizei)
{
	dim3 dimG(sizeo, sizeo, 1);
	dim3 dimB(sizei, sizei, 1);

	assignment<<<dimG,dimB>>>(dOutput);
}


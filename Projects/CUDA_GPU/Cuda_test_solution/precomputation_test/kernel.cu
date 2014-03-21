#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"

#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(double *input, double *output, int size)
{
	int row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if(row<size && col<size){
		output[row*size+col] = 0.5*(input[(row+1)*(size+2)+col+2]-input[(row+1)*(size+2)+col]);
	}
}

void launchkernel(double *h_input, double *h_output, int size)
{
	dim3 dimG((size-1)/BLOCK_SIZE+1, (size-1)/BLOCK_SIZE+1,1);
	dim3 dimB(BLOCK_SIZE, BLOCK_SIZE,1);

	double *d_input, *d_output;

	cudaMalloc((void**)&d_input, (size+2)*(size+2)*sizeof(double));
	cudaMalloc((void**)&d_output, size*size*sizeof(double));

	cudaMemcpy(d_input, h_input, (size+2)*(size+2)*sizeof(double), cudaMemcpyHostToDevice);

	kernel<<<dimG, dimB>>>(d_input,d_output,size);
	cudaDeviceSynchronize();

	cudaMemcpy(h_output,d_output,size*size*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_input);
	cudaFree(d_output);
}


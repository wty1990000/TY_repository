
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include "Random.h"

#define BLOCK_SIZE 4
#define INPUT_SIZE 7
#define OUTPUT_SIZE ((INPUT_SIZE-1)/(BLOCK_SIZE*2)+1)

__global__ void reduction_kernel(float *input, float *output, int len)
{
	__shared__ float partialSum[2*BLOCK_SIZE];
	
	unsigned int t = threadIdx.x;
	unsigned start = 2*blockIdx.x*blockDim.x;
	
	if((start + t) < len){
		partialSum[t] = input[start + t];
	} else {
		partialSum[t] = 0.0f;
	}
	
	if((start + blockDim.x + t) < len){
		partialSum[blockDim.x + t] = input[start + blockDim.x + t];
	} else {
		partialSum[blockDim.x + t] = 0.0f;
	}
	

	for (unsigned int stride = blockDim.x; stride > 0; stride /=2)
	{
		__syncthreads();
		if (t <stride) {
			partialSum[t] += partialSum[t + stride];
		}
	}
	
//__syncthreads();
	
	if(t == 0)
	{
		output[blockIdx.x] = partialSum[t];
	}
}

int main()
{
	float *hInput, *hOutput;
	float *dInput, *dOutput;

	Random rr;


	hInput = (float*)malloc(INPUT_SIZE*sizeof(float));
	hOutput = (float*)malloc(INPUT_SIZE*sizeof(float));

	for(int i=0; i<INPUT_SIZE; i++)
		hInput[i] = float(rr.random_integer(1,10));

	cudaMalloc((void**)&dInput, INPUT_SIZE * sizeof(float));
	cudaMalloc((void**)&dOutput, OUTPUT_SIZE * sizeof(float));

	cudaMemcpy(dInput, hInput, INPUT_SIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 dimGrid(OUTPUT_SIZE, 1,1);
	dim3 dimBlock(BLOCK_SIZE,1,1);

	unsigned int len = INPUT_SIZE;

	reduction_kernel<<<dimGrid, dimBlock>>>(dInput, dOutput, len);

	cudaMemcpy(hOutput, dOutput, OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

	for(int i=1; i<OUTPUT_SIZE; i++)
		hOutput[0] += hOutput[i];
	printf("The final result of reduction is: %3.1f",hOutput[0]);

	cudaFree(dInput);
	cudaFree(dOutput);

	return 0;
}
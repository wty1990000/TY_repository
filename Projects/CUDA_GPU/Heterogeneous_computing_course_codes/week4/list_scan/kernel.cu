
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdio>
#include <cstdlib>
#include "Random.h"

#define BLOCK_SIZE 512
#define SECTION_SIZE (2*BLOCK_SIZE)
#define INPUT_SIZE 4569
#define OUTPUT_SIZE INPUT_SIZE//((INPUT_SIZE-1)/(BLOCK_SIZE*2)+1)

__global__ void list_scan_kernel(float *input, float *output, float *sum, int len)
{
	__shared__ float XY[SECTION_SIZE];
	
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < len){
		XY[threadIdx.x] = input[i];
	}

	//Reduction
	for( int stride =1; stride < blockDim.x; stride *=2){
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2-1;
		if(index<blockDim.x)
			XY[index] += XY[index-stride];
	}
	
	//Post reduction
	for(int stride = SECTION_SIZE/4; stride>0; stride /= 2){
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2-1;
		if(index+stride < blockDim.x){
			XY[index+stride] += XY[index];
		}
	}
	__syncthreads();
	output[i] = XY[threadIdx.x];  //results for each individual sections

	__syncthreads();
	if(threadIdx.x ==SECTION_SIZE-1){
		sum[blockIdx.x] = XY[SECTION_SIZE-1];
	}	
}
__global__ void list_scan_kernel2(float *input, float *intermediate, int len)
{
	__shared__ float XY[SECTION_SIZE];
	
	unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
	if(i < len){
		XY[threadIdx.x] = input[i];
	}

	//Reduction
	for( int stride =1; stride < blockDim.x; stride *=2){
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2-1;
		if(index<blockDim.x)
			XY[index] += XY[index-stride];
	}
	
	//Post reduction
	for(int stride = SECTION_SIZE/4; stride>0; stride /= 2){
		__syncthreads();
		int index = (threadIdx.x+1)*stride*2-1;
		if(index+stride < blockDim.x){
			XY[index+stride] += XY[index];
		}
	}
	__syncthreads();
	intermediate[i] = XY[threadIdx.x];  //results for each individual sections
	
}
__global__ void consolidation_kernel(float *input, float *output)
{
	int i = blockIdx.x;
	int t = (blockIdx.x+1) * blockDim.x + threadIdx.x;
	if(i<(OUTPUT_SIZE-1)/(2*BLOCK_SIZE))
		output[t] += input[i];
}

int main()
{
	float *hInput, *hOutput;
	float *dInput, *dOutput, *dSum;

	//Random rr;


	hInput = (float*)malloc(INPUT_SIZE*sizeof(float));
	hOutput = (float*)malloc(INPUT_SIZE*sizeof(float));

	for(int i=0; i<INPUT_SIZE; i++)
		hInput[i] = float(i+1);//rr.random_integer(1,10));

	cudaMalloc((void**)&dInput, INPUT_SIZE * sizeof(float));
	cudaMalloc((void**)&dOutput, OUTPUT_SIZE * sizeof(float));
	cudaMalloc((void**)&dSum, ((OUTPUT_SIZE-1)/(2*BLOCK_SIZE)+1)*sizeof(float));

	cudaMemcpy(dInput, hInput, INPUT_SIZE * sizeof(float),cudaMemcpyHostToDevice);

	dim3 dimGrid((OUTPUT_SIZE-1)/SECTION_SIZE+1, 1,1);
	dim3 dimBlock(SECTION_SIZE,1,1);

	unsigned int len = INPUT_SIZE;
	unsigned int sum_len = (OUTPUT_SIZE-1)/(2*BLOCK_SIZE)+1;

	list_scan_kernel<<<dimGrid, dimBlock>>>(dInput, dOutput, dSum, len);
	list_scan_kernel2<<<dimGrid, dimBlock>>>(dSum, dSum, sum_len);
	consolidation_kernel<<<dimGrid,dimBlock>>>(dSum,dOutput);

	cudaMemcpy(hOutput, dOutput, OUTPUT_SIZE*sizeof(float),cudaMemcpyDeviceToHost);

	printf("The final result of list scan is:\n") ;
	for(int i=0; i<OUTPUT_SIZE; i++){
		printf("%3.0f,",hOutput[i]);
	}
		
	

	cudaFree(dInput);
	cudaFree(dOutput);
	cudaFree(dSum);

	free(hInput);
	free(hOutput);

	return 0;
}
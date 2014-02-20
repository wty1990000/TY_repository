#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#include <stdio.h>
#include <cstdlib>
#include <cstring>

#define BLOCK_SIZE 256
#define HISTOGRAM_LENGTH 256

__global__ void histo(unsigned char* buffer, unsigned int* histo, long size)
{
	__shared__ unsigned int private_histo[256];
	if(threadIdx.x < 256)
		private_histo[threadIdx.x] = 0;

	__syncthreads();

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	while (i<size)
	{
		atomicAdd(&(private_histo[buffer[i]]),1);
		i += stride;
	}
	__syncthreads();

	if(threadIdx.x < 256)
		atomicAdd(&(histo[threadIdx.x]), private_histo[threadIdx.x]);

}

int main()
{
	unsigned char h_buffer[] = "AAAAAA bbbbbb ccccccccc abcdefg hijklmn HAHAHA wojiushi HAHAH AAAAAA bbbbbb ccccccccc abcdefg hijklmn HAHAHA wojiushi HAHAH";
	unsigned int* h_histo;
	unsigned char* d_buffer;
	unsigned int*  d_histo;

	long size = strlen((char*)h_buffer);
	
	h_histo = (unsigned int*)malloc(HISTOGRAM_LENGTH*sizeof(unsigned int));

	cudaMalloc((void**)&d_buffer, size*sizeof(unsigned char));
	cudaMalloc((void**)&d_histo,  HISTOGRAM_LENGTH*sizeof(unsigned int));

	cudaMemcpy(d_buffer, h_buffer, size*sizeof(unsigned char), cudaMemcpyHostToDevice);

	dim3 dimBlock(BLOCK_SIZE,1,1);
	dim3 dimGrid((size-1)/BLOCK_SIZE+1,1,1);

	histo<<<dimGrid,dimBlock>>>(d_buffer, d_histo, size);

	cudaMemcpy(h_histo,d_histo,HISTOGRAM_LENGTH*sizeof(unsigned int),cudaMemcpyDeviceToHost);

	cudaFree(d_histo);
	cudaFree(d_buffer);

	printf("The result is:");
	for(int i = 0; i< HISTOGRAM_LENGTH; ++i)
		printf("%d,",h_histo[i]);

	free(h_histo);
	

	return 0;
}
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

const int iSize = 50;

__global__ void vector_addition_kernel(float *A, float *B, float *C, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n)
		C[i] = A[i] + B[i];
}
float vector_addition(float *h_A, float *h_B, float *h_C, int n)
{
	dim3 DimGrid((n-1)/16+1,1,1);
	dim3 DimBlock(16,1,1);
	int size = n *sizeof(float);
	float *d_A, *d_B, *d_C;
	cudaMalloc((void**)&d_A,size);
	cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_B,size);
	cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_C,size);

	vector_addition_kernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n);
	cudaMemcpy(h_C, d_C,size,cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	return *h_C;
}

int main()
{
	float h_A[iSize], h_B[iSize], h_C[iSize];

	for(int i=0; i<iSize; i++){
		h_A[i] = float(i);
		h_B[i] = float(i*i);
	}
	vector_addition(h_A,h_B,h_C,iSize);
	
	printf("The results are:\n");
	for(int i=0; i<iSize; i++)
		printf("%3.1f\t",h_C[i]);

	return 0;
}
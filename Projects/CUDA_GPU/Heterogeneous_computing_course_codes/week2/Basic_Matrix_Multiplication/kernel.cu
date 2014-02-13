
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

const int iTileSize = 2;

__global__ void matrixMultiplyKernel(float *A, float *B, float *C,
									 int m, int n, int k)
{
	int Row = threadIdx.y + blockDim.y * blockIdx.y;
	int Col = threadIdx.x + blockDim.x * blockIdx.x;

	 if(Row <m && Col <k){
		 float CValue = 0.0;
		 for(int i=0; i<n; i++)
			 CValue += A[Row*n+i]*B[i*k+Col];
		 C[Row*k+Col] = CValue;
	 }
}

void matrixMultiplication(float *h_A, float *h_B, float *h_C, int m, int n, int k)
{
	dim3 dimGrid((k-1)/iTileSize+1,(m-1)/iTileSize+1,1);
	dim3 dimBlock(iTileSize,iTileSize);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void**)&d_A, m*n*sizeof(float));
	cudaMemcpy(d_A,h_A,m*n*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_B, n*k*sizeof(float));
	cudaMemcpy(d_B,h_B,n*k*sizeof(float),cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_C, m*k*sizeof(float));

	matrixMultiplyKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, m, n, k);
	cudaMemcpy(h_C, d_C, k*m*sizeof(float),cudaMemcpyHostToDevice);
	
	cudaFree(h_A);
	cudaFree(h_B);
	cudaFree(h_C);
}

int main()
{
	float h_A[16], h_B[16], h_C[16];
	
	for(int y=0; y<16; y++){
		h_A[y] = float(y);
		h_B[y] = 1.0;
	}
	matrixMultiplication(h_A,h_B,h_C,4,4,4);	
	printf("The result is:\n");
	for(int j=0; j<4*4; j++){
		if((j+1)%4 !=0){
			printf("%3.1f\t",h_C[j]);
		}
		else{
			printf("%3.1f\n",h_C[j]);
		}
	}

	
	return 0;

}

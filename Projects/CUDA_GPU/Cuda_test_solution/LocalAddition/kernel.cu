
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>


__global__ void addKernel()
{
   __shared__ int Sum1[10][10];
   int Sum[10][10];
   
   if( threadIdx.x ==0 && threadIdx.y ==0){
	    for(int i=0; i<10; i++){
			 for(int j=0; j<10; j++){
				 Sum1[i][j] = 0;
			 }
		}
   }
   __syncthreads();
   for(int i=0; i<10; i++){
	   for(int j=0; j<10; j++){
		   Sum[i][j] = (i+1)*(j+1);
		   atomicAdd(&Sum1[i][j], Sum[i][j]);
	   }
   }
   __syncthreads();
	  if( threadIdx.x ==0 && threadIdx.y ==0){
		   for(int i=0; i<10; i++){
	   for(int j=0; j<10; j++){
		   printf("%d, ", Sum1[i][j]);
	   }
		   }
   }
}

int main()
{
	addKernel<<<1,16>>>();
	return 0;
}

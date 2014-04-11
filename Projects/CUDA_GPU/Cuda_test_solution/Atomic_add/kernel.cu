
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void test(float* output)
{
	__shared__ float temp, tmp1;
	float partial, cc;
	int tx = threadIdx.x;

	if(tx ==0){
		temp = 0.0;
		tmp1 =0.0;
	}

	partial = tx+1;

	//__syncthreads();
	//for(int i=0; i<2; i++){
	//	temp[blockIdx.x] += float(i);
	//}

	atomicAdd(&temp,partial);
	//atomicAdd(&(output[1]),partial);
	cc = partial - temp;

	atomicAdd(&tmp1,cc);

	if(tx ==0){
		output[blockIdx.x] = tmp1;
	}
	//output[blockIdx.x] = temp[blockIdx.x];
	//output[tx] = 0.0;
	//output[tx] = 0.0;
}

void launch(float* output)
{
	test<<<2,10>>>(output);
}

int main()
{
	float *out = (float*)malloc(2*sizeof(float));
	float *output;

	cudaMalloc((void**)&output, 2*sizeof(float));

	launch(output);

	cudaMemcpy(out,output,2*sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(output);
	for(int i=0; i<2; i++){
		printf("the result is: %f\t", out[i]);
	}
	free(out);
	

	return 0;
}
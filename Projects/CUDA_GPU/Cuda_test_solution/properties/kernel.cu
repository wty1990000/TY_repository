
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int main()
{
	int dev_count;
	cudaDeviceProp prop;
	cudaGetDeviceCount( &dev_count);
	for (int i = 0; i < dev_count; i++) {
		cudaGetDeviceProperties(&prop, i);
	}
	if (prop.deviceOverlap){
		printf("Device support CUDA streams\n");
	}
	printf("Device has %d SMs\n",prop.multiProcessorCount);
	printf("Device has %d threads per SMs",prop.maxThreadsPerMultiProcessor);
	printf("Device has %d threads per block",prop.maxThreadsPerBlock);
	
	return 0;
}
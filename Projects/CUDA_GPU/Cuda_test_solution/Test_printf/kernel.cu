#include "CudaComputing.cuh"

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

 

cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);

__global__ void addKernel(int *c, const int *a, const int *b)

{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}
int vvmain(int* c)
{
	const int arraySize = 5;
    const int a[arraySize] = {1, 2, 3, 4, 5 };
    const int b[arraySize] = {10, 20, 30, 40, 50 };
    // Add vectors inparallel.

    cudaError_t cudaStatus =addWithCuda(c, a, b, arraySize);
    if (cudaStatus !=cudaSuccess) {
        return 1;
    }

 

    // cudaDeviceReset must becalled before exiting in order for profiling and

    // tracing tools such asNsight and Visual Profiler to show complete traces.

    cudaStatus =cudaDeviceReset();
    if (cudaStatus !=cudaSuccess) {
        return 1;
    }
    return 0;
}
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{

    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;

    cudaError_t cudaStatus;
    // Choose which GPU to runon, change this on a multi-GPU system
    cudaStatus =cudaSetDevice(0);
    if (cudaStatus !=cudaSuccess) {
       goto Error;
    }
    // Allocate GPU buffersfor three vectors (two input, one output)   .
    cudaStatus =cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus !=cudaSuccess) {
        goto Error;
    }
    cudaStatus =cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus !=cudaSuccess) {
        goto Error;
    }
    cudaStatus =cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus !=cudaSuccess) {
        goto Error;

    }
    // Copy input vectors fromhost memory to GPU buffers.
    cudaStatus =cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus !=cudaSuccess) {
        goto Error;
    }
    cudaStatus =cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus !=cudaSuccess) {
        goto Error;
    }
    // Launch a kernel on theGPU with one thread for each element.
    addKernel<<<(size-1)/128+1,128>>>(dev_c, dev_a, dev_b);
    // cudaDeviceSynchronizewaits for the kernel to finish, and returns
    // any errors encounteredduring the launch.
    cudaStatus =cudaDeviceSynchronize();
    if (cudaStatus !=cudaSuccess) {
        goto Error;
    }
    // Copy output vector fromGPU buffer to host memory.
    cudaStatus = cudaMemcpy(c,dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus !=cudaSuccess) {
        goto Error;
    }
Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    return cudaStatus;
}
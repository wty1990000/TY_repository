
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "helper_cuda.h"
#include "helper_functions.h"

#include <stdio.h>

#include "kernel.cuh"

__global__ void RGradient_kernel(const double *d_InputIMGR, const double *d_InputIMGT,
								 double *d_OutputIMGR, double *d_OutputIMGT, 
								 double *d_OutputIMGRx, double *d_OutputIMGRy,
								 double *d_OutputIMGTx, double *d_OutputIMGTy, double *d_OutputIMGTxy,
								 int width, int height)
{
	//Map the threads to the pixel positions
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

	unsigned int i  = row * (width+2);		unsigned int j  = col;
	unsigned int i1 = (row+1) * (width+2);	unsigned int j1 = col +1;
	unsigned int i2 = (row+2) * (width+2);  unsigned int j2 = col +2;

	if(row < height && col < width){
		d_OutputIMGR[row*width+col]  = d_InputIMGR[i1+j1];
		d_OutputIMGRx[row*width+col] = 0.5 * (d_InputIMGR[i1+j2] - d_InputIMGR[i1+j]);
		d_OutputIMGRy[row*width+col] = 0.5 * (d_InputIMGR[i2+j1] - d_InputIMGR[i+j1]);

		d_OutputIMGT[row*width+col]  = d_InputIMGT[i1+j1];
		d_OutputIMGTx[row*width+col] = 0.5 * (d_InputIMGT[i1+j2] - d_InputIMGT[i1+j]);
		d_OutputIMGTy[row*width+col] = 0.5 * (d_InputIMGT[i2+j1] - d_InputIMGT[i+j1]);
		d_OutputIMGTxy[row*width+col]= 0.25 * (d_InputIMGT[i2+j2] - d_InputIMGT[i+j2] - d_InputIMGT[i2+j] + d_InputIMGT[i+j]);
	}
	

}

void launch_kernel(const double *h_InputIMGR, const double *h_InputIMGT,
								 double *h_OutputIMGR, double *h_OutputIMGT, 
								 double *h_OutputIMGRx, double *h_OutputIMGRy,
								 double *h_OutputIMGTx, double *h_OutputIMGTy, double *h_OutputIMGTxy,
								 int width, int height)
{
	double *d_InputIMGR, *d_InputIMGT;
	double *d_OutputIMGR, *d_OutputIMGT, *d_OutputIMGRx, *d_OutputIMGRy, *d_OutputIMGTx, *d_OutputIMGTy, *d_OutputIMGTxy;

	checkCudaErrors(cudaMalloc((void**)&d_InputIMGR, (width+2)*(height+2)*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_InputIMGT, (width+2)*(height+2)*sizeof(double)));

	checkCudaErrors(cudaMemcpy(d_InputIMGR,h_InputIMGR,(width+2)*(height+2)*sizeof(double),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_InputIMGT,h_InputIMGT,(width+2)*(height+2)*sizeof(double),cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGR, width*height*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGT, width*height*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGRx, width*height*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGRy, width*height*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTx, width*height*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTy, width*height*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_OutputIMGTxy, width*height*sizeof(double)));

	dim3 dimB(BLOCK_SIZE,BLOCK_SIZE,1);
	dim3 dimG((width-1)/BLOCK_SIZE+1,(height-1)/BLOCK_SIZE+1,1);

	RGradient_kernel<<<dimG, dimB>>>(d_InputIMGR,d_InputIMGT,
								d_OutputIMGR, d_OutputIMGT, 
								 d_OutputIMGRx, d_OutputIMGRy,
								 d_OutputIMGTx, d_OutputIMGTy, d_OutputIMGTxy,
								 width, height);

	//cudaDeviceSynchronize();

	checkCudaErrors(cudaMemcpy(h_OutputIMGR,d_OutputIMGR,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_OutputIMGT,d_OutputIMGT,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_OutputIMGRx,d_OutputIMGRx,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_OutputIMGRy,d_OutputIMGRy,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_OutputIMGTx,d_OutputIMGTx,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_OutputIMGTy,d_OutputIMGTy,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_OutputIMGTxy,d_OutputIMGTxy,width*height*sizeof(double),cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_OutputIMGR));
	checkCudaErrors(cudaFree(d_OutputIMGT));
	checkCudaErrors(cudaFree(d_OutputIMGRx));
	checkCudaErrors(cudaFree(d_OutputIMGRy));
	checkCudaErrors(cudaFree(d_OutputIMGTx));
	checkCudaErrors(cudaFree(d_OutputIMGTy));
	checkCudaErrors(cudaFree(d_OutputIMGTxy));
}


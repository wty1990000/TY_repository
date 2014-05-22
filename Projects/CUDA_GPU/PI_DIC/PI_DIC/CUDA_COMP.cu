#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "CUDA_COMP.cuh"
#include <iostream>
#include <vector>
#include <stdio.h>

//Initialize the CUDA runtime library
void cudaInit()
{
	cudaFree(0);
}


/*
--------------CUDA Kernels used for GPU computing--------------
*/
//Precopmutation kernel
__global__ void precomputation_kernel()
{

}

/*
--------------Interface Functions Declarition for code integration in .cu file--------------
*/
//Precomputation Interface
void precompoutation_interface(const std::vector<float>& hInput_Img1, const std::vector<float>& hInput_Img2, int iWidth, int iHeight,
							  float *fhOutputR, float *fhOutputRx, float *fhOutputRy,
							  float *fhOutputT, float *fhOutputTx, float *fhOutputTy,
							  float *fhOutputBicubicMatrix);

/*
--------------Interface Function CombinedComputation for host use--------------
*/


/*
--------------Interface Functions Definition for code integration in .cu file--------------
*/
//Precomputation Interface
void precompoutation_interface(const std::vector<float>& hInput_Img1, const std::vector<float>& hInput_Img2, int iWidth, int iHeight,
							  float *fhOutputR, float *fhOutputRx, float *fhOutputRy,
							  float *fhOutputT, float *fhOutputTx, float *fhOutputTy,
							  float *fhOutputBicubicMatrix)
/*Input: vector of image intensity values, image width and height (with 1 pixel border).
 Output: Image gradients: Rx, Ry, Tx, Ty, Txy, BicubicMatrix
Purpose: Precomputation Interface function for CPU use
*/
{
	float *fdOutputR, *fdOutputRx, *fdOutputRy;
	float *fdOutputT, *fdOutputTx, *fdOutputTy, *fTxy;
	float *fdOutputBicubicMatrix;
	const static float fhAlfaMatrix[16*16] = {
											1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
											0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ,
											-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
											2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
											0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 
											0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ,
											0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0,
											0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0 , 
											-3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0, 0, 0, 0, 0, 
											0, 0, 0, 0, -3, 0, 3, 0, 0, 0, 0, 0, -2, 0, -1, 0,  
											9, -9, -9, 9, 6, 3, -6, -3, 6, -6, 3, -3, 4, 2, 2, 1 , 
											-6, 6, 6, -6, -3, -3, 3, 3, -4, 4, -2, 2, -2, -2, -1, -1, 
											2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 , 
											0, 0, 0, 0, 2, 0, -2, 0, 0, 0, 0, 0, 1, 0, 1, 0 , 
											-6, 6, 6, -6, -4, -2, 4, 2, -3, 3, -3, 3, -2, -1, -2, -1,
											4, -4, -4, 4, 2, 2, -2, -2, 2, -2, 2, -2, 1, 1, 1, 1 
											};
	StopWatchWin precomputation;
}



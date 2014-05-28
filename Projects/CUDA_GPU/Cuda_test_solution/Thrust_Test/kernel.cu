#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.cuh"
#include "helper_cuda.h"
#include "helper_functions.h"
#include "thrust\host_vector.h"
#include "thrust\device_vector.h"

#include <stdio.h>

void hostVersion(float& time)
{
	StopWatchWin w;
	thrust::host_vector<float> hV;
	w.start();
	for(int i=0; i<21; i++){
		for(int j=0; j<21; j++){
		for(int l=0; l<33; l++)
			for(int m=0; m<33; m++)
			{
				hV.push_back(26.0f-16.0f+(float)l);
			}
		}
	}
	thrust::device_vector<float>dV = hV;
	w.stop();
	time = w.getTime();
}

void deViceVersion(float& time)
{
	StopWatchWin w;
	thrust::device_vector<float> hV;
	w.start();
	for(int i=0; i<21; i++){
		for(int j=0; j<21; j++){
			for(int l=0; l<33; l++)
				for(int m=0; m<33; m++)
				{
					hV.push_back(26.0f-16.0f+(float)l);
				}
		}
	}
	w.stop();
	time = w.getTime();
}

void init()
{
	cudaFree(0);
}

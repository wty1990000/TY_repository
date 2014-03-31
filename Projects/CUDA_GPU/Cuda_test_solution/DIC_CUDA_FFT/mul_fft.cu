#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Used 3rd libraries
#include "cufft.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
//CUDA helper fucntions
#include "helper_cuda.h"
#include "helper_functions.h"

#include <iostream>
#include <cassert>

using namespace std;

#ifndef REAL
#define REAL float
#endif // !REAL

template<typename Real>
class TestFFT2D
{
protected:
	int nGPU;	//Number of GPUs
	cudaStream_t *streams;
	int nPerCall;	//Number of FFTs per call
	thrust::host_vector<cufftHandle>fftPlanMany;
	int dim[2]	//2D FFT
	Real *h_data;
	long h_data_elements;
	long nfft, n2ft2d, h_memsize, nelements;
	long totalFFT;
	thrust::device_vector<Real*>d_data;
	long bytesPerGPU;
public:
	TestFFT2D(int _nGPU, Real* h_data, long _h_data_elements,
		int *_dim, int _nPerCall, cudaStream_t *_streams){
			nGPU = _nGPU;
			h_data = _h_data;
			h_data_elements = _h_data_elements;
			dim[0] = _dim[0]; dim[1] = _dim[1];
			nfft = dim[0] * dim[1];
			n2ft2d = 2*dim[0]*dim[1];
			totalFFT = h_data_elements/n2ft2d;
	}
};





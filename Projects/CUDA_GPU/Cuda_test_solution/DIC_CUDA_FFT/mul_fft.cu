#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//Used 3rd libraries
#include "cufft.h"
#include "cufftw.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
//CUDA helper fucntions
#include "helper_cuda.h"
#include "helper_functions.h"

using namespace std;

static __device__ __host__ inline cufftDoubleComplex ComplexMul(cufftDoubleComplex a, cufftDoubleComplex b)
{
	cufftDoubleComplex c;
	c.x = a.x * b.x - a.y * b.y;
	c.y = a.x *b.y + a.y * b.x;
	return c;
}
static __device__ __host__ inline cufftDoubleComplex ComplexScale(cufftDoubleComplex a, double s)
{
	cufftDoubleComplex c;
	c.x = s * a.x;
	c.y = s * a.y;
	return c;
}
__global__ void pointwiseMul(const cufftDoubleComplex* a, const cufftDoubleComplex* b, cufftDoubleComplex* c, int width, int height, double scale)
{
	unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

	if((row < height) && (col < width)){
		c[row*width+col] = ComplexScale(ComplexMul(a[row*width+col],b[row*width+col]),scale);
	}

}


void init()
{
	cudaFree(0);
}
void FFT2D_1(double* h_OutputC, float& time)
{
	StopWatchWin timer; 
	thrust::host_vector<double> h_InputA;
	thrust::host_vector<double> h_InputB;
	thrust::device_vector<double> d_InputA;  double *raw_ptrA;
	thrust::device_vector<double> d_InputB;  double *raw_ptrB;
	double* d_OutputC;

	cufftDoubleComplex *freqDomA, *freqDomB, *freqDomfg;
	cufftHandle plan, rplan;

	dim3 dimBlock(16,16,1);
	dim3 dimGrid((32-1)/16+1,(32-1)/16+1,1);

	for(int i=0; i<32*32; i++){
		h_InputA.push_back(double(i));
		h_InputB.push_back(double(i));
	}

	d_InputA = h_InputA;	raw_ptrA = thrust::raw_pointer_cast(d_InputA.data());
	d_InputB = h_InputB;	raw_ptrB = thrust::raw_pointer_cast(d_InputB.data());

	cudaMalloc((void**)&freqDomA, 32*(32/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&freqDomB, 32*(32/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&freqDomfg, 32*(32/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&d_OutputC, 32*32*sizeof(cufftDoubleComplex));

	//cuFFT plan
	cufftPlan2d(&plan,32,32,CUFFT_D2Z);
	cufftPlan2d(&rplan,32,32,CUFFT_Z2D);
	cufftSetCompatibilityMode(plan,CUFFT_COMPATIBILITY_NATIVE);
	cufftSetCompatibilityMode(rplan, CUFFT_COMPATIBILITY_NATIVE);
	
	timer.start();
	cufftExecD2Z(plan, raw_ptrA, freqDomA);
	cufftExecD2Z(plan, raw_ptrB, freqDomB);

	//Multiplication
	pointwiseMul<<<dimGrid,dimBlock>>>(freqDomA, freqDomB, freqDomfg, 32/2+1, 32, 1.0/(32*(32/2+1)));
	
	cufftExecZ2D(rplan, freqDomfg, d_OutputC);
	cudaMemcpy(h_OutputC,d_OutputC, 32*32*sizeof(double), cudaMemcpyDeviceToHost);
	
	cufftDestroy(plan);
	cufftDestroy(rplan);

	cudaFree(freqDomA);
	cudaFree(freqDomB);
	cudaFree(freqDomfg);
	cudaFree(d_OutputC);
	timer.stop();
	time = timer.getTime();
}
void FFT2D_FR(double* h_OutputC, float& time)
{
	StopWatchWin timer; 
	thrust::host_vector<double> h_InputA;
	thrust::device_vector<double> d_InputA;  double *raw_ptrA;
	double* d_OutputC;

	cufftDoubleComplex *freqDomA;
	cufftHandle plan, rplan;

	for(int i=0; i<32*32; i++){
		h_InputA.push_back(double(i));
	}

	d_InputA = h_InputA;	raw_ptrA = thrust::raw_pointer_cast(d_InputA.data());

	cudaMalloc((void**)&freqDomA, 32*(32/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&d_OutputC, 32*32*sizeof(cufftDoubleReal));

	//cuFFT plan
	cufftPlan2d(&plan,32,32,CUFFT_D2Z);
	cufftPlan2d(&rplan,32,32,CUFFT_Z2D);
	cufftSetCompatibilityMode(plan,CUFFT_COMPATIBILITY_NATIVE);
	cufftSetCompatibilityMode(rplan, CUFFT_COMPATIBILITY_NATIVE);
	
	timer.start();
	cufftExecD2Z(plan, raw_ptrA, freqDomA);
	cufftExecZ2D(rplan, freqDomA, d_OutputC);
	cudaMemcpy(h_OutputC,d_OutputC, 32*32*sizeof(double), cudaMemcpyDeviceToHost);
	
	cufftDestroy(plan);
	cufftDestroy(rplan);

	cudaFree(freqDomA);
	cudaFree(d_OutputC);
	timer.stop();
	time = timer.getTime();
}
void FFT2D_batchedIn(double* h_OutputC, float& time)
{
	StopWatchWin timer; 
	thrust::host_vector<double> h_InputA;
	thrust::device_vector<double> d_InputA;  double *raw_ptrA;
	//double* d_OutputC;
	cufftDoubleComplex* d_Freq;

	int batch = 4; int nRows = 32; int nCols = 32; 
	int n[2] = {nRows, nCols};

	int cols_padded = 2*(nCols/2+1);
	int inembed[2] = {nRows, 2*(nCols/2+1)};
	int onembed[2] = {nRows, (nCols/2+1)};

	
	cufftHandle forward_plan, inverse_plan;

	for(int i=0; i<nRows*cols_padded*batch; i++){
		h_InputA.push_back(double(0));
	}
	d_InputA = h_InputA;	raw_ptrA = thrust::raw_pointer_cast(h_InputA.data());
	timer.start();
	cufftPlanMany(&forward_plan,2,n,inembed,batch,1,onembed,batch,1,CUFFT_D2Z,batch);
	cufftPlanMany(&inverse_plan,2,n,onembed,batch,1,inembed,batch,1,CUFFT_Z2D,batch);

	d_Freq = reinterpret_cast<cufftDoubleComplex*>(raw_ptrA);

	cufftExecD2Z(forward_plan, raw_ptrA, d_Freq);
	cufftExecZ2D(inverse_plan, d_Freq, raw_ptrA);

	cudaMemcpy(h_OutputC,raw_ptrA,nRows*cols_padded*batch*sizeof(double),cudaMemcpyDeviceToHost);

	timer.stop();
	time = timer.getTime();
}
void FFT2D_batchedOut(double* h_OutputC, float& time)
{
	StopWatchWin timer; 
	thrust::host_vector<double> h_InputA, h_InputB;
	thrust::device_vector<double> d_InputA, d_InputB;  double *raw_ptrA, *raw_ptrB;
	//double* d_OutputC;
	cufftDoubleComplex* d_FreqA, *d_FreqB, *d_Freqfg;

	int batch = 441; int nRows = 32; int nCols = 32; 
	int n[2] = {nRows, nCols};

	int inembed[2] = {nRows, nCols};
	int onembed[2] = {nRows, (nCols/2+1)};

	
	cufftHandle forward_plan, inverse_plan;
	dim3 dimBlock(16,16,1);
	dim3 dimGrid((nRows-1)/16+1,(nCols/2)/16+1,1);


	for(int i=0; i<nRows*nCols*batch; i++){
		h_InputA.push_back(double(1));
		h_InputB.push_back(double(1));
	}
	d_InputA = h_InputA;	raw_ptrA = thrust::raw_pointer_cast(d_InputA.data());
	d_InputB = h_InputB;	raw_ptrB = thrust::raw_pointer_cast(d_InputB.data());

	cudaMalloc((void**)&d_FreqA, nRows*(nCols/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&d_FreqB, nRows*(nCols/2+1)*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&d_Freqfg, nRows*(nCols/2+1)*sizeof(cufftDoubleComplex));

	timer.start();
	cufftPlanMany(&forward_plan,2,n,inembed,batch,1,onembed,batch,1,CUFFT_D2Z,batch);
	cufftPlanMany(&inverse_plan,2,n,onembed,batch,1,inembed,batch,1,CUFFT_Z2D,batch);
	cufftExecD2Z(forward_plan, raw_ptrA, d_FreqA);
	cufftExecD2Z(forward_plan, raw_ptrB, d_FreqB);

	//pointwiseMul<<<dimGrid,dimBlock>>>(d_FreqA, d_FreqB, d_Freqfg,nCols/2+1, nRows, 1.0/((nCols/2+1)*nRows));

	cufftExecZ2D(inverse_plan, d_Freqfg, raw_ptrA);

	timer.stop();
	time = timer.getTime();
	cudaMemcpy(h_OutputC,raw_ptrA,nRows*nCols*batch*sizeof(double),cudaMemcpyDeviceToHost);

	cudaFree(d_FreqA);
	cudaFree(d_FreqB);
	cudaFree(d_Freqfg);
}

#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

const int BLOCK_SIZE = 16;

void precompute_kernel(const double *h_InputIMGR, const double *h_InputIMGT,
								 double *h_OutputIMGR, double *h_OutputIMGT, 
								 double *h_OutputIMGRx, double *h_OutputIMGRy,
								 /*double *h_OutputIMGTx, double *h_OutputIMGTy, double *h_OutputIMGTxy,*/double *h_OutputdTBicubic,
								 int width, int height, float& time);
void initialize_CUDA();
#endif // !_KENEL_CUH_

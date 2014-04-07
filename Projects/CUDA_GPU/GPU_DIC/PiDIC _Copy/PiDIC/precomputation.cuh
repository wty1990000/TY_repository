#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

const int BLOCK_SIZE = 16;

void precompute_kernel(const double *h_InputIMGR, const double *h_InputIMGT,
								 double *d_OutputIMGR, double *d_OutputIMGT, 
								 double *d_OutputIMGRx, double *d_OutputIMGRy,
								 double *d_OutputdTBicubic,
								 int width, int height, float& time)
#endif // !_KENEL_CUH_

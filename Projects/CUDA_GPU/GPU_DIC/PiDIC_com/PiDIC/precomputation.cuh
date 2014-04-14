#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

const int BLOCK_SIZE = 16;

void precompute_kernel(const float *h_InputIMGR, const float *h_InputIMGT,
								 float *d_OutputIMGR, float *d_OutputIMGT, 
								 float *d_OutputIMGRx, float *d_OutputIMGRy,
								 float *d_OutputdTBicubic,
								 int width, int height, float& time);
#endif // !_KENEL_CUH_

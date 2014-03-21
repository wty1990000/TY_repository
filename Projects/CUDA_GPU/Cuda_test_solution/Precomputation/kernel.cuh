#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

#define BLOCK_SIZE 6


void launch_kernel(const double *h_InputIMGR, const double *h_InputIMGT,
								 double *h_OutputIMGR, double *h_OutputIMGT, 
								 double *h_OutputIMGRx, double *h_OutputIMGRy,
								 double *h_OutputIMGTx, double *h_OutputIMGTy, double *h_OutputIMGTxy,
								 int width, int height);

#endif // !_KENEL_CUH_

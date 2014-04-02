#ifndef _KERNEL_H_
#define _KERNEL_H_

#define BLOCK_WIDTH 16

void launch_kernel(const double *h_InputIMGR, const double *h_InputIMGT,
								 double *h_OutputIMGR, double *h_OutputIMGT, 
								 double *h_OutputIMGRx, double *h_OutputIMGRy,
								 double *h_OutputIMGTx, double *h_OutputIMGTy, double *h_OutputIMGTxy,double *h_OutputdTBicubic,
								 int width, int height);
void freeing(void* pointer);

#endif // !_KERNEL_H_

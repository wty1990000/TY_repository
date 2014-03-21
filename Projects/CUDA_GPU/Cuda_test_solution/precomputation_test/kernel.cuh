#ifndef _KERNEL_CUH_
#define _KERNEL_CUH_

const int BLOCK_SIZE = 2;

void launchkernel(double *h_input, double *h_output, int size);

#endif // !_KENEL_CUH_

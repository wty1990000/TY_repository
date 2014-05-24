#ifndef PIDIC_CUH_
#define PIDIC_CUH_

//Initialize CUDA Runtime Library
void InitCuda();
void computation(const float* ImgR, const float* ImgT, int iWidth, int iHeight);
#endif // !PIDIC_CUH_

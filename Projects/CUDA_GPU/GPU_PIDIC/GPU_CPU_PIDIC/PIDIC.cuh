#ifndef PIDIC_CUH_
#define PIDIC_CUH_

//Initialize CUDA Runtime Library
void InitCuda();
void computation_interface(const std::vector<float>& ImgR, const std::vector<float>& ImgT, int iWidth, int iHeight);

#endif // !PIDIC_CUH_

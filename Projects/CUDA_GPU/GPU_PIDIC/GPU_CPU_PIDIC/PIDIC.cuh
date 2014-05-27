#ifndef PIDIC_CUH_
#define PIDIC_CUH_

//Initialize CUDA Runtime Library
void InitCuda();
void precomputation_interface(const std::vector<float>& h_InputIMGR, const std::vector<float>& h_InputIMGT, int width, int height, float& time,
							 float* fOutputIMGR, float* fOutputIMGT, float* fOutputIMGRx, float* fOutputIMGRy, float* fOutputBicubic);

#endif // !PIDIC_CUH_

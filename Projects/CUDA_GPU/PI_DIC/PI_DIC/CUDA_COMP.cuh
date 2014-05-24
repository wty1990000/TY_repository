#ifndef CUDA_COMP_CUH
#define CUDA_COMP_CUH

#include <vector>

void cudaInit();		//Initilize the CUDA runtime library
//Combined Computation Interface function
void combined_function(const std::vector<float>& hInput_IMGR, const std::vector<float>& h_InputIMGT, float* hInput_dPXY, int width, int height,
					   int iSubsetH, int iSubsetW, int iSubsetX, int iSubsetY, int iNumberX, int iNumberY, int iFFTSubH, int iFFTSubW, int iIterationNum, float fDeltaP, 
					   int* iU, int* iV, float* fZNCC, float* fdP, float& fTimePrecomputation, float& fTimeFFT, float& fTimeICGN, float& fTimeTotal);



#endif // !CUDA_COMP_CUH

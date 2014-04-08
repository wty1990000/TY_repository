#ifndef _COMBINATION_H_
#define _COMBINATION_H_

void initialize_CUDA();

void combined_functions(const double* h_InputIMGR, const double* h_InputIMGT, double* h_OutputFFTcc);

#endif // !_COMBINATION_H_

#ifndef _MUL_FFT_H_
#define _MUL_FFT_H_

void init();
void FFT2D_1(double* h_OutputC, float& time);
void FFT2D_FR(double* h_OutputC, float& time);
void FFT2D_batchedIn(double* h_OutputC, float& time);
void FFT2D_batchedOut(double* h_OutputC, float& time);

#endif // !_MUL_FFT_H_

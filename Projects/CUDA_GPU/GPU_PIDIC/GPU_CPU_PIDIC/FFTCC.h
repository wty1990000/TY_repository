#ifndef FFTCC_H_
#define FFTCC_H_

void FFT_CC_interface(float* hInput_dR, float* hInput_dT, float* hInput_dPXY, int iNumberY, int iNumberX, int iFFTSubH, int iFFTSubW, 
					  int width, int height, int iSubsetY, int iSubsetX,
					  float* fZNCC, int* iCorrPeakX, int* iCorrPeakY, float& time);

#endif // !FFTCC_H_
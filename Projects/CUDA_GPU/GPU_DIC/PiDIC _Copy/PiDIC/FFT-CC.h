#ifndef FFT_CC_H
#define FFT_CC_H

void FFT_CC(const double* m_dR, const double* m_dT, const double* m_dPXY, const int& m_iNumberY, const int& m_iNumberX, 
			const int& m_iFFTSubH, const int& m_iFFTSubW, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetY, const int& m_iSubsetX,
			int* m_iFlag1, double* m_dZNCC, double* m_dP, int* m_iCorrPeakX, int* m_iCorrPeakY,float& time);

#endif // !FFT_CC_H

#ifndef FFT_CC_H
#define FFT_CC_H

void FFT_CC(const float* m_dR, const float* m_dT, const float* m_dPXY, const int& m_iNumberY, const int& m_iNumberX, 
			const int& m_iFFTSubH, const int& m_iFFTSubW, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetY, const int& m_iSubsetX,
			float* m_dZNCC, int* m_iCorrPeakX, int* m_iCorrPeakY, float& time);

#endif // !FFT_CC_H

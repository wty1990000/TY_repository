#ifndef _COMBINATION_H_
#define _COMBINATION_H_

void initialize();

void combined_functions(const float* h_InputIMGR, const float* h_InputIMGT, const float* m_dPXY, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetH, const int& m_iSubsetW, const float& m_dNormDeltaP,
						const int& m_iSubsetX, const int& m_iSubsetY, const int& m_iNumberX, const int& m_iNumberY, const int& m_iFFTSubH, const int& m_iFFTSubW, const int& m_maxIteration,
						int* m_iU, int *m_iV, float* m_dZNCC, float* m_dP, int* m_iIterationNum,
						float& precompute_tme, float& fft_time, float& icgn_time, float& total_time);

#endif // !_COMBINATION_H_

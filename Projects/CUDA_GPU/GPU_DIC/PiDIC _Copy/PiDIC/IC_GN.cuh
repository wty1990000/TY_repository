#ifndef _IC_GN_H
#define _IC_GN_H

void launch_ICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, const float& m_dNormDeltaP,
				 const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV,
				 const int& m_iNumberY, const int& m_iNumberX, const int& m_iSubsetH, const int& m_iSubsetW, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetY, const int& m_iSubsetX, const int& m_iMaxiteration,
				 float* output_dP, int* dm_iIterationNum, float& time);

#endif // !_IC_GN_H

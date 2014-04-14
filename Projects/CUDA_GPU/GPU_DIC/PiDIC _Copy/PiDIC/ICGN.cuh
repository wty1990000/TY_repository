#ifndef _IC_GN_H_
#define _IC_GN_H_

void launch_ICGN(const float* input_dPXY, const float* input_mdR, const float* input_mdRx, const float* input_mdRy, float m_dNormDeltaP,
				 const float* input_mdT, const float* input_mBicubic, const int* input_iU, const int* input_iV,
				 int m_iNumberY, int m_iNumberX, int m_iSubsetH, int m_iSubsetW, int m_iWidth, int m_iHeight, int m_iSubsetY, int m_iSubsetX, int m_iMaxiteration,
				 float* output_dP, int* dm_iIterationNum, float& time);

#endif // !_IC_GN_H_

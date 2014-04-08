#ifndef _FFTCC_H_
#define _FFTCC_H_

#define BLOCK_SIZE 16

void FFTCC_kernel(const double* dInput_mdR, const double* dInput_mdT, int m_iWidth, int m_iHeight,
				  double* dOutputm_dPXY,  int* dOutput_iFlag1, double* dm_SubsetC,
				  int m_iMarginX, int m_iMarginY, int m_iGridSpaceY, int m_iGridSapceX,
				  int m_iSubsetX, int m_iSubsetY, int m_iNumberX, int m_iNumberY,int m_iFFTSubW, int m_iFFTSubH,float& time);

#endif // !_FFTCC_H_

#include "FFT-CC.h"
#include "stdafx.h"
#include "afxdialogex.h"
#include "fftw3.h"
#include "helper_functions.h"


void FFT_CC(const float* m_dR, const float* m_dT, const float* m_dPXY, const int& m_iNumberY, const int& m_iNumberX, 
			const int& m_iFFTSubH, const int& m_iFFTSubW, const int& m_iWidth, const int& m_iHeight, const int& m_iSubsetY, const int& m_iSubsetX,
			float* m_dZNCC, int* m_iCorrPeakX, int* m_iCorrPeakY, float& time)
/*Input: m_dR, m_dT, m_dPXY, m_*'s
 Output: m_iFlag1, m_dZNCC, m_dP, m_iCorrPeakX, m_iCorrPeakY
Purpose: FFT-CC computation, only needs to transfre m_dR and m_dT from GPU
*/
{
	StopWatchWin sw;
	int m_dAvef = 0; // R_m
	int m_dAveg = 0; // T_m
	int m_dModf = 0; // sqrt (Sigma(R_i - R_m)^2)
	int m_dModg = 0; // sqrt (Sigma(T_i - T_m)^2)
	int m_dCorrPeak = -2; // maximum C
	int m_iCorrPeakXY = 0; // loacatoin of maximum C
	float m_dTemp;//parameter for normalization

	float *m_Subset1 = (float*)malloc(m_iFFTSubH*m_iFFTSubW*sizeof(float));
	float *m_Subset2 = (float*)malloc(m_iFFTSubH*m_iFFTSubW*sizeof(float));
	float *m_SubsetC = (float*)malloc(m_iFFTSubH*m_iFFTSubW*sizeof(float));

	fftwf_complex *m_FreqDom1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)* m_iFFTSubW * (m_iFFTSubH / 2 + 1));
	fftwf_complex *m_FreqDom2 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)* m_iFFTSubW * (m_iFFTSubH / 2 + 1));
	fftwf_complex *m_FreqDomfg = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)* m_iFFTSubW * (m_iFFTSubH / 2 + 1));

	fftwf_plan m_fftwPlan1 = fftwf_plan_dft_r2c_2d(m_iFFTSubW, m_iFFTSubH, m_Subset1, m_FreqDom1, FFTW_ESTIMATE);
	fftwf_plan m_fftwPlan2 = fftwf_plan_dft_r2c_2d(m_iFFTSubW, m_iFFTSubH, m_Subset2, m_FreqDom2, FFTW_ESTIMATE);
	fftwf_plan m_rfftwPlan = fftwf_plan_dft_c2r_2d(m_iFFTSubW, m_iFFTSubH, m_FreqDomfg, m_SubsetC, FFTW_ESTIMATE);

	sw.start();
	for (int i = 0; i < m_iNumberY; i++)
	{
		for (int j = 0; j < m_iNumberX; j++)
		{
			m_dAvef = 0; // R_m
			m_dAveg = 0; // T_m
			// Feed the gray intensity values into subsets
			for (int l = 0; l < m_iFFTSubH; l++)
			{
				for (int m = 0; m < m_iFFTSubW; m++)
				{
					m_Subset1[(l * m_iFFTSubW + m)] = m_dR[int(m_dPXY[(i*m_iNumberX+j)*2+0] - m_iSubsetY + l)*m_iWidth+int(m_dPXY[(i*m_iNumberX+j)*2+1] - m_iSubsetX + m)];
					m_dAvef += (m_Subset1[l * m_iFFTSubW + m] / (m_iFFTSubH * m_iFFTSubW));
					m_Subset2[(l * m_iFFTSubW + m)] = m_dT[int(m_dPXY[(i*m_iNumberX+j)*2+0] - m_iSubsetY + l)*m_iWidth+int(m_dPXY[(i*m_iNumberX+j)*2+1] - m_iSubsetX + m)];
					m_dAveg += (m_Subset2[l * m_iFFTSubW + m] / (m_iFFTSubH * m_iFFTSubW));
				}
			}
			m_dModf = 0; // sqrt (Sigma(R_i - R_m)^2)
			m_dModg = 0; // sqrt (Sigma(T_i - T_m)^2)
			for (int l = 0; l < m_iFFTSubH; l++)
			{
				for (int m = 0; m < m_iFFTSubW; m++)
				{
					m_Subset1[(l * m_iFFTSubW + m)] -= m_dAvef;
					m_Subset2[(l * m_iFFTSubW + m)] -= m_dAveg;
					m_dModf += pow((m_Subset1[l * m_iFFTSubW + m]), 2);
					m_dModg += pow((m_Subset2[l * m_iFFTSubW + m]), 2);
				}
			}
			//FFT-CC algorithm accelerated by FFTW
			fftwf_execute(m_fftwPlan1);
			fftwf_execute(m_fftwPlan2);
			for (int n = 0; n < m_iFFTSubW * (m_iFFTSubH / 2 + 1); n++)
			{
				m_FreqDomfg[n][0] = (m_FreqDom1[n][0] * m_FreqDom2[n][0]) + (m_FreqDom1[n][1] * m_FreqDom2[n][1]);
				m_FreqDomfg[n][1] = (m_FreqDom1[n][0] * m_FreqDom2[n][1]) - (m_FreqDom1[n][1] * m_FreqDom2[n][0]);
			}
			fftwf_execute(m_rfftwPlan);
			m_dCorrPeak = -2; // maximum C
			m_iCorrPeakXY = 0; // loacatoin of maximum C
			m_dTemp = sqrt(m_dModf * m_dModg) * m_iFFTSubW * m_iFFTSubH; //parameter for normalization

			// Search for maximum C, meanwhile normalize C
			for (int k = 0; k < m_iFFTSubW * m_iFFTSubH; k++)
			{
				m_SubsetC[k] /= m_dTemp;
				if (m_dCorrPeak < m_SubsetC[k])
				{
					m_dCorrPeak = m_SubsetC[k];
					m_iCorrPeakXY = k;
				}
			}
			// calculate the loacation of maximum C
			m_iCorrPeakX[i*m_iNumberX+j] = m_iCorrPeakXY % m_iFFTSubW;
			m_iCorrPeakY[i*m_iNumberX+j] = int(m_iCorrPeakXY / m_iFFTSubW);

			// Shift the C peak to the right quadrant 
			if (m_iCorrPeakX[i*m_iNumberX+j] > m_iSubsetX)
			{
				m_iCorrPeakX -= m_iFFTSubW;
			}
			if (m_iCorrPeakY[i*m_iNumberX+j] > m_iSubsetY)
			{
				m_iCorrPeakY -= m_iFFTSubH;
			}
			m_dZNCC[i*m_iNumberX+j] = m_dCorrPeak; // save the ZNCC
		}
	}
	sw.stop();
	time = sw.getTime();

	//Free FFTW data sets
	fftwf_destroy_plan(m_fftwPlan1);
	fftwf_destroy_plan(m_fftwPlan2);
	fftwf_destroy_plan(m_rfftwPlan);
	fftwf_free(m_FreqDom1);
	fftwf_free(m_FreqDom2);
	fftwf_free(m_FreqDomfg);
	free(m_Subset1);
	free(m_Subset2);
	free(m_SubsetC);
}
 
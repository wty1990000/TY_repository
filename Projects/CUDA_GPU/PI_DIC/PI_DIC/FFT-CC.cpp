#include "fftw3.h"
#include "FFT-CC.h"
#include "helper_functions.h"

void FFT_CC_interface(float* hInput_dR, float* hInput_dT, float* hInput_dPXY, int iNumberY, int iNumberX, int iFFTSubH, int iFFTSubW, 
					  int width, int height, int iSubsetY, int iSubsetX,
					  float* fZNCC, int* iCorrPeakX, int* iCorrPeakY, float& time)
{
	StopWatchWin FFTWatch;

	float fAvef = 0.0f;	//Rm
	float fAveg = 0.0f;	//Tm
	float fModf = 0.0f;	//sqrt(Sigma(Ri - Rm)^2)
	float fModg = 0.0f;	//sqrt(Sigma(Ti - Tm)^2)
	float fCorrPeak = -2.0f;	//Maximum C
	int iCorrPeakXY = 0;	//location of maximum C
	float fTempNorm;	//parameter for normalization

	float *fSubset1 = (float*)malloc(iFFTSubH*iFFTSubW*sizeof(float));
	float *fSubset2 = (float*)malloc(iFFTSubH*iFFTSubW*sizeof(float));
	float *fSubsetC = (float*)malloc(iFFTSubH*iFFTSubW*sizeof(float));

	fftwf_complex *FreqDom1 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iFFTSubW*(iFFTSubH/2+1));
	fftwf_complex *FreqDom2 = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iFFTSubW*(iFFTSubH/2+1));
	fftwf_complex *FreqDomfg = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex)*iFFTSubW*(iFFTSubH/2+1));

	fftwf_plan fftwPlan1 = fftwf_plan_dft_r2c_2d(iFFTSubW, iFFTSubH, fSubset1, FreqDom1, FFTW_ESTIMATE);
	fftwf_plan fftwPlan2 = fftwf_plan_dft_r2c_2d(iFFTSubW, iFFTSubH, fSubset2, FreqDom2, FFTW_ESTIMATE);
	fftwf_plan rfftwPlan = fftwf_plan_dft_c2r_2d(iFFTSubW, iFFTSubH, FreqDomfg, fSubsetC, FFTW_ESTIMATE);

	FFTWatch.start();

	for (int i = 0; i < iNumberY; i++)
	{
		for (int j = 0; j < iNumberX; j++)
		{
			fAvef = 0.0; // R_m
			fAveg = 0.0; // T_m
			// Feed the gray intensity values into subsets
			for (int l = 0; l < iFFTSubH; l++)
			{
				for (int m = 0; m < iFFTSubW; m++)
				{
					fSubset1[(l * iFFTSubW + m)] = hInput_dR[int(hInput_dPXY[(i*iNumberX+j)*2+0] - iSubsetY + l)*width+int(hInput_dPXY[(i*iNumberX+j)*2+1] - iSubsetX + m)];
					fAvef += (fSubset1[l * iFFTSubW + m] / (iFFTSubH * iFFTSubW));
					fSubset2[(l * iFFTSubW + m)] = hInput_dT[int(hInput_dPXY[(i*iNumberX+j)*2+0] - iSubsetY + l)*width+int(hInput_dPXY[(i*iNumberX+j)*2+1] - iSubsetX + m)];
					fAveg += (fSubset2[l * iFFTSubW + m] / (iFFTSubH * iFFTSubW));
				}
			}
			fModf = 0; // sqrt (Sigma(R_i - R_m)^2)
			fModg = 0; // sqrt (Sigma(T_i - T_m)^2)
			for (int l = 0; l < iFFTSubH; l++)
			{
				for (int m = 0; m < iFFTSubW; m++)
				{
					fSubset1[(l * iFFTSubW + m)] -= fAvef;
					fSubset2[(l * iFFTSubW + m)] -= fAveg;
					fModf += pow((fSubset1[l * iFFTSubW + m]), 2);
					fModg += pow((fSubset2[l * iFFTSubW + m]), 2);
				}
			}
			//FFT-CC algorithm accelerated by FFTW
			fftwf_execute(fftwPlan1);
			fftwf_execute(fftwPlan2);
			for (int n = 0; n < iFFTSubW * (iFFTSubH / 2 + 1); n++)
			{
				FreqDomfg[n][0] = (FreqDom1[n][0] * FreqDom2[n][0]) + (FreqDom1[n][1] * FreqDom2[n][1]);
				FreqDomfg[n][1] = (FreqDom1[n][0] * FreqDom2[n][1]) - (FreqDom1[n][1] * FreqDom2[n][0]);
			}
			fftwf_execute(rfftwPlan);
			fCorrPeak = -2; // maximum C
			iCorrPeakXY = 0; // loacatoin of maximum C
			iCorrPeakXY = sqrt(fModf * fModg) * float(iFFTSubW * iFFTSubH); //parameter for normalization

			// Search for maximum C, meanwhile normalize C
			for (int k = 0; k < iFFTSubW * iFFTSubH; k++)
			{
				fSubsetC[k] /= iCorrPeakXY;
				if (fCorrPeak < fSubsetC[k])
				{
					fCorrPeak = fSubsetC[k];
					iCorrPeakXY = k;
				}
			}
			// calculate the loacation of maximum C
			iCorrPeakX[i*iNumberX+j] = iCorrPeakXY % iFFTSubW;
			iCorrPeakY[i*iNumberX+j] = int(iCorrPeakXY / iFFTSubW);

			// Shift the C peak to the right quadrant 
			if (iCorrPeakX[i*iNumberX+j] > iSubsetX)
			{
				iCorrPeakX[i*iNumberX+j] -= iFFTSubW;
			}
			if (iCorrPeakY[i*iNumberX+j] > iSubsetY)
			{
				iCorrPeakY[i*iNumberX+j] -= iFFTSubH;
			}
			fZNCC[i*iNumberX+j] = fCorrPeak; // save the ZNCC
		}
	}
	FFTWatch.stop();
	time = FFTWatch.getTime();

	//Free FFTW data sets
	fftwf_destroy_plan(fftwPlan1);
	fftwf_destroy_plan(fftwPlan2);
	fftwf_destroy_plan(rfftwPlan);
	fftwf_free(FreqDom1);
	fftwf_free(FreqDom2);
	fftwf_free(FreqDomfg);
	free(fSubset1);
	free(fSubset2);
	free(fSubsetC);


}
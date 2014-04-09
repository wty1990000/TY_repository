#include <iostream>
#include "loop_plan.h"
#include "fftw3.h"
#include "Random.h"
#include "helper_timer.h"

void loopplan(float &t)
{
	int SubW = 32;
	int SubH = 32;
	int batch = 21;
	
	Random r;
	StopWatchWin ti;

	double *data1 = new double[SubW*SubH];
	double *data2 = new double[SubW*SubH];
	double *result = new double[SubW*SubH];

	fftw_complex *freq1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SubW*(SubH/2+1));
	fftw_complex *freq2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SubW*(SubH/2+1));
	fftw_complex *freqfg = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*SubW*(SubH/2+1));

	fftw_plan Plan1 = fftw_plan_dft_r2c_2d(SubW,SubH,data1,freq1,FFTW_ESTIMATE);
	fftw_plan Plan2 = fftw_plan_dft_r2c_2d(SubW,SubH,data2,freq2,FFTW_ESTIMATE);
	fftw_plan rPlan = fftw_plan_dft_c2r_2d(SubW,SubH,freqfg,result,FFTW_ESTIMATE);

	for(int i=0; i<batch; i++){
		for(int j=0; j<batch; j++){
			for(int l=0; l<SubH; l++){
				for(int m=0; m<SubW; m++){
					data1[l*SubW+m] = double(r.random_integer(0,255));
					data2[l*SubW+m] = double(r.random_integer(0,255));
				}
			}
			ti.start();
			fftw_execute(Plan1);
			fftw_execute(Plan2);
			ti.stop();
			t=ti.getTime();
			for(int n=0; n<SubW*(SubH/2+1); n++){
				freqfg[n][0] = (freq1[n][0]*freq2[n][0])+(freq1[n][1]*freq2[n][1]);
				freqfg[n][1] = (freq1[n][0]*freq2[n][1])-(freq1[n][1]*freq2[n][0]);
			}
			fftw_execute(rPlan);
		}
	}

	delete data1;
	delete data2;
	delete result;
	fftw_destroy_plan(Plan1);
	fftw_destroy_plan(Plan2);
	fftw_destroy_plan(rPlan);
	fftw_free(freq1);
	fftw_free(freq2);
	fftw_free(freqfg);
}
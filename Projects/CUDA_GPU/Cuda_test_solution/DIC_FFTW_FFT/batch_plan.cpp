#include "batch_plan.h"
#include <iostream>
#include "fftw3.h"
#include "Random.h"
#include "helper_timer.h"

void batchedplan(float &t)
{
	Random r;
	StopWatchWin ti;

	int SubW = 32;
    int SubH = 32;
    int batch = 21;

	int n[2] = {SubW,SubH};
	int inembed[2] = {SubW, SubH};
	int onembed[2] = {SubW, (SubH/2+1)};

	double *data1 = new double[batch*batch*SubW*SubH];
	double *data2 = new double[batch*batch*SubW*SubH];
	double *result = new double[batch*batch*SubW*SubH];

	fftw_complex *freq1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*batch*batch*SubW*(SubH/2+1));
	fftw_complex *freq2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*batch*batch*SubW*(SubH/2+1));
	fftw_complex *freqfg = (fftw_complex*)fftw_malloc(sizeof(fftw_complex)*batch*batch*SubW*(SubH/2+1));

	fftw_plan Plan1 = fftw_plan_many_dft_r2c(2,n,batch*batch,data1,inembed,batch*batch,1,freq1,onembed,batch*batch,1,FFTW_ESTIMATE);
	fftw_plan Plan2 = fftw_plan_many_dft_r2c(2,n,batch*batch,data2,inembed,batch*batch,1,freq2,onembed,batch*batch,1,FFTW_ESTIMATE);
	fftw_plan rPlan = fftw_plan_many_dft_c2r(2,n,batch*batch,freqfg,onembed,batch*batch,1,result,inembed,batch*batch,1,FFTW_ESTIMATE);

//#pragma omp parallel for
	for(int i=0; i<batch; i++){
		for(int j=0; j<batch; j++){
			for(int l=0; l<SubH; l++){
				for(int m=0; m<SubW; m++){
					data1[(((i*batch)+j)*SubH+l)*SubW+m] = double(r.random_integer(0,255));
					data2[(((i*batch)+j)*SubH+l)*SubW+m] = double(r.random_integer(0,255));
				}
			}
		}
	}

	ti.start();
	fftw_execute_dft_r2c(Plan1,data1,freq1);
	fftw_execute_dft_r2c(Plan2,data2,freq2);
	ti.stop();
	t = ti.getTime();
	
//#pragma omp parallel for
	for(int i=0; i<batch; i++){
		for(int j=0; j<batch; j++){
			for(int n=0; n<SubW*(SubH/2+1); n++){
				freqfg[((i*batch)+j)*SubW*(SubH/2+1)+n][0] = (freq1[((i*batch)+j)*SubW*(SubH/2+1)+n][0]*freq2[((i*batch)+j)*SubW*(SubH/2+1)+n][0])+(freq1[((i*batch)+j)*SubW*(SubH/2+1)+n][1]*freq2[((i*batch)+j)*SubW*(SubH/2+1)+n][1]);
				freqfg[((i*batch)+j)*SubW*(SubH/2+1)+n][1] = (freq1[((i*batch)+j)*SubW*(SubH/2+1)+n][0]*freq2[((i*batch)+j)*SubW*(SubH/2+1)+n][1])-(freq1[((i*batch)+j)*SubW*(SubH/2+1)+n][1]*freq2[((i*batch)+j)*SubW*(SubH/2+1)+n][0]);
			}
		}
	}



	fftw_execute_dft_c2r(Plan1,freqfg,result);

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
#include <iostream>
#include "kernel.cuh"
#include <stdio.h>

using namespace std;

int main()
{
	double *h_InputIMGR, *h_InputIMGT;
	double *h_OutputIMGR,*h_OutputIMGT; 
	double *h_OutputIMGRx, *h_OutputIMGRy;
	double *h_OutputIMGTx, *h_OutputIMGTy, *h_OutputIMGTxy;

	h_InputIMGR = (double*)malloc(16*16*sizeof(double));
	h_InputIMGT = (double*)malloc(16*16*sizeof(double));


	for(int i=0; i<16; i++){
		for(int j=0; j<16; j++){
			h_InputIMGR[i*16+j] = 0.0;
			h_InputIMGT[i*16+j] = 0.0;
		}
	}

	for(int i=0; i<16; i++){
		for(int j=0; j<16; j++){
			cout<<h_InputIMGR[i*16+j]<<","<<h_InputIMGT[i*16+j]<<endl;
		}
	}

	h_OutputIMGR = (double*)malloc(14*14*sizeof(double));
	h_OutputIMGT = (double*)malloc(14*14*sizeof(double));
	h_OutputIMGRx = (double*)malloc(14*14*sizeof(double));
	h_OutputIMGRy = (double*)malloc(14*14*sizeof(double));
	h_OutputIMGTx = (double*)malloc(14*14*sizeof(double));
	h_OutputIMGTy = (double*)malloc(14*14*sizeof(double));
	h_OutputIMGTxy = (double*)malloc(14*14*sizeof(double));

	launch_kernel(h_InputIMGR,h_InputIMGT,h_OutputIMGR,h_OutputIMGT,
				  h_OutputIMGRx, h_OutputIMGRy,h_OutputIMGTx,h_OutputIMGTy,h_OutputIMGTxy,14,14);


	for(int i=0; i<14; i++)
		for(int j=0; j<14; j++){
			cout<<h_OutputIMGR[i*14+j]<<","<<h_OutputIMGT[i*14+j]
			<<","<<h_OutputIMGRx[i*14+j]<<","<<h_OutputIMGRy[i*14+j]
			<<","<<h_OutputIMGTx[i*14+j]<<","<<h_OutputIMGTy[i*14+j]<<","<<h_OutputIMGTxy[i*14+j]<<endl;
		}
	cout<<"Done!"<<endl;

	free(h_OutputIMGR);
	free(h_OutputIMGT);
	free(h_OutputIMGRx);
	free(h_OutputIMGRy);
	free(h_OutputIMGTx);
	free(h_OutputIMGTy);
	free(h_OutputIMGTxy);


	return 0;
}


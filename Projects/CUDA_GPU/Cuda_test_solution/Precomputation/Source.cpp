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

	h_InputIMGR = (double*)malloc(256*256*sizeof(double));
	h_InputIMGT = (double*)malloc(256*256*sizeof(double));


	for(int i=0; i<256*256; i++){
		h_InputIMGR[i] = double(i);
		h_InputIMGT[i] = double(i*i);
		}

	for(int i=0; i<256; i++){
		for(int j=0; j<256; j++){
			cout<<h_InputIMGR[i*256+j]<<","<<h_InputIMGT[i*256+j]<<endl;
		}
	}

	h_OutputIMGR = (double*)malloc(254*254*sizeof(double));
	h_OutputIMGT = (double*)malloc(254*254*sizeof(double));
	h_OutputIMGRx = (double*)malloc(254*254*sizeof(double));
	h_OutputIMGRy = (double*)malloc(254*254*sizeof(double));
	h_OutputIMGTx = (double*)malloc(254*254*sizeof(double));
	h_OutputIMGTy = (double*)malloc(254*254*sizeof(double));
	h_OutputIMGTxy = (double*)malloc(254*254*sizeof(double));

	launch_kernel(h_InputIMGR,h_InputIMGT,h_OutputIMGR,h_OutputIMGT,
				  h_OutputIMGRx, h_OutputIMGRy,h_OutputIMGTx,h_OutputIMGTy,h_OutputIMGTxy,254,254);


	for(int i=0; i<254; i++)
		for(int j=0; j<254; j++){
			cout<<h_OutputIMGR[i*254+j]<<","<<h_OutputIMGT[i*254+j]
			<<","<<h_OutputIMGRx[i*254+j]<<","<<h_OutputIMGRy[i*254+j]
			<<","<<h_OutputIMGTx[i*254+j]<<","<<h_OutputIMGTy[i*254+j]<<","<<h_OutputIMGTxy[i*254+j]<<endl;
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


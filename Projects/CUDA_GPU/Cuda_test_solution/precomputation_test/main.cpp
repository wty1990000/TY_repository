
#include "precomputation.cuh"
#include "fft_cc.cuh"

#include <stdio.h>
#include <iostream>

using namespace std;

int main()
{
	double *h_InputIMGR, *h_InputIMGT;
	double *h_OutputIMGR,*h_OutputIMGT; 
	double *h_OutputIMGRx, *h_OutputIMGRy;
	double *h_OutputIMGTx, *h_OutputIMGTy, *h_OutputIMGTxy, *h_OutputdTBicubic;

	h_InputIMGR = (double*)malloc(8*8*sizeof(double));
	h_InputIMGT = (double*)malloc(8*8*sizeof(double));


	for(int i=0; i<8*8; i++){
		h_InputIMGR[i] = double(i);
		h_InputIMGT[i] = double(i*i);
	}

	for(int i=0; i<8; i++){
		for(int j=0; j<8; j++){
			cout<<h_InputIMGR[i*8+j]<<","<<h_InputIMGT[i*8+j]<<endl;
		}
	}

	cout<<endl;

	h_OutputIMGR = (double*)malloc(6*6*sizeof(double));
	h_OutputIMGT = (double*)malloc(6*6*sizeof(double));
	h_OutputIMGRx = (double*)malloc(6*6*sizeof(double));
	h_OutputIMGRy = (double*)malloc(6*6*sizeof(double));
	h_OutputIMGTx = (double*)malloc(6*6*sizeof(double));
	h_OutputIMGTy = (double*)malloc(6*6*sizeof(double));
	h_OutputIMGTxy = (double*)malloc(6*6*sizeof(double));
	h_OutputdTBicubic = (double*)malloc((6-1)*(6-1)*4*4*sizeof(double));

	launch_kernel(h_InputIMGR,h_InputIMGT,h_OutputIMGR,h_OutputIMGT,
				  h_OutputIMGRx, h_OutputIMGRy,h_OutputIMGTx,h_OutputIMGTy,h_OutputIMGTxy,h_OutputdTBicubic,6,6);

	for(int i=0; i<6; i++){
		for(int j=0; j<6; j++){
			cout<<h_OutputIMGR[i*6+j]<<","<<h_OutputIMGT[i*6+j]
				<<","<<h_OutputIMGRx[i*6+j]<<","<<h_OutputIMGRy[i*6+j]
				<<","<<h_OutputIMGTx[i*6+j]<<","<<h_OutputIMGTy[i*6+j]<<","<<h_OutputIMGTxy[i*6+j]<<endl;
		}
	}

	cout<<endl;

	for(int i=0; i<(6-1); i++){
		for(int j=0; j<(6-1); j++){
			for(int k=0; k<4; k++){
				for(int l=0; l<4; l++){
					cout<<h_OutputdTBicubic[((i*(6-1)+j)*4+k)*4+l]<<",\t";
				}
				cout<<endl;
			}
			cout<<endl;
		}
	}

	cout<<endl;
	cout<<"Done!"<<endl;

	free(h_OutputIMGR);
	free(h_OutputIMGT);
	free(h_OutputIMGRx);
	free(h_OutputIMGRy);
	free(h_OutputIMGTx);
	free(h_OutputIMGTy);
	free(h_OutputIMGTxy);
	free(h_OutputdTBicubic);


	return 0;
}


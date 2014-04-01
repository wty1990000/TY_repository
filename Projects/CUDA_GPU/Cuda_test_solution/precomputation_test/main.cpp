
#include "precomputation.cuh"
#include "Random.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int main()
{
	Random r;
	double *h_InputIMGR, *h_InputIMGT;
	double *h_OutputIMGR,*h_OutputIMGT; 
	double *h_OutputIMGRx, *h_OutputIMGRy;
	double *h_OutputIMGTx, *h_OutputIMGTy, *h_OutputIMGTxy, *h_OutputdTBicubic;

	h_InputIMGR = (double*)malloc(4*4*sizeof(double));
	h_InputIMGT = (double*)malloc(4*4*sizeof(double));


	for(int i=0; i<4*4; i++){
		h_InputIMGR[i] = double(r.random_integer(0,255));
		h_InputIMGT[i] = double(r.random_integer(0,255));
	}

	//for(int i=0; i<4; i++){
	//	for(int j=0; j<4; j++){
	//		cout<<h_InputIMGR[i*4+j]<<","<<h_InputIMGT[i*4+j]<<endl;
	//	}
	//}

	cout<<endl;

	h_OutputIMGR = (double*)malloc(2*2*sizeof(double));
	h_OutputIMGT = (double*)malloc(2*2*sizeof(double));
	h_OutputIMGRx = (double*)malloc(2*2*sizeof(double));
	h_OutputIMGRy = (double*)malloc(2*2*sizeof(double));
	h_OutputIMGTx = (double*)malloc(2*2*sizeof(double));
	h_OutputIMGTy = (double*)malloc(2*2*sizeof(double));
	h_OutputIMGTxy = (double*)malloc(2*2*sizeof(double));
	h_OutputdTBicubic = (double*)malloc((2)*(2)*4*4*sizeof(double));

	launch_kernel(h_InputIMGR,h_InputIMGT,h_OutputIMGR,h_OutputIMGT,
				  h_OutputIMGRx, h_OutputIMGRy,h_OutputIMGTx,h_OutputIMGTy,h_OutputIMGTxy,h_OutputdTBicubic,2,2);

	//for(int i=0; i<2; i++){
	//	for(int j=0; j<2; j++){
	//		cout<<h_OutputIMGR[i*2+j]<<","<<h_OutputIMGT[i*2+j]
	//			<<","<<h_OutputIMGRx[i*2+j]<<","<<h_OutputIMGRy[i*2+j]
	//			<<","<<h_OutputIMGTx[i*2+j]<<","<<h_OutputIMGTy[i*2+j]<<","<<h_OutputIMGTxy[i*2+j]<<endl;
	//	}
	//}

	cout<<endl;

	for(int i=0; i<2; i++){
		for(int j=0; j<2; j++){
			for(int k=0; k<4; k++){
				for(int l=0; l<4; l++){
					cout<<h_OutputdTBicubic[((i*2+j)*4+k)*4+l]<<",\t";
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


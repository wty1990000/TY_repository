
#include "combination.cuh"
#include "Random.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int main()
{
	Random r;
	double *h_InputIMGR, *h_InputIMGT, *h_OutputFFTcc;

	h_InputIMGR = (double*)malloc(256*256*sizeof(double));
	h_InputIMGT = (double*)malloc(256*256*sizeof(double));
	h_OutputFFTcc = (double*)malloc(21*21*32*32*sizeof(double));
	
	for(int i=0; i<256*256; i++){
		h_InputIMGR[i] = double(1);
		h_InputIMGT[i] = double(1);
	}

	initialize_CUDA();

	//Invoke combined function
	combined_functions(h_InputIMGR,h_InputIMGT, h_OutputFFTcc);
	

	cout<<endl;


	for(int k=0; k<32; k++){
				for(int l=0; l<32; l++){
					cout<<h_OutputFFTcc[(((2*21+1)*32)+k)*32+l]/double(32*32)<<",\t";
				}
				cout<<endl;
	}
	cout<<endl;
	cout<<"Done!"<<endl;

	free(h_InputIMGR);
	free(h_InputIMGT);
	free(h_OutputFFTcc);
	return 0;
}


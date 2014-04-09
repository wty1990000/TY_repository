#include <iostream>
#include "mul_fft.cuh"

using namespace std;

int main()
{
	float tt=0.0;;
	double* h_OutputC;

	h_OutputC = (double*)malloc(441*32*32*sizeof(double));
	
	init();

	FFT2D_batchedOut(h_OutputC,tt);
	
	//for(int ii=0; ii<2; ii++){
	//	for(int jj=0; jj<2; jj++){
	//		for(int i=0; i<32; i++){
	//			for(int j=0; j<32; j++){
	//				cout<<h_OutputC[((ii*2+jj)*32+i)*32+j]/double(32*32)<<"\t";
	//			}
	//			cout<<endl;
	//		}
	//		cout<<endl;
	//	}
	//}

	cout<<"One FFT time:"<<tt<<endl;
	free(h_OutputC);

	return 0;
}
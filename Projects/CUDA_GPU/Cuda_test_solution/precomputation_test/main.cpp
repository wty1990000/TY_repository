#include <iostream>
#include "kernel.cuh"
#include <stdio.h>

using namespace std;

int main()
{
	double h_input[16]; 
	double h_output[16];

	for(int i=0; i<16; i++){
		h_input[i] = double(i*i);
	}

	for(int j=0; j<4*4; j++){
		if((j+1)%4 !=0){
			printf("%3.1f\t",h_input[j]);
		}
		else{
			printf("%3.1f\n",h_input[j]);
		}
	}

	launchkernel(h_input,h_output,2);

	printf("The result is:\n");
	for(int j=0; j<2*2; j++){
		if((j+1)%2 !=0){
			printf("%3.1f\t",h_output[j]);
		}
		else{
			printf("%3.1f\n",h_output[j]);
		}
	}
	return 0;
}


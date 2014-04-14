#include <stdio.h>
#include <stdlib.h>

#include "comb.cuh"

int main()
{
	float *output;

	int sizeo, sizei;

	sizeo = 21; sizei = 32;

	output = (float*)malloc(sizeo*sizeo*sizeof(float));

	combine(output,sizeo,sizei);

	printf("The results are::\n");
	for(int i=0; i< sizeo; i++)
		for(int j=0; j<sizeo; j++)
		{
			printf("%f,", output[i*sizeo+j]);
		}

	free(output);
}
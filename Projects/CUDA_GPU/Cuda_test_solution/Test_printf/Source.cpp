#include "CudaComputing.cuh"
#include <stdio.h>

int main()
{
	int c[5];

	if( vvmain(c)==0 ){
			printf("%d,%d,%d,%d,%d",c[0],c[1],c[2],c[3],c[4]);
		}

	return 0;
}
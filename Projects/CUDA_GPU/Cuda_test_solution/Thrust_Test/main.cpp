#include "kernel.cuh"
#include "helper_cuda.h"
#include "helper_functions.h"

#include <iostream>


int main()
{
	float hWatch, dWatch;
	
	init();
	hostVersion(hWatch);
	deViceVersion(dWatch);

	std::cout<<"hWatch: "<<hWatch<<"\t"<<"dWatch: "<<dWatch<<std::endl;

	return 0;

}


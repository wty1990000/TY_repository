/*******This is for the test of array-of-struct-of-fixed-array structure******/
//

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>

#define REGIONS 20
#define YEARS 5

__inline __host__ void gpuAssert(cudaError_t code, char *file, int line, 
                 bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code),
          file, line);
      if (abort) exit(code);
   }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

struct AnimalPopulationForYear_s
{
   bool isYearEven;
   int * rabbits;
   int * hyenas;
};

AnimalPopulationForYear_s * dev_pop;

__global__ void RunSim(AnimalPopulationForYear_s dev_pop[],
               int year)
{
   int idx = blockIdx.x*blockDim.x+threadIdx.x;
   int rabbits, hyenas;
   int arrEl = year-1;

   rabbits = (idx+1) * year * year; 
   hyenas = rabbits / 10;

   if ( rabbits > 100000 ) rabbits = 100000;   
   if ( hyenas < 2 ) hyenas = 2;

   if ( idx < REGIONS ) dev_pop[arrEl].rabbits[idx] = rabbits;
   if ( idx < REGIONS ) dev_pop[arrEl].hyenas[idx] = hyenas;

   if (threadIdx.x == 0 && blockIdx.x == 0)
      dev_pop[arrEl].isYearEven = (year & 0x01 == 0x0);
}

int main()
{
   //Various reused sizes...
   const size_t fullArrSz = size_t(YEARS) * size_t(REGIONS) * sizeof(int);
   const size_t structArrSz = size_t(YEARS) * sizeof(AnimalPopulationForYear_s);

   //Vars to hold struct and merged subarray memory inside it.
   AnimalPopulationForYear_s * h_pop;
   int * dev_hyenas, * dev_rabbits, * h_hyenas, * h_rabbits, arrEl;

   //Alloc. memory.
   h_pop = (AnimalPopulationForYear_s *) malloc(structArrSz);
   h_rabbits = (int *) malloc(fullArrSz);
   h_hyenas = (int *) malloc(fullArrSz);
   gpuErrchk(cudaMalloc((void **) &dev_pop,structArrSz));
   gpuErrchk(cudaMalloc((void **) &dev_rabbits,fullArrSz));
   gpuErrchk(cudaMalloc((void **) &dev_hyenas,fullArrSz));

   //Offset ptrs.
   for (int i = 0; i < YEARS; i++)
   {
      h_pop[i].rabbits = dev_rabbits+i*REGIONS;
      h_pop[i].hyenas = dev_hyenas+i*REGIONS;
   }

   //Copy host struct with dev. pointers to device.
   gpuErrchk
      (cudaMemcpy(dev_pop,h_pop, structArrSz, cudaMemcpyHostToDevice));

   //Call kernel
   for(int i=1; i < YEARS+1; i++) RunSim<<<REGIONS/128+1,128>>>(dev_pop,i);

   //Make sure nothing went wrong.
   gpuErrchk(cudaPeekAtLastError());
   gpuErrchk(cudaDeviceSynchronize());

   gpuErrchk(cudaMemcpy(h_pop,dev_pop,structArrSz, cudaMemcpyDeviceToHost));
   gpuErrchk
      (cudaMemcpy(h_rabbits, dev_rabbits,fullArrSz, cudaMemcpyDeviceToHost));
   gpuErrchk(cudaMemcpy(h_hyenas,dev_hyenas,fullArrSz, cudaMemcpyDeviceToHost));

   for(int i=0; i < YEARS; i++)
   {
      h_pop[i].rabbits = h_rabbits + i*REGIONS;
      h_pop[i].hyenas = h_hyenas + i*REGIONS;
   }

   for(int i=1; i < YEARS+1; i++)
   {
      arrEl = i-1;
      printf("\nYear %i\n=============\n\n", i);      
      printf("Rabbits\n-------------\n");
      for (int j=0; j < REGIONS; j++)
     printf("Region: %i  Pop: %i\n", j, h_pop[arrEl].rabbits[j]);;      
      printf("Hyenas\n-------------\n");
      for (int j=0; j < REGIONS; j++)
     printf("Region: %i  Pop: %i\n", j, h_pop[arrEl].hyenas[j]);
   }

   //Free on device and host
   cudaFree(dev_pop);
   cudaFree(dev_rabbits);
   cudaFree(dev_hyenas);

   free(h_pop);
   free(h_rabbits);
   free(h_hyenas);

   return 0;
}
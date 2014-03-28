#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	__shared__ float ds_A[16][16];
	__shared__ float ds_B[16][16];
	
	int ty = threadIdx.y;  int tx = threadIdx.x;
	
	int Row = ty + blockDim.y*blockIdx.y;
	int Col = tx + blockDim.x*blockIdx.x;
	float CValue = 0.0;
	
	for(int i=0; i < (numAColumns-1)/16+1; ++i){
		if((Row<numARows) && (i*16+tx < numAColumns))
		{ds_A[ty][tx] = A[numAColumns*Row + i*16+tx];}
		else{
			ds_A[ty][tx] = 0.0;}
		if((i*16+ty < numBRows) && (Col<numBColumns)){
			ds_B[ty][tx] = B[(i*16+ty)*numBColumns+Col];}
		else{
			ds_B[ty][tx] = 0.0;}
		__syncthreads();
		for(int j=0; j<16; ++j){
			CValue += ds_A[ty][j] * ds_B[j][tx];}
	__syncthreads();	
	}
	if((Row<numCRows) && (Col<numCColumns)){
		C[Row*numCColumns+Col] = CValue;}
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
	hostC = (float *) malloc(numCRows*numCColumns*sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
	wbCheck(cudaMalloc((void**)&deviceA, numARows*numAColumns*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceB, numBRows*numBColumns*sizeof(float)));
	wbCheck(cudaMalloc((void**)&deviceC, numCRows*numCColumns*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
	wbCheck(cudaMemcpy(deviceA,hostA,numARows*numAColumns*sizeof(float),cudaMemcpyHostToDevice));
	wbCheck(cudaMemcpy(deviceB,hostB,numBRows*numBColumns*sizeof(float),cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((numBColumns-1)/16+1, (numARows-1)/16+1,1);
	dim3 DimBlock(16,16,1);
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	matrixMultiplyShared<<<DimGrid,DimBlock>>>(deviceA, deviceB, deviceC,
                   numARows, numAColumns,
                   numBRows, numBColumns,
                   numCRows, numCColumns);
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	wbCheck(cudaMemcpy(hostC,deviceC,numCRows*numCColumns*sizeof(float),cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
	wbCheck(cudaFree(deviceA));
	wbCheck(cudaFree(deviceB));
	wbCheck(cudaFree(deviceC));
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
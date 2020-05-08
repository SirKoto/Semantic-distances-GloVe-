#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <stdio.h>
#include "GlobalHeader.h"
#include <assert.h>


// Reserve pinned memory
extern "C"
void reservePinnedMemory(embed_t* &ptr, size_t bytes) {

    cudaError_t status = cudaMallocHost((void**)&ptr, bytes);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(status));

		assert(status == cudaSuccess);
	}
}

extern "C"
void reservePinnedMemoryV(embedV_t * &ptr, size_t bytes) {
    
	cudaError_t status = cudaMallocHost((void**)&ptr, bytes);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(status));

		assert(status == cudaSuccess);
	}
}


// Free all data from pinned
extern "C"
void freePinnedMemory(void* ptr) {
	cudaFreeHost(ptr);
}

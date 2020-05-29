#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <stdio.h>
#include "GlobalHeader.h"
#include <assert.h>


// Reserve pinned memory
extern "C"
void reservePinnedMemory(embed_t* &ptr, size_t bytes) {

	#ifdef NOT_PINNED_MEMORY
	ptr = (embed_t *) malloc(bytes);
	#else
	cudaError_t status = cudaMallocHost((void**)&ptr, bytes);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(status));

		assert(status == cudaSuccess);
	}
	#endif
}

extern "C"
void reservePinnedMemoryV(embedV_t * &ptr, size_t bytes) {
	#ifdef NOT_PINNED_MEMORY
	ptr = (embedV_t *) malloc(bytes);
	#else
	cudaError_t status = cudaMallocHost((void**)&ptr, bytes);
	if (status != cudaSuccess)
	{
		fprintf(stderr, "CUDA Runtime Error: %s\n",
			cudaGetErrorString(status));

		assert(status == cudaSuccess);
	}
	#endif
}


// Free all data from pinned
extern "C"
void freePinnedMemory(void* ptr) {
	#ifdef NOT_PINNED_MEMORY
	free(ptr);
	#else
	cudaFreeHost(ptr);
	#endif
}

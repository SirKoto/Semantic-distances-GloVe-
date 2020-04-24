
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <stdio.h>
#include "GlobalHeader.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//rows determined as the amount of rows in a block
// A is query vector, B is the model ( rows ), C is output matrix
// Rows should be 300 for proper usage of this access method
__global__ void DotProduct
(int rows, embed_t *A, embed_t *B, embed_t *C, embed_t normA, embed_t *normsB) {
  __shared__ embed_t fastA[300];
  
  unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id<300) {
      fastA[id]= A[id]; // only one embeding is on A
  }
  __syncthreads();
  if (id<rows) {
    
  unsigned long identifier=id*numEmbeds;
  embed_t acum=0;
  for(unsigned long i=0;i<300;++i) {
      acum+=fastA[i] * B[identifier + i];
  }
  C[id]=acum/(normA*normsB[id]);
  }
}


__global__ void FirstMerge
(int N, float *sims, int length) {

  
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int start=id*N;
    int end=start+N;
    if (start<length) { 
    
    // Insertion sort, as N SHOULD be small
    
	for(int i=start+1; i<end; i++)
	{
    if (i<length){
		embed_t temp=sims[i];
		int j=i-1;
		while((temp>sims[j]) && (j>=start))
		{
			sims[j+1]=sims[j];
			j=j-1;
		}
		sims[j+1]=temp;
	}
    }
}
}

__global__ void BotchedMergeSort
(int N, float *sims, unsigned long stride) {

  
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    id=id*N;
    int currentPos=0, posB=0;
    if (id<stride) { 
        embed_t elem=sims[id+stride];
        while(currentPos<N) {
            if (id+currentPos<stride)
            if (sims[id+currentPos]<elem) {
                sims[id+currentPos]=elem;
                ++posB;
                elem=sims[id+posB+stride];
            }
            currentPos++;
        }
   
}
}


extern "C"
void reservePinnedMemory(embed_t * &ptr, int32_t bytes);


extern "C"
int runCuda(embed_t* norms, embedV_t* model, int32_t numRows, int32_t queryTermPos,int32_t N)
{

	embedV_t queryTerm;
	embed_t* A_d, *B_d;
	embed_t* C_d, *norms_d;
	int nBlocks=(numRows/512)+1;
	int nThreads=512;
	float elapsedTime;

	embed_t* similarities;
	reservePinnedMemory(similarities, sizeof(embed_t) * numRows);

	cudaEvent_t start, stop;

	queryTerm = model[queryTermPos]; // request the model to look for
    
	embed_t normA = norms[queryTermPos];



	int numBytesQuery = sizeof(embedV_t);
	int numBytesModel = sizeof(embedV_t) * numRows;
	int numBytesSimsAndNorms = sizeof(embed_t) * numRows;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
	cudaMalloc((embed_t**)&A_d, numBytesQuery); 
	cudaMalloc((embed_t**)&B_d, numBytesModel); 
	cudaMalloc((embed_t**)&C_d, numBytesSimsAndNorms); 
	cudaMalloc((embed_t**)&norms_d, numBytesSimsAndNorms); 

	cudaMemcpyAsync(A_d, queryTerm.data, numBytesQuery, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_d, model, numBytesModel, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(norms_d, norms, numBytesSimsAndNorms, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	DotProduct<<<nBlocks, nThreads >>>(numRows, A_d, B_d, C_d, normA, norms_d);


    FirstMerge<<<nBlocks, nThreads >>>(N,C_d,numRows);
  
    unsigned long toReduce=((numRows/N)/2);
    while(toReduce>0) {
        nBlocks=((toReduce*N)/nThreads)+1;
        printf("%lu\n",toReduce*N);
        BotchedMergeSort<<<nBlocks, nThreads >>>(N, C_d, toReduce*N);
        toReduce=toReduce/2;
    }
    
    
	cudaMemcpyAsync(similarities, C_d, numBytesSimsAndNorms, cudaMemcpyDeviceToHost); 
  
  cudaError_t error = cudaGetLastError();
if (error != cudaSuccess) {
  fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
}
	cudaFree(B_d);
	cudaFree(norms_d);
	cudaFree(A_d);
	//cudaFreeHost(similarities);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
    printf("\nSimilarity vector\n");
    
    for(int i=0;i<numRows;++i) {
    printf("%f ",similarities[i]);
    }

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\nSimilarities\n");
	printf("Vector Size: %d\n", numRows);
	printf("nThreads: %d\n", nThreads);
	printf("nBlocks: %d\n", nBlocks);
	printf("Tiempo Total %4.6f ms\n", elapsedTime);
	printf("Ancho de Banda %4.3f GB/s\n", (numRows *300* sizeof(float)) / (1000000 * elapsedTime));
  

	return 0;

}




#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <stdio.h>
#include <vector>
#include "GlobalHeader.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//rows determined as the amount of rows in a block
// A is query vector, B is the model ( rows ), C is output matrix
// Rows should be 300 for proper usage of this access method
__global__ void DotProduct
(int rows, embed_t *A, embedV_t *B, embed_t *C,unsigned int *pos, embed_t normA, embed_t *normsB) {
  __shared__ embed_t fastA[numEmbeds];
  
  unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id<numEmbeds) {
      fastA[id]= A[id]; // only one embeding is on A
  }
  __syncthreads();
  if (id<rows) {
    
  //unsigned long identifier=id*numEmbeds;
  embed_t acum=0;
  for(unsigned long i=0;i<numEmbeds;++i) {
      acum+=fastA[i] * B[id].data[i];
  }
  C[id]=acum/(normA*normsB[id]);
  pos[id]=id;
  }
}


__global__ void FirstMerge
(int N, embed_t *sims,unsigned int* pos, int length, int pad) {

  
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int start=id*N;
    int end=start+N;
    if (start<length) { 
    
    // Insertion sort, as N SHOULD be small
    
	for(int i=start+1; i<end; i++)
	{
    if (i<length){
		embed_t temp=sims[i];
        unsigned int position=pos[i];
		int j=i-1;
		while((temp>sims[j]) && (j>=start))
		{
			sims[j+1]=sims[j];
            pos[j+1]=pos[j];
			j=j-1;
		}
		sims[(j+1)]=temp;
        pos[(j+1)]=position;
	}
    else if (i<pad) {
        for (int i=0;i<N;++i) {
    	sims[id+i]=0;
        pos[id+i]=0;
        }
    }
    }
}
}

__global__ void BotchedMergeSort
(int N, embed_t *sims,unsigned int* pos,unsigned long stride) {

  
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    id=id*N;
    int currentPos=0, posB=0;
    if (id<stride) { 
        embed_t elem=sims[(id+stride)];
        unsigned int posAux=pos[(id+stride)];
        sims[(id+stride)]=0;
        pos[(id+stride)]=0;
        while(currentPos<N) {
            if (sims[(id+currentPos)]<elem) {
                sims[(id+currentPos)]=elem;
                pos[(id+currentPos)]=posAux;
                ++posB;
                elem=sims[(id+posB+stride)];
                posAux=pos[(id+posB+stride)];
                sims[(id+posB+stride)]=0;
                pos[(id+posB+stride)]=0;

            }
            ++currentPos;
        }
   
}
}


extern "C"
void reservePinnedMemory(embed_t * &ptr, int32_t bytes);


extern "C"
std::vector<unsigned int> runCuda(embed_t* norms, embedV_t* model, int32_t numRows, int32_t queryTermPos,int32_t N, int &returnCode)
{

	embedV_t queryTerm;
	embed_t* A_d;
	embed_t* norms_d;
    embed_t* C_d;
    embedV_t* B_d;
    unsigned int *positions,*pos_d;
	int nBlocks=(numRows/512)+1;
	int nThreads=512;
	float elapsedTime;
    
    unsigned int numRowsMod=numRows;
    if (!numRows%N) {
        numRowsMod=(N-numRows%N)+numRows;
        numRowsMod+=numRowsMod%2*N;
      }


	embed_t* similarities;
	reservePinnedMemory(similarities, sizeof(embed_t) * numRowsMod);
    cudaMallocHost((void**)&positions, sizeof(embed_t) * numRowsMod);


	cudaEvent_t start, stop;

	queryTerm = model[queryTermPos]; // request the model to look for
    
	embed_t normA = norms[queryTermPos];


	unsigned int numBytesQuery = sizeof(embedV_t);
	unsigned int numBytesModel = sizeof(embedV_t) * numRows;
	unsigned int numBytesNorms = sizeof(embed_t) * numRows;
	unsigned int numBytesSims = sizeof(unsigned int) * numRowsMod;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    
	cudaMalloc((embed_t**)&A_d, numBytesQuery); 
	cudaMalloc((embed_t**)&B_d, numBytesModel); 
	cudaMalloc((embed_t**)&C_d, numBytesSims); 
	cudaMalloc((unsigned int**)&pos_d, numBytesSims); 
	cudaMalloc((embed_t**)&norms_d, numBytesNorms); 

	cudaMemcpyAsync(A_d, queryTerm.data, numBytesQuery, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_d, model, numBytesModel, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(norms_d, norms, numBytesNorms, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	DotProduct<<<nBlocks, nThreads >>>(numRows, A_d, B_d, C_d, pos_d,normA, norms_d);

    
    FirstMerge<<<nBlocks, nThreads >>>(N,C_d,pos_d,numRows,numRowsMod);
  
    unsigned long toReduce=((numRowsMod/N)/2);
    while(toReduce>0) {
        nBlocks=((toReduce*N)/nThreads)+1;
        printf("%lu\n",toReduce*N);
        BotchedMergeSort<<<nBlocks, nThreads >>>(N, C_d, pos_d,toReduce*N);
        if (toReduce>1) toReduce=toReduce/2+toReduce%2;
        else toReduce=toReduce/2;
    }
    
    
    
	cudaMemcpyAsync(similarities, C_d, numBytesSims, cudaMemcpyDeviceToHost); 
  	cudaMemcpyAsync(positions, pos_d, numBytesSims, cudaMemcpyDeviceToHost); 

  cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
      returnCode=1;
    }
	cudaFree(B_d);
	cudaFree(norms_d);
	cudaFree(A_d);
	//cudaFreeHost(similarities);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
    printf("\nSimilarity vector\n");
    
    for(int i=0;i<numRowsMod;++i) {
        if (i%N==0 && i!=0)printf("\n");
    printf("[ %f , %i ]",similarities[i],positions[i]);

    }
    
    

	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("\nSimilarities\n");
	printf("Vector Size: %d\n", numRows);
	printf("nThreads: %d\n", nThreads);
	printf("nBlocks: %d\n", nBlocks);
	printf("Tiempo Total %4.6f ms\n", elapsedTime);
	printf("Ancho de Banda %4.3f GB/s\n", (numRows *300* sizeof(float)) / (1000000 * elapsedTime));
  
    std::vector<unsigned int> results;
    for (int i=0;i<N;++i) {
    results.push_back(positions[i]);
    }



	return results;

}



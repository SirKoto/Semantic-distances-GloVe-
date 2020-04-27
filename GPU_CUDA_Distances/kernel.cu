
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
  if (threadIdx.x<numEmbeds) {
      fastA[threadIdx.x]= A[threadIdx.x]; // only one embeding is on A
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
(int N, embed_t *sims,unsigned int* pos,embed_t *simsAux,unsigned int* posAux,unsigned long stride) {

  
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    id=id*N;
    int posA=0,posB=0;
    if (id<stride) { 
        embed_t elemA=sims[(id+stride)];
        unsigned int posAuxA=pos[(id+stride)];
        embed_t elemB=sims[id];
        unsigned int posAuxB=pos[id];

        sims[(id+stride)]=0;
        for(int i=0;i<N;++i) {
            if (elemA>elemB) {
                ++posA;
                simsAux[id+i]=elemA;
                posAux[id+i]=posAuxA;
                
                elemA=sims[(id+posA+stride)];
                posAuxA=pos[(id+posA+stride)];
                sims[(id+posA+stride)]=0;
            }
            else {
                ++posB;
                simsAux[id+i]=elemB;
                posAux[id+i]=posAuxB;
                
                elemB=sims[id+posB];
                posAuxB=pos[id+posB];

            }
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
    embed_t *C_d,*CAux_d;
    embedV_t* B_d;
    unsigned int *positions,*pos_d,*posAux_d;
	unsigned int nBlocks=(numRows/512)+1;
	int nThreads=512;
	float elapsedTime;
    
    unsigned int numRowsMod=numRows;
    if (numRows%N!=0) numRowsMod=(N-numRows%N)+numRows;
    numRowsMod+=numRowsMod%2*N;

    printf("%u\n",numRows);
	embed_t* similarities;
	cudaMallocHost((void**)&similarities, sizeof(embed_t) * numRowsMod);
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


	cudaMalloc((embed_t**)&CAux_d, numBytesSims); 
	cudaMalloc((unsigned int**)&posAux_d, numBytesSims); 


	cudaMemcpyAsync(A_d, queryTerm.data, numBytesQuery, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(B_d, model, numBytesModel, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(norms_d, norms, numBytesNorms, cudaMemcpyHostToDevice);

	cudaEventRecord(start, 0);

	DotProduct<<<nBlocks, nThreads >>>(numRows, A_d, B_d, C_d, pos_d,normA, norms_d);

    
    FirstMerge<<<nBlocks, nThreads >>>(N,C_d,pos_d,numRows,numRowsMod);
  
    unsigned long toReduce=((numRowsMod/N)/2);
    bool alternate=true;
    while(toReduce>0) {
        nBlocks=((toReduce*N)/nThreads)+1;
        //printf("%lu\n",toReduce*N);
        if (alternate) BotchedMergeSort<<<nBlocks, nThreads >>>(N, C_d, pos_d,CAux_d,posAux_d,toReduce*N);
        else BotchedMergeSort<<<nBlocks, nThreads >>>(N, CAux_d,posAux_d,C_d, pos_d,toReduce*N);
        if (toReduce>1){
            toReduce+=toReduce%2;
            }
        toReduce=toReduce/2;
        alternate=!alternate;
    }
    
    if (alternate){
	cudaMemcpyAsync(similarities, C_d, sizeof(embed_t)*N, cudaMemcpyDeviceToHost); 
  	cudaMemcpyAsync(positions, pos_d, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost); 
    }
    else {
    cudaMemcpyAsync(similarities, CAux_d, sizeof(embed_t)*N, cudaMemcpyDeviceToHost); 
  	cudaMemcpyAsync(positions, posAux_d, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost); 

    }

  cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
      returnCode=1;
    }
	cudaFree(B_d);
	cudaFree(norms_d);
	cudaFree(A_d);
	cudaFree(CAux_d);
	cudaFree(C_d);
	cudaFree(pos_d);
	cudaFree(posAux_d);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
    printf("\nSimilarity vector\n");
    
   for(int i=0;i<N;++i) {
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

	cudaFreeHost(similarities);
	cudaFreeHost(positions);


	return results;

}



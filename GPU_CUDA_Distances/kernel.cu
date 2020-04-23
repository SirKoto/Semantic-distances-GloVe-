
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "GlobalHeader.h"
#include <vector>

#include <stdio.h>
#define CUDA_INCLUDE
#include "GlobalHeader.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

//rows determined as the amount of rows in a block
// A is query vector, B is the model ( rows ), C is output matrix
// Rows should be 300 for proper usage of this access method
__global__ void DotProduct
(int rows, embed_t *A, embed_t *B, embed_t *C, embed_t normA, embed_t *normsB) {
  __shared__ embed_t fastA[300];
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id<300) {
      fastA[id]=A[id];
  }
  __syncthreads();
  embed_t acum=0;
  for(int i=0;i<300;++i) {
      acum+=fastA[i]*B[id*300+i];
  }
  C[id]=acum/(normA*normsB[id]);
}


__global__ void FirstMerge
(int N, float *sims, int length) {

  
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int start=id*N;
    int end=start+N;
    if (!(start>length)) { 
    
    // Insertion sort, as N SHOULD be small
   int key, j;
   for(int i = start+1; i<end; i++) {
      key = sims[i];
      j = i;
      while(j > 0 && sims[j-1]<key) {
         sims[j] = sims[j-1];
         j--;
      }
      sims[j] = key;  
   }
}
}

int binary_search(std::vector<std::string> words, int length, std::string to_be_found){
    
    int p = 0;
    int r = length - 1;
    int q = (r + p) / 2;
    int counter = 0;

    while (p <= r)
    {
        counter++;
        if (words[q] == to_be_found)
            return q;
        else
        {
            if (words[q] < to_be_found) 
            {
                p = q + 1;
                q = (r + p) / 2;
            }
            else
            {
                r = q - 1;
                q = (r + p) / 2;    
            }
        }
    }
    return -1;
}



extern "C"
int runCuda(std::vector<std::string> words,std::vector<embed_t> norms,std::vector<embedV_t> model,int rows,std::string toFind)
{

  embed_t* queryTerm,similarities;
  embedV_t* A_d,B_d;
  embed_t* C_d,norms_d;
  int nBlocks=1;
  int nThreads=10;
  float elapsedTime;

  embed_t similarities[300];

  cudaEvent_t start,stop;

  int queryTermPos=binary_search(words,rows,toFind);
  for (int i=0;i<300;++i) {
      queryTerm[i]=model[queryTermPos*300+i];
  }
  
  embed_t normA=norms[queryTermPos];
  

  int numBytesQuery=sizeof(embed_t)*300;
  int numBytesModel=numBytesQuery*rows;
  int numBytesSimsAndNorms=sizeof(embed_t)*rows;

  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  cudaMalloc((embed_t**)&A_d, numBytesQuery); 
  cudaMalloc((embed_t**)&B_d, numBytesModel); 
  cudaMalloc((embed_t**)&C_d, numBytesSimsAndNorms); 
  cudaMalloc((embed_t**)&norms_d, numBytesSimsAndNorms); 

  cudaMemcpyAsync(A_d,queryTerm, numBytesQuery, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(B_d,model, numBytesModel, cudaMemcpyHostToDevice);
  cudaMemcpyAsync(norms_d,norms, numBytesSimsAndNorms, cudaMemcpyHostToDevice);

  cudaEventRecord(start, 0);
  
  DotProduct<<<nBlocks,nThreads >>>(rows,A_d,B_d,C_d,normA,norms_d);

  cudaMemcpyAsync(similarities, C_d, numBytesSimsAndNorms, cudaMemcpyDeviceToHost); 
  
  cudaFree(B_d);
  cudaFree(norms_d);
  cudaFree(A_d);

  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);


  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf("\nSimilarities\n");
  printf("Vector Size: %d\n", rows);
  printf("nThreads: %d\n", nThreads);
  printf("nBlocks: %d\n", nBlocks);
  printf("Tiempo Total %4.6f ms\n", elapsedTime);
  printf("Ancho de Banda %4.3f GB/s\n", (rows *300* sizeof(float)) / (1000000 * elapsedTime));
  
  return 0;

}



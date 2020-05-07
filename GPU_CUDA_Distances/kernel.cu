
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <stdio.h>
#include <vector>
#include "GlobalHeader.h"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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
(int64_t N, embed_t *sims, unsigned int* pos, int64_t length, int64_t pad) {
	int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
	int64_t start=id*N;
	int64_t end=start+N;
    if (start<length) { 
    
    // Insertion sort, as N SHOULD be small
    
		for(int64_t i=start+1; i<end; i++)
		{
			if (i<length){
                /*if (i >= pad || i < 0) {
                    printf("ERRORR1 %i\n", i);
                }*/
				embed_t temp=sims[i];
				int64_t position=pos[i];
                int64_t j=i-1;
                
				while((j>=start) && (temp>sims[j]) )
				{
					sims[j+1]=sims[j];
					pos[j+1]=pos[j];
                    j=j-1;
                    /*if (j >= pad || j < -1) {
						printf("ERRORR3 %i\n", j);
					}*/
				}
				sims[(j+1)]=temp;
				pos[(j+1)]=position;
			}
			else if (i<pad) {
				for (int64_t i=0;i<N;++i) {
                    /*if (id+i >= pad || id+i < -1) {
						printf("ERRORR4 %i\n", i);
					}*/
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
	unsigned int posA=0,posB=0;
    if (id<stride) { 
        embed_t elemA=sims[(id+stride)];
        unsigned int posAuxA=pos[(id+stride)];
        embed_t elemB=sims[id];
        unsigned int posAuxB=pos[id];

        sims[(id+stride)]=0;
        for(unsigned int i=0;i<N;++i) {
            if (posAuxA==posAuxB) {
                ++posA;
                elemA=sims[(id+posA+stride)];
                posAuxA=pos[(id+posA+stride)];
                sims[(id+posA+stride)]=0;
            }
            if (elemA>elemB && posA<N) {
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
void loadModel(embed_t * norms, embedV_t * model, uint32_t numRows, embedV_t* &B_d, embed_t* & norms_d)
{
	size_t numBytesModel = sizeof(embedV_t) * numRows;
	size_t numBytesNorms = sizeof(embed_t) * numRows;

	gpuErrchk(cudaMalloc((embed_t**)&B_d, numBytesModel));
	gpuErrchk(cudaMalloc((embed_t**)&norms_d, numBytesNorms));

	gpuErrchk(cudaMemcpy(B_d, model, numBytesModel, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(norms_d, norms, numBytesNorms, cudaMemcpyHostToDevice));
}

extern "C"
void freeAll(embedV_t * &B_d, embed_t * &norms_d)
{
	gpuErrchk(cudaFree(norms_d));
	gpuErrchk(cudaFree(B_d));
}


extern "C"
void runCuda(embed_t* norms, embedV_t* model, uint32_t numRows, uint32_t queryTermPos, uint32_t N, embedV_t * B_d, embed_t * norms_d, int &returnCode, std::vector<unsigned int> &res)
{
	if (!B_d || !norms_d) {
		fprintf(stderr, "Memory not initialized\n");
		returnCode = 1;
		res = {};
		return;
	}

	embedV_t queryTerm;
	embed_t* A_d;
    embed_t *C_d,*CAux_d;
    unsigned int *positions,*pos_d,*posAux_d;
	unsigned int nBlocks=(numRows/512)+1;
	int nThreads=512;
	float elapsedTime;
    
    unsigned int numRowsMod=numRows;
    if (numRows%N!=0) numRowsMod=(N-numRows%N)+numRows;
    numRowsMod+=numRowsMod%2*N;

    //printf("%u\n",numRows);
	embed_t* similarities;
	gpuErrchk(cudaMallocHost((void**)&similarities, sizeof(embed_t) * numRowsMod));
	gpuErrchk(cudaMallocHost((void**)&positions, sizeof(embed_t) * numRowsMod));


	cudaEvent_t start, stop;

	queryTerm = model[queryTermPos]; // request the model to look for
    
	embed_t normA = norms[queryTermPos];


	unsigned int numBytesQuery = sizeof(embedV_t);
	unsigned int numBytesSims = sizeof(unsigned int) * numRowsMod;

	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
    
	gpuErrchk(cudaMalloc((embed_t**)&A_d, numBytesQuery));
	gpuErrchk(cudaMalloc((embed_t**)&C_d, numBytesSims));
	gpuErrchk(cudaMalloc((unsigned int**)&pos_d, numBytesSims));


	gpuErrchk(cudaMalloc((embed_t**)&CAux_d, numBytesSims));
	gpuErrchk(cudaMalloc((unsigned int**)&posAux_d, numBytesSims));


	gpuErrchk(cudaMemcpyAsync(A_d, queryTerm.data, numBytesQuery, cudaMemcpyHostToDevice));

	gpuErrchk(cudaEventRecord(start, 0));

	DotProduct<<<nBlocks, nThreads >>>(numRows, A_d, B_d, C_d, pos_d,normA, norms_d);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());// Coment this on release
    
    FirstMerge<<<nBlocks, nThreads >>>(N,C_d,pos_d,numRows,numRowsMod);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());// Coment this on release

    unsigned long toReduce=((numRowsMod/N)/2);
    bool alternate=true;
    while(toReduce>0) {
        nBlocks=((toReduce*N)/nThreads)+1;
        //printf("%lu\n",toReduce*N);
		if (alternate) { 
			BotchedMergeSort <<<nBlocks, nThreads >>> (N, C_d, pos_d, CAux_d, posAux_d, toReduce * N); 
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize());// Coment this on release
		}
		else {
			BotchedMergeSort <<<nBlocks, nThreads >>> (N, CAux_d, posAux_d, C_d, pos_d, toReduce * N);
			gpuErrchk(cudaPeekAtLastError());
			gpuErrchk(cudaDeviceSynchronize()); // Coment this on release
		}
        if (toReduce>1){
            toReduce+=toReduce%2;
            }
        toReduce=toReduce/2;
        alternate=!alternate;
    }
    
    if (alternate) {
		gpuErrchk(cudaMemcpyAsync(similarities, C_d, sizeof(embed_t)*N, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpyAsync(positions, pos_d, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));
    }
    else {
		gpuErrchk(cudaMemcpyAsync(similarities, CAux_d, sizeof(embed_t)*N, cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpyAsync(positions, posAux_d, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));
    }

  cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
      returnCode=1;
    }

	gpuErrchk(cudaFree(A_d));
	gpuErrchk(cudaFree(CAux_d));
	gpuErrchk(cudaFree(C_d));
	gpuErrchk(cudaFree(pos_d));
	gpuErrchk(cudaFree(posAux_d));

	gpuErrchk(cudaEventRecord(stop, 0));
	gpuErrchk(cudaEventSynchronize(stop));
	
    //printf("\nSimilarity vector\n");
    
   /*for(int i=0;i<N;++i) {
    printf("[ %f , %i ]",similarities[i],positions[i]);

    }*/
    
    

	gpuErrchk(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("\nSimilarities\n");
	printf("Vector Size: %d\n", numRows);
	printf("nThreads: %d\n", nThreads);
	printf("nBlocks: %d\n", (numRows/nThreads)+1);
	printf("Tiempo Total %4.6f ms\n", elapsedTime);
	printf("Ancho de Banda %4.3f GB/s\n", (numRows *numEmbeds* sizeof(float)) / (1000000 * elapsedTime));
  
    std::vector<unsigned int> results;
    for (unsigned int i=0;i<N;++i) {
		results.push_back(positions[i]);
    }

	gpuErrchk(cudaFreeHost(similarities));
	gpuErrchk(cudaFreeHost(positions));

	res = results;
}




#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include <stdio.h>
#include <vector>
#include <assert.h>
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

__device__ __constant__
embedV_t* c_model;
__device__ __constant__
embed_t* c_norms;

//rows determined as the amount of rows in a block
// A is query vector, B is the model ( rows ), C is output matrix
// Rows should be 300 for proper usage of this access method
__global__ void DotProduct
(const int rows, const embed_t *A, embed_t *C,unsigned int *pos, const embed_t normA) {
  __shared__ embed_t fastA[numEmbeds];
  __shared__ embed_t partial[8*128];
  unsigned long id = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.x<numEmbeds) {
      fastA[threadIdx.x]= A[threadIdx.x]; // only one embeding is on A
  }
  __syncthreads();
  if (id<rows*8) {
    
    unsigned long row=id/8; // Get row
    unsigned long interiorId=id%8;  // Get id within row
    unsigned long interiorRow=(row%128)*8;  // Get row within shared memory, shared memory acting as a local cache
  partial[interiorRow+interiorId]=0;  // Initialize section of cache to be used as acumulator as 0
  for(unsigned int i=interiorId;i<numEmbeds;i+=8) {
      partial[interiorRow+interiorId]+=fastA[i] * c_model[row].data[i]; // Accumulate within the shared memory space
  }
  
  if (interiorId<4) {  // Unrolling to reduce the 8 elements within a row to 1
      partial[interiorRow+interiorId]+=partial[interiorRow+interiorId+4];
  }
  if (interiorId<2) {
      partial[interiorRow+interiorId]+=partial[interiorRow+interiorId+2];
  }
    if (interiorId==0) { // Final step and write results
        embed_t acum=0;
      acum=partial[interiorRow+interiorId]+partial[interiorRow+interiorId+1];
        C[row]=acum/(normA * c_norms[row]);
      pos[row]=row;

  }
  }
}



__global__ void FirstMerge
(const int64_t N, embed_t *sims, unsigned int* pos, const int64_t length, const int64_t pad) {
	const int64_t id = blockIdx.x * blockDim.x + threadIdx.x;
	const int64_t start=id*N;
	const int64_t end=start+N;
    if (start<length) { 
    
    // Insertion sort, as N SHOULD be small
    
		for(int64_t i=start+1; i<end; i++)
		{
			if (i<length){
                /*if (i >= pad || i < 0) {
                    printf("ERRORR1 %i\n", i);
                }*/
				const embed_t temp=sims[i];
				const int64_t position=pos[i];
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
(const int N, embed_t *sims, unsigned int* pos, const unsigned long stride) {

    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    id=id*N;
	unsigned int posA=0,posB=0;
    if (id<stride) { 
		unsigned int buffPos[maxN];
		embed_t buffSims[maxN];
		
        embed_t elemA=sims[(id+stride)];
        unsigned int posAuxA=pos[(id+stride)];
        embed_t elemB=sims[id];
        unsigned int posAuxB=pos[id];

        for(unsigned int i=0;i<N;++i) {
            if (posAuxA==posAuxB) {
                ++posA;
                elemA=sims[(id+posA+stride)];
                posAuxA=pos[(id+posA+stride)];
            }
            if (elemA>elemB && posA<N) {
                ++posA;
                buffSims[i]=elemA;
                buffPos[i]=posAuxA;
                
                elemA=sims[(id+posA+stride)];
                posAuxA=pos[(id+posA+stride)];
            }
            else {
                ++posB;
                buffSims[i]=elemB;
                buffPos[i]=posAuxB;
                
                elemB=sims[id+posB];
                posAuxB=pos[id+posB];

            }
		}

		memcpy(sims + id, buffSims, N * sizeof(embed_t));
		memcpy(pos + id, buffPos, N * sizeof(unsigned int));
	
}
}



// Load memory into cuda constants. This memory will be freed automatically at the end of the cuda context
extern "C"
void loadModel(embed_t * norms, embedV_t * model, uint32_t numRows)
{
	size_t numBytesModel = sizeof(embedV_t) * numRows;
	size_t numBytesNorms = sizeof(embed_t) * numRows;
	embedV_t* modelSym;
	embed_t* normsSym;

	gpuErrchk(cudaMalloc((embed_t**)&modelSym, numBytesModel));
	gpuErrchk(cudaMalloc((embed_t**)&normsSym, numBytesNorms));

	gpuErrchk(cudaMemcpy(modelSym, model, numBytesModel, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(normsSym, norms, numBytesNorms, cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpyToSymbol(c_model, (void**)&modelSym, sizeof(modelSym)));
	gpuErrchk(cudaMemcpyToSymbol(c_norms, (void**)&normsSym, sizeof(normsSym)));

	gpuErrchk(cudaDeviceSynchronize());// Coment this on release

}

// FUNCTIONS DEFINED IN CUDAHELP.CU
extern "C"
void reservePinnedMemory(embed_t* &ptr, size_t bytes);

extern "C"
void freePinnedMemory(void* ptr);

// MAIN FUNCTION TO RUN

extern "C"
void runCuda(embed_t* norms, embedV_t* model, uint32_t numRows, embedV_t A, embed_t normA, uint32_t N, int &returnCode, std::vector<unsigned int> &res)
{

	assert(N <= maxN);

	embedV_t queryTerm;
	embed_t* A_d;
    embed_t *C_d;
    unsigned int *positions,*pos_d;
	unsigned int nBlocks=(numRows/128)+1;
    unsigned int nBlocksOriginal=nBlocks;
	int nThreads=1024;
	float elapsedTime;
    
    unsigned int numRowsMod=numRows;
    if (numRows%N!=0) numRowsMod=(N-numRows%N)+numRows;
    numRowsMod+=numRowsMod%2*N;

    //printf("%u\n",numRows);
	embed_t* similarities;
	reservePinnedMemory(similarities, sizeof(embed_t) * numRowsMod);
	positions = reinterpret_cast<unsigned int*>(similarities);
	similarities = nullptr;
	reservePinnedMemory(similarities, sizeof(embed_t) * numRowsMod);


	cudaEvent_t start, stop;
    // request the model to look for
	//queryTerm = model[queryTermPos]; 
    queryTerm=A;
	//embed_t normA = norms[queryTermPos];


	unsigned int numBytesQuery = sizeof(embedV_t);
	unsigned int numBytesSims = sizeof(unsigned int) * numRowsMod;

	gpuErrchk(cudaEventCreate(&start));
	gpuErrchk(cudaEventCreate(&stop));
    
	gpuErrchk(cudaMalloc((embed_t**)&A_d, numBytesQuery));
	gpuErrchk(cudaMalloc((embed_t**)&C_d, numBytesSims));
	gpuErrchk(cudaMalloc((unsigned int**)&pos_d, numBytesSims));



	gpuErrchk(cudaMemcpyAsync(A_d, queryTerm.data, numBytesQuery, cudaMemcpyHostToDevice));

	gpuErrchk(cudaEventRecord(start, 0));

	DotProduct<<<nBlocks, nThreads >>>(numRows, A_d,  C_d, pos_d,normA);
    
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());// Coment this on release
    
    FirstMerge<<<nBlocks, nThreads >>>(N,C_d,pos_d,numRows,numRowsMod);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());// Coment this on release

	unsigned long toReduce=((numRowsMod/N)/2);
	
    while(toReduce>0) {
        nBlocks=((toReduce*N)/nThreads)+1;
		BotchedMergeSort <<<nBlocks, nThreads >>> (N, C_d, pos_d, toReduce * N);
		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize()); // Coment this on release
        if (toReduce>1){
            toReduce+=toReduce%2;
		}
        toReduce=toReduce/2;
    }
	
	// Because we don't use the similarities rigt now...
	// gpuErrchk(cudaMemcpyAsync(similarities, C_d, sizeof(embed_t)*N, cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpyAsync(positions, pos_d, sizeof(unsigned int)*N, cudaMemcpyDeviceToHost));

  cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      fprintf(stderr, "ERROR: %s \n", cudaGetErrorString(error));
      returnCode=1;
    }

	gpuErrchk(cudaFree(A_d));
	gpuErrchk(cudaFree(C_d));
	gpuErrchk(cudaFree(pos_d));

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
	printf("nBlocks: %d\n", nBlocksOriginal+1);
	printf("Tiempo Total %4.6f ms\n", elapsedTime);
	printf("Ancho de Banda %4.3f GB/s\n", (numRows *numEmbeds* sizeof(float)) / (1000000 * elapsedTime));
  
    std::vector<unsigned int> results;
    for (unsigned int i=0;i<N;++i) {
		results.push_back(positions[i]);
    }

	freePinnedMemory(similarities);
	freePinnedMemory(positions);

	res = results;
}



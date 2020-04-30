#include <iostream>
#include <stdlib.h>

#include <algorithm> 
#include <cuda_runtime.h>
#include <vector_types.h>
#include "GlobalHeader.h"
#include "loader.h"
#include <chrono> 

extern "C" std::vector<unsigned int> runCuda(embed_t * norms, embedV_t * model, int32_t numRows, int32_t queryTermPos,int32_t N, int &returnCode);

extern "C"
void loadModel(embed_t * norms, embedV_t * model, int32_t numRows);

extern "C"
void freeAll();

bool customCompare (embed_p i,embed_p j) { return (i.data>=j.data); }
std::vector<unsigned int> sequentialSearch(embed_t* norms,embedV_t* embeddings,unsigned int numElems,unsigned int idx,unsigned int N ) {
        
   std::vector<embed_p> similarities;
   embedV_t A=embeddings[idx];
   for(unsigned int i=0;i<numElems;++i) {
       embedV_t B=embeddings[i];
       float acum=0;
       for(int j=0;j<numEmbeds;++j) {
           acum+=A[j]*B[j];
       }
       embed_p res;
       res.data=acum/(norms[idx]*norms[i]);
       res.id=i;
       similarities.push_back(res);
    }
  std::vector<embed_p> orderedResults;
  for (unsigned int i=0;i<=N;++i) {
      orderedResults.push_back(similarities[i]);

  }
  std::sort (orderedResults.begin(), orderedResults.end(), customCompare);  
  for (unsigned int i=N;i<numElems;++i) {
      embed_p elem=similarities[i];
      for (int j=0;j<N;++j) {
          if (orderedResults[j].data<elem.data) {
              orderedResults[N]=elem;
              std::sort (orderedResults.begin(), orderedResults.end(), customCompare);  
              break;
          }
      }
  }
  std::vector<unsigned int> result;
  for (int i=0;i<N;++i) {
      result.push_back(orderedResults[i].id);
  }
  return result;
}


int main(int argc, char* argv[]) {

	if (argc != 2) {
		std::cout << "Needs a file to load" << std::endl;
		return 0;
	}

	std::vector<std::string> words;
	embed_t* norms;
	embedV_t* embeddings;
	size_t numElems;

	int res = loader::loadData(argv[1], numElems, words, norms, embeddings);

	if (res) {
		std::cout << "Embedings loaded" << std::endl;
	}
	else {
		std::cout << "ERROR::EMBEDINGS NOT LOADED!" << std::endl;
		return 1;
	}

	// load model
	loadModel(norms, embeddings, numElems);

	std::string word;
	bool runCPU;
	int returnCode = 0;
	std::vector<unsigned int> results;

	std::cout << "Enter word to look for similarities  [(bool) run CPU]" << std::endl;
	while (returnCode == 0 && std::cin >> word >> runCPU) {
		// Search word
		unsigned int idx = loader::binary_search(words, word);
		if (idx == -1) {
			std::cout << "Could not find word!!!!" << std::endl;
			continue;
		}
		
		std::cout << "Found word \"" << word << "\" in position " << idx << std::endl;
		if (runCPU) {
			auto start = std::chrono::steady_clock::now();
			results = sequentialSearch(norms, embeddings, numElems, idx, 11);
			auto stop = std::chrono::steady_clock::now();
			std::cout << "Most similar N words with CPU:" << std::endl;
			for (int i = 0; i < 11; ++i) {
				if (results[i] != idx)
					std::cout << words[results[i]] << std::endl;
			}

			std::cout << "CPU execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()
				<< " milliseconds\n";
		}
        auto startGPU = std::chrono::steady_clock::now();
        results  = runCuda(norms, embeddings, numElems, idx,11, returnCode);
        auto stopGPU = std::chrono::steady_clock::now();
std::cout << "GPU execution took: "<< std::chrono::duration_cast<std::chrono::milliseconds>(stopGPU - startGPU).count()
    << " milliseconds\n";   

       	std::cout << "Most similar N words:" << std::endl;
        for (int i=0;i<11;++i){
            if (results[i]!=idx)
				std::cout << words[results[i]] << std::endl;
        }
        
       
		std::cout << "Enter word to look for similarities  [(bool) run CPU]" << std::endl;
	}

	freeAll();
	// free data
	loader::freeData(norms, embeddings);

	return returnCode;
}

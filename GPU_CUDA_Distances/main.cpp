#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include "GlobalHeader.h"
#include "loader.h"

extern "C" int runCuda(embed_t * norms, embedV_t * model, int32_t numRows, int32_t queryTermPos,int32_t N);


int binary_search(std::vector<std::string> words, int length, std::string to_be_found) {

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

int main(int argc, char* argv[]) {

	if (argc != 2) {
		std::cout << "Needs a file to load" << std::endl;
		return 0;
	}

	std::vector<std::string> words;
	embed_t* norms;
	embedV_t* embeddings;
	int numElems;

	int res = loader::loadData(argv[1], numElems, words, norms, embeddings);

	if (res) {
		std::cout << "Embedings loaded" << std::endl;
	}
	else {
		std::cout << "ERROR::EMBEDINGS NOT LOADED!" << std::endl;
		return 1;
	}

	std::string word;
	int returnCode = 0;
	std::cout << "Enter word to look for similarities" << std::endl;
	while (returnCode == 0 && std::cin >> word) {
		returnCode  = runCuda(norms, embeddings, numElems, 30, 10);
		std::cout << "Enter word to look for similarities" << std::endl;
	}


	// free data
	loader::freeData(norms, embeddings);

	return returnCode;
}

#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include "GlobalHeader.h"
#include "loader.h"

extern "C" int runCuda(embed_t * norms, embedV_t * model, int32_t numRows, int32_t queryTermPos,int32_t N);


unsigned int binary_search(const std::vector<std::string>& words, const std::string& to_be_found) {

	unsigned int p = 0;
	unsigned int r = static_cast<unsigned int>(words.size()) - 1;
	unsigned int q = (r + p) / 2;
	unsigned int counter = 0;

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
		// Search word
		unsigned int idx = binary_search(words, word);
		if (idx == -1) {
			std::cout << "Could not find word!!!!" << std::endl;
			continue;
		}
		
		std::cout << "Found word " << word << " in position " << idx << std::endl;

		returnCode  = runCuda(norms, embeddings, numElems, idx, 5);
		std::cout << "Enter word to look for similarities" << std::endl;
	}


	// free data
	loader::freeData(norms, embeddings);

	return returnCode;
}

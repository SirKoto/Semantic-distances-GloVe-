#include <iostream>
#include <stdlib.h>

#include <algorithm> 
#include <cuda_runtime.h>
#include <vector_types.h>
#include "GlobalHeader.h"
#include "loader.h"
#include <chrono> 
#include <math.h> 
extern "C"
void loadModel(embed_t * norms, embedV_t * model, uint32_t numRows);


extern "C"
void runCuda(embed_t * norms, embedV_t * model, uint32_t numRows, embedV_t A, embed_t normA, uint32_t N, int& returnCode, std::vector<unsigned int> & res);

bool customCompare(const embed_p& i, const embed_p& j) { return (i.data >= j.data); }
std::vector<unsigned int> sequentialSearch(embed_t* norms, embedV_t* embeddings, unsigned int numElems, embedV_t A, unsigned int N) {
	float normA = 0;
	for (int i = 0; i < numEmbeds; ++i) {
		normA += sqrt(A[i] * A[i]);
	}
	std::vector<embed_p> similarities;
	for (unsigned int i = 0; i < numElems; ++i) {
		embedV_t B = embeddings[i];
		float acum = 0;
		for (int j = 0; j < numEmbeds; ++j) {
			acum += A[j] * B[j];
		}
		embed_p res;
		res.data = acum / (normA * norms[i]);
		res.id = i;
		similarities.push_back(res);
	}
	std::vector<embed_p> orderedResults;
	for (unsigned int i = 0; i <= N; ++i) {
		orderedResults.push_back(similarities[i]);

	}
	std::sort(orderedResults.begin(), orderedResults.end(), customCompare);
	for (unsigned int i = N; i < numElems; ++i) {
		embed_p elem = similarities[i];
		for (unsigned int j = 0; j < N; ++j) {
			if (orderedResults[j].data < elem.data) {
				orderedResults[N] = elem;
				std::sort(orderedResults.begin(), orderedResults.end(), customCompare);
				break;
			}
		}
	}
	std::vector<unsigned int> result;
	for (unsigned int i = 0; i < N; ++i) {
		result.push_back(orderedResults[i].id);
	}
	return result;
}


int main(int argc, char* argv[]) {

	if (argc != 2 && argc != 3) {
		std::cout << "Needs a file to load" << std::endl;
		std::cout << "./exe [txt full file]" << std::endl;
		std::cout << "./exe [txt strings] [bin file with embeddings and norms]" << std::endl;
		return 0;
	}

	std::vector<std::string> words;
	embed_t* norms;
	embedV_t* embeddings;
	size_t numElems;

	int res;
	if (argc == 2) {
		res = loader::loadData(argv[1], numElems, words, norms, embeddings);
	}
	else { // argc == 3
		res = loader::loadData(argv[1], argv[2], numElems, words, norms, embeddings);
	}

	if (res) {
		std::cout << "Embedings loaded" << std::endl;
	}
	else {
		std::cout << "ERROR::EMBEDINGS NOT LOADED!" << std::endl;
		return 1;
	}

	// load model
	auto startPreloadData = std::chrono::steady_clock::now();
	loadModel(norms, embeddings, static_cast<uint32_t>(numElems));
	auto endPreloadData = std::chrono::steady_clock::now();
	std::cout << "Data preloading took: " << std::chrono::duration_cast<std::chrono::milliseconds>(endPreloadData - startPreloadData).count()
		<< " milliseconds\n";

	std::string word;
	bool runCPU;
	int returnCode = 0;
	std::vector<unsigned int> results;

	std::cout << "Enter word to look for similarities in this input (word +/- words !) and (0/1) to run CPU" << std::endl;
	embedV_t A;
	while (returnCode == 0 && std::cin >> word) {
		unsigned int idx = loader::binary_search(words, word);
		if (idx == -1) {
			std::cout << "Could not find word!!!!" << std::endl;
			continue;
		}
		A = embeddings[idx];
		// Check that the next caracter is not a zero or a one
		std::cin >> std::ws;
		if (std::cin.peek() != '0' && std::cin.peek() != '1') {
			while (std::cin >> word) {
				if (word == "!") break;
				else if (word == "+") {
					std::cin >> word;
					unsigned int idx = loader::binary_search(words, word);
					if (idx == -1) {
						std::cout << "Could not find word!!!!" << std::endl;
						continue;
					}
					else {
						embedV_t aux = embeddings[idx];
						for (int i = 0; i < numEmbeds; ++i) {
							A[i] = A[i] + aux[i];
						}
					}
				}
				else if (word == "-") {
					std::cin >> word;
					unsigned int idx = loader::binary_search(words, word);
					if (idx == -1) {
						std::cout << "Could not find word!!!!" << std::endl;
						continue;
					}
					else {
						embedV_t aux = embeddings[idx];
						for (int i = 0; i < numEmbeds; ++i) {
							A[i] = A[i] - aux[i];
						}
					}
				}
			}
		}

		float normA = 0;
		for (int i = 0; i < numEmbeds; ++i) {
			normA += sqrt(A[i] * A[i]);
		}

		std::cin >> runCPU;
		// Search word
		//unsigned int idx = loader::binary_search(words, word);


		//std::cout << "Found word \"" << word << "\" in position " << idx << std::endl;
		if (runCPU) {
			auto start = std::chrono::steady_clock::now();
			results = sequentialSearch(norms, embeddings, static_cast<uint32_t>(numElems), A, 11);
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
		runCuda(norms, embeddings, static_cast<uint32_t>(numElems), A, normA, 11, returnCode, results);
		auto stopGPU = std::chrono::steady_clock::now();
		std::cout << "GPU execution took: " << std::chrono::duration_cast<std::chrono::milliseconds>(stopGPU - startGPU).count()
			<< " milliseconds\n";

		std::cout << "Most similar N words:" << std::endl;
		int counter = 0;
		for (int i = 0; i < 11; ++i) {
			if (results[i] != idx && counter < 10) {
				std::cout << words[results[i]] << std::endl;
				++counter;
			}
		}


		std::cout << "Enter word to look for similarities  and (0/1) to run CPU" << std::endl;
	}

	// free data
	loader::freeData(norms, embeddings);

	return returnCode;
}

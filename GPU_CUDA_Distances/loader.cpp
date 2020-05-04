#include "loader.h"

#include <fstream>
#include <chrono>
#include <iostream>

extern "C"
void reservePinnedMemory(embed_t * &ptr, size_t bytes);
extern "C"
void reservePinnedMemoryV(embedV_t * &ptr, size_t bytes);
extern "C"
void freePinnedMemory(void* ptr);

bool loader::loadData(const std::string& filename,
	size_t& numWords,
	std::vector<std::string>& words,
	embed_t*& norms,
	embedV_t*& embedings)
{
	std::ifstream stream(filename);
	if (!stream.is_open())
	{
		return false;
	}

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	// count lines 
	numWords = 0;
	{
		std::string buff;
		while (std::getline(stream, buff)) {
			numWords += 1;
		}
	}
	std::chrono::steady_clock::time_point linesCounted = std::chrono::steady_clock::now();

	std::cout << "Lines counted: " << numWords << std::endl;
	std::cout << "Count lines = " << std::chrono::duration_cast<std::chrono::microseconds>(linesCounted - begin).count() << " us " << std::endl;


	// reserve
	words.resize(numWords);
	reservePinnedMemory(norms, numWords * sizeof(embed_t));
	reservePinnedMemoryV(embedings,numWords * sizeof(embedV_t));


	// back to the begining
	stream.clear(); // must clear error flags (eof)
	stream.seekg(0);
	size_t idx = 0;
	while (idx < numWords) {
		stream >> words[idx] >> norms[idx]; // load word and precomputed norm

		for (int i = 0; i < numEmbeds; ++i) {
			stream >> embedings[idx][i]; // load embeding
		}
		idx += 1;
	}

	std::chrono::steady_clock::time_point dataLoaded = std::chrono::steady_clock::now();

	stream.close();

	std::cout << "Load data = " << std::chrono::duration_cast<std::chrono::microseconds>(dataLoaded - linesCounted).count() << " us " << std::endl;
	std::cout << "Total loading time = " << std::chrono::duration_cast<std::chrono::microseconds>(dataLoaded - begin).count() << " us " << std::endl;


	return true;
}



void loader::freeData(embed_t* norms, embedV_t* embedings)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	freePinnedMemory(norms);
	freePinnedMemory(embedings);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Unloaded data = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us " << std::endl;


}

unsigned int loader::binary_search(const std::vector<std::string>& words, const std::string& to_be_found) {

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	unsigned int p = 0;
	unsigned int r = static_cast<unsigned int>(words.size()) - 1;
	unsigned int q = (r + p) / 2;
	unsigned int counter = 0;

	while (p <= r)
	{
		counter++;
		if (words[q] == to_be_found) {
			break;
		} else {
			if (words[q] < to_be_found) {
				p = q + 1;
				q = (r + p) / 2;
			}
			else {
				r = q - 1;
				q = (r + p) / 2;
			}
		}
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	std::cout << "Binary Search = " << counter << " iterations, " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " us " << std::endl;

	if (p > r) {
		return -1;
	}
	else {
		return q;
	}

}

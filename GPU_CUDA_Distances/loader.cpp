#include "loader.h"

#include <fstream>

extern "C"
void reservePinnedMemory(embed_t * &ptr, int32_t bytes);
extern "C"
void reservePinnedMemoryV(embedV_t * &ptr, int32_t bytes);
extern "C"
void freePinnedMemory(void* ptr);

bool loader::loadData(const std::string& filename,
	int& numWords,
	std::vector<std::string>& words,
	embed_t*& norms,
	embedV_t*& embedings)
{
	std::ifstream stream(filename);
	if (!stream.is_open())
	{
		return false;
	}

	// count lines 
	numWords = 0;
	{
		std::string buff;
		while (std::getline(stream, buff)) {
			numWords += 1;
		}
	}

	// reserve
	words.resize(numWords);
	reservePinnedMemory(norms, numWords * sizeof(embed_t));
	reservePinnedMemoryV(embedings,numWords * sizeof(embedV_t));


	// back to the begining
	stream.clear(); // must clear error flags (eof)
	stream.seekg(0);
	int idx = 0;
	while (idx < numWords) {
		stream >> words[idx] >> norms[idx]; // load word and precomputed norm

		for (int i = 0; i < numEmbeds; ++i) {
			stream >> embedings[idx][i]; // load embeding
		}
		idx += 1;
	}


	stream.close();
	return true;
}



void loader::freeData(embed_t* norms, embedV_t* embedings)
{
	freePinnedMemory(norms);
	freePinnedMemory(embedings);
}

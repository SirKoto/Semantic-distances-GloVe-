#include "loader.h"

#include <fstream>


bool loader::loadData(const std::string& filename, std::vector<std::string>& words, std::vector<embed_t>& norms, std::vector<embedV_t>& embedings)
{
	std::ifstream stream(filename);
	if (!stream.is_open())
	{
		return false;
	}

	// count lines 
	int num = 0;
	{
		std::string buff;
		while (std::getline(stream, buff)) {
			num += 1;
		}
	}
	// reserve
	words.resize(num);
	norms.resize(num);

	embedings.resize(num);

	// back to the begining
	stream.clear(); // must clear error flags (eof)
	stream.seekg(0);

	int idx = 0;
	while (idx < num) {
		stream >> words[idx] >> norms[idx]; // load word and precomputed norm

		for (int i = 0; i < numEmbeds; ++i) {
			stream >> embedings[idx][i]; // load embeding
		}
		idx += 1;
	}


	stream.close();
	return num;
}

#pragma once

#include "GlobalHeader.h"
#include <string>
#include <vector>


class loader
{
public:
	// Load all necessary data
	static bool loadData(const std::string& filename,
		int& numWords,
		std::vector<std::string>& words,
		embed_t*& norms,
		embedV_t*& embedings);

	// Free from memory
	static void freeData(
		embed_t* norms,
		embedV_t* embedings);

};

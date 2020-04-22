#pragma once

#include "GlobalHeader.h"
#include <string>
#include <vector>


class loader
{
public:
	static bool loadData(const std::string &filename, 
		std::vector<std::string> &words,
		std::vector<embed_t> &norms,
		std::vector<embedV_t> &embedings);
};


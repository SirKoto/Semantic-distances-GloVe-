#pragma once
// embeding type
typedef float embed_t;
#define numEmbeds 300
// full vector of embedings type
#ifndef CUDA_INCLUDE
#include <array>
typedef std::array<embed_t, numEmbeds> embedV_t;
#endif
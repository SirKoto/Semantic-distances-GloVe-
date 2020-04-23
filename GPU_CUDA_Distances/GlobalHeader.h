#pragma once
// embeding type
typedef float embed_t;
#define numEmbeds 300

// This defines the description of a word in a vector
struct embedV_t {
	embed_t data[numEmbeds];

	embed_t& operator[](const int& i) {
		return data[i];
	}
};
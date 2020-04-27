#ifndef GLOBAL_HEADER_H
#define GLOBAL_HEADER_H


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


struct embed_p {
    embed_t data;
	unsigned int id; // unsigned int fits the entire number of words to load
};

#endif // !GLOBAL_HEADER_H
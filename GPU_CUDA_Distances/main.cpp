#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <vector_types.h>
#include <helper_cuda.h>


extern "C" int runCuda();

int main() {
	return runCuda();
}
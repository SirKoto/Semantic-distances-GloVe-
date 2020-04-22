#include <iostream>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <vector_types.h>



extern "C" int runCuda();

int main() {
	return runCuda();
}

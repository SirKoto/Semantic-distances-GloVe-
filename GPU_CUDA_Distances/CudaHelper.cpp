#include <iostream>

extern "C" void sayHelloWorld();

void sayHelloWorld() {
	std::cout << "Hello World" << std::endl;
}
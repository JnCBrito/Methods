#include "examples.h"
#include <string>
#include <iostream>

int main(int argc, char* argv[]) {
	std::string task;
	bool run_all(false);
	if (argc < 2) {
		std::cout << "Run newpricing <example_index>" << std::endl;
		run_all = true;
	} else {
		task = argv[1];
	}
	if (run_all || task == "1") option_pricing_1_Proj();
	if (run_all || task == "2") option_pricing_2_Proj();
	if (run_all || task == "3") option_pricing_3_Proj();
	return 0;
}

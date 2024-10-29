#include "Header.h"

#include "LayerId.h"

std::vector<LayerId> load_model(const std::string& Model_file_name) {
	std::vector<LayerId> Model;
	std::ifstream Model_file(Model_file_name);
	while (!Model_file.eof()) {
		int input1, input2; Model_file >> input1 >> input2;
		std::string setting; std::getline(Model_file, setting);
		Model.push_back(LayerId(Layer::type(input1), input2, setting));
	}
	Model_file.close();
	return Model;
}

std::vector<Matrix<double>> load_data(const std::string& DataBase_file_name, std::size_t input_size) {
	std::vector<Matrix<double>> Data;
	std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof()) {
		Data.push_back(Matrix<double>(input_size, 1));
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> Data.back()[i][0];
		}
	}
	return Data;
}

std::vector<Matrix<double>> load_data(const std::string&& DataBase_file_name, std::size_t input_size, std::size_t data_range) {
	int loop = 0;
	std::vector<Matrix<double>> Data;
	static std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof() && loop < data_range) {
		Data.push_back(Matrix<double>(input_size, 1));
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> Data.back()[i][0];
		}
		loop++;
	}
	//DataBase_file.close();
	return Data;
}
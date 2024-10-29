#include "Header.h"

#include "LayerId.h"

std::vector<LayerId> load_model(const std::string& Model_file_name);

std::vector<Matrix<double>> load_data(const std::string& DataBase_file_name, std::size_t input_size);

std::vector<Matrix<double>> load_data(const std::string&& DataBase_file_name, std::size_t input_size, std::size_t data_range);
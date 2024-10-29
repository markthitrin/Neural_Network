#include "Header.cuh"
#include "Genetive_algorithm.cu"
#include <thread>
#include <chrono>
#include <queue>
#include <iomanip>

// file name
const char* Model_file_name = "file/Model.txt";
const char* DataBase_file_name = "file/Data6.txt";
const char* output_file_name = "file/Output1.txt";
const char* RandWeightSetting_file_name = "file/Rand_weight_setting.txt";
const char* RandBiasSetting_file_name = "file/Rand_bias_setting.txt";
const char* Lost_file_name = "file/lost.txt";

// variable
int data_range;
int learning_range;
int testing_range;
int input_range;
int output_range;
int have_trained = 0;
double train_speed = 1;
int load_range = 0;
int batch = 1;
int batch_count = 0;


//runtime command
bool print_weight = false;
bool print_value = false;
bool print_bias = false;
bool pause = false;
int print_point = 300;
int print_lost = 0;
bool do_set_learning_rate = false;
double set_learning_rate_to = 0;
bool check_terminate = true;
bool do_save = false;
bool do_terminate = false;
std::string save_file_name;
void check_input() {
	while (check_terminate) {
		int i = 0;
		std::string input;
		std::getline(std::cin, input);
		std::string command = get_text(input, i);
		if (command == "pause")
			pause = true;
		else if (command == "print_point") {
			int number = get_number(input, i);
			print_point += number;
		}
		else if (command == "unpase")
			pause = false;
		else if (command == "terminate") {
			do_terminate = true;
		}
	}
}

std::vector<LayerId> load_model(const int max_size = 2000000000) {
	std::vector<LayerId> Model;
	std::ifstream Model_file(Model_file_name);
	int layer_number = 0;
	while (!Model_file.eof() && layer_number < max_size) {
		int input1, input2; Model_file >> input1 >> input2;
		std::string setting; std::getline(Model_file, setting);
		Model.push_back(LayerId(Layer::type(input1), input2, setting));
		layer_number++;
	}
	Model_file.close();
	return Model;
}

std::vector<Matrix<double>> load_data(std::size_t input_size) { // loas the whole data
	std::vector<Matrix<double>> Data;
	std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof()) {
		Data.push_back(Matrix<double>(input_size,1));
		double* get_input = new double[input_size];
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> get_input[i];
		}
		cudaMemcpy(Data.back().value, get_input, input_size * sizeof(double), cudaMemcpyHostToDevice);
		delete[] get_input;
	}
	return Data;
}

std::vector<Matrix<double>> load_data(std::size_t input_size, std::size_t data_range) { // load the data in specific range
	int loop = 0;
	std::vector<Matrix<double>> Data;
	static std::ifstream DataBase_file(DataBase_file_name);
	while (!DataBase_file.eof() && loop < data_range) {
		Data.push_back(Matrix<double>(input_size, 1));
		double* get_input = new double[input_size];
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> get_input[i];
		}
		cudaMemcpy(Data.back().value, get_input, input_size * sizeof(double), cudaMemcpyHostToDevice);
		delete[] get_input;
		loop++;
	}
	return Data;
}

std::vector<Matrix<double>> load_data(std::ifstream& DataBase_file, std::size_t input_size, std::size_t data_range) { // load the data in specific range
	int loop = 0;
	std::vector<Matrix<double>> Data;
	while (!DataBase_file.eof() && loop < data_range) {
		Data.push_back(Matrix<double>(input_size, 1));
		double* get_input = new double[input_size];
		for (int i = 0; i < input_size; i++) {
			DataBase_file >> get_input[i];
		}
		cudaMemcpy(Data.back().value, get_input, input_size * sizeof(double), cudaMemcpyHostToDevice);
		delete[] get_input;
		loop++;
	}
	return Data;
}

std::vector<std::pair<double, double>> load_rand_weight_setting() {
	std::vector<std::pair<double, double>> setting;
	std::ifstream RandWeightFile(RandWeightSetting_file_name);
	while (!RandWeightFile.eof()) {
		double input1, input2;
		RandWeightFile >> input1 >> input2;
		setting.push_back({ input1, input2 });
	}
	RandWeightFile.close();
	return setting;
}

std::vector<std::pair<double, double>> load_rand_bias_setting() {
	std::vector<std::pair<double, double>> setting;
	std::ifstream RandBiasFile(RandBiasSetting_file_name);
	while (!RandBiasFile.eof()) {
		double input1, input2;
		RandBiasFile >> input1 >> input2;
		setting.push_back({ input1,input2 });
	}
	RandBiasFile.close();
	return setting;
}



std::function<double(Neural_Network&)>
simulator = [](Neural_Network& AI) {
	double lost = 0;
	std::ifstream Data_file(DataBase_file_name);
	for (int i = 0; i < learning_range; i+=load_range) {
		std::vector<Matrix<double>> data = load_data(Data_file, AI.get_input_size(), load_range);
		for (int j = 0; j < load_range; j++) {
			int start_pos = rand() % (data.size() - input_range + output_range - 1);
			for (int k = 0; k < input_range; k++ ) {
				AI.feedforward(data[start_pos + k]);
			}
			for (int k = 0; k < output_range; k++) {
				lost += AI.get_loss(data[start_pos + input_range + k]);
				Matrix<double> get_output = AI.get_output();
				AI.feedforward(get_output);
			}
		}
		AI.fogot_all();
	}
	Data_file.close();
	return double(100000) / lost;
};

std::function<double(Neural_Network&)>
dense_simulator = [](Neural_Network& AI) {
	double lost = 0;
	double count = 0;
	std::ifstream Data_file(DataBase_file_name);
	for (int i = 0; i < learning_range; i += load_range) {
		std::vector<Matrix<double>> data = load_data(Data_file, AI.get_input_size(), load_range); // input ,output,input,output,input ...
		for (int j = 0; j < load_range; j += 2) {
			int start_pos = rand() % (data.size() - input_range + output_range - 1) & (!1);
			AI.feedforward(data[start_pos + j]);
			lost += AI.get_loss(data[start_pos + j + 1]);
			++count;
		}
		AI.fogot_all();
	}
	Data_file.close();
	double score = double(1000) / ( lost /  count);
	return score == score ? score : 0;
};

std::function<double(double)>
probability_func = [](double a) {
	return a;
};
// weight and bias initialization function
std::function<double(std::size_t, std::size_t)>
n_random_func = [](std::size_t size, std::size_t next) {
	//return std::pow(-1, rand() % 2) * double(rand() % 20000) / 20000;
	return std::pow(-1, rand() % 2) *  double(rand() % 30000) / 10000;
};
std::function<double()>
random_func = []() {
	//return std::pow(-1, rand() % 2) * double(rand() % 20000) / 20000;
	return std::pow(-1, rand() % 2) * std::tanh(double(rand() % 30000) / 10000);
};
std::function<double(std::size_t, std::size_t)> 
random_func2 = [](std::size_t size, std::size_t next) {
	return std::pow(-1, rand() % 2) * (double(rand() % 20000) / 10000) * std::sqrt(double(2) / size);
};
std::function<double()> 
zero = []() {
	return 0;
};





std::vector<Matrix<double>> predict(Neural_Network& AI, std::vector<Matrix<double>> Data, int start) {
	std::vector<Matrix<double>> result;
	for (int i = start; i < start + input_range; i++) {
		AI.predict(Data[i]);
	}

	for (int i = start + input_range; i < start + input_range + output_range; i++) {
		result.push_back(Matrix<double>(AI.get_input_size(), 1)); 
		result.back() = AI.get_output();
		AI.predict(result.back());
	}

	AI.fogot_all();
	return result;
}

char get_char(const Matrix<double>& M) {
	double max = -1000000;
	int pos = 0;
	double* a = new double[M.get_size()];
	cudaMemcpy(a, M.value, M.get_sizeb(), cudaMemcpyDeviceToHost);
	for(int i = 0 ;i<M.get_size();i++) {
		if (max < a[i]) {
			max = a[i];
			pos = i;
		}
	}
	delete[] a;
	return (char)pos;
}

int main() {
	try {
		std::cout << std::fixed;
		std::cout << std::setprecision(6);
		std::srand(std::time(0));
		std::vector<LayerId> get_model = load_model();
		std::cout << "Model was leaded successfully\n";
		
		
		std::vector<Matrix<double>> Data;
		std::cout << "data range : "; std::cin >> data_range;													// get setting 
		std::cout << "load_range : "; std::cin >> load_range; 
		std::cout << "learing range : ";std::cin >> learning_range;	
		std::cout << "input range : "; std::cin >> input_range;
		std::cout << "output range : "; std::cin >> output_range;
		
		testing_range = data_range - learning_range;
		
		std::vector<std::function<double(std::size_t, std::size_t)>> Weight_setting; 
		//Weight_setting.push_back(n_random_func);
		for (int i = 0; i < get_model.size() - 1; i++) { Weight_setting.push_back(random_func2); }
		std::cout << "Load rand weigth setting successfully\n";
		std::vector<std::function<double()>> Bias_setting;
		for (int i = 0; i < get_model.size() - 1; i++) { Bias_setting.push_back(zero); }
		std::cout << "Load rand bias setting successfully\n";

		if (Weight_setting.size() < get_model.size() - 1) {													// check for error setting
			std::cout << "Weight setting doesn't match AIsize\n"; return 0;
		}
		if (Bias_setting.size() < get_model.size() - 1) {
			std::cout << "Bias setting doesn't maych AIsize\n"; return 0;
		}



		Genetive_algorithm ML(get_model, dense_simulator, probability_func, 100, 0.1, 0.2);
		
		ML.rand_weight(Weight_setting);
		ML.rand_bias(Bias_setting);

		std::ofstream output_file(output_file_name);
		std::ofstream lost_file(Lost_file_name);

		//std::thread runtime(check_input);
		
		check_terminate = false;
		while (print_point) {
			ML.set_point(0);
			ML.simulate();
			ML.mutate();
			if (print_point) {
				print_point--;
				std::cout << "ML generation : " << ML.get_generetion_number() << " got max point : " << ML.get_max_point() << "\n";
				ML.print_point();
			}
			while (pause) {

			}
		}

		std::cout << "started testing\n";		

		//runtime.join();
		
		std::ofstream model_output("file/AI2.txt");
		ML.get_max_model().save_as(model_output);
		return 0;
	}
	catch (std::string Error) {																			
		std::cout << Error << std::endl;
		std::cin.get();
		return 0;
	}
	std::cin.get();
}
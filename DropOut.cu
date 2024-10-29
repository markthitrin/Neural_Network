#pragma once
#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"
#include "Variable.cuh"

class DropOut : public Layer {
public:
	DropOut () : Layer() { ; };

	DropOut(const std::size_t& size,
		std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) : Layer(Layer::DROPOUT, size) {
		setup_random_state(size);
	}

	DropOut(const LayerId& set,
		std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) : Layer(Layer::DROPOUT, set.Layer_size) {
		setup_random_state(set.Layer_size);

		set_layer(set.setting);
	}

	DropOut(const DropOut& copy) : Layer(copy) {
		setup_random_state(copy.value.get_size());

		drop_out_rate = copy.drop_out_rate;
	}

	DropOut(DropOut&& other) : Layer(std::move(other)) {
		random_state = other.random_state;
		other.random_state = nullptr;

		drop_out_rate = other.drop_out_rate;
	}

	~DropOut() {
		cudaFree(random_state);
	}



	Matrix<double> feed() {
		Matrix<double> filter(value.get_size(), 1);

		int blockPergrid = upper_value(double(value.get_size()) / 1024);
		int threadPerblock = std::min(value.get_size(), 1024);
		get_random << <blockPergrid, threadPerblock >> > (filter.value, random_state, value.get_size());
		cudaDeviceSynchronize();
		drop_out << <blockPergrid, threadPerblock >> > (filter.value, filter.value, drop_out_rate, value.get_size());
		cudaDeviceSynchronize();

		v.push_back(filter);

		return mul_each(value,filter);
	}

	Matrix<double> predict() {
		int blockPergrid = upper_value(double(value.get_size()) / 1024);
		int threadPerblock = std::min(value.get_size(), 1024);
		device_predict_computeDROPOUT<< <blockPergrid, threadPerblock >> > (value.value, double(1) - drop_out_rate, value.get_size());
		cudaDeviceSynchronize();

		return value;
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient) {
		std::vector<Matrix<double>> result;

		int start_pos = v.size() - gradient.size();

		for (int round = 0; round < gradient.size(); round++) {	
			result.push_back(Matrix<double>(gradient[round].row,1));

			int blockPergrid = upper_value(double(result[round].get_size()) / 1024);
			int threadPerblock = std::min(result[round].get_size(), 1024);
			device_valuechange_computeDROPOUT << <blockPergrid, threadPerblock >> > (result[round].value, gradient[round].value, v[start_pos + round].value, result[round].get_size());
			cudaDeviceSynchronize();
		}

		return result;
	}

	void mutate(const double& mutation_chance, const double& mutation_rate) {
		
	}



	void fogot(const std::size_t& number) {																		// delete old memory and shift the new memory
		int h = number;
		if (number > v.size())
			h = v.size();
		for (int i = 0; i < v.size() - h; i++) {
			v[i] = v[i + h];
		}
		for (int i = 0; i < h; i++) {
			v.pop_back();
		}
	}

	void fogot_all() {																							// delete all memory
		fogot(v.size());
	}



	void change_dependencies() {
		
	}

	void set_change_dependencies(const double& number) {

	}

	void mul_change_dependencies(const double& number) {

	}

	void set_drop_out_rate(const double& number) {
		drop_out_rate = number;
	}



	void reconstruct(const std::size_t& size, std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) {
		fogot_all();

		Layer::reconstruct(Layer::DROPOUT, size);
		
		setup_random_state(size);
	}

	void reconstruct(const LayerId& set, std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) {
		fogot_all();
		
		Layer::reconstruct(Layer::DROPOUT, set.Layer_size);

		setup_random_state(set.Layer_size);

		set_layer(set.setting);
	}

	void reconstruct(const DropOut& copy) {
		fogot_all();
		
		Layer::reconstruct(copy);

		setup_random_state(copy.value.get_size());

		drop_out_rate = copy.drop_out_rate;
	}

	void reconstruct(DropOut&& other) {
		fogot_all();
		
		Layer::reconstruct(std::move(other));

		random_state = other.random_state;
		other.random_state = nullptr;

		drop_out_rate = other.drop_out_rate;
	}



	void rand_weight(const double& min, const double& max) {
		
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		
	}

	void rand_weight(std::function<double()> func) {
	
	}

	void rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {

	}

	void rand_bias(const double& min, const double& max) {
		
	}

	void rand_bias(std::pair<const double&, const double&> setting) {
		
	}

	void rand_bias(std::function<double()> func) {
		
	}

	void rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
		
	}



	void print_value() {
		std::cout << "---------DropOut Layer----------\n";
		value.print();
	}



	double get_drop_out_rate() {
		return drop_out_rate;
	}

	void save_as(std::ofstream& output_file) {

	}

	void load(std::ifstream& input_file) {

	}

private:
	void setup_random_state(const int size) {
		if (random_state != nullptr)
			cudaFree(random_state);
		cudaMalloc(&random_state, sizeof(curandState) * size);
		int blockPergrid = upper_value(double(size) / 1024);
		int threadPerblock = std::min(size, 1024);
		set_random << <blockPergrid, threadPerblock >> > (random_state, rand(), size);
		cudaDeviceSynchronize();
	}

	void set_layer(const std::string& setting) {															// set the layer using command
		int size = setting.size();
		int i = 0;
		while (i < size) {
			std::string command = get_text(setting, i);
			if (command == "drop_out_rate")
				set_drop_out_rate(setting, i);
			else if (command == "")
				;
			else throw "command not found";
		}
	}

	void set_drop_out_rate(const std::string& str, int& i) {												// set drop out rate using command
		double a = get_number(str, i);
		drop_out_rate = a;
	}
																											// containing random function

	double drop_out_rate = 0.1;																				// dropout rate = ( 0 , 1 )
	curandState* random_state;
};
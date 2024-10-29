#pragma once

#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"
#include "Variable.cuh"


class Power : public Layer {
public:
	Power() { Layer_type = Layer::POWER; }

	Power(const std::size_t& size) {
		Layer_type = Layer::POWER;

		value.reconstruct(size, 1);
		exponent.reconstruct(size, 1);
		exponent_change.reconstruct(size, 1);

		set_random_state();
	}

	Power(const LayerId& set) {
		Layer_type = Layer::POWER;

		value.reconstruct(set.Layer_size, 1);
		exponent.reconstruct(set.Layer_size, 1);
		exponent_change.reconstruct(set.Layer_size, 1);

		set_random_state();

		set_Layer(set.setting);
	}

	Power(const Power& copy) {
		Layer_type = Layer::POWER;

		value.reconstruct(copy.value);
		exponent.reconstruct(copy.exponent);
		exponent_change.reconstruct(copy.exponent_change);

		learning_rate = copy.learning_rate;
		optimizer = copy.optimizer;
		decay_rate = copy.decay_rate;

		set_random_state();
	}

	Power(Power&& other) {
		Layer_type = Layer::POWER;

		value.reconstruct(std::move(other.value));
		exponent.reconstruct(std::move(other.exponent));
		exponent_change.reconstruct(std::move(other.exponent_change));

		learning_rate = other.learning_rate;
		optimizer = other.optimizer;
		decay_rate = other.decay_rate;

		random_state = other.random_state;
		other.random_state = nullptr;
	}

	~Power() {
		cudaFree(random_state);
	}



	Matrix<double> feed() {
		Matrix<double> copy = value;

		int blockPergrid = upper_value(double(value.get_size()) / 1024);
		int threadPerblock = std::min(value.get_size(), 1024);
		device_pow_func << <blockPergrid, threadPerblock >> > (copy.value, value.value, exponent.value, value.get_size());
		cudaDeviceSynchronize();

		v.push_back(value);

		return copy;
	}

	Matrix<double> predict() {
		Matrix<double> copy = value;

		int blockPergrid = upper_value(double(value.get_size()) / 1024);
		int threadPerblock = std::min(value.get_size(), 1024);
		device_pow_func << <blockPergrid, threadPerblock >> > (copy.value, value.value, exponent.value, value.get_size());
		cudaDeviceSynchronize();

		return copy;
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient) {
		std::vector<Matrix<double>> value_change;

		int start_pos = v.size() - gradient.size();

		for (int round = 0; round < gradient.size(); round++) {
			int blockPergrid = upper_value(double(value.get_size()) / 1024);
			int threadPerblock = std::min(value.get_size(), 1024);
			device_expochange_computePOWER << <blockPergrid, threadPerblock >> > (exponent_change.value, gradient[round].value, exponent.value, v[start_pos + round].value, value.get_size());
			cudaDeviceSynchronize();
		}

		for (int round = 0; round < gradient.size(); round++) {
			value_change.push_back(Matrix<double>(value.get_size(), 1));
			set_Matrix(value_change.back(), 0);

			int blockPergrid = upper_value(double(value.get_size()) / 1024);
			int threadPerblock = std::min(value.get_size(), 1024);
			device_valuechange_computePOWER << <blockPergrid, threadPerblock >> > (value_change.back().value, gradient[round].value, exponent.value, v[start_pos + round].value, value.get_size());
			cudaDeviceSynchronize();
		}

		return value_change;
	}

	void mutate(const double& mutation_chance, const double& mutation_rate) {
		int blockPergrid = upper_value(double(exponent.get_size()) / 1024);
		int threadPerblock = std::min(exponent.get_size(), 1024);
		mutate_array << <blockPergrid, threadPerblock >> > (exponent.value, exponent.value, random_state, mutation_chance, mutation_rate, exponent.get_size());
		cudaDeviceSynchronize();
	}



	void fogot(const std::size_t& number) {																	// delete old memory and shift the new memory
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

	void fogot_all() {																						// delete all memory
		fogot(v.size());
	}



	void change_dependencies() {
		++t;
		if (optimizer == SGD) {
			exponent = exponent + exponent_change * learning_rate;
		}
		else if (optimizer == MOMENTUM) {
			set_up_Matrix(s_exponent_change, exponent);

			s_exponent_change = s_exponent_change * decay_rate + exponent_change * (double(1) - decay_rate);

			exponent = exponent + s_exponent_change * learning_rate;
		}
		else if (optimizer == ADAM) {
			set_up_Matrix(s_exponent_change, exponent);
			set_up_Matrix(ss_exponent_change, exponent);

			s_exponent_change = s_exponent_change * decay_rate + exponent_change * (double(1) - decay_rate);

			ss_exponent_change = ss_exponent_change * decay_rate + mul_each(exponent_change, exponent_change) * (double(1) - decay_rate);

			exponent = exponent + devide_each(s_exponent_change * learning_rate, (pow_each(ss_exponent_change, 0.5) + 0.000001));
		}
	}

	void set_change_dependencies(const double& value) {														// set changing weight and chaing bias to specifc value
		set_Matrix(exponent_change, value);
	}

	void mul_change_dependencies(const double& value) {														// multiply changing weight and ching bias with specific value
		exponent_change = exponent_change * value;
	}

	void set_learning(const double& value) {
		learning_rate = value;
	}



	void reconstruct(const std::size_t& size) { // not allowed
		fogot_all();

		value.reconstruct(size, 1);
		exponent.reconstruct(size, 1);
		exponent_change.reconstruct(size, 1);

		set_random_state();
	}

	void reconstruct(const LayerId& set, const size_t& next) {
		fogot_all();

		value.reconstruct(set.Layer_size, 1);
		exponent.reconstruct(set.Layer_size, 1);
		exponent_change.reconstruct(set.Layer_size, 1);

		set_random_state();

		set_Layer(set.setting);
	}

	void reconstruct(const Power& copy) {
		fogot_all();
		Layer_type = Layer::POWER;

		value.reconstruct(copy.value);
		exponent.reconstruct(copy.exponent);
		exponent_change.reconstruct(copy.exponent_change);

		learning_rate = copy.learning_rate;
		optimizer = copy.optimizer;
		decay_rate = copy.decay_rate;

		set_random_state();
	}

	void reconstruct(Power&& other) {
		fogot_all();
		Layer_type = Layer::POWER;

		value.reconstruct(std::move(other.value));
		exponent.reconstruct(std::move(other.exponent));
		exponent_change.reconstruct(std::move(other.exponent_change));

		learning_rate = other.learning_rate;
		optimizer = other.optimizer;
		decay_rate = other.decay_rate;

		random_state = other.random_state;
		other.random_state = nullptr;
	}



	void rand_weight(const double& min, const double& max) {
		if (!exponent.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[exponent.get_size()];
		for (int i = 0; i < exponent.get_size(); i++)
			r[i] = mapping(rand() % 10000, 0, 10000, min, max);
		cudaMemcpy(exponent.value, r, exponent.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		if (!exponent.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[exponent.get_size()];
		for (int i = 0; i < exponent.get_size(); i++)
			r[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		cudaMemcpy(exponent.value, r, exponent.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::function<double()> func) {
		if (!exponent.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[exponent.get_size()];
		for (int i = 0; i < exponent.get_size(); i++)
			r[i] = func();
		cudaMemcpy(exponent.value, r, exponent.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
		if (!exponent.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[exponent.get_size()];
		for (int i = 0; i < exponent.get_size(); i++)
			r[i] = func(value.row, next);
		cudaMemcpy(exponent.value, r, exponent.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(const double& min, const double& max) {
		
	}

	void rand_bias(std::pair<const double&, const double&> setting) {
		
	}

	void rand_bias(std::function<double()> func) {
		
	}

	void rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
		
	}



	void print_weight() {
		std::cout << "--------Power layer---------\n";
		exponent.print();
	}

	void print_value() {
		std::cout << "--------Power layer----------\n";
		value.print();
	}

	void print_bias() {
		std::cout << "------Power layer------------\n";
	}



	void save_as(std::ofstream& output_file) {
		exponent.print(output_file);
	}

	void load(std::ifstream& input_file) {
		exponent.load(input_file);
	}

protected:
private:
	void set_random_state() {
		if (random_state != nullptr)
			cudaFree(random_state);
		cudaMalloc(&random_state, value.get_size() * sizeof(curandState));
		int blockPergrid = upper_value(double(value.get_size()) / 1024);
		int threadPerblock = std::min(value.get_size(), 1024);
		set_random << <blockPergrid, threadPerblock >> > (random_state, rand(), value.get_size());
		cudaDeviceSynchronize();
	}

	void set_Layer(const std::string& setting) {															// set the layer using command
		int size = setting.size();
		int i = 0;
		while (i < size) {
			std::string command = get_text(setting, i);
			if (command == "learning_rate")
				set_learning_rate(setting, i);
			else if (command == "opt")
				set_optimizer(setting, i);
			else if (command == "decay_rate")
				set_decay_rate(setting, i);
			else if (command == "")
				;
			else throw "command not found";
		}
	}

	void set_decay_rate(const std::string& str, int& i) {
		double a = get_number(str, i);
		decay_rate = a;
	}

	void set_optimizer(const std::string& str, int& i) {
		std::string _opt = get_text(str, i);
		if (_opt == "SGD")
			optimizer = Layer::SGD;
		else if (_opt == "MOMENTUM")
			optimizer = Layer::MOMENTUM;
		else if (_opt == "ADAM")
			optimizer = Layer::ADAM;
		else if (_opt == "")
			;
		else throw "optimizer not found";
	}

	void set_learning_rate(const std::string& str, int& i) {
		double a = get_number(str, i);
		learning_rate = a;
	}



	Matrix<double> exponent;

	Matrix<double> exponent_change;

	Matrix<double> s_exponent_change;

	Matrix<double> ss_exponent_change;

	curandState* random_state;
};
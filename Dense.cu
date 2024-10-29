#pragma once

#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"
#include "Variable.cuh"

class Dense : public Layer {
public:
	Dense() : Layer() { Layer_type = Layer::DENSE; }

	Dense(const std::size_t& size) : Layer(Layer::DENSE, size) {
		act_func = sigmoid_func;
		dact_func = dsigmoid_func;

		set_random_state();
	}

	Dense(const std::size_t& size, const std::size_t& next, 
		nvstd::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		nvstd::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func)  : 
		Layer(Layer::DENSE, size) {
		weight.reconstruct(next, size);
		weight_change.reconstruct(next, size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);

		act_func = _act_func;
		dact_func = _dact_func;

		set_random_state();
	}

	Dense(const LayerId& set, const std::size_t& next) : 
		Layer(Layer::DENSE, set.Layer_size) {
		weight.reconstruct(next, set.Layer_size);
		weight_change.reconstruct(next, set.Layer_size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);

		act_func = sigmoid_func;
		dact_func = dsigmoid_func;

		set_random_state();
		
		set_Layer(set.setting);	
	}

	Dense(const Dense& copy) : Layer(copy) {
		weight.reconstruct(copy.weight);
		weight_change.reconstruct(copy.weight_change);
		bias.reconstruct(copy.bias);
		bias_change.reconstruct(copy.bias_change);

		act_func = copy.act_func;
		dact_func = copy.dact_func;

		set_random_state();
	}

	Dense(Dense&& other) : Layer(std::move(other)) {
		weight.reconstruct(std::move(other.weight));
		weight_change.reconstruct(std::move(other.weight_change));
		bias.reconstruct(std::move(other.bias));
		bias_change.reconstruct(std::move(other.bias_change));

		act_func = other.act_func;
		dact_func = other.dact_func;

		set_random_state();
		cudaFree(other.random_state);
	}

	~Dense() {
		cudaFree(random_state);
	}


	Matrix<double> feed()  {																				// feedforward
		if (!weight.is_constructed())	
			throw "Undefined weight";

		v.push_back(value);

		return act_func((weight * value) + bias);
	}

	Matrix<double> predict() {
		if (!weight.is_constructed())
			throw "Undefined weight";

		return act_func((weight * value) + bias);
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient)  {
		if (gradient.size() > v.size())
			throw "invalid gradient size for backpropagation";

		std::vector<Matrix<double>> value_change;
		std::vector<Matrix<double>> doutput;

		const std::size_t start_pos = v.size() - gradient.size();

		for (int round = 0; round < gradient.size(); round++) {	
			doutput.push_back(dact_func((weight * v[round + start_pos] + bias), gradient[round]));
		}

		for (int round = 0; round < gradient.size(); round++) {
			int size = weight.get_size();
			int blockPergrid = upper_value(double(size) / 1024);
			int threadPerblock = std::min(size, 1024);
			device_weightchange_computeDENSE << <blockPergrid, threadPerblock >> > (weight_change.value, doutput[round].value, v[round].value, doutput[round].row, value.row);
			cudaDeviceSynchronize();
		}

		for (int round = 0; round < gradient.size(); round++) {	
			bias_change = bias_change + doutput[round];
		}

		for (int round = 0; round < gradient.size(); round++) {	
			value_change.push_back(Matrix<double>(value.row, 1));										
			set_Matrix(value_change.back(), 0);																

			int blockPergrid = upper_value(double(value.get_size()) / 1024);
			int threadPerblock = std::min(value.get_size(), 1024);
			device_flow_computeDENSE << <blockPergrid, threadPerblock >> > (value_change.back().value, doutput[round].value, weight.value, weight.row, weight.column);
			cudaDeviceSynchronize();
		}

		return value_change;
	}
	
	void mutate(const double& mutation_chance, const double& mutation_rate) {
		int blockPergrid = upper_value(double(weight.get_size()) / 1024);
		int threadPerblock = std::min(weight.get_size(), 1024);
		mutate_array << <blockPergrid, threadPerblock >> > (weight.value, weight.value, random_state, mutation_chance, mutation_rate, weight.get_size());
		cudaDeviceSynchronize();

		blockPergrid = upper_value(double(bias.get_size()) / 1024);
		threadPerblock = std::min(bias.get_size(), 1024);
		mutate_array << <blockPergrid, threadPerblock >> > (bias.value, bias.value, random_state, mutation_chance, mutation_rate, bias.get_size());
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
			weight = weight + weight_change * learning_rate;
			bias = bias + bias_change * learning_rate;
		}
		else if (optimizer == MOMENTUM) {
			set_up_Matrix(s_weight_change, weight);
			set_up_Matrix(s_bias_change, bias);

			s_weight_change = s_weight_change * decay_rate + weight_change * (double(1) - decay_rate);
			s_bias_change = s_bias_change * decay_rate + bias_change * (double(1) - decay_rate);

			weight = weight + s_weight_change * learning_rate;
			bias = bias + bias_change * learning_rate;
		}
		else if (optimizer == ADAM) {
			set_up_Matrix(s_weight_change, weight);
			set_up_Matrix(s_bias_change, bias);
			set_up_Matrix(ss_weight_change, weight);
			set_up_Matrix(ss_bias_change, bias);

			s_weight_change = s_weight_change * decay_rate + weight_change * (double(1) - decay_rate);
			s_bias_change = s_bias_change * decay_rate + bias_change * (double(1) - decay_rate);

			ss_weight_change = ss_weight_change * decay_rate + mul_each(weight_change, weight_change) * (double(1) - decay_rate);
			ss_bias_change = ss_bias_change * decay_rate + mul_each(bias_change, bias_change) * (double(1) - decay_rate);

			weight = weight + devide_each(s_weight_change * learning_rate, (pow_each(ss_weight_change, 0.5) + 0.000001));
			bias = bias + devide_each(s_bias_change * learning_rate, (pow_each(ss_bias_change, 0.5) + 0.000001));
		}
	}

	void set_change_dependencies(const double& value) {														// set changing weight and chaing bias to specifc value
		set_Matrix(weight_change, value);
		set_Matrix(bias_change, value);
	}

	void mul_change_dependencies(const double& value) {														// multiply changing weight and ching bias with specific value
		weight_change = weight_change * value;
		bias_change = bias_change * value;
	}

	void set_learning(const double& value) {
		learning_rate = value;
	}
	


	void reconstruct(const std::size_t& size, const std::size_t& next,
		nvstd::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		nvstd::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func) {
		fogot_all();

		Layer::reconstruct(Layer::DENSE, size);

		weight.reconstruct(next, size);
		weight_change.reconstruct(next, size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);

		act_func = _act_func;
		dact_func = _dact_func;

		set_random_state();
	}

	void reconstruct(const LayerId& set,const size_t& next) {
		fogot_all();

		Layer::reconstruct(Layer::DENSE, set.Layer_size);

		weight.reconstruct(next, set.Layer_size);
		weight_change.reconstruct(next, set.Layer_size);
		bias.reconstruct(next, 1);
		bias_change.reconstruct(next, 1);

		act_func = sigmoid_func;
		dact_func = dsigmoid_func;

		set_random_state();

		set_Layer(set.setting);
	}

	void reconstruct(const Dense& copy) {
		fogot_all();
		
		Layer::reconstruct(copy);

		weight.reconstruct(copy.weight);
		weight_change.reconstruct(copy.weight_change);
		bias.reconstruct(copy.bias);
		bias_change.reconstruct(copy.bias_change);

		act_func = copy.act_func;
		dact_func = copy.dact_func;

		set_random_state();
	}

	void reconstruct(Dense&& other) {
		fogot_all();
		
		Layer::reconstruct(std::move(other));

		weight.reconstruct(std::move(other.weight));
		weight_change.reconstruct(std::move(other.weight_change));
		bias.reconstruct(std::move(other.bias));
		bias_change.reconstruct(std::move(other.bias_change));

		act_func = other.act_func;
		dact_func = other.dact_func;

		cudaFree(random_state);

		random_state = other.random_state;
		other.random_state = nullptr;
	}



	void rand_weight(const double& min, const double& max) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for(int i = 0 ;i<weight.get_size();i++)
			r[i] = mapping(rand() % 10000, 0, 10000, min, max);
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for(int i = 0 ;i<weight.get_size();i++)
			r[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::function<double()> func) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		double* r = new double[weight.get_size()];
		for(int i = 0;i<weight.get_size();i++)
			r[i] = func();
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_weight(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		if (!weight.is_constructed())
			throw "cant set undefined weight value";

		int size = weight.get_size();
		double* r = new double[weight.get_size()];
		for (int i = 0; i < weight.get_size(); i++)
			r[i] = func(value.row, next);
		cudaMemcpy(weight.value, r, weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(const double& min, const double& max) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for (int i = 0; i < bias.get_size();i++)
			r[i] = mapping(rand() % 10000, 0, 10000, min, max);
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(std::pair<const double&, const double&> setting) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for (int i = 0; i < bias.get_size(); i++) 
			r[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(std::function<double()> func) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";
		
		double* r = new double[bias.get_size()];
		for(int i =0 ;i<bias.get_size();i++)
			r[i] = func();
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}

	void rand_bias(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		if (!bias.is_constructed())
			throw "cant set undefined bias value";

		double* r = new double[bias.get_size()];
		for (int i = 0; i < bias.get_size();i++)
			r[i] = func(value.row, next);
		cudaMemcpy(bias.value, r, bias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] r;
	}



	void print_weight() {
		std::cout << "---------Dense Layer----------\n";
		weight.print();
	}

	void print_value() {
		std::cout << "---------Dense Layer----------\n";
		value.print();
	}

	void print_bias(){
		std::cout << "---------Dense Layer---------\n";
		bias.print();
	}



	nvstd::function<Matrix<double>(const Matrix<double>&)> get_act_func() {
		return act_func;
	}

	nvstd::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dact_func() {
		return dact_func;
	}


	void save_as(std::ofstream& output_file) {
		weight.print(output_file);
		bias.print(output_file);
	}
	
	void load(std::ifstream& input_file) {
		weight.load(input_file);
		bias.load(input_file);
	}

protected:
private:
	void set_random_state() {
		if (random_state != nullptr)
			cudaFree(random_state);
		cudaMalloc(&random_state, weight.get_size() * sizeof(curandState));
		int blockPergrid = upper_value(double(weight.get_size()) / 1024);
		int threadPerblock = std::min(weight.get_size(), 1024);
		set_random << <blockPergrid, threadPerblock >> > (random_state, rand(), weight.get_size());
		cudaDeviceSynchronize();
	}

	void set_Layer(const std::string& setting) {															// set the layer using command
		int size = setting.size();
		int i = 0;
		while (i < size) {
			std::string command = get_text(setting, i);
			if (command == "act")
				universal_set_func(act_func, setting, i);
			else if (command == "dact")
				universal_set_func(dact_func, setting, i);
			else if (command == "learning_rate")
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



	Matrix<double> weight_change;
	Matrix<double> bias_change;

	Matrix<double> weight;
	Matrix<double> bias;

	Matrix<double> s_weight_change;
	Matrix<double> s_bias_change;

	Matrix<double> ss_weight_change;
	Matrix<double> ss_bias_change;

	std::function<Matrix<double>(const Matrix<double>&)> act_func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dact_func;

	curandState* random_state;
};
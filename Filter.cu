#pragma once
#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"
#include "Variable.cuh"


class Filter : public Layer {
public:
	Filter() : Layer() { ; };

	Filter(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _func = descale_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc = ddescale_func) :
		Layer(Layer::FILTER, size), func(_func) , dfunc(_dfunc)
	{

	}

	Filter(const LayerId& set) : Layer(Layer::FILTER, set.Layer_size) {
		func = descale_func;
		dfunc = ddescale_func;

		set_Layer(set.setting);
	}

	Filter(const Filter& copy) : Layer(copy) {
		func = copy.func;
		dfunc = copy.dfunc;
	}

	Filter(Filter&& other) : Layer(std::move(other)) {
		func = other.func;
		dfunc = other.dfunc;
	}

	~Filter() {

	}


	Matrix<double> feed() {																					// feedforward
		v.push_back(value);																					// remember value
		return func(value);																					// return output
	}

	Matrix<double> predict() {
		return func(value);
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient) {					// backpropagation
		int start_pos = v.size() - gradient.size();															// rearrange the gradient In case givven gradient is shorter than memory

		std::vector<Matrix<double>> result;																	// flow gradient

		for (int round = 0; round < gradient.size(); round++) {												// loop though every time step
			result.push_back(Matrix<double>(value.row, 1));
			result.back() = dfunc(v[round + start_pos], gradient[round]);									// compute gradient
		}
		return result;
	}

	void mutate(const double& mutation_chance, const double& mutation_rate) {

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

	}

	void set_change_dependencies(const double& number) {

	}

	void mul_change_dependencies(const double& number) {

	}



	void reconstruct(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _func = descale_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc = ddescale_func) {
		fogot_all();
		
		Layer::reconstruct(Layer::FILTER, size);
		
		func = _func;
		dfunc = _dfunc;
	}

	void reconstruct(const LayerId& set) {
		fogot_all();
		
		Layer::reconstruct(Layer::FILTER, set.Layer_size);

		func = descale_func;
		dfunc = ddescale_func;

		set_Layer(set.setting);
	}

	void reconstruct(const Filter& copy) {
		fogot_all();
		
		Layer::reconstruct(copy);

		func = copy.func;
		dfunc = copy.dfunc;
	}

	void reconstruct(Filter&& other) {
		fogot_all();
		
		Layer::reconstruct(std::move(other));

		func = other.func;
		dfunc = other.dfunc;
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
		std::cout << "---------Filter Layer----------\n";
		value.print();
	}

	void save_as(std::ofstream& output_file) {

	}

	void load(std::ifstream& input_file) {

	}

private:
	void set_Layer(const std::string& setting) {															// set layer using command
		int size = setting.size();
		int i = 0;
		std::string a;
		while (i < size) {
			a = get_text(setting, i);
			if (a == "func")
				universal_set_func(func, setting, i);
			else if (a == "dfunc")
				universal_set_func(dfunc, setting, i);
			else if (a == "")
				;
			else throw "command not found";
		}
	}



	std::function<Matrix<double>(const Matrix<double>&)> func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dfunc;
};
#pragma once
#include "Header.h"
#include "Layer.cpp"
#include "LayerId.cpp"

#include "Func.h"



class Filter : public Layer {
public:
	Filter() { ; };

	Filter(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _func = descale_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc = ddescale_func) :
		func(_func) , dfunc(_dfunc)
	{
		Layer_type = Layer::FILTER;

		value.reconstruct(size, 1);
	}

	Filter(const LayerId& set) {
		Layer_type = Layer::FILTER;

		value.reconstruct(set.Layer_size, 1);

		func = descale_func;
		dfunc = ddescale_func;

		set_Layer(set.setting);
	}



	Matrix<double> feed() {																					// feedforward
		v.push_back(value);																					// remember value
		return func(value);																					// return output
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) {					// backpropagation
		int start_pos = v.size() - gadient.size();															// rearrange the gadient In case givven gadient is shorter than memory

		std::vector<Matrix<double>> result;																	// flow gadient

		for (int round = 0; round < gadient.size(); round++) {												// loop though every time step
			result.push_back(Matrix<double>(value.get_row(), 1));
			result.back() = dfunc(v[round + start_pos], gadient[round]);									// compute gadient
		}
		return result;
	}



	void forgot(const std::size_t& number) {																	// delete old memory and shift the new memory
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

	void forgot_all() {																						// delete all memory
		forgot(v.size());
	}



	void change_dependencies() {

	}

	void set_change_dependencies(const double& number) {

	}

	void mul_change_dependencies(const double& number) {

	}



	void reconstruct(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc) {
		forgot_all();
		
		value.reconstruct(size, 1);
		
		func = _func;
		dfunc = _dfunc;
	}

	void reconstruct(const LayerId& set) {
		forgot_all();
		
		value.reconstruct(set.Layer_size, 1);

		func = descale_func;
		dfunc = ddescale_func;

		set_Layer(set.setting);
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
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << value[i][0] << "    \t";
		}std::cout << std::endl;
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
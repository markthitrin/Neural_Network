#pragma once

#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Function.h"
#include "Variable.h"


class Dense : public Layer {
public:

	enum weight_type {
		WEIGHT = 0
	};

	enum bias_type {
		BIAS = 0
	};

	Dense();

	Dense(const std::size_t& size);

	Dense(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func);

	Dense(const LayerId& set, const std::size_t& next);

	Dense(const Dense& copy);

	Dense(Dense&& move);



	Matrix<double> feed();

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient);



	void change_dependencies();

	void set_change_dependencies(const double& value);

	void mul_change_dependencies(const double& value);

	void map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func);

	bool found_nan();



	void reconstruct(const std::size_t& size);

	void reconstruct(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _act_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func = dsigmoid_func);

	void reconstruct(const LayerId& set, const size_t& next);

	void reconstruct(const Dense& copy);

	void reconstruct(Dense&& move);



	void set_weight(const int& weight_type, const Matrix<double>& _weight);

	void set_bias(const int& bias_type, const Matrix<double>& _bias);

	void rand_weight(const double& min, const double& max);

	void rand_weight(std::pair<const double&, const double&> setting);

	void rand_weight(std::function<double()> func);

	void rand_weight(std::function<double(std::size_t, std::size_t)> func,const std::size_t& next);

	void rand_bias(const double& min, const double& max);

	void rand_bias(std::pair<const double&, const double&> setting);

	void rand_bias(std::function<double()> func);

	void rand_bias(std::function<double(std::size_t, std::size_t)> func,const std::size_t& next);



	void print_weight();

	void print_value();

	void print_bias();



	std::function<Matrix<double>(const Matrix<double>&)> get_act_func();

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dact_func();

	double get_max_abs_change_dependencies();


	void save_as(std::ofstream& output_file);

	void load(std::ifstream& input_file);


	void set_optimizer(const Layer::opt _opt);

protected:
private:

	void set_Layer(const std::string& setting);														// containing changing buas computed by backpropagation and will be added to bias \

	Matrix<double> weight;																					// containing weight
	Matrix<double> bias;

	Matrix<double> weight_change;																			// containing changing weight computed by backpropagation and will be added to weight															
	Matrix<double> bias_change;

	Matrix<double> s_weight_change;
	Matrix<double> s_bias_change;

	Matrix<double> ss_weight_change;
	Matrix<double> ss_bias_change;																					// containing bias

	std::function<Matrix<double>(const Matrix<double>&)> act_func;											// activate function
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dact_func;					// derivatives of activate function
};
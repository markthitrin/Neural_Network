#pragma once
#include "Header.h"
#include "Matrix.h"

extern class Neural_Network;

class LayerId;

class Layer {
public:
	enum type { UNDEFINED, DENSE, RNN, LSTM, DROPOUT, FILTER, POWER };

	enum opt { SGD, MOMENTUM, ADAM };

	Layer();

	Layer(const std::size_t& size);

	Layer(const LayerId _layer_id);

	Layer(const Layer& copy);

	Layer(Layer&& move);

	virtual ~Layer();



	virtual Matrix<double> feed() = 0;

	virtual std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) = 0;

	virtual void forgot(const std::size_t& number);

	virtual void forgot_all();

	virtual void set_drop_out_rate(const double& number);

	virtual void change_dependencies();

	virtual void set_change_dependencies(const double& value);

	virtual void mul_change_dependencies(const double& value);

	virtual void map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func);

	virtual bool found_nan();



	virtual void reconstruct(const std::size_t& size);

	virtual void reconstruct(const LayerId _layer_id);

	virtual void reconstruct(const Layer& copy);

	virtual void reconstruct(Layer&& move);


	virtual void set_weight(const int& weight_type, const Matrix<double>& _weight);

	virtual void set_bias(const int& bias_type, const Matrix<double>& _bias);

	virtual void rand_weight(const double& min, const double& max);

	virtual void rand_weight(const std::pair<double, double>& setting);

	virtual void rand_weight(const std::function<double()>& setting);

	virtual void rand_weight(const std::function<double(std::size_t, std::size_t)>& setting, const std::size_t& next);

	virtual void rand_bias(const double& min, const double& max);

	virtual void rand_bias(const std::pair<double, double>& setting);

	virtual void rand_bias(const std::function<double()>& setting);

	virtual void rand_bias(const std::function<double(std::size_t, std::size_t)>& setting, const std::size_t& next);


	virtual void print_weight();

	virtual void print_bias();

	virtual void print_value();



	virtual std::size_t get_size();

	virtual Layer::type get_type();

	virtual double get_learning_rate();

	virtual void set_value(const Matrix<double>& _value);

	virtual Matrix<double> get_value();

	virtual double get_max_abs_change_dependencies();



	virtual void set_learning_rate(const double& _learning_rate);

	virtual void set_optimizer(const Layer::opt _opt);


	virtual void save_as(std::ofstream& output_file);

	virtual void load(std::ifstream& input_file);


	Matrix<double> operator=(const Matrix<double>& rhs);

protected:

	Matrix<double> value;
	std::vector<Matrix<double>> v;
	type Layer_type = UNDEFINED;
	double learning_rate = 0.1;
	opt optimizer = SGD;
	double decay_rate = 0.9;
	bool do_record = false;

	int t = 0;
	friend Neural_Network;
};
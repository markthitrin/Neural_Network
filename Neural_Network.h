#pragma once

#include "Header.h"
#include "Dense.h"
#include "LSTM.h"
#include "LayerId.h"
#include "DropOut.h"
#include "Filter.h"
#include "Power.h"
#include "Variable.h"

class Neural_Network {
public:
	Neural_Network();

	Neural_Network(std::vector<LayerId> _layer,
		std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func) ;

	Neural_Network(const Neural_Network& copy);

	Neural_Network(Neural_Network&& move);



	void set_weight(const int& layer_number, const int& weight_type, const Matrix<double>& _weight);

	void set_bias(const int& layer_number, const int& bias_type, const Matrix<double>& _bias);

	void rand_weight(const int& layer_number, const double& min, const double& max);
	void rand_weight(const double& min, const double& max);

	void rand_weight(const int& layer_number, const std::pair<double, double>& setting);
	void rand_weight(const std::vector<std::pair<double, double>>& setting);

	void rand_weight(const int& layer_number, const std::function<double()>& setting);
	void rand_weight(const std::vector<std::function<double()>>& setting);

	void rand_weight(const int& layer_number, const std::function<double(std::size_t, std::size_t)>& setting);
	void rand_weight(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting);

	void rand_bias(const int& layer_number, const double& min, const double& max);
	void rand_bias(const double& min, const double& max);

	void rand_bias(const int& layer_number, const std::pair<double, double>& setting);
	void rand_bias(const std::vector<std::pair<double, double>>& setting);

	void rand_bias(const int& layer_number, const std::function<double()>& setting);
	void rand_bias(const std::vector<std::function<double()>>& setting);

	void rand_bias(const int& layer_number, const std::function<double(std::size_t, std::size_t)>& setting);
	void rand_bias(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting);




	void print_weight();

	void print_value();

	void print_bias();



	Matrix<double> feedforward(const Matrix<double>& input);

	void backpropagation(const Matrix<double>& target);

	

	void reconstruct(std::vector<LayerId> _layer,
		std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func);

	void reconstruct(const Neural_Network& copy);

	void reconstruct(Neural_Network&& move);



	void forgot(const std::size_t& number);

	void forgot_all();



	

	void change_dependencies(const std::size_t& pos);
	void change_dependencies();

	void set_change_dependencies(const std::size_t& layer_number,const double& number);
	void set_change_dependencies(const double& number);


	void mul_change_dependencies(const std::size_t& layer_number, const double& number);
	void mul_change_dependencies(const double& number);

	bool found_nan(const std::size_t& layer_number);
	bool found_nan();


	void set_learning_rate(const std::size_t& layer_number, const double& number);
	void set_learning_rate(const double& number);

	void set_do_record(const bool& value);

	void set_drop_out_rate(const std::size_t& layer_number, const double& number);
	void set_drop_out_rate(const double& number);

	std::size_t get_layer_size();

	Matrix<double> get_output();

	std::size_t get_input_size();

	double get_loss(const Matrix<double>& target);

	double get_max_abs_change_dependencies(const std::size_t& layer_number);
	double get_max_abs_change_dependencies();



	void save_as(std::ofstream& output_file);

	void load(std::ifstream& input_file);


private:

	std::vector<Layer*> layer;
	std::function<double(const Matrix<double>&, const Matrix<double>&)> loss_func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dloss_func;
};
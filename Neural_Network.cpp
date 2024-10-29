#pragma once

#include "Header.h"
#include "Dense.h"
#include "LSTM.h"
#include "LayerId.h"
#include "DropOut.h"
#include "Filter.h"
#include "Power.h"
#include "Neural_Network.h"

Neural_Network::Neural_Network() {}

Neural_Network::Neural_Network(std::vector<LayerId> _layer,
	std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func) {
	reconstruct(_layer, _loss_func, _dloss_func);
}

Neural_Network::Neural_Network(const Neural_Network& copy) {
	reconstruct(copy);
}

Neural_Network::Neural_Network(Neural_Network&& move) {
	reconstruct(std::move(move));
}


void Neural_Network::set_weight(const int& layer_number, const int& weight_type, const Matrix<double>& _weight) {
	layer[layer_number]->set_weight(weight_type, _weight);
}

void Neural_Network::set_bias(const int& layer_number, const int& bias_type, const Matrix<double>& _bias) {
	layer[layer_number]->set_bias(bias_type, _bias);
}

void Neural_Network::rand_weight(const int& layer_number, const double& min, const double& max) {
	if (min > max) {
		throw std::runtime_error("Invalid random range");
	}
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_weight(min, max);
}
void Neural_Network::rand_weight(const double& min, const double& max) {
	for (int i = 0; i < layer.size() - 1; i++) {
		rand_weight(i, min, max);
	}
}

void Neural_Network::rand_weight(const int& layer_number, const std::pair<double, double>& setting) {
	if (setting.first > setting.second) {
		throw std::runtime_error("Invalid random range");
	}
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_weight(setting);
}
void Neural_Network::rand_weight(const std::vector<std::pair<double, double>>& setting) {
	if (setting.size() != layer.size() - 1)
		throw std::runtime_error("Invalid random weight value");

	for (int i = 0; i < layer.size() - 1; i++) {
		rand_weight(i, setting[i]);
	}
}

void Neural_Network::rand_weight(const int& layer_number, const std::function<double()>& setting) {
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_weight(setting);
}
void Neural_Network::rand_weight(const std::vector<std::function<double()>>& setting) {
	if (setting.size() != layer.size() - 1)
		throw std::runtime_error("Invalid random weight value");

	for (int i = 0; i < layer.size() - 1; i++) {
		rand_weight(i, setting[i]);
	}
}

void Neural_Network::rand_weight(const int& layer_number, const std::function<double(std::size_t, std::size_t)>& setting) {
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_weight(setting, layer[layer_number + 1]->get_size());
}
void Neural_Network::rand_weight(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
	if (setting.size() != layer.size() - 1)
		throw std::runtime_error("Invalid random weight value");

	for (int i = 0; i < layer.size() - 1; i++) {
		rand_weight(i, setting[i]);
	}
}

void Neural_Network::rand_bias(const int& layer_number, const double& min, const double& max) {
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_bias(min, max);
}
void Neural_Network::rand_bias(const double& min, const double& max) {
	for (int i = 0; i < layer.size() - 1; i++) {
		rand_bias(i, min, max);
	}
}

void Neural_Network::rand_bias(const int& layer_number, const std::pair<double, double>& setting) {
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_bias(setting);
}
void Neural_Network::rand_bias(const std::vector<std::pair<double, double>>& setting) {
	if (setting.size() != layer.size() - 1)
		throw std::runtime_error("invalid random bias value");

	for (int i = 0; i < layer.size() - 1; i++) {
		rand_bias(i, setting[i]);
	}
}

void Neural_Network::rand_bias(const int& layer_number, const std::function<double()>& setting) {
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_bias(setting);
}
void Neural_Network::rand_bias(const std::vector<std::function<double()>>& setting) {
	if (setting.size() != layer.size() - 1)
		throw std::runtime_error("invalid random bias value");

	for (int i = 0; i < layer.size() - 1; i++) {
		rand_bias(i, setting[i]);
	}
}

void Neural_Network::rand_bias(const int& layer_number, const std::function<double(std::size_t, std::size_t)>& setting) {
	if (layer_number >= layer.size() - 1) {
		throw std::runtime_error("Invalid layer number");
	}
	layer[layer_number]->rand_bias(setting, layer[layer_number + 1]->get_size());
}
void Neural_Network::rand_bias(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
	if (setting.size() != layer.size() - 1)
		throw std::runtime_error("invalid random bias value");

	for (int i = 0; i < layer.size() - 1; i++) {
		rand_bias(i, setting[i]);
	}
}



void Neural_Network::print_weight() {
	std::cout << "======== weight ========\n";
	for (int i = 0; i < layer.size() - 1; i++) {
		layer[i]->print_weight();
	}
}

void Neural_Network::print_bias() {
	std::cout << "========= bias ==========\n";
	for (int i = 0; i < layer.size() - 1; i++) {
		layer[i]->print_bias();
	}
}

void Neural_Network::print_value() {
	std::cout << "======== value =========\n";
	for (int i = 0; i < layer.size(); i++) {
		layer[i]->print_value();
	}
}


Matrix<double> Neural_Network::feedforward(const Matrix<double>& input) {
	if (input.get_row() != layer[0]->get_size() || input.get_column() != 1)
		throw std::runtime_error("invalid input Matrix");

	layer[0]->set_value(input);

	for (int i = 1; i < layer.size(); i++) {
		layer[i]->set_value(layer[i - 1]->feed());
	}
	
	return static_cast<Dense*>(layer.back())->get_value();
}

void Neural_Network::backpropagation(const Matrix<double>& target) {
	Matrix<double> output = static_cast<Dense*>(layer.back())->get_value();	

	std::vector<Matrix<double>> error;
	error.push_back(dloss_func(output, target));

	for (int i = layer.size() - 2; i >= 0; i--) {
		error = layer[i]->propagation(error);
	}
}


void Neural_Network::reconstruct(std::vector<LayerId> _layer,
	std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func) {
	loss_func = _loss_func;
	dloss_func = _dloss_func;

	layer.clear();
	std::size_t _Layer_size = _layer.size();

	for (int i = 0; i < _Layer_size - 1; i++) {
		switch (_layer[i].Layer_type) {
		case Layer::type::DENSE :
			layer.push_back(new Dense(_layer[i], _layer[i + 1].Layer_size));
			break;
		case Layer::type::LSTM :
			layer.push_back(new LSTM(_layer[i]));
			if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
				throw std::runtime_error("invalid size Layer afte lstm");
			break;
		case Layer::type::DROPOUT :
			layer.push_back(new DropOut(_layer[i]));
			if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
				throw std::runtime_error("invalid Layer size after Dropout");
			break;
		case Layer::type::FILTER :
			layer.push_back(new Filter(_layer[i]));
			if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
				throw std::runtime_error("invalid Layer size after Filter");
			break;
		case Layer::type::POWER :
			layer.push_back(new Power(_layer[i]));
			if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
				throw std::runtime_error("invalid layer size after Power");
			break;
		}
	}
	if (_layer[_Layer_size - 1].Layer_type != Layer::DENSE)
		throw "the output layer must be Dense layer";
	layer.push_back(new Dense(_layer[_Layer_size - 1].Layer_size));
}

void Neural_Network::reconstruct(const Neural_Network& copy) {
	loss_func = copy.loss_func;
	dloss_func = copy.dloss_func;

	layer = copy.layer;
}

void Neural_Network::reconstruct(Neural_Network&& move) {
	loss_func = move.loss_func;
	dloss_func = move.dloss_func;
	
	layer = std::move(move.layer);
}



void Neural_Network::forgot(const std::size_t& number) {																	// delete the past value memory
	for (int i = 0; i < layer.size() - 1; i++) {
		layer[i]->forgot(number);
	}
}

void Neural_Network::forgot_all() {																						// call forgot function with memory lenght
	forgot(layer[0]->v.size());
}



void Neural_Network::change_dependencies(const std::size_t& layer_number) {
	layer[layer_number]->change_dependencies();
}
void Neural_Network::change_dependencies() {
	for (int i = 0; i < layer.size() - 1; i++) {
		change_dependencies(i);
	}
}

void Neural_Network::set_change_dependencies(const std::size_t& layer_number, const double& number) {
	layer[layer_number]->set_change_dependencies(number);
}
void Neural_Network::set_change_dependencies(const double& number) {													// set changing value to specific value
	for (int i = 0; i < layer.size() - 1; i++) {
		set_change_dependencies(i, number);
	}
}

void Neural_Network::mul_change_dependencies(const std::size_t& layer_number, const double& number) {
	layer[layer_number]->mul_change_dependencies(number);
}
void Neural_Network::mul_change_dependencies(const double& number) {													// multiply changing value with specific value
	for (int i = 0; i < layer.size() - 1; i++) {
		mul_change_dependencies(i, number);
	}
}

bool Neural_Network::found_nan(const std::size_t& layer_number) {
	return layer[layer_number]->found_nan();
}
bool Neural_Network::found_nan() {
	for (int i = 0; i < layer.size() - 1; i++) {
		if (found_nan(i)) {
			return true;
		}
	}
	return true;
}


void Neural_Network::set_learning_rate(const std::size_t& layer_nnumber, const double& number) {
	layer[layer_nnumber]->set_learning_rate(number);
}
void Neural_Network::set_learning_rate(const double& number) {														// set every layer's learning rate to specific value
	for (int i = 0; i < layer.size(); i++) {
		set_learning_rate(i, number);
	}
}

void Neural_Network::set_do_record(const bool& value) {
	for (int i = 0; i < layer.size(); i++) {
		layer[i]->do_record = value;
	}
}

void Neural_Network::set_drop_out_rate(const std::size_t& layer_number, const double& number) {
	layer[layer_number]->set_drop_out_rate(number);
}
void Neural_Network::set_drop_out_rate(const double& number) {														// set every drop layer's drop out rate to specific layer
	for (int i = 0; i < layer.size(); i++) {
		set_drop_out_rate(i, number);
	}
}

std::size_t Neural_Network::get_layer_size() {
	return layer.size();
}

Matrix<double> Neural_Network::get_output() {
	return layer.back()->value;
}

std::size_t Neural_Network::get_input_size() {
	return layer[0]->get_size();
}

double Neural_Network::get_loss(const Matrix<double>& target) {
	return loss_func(layer.back()->value, target);
}

double Neural_Network::get_max_abs_change_dependencies(const std::size_t& layer_number) {
	return layer[layer_number]->get_max_abs_change_dependencies();
}
double Neural_Network::get_max_abs_change_dependencies() {
	double result = 0;
	for (int i = 0; i < layer.size() - 1; i++) {
		result = std::max(result, get_max_abs_change_dependencies(i));
	}

	return result;
}



void Neural_Network::save_as(std::ofstream& output_file) {
	for (int i = 0; i < layer.size() - 1; i++) {
		layer[i]->save_as(output_file);
	}
}

void Neural_Network::load(std::ifstream& input_file) {
	for (int i = 0; i < layer.size() - 1; i++) {
		layer[i]->load(input_file);
	}
}
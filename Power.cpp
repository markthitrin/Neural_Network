#pragma once

#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Power.h"
#include "Function.h"
#include "Variable.h"

Power::Power() { Layer_type = Layer::POWER; }

Power::Power(const std::size_t& size) {
	reconstruct(size);
}

Power::Power(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _act_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func) {
	reconstruct(size, _act_func, _dact_func);
}

Power::Power(const LayerId& set) {
	reconstruct(set);
}

Power::Power(const Power& copy) {
	reconstruct(copy);
}

Power::Power(Power&& move) {
	reconstruct(move);
}



Matrix<double> Power::feed() {																				// feedforward
	if (!power.is_constructed())
		throw std::runtime_error("Undefined power");

	if (do_record || v.size() == 0)
		v.push_back(value);
	else
		v[0] = value;

	return act_func(pow(value,power) + bias);
}

std::vector<Matrix<double>> Power::propagation(const std::vector<Matrix<double>>& gadient) {
	if (gadient.size() > v.size())
		throw std::runtime_error("invalid gadient size for  backpropagation");

	std::vector<Matrix<double>> value_change;
	std::vector<Matrix<double>> doutput;

	const std::size_t start_pos = v.size() - gadient.size();

	for (int round = 0; round < gadient.size(); round++) {
		doutput.push_back(dact_func(pow(v[round + start_pos],power) + bias, gadient[round]));
	}

	for (int round = 0; round < gadient.size(); round++) {
		for (int i = 0; i < power.get_row(); i++) {
			power_change[i][0] += doutput[round][i][0] * std::pow(v[round + start_pos][i][0], power[i][0]) * learning_rate;
		}
	}

	for (int round = 0; round < gadient.size(); round++) {
		for (int i = 0; i < bias.get_row(); i++) {
			bias_change[i][0] += doutput[round][i][0] * learning_rate;
		}
	}

	for (int round = 0; round < gadient.size(); round++) {
		value_change.push_back(Matrix<double>(value.get_row(), 1));
		value_change.back() = 0;

		for (int i = 0; i < value.get_row(); i++) {
			value_change.back()[i][0] += doutput[round][i][0] * power[i][0] * std::pow(v[round + start_pos][i][0], power[i][0] - 1);
		}
	}

	return value_change;
}



void Power::change_dependencies() {
	++t;
	switch (optimizer) {
	case SGD :
		power += power_change;
		bias += bias_change;
		break;
	case MOMENTUM :
		s_power_change = s_power_change * decay_rate + power_change * (double(1) - decay_rate);
		s_bias_change = s_bias_change * decay_rate + bias_change * (double(1) - decay_rate);

		power += s_power_change;
		bias += bias_change;
		break;
	case ADAM :
		s_power_change = s_power_change * decay_rate + power_change * (double(1) - decay_rate);
		s_bias_change = s_bias_change * decay_rate + bias_change * (double(1) - decay_rate);

		ss_power_change = ss_power_change * decay_rate + mul_each(power_change, power_change) * (double(1) - decay_rate);
		ss_bias_change = ss_bias_change * decay_rate + mul_each(bias_change, bias_change) * (double(1) - decay_rate);

		power += devide_each(s_power_change * learning_rate, (pow(ss_power_change, 0.5) + 0.000001));
		bias += devide_each(s_bias_change * learning_rate, (pow(ss_power_change, 0.5) + 0.000001));
		break;
	}
}

void Power::set_change_dependencies(const double& value) {														// set changing weight and chaing bias to specifc value
	power_change = value;
	bias_change = value;
}

void Power::mul_change_dependencies(const double& value) {														// multiply changing weight and ching bias with specific value
	power_change *= value;
	bias_change *= value;
}

void Power::map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func) {
	power_change = func(power_change);
	bias_change = func(bias_change);
}

bool Power::found_nan() {
	return power != power || bias != bias;
}



void Power::reconstruct(const std::size_t& size) {
	reconstruct(size, sigmoid_func, dsigmoid_func);
}

void Power::reconstruct(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _act_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func) {
	Layer::reconstruct(size);

	Layer::Layer_type = Layer::type::POWER;

	power.reconstruct(size, 1);
	bias.reconstruct(size, 1);

	power_change.reconstruct(size, 1);
	bias_change.reconstruct(size, 1);

	s_power_change.reconstruct(0, 0);
	s_bias_change.reconstruct(0, 0);

	ss_power_change.reconstruct(0, 0);
	ss_bias_change.reconstruct(0, 0);


	act_func = _act_func;
	dact_func = _dact_func;
}

void Power::reconstruct(const LayerId& set) {
	reconstruct(set.Layer_size, sigmoid_func, dsigmoid_func);

	set_Layer(set.setting);
}

void Power::reconstruct(const Power& copy) {
	Layer::reconstruct(copy);

	power.reconstruct(copy.power);
	bias.reconstruct(copy.bias);

	s_power_change.reconstruct(copy.s_power_change);
	s_bias_change.reconstruct(copy.s_bias_change);

	ss_power_change.reconstruct(copy.ss_power_change);
	ss_bias_change.reconstruct(copy.ss_bias_change);

	act_func = copy.act_func;
	dact_func = copy.dact_func;
}

void Power::reconstruct(Power&& move) {
	Layer::reconstruct(std::move(move));

	power.reconstruct(std::move(move.power));
	bias.reconstruct(std::move(move.bias));

	s_power_change.reconstruct(std::move(move.s_power_change));
	s_bias_change.reconstruct(std::move(move.s_bias_change));

	ss_power_change.reconstruct(std::move(move.ss_power_change));
	ss_bias_change.reconstruct(std::move(move.ss_bias_change));

	act_func = std::move(move.act_func);
	dact_func = std::move(move.dact_func);
}


void Power::set_weight(const int& weight_type, const Matrix<double>& _weight) {
	switch (weight_type) {
	case weight_type::POWER :
		power = _weight;
		break;
	default :
		throw std::runtime_error("Invalid weight_type to be set");
		break;
	}
}

void Power::set_bias(const int& bias_type, const Matrix<double>& _bias) {
	switch (bias_type) {
	case bias_type::BIAS :
		bias = _bias;
		break;
	default:
		throw std::runtime_error("Invalid bias_type to be set");
		break;
	}
}

void Power::rand_weight(const double& min, const double& max) {
	if (!power.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < power.get_row(); i++) {
		power[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
	}
}

void Power::rand_weight(std::pair<const double&, const double&> setting) {
	if (!power.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < power.get_row(); i++) {
		power[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
	}
}

void Power::rand_weight(std::function<double()> func) {
	if (!power.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < power.get_row(); i++) {
		power[i][0] = func();
	}
}

void Power::rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
	if (!power.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < power.get_row(); i++) {
		power[i][0] = func(value.get_row(), next);
	}
}

void Power::rand_bias(const double& min, const double& max) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
	}
}

void Power::rand_bias(std::pair<const double&, const double&> setting) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
	}
}

void Power::rand_bias(std::function<double()> func) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = func();
	}
}

void Power::rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = func(value.get_row(), next);
	}
}



void Power::print_weight() {
	std::cout << "---------Power Layer----------\n";
	for (int i = 0; i < power.get_row(); i++) {
		std::cout << power[i][0] << "\n";
	}
}

void Power::print_value() {
	std::cout << "---------Power Layer----------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << value[i][0] << "    \t";
	}std::cout << std::endl;
}

void Power::print_bias() {
	std::cout << "---------Poower Layer---------\n";
	for (int i = 0; i < bias.get_row(); i++) {
		std::cout << bias[i][0] << "    \t";
	}std::cout << std::endl;
}



std::function<Matrix<double>(const Matrix<double>&)> Power::get_act_func() {
	return act_func;
}

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> Power::get_dact_func() {
	return dact_func;
}

double Power::get_max_abs_change_dependencies() {
	double result = 0;
	auto get_max_abs_matrix_value = [](const Matrix<double>& M) {
		if (!M.is_constructed()) {
			return 0.0;
		}
		double result = 0;
		for (int i = 0; i < M.get_row(); i++) {
			for (int j = 0; j < M.get_column(); j++) {
				result = std::max(result, std::abs(M[i][j]));
			}
		}
		return result;
	};

	result = std::max(result, get_max_abs_matrix_value(power_change));
	result = std::max(result, get_max_abs_matrix_value(bias_change));

	result = std::max(result, get_max_abs_matrix_value(s_power_change));
	result = std::max(result, get_max_abs_matrix_value(s_bias_change));

	result = std::max(result, get_max_abs_matrix_value(ss_power_change));
	result = std::max(result, get_max_abs_matrix_value(ss_bias_change));

	return result;
}



void Power::save_as(std::ofstream& output_file) {
	power.save_as(output_file);
	bias.save_as(output_file);
}

void Power::load(std::ifstream& input_file) {
	power.load(input_file);
	bias.load(input_file);
}


void Power::set_optimizer(const Layer::opt& _optimizer) {
	switch(optimizer){
	case Layer::opt::SGD: break;
	case Layer::opt::MOMENTUM:
		s_power_change.reconstruct(0, 0);

		s_bias_change.reconstruct(0, 0);
		break;
	case Layer::opt::ADAM:
		s_power_change.reconstruct(0, 0);
		ss_power_change.reconstruct(0, 0);

		s_bias_change.reconstruct(0, 0);
		ss_bias_change.reconstruct(0, 0);
		break;
	}

	optimizer = _optimizer;
	switch (optimizer) {
	case Layer::opt::SGD: break;
	case Layer::opt::MOMENTUM:
		s_power_change.reconstruct(power_change.get_row(), power_change.get_column());
		s_power_change *= 0;

		s_bias_change.reconstruct(bias_change.get_row(), bias.get_column());
		s_bias_change *= 0;
		break;
	case Layer::opt::ADAM:
		s_power_change.reconstruct(power_change.get_row(), power_change.get_column());
		ss_power_change.reconstruct(power_change.get_row(), power_change.get_column());
		s_power_change *= 0;
		ss_power_change *= 0;

		s_bias_change.reconstruct(bias_change.get_row(), bias_change.get_column());
		ss_bias_change.reconstruct(bias_change.get_row(), bias_change.get_column());
		s_bias_change *= 0;
		ss_bias_change *= 0;
		break;
	}
}



void Power::set_Layer(const std::string& setting) {															// set the layer using command
	int size = setting.size();
	int i = 0;
	
	auto set_optimizer_text = [&]() {
		std::string _optimizer = get_text(setting, i);
		if (_optimizer == "SGD") 
			set_optimizer(Layer::SGD);
		else if (_optimizer == "MOMENTUM") 
			set_optimizer(Layer::MOMENTUM);
		else if (_optimizer == "ADAM") 
			set_optimizer(Layer::ADAM);
	};
	auto set_learning_rate_text = [&]() {
		double a = get_number(setting, i);
		set_learning_rate(a);
	};


	while (i < size) {
		std::string command = get_text(setting, i);
		if (command == "act")
			universal_set_func(act_func, setting, i);
		else if (command == "dact")
			universal_set_func(dact_func, setting, i);
		else if (command == "learning_rate")
			set_learning_rate_text();
		else if (command == "optimizer")
			set_optimizer_text();
		else if (command == "")
			;
		else throw std::runtime_error("command not found");
	}
}

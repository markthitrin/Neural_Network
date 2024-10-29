#pragma once

#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Dense.h"
#include "Function.h"
#include "Variable.h"


Dense::Dense() { Layer_type = Layer::DENSE; }

Dense::Dense(const std::size_t& size) {
	reconstruct(size);
}

Dense::Dense(const std::size_t& size, const std::size_t& next, 
	std::function<Matrix<double>(const Matrix<double>&)> _act_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func) {
	reconstruct(size, next, act_func, dact_func);
}

Dense::Dense(const LayerId& set,const std::size_t& next) {
	reconstruct(set, next);
}

Dense::Dense(const Dense& copy) {
	reconstruct(copy);
}

Dense::Dense(Dense&& move) {
	reconstruct(std::move(move));
}



Matrix<double> Dense::feed()  {
	if (!weight.is_constructed())
		throw std::runtime_error("Undefined weight");
	
	if (do_record || v.size() == 0)
		v.push_back(value);
	else
		v[0] = value;

	return act_func((weight * value) + bias);
}

std::vector<Matrix<double>> Dense::propagation(const std::vector<Matrix<double>>& gadient)  {
	if (gadient.size() > v.size())
		throw std::runtime_error("invalid gadient size for  backpropagation");

	std::vector<Matrix<double>> value_change;
	std::vector<Matrix<double>> doutput;

	const std::size_t start_pos = v.size() - gadient.size();

	for (int round = 0; round < gadient.size(); round++) {
		doutput.push_back(dact_func((weight * v[round + start_pos]) + bias, gadient[round]));
	}

	for (int round = 0; round < gadient.size(); round++) {
		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				weight_change[i][j] += doutput[round][i][0] * v[round + start_pos][j][0] * learning_rate;
			}
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

		for (int i = 0; i < weight.get_row(); i++) {
			for (int j = 0; j < weight.get_column(); j++) {
				value_change.back()[j][0] += doutput[round][i][0] * weight[i][j];
			}
		}
	}

	return value_change;
}



void Dense::change_dependencies() {
	++t;
	switch (optimizer) {
	case Layer::opt::SGD :
		weight += weight_change;
		bias += bias_change;
		break;
	case Layer::opt::MOMENTUM:
		s_weight_change = s_weight_change * decay_rate + weight_change * (double(1) - decay_rate);
		s_bias_change = s_bias_change * decay_rate + bias_change * (double(1) - decay_rate);

		weight += s_weight_change;
		bias += bias_change;
		break;
	case Layer::opt::ADAM:
		s_weight_change = s_weight_change * decay_rate + weight_change * (double(1) - decay_rate);
		s_bias_change = s_bias_change * decay_rate + bias_change * (double(1) - decay_rate);

		ss_weight_change = ss_weight_change * decay_rate + mul_each(weight_change, weight_change) * (double(1) - decay_rate);
		ss_bias_change = ss_bias_change * decay_rate + mul_each(bias_change, bias_change) * (double(1) - decay_rate);

		weight += devide_each(s_weight_change * learning_rate, (pow(ss_weight_change, 0.5) + 0.000001));
		bias += devide_each(s_bias_change * learning_rate, (pow(ss_weight_change, 0.5) + 0.000001));
		break;
	}
}

void Dense::set_change_dependencies(const double& value) {
	weight_change = value;
	bias_change = value;
}

void Dense::mul_change_dependencies(const double& value) {
	weight_change *= value;
	bias_change *= value;
}

void Dense::map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func) {
	weight_change = func(weight_change);
	bias_change = func(bias_change);
}

bool Dense::found_nan() {
	return weight != weight || bias != bias;
}
	


void Dense::reconstruct(const std::size_t& size) {
	reconstruct(size, 0, sigmoid_func, dsigmoid_func);
}

void Dense::reconstruct(const std::size_t& size, const std::size_t& next,
	std::function<Matrix<double>(const Matrix<double>&)> _act_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dact_func) {
	Layer::reconstruct(size);

	Layer::Layer_type = Layer::type::DENSE;

	weight.reconstruct(next, size);
	bias.reconstruct(next, 1);

	weight_change.reconstruct(next, size);
	bias_change.reconstruct(next, 1);

	s_weight_change.reconstruct(0, 0);
	s_bias_change.reconstruct(0, 0);

	ss_weight_change.reconstruct(0, 0);
	ss_bias_change.reconstruct(0, 0);

	act_func = _act_func;
	dact_func = _dact_func;
}

void Dense::reconstruct(const LayerId& set,const size_t& next) {
	reconstruct(set.Layer_size, next, sigmoid_func, dsigmoid_func);

	set_Layer(set.setting);
}

void Dense::reconstruct(const Dense& copy) {
	Layer::reconstruct(copy);

	weight = copy.weight;
	bias = copy.bias;

	weight_change = copy.weight_change;
	bias_change = copy.bias_change;

	s_weight_change = copy.s_weight_change;
	s_bias_change = copy.s_bias_change;

	ss_weight_change = copy.ss_weight_change;
	ss_bias_change = copy.ss_bias_change;

	act_func = copy.act_func;
	dact_func = copy.dact_func;
}

void Dense::reconstruct(Dense&& move) {
	Layer::reconstruct(std::move(move));

	weight = std::move(move.weight);
	bias = std::move(move.bias);

	weight_change = std::move(move.weight_change);
	bias_change = std::move(move.bias_change);

	s_weight_change = std::move(move.s_weight_change);
	s_bias_change = std::move(move.s_bias_change);

	ss_weight_change = std::move(move.ss_weight_change);
	ss_bias_change = std::move(move.ss_bias_change);

	act_func = move.act_func;
	dact_func = move.dact_func;
}



void Dense::set_weight(const int& weight_type, const Matrix<double>& _weight) {
	switch (weight_type) {
	case WEIGHT :
		weight = _weight;
		break;
	default:
		throw std::runtime_error("Invalid weight_type to be set");
		break;
	}
}

void Dense::set_bias(const int& bias_type, const Matrix<double>& _bias) {
	switch (bias_type) {
	case BIAS :
		bias = _bias;
		break;
	default:
		throw std::runtime_error("Invalid bias_number to be set");
		break;
	}
}

void Dense::rand_weight(const double& min, const double& max) {
	if (!weight.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < weight.get_row(); i++) {
		for (int j = 0; j < weight.get_column(); j++) {
			weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
		}
	}
}

void Dense::rand_weight(std::pair<const double&, const double&> setting) {
	if (!weight.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < weight.get_row(); i++){
		for (int j = 0; j < weight.get_column(); j++) {
			weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		}
	}
}

void Dense::rand_weight(std::function<double()> func) {
	if (!weight.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < weight.get_row(); i++) {
		for (int j = 0; j < weight.get_column(); j++) {
			weight[i][j] = func();
		}
	}
}

void Dense::rand_weight(std::function<double(std::size_t,std::size_t)> func,const std::size_t& next) {
	if (!weight.is_constructed())
		throw std::runtime_error("cant set undefined weight value");

	for (int i = 0; i < weight.get_row(); i++) {
		for (int j = 0; j < weight.get_column(); j++) {
			weight[i][j] = func(value.get_row(), next);
		}
	}
}

void Dense::rand_bias(const double& min, const double& max) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
	}
}

void Dense::rand_bias(std::pair<const double&, const double&> setting) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
	}
}

void Dense::rand_bias(std::function<double()> func) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = func();
	}
}

void Dense::rand_bias(std::function<double(std::size_t,std::size_t)> func,const std::size_t& next) {
	if (!bias.is_constructed())
		throw std::runtime_error("cant set undefined bias value");

	for (int i = 0; i < bias.get_row(); i++) {
		bias[i][0] = func(value.get_row(), next);
	}
}



void Dense::print_weight() {
	std::cout << "---------Dense Layer----------\n";
	for (int i = 0; i < weight.get_row(); i++) {
		for (int j = 0; j < weight.get_column(); j++) {
			std::cout << weight[i][j] << "    \t";
		}std::cout << std::endl;
	}
}

void Dense::print_value() {
	std::cout << "---------Dense Layer----------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << value[i][0] << "    \t";
	}std::cout << std::endl;
}

void Dense::print_bias(){
	std::cout << "---------Dense Layer---------\n";
	for (int i = 0; i < bias.get_row(); i++) {
		std::cout << bias[i][0] << "    \t";
	}std::cout << std::endl;
}



std::function<Matrix<double>(const Matrix<double>&)> Dense::get_act_func() {
	return act_func;
}

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> Dense::get_dact_func() {
	return dact_func;
}

double Dense::get_max_abs_change_dependencies() {
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

	result = std::max(result, get_max_abs_matrix_value(weight_change));
	result = std::max(result, get_max_abs_matrix_value(bias_change));

	result = std::max(result, get_max_abs_matrix_value(s_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_bias_change));

	result = std::max(result, get_max_abs_matrix_value(ss_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_bias_change));

	return result;
}
	


void Dense::save_as(std::ofstream& output_file) {
	weight.save_as(output_file);
	bias.save_as(output_file);
}

void Dense::load(std::ifstream& input_file) {
	weight.load(input_file);
	bias.load(input_file);
}



void Dense::set_optimizer(const Layer::opt _opt) {
	switch (optimizer) {
	case Layer::opt::SGD: break;
	case Layer::opt::MOMENTUM:
		s_weight_change.reconstruct(0, 0);

		s_bias_change.reconstruct(0, 0);
		break;
	case Layer::opt::ADAM:
		s_weight_change.reconstruct(0, 0);
		ss_weight_change.reconstruct(0, 0);

		s_bias_change.reconstruct(0, 0);
		ss_bias_change.reconstruct(0, 0);
		break;
	}

	optimizer = _opt;
	switch (optimizer) {
	case Layer::opt::SGD: break;
	case Layer::opt::MOMENTUM:
		s_weight_change.reconstruct(weight_change.get_row(),weight_change.get_column());
		s_weight_change *= 0;

		s_bias_change.reconstruct(bias_change.get_row(),bias.get_column());
		s_bias_change *= 0;
		break;
	case Layer::opt::ADAM:
		s_weight_change.reconstruct(weight_change.get_row(),weight_change.get_column());
		ss_weight_change.reconstruct(weight_change.get_row(),weight_change.get_column());
		s_weight_change *= 0;
		ss_weight_change *= 0;

		s_bias_change.reconstruct(bias_change.get_row(), bias_change.get_column());
		ss_bias_change.reconstruct(bias_change.get_row(), bias_change.get_column());
		s_bias_change *= 0;
		ss_bias_change *= 0;
		break;
	}
}



void Dense::set_Layer(const std::string& setting) {
	int size = setting.size();
	int i = 0;
	
	auto set_optimizer_text = [&]() {
		std::string _optimizer = get_text(setting, i);
		if (_optimizer == "SGD")
			set_optimizer(Layer::opt::SGD);
		else if (_optimizer == "MOMENTUM")
			set_optimizer(Layer::opt::MOMENTUM);
		else if (_optimizer == "ADAM")
			set_optimizer(Layer::opt::ADAM);
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
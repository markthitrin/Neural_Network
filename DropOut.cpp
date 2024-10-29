#pragma once
#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "DropOut.h"
#include "Function.h"
#include "Variable.h"


DropOut::DropOut () { ; };

DropOut::DropOut(const std::size_t& size) {
	reconstruct(size);
}

DropOut::DropOut(const std::size_t& size,
	std::function<double()> _rand_func) {
	reconstruct(size, _rand_func);
}

DropOut::DropOut(const LayerId& set) {
	reconstruct(set);
}

DropOut::DropOut(const DropOut& copy) {
	reconstruct(copy);
}

DropOut::DropOut(DropOut&& move) {
	reconstruct(std::move(move));
}



Matrix<double> DropOut::feed() {
	Matrix<double> filter(value.get_row(), 1);

	for (int i = 0; i < value.get_row(); i++) {
		filter[i][0] = rand_func() < drop_out_rate ? 0 : 1;
	}

	if (do_record || v.size() == 0)
		v.push_back(filter);
	else
		v[0] = filter;
	
	return mul_each(value, filter);
}

std::vector<Matrix<double>> DropOut::propagation(const std::vector<Matrix<double>>& gadient) {
	std::vector<Matrix<double>> result;

	int start_pos = v.size() - gadient.size();

	for (int i = 0; i < gadient.size(); i++) {	
		result.push_back(Matrix<double>(gadient[i].get_row(), 1));

		for (int j = 0; j < gadient[i].get_row(); j++) {
			for (int k = 0; k < gadient[i].get_column(); k++) {
				result[i][j][k] = gadient[i][j][k] * v[start_pos + i][j][k];
			}
		}
	}
	return result;
}



void DropOut::set_drop_out_rate(const double& number) {
	drop_out_rate = number;
}

void DropOut::set_rand_func(std::function<double()> _rand_func) {
	rand_func = _rand_func;
}



void DropOut::reconstruct(const std::size_t& size) {
	reconstruct(size, []() {return double(rand() % 10000) / 10000; });
}

void DropOut::reconstruct(const std::size_t& size, std::function<double()> _rand_func) {
	Layer::reconstruct(size);

	Layer::Layer_type = Layer::type::DROPOUT;

	drop_out_rate = 0.1;
	rand_func = _rand_func;
}

void DropOut::reconstruct(const LayerId& set) {
	Layer::reconstruct(set.Layer_size);

	Layer::Layer_type = DROPOUT;

	rand_func = []() {return double(rand() % 10000) / 10000; };
	
	set_layer(set.setting);
}

void DropOut::reconstruct(const DropOut& copy) {
	Layer::reconstruct(copy);

	rand_func = copy.rand_func;
	drop_out_rate = copy.drop_out_rate;
}

void DropOut::reconstruct(DropOut&& move) {
	Layer::reconstruct(std::move(move));

	rand_func = move.rand_func;
	drop_out_rate = move.drop_out_rate;
}


void DropOut::print_weight() {
	std::cout << "---------DropOut Layer----------\n";
	std::cout << "Drop out rate : " << drop_out_rate << "\n";
}

void DropOut::print_bias() {
	std::cout << "---------DropOut Layer----------\n";
}

void DropOut::print_value() {
	std::cout << "---------DropOut Layer----------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << value[i][0] << "    \t";
	}std::cout << std::endl;
}



std::function<double()> DropOut::get_rand_func() {
	return rand_func;
}

double DropOut::get_drop_out_rate() {
	return drop_out_rate;
}



void DropOut::set_layer(const std::string& setting) {															// set the layer using command
	int size = setting.size();
	int i = 0;

	auto set_rand_func_text = [&]() {
		std::string a = get_text(setting, i);
		if (a == "normal")
			rand_func = normal_rand_func;
		else throw std::runtime_error("function not found");
	};
	auto set_drop_out_rate_text = [&]() {
		double a = get_number(setting, i);
		drop_out_rate = a;
	};

	while (i < size) {
		std::string command = get_text(setting, i);
		if (command == "rand")
			set_rand_func_text();
		else if (command == "drop_out_rate")
			set_drop_out_rate_text();
		else if (command == "")
			;
		else throw std::runtime_error("command not found");
	}
}

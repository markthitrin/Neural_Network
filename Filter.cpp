#pragma once
#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Filter.h"
#include "Function.h"
#include "Variable.h"


Filter::Filter() { ; };

Filter::Filter(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc) {
	reconstruct(size, func, dfunc);
}

Filter::Filter(const LayerId& set) {
	reconstruct(set);
}

Filter::Filter(const Filter& copy) {
	reconstruct(copy);
}

Filter::Filter(Filter&& move) {
	reconstruct(std::move(move));
}



Matrix<double> Filter::feed() {											
	if (do_record || v.size() == 0)
		v.push_back(value);
	else
		v[0] = value;													

	return func(value);																					// return output
}

std::vector<Matrix<double>> Filter::propagation(const std::vector<Matrix<double>>& gadient) {
	int start_pos = v.size() - gadient.size();

	std::vector<Matrix<double>> result;

	for (int round = 0; round < gadient.size(); round++) {
		result.push_back(dfunc(v[round + start_pos], gadient[round]));
	}
	return result;
}



void Filter::reconstruct(const std::size_t& size) {
	reconstruct(size, descale_func, ddescale_func);
}

void Filter::reconstruct(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc) {
	Layer::reconstruct(size);

	Layer::Layer_type = Layer::type::FILTER;

	func = _func;
	dfunc = _dfunc;
}

void Filter::reconstruct(const LayerId& set) {
	reconstruct(set.Layer_size, descale_func, ddescale_func);

	set_Layer(set.setting);
}

void Filter::reconstruct(const Filter& copy) {
	Layer::reconstruct(copy);

	func = copy.func;
	dfunc = copy.dfunc;
}

void Filter::reconstruct(Filter&& move) {
	Layer::reconstruct(move);

	func = move.func;
	dfunc = move.dfunc;
}



void Filter::print_weight() {
	std::cout << "---------Filter Layer----------\n";
}

void Filter::print_bias() {
	std::cout << "---------Filter Layer----------\n";
}

void Filter::print_value() {
	std::cout << "---------Filter Layer----------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << value[i][0] << "    \t";
	}std::cout << std::endl;
}



void Filter::set_Layer(const std::string& setting) {															// set layer using command
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
		else throw std::runtime_error("command not found");
	}
}
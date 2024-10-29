#pragma once
#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Function.h"
#include "Variable.h"


class Filter : public Layer {
public:
	Filter();

	Filter(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _func = descale_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc = ddescale_func);

	Filter(const LayerId& set);

	Filter(const Filter& copy);

	Filter(Filter&& move);



	Matrix<double> feed();

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient);



	void reconstruct(const std::size_t& size);

	void reconstruct(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dfunc);

	void reconstruct(const LayerId& set);

	void reconstruct(const Filter& copy);

	void reconstruct(Filter&& move);



	void print_weight();

	void print_bias();

	void print_value();
private:
	void set_Layer(const std::string& setting);



	std::function<Matrix<double>(const Matrix<double>&)> func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dfunc;
};
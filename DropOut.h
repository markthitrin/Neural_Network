#pragma once
#pragma once
#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Function.h"
#include "Variable.h"



class DropOut : public Layer {
public:
	DropOut();

	DropOut(const std::size_t& size);

	DropOut(const std::size_t& size,
		std::function<double()> _rand_func);

	DropOut(const LayerId& set);

	DropOut(const DropOut& copy);

	DropOut(DropOut&& move);



	Matrix<double> feed();

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient);



	void set_drop_out_rate(const double& number);

	void set_rand_func(std::function<double()> _rand_func);



	void reconstruct(const std::size_t& size);

	void reconstruct(const std::size_t& size, std::function<double()> _rand_func);

	void reconstruct(const LayerId& set);

	void reconstruct(const DropOut& copy);

	void reconstruct(DropOut&& move);


	void print_weight();

	void print_bias();

	void print_value();



	std::function<double()> get_rand_func();

	double get_drop_out_rate();

private:
	void set_layer(const std::string& setting);



	std::function<double()> rand_func;
	double drop_out_rate = 0.1;
};
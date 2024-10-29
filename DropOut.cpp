#pragma once
#include "Header.h"
#include "Layer.cpp"
#include "LayerId.cpp"

#include "Func.h"



class DropOut : public Layer {
public:
	DropOut () { ; };

	DropOut(const std::size_t& size,
		std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) {
		Layer_type = DROPOUT;

		value.reconstruct(size, 1);

		rand_func = _rand_func;
	}

	DropOut(const LayerId& set) {
		Layer_type = DROPOUT;

		value.reconstruct(set.Layer_size, 1);

		rand_func = []() {return double(rand() & 10000) / 10000; };
		set_layer(set.setting);
	}



	Matrix<double> feed() {																						// feedforward
		Matrix<double> filter(value.get_row(),1);																// create Matrix containing filter				

		for (int i = 0; i < value.get_row(); i++) {
			filter[i][0] = rand_func() < drop_out_rate ? 0 : 1;													// randomly put 0 or 1 into filter
		}
		v.push_back(filter);																					// remember the filter
		return mul_each(value,filter);																			// return Matrix for each value[i][j] if filter[i][j] = 1, otherwise 0
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) {						// backpropagation
		std::vector<Matrix<double>> result;																		// the output of this function(flow gadient)
		
		int start_pos = v.size() - gadient.size();																// rearrange in case givven gadient is shorter than layer's memory

		for (int i = 0; i < gadient.size(); i++) {																// loop though every time step
			result.push_back(Matrix<double>(gadient[i].get_row(),1));

			for (int j = 0; j < gadient[i].get_row(); j++) {													// loop though every gadient
				for (int k = 0; k < gadient[i].get_column(); k++) {
					result[i][j][k] = gadient[i][j][k] * v[start_pos + i][j][k];								// compute flow gadient
				}
			}
		}
		return result;
	}



	void forgot(const std::size_t& number) {																		// delete old memory and shift the new memory
		int h = number;
		if (number > v.size())
			h = v.size();
		for (int i = 0; i < v.size() - h; i++) {
			v[i] = v[i + h];
		}
		for (int i = 0; i < h; i++) {
			v.pop_back();
		}
	}

	void forgot_all() {																							// delete all memory
		forgot(v.size());
	}



	void change_dependencies() {
		
	}

	void set_change_dependencies(const double& number) {

	}

	void mul_change_dependencies(const double& number) {

	}

	void set_drop_out_rate(const double& number) {
		drop_out_rate = number;
	}

	void set_rand_func(std::function<double()> _rand_func) {
		rand_func = _rand_func;
	}



	void reconstruct(const std::size_t& size, std::function<double()> _rand_func = []() {return double(rand() % 10000) / 10000; }) {
		forgot_all();
		
		value.reconstruct(size, 1);

		rand_func = _rand_func;
	}

	void reconstruct(const LayerId& set) {
		Layer_type = DROPOUT;

		value.reconstruct(set.Layer_size, 1);

		rand_func = []() {return double(rand() & 10000) / 10000; };
		set_layer(set.setting);
	}



	void rand_weight(const double& min, const double& max) {
		
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		
	}

	void rand_weight(std::function<double()> func) {
	
	}

	void rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {

	}

	void rand_bias(const double& min, const double& max) {
		
	}

	void rand_bias(std::pair<const double&, const double&> setting) {
		
	}

	void rand_bias(std::function<double()> func) {
		
	}

	void rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
		
	}



	void print_value() {
		std::cout << "---------DropOut Layer----------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << value[i][0] << "    \t";
		}std::cout << std::endl;
	}



	std::function<double()> get_rand_func() {
		return rand_func;
	}

	double get_drop_out_rate() {
		return drop_out_rate;
	}

private:
	void set_layer(const std::string& setting) {															// set the layer using command
		int size = setting.size();
		int i = 0;
		while (i < size) {
			std::string command = get_text(setting, i);
			if (command == "rand")
				set_rand_func(setting, i);
			else if (command == "drop_out_rate")
				set_drop_out_rate(setting, i);
			else if (command == "")
				;
			else throw "command not found";
		}
	}

	void set_rand_func(const std::string& str, int& i) {													// set rand function usnig command
		std::string a = get_text(str, i);
		if (a == "normal")
			rand_func = normal_rand_func;
		else throw "function not found";
	}

	void set_drop_out_rate(const std::string& str, int& i) {												// set drop out rate using command
		double a = get_number(str, i);
		drop_out_rate = a;
	}



	std::function<double()> rand_func;																		// containing random function
	double drop_out_rate = 0.1;																				// dropout rate = ( 0 , 1 )
};
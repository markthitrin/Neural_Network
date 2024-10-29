#pragma once

#include "Header.h"
#include "Dense.cpp"
#include "LSTM.cpp"
#include "LayerId.cpp"
#include "DropOut.cpp"
#include "Filter.cpp"

class Neural_Network {
public:
	Neural_Network() {}

	Neural_Network(std::vector<LayerId> _layer,
		std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func) :
		loss_func(_loss_func), dloss_func(_dloss_func) {

		std::size_t _Layer_size = _layer.size();

		for (int i = 0; i < _Layer_size - 1; i++) {
			if (_layer[i].Layer_type == Layer::type::DENSE) {
				layer.push_back(new Dense(_layer[i],_layer[i + 1].Layer_size));
			}
			else if (_layer[i].Layer_type == Layer::type::LSTM) {
				layer.push_back(new LSTM(_layer[i]));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid size Layer afte lstm";
			}
			else if (_layer[i].Layer_type == Layer::type::DROPOUT) {
				layer.push_back(new DropOut(_layer[i]));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid Layer size after Dropout";
			}
			else if (_layer[i].Layer_type == Layer::type::FILTER) {
				layer.push_back(new Filter(_layer[i]));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid Layer size after Filter";
			}
		}
		if (_layer[_Layer_size - 1].Layer_type != Layer::DENSE)
			throw "the output layer must be Dense layer";
		layer.push_back(new Dense(_layer[_Layer_size - 1].Layer_size));
	}

	Neural_Network(const Neural_Network& copy) { // do not use, not finish dont use it
		for (int i = 0; i < copy.layer.size() - 1; i++) {
			if (copy.layer[i]->get_type() == Layer::DENSE) {
				Dense* _layer = static_cast<Dense*>(copy.layer[i]);
				Dense* _next_layer = static_cast<Dense*>(copy.layer[i + 1]);
				layer.push_back(new Dense(_layer->get_size(), _next_layer->get_size(),
					_layer->get_act_func(), _layer->get_dact_func()));
			}
			else if (copy.layer[i]->get_type() == Layer::LSTM) {
				LSTM* _layer = static_cast<LSTM*>(copy.layer[i]);
				layer.push_back(new LSTM(_layer->get_size()));
			}
			else if (copy.layer[i]->get_type() == Layer::DROPOUT) {
				DropOut* _layer = static_cast<DropOut*>(copy.layer[i]);
				layer.push_back(new DropOut(_layer->get_size(), _layer->get_rand_func()));
			}
			else if (copy.layer[i]->get_type() == Layer::FILTER) {
				Filter* _layer = static_cast<Filter*>(copy.layer[i]);
				layer.push_back(new Filter(_layer->get_size()));
			}
		}
		layer.push_back(new Dense(copy.layer.back()->get_size()));
	}



	void rand_weight(const std::vector<std::pair<double, double>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "Invalid random weight value";

		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_weight(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_weight(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// drop out layer dotn have weight
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
		}
	}

	void rand_weight(const std::vector<std::function<double()>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "Invalid random weight value";

		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_weight(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_weight(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have weight
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
		}
	}

	void rand_weight(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "Invalid random weight value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_weight(setting[i], layer[i + 1]->value.get_row());
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_weight(setting[i], layer[i + 1]->value.get_row());
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have weight
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
		}
	}

	void rand_bias(const std::vector<std::pair<double, double>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "invalid random bias value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_bias(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_bias(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have bias
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
		}
	}

	void rand_bias(const std::vector<std::function<double()>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "invalid random bias value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_bias(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_bias(setting[i]);
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have bias
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
		}
	}

	void rand_bias(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "invalid random bias value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_bias(setting[i], layer[i + 1]->value.get_row());
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_bias(setting[i], layer[i + 1]->value.get_row());
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have bias
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
		}
	}



	void print_weight() {
		std::cout << "======== weight ========\n";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_weight();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->print_weight();
			else if (layer[i]->get_type() == Layer::DROPOUT)
				std::cout << "Drop Out : " << static_cast<DropOut*>(layer[i])->get_drop_out_rate() << std::endl;
			else if (layer[i]->Layer_type == Layer::FILTER)
				std::cout << "Filter layer\n";
		}
	}

	void print_value() {
		std::cout << "======== value =========\n";
		for (int i = 0; i < layer.size(); i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_value();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->print_value();
			else if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->print_value();
			else if (layer[i]->Layer_type == Layer::FILTER)
				static_cast<Filter*>(layer[i])->print_value();
		}
	}

	void print_bias() {
		std::cout << "========= bias ==========\n";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->print_bias();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->print_bias();
			else if (layer[i]->get_type() == Layer::DROPOUT)
				std::cout << "Drop Out : " << static_cast<DropOut*>(layer[i])->get_drop_out_rate() << std::endl;
			else if (layer[i]->Layer_type == Layer::FILTER)
				std::cout << "Filter layer\n";
		}
	}



	Matrix<double> feedforward(Matrix<double> input) { 
		if (input.get_row() != layer[0]->get_size() || input.get_column() != 1)
			throw "invalid input Matrix";

		layer[0]->set_value(input);																			// set input into the first layer

		for (int i = 1; i < layer.size(); i++) {															// loop though every layer
			layer[i]->set_value(layer[i - 1]->feed());														// get output from previous layer then set into the next layer
		}
		return static_cast<Dense*>(layer.back())->get_value();												// get output from last layer
	}

	void backpropagation(Matrix<double> target) {
		Matrix<double> output = static_cast<Dense*>(layer.back())->get_value();								// get output form last layer

		std::vector<Matrix<double>> error;
		error.push_back(dloss_func(output, target));														// compute the gadient from output and target value

		for (int i = layer.size() - 2; i >= 0; i--) {														// loop though every layer (from the back)
			error = layer[i]->propagation(error);															// get gadient flow from back layer and set into vector error
		}
	}



	void forgot(const std::size_t& number) {																	// delete the past value memory
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->forgot(number);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->forgot(number);
			else if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->forgot(number);
			else if (layer[i]->get_type() == Layer::FILTER)
				static_cast<Filter*>(layer[i])-> forgot(number);
		}
	}

	void forgot_all() {																						// call forgot function with memory lenght
		forgot(layer[0]->v.size());
	}



	void change_dependencies() {																			// change weight and bias with computed changeing value
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->change_dependencies();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->change_dependencies();
			else if (layer[i]->get_type() == Layer::DROPOUT)
				;
			else if (layer[i]->get_type() == Layer::FILTER)
				;
		}
	}

	void set_change_dependencies(const double& number) {													// set changing value to specific value
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->set_change_dependencies(number);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->set_change_dependencies(number);
			else if (layer[i]->get_type() == Layer::DROPOUT)
				;
			else if (layer[i]->get_type() == Layer::FILTER)
				;
		}
	}

	void mul_change_dependencies(const double& number) {													// multiply changing value with specific value
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->mul_change_dependencies(number);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->mul_change_dependencies(number);
			else if (layer[i]->get_type() == Layer::DROPOUT)
				;
			else if (layer[i]->get_type() == Layer::FILTER)
				;
		}
	}



	void set_all_learning_rate(const double& number) {														// set every layer's learning rate to specific value
		for (int i = 0; i < layer.size(); i++) {
			layer[i]->set_learning_rate(number);
		}
	}

	void set_all_drop_out_rate(const double& number) {														// set every drop layer's drop out rate to specific layer
		for (int i = 0; i < layer.size(); i++) {
			if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->set_drop_out_rate(number);
		}
	}

	std::size_t get_layer_size() {
		return layer.size();
	}

	Matrix<double> get_output() {
		return layer.back()->value;
	}

	std::size_t get_input_size() {
		return layer[0]->get_size();
	}

	double get_loss(const Matrix<double>& target) {
		return loss_func(layer.back()->value, target);
	}

private:
	std::vector<Layer*> layer;																				// pointer containing all of the layers
	std::function<double(const Matrix<double>&, const Matrix<double>&)> loss_func;							// function computing loss
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dloss_func;					// derivatives loss funtio
};
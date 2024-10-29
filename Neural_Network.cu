#pragma once

#include "Header.cuh"
#include "Dense.cu"
#include "LSTM.cu"
#include "LayerId.cu"
#include "DropOut.cu"
#include "Filter.cu"
#include "Power.cu"

class Neural_Network {
public:
	Neural_Network() {}

	Neural_Network(std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func) :
		loss_func(_loss_func), dloss_func(_dloss_func) {

	}

	Neural_Network(std::vector<LayerId> _layer,
		std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func = squere_mean_loss_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func = dsquere_mean_loss_func) :
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
			else if (_layer[i].Layer_type == Layer::type::POWER) {
				layer.push_back(new Power(_layer[i]));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid Layer size after Power";
			}
		}
		if (_layer[_Layer_size - 1].Layer_type != Layer::DENSE)
			throw "the output layer must be Dense layer";
		layer.push_back(new Dense(_layer[_Layer_size - 1].Layer_size));
	}

	Neural_Network(const Neural_Network& copy) { // do not use, not finish dont use it
		std::size_t Layer_size = copy.layer.size();

		loss_func = copy.loss_func;
		dloss_func = copy.dloss_func;
		
		for (int i = 0; i < layer.size(); i++) {
			delete layer[i];
		}
		layer.clear();

		for (int i = 0; i < copy.layer.size(); i++) {
			layer.push_back(new Dense(*static_cast<Dense*>(copy.layer[i])));
		}
	}

	Neural_Network(Neural_Network&& other) {
		loss_func = other.loss_func;
		dloss_func = other.dloss_func;

		layer = std::move(other.layer);
	}

	~Neural_Network() {
		for (int i = 0; i < layer.size(); i++) {
			delete layer[i];
		}layer.clear();
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
			else if (layer[i]->Layer_type == Layer::POWER) {
				static_cast<Power*>(layer[i])->rand_weight(setting[i]);
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
			else if (layer[i]->Layer_type == Layer::POWER) {
				static_cast<Power*>(layer[i])->rand_weight(setting[i]);
			}
		}
	}

	void rand_weight(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "Invalid random weight value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_weight(setting[i], layer[i + 1]->value.row);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_weight(setting[i], layer[i + 1]->value.row);
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have weight
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
			else if (layer[i]->Layer_type == Layer::POWER) {
				static_cast<Power*>(layer[i])->rand_weight(setting[i], layer[i + 1]->value.row);
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
			else if (layer[i]->Layer_type == Layer::POWER) {
				;// Power layer dont have power
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
			else if (layer[i]->Layer_type == Layer::POWER) {
				;// Power layer dont have power
			}
		}
	}

	void rand_bias(const std::vector<std::function<double(std::size_t, std::size_t)>>& setting) {
		if (setting.size() != layer.size() - 1)
			throw "invalid random bias value";
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->Layer_type == Layer::DENSE) {
				static_cast<Dense*>(layer[i])->rand_bias(setting[i], layer[i + 1]->value.row);
			}
			else if (layer[i]->Layer_type == Layer::LSTM) {
				static_cast<LSTM*>(layer[i])->rand_bias(setting[i], layer[i + 1]->value.row);
			}
			else if (layer[i]->Layer_type == Layer::DROPOUT) {
				;// dropout layer dont have bias
			}
			else if (layer[i]->Layer_type == Layer::FILTER) {
				;// Filter layer dont have weight
			}
			else if (layer[i]->Layer_type == Layer::POWER) {
				;// Power layer dont have power
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
			else if (layer[i]->Layer_type == Layer::POWER)
				static_cast<Power*>(layer[i])->print_weight();
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
			else if (layer[i]->Layer_type == Layer::POWER)
				static_cast<Power*>(layer[i])->print_value();
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
			else if (layer[i]->Layer_type == Layer::POWER)
				static_cast<Power*>(layer[i])->print_bias();
		}
	}



	Matrix<double> feedforward(Matrix<double> input) { 
		if (input.row != layer[0]->get_size() || input.column != 1)
			throw "invalid input Matrix";

		layer[0]->set_value(input);																			// set input into the first layer

		for (int i = 1; i < layer.size(); i++) {															// loop though every layer
			layer[i]->set_value(layer[i - 1]->feed());														// get output from previous layer then set into the next layer
		}

		return layer.back()->value;																			// get output from last layer
	}

	Matrix<double> predict(Matrix<double> input) {
		if (input.row != layer[0]->get_size() || input.column != 1)
			throw "invalid input Matrix";

		layer[0]->set_value(input);																			// set input into the first layer

		for (int i = 1; i < layer.size(); i++) {															// loop though every layer
			layer[i]->set_value(layer[i - 1]->predict());														// get output from previous layer then set into the next layer
		}
		return static_cast<Dense*>(layer.back())->value;												// get output from last layer
	}

	void backpropagation(Matrix<double> target) {
		Matrix<double> output = static_cast<Dense*>(layer.back())->value;								// get output form last layer

		std::vector<Matrix<double>> error;
		error.push_back(dloss_func(output, target));														// compute the gadient from output and target value

		for (int i = layer.size() - 2; i >= 0; i--) {														// loop though every layer (from the back)
			error = layer[i]->propagation(error);															// get gadient flow from back layer and set into vector error
		}
	}

	void mutate(const double& mutation_chance, const double& mutation_rate) {
		for (int i = 0; i < layer.size() - 1; i++) {
			layer[i]->mutate(mutation_chance, mutation_rate);
		}
	}



	void fogot(const std::size_t& number) {																	// delete the past value memory
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->fogot(number);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->fogot(number);
			else if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->fogot(number);
			else if (layer[i]->get_type() == Layer::FILTER)
				static_cast<Filter*>(layer[i])->fogot(number);
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->fogot(number);
		}
	}

	void fogot_all() {																						// call fogot function with memory lenght
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->fogot_all();
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->fogot_all();
			else if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->fogot_all();
			else if (layer[i]->get_type() == Layer::FILTER)
				static_cast<Filter*>(layer[i])->fogot_all();
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->fogot_all();
		}
	}


	void reconstruct(std::vector<LayerId> _layer,
		std::function<double(const Matrix<double>&, const Matrix<double>&)> _loss_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dloss_func) {
		layer.clear();
		loss_func = _loss_func;
		dloss_func = _dloss_func;

		std::size_t _Layer_size = _layer.size();

		for (int i = 0; i < _Layer_size - 1; i++) {
			if (_layer[i].Layer_type == Layer::type::DENSE) {
				layer.push_back(new Dense(_layer[i], _layer[i + 1].Layer_size));
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
			else if (_layer[i].Layer_type == Layer::type::POWER) {
				layer.push_back(new Power(_layer[i]));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid Layer size after Power";
			}
		}
		if (_layer[_Layer_size - 1].Layer_type != Layer::DENSE)
			throw "the output layer must be Dense layer";
		layer.push_back(new Dense(_layer[_Layer_size - 1].Layer_size));
	}

	void reconstruct(const Neural_Network& copy) {
		std::size_t Layer_size = copy.layer.size();

		loss_func = copy.loss_func;
		dloss_func = copy.dloss_func;

		for (int i = 0; i < Layer_size; i++) {
			if (copy.layer[i]->get_type() == Layer::DENSE) {
				layer.push_back(new Dense(*static_cast<Dense*>(copy.layer[i])));
			}
			else if (copy.layer[i]->get_type() == Layer::LSTM) {
				layer.push_back(new LSTM(*static_cast<LSTM*>(copy.layer[i])));
			}
			else if (copy.layer[i]->get_type() == Layer::DROPOUT) {
				layer.push_back(new DropOut(*static_cast<DropOut*>(copy.layer[i])));
			}
			else if (copy.layer[i]->get_type() == Layer::FILTER) {
				layer.push_back(new Filter(*static_cast<Filter*>(copy.layer[i])));
			}
		}
	}

	void reconstruct(Neural_Network&& other) {
		loss_func = other.loss_func;
		dloss_func = other.dloss_func;

		layer = std::move(other.layer);
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
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->change_dependencies();
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
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->set_change_dependencies(number);
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
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->mul_change_dependencies(number);
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

	std::size_t get_input_size() const  {
		return layer[0]->get_size();
	}

	double get_loss(const Matrix<double>& target) {
		return loss_func(layer.back()->value, target);
	}

	void load_model(std::ifstream& model_file,int model_size = -1) {
		int type, node;
		std::string setting;
		std::vector<LayerId> _layer;
		int i = 0;
		while (model_size-- != 0 && !model_file.eof()) {
			model_file >> type >> node;
			std::getline(model_file, setting);
			_layer.push_back(LayerId(Layer::type(type), node, setting));

			if (_layer[i].Layer_type == Layer::type::DENSE) {
				layer.push_back(new Dense(_layer[i], _layer[i + 1].Layer_size));
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
			else if (_layer[i].Layer_type == Layer::type::POWER) {
				layer.push_back(new Power(_layer[i]));
				if (_layer[i + 1].Layer_size != _layer[i].Layer_size)
					throw "invalid Layer size after Power";
			}
			++i;
		}
		if(_layer.back().Layer_type != Layer::type::DENSE)
			throw "the output layer must be Dense layer";
		_layer.clear();
	}

	void save_as(std::ofstream& output_file) {
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->save_as(output_file);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->save_as(output_file);
			else if (layer[i]->get_type() == Layer::FILTER)
				static_cast<Filter*>(layer[i])->save_as(output_file);
			else if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->save_as(output_file);
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->save_as(output_file);
		}
	}

	void load(std::ifstream& input_file) {
		for (int i = 0; i < layer.size() - 1; i++) {
			if (layer[i]->get_type() == Layer::DENSE)
				static_cast<Dense*>(layer[i])->load(input_file);
			else if (layer[i]->get_type() == Layer::LSTM)
				static_cast<LSTM*>(layer[i])->load(input_file);
			else if (layer[i]->get_type() == Layer::FILTER)
				static_cast<Filter*>(layer[i])->load(input_file);
			else if (layer[i]->get_type() == Layer::DROPOUT)
				static_cast<DropOut*>(layer[i])->load(input_file);
			else if (layer[i]->get_type() == Layer::POWER)
				static_cast<Power*>(layer[i])->load(input_file);
		}
	}
	
private:
	std::vector<Layer*> layer;																				// pointer containing all of the layers
	std::function<double(const Matrix<double>&, const Matrix<double>&)> loss_func;							// function computing loss
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dloss_func;					// derivatives loss funtio
};
#pragma once

#include "Header.cuh"
#include "Matrix.cu"
#include "Func.cuh"

class Layer {
public:
	enum type {UNDEFINED,DENSE,RNN,LSTM,DROPOUT,FILTER,POWER};													// Layer type

	enum opt { SGD, MOMENTUM, ADAM };

	Layer() {};

	Layer(const type& _Layer_type,const std::size_t& size,const double& _learning_rate = 0.1) {
		Layer_type = _Layer_type;
		value.reconstruct(size, 1);
		learning_rate = _learning_rate;
	}

	Layer(const Layer& copy) {
		Layer_type = copy.Layer_type;
		learning_rate = copy.learning_rate;
		decay_rate = copy.decay_rate;
		optimizer = copy.optimizer;
		t = copy.t;

		value.reconstruct(copy.value);

		v = copy.v;
	}

	Layer(Layer&& other) {
		Layer_type = other.Layer_type;
		learning_rate = other.learning_rate;
		decay_rate = other.decay_rate;
		optimizer = other.optimizer;
		t = other.t;

		value = std::move(other.value);

		v = std::move(other.v);
	}

	virtual ~Layer() {}



	virtual Matrix<double> feed() = 0;																		// required feed funtion to feedforward

	virtual Matrix<double> predict() = 0;

	virtual std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient) = 0;		// required propagation function to backpropagation

	virtual void mutate(const double& mutation_chance, const double& mutation_rate) = 0;

	virtual void fogot(const std::size_t& number) = 0;														// required forgot function

	virtual void fogot_all() = 0;																			// required forgot all function



	std::size_t get_size() const {
		return value.row;
	}

	Layer::type get_type() {
		return Layer_type;
	}

	double get_learning_rate() {
		return learning_rate;
	}

	Matrix<double> get_value() {
		return value;
	}



	void set_learning_rate(const double& _learning_rate) {
		learning_rate = _learning_rate;
	}

	void set_value(const Matrix<double>& _value) {
		value = _value;
	}


	void reconstruct(const type& _Layer_type, const std::size_t& size, const double& _learning_rate = 0.1) {
		Layer_type = _Layer_type;
		learning_rate = _learning_rate;
		value.reconstruct(size, 1);
		v.clear();
	}

	void reconstruct(const Layer& copy) {
		Layer_type = copy.Layer_type;
		learning_rate = copy.learning_rate;
		optimizer = copy.optimizer;
		decay_rate = copy.decay_rate;
		t = copy.t;

		value.reconstruct(copy.value);
		v = copy.v;
	}

	void reconstruct(Layer&& other) {
		Layer_type = other.Layer_type;
		learning_rate = other.learning_rate;
		optimizer = other.optimizer;
		decay_rate = other.decay_rate;
		t = other.t;

		value.reconstruct(std::move(other.value));
		v = std::move(other.v);
	}


	
	Matrix<double> operator=(const Matrix<double>& rhs) {
		return value = rhs;
	}
	
protected:
	Matrix<double> value;																					// pointer containing value to be feeded
	std::vector<Matrix<double>> v;																			// vector containing past value
	type Layer_type = UNDEFINED;																			
	double learning_rate = 0.1;

	opt optimizer = SGD;
	double decay_rate = 0.9;

	int t = 0;

	friend class Neural_Network;
};
#pragma once

#include "Header.h"
#include "Matrix.h"
#include "Layer.h"
#include "LayerId.h"
#include "Function.h"
#include "Variable.h"

Layer::Layer() {};

Layer::Layer(const std::size_t& size) {
	Layer::reconstruct(size);
}

Layer::Layer(const LayerId _layer_id) {
	Layer::reconstruct(_layer_id);
}

Layer::Layer(const Layer& copy) {
	Layer::reconstruct(copy);
}

Layer::Layer(Layer&& move) {
	Layer::reconstruct(move);
}

Layer::~Layer() {}



void Layer::forgot(const std::size_t& number) {
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

void Layer::forgot_all() {
	Layer::forgot(v.size());
}

void Layer::set_drop_out_rate(const double& number) {
	// No implementation for default
}

void Layer::change_dependencies() {
	// No implementation for default
}

void Layer::set_change_dependencies(const double& value) {
	// No implementation for default
}

void Layer::mul_change_dependencies(const double& value) {
	// No implementation for default
}

void Layer::map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func) {
	// No implementation for default
}

bool Layer::found_nan() {
	return false;
}



void Layer::reconstruct(const std::size_t& size) {
	value.reconstruct(size, 1);

	learning_rate = 0.1;
	v.clear();
	Layer_type = UNDEFINED;
	optimizer = SGD;
	decay_rate = 0.9;
	do_record = false;
	t = 0;
}

void Layer::reconstruct(const LayerId _layer_id) {
	Layer::reconstruct(_layer_id.Layer_size);

	Layer_type = _layer_id.Layer_type;
}

void Layer::reconstruct(const Layer& copy) {
	value = copy.value;
	v = copy.v;
	Layer_type = copy.Layer_type;
	learning_rate = copy.learning_rate;
	optimizer = copy.optimizer;
	decay_rate = copy.decay_rate;
	do_record = copy.do_record;

	t = copy.t;
}

void Layer::reconstruct(Layer&& move) {
	value = std::move(move.value);
	v = std::move(move.v);
	Layer_type = move.Layer_type;
	learning_rate = move.learning_rate;
	optimizer = move.optimizer;
	decay_rate = move.decay_rate;
	do_record = move.do_record;

	t = move.t;
}


void Layer::set_weight(const int& weight_type, const Matrix<double>& _weight) {
	// No implementation for default
}

void Layer::set_bias(const int& bias_type, const Matrix<double>& _bias) {
	// No implementation for default
}

void Layer::rand_weight(const double& min, const double& max) {
	// No implementation for default
}

void Layer::rand_weight(const std::pair<double, double>& setting) {
	// No implementation for default
}

void Layer::rand_weight(const std::function<double()>& setting) {
	// No implementation for default
}

void Layer::rand_weight(const std::function<double(std::size_t, std::size_t)>& setting,const std::size_t& next) {
	// No implementation for default
}

void Layer::rand_bias(const double& min, const double& max) {
	// No implementation for default
}

void Layer::rand_bias(const std::pair<double, double>& setting) {
	// No implementation for default
}

void Layer::rand_bias(const std::function<double()>& setting) {
	// No implementation for default
}

void Layer::rand_bias(const std::function<double(std::size_t, std::size_t)>& setting,const std::size_t& next) {
	// No implementation for default
}



void Layer::print_weight() {
	// No implementation for default
}


void Layer::print_bias() {
	// No implementation for default
}

void Layer::print_value() {
	value.print();
}



std::size_t Layer::get_size() {
	return value.get_row();
}

Layer::type Layer::get_type() {
	return Layer_type;
}

double Layer::get_learning_rate() {
	return learning_rate;
}

void Layer::set_value(const Matrix<double>& _value) {
	value = _value;
}

Matrix<double> Layer::get_value() {
	return value;
}

double Layer::get_max_abs_change_dependencies() {
	return 0;
}


void Layer::set_learning_rate(const double& _learning_rate) {
	learning_rate = _learning_rate;

	if (learning_rate < 0) {
		learning_rate = 0;
	}
}

void Layer::set_optimizer(const Layer::opt _opt) {
	optimizer = _opt;
}



void Layer::save_as(std::ofstream& ouput_file) {
	// No implementation for default
}

void Layer::load(std::ifstream& input_file) {
	// No implementation for default;
}


	
Matrix<double> Layer::operator=(const Matrix<double>& rhs) {
	return value = rhs;
}
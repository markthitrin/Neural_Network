#pragma once
#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "LSTM.h"
#include "Function.h"
#include "Variable.h"


LSTM::LSTM() { Layer_type = Layer::LSTM; };

LSTM::LSTM(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _Iact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func,
	std::function<Matrix<double>(const Matrix<double>&)> _Fact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func,
	std::function<Matrix<double>(const Matrix<double>&)> _Oact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func,
	std::function<Matrix<double>(const Matrix<double>&)> _Kact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func) {
	reconstruct(size, _Iact_func, _dIact_func, _Fact_func, _dFact_func, _Kact_func, _dKact_func, _Kact_func, _dKact_func);
}

LSTM::LSTM(const LayerId& set) {
	reconstruct(set);
}

LSTM::LSTM(const LSTM& copy) {
	reconstruct(copy);
}

LSTM::LSTM(LSTM&& move) {
	reconstruct(move);
}


		
Matrix<double> LSTM::feed() {
	Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);
	Matrix<double> forgot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);
	Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);
	Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);

	c.push_back(mul_each(forgot_gate, c.back()) + mul_each(input_gate, K));
	h.push_back(mul_each(output_gate, c.back()));
	
	if (do_record || v.size() == 0)
		v.push_back(value);
	else
		v[0] = value;

	return h.back();
}

std::vector<Matrix<double>> LSTM::propagation(const std::vector<Matrix<double>>& gadient) {
	if (gadient.size() > c.size())
		throw std::runtime_error("invalid gaadient  size for  backpropagatoin lstm");

	Matrix<double> dc(value.get_row(), 1);
	Matrix<double> dh(value.get_row(), 1);

	Matrix<double> next_dc(value.get_row(), 1);
	Matrix<double> next_dh(value.get_row(), 1);

	dc = 0;
	dh = 0;

	std::vector<Matrix<double>> _gadient;
	std::vector<Matrix<double>> flow_gadient;



	for (int i = 0; i < v.size(); i++) {
		flow_gadient.push_back(Matrix<double>(value.get_row(), 1));
		flow_gadient.back() = 0;
	}

	for (int i = 0; i + gadient.size() < v.size(); i++) {
		_gadient.push_back(Matrix<double>(value.get_row(), 1));
		_gadient.back() = 0;
	}

	for (int i = 0; i < gadient.size(); i++) {
		_gadient.push_back(gadient[i]);
	}



	for (int round = v.size() - 1; round >= 0; round--) {
		next_dc = 0;
		next_dh = 0;

		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h[round]) + Ibias);
		Matrix<double> forgot_gate = Fact_func((xF_weight * value) + (hF_weight * h[round]) + Fbias);
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h[round]) + Obias);
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h[round]) + Kbias);

		dh = dh + _gadient[round];
		dc = dc + mul_each(dh, output_gate);

		Matrix<double> dinput_gate = dIact_func((xI_weight * value) + (hI_weight * h[round]) + Ibias, mul_each(dc, K));
		Matrix<double> dforgot_gate = dFact_func((xF_weight * value) + (hF_weight * h[round]) + Fbias, mul_each(dc, c[round]));
		Matrix<double> doutput_gate = dOact_func((xO_weight * value) + (hO_weight * h[round]) + Obias, mul_each(dh, c[round + 1]));
		Matrix<double> dK = dKact_func((xK_weight * value) + (hK_weight * h[round]) + Kbias, mul_each(dc, input_gate));

		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				xO_weight_change[i][j] += check_nan(doutput_gate[i][0] * v[round][j][0],0.0) * learning_rate;
				hO_weight_change[i][j] += check_nan(doutput_gate[i][0] * h[round][j][0],0.0) * learning_rate;
				xI_weight_change[i][j] += check_nan(dinput_gate[i][0] * v[round][j][0], 0.0) * learning_rate;
				hI_weight_change[i][j] += check_nan(dinput_gate[i][0] * h[round][j][0], 0.0) * learning_rate;
				xF_weight_change[i][j] += check_nan(dforgot_gate[i][0] * v[round][j][0], 0.0) * learning_rate;
				hF_weight_change[i][j] += check_nan(dforgot_gate[i][0] * h[round][j][0], 0.0) * learning_rate;
				xK_weight_change[i][j] += check_nan(dK[i][0] * v[round][j][0], 0.0) * learning_rate;
				hK_weight_change[i][j] += check_nan(dK[i][0] * h[round][j][0], 0.0) * learning_rate;

				next_dh[i][0] += check_nan(doutput_gate[i][0] * hO_weight_change[i][j], 0.0);
				next_dh[i][0] += check_nan(dinput_gate[i][0] * hI_weight_change[i][j], 0.0);
				next_dh[i][0] += check_nan(dforgot_gate[i][0] * hF_weight_change[i][j], 0.0);
				next_dh[i][0] += check_nan(dK[i][0] * hK_weight_change[i][j], 0.0);

				flow_gadient[round][i][0] += check_nan(doutput_gate[i][0] * xO_weight_change[i][j], 0.0);
				flow_gadient[round][i][0] += check_nan(dinput_gate[i][0] * xI_weight_change[i][j], 0.0);
				flow_gadient[round][i][0] += check_nan(dforgot_gate[i][0] * xF_weight_change[i][j], 0.0);
				flow_gadient[round][i][0] += check_nan(dK[i][0] * hK_weight_change[i][j], 0.0);
				
			}
			Obias_change[i][0] += doutput_gate[i][0] * learning_rate;
			Ibias_change[i][0] += dinput_gate[i][0] * learning_rate;
			Fbias_change[i][0] += dforgot_gate[i][0] * learning_rate;
			Kbias_change[i][0] += dK[i][0] * learning_rate;

		}

		next_dc = mul_each(dc, forgot_gate);

		dh = next_dh;																						
		dc = next_dc;
			
		// try to descale exploding gadient
		double max_dh_value = std::max(get_max(dh), std::abs(get_min(dh)));									
		double max_dc_value = std::max(get_max(dc), std::abs(get_min(dc)));
			
		double flow_cap = std::sqrt(double(2) / v.size());
		if (max_dh_value > flow_cap) dh = dh * (flow_cap / max_dh_value);
		if (max_dc_value > flow_cap) dc = dc * (flow_cap / max_dc_value);
	}

	for (int i = 0; i < value.get_row(); i++) {
		init_h_change[i][0] += dh[i][0] * learning_rate;
		init_c_change[i][0] += dc[i][0] * learning_rate;
	}

	return flow_gadient;
}
	


void LSTM::forgot(const std::size_t& number) {
	Layer::forgot(number);

	std::size_t _number = number;
	if (number > c.size())
		_number = c.size();
	for (int i = 0; i < c.size() - _number; i++) {
		c[i] = c[i + _number];
		h[i] = h[i + _number];
	}
	for (int i = 0; i < _number; i++) {
		h.pop_back();
		c.pop_back();
	}
	if (c.size() == 0) {
		c.push_back(init_c);
		h.push_back(init_h);
	}
}

void LSTM::forgot_all() {
	forgot(v.size());
}



void LSTM::change_dependencies() {
	++t;
	switch (optimizer) {
	case Layer::SGD : 
		xO_weight += xO_weight_change;
		xF_weight += xF_weight_change;
		xI_weight += xI_weight_change;
		xK_weight += xK_weight_change;
		hO_weight += hO_weight_change;
		hF_weight += hF_weight_change;
		hI_weight += hI_weight_change;
		hK_weight += hK_weight_change;

		Obias += Obias_change;
		Fbias += Fbias_change;
		Ibias += Ibias_change;
		Kbias += Kbias_change;

		init_c += init_c_change;
		init_h += init_h_change;
		break;
	case Layer::MOMENTUM :
		s_xO_weight_change = s_xO_weight_change * decay_rate + xO_weight_change * (double(1) - decay_rate);
		s_xF_weight_change = s_xF_weight_change * decay_rate + xF_weight_change * (double(1) - decay_rate);
		s_xI_weight_change = s_xI_weight_change * decay_rate + xI_weight_change * (double(1) - decay_rate);
		s_xK_weight_change = s_xK_weight_change * decay_rate + xK_weight_change * (double(1) - decay_rate);
		s_hO_weight_change = s_hO_weight_change * decay_rate + xO_weight_change * (double(1) - decay_rate);
		s_hF_weight_change = s_hF_weight_change * decay_rate + xF_weight_change * (double(1) - decay_rate);
		s_hI_weight_change = s_hI_weight_change * decay_rate + xI_weight_change * (double(1) - decay_rate);
		s_hK_weight_change = s_hK_weight_change * decay_rate + xK_weight_change * (double(1) - decay_rate);

		s_Obias_change = s_Obias_change * decay_rate + Obias_change * (double(1) - decay_rate);
		s_Fbias_change = s_Fbias_change * decay_rate + Fbias_change * (double(1) - decay_rate);
		s_Ibias_change = s_Ibias_change * decay_rate + Ibias_change * (double(1) - decay_rate);
		s_Kbias_change = s_Kbias_change * decay_rate + Kbias_change * (double(1) - decay_rate);

		s_init_c_change = s_init_c_change * decay_rate + init_c_change * (double(1) - decay_rate);
		s_init_h_change = s_init_h_change * decay_rate + init_h_change * (double(1) - decay_rate);

		xO_weight = s_xO_weight_change + xO_weight;
		xF_weight = s_xF_weight_change + xF_weight;
		xI_weight = s_xI_weight_change + xI_weight;
		xK_weight = s_xK_weight_change + xK_weight;
		hO_weight = s_hO_weight_change + hO_weight;
		hF_weight = s_hF_weight_change + hF_weight;
		hI_weight = s_hI_weight_change + hI_weight;
		hK_weight = s_hK_weight_change + hK_weight;

		Obias = s_Obias_change + Obias;
		Fbias = s_Fbias_change + Fbias;
		Ibias = s_Ibias_change + Ibias;
		Kbias = s_Kbias_change + Kbias;

		init_c = init_c + s_init_c_change;
		init_h = init_h + s_init_h_change;
		break;
	case Layer::ADAM :
		s_xO_weight_change = s_xO_weight_change * decay_rate + xO_weight_change * (double(1) - decay_rate);
		s_xF_weight_change = s_xF_weight_change * decay_rate + xF_weight_change * (double(1) - decay_rate);
		s_xI_weight_change = s_xI_weight_change * decay_rate + xI_weight_change * (double(1) - decay_rate);
		s_xK_weight_change = s_xK_weight_change * decay_rate + xK_weight_change * (double(1) - decay_rate);
		s_hO_weight_change = s_hO_weight_change * decay_rate + xO_weight_change * (double(1) - decay_rate);
		s_hF_weight_change = s_hF_weight_change * decay_rate + xF_weight_change * (double(1) - decay_rate);
		s_hI_weight_change = s_hI_weight_change * decay_rate + xI_weight_change * (double(1) - decay_rate);
		s_hK_weight_change = s_hK_weight_change * decay_rate + xK_weight_change * (double(1) - decay_rate);

		s_Obias_change = s_Obias_change * decay_rate + Obias_change * (double(1) - decay_rate);
		s_Fbias_change = s_Fbias_change * decay_rate + Fbias_change * (double(1) - decay_rate);
		s_Ibias_change = s_Ibias_change * decay_rate + Ibias_change * (double(1) - decay_rate);
		s_Kbias_change = s_Kbias_change * decay_rate + Kbias_change * (double(1) - decay_rate);

		s_init_c_change = s_init_c_change * decay_rate + init_c_change * (double(1) - decay_rate);
		s_init_h_change = s_init_h_change * decay_rate + init_h_change * (double(1) - decay_rate);

		ss_xO_weight_change = ss_xO_weight_change * decay_rate + mul_each(xO_weight_change, xO_weight_change) * (double(1) - decay_rate);
		ss_xF_weight_change = ss_xF_weight_change * decay_rate + mul_each(xF_weight_change, xF_weight_change) * (double(1) - decay_rate);
		ss_xI_weight_change = ss_xI_weight_change * decay_rate + mul_each(xI_weight_change, xI_weight_change) * (double(1) - decay_rate);
		ss_xK_weight_change = ss_xK_weight_change * decay_rate + mul_each(xK_weight_change, xK_weight_change) * (double(1) - decay_rate);
		ss_hO_weight_change = ss_hO_weight_change * decay_rate + mul_each(hO_weight_change, hO_weight_change) * (double(1) - decay_rate);
		ss_hF_weight_change = ss_hF_weight_change * decay_rate + mul_each(hF_weight_change, hF_weight_change) * (double(1) - decay_rate);
		ss_hI_weight_change = ss_hI_weight_change * decay_rate + mul_each(hI_weight_change, hI_weight_change) * (double(1) - decay_rate);
		ss_hK_weight_change = ss_hK_weight_change * decay_rate + mul_each(hK_weight_change, hK_weight_change) * (double(1) - decay_rate);

		ss_Obias_change = ss_Obias_change * decay_rate + mul_each(Obias_change, Obias_change) * (double(1) - decay_rate);
		ss_Fbias_change = ss_Fbias_change * decay_rate + mul_each(Fbias_change, Fbias_change) * (double(1) - decay_rate);
		ss_Ibias_change = ss_Ibias_change * decay_rate + mul_each(Ibias_change, Ibias_change) * (double(1) - decay_rate);
		ss_Kbias_change = ss_Kbias_change * decay_rate + mul_each(Kbias_change, Kbias_change) * (double(1) - decay_rate);

		ss_init_c_change = ss_init_c_change * decay_rate + mul_each(init_c_change, init_c_change) * (double(1) - decay_rate);
		ss_init_h_change = ss_init_h_change * decay_rate + mul_each(init_h_change, init_h_change) * (double(1) - decay_rate);

		xO_weight = xO_weight + devide_each(s_xO_weight_change * learning_rate, pow(ss_xO_weight_change, 0.5) + 0.0000001);
		xF_weight = xF_weight + devide_each(s_xF_weight_change * learning_rate, pow(ss_xF_weight_change, 0.5) + 0.0000001);
		xI_weight = xI_weight + devide_each(s_xI_weight_change * learning_rate, pow(ss_xI_weight_change, 0.5) + 0.0000001);
		xK_weight = xK_weight + devide_each(s_xK_weight_change * learning_rate, pow(ss_xK_weight_change, 0.5) + 0.0000001);
		hO_weight = hO_weight + devide_each(s_hO_weight_change * learning_rate, pow(ss_hO_weight_change, 0.5) + 0.0000001);
		hF_weight = hF_weight + devide_each(s_hF_weight_change * learning_rate, pow(ss_hF_weight_change, 0.5) + 0.0000001);
		hI_weight = hI_weight + devide_each(s_hI_weight_change * learning_rate, pow(ss_hI_weight_change, 0.5) + 0.0000001);
		hK_weight = hK_weight + devide_each(s_hK_weight_change * learning_rate, pow(ss_hK_weight_change, 0.5) + 0.0000001);

		Obias = Obias + devide_each(s_Obias_change * learning_rate, pow(ss_Obias_change, 0.5) + 0.0000001);
		Fbias = Fbias + devide_each(s_Fbias_change * learning_rate, pow(ss_Fbias_change, 0.5) + 0.0000001);
		Ibias = Ibias + devide_each(s_Ibias_change * learning_rate, pow(ss_Ibias_change, 0.5) + 0.0000001);
		Kbias = Kbias + devide_each(s_Kbias_change * learning_rate, pow(ss_Kbias_change, 0.5) + 0.0000001);

		init_c = init_c + devide_each(s_init_c_change * learning_rate, pow(ss_init_c_change, 0.5) + 0.0000001);
		init_h = init_h + devide_each(s_init_h_change * learning_rate, pow(ss_init_h_change, 0.5) + 0.0000001);
		break;
	}
}

void LSTM::set_change_dependencies(const double& number) {
	xO_weight_change = number;
	xF_weight_change = number;
	xI_weight_change = number;
	xK_weight_change = number;
	hO_weight_change = number;
	hF_weight_change = number;
	hI_weight_change = number;
	hK_weight_change = number;

	Obias_change = number;
	Fbias_change = number;
	Ibias_change = number;
	Kbias_change = number;

	init_c_change = number;
	init_h_change = number;
}

void LSTM::mul_change_dependencies(const double& number) {
	xO_weight_change *= number;
	xF_weight_change *= number;
	xI_weight_change *= number;
	xK_weight_change *= number;
	hO_weight_change *= number;
	hF_weight_change *= number;
	hI_weight_change *= number;
	hK_weight_change *= number;

	Obias_change *= number;
	Fbias_change *= number;
	Ibias_change *= number;
	Kbias_change *= number;

	init_c_change *= number;
	init_h_change *= number;
}

void LSTM::map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func) {
	xO_weight_change = func(xO_weight_change);
	xF_weight_change = func(xF_weight_change);
	xI_weight_change = func(xI_weight_change);
	xK_weight_change = func(xK_weight_change);
	hO_weight_change = func(hO_weight_change);
	hF_weight_change = func(hF_weight_change);
	hI_weight_change = func(hI_weight_change);
	hK_weight_change = func(hK_weight_change);

	Obias_change = func(Obias_change);
	Fbias_change = func(Fbias_change);
	Ibias_change = func(Ibias_change);
	Kbias_change = func(Kbias_change);

	init_c_change = func(init_c_change);
	init_h_change = func(init_h_change);
}

bool LSTM::found_nan() {
	return
		xO_weight != xO_weight ||
		xF_weight != xF_weight ||
		xI_weight != xI_weight ||
		xK_weight != xK_weight ||
		hO_weight != hO_weight ||
		hF_weight != hF_weight ||
		hI_weight != hI_weight ||
		hK_weight != hK_weight ||
		Obias != Obias ||
		Fbias != Fbias ||
		Ibias != Ibias ||
		Kbias != Kbias ||
		init_c_change != init_c_change ||
		init_h_change != init_h_change;
}



void LSTM::reconstruct(const std::size_t& size) {
	reconstruct(size, tanh_func, dtanh_func, sigmoid_func, dsigmoid_func, sigmoid_func, dsigmoid_func, tanh_func, dtanh_func);
}

void LSTM::reconstruct(const std::size_t& size,
	std::function<Matrix<double>(const Matrix<double>&)> _Iact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func,
	std::function<Matrix<double>(const Matrix<double>&)> _Fact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func,
	std::function<Matrix<double>(const Matrix<double>&)> _Oact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func,
	std::function<Matrix<double>(const Matrix<double>&)> _Kact_func,
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func) {
	Layer::reconstruct(size);

	Layer::Layer_type = Layer::type::LSTM;

	Iact_func = _Iact_func;
	Fact_func = _Fact_func;
	Oact_func = _Oact_func;
	Kact_func = _Kact_func;
	dIact_func = _dIact_func;
	dFact_func = _dFact_func;
	dOact_func = _dOact_func;
	dKact_func = _dKact_func;

	xO_weight.reconstruct(size, size);
	xF_weight.reconstruct(size, size);
	xI_weight.reconstruct(size, size);
	xK_weight.reconstruct(size, size);
	hO_weight.reconstruct(size, size);
	hF_weight.reconstruct(size, size);
	hI_weight.reconstruct(size, size);
	hK_weight.reconstruct(size, size);

	Obias.reconstruct(size, 1);
	Fbias.reconstruct(size, 1);
	Ibias.reconstruct(size, 1);
	Kbias.reconstruct(size, 1);

	xO_weight_change.reconstruct(size, size);
	xF_weight_change.reconstruct(size, size);
	xI_weight_change.reconstruct(size, size);
	xK_weight_change.reconstruct(size, size);
	hO_weight_change.reconstruct(size, size);
	hF_weight_change.reconstruct(size, size);
	hI_weight_change.reconstruct(size, size);
	hK_weight_change.reconstruct(size, size);

	Obias_change.reconstruct(size, 1);
	Fbias_change.reconstruct(size, 1);
	Ibias_change.reconstruct(size, 1);
	Kbias_change.reconstruct(size, 1);

	init_c.reconstruct(size, 1);
	init_h.reconstruct(size, 1);

	init_c = 0;
	init_h = 0;

	c.push_back(init_c);
	h.push_back(init_h);
}

void LSTM::reconstruct(const LayerId& set) {
	reconstruct(set.Layer_size);
	
	set_Layer(set.setting);
}

void LSTM::reconstruct(const LSTM& copy) {
	Layer::reconstruct(copy);

	Iact_func = copy.Iact_func;
	Fact_func = copy.Fact_func;
	Oact_func = copy.Oact_func;
	Kact_func = copy.Kact_func;
	dIact_func = copy.dIact_func;
	dFact_func = copy.dFact_func;
	dOact_func = copy.dOact_func;
	dKact_func = copy.dKact_func;

	xO_weight.reconstruct(copy.xO_weight);
	xF_weight.reconstruct(copy.xF_weight);
	xI_weight.reconstruct(copy.xI_weight);
	xK_weight.reconstruct(copy.xK_weight);
	hO_weight.reconstruct(copy.hO_weight);
	hF_weight.reconstruct(copy.hF_weight);
	hI_weight.reconstruct(copy.hI_weight);
	hK_weight.reconstruct(copy.hK_weight);

	Obias.reconstruct(copy.Obias);
	Fbias.reconstruct(copy.Fbias);
	Ibias.reconstruct(copy.Ibias);
	Kbias.reconstruct(copy.Kbias);
	
	init_c.reconstruct(copy.init_c);
	init_h.reconstruct(copy.init_h);

	xO_weight_change.reconstruct(copy.xO_weight_change);
	xF_weight_change.reconstruct(copy.xF_weight_change);
	xI_weight_change.reconstruct(copy.xI_weight_change);
	xK_weight_change.reconstruct(copy.xK_weight_change);
	hO_weight_change.reconstruct(copy.hO_weight_change);
	hF_weight_change.reconstruct(copy.hF_weight_change);
	hI_weight_change.reconstruct(copy.hI_weight_change);
	hK_weight_change.reconstruct(copy.hK_weight_change);

	Obias_change.reconstruct(copy.Obias_change);
	Fbias_change.reconstruct(copy.Fbias_change);
	Ibias_change.reconstruct(copy.Ibias_change);
	Kbias_change.reconstruct(copy.Kbias_change);

	init_c_change.reconstruct(copy.init_c_change);
	init_h_change.reconstruct(copy.init_h_change);

	s_xO_weight_change.reconstruct(copy.xO_weight_change);
	s_xF_weight_change.reconstruct(copy.xF_weight_change);
	s_xI_weight_change.reconstruct(copy.xI_weight_change);
	s_xK_weight_change.reconstruct(copy.xK_weight_change);
	s_hO_weight_change.reconstruct(copy.hO_weight_change);
	s_hF_weight_change.reconstruct(copy.hF_weight_change);
	s_hI_weight_change.reconstruct(copy.hI_weight_change);
	s_hK_weight_change.reconstruct(copy.hK_weight_change);

	s_Obias_change.reconstruct(copy.Obias_change);
	s_Fbias_change.reconstruct(copy.Fbias_change);
	s_Ibias_change.reconstruct(copy.Ibias_change);
	s_Kbias_change.reconstruct(copy.Kbias_change);

	s_init_c_change.reconstruct(copy.init_c);
	s_init_h_change.reconstruct(copy.init_h);

	ss_xO_weight_change.reconstruct(copy.xO_weight_change);
	ss_xF_weight_change.reconstruct(copy.xF_weight_change);
	ss_xI_weight_change.reconstruct(copy.xI_weight_change);
	ss_xK_weight_change.reconstruct(copy.xK_weight_change);
	ss_hO_weight_change.reconstruct(copy.hO_weight_change);
	ss_hF_weight_change.reconstruct(copy.hF_weight_change);
	ss_hI_weight_change.reconstruct(copy.hI_weight_change);
	ss_hK_weight_change.reconstruct(copy.hK_weight_change);

	ss_Obias_change.reconstruct(copy.Obias_change);
	ss_Fbias_change.reconstruct(copy.Fbias_change);
	ss_Ibias_change.reconstruct(copy.Ibias_change);
	ss_Kbias_change.reconstruct(copy.Kbias_change);

	ss_init_c_change.reconstruct(copy.init_c);
	ss_init_h_change.reconstruct(copy.init_h);

	c = copy.c;
	h = copy.h;
}

void LSTM::reconstruct(LSTM&& move) {
	Layer::reconstruct(std::move(move));

	Iact_func = std::move(move.Iact_func);
	Fact_func = std::move(move.Fact_func);
	Oact_func = std::move(move.Oact_func);
	Kact_func = std::move(move.Kact_func);
	dIact_func = std::move(move.dIact_func);
	dFact_func = std::move(move.dFact_func);
	dOact_func = std::move(move.dOact_func);
	dKact_func = std::move(move.dKact_func);

	xO_weight.reconstruct(std::move(move.xO_weight));
	xF_weight.reconstruct(std::move(move.xF_weight));
	xI_weight.reconstruct(std::move(move.xI_weight));
	xK_weight.reconstruct(std::move(move.xK_weight));
	hO_weight.reconstruct(std::move(move.hO_weight));
	hF_weight.reconstruct(std::move(move.hF_weight));
	hI_weight.reconstruct(std::move(move.hI_weight));
	hK_weight.reconstruct(std::move(move.hK_weight));

	Obias.reconstruct(std::move(move.Obias));
	Fbias.reconstruct(std::move(move.Fbias));
	Ibias.reconstruct(std::move(move.Ibias));
	Kbias.reconstruct(std::move(move.Kbias));

	xO_weight_change.reconstruct(std::move(move.xO_weight_change));
	xF_weight_change.reconstruct(std::move(move.xF_weight_change));
	xI_weight_change.reconstruct(std::move(move.xI_weight_change));
	xK_weight_change.reconstruct(std::move(move.xK_weight_change));
	hO_weight_change.reconstruct(std::move(move.hO_weight_change));
	hF_weight_change.reconstruct(std::move(move.hF_weight_change));
	hI_weight_change.reconstruct(std::move(move.hI_weight_change));
	hK_weight_change.reconstruct(std::move(move.hK_weight_change));

	Obias_change.reconstruct(std::move(move.Obias_change));
	Fbias_change.reconstruct(std::move(move.Fbias_change));
	Ibias_change.reconstruct(std::move(move.Ibias_change));
	Kbias_change.reconstruct(std::move(move.Kbias_change));

	init_c.reconstruct(std::move(move.init_c));
	init_h.reconstruct(std::move(move.init_h));

	c = std::move(move.c);
	h = std::move(move.h);
}


void LSTM::set_weight(const int& weight_type, const Matrix<double>& _weight) {
	switch (weight_type) {
	case weight_type::XO_WEIGHT :
		xO_weight = _weight;
		break;
	case weight_type::XF_WEIGHT:
		xF_weight = _weight;
		break;
	case weight_type::XI_WEIGHT:
		xI_weight = _weight;
		break;
	case weight_type::XK_WEIGHT:
		xK_weight = _weight;
		break;
	case weight_type::HO_WEIGHT:
		hO_weight = _weight;
		break;
	case weight_type::HF_WEIGHT:
		hF_weight = _weight;
		break;
	case weight_type::HI_WEIGHT:
		hI_weight = _weight;
		break;
	case weight_type::HK_WEIGHT:
		hK_weight = _weight;
		break;
	case weight_type::INIT_C:
		init_c = _weight;
		break;
	case weight_type::INIT_H:
		init_h = _weight;
		break;
	default :
		throw std::runtime_error("Invalid weight_type to be set");
		break;
	}
}

void LSTM::set_bias(const int& bias_type, const Matrix<double>& _bias) {
	switch (bias_type) {
	case bias_type::OBIAS:
		Obias = _bias;
		break;
	case bias_type::FBIAS:
		Fbias = _bias;
		break;
	case bias_type::IBIAS:
		Ibias = _bias;
		break;
	case bias_type::KBIAS:
		Kbias = _bias;
		break;
	default:
		throw std::runtime_error("Invalid bias_type to be set");
		break;
	}
}

void LSTM::rand_weight(const double& min, const double& max) {
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			xO_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
			xF_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
			xI_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);
			xK_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max);

			hO_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
			hF_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
			hI_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
			hK_weight[i][j] = mapping(rand() % 10000, 0, 10000, min, max) / 10;
		}
	}
}

void LSTM::rand_weight(std::pair<const double&, const double&> setting) {
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			xO_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xF_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xI_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xK_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);

			hO_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
			hF_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
			hI_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
			hK_weight[i][j] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) / 10;
		}
	}
}
	
void LSTM::rand_weight(std::function<double()> func) {
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			xO_weight[i][j] = func();
			xF_weight[i][j] = func();
			xI_weight[i][j] = func();
			xK_weight[i][j] = func();

			hO_weight[i][j] = func() / 10;
			hF_weight[i][j] = func() / 10;
			hI_weight[i][j] = func() / 10;
			hK_weight[i][j] = func() / 10;
		}
	}
}

void LSTM::rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			xO_weight[i][j] = func(value.get_row(), next);
			xF_weight[i][j] = func(value.get_row(), next);
			xI_weight[i][j] = func(value.get_row(), next);
			xK_weight[i][j] = func(value.get_row(), next);

			hO_weight[i][j] = func(value.get_row(), next) / 10;
			hF_weight[i][j] = func(value.get_row(), next) / 10;
			hI_weight[i][j] = func(value.get_row(), next) / 10;
			hK_weight[i][j] = func(value.get_row(), next) / 10;
		}
	}
}

void LSTM::rand_bias(const double& min, const double& max) {
	for (int i = 0; i < value.get_row(); i++) {
		Obias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
		Fbias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
		Ibias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
		Kbias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
	}
}

void LSTM::rand_bias(std::pair<const double&, const double&> setting) {
	for (int i = 0; i < value.get_row(); i++) {
		Obias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		Fbias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		Ibias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		Kbias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
	}
}
	
void LSTM::rand_bias(std::function<double()> func) {
	for (int i = 0; i < value.get_row(); i++) {
		Obias[i][0] = func();
		Fbias[i][0] = func();
		Ibias[i][0] = func();
		Kbias[i][0] = func();
	}
}

void LSTM::rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next) {
	for (int i = 0; i < value.get_row(); i++) {
		Obias[i][0] = func(value.get_row(), next);
		Fbias[i][0] = func(value.get_row(), next);
		Ibias[i][0] = func(value.get_row(), next);
		Kbias[i][0] = func(value.get_row(), next);
	}
}



std::function<Matrix<double>(const Matrix<double>&)> LSTM::get_Oact_func() {
	return Oact_func;
}

std::function<Matrix<double>(const Matrix<double>&)> LSTM::get_Fact_func() {
	return Fact_func;
}

std::function<Matrix<double>(const Matrix<double>&)> LSTM::get_Iact_func() {
	return Iact_func;
}

std::function<Matrix<double>(const Matrix<double>&)> LSTM::get_Kact_func() {
	return Kact_func;
}

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> LSTM::get_dOact_func() {
	return dOact_func;
}

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> LSTM::get_dFact_func() {
	return dFact_func;
}

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> LSTM::get_dIact_func() {
	return dIact_func;
}

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> LSTM::get_dKact_func() {
	return dKact_func;
}

double LSTM::get_max_abs_change_dependencies() {
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

	result = std::max(result, get_max_abs_matrix_value(xO_weight_change));
	result = std::max(result, get_max_abs_matrix_value(xF_weight_change));
	result = std::max(result, get_max_abs_matrix_value(xI_weight_change));
	result = std::max(result, get_max_abs_matrix_value(xK_weight_change));
	result = std::max(result, get_max_abs_matrix_value(hO_weight_change));
	result = std::max(result, get_max_abs_matrix_value(hF_weight_change));
	result = std::max(result, get_max_abs_matrix_value(hI_weight_change));
	result = std::max(result, get_max_abs_matrix_value(hK_weight_change));

	result = std::max(result, get_max_abs_matrix_value(Obias_change));
	result = std::max(result, get_max_abs_matrix_value(Fbias_change));
	result = std::max(result, get_max_abs_matrix_value(Ibias_change));
	result = std::max(result, get_max_abs_matrix_value(Kbias_change));


	result = std::max(result, get_max_abs_matrix_value(s_xO_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_xF_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_xI_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_xK_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_hO_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_hF_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_hI_weight_change));
	result = std::max(result, get_max_abs_matrix_value(s_hK_weight_change));

	result = std::max(result, get_max_abs_matrix_value(s_Obias_change));
	result = std::max(result, get_max_abs_matrix_value(s_Fbias_change));
	result = std::max(result, get_max_abs_matrix_value(s_Ibias_change));
	result = std::max(result, get_max_abs_matrix_value(s_Kbias_change));


	result = std::max(result, get_max_abs_matrix_value(ss_xO_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_xF_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_xI_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_xK_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_hO_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_hF_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_hI_weight_change));
	result = std::max(result, get_max_abs_matrix_value(ss_hK_weight_change));

	result = std::max(result, get_max_abs_matrix_value(ss_Obias_change));
	result = std::max(result, get_max_abs_matrix_value(ss_Fbias_change));
	result = std::max(result, get_max_abs_matrix_value(ss_Ibias_change));
	result = std::max(result, get_max_abs_matrix_value(ss_Kbias_change));


	result = std::max(result, get_max_abs_matrix_value(init_c_change));
	result = std::max(result, get_max_abs_matrix_value(init_h_change));

	result = std::max(result, get_max_abs_matrix_value(s_init_c_change));
	result = std::max(result, get_max_abs_matrix_value(s_init_h_change));

	result = std::max(result, get_max_abs_matrix_value(ss_init_c_change));
	result = std::max(result, get_max_abs_matrix_value(ss_init_h_change));


	return result;
}



void LSTM::save_as(std::ofstream& output_file) {
	xO_weight.save_as(output_file);
	xF_weight.save_as(output_file);
	xI_weight.save_as(output_file);
	xK_weight.save_as(output_file);
	hO_weight.save_as(output_file);
	hF_weight.save_as(output_file);
	hI_weight.save_as(output_file);
	hK_weight.save_as(output_file);

	Obias.save_as(output_file);
	Fbias.save_as(output_file);
	Ibias.save_as(output_file);
	Kbias.save_as(output_file);

	init_c.save_as(output_file);
	init_h.save_as(output_file);
}

void LSTM::load(std::ifstream& input_file) {
	xO_weight.load(input_file);
	xF_weight.load(input_file);
	xI_weight.load(input_file);
	xK_weight.load(input_file);
	hO_weight.load(input_file);
	hF_weight.load(input_file);
	hI_weight.load(input_file);
	hK_weight.load(input_file);

	Obias.load(input_file);
	Fbias.load(input_file);
	Ibias.load(input_file);
	Kbias.load(input_file);

	init_c.load(input_file);
	init_h.load(input_file);
}



void LSTM::print_weight() {
	std::cout << "--------------LSTM Layer----------\n\n";
	print_xI_weight(); std::cout << std::endl;
	print_xF_weight(); std::cout << std::endl;
	print_xK_weight(); std::cout << std::endl;
	print_xO_weight(); std::cout << std::endl;
	print_hI_weight(); std::cout << std::endl;
	print_hF_weight(); std::cout << std::endl;
	print_hK_weight(); std::cout << std::endl;
	print_hO_weight(); std::cout << std::endl;
}

void LSTM::print_bias() {
	std::cout << "--------------LSTM Layer----------\n\n";
	print_Ibias(); std::cout << std::endl;
	print_Fbias(); std::cout << std::endl;
	print_Kbias(); std::cout << std::endl;
	print_Obias(); std::cout << std::endl;

	print_init();
}

void LSTM::print_value() {
	Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);
	Matrix<double> forgot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);
	Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);
	Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);
	std::cout << "--------------LSTM Layer----------\n\n";
	std::cout << "--------value--------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << value[i][0] << "    \t";
	}std::cout << std::endl;
	std::cout << "--------input--------\n";
	for (int i = 0; i < input_gate.get_row(); i++) {
		std::cout << input_gate[i][0] << "    \t";
	}std::cout << std::endl;
	std::cout << "--------forgot--------\n";
	for (int i = 0; i < forgot_gate.get_row(); i++) {
		std::cout << forgot_gate[i][0] << "    \t";
	}std::cout << std::endl;
	std::cout << "--------output-------\n";
	for (int i = 0; i < output_gate.get_row(); i++) {
		std::cout << output_gate[i][0] << "    \t";
	}std::cout << std::endl;
	std::cout << "----------K----------\n";
	for (int i = 0; i < K.get_row(); i++) {
		std::cout << K[i][0] << "    \t";
	}std::cout << std::endl;
}

void LSTM::print_xO_weight() {
	std::cout << "  -----x-output weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << xO_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_xF_weight() {
	std::cout << "  -----x-forgot weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << xF_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_xI_weight() {
	std::cout << "  -----x-input weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << xI_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_xK_weight() {
	std::cout << "  -----x-k    weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << xK_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_hO_weight() {
	std::cout << "  -----h-output weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << hO_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_hF_weight() {
	std::cout << "  -----h-forgot weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << hF_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_hI_weight() {
	std::cout << "  -----h-input weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << hI_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_hK_weight() {
	std::cout << "  -----h-K     weight----\n";
	for (int i = 0; i < value.get_row(); i++) {
		for (int j = 0; j < value.get_row(); j++) {
			std::cout << hK_weight[i][j] << "    \t";
		}
		std::cout << std::endl;
	}
}

void LSTM::print_Obias() {
	std::cout << "   ---output bias------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << Obias[i][0] << "    \t";
	}std::cout << std::endl;
}

void LSTM::print_Fbias() {
	std::cout << "   ---forgot bias------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << Fbias[i][0] << "    \t";
	}std::cout << std::endl;
}

void LSTM::print_Ibias() {
	std::cout << "   ---input bias------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << Ibias[i][0] << "    \t";
	}std::cout << std::endl;
}

void LSTM::print_Kbias() {
	std::cout << "   ---K     bias------\n";
	for (int i = 0; i < value.get_row(); i++) {
		std::cout << Kbias[i][0] << "    \t";
	}std::cout << std::endl;
}

void LSTM::print_init() {
	std::cout << " -------- init----------\n";
	for (int i = 0; i < init_c.get_row(); i++) {
		for (int j = 0; j < init_c.get_column(); j++) {
			std::cout << init_c[i][j] << "    \t";
		}std::cout << std::endl;
	}

	for (int i = 0; i < init_h.get_row(); i++) {
		for (int j = 0; j < init_h.get_column(); j++) {
			std::cout << init_h[i][j] << "    \t";
		}std::cout << std::endl;
	}
}


void LSTM::set_optimizer(const Layer::opt& _optimizer) {
	switch (optimizer) {
	case Layer::SGD: break;
	case Layer::MOMENTUM :s_xO_weight_change.reconstruct(0,0);
		s_xF_weight_change.reconstruct(0, 0);
		s_xI_weight_change.reconstruct(0, 0);
		s_xK_weight_change.reconstruct(0, 0);
		s_hO_weight_change.reconstruct(0, 0);
		s_hF_weight_change.reconstruct(0, 0);
		s_hI_weight_change.reconstruct(0, 0);
		s_hK_weight_change.reconstruct(0, 0);

		s_Obias_change.reconstruct(0, 0);
		s_Fbias_change.reconstruct(0, 0);
		s_Ibias_change.reconstruct(0, 0);
		s_Kbias_change.reconstruct(0, 0);

		s_init_c_change.reconstruct(0, 0);
		s_init_h_change.reconstruct(0, 0);
		break;
	case Layer::ADAM :
		s_xO_weight_change.reconstruct(0, 0);
		s_xF_weight_change.reconstruct(0, 0);
		s_xI_weight_change.reconstruct(0, 0);
		s_xK_weight_change.reconstruct(0, 0);
		s_hO_weight_change.reconstruct(0, 0);
		s_hF_weight_change.reconstruct(0, 0);
		s_hI_weight_change.reconstruct(0, 0);
		s_hK_weight_change.reconstruct(0, 0);

		s_Obias_change.reconstruct(0, 0);
		s_Fbias_change.reconstruct(0, 0);
		s_Ibias_change.reconstruct(0, 0);
		s_Kbias_change.reconstruct(0, 0);

		s_init_c_change.reconstruct(0, 0);
		s_init_h_change.reconstruct(0, 0);


		ss_xO_weight_change.reconstruct(0, 0);
		ss_xF_weight_change.reconstruct(0, 0);
		ss_xI_weight_change.reconstruct(0, 0);
		ss_xK_weight_change.reconstruct(0, 0);
		ss_hO_weight_change.reconstruct(0, 0);
		ss_hF_weight_change.reconstruct(0, 0);
		ss_hI_weight_change.reconstruct(0, 0);
		ss_hK_weight_change.reconstruct(0, 0);

		ss_Obias_change.reconstruct(0, 0);
		ss_Fbias_change.reconstruct(0, 0);
		ss_Ibias_change.reconstruct(0, 0);
		ss_Kbias_change.reconstruct(0, 0);

		ss_init_c_change.reconstruct(0, 0);
		ss_init_h_change.reconstruct(0, 0);
		break;
	}

	optimizer = _optimizer;
	switch (optimizer) {
	case Layer::SGD: break;
	case Layer::MOMENTUM:
		s_xO_weight_change.reconstruct(xO_weight_change.get_row(), xO_weight_change.get_column());
		s_xF_weight_change.reconstruct(xF_weight_change.get_row(), xF_weight_change.get_column());
		s_xI_weight_change.reconstruct(xI_weight_change.get_row(), xI_weight_change.get_column());
		s_xK_weight_change.reconstruct(xK_weight_change.get_row(), xK_weight_change.get_column());
		s_hO_weight_change.reconstruct(hO_weight_change.get_row(), hO_weight_change.get_column());
		s_hF_weight_change.reconstruct(hF_weight_change.get_row(), hF_weight_change.get_column());
		s_hI_weight_change.reconstruct(hI_weight_change.get_row(), hI_weight_change.get_column());
		s_hK_weight_change.reconstruct(hK_weight_change.get_row(), hK_weight_change.get_column());
		s_xO_weight_change *= 0;
		s_xF_weight_change *= 0;
		s_xI_weight_change *= 0;
		s_xK_weight_change *= 0;
		s_hO_weight_change *= 0;
		s_hF_weight_change *= 0;
		s_hI_weight_change *= 0;
		s_hK_weight_change *= 0;

		s_Obias_change.reconstruct(Obias_change.get_row(), Obias_change.get_column());
		s_Fbias_change.reconstruct(Fbias_change.get_row(), Fbias_change.get_column());
		s_Ibias_change.reconstruct(Ibias_change.get_row(), Ibias_change.get_column());
		s_Kbias_change.reconstruct(Kbias_change.get_row(), Kbias_change.get_column());
		s_Obias_change *= 0;
		s_Fbias_change *= 0;
		s_Ibias_change *= 0;
		s_Kbias_change *= 0;

		s_init_c_change.reconstruct(init_c_change.get_row(),init_c_change.get_column());
		s_init_h_change.reconstruct(init_h_change.get_row(),init_h_change.get_column());
		s_init_c_change *= 0;
		s_init_h_change *= 0;
		break;
	case Layer::ADAM:
		s_xO_weight_change.reconstruct(xO_weight_change.get_row(), xO_weight_change.get_column());
		s_xF_weight_change.reconstruct(xF_weight_change.get_row(), xF_weight_change.get_column());
		s_xI_weight_change.reconstruct(xI_weight_change.get_row(), xI_weight_change.get_column());
		s_xK_weight_change.reconstruct(xK_weight_change.get_row(), xK_weight_change.get_column());
		s_hO_weight_change.reconstruct(hO_weight_change.get_row(), hO_weight_change.get_column());
		s_hF_weight_change.reconstruct(hF_weight_change.get_row(), hF_weight_change.get_column());
		s_hI_weight_change.reconstruct(hI_weight_change.get_row(), hI_weight_change.get_column());
		s_hK_weight_change.reconstruct(hK_weight_change.get_row(), hK_weight_change.get_column());
		s_xO_weight_change *= 0;
		s_xF_weight_change *= 0;
		s_xI_weight_change *= 0;
		s_xK_weight_change *= 0;
		s_hO_weight_change *= 0;
		s_hF_weight_change *= 0;
		s_hI_weight_change *= 0;
		s_hK_weight_change *= 0;

		s_Obias_change.reconstruct(Obias_change.get_row(), Obias_change.get_column());
		s_Fbias_change.reconstruct(Fbias_change.get_row(), Fbias_change.get_column());
		s_Ibias_change.reconstruct(Ibias_change.get_row(), Ibias_change.get_column());
		s_Kbias_change.reconstruct(Kbias_change.get_row(), Kbias_change.get_column());
		s_Obias_change *= 0;
		s_Fbias_change *= 0;
		s_Ibias_change *= 0;
		s_Kbias_change *= 0;

		s_init_c_change.reconstruct(init_c_change.get_row(), init_c_change.get_column());
		s_init_h_change.reconstruct(init_h_change.get_row(), init_h_change.get_column());
		s_init_c_change *= 0;
		s_init_h_change *= 0;


		ss_xO_weight_change.reconstruct(xO_weight_change.get_row(), xO_weight_change.get_column());
		ss_xF_weight_change.reconstruct(xF_weight_change.get_row(), xF_weight_change.get_column());
		ss_xI_weight_change.reconstruct(xI_weight_change.get_row(), xI_weight_change.get_column());
		ss_xK_weight_change.reconstruct(xK_weight_change.get_row(), xK_weight_change.get_column());
		ss_hO_weight_change.reconstruct(hO_weight_change.get_row(), hO_weight_change.get_column());
		ss_hF_weight_change.reconstruct(hF_weight_change.get_row(), hF_weight_change.get_column());
		ss_hI_weight_change.reconstruct(hI_weight_change.get_row(), hI_weight_change.get_column());
		ss_hK_weight_change.reconstruct(hK_weight_change.get_row(), hK_weight_change.get_column());
		ss_xO_weight_change *= 0;
		ss_xF_weight_change *= 0;
		ss_xI_weight_change *= 0;
		ss_xK_weight_change *= 0;
		ss_hO_weight_change *= 0;
		ss_hF_weight_change *= 0;
		ss_hI_weight_change *= 0;
		ss_hK_weight_change *= 0;

		ss_Obias_change.reconstruct(Obias_change.get_row(), Obias_change.get_column());
		ss_Fbias_change.reconstruct(Fbias_change.get_row(), Fbias_change.get_column());
		ss_Ibias_change.reconstruct(Ibias_change.get_row(), Ibias_change.get_column());
		ss_Kbias_change.reconstruct(Kbias_change.get_row(), Kbias_change.get_column());
		ss_Obias_change *= 0;
		ss_Fbias_change *= 0;
		ss_Ibias_change *= 0;
		ss_Kbias_change *= 0;

		ss_init_c_change.reconstruct(init_c_change.get_row(), init_c_change.get_column());
		ss_init_h_change.reconstruct(init_h_change.get_row(), init_h_change.get_column());
		ss_init_c_change *= 0;
		ss_init_h_change *= 0;
		break;
	}
}



void LSTM::set_Layer(const std::string& setting) {
	int size = setting.size();
	int i = 0;

	auto set_optimizer_text = [&]() {
		std::string _optimizer = get_text(setting, i);
		if (_optimizer == "SGD")
			set_optimizer(Layer::SGD);
		else if (_optimizer == "MOMENTUM")
			set_optimizer(Layer::MOMENTUM);
		else if (_optimizer == "ADAM")
			set_optimizer(Layer::ADAM);
	};
	auto set_learning_rate_text = [&]() {
		double a = get_number(setting, i);
		set_learning_rate(a);
	};
	
	std::string a;
	while (i < size) {
		a = get_text(setting, i);
		if (a == "Iact")
			universal_set_func(Iact_func, setting, i);
		else if (a == "Fact")
			universal_set_func(Fact_func, setting, i);
		else if (a == "Oact")
			universal_set_func(Oact_func, setting, i);
		else if (a == "Kact")
			universal_set_func(Kact_func, setting, i);
		else if (a == "dIact")
			universal_set_func(dIact_func, setting, i);
		else if (a == "dFact")
			universal_set_func(dFact_func, setting, i);
		else if (a == "dOact")
			universal_set_func(dOact_func, setting, i);
		else if (a == "dKact")
			universal_set_func(dKact_func, setting, i);
		else if (a == "learning_rate")
			set_learning_rate_text();
		else if (a == "optimizer")
			set_optimizer_text();
		else if (a == "")
			;
		else throw std::runtime_error("command not found");
	}
}
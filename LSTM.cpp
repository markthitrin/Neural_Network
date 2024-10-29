#pragma once
#include "Header.h"
#include "Layer.cpp"
#include "LayerId.cpp"

#include "Func.h"


class LSTM : public Layer {
public:
	LSTM() { Layer_type = Layer::LSTM; };

	LSTM(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func = dtanh_func) :
		Iact_func(_Iact_func) ,dIact_func(_dIact_func),
		Fact_func(_Fact_func), dFact_func(_dFact_func), 
		Oact_func(_Oact_func), dOact_func(_dOact_func),
		Kact_func(_Kact_func), dKact_func(_dKact_func) {
		
		Layer_type = Layer::LSTM;

		value.reconstruct(size, 1);

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

		init_c_change.reconstruct(size, 1);
		init_h_change.reconstruct(size, 1);

		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		c.push_back(init_c);
		h.push_back(init_h);
	}

	LSTM(const LayerId& set) {
		value.reconstruct(set.Layer_size, 1);
		Layer_type = Layer::LSTM;

		xO_weight.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight.reconstruct(set.Layer_size, set.Layer_size);

		Obias.reconstruct(set.Layer_size, 1);
		Fbias.reconstruct(set.Layer_size, 1);
		Ibias.reconstruct(set.Layer_size, 1);
		Kbias.reconstruct(set.Layer_size, 1);

		xO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight_change.reconstruct(set.Layer_size, set.Layer_size);

		Obias_change.reconstruct(set.Layer_size, 1);
		Fbias_change.reconstruct(set.Layer_size, 1);
		Ibias_change.reconstruct(set.Layer_size, 1);
		Kbias_change.reconstruct(set.Layer_size, 1);

		init_c.reconstruct(set.Layer_size, 1);
		init_h.reconstruct(set.Layer_size, 1);

		init_c_change.reconstruct(set.Layer_size, 1);
		init_h_change.reconstruct(set.Layer_size, 1);

		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		c.push_back(init_c);
		h.push_back(init_h);

		Iact_func = tanh_func;
		Fact_func = sigmoid_func;
		Oact_func = sigmoid_func;
		Kact_func = tanh_func;

		dIact_func = dtanh_func;
		dFact_func = dsigmoid_func;
		dOact_func = dsigmoid_func;
		dKact_func = dtanh_func;

		set_Layer(set.setting);
	}


		
	Matrix<double> feed() {																						// feedforward
		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);			// compute input gate
		Matrix<double> forgot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);			// compute forgot gate
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);			// compute output gate
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);

		c.push_back(mul_each(forgot_gate, c.back()) + mul_each(input_gate, K));									// compute and remember cell state
		h.push_back(mul_each(output_gate, c.back()));															// compute and remember output fo the cell
		v.push_back(value);																						// remember given input

		return h.back();																						// return output
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient) {						// backpropagation
		if (gadient.size() > c.size())
			throw "invalid gaadient  size for  backpropagatoin lstm";

		Matrix<double> dc(value.get_row(), 1);																	// containing error of cell state for each time step
		Matrix<double> dh(value.get_row(), 1);																	// containing error of output for each time step

		Matrix<double> next_dc(value.get_row(), 1);																// containing error of next time step
		Matrix<double> next_dh(value.get_row(), 1);

		set_Matrix(dc, 0);
		set_Matrix(dh, 0);

		std::vector<Matrix<double>> _gadient;
		std::vector<Matrix<double>> flow_gadient;



		for (int i = 0; i < v.size(); i++) {																	// rearrange the gadient and put into _gadient
			flow_gadient.push_back(Matrix<double>(value.get_row(), 1)); set_Matrix(flow_gadient.back(), 0); 
		}

		for (int i = 0; i + gadient.size() < v.size(); i++) {
			_gadient.push_back(Matrix<double>(value.get_row(), 1));
			set_Matrix(_gadient.back(),0);
		}

		for (int i = 0; i < gadient.size(); i++) {
			_gadient.push_back(gadient[i]);
		}



		for (int round = v.size() - 1; round >= 0; round--) {													// loop thougj eery time step
			set_Matrix(next_dc, 0);
			set_Matrix(next_dh, 0);

			Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h[round]) + Ibias);		// compute input gate
			Matrix<double> forgot_gate = Fact_func((xF_weight * value) + (hF_weight * h[round]) + Fbias);		// compute forgot gate
			Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h[round]) + Obias);		// comput output gate
			Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h[round]) + Kbias);

			dh = dh + _gadient[round];																			// add up output error
			dc = dc + mul_each(dh, output_gate);																// add up cell state error

			Matrix<double> dinput_gate = dIact_func((xI_weight * value) + (hI_weight * h[round]) + Ibias, mul_each(dc, K));// derivative of input gate
			Matrix<double> dforgot_gate = dFact_func((xF_weight * value) + (hF_weight * h[round]) + Fbias, mul_each(dc, c[round]));// derivative of forgot gate
			Matrix<double> doutput_gate = dOact_func((xO_weight * value) + (hO_weight * h[round]) + Obias, mul_each(dh, c[round + 1]));// derivative of output
			Matrix<double> dK = dKact_func((xK_weight * value) + (hK_weight * h[round]) + Kbias, mul_each(dc, input_gate));

			for (int i = 0; i < value.get_row(); i++) {															// loop though every output row
				for (int j = 0; j < value.get_row(); j++) {														// loop though every input row
					xO_weight_change[i][j] += doutput_gate[i][0] * v[round][j][0];				// comupute changing weight
					hO_weight_change[i][j] += doutput_gate[i][0] * h[round][j][0];
					xI_weight_change[i][j] += dinput_gate[i][0] * v[round][j][0];
					hI_weight_change[i][j] += dinput_gate[i][0] * h[round][j][0];
					xF_weight_change[i][j] += dforgot_gate[i][0] * v[round][j][0];
					hF_weight_change[i][j] += dforgot_gate[i][0] * h[round][j][0];
					xK_weight_change[i][j] += dK[i][0] * v[round][j][0];
					hK_weight_change[i][j] += dK[i][0] * h[round][j][0];

					next_dh[j][0] += doutput_gate[i][0] * hO_weight_change[i][j];								// comupute next time step output error
					next_dh[j][0] += dinput_gate[i][0] * hI_weight_change[i][j];
					next_dh[j][0] += dforgot_gate[i][0] * hF_weight_change[i][j];
					next_dh[j][0] += dK[i][0] * hK_weight_change[i][j];

					flow_gadient[round][j][0] += doutput_gate[i][0] * xO_weight_change[i][j];					// computer flow gadient
					flow_gadient[round][j][0] += dinput_gate[i][0] * xI_weight_change[i][j];
					flow_gadient[round][j][0] += dforgot_gate[i][0] * xF_weight_change[i][j];
					flow_gadient[round][j][0] += dK[i][0] * hK_weight_change[i][j];
				
				}
				Obias_change[i][0] += doutput_gate[i][0];										// compute changing bias
				Ibias_change[i][0] += dinput_gate[i][0];
				Fbias_change[i][0] += dforgot_gate[i][0];
				Kbias_change[i][0] += dK[i][0];

			}

			next_dc = mul_each(dc, forgot_gate);																	// compute next time step cell state error

			dh = next_dh;																						
			dc = next_dc;
			
			// try to descale exploding gadient
			double max_dh_value = std::max(get_max(dh), std::abs(get_min(dh)));									
			double max_dc_value = std::max(get_max(dc), std::abs(get_min(dc)));
			
			double flow_cap = std::sqrt(double(2) / v.size());
			if (max_dh_value > flow_cap) dh = dh * (flow_cap / max_dh_value);
			if (max_dc_value > flow_cap) dc = dc * (flow_cap / max_dc_value);
		}

		for (int i = 0; i < value.get_row(); i++) {																// compute initial cell state
			init_h_change[i][0] += dh[i][0];
			init_c_change[i][0] += dc[i][0];
		}

		return flow_gadient;
	}
	


	void forgot(const std::size_t& number) {
		std::size_t _number = number;
		if (number > v.size())
			_number = v.size();
		for (int i = 0; i < v.size() - _number; i++) {
			c[i] = c[i + _number];
			h[i] = h[i + _number];
			v[i] = v[i + _number];
		}
		for (int i = 0; i < _number; i++) {
			v.pop_back();
			h.pop_back();
			c.pop_back();
		}
		if (c.size() == 0) {
			c.push_back(init_c);
			h.push_back(init_h);
		}
	}

	void forgot_all() {
		forgot(v.size());
	}



	void change_dependencies() {
		++t;
		if (optimizer == Layer::SGD) {
			xO_weight = xO_weight_change * learning_rate + xO_weight;
			xF_weight = xF_weight_change * learning_rate + xF_weight;
			xI_weight = xI_weight_change * learning_rate + xI_weight;
			xK_weight = xK_weight_change * learning_rate + xK_weight;
			hO_weight = hO_weight_change * learning_rate + hO_weight;
			hF_weight = hF_weight_change * learning_rate + hF_weight;
			hI_weight = hI_weight_change * learning_rate + hI_weight;
			hK_weight = hK_weight_change * learning_rate + hK_weight;

			Obias = Obias_change * learning_rate + Obias;
			Fbias = Fbias_change * learning_rate + Fbias;
			Ibias = Ibias_change * learning_rate + Ibias;
			Kbias = Kbias_change * learning_rate + Kbias;

			init_c = init_c + init_c_change * learning_rate;
			init_h = init_h + init_h_change * learning_rate;
		}
		else if (optimizer == Layer::MOMENTUM) {
			set_up_Matrix(s_xO_weight_change, xO_weight);
			set_up_Matrix(s_xF_weight_change, xF_weight);
			set_up_Matrix(s_xI_weight_change, xI_weight);
			set_up_Matrix(s_xK_weight_change, xK_weight);
			set_up_Matrix(s_hO_weight_change, hO_weight);
			set_up_Matrix(s_hF_weight_change, hF_weight);
			set_up_Matrix(s_hI_weight_change, hI_weight);
			set_up_Matrix(s_hK_weight_change, hK_weight);

			set_up_Matrix(s_Obias_change, Obias);
			set_up_Matrix(s_Fbias_change, Fbias);
			set_up_Matrix(s_Ibias_change, Ibias);
			set_up_Matrix(s_Kbias_change, Kbias);

			set_up_Matrix(s_init_c_change, init_c);
			set_up_Matrix(s_init_h_change, init_h);

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

			xO_weight = s_xO_weight_change * learning_rate + xO_weight;
			xF_weight = s_xF_weight_change * learning_rate + xF_weight;
			xI_weight = s_xI_weight_change * learning_rate + xI_weight;
			xK_weight = s_xK_weight_change * learning_rate + xK_weight;
			hO_weight = s_hO_weight_change * learning_rate + hO_weight;
			hF_weight = s_hF_weight_change * learning_rate + hF_weight;
			hI_weight = s_hI_weight_change * learning_rate + hI_weight;
			hK_weight = s_hK_weight_change * learning_rate + hK_weight;

			Obias = s_Obias_change * learning_rate + Obias;
			Fbias = s_Fbias_change * learning_rate + Fbias;
			Ibias = s_Ibias_change * learning_rate + Ibias;
			Kbias = s_Kbias_change * learning_rate + Kbias;

			init_c = init_c + s_init_c_change * learning_rate;
			init_h = init_h + s_init_h_change * learning_rate;
		}
		else if (optimizer == Layer::ADAM) {
			set_up_Matrix(s_xO_weight_change, xO_weight);
			set_up_Matrix(s_xF_weight_change, xF_weight);
			set_up_Matrix(s_xI_weight_change, xI_weight);
			set_up_Matrix(s_xK_weight_change, xK_weight);
			set_up_Matrix(s_hO_weight_change, hO_weight);
			set_up_Matrix(s_hF_weight_change, hF_weight);
			set_up_Matrix(s_hI_weight_change, hI_weight);
			set_up_Matrix(s_hK_weight_change, hK_weight);

			set_up_Matrix(s_Obias_change, Obias);
			set_up_Matrix(s_Fbias_change, Fbias);
			set_up_Matrix(s_Ibias_change, Ibias);
			set_up_Matrix(s_Kbias_change, Kbias);

			set_up_Matrix(s_init_c_change, init_c);
			set_up_Matrix(s_init_h_change, init_h);

			set_up_Matrix(ss_xO_weight_change, xO_weight);
			set_up_Matrix(ss_xF_weight_change, xF_weight);
			set_up_Matrix(ss_xI_weight_change, xI_weight);
			set_up_Matrix(ss_xK_weight_change, xK_weight);
			set_up_Matrix(ss_hO_weight_change, hO_weight);
			set_up_Matrix(ss_hF_weight_change, hF_weight);
			set_up_Matrix(ss_hI_weight_change, hI_weight);
			set_up_Matrix(ss_hK_weight_change, hK_weight);

			set_up_Matrix(ss_Obias_change, Obias);
			set_up_Matrix(ss_Fbias_change, Fbias);
			set_up_Matrix(ss_Ibias_change, Ibias);
			set_up_Matrix(ss_Kbias_change, Kbias);

			set_up_Matrix(ss_init_c_change, init_c);
			set_up_Matrix(ss_init_h_change, init_h);

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
			ss_xF_weight_change = ss_xF_weight_change * decay_rate + mul_each(xF_weight_change, xF_weight_change)* (double(1) - decay_rate);
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

			xO_weight = xO_weight + devide_each(s_xO_weight_change * learning_rate , pow(ss_xO_weight_change, 0.5) + 0.0000001);
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
		}
	}

	void set_change_dependencies(const double& number) {
		set_Matrix(xO_weight_change,number);
		set_Matrix(xF_weight_change,number);
		set_Matrix(xI_weight_change,number);
		set_Matrix(xK_weight_change,number);
		set_Matrix(hO_weight_change,number);
		set_Matrix(hF_weight_change,number);
		set_Matrix(hI_weight_change,number);
		set_Matrix(hK_weight_change,number);

		set_Matrix(Obias_change,number);
		set_Matrix(Fbias_change,number);
		set_Matrix(Ibias_change,number);
		set_Matrix(Kbias_change,number);

		set_Matrix(init_c_change,number);
		set_Matrix(init_h_change,number);
	}

	void mul_change_dependencies(const double& number) {
		xO_weight_change = xO_weight_change * number;
		xF_weight_change = xF_weight_change * number;
		xI_weight_change = xI_weight_change * number;
		xK_weight_change = xK_weight_change * number;
		hO_weight_change = hO_weight_change * number;
		hF_weight_change = hF_weight_change * number;
		hI_weight_change = hI_weight_change * number;
		hK_weight_change = hK_weight_change * number;

		Obias_change = Obias_change * number;
		Fbias_change = Fbias_change * number;
		Ibias_change = Ibias_change * number;
		Kbias_change = Kbias_change * number;

		init_c_change = init_c_change * number;
		init_h_change = init_h_change * number;
	}



	void reconstruct(const std::size_t& size, const std::size_t& next,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func = dtanh_func) {
		value.reconstruct(size, 1);

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
		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		forgot_all();
		c.push_back(init_c);
		h.push_back(init_h);
	}

	void reconstruct(const LayerId& set) {
		value.reconstruct(set.Layer_size, 1);
		Layer_type = Layer::LSTM;

		xO_weight.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight.reconstruct(set.Layer_size, set.Layer_size);

		Obias.reconstruct(set.Layer_size, 1);
		Fbias.reconstruct(set.Layer_size, 1);
		Ibias.reconstruct(set.Layer_size, 1);
		Kbias.reconstruct(set.Layer_size, 1);

		xO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		xK_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hO_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hF_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hI_weight_change.reconstruct(set.Layer_size, set.Layer_size);
		hK_weight_change.reconstruct(set.Layer_size, set.Layer_size);

		Obias_change.reconstruct(set.Layer_size, 1);
		Fbias_change.reconstruct(set.Layer_size, 1);
		Ibias_change.reconstruct(set.Layer_size, 1);
		Kbias_change.reconstruct(set.Layer_size, 1);

		init_c.reconstruct(set.Layer_size, 1);
		init_h.reconstruct(set.Layer_size, 1);

		init_c_change.reconstruct(set.Layer_size, 1);
		init_h_change.reconstruct(set.Layer_size, 1);
		set_Matrix(init_c, 0);
		set_Matrix(init_h, 0);

		c.push_back(init_c);
		h.push_back(init_h);

		Iact_func = tanh_func;
		Fact_func = sigmoid_func;
		Oact_func = sigmoid_func;
		Kact_func = tanh_func;

		dIact_func = dtanh_func;
		dFact_func = dsigmoid_func;
		dOact_func = dsigmoid_func;
		dKact_func = dtanh_func;

		set_Layer(set.setting);
	}



	void rand_weight(const double& min,const double& max) {
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

	void rand_weight(std::pair<const double&, const double&> setting) {
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
	
	void rand_weight(std::function<double()> func) {
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

	void rand_weight(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				xO_weight[i][j] = func(value.get_row(),next);
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

	void rand_bias(const double& min, const double& max) {
		for (int i = 0; i < value.get_row(); i++) {
			Obias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
			Fbias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
			Ibias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
			Kbias[i][0] = mapping(rand() % 10000, 0, 10000, min, max);
		}
	}

	void rand_bias(std::pair<const double&,const double&> setting) {
		for (int i = 0; i < value.get_row(); i++) {
			Obias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			Fbias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			Ibias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			Kbias[i][0] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		}
	}
	
	void rand_bias(std::function<double()> func) {
		for (int i = 0; i < value.get_row(); i++) {
			Obias[i][0] = func();
			Fbias[i][0] = func();
			Ibias[i][0] = func();
			Kbias[i][0] = func();
		}
	}

	void rand_bias(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		for (int i = 0; i < value.get_row(); i++) {
			Obias[i][0] = func(value.get_row(),next);
			Fbias[i][0] = func(value.get_row(), next);
			Ibias[i][0] = func(value.get_row(), next);
			Kbias[i][0] = func(value.get_row(), next);
		}
	}



	std::function<Matrix<double>(const Matrix<double>&)> get_Oact_func() {
		return Oact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&)> get_Fact_func() {
		return Fact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&)> get_Iact_func() {
		return Iact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&)> get_Kact_func() {
		return Kact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dOact_func() {
		return dOact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dFact_func() {
		return dFact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dIact_func() {
		return dIact_func;
	}

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dKact_func() {
		return dKact_func;
	}



	void print_weight() {
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

	void print_bias() {
		std::cout << "--------------LSTM Layer----------\n\n";
		print_Ibias(); std::cout << std::endl;
		print_Fbias(); std::cout << std::endl;
		print_Kbias(); std::cout << std::endl;
		print_Obias(); std::cout << std::endl;

		print_init();
	}

	void print_value() {
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
protected:
	void print_xO_weight() {
		std::cout << "  -----x-output weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xO_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_xF_weight() {
		std::cout << "  -----x-forgot weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xF_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_xI_weight() {
		std::cout << "  -----x-input weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xI_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_xK_weight() {
		std::cout << "  -----x-k    weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << xK_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_hO_weight() {
		std::cout << "  -----h-output weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hO_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_hF_weight() {
		std::cout << "  -----h-forgot weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hF_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_hI_weight() {
		std::cout << "  -----h-input weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hI_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_hK_weight() {
		std::cout << "  -----h-K     weight----\n";
		for (int i = 0; i < value.get_row(); i++) {
			for (int j = 0; j < value.get_row(); j++) {
				std::cout << hK_weight[i][j] << "    \t";
			}
			std::cout << std::endl;
		}
	}

	void print_Obias() {
		std::cout << "   ---output bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Obias[i][0] << "    \t";
		}std::cout << std::endl;
	}

	void print_Fbias() {
		std::cout << "   ---forgot bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Fbias[i][0] << "    \t";
		}std::cout << std::endl;
	}

	void print_Ibias() {
		std::cout << "   ---input bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Ibias[i][0] << "    \t";
		}std::cout << std::endl;
	}

	void print_Kbias() {
		std::cout << "   ---K     bias------\n";
		for (int i = 0; i < value.get_row(); i++) {
			std::cout << Kbias[i][0] << "    \t";
		}std::cout << std::endl;
	}

	void print_init() {
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



	void set_Layer(const std::string& setting) {
		int size = setting.size();
		int i = 0;
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
				set_learning_rate(setting, i);
			else if (a == "optimizer")
				set_optimizer(setting, i);
			else if (a == "")
				;
			else throw "command not found";
		}
	}

	void set_optimizer(const std::string& str, int& i) {
		std::string _optimizer = get_text(str, i);
		if (_optimizer == "SGD") {
			optimizer = Layer::SGD;
		}
		else if (_optimizer == "MOMENTUM") {
			optimizer = Layer::MOMENTUM;
		}
		else if (_optimizer == "ADAM") {
			optimizer = Layer::ADAM;
		}
	}

	void set_up_Matrix(Matrix<double>& M, const Matrix<double>& B) {
		if (!M.is_constructed()) {
			M.reconstruct(B.get_row(), B.get_column());
			set_Matrix(M, 0);
		}
	}

	void set_learning_rate(const std::string& str, int& i) {
		double a = get_number(str, i);
		learning_rate = a;
	}


	Matrix<double> xO_weight;																				// weight for input -> output gate
	Matrix<double> xF_weight;																				// weight for input -> forgot gate
	Matrix<double> xI_weight;																				// weight for input -> input gate
	Matrix<double> xK_weight;																				// weight for input -> K
	Matrix<double> hO_weight;																				// weight for hidden -> output gate
	Matrix<double> hF_weight;																				// weight for hidden -> forgot gate
	Matrix<double> hI_weight;																				// weight for hidden -> input gate
	Matrix<double> hK_weight;																				// weight for hidden -> K

	Matrix<double> Obias;																					// bias for output gate
	Matrix<double> Fbias;																					// bias for forgot gate	
	Matrix<double> Ibias;																					// bias for input gate
	Matrix<double> Kbias;																					// bias for K

	Matrix<double> init_c;																					// initial cell state
	Matrix<double> init_h;																					// initial hidden

	Matrix<double> s_xO_weight_change;
	Matrix<double> s_xF_weight_change;
	Matrix<double> s_xI_weight_change;
	Matrix<double> s_xK_weight_change;
	Matrix<double> s_hO_weight_change;
	Matrix<double> s_hF_weight_change;
	Matrix<double> s_hI_weight_change;
	Matrix<double> s_hK_weight_change;

	Matrix<double> s_Obias_change;
	Matrix<double> s_Fbias_change;
	Matrix<double> s_Ibias_change;
	Matrix<double> s_Kbias_change;

	Matrix<double> s_init_c_change;
	Matrix<double> s_init_h_change;

	Matrix<double> ss_xO_weight_change;
	Matrix<double> ss_xF_weight_change;
	Matrix<double> ss_xI_weight_change;
	Matrix<double> ss_xK_weight_change;
	Matrix<double> ss_hO_weight_change;
	Matrix<double> ss_hF_weight_change;
	Matrix<double> ss_hI_weight_change;
	Matrix<double> ss_hK_weight_change;

	Matrix<double> ss_Obias_change;
	Matrix<double> ss_Fbias_change;
	Matrix<double> ss_Ibias_change;
	Matrix<double> ss_Kbias_change;

	Matrix<double> ss_init_c_change;
	Matrix<double> ss_init_h_change;

	Matrix<double> xO_weight_change;																		
	Matrix<double> xF_weight_change;																		
	Matrix<double> xI_weight_change;																		
	Matrix<double> xK_weight_change;																		
	Matrix<double> hO_weight_change;																		
	Matrix<double> hF_weight_change;																		
	Matrix<double> hI_weight_change;																		
	Matrix<double> hK_weight_change;																		

	Matrix<double> Obias_change;																																		
	Matrix<double> Fbias_change;																			
	Matrix<double> Ibias_change;																			
	Matrix<double> Kbias_change;																			

	Matrix<double> init_c_change;																			
	Matrix<double> init_h_change;																			
	
	std::vector<Matrix<double>> c;																			// cell state memory
	std::vector<Matrix<double>> h;																			// output memory

	std::function<Matrix<double>(const Matrix<double>&)> Oact_func = sigmoid_func;							// activate function for output gate
	std::function<Matrix<double>(const Matrix<double>&)> Fact_func = sigmoid_func;							// activate function for forgot gate
	std::function<Matrix<double>(const Matrix<double>&)> Iact_func = tanh_func;								// activate function for input gate
	std::function<Matrix<double>(const Matrix<double>&)> Kact_func = tanh_func;								// activate function for K

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dOact_func = dsigmoid_func;	// derivative of activate function for output gate
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dFact_func = dsigmoid_func;	// derivative of activate function for forgot gate
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dIact_func = dtanh_func;	// derivative of activate function for input gate
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dKact_func = dtanh_func;	// derivative of activate function for K
};
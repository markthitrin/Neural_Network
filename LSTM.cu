#pragma once
#include "Header.cuh"
#include "Layer.cu"
#include "LayerId.cu"

#include "Func.cuh"
#include "Variable.cuh"

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
		Layer(Layer::LSTM, size),
		Iact_func(_Iact_func) ,dIact_func(_dIact_func),
		Fact_func(_Fact_func), dFact_func(_dFact_func), 
		Oact_func(_Oact_func), dOact_func(_dOact_func),
		Kact_func(_Kact_func), dKact_func(_dKact_func) {
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

		set_ranndom_state();
	}

	LSTM(const LayerId& set) : Layer(Layer::LSTM, set.Layer_size) {
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

		set_ranndom_state();

		set_Layer(set.setting);
	}

	LSTM(const LSTM& copy) : Layer(copy) {
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

		init_c.reconstruct(copy.init_c);
		init_h.reconstruct(copy.init_h);

		init_c_change.reconstruct(copy.init_c_change);
		init_h_change.reconstruct(copy.init_h_change);

		c.push_back(init_c);
		h.push_back(init_h);

		Iact_func = copy.Iact_func;
		Fact_func = copy.Fact_func;
		Oact_func = copy.Oact_func;
		Kact_func = copy.Kact_func;

		dIact_func = copy.dIact_func;
		dFact_func = copy.dFact_func;
		dOact_func = copy.dOact_func;
		dKact_func = copy.dKact_func;
		
		set_ranndom_state();
	}

	LSTM(LSTM&& other) : Layer(std::move(other)) {
		xO_weight.reconstruct(std::move(other.xO_weight));
		xF_weight.reconstruct(std::move(other.xF_weight));
		xI_weight.reconstruct(std::move(other.xI_weight));
		xK_weight.reconstruct(std::move(other.xK_weight));
		hO_weight.reconstruct(std::move(other.hO_weight));
		hF_weight.reconstruct(std::move(other.hF_weight));
		hI_weight.reconstruct(std::move(other.hI_weight));
		hK_weight.reconstruct(std::move(other.hK_weight));

		Obias.reconstruct(std::move(other.Obias));
		Fbias.reconstruct(std::move(other.Fbias));
		Ibias.reconstruct(std::move(other.Ibias));
		Kbias.reconstruct(std::move(other.Kbias));

		xO_weight_change.reconstruct(std::move(other.xO_weight_change));
		xF_weight_change.reconstruct(std::move(other.xF_weight_change));
		xI_weight_change.reconstruct(std::move(other.xI_weight_change));
		xK_weight_change.reconstruct(std::move(other.xK_weight_change));
		hO_weight_change.reconstruct(std::move(other.hO_weight_change));
		hF_weight_change.reconstruct(std::move(other.hF_weight_change));
		hI_weight_change.reconstruct(std::move(other.hI_weight_change));
		hK_weight_change.reconstruct(std::move(other.hK_weight_change));

		Obias_change.reconstruct(std::move(other.Obias_change));
		Fbias_change.reconstruct(std::move(other.Fbias_change));
		Ibias_change.reconstruct(std::move(other.Ibias_change));
		Kbias_change.reconstruct(std::move(other.Kbias_change));

		init_c.reconstruct(std::move(other.init_c));
		init_h.reconstruct(std::move(other.init_h));

		init_c_change.reconstruct(std::move(other.init_c_change));
		init_h_change.reconstruct(std::move(other.init_h_change));

		c = std::move(other.c);
		h = std::move(other.h);

		Iact_func = other.Iact_func;
		Fact_func = other.Fact_func;
		Oact_func = other.Oact_func;
		Kact_func = other.Kact_func;

		dIact_func = other.dIact_func;
		dFact_func = other.dFact_func;
		dOact_func = other.dOact_func;
		dKact_func = other.dKact_func;

		random_state = std::move(other.random_state);

		other.random_state = nullptr;
	}

	~LSTM() {
		cudaFree(random_state);
	}



	Matrix<double> feed() {																						// feedforward
		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);			// compute input gate
		Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);			// compute forgot gate
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);			// compute output gate
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);

		c.push_back(mul_each(fogot_gate, c.back()) + mul_each(input_gate, K));									// compute and remember cell state
		h.push_back(mul_each(output_gate, c.back()));															// compute and remember output fo the cell
		v.push_back(value);																						// remember given input

		return h.back();																						// return output
	}

	Matrix<double> predict() {
		Matrix<double> input_gate = Iact_func((xI_weight * value) + (hI_weight * h.back()) + Ibias);			// compute input gate
		Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);			// compute forgot gate
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);			// compute output gate
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);

		c.push_back(mul_each(fogot_gate, c.back()) + mul_each(input_gate, K));									// compute and remember cell state
		h.push_back(mul_each(output_gate, c.back()));															// compute and remember output fo the cell
		v.push_back(value);																						// remember given input

		return h.back();
	}

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gradient) {						// backpropagation
		if (gradient.size() > c.size())
			throw "invalid gaadient  size for  backpropagatoin lstm";

		Matrix<double> dc(value.row, 1);																	// containing error of cell state for each time step
		Matrix<double> dh(value.row, 1);																	// containing error of output for each time step

		Matrix<double> next_dc(value.row, 1);																// containing error of next time step
		Matrix<double> next_dh(value.row, 1);

		set_Matrix(dc, 0);
		set_Matrix(dh, 0);

		std::vector<Matrix<double>> _gradient;
		std::vector<Matrix<double>> flow_gradient;



		for (int i = 0; i < v.size(); i++) {																	// rearrange the gradient and put into _gradient
			flow_gradient.push_back(Matrix<double>(value.row, 1)); set_Matrix(flow_gradient.back(), 0); 
		}

		for (int i = 0; i + gradient.size() < v.size(); i++) {
			_gradient.push_back(Matrix<double>(value.row, 1));
			set_Matrix(_gradient.back(),0);
		}

		for (int i = 0; i < gradient.size(); i++) {
			_gradient.push_back(gradient[i]);
		}



		for (int round = v.size() - 1; round >= 0; round--) {
			set_Matrix(next_dc, 0);
			set_Matrix(next_dh, 0);

			Matrix<double> input_gate = Iact_func((xI_weight * v[round]) + (hI_weight * h[round]) + Ibias);
			Matrix<double> fogot_gate = Fact_func((xF_weight * v[round]) + (hF_weight * h[round]) + Fbias);
			Matrix<double> output_gate = Oact_func((xO_weight * v[round]) + (hO_weight * h[round]) + Obias);
			Matrix<double> K = Kact_func((xK_weight * v[round]) + (hK_weight * h[round]) + Kbias);

			dh = dh + _gradient[round];
			dc = dc + mul_each(dh, output_gate);

			Matrix<double> dinput_gate = dIact_func((xI_weight * v[round]) + (hI_weight * h[round]) + Ibias, mul_each(dc, K));
			Matrix<double> dfogot_gate = dFact_func((xF_weight * v[round]) + (hF_weight * h[round]) + Fbias, mul_each(dc, c[round]));
			Matrix<double> doutput_gate = dOact_func((xO_weight * v[round]) + (hO_weight * h[round]) + Obias, mul_each(dh, c[round + 1]));
			Matrix<double> dK = dKact_func((xK_weight * v[round]) + (hK_weight * h[round]) + Kbias, mul_each(dc, input_gate));


			int blockPergrid = upper_value(double(value.get_size() * value.get_size()) / 1024);
			int threadPerblock = std::min(value.get_size() * value.get_size(), 1024);

			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xO_weight_change.value, doutput_gate.value, v[round].value, xO_weight.row, xO_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hO_weight_change.value, doutput_gate.value, h[round].value, hO_weight.row, hO_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xI_weight_change.value, dinput_gate.value, v[round].value, xI_weight.row, xI_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hI_weight_change.value, dinput_gate.value, h[round].value, hI_weight.row, hI_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xF_weight_change.value, dfogot_gate.value, v[round].value, xF_weight.row, xF_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hF_weight_change.value, dfogot_gate.value, h[round].value, hF_weight.row, hF_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (xK_weight_change.value, dK.value, v[round].value, xK_weight.row, xK_weight.column);
			device_weightchange_computeLSTM << <blockPergrid, threadPerblock >> > (hK_weight_change.value, dK.value, h[round].value, hK_weight.row, hK_weight.column);
			cudaDeviceSynchronize();

			blockPergrid = upper_value(double(value.get_size()) / 1024);
			threadPerblock = std::min(value.get_size(), 1024);

			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (next_dh.value, doutput_gate.value, hO_weight.value, hO_weight.row, hO_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (next_dh.value, dinput_gate.value, hI_weight.value, hI_weight.row, hI_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (next_dh.value, dfogot_gate.value, hF_weight.value, hF_weight.row, hF_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (next_dh.value, dK.value, hK_weight.value, hK_weight.row, hK_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (flow_gradient[round].value, doutput_gate.value, xO_weight.value, xO_weight.row, hO_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (flow_gradient[round].value, dinput_gate.value, xI_weight.value, xI_weight.row, hI_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (flow_gradient[round].value, dfogot_gate.value, xF_weight.value, xF_weight.row, hF_weight.column);
			cudaDeviceSynchronize();
			device_flow_computeLSTM << <blockPergrid, threadPerblock >> > (flow_gradient[round].value, dK.value, xK_weight.value, xK_weight.row, hK_weight.column);
			cudaDeviceSynchronize();

			Obias_change = Obias_change + doutput_gate;
			Ibias_change = Ibias_change + dinput_gate;
			Fbias_change = Fbias_change + dfogot_gate;
			Kbias_change = Kbias_change + dK;

			next_dc = mul_each(dc, fogot_gate);

			dh = next_dh;																						
			dc = next_dc;
			
			// try to descale exploding gradient
			double max_dh_value = std::max(get_max(dh), std::abs(get_min(dh)));
			double max_dc_value = std::max(get_max(dc), std::abs(get_min(dc)));
			
			double flow_cap = std::sqrt(double(2) / v.size());
			if (max_dh_value > flow_cap) 
				dh = dh * (flow_cap / max_dh_value);
			if (max_dc_value > flow_cap) 
				dc = dc * (flow_cap / max_dc_value);
		}
					
		init_h_change = init_h_change + dh;
		init_c_change = init_c_change + dc;

		return flow_gradient;
	}


	void mutate(const double& mutation_chance, const double& mutation_rate) {
		int blockPergrid = upper_value(double(value.get_size() * value.get_size()) / 1024);
		int threadPerblock = std::min(int(value.get_size() * value.get_size()), 1024);
		mutate_array << <blockPergrid, threadPerblock >> > (xO_weight.value, xO_weight.value, random_state, mutation_chance, mutation_rate, xO_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (xF_weight.value, xF_weight.value, random_state, mutation_chance, mutation_rate, xI_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (xI_weight.value, xI_weight.value, random_state, mutation_chance, mutation_rate, xF_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (xK_weight.value, xK_weight.value, random_state, mutation_chance, mutation_rate, xK_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (hO_weight.value, hO_weight.value, random_state, mutation_chance, mutation_rate, hO_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (hF_weight.value, hF_weight.value, random_state, mutation_chance, mutation_rate, hF_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (hI_weight.value, hI_weight.value, random_state, mutation_chance, mutation_rate, hI_weight.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (hK_weight.value, hK_weight.value, random_state, mutation_chance, mutation_rate, hK_weight.get_size());
		cudaDeviceSynchronize();

		blockPergrid = upper_value(double(value.get_size()) / 1024);
		threadPerblock = std::min(int(value.get_size()), 1024);
		mutate_array << <blockPergrid, threadPerblock >> > (Obias.value, Obias.value, random_state, mutation_chance, mutation_rate, Obias.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (Fbias.value, Fbias.value, random_state, mutation_chance, mutation_rate, Fbias.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (Ibias.value, Ibias.value, random_state, mutation_chance, mutation_rate, Ibias.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (Kbias.value, Kbias.value, random_state, mutation_chance, mutation_rate, Kbias.get_size());
		cudaDeviceSynchronize();

		blockPergrid = upper_value(double(value.get_size()) / 1024);
		threadPerblock = std::min(int(value.get_size()), 1024);
		mutate_array << <blockPergrid, threadPerblock >> > (init_c.value, init_c.value, random_state, mutation_chance, mutation_rate, init_c.get_size());
		cudaDeviceSynchronize();
		mutate_array << <blockPergrid, threadPerblock >> > (init_h.value, init_h.value, random_state, mutation_chance, mutation_rate, init_h.get_size());
		cudaDeviceSynchronize();
	}
	


	void fogot(const std::size_t& number) {
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

	void fogot_all() {
		fogot(v.size());
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

			xO_weight = xO_weight + devide_each(s_xO_weight_change * learning_rate, pow_each(ss_xO_weight_change, 0.5) + 0.0000001);
			xF_weight = xF_weight + devide_each(s_xF_weight_change * learning_rate, pow_each(ss_xF_weight_change, 0.5) + 0.0000001);
			xI_weight = xI_weight + devide_each(s_xI_weight_change * learning_rate, pow_each(ss_xI_weight_change, 0.5) + 0.0000001);
			xK_weight = xK_weight + devide_each(s_xK_weight_change * learning_rate, pow_each(ss_xK_weight_change, 0.5) + 0.0000001);
			hO_weight = hO_weight + devide_each(s_hO_weight_change * learning_rate, pow_each(ss_hO_weight_change, 0.5) + 0.0000001);
			hF_weight = hF_weight + devide_each(s_hF_weight_change * learning_rate, pow_each(ss_hF_weight_change, 0.5) + 0.0000001);
			hI_weight = hI_weight + devide_each(s_hI_weight_change * learning_rate, pow_each(ss_hI_weight_change, 0.5) + 0.0000001);
			hK_weight = hK_weight + devide_each(s_hK_weight_change * learning_rate, pow_each(ss_hK_weight_change, 0.5) + 0.0000001);

			Obias = Obias + devide_each(s_Obias_change * learning_rate, pow_each(ss_Obias_change, 0.5) + 0.0000001);
			Fbias = Fbias + devide_each(s_Fbias_change * learning_rate, pow_each(ss_Fbias_change, 0.5) + 0.0000001);
			Ibias = Ibias + devide_each(s_Ibias_change * learning_rate, pow_each(ss_Ibias_change, 0.5) + 0.0000001);
			Kbias = Kbias + devide_each(s_Kbias_change * learning_rate, pow_each(ss_Kbias_change, 0.5) + 0.0000001);

			init_c = init_c + devide_each(s_init_c_change * learning_rate, pow_each(ss_init_c_change, 0.5) + 0.0000001);
			init_h = init_h + devide_each(s_init_h_change * learning_rate, pow_each(ss_init_h_change, 0.5) + 0.0000001);
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
		fogot_all();

		Layer::reconstruct(Layer::LSTM, size);

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

		fogot_all();
		c.push_back(init_c);
		h.push_back(init_h);

		set_ranndom_state();
	}

	void reconstruct(const LayerId& set) { 
		fogot_all();
		
		Layer::reconstruct(Layer::LSTM, set.Layer_size);

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

		set_ranndom_state();

		set_Layer(set.setting);
	}

	void reconstruct(const LSTM& copy) {
		fogot_all();

		Layer::reconstruct(copy);

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

		init_c.reconstruct(copy.init_c);
		init_h.reconstruct(copy.init_h);

		init_c_change.reconstruct(copy.init_c_change);
		init_h_change.reconstruct(copy.init_h_change);

		c.push_back(init_c);
		h.push_back(init_h);

		Iact_func = copy.Iact_func;
		Fact_func = copy.Fact_func;
		Oact_func = copy.Oact_func;
		Kact_func = copy.Kact_func;

		dIact_func = copy.dIact_func;
		dFact_func = copy.dFact_func;
		dOact_func = copy.dOact_func;
		dKact_func = copy.dKact_func;

		set_ranndom_state();
	}

	void reconstruct(LSTM&& other) {
		fogot_all();
		
		Layer::reconstruct(std::move(other));

		xO_weight.reconstruct(std::move(other.xO_weight));
		xF_weight.reconstruct(std::move(other.xF_weight));
		xI_weight.reconstruct(std::move(other.xI_weight));
		xK_weight.reconstruct(std::move(other.xK_weight));
		hO_weight.reconstruct(std::move(other.hO_weight));
		hF_weight.reconstruct(std::move(other.hF_weight));
		hI_weight.reconstruct(std::move(other.hI_weight));
		hK_weight.reconstruct(std::move(other.hK_weight));

		Obias.reconstruct(std::move(other.Obias));
		Fbias.reconstruct(std::move(other.Fbias));
		Ibias.reconstruct(std::move(other.Ibias));
		Kbias.reconstruct(std::move(other.Kbias));

		xO_weight_change.reconstruct(std::move(other.xO_weight_change));
		xF_weight_change.reconstruct(std::move(other.xF_weight_change));
		xI_weight_change.reconstruct(std::move(other.xI_weight_change));
		xK_weight_change.reconstruct(std::move(other.xK_weight_change));
		hO_weight_change.reconstruct(std::move(other.hO_weight_change));
		hF_weight_change.reconstruct(std::move(other.hF_weight_change));
		hI_weight_change.reconstruct(std::move(other.hI_weight_change));
		hK_weight_change.reconstruct(std::move(other.hK_weight_change));

		Obias_change.reconstruct(std::move(other.Obias_change));
		Fbias_change.reconstruct(std::move(other.Fbias_change));
		Ibias_change.reconstruct(std::move(other.Ibias_change));
		Kbias_change.reconstruct(std::move(other.Kbias_change));

		init_c.reconstruct(std::move(other.init_c));
		init_h.reconstruct(std::move(other.init_h));

		init_c_change.reconstruct(std::move(other.init_c_change));
		init_h_change.reconstruct(std::move(other.init_h_change));

		c = std::move(other.c);
		h = std::move(other.h);

		Iact_func = other.Iact_func;
		Fact_func = other.Fact_func;
		Oact_func = other.Oact_func;
		Kact_func = other.Kact_func;

		dIact_func = other.dIact_func;
		dFact_func = other.dFact_func;
		dOact_func = other.dOact_func;
		dKact_func = other.dKact_func;

		random_state = std::move(other.random_state);

		other.random_state = nullptr;
	}



	void rand_weight(const double& min,const double& max) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			xF_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			xI_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			xK_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			hO_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) ;
			hF_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) ;
			hI_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) ;
			hK_weightHost[i] = mapping(rand() % 10000, 0, 10000, min, max) ;
		}
		cudaMemcpy(xO_weight.value, xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.value, xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.value, xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.value, xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.value, hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.value, hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.value, hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.value, hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}

	void rand_weight(std::pair<const double&, const double&> setting) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xF_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xI_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			xK_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			hO_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) ;
			hF_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) ;
			hI_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) ;
			hK_weightHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second) ;
		}
		cudaMemcpy(xO_weight.value, xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.value, xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.value, xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.value, xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.value, hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.value, hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.value, hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.value, hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}
	
	void rand_weight(std::function<double()> func) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = func();
			xF_weightHost[i] = func();
			xI_weightHost[i] = func();
			xK_weightHost[i] = func();
			hO_weightHost[i] = func();
			hF_weightHost[i] = func();
			hI_weightHost[i] = func();
			hK_weightHost[i] = func();
		}
		cudaMemcpy(xO_weight.value, xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.value, xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.value, xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.value, xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.value, hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.value, hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.value, hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.value, hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}

	void rand_weight(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		double* xO_weightHost = new double[value.get_size() * value.get_size()];
		double* xF_weightHost = new double[value.get_size() * value.get_size()];
		double* xI_weightHost = new double[value.get_size() * value.get_size()];
		double* xK_weightHost = new double[value.get_size() * value.get_size()];
		double* hO_weightHost = new double[value.get_size() * value.get_size()];
		double* hF_weightHost = new double[value.get_size() * value.get_size()];
		double* hI_weightHost = new double[value.get_size() * value.get_size()];
		double* hK_weightHost = new double[value.get_size() * value.get_size()];
		for (int i = 0; i < value.get_size() * value.get_size(); i++) {
			xO_weightHost[i] = func(value.row, next);
			xF_weightHost[i] = func(value.row, next);
			xI_weightHost[i] = func(value.row, next);
			xK_weightHost[i] = func(value.row, next);
			hO_weightHost[i] = func(value.row, next);
			hF_weightHost[i] = func(value.row, next);
			hI_weightHost[i] = func(value.row, next);
			hK_weightHost[i] = func(value.row, next);
		}
		cudaMemcpy(xO_weight.value, xO_weightHost, xO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xI_weight.value, xI_weightHost, xI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xF_weight.value, xF_weightHost, xF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(xK_weight.value, xK_weightHost, xK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hO_weight.value, hO_weightHost, hO_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hI_weight.value, hI_weightHost, hI_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hF_weight.value, hF_weightHost, hF_weight.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(hK_weight.value, hK_weightHost, hK_weight.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] xO_weightHost;
		delete[] xF_weightHost;
		delete[] xI_weightHost;
		delete[] xK_weightHost;
		delete[] hO_weightHost;
		delete[] hF_weightHost;
		delete[] hI_weightHost;
		delete[] hK_weightHost;
	}

	void rand_bias(const double& min, const double& max) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			FbiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			IbiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
			KbiasHost[i] = mapping(rand() % 10000, 0, 10000, min, max);
		}
		cudaMemcpy(Obias.value, ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.value, IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.value, FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.value, KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
	}

	void rand_bias(std::pair<const double&,const double&> setting) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			FbiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			IbiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
			KbiasHost[i] = mapping(rand() % 10000, 0, 10000, setting.first, setting.second);
		}
		cudaMemcpy(Obias.value, ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.value, IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.value, FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.value, KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
	}
	
	void rand_bias(std::function<double()> func) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = func();
			FbiasHost[i] = func();
			IbiasHost[i] = func();
			KbiasHost[i] = func();;
		}
		cudaMemcpy(Obias.value, ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.value, IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.value, FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.value, KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
	}

	void rand_bias(std::function<double(std::size_t,std::size_t)> func,std::size_t next) {
		double* ObiasHost = new double[value.get_size()];
		double* FbiasHost = new double[value.get_size()];
		double* IbiasHost = new double[value.get_size()];
		double* KbiasHost = new double[value.get_size()];
		for (int i = 0; i < value.get_size(); i++) {
			ObiasHost[i] = func(value.row, next);
			FbiasHost[i] = func(value.row, next);
			IbiasHost[i] = func(value.row, next);
			KbiasHost[i] = func(value.row, next);
		}
		cudaMemcpy(Obias.value, ObiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Ibias.value, IbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Fbias.value, FbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		cudaMemcpy(Kbias.value, KbiasHost, Obias.get_sizeb(), cudaMemcpyHostToDevice);
		delete[] ObiasHost;
		delete[] IbiasHost;
		delete[] FbiasHost;
		delete[] KbiasHost;
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
		Matrix<double> fogot_gate = Fact_func((xF_weight * value) + (hF_weight * h.back()) + Fbias);
		Matrix<double> output_gate = Oact_func((xO_weight * value) + (hO_weight * h.back()) + Obias);
		Matrix<double> K = Kact_func((xK_weight * value) + (hK_weight * h.back()) + Kbias);
		std::cout << "--------------LSTM Layer----------\n\n";
		value.print();
		std::cout << "--------input--------\n";
		input_gate.print();
		std::cout << "--------fogot--------\n";
		fogot_gate.print();
		std::cout << "--------output-------\n";
		output_gate.print();
		std::cout << "----------K----------\n";
		K.print();
	}


	void save_as(std::ofstream& output_file) {
		xI_weight.print(output_file);
		hI_weight.print(output_file);
		xF_weight.print(output_file);
		hF_weight.print(output_file);
		xO_weight.print(output_file);
		hO_weight.print(output_file);
		xK_weight.print(output_file);
		hK_weight.print(output_file);

		Ibias.print(output_file);
		Fbias.print(output_file);
		Obias.print(output_file);
		Kbias.print(output_file);
		
		init_c.print(output_file);
		init_h.print(output_file);
	}

	void load(std::ifstream& input_file) {
		xI_weight.load(input_file);
		hI_weight.load(input_file);
		xF_weight.load(input_file);
		hF_weight.load(input_file);
		xO_weight.load(input_file);
		hO_weight.load(input_file);
		xK_weight.load(input_file);
		hK_weight.load(input_file);

		Ibias.load(input_file);
		Fbias.load(input_file);
		Obias.load(input_file);
		Kbias.load(input_file);

		init_c.load(input_file);
		init_h.load(input_file);
	}

protected:
	void print_xO_weight() {
		std::cout << "  -----x-output weight----\n";
		xO_weight.print();
	}

	void print_xF_weight() {
		std::cout << "  -----x-fogot weight----\n";
		xF_weight.print();
	}

	void print_xI_weight() {
		std::cout << "  -----x-input weight----\n";
		xI_weight.print();
	}

	void print_xK_weight() {
		std::cout << "  -----x-k    weight----\n";
		xK_weight.print();
	}

	void print_hO_weight() {
		std::cout << "  -----h-output weight----\n";
		hO_weight.print();
	}

	void print_hF_weight() {
		std::cout << "  -----h-fogot weight----\n";
		hF_weight.print();
	}

	void print_hI_weight() {
		std::cout << "  -----h-input weight----\n";
		hI_weight.print();
	}

	void print_hK_weight() {
		std::cout << "  -----h-K     weight----\n";
		hK_weight.print();
	}

	void print_Obias() {
		std::cout << "   ---output bias------\n";
		Obias.print();
	}

	void print_Fbias() {
		std::cout << "   ---fogot bias------\n";
		Fbias.print();
	}

	void print_Ibias() {
		std::cout << "   ---input bias------\n";
		Ibias.print();
	}

	void print_Kbias() {
		std::cout << "   ---K     bias------\n";
		Kbias.print();
	}

	void print_init() {
		std::cout << " -------- init----------\n";
		init_c.print();

		init_h.print();
	}



	void set_ranndom_state() {
		if (random_state != nullptr)
			cudaFree(random_state);
		cudaMalloc(&random_state, value.get_size() * value.get_size() * sizeof(curandState));
		int blockPergrid = upper_value(double(value.get_size() * value.get_size()) / 1024);
		int threadPerblock = std::min(value.get_size() * value.get_size(), 1024);
		set_random << <blockPergrid, threadPerblock >> > (random_state, rand(), value.get_size() * value.get_size());
		cudaDeviceSynchronize();
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
			else if (a == "decay_rate")
				set_decay_rate(setting, i);
			else if (a == "opt")
				set_optimizer(setting, i);
			else if (a == "")
				;
			else throw "command not found";
		}
	}

	void set_decay_rate(const std::string& str, int& i) {
		double a = get_number(str, i);
		decay_rate = a;
	}

	void set_optimizer(const std::string& str, int& i) {
		std::string _opt = get_text(str, i);
		if (_opt == "SGD")
			optimizer = Layer::SGD;
		else if (_opt == "MOMENTUM")
			optimizer = Layer::MOMENTUM;
		else if (_opt == "ADAM")
			optimizer = Layer::ADAM;
		else if (_opt == "")
			;
		else throw "optimizer not found";
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

	Matrix<double> xO_weight_change;																		// changing weight for input -> output gate
	Matrix<double> xF_weight_change;																		// changing weight for input -> forgot gate
	Matrix<double> xI_weight_change;																		// changing weight for input -> input gate
	Matrix<double> xK_weight_change;																		// changing weight for input -> K
	Matrix<double> hO_weight_change;																		// changing hidden for input -> output gate
	Matrix<double> hF_weight_change;																		// changing hidden for input -> forgot gate
	Matrix<double> hI_weight_change;																		// changing hidden for input -> input gate
	Matrix<double> hK_weight_change;																		// changing hidden for input -> K

	Matrix<double> Obias_change;																			// changing bias for output gate																	
	Matrix<double> Fbias_change;																			// changing bias for forgot gate
	Matrix<double> Ibias_change;																			// changing bias for input gate
	Matrix<double> Kbias_change;																			// changing bias for K

	Matrix<double> init_c_change;																			// changing initial cell state
	Matrix<double> init_h_change;																			// changing initial hidden

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
	
	std::vector<Matrix<double>> c;
	std::vector<Matrix<double>> h;

	std::function<Matrix<double>(const Matrix<double>&)> Oact_func = sigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> Fact_func = sigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&)> Iact_func = tanh_func;
	std::function<Matrix<double>(const Matrix<double>&)> Kact_func = tanh_func;

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dOact_func = dsigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dFact_func = dsigmoid_func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dIact_func = dtanh_func;
	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> dKact_func = dtanh_func;

	curandState* random_state;
};
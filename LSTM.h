#pragma once
#include "Header.h"
#include "Layer.h"
#include "LayerId.h"
#include "Function.h"
#include "Variable.h"


class LSTM : public Layer {
public:
	LSTM();

	LSTM(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func = dtanh_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func = sigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func = dsigmoid_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func = tanh_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func = dtanh_func);

	LSTM(const LayerId& set);

	LSTM(const LSTM& copy);

	LSTM(LSTM&& move);



	Matrix<double> feed();

	std::vector<Matrix<double>> propagation(const std::vector<Matrix<double>>& gadient);



	void forgot(const std::size_t& number);

	void forgot_all();



	void change_dependencies();

	void set_change_dependencies(const double& number);

	void mul_change_dependencies(const double& number);

	void map_change_dependencies(const std::function<Matrix<double>(Matrix<double>)>& func);

	bool found_nan();



	void reconstruct(const std::size_t& size);

	void reconstruct(const std::size_t& size,
		std::function<Matrix<double>(const Matrix<double>&)> _Iact_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dIact_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Fact_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dFact_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Oact_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dOact_func,
		std::function<Matrix<double>(const Matrix<double>&)> _Kact_func,
		std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> _dKact_func);

	void reconstruct(const LayerId& set);

	void reconstruct(const LSTM& copy);

	void reconstruct(LSTM&& move);



	void set_weight(const int& weight_type, const Matrix<double>& _weight);

	void set_bias(const int& bias_type, const Matrix<double>& _bias);

	void rand_weight(const double& min, const double& max);

	void rand_weight(std::pair<const double&, const double&> setting);

	void rand_weight(std::function<double()> func);

	void rand_weight(std::function<double(std::size_t, std::size_t)> func, std::size_t next);

	void rand_bias(const double& min, const double& max);

	void rand_bias(std::pair<const double&, const double&> setting);

	void rand_bias(std::function<double()> func);

	void rand_bias(std::function<double(std::size_t, std::size_t)> func, std::size_t next);



	std::function<Matrix<double>(const Matrix<double>&)> get_Oact_func();

	std::function<Matrix<double>(const Matrix<double>&)> get_Fact_func();

	std::function<Matrix<double>(const Matrix<double>&)> get_Iact_func();

	std::function<Matrix<double>(const Matrix<double>&)> get_Kact_func();

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dOact_func();

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dFact_func();

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dIact_func();

	std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> get_dKact_func();

	double get_max_abs_change_dependencies();


	void save_as(std::ofstream& output_file);

	void load(std::ifstream& input_file);



	void print_weight();

	void print_bias();

	void print_value();



	void set_optimizer(const Layer::opt& _optimizer);

	enum weight_type {
		XO_WEIGHT = 0,
		XF_WEIGHT,
		XI_WEIGHT,
		XK_WEIGHT,
		HO_WEIGHT,
		HF_WEIGHT,
		HI_WEIGHT,
		HK_WEIGHT,

		INIT_H,
		INIT_C
	};

	enum bias_type {
		OBIAS = 0,
		FBIAS,
		IBIAS,
		KBIAS
	};
protected:
private:

	void print_xO_weight();

	void print_xF_weight();

	void print_xI_weight();

	void print_xK_weight();

	void print_hO_weight();

	void print_hF_weight();

	void print_hI_weight();

	void print_hK_weight();

	void print_Obias();

	void print_Fbias();

	void print_Ibias();

	void print_Kbias();

	void print_init();



	void set_Layer(const std::string& setting);


	Matrix<double> xO_weight;
	Matrix<double> xF_weight;
	Matrix<double> xI_weight;
	Matrix<double> xK_weight;
	Matrix<double> hO_weight;
	Matrix<double> hF_weight;
	Matrix<double> hI_weight;
	Matrix<double> hK_weight;

	Matrix<double> Obias;
	Matrix<double> Fbias;
	Matrix<double> Ibias;
	Matrix<double> Kbias;

	Matrix<double> init_c;
	Matrix<double> init_h;

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
};
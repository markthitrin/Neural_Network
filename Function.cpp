#pragma once
#include "Header.h"
#include "Matrix.cpp"

std::function<Matrix<double>(const Matrix<double>&)> 
sigmoid_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = double(1) / (double(1) + std::exp(-input[i][j]));

			if (result[i][j] != result[i][j]) {																			// In case result = nan. std::exp goes too high
				result[i][j] = 0.000001;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dsigmoid_func = [](const Matrix<double>& input, const Matrix<double> gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = std::exp(-input[i][j]) / std::pow(double(1) + std::exp(-input[i][j]), 2.0);
			result[i][j] *= gadient[i][j];

			if (result[i][j] != result[i][j])
				result[i][j] = 0.000001;
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
tanh_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = std::tanh(input[i][j]);

			if (result[i][j] != result[i][j]) { // In case result[i][j] = nan
				result[i][j] = 0.000001;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dtanh_func = [](const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = double(1) - std::pow(std::tanh(input[i][j]), 2.0);
			result[i][j] *= gadient[i][j];

			if (result[i][j] != result[i][j]) { // In case result[i][j] = nan
				result[i][j] = 0.0000001;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
linear_func = [](const Matrix<double>& input) {
	return input;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dlinear_func = [](const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			result[i][j] = gadient[i][j];
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)>
ReLU_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			result[i][j] = std::max(0.0, input[i][j]);
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dReLU_func = [](const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			if (input[i][j] >= 0) {
				result[i][j] = gadient[i][j];
			}
			else {
				result[i][j] = 0;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)>
leakReLU_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			if (input[i][j] >= 0) {
				result[i][j] = std::max(0.1 * input[i][j], input[i][j]);
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dleakReLU_func = [](const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			if (input[i][j] >= 0) {
				result[i][j] = gadient[i][j];
			}
			else {
				result[i][j] = gadient[i][j] * 0.1;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
soft_max = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	long double sum = 0;
	for (int i = 0; i < input.get_row(); i++) {
		sum += std::exp(input[i][0]);
	}
	for (int i = 0; i < input.get_row(); i++) {
		result[i][0] = std::exp(input[i][0]) / sum;

		if (result[i][0] != result[i][0]) { // In case result[i][j] = nan
			result[i][0] = 0.000001;
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dsoft_max = [](const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	Matrix<double> y = soft_max(input);
	for (int i = 0; i < input.get_row(); i++) {
		result[i][0] = 0;
	}
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_row(); j++) {
			result[i][0] += (y[i][0] * ((i == j) - y[j][0])) * gadient[j][0];

			if (result[i][0] != result[i][0]) { // In case result = nan
				result[i][0] = 0.000001;
			}
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
descale_func = [](const Matrix<double>& input) {
	Matrix<double> result(input.get_row(), input.get_column());
	double max_value = input[0][0];
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			max_value = std::max(input[i][j], max_value);
		}
	}
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = input[i][j] - max_value;
		}
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
ddescale_func = [](const Matrix<double>& input, const Matrix<double>& gadient) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		for (int j = 0; j < input.get_column(); j++) {
			result[i][j] = gadient[i][j];
		}
	}
	return result;
};

std::function<double(const Matrix<double>&, const Matrix<double>&)> 
catagorical_CEnt_loss_func = [](const Matrix<double>& input, const Matrix<double>& target) {
	double result = 0;
	for (int i = 0; i < input.get_row(); i++) {
		result += target[i][0] * std::log(input[i][0]);
	}
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> 
dcatagorical_CEnt_loss_func = [](const Matrix<double>& input,const Matrix<double>& target) {
	Matrix<double> result(input.get_row(), input.get_column());
	for (int i = 0; i < input.get_row(); i++) {
		result[i][0] = target[i][0] / input[i][0];
		if (result[i][0] != result[i][0])
			result[i][0] = 0.0000001;
	}
	return result;
};

std::function<double()> 
normal_rand_func = []() {
	return double(std::rand() % 10000) / 10000;
};



double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2) {
	return ((value - min1) / (max1 - min1) * (max2 - min2)) + min2;
}

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right) { // directly multiply a[i][j] and b[i][j]
	if (left.get_row() != right.get_row() || left.get_column() != right.get_column())
		throw "invalid multiply each elemenet";
	Matrix<double> result(left.get_row(),left.get_column());
	for (int i = 0; i < left.get_row(); i++) {
		for (int j = 0; j < left.get_column(); j++) {
			result[i][j] = left[i][j] * right[i][j];
		}
	}
	return result;
}

Matrix<double> devide_each(const Matrix<double>& left, const Matrix<double>& right) { // directly multiply a[i][j] and b[i][j]
	if (left.get_row() != right.get_row() || left.get_column() != right.get_column())
		throw "invalid multiply each elemenet";
	Matrix<double> result(left.get_row(), left.get_column());
	for (int i = 0; i < left.get_row(); i++) {
		for (int j = 0; j < left.get_column(); j++) {
			result[i][j] = left[i][j] / right[i][j];
		}
	}
	return result;
}

Matrix<double> pow(const Matrix<double>& M, const double& number) {
	Matrix<double> result(M.get_row(),M.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			result[i][j] = std::pow(M[i][j], number);
		}
	}
	return result;
}

void set_Matrix(Matrix<double>& M, double value) { // set every Matrix's member to specific number
	for (int i = 0; i < M.get_row(); i++) {
		for (int j = 0; j < M.get_column(); j++) {
			M[i][j] = value;
		}
	}
}

double get_max(const Matrix<double>& M) { // get max value of the Matrix
	double max_value = M[0][0];
	for (int i = 0; i < M.get_row(); i++) {
		for (int j = 0; j < M.get_column(); j++) {
			max_value = std::max(max_value, M[i][j]);
		}
	}
	return max_value;
}

double get_min(const Matrix<double>& M) { // get min value of the Matrix
	double min_value = M[0][0];
	for (int i = 0; i < M.get_row(); i++) {
		for (int j = 0; j < M.get_column(); j++) {
			min_value = std::min(min_value, M[i][j]);
		}
	}
	return min_value;
}

double check_nan(const double& test, const double& value) {
	return test != test ? value : test;
}

std::string get_text(const std::string& str, int& i) {
	std::string result;
	while (str[i] != '\0' && str[i] != ':' && str[i] != ' ' && str[i] != ',') {
		result.insert(result.end(), str[i]);
		++i;
	}
	++i;
	return result;
}

double get_number(const std::string& str, int& i) { // change number in string to double
	double result = 0;
	int dot_pos = -1;
	while (str[i] != '\0' && str[i] != ':' && str[i] != ' ' && str[i] != ',') {
		if (dot_pos == -1) {
			if (str[i] == '.')
				dot_pos = 1;
			else {
				result = result * 10 + (str[i] - '0');
			}
		}
		else if (dot_pos != -1) {
			result += double(str[i] - '0') * std::pow(10, -dot_pos);
			dot_pos++;
		}
		++i;
	}
	++i;
	return result;
}

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i) { // return function from string name (func)
	std::string a = get_text(setting, i);
	if (a == "sigmoid")
		func = sigmoid_func;
	else if (a == "tanh")
		func = tanh_func;
	else if (a == "linear")
		func = linear_func;
	else if (a == "soft_max")
		func = soft_max;
	else if (a == "ReLU")
		func = ReLU_func;
	else if (a == "leakReLU")
		func = leakReLU_func;
	else throw "function not found";
}

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)>& func, const std::string& setting, int& i) { // return function from string name (dfunc)
	std::string a = get_text(setting, i);
	if (a == "dsigmoid")
		func = dsigmoid_func;
	else if (a == "dtanh")
		func = dtanh_func;
	else if (a == "dlinear")
		func = dlinear_func;
	else if (a == "dsoft_max")
		func = dsoft_max;
	else if (a == "dReLU")
		func = dReLU_func;
	else if (a == "dleakReLU")
		func = dleakReLU_func;
	else throw "function not found";
}
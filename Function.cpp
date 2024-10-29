#pragma once
#include "Header.h"
#include "Matrix.h"
#include "Variable.h"

double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2) {
	return ((value - min1) / (max1 - min1) * (max2 - min2)) + min2;
}

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right) { // directly multiply a[i][j] and b[i][j]
	if (left.get_row() != right.get_row() || left.get_column() != right.get_column())
		throw std::runtime_error("invalid multiply each elemenet");
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
		throw std::runtime_error("invalid multiply each elemenet");
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

Matrix<double> pow(const Matrix<double>& M, const Matrix<double>& P) {
	Matrix<double> result(M.get_row(), M.get_column());
	for (int i = 0; i < result.get_row(); i++) {
		for (int j = 0; j < result.get_column(); j++) {
			result[i][j] = std::pow(M[i][j], P[i][j]);
		}
	}
	return result;
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
	else throw std::runtime_error("function not found");
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
	else throw std::runtime_error("function not found");
}
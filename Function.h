#pragma once
#pragma once
#include "Header.h"
#include "Matrix.h"

double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right);

Matrix<double> devide_each(const Matrix<double>& left, const Matrix<double>& right);

Matrix<double> pow(const Matrix<double>& M, const double& number);

Matrix<double> pow(const Matrix<double>& M, const Matrix<double>& P);



double get_max(const Matrix<double>& M);

double get_min(const Matrix<double>& M);

double check_nan(const double& test, const double& value);



std::string get_text(const std::string& str, int& i);

double get_number(const std::string& str, int& i);

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i);

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>& func, const std::string& setting, int& i);
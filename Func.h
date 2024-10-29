#pragma once

#include "Header.h"
#include "Matrix.cpp"

extern std::function<Matrix<double>(const Matrix<double>&)>
sigmoid_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsigmoid_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
tanh_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dtanh_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
linear_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dlinear_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
ReLU_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dReLU_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
leakReLU_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dleakReLU_func;

extern std::function<Matrix<double>(const Matrix<double>&)>
soft_max;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsoft_max;

extern std::function<Matrix<double>(const Matrix<double>&)>
descale_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
ddescale_func;

extern std::function<double(const Matrix<double>&, const Matrix<double>&)>
catagorical_CEnt_loss_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dcatagorical_CEnt_loss_func;

extern std::function<double()>
normal_rand_func;


double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);

Matrix<double> mul_each(const Matrix<double>& left, const Matrix<double>& right);

Matrix<double> devide_each(const Matrix<double>& left, const Matrix<double>& right);

Matrix<double> pow(const Matrix<double>& M, const double& number);

void set_Matrix(Matrix<double>& M, double value);;

double get_max(const Matrix<double>& M);

double get_min(const Matrix<double>& M);

double check_nan(const double& test, const double& value);

std::string get_text(const std::string& str, int& i);

double get_number(const std::string& str, int& i);

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i);

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>& func, const std::string& setting, int& i);
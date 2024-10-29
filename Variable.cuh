#pragma once

#include "Header.cuh"
#include "Matrix.cu";

extern std::function<Matrix<double>(const Matrix<double>&)>
squere_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsquere_func;

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

extern std::function<double(const Matrix<double>&, const Matrix<double>&)>
squere_mean_loss_func;

extern std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsquere_mean_loss_func;

extern std::function<double()>
normal_rand_func;
#pragma once

#include "Header.cuh"
#include "Matrix.cu"

int upper_value(const double& a);

__global__ void get_random(double* result, curandState* state, const int sizes);

__global__ void set_random(curandState* state, const double seed, const int size);

__global__ void mutate_array(double* result, const double* value, curandState* random_state, const double mutation_chance, const double mutation_rate, const int size);

__global__ void drop_out(double* result, const double* value, const double drop_rate, const int size);

__global__ void device_sigmoid_func(double* result, const double* value, const int size);

__global__ void device_dsigmoid_func(double* result, const double* value, const double* gradient, const int size);

__global__ void device_tanh_func(double* result, const double* value, const int size);

__global__ void device_dtanh_func(double* result, const double* value, const double* gradient, const int size);

__global__ void device_softmax_func(double* result, const double* value, const int sum, const int size);

__global__ void device_dsoftmax_func(double* result, const double* output, const double* gradient, const int size);

__global__ void device_exp_func(double* result, const double* value, const int size);

__global__ void device_getsumBin_func(double* sum, const double* value, const int size);

__global__ void device_getsumBru_func(double* sum, const double* value, const int size);

__global__ void device_getmaxBin_func(double* getmax, const double* value, const int size);

__global__ void device_getmaxBru_func(double* getmax, const double* value, const int size);

__global__ void device_getminBin_func(double* getmin, const double* value, const int size);

__global__ void device_getminBru_func(double* getmin, const double* value, const int size);

__global__ void device_mul_func(double* result, const double* value1, const double* value2, const int size);

__global__ void device_mul_func(double* result, const double* value, const double number, const int size);

__global__ void device_devide_func(double* result, const double* value, const double* devisor, const int size);

__global__ void device_pow_func(double* result, const double* value, const double exponent, const int size);

__global__ void device_pow_func(double* result, const double* value, const double* exponent, const int size);

__global__ void device_plus_func(double* result, const double* value, const double number, const int size);

__global__ void device_ccentloss_func(double* result, const double* output, const double* target, const int size);

__global__ void device_dccentloss_func(double* result, const double* output, const double* target, const int size);

__global__ void device_smeanloss_func(double* result, const double* output, const double* target, const int size);

__global__ void device_dsmeanloss_func(double* result, const double* output, const double* target, const int size);

__global__ void device_set_matrix(double* value, const double number, const int size);

void get_sum(double* sum, const double* value, const int _size);

void get_max(double* result, const double* value, const int _size);

void get_min(double* result, const double* value, const int _size);



__host__ __device__ double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2);

Matrix<double> mul_each(const Matrix<double>& value1, const Matrix<double>& value2);

Matrix<double> devide_each(const Matrix<double>& value, const Matrix<double>& divisor);

Matrix<double> pow_each(const Matrix<double>& value, const double exponent);

void set_Matrix(Matrix<double>& M, double value);

void set_up_Matrix(Matrix<double>& M, const Matrix<double>& B);

double get_max(const Matrix<double>& M);

double get_min(const Matrix<double>& M);



std::string get_text(const std::string& str, int& i);

double get_number(const std::string& str, int& i);



void universal_set_func(std::function<Matrix<double>(const Matrix<double>&)>& func, const std::string& setting, int& i);

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>& func, const std::string& setting, int& i);

__global__ void device_weightchange_computeLSTM(double* weight_change, const double* dgate, const double* value, const int weight_row, const int weight_column);

__global__ void device_flow_computeLSTM(double* value_change, const double* dgate, const double* weight, const int weight_row, const int weight_column);

__global__ void device_weightchange_computeDENSE(double* weight_change, const double* doutput, const double* value, const int weight_row, const int weight_column);

__global__ void device_flow_computeDENSE(double* value_change, const double* doutput, const double* weight, const int weightrow, const int weightcolumn);

__global__ void device_predict_computeDROPOUT(double* value, const double keep_rate, const int size);

__global__ void device_valuechange_computeDROPOUT(double* value_change, const double* gradient, const double* value, const int size);

__global__ void device_expochange_computePOWER(double* exponent_change, const double* gradient, const double* exponent, const double* value, const int size);

__global__ void device_valuechange_computePOWER(double* value_change, const double* gradient, const double* exponent, const double* value, const int size);
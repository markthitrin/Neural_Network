#pragma once
#include "Header.cuh"
#include "Matrix.cu"
#include "Func.cuh"

int upper_value(const double& a) {
	if (a != int(a))
		return int(a) + 1;
	return int(a);
}


__global__ void get_random(double* result, curandState* state, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	result[pos] = curand_uniform((&state[pos]));
}

__global__ void set_random(curandState* state, const double seed, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	curand_init(seed, pos, 0, &state[pos]);
}

__global__ void mutate_array(double* result, const double* value, curandState* random_state,const double mutation_chance, const double mutation_rate, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	if (curand_uniform((&random_state[pos])) <= mutation_chance) {
		result[pos] += ((curand_uniform((&random_state[pos])) * 2 - 1) * mutation_rate);
	}
}

__global__ void drop_out(double* result, const double* value, const double drop_rate, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	if (value[pos] <= drop_rate)
		result[pos] = 0;
	else
		result[pos] = 1;
}

__global__ void device_sigmoid_func(double* result, const double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double get_result = double(1) / (double(1) + exp(-value[pos]));
	if (get_result != get_result)
		get_result = 0.000001;

	result[pos] = get_result;
}

__global__ void device_dsigmoid_func(double* result, const double* value, const double* gradient, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double get_result = exp(-value[pos]) / pow(double(1) + exp(-value[pos]), 2.0);
	get_result *= gradient[pos];
	if (get_result != get_result)
		get_result = 0.000001;

	result[pos] = get_result;
}

__global__ void device_tanh_func(double* result, const double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double get_result = tanh(value[pos]);
	if (get_result != get_result)
		get_result = 0.000001;
	
	result[pos] = get_result;
}

__global__ void device_dtanh_func(double* result, const double* value, const double* gradient, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double get_result = double(1) - pow(tanh(value[pos]), 2.0);
	get_result *= gradient[pos];
	if (get_result != get_result)
		get_result = 0.000001;

	result[pos] = get_result;
}

__global__ void device_softmax_func(double* result, const double* value,const int sum, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	double get_result = value[pos] / sum;
	if (get_result != get_result)
		get_result = 0.000001;

	result[pos] = get_result;
}

__global__ void device_dsoftmax_func(double* result, const double* output, const double* gradient, const int size) {
	int pos = blockDim.x * blockIdx.x + threadIdx.x;
	if (pos >= size)
		return;

	double get_result = 0;
	for (int i = 0; i < size; i++) {
		get_result += output[i] * ((i == pos) - output[pos]) * gradient[i];
	}
	if (get_result != get_result)
		get_result = 0.000001;

	result[pos] = get_result;
}

__global__ void device_exp_func(double* result, const double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos >= size)
		return;

	result[pos] = exp(value[pos]);
}

__global__ void device_getsumBin_func(double* sum, const double* value,  const int size) {
	__shared__ double cpy[1024];

	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int id = threadIdx.x;

	if (pos < size) {
		cpy[id] = value[pos];
	}
	__syncthreads();

	for (int i = 0; i < 10; i++) {
		int des = (1 << i);
		if (!(id & ((des << 1) - 1)) && pos + des < size) {
			cpy[id] += cpy[id + des];
		}
		__syncthreads();
	}

	sum[blockIdx.x] = cpy[0];
}

__global__ void device_getsumBru_func(double* sum, const double* value, const int size) {
	double result = 0;
	for (int q = 0; q < size; q++) {
		result += value[q];
	}
	(*sum) = result;
}

__global__ void device_getmaxBin_func(double* getmax, const double* value, const int size) {
	__shared__ double cpy[1024];

	int id = threadIdx.x;
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos < size) {
		cpy[id] = value[pos];
	}
	__syncthreads();

	for (int i = 0; i < 10; i++) {
		int filter = (1 << i);
		if (!(id & ((filter << 1) - 1)) && pos + filter < size) {
			cpy[id] = max(cpy[id], cpy[id + filter]);
		}
		__syncthreads();
	}

	getmax[blockIdx.x] = cpy[0];
}

__global__ void device_getmaxBru_func(double* getmax, const double* value, const int size) {
	double _max = value[0];
	for (int i = 1; i < size; i++) {
		_max = max(_max, value[i]);
	}

	(*getmax) = _max;
}

__global__ void device_getminBin_func(double* getmin, const double* value,  const int size) {
	__shared__ double cpy[1024];

	int id = threadIdx.x;
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos < size) {
		cpy[id] = value[pos];
	}
	__syncthreads();

	for (int i = 0; i < 10; i++) {
		int filter = (1 << i);
		if (!(id & ((filter << 1) - 1)) && pos + filter < size) {
			cpy[id] = min(cpy[id], cpy[id + filter]);
		}
		__syncthreads();
	}

	getmin[blockIdx.x] = cpy[0];
}

__global__ void device_getminBru_func(double* getmin, const double* value, const int size) {
	double _min = value[0];
	for (int i = 1; i < size; i++) {
		_min = min(_min, value[i]);
	}

	(*getmin) = _min;
}

__global__ void device_mul_func(double* result, const double* value1, const double* value2, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= size)
		return;

	result[pos] = value1[pos] * value2[pos];
}

__global__ void device_mul_func(double* result, const double* value, const double number, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= size)
		return;

	result[pos] = value[pos] * number;
}

__global__ void device_devide_func(double* result, const double* value, const double* devisor, const int size) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (pos >= size)
		return;

	result[pos] = value[pos] / devisor[pos];
}

__global__ void device_pow_func(double* result, const double* value, const double exponent, const int size) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;

	if (pos >= size)
		return;

	result[pos] = pow(value[pos], exponent);
}

__global__ void device_pow_func(double* result, const double* value, const double* exponent, const int size) {
	int pos = threadIdx.x + blockIdx.x * blockDim.x;

	if (pos >= size)
		return;

	result[pos] = pow(value[pos], exponent[pos]);
}

__global__ void device_plus_func(double* result, const double* value, const double number, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (pos >= size)
		return;

	result[pos] += number;
}

__global__ void device_ccentloss_func(double* result, const double* output, const double* target, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= size)
		return;

	result[pos] = target[pos] * log(output[pos]);
}

__global__ void device_dccentloss_func(double* result,const double* output, const double* target, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= size)
		return;

	if (result[pos] < 0.00001)
		result[pos] = target[pos] / 0.00001;
	else
		result[pos] = target[pos] / output[pos];

}

__global__ void device_smeanloss_func(double* result,const double* output, const double* target, const int size) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;

	if (pos >= size)
		return;

	result[pos] = pow(target[pos] - output[pos],2.0);
}

__global__ void device_dsmeanloss_func(double* result,const double* output, const double* target, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;

	if (pos >= size)
		return;

	result[pos] = 2 * (target[pos] - output[pos]);
}

__global__ void device_set_matrix(double* value, const double number, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (pos >= size)
		return;

	value[pos] = number;
}

void get_sum(double* sum ,const double* value, const int _size) {
	int size = _size;
	int pos = 0;

	double* getsum[2];
	cudaMalloc(&(getsum[0]), size * sizeof(double));
	cudaMalloc(&(getsum[1]), size * sizeof(double));
	cudaMemcpy(getsum[0], value, size * sizeof(double), cudaMemcpyDeviceToDevice);

	int blockPergrid = upper_value(double(size) / 1024);
	int threadPerblock = std::min(size, 1024);
	while (size > 256) {
		device_getsumBin_func << <blockPergrid, threadPerblock >> > (getsum[1 - pos], getsum[pos], size);
		cudaDeviceSynchronize();
		size = blockPergrid;
		blockPergrid = upper_value(double(size) / 1024);
		threadPerblock = std::min(size, 1024);
		pos = 1 - pos;
	}

	double* get_result;
	cudaMalloc(&get_result, sizeof(double));
	device_getsumBru_func << <1, 1 >> > (get_result, getsum[pos], size);
	cudaDeviceSynchronize();

	cudaMemcpy(sum, get_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(getsum[0]);
	cudaFree(getsum[1]);
	cudaFree(get_result);
}

void get_max(double* result, const double* value, const int _size) {
	int size = _size;
	int pos = 0;

	double* getmax[2];
	cudaMalloc(&(getmax[0]), size * sizeof(double));
	cudaMalloc(&(getmax[1]), size * sizeof(double));
	cudaMemcpy(getmax[0], value, size * sizeof(double), cudaMemcpyDeviceToDevice);

	int blockPergrid = upper_value(double(size) / 1024);
	int threadPerblock = std::min(size, 1024);
	while (size > 256) {
		device_getmaxBin_func << <blockPergrid, threadPerblock >> > (getmax[1 - pos], getmax[pos], size);
		cudaDeviceSynchronize();
		size = blockPergrid;
		blockPergrid = upper_value(double(size) / 1024);
		threadPerblock = std::min(size, 1024);
		pos = 1 - pos;
	}

	double* get_result;
	cudaMalloc(&get_result, sizeof(double));
	device_getmaxBru_func << <1, 1 >> > (get_result, getmax[pos], size);
	cudaDeviceSynchronize();

	cudaMemcpy(result, get_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(getmax[0]);
	cudaFree(getmax[1]);
	cudaFree(get_result);
}

void get_min(double* result, const double* value, const int _size) {
	int size = _size;
	int pos = 0;

	double* getmin[2];
	cudaMalloc(&(getmin[0]), size * sizeof(double));
	cudaMalloc(&(getmin[1]), size * sizeof(double));

	cudaMemcpy(getmin[0], value, size * sizeof(double), cudaMemcpyDeviceToDevice);
	int blockPergrid = upper_value(double(size) / 1024);
	int threadPerblock = std::min(size, 1024);
	while (size > 256) {
		device_getminBin_func << <blockPergrid, threadPerblock >> > (getmin[1 - pos], getmin[pos], size);
		cudaDeviceSynchronize();
		size = blockPergrid;
		blockPergrid = upper_value(double(size) / 1024);
		threadPerblock = std::min(size, 1024);
		pos = 1 - pos;
	}

	double* get_result;
	cudaMalloc(&get_result, sizeof(double));
	device_getminBru_func << <1, 1 >> > (get_result, getmin[pos], size);
	cudaDeviceSynchronize();

	cudaMemcpy(result, get_result, sizeof(double), cudaMemcpyDeviceToHost);
	
	cudaFree(getmin[0]);
	cudaFree(getmin[1]);
	cudaFree(get_result);
}

std::function<Matrix<double>(const Matrix<double>&)>
squere_func = [](const Matrix<double>& input) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_pow_func << <blockPergrid, threadPerblock >> > (result.value, result.value, 2.0, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsquere_func = [](const Matrix<double>& input, const Matrix<double> gradient) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_mul_func << <blockPergrid, threadPerblock >> > (result.value, result.value, 2.0, result.get_size());
	cudaDeviceSynchronize();
	result = mul_each(result, gradient);
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
sigmoid_func = [] (const Matrix<double>& input) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_sigmoid_func << <blockPergrid, threadPerblock >> > (result.value, result.value, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dsigmoid_func = [] (const Matrix<double>& input, const Matrix<double> gradient) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_dsigmoid_func << <blockPergrid, threadPerblock >> > (result.value, result.value, gradient.value, input.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
tanh_func = [] (const Matrix<double>& input) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_tanh_func << <blockPergrid, threadPerblock >> > (result.value, result.value, input.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dtanh_func = [] (const Matrix<double>& input, const Matrix<double>& gradient) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_dtanh_func << <blockPergrid, threadPerblock >> > (result.value, result.value, gradient.value, input.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
linear_func = [] (const Matrix<double>& input) {
	return input;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dlinear_func = [] (const Matrix<double>& input, const Matrix<double>& gradient) {
	return gradient;
};

std::function<Matrix<double>(const Matrix<double>&)> 
soft_max = [] (const Matrix<double>& input) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	
	device_exp_func << <blockPergrid, threadPerblock >> > (result.value, result.value, result.get_size());
	cudaDeviceSynchronize();

	double sum = 0;
	get_sum(&sum, result.value, result.get_size());

	device_softmax_func << <blockPergrid, threadPerblock >> > (result.value, result.value, sum, result.get_size());
	cudaDeviceSynchronize();
	
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
dsoft_max = [] (const Matrix<double>& input, const Matrix<double>& gradient) {
	Matrix<double> get_soft = soft_max(input);
	Matrix<double> result(input.row, input.column);
	set_Matrix(result, 0);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_dsoftmax_func << <blockPergrid, threadPerblock >> > (result.value, get_soft.value, gradient.value, result.get_size());
	cudaDeviceSynchronize();

	
	return result;
};

std::function<Matrix<double>(const Matrix<double>&)> 
descale_func = [] (const Matrix<double>& input) {
	Matrix<double> result(input);
	double max_value = 0;
	get_max(&max_value, result.value, result.get_size());
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_plus_func << <blockPergrid, threadPerblock >> > (result.value, result.value, -max_value, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)> 
ddescale_func = [] (const Matrix<double>& input, const Matrix<double>& gradient) {
	return gradient;
};

std::function<double(const Matrix<double>&, const Matrix<double>&)> 
catagorical_CEnt_loss_func = [] (const Matrix<double>& input, const Matrix<double>& target) {
	double* cpy;
	cudaMalloc(&cpy, input.get_sizeb());
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_ccentloss_func << <blockPergrid, threadPerblock >> > (cpy, input.value, target.value, input.get_size());
	cudaDeviceSynchronize();
	double result = 0;
	get_sum(&result, cpy, input.get_size());
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)> 
dcatagorical_CEnt_loss_func = [] (const Matrix<double>& input,const Matrix<double>& target) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_dccentloss_func << <blockPergrid, threadPerblock >> > (result.value, result.value, target.value, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<double(const Matrix<double>&, const Matrix<double>&)>
squere_mean_loss_func = [](const Matrix<double>& input, const Matrix<double>& target) {
	double* cpy;
	cudaMalloc(&cpy, input.get_sizeb());
	int blockPergrid = upper_value(double(input.get_size()) / 1024);
	int threadPerblock = std::min(input.get_size(), 1024);
	device_smeanloss_func << <blockPergrid, threadPerblock >> > (cpy, input.value, target.value, input.get_size());
	cudaDeviceSynchronize();
	double result = 0;
	get_sum(&result, cpy, input.get_size());
	cudaFree(cpy);
	return result;
};

std::function<Matrix<double>(const Matrix<double>&, const Matrix<double>&)>
dsquere_mean_loss_func = [](const Matrix<double>& input, const Matrix<double>& target) {
	Matrix<double> result(input);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_dsmeanloss_func << <blockPergrid, threadPerblock >> > (result.value, result.value, target.value, result.get_size());
	cudaDeviceSynchronize();
	return result;
};

std::function<double()> 
normal_rand_func = [] () {
	return double(std::rand() % 10000) / 10000;
};



__host__ __device__ double mapping(const double& value, const double& min1, const double& max1, const double& min2, const double& max2) {
	return ((value - min1) / (max1 - min1) * (max2 - min2)) + min2;
}

Matrix<double> mul_each(const Matrix<double>& value1, const Matrix<double>& value2) { // directly multiply a[i][j] and b[i][j]
	if (value1.row != value2.row || value1.column != value2.column)
		throw "invalid multiply each elemenet";
	Matrix<double> result(value1.row,value1.column);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_mul_func << <blockPergrid, threadPerblock >> > (result.value, value1.value, value2.value, result.get_size());
	cudaDeviceSynchronize();
	return result;
}

Matrix<double> devide_each(const Matrix<double>& value, const Matrix<double>& divisor) {
	if(value.row != divisor.row || value.column != divisor.column)
		throw "invalid devision each elemenet";
	Matrix<double> result(value.row, value.column);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_devide_func << <blockPergrid, threadPerblock >> > (result.value, value.value, divisor.value, result.get_size());
	cudaDeviceSynchronize();
	return result;
}

Matrix<double> pow_each(const Matrix<double>& value, const double exponent) {
	Matrix<double> result(value.row, value.column);
	int blockPergrid = upper_value(double(result.get_size()) / 1024);
	int threadPerblock = std::min(result.get_size(), 1024);
	device_pow_func << <blockPergrid, threadPerblock >> > (result.value, value.value, exponent, result.get_size());
	cudaDeviceSynchronize();
	return result;
}

void set_Matrix(Matrix<double>& M, double value) { // set every Matrix's member to specific number
	int blockPergrid = upper_value(double(M.get_size()) / 1024);
	int threadPerblock = std::min(M.get_size(), 1024);
	device_set_matrix << <blockPergrid, threadPerblock >> > (M.value, value, M.get_size());
	cudaDeviceSynchronize();
}

void set_up_Matrix(Matrix<double>& M, const Matrix<double>& B) {
	if (!M.is_constructed()) {
		M.reconstruct(B.row, B.column);
		set_Matrix(M, 0);
	}
}

double get_max(const Matrix<double>& M) { // get max value of the Matrix
	double max_value = 0;
	get_max(&max_value, M.value, M.get_size());
	return max_value;
}

double get_min(const Matrix<double>& M) { // get min value of the Matrix
	double min_value = 0;
	get_min(&min_value, M.value, M.get_size());
	return min_value;
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
	if (a == "squere")
		func = squere_func;
	else if (a == "sigmoid")
		func = sigmoid_func;
	else if (a == "tanh")
		func = tanh_func;
	else if (a == "linear")
		func = linear_func;
	else if (a == "soft_max")
		func = soft_max;
	else throw "function not found";
}

void universal_set_func(std::function<Matrix<double>(const Matrix<double>&,const Matrix<double>&)>& func, const std::string& setting, int& i) { // return function from string name (dfunc)
	std::string a = get_text(setting, i);
	if (a == "dsquere")
		func = dsquere_func;
	else if (a == "dsigmoid")
		func = dsigmoid_func;
	else if (a == "dtanh")
		func = dtanh_func;
	else if (a == "dlinear")
		func = dlinear_func;
	else if (a == "dsoft_max")
		func = dsoft_max;
	else throw "function not found";
}

__global__ void device_weightchange_computeLSTM(double* weight_change, const double* dgate, const double* value, const int weight_row, const int weight_column) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int r = pos / weight_column;
	int c = pos % weight_column;
	if (r < weight_row && c < weight_column) {
		weight_change[pos] += dgate[r] * value[c];
	}
}

__global__ void device_flow_computeLSTM(double* value_change, const double* dgate, const double* weight, const int weight_row, const int weight_column) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < weight_column) {
		for (int r = 0; r < weight_row; r++) {
			value_change[pos] += dgate[r] * weight[r * weight_column + pos];
		}
	}
}

__global__ void device_weightchange_computeDENSE(double* weight_change, const double* doutput, const double* value, const int weight_row, const int weight_column) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	int r = pos / weight_column;
	int c = pos % weight_column;
	if (r < weight_row && c < weight_column) {
		weight_change[pos] += doutput[r] * value[c];
	}
}

__global__ void device_flow_computeDENSE(double* value_change, const double* doutput, const double* weight, const int weightrow, const int weightcolumn) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < weightcolumn) {
		for (int r = 0; r < weightrow; r++) {
			value_change[pos] += doutput[r] * weight[r * weightcolumn + pos];
		}
	}
}

__global__ void device_predict_computeDROPOUT(double* value, const double keep_rate, const int size) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	if (pos < size) {
		value[pos] = value[pos] * keep_rate;
	}
}

__global__ void device_valuechange_computeDROPOUT(double* value_change, const double* gradient, const double* value, const int size) {
	int pos = blockIdx.x * blockDim.x + threadIdx.x;
	if (pos < size) {
		value_change[pos] = gradient[pos] * value[pos];
	}
}

__global__ void device_expochange_computePOWER(double* exponent_change, const double* gradient, const double* exponent, const double* value, const int size) {
	int pos = threadIdx.x + blockDim.x * blockIdx.x;
	if (pos < size) {
		exponent_change[pos] = pow(value[pos], exponent[pos]) * log(value[pos]) * gradient[pos];
	}
}

__global__ void device_valuechange_computePOWER(double* value_change, const double* gradient, const double* exponent, const double* value, const int size) {
	int pos = threadIdx.x + blockIdx.x * blockDim.x;
	if (pos < size) {
		if (exponent[pos] == 1.0) {
			value_change[pos] = log(value[pos]) * gradient[pos];
		}
		else {
			value_change[pos] = exponent[pos] * pow(value[pos], exponent[pos] - 1.0) * gradient[pos];
		}
	}
}
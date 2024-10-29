#pragma once

#include "Header.cuh"

int upper_value(const double& a);

template<typename t>
__global__ void thread_matrix_mul(t* result, const t* lhs, const t* rhs, const int _row, const int _k, const int _column) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int column = blockDim.y * blockIdx.y + threadIdx.y;
	if (row >= _row || column >= _column)
		return;

	t sum = 0;
	for (int i = 0; i < _k; i++) {
		sum += lhs[row * _k + i] * rhs[i * _column + column];
	}
	result[row * _column + column] = sum;
}

template <typename t>
__global__ void thread_matrix_plus(t* result, const t* lhs, const t* rhs, const int _row, const int _column) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int column = blockDim.y * blockIdx.y + threadIdx.y;
	if (row >= _row || column >= _column)
		return;

	result[row * _column + column] = lhs[row * _column + column] + rhs[row * _column + column];
}

template <typename t>
__global__ void thread_matrix_minus(t* result, const t* lhs, const t* rhs, const int _row, const int _column) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int column = blockDim.y * blockIdx.y + threadIdx.y;
	if (row >= _row || column >= _column)
		return;
	result[row * _column + column] = lhs[row * _column + column] - rhs[row * _column + column];
}

template<typename t>
__global__ void thread_number_mul(t* result, const t* lhs, const t number, const int _row, const int _column) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int column = blockDim.y * blockIdx.y + threadIdx.y;
	if (row >= _row || column >= _column)
		return;
	result[row * _column + column] = lhs[row * _column + column] * number;
}

template<typename t>
__global__ void thread_number_devide(t* result, const t* lhs, const t number, const int _row, const int _column) {
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int column = blockDim.y * blockIdx.y + threadIdx.y;
	if (row >= _row || column >= _column)
		return;
	result[row * _column + column] = lhs[row * _column + column] / number;
}

template<typename t>
__global__ void thread_number_plus(t* result, const t* M,const t p, const int size) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < size) {
		result[i] = M[i] + p;
	}
}

template <typename t>
__global__ void set_zero(t* value, const int _i, const int _j) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;
	if (i >= _i || j >= _j)
		return;

	value[i * _j + j] = 0;
}

template<typename t>
class Matrix {
public:
	Matrix() {};

	Matrix(const std::size_t& _row, const std::size_t& _column) : row(_row), column(_column) {
		declare();
	}

	Matrix(std::vector<std::vector<t>> _value) : row(_value.size()), column(_value[0].size()) {
		declare();

		for (int i = 0; i < _value.size(); i++) {
			if (_value[i].size() != column) {
				destroy();
				throw "Cant declare a Matrix with Non-squre shape";
			}
			cudaMemcpy(&(value[i * column]), &(_value[i][0]), column * sizeof(t), cudaMemcpyHostToDevice);
		}
	}

	Matrix(const Matrix& copy) : row(copy.row), column(copy.column) {
		declare();

		cudaMemcpy(value, copy.value, row * column * sizeof(t), cudaMemcpyDeviceToDevice);
	}

	Matrix(Matrix&& other) : row(std::move(other.row)), column(std::move(other.column)) {
		value = other.value;

		other.value = nullptr;
	}

	~Matrix() {
		destroy();
	}



	Matrix<t>& operator=(const Matrix<t>& rhs) {
		if (row != rhs.row || column != rhs.column) {
			destroy();
			declare(rhs.row, rhs.column);
		}

		cudaMemcpy(value, rhs.value, row * column * sizeof(t), cudaMemcpyDeviceToDevice);
		return (*this);
	}

	Matrix<t>& operator=(Matrix<t>&& other) {
		destroy();

		row = std::move(other.row);
		column = std::move(other.column);

		value = other.value;

		other.value = nullptr;

		return (*this);
	}

	Matrix<t> operator*(const Matrix<t>& rhs) {
		if (column != rhs.row)
			throw "illegal Matrix multification(lhs.column != rhs.row)";
		Matrix<t> result(row, rhs.column);

		int x_b = upper_value(double(row) / 32);
		int y_b = upper_value(double(rhs.column) / 32);
		int x_t = std::min(row, std::size_t(32));
		int y_t = std::min(rhs.column, std::size_t(32));
		dim3 blockPergrid(x_b, y_b);
		dim3 threadPerblock(x_t, y_t);
		set_zero << <blockPergrid, threadPerblock >> > (result.value, row, rhs.column);
		cudaDeviceSynchronize();
		thread_matrix_mul << <blockPergrid, threadPerblock >> > (result.value, value, rhs.value, row, column, rhs.column);
		cudaDeviceSynchronize();
		return result;
	}

	Matrix<t> operator*(const t& rhs) {
		Matrix<t> result(row, column);
		int x_b = upper_value(double(row) / 32);
		int y_b = upper_value(double(column) / 32);
		int x_t = std::min(row, std::size_t(32));
		int y_t = std::min(column, std::size_t(32));
		dim3 blockPergrid(x_b, y_b);
		dim3 threadPerblock(x_t, y_t);
		thread_number_mul << <blockPergrid, threadPerblock >> > (result.value, value, rhs, row, column);
		return result;
	}

	Matrix<t> operator+(const Matrix<t>& rhs) {
		if (row != rhs.row || column != rhs.column)
			throw "Illegal Matrix sumasion(size are not equal)";
		Matrix<t> result(row, column);
		int x_b = upper_value(double(row) / 32);
		int y_b = upper_value(double(column) / 32);
		int x_t = std::min(row, std::size_t(32));
		int y_t = std::min(column, std::size_t(32));
		dim3 blockPergrid(x_b, y_b);
		dim3 threadPerblock(x_t, y_t);
		thread_matrix_plus << <blockPergrid, threadPerblock >> > (result.value, value, rhs.value, row, column);
		cudaDeviceSynchronize();
		return result;
	}

	Matrix<t> operator+(const t& rhs) {
		Matrix<t> result(row, column);
		int blockPergrid = upper_value(double(result.get_size()) / 1024);
		int threadPerblock = std::min(result.get_size(), 1024);
		thread_number_plus << <blockPergrid, threadPerblock >> > (result.value, value, rhs, result.get_size());
		cudaDeviceSynchronize();
		return result;
	}
	
	Matrix<t> operator-(const Matrix<t>& rhs) {
		if (row != rhs.row || column != rhs.column)
			throw "Illegal Matrix subtract(size are not equal)";
		Matrix<t> result(row, column);
		int x_b = upper_value(double(row) / 32);
		int y_b = upper_value(double(column) / 32);
		int x_t = std::min(row, std::size_t(32));
		int y_t = std::min(column, std::size_t(32));
		dim3 blockPergrid(x_b, y_b);
		dim3 threadPerblock(x_t, y_t);
		thread_matrix_minus << <blockPergrid, threadPerblock >> > (result.value, value, rhs.value, row, column);
		return result;
	}

	Matrix<t> operator/(const t& rhs) {
		Matrix<t> result(row, column);
		int x_b = upper_value(double(row) / 32);
		int y_b = upper_value(double(column) / 32);
		int x_t = std::min(row, std::size_t(32));
		int y_t = std::min(column, std::size_t(32));
		dim3 blockPergrid(x_b, y_b);
		dim3 threadPerblock(x_t, y_t);
		thread_number_devide << <blockPergrid, threadPerblock >> > (result.value, value, rhs, row, column);
		return result;
	}



	void print() const {
		t* copy = new t[row * column];
		cudaMemcpy(copy, value, sizeof(t) * row * column, cudaMemcpyDeviceToHost);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				std::cout << copy[i * column + j] << "\t";
				if ((i * column + j) % 6 == 5)
					std::cout << std::endl;
			}
		}
		std::cout << std::endl;
	}

	void print(std::ofstream& output_file) const {
		t* copy = new t[row * column];
		cudaMemcpy(copy, value, sizeof(t) * row * column, cudaMemcpyDeviceToHost);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				output_file << copy[i * column + j] << "\t";
			}output_file << "\n";
		}
		output_file << std::endl;
	}


	void load(std::ifstream& input_file) {
		t* copy = new t[row * column];
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				input_file >> copy[i * column + j];
			}
		}
		cudaMemcpy(value, copy, sizeof(t) * row * column, cudaMemcpyHostToDevice);
		delete copy;
	}


	void reconstruct(const std::size_t& _row, const std::size_t& _column) {
		destroy();
		declare(_row, _column);
	}

	void reconstruct(const Matrix<t>& copy) {
		destroy();

		row = copy.row;
		column = copy.column;

		declare();

		cudaMemcpy(value, copy.value, row * column * sizeof(t), cudaMemcpyDeviceToDevice);
	}

	void reconstruct(Matrix<t>&& other) {
		destroy();

		row = std::move(other.row);
		column = std::move(other.column);

		value = other.value;

		other.value = nullptr;
	}



	constexpr int get_size() const { return column * row; }

	constexpr int get_sizeb() const { return column * row * sizeof(t); }

	bool is_constructed() const {
		return value != nullptr;
	}
	
	std::size_t row = NULL;
	std::size_t column = NULL;
	t* value = nullptr;
private:


	void destroy() {
		if (value != nullptr)
			cudaFree(value);
	}

	void declare() {
		if (row == NULL || column == NULL)
			value = nullptr;

		cudaMalloc(&value, row * column * sizeof(t));
	}

	void declare(const std::size_t& _row, const std::size_t& _column) {
		row = _row;
		column = _column;
		declare();
	}

	
};
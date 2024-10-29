#pragma once

#include "Header.h"

template<typename t>
class Matrix {
public:
	Matrix();

	Matrix(const std::size_t& _row, const std::size_t& _column);

	Matrix(const std::initializer_list<std::initializer_list<t>>& init);

	Matrix(const std::vector<std::vector<t>>& _value);

	Matrix(const Matrix& copy);

	Matrix(Matrix&& move) noexcept;

	~Matrix();



	Matrix<t>& operator=(const Matrix<t>& rhs);

	Matrix<t>& operator=(const t& rhs);

	Matrix<t>& operator=(Matrix<t>&& move);



	Matrix<t> operator+(const Matrix<t>& rhs) const;

	Matrix<t> operator+(const t& rhs) const;

	Matrix<t> operator-(const Matrix<t>& rhs) const;

	Matrix<t> operator-(const t& rhs) const;

	Matrix<t> operator*(const Matrix<t>& rhs) const;

	Matrix<t> operator*(const t& rhs) const;

	Matrix<t> operator/(const Matrix<t>& rhs) const;

	Matrix<t> operator/(const t& rhs) const;




	Matrix<t> operator+=(const Matrix<t>& rhs);

	Matrix<t> operator+=(const t& rhs);

	Matrix<t> operator-=(const Matrix<t>& rhs);

	Matrix<t> operator-=(const t& rhs);

	Matrix<t> operator*=(const Matrix<t>& rhs);

	Matrix<t> operator*=(const t& rhs);

	Matrix<t> operator/=(const Matrix<t>& rhs);

	Matrix<t> operator/=(const t& rhs);



	bool operator==(const Matrix<t>& rhs) const;

	bool operator==(const t& rhs) const;

	bool operator!=(const Matrix<t>& rhs) const;

	bool operator!=(const t& rhs) const;



	t*& operator[](const std::size_t& pos);

	const t* operator[](const std::size_t& pos) const;



	void print() const;



	void reconstruct(const std::size_t& _row, const std::size_t& _column);

	void reconstruct(const std::initializer_list<t>& init);

	void reconstruct(const std::initializer_list<std::initializer_list<t>>& init);

	void reconstruct(const std::vector<std::vector<t>>& _value);

	void reconstruct(const Matrix<t>& copy);

	void reconstruct(Matrix<t>&& move) noexcept;


	Matrix<t> transpose() const;

	Matrix<t> adjugate() const;

	const t det() const;

	Matrix<t> inverse() const;



	std::size_t get_row() const;

	std::size_t get_column() const;



	void save_as(std::ofstream& output_file);

	void load(std::ifstream& input_file);



	bool is_constructed() const;
private:
	const t det(bool* considered_row, bool* considering_col, const std::size_t& size) const;

	void destroy();

	void declare();

	void declare(const std::size_t& _row, const std::size_t& _column);

	std::size_t row = NULL;
	std::size_t column = NULL;
	t** value = nullptr;
};


template<typename t>
Matrix<t>::Matrix() {}

template<typename t>
Matrix<t>::Matrix(const std::size_t& _row, const std::size_t& _column) {
	reconstruct(_row, _column);
}

template<typename t>
Matrix<t>::Matrix(const std::initializer_list<std::initializer_list<t>>& init) {
	reconstruct(init);
}

template<typename t>
Matrix<t>::Matrix(const std::vector<std::vector<t>>& _value) {
	reconstruct(_value);
}

template<typename t>
Matrix<t>::Matrix(const Matrix& copy) {
	reconstruct(copy);
}

template<typename t>
Matrix<t>::Matrix(Matrix&& move) noexcept {
	reconstruct(move);
}

template<typename t>
Matrix<t>::~Matrix() {
	destroy();
}


template<typename t>
Matrix<t>& Matrix<t>::operator=(const Matrix<t>& rhs) {
	reconstruct(rhs);
	return (*this);
}

template<typename t>
Matrix<t>& Matrix<t>::operator=(const t& rhs) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			value[i][j] = rhs;
		}
	}
	return *this;
}

template<typename t>
Matrix<t>& Matrix<t>::operator=(Matrix<t>&& move) {
	reconstruct(std::move(move));
	return (*this);
}




template<typename t>
Matrix<t> Matrix<t>::operator+(const Matrix<t>& rhs) const {
	if (row != rhs.row || column != rhs.column) {
		throw std::runtime_error("Can not add Matrix with different size");
	}

	Matrix<t> result(row, column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			result[i][j] = value[i][j] + rhs[i][j];
		}
	}
	return result;
}

template<typename t>
Matrix<t> Matrix<t>::operator+(const t& rhs) const {
	Matrix<t> result(row, column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			result[i][j] = value[i][j] + rhs;
		}
	}
	return result;
}

template<typename t>
Matrix<t> Matrix<t>::operator-(const Matrix<t>& rhs) const {
	Matrix<t> result(row, column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			result[i][j] = value[i][j] - rhs[i][j];
		}
	}
	return result;
}

template<typename t>
Matrix<t> Matrix<t>::operator-(const t& rhs) const {
	Matrix<t> result(row, column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			result[i][j] = value[i][j] - rhs;
		}
	}
	return result;
}

template<typename t>
Matrix<t> Matrix<t>::operator*(const Matrix<t>& rhs) const {
	if (column != rhs.row) {
		throw std::runtime_error("Illegal Matrix multification");
	}

	Matrix<double> result(row, rhs.column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < rhs.column; j++) {
			t sum = 0;
			for (int k = 0; k < column; k++) {
				sum += value[i][k] * rhs.value[k][j];
			}
			result[i][j] = sum;
		}
	}
	return result;
}

template<typename t>
Matrix<t> Matrix<t>::operator*(const t& rhs) const {
	Matrix<t> result(row, column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			result[i][j] = value[i][j] * rhs;
		}
	}
	return result;
}

template<typename t>
Matrix<t> Matrix<t>::operator/(const Matrix<t>& rhs) const {
	return (*this) * rhs.inverse();
}

template<typename t>
Matrix<t> Matrix<t>::operator/(const t& rhs) const {
	Matrix<t> result(row, column);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			result[i][j] = value[i][j] / rhs;
		}
	}
	return result;
}




template<typename t>
Matrix<t> Matrix<t>::operator+=(const Matrix<t>& rhs) {
	return (*this) = (*this) + rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator+=(const t& rhs) {
	return (*this) = (*this) + rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator-=(const Matrix<t>& rhs) {
	return (*this) = (*this) - rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator-=(const t& rhs) {
	return (*this) = (*this) - rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator*=(const Matrix<t>& rhs) {
	return (*this) = (*this) * rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator*=(const t& rhs) {
	return (*this) = (*this) * rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator/=(const t& rhs) {
	return (*this) = (*this) / rhs;
}

template<typename t>
Matrix<t> Matrix<t>::operator/=(const Matrix<t>& rhs) {
	return (*this) = (*this) / rhs;
}



template<typename t>
bool Matrix<t>::operator==(const Matrix<t>& rhs) const {
	if (row != rhs.row || column != rhs.column) {
		return false;
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if (value[i][j] != rhs[i][j]) {
				return false;
			}
		}
	}
	return true;
}

template<typename t>
bool Matrix<t>::operator==(const t& rhs) const {
	if (row == NULL || column == NULL) {
		return false;
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if (value[i][j] != rhs) {
				return false;
			}
		}
	}
	return true;
}

template<typename t>
bool Matrix<t>::operator!=(const Matrix<t>& rhs) const {
	if (row != rhs.row || column != rhs.column) {
		return true;
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if (value[i][j] == rhs[i][j]) {
				return false;
			}
		}
	}
	return true;
}

template<typename t>
bool Matrix<t>::operator!=(const t& rhs) const {
	if (row == NULL || column == NULL) {
		return true;
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if (value[i][j] != rhs) {
				return true;
			}
		}
	}
	return false;
}


template<typename t>
t*& Matrix<t>::operator[](const std::size_t& pos) {
	return value[pos];
}

template<typename t>
const t* Matrix<t>::operator[](const std::size_t& pos) const {
	return value[pos];
}



template<typename t>
void Matrix<t>::print() const {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			std::cout << value[i][j] << ' ';
		}
		std::cout << std::endl;
	}
}


template<typename t>
void Matrix<t>::reconstruct(const std::size_t& _row, const std::size_t& _column) {
	if (row != _row || column != _column) {
		row = _row;
		column = _column;
		destroy();
		declare(_row, _column);
	}
}

template<typename t>
void Matrix<t>::reconstruct(const std::initializer_list<t>& init) {
	reconstruct(init.size(), 1);

	for (int i = 0; i < row; i++) {
		value[i][0] = *(init.begin() + i);
	}
}

template<typename t>
void Matrix<t>::reconstruct(const std::initializer_list<std::initializer_list<t>>& init) {
	reconstruct(init.size(), init.begin()->size());

	for (int i = 0; i < row; i++) {
		if ((*(init.begin() + i)).size() != column) {
			destroy();
			throw std::runtime_error("Non-squre initializer can not be used to initilize");
		}
		for (int j = 0; j < column; j++) {
			value[i][j] = *((*(init.begin() + i)).begin() + j);
		}
	}
}

template<typename t>
void Matrix<t>::reconstruct(const std::vector<std::vector<t>>& _value) {
	destroy();
	declare(_value.size(), value[0].size());

	for (int i = 0; i < _value.size(); i++) {
		if (_value[i].size() != column) {
			destroy();
			throw  std::runtime_error("Non-squre Matrix can not be used to initilize");
		}
		for (int j = 0; j < _value[i].size(); j++) {
			value[i][j] = _value[i][j];
		}
	}
}

template<typename t>
void Matrix<t>::reconstruct(const Matrix<t>& copy) {
	reconstruct(copy.row, copy.column);

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			value[i][j] = copy.value[i][j];
		}
	}
}

template<typename t>
void Matrix<t>::reconstruct(Matrix<t>&& move) noexcept {
	destroy();
	
	row = move.row;
	column = move.column;

	value = move.value;

	move.value = nullptr;
	move.row = NULL;
	move.column = NULL;
}


template<typename t>
Matrix<t> Matrix<t>::transpose() const {
	Matrix<t> result(column, row);

	for (int i = 0; i < result.row; i++) {
		for (int j = 0; j < result.column; j++) {
			result.value[i][j] = value[j][i];
		}
	}

	return result;
}

template <typename t>
Matrix<t> Matrix<t>::adjugate() const {
	if (row != column) {
		throw std::runtime_error("Non-squre Matrix can not find det");
	}

	Matrix<t> result(row, column);

	const std::size_t size = row;
	bool* considered_row = new bool[size];
	bool* considered_col = new bool[size];
	for (int i = 0; i < size; i++) {
		considered_row[i] = true;
		considered_col[i] = true;
	}

	int a;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			if ((i + j) % 2 == 0) {
				a = 1;
			}
			else {
				a = -1;
			}

			considered_row[i] = false;
			considered_col[j] = false;

			result.value[i][j] = a * det(considered_row, considered_col, size - 1);

			considered_row[i] = true;
			considered_col[j] = true;
		}
	}

	return result;
}

template <typename t>
const t Matrix<t>::det() const {
	if (row != column) {
		throw std::runtime_error("Non-squre Matrix can not find det");
	}

	const std::size_t size = row;
	bool* considered_row = new bool[size];
	bool* considered_col = new bool[size];
	for (int i = 0; i < size; i++) {
		considered_row[i] = true;
		considered_col[i] = true;
	}

	return det(considered_row, considered_col, size);
}

template<typename t>
Matrix<t> Matrix<t>::inverse() const {
	if (row != column) {
		throw std::runtime_error("Non-squre Matrix can not find det");
	}

	const t d = det();
	if (d == 0 || d != d) {
		throw std::runtime_error("0 det Matrix does not have inverse");
	}
	return this->adjugate().transpose() / d;
}


template<typename t>
std::size_t Matrix<t>::get_row() const { return row; }

template<typename t>
std::size_t Matrix<t>::get_column() const { return column; }



template<typename t>
void Matrix<t>::save_as(std::ofstream& output_file) {
	output_file << row << " " << column << "\n";

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			output_file << value[i][j] << " ";
		}
		output_file << "\n";
	}
}

template<typename t>
void Matrix<t>::load(std::ifstream& input_file) {
	int get_row, get_column;
	input_file >> get_row >> get_column;
	if (row != get_row || column == get_column) {
		destroy();
		declare(row, column);
	}

	for (int i = 0; i < row; i++) {
		for (int j = 0; j < column; j++) {
			input_file >> value[i][j];
		}
	}
}

template<typename t>
bool Matrix<t>::is_constructed() const {
	return value != nullptr;
}

template <typename t>
const t Matrix<t>::det(bool* considering_row, bool* considering_col, const std::size_t& size) const {
	if (size == 0) {
		return 1;
	}

	t result = 0;

	int a = -1;

	int r = 0;
	while (!considering_row[r]) r++;

	for (int i = 0; i < column; i++) {
		if (considering_row[r] && considering_col[i]) {
			a *= -1;
			considering_row[r] = false;
			considering_col[i] = false;

			result += a * value[r][i] * det(considering_row, considering_col, size - 1);

			considering_row[r] = true;
			considering_col[i] = true;
		}
	}

	return result;
}

template<typename t>
void Matrix<t>::destroy() {
	if (value != nullptr) {
		for (int i = 0; i < row; i++) {
			if (value[i] != nullptr) {
				delete value[i];
			}
		}
		delete[] value;
	}

	value = nullptr;
}

template<typename t>
void Matrix<t>::declare() {
	if (row < 0 || column < 0) {
		row = NULL;
		column = NULL;
		throw std::runtime_error("Negative size Matrix declaration");
	}

	if (row == NULL || column == NULL)
		return;

	value = new t * [row];
	for (int i = 0; i < row; i++) {
		value[i] = new t[column];
	}
}

template<typename t>
void Matrix<t>::declare(const std::size_t& _row, const std::size_t& _column) {
	row = _row;
	column = _column;
	declare();
}
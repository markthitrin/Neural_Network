#pragma once

#include "Header.h"

template<typename t>
class Matrix {
public:
	Matrix() {};

	Matrix(const std::size_t& _row, const std::size_t& _column) : row(_row), column(_column) {
		declare();
	}

	Matrix(std::vector<std::vector<t>> _value) : row(_value.size()) , column(_value[0].size()) {
		declare();

		for (int i = 0; i < _value.size(); i++) {
			if (_value[i].size() != column) {
				destroy();
				throw "Cant declare a Matrix with Non-squre shape";
			}
			for (int j = 0; j < _value[i].size(); j++) {
				value[i][j] = _value[i][j];
			}
		}
	}

	Matrix(const Matrix& copy) : row(copy.row) , column(copy.column) {
		declare();

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				value[i][j] = copy.value[i][j];
			}
		}
	}

	~Matrix() {
		for (int i = 0; i < row; i++) {
			delete value[i];
		}
		delete[] value;
	}



	Matrix<t>& operator=(const Matrix<t>& rhs) {
		destroy();
		declare(rhs.row, rhs.column);

		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				value[i][j] = rhs.value[i][j];
			}
		}
		return (*this);
	}

	Matrix<t> operator*(const Matrix<t>& rhs) {
		if (column != rhs.row)
			throw "illegal Matrix multification(lhs.column != rhs.row)";
		Matrix<t> result(row,rhs.column);
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

	Matrix<t> operator*(const double& rhs) {
		Matrix<t> result(row, column);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				result[i][j] = value[i][j] * rhs;
			}
		}
		return result;
	}

	Matrix<t> operator+(const Matrix<t>& rhs) {
		if (row != rhs.row || column != rhs.column) {
			throw "Illegal Matrix sumasion(size are not equal)";
		}
		Matrix<t> result(row, column);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				result[i][j] = value[i][j] + rhs[i][j];
			}
		}
		return result;
	}

	Matrix<t> operator+(const double& rhs) {
		Matrix<t> result(row, column);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				result[i][j] = value[i][j] + rhs;
			}
		}
		return result;
	}

	Matrix<t> operator-(const Matrix<t>& rhs) {
		Matrix<double> result(row, column);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				result[i][j] = value[i][j] - rhs[i][j];
			}
		}
		return result;
	}

	Matrix<t> operator/(const double& rhs) {
		Matrix<t> result(row, column);
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				result[i][j] = value[i][j] / rhs;
			}
		}
		return result;
	}



	t*& operator[](const std::size_t& pos) {
		return value[pos];
	}

	const t* operator[](const std::size_t& pos) const {
		return value[pos];
	}

	void print() const {
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < column; j++) {
				std::cout << value[i][j] << ' ';
			}
			std::cout << std::endl;
		}
	}



	void reconstruct(const std::size_t& _row, const std::size_t& _column) {
		destroy();
		declare(_row, _column);
	}


	
	std::size_t get_row() const { return row; }

	std::size_t get_column() const { return column; }


	
	bool is_constructed() const  {
		return value != nullptr;
	}
private:
	void destroy() {
		if (value != nullptr) {
			for (int i = 0; i < row; i++) {
				delete value[i];
			}
			delete[] value;
		}
		row = NULL;
		column = NULL;
		value = nullptr;
	}

	void declare() {
		if (row == NULL || column == NULL)
			throw "undefine Matrix cant been declared";
		value = new t * [row];
		for (int i = 0; i < row; i++) {
			value[i] = new t[column];
		}
	}

	void declare(const std::size_t& _row, const std::size_t& _column) {
		row = _row;
		column = _column;
		declare();
	}

	std::size_t row = NULL;
	std::size_t column = NULL;
	t** value = nullptr;
};
#include "Matrix.h"
#include <memory>
Matrix::Matrix(int rows, int columns)
{
	_Cmatrix.columns = columns;
	_Cmatrix.rows = rows;
}
matrix Matrix::getCMatrix() const
{
	return _Cmatrix;
}
const int Matrix::getRows() const
{
	return _Cmatrix.rows;
}

const int Matrix::getColumns() const
{
	return _Cmatrix.columns;
}

const long Matrix::getElementsCount() const
{
	return (long)getRows() * getRows();
}

void Matrix::allocateMatrix()
{
	if (_allocated == true) {
		throw std::exception();
		return;
	}
	_Cmatrix.elements = new double[getElementsCount()];
	_allocated = true;
}

void Matrix::deallocateMatrix()
{
	if (_allocated == false) {
		throw std::exception();
		return;
	}
	delete[] _Cmatrix.elements;
	_allocated = false;
}

Matrix::~Matrix()
{
}
#include "Matrix.h"
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

const int Matrix::matrixMultiply() const
{
	return 0;
}

const long Matrix::getElementsCount() const
{
	return (long)getRows() * getRows();
}

Matrix::~Matrix()
{
}
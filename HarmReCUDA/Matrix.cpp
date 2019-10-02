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

const long Matrix::getElementsCount() const
{
	return (long)getRows() * getRows();
}

const void Matrix::allocateMatrix() const
{
	
}

const void Matrix::deallocateMatrix() const
{
	
}

Matrix::~Matrix()
{
}
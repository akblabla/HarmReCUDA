#include "AMatrix.hpp"
AMatrix::AMatrix(int rows, int columns)
{
	_Cmatrix.columns = columns;
	_Cmatrix.rows = rows;
}
matrix AMatrix::getCMatrix() const
{
	return _Cmatrix;
}
int AMatrix::getRows() const
{
	return _Cmatrix.rows;
}

int AMatrix::getColumns() const
{
	return _Cmatrix.columns;
}

long AMatrix::getElementsCount() const
{
	return getRows() * getColumns();
}
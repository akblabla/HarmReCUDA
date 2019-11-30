#include "AMatrix.hpp"
AMatrix::AMatrix(int rows, int columns)
{
	_Cmatrix.columns = columns;
	_Cmatrix.rows = rows;
	_Cmatrix.ld = rows;
}
AMatrix::~AMatrix()
{
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

int AMatrix::getLeadingDimension() const
{
	return _Cmatrix.ld;
}

long AMatrix::getElementsCount() const
{
	return getRows() * getColumns();
}

bool AMatrix::isAllocated() const
{
	return _allocated;
}

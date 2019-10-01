#include "AMatrix.h"

AMatrix::AMatrix(int rows, int columns)
{
	_CMatrix.columns = columns;
	_CMatrix.rows = rows;
}

matrix AMatrix::getCMatrix() const
{
	return _CMatrix;
}
const int AMatrix::getRows() const
{
	return _CMatrix.rows;
}

const int AMatrix::getColumns() const
{
	return _CMatrix.columns;
}

const long AMatrix::getElementsCount() const
{
	return (long)getRows() * getRows();
}
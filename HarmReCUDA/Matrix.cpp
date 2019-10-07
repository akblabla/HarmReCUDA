#include "Matrix.h"
#include <memory>
#include <iostream>
#include <algorithm>
Matrix::Matrix(int rows, int columns, matrixInitialisation::matrixInitialisation initialisation)
{
	_Cmatrix.columns = columns;
	_Cmatrix.rows = rows;
	switch (initialisation)
	{
	case matrixInitialisation::no_init:
		break;
	case matrixInitialisation::allocate:
		allocate();
		break;
	case matrixInitialisation::assign:
		allocate();
		break;
	default:
		break;
	}

}
matrix Matrix::getCMatrix() const
{
	return _Cmatrix;
}
int Matrix::getRows() const
{
	return _Cmatrix.rows;
}

int Matrix::getColumns() const
{
	return _Cmatrix.columns;
}

long Matrix::getElementsCount() const
{
	return getRows() * getColumns();
}

void Matrix::allocate()
{
	if (_allocated == true) {
		return;
	}
	_Cmatrix.elements = new double[getElementsCount()];
	_allocated = true;
}

void Matrix::deallocate()
{
	if (_allocated == false) {
		return;
	}
	delete[] _Cmatrix.elements;
	_allocated = false;
}

double Matrix::getElement(int row, int column)
{
	return _Cmatrix.elements[column *getRows()+ row];
}

void Matrix::setElement(double value, int row, int column)
{
	_Cmatrix.elements[column * getRows() + row] = value;
}

void Matrix::print(int rows, int columns)
{
	for (int j = 0; j < std::min(getRows(), rows); j++) {
		for (int i = 0; i < std::min(getColumns(), columns); i++) {
			printf("%10.3f", getElement(j, i));
		}
		printf("\n");
	}
}

Matrix::~Matrix()
{
}
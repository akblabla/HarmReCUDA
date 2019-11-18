#include "Matrix.hpp"
#include <memory>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>
#include "helper_cuda.h"
Matrix::Matrix(int rows, int columns, matrixInitialisation initialisation) : AMatrix(rows, columns)
{
	switch (initialisation)
	{
	case matrixInitialisation::M_NO_INIT:
		break;
	case matrixInitialisation::M_ALLOCATE:
		allocate();
		break;
	case matrixInitialisation::M_ASSIGN:
		allocate();
		break;
	default:
		break;
	}

}

Matrix::Matrix(const Matrix_d& src, matrixInitialisation initialisation) : AMatrix(src.getRows(), src.getColumns())
{
	switch (initialisation)
	{
	case matrixInitialisation::M_NO_INIT:
		break;
	case matrixInitialisation::M_ALLOCATE:
		allocate();
		break;
	case matrixInitialisation::M_ASSIGN:
		allocate();
		downloadFromDevice(src);
		break;
	default:
		break;
	}
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
	print(0, rows, 0, columns);
}

void Matrix::print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd)
{
	//treat negative numbers as uncapped
	if (rowsEnd < 0) {
		rowsEnd = getRows();
	}
	if (columnsEnd < 0) {
		columnsEnd = getColumns();
	}
	rowsStart = std::min(getRows(), rowsStart);
	rowsEnd = std::min(getRows(), rowsEnd);
	columnsStart = std::min(getColumns(), columnsStart);
	columnsEnd = std::min(getColumns(), columnsEnd);
	for (int j = rowsStart; j < rowsEnd; j++) {
		for (int i = columnsStart; i < columnsEnd; i++) {
			printf("%10.6lf", getElement(j, i));
		}
		printf("\n");
	}
}

void Matrix::printToFile(const char* filename, int rows, int columns)
{
	//treat negative numbers as uncapped
	if (rows < 0) {
		rows = INT_MAX;
	}
	if (columns < 0) {
		columns = INT_MAX;
	}
	auto fp = fopen(filename, "w");
	for (int j = 0; j < std::min(getRows(), rows); j++) {
		fprintf(fp,"\t%.6f", getElement(j, 0));
		for (int i = 1; i < std::min(getColumns(), columns); i++) {
			fprintf(fp, "\t%.6f", getElement(j, i));
		}
		if (j < getRows()-1){
			fprintf(fp, "\n");
		}
	}
}

void Matrix::downloadFromDevice(const Matrix_d& src)
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception("Matrix dimension mismatch");
		return;
	}
	auto const cudaStat = cudaMemcpy(getCMatrix().elements, src.getCMatrix().elements, getElementsCount() * sizeof(*_Cmatrix.elements), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
}

Matrix::~Matrix()
{
}
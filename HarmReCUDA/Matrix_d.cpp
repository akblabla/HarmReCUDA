#include "Matrix_d.h"
#include <cuda_runtime.h>
#include <exception>

Matrix_d::Matrix_d(int rows, int columns) : Matrix(rows,columns)
{
	cudaError_t cudaStat;
	_Cmatrix.columns = columns;
	_Cmatrix.rows = rows;
}

matrix Matrix_d::getCMatrix() const
{
	return _Cmatrix;
}
const int Matrix_d::getRows() const
{
	return _Cmatrix.rows;
}

const int Matrix_d::getColumns() const
{
	return _Cmatrix.columns;
}

const long Matrix_d::getElementsCount() const
{
	return (long)getRows() * getRows();
}

void Matrix_d::allocateMatrixOnDevice()
{
	if (_allocatedToDevice == true) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMalloc((void**)& _Cmatrix.elements, getElementsCount() * sizeof(*_Cmatrix.elements));
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocatedToDevice = true;
}

void Matrix_d::deallocateMatrixOnDevice()
{
	if (_allocatedToDevice == false) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaFree((void**)& _Cmatrix.elements);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocatedToDevice = false;
}

void Matrix_d::uploadMatrixToDevice(Matrix& src) const
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMemcpy(_Cmatrix.elements, src.getCMatrix().elements, src.getElementsCount() * sizeof(*src.getCMatrix().elements), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
}

void Matrix_d::downloadMatrixFromDevice(Matrix& dest) const
{
	if (getRows() != dest.getRows() || getColumns() != dest.getColumns()) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMemcpy(dest.getCMatrix().elements, _Cmatrix.elements, getElementsCount() * sizeof(*_Cmatrix.elements), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
}
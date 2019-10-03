#include "Matrix_d.h"
#include <cuda_runtime.h>
#include <exception>

Matrix_d::Matrix_d(int rows, int columns) : Matrix(rows,columns)
{
}

void Matrix_d::allocateMatrix()
{
	if (_allocated == true) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMalloc((void**)&_Cmatrix.elements, getElementsCount() * sizeof(*_Cmatrix.elements));
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocated = true;
}

void Matrix_d::deallocateMatrix()
{
	if (_allocated == false) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaFree((void*) _Cmatrix.elements);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocated = false;
}

void Matrix_d::uploadMatrixToDevice(Matrix& src) const
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMemcpy(_Cmatrix.elements, src.getCMatrix().elements, src.getElementsCount() * sizeof(*src.getCMatrix().elements), cudaMemcpyHostToDevice);
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
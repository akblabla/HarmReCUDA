#include "Matrix_d.h"
#include <cuda_runtime.h>
#include <exception>

Matrix_d::Matrix_d(int rows, int columns) : AMatrix(rows, columns)
{
}

void Matrix_d::allocateMatrixOnDevice()
{
	if (_allocatedToDevice == true) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMalloc((void**)& _CMatrix.elements, getElementsCount() * sizeof(*_CMatrix.elements));
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
	auto const cudaStat = cudaFree((void**)& _CMatrix.elements);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocatedToDevice = false;
}

void Matrix_d::uploadMatrixToDevice(Matrix_cpu src) const
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMemcpy(_CMatrix.elements, src.getCMatrix().elements, src.getElementsCount() * sizeof(*src.getCMatrix().elements), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
}

void Matrix_d::downloadMatrixFromDevice(Matrix_cpu dest) const
{
	if (getRows() != dest.getRows() || getColumns() != dest.getColumns()) {
		throw std::exception();
		return;
	}
	auto const cudaStat = cudaMemcpy(dest.getCMatrix().elements, _CMatrix.elements, getElementsCount() * sizeof(*_CMatrix.elements), cudaMemcpyDeviceToHost);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
}
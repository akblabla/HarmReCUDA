#include "Matrix_d.h"
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include "cublas_v2.h"
Matrix_d::Matrix_d(int rows, int columns) : Matrix(rows,columns)
{
}

void Matrix_d::allocate()
{
	if (_allocated == true) {
		return;
	}
	auto const cudaStat = cudaMalloc((void**)&_Cmatrix.elements, getElementsCount() * sizeof(*_Cmatrix.elements));
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocated = true;
}

void Matrix_d::deallocate() 
{
	if (_allocated == false) {
		return;
	}
	auto const cudaStat = cudaFree((void*) _Cmatrix.elements);
	if (cudaStat != cudaSuccess) {
		throw std::exception();
		return;
	}
	_allocated = false;
}

void Matrix_d::uploadToDevice(Matrix& src)
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

void Matrix_d::downloadFromDevice(Matrix& dest)
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

void Matrix_d::GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta)
{
	if (A.getColumns() != B.getRows()) {
		throw std::exception("Matrix multiplication failed: Incompatible matrix dimensions");
		return;
	}
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);

	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;
	int m = A.getRows();
	int n = B.getColumns();
	int k = B.getRows();
	const double* A_ = A.getCMatrix().elements;
	int lda = A.getRows();
	const double* B_ = B.getCMatrix().elements;
	int ldb = B.getRows();
	double* C_ = _Cmatrix.elements;
	int ldc = getRows();

	stat = cublasDgemm(handle, transa, transb, m, n, k, &alpha, A_, lda, B_, ldb, &beta, C_, ldc);
	cublasDestroy(handle);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception("Matrix multiplication failed: CuBLAS error");
		return;
	}
}


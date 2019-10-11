#include "Matrix_d.h"
#include <cuda_runtime.h>
#include <exception>
#include <iostream>
#include "cublas_v2.h"
#include "helper_cuda.h"
Matrix_d::Matrix_d(int rows, int columns, matrixInitialisation initialisation) : AMatrix(rows,columns)
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

Matrix_d::Matrix_d(const Matrix& src, matrixInitialisation initialisation) : AMatrix(src.getRows(), src.getColumns())
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
		uploadToDevice(src);
		break;
	default:
		break;
	}
}

Matrix_d::Matrix_d(const Matrix_d& src, matrixInitialisation initialisation) : AMatrix(src.getRows(), src.getColumns())
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
		copyFromDevice(src);
		break;
	default:
		break;
	}
}

void Matrix_d::allocate()
{
	if (_allocated == true) {
		return;
	}
	auto const cudaStat = cudaMalloc((void**)&_Cmatrix.elements, getElementsCount() * sizeof(*_Cmatrix.elements));
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
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
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
	_allocated = false;
}

void Matrix_d::uploadToDevice(const Matrix& src)
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception("Matrix dimension mismatch");
		return;
	}
	auto const cudaStat = cudaMemcpy(_Cmatrix.elements, src.getCMatrix().elements, src.getElementsCount() * sizeof(*src.getCMatrix().elements), cudaMemcpyHostToDevice);
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
}

void Matrix_d::copyFromDevice(const Matrix_d& src)
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception("Matrix dimension mismatch");
		return;
	}
	auto const cudaStat = cudaMemcpy(_Cmatrix.elements, src.getCMatrix().elements, src.getElementsCount() * sizeof(*src.getCMatrix().elements), cudaMemcpyDeviceToDevice);
	if (cudaStat != cudaSuccess) {
		throw std::exception(cudaGetErrorString(cudaStat));
		return;
	}
}

void Matrix_d::GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta)
{
	matrixTranspose transposeA = NO_TRANS;
	matrixTranspose transposeB = NO_TRANS;
	GeneralMatrixToMatrixMultiply(A, B, alpha, beta, transposeA, transposeB);
}

void Matrix_d::GeneralMatrixToMatrixMultiply(Matrix_d& A, Matrix_d& B, double alpha, double beta, matrixTranspose transposeA, matrixTranspose transposeB)
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m = A.getRows();
	int n = B.getColumns();
	int k = B.getRows();
	int k2;
	const double* A_ = A.getCMatrix().elements;
	int lda = A.getRows();
	const double* B_ = B.getCMatrix().elements;
	int ldb = B.getRows();
	double* C_ = _Cmatrix.elements;
	int ldc = getRows();

	switch (transposeA)
	{
	case AMatrix::NO_TRANS:
		transa = CUBLAS_OP_N;
		m = A.getRows();
		k2 = A.getColumns();
		break;
	case AMatrix::TRANS:
		transa = CUBLAS_OP_T;
		m = A.getColumns();
		k2 = A.getRows();
		break;
	default:
		break;
	}
	switch (transposeB)
	{
	case AMatrix::NO_TRANS:
		transb = CUBLAS_OP_N;
		n = B.getColumns();
		k = B.getRows();
		break;
	case AMatrix::TRANS:
		transb = CUBLAS_OP_T;
		n = B.getRows();
		k = B.getColumns();
		break;
	default:
		break;
	}
//	lda = m;
//	ldb = k;

	if (k != k2) {
		throw std::exception("Matrix multiplication failed: Incompatible matrix dimensions");
		return;
	}

	stat = cublasDgemm(handle, transa, transb, m, n, k, &alpha, A_, lda, B_, ldb, &beta, C_, ldc);
	cublasDestroy(handle);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
}

void Matrix_d::print(int rows, int columns)
{
	const Matrix_d& m = *this;
	Matrix cpuMatrix(m,matrixInitialisation::M_ASSIGN);
	cpuMatrix.print(rows, columns);
	cpuMatrix.deallocate();
}

void Matrix_d::print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd)
{
	const Matrix_d& m = *this;
	Matrix cpuMatrix(m, matrixInitialisation::M_ASSIGN);
	cpuMatrix.print(rowsStart, rowsEnd, columnsStart, columnsEnd);
	cpuMatrix.deallocate();
}


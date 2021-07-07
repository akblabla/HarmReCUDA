#include "Matrix_d.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <exception>
#include <iostream>
#include "cublas_v2.h"
#include "helper_cuda.h"
Matrix_d::Matrix_d() : AMatrix(0, 0)
{
}
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

Matrix_d::Matrix_d(const Matrix& src, cudaStream_t& stream, matrixInitialisation initialisation) : AMatrix(src.getRows(), src.getColumns())
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

void Matrix_d::uploadToDevice(const Matrix& src, cudaStream_t& stream)
{
	if (getRows() != src.getRows() || getColumns() != src.getColumns()) {
		throw std::exception("Matrix dimension mismatch");
		return;
	}
	auto const cudaStat = cudaMemcpyAsync(_Cmatrix.elements, src.getCMatrix().elements, src.getElementsCount() * sizeof(*src.getCMatrix().elements), cudaMemcpyHostToDevice, stream);
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

void Matrix_d::GeneralMatrixToMatrixMultiply(cublasHandle_t handle, const Matrix_d& A, const Matrix_d& B, double alpha, double beta)
{
	cublasStatus_t stat;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m = A.getRows();
	int n = B.getColumns();
	int k = B.getRows();
	int k2;
	const double* A_ = A.getCMatrix().elements;
	int lda = A.getLeadingDimension();
	const double* B_ = B.getCMatrix().elements;
	int ldb = B.getLeadingDimension();
	double* C_ = _Cmatrix.elements;
	int ldc = getLeadingDimension();

	n = B.getColumns();
	k = B.getRows();
	m = A.getRows();
	k2 = A.getColumns();
	switch (A.isTransposed())
	{
	case false:
		transa = CUBLAS_OP_N;
		break;
	case true:
		transa = CUBLAS_OP_T;
		break;
	default:
		break;
	}
	switch (B.isTransposed())
	{
	case false:
		transb = CUBLAS_OP_N;
		break;
	case true:
		transb = CUBLAS_OP_T;
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

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
}

void Matrix_d::GeneralMatrixToMatrixAddition(cublasHandle_t handle, const Matrix_d& A, const Matrix_d& B, double alpha, double beta)
{
	cublasStatus_t stat;
	cublasOperation_t transa;
	cublasOperation_t transb;
	int m, n, m2, n2;

	const double* A_ = A.getCMatrix().elements;
	int lda = A.getLeadingDimension();
	const double* B_ = B.getCMatrix().elements;
	int ldb = B.getLeadingDimension();
	double* C_ = _Cmatrix.elements;
	int ldc = getLeadingDimension();


	m = A.getRows();
	n = A.getColumns();
	m2 = B.getRows();
	n2 = B.getColumns();
	switch (A.isTransposed())
	{
	case false:
		transa = CUBLAS_OP_N;
		break;
	case true:
		transa = CUBLAS_OP_T;
		break;
	default:
		break;
	}
	switch (B.isTransposed())
	{
	case false:
		transb = CUBLAS_OP_N;
		break;
	case true:
		transb = CUBLAS_OP_T;
		break;
	default:
		break;
	}

	if ((m != m2)|| (n != n2)) {
		throw std::exception("Matrix addition failed: Incompatible matrix dimensions");
		return;
	}

	stat = cublasDgeam(handle, transa, transb, m, n, &alpha, A_, lda, &beta, B_, ldb, C_, ldc);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
}

void Matrix_d::multiplyWithScalar(double alpha)
{
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	cublasOperation_t transa = CUBLAS_OP_N;
	cublasOperation_t transb = CUBLAS_OP_N;
	int m, n;

	double* A_ = getCMatrix().elements;
	int lda = getLeadingDimension();

	m = getRows();
	n = getColumns();
	if (isTransposed())
	{
		n = getRows();
		m = getColumns();
	}
	double beta = 0;
	stat = cublasDgeam(handle,
		transa, transb,
		m, n,
		&alpha,
		A_, lda,
		&beta,
		A_, lda,
		A_, lda);
	cublasDestroy(handle);

	if (stat != CUBLAS_STATUS_SUCCESS) {
		throw std::exception(_cudaGetErrorEnum(stat));
		return;
	}
	
}

void Matrix_d::getSubMatrix(Matrix_d& dest, int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) const
{
	//treat negative numbers as uncapped
	if (rowsEnd < 0) {
		rowsEnd = getRows();
	}
	else {
		rowsEnd++;
	}
	if (columnsEnd < 0) {
		columnsEnd = getColumns();
	}
	else {
		columnsEnd++;
	}
	rowsStart = std::min(getRows(), rowsStart);
	rowsEnd = std::min(getRows(), rowsEnd);
	columnsStart = std::min(getColumns(), columnsStart);
	columnsEnd = std::min(getColumns(), columnsEnd);

	Matrix_d result(rowsEnd - rowsStart, columnsEnd - columnsStart, AMatrix::M_NO_INIT);
	result._Cmatrix.elements = _Cmatrix.elements + (rowsStart + columnsStart * _Cmatrix.ld);
	result._Cmatrix.ld = _Cmatrix.ld;
	dest = result;
}

void Matrix_d::print(int rows, int columns) const
{
	const Matrix_d& m = *this;
	Matrix cpuMatrix(m,matrixInitialisation::M_ASSIGN);
	cpuMatrix.print(rows, columns);
	cpuMatrix.deallocate();
}

void Matrix_d::print(int rowsStart, int rowsEnd, int columnsStart, int columnsEnd) const
{
	const Matrix_d& m = *this;
	Matrix cpuMatrix(m, matrixInitialisation::M_ASSIGN);
	cpuMatrix.print(rowsStart, rowsEnd, columnsStart, columnsEnd);
	cpuMatrix.deallocate();
}

Matrix_d::~Matrix_d()
{
	deallocate();
}